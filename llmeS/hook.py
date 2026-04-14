# LLM ENERGY MODULE (PROVIDER-AGNOSTIC SEMANTIC ENERGY)
#
# This module defines an LLM-based energy estimator that interprets
# structured visual representations as probabilistic signals.
#
# DESIGN PRINCIPLE:
# The LLM is treated as a black-box semantic energy oracle.
# No assumptions are made about the underlying provider.
#
# All interaction is performed through a global client:
#
#     raw = client.generate(prompt)
#
# The client MUST return raw text output.
# Any API-specific logic is handled externally.
#
# This module is responsible for:
#   - constructing structured prompts from sparse representations
#   - extracting probability distributions from raw LLM output
#   - validating and normalizing probability vectors
#
# FAILURE MODE:
# If the LLM backend fails or returns invalid output:
#   - the function returns (None, error_code)
#   - the main system continues without interruption
#
# ARCHITECTURAL ROLE:
# This module provides a semantic energy term that augments
# the thermodynamic model with external probabilistic signals.

import json
import torch
import numpy as np
import hashlib
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_json_block(text: str):
    stack = []
    start_idx = None
    for into, ch in enumerate(text):
        if ch == "{":
            if start_idx is None:
                start_idx = into
            stack.append(ch)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    return text[start_idx:into + 1]
    return None


def upscale_to_64(grid_2d):
    if grid_2d.dim() == 2:
        grid_2d = grid_2d.unsqueeze(0).unsqueeze(0)
    upscaled = F.interpolate(grid_2d, size=(64, 64), mode="bilinear", align_corners=False)
    return upscaled.squeeze(0).squeeze(0)


def _component_direction_pca(coordinates_np):
    if len(coordinates_np) < 3:
        return [0.0, 0.0]

    cov = np.cov(coordinates_np.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    main_vec = eigenvectors[:, np.argmax(eigenvalues)]

    norm = np.linalg.norm(main_vec) + 1e-12
    main_vec = main_vec / norm

    if main_vec[0] < 0:
        main_vec = -main_vec

    return [float(main_vec[0]), float(main_vec[1])]


def _component_features(coordinates_list):
    coordinate = np.array(coordinates_list)

    if coordinate.size == 0:
        return {
            "size": 0,
            "bbox": [0, 0, 0, 0],
            "center": [0.0, 0.0],
            "direction": [0.0, 0.0],
            "aspect_ratio": 1.0
        }

    ys = coordinate[:, 0]
    xs = coordinate[:, 1]

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())

    center = [float(ys.mean()), float(xs.mean())]
    direction = _component_direction_pca(coordinate)

    aspect_ratio = (y2 - y1 + 1) / (x2 - x1 + 1 + 1e-6)

    return {
        "size": int(len(coordinates_list)),
        "bbox": [y1, x1, y2, x2],
        "center": center,
        "direction": direction,
        "aspect_ratio": float(aspect_ratio)
    }


def find_connected_components_fast(grid_2d, threshold=0.15):
    from scipy.ndimage import label

    binary = (grid_2d > threshold).detach().cpu().numpy()
    labeled, num = label(binary)

    components = []
    for into in range(1, num + 1):
        coordinates_plus = np.argwhere(labeled == into)

        components.append(coordinates_plus.tolist())

    return components


def to_sparse_gpu(grid_2d, threshold=0.15):
    grid_2d = grid_2d.to(device)
    mask = grid_2d > threshold

    coordinate_grow = torch.nonzero(mask, as_tuple=False)

    values = grid_2d[mask]

    coordinates_cpu = coordinate_grow.cpu().numpy()
    values_cpu = values.cpu().numpy()

    active_pixels = [
        [int(r), int(clm), round(float(v), 3)]
        for (r, clm), v in zip(coordinates_cpu, values_cpu)
    ]

    if len(active_pixels) > 800:
        step = max(1, len(active_pixels) // 800)
        active_pixels = active_pixels[::step]

    raw_components = find_connected_components_fast(grid_2d, threshold)

    comp_objs = []
    for comp in raw_components:
        feats = _component_features(comp)

        step = max(1, len(comp) // 50)
        simplified = comp[::step]

        comp_objs.append({
            "points": simplified,
            "size": feats["size"],
            "bbox": feats["bbox"],
            "center": feats["center"],
            "direction": feats["direction"],
            "aspect_ratio": feats["aspect_ratio"]
        })

    return json.dumps({
        "size": list(grid_2d.shape),
        "pixels": active_pixels,
        "components": comp_objs
    })


def hash_repr(s: str):
    return hashlib.md5(s.encode()).hexdigest()


_cache = {}


def LLMEnergy(image_repr: str, client=None):
    if client is None:
        return None, "no_client"

    key = hash_repr(image_repr)
    if key in _cache:
        return _cache[key], None

    prompt = f"""
    You MUST follow:
    1. Use components to infer topology
    2. Use pixels to infer intensity
    3. Use component directions to infer stroke flow
    4. Use aspect ratios to refine structural interpretation
    5. Output calibrated probabilities
    
    Return ONLY JSON:
    {{"probs": [p0, p1, ..., p9]}}
    
    Image:
    {image_repr}
    """

    try:
        raw_output = client.generate(prompt)

        if raw_output is None or not str(raw_output).strip():
            return None, "empty_response"

        raw_output = raw_output.strip()

    except Exception as e:
        return None, f"client_error: {e}"

    try:
        parsed_json = json.loads(raw_output)
    except json.JSONDecodeError:
        json_block = extract_json_block(raw_output)
        if json_block is None:
            return None, "json_not_found"
        try:
            parsed_json = json.loads(json_block)
        except json.JSONDecodeError:
            return None, "json_parse_error"

    probs = parsed_json.get("probs")

    if not isinstance(probs, list) or len(probs) != 10:
        return None, "invalid_probs"

    try:
        prob_vector = np.array(probs, dtype=np.float32)
    except ValueError:
        return None, "non_numeric_probs"

    total = prob_vector.sum()
    if total <= 0:
        return None, "invalid_distribution"

    prob_vector /= total

    result = {"probs": prob_vector.tolist()}
    _cache[key] = result

    return result, None


def LIES_gpu(prob_vector, alpha=5.0, beta=2.0, gamma=2.0):
    prob = torch.tensor(prob_vector, device=device)

    eps = 1e-12
    entropy = -torch.sum(prob * torch.log(prob + eps))
    confidence = torch.max(prob)

    top2 = torch.topk(prob, 2).values
    margin_out = top2[0] - top2[1]

    energy = alpha * entropy - beta * confidence - gamma * margin_out
    return energy.item()


def process_digit(grid_2d, client=None):
    grid_2d = grid_2d.to(device)

    if grid_2d.numel() == 784:
        grid_2d = grid_2d.view(28, 28)
        grid_2d = upscale_to_64(grid_2d)
    elif grid_2d.numel() == 4096:
        grid_2d = grid_2d.view(64, 64)
    else:
        raise ValueError("Unexpected input size")

    image_repr = to_sparse_gpu(grid_2d, threshold=0.15)

    result, err = LLMEnergy(image_repr, client)
    if err:
        return None, err

    energy = LIES_gpu(result["probs"])

    return {"probs": result["probs"], "energy": energy}, None