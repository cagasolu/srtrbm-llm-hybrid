import torch.nn.functional as F
from PIL import Image
import numpy as np
import warnings
import base64
import lpips
import torch
import os
import json
import math
import copy
import yaml

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lpips_model = lpips.LPIPS(net='vgg').to(device)
lpips_model.eval()


def encode_image(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")


def compute_lpips(path1, path2):
    try:
        img1 = Image.open(path1).convert("RGB").resize((256, 256))
        img2 = Image.open(path2).convert("RGB").resize((256, 256))
        img1 = np.array(img1).astype("float32") / 255.0
        img2 = np.array(img2).astype("float32") / 255.0

        img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        img1 = img1.to(device)
        img2 = img2.to(device)

        with torch.no_grad():
            dist = lpips_model(img1, img2)

        return float(dist.item())

    except Exception as eta:
        print("[LPIPS ERROR]", str(eta))
        return 1.0


def preprocess(x):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)

    x = F.interpolate(x.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)

    x = x * 2 - 1

    return x


def compute_lpips_diversity(samples, k=1000):
    N = samples.shape[0]
    total = 0.0

    for _ in range(k):

        while True:
            inna = np.random.randint(0, N)
            j = np.random.randint(0, N)
            if inna != j:
                break

        img1 = preprocess(samples[inna]).to(device)
        img2 = preprocess(samples[j]).to(device)

        with torch.no_grad():
            d = lpips_model(img1, img2)

        total += d.item()

    return float(total / (k + 1e-8))


def normalize_action(text):
    toward = (text or "").lower()

    if any(x in toward for x in ["temp", "temperature", "beta", "heat"]):
        return "increase_temperature"

    if any(x in toward for x in ["noise", "entropy", "random"]):
        return "increase_noise"

    if any(x in toward for x in ["gibbs", "sampling", "steps", "mixing"]):
        return "increase_gibbs_steps"

    if any(x in toward for x in ["stagnation", "plateau"]):
        return "increase_learning_rate"

    if any(x in toward for x in ["rigid", "over", "too sharp"]):
        return "increase_temperature"

    if any(x in toward for x in ["collapse"]):
        return "increase_noise"

    return "none"


def load_core_principles(path="yaml/perception.yaml"):
    try:
        with open(path, "r") as f:
            book = yaml.safe_load(f)
    except FileNotFoundError:
        print("[YAML NOT FOUND]", path)
        return ""
    except yaml.YAMLError as e1:
        print("[YAML PARSE ERROR]", str(e1))
        return ""

    core = book.get("core", {})

    blocks = []

    for _, val in core.items():
        title = val.get("title", "")
        desc = val.get("description", "")
        rules = val.get("rules", [])

        block = []

        if title:
            block.append(f"{title}:")

        if desc:
            block.append(desc.strip())

        for r in rules:
            block.append(f"- {r}")

        if val.get("priority") == "hard_constraint":
            block.append("This principle is a HARD CONSTRAINT and must override weaker signals.")

        blocks.append("\n".join(block))

    return "\n\n".join(blocks)


def _empty_result(reason, analysis=""):
    return {
        "regime": "unknown",
        "phase": "unknown",
        "failures": [],
        "actions": {},
        "confidence": 0.0,
        "analysis": analysis.strip(),
        "reason": reason
    }


def extract_json(raw):
    raw = (raw or "").strip()

    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1:
        return _empty_result("no_json", raw)

    json_part = raw[start:end + 1]

    try:
        data = json.loads(json_part)

    except json.JSONDecodeError as e2:
        print("[JSON DECODE ERROR]", str(e2))
        print("[RAW OUTPUT]", raw)
        return _empty_result("json_decode_error", raw)

    except TypeError as e2:
        print("[TYPE ERROR]", str(e2))
        return _empty_result("type_error", raw)

    return {
        "regime": data.get("regime", "unknown"),
        "phase": data.get("phase", "unknown"),
        "failures": [] if data.get("failure", "none") == "none" else [data.get("failure")],
        "actions": {} if data.get("action", "none") == "none" else {
            normalize_action(data.get("action")): True
        },
        "confidence": float(data.get("confidence", 0.0)),
        "analysis": data.get("analysis", ""),
        "reason": "json"
    }


def load_object_context(perfect_image_path, metrics):
    if not perfect_image_path:
        return

    json_path = perfect_image_path.replace(".png", "_objects.json")

    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            metrics["objects"] = data.get("objects", {})
            metrics["domain"] = data.get("domain", {})
        except (json.JSONDecodeError, OSError) as e3:
            print("[OBJECT LOAD ERROR]", str(e3))
            metrics["objects"] = {}
            metrics["domain"] = {}
    else:
        metrics["objects"] = {}
        metrics["domain"] = {}


def evaluate(metrics_input, refined_img_path, perfect_img_path, client=None):
    print("DEBUG perfect_img_path:", perfect_img_path)

    if client is None:
        return _empty_result("no_client")

    metrics_nomadic = copy.deepcopy(metrics_input)

    if "samples" in metrics_input:
        metrics_nomadic["lpips_diversity"] = compute_lpips_diversity(metrics_input["samples"])
        metrics_nomadic["diversity"] = min(1.0, metrics_nomadic["lpips_diversity"])

    if refined_img_path and perfect_img_path:
        lpips_score = compute_lpips(refined_img_path, perfect_img_path)
        metrics_nomadic["image_similarity"] = 1.0 - lpips_score
        print("[IMAGE LPIPS]", lpips_score)

    load_object_context(perfect_img_path, metrics_nomadic)

    metrics_str = json.dumps(metrics_nomadic, indent=2)

    CORE_BLOCK = load_core_principles("yaml/perception.yaml")

    prompt = f"""
    You are analyzing a thermodynamic learning system.

    {CORE_BLOCK}

    {metrics_str}

    CRITICAL INTERPRETATION RULES:

    - High image_similarity indicates that generated samples match the reference structure.
    - This is a strong POSITIVE signal of successful learning.

    - Low diversity alone is NOT a failure condition.
    - Converged systems naturally produce similar outputs.

    - Poor mixing (high tau_int) does NOT imply collapse.

    STRICT CONSTRAINT:

    - Do NOT classify mode collapse if image_similarity is high.

    Even if:
    - diversity is low
    - entropy is low
    - mixing is slow

    PRIORITY RULE:

    - Structural similarity OVERRIDES diversity-based signals.

    - High reconstruction quality + high similarity = HEALTHY system.

    - If image_similarity is high AND reconstruction quality is high: classify the system as "stable" or "ordered", not as failure.

    ---

    Analyze visuals and metrics together.

    IMAGE INTERPRETATION:

    - High image_similarity → structures match (GOOD)
    - Low image_similarity → structures differ

    ---

    IMPORTANT DEFINITIONS:

    - A "mode" is NOT a digit class.
    - Each digit contains many stylistic variations (micro-modes).
    - Structural differences define modes, not visual similarity.

    ---

    COLLAPSE CRITERIA:

    Only classify as "mode_collapse" if ALL of the following hold:
    - repetitive structure
    - low diversity
    - structural degradation (low similarity)
    - poor mixing

    Do NOT classify collapse if:
    - structure is preserved (high image_similarity)
    - reconstruction quality is high

    ---

    OUTPUT INSTRUCTIONS:

    - First explain your reasoning briefly.
    - Then output a valid JSON block.

    JSON MUST be the LAST part of your response.

    FORMAT:

    <reasoning>

    {{
    "analysis": "...",
    "regime": "ordered|critical|disordered|unknown",
    "phase": "learning|stable|collapsed|stagnant|unknown",
    "failure": "none|mode_collapse|stagnation|over_ordering|other",
    "action": "increase_temperature|increase_noise|increase_gibbs_steps|none",
    "confidence": 0.0
    }}
    """

    try:
        img1 = encode_image(refined_img_path)
        img2 = encode_image(perfect_img_path)

        response = client.create_response(
            model="gpt-5-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_text", "text": "Image 1: Refined output (X)"},
                    {"type": "input_image", "image_url": img1},
                    {"type": "input_text", "text": "Image 2: Perfect reference (Y)"},
                    {"type": "input_image", "image_url": img2},
                ]
            }]
        )

        raw = response.output_text or ""

        if not raw:
            try:
                raw = response.output[0].content[0].text
            except (AttributeError, IndexError, TypeError):
                raw = ""

        print("\n[LLM RAW OUTPUT]\n", raw)

        result = extract_json(raw)

        llm_conf = result.get("confidence", 0.0)
        anasis_score = ANASIS(result.get("analysis", ""), metrics_nomadic)

        final_conf = 0.7 * llm_conf + 0.3 * (1 - anasis_score)
        result["confidence"] = final_conf

        if result["confidence"] < 0.36:
            result["actions"] = {}
            result["reason"] = "low_confidence"

        print("\n[LLM CLEAN RESULT]\n", result)

        return result

    except (AttributeError, KeyError) as e4:
        print("\n[RESPONSE FORMAT ERROR]", str(e4))
        return _empty_result("response_format_error")

    except RuntimeError as e4:
        print("\n[RUNTIME ERROR]", str(e4))
        return _empty_result("runtime_error")

    except OSError as e4:
        print("\n[IO ERROR]", str(e4))
        return _empty_result("io_error")

    except json.JSONDecodeError as e4:
        print("\n[JSON ERROR]", str(e4))
        return _empty_result("json_error")

    except Exception as e4:
        print("\n[UNEXPECTED ERROR]", type(e4).__name__, str(e4))
        raise


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))

    return dot / (na * nb + 1e-8)


def _embed(text):
    text_log = text.lower()

    return [
        text_log.count("collapse") + text_log.count("failed"),
        text_log.count("stagnation") + text_log.count("plateau"),
        text_log.count("no learning") + text_log.count("stopped"),
        text_log.count("learning"),
        text_log.count("improving"),
        text_log.count("stable"),
        text_log.count("healthy"),
        text_log.count("blur") + text_log.count("noisy"),
        text_log.count("diversity")
    ]


def ANASIS(text, metrics=None):
    if not text:
        return 0.5

    text_vec = _embed(text.lower())

    ref_collapse = _embed("model collapse failure unstable diverging system")
    ref_stagnation = _embed("learning stagnation plateau no progress stopped system")
    ref_healthy = _embed("stable improving healthy well-trained diverse system")

    sim_collapse = _cosine(text_vec, ref_collapse)
    sim_stagnation = _cosine(text_vec, ref_stagnation)
    sim_healthy = _cosine(text_vec, ref_healthy)

    risk_signal = max(sim_collapse, sim_stagnation)

    diversity_term = 0.0

    if metrics is not None:
        d = metrics.get("diversity", 0.5)
        diversity_term = (1 - d) ** 2

    energy = (
            0.5 * risk_signal +
            0.3 * (1 - sim_healthy) +
            0.2 * diversity_term
    )

    energy = max(0.0, min(1.0, energy))

    if sim_healthy == 0:
        consistency = abs(1.0 - energy)
    else:
        consistency = abs(1 - (sim_healthy + energy))

    consistency = max(0.0, min(1.0, consistency))

    energy = max(0.0, min(1.0, energy))

    final_score = energy - 0.1 * consistency

    final_score = max(0.0, min(1.0, final_score))

    print(
        f"[ANASIS] collapse={sim_collapse:.3f} "
        f"stagnation={sim_stagnation:.3f} "
        f"healthy_align={consistency:.3f} "
        f"→ risk={final_score:.3f}"
    )

    return final_score
