# LLM GATEWAY (PROVIDER-AGNOSTIC INTERFACE)
#
# This module provides a high-level evaluation interface that integrates
# thermodynamic metrics with LLM-based semantic interpretation.
#
# IMPORTANT DESIGN PRINCIPLE:
# The system is strictly LLM-provider agnostic.
# No specific API (OpenAI, Anthropic, Gemini, etc.) is assumed.
#
# All LLM interaction is routed through a global client:
#
#     raw = client.generate(prompt, images=[...])
#
# The client is expected to return raw text output only.
# Any provider-specific formatting, API calls, or response schemas
# must be handled inside the client backend implementation.
#
# This module does NOT depend on:
#   - response objects
#   - structured API outputs
#   - provider-specific message formats
#
# Instead, it assumes:
#   - raw string output from an LLM
#   - JSON content embedded within that output
#
# The gateway is responsible for:
#   1. Constructing structured prompts from system metrics
#   2. Passing inputs (text + images) to the LLM client
#   3. Extracting and validating JSON from raw responses
#   4. Enforcing epistemic and structural constraints
#
# If no LLM backend is provided:
#   - The system will gracefully degrade
#   - A fallback result will be returned
#
# This design ensures full decoupling between:
#   - thermodynamic learning (SR-TRBM)
#   - semantic interpretation (LLM)
#
# Result:
#   The system remains stable, extensible, and backend-independent.

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
            block.append("This principle is a HARD CONSTRAINT and ought to override weaker signals.")

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

    scores = data.get("scores") or {
        "temperature": 0.0,
        "gibbs": 0.0
    }

    return {
        "regime": data.get("regime", "unknown"),
        "phase": data.get("phase", "unknown"),
        "failures": [] if data.get("failure", "none") == "none" else [data.get("failure")],
        "scores": scores,
        "risk": data.get("risk", {}),
        "actions": {},
        "confidence": float(data.get("confidence", 0.0)),
        "analysis": data.get("analysis", ""),
        "reason": "json"
    }


def extract_json_safe(raw):
    raw = (raw or "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return extract_json(raw)


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


def Evaluate(metrics_input, refined_img_path, perfect_img_path, client=None):
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

    metrics_nomadic = enforce_hierarchy(metrics_nomadic)
    load_object_context(perfect_img_path, metrics_nomadic)

    metrics_str = json.dumps(metrics_nomadic, indent=2)

    CORE_BLOCK = load_core_principles("yaml/perception.yaml")

    prompt_main = f"""
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

    ADDITIONAL CLASSIFICATION TASK:

    Classify the system into the following categories:

    - regime: one of ["learning", "stable", "stagnant", "converged"]
    - phase: one of ["ordered", "critical", "disordered"]

    Guidelines:

    - ordered: low entropy, high beta, structured outputs
    - disordered: high entropy, weak structure
    - critical: balance between structure and diversity

    - learning: weights are still changing (delta_w noticeable)
    - stable: learning slowed but system remains healthy
    - stagnant: little to no progress
    - converged: learning has effectively finished

    Notes:

    - Choose the closest matching category based on the evidence.
    - Avoid leaving these fields undefined.
    - When uncertain, select the most plausible category rather than abstaining.

    ---

    OUTPUT INSTRUCTIONS:

    - First explain your reasoning briefly.
    - Then output a valid JSON block.

    JSON ought to be the LAST part of your response.

    RULES:

    - Scores ought to be between 0 and 1
    - Do NOT output discrete actions
    - Higher score = stronger recommendation
    """

    prompt_format = """
    FORMAT:

    <reasoning>

    {
    "analysis": "...",

    "regime": "...",
    "phase": "...",

    "scores": {
    "temperature": 0.0,
    "gibbs": 0.0
    },

    "risk": {
    "stagnation": 0.0,
    "collapse": 0.0,
    "over_ordering": 0.0
    },

    "confidence": 0.0
    }

    EPISTEMIC INTERPRETATION (MANDATORY):

    - You ought to interpret confidence as bounded by evidence.
    - Confidence is not an independent belief; it is a function of evidence.

    - If your internal belief is high but evidence is limited, you ought to explain the discrepancy.

    - You ought to explicitly reflect this relationship in your analysis:

        - High belief + low evidence → constrained confidence
        - Low evidence → weak epistemic justification

    - Do NOT just output a number. You ought to justify confidence in terms of evidence.

    FORMAL RULE:
    confidence ≤ evidence

    INTERPRETATION RULE:
    confidence represents only what is supported by evidence, not what is intuitively likely.

    STRICT OUTPUT RULES:
    - Output ONLY one JSON object
    - Do NOT include text after JSON
    - Do NOT include markdown (no ```json)
    - JSON ought to be directly parseable by json.loads()
    """

    prompt = prompt_main + prompt_format

    try:
        img1 = encode_image(refined_img_path)
        img2 = encode_image(perfect_img_path)

        raw = client.generate(
            prompt,
            images=[img1, img2]
        )

        if raw is None or not str(raw).strip():
            return _empty_result("empty_response")

        if len(raw) < 2000:
            print("\n[LLM RAW OUTPUT]\n", raw)
        else:
            print("\n[LLM RAW OUTPUT] (truncated)\n", raw[:2000])

        result = extract_json_safe(raw)
        result = validate_llm_output(result, metrics_nomadic)

        if not result.get("analysis") or len(result.get("analysis", "").strip()) < 5:
            result["analysis"] = "No meaningful analysis provided by LLM"

        llm_conf = result.get("confidence", 0.0)
        evidence = compute_evidence(metrics_nomadic)

        final_conf = min(llm_conf, evidence)

        sim = metrics_nomadic.get("image_similarity", 0.0)

        if evidence < 0.1 and sim < 0.7:
            final_conf = 0.0
            result["reason"] = "no_evidence"

        final_conf = max(0.0, min(1.0, final_conf))
        result["confidence"] = final_conf
        result["evidence"] = evidence
        result["llm_conf_raw"] = llm_conf

        print(f"[SYS] EVIDENCE={evidence:.3f} | LLM={llm_conf:.3f} → FINAL={final_conf:.3f}")

        if result["confidence"] < 0.5:
            for k in result.get("scores", {}):
                result["scores"][k] = max(0.0, min(1.0, result["scores"][k] * 0.5))
            if result.get("reason"):
                result["reason"] += "|low_confidence_scaled"
            else:
                result["reason"] = "low_confidence_scaled"

        print("\n[LLM CLEAN RESULT]\n", result)

        if result.get("phase") == "unknown":
            print("[WARNING] phase unresolved after parsing")

        if result.get("regime") == "unknown":
            print("[WARNING] regime unresolved after parsing")

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


def compute_evidence(metrics):
    signals = [
        metrics.get("image_similarity", 0.0),
        1 - metrics.get("std", 1.0),
        metrics.get("flip_rate", 0.0),
        metrics.get("diversity", 0.5),
    ]

    delta_w = metrics.get("delta_w", 0.0)
    signals.append(min(1.0, delta_w * 1000))

    return sum(signals) / len(signals)


def enforce_hierarchy(metrics):
    sim = metrics.get("image_similarity", 0.0)

    if sim > 0.8:
        if "diversity" in metrics:
            metrics["diversity"] *= 0.3

    return metrics


def validate_llm_output(result, metrics):
    sim = metrics.get("image_similarity", 0.0)

    if sim > 0.8:
        analysis = str(result.get("analysis", "")).lower()

        if "collapse" in analysis and "no collapse" not in analysis:
            result.setdefault("risk", {})
            result["risk"]["collapse"] = 0.0
            result["analysis"] += " [OVERRIDDEN: high structural similarity]"

    return result


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