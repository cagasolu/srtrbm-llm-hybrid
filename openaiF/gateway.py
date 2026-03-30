import base64
import os
import json
import math
import copy


def encode_image(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")


def normalize_action(text):
    toward = text.lower()

    if any(x in toward for x in ["temp", "temperature", "beta", "heat"]):
        return "increase_temperature"

    if any(x in toward for x in ["noise", "entropy", "random"]):
        return "increase_noise"

    if any(x in toward for x in ["gibbs", "sampling", "steps"]):
        return "increase_gibbs_steps"

    return "none"


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


def clean_json(raw):
    raw = raw.strip()

    # cut garbage before/after JSON
    if not raw.startswith("{"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end + 1]

    # fix python dict → json
    raw = raw.replace("'", '"')

    return raw


def extract_json(raw):
    try:
        raw = clean_json(raw)
        data = json.loads(raw)

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

    except Exception as ece:
        print("[JSON FIX FAILED]", str(ece))
        print("[RAW]", raw)
        return _empty_result("json_error", raw)


def load_object_context(perfect_image_path, metrics):
    json_path = perfect_image_path.replace(".png", "_objects.json")

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        metrics["objects"] = data.get("objects", {})
        metrics["domain"] = data.get("domain", {})
    else:
        metrics["objects"] = {}
        metrics["domain"] = {}


# noinspection PyBroadException
def evaluate(metrics_input, refined_img_path, perfect_img_path, client=None):
    print("DEBUG perfect_img_path:", perfect_img_path)

    if client is None:
        return _empty_result("no_client")

    metrics_nomadic = copy.deepcopy(metrics_input)
    load_object_context(perfect_img_path, metrics_nomadic)

    metrics_str = json.dumps(metrics_nomadic, indent=2)

    prompt = f"""
You are analyzing a thermodynamic learning system.

{metrics_str}

Analyze visuals and metrics together.

Return ONLY JSON.

Rules:
- Start with {{
- End with }}
- Use ONLY double quotes
- No text before or after
- No explanations

JSON:

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
                    {"type": "input_image", "image_url": img1},
                    {"type": "input_image", "image_url": img2},
                ]
            }]
        )

        raw = response.output_text or ""

        if not raw:
            try:
                raw = response.output[0].content[0].text
            except:
                raw = ""

        print("\n[LLM RAW OUTPUT]\n", raw)

        result = extract_json(raw)

        print("\n[LLM CLEAN RESULT]\n", result)

        return result

    except Exception as eight:
        print("\n[LLM FAILURE]", str(eight))
        return _empty_result(str(eight))


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


def ANASIS(text):
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
    stability_signal = sim_healthy

    risk_score = 0.1 + 0.8 * risk_signal
    stability_score = 0.1 + 0.8 * stability_signal

    final_score = 0.65 * risk_score + 0.35 * (1.0 - stability_score)
    final_score = max(0.0, min(1.0, final_score))

    print(
        f"[ANASIS] collapse={sim_collapse:.3f} "
        f"stagnation={sim_stagnation:.3f} "
        f"healthy={sim_healthy:.3f} "
        f"→ final={final_score:.3f}"
    )

    return final_score