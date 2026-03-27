import base64
import os
import json
import sys
import re
import copy
from openai import OpenAI

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) or the alternative transition is as follows.

client = OpenAI(
    api_key="...")


def encode_image(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")


def normalize_action(text):
    t = text.lower()

    if "temperature" in t or "beta" in t:
        return "increase_temperature"

    if "noise" in t or "entropy" in t:
        return "increase_noise"

    if "gibbs" in t or "sampling" in t:
        return "increase_gibbs_steps"

    return "none"


# noinspection PyBroadException
def extract_final(text):
    if "FINAL:" not in text:
        raise ValueError("No FINAL block")

    analysis_part = text.split("FINAL:")[0].strip()
    block = text.split("FINAL:")[-1]

    result_lite = {
        "regime": "unknown",
        "phase": "unknown",
        "failures": [],
        "actions": {},
        "confidence": 0.0,
        "reason": "final_block",
        "analysis": analysis_part
    }

    for line in block.splitlines():
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key == "regime":
            result_lite["regime"] = value

        elif key == "phase":
            result_lite["phase"] = value

        elif key == "failure":
            result_lite["failures"].append(value)

        elif key == "action":
            norm = normalize_action(value)
            result_lite["actions"][norm] = True

        elif key == "confidence":
            try:
                result_lite["confidence"] = float(value)
            except:
                pass

    return result_lite


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
        print("[LLM INFO] Using image-based object inference")


def evaluate(metrics_input, refined_img_path, perfect_img_path):
    metrics_nomadic = copy.deepcopy(metrics_input)
    load_object_context(perfect_img_path, metrics_nomadic)

    prompt = f"""
We analyze a thermodynamic learning system.

We consider:
- thermodynamic state
- sampling behavior
- representation quality
- image structure
- energy-based representation learning

Metrics:
{json.dumps(metrics_nomadic, indent=2)}

----------------------------------------

INTERPRETATION PRINCIPLES:

- Low entropy or low diversity does NOT automatically indicate failure.
- In thermodynamic systems, concentrated modes may reflect a valid low-temperature ordered phase.
- Prefer physical interpretation over generic machine learning failure labels.
- Focus on how the energy landscape organizes representations rather than how many distinct samples are generated.

Use the following interpretations when appropriate:
- ordered phase
- attractor concentration
- phase dominance
- reduced configurational entropy

Representation Learning Focus:

- Evaluate whether stable and well-separated energy minima (attractors) correspond to meaningful structural patterns.
- Prioritize representation quality over generative diversity.
- Determine whether the system forms distinct basins for different geometric classes.
- Pay special attention to digits with high geometric complexity and curvature (e.g., 2, 3, 5, 8).
- Check whether such complex structures are underrepresented or collapsed into simpler attractors.
- Interpret dominance of simple shapes (e.g., 0, 1) as potential energy bias toward low-complexity configurations.

Only classify as failure if:
- the system is inconsistent with its thermodynamic regime
- or sampling dynamics are insufficient given temperature
- or the system is trapped despite conditions that should allow exploration
- or geometrically distinct patterns are improperly merged into the same energy basin

If the system is stable and consistent:
- set failure=none
- set action=none

Avoid defaulting to "mode collapse" unless clearly justified.

----------------------------------------

In ANALYSIS:
- Start sentences with subject plus verb such as "We observe", "We see", or "We detect"
- Describe visual patterns in the images
- Identify dominant structures and underrepresented patterns
- Pay explicit attention to geometric complexity (curvature, stroke variation, topology)
- Connect observations to energy landscape structure
- Interpret results in terms of phase behavior (ordered / critical / disordered)
- Explicitly state whether the system behavior is consistent with its temperature regime
- Evaluate whether complex patterns are sufficiently represented as distinct attractors

Think freely.

----------------------------------------

At the end you MUST output:

FINAL:
regime=<value>
phase=<value>
failure=<value>
action=<value>
confidence=<value>

Allowed actions:
- increase_temperature
- decrease_temperature
- increase_noise
- increase_gibbs_steps
- none

Rules:
- FINAL must be short
- One line per field
- No explanations inside FINAL
"""

    try:
        img1 = encode_image(refined_img_path)
        img2 = encode_image(perfect_img_path)

        response = client.responses.create(
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
            raw = response.output[0].content[0].text

        print("\n[LLM RAW OUTPUT PREVIEW]")
        print(raw[:500])

        parsed = extract_final(raw)

        print("[LLM PARSED]", parsed)

        return parsed

    except Exception as enigma:
        print("\n[LLM FAILURE]", str(enigma))

        return {
            "regime": "unknown",
            "phase": "unknown",
            "failures": [],
            "actions": {},
            "confidence": 0.0,
            "analysis": "",
            "reason": str(enigma)
        }