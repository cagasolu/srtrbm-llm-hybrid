from openai import OpenAI
import base64
import os
import json
import sys

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) or the alternative transition is as follows.

client = OpenAI(
    api_key="...")


def encode_image(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:image/png;base64,{b64}"


def evaluate_metrics(metrics, image_refined, image_perfect):
    prompt = f"""
You are interpreting a thermodynamic generative system.

Two images are provided:

- Image A: refined generative output
- Image B: stabilized / perfect output

Metrics:
{json.dumps(metrics, indent=2)}

Your task is NOT simple evaluation.
You must interpret the underlying generative dynamics.

Analyze:

1. Attractor behavior:
   - Do outputs converge toward a stable manifold?

2. Diversity transformation:
   - Is diversity reduced, preserved, or refined?

3. Convergence dynamics:
   - Do multiple variations collapse into a consistent form?

4. Structural improvement:
   - Does Image B improve clarity and separability?

Important:
- Repetition is NOT automatically bad.
- Convergence may indicate strong attractor dynamics.

Return ONLY raw JSON:

{{
  "regime": "...",
  "attractor_strength": "low | medium | high",
  "convergence_type": "weak | moderate | strong",
  "diversity_shift": "lost | preserved | refined",
  "structural_improvement": "none | moderate | strong",
  "confidence": 0-1,
  "reason": "deep technical explanation comparing both images"
}}
"""

    try:
        img1 = encode_image(image_refined)
        img2 = encode_image(image_perfect)

        response = client.responses.create(
            model="gpt-5-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": img1
                        },
                        {
                            "type": "input_image",
                            "image_url": img2
                        }
                    ]
                }
            ]
        )

        raw = response.output_text

        if not raw:
            raise ValueError("Empty LLM response")

        raw = raw.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        start = raw.find("{")
        end = raw.rfind("}") + 1
        raw = raw[start:end]

        parsed = json.loads(raw)

        return parsed

    except Exception as e:
        return {
            "regime": "unknown",
            "attractor_strength": "unknown",
            "convergence_type": "unknown",
            "diversity_shift": "unknown",
            "structural_improvement": "unknown",
            "confidence": 0.0,
            "reason": f"LLM error: {str(e)}"
        }


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Usage: python3 call_openai.py <input_json> <output_json> <refined_img> <perfect_img>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    image_refined = sys.argv[3]
    image_perfect = sys.argv[4]

    with open(input_path, "r") as f:
        metrics = json.load(f)

    result = evaluate_metrics(metrics, image_refined, image_perfect)

    print("\n[LLM RESULT]")
    print(result)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved JSON → {output_path}")