import json

def LLMRefinementSignal(payload, client=None):
    if client is None:
        print("[LLM] client is None → skipping")
        return {
            "points_add": [],
            "points_remove": [],
            "confidence": 0.0,
            "abstained": True,
            "reason": "no_client",
            "raw": None
        }

    if not isinstance(payload, dict):
        raise TypeError("payload must be dict")

    if "image_ascii" in payload:

        ascii_img = payload["image_ascii"]

        prompt = f"""
        You are improving a handwritten digit.

        Image:
        {ascii_img}

        Task:
        - Fix broken strokes
        - Connect nearby parts
        - Remove isolated noise

        IMPORTANT:
        - Do NOT change the digit identity
        - Prefer adding structure over removing

        Return ONLY JSON:

        {{
        "points_add": [[row, col], ...],
        "points_remove": [[row, col], ...],
        "confidence": float,
        "abstained": false
        }}
        """

        try:
            response = client.create_response(
                model="gpt-5-mini",
                input=[{
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}]
                }]
            )
        except Exception as e:
            print("[LLM ERROR]", e)
            return {
                "points_add": [],
                "points_remove": [],
                "confidence": 0.0,
                "abstained": True,
                "reason": "llm_call_failed",
                "raw": None
            }

        try:
            raw = response.output[0].content[0].text
        except:
            raw = getattr(response, "output_text", "") or ""

        raw = (raw or "").strip()
        print("\n[LLM RAW POINT MODE]", raw)

        try:
            result = json.loads(raw)
        except:
            try:
                start = raw.find("{")
                end = raw.rfind("}")
                result = json.loads(raw[start:end+1])
            except:
                print("[PARSE FAIL POINT MODE]")
                return {
                    "points_add": [],
                    "points_remove": [],
                    "confidence": 0.0,
                    "abstained": True,
                    "reason": "json_parse_fail",
                    "raw": raw
                }

        return {
            "points_add": result.get("points_add", []),
            "points_remove": result.get("points_remove", []),
            "confidence": float(result.get("confidence", 0.0)),
            "abstained": bool(result.get("abstained", False)),
            "reason": "point_mode",
            "raw": result
        }

    active_pixels = payload.get("active_pixels")

    if not isinstance(active_pixels, list):
        raise TypeError("active_pixels must be list")

    return {
        "points_add": [],
        "points_remove": [],
        "confidence": 0.0,
        "abstained": True,
        "reason": "fallback_old_mode"
    }