import re

DEBUG_LLM_OUTPUT = False

recent_action_intensity = {
    "temperature": 0.0,
    "noise": 0.0,
    "gibbs_steps": 0.0
}

ACTION_KEY_MAPPING = {
    "increase_temperature": "temperature",
    "increase_noise": "noise",
    "increase_gibbs_steps": "gibbs_steps"
}


# noinspection PyBroadException
def extract_response_text(response):
    if response is None:
        return ""

    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()

    try:
        collected_text = []
        for item in getattr(response, "output", []):
            for content in getattr(item, "content", []):
                if hasattr(content, "text") and content.text:
                    collected_text.append(str(content.text))
                elif isinstance(content, dict) and "text" in content:
                    collected_text.append(str(content["text"]))
        return "\n".join(collected_text).strip()
    except Exception:
        return ""


def parse_llm_output_strict(raw_text):
    segments = re.split(r"[;\n]+", raw_text.strip().lower())

    if len(segments) < 3:
        raise ValueError

    decision_token = segments[0].split("=")[1].strip()
    confidence_value = float(segments[1].split("=")[1].strip())
    action_segment = segments[2].split("=")[1].strip()

    action_flags = {
        "increase_temperature": False,
        "increase_noise": False,
        "increase_gibbs_steps": False
    }

    for token in action_segment.split(","):
        token = token.strip()
        if token == "temp":
            action_flags["increase_temperature"] = True
        elif token == "noise":
            action_flags["increase_noise"] = True
        elif token == "gibbs":
            action_flags["increase_gibbs_steps"] = True

    decision_mapping = {
        "continue": "continue",
        "adjust": "adjust",
        "abort": "abort_and_restart"
    }

    parsed_decision = decision_mapping.get(decision_token, "continue")

    return parsed_decision, confidence_value, action_flags


# noinspection PyBroadException
def parse_llm_output_fallback(raw_text):
    parsed_decision = "continue"
    parsed_confidence = 0.0

    action_flags = {
        "increase_temperature": False,
        "increase_noise": False,
        "increase_gibbs_steps": False
    }

    raw_text = raw_text.lower()

    if "d=adjust" in raw_text:
        parsed_decision = "adjust"
    elif "d=abort" in raw_text:
        parsed_decision = "abort_and_restart"

    confidence_match = re.search(r"c=([0-9]*\.?[0-9]+)", raw_text)
    if confidence_match:
        try:
            parsed_confidence = float(confidence_match.group(1))
        except:
            parsed_confidence = 0.0

    if "a=" in raw_text:
        action_part = raw_text.split("a=")[-1]
        tokens = re.split(r"[,\s;]+", action_part)

        for token in tokens:
            if token == "temp":
                action_flags["increase_temperature"] = True
            elif token == "noise":
                action_flags["increase_noise"] = True
            elif token == "gibbs":
                action_flags["increase_gibbs_steps"] = True

    return parsed_decision, parsed_confidence, action_flags


# noinspection PyBroadException
def interpret_llm_output(raw_text):
    try:
        return parse_llm_output_strict(raw_text)
    except Exception:
        return parse_llm_output_fallback(raw_text)


def apply_action_dampening(action_flags):
    global recent_action_intensity

    decay = 0.8
    growth = 1.0
    max_intensity = 3.0

    for action_name, is_active in action_flags.items():
        mapped_key = ACTION_KEY_MAPPING[action_name]

        recent_action_intensity[mapped_key] *= decay

        if is_active:
            recent_action_intensity[mapped_key] += growth

            if recent_action_intensity[mapped_key] > max_intensity:
                recent_action_intensity[mapped_key] = max_intensity

    return action_flags


# noinspection PyBroadException
def request_llm_guidance(metric_snapshot, trend_snapshot, client):
    try:
        response = client.create_response(
            model="gpt-5-mini",
            input=f"""
            You observe a thermodynamic RBM training process.
            
            ROLE:
            - You are a global observer
            - You DO NOT control training directly
            - You only decide whether to CONTINUE or ABORT
            
            GOAL:
            - Detect stagnation
            - Preserve meaningful learning dynamics
            - Avoid premature stopping
            
            IMPORTANT:
            - Low entropy can be valid (ordered phase)
            - Use trends, not single values
            - Do NOT overreact
            
            CORE STATE
            
            entropy={metric_snapshot['entropy']:.6f}
            diversity={metric_snapshot['diversity']:.6f}
            energy_gap={metric_snapshot['energy_gap']:.6f}
            std={metric_snapshot['std']:.6f}
            
            temperature={metric_snapshot['temperature']:.6f}
            flip_rate={metric_snapshot['flip_rate']:.6f}
            delta_w={metric_snapshot['delta_w']:.6f}
            beta_eff={metric_snapshot['beta_eff']:.6f}
            
            HISTORY
            
            gap_global={metric_snapshot['history']['gap_global']:.6f}
            entropy_global={metric_snapshot['history']['entropy_global']:.6f}
            temp_global={metric_snapshot['history']['temp_global']:.6f}
            beta_global={metric_snapshot['history']['beta_global']:.6f}
            
            gap_local={metric_snapshot['history']['gap_local']:.6f}
            entropy_local={metric_snapshot['history']['entropy_local']:.6f}
            temp_local={metric_snapshot['history']['temp_local']:.6f}
            beta_local={metric_snapshot['history']['beta_local']:.6f}
            
            gap_std={metric_snapshot['history']['gap_std']:.6f}
            entropy_std={metric_snapshot['history']['entropy_std']:.6f}
            temp_std={metric_snapshot['history']['temp_std']:.6f}
            
            delta_w_hist={metric_snapshot['history']['delta_w']:.6f}
            
            stagnation={metric_snapshot['history']['stagnation']}
            learning_active={metric_snapshot['history']['learning_active']}
            
            TREND
            
            energy_gap_slope={trend_snapshot['energy_gap_slope']:.6f}
            entropy_slope={trend_snapshot['entropy_slope']:.6f}
            temperature_slope={trend_snapshot['temperature_slope']:.6f}
            beta_slope={trend_snapshot['beta_slope']:.6f}
            
            INTERPRETATION RULES
            
            - stagnation = low std AND delta_w ≈ 0
            - learning_active = weights still changing
            - gap_global < 0 → long-term improvement
            - gap_local < 0 → recent improvement
            - entropy ↓ → ordering phase
            - entropy ↑ → exploration phase
            
            DECISION POLICY
            
            CONTINUE if:
            - learning_active is True
            - OR energy_gap is improving (global or local)
            - OR system is stable but not stuck
            
            ABORT ONLY IF ALL TRUE:
            - stagnation = True
            - learning_active = False
            - energy_gap_slope ≈ 0
            
            OUTPUT FORMAT (STRICT)
            
            D=<continue|abort>;
            C=<0.0-1.0>;
            A=none
            """,
            reasoning={"effort": "medium"}
        )

        raw_output = extract_response_text(response)

        if DEBUG_LLM_OUTPUT:
            print("[GPT RAW OUTPUT]", raw_output)

        if not raw_output or len(raw_output) < 5:
            return None

        decision, confidence, action_flags = interpret_llm_output(raw_output)

        action_flags = apply_action_dampening(action_flags)

        return {
            "decision": decision,
            "confidence": confidence,
            "actions": action_flags,
            "source": "OpenAI gpt-5-mini"
        }

    except Exception as emerald:
        print("\n[LLM ERROR]", str(emerald))

        return None


def llm_determine(metrics, trend, client=None):
    if client is None:
        return {
            "decision": "continue",
            "confidence": 0.0,
            "actions": {},
            "source": "no_client"
        }

    result = request_llm_guidance(metrics, trend, client)

    if result is None:
        return {
            "decision": "continue",
            "confidence": 0.0,
            "actions": {},
            "source": "fallback_safe"
        }

    return result