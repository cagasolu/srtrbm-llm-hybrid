import re
import yaml
import json
import random
from typing import Dict, Any, Tuple
from openaiF.philosophy import EpistemicController

ACTION_KEY_MAPPING = {
    "increase_temperature": "temperature",
    "increase_noise": "noise",
    "increase_gibbs_steps": "gibbs_steps"
}

controller = EpistemicController()


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_acting_yaml(path: str = "yaml/acting.yaml") -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print("[YAML ERROR] File not found:", path)
    except yaml.YAMLError as e:
        print("[YAML ERROR]", e)
    return {}


def build_acting_block(acting_dict: Dict[str, Any]) -> str:
    try:
        return yaml.dump(acting_dict, allow_unicode=True)
    except yaml.YAMLError:
        return ""


def evaluate_condition(cond: str, metrics: Dict[str, Any]) -> bool:
    if not cond:
        return False

    cond = cond.lower()

    history = metrics.get("history", {})
    delta_w = safe_float(metrics.get("delta_w", 0.0), 0.0)
    std = safe_float(metrics.get("std", 0.0), 0.0)
    flip_rate = safe_float(metrics.get("flip_rate", 1.0), 1.0)

    stagnation = history.get("stagnation", False)
    learning_active = history.get("learning_active", False)
    converged = history.get("converged", False)

    base_checks = {
        "stagnation == true": stagnation,
        "learning_active == true": learning_active,
        "converged == true": converged,
        "low std": std < metrics.get("std_threshold", 0.1),
        "slow mixing": flip_rate < 0.05,
    }

    for key, val in base_checks.items():
        if key in cond:
            return val

    if "delta_w <" in cond:
        return delta_w < 1e-4

    if " and " in cond:
        return all(evaluate_condition(c.strip(), metrics) for c in cond.split("and"))

    if " or " in cond:
        return any(evaluate_condition(c.strip(), metrics) for c in cond.split("or"))

    return False


def interpret_llm_output(raw_text: str) -> Tuple[str, float, Dict[str, bool]]:
    # noinspection PyBroadException
    try:
        data = json.loads(raw_text)
        decision = data.get("decision", "continue")
        confidence = float(data.get("confidence", 0.0))

        actions = data.get("actions", {})

        return decision, confidence, actions

    except Exception:
        pass

    raw = (raw_text or "").lower()

    decision = "continue"
    confidence = 0.0

    actions = {
        "increase_temperature": False,
        "increase_noise": False,
        "increase_gibbs_steps": False
    }

    if "abort" in raw:
        decision = "abort_and_restart"
    elif "adjust" in raw:
        decision = "adjust"

    match = re.search(r"(0\.\d+|1\.0+)", raw)
    if match:
        confidence = float(match.group(1))

    if "temp" in raw:
        actions["increase_temperature"] = True
    if "noise" in raw:
        actions["increase_noise"] = True
    if "gibbs" in raw:
        actions["increase_gibbs_steps"] = True

    return decision, confidence, actions


def compute_continuous_intensity(actions, metrics, confidence):
    intensity = {
        "temperature": 0.0,
        "noise": 0.0,
        "gibbs_steps": 0.0
    }

    std = float(metrics.get("std", 0.0))
    entropy = float(metrics.get("entropy", 0.0))
    delta_w = float(metrics.get("delta_w", 0.0))

    flip_rate = float(metrics.get("flip_rate", 1.0))

    stagnation = metrics.get("history", {}).get("stagnation", False)

    temp_signal = (1 - std + 1 - entropy) / 2

    noise_signal = (1 - delta_w) + (0.6 if stagnation else 0.0)

    noise_signal = min(noise_signal, 1.0)

    gibbs_signal = (1 - flip_rate)

    signals = {
        "increase_temperature": temp_signal,
        "increase_noise": noise_signal,
        "increase_gibbs_steps": gibbs_signal
    }

    for action_name in actions:
        key = ACTION_KEY_MAPPING[action_name]
        intensity[key] = signals[action_name] * confidence

    return intensity


def apply_acting_policy(metrics: Dict[str, Any], result: Dict[str, Any], acting_yaml: Dict[str, Any]) -> Dict[str, Any]:
    policies = acting_yaml.get("acting_system", {}).get("acting_policy", [])

    actions = result.get("actions", {})

    for policy in policies:
        if evaluate_condition(policy.get("condition"), metrics):

            action = policy.get("action")

            if action == "none":
                return result

            if action in actions:
                result["actions"][action] = True
                result.setdefault("notes", []).append(policy.get("id"))
                return result

    return result


def enforce_constraints(metrics: Dict[str, Any], result: Dict[str, Any], acting_yaml: Dict[str, Any]) -> Dict[str, Any]:
    actions = result.get("actions", {})

    confidence = safe_float(result.get("confidence", 0.0), 0.0)

    system = acting_yaml.get("acting_system", {})
    params = system.get("control_parameters", {})
    constraints = system.get("constraints", [])

    conf_th = safe_float(params.get("confidence_threshold", 0.4), 0.4)
    delta_th = safe_float(params.get("convergence_delta_w", 1e-4), 1e-4)

    history = metrics.get("history", {})

    stagnation = history.get("stagnation", False)
    std = float(metrics.get("std", 0.0))

    base_rate = (
            0.12
            + 0.55 * float(stagnation)
            + 0.18 * (1.0 - std)
    )
    base_rate = max(0.0, min(base_rate, 1.0))

    if history.get("converged", False):
        base_rate = 0.0

    delta_w = safe_float(metrics.get("delta_w", 0.0), 0.0)

    for rule in constraints:
        rtype = rule.get("type")
        action_type = rule.get("action")

        if rtype == "confidence" and confidence < conf_th:

            if action_type == "block":
                result.setdefault("notes", []).append(rule.get("id"))

            elif action_type == "dampen":
                for k in actions:
                    if actions[k]:
                        actions[k] = (random.random() < base_rate)
                result.setdefault("notes", []).append(rule.get("id"))

        elif rtype == "convergence":
            if history.get("converged") or delta_w < delta_th:

                if action_type == "block_all":
                    result["decision"] = "continue"

                    result.setdefault("notes", []).append(rule.get("id"))

                    return result

        elif rtype == "state":
            if history.get("learning_active") and not history.get("stagnation"):

                if action_type == "block_all":
                    result.setdefault("notes", []).append(rule.get("id"))

                elif action_type == "dampen":
                    for k in actions:
                        if actions[k]:
                            actions[k] = (random.random() < base_rate)

                    result.setdefault("notes", []).append(rule.get("id"))


        elif rtype == "structure":
            if sum(actions.values()) > 1 and action_type == "clamp_single":
                result["actions"] = {
                    "increase_temperature": False,
                    "increase_noise": False,
                    "increase_gibbs_steps": True
                }
                result.setdefault("notes", []).append(rule.get("id"))

    return result


def request_llm_guidance(metrics: Dict[str, Any], client) -> Dict[str, Any]:
    acting_yaml = load_acting_yaml()

    acting_block = build_acting_block(acting_yaml)

    response = client.create_response(
        model="gpt-5-mini",
        input=f"""
        You observe a thermodynamic RBM training process.

        ACTING SYSTEM:
        {acting_block}

        entropy={metrics['entropy']}
        diversity={metrics['diversity']}
        energy_gap={metrics['energy_gap']}
        std={metrics['std']}
        temperature={metrics['temperature']}
        flip_rate={metrics['flip_rate']}
        delta_w={metrics['delta_w']}
        beta_eff={metrics['beta_eff']}

        OUTPUT:
        D=<continue|abort>;
        C=<0.0-1.0>;
        A=<multiple allowed: temp, noise, gibbs>;
        """
    )

    raw = response.output_text.strip()

    decision, confidence, actions = interpret_llm_output(raw)

    return {
        "decision": decision,
        "confidence": confidence,
        "actions": actions,
    }


def llm_determine(metrics: Dict[str, Any], _trend=None, client=None) -> Dict[str, Any]:
    if client is None:
        return {"decision": "continue", "confidence": 0.0, "actions": {}}

    acting_yaml = load_acting_yaml()

    result = request_llm_guidance(metrics, client)

    result["temperature"] = metrics.get("temperature", 1.0)

    result = apply_acting_policy(metrics, result, acting_yaml)

    result = enforce_constraints(metrics, result, acting_yaml)

    result["intensity"] = compute_continuous_intensity(
        result["actions"],
        metrics,
        result["confidence"]
    )

    ctrl = controller.decide(result)

    result["epistemic"] = ctrl

    return result