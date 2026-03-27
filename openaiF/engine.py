import copy
import math


class LLMController:

    def __init__(self):
        self.history = []

    def smooth_scale(self, x, center=0.0, sharpness=1.0):
        return 1 / (1 + math.exp(-sharpness * (x - center)))

    def compute_control(self, model, metrics):
        energy_gap = metrics.get("energy_gap", 0.0)
        
        entropy = metrics.get("quality", {}).get("entropy", 1.0)

        temp_scale = model.energy_temp_scale

        gap_signal = self.smooth_scale(energy_gap, center=50.0, sharpness=0.05)

        low_entropy = self.smooth_scale(0.3 - entropy, sharpness=10.0)

        instability = gap_signal * low_entropy

        delta_temp = instability * 0.01
        delta_noise = instability * 0.015

        delta_temp *= (1 + temp_scale * 50)

        return {
            "delta_lambda_gain": 0.0,
            "delta_temp_scale": delta_temp,
            "delta_noise": delta_noise,
            "increase_gibbs": gap_signal > 0.6
        }

    def blend(self, llm_actions, control_actions, confidence):

        w = max(0.0, min(1.0, confidence))

        blended = {}

        for k in control_actions:
            if isinstance(control_actions[k], bool):
                blended[k] = control_actions[k] or llm_actions.get(k, False)
            else:
                blended[k] = (1 - w) * control_actions[k] + w * llm_actions.get(k, 0.0)

        return blended

    def apply(self, model, llm_output, metrics):

        confidence = float(llm_output.get("confidence", 0))
        raw_actions = llm_output.get("actions", {})

        control_actions = self.compute_control(model, metrics)

        llm_mapped = {
            "delta_lambda_gain": 0.0,
            "delta_temp_scale": 0.005 if "increase_temperature" in raw_actions else 0.0,
            "delta_noise": 0.01 if "increase_noise" in raw_actions else 0.0,
            "increase_gibbs": "increase_gibbs_steps" in raw_actions
        }

        actions = self.blend(llm_mapped, control_actions, confidence)

        alpha = 0.1

        before = {
            "lambda_gain": model.lambda_gain,
            "temp_scale": model.energy_temp_scale,
            "flip_smoothing": model.flip_smoothing,
            "gibbs_steps": model.gibbs_steps
        }

        model.lambda_gain += alpha * actions["delta_lambda_gain"]
        model.energy_temp_scale += alpha * actions["delta_temp_scale"]
        model.flip_smoothing += alpha * actions["delta_noise"]

        if actions["increase_gibbs"]:
            model.gibbs_steps = min(model.gibbs_steps + 1, 10)

        model.lambda_gain = max(1e-5, min(0.1, model.lambda_gain))
        model.energy_temp_scale = max(1e-6, min(0.01, model.energy_temp_scale))
        model.flip_smoothing = max(0.0, min(0.1, model.flip_smoothing))

        after = {
            "lambda_gain": model.lambda_gain,
            "temp_scale": model.energy_temp_scale,
            "flip_smoothing": model.flip_smoothing,
            "gibbs_steps": model.gibbs_steps
        }

        self.history.append({
            "before": before,
            "after": after,
            "confidence": confidence,
            "actions": actions
        })

        print("\n[CONTROL CONTINUOUS]")
        print("Before:", before)
        print("After :", after)