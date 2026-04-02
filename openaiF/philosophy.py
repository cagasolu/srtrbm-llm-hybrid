import random
import math


class EpistemicController:
    def __init__(self):
        self.history = []
        self.max_len = 5

    def update(self, result):
        g = result["intensity"]["gibbs_steps"]
        self.history.append(g)
        self.history = self.history[-self.max_len:]

    def average(self):
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def persistent_doubt(self):
        if len(self.history) < 3:
            return False
        return all(into > 0.6 for into in self.history[-3:])

    @staticmethod
    def _stochastic_apply(strength, confidence, temperature):

        # epistemic pressure
        doubt = strength * confidence

        # Boltzmann probability
        probability = 1 / (1 + math.exp(-doubt / (temperature + 1e-8)))

        # sample
        manifold_likely = random.random()
        print(f"[STOCHASTIC] prob={probability} | rand={manifold_likely}")

        return (manifold_likely < probability), probability

    def decide(self, result):
        self.update(result)

        average = self.average()
        actions = result["actions"].copy()
        confidence = result.get("confidence", 0.0)

        # persistent epistemic push

        if self.persistent_doubt():
            actions["increase_gibbs_steps"] = True
            average = max(average, 0.7)

        strength = average

        if not any(actions.values()):
            apply_flag, prob = self._stochastic_apply(
                strength,
                confidence,
                temperature=result.get("temperature", 1.0)
            )

            return {
                "apply": False,
                "prob": prob,
                "strength": strength,
                "actions": actions
            }

        temperature = result.get("temperature", 1.0)

        apply_flag, prob = self._stochastic_apply(
            strength,
            confidence,
            temperature
        )

        return {
            "apply": apply_flag,
            "prob": prob,
            "strength": strength,
            "actions": actions
        }