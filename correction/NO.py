import torch


class Refinement:

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def mh_step(self, states, steps=6):
        T = self.model.temperature()
        current = states.clone()

        for _ in range(steps):
            proposal = self.model.gibbs_chain(current, T, steps=2)

            F_cur = self.model.free_energy(current, T)
            F_prop = self.model.free_energy(proposal, T)

            delta = F_prop - F_cur

            prob = torch.exp(torch.clamp(-delta / T, max=0))
            accept = torch.rand_like(prob) < prob

            accept = accept.view(-1, *([1] * (current.dim() - 1)))
            current = torch.where(accept, proposal, current)

        return current

    @torch.no_grad()
    def energy_guided_refine(self, states, steps=5):
        T = self.model.temperature()
        current = states.clone()

        for _ in range(steps):
            proposal = self.model.gibbs_chain(current, T, steps=2)

            F_cur = self.model.free_energy(current, T)
            F_prop = self.model.free_energy(proposal, T)

            better = F_prop < F_cur

            delta = F_prop - F_cur

            prob = torch.exp(torch.clamp(-delta / T, max=0))
            stochastic_accept = torch.rand_like(prob) < prob

            accept = better | stochastic_accept
            accept = accept.view(-1, *([1] * (current.dim() - 1)))

            current = torch.where(accept, proposal, current)

        return current

    @torch.no_grad()
    def soft_refine(self, states, steps=10):
        T = self.model.temperature()
        v = states.clone()

        for _ in range(steps):
            h_prob = torch.sigmoid((v @ self.model.W + self.model.hidden_bias) / T)
            h = torch.bernoulli(h_prob)

            v_prob = torch.sigmoid((h @ self.model.W.T + self.model.visible_bias) / T)

            v = 0.7 * v + 0.3 * v_prob
        return v

    @torch.no_grad()
    def myra_refine(self, states):
        states = self.mh_step(states, steps=6)
        states = self.energy_guided_refine(states, steps=5)
        states = self.soft_refine(states, steps=8)

        return states