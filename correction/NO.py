import torch
import torch.nn.functional as F


class Refinement:

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def exact_energy_refinement(self, v, passes=1, verbose=False):

        T = self.model.temperature().to(v.device)
        v = v.clone()

        if verbose:
            F_before = self.model.free_energy(v, T).mean()

        for _ in range(passes):
            indices = torch.randperm(self.model.n_visible, device=v.device).tolist()

            for into in indices:
                v[:, into] = 0.0
                F0 = self.model.free_energy(v, T)

                v[:, into] = 1.0
                F1 = self.model.free_energy(v, T)

                delta_F = F1 - F0
                prob = torch.sigmoid(-delta_F)

                v[:, into] = (torch.rand_like(prob) < prob).float()

        if verbose:
            F_after = self.model.free_energy(v, T).mean()
            print((F_after - F_before).item())

        return v

    @staticmethod
    @torch.no_grad()
    def soft_connectivity(v, steps=1):

        image = v.view(-1, 1, 28, 28)

        for _ in range(steps):
            neighbors = (
                F.pad(image[:, :, :, 1:], (0, 1, 0, 0)) +
                F.pad(image[:, :, :, :-1], (1, 0, 0, 0)) +
                F.pad(image[:, :, 1:, :], (0, 0, 0, 1)) +
                F.pad(image[:, :, :-1, :], (0, 0, 1, 0))
            )

            turn_on = neighbors >= 4
            turn_off = neighbors == 0

            image = torch.where(turn_on, torch.ones_like(image), image)
            image = torch.where(turn_off, torch.zeros_like(image), image)

        return image.view(-1, 784)

    @torch.no_grad()
    def myra_refine(
        self,
        v,
        energy_passes=1,
        use_connectivity=True
    ):

        if energy_passes > 0:
            v = self.exact_energy_refinement(v, passes=energy_passes)

        if use_connectivity:
            v = self.soft_connectivity(v, steps=1)

        return v