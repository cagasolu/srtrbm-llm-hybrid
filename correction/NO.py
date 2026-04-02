import torch
import torch.nn.functional as F


class Refinement:

    @staticmethod
    @torch.no_grad()
    def exact_energy_refinement(model, v, verbose=False):

        T = model.temperature()

        v = v.clone()

        if verbose:
            energy_before = model.free_energy(v, T).mean()
        else:
            energy_before = None

        for i in range(model.n_visible):
            v0 = v.clone()
            v1 = v.clone()

            v0[:, i] = 0
            v1[:, i] = 1

            F0 = model.free_energy(v0, T)
            F1 = model.free_energy(v1, T)

            v[:, i] = (F1 < F0).float()

        if verbose:
            energy_after = model.free_energy(v, T).mean()
            delta_F = energy_after - energy_before

            print("\nExact Energy Refinement Diagnostics")
            print("-----------------------------------")
            print(f"Free Energy Before : {energy_before.item():.6f}")
            print(f"Free Energy After  : {energy_after.item():.6f}")
            print(f"ΔF (After-Before)  : {delta_F.item():.6f}")

        return v

    @staticmethod
    @torch.no_grad()
    def pixel_connectivity_refine(v, steps=2):

        image = v.view(-1, 1, 28, 28)

        for _ in range(steps):
            neighbors = (
                    F.pad(image[:, :, :, 1:], (0, 1, 0, 0)) +
                    F.pad(image[:, :, :, :-1], (1, 0, 0, 0)) +
                    F.pad(image[:, :, 1:, :], (0, 0, 0, 1)) +
                    F.pad(image[:, :, :-1, :], (0, 0, 1, 0))
            )

            turn_on = (neighbors >= 4)

            turn_off = (neighbors <= 1)

            image = torch.where(turn_on, torch.ones_like(image), image)
            image = torch.where(turn_off, torch.zeros_like(image), image)

        return image.view(-1, 784)