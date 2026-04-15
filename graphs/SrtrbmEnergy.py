import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from graphs.SrtrbmVisualization import save_digit_grid

# Modern plotting style

plt.style.use("seaborn-v0_8-whitegrid")

COLORS = {
    "data": "#1f77b4",  # deep blue
    "model": "#ff7f0e",  # orange
    "mean_data": "#1f77b4",
    "mean_model": "#ff7f0e"
}


# Energy ranking

@torch.no_grad()
def compute_energy_ranking(model, data):
    temperature = model.temperature()

    energies = model.free_energy(data, temperature).detach()

    idx = torch.argsort(energies)

    return energies.cpu(), idx.cpu()


# Energy landscape visualization

@torch.no_grad()
def visualize_energy_extremes(model, data, k=100):
    _, idx = compute_energy_ranking(model, data)

    N = len(data)

    best = data[idx[:k]]
    mid = data[idx[N // 2 - k // 2: N // 2 + k // 2]]
    worst = data[idx[-k:]]

    save_digit_grid(best, "lowest_energy_digits.png")
    save_digit_grid(mid, "median_energy_digits.png")
    save_digit_grid(worst, "highest_energy_digits.png")


# Energy distribution analysis

@torch.no_grad()
def plot_data_vs_model_energy(model, data):
    temperature = model.temperature()

    # full dataset
    n = int(data.shape[0])

    # Data energy

    F_data = model.free_energy(data[:n], temperature)
    F_data = F_data.detach().cpu().numpy()

    # Model sampling

    samples = model.generate_ensemble_samples(
        n_chains=n,
        steps=2000,
    )

    F_model = model.free_energy(samples, temperature)
    F_model = F_model.detach().cpu().numpy()

    # Histogram range

    lo = min(F_data.min(), F_model.min())
    hi = max(F_data.max(), F_model.max())

    bins = np.linspace(lo, hi, 120)

    # Kernel density estimation

    kde_data = gaussian_kde(F_data)
    kde_model = gaussian_kde(F_model)

    x = np.linspace(lo, hi, 500)

    # Plot

    plt.figure(figsize=(7, 5))

    plt.hist(
        F_data,
        bins=bins,
        density=True,
        alpha=0.35,
        color=COLORS["data"],
        label="Data"
    )

    plt.hist(
        F_model,
        bins=bins,
        density=True,
        alpha=0.35,
        color=COLORS["model"],
        label="Model"
    )

    # KDE curves

    plt.plot(
        x,
        kde_data(x),
        color=COLORS["data"],
        linewidth=2.5
    )

    plt.plot(
        x,
        kde_model(x),
        color=COLORS["model"],
        linewidth=2.5
    )

    # mean lines

    mean_data = F_data.mean()
    mean_model = F_model.mean()

    plt.axvline(
        mean_data,
        linestyle="--",
        linewidth=2,
        color=COLORS["mean_data"]
    )

    plt.axvline(
        mean_model,
        linestyle=":",
        linewidth=2,
        color=COLORS["mean_model"]
    )

    energy_gap = abs(mean_data - mean_model)

    plt.xlabel("Free Energy")

    plt.ylabel("Probability Density")

    plt.title(
        f"Free Energy Distribution (ΔF = {energy_gap:.2f})"
    )

    plt.legend(frameon=False)

    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        "srtrbm_energy_data_vs_model_modern.pdf",
        bbox_inches="tight"
    )

    plt.close()

    # Return statistics

    return {

        "mean_data_energy": float(mean_data),

        "mean_model_energy": float(mean_model),

        "energy_gap": float(energy_gap)

    }