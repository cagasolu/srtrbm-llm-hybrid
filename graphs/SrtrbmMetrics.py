import torch
import numpy as np
import matplotlib.pyplot as plt

# Modern plotting style

plt.style.use("seaborn-v0_8-whitegrid")

COLORS = {
    "flip": "#1f77b4",
    "smooth": "#d62728",
    "sus": "#2ca02c"
}


# Sample quality diagnostics

@torch.no_grad()
def sample_quality_metrics(model, real_data, n_samples=5000, diversity_pairs=3000):
    samples = model.generate_ensemble_samples(
        n_chains=n_samples,
        steps=2000
    ).float()

    # Pixel entropy

    p = samples.mean(0)

    entropy = -(p * torch.log(p.clamp(min=1e-8)) +
                (1 - p) * torch.log((1 - p).clamp(min=1e-8))).mean()

    # Diversity

    flat = samples.view(n_samples, -1)

    device = flat.device

    idx1 = torch.randint(0, n_samples, (diversity_pairs,), device=device)
    idx2 = torch.randint(0, n_samples, (diversity_pairs,), device=device)

    dists = torch.abs(flat[idx1] - flat[idx2]).mean(1)

    diversity = dists.mean().item()

    # Mean distribution distance

    real_mean = real_data.float().mean(0)
    gen_mean = samples.mean(0)

    mean_l2 = torch.norm(real_mean - gen_mean) / np.sqrt(real_mean.numel())

    return {
        "pixel_entropy": entropy.item(),
        "diversity": diversity,
        "mean_l2": mean_l2.item()
    }


# Phase transition detection

def detect_critical_beta(beta, flip):
    beta = np.array(beta)
    flip = np.array(flip)

    if len(beta) < 5:
        return beta.mean()

    order = np.argsort(beta)

    beta = beta[order]
    flip = flip[order]

    window = 7
    kernel = np.ones(window) / window

    flip_smooth = np.convolve(flip, kernel, mode="same")

    susceptibility = -np.gradient(flip_smooth, beta)

    idx = np.argmax(susceptibility)

    return beta[idx]


# Phase diagram visualization

def plot_flip_beta(
        model,
        title,
        filename,
        fig_size=(10, 8),
        density=True
):
    beta = np.array(model.spectral_beta_hist)
    flip = np.array(model.flip_hist)

    if len(beta) < 5:
        return

    # Sort

    order = np.argsort(beta)

    beta = beta[order]
    flip = flip[order]

    # Smooth

    window = 9
    kernel = np.ones(window) / window

    flip_smooth = np.convolve(flip, kernel, mode="same")

    # Susceptibility

    susceptibility = -np.gradient(flip_smooth, beta)

    beta_c = beta[np.argmax(susceptibility)]

    # Figure

    fig, axes = plt.subplots(3, 1, figsize=fig_size)

    # Panel 1 — Phase diagram

    if density:

        hb = axes[0].hexbin(
            beta,
            flip,
            gridsize=40,
            cmap="viridis",
            mincnt=1
        )

        fig.colorbar(
            hb,
            ax=axes[0],
            label="Density",
            fraction=0.045,
            pad=0.03
        )

    else:

        axes[0].scatter(
            beta,
            flip,
            s=20,
            alpha=0.35,
            color=COLORS["flip"]
        )

    axes[0].plot(
        beta,
        flip_smooth,
        linewidth=2.5,
        color=COLORS["smooth"],
        label="Smoothed flip rate"
    )

    axes[0].axvline(
        beta_c,
        linestyle="--",
        linewidth=2,
        color="black",
        label=f"β_c ≈ {beta_c:.2f}"
    )

    axes[0].set_xlabel(r"$\beta_{\mathrm{spectral}}$")
    axes[0].set_ylabel("Flip Rate")
    axes[0].set_title(title)
    axes[0].legend()

    # Panel 2 — Smoothed flip

    axes[1].plot(
        beta,
        flip_smooth,
        linewidth=2.5,
        color=COLORS["smooth"]
    )

    axes[1].axvline(
        beta_c,
        linestyle="--",
        color="black"
    )

    axes[1].set_ylabel("Smoothed Flip")

    # Panel 3 — Susceptibility

    axes[2].plot(
        beta,
        susceptibility,
        linewidth=2.5,
        color=COLORS["sus"]
    )

    axes[2].axvline(
        beta_c,
        linestyle="--",
        color="black"
    )

    axes[2].set_xlabel(r"$\beta_{\mathrm{spectral}}$")
    axes[2].set_ylabel("Susceptibility")

    # Layout

    fig.subplots_adjust(
        left=0.12,
        right=0.96,
        hspace=0.35
    )

    plt.savefig(filename, bbox_inches="tight")

    plt.close()