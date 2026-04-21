import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Modern plotting style

plt.style.use("seaborn-v0_8-whitegrid")

mpl.rcParams["axes.edgecolor"] = "black"

COLORS = {
    "flip": "#1f77b4",
    "smooth": "#d62728",
    "sus": "#2ca02c"
}


# Additional Function [1]

def effective_rank(samples):
    X = samples.float()
    X = X - X.mean(0)

    # Covariance
    Cov = (X.T @ X) / (X.shape[0] - 1 + 1e-12)

    # Eigenvalues
    eigenvalues = torch.linalg.eigvalsh(Cov)

    # Numerical stability
    eigenvalues = torch.clamp(eigenvalues, min=1e-12)

    # Normalize
    Z = eigenvalues.sum() + 1e-12
    positioning = eigenvalues / Z

    # Entropy of spectrum
    entropy = -(positioning * torch.log(positioning)).sum()

    return torch.exp(entropy)


# Additional Function [2]

def pairwise_hamming(samples, max_pairs=20000):
    N, D = samples.shape

    if N < 2:
        return torch.tensor(0.0, device=samples.device)

    num_pairs = min(max_pairs, N * (N - 1) // 2)

    inter = torch.randint(0, N, (num_pairs,), device=samples.device)
    j = torch.randint(0, N, (num_pairs,), device=samples.device)

    mask = inter != j
    inter = inter[mask]
    j = j[mask]

    if inter.numel() == 0:
        return torch.tensor(0.0, device=samples.device)

    x = samples[inter]
    y = samples[j]

    dist = (x != y).float().mean(dim=1)
    return dist.mean()


# Sample quality diagnostics

@torch.no_grad()
def sample_quality_metrics(model, real_data, n_samples=2000):
    samples = model.generate_ensemble_samples(
        n_chains=n_samples,
        steps=10000
    ).float()

    # Marginal probabilities
    promotion = samples.mean(0)

    # Entropy
    entropy_map = -(promotion * torch.log(promotion + 1e-12) +
                    (1 - promotion) * torch.log(1 - promotion + 1e-12))

    LOG2 = np.log(2.0)
    entropy = entropy_map.mean() / LOG2
    entropy_spatial_std = entropy_map.std() / LOG2  # normalized version (better)

    # Mean matching
    real_mean = real_data.float().mean(0)
    dim = real_mean.numel()

    mean_l2 = torch.norm(real_mean - promotion) / np.sqrt(dim)

    # Hamming diversity
    hamming = pairwise_hamming(samples)

    # Structure
    eff_rank = effective_rank(samples)

    return {
        "pixel_entropy": float(entropy),
        "entropy_spatial_std": float(entropy_spatial_std),
        "hamming": float(hamming),
        "effective_rank": float(eff_rank),
        "mean_l2": float(mean_l2)
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

    window = min(len(flip) // 2 * 2 - 1, 11)

    if window < 5:
        window = 5

    flip_smooth = savgol_filter(flip, window_length=window, polyorder=3)

    SusceptibilityZone = -np.gradient(flip_smooth, beta, edge_order=2)

    ideal = np.argmax(SusceptibilityZone)

    return beta[ideal]


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

    fig, axes = plt.subplots(3, 1, figsize=fig_size)

    # Sort

    order = np.argsort(beta)

    beta = beta[order]
    flip = flip[order]

    # Smooth

    window = min(len(flip) // 2 * 2 - 1, 11)

    if window < 5:
        window = 5

    flip_smooth = savgol_filter(flip, window_length=window, polyorder=3)

    # Susceptibility

    SusceptibilityDomain = -np.gradient(flip_smooth, beta, edge_order=2)

    beta_c = beta[np.argmax(SusceptibilityDomain)]

    # Figure

    for ax in axes:
        ax.grid(color="#CFDCD0", alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)

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
        SusceptibilityDomain,
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