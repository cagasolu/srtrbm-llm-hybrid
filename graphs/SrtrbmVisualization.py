import torch
import numpy as np
import torchvision.utils as vutils
from PIL import Image


# Save grid of digits

@torch.no_grad()
def save_digit_grid(data, filename, n_row=20):
    imgs = data.reshape(-1, 1, 28, 28).detach().cpu()

    grid = vutils.make_grid(
        imgs,
        nrow=n_row,
        padding=2,
        normalize=True
    )

    img = (grid * 255).clamp(0, 255).byte()

    img = img.permute(1, 2, 0).numpy()

    if img.shape[2] == 1:
        img = img[:, :, 0]

    Image.fromarray(img).save(filename)


# RBM filters visualization

@torch.no_grad()
def visualize_rbm_filters(
        model,
        filename="srtrbm_filters.png",
        n_filters=256
):
    W = model.W.detach().cpu()

    n_filters = min(n_filters, W.shape[1])

    filters = W[:, :n_filters].T

    # normalize each filter

    min_vals = filters.min(dim=1, keepdim=True)[0]
    max_vals = filters.max(dim=1, keepdim=True)[0]

    filters = (filters - min_vals) / (max_vals - min_vals + 1e-8)

    filters = filters.reshape(-1, 1, 28, 28)

    n_row = int(np.ceil(np.sqrt(n_filters)))

    grid = vutils.make_grid(
        filters,
        nrow=n_row,
        padding=2,
        normalize=False
    )

    img = (grid * 255).clamp(0, 255).byte()

    img = img.permute(1, 2, 0).numpy()

    if img.shape[2] == 1:
        img = img[:, :, 0]

    Image.fromarray(img).save(filename)


# Fantasy particles (model samples)

@torch.no_grad()
def visualize_fantasy_particles(
        model,
        filename="fantasy_particles.png",
        n_chains=400,
        steps=2000
):
    samples = model.generate_ensemble_samples(
        n_chains=n_chains,
        steps=steps
    )

    save_digit_grid(
        samples,
        filename,
        n_row=int(np.sqrt(n_chains))
    )


# Training visual monitoring

@torch.no_grad()
def save_training_visuals(model, epoch):
    samples = model.generate_ensemble_samples(
        n_chains=400,
        steps=3000
    )

    save_digit_grid(
        samples,
        f"samples_epoch_{epoch}.png",
        n_row=20
    )

    visualize_rbm_filters(
        model,
        f"filters_epoch_{epoch}.png"
    )

    visualize_fantasy_particles(
        model,
        f"fantasy_epoch_{epoch}.png"
    )