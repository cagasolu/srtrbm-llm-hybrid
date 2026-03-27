#   Self-Regulated Thermodynamic RBM (SR-TRBM) Project Core
#
#   SPDX-License-Identifier: BSD-3-Clause
#
#   Copyright © 2026 Görkem Can Süleymanoğlu

from graphs import SrtrbmVisualization
from graphs import SrtrbmEnergy
from graphs import SrtrbmMetrics

from analysis.AutoGPU import GPUEnergyTracker
from correction.NO import Refinement

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.utils as vutils
from tqdm import tqdm
from PIL import Image
import numpy as np
import subprocess
import threading
import json
import math
import time
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

from openaiF.engine import LLMController
from openaiF.gateway import evaluate

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# Data Model

def load_mnist(device_exact):
    dataset = fetch_openml("mnist_784", version=1, cache=True)
    X_prime = dataset.data.to_numpy(dtype="float32") / 255.0
    X_prime = (X_prime > 0.5).astype("float32")

    return torch.tensor(X_prime).to(device_exact)


def susceptibility(samples):
    spins = 2 * samples - 1

    magnetization = spins.mean(dim=1)

    chi = magnetization.var(unbiased=False)

    return chi.item()


def binder_cumulant(samples):
    spins = 2 * samples - 1

    spins = spins.reshape(spins.size(0), -1)

    m = spins.mean(dim=1)

    m2 = torch.mean(m ** 2)
    m4 = torch.mean(m ** 4)

    U = 1 - m4 / (3 * (m2 ** 2) + 1e-12)

    return U.item()


# Model Point

class HybridThermodynamicRBM:

    def __init__(
            self,
            n_visible,
            n_hidden,
            device_type="cuda",
            learning_rate=5e-4,
            epochs=400,
            gibbs_steps=1,
            batch_size=128,
            lambda_gain=0.01,
            flip_smoothing=0.01,
            energy_temp_scale=0.001,
            weight_decay=1e-4,
            fixed_temperature=None
    ):

        self.device = torch.device(device_type)

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.epochs = epochs
        self.gibbs_steps = gibbs_steps
        self.batch_size = batch_size

        # Thermodynamic parameters

        self.lambda_gain = lambda_gain
        self.flip_smoothing = flip_smoothing
        self.energy_temp_scale = energy_temp_scale
        self.weight_decay = weight_decay

        self.fixed_temperature = fixed_temperature

        self.energy_count = 0

        self.bias_decay = 0.0

        # Model parameters

        self.W = torch.randn(n_visible, n_hidden, device=self.device) * 0.05
        self.visible_bias = torch.zeros(n_visible, device=self.device)
        self.hidden_bias = torch.zeros(n_hidden, device=self.device)

        # Thermodynamic states

        self.log_temperature = torch.tensor(0.0, device=self.device)
        self.flip_reference = torch.tensor(0.0, device=self.device)
        self.energy_avg = torch.tensor(0.0, device=self.device)

        self.persistent_sampling = None

        # Monitoring histories

        self.flip_hist = []
        self.c_hist = []
        self.weight_norm_hist = []
        self.beta_eff_hist = []
        self.temp_hist = []
        self.delta_w_hist = []
        self.persistent_div_hist = []
        self.T_micro_hist = []
        self.T_macro_hist = []
        self.F_data_hist = []
        self.F_model_hist = []
        self.F_gap_hist = []
        self.true_beta_hist = []
        self.spectral_beta_hist = []

    # Temperature (Hybrid Rule)

    def temperature(self):
        if self.fixed_temperature is not None:
            return torch.tensor(self.fixed_temperature, device=self.device, dtype=torch.float32)

        micro = torch.exp(self.log_temperature)
        macro = self.energy_temp_scale * self.energy_avg

        T = micro + macro

        return torch.clamp(T, min=1e-6, max=1e3)

    # Energy

    def raw_energy(self, v, h):
        return - (v @ self.W * h).sum(1) \
            - (v * self.visible_bias).sum(1) \
            - (h * self.hidden_bias).sum(1)

    def free_energy(self, v, T):
        activation = (v @ self.W + self.hidden_bias) / T
        return -(v * self.visible_bias).sum(1) / T \
            - F.softplus(activation).sum(1)

    @torch.no_grad()
    def reconstruction_accuracy(self, data):

        recon_prob = self.reconstruct(data)
        recon_bin = (recon_prob > 0.5).float()

        correct = (recon_bin == data).float().sum()
        total = data.numel()

        return (correct / total).item()

    # Gibbs Sampling

    @torch.no_grad()
    def gibbs_chain(self, v, T, steps=None):

        steps = steps if steps is not None else self.gibbs_steps

        for _ in range(steps):
            h = torch.bernoulli(torch.sigmoid((v @ self.W + self.hidden_bias) / T))
            v = torch.bernoulli(torch.sigmoid((h @ self.W.T + self.visible_bias) / T))

        return v

    # Training

    def train(self, data, energy_tracker=None):

        N = data.shape[0]

        persistent_v = torch.bernoulli(
            torch.full((self.batch_size, self.n_visible), 0.5, device=self.device)
        )

        epoch_bar = tqdm(range(self.epochs), desc="Training", leave=True)

        for _ in epoch_bar:

            W_before = self.W.clone()

            perm = torch.randperm(N, device=self.device)
            data = data[perm]

            flip_rates = []
            energy_gaps = []
            F_data_batches = []
            F_model_batches = []

            for algebraic in range(0, N, self.batch_size):

                T = self.temperature()

                v_pos = data[algebraic:algebraic + self.batch_size]
                if v_pos.shape[0] != self.batch_size:
                    continue

                # Negative phase (Persistent Contrastive Divergence - PCD-k)

                v_prev = persistent_v

                v_neg = self.gibbs_chain(v_prev, T, steps=self.gibbs_steps)

                persistent_v = v_neg

                flip = (v_neg != v_prev).float().mean()
                flip_rates.append(flip)

                # Expectations

                h_pos = torch.sigmoid((v_pos @ self.W + self.hidden_bias) / T)
                h_neg = torch.sigmoid((v_neg @ self.W + self.hidden_bias) / T)

                dW = (v_pos.T @ h_pos - v_neg.T @ h_neg) / self.batch_size
                dW -= self.weight_decay * self.W

                self.W += self.lr * dW

                # ℓ2 regularization on biases (bias decay) b_{t+1} = (1 - ηλ) b_t + η g_t

                db_v = (v_pos - v_neg).mean(0) - self.bias_decay * self.visible_bias
                db_h = (h_pos - h_neg).mean(0) - self.bias_decay * self.hidden_bias

                self.visible_bias += self.lr * db_v
                self.hidden_bias += self.lr * db_h

                # Energy gap

                F_data_batch = self.free_energy(v_pos, T).mean()
                F_model_batch = self.free_energy(v_neg, T).mean()

                energy_gaps.append(F_data_batch - F_model_batch)
                F_data_batches.append(F_data_batch)
                F_model_batches.append(F_model_batch)

                if energy_tracker is not None:
                    energy_tracker.step()

            # Epoch statistics

            flip_epoch = torch.stack(flip_rates).mean()
            energy_gap_epoch = torch.stack(energy_gaps).mean()

            F_data_epoch = torch.stack(F_data_batches).mean().item()
            F_model_epoch = torch.stack(F_model_batches).mean().item()

            # Microscopic control

            self.flip_reference = (
                    (1 - self.flip_smoothing) * self.flip_reference
                    + self.flip_smoothing * flip_epoch
            )

            error = flip_epoch - self.flip_reference

            self.log_temperature = (
                    self.log_temperature -
                    self.lambda_gain * error
            )

            # Macroscopic control (Exact Cesàro average)

            self.energy_count += 1

            self.energy_avg = self.energy_avg + (
                    energy_gap_epoch.detach() - self.energy_avg
            ) / self.energy_count

            # Diagnostics

            current_T = self.temperature()

            T_micro = torch.exp(self.log_temperature).item()
            T_macro = (self.energy_temp_scale * self.energy_avg).item()

            weight_norm = torch.norm(self.W).item()

            with torch.no_grad():
                u = torch.randn(self.n_visible, 1, device=self.device)
                u = u / (torch.norm(u) + 1e-8)

                for _ in range(5):
                    v = torch.matmul(self.W.T, u)
                    v = v / (torch.norm(v) + 1e-8)
                    u = torch.matmul(self.W, v)
                    u = u / (torch.norm(u) + 1e-8)

                spectral_norm = torch.norm(torch.matmul(self.W, v)).item()

            beta_eff = weight_norm / current_T.item()

            spectral_beta = spectral_norm / current_T.item()

            true_beta = 1.0 / current_T.item()

            delta_w = torch.norm(self.W - W_before).item()

            # Store histories

            self.flip_hist.append(flip_epoch.item())
            self.c_hist.append(self.flip_reference.item())
            self.weight_norm_hist.append(weight_norm)
            self.beta_eff_hist.append(beta_eff)
            self.temp_hist.append(current_T.item())
            self.delta_w_hist.append(delta_w)
            self.T_micro_hist.append(T_micro)
            self.T_macro_hist.append(T_macro)
            self.F_data_hist.append(F_data_epoch)
            self.F_model_hist.append(F_model_epoch)
            self.true_beta_hist.append(true_beta)
            self.spectral_beta_hist.append(spectral_beta)
            self.F_gap_hist.append(F_data_epoch - F_model_epoch)

            # Progress bar

            epoch_bar.set_postfix({
                "T": f"{current_T.item():.3f}",
                "flip": f"{flip_epoch.item():.3f}",
                "beta": f"{beta_eff:.3f}"
            })

            persistent_div = torch.var(persistent_v.float())
            self.persistent_div_hist.append(persistent_div.item())

    # Sampling

    @torch.no_grad()
    def generate_ensemble_samples(
            self,
            n_chains=32,
            steps=6000,
            temp_noise=0.01,
            final_cool_fraction=0.4,
            energy_tracker=None,
    ):
        """
        Multichain Gibbs sampler with annealed stochastic temperature.
        """

        device = self.device

        T_base = self.temperature().item()

        final_cool_fraction = float(min(max(final_cool_fraction, 0.0), 1.0))

        if (
                hasattr(self, "persistent_sampling")
                and self.persistent_sampling is not None
                and self.persistent_sampling.shape == (n_chains, self.n_visible)
        ):
            v = self.persistent_sampling.clone()
        else:
            # Otherwise initialize chains from uniform Bernoulli noise

            v = torch.bernoulli(
                torch.rand(n_chains, self.n_visible, device=device)
            )

        # Determine when deterministic cooling begins

        final_cool_start = int((1.0 - final_cool_fraction) * steps)

        for t in range(steps):

            # Temperature schedule:

            if t < final_cool_start and final_cool_start > 0:
                anneal_factor = 1.0 - (t / final_cool_start)
                noise_scale = temp_noise * anneal_factor

                T = T_base * torch.exp(
                    noise_scale * torch.randn(n_chains, 1, device=device)
                )
            else:
                T = T_base

            h = torch.bernoulli(
                torch.sigmoid((v @ self.W + self.hidden_bias) / T)
            )

            for _ in range(1):
                h = torch.bernoulli(
                    torch.sigmoid((v @ self.W + self.hidden_bias) / T)
                )

            v = torch.bernoulli(
                torch.sigmoid((h @ self.W.T + self.visible_bias) / T)
            )

            if energy_tracker is not None:
                energy_tracker.step()

        # Store final state for persistent continuation

        self.persistent_sampling = v.detach()

        return v

    @torch.no_grad()
    def save_ensemble_samples(
            self,
            filename="samples_ensemble.png",
            n_display=100,
            steps=6000,
            energy_tracker=None
    ):
        """
        Generates ensemble samples and saves a grid.
        """
        samples = self.generate_ensemble_samples(
            n_chains=n_display,
            steps=steps,
            energy_tracker=energy_tracker
        )

        samples_original = samples.clone().detach()

        symbol_samples = samples[:2].clone().detach()

        samples = Refinement.exact_energy_refinement(self, samples, verbose=True)
        samples = Refinement.pixel_connectivity_refine(samples)

        mean_forward = samples.mean(dim=1, keepdim=True)
        std_aqua = samples.std(dim=1, keepdim=True)

        threshold_mean = mean_forward + 0.5 * std_aqua * torch.tanh(std_aqua * 2.0)

        samples = (samples > threshold_mean).float()
        samples = samples.view(-1, 1, 28, 28)
        samples_refined = samples.clone()

        chi = susceptibility(samples_refined)
        binder = binder_cumulant(samples_refined)

        print("\nPhysical Refinement Diagnostics")
        print("Primary magnetic susceptibility:", chi)
        print("Binder cumulant (1):", binder)

        grid = vutils.make_grid(
            samples_refined,
            nrow=50,
            padding=2
        )

        grid = (grid * 255).clamp(0, 255).byte()
        nd_arr = grid.permute(1, 2, 0).cpu().numpy()

        Image.fromarray(nd_arr).save(
            filename.replace(".png", "_refined.png")
        )

        samples_original = samples_original.view(-1, 1, 28, 28)

        grid_0 = vutils.make_grid(
            samples_original,
            nrow=50,
            padding=2
        )

        grid_0 = (grid_0 * 255).clamp(0, 255).byte()
        nd_arr1 = grid_0.permute(1, 2, 0).cpu().numpy()

        Image.fromarray(nd_arr1).save(
            filename.replace(".png", "_symbol.png")
        )

    @torch.no_grad()
    def ensemble_diagnostics(
            self,
            n_chains=16,
            steps=6000,
            thinning=5,
            max_lag=200
    ):
        """
        Chain-wise autocorrelation analysis using free energy.
        Provides robust ESS and integrated autocorrelation time.
        """

        device = self.device
        T = self.temperature()

        v = torch.bernoulli(
            torch.full((n_chains, self.n_visible), 0.5, device=device)
        )

        energies = []

        for t in range(steps):

            h = torch.bernoulli(
                torch.sigmoid((v @ self.W + self.hidden_bias) / T)
            )

            v = torch.bernoulli(
                torch.sigmoid((h @ self.W.T + self.visible_bias) / T)
            )

            if t % thinning == 0:
                Fv = self.free_energy(v, T)  # shape: (n_chains,)
                energies.append(Fv.detach())

        energies = torch.stack(energies)  # (time, chains)

        taus = []

        for c in range(n_chains):

            series = energies[:, c]
            series = series - series.mean()

            var = torch.var(series, unbiased=False)

            autocorr = []

            for lag in range(max_lag):
                if lag == 0:
                    autocorr.append(1.0)
                else:
                    if lag >= len(series):
                        break

                    c_val = torch.dot(
                        series[:-lag],
                        series[lag:]
                    ) / (len(series) - lag)

                    autocorr.append((c_val / var).item())

            tau_int = 1.0
            for k in range(1, len(autocorr)):
                if autocorr[k] <= 0:
                    break
                tau_int += 2.0 * autocorr[k]

            taus.append(tau_int)

        tau_int_mean = float(np.mean(taus))
        ess = len(energies) * n_chains / tau_int_mean

        return {
            "tau_int": tau_int_mean,
            "ess": ess
        }

    @torch.no_grad()
    def save_professional_samples(
            self,
            filename="samples_professional.png",
            n_display=100,
            steps=6000,
            energy_tracker=None
    ):
        """
        Generates high-quality samples and returns robust diagnostics.
        """

        samples = self.generate_ensemble_samples(
            n_chains=n_display,
            steps=steps,
            energy_tracker=energy_tracker
        )

        samples_prof_original = samples.clone().detach()

        symbol_samples = samples[:2].clone().detach()

        samples = Refinement.exact_energy_refinement(self, samples)
        samples = Refinement.pixel_connectivity_refine(samples)

        mean_forward_pro = samples.mean(dim=1, keepdim=True)
        std_aqua_pro = samples.std(dim=1, keepdim=True)

        threshold_proMean = mean_forward_pro + 0.5 * std_aqua_pro * torch.tanh(std_aqua_pro * 2.0)

        samples = (samples > threshold_proMean).float()
        samples = samples.view(-1, 1, 28, 28)

        diagnostics = self.ensemble_diagnostics()

        samples_refined = samples.clone()

        chi = susceptibility(samples_refined)
        binder = binder_cumulant(samples_refined)

        print("\nSecondary magnetic susceptibility:", chi)
        print("Binder cumulant (2):", binder)

        grid = vutils.make_grid(
            samples_refined,
            nrow=50,
            padding=2
        )

        grid = (grid * 255).clamp(0, 255).byte()
        nd_arr = grid.permute(1, 2, 0).cpu().numpy()

        Image.fromarray(nd_arr).save(
            filename.replace(".png", "_refined.png")
        )

        samples_prof_original = samples_prof_original.view(-1, 1, 28, 28)

        grid1 = vutils.make_grid(
            samples_prof_original,
            nrow=50,
            padding=2
        )

        grid1 = (grid1 * 255).clamp(0, 255).byte()
        nd_arr2 = grid1.permute(1, 2, 0).cpu().numpy()

        Image.fromarray(nd_arr2).save(
            filename.replace(".png", "_symbol.png")
        )

        return diagnostics

    # Ais Calculation

    @torch.no_grad()
    def ais_log_partition(self, n_runs=1000, n_intermediate=2000, energy_tracker=None):

        device = self.device
        T = self.temperature().item()

        W = self.W
        bv = self.visible_bias
        bh = self.hidden_bias

        nv, nh = self.n_visible, self.n_hidden

        betas = torch.linspace(0.0, 1.0, n_intermediate, device=device)

        # Base model: uniform Bernoulli(0.5)

        logZ0 = (nv + nh) * math.log(2.0)

        v = torch.bernoulli(torch.full((n_runs, nv), 0.5, device=device))
        h = torch.bernoulli(torch.full((n_runs, nh), 0.5, device=device))

        log_weights = torch.zeros(n_runs, device=device)

        for k in range(1, n_intermediate):
            beta_prev = betas[k - 1]
            beta_curr = betas[k]

            energy = (- (v @ W * h).sum(1)
                      - (v * bv).sum(1)
                      - (h * bh).sum(1)
                      ) / T  # SINGLE temperature scaling point

            log_weights += (beta_curr - beta_prev) * energy

            h_prob = torch.sigmoid(beta_curr * (v @ W + bh) / T)
            h = torch.bernoulli(h_prob)

            v_prob = torch.sigmoid(beta_curr * (h @ W.T + bv) / T)
            v = torch.bernoulli(v_prob)

            if energy_tracker is not None:
                energy_tracker.step()

        logZ = logZ0 + torch.logsumexp(log_weights, dim=0) - math.log(n_runs)

        log_w = log_weights - torch.max(log_weights).detach()

        w = torch.exp(log_w)
        w = w / torch.sum(w)
        ess = 1.0 / torch.sum(w ** 2)

        log_weight_var = torch.var(log_weights)

        return logZ.item(), log_weight_var.item(), ess.item()

    # Log-Likelihood

    @torch.no_grad()
    def log_likelihood(self, data, log_Z_2ox):
        T = self.temperature()
        Fv = self.free_energy(data, T)
        return (-Fv - log_Z_2ox).mean().item()

    @torch.no_grad()
    def pseudo_likelihood(self, data):

        T = self.temperature()

        N, D = data.shape
        id_Xx = torch.randint(0, D, (N,), device=self.device)

        v_flip = data.clone()
        v_flip[torch.arange(N), id_Xx] = 1 - v_flip[torch.arange(N), id_Xx]

        F_v = self.free_energy(data, T)
        F_flip = self.free_energy(v_flip, T)

        return torch.mean(
            -torch.logaddexp(
                torch.zeros_like(F_v),
                F_flip - F_v
            )
        ).item()

    @torch.no_grad()
    def reconstruct(self, v):

        T = self.temperature()

        h = torch.sigmoid((v @ self.W + self.hidden_bias) / T)
        v_recon = torch.sigmoid((h @ self.W.T + self.visible_bias) / T)

        return v_recon


# Multi-GPU Worker

def worker(Gpu_Id, seed_round, consequences):
    torch.cuda.set_device(Gpu_Id)
    torch.manual_seed(seed_round)
    torch.cuda.manual_seed_all(seed_round)
    energy_tracker = GPUEnergyTracker(Gpu_Id)

    device_warm = torch.device(f"cuda:{Gpu_Id}")

    # Data

    data = load_mnist(device_warm)

    train_data = data[:60000]
    test_data = data[60000:]

    last_model = HybridThermodynamicRBM(
        n_visible=784,
        n_hidden=512,
        device_type=f"cuda:{Gpu_Id}",
        fixed_temperature=None
    )

    controller = LLMController()

    last_model.train(train_data, energy_tracker)

    def dots_spinner(stop_event_choose, label):
        """
        Lightweight terminal spinner used to indicate that a diagnostic
        computation is currently running.
        """
        frames = ["   ", ".  ", ".. ", "..."]

        i = 0

        while not stop_event_choose.is_set():
            sys.stdout.write(f"\r▶ {label}{frames[i % len(frames)]}")
            sys.stdout.flush()
            time.sleep(0.4)

            i += 1

    print(f"\n[GPU {Gpu_Id}] Running AIS...")

    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=dots_spinner, args=(stop_event, "AIS"))
    spinner_thread.start()

    ais_start = time.time()

    try:
        log_Z_3ox, ais_var, ais_ess = last_model.ais_log_partition(
            n_runs=3900,
            n_intermediate=7200,
            energy_tracker=energy_tracker
        )
    except Exception as e:
        print(f"\n✖ AIS failed: {e}")

        log_Z_3ox, ais_var, ais_ess = float("nan"), float("nan"), float("nan")

    ais_duration = time.time() - ais_start

    stop_event.set()
    spinner_thread.join()

    sys.stdout.write(f"\r✔ AIS finished in {ais_duration:.2f} seconds\n")

    train_ll = last_model.log_likelihood(train_data, log_Z_3ox)
    Test_ll = last_model.log_likelihood(test_data, log_Z_3ox)
    Train_pl = last_model.pseudo_likelihood(train_data)
    Test_pl = last_model.pseudo_likelihood(test_data)

    # Reconstruction

    reconstruction = last_model.reconstruct(test_data)
    reconstruction_mse = torch.mean(
        (test_data - reconstruction) ** 2
    ).item()

    recon_acc = last_model.reconstruction_accuracy(test_data)

    # Thermodynamic diagnostics

    final_temperature = last_model.temperature().item()
    weight_norm = torch.norm(last_model.W).item()

    effective_beta = weight_norm / final_temperature

    sample_filename = f"samples_gpu{Gpu_Id}_seed{seed_round}.png"
    prof_filename = f"samples_prof_gpu{Gpu_Id}_seed{seed_round}.png"

    # Basit ensemble save

    last_model.save_ensemble_samples(
        filename=sample_filename,
        n_display=1200,
        steps=6000,
        energy_tracker=energy_tracker
    )

    # Professional ensemble + diagnostics

    diagnostics = last_model.save_professional_samples(
        filename=prof_filename,
        n_display=1200,
        steps=6000,
        energy_tracker=energy_tracker
    )

    print("\nRunning energy analysis ↓\n")

    analysis_steps = [

        ("Energy distribution",
         lambda: SrtrbmEnergy.plot_data_vs_model_energy(last_model, train_data)),

        ("Energy landscape extremes",
         lambda: SrtrbmEnergy.visualize_energy_extremes(last_model, train_data)),

        ("Phase diagram",
         lambda: SrtrbmMetrics.plot_flip_beta(
             last_model,
             "SR-TRBM Phase Diagram",
             filename="srtrbm_phase_diagram.pdf"
         )),

        ("RBM filters",
         lambda: SrtrbmVisualization.visualize_rbm_filters(
             last_model,
             filename="srtrbm_filters.png",
             n_filters=256
         )),

        ("Sample quality metrics",
         lambda: SrtrbmMetrics.sample_quality_metrics(last_model, train_data)),
    ]

    quality = None

    energy_stats = None

    print("\nRunning diagnostics with the following calculations ↙\n")

    total_start = time.time()

    for name, fn in analysis_steps:

        stop_event = threading.Event()
        spinner_thread = threading.Thread(target=dots_spinner, args=(stop_event, name))
        spinner_thread.start()

        start = time.time()

        try:
            result_cache = fn()
        except Exception as e:

            print(f"\n✖ {name} failed:", e)

            result_cache = None

        duration = time.time() - start

        stop_event.set()
        spinner_thread.join()

        sys.stdout.write(f"\r✔ {name} finished in {duration:.2f} seconds\n")

        if name == "Energy distribution" and result_cache is not None:
            energy_stats = result_cache
        if name == "Sample quality metrics":
            quality = result_cache

    total_time = time.time() - total_start

    print(f"\nDiagnostics completed in {total_time:.2f} seconds\n")

    if quality is None:
        quality = {
            "pixel_entropy": float("nan"),
            "diversity": float("nan"),
            "mean_l2": float("nan")
        }

    if energy_stats is None:
        energy_stats = {
            "mean_data_energy": float("nan"),
            "mean_model_energy": float("nan"),
            "energy_gap": float("nan")
        }

    gpu_energy = energy_tracker.total_energy()

    print(f"GPU Energy Used : {gpu_energy:.2f} Joules\n")

    consequences.append({
        "seed": seed_round,
        "gpu": Gpu_Id,
        "gpu_energy": gpu_energy,
        "temperature": final_temperature,
        "weight_norm": weight_norm,
        "beta_effective": effective_beta,
        "train_log_likelihood": train_ll,
        "test_log_likelihood": Test_ll,
        "train_pseudo_likelihood": Train_pl,
        "test_pseudo_likelihood": Test_pl,
        "reconstruction_mse": reconstruction_mse,
        "reconstruction_accuracy": recon_acc,
        "mean_data_energy": energy_stats["mean_data_energy"],
        "mean_model_energy": energy_stats["mean_model_energy"],
        "energy_gap": energy_stats["energy_gap"],
        "logZ": log_Z_3ox,
        "ais_ess": ais_ess,
        "pixel_entropy": quality["pixel_entropy"],
        "diversity": quality["diversity"],
        "mean_l2": quality["mean_l2"],
        "ais_log_weight_variance": ais_var,
        "mcmc_tau_int": diagnostics["tau_int"],
        "mcmc_ess": diagnostics["ess"],
        "prof_sample_file": prof_filename,
        "sample_file": sample_filename
    })

    plt.figure(figsize=(10, 12))

    # Flip rate

    plt.subplot(3, 1, 1)
    plt.plot(last_model.flip_hist, linewidth=2, label="Flip rate")
    plt.plot(last_model.c_hist, linestyle="--", linewidth=2, label="Adaptive reference")
    plt.title("Microscopic Flip Dynamics")
    plt.ylabel("Flip Rate")
    plt.grid(alpha=0.3)
    plt.legend()

    # Temperature evolution

    plt.subplot(3, 1, 2)
    plt.plot(last_model.temp_hist, linewidth=2)
    plt.title(r"Global Temperature $T$")
    plt.ylabel("Temperature")
    plt.grid(alpha=0.3)

    # Micro vs Macro

    plt.subplot(3, 1, 3)
    plt.plot(last_model.T_micro_hist, linewidth=2, label=r"$T_{micro}$")
    plt.plot(last_model.T_macro_hist, linewidth=2, label=r"$T_{macro}$")
    plt.title("Micro–Macro Temperature Decomposition")
    plt.xlabel("Epoch")
    plt.ylabel("Temperature Components")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("srtrbm_core_dynamics.pdf", bbox_inches="tight")
    plt.close()

    # Energy & Spectral Diagnostics

    plt.figure(figsize=(10, 14))

    # Weight norm

    plt.subplot(4, 1, 1)
    plt.plot(last_model.weight_norm_hist, linewidth=2)
    plt.title("Weight Norm Evolution")
    plt.ylabel(r"$||W||$")
    plt.grid(alpha=0.3)

    # Effective beta

    plt.subplot(4, 1, 2)
    plt.plot(last_model.beta_eff_hist, linewidth=2)
    plt.title(r"Effective Inverse Temperature $\beta_{\mathrm{eff}}$")
    plt.ylabel(r"$\beta_{\mathrm{eff}}$")
    plt.grid(alpha=0.3)

    # Free energy comparison

    plt.subplot(4, 1, 3)
    plt.plot(last_model.F_data_hist, linewidth=2, label=r"$F_{data}$")
    plt.plot(last_model.F_model_hist, linewidth=2, label=r"$F_{model}$")
    plt.title("Free Energy: Data vs Model")
    plt.ylabel("Free Energy")
    plt.grid(alpha=0.3)
    plt.legend()

    # Spectral beta

    plt.subplot(4, 1, 4)
    plt.plot(last_model.spectral_beta_hist)

    plt.axhline(1.0, linestyle="--")

    plt.title(r"Spectral Inverse Temperature $\beta_{spectral}$")
    plt.xlabel("Epoch")
    plt.ylabel(r"$\beta_{spectral}$")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("srtrbm_energy_diagnostics.pdf", bbox_inches="tight")
    plt.close()

    sample_refined = sample_filename.replace(".png", "_refined.png")
    prof_refined = prof_filename.replace(".png", "_refined.png")

    samples_for_phase = last_model.generate_ensemble_samples(n_chains=256, steps=2000)
    samples_for_phase = samples_for_phase.view(-1, 1, 28, 28)

    chi = susceptibility(samples_for_phase)

    llm_metrics = {
        "temperature": float(final_temperature),
        "beta_effective": float(effective_beta),
        "energy_gap": float(energy_stats["energy_gap"]),

        "quality": {
            "diversity": float(quality["diversity"]),
            "entropy": float(quality["pixel_entropy"])
        },

        "sampling": {
            "mcmc_tau_int": float(diagnostics["tau_int"])
        },

        "trend": {
            "temp_slope": float(last_model.temp_hist[-1] - last_model.temp_hist[-10]),
            "beta_slope": float(last_model.beta_eff_hist[-1] - last_model.beta_eff_hist[-10])
        },

        "phase": {
            "susceptibility": float(chi)
        }
    }

    subprocess.run([
        "python3",
        "supplement/cluster.py",
        sample_refined,
        sample_refined.replace("_refined.png", "_perfect.png")
    ])

    subprocess.run([
        "python3",
        "supplement/cluster.py",
        prof_refined,
        prof_refined.replace("_refined.png", "_perfect.png")
    ])

    prof_perfect = prof_refined.replace("_refined.png", "_perfect.png")

    output_json = f"llm_output_gpu{Gpu_Id}_seed{seed_round}.json"

    result = evaluate(
        llm_metrics,
        prof_refined,
        prof_perfect
    )

    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[LLM OUTPUT SAVED] → {output_json}")

    print("\n[LLM RESULT]")
    print(json.dumps(result, indent=2))

    if result.get("confidence", 0.0) > 0.7:
        controller.apply(last_model, result, llm_metrics)
    else:
        print("[CONTROL] Skipped (low confidence)")


# Main Section

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    available_gpus = torch.cuda.device_count()

    seeds = [1]

    if available_gpus < len(seeds):
        raise RuntimeError("Not enough GPUs available.")

    manager = mp.Manager()
    results = manager.list()

    processes = []

    for gpu_id, seed in enumerate(seeds):
        p = mp.Process(
            target=worker,
            args=(gpu_id, seed, results)
        )

        p.start()

        processes.append(p)

    for p in processes:
        p.join()

    print("Hybrid Thermodynamic RBM Final Results")

    for result in list(results):
        print(f"Seed: {result['seed']} | GPU: {result['gpu']}")
        print(f"Final Temperature        : {result['temperature']:.6f}")
        print(f"Weight Norm              : {result['weight_norm']:.6f}")
        print(f"Effective Beta           : {result['beta_effective']:.6f}")
        print(f"Train Log-Likelihood     : {result['train_log_likelihood']:.6f}")
        print(f"Test Log-Likelihood      : {result['test_log_likelihood']:.6f}")
        print(f"Train Pseudo-Likelihood  : {result['train_pseudo_likelihood']:.6f}")
        print(f"Test Pseudo-Likelihood   : {result['test_pseudo_likelihood']:.6f}")
        print(f"Reconstruction MSE       : {result['reconstruction_mse']:.6f}")
        print(f"Reconstruction Accuracy  : {result['reconstruction_accuracy']:.6f}")
        print(f"Mean Data Energy         : {result['mean_data_energy']:.6f}")
        print(f"Mean Model Energy        : {result['mean_model_energy']:.6f}")
        print(f"Energy Gap               : {result['energy_gap']:.6f}")
        print(f"GPU Energy Used          : {result['gpu_energy']:.2f} J")
        print(f"Log Partition (AIS)      : {result['logZ']:.6f}")
        print(f"AIS ESS                  : {result['ais_ess']:.2f}")
        print(f"AIS Log-Weight Variance  : {result['ais_log_weight_variance']:.6f}")
        print(f"MCMC tau_int             : {result['mcmc_tau_int']:.2f}")
        print(f"MCMC ESS                 : {result['mcmc_ess']:.2f}")
        print(f"Pixel Entropy            : {result['pixel_entropy']:.6f}")
        print(f"Sample Diversity         : {result['diversity']:.6f}")
        print(f"Mean Distribution L2     : {result['mean_l2']:.6f}")
        print(f"Professional Samples File: {result['prof_sample_file']}")
        print(f"Generated Samples File   : {result['sample_file']}")

        print()

# Self-Regulated Thermodynamic RBM (SR-TRBM) with Endogenous Micro–Macro Temperature Dynamics