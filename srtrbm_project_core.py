#   Self-Regulated Thermodynamic RBM (SR-TRBM)
#
#   SPDX-License-Identifier: BSD-3-Clause
#
#   Copyright © 2026 Görkem Can Süleymanoğlu
#
#   This implementation realizes a thermodynamically self-regulated
#   energy-based model operating in a finite-time stochastic regime.
#   The framework extends classical Restricted Boltzmann Machines by
#   introducing endogenous control of sampling dynamics and bounded
#   external energy corrections.
#
#   Training is performed using Persistent Contrastive Divergence
#   (PCD-k), where the negative phase is approximated by short-run
#   Gibbs chains. To use conductance collapse and Gibbs freezing,
#   the model employs a hybrid temperature mechanism:
#
#       T = T_micro + T_macro
#
#   The microscopic component follows a feedback control law based
#   on the empirical flip rate r_t:
#
#       λ_{t+1} = λ_t - η (r_t - c_t)
#
#   where c_t is an exponentially smoothed reference, inducing
#   closed-loop stabilization of stochastic transition intensity.
#
#   The model incorporates a localized LLM-guided refinement mechanism
#   for proposals within an uncertainty band (ΔE ∈ [0, 2]).
#
#   LLM contributions are normalized via a running variance estimate:
#       σ² ← EMA(ΔE_model²),   ΔE_llm ← ΔE_llm / σ
#
#   Designed for CUDA-enabled GPU execution with fixed seeds.
#   Average runtime per seed: ~20–30 minutes on RTX-class GPUs.
#   AIS estimation and multichain Gibbs sampling are dominant costs.
#
#   Thermal monitoring is recommended (e.g., nvidia-smi).
#   Multi-seed runs should be executed sequentially.
#
#   Full license text is provided in the root LICENSE file.

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
import textwrap
import json
import math
import time
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

from openaiF.hook import LLMEnergy, to_sparse_gpu, LIES_gpu
from openaiF.gateway import Evaluate, ANASIS
from openaiF.client import SafeOpenAIClient

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
    dataSet = fetch_openml("mnist_784", version=1, cache=True)
    X_prime = dataSet.data.to_numpy(dtype="float32") / 255.0
    X_prime = (X_prime > -0.0).astype("float32")

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
        self.v_bias = None
        self.h_bias = None

        self.lag_step = 8

        self.dictionary = {}

        self.analysis_history = []

        self.abort_counter = getattr(self, "abort_counter", 0)

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

        self.command = 1.0
        self.bias_decay = 0.0
        self.energy_count = 0
        self.lambda_llm = 0.7
        self.llm_scale = 1.0

        self.llm_scale_initialized = False

        # Model parameters

        self.coefficient_zeta = 0.05

        self.W = torch.randn(n_visible, n_hidden, device=self.device) * self.coefficient_zeta

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

        return torch.clamp(T, min=1e-12, max=1e3)

    # Energy

    def free_energy(self, v, T):
        activation = (v @ self.W + self.hidden_bias) / T
        return -(v * self.visible_bias).sum(1) / T \
            - F.softplus(activation).sum(1)

    @torch.no_grad()
    def reconstruction_accuracy(self, data):

        recon_prob = self.reconstruct(data)
        recon_bin = (recon_prob > 0.4).float()

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

            v_neg_all = []
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

                v_neg_all.append(v_neg)

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

            self.log_temperature = self.command * (
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
                u_vec = torch.randn(self.n_visible, 1, device=self.device)
                u_vec = u_vec / (torch.norm(u_vec) + 1e-12)

                for _ in range(self.gibbs_steps * 2):
                    v_vec = torch.matmul(self.W.T, u_vec)
                    v_vec = v_vec / (torch.norm(v_vec) + 1e-12)
                    u_vec = torch.matmul(self.W, v_vec)
                    u_vec = u_vec / (torch.norm(u_vec) + 1e-12)

                spectral_norm = torch.norm(torch.matmul(self.W, v_vec)).item()

            v_neg_all = torch.cat(v_neg_all, dim=0)

            activations = (v_neg_all @ self.W + self.hidden_bias) / current_T
            beta_eff = activations.std(dim=1).mean().item()

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

        return None

    # Sampling

    @torch.no_grad()
    def generate_ensemble_samples(
            self,
            n_chains=2000,
            steps=10000,
            energy_tracker=None,
    ):
        """
        Multichain Gibbs sampler with STABLE temperature (debug version).
        """

        device = self.device

        T_base = self.temperature().item()
        T = T_base

        # Persistent init
        if (
                hasattr(self, "persistent_sampling")
                and self.persistent_sampling is not None
                and self.persistent_sampling.shape == (n_chains, self.n_visible)
        ):
            v = self.persistent_sampling
        else:
            v = torch.bernoulli(
                torch.rand(n_chains, self.n_visible, device=device)
            )

        for _ in range(steps):
            # h sample
            h = torch.bernoulli(
                torch.sigmoid((v @ self.W + self.hidden_bias) / T)
            )

            # additional Gibbs steps
            for _ in range(self.gibbs_steps):
                v = torch.bernoulli(
                    torch.sigmoid((h @ self.W.T + self.visible_bias) / T)
                )
                h = torch.bernoulli(
                    torch.sigmoid((v @ self.W + self.hidden_bias) / T)
                )

            # final v update
            v = torch.bernoulli(
                torch.sigmoid((h @ self.W.T + self.visible_bias) / T)
            )

            if energy_tracker is not None:
                energy_tracker.step()

        return v

    @torch.no_grad()
    def llm_uncertainty_refinement(self, state, client=None):
        """
        LLM-guided refinement with physically consistent energy coupling.

        The refinement operates on locally uncertain proposals determined by the 
        model. Only proposals with ΔE_model in the interval [0, 2] are considered, 
        which corresponds to the uncertainty region of the energy landscape.

        The language model contributes as a bounded perturbation to the energy.
        Its influence is normalized and constrained relative to the model energy,
        ensuring that it cannot dominate the underlying distribution but can 
        still provide meaningful local corrections.

        The same bit-flip is used during both selection and proposal 
        stages to preserve consistency in the transition dynamics.
        """

        batch_size, dim = state.shape
        temperature = self.temperature()

        debug = getattr(self, "debug", True)

        if debug:
            print(f"\n[LLM] Start | B={batch_size} | T={temperature:.4f}", flush=True)

        refined_state = state.clone()

        llm_cache = {}

        def get_llm_energy(repr_str):
            if repr_str in llm_cache:
                return llm_cache[repr_str]

            consequence, error = LLMEnergy(repr_str, client)
            llm_cache[repr_str] = (consequence, error)

            return consequence, error

        # Generate single-bit proposals
        flip_indices = torch.randint(0, dim, (batch_size,), device=state.device)

        state_prop = state.clone()
        state_prop[torch.arange(batch_size), flip_indices] = \
            1 - state_prop[torch.arange(batch_size), flip_indices]

        # Signed energy difference (no absolute value)
        E_cur = self.free_energy(state, temperature)
        E_prop = self.free_energy(state_prop, temperature)
        delta_E = E_prop - E_cur

        # Select uncertain samples where ΔE ∈ [0, 2]
        low_num, high_num = 0.0, 2.0
        center = 1.0

        mask = (delta_E >= low_num) & (delta_E <= high_num)
        band_indices = torch.nonzero(mask).squeeze(1)

        # Adaptive selection size using effective sample size (ESS)
        weights = torch.softmax(-E_cur / (temperature + 1e-12), dim=0)

        ess = 1.0 / torch.sum(weights ** 2)
        ess_ratio = ess.item() / batch_size

        # Presence of 'k' (type: fixed)
        k = int(batch_size * (1.0 - ess_ratio))

        # Settle the 'k' norm
        k = max(2, min(batch_size, round(k / 1)))

        # Selection (band-first)
        if band_indices.numel() >= k:
            band_delta = delta_E[band_indices]
            _, order = torch.sort(torch.abs(band_delta - center))
            selected_indices = band_indices[order[:k]]
        else:
            if band_indices.numel() > 0:
                selected_indices = band_indices
            else:
                all_dist = torch.abs(delta_E - center)
                _, order = torch.sort(all_dist)
                selected_indices = order[:k]

        selected_flips = flip_indices[selected_indices]

        if debug:
            print(
                f"[Debug] ESS={ess:.2f} | k={k} | "
                f"deltaE_band=[{low_num:.2f},{high_num:.2f}] | "
                f"in_band={band_indices.numel()} | selected={selected_indices.numel()}"
            )

        accepted_count = 0
        llm_usage_count = 0

        for integer, ideal in enumerate(selected_indices):

            current_sample = refined_state[ideal]
            proposed_sample = current_sample.clone()

            flip_idx = selected_flips[integer].item()
            proposed_sample[flip_idx] = 1 - proposed_sample[flip_idx]

            E_model_cur = self.free_energy(current_sample.unsqueeze(0), temperature)[0]
            E_model_prop = self.free_energy(proposed_sample.unsqueeze(0), temperature)[0]

            delta_model = E_model_prop - E_model_cur
            delta_total = delta_model

            if client is not None:

                current_repr = to_sparse_gpu(current_sample.view(28, 28))
                proposed_repr = to_sparse_gpu(proposed_sample.view(28, 28))

                current_llm, err_cur = get_llm_energy(current_repr)
                proposed_llm, err_prop = get_llm_energy(proposed_repr)

                if not (err_cur or err_prop or current_llm is None or proposed_llm is None):

                    llm_usage_count += 1

                    probs_cur = current_llm["probs"]
                    probs_prop = proposed_llm["probs"]

                    E_llm_cur = LIES_gpu(probs_cur)
                    E_llm_prop = LIES_gpu(probs_prop)

                    delta_llm = torch.tensor(
                        E_llm_prop - E_llm_cur,
                        device=temperature.device
                    )

                    # Normalize relative to model energy scale
                    scale_est = torch.abs(delta_model.detach()) + 1e-12

                    if not self.llm_scale_initialized:
                        self.llm_scale = scale_est ** 2
                        self.llm_scale_initialized = True
                    else:
                        self.llm_scale = 0.97 * self.llm_scale + 0.03 * (delta_model ** 2)

                    scale = torch.sqrt(self.llm_scale + 1e-12)

                    delta_llm = delta_llm / scale

                    # consistent bound
                    bound = torch.abs(delta_model.detach()) / scale + 1e-12
                    delta_llm = torch.clamp(delta_llm, -bound, bound)
                    delta_total = delta_model + self.lambda_llm * delta_llm

                    if debug:
                        print(
                            f"[Debug] d_model={delta_model.item():.4f}, "
                            f"d_llm={delta_llm.item():.4f}"
                        )

            # Numerical stability for exponentiation
            delta_total = torch.clamp(delta_total, -10.0, 10.0)

            accept_prob = torch.minimum(
                torch.tensor(1.0, device=delta_total.device),
                torch.exp(-delta_total)
            )

            rand_val = torch.rand(1, device=temperature.device)

            if rand_val < accept_prob:
                refined_state[ideal] = proposed_sample
                accepted_count += 1
                accepted = True
            else:
                accepted = False

            if debug:
                status = "ACCEPT" if accepted else "REJECT"
                print(
                    f"[Step] Sample={ideal} | {status} | "
                    f"d_model={delta_model.item():.3f} | "
                    f"d_total={delta_total.item():.3f} | "
                    f"p={accept_prob.item():.3f}"
                )

        if debug:
            print("\n[LLM Summary]")
            print(f"LLM used  : {llm_usage_count}/{batch_size}")
            print(f"Accepted  : {accepted_count}/{k}")
            print(f"T         : {temperature:.4f}")
            print("[LLM Done]\n", flush=True)

        return refined_state

    @torch.no_grad()
    def save_ensemble_samples(
            self,
            filename="samples_ensemble.png",
            n_display=100,
            steps=10000,
            energy_tracker=None,
            client=None
    ):
        samples = self.generate_ensemble_samples(
            n_chains=n_display,
            steps=steps,
            energy_tracker=energy_tracker
        )

        samples_original = samples.clone().detach()

        samples = self.llm_uncertainty_refinement(samples, client=client)

        refiner = Refinement(self)

        samples = refiner.myra_refine(samples)

        samples = samples.view(-1, 1, 28, 28)
        original_tmp = samples_original.view(-1, 1, 28, 28)

        chi_final = susceptibility(samples)
        chi_raw = susceptibility(original_tmp)

        binder_final = binder_cumulant(samples)
        binder_raw = binder_cumulant(original_tmp)

        print("\nPhysical Refinement Diagnostics:", chi_final)

        print("Binder cumulant (1):", binder_final)
        print("Chi improvement:", chi_final - chi_raw)

        print("Binder improvement:", binder_final - binder_raw)

        grid = vutils.make_grid(
            samples,
            nrow=50,
            padding=2
        )

        grid = (grid * 255).clamp(0, 255).byte()
        nd_arr = grid.permute(1, 2, 0).cpu().numpy()

        Image.fromarray(nd_arr).save(
            filename.replace(".png", "_refined.png")
        )

        grid_0 = vutils.make_grid(
            original_tmp,
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
            n_chains=2000,
            steps=100000,
            burn_in_ratio=0,
            thinning=1,
            lag_steps=None
    ):
        device = self.device

        if lag_steps is None:
            lag_steps = self.lag_step

        temperature = self.temperature().item()
        T = temperature

        # Initialize chains
        visible = torch.bernoulli(
            torch.full((n_chains, self.n_visible), 0.5, device=device)
        )

        energy_trace = []

        # MCMC sampling
        for _ in range(steps):
            # h sample
            hidden = torch.bernoulli(
                torch.sigmoid((visible @ self.W + self.hidden_bias) / T)
            )

            # internal Gibbs steps
            for _ in range(self.gibbs_steps):
                visible = torch.bernoulli(
                    torch.sigmoid((hidden @ self.W.T + self.visible_bias) / T)
                )
                hidden = torch.bernoulli(
                    torch.sigmoid((visible @ self.W + self.hidden_bias) / T)
                )

            # final visible update
            visible = torch.bernoulli(
                torch.sigmoid((hidden @ self.W.T + self.visible_bias) / T)
            )

            energy_trace.append(
                self.free_energy(visible, T).detach()
            )

        energy_trace = torch.stack(energy_trace)  # (time, chains)

        # Burn-in
        burn = int(burn_in_ratio * steps)

        if burn >= steps - 10:
            burn = 0

        energy_trace = energy_trace[burn:]

        # Thinning
        energy_trace = energy_trace[::thinning]
        n_steps, n_chains = energy_trace.shape

        max_lag = n_steps // 2
        if max_lag < 2:
            return {
                "tau_int": float("nan"),
                "tau_std": float("nan"),
                "tau_max": float("nan"),
                "tau_min": float("nan"),
                "tau_rms": float("nan"),
                "tau_geo": float("nan"),
                "tau_harm": float("nan"),
                "ess": 0.0,
                "r_hat": float("nan"),
                "n_eff_per_chain": 0.0,
                "autocorr_len": 0,
                "mix_index_ratio": float("nan")
            }

        # Center
        centered = energy_trace - energy_trace.mean(dim=0, keepdim=True)
        var = torch.var(centered, dim=0, unbiased=True)

        if torch.mean(var) < 1e-12:
            return {
                "tau_int": float("inf"),
                "tau_std": 0.0,
                "tau_max": float("inf"),
                "tau_min": float("inf"),
                "tau_rms": float("inf"),
                "tau_geo": float("inf"),
                "tau_harm": float("inf"),
                "ess": 0.0,
                "r_hat": float("nan"),
                "n_eff_per_chain": 0.0,
                "autocorr_len": 0,
                "mix_index_ratio": float("nan")
            }

        lag_values = torch.unique(
            torch.logspace(0, math.log10(max_lag), steps=lag_steps, device=device).long()
        )

        samples_per_lag = 10000
        acf_list = []

        for lag in lag_values:
            if lag >= n_steps:
                break

            ideal = torch.randint(0, n_steps - lag, (samples_per_lag,), device=device)

            x_t = centered[ideal]
            x_t_lag = centered[ideal + lag]

            rho = (x_t * x_t_lag).mean(dim=0) / (var + 1e-12)
            acf_list.append(rho)

        acf_tensor = torch.stack(acf_list)  # (lags, chains)

        tau_list = []

        for chain in range(n_chains):
            rho = acf_tensor[:, chain]
            neg_idx = torch.where(rho < 0)[0]

            if len(neg_idx) > 0:
                cutoff = neg_idx[0].item()
                rho = rho[:cutoff]

            tau = 1.0 + torch.sum(rho)
            tau_list.append(tau)

        min_value = 1e-12

        tau_per_chain = torch.stack(tau_list)
        tau_per_chain = torch.clamp(tau_per_chain, min=min_value)

        tau_mean = tau_per_chain.mean().item()
        tau_std = tau_per_chain.std(unbiased=False).item()
        tau_max = tau_per_chain.max().item()
        tau_min = tau_per_chain.min().item()

        tau_rms = torch.sqrt(torch.mean(tau_per_chain ** 2)).item()
        tau_geo = torch.exp(torch.mean(torch.log(tau_per_chain))).item()
        tau_harm = (1.0 / torch.mean(1.0 / tau_per_chain)).item()

        # ESS
        ess_per_chain = n_steps / (tau_per_chain + 1e-12)
        total_ess = torch.sum(ess_per_chain).item()

        # R-hat (unchanged)
        half = n_steps // 2

        if half < 10:
            r_hat = float("nan")
        else:
            first_half = energy_trace[:half]
            second_half = energy_trace[half:2 * half]

            # (half, 2 * chains)
            split = torch.cat([first_half, second_half], dim=1)

            n_half, m = split.shape
            device = split.device

            # Rank normalization
            flat = split.flatten()

            sorted_idx = torch.argsort(flat)
            ranks = torch.empty_like(flat, dtype=torch.float32)

            ranks[sorted_idx] = torch.arange(
                1, len(flat) + 1, device=device, dtype=torch.float32
            )

            ranks = (ranks - 0.5) / len(flat)
            ranks = torch.clamp(ranks, 1e-6, 1 - 1e-6)

            # Gaussian transform
            z_flat = torch.erfinv(2.0 * ranks - 1.0) * (2.0 ** 0.5)
            z = z_flat.view(n_half, m)

            # Variance guard
            chain_vars = torch.var(z, dim=0, unbiased=True)
            W = torch.mean(chain_vars + 1e-12)

            chain_means = z.mean(dim=0)
            B = n_half * torch.var(chain_means, unbiased=True)

            var_hat = (1 - 1 / n_half) * W + (1 / n_half) * B

            r_hat = torch.sqrt(var_hat / (W + 1e-12)).item()

        autocorr_length_mask = int(acf_tensor.shape[0])

        mix_index_ratio = ((
                                   (tau_rms + tau_max + tau_min) / 3)
                           / autocorr_length_mask)

        return {
            "tau_int": tau_mean,
            "tau_std": tau_std,
            "tau_max": tau_max,
            "tau_min": tau_min,
            "tau_rms": tau_rms,
            "tau_geo": tau_geo,
            "tau_harm": tau_harm,
            "ess": float(total_ess),
            "r_hat": float(r_hat),
            "n_eff_per_chain": float(ess_per_chain.mean().item()),
            "autocorr_len": autocorr_length_mask,
            "mix_index_ratio": mix_index_ratio
        }

    @torch.no_grad()
    def save_professional_samples(
            self,
            filename="samples_professional.png",
            n_display=100,
            steps=10000,
            energy_tracker=None,
            client=None
    ):
        samples = self.generate_ensemble_samples(
            n_chains=n_display,
            steps=steps,
            energy_tracker=energy_tracker
        )

        samples_original = samples.clone().detach()

        samples = self.llm_uncertainty_refinement(samples, client=client)

        refiner = Refinement(self)

        samples = refiner.myra_refine(samples)

        samples = samples.view(-1, 1, 28, 28)
        original_tmp = samples_original.view(-1, 1, 28, 28)

        grid = vutils.make_grid(
            samples,
            nrow=50,
            padding=2
        )

        grid = (grid * 255).clamp(0, 255).byte()
        nd_arr = grid.permute(1, 2, 0).cpu().numpy()

        Image.fromarray(nd_arr).save(
            filename.replace(".png", "_refined.png")
        )

        grid1 = vutils.make_grid(
            original_tmp,
            nrow=50,
            padding=2
        )

        grid1 = (grid1 * 255).clamp(0, 255).byte()
        nd_arr2 = grid1.permute(1, 2, 0).cpu().numpy()

        Image.fromarray(nd_arr2).save(
            filename.replace(".png", "_symbol.png")
        )

        diagnostics = self.ensemble_diagnostics()

        chi_final = susceptibility(samples)
        chi_raw = susceptibility(original_tmp)

        binder_final = binder_cumulant(samples)
        binder_raw = binder_cumulant(original_tmp)

        print("\nSecondary magnetic susceptibility:", chi_final)

        print("Binder cumulant (2):", binder_final)
        print("Chi improvement:", chi_final - chi_raw)

        print("Binder improvement:", binder_final - binder_raw)

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

        betas = torch.linspace(0.0, 1.0, n_intermediate, device=device) ** 2

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
                      ) / T
            # SINGLE temperature scaling point

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
        W = self.W
        b = self.visible_bias
        h_bias = self.hidden_bias
        N, D = data.shape

        # hidden pre-activation
        wx = (data @ W + h_bias) / T  # (N, H)
        log_probs = []

        for interaction in range(D):
            v_i = data[:, interaction]  # (N,)
            W_i = W[interaction]  # (H,)

            # flip effect
            delta = (1 - 2 * v_i).unsqueeze(1) * W_i / T
            wx_flip = wx + delta

            # free energy difference
            term = torch.sum(
                F.softplus(wx_flip) - F.softplus(wx),
                dim=1
            )

            logits = (b[interaction] / T) + term

            # log P(v_i | rest)
            log_prob_i = -F.binary_cross_entropy_with_logits(
                logits,
                v_i,
                reduction='none'
            )

            log_probs.append(log_prob_i)

        # average over dimensions
        log_prob = torch.stack(log_probs, dim=1).mean(dim=1)

        print("coefficient (kappa):", D)

        # average over batch
        return log_prob.mean().item()

    @torch.no_grad()
    def reconstruct(self, v):

        T = self.temperature()
        h = torch.sigmoid((v @ self.W + self.hidden_bias) / T)
        v_recon = torch.sigmoid((h @ self.W.T + self.visible_bias) / T)

        return v_recon


# Multi-GPU Worker

def worker(Gpu_Id, seed_round, consequences, reputational):
    client = SafeOpenAIClient()
    torch.cuda.set_device(Gpu_Id)
    torch.manual_seed(seed_round)
    torch.cuda.manual_seed_all(seed_round)
    energy_tracker = GPUEnergyTracker(Gpu_Id)

    device_warm = torch.device(f"cuda:{Gpu_Id}")

    # Data

    data = load_mnist(device_warm)
    train_data_um = data[:60000]
    test_data_um = data[60000:]

    last_model = HybridThermodynamicRBM(
        n_visible=784,
        n_hidden=512,
        device_type=f"cuda:{Gpu_Id}",
        fixed_temperature=None
    )

    train_result = last_model.train(
        train_data_um,
        energy_tracker=energy_tracker
    )

    if train_result == "ABORT":
        print("[WORKER] Training aborted early")
        return

    def dots_spinner(stop_event_choose, elastic_label):
        """
        Lightweight terminal spinner used to indicate that a diagnostic
        computation is currently running.
        """
        frames = ["   ", ".  ", ".. ", "..."]
        integer = 0

        while not stop_event_choose.is_set():
            sys.stdout.write(f"\r▶ {elastic_label}{frames[integer % len(frames)]}")
            sys.stdout.flush()
            time.sleep(0.4)

            integer += 1

    print(f"\n[GPU {Gpu_Id}] Running AIS...")

    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=dots_spinner, args=(stop_event, "AIS"))
    spinner_thread.start()
    ais_start = time.time()

    try:
        log_Z_3ox, ais_var, ais_ess = last_model.ais_log_partition(
            n_runs=8000,
            n_intermediate=12000,
            energy_tracker=energy_tracker
        )
    except Exception as emerald:
        print(f"\n✖ AIS failed: {emerald}")
        log_Z_3ox, ais_var, ais_ess = float("nan"), float("nan"), float("nan")

    ais_duration = time.time() - ais_start

    stop_event.set()
    spinner_thread.join()

    sys.stdout.write(f"\r✔ AIS finished in {ais_duration:.2f} seconds\n")

    train_ll = last_model.log_likelihood(train_data_um, log_Z_3ox)
    Test_ll = last_model.log_likelihood(test_data_um, log_Z_3ox)
    Train_pl = last_model.pseudo_likelihood(train_data_um)
    Test_pl = last_model.pseudo_likelihood(test_data_um)

    # Reconstruction

    reconstruction = last_model.reconstruct(test_data_um)

    reconstruction_mse = torch.mean(
        (test_data_um - reconstruction) ** 2
    ).item()

    recon_acc = last_model.reconstruction_accuracy(test_data_um)

    # Thermodynamic diagnostics

    final_temperature = last_model.temperature().item()
    weight_norm = torch.norm(last_model.W).item()
    gain_spectral = weight_norm / final_temperature

    sample_filename = f"samples_gpu{Gpu_Id}_seed{seed_round}.png"
    prof_filename = f"samples_prof_gpu{Gpu_Id}_seed{seed_round}.png"

    # First ensemble save

    last_model.save_ensemble_samples(
        filename=sample_filename,
        n_display=1200,
        steps=100000,
        energy_tracker=energy_tracker,
        client=client
    )

    # Professional ensemble + diagnostics

    diagnostics = last_model.save_professional_samples(
        filename=prof_filename,
        n_display=1200,
        steps=100000,
        energy_tracker=energy_tracker,
        client=client
    )

    print("\nRunning energy analysis ↓\n")

    analysis_steps = [

        ("Energy distribution",
         lambda: SrtrbmEnergy.plot_data_vs_model_energy(last_model, train_data_um)),

        ("Energy landscape extremes",
         lambda: SrtrbmEnergy.visualize_energy_extremes(last_model, train_data_um)),

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
         lambda: SrtrbmMetrics.sample_quality_metrics(last_model, train_data_um)),
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
        except Exception as emerald:

            print(f"\n✖ {name} failed:", emerald)

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
            "entropy_spatial_std": float("nan"),
            "hamming": float("nan"),
            "effective_rank": float("nan"),
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
        "spectral_gain": gain_spectral,
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
        "entropy_spatial_std": quality["entropy_spatial_std"],
        "hamming": quality["hamming"],
        "effective_rank": quality["effective_rank"],
        "mean_l2": quality["mean_l2"],
        "ais_log_weight_variance": ais_var,
        "mcmc_tau_int": diagnostics["tau_int"],
        "mcmc_tau_std": diagnostics["tau_std"],  # heterogeneity
        "mcmc_tau_max": diagnostics["tau_max"],  # worst-case chain
        "mcmc_tau_min": diagnostics["tau_min"],  # fastest chain
        "mcmc_tau_rms": diagnostics["tau_rms"],
        "mcmc_tau_geo": diagnostics["tau_geo"],
        "mcmc_tau_harm": diagnostics["tau_harm"],
        "mcmc_ess": diagnostics["ess"],
        "mcmc_ess_per_chain": diagnostics["n_eff_per_chain"],
        "mcmc_r_hat": diagnostics["r_hat"],
        "mcmc_acf_len": diagnostics["autocorr_len"],
        "mcmc_mix_index": diagnostics["mix_index_ratio"],
        "prof_sample_file": prof_filename,
        "sample_file": sample_filename
    })

    plt.figure(figsize=(10, 12))

    # Flip rate

    plt.subplot(3, 1, 1)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)

    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1.2)

    plt.plot(last_model.flip_hist, linewidth=2, label="Flip rate")
    plt.plot(last_model.c_hist, linestyle="--", linewidth=2, label="Adaptive reference")
    plt.title("Microscopic Flip Dynamics")
    plt.ylabel("Flip Rate")
    plt.grid(color="#CFDCD0", alpha=0.25, linewidth=0.8)
    plt.legend()

    # Temperature evolution

    plt.subplot(3, 1, 2)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)

    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1.2)

    plt.plot(last_model.temp_hist, linewidth=2)
    plt.title(r"Global Temperature $T$")
    plt.ylabel("Temperature")
    plt.grid(color="#CFDCD0", alpha=0.25, linewidth=0.8)

    # Micro vs Macro

    plt.subplot(3, 1, 3)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)

    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1.2)

    plt.plot(last_model.T_micro_hist, linewidth=2, label=r"$T_{micro}$")
    plt.plot(last_model.T_macro_hist, linewidth=2, label=r"$T_{macro}$")
    plt.title("Micro–Macro Temperature Decomposition")
    plt.xlabel("Epoch")
    plt.ylabel("Temperature Components")
    plt.grid(color="#CFDCD0", alpha=0.25, linewidth=0.8)
    plt.legend()

    plt.tight_layout()
    plt.savefig("srtrbm_core_dynamics.pdf", bbox_inches="tight")
    plt.close()

    # Energy & Spectral Diagnostics

    plt.figure(figsize=(10, 14))

    # Weight norm

    plt.subplot(4, 1, 1)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)

    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1.2)

    plt.plot(last_model.weight_norm_hist, linewidth=2)
    plt.title("Weight Norm Evolution")
    plt.ylabel(r"$||W||$")
    plt.grid(color="#CFDCD0", alpha=0.25, linewidth=0.8)

    # Effective beta

    plt.subplot(4, 1, 2)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)

    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1.2)

    plt.plot(last_model.beta_eff_hist, linewidth=2)
    plt.title(r"Effective Inverse Temperature $\beta_{\mathrm{eff}}$")
    plt.ylabel(r"$\beta_{\mathrm{eff}}$")
    plt.grid(color="#CFDCD0", alpha=0.25, linewidth=0.8)

    # Free energy comparison

    plt.subplot(4, 1, 3)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)

    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1.2)

    plt.plot(last_model.F_data_hist, linewidth=2, label=r"$F_{data}$")
    plt.plot(last_model.F_model_hist, linewidth=2, label=r"$F_{model}$")
    plt.title("Free Energy: Data vs Model")
    plt.ylabel("Free Energy")
    plt.grid(color="#CFDCD0", alpha=0.25, linewidth=0.8)
    plt.legend()

    # Spectral beta

    plt.subplot(4, 1, 4)

    ax = plt.gca()

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)

    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1.2)

    plt.plot(last_model.spectral_beta_hist)

    plt.axhline(1.0, linestyle="--")

    plt.title(r"Spectral Inverse Temperature $\beta_{spectral}$")
    plt.xlabel("Epoch")
    plt.ylabel(r"$\beta_{spectral}$")
    plt.grid(color="#CFDCD0", alpha=0.25, linewidth=0.8)

    plt.tight_layout()
    plt.savefig("srtrbm_energy_diagnostics.pdf", bbox_inches="tight")
    plt.close()

    # LLM Section Phase 1-0

    sample_refined = sample_filename.replace(".png", "_refined.png")
    prof_refined = prof_filename.replace(".png", "_refined.png")

    samples_for_phase = last_model.generate_ensemble_samples(n_chains=2000, steps=100000)
    samples_for_phase = samples_for_phase.view(-1, 1, 28, 28)

    chi = susceptibility(samples_for_phase)

    llm_metrics = {
        "temperature": float(final_temperature),
        "spectral_gain": float(gain_spectral),
        "energy_gap": float(energy_stats["energy_gap"]),

        "quality": {
            "pixel_entropy": float(quality["pixel_entropy"]),
            "entropy_spatial_std": float(quality["entropy_spatial_std"]),
            "hamming": float(quality["hamming"]),
            "effective_rank": float(quality["effective_rank"]),
            "mean_l2": float(quality["mean_l2"])
        },

        "sampling": {
            "mcmc_tau_int": float(diagnostics["tau_int"]),
            "mcmc_tau_rms": float(diagnostics["tau_rms"]),
            "mcmc_r_hat": float(diagnostics["r_hat"])
        },

        "trend": {
            "temp_slope": float(last_model.temp_hist[-1] - last_model.temp_hist[0]),
            "beta_slope": float(last_model.beta_eff_hist[-1] - last_model.beta_eff_hist[0])
        },

        "phase": {
            "susceptibility": float(chi)
        },

        "history": {
            "gap_trend": float(last_model.F_gap_hist[-1] - last_model.F_gap_hist[0]),
            "entropy_trend": float(last_model.persistent_div_hist[-1] - last_model.persistent_div_hist[0]),
            "temp_trend": float(last_model.temp_hist[-1] - last_model.temp_hist[0]),
            "beta_trend": float(last_model.beta_eff_hist[-1] - last_model.beta_eff_hist[0]),
            "gap_std": float(np.std(last_model.F_gap_hist)),
            "entropy_std": float(np.std(last_model.persistent_div_hist)),
            "temp_std": float(np.std(last_model.temp_hist)),

            "current": {
                "entropy": float(last_model.persistent_div_hist[-1]),
                "temperature": float(last_model.temp_hist[-1]),
                "beta": float(last_model.beta_eff_hist[-1]),
                "gap": float(last_model.F_gap_hist[-1])
            },

            "learning_signal": float(last_model.delta_w_hist[-1]),
            "stagnation": bool(last_model.delta_w_hist[-1] < 1e-12),
            "learning_active": bool(last_model.delta_w_hist[-1] > 1e-12),
            "cooling": bool(last_model.temp_hist[-1] < last_model.temp_hist[0]),
            "heating": bool(last_model.temp_hist[-1] > last_model.temp_hist[0]),

            "trend_signature": [
                float(np.sign(last_model.F_gap_hist[-1] - last_model.F_gap_hist[0])),
                float(np.sign(last_model.persistent_div_hist[-1] - last_model.persistent_div_hist[0])),
                float(np.sign(last_model.temp_hist[-1] - last_model.temp_hist[0]))
            ]
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

    consequence = Evaluate(
        llm_metrics,
        prof_refined,
        prof_perfect,
        client=client
    )

    analysis_signal = ANASIS(consequence.get("analysis", ""))

    print(f"[ANALYSIS SIGNAL] {analysis_signal:.3f}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(consequence, f, indent=2, ensure_ascii=False)

    print(f"[LLM OUTPUT SAVED] → {output_json}")

    print("\n[LLM RESULT]\n")

    json_text = json.dumps(consequence, indent=2, ensure_ascii=False)

    for line in json_text.split("\n"):
        wrapped_lines = textwrap.wrap(line, width=90) or [""]

        for word_line in wrapped_lines:
            print(word_line)
            sys.stdout.flush()
            time.sleep(0.005)

    # TODO: Clear everything from here up to the reputational.append stmt
    # (including the section) until a 'True' lag_steps is encountered. Once
    # encountered, re-add this section, define the interval, and execute.

    cuda_state = torch.cuda.get_rng_state_all()
    cpu_state = torch.get_rng_state()

    result_was_plural = {}
    entropy_results = {}

    print(f"\nLag[{last_model.lag_step}]_step ↓\n")

    for ls in range(
            5 if last_model.lag_step - 5 < 5 else last_model.lag_step - 5,
            last_model.lag_step + 7
    ):
        torch.cuda.set_rng_state_all(cuda_state)
        torch.set_rng_state(cpu_state)

        # Run MCMC diagnostics

        d = last_model.ensemble_diagnostics(lag_steps=ls)

        result_was_plural[ls] = d
        quality = SrtrbmMetrics.sample_quality_metrics(last_model, train_data_um)

        pixel_entropy = quality["pixel_entropy"]
        spatial_entropy = quality["entropy_spatial_std"]

        entropy_results[ls] = {
            "pixel_entropy": pixel_entropy,
            "spatial_entropy": spatial_entropy
        }

        low_U, high_Q = sorted([pixel_entropy, spatial_entropy])
        band_consistency = low_U <= d['mix_index_ratio'] <= high_Q

        last_model.dictionary[ls] = {
            "mix": d['mix_index_ratio'],
            "pixel": pixel_entropy,
            "spatial": spatial_entropy,
            "low": low_U,
            "high": high_Q,
            "band": band_consistency
        }

        print(
            f"LagSteps={ls:<2} | Mix={d['mix_index_ratio']:.6f} | "
            f"PixelH={pixel_entropy:.4f} | SpatialH={spatial_entropy:.4f} | "
            f"BandConsistent={band_consistency}"
        )

    print("\n")

    reputational.append({
        "seed": seed_round,
        "gpu": Gpu_Id,
        "expected_lag": last_model.lag_step,
        "lag_dict": dict(last_model.dictionary)
    })


# Main Section

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    available_gpus = torch.cuda.device_count()

    seeds = [2]

    if available_gpus < len(seeds):
        raise RuntimeError("Not enough GPUs available.")

    manager = mp.Manager()
    results = manager.list()
    reputation = manager.list()

    processes = []

    for gpu_id, seed in enumerate(seeds):
        p = mp.Process(
            target=worker,
            args=(gpu_id, seed, results, reputation)
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
        print(f"Spectral Gain            : {result['spectral_gain']:.6f}")
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
        print(f"MCMC tau_std             : {result['mcmc_tau_std']:.2f}")
        print(f"MCMC tau_max             : {result['mcmc_tau_max']:.2f}")
        print(f"MCMC tau_min             : {result['mcmc_tau_min']:.2f}")
        print(f"MCMC tau_rms             : {result['mcmc_tau_rms']:.2f}")
        print(f"MCMC tau_geo             : {result['mcmc_tau_geo']:.2f}")
        print(f"MCMC tau_harm            : {result['mcmc_tau_harm']:.2f}")
        print(f"MCMC ESS                 : {result['mcmc_ess']:.2f}")
        print(f"MCMC ESS / chain         : {result['mcmc_ess_per_chain']:.2f}")
        print(f"MCMC R-hat               : {result['mcmc_r_hat']:.4f}")
        print(f"MCMC ACF Length          : {result['mcmc_acf_len']}")
        print(f"MCMC Mix Index           : {result['mcmc_mix_index']:.6f}")

        entropy_a = result['entropy_spatial_std']
        entropy_b = result['pixel_entropy']

        low_down = min(entropy_a, entropy_b)
        high_up = max(entropy_a, entropy_b)

        mix_index = result['mcmc_mix_index']
        in_band = low_down <= mix_index <= high_up

        if entropy_a > entropy_b:
            print("\nNote: Entropy bounds reordered—metric definitions are well-formed.\n")

        print(f"Entropy Band             : Closed interval [{low_down:.6f}, {high_up:.6f}]")
        print(
            f"Choice(Mix | Band)       : C({mix_index:.6f} | [{low_down:.6f}, {high_up:.6f}]) = {mix_index:.6f}"
        )
        print(f"Band Consistency         : {in_band}")

        rep_match = next(
            (r for r in reputation if r["seed"] == result["seed"]),
            None
        )

        if not rep_match:
            print("\n>>> SEED EXPERIMENT: FAILED | No reputational data\n")
        else:
            expected_lag = rep_match["expected_lag"]
            lag_dict = rep_match["lag_dict"]

            lag_band_ok = lag_dict.get(expected_lag, {}).get("band", False)

            true_lags = [
                lag for lag, info in lag_dict.items()
                if info.get("band", False)
            ]

            true_count = len(true_lags)

            condition = in_band and lag_band_ok and (true_count == 1)

            if condition:
                print("\n>>> SEED EXPERIMENT: SUCCESS\n")
            else:
                print(
                    f"\n>>> SEED EXPERIMENT: FAILED | "
                    f"Band={in_band} | "
                    f"Lag[{expected_lag}]={lag_band_ok} | "
                    f"TrueCount={true_count} | "
                    f"TrueLags={true_lags}\n"
                )

        print(f"Hamming                  : {result['hamming']:.6f}")
        print(f"Effective Rank           : {result['effective_rank']:.6f}")
        print(f"Mean Distribution L2     : {result['mean_l2']:.6f}")
        print(f"Professional Samples File: {result['prof_sample_file']}")
        print(f"Generated Samples File   : {result['sample_file']}")

        print()

# MYRA: A Feedback-Controlled Thermodynamic RBM for Hybrid Intelligence