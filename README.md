# MYRA: Hybrid Intelligence System

A hybrid intelligence system where thermodynamic energy-based generation (SR-TRBM) and LLM-assisted multimodal interpretation are unified under the MYRA architecture within a closed-loop framework.

> **“What did the model actually learn?”**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19211121.svg)](https://doi.org/10.5281/zenodo.19211121)

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![CUDA](https://img.shields.io/badge/CUDA-enabled-green)
![Python](https://img.shields.io/badge/python-3.10-blue)

> MYRA generates.  
> Energy guides.  
> Structure emerges.  
> The system understands.

---

## 📦 Additional Files and Models

Example outputs are in the `artifacts/` directory (PCD-1, seed=1, adaptive temperature), together with a corresponding `run.log`.

The following files support reproducibility and evaluation:

- **`stan.7z`**  
  Compressed dataset archive. After extraction, it yields `stan.dgts`, used as the primary numerical dataset.

- **`zeta_mnist_hybrid.pt`**  
  Pretrained model checkpoint for inference, sampling, and energy-based analysis.

### 🔧 Usage

Extract the dataset before running experiments:

```bash
7z x stan.7z
```

Alternatively, you can use WinRAR or other compatible tools. The model's entire size is approximately 190 MB.

---

### 🔧 Usage

Extract the dataset before running experiments:

\`\`\`bash

7z x stan.7z

\`\`\`

Alternatively, you can use WinRAR or other compatible tools. The model's entire size is approximately 190 MB.

---

## 🔬 Experiment Protocol: Single-Seed Band Uniqueness Criterion

The MYRA experiment protocol does not rely on multi-seed averaging or aggregate statistics across runs. Instead, each seed is evaluated independently through a band uniqueness criterion applied over a local lag sweep. The purpose is to obtain a truth value for the seed level in the execution results. The concept is a new idea in the literature.

For reference outputs, see the following:

- `artifacts/run.log`—example of a successful run with a valid ground truth
- `artifacts/run_false_example.log`—example of a run where the criterion is not satisfied

After training and sampling, the system sweeps lag steps in the range:

`[lag_step − 5, lag_step + 6]`

At each step, the MCMC Mix Index is compared against the closed entropy interval:

`[min(PixelH, SpatialH), max(PixelH, SpatialH)]`

A seed experiment is considered successful if and only if all three conditions hold simultaneously:

1. Global Mix Index ∈ Entropy Band
2. BandConsistent = True at the characteristic lag step
3. Exactly one lag step across the full sweep satisfies \`BandConsistent = True\`.

---

### 🎯 Key Criterion: Uniqueness

Condition (iii) is the structurally decisive one.

If multiple lag steps produce band-consistent results, the mixing signal is diffuse—the system has not converged to a sharp, well-localized attractor.

> A system that converges everywhere has converged nowhere in particular.

Uniqueness of the band-consistent lag is therefore not a byproduct of the evaluation; it is the criterion itself.

---

### 🔥 Interpretation

This design reflects a thermodynamic intuition:

- A well-mixed chain should exhibit band consistency precisely at the characteristic autocorrelation scale of its energy landscape:
- Not broadly
- Not sporadically

The goal is sharp localization, not widespread agreement.

---

## 🚀 Overview

MYRA (Model Representation Anatomy) is a hybrid system that combines energy-based generation with structural analysis and LLM-assisted interpretation.

At its core, MYRA uses a thermodynamically regulated Boltzmann machine (SR-TRBM) as its generative component. This model learns data by organizing it within an energy landscape and sampling stable configurations through stochastic dynamics. MYRA is not only a generative system. It is designed to analyze what the model has learned. Generated states are evaluated, refined, and interpreted within a unified pipeline.

### System Loop

- SR-TRBM generates samples from the learned energy landscape  
- Structural modules refine and stabilize these samples  
- Metrics measure entropy, diversity, and similarity  
- The LLM interprets the results and provides higher-level analysis  

These components form a single system focused on understanding model behavior.

---

## 📚 Theoretical Origin

The SR-TRBM model used in this project originates from prior work:

👉 https://github.com/cagasolu/sr-trbm  

The associated arXiv publication serves as the theoretical foundation.

---

## 🧠 Core Idea

> **What did the model actually learn?**

The model does not fail to learn. It learns the structure of the data within its own representation. However, this structure is not directly visible in the generated outputs. During generation, the model recombines learned patterns within its energy landscape. This can produce outputs that are coherent but do not exist in the original dataset.

This creates a gap between what the model has learned and what we observe. MYRA is designed to analyze this gap. It focuses not only on generated outputs but on revealing the structure that the model has internalized, using energy-based modeling, structural analysis, and LLM-assisted interpretation.

---

## ⚙️ System Topology

```
Latent Space
↓
SR-TRBM (Generative Engine)
↓
Sampled States
↓
MYRA (Structural Reasoning & Refinement)
↓
Refined States
↓
Metrics Extraction
↓
LLM (Interpretation Layer)
↓
Convergence & Structure Analysis
↓
Epistemic Control (implicit, MYRA-governed)
↓
Final Output / Conclusion
```

---

## 📦 Project Structure

```
Main Core
├── srtrbm_project_core.py
│   Central orchestration module handling generation, sampling, and energy-based metrics
│   of the Boltzmann machine.

LLM Integration
├── openaiF/
│   ├── __init__.py
│   │   Unified interface for LLM-based evaluation and control.
│   │
│   ├── client.py
│   │   Fault-tolerant LLM client with retry and fallback handling.
│   │
│   ├── gateway.py
│   │   Multimodal interpretation engine combining metrics and LPIPS-based similarity.
│   │
│   ├── hook.py
│   │   Epistemic controller for evidence-bounded, uncertainty-aware decisions.

Refinement
├── supplement/
│   └── cluster.py
│       Embedding-based structural refinement module. Projects generated samples onto a
│       learned manifold and reconstructs them via prototype similarity and spatial filtering.

Analysis & Diagnostics
├── analysis/
│   ├── __init__.py
│   │   Exposes analysis utilities for integration with the main pipeline.
│   │
│   └── AutoGPU.py
│       GPU energy tracking via NVML, monitoring real-time power usage, and estimating
│       total energy consumption during execution.

Visualization & Diagnostics
├── graphs/
│   ├── __init__.py
│   │   Aggregates visualization and diagnostic utilities into a unified interface.
│   │
│   ├── SrtrbmEnergy.py
│   │   Energy landscape analysis and data–model energy distribution comparison.
│   │
│   ├── SrtrbmMetrics.py
│   │   Statistical diagnostics, including entropy, diversity, and phase transition analysis.
│   │
│   └── SrtrbmVisualization.py
│       Visualization of generated samples, RBM filters, and fantasy particles.

Correction Modules
├── correction/
│   ├── __init__.py
│   │   Exposes refinement and correction utilities for integration into the pipeline.
│   │
│   └── NO.py
│       Implements energy-based and spatial refinement methods to stabilize and
│       denoise generated samples.

LLM Configuration
├── yaml/
│   ├── perception.yaml
│   │   Specifies structural reasoning rules for interpreting model outputs and
│   │   distinguishing collapse from structured convergence.

Assets
├── zeta_mnist_hybrid.pt
│   Pretrained hybrid model used for generation and evaluation.
│
└── stan.dgts
    Structured dataset used for training and inference.
```

<details>
<summary>Full directory structure</summary>

```
.
├── analysis
├── correction
├── graphs
├── openaiF
├── supplement
├── yaml
├── srtrbm_project_core.py
├── stan.dgts
└── zeta_mnist_hybrid.pt
```

</details>

---

## 🔍 LLM Integration

MYRA uses an LLM as an external interpretive layer.

The LLM is not used for generation. Instead, it analyzes model behavior by looking at outputs, metrics, and structural patterns.

---

## ⚙️ Implementation

The current setup uses the OpenAI API (`openaiF` module), mainly because it is easy to run and requires minimal setup. That said, the system is not tied to OpenAI.

The interface—especially `client.py`—is kept simple on purpose so it can be adapted to other providers with minimal effort. This includes:

- Anthropic (Claude)
- Google (Gemini)
- Meta (Llama / local setups)
- Mistral
- DeepSeek
- Qwen
- or other LLMs

You can rename or replace the `openaiF` module to use a different LLM backend.

In most cases, adapting the system only requires updating the module import and the `SafeOpenAIClient()` reference in `srtrbm_project_core.py`. Depending on the backend, minor adjustments may also be needed in the module’s `__init__.py` to align the interface.

This keeps the integration simple and makes it easy to switch between different LLM providers.

In practice, the interpretive layer can be swapped depending on your needs (e.g., performance, privacy, or local execution).

---

## 🧪 Practical Notes

- OpenAI is used here mainly for convenience.
- Other models can be plugged in with small changes.
- Different models will not behave the same—some are more precise, some more generic.

---

> [!NOTE]

> The LLM is only used for interpretation.  
> It should not be treated as ground truth, but as an additional layer of analysis.

---

### 🔁 Epistemic Constraint

The LLM’s confidence is constrained by available evidence:

> confidence ≤ evidence

This prevents overconfident interpretations and enforces alignment between:

- observed metrics
- visual structure
- inferred system state

---

### ⚙️ What the LLM receives

The LLM processes both:

#### Metrics

* temperature
* beta_effective
* energy_gap
* reconstruction error
* entropy & diversity

#### Images

* refined samples
* stabilized outputs

---

### 📊 Example Output

```json
{
  "regime": "attractor-dominated refinement",
  "attractor_strength": "high",
  "convergence_type": "strong",
  "diversity_shift": "refined",
  "structural_improvement": "strong"
}
```

---

## 🧬 Key Insight

High repetition is not necessarily a failure. In some cases, it may indicate the formation of stable internal structures.

```
multiple stochastic variations → converge → stable prototypes
```

---

## 🔒 Model & Training Policy

This repository provides controlled access to MYRA, including its inference pipeline and evaluation framework.

The project follows a semi-open paradigm: core system behavior is observable, but full training dynamics and optimization strategies remain restricted.

---

### ✅ Available

The following components are fully accessible:

- Pre-trained model weights (SR-TRBM core)
- Inference and generation pipeline
- MYRA structural refinement mechanism (inference-time behavior)
- LLM-based multimodal evaluation framework
- Metrics computation (e.g., diversity, similarity, convergence signals)
- Epistemic evaluation logic (evidence-constrained interpretation)
- YAML-based structural principles and constraint definitions

---

### ⚠️ Partially Documented

These components are visible at a high level but not fully disclosed:

- Training objectives and loss shaping strategy
- Energy scheduling and thermodynamic control logic
- Refinement dynamics during training vs inference
- Internal calibration of metrics and thresholds
- LLM prompting strategy and constraint tuning

---

### 🔒 Not Included

The following components are not included in this release:

- Full training pipeline and data flow
- Optimization strategies and hyperparameter search
- Internal training dynamics and update rules
- Proprietary heuristics for stability and convergence
- Dataset construction and curation pipeline
- CNN-based supporting architectures used within the model
- Embedding mechanisms and DGTS-based internal data structures
- Exact prompt engineering and internal processing details of LLM usage
- Prompt-to-response transformation strategies and orchestration logic

---

### 🧠 Rationale

MYRA is designed as a research-grade hybrid intelligence system, where

- generation (SR-TRBM)
- structural reasoning (MYRA)
- interpretation (LLM)

are tightly coupled.

Releasing only the inference layer allows:
- reproducibility of observed behavior
- evaluation of system outputs
- analysis of structural and epistemic properties

while preserving the integrity of the underlying training methodology.

---

### ⚖️ Philosophy

This project adopts a controlled transparency approach:

> Behavior is open.  
> Mechanism is partially open.  
> Optimization remains closed.

---

## ⚡ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
python3 srtrbm_project_core.py
```

---

## 🔮 Future Work

* LLM-driven adaptive control (temperature/beta tuning)
* Closed-loop generative systems
* Real-time interpretability
* Multi-sample evaluation

---

## 📌 Summary

This is not just a generative model.

It is a system that:

* generates
* refines
* interprets
* and reveals aspects of its own behavior

---

## 📖 Citation

If you use this implementation, please cite:

### Software (Zenodo)
Süleymanoğlu, G. C. (2026). SRTRBM-LLM-HYBRID: A Hybrid Intelligence System for Attractor Dynamics Analysis (Version 1.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.19211121

---

## License

BSD 3-Clause License  

Copyright (c) 2026 Görkem Can Süleymanoğlu
