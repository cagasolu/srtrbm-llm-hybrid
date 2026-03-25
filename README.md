# SR-TRBM + LLM Hybrid Intelligence System

A hybrid AI system combining thermodynamic generative modeling (SR-TRBM), neural refinement, and LLM-based multimodal interpretation.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19211496.svg)](https://doi.org/10.5281/zenodo.19211496)

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![CUDA](https://img.shields.io/badge/CUDA-enabled-green)
![Python](https://img.shields.io/badge/python-3.10-blue)

> The model generates. The system interprets. The pipeline understands.

---

## 📦 Artifacts

Pretrained artifacts are available on Zenodo:

👉 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19211602.svg)](https://doi.org/10.5281/zenodo.19211602)

⚠️ Note: The complete setup, including the GitHub source code and pretrained artifacts, requires ~190 MB of disk space.

---

## 🚀 Overview

This project implements a hybrid intelligence pipeline that brings together:

* Energy-based generative modeling (SR-TRBM)
* CNN-based structural refinement
* Multimodal LLM interpretation

Instead of treating low diversity as a failure, we look at it differently —
it can actually indicate **attractor-driven convergence**.

---

## 📚 Theoretical Origin

The SR-TRBM model used in this project originates from prior work:

👉 https://github.com/cagasolu/sr-trbm

The associated arXiv publication can be regarded as the theoretical precursor to this system.

---

## 🧠 Core Idea

The system analyzes generative behavior through:

* Attractor dynamics
* Convergence behavior
* Structured diversity
* Energy landscape properties

This allows us to distinguish between:

* ❌ Mode collapse
* ✅ Attractor-driven refinement

---

## ⚙️ System Topology

```
Latent Space → SR-TRBM → Generated Samples
                          ↓
                     Cluster Refinement
                          ↓
               LLM Multimodal Interpretation
                          ↓
           Attractor / Convergence Analysis
```

---

## 📦 Project Structure

```
Core
├── srtrbm_project_core.py        # main pipeline (generation + metrics)

LLM Integration
├── openai/
│   └── call_openai.py            # multimodal LLM evaluation

Refinement
├── supplement/
│   └── cluster.py                # structural refinement

Analysis & Diagnostics
├── analysis/
│   └── AutoGPU.py                # GPU orchestration

Visualization
├── graphs/
│   ├── SrtrbmEnergy.py
│   ├── SrtrbmMetrics.py
│   └── SrtrbmVisualization.py

Correction Modules
├── correction/
│   └── NO.py

Artifacts
├── zeta_mnist_hybrid.pt
├── stan.dgts
```

<details>
<summary>Full directory structure</summary>

```
.
├── analysis
├── correction
├── graphs
├── openai
├── supplement
├── srtrbm_project_core.py
├── stan.dgts
└── zeta_mnist_hybrid.pt
```

</details>

---

## 🔍 LLM Integration (OpenAI)

This system uses an **OpenAI LLM (multimodal)** as an interpretation layer.

The LLM is not used for generation, but for **analyzing the behavior of the model**.

---

### 🧠 Why use an LLM?

Traditional metrics (MSE, entropy, etc.) are not always enough to understand:

* whether the system is collapsing
* whether diversity is meaningful
* whether outputs are structurally consistent

The LLM provides:

* semantic interpretation
* pattern-level reasoning
* comparison between visual outputs

In short:

> The model generates.
> The LLM interprets.

---

### 🔑 API Key Requirement

To run the LLM analysis, you must provide an OpenAI API key.

Set it using:

```bash
export OPENAI_API_KEY=your_key_here
```

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
* stabilized ("perfect") outputs

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

### ⚠️ Note

* The LLM is used only for **analysis and interpretation**
* It does not influence training (yet)
* Future versions may include **LLM-driven control loops**

---

## 🧬 Key Insight

High repetition is not always a problem.

In many cases, it actually means the system has learned a strong internal structure.

```
multiple stochastic variations → converge → stable prototypes
```

---

## 🔒 Model & Training Policy

This repository provides access to the **trained model and inference pipeline**.

However, the **full training process and optimization details are not publicly released**.

### Available

* ✅ Pre-trained model weights
* ✅ Inference / generation pipeline
* ✅ LLM-based multimodal analysis

### Not Included

* ❌ Full training procedure
* ❌ Optimization strategies
* ❌ Internal training dynamics

---

## 🧠 Model Overview

The system is built on a custom embedding-based neural architecture, including:

* Residual convolutional blocks
* Metric learning (similarity-based loss)
* Embedding normalization
* Nearest-neighbor reasoning

Training involves:

* Data augmentation (affine + radial attention)
* Metric learning constraints
* Dataset correction loop
* Iterative refinement and fine-tuning

---

## ⚡ Installation

```bash
pip install -r requirements.txt
```

---

## 🔑 Setup

```bash
export OPENAI_API_KEY=your_key_here
```

---

## ▶️ Run

```bash
python3 srtrbm_project_core.py
```

---

## 🧪 What Makes This Different?

This system is not just measuring performance.

It introduces:

* An LLM-based interpretation layer
* A combination of physics, perception, and reasoning
* A shift from:

  error-based evaluation
  → to dynamic system analysis

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
* and starts to **understand its own behavior**

---

## 📖 Citation

If you use this implementation, please cite:

### Software (Zenodo)
Süleymanoğlu, G. C. (2026). SRTRBM-LLM-HYBRID: A Hybrid Intelligence System for Attractor Dynamics Analysis (Version 1.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.19211496

---

## License

BSD 3-Clause License  

Copyright (c) 2026 Görkem Can Süleymanoğlu
