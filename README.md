# SR-TRBM + LLM Hybrid Intelligence System

A hybrid intelligence system that integrates thermodynamic energy-based generation (SR-TRBM), deterministic neural refinement, and LLM-driven multimodal interpretation within a closed-loop framework.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19211121.svg)](https://doi.org/10.5281/zenodo.19211121)

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![CUDA](https://img.shields.io/badge/CUDA-enabled-green)
![Python](https://img.shields.io/badge/python-3.10-blue)

> The model generates. The system interprets. The pipeline understands.

---

## 📦 Artifacts

The following artifacts are included to support training, evaluation, and reproducibility of the experimental pipeline:

- **`stan.7z`**  
  Compressed dataset archive. After extraction, it provides `stan.dgts`, which is used as the primary numerical dataset.

- **`zeta_mnist_hybrid.pt`**  
  Pretrained model checkpoint used for inference, sampling, and energy-based analysis.

### 🔧 Usage

Extract the dataset before running experiments:

```bash
7z x stan.7z
```

Alternatively, you can use WinRAR or other compatible tools. The model's entire size is approximately 190 MB.

---

## 🚀 Overview

This project implements a hybrid intelligence pipeline that brings together:

* Energy-based generative modeling (SR-TRBM)
* CNN-based structural refinement
* Multimodal LLM interpretation

At the core of the system, a Boltzmann machine learns geometric representations of real-world objects. Instead of relying on large-scale data, it focuses on capturing structure from meaningful inputs observed on the fly, using energy-based dynamics to reach stable representations.

In parallel, the learning process is continuously observed by a multimodal LLM (OpenAI GPT-5). The goal here is not generation but interpretation—translating what the model learns into something understandable, while also tracking convergence behavior and structural consistency.

Rather than a simple pipeline, this forms a feedback loop. Generation, refinement, and interpretation interact with each other, making the system behave more like a dynamic process than a static model.

Our model is an evolving system. The broader goal is to scale the system into a larger and more efficient architecture. We aim to organize real-world data at a lower cost while maintaining strong interpretability.

The code is shared in a semi-open structure, and the use of different parts of the system is encouraged within the scope of the provided license and legal framework.

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
Latent Space → SR-TRBM → Generated Samples → Conclusion
↓
SR-TRBM
↓
Structural Refinement
↓
Metrics Extraction + LLM
↓
Generated Samples
↓
Interpretation → Convergence Analysis
↓
Epistemic Control (implicit)
↓
Conclusion
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
│   │   Fault-tolerant OpenAI client with retry and fallback handling.
│   │
│   ├── gateway.py
│   │   Core interpretation engine combining metrics, images, and perceptual similarity (LPIPS)
│   │   into structured semantic analysis.
│   │
│   ├── philosophy.py
│   │   Epistemic controller enabling probabilistic, uncertainty-driven decision making.
│   │
│   └── progress.py
│       The decision layer translates LLM outputs into actionable control signals using
│       policies, constraints, and multimodal evaluation.

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
│       GPU energy tracking via NVML, monitoring real-time power usage and estimating
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

GPT Configuration
├── yaml/
│   ├── acting.yaml
│   │   Defines epistemic control policies that translate uncertainty into
│   │   probabilistic system interventions.
│   │
│   └── perception.yaml
│       Specifies structural reasoning rules for interpreting model outputs and
│       distinguishing collapse from structured convergence.

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

## 🔍 LLM Integration (OpenAI)

This system uses an **OpenAI LLM (multimodal)** as an interpretation layer.

The LLM is not used for generation, but for **analyzing the behavior of the model**.

---

### 🧠 Why use an LLM?

🔍 LLM Integration (OpenAI)

An OpenAI multimodal LLM acts as an interpretive observer, analyzing the model’s behavior beyond traditional metrics. It evaluates collapse, diversity, and structural consistency through semantic and pattern-level reasoning.

In short:

The model generates. The LLM interprets.

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
* ✅ Inference/generation pipeline
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
Süleymanoğlu, G. C. (2026). SRTRBM-LLM-HYBRID: A Hybrid Intelligence System for Attractor Dynamics Analysis (Version 1.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.19211121

---

## License

BSD 3-Clause License  

Copyright (c) 2026 Görkem Can Süleymanoğlu
