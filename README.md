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

This project implements MYRA (Model-Yielded Reasoning Architecture)—a hybrid intelligence system that integrates energy-based generation, structural self-refinement, and LLM-assisted interpretation within a closed-loop framework. At its core, MYRA incorporates a thermodynamically regulated Boltzmann machine (SR-TRBM) as its generative engine. This component learns structured representations by organizing data within an energy landscape, converging toward stable configurations through stochastic dynamics. MYRA is a post-processing module and the governing architecture of the system. It defines how generated states are evaluated, refined, and interpreted.

The system operates as a continuous feedback loop:

- MYRA generates (via SR-TRBM)
- Energy constraints guide the state space.
- MYRA refines structural consistency.
- The LLM interprets system behavior.

This results in a unified intelligence framework where generation, refinement, and interpretation are not separate stages but tightly coupled processes. The project is a semi-open, complex AI system designed to learn from real-world data through the interaction of thermodynamic modeling, structural reasoning, and external interpretive intelligence.

---

## 📚 Theoretical Origin

The SR-TRBM model used in this project originates from prior work:

👉 https://github.com/cagasolu/sr-trbm

The associated arXiv publication can be regarded as the theoretical precursor to this system.

---

## 🧠 Core Philosophy

> **“What did the model actually learn?”**

This project is built to answer that question.

Unlike modern LLMs that focus on completing human tasks, MYRA focuses on **understanding learned structure**—not just output.

It aims at the root of artificial intelligence: revealing what a model has actually learned, not what it can convincingly generate.

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
├── llmeS/
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
├── llmeS
├── supplement
├── yaml
├── srtrbm_project_core.py
├── stan.dgts
└── zeta_mnist_hybrid.pt
```

</details>

---

## 🔍 LLM Integration (Claude 3.5 Sonnet (Anthropic), Gemini 1.5 Pro (Google), Llama 3.1 (Meta), Mistral Large 2 (Mistral AI), DeepSeek-V3, Grok-2 (xAI), Command R+ (Cohere), Qwen 2.5 (Alibaba))

MYRA incorporates a multimodal LLM as an external interpretive layer.

The LLM is not used for generation. 

Its role is to analyze the system behavior under explicit structural and epistemic constraints.

---

# ⚠️ Model Compatibility & Performance Constraints

While **MYRA** supports a wide range of LLMs, internal benchmarks indicate significant variance in **confidence scoring** and **epistemic reliability** across different architectures.

---

## 📉 Critical Observations on GPT-Series Models

Extensive testing shows that **OpenAI GPT models may not be optimal** for the rigorous structural auditing required by MYRA:

- **Confidence Instability**  
  GPT models often exhibit overconfidence on diagnostic tasks, failing to maintain stable probability calibration when analyzing complex structural anomalies.

- **Heuristic Bias**  
  These models tend to default to generic, "safe" explanations rather than identifying **precise mathematical deviations** within RBM layers.

- **Deterministic Limitations**  
  Limited access to internal logit distributions in proprietary APIs restricts the depth of interpretability and fine-grained behavioral analysis.

---

> [!CAUTION]  
> **Reference Audit Notice**  
> The provided `run.log` file was generated using a GPT model via OpenAI API.
> Although it serves as a functional reference point, using other LLMs is recommended due to known **confidence interval issues**.

---

### 🧠 Role of the LLM

Within MYRA, the LLM functions as a constraint-aware evaluator, combining:

- quantitative metrics (energy, diversity, similarity)
- structural signals (refined outputs)
- multimodal inputs (image + statistics)

It performs system-level interpretation by:

- evaluating convergence dynamics
- distinguishing collapse vs. structured convergence
- assessing structural consistency across samples
- reasoning over attractor behavior and energy organization

Importantly, interpretation is not free-form.

It is guided by strict principles defined in the system:

- structural similarity overrides weak diversity signals
- refinement reveals true structural model
- collapse requires simultaneous structural degradation

These constraints ensure that the LLM behaves as a bounded reasoning component, not a generative oracle.

---

### 🔁 Epistemic Constraint

The LLM’s confidence is explicitly limited by available evidence:

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

High repetition is not always a problem. In many cases, it actually means the system has learned a strong internal structure.

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

### ⚠️ Partially Exposed

These components are visible at a high level but not fully disclosed:

- Training objectives and loss shaping strategy
- Energy scheduling and thermodynamic control logic
- Refinement dynamics during training vs inference
- Internal calibration of metrics and thresholds
- LLM prompting strategy and constraint tuning

---

### ❌ Not Included

The following are intentionally withheld:

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
