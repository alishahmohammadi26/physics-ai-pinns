# Track 1: Physics-Based AI — Solving ODEs, Fluid Dynamics & Reaction Engineering

> **Author:** Ali Shahmohammadi, Ph.D.  
> **Portfolio:** [alishahmohammadi22.github.io](https://alishahmohammadi22.github.io)  
> **Track:** 1 of 6 — Physics-Based AI

---

## Overview

This track is a structured, bottom-up series covering **Physics-Informed Neural Networks (PINNs)** and physics-based AI methods applied to chemical engineering, fluid dynamics, and reaction engineering. It starts from first principles and builds toward real-world industrial applications.

No prior knowledge of PINNs is required — each module includes conceptual explanations, derivations, and hands-on Jupyter notebooks.

---

## Modules

| # | Topic | Status |
|---|-------|--------|
| 01 | Introduction to Physics-Informed Neural Networks (PINNs) | ✅ In progress |
| 02 | Solving simple ODEs with neural networks: step-by-step | 🔲 Planned |
| 03 | Key tools and frameworks: NVIDIA Modulus for physics-based AI | 🔲 Planned |
| 04 | Boundary conditions and constraints in PINNs: best practices | 🔲 Planned |
| 05 | Intermediate applications: heat transfer with physics-informed nets | 🔲 Planned |
| 06 | Fluid dynamics and Navier–Stokes equations: solving complex PDEs with AI | 🔲 Planned |
| 07 | Real-world example: predicting fluid flow in microreactors with PINNs | 🔲 Planned |
| 08 | Combining physics-based AI with traditional CFD: a hybrid approach | 🔲 Planned |
| 09 | Reaction kinetics: using neural networks to model chemical reactions | 🔲 Planned |
| 09 | Solving stiff ODEs in reaction engineering with AI | 🔲 Planned |
| 10 | Stiff ODEs in reaction engineering with AI | 🔲 Planned |
| 11 | Inverse problems: using PINNs to infer reaction parameters from data | 🔲 Planned |
| 12 | AI for multiphase flow modeling: applications in chemical engineering | 🔲 Planned |
| 13 | Applying PINNs to optimize reaction yields in catalytic processes | 🔲 Planned |
| 14 | Best practices for training and validating physics-informed models | 🔲 Planned |
| 15 | Tools comparison: NVIDIA Modulus vs. DeepXDE vs. PyDEns | 🔲 Planned |
| 16 | Integrating experimental data with physics-based AI models | 🔲 Planned |
| 17 | Addressing scalability: large-scale systems with PINNs | 🔲 Planned |
| 18 | Combining physics-based models with generative AI for predictive simulations | 🔲 Planned |
| 19 | Case study: fluid dynamics optimization with AI in bioreactors | 🔲 Planned |
| 20 | Future trends: where physics-based AI is heading in reaction engineering | 🔲 Planned |

---

## Environment Setup

This track uses a Python virtual environment managed by **[uv](https://github.com/astral-sh/uv)** — a fast Python package manager.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create the virtual environment

```bash
cd projects/Track1_Physics_AI
uv venv --python 3.11
```

### 3. Install dependencies

```bash
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

uv pip install -r requirements.txt
```

### 4. Register the kernel for Jupyter

```bash
python -m ipykernel install --user \
  --name track1-physics-ai \
  --display-name "Python (Track1 — Physics AI)"
```

### 5. Launch Jupyter

```bash
# Option A — launch via uv (no manual activation needed)
uv run jupyter lab

# Option B — standard launch after activation
jupyter lab
```

In VS Code, open any `.ipynb` notebook, click the kernel picker in the top-right corner, and select **Python (Track1 — Physics AI)**.

### 6. Verify the environment

```python
import sys, torch, numpy, matplotlib
print("Python :", sys.executable)
print("PyTorch:", torch.__version__)
print("NumPy  :", numpy.__version__)
print("Matplotlib:", matplotlib.__version__)
```

---

## Key Concepts Covered

- **Physics-Informed Neural Networks (PINNs)** — encoding differential equations directly into the training objective
- **Automatic differentiation** — computing exact derivatives through the network's computational graph
- **Governing equations** — heat equation, Navier–Stokes, wave equation, reaction–diffusion systems
- **Forward problems** — solving PDEs without a mesh
- **Inverse problems** — inferring unknown physical parameters from sparse observations
- **Boundary and initial conditions** — soft and hard enforcement strategies
- **Collocation-point sampling** — adaptive and residual-based strategies
- **Loss function design** — balancing physics, boundary, initial, and data terms
- **Hybrid methods** — combining PINNs with traditional CFD solvers

---

## Stack

| Tool | Purpose |
|------|---------|
| `torch` | Automatic differentiation and neural-network training |
| `numpy` | Numerical arrays and preprocessing |
| `matplotlib` | Visualization of fields, losses, and comparisons |
| `jupyter` / `jupyterlab` | Interactive notebooks |
| `ipykernel` | Jupyter kernel registration |
| `uv` | Fast, reproducible environment management |

---

## Related Tracks

| Track | Topic |
|-------|-------|
| **Track 2** | Sequential Model-Based Design of Experiments with AI |
| **Track 3** | Agentic AI — Production Pipelines & Multi-Agent Systems |
| **Track 4** | Data Governance, FAIR Data & Knowledge Graphs |
| **Track 5** | Scientific ML & Regulatory AI |
| **Track 6** | Presentations & Learning Resources |
