# Building Physics-Informed Neural Networks with NVIDIA PhysicsNeMo and BioNeMo

## A CPU-First, Beginner-to-Advanced Hands-On Course

---

## Table of Contents

1. [Course Purpose](#1-course-purpose)
2. [PhysicsNeMo vs BioNeMo](#2-physicsnemo-vs-bionemo)
3. [CPU-First Learning Strategy](#3-cpu-first-learning-strategy)
4. [Complete Learning Roadmap](#4-complete-learning-roadmap)
5. [Course Folder Structure](#5-course-folder-structure)
6. [Environment Setup](#6-environment-setup)
7. [Python and PyTorch Foundations](#7-python-and-pytorch-foundations)
8. [How a PINN Works](#8-how-a-pinn-works)
9. [First PINN: Exponential Decay](#9-first-pinn-exponential-decay)
10. [Biological Interpretation: Drug Elimination](#10-biological-interpretation-drug-elimination)
11. [PhysicsNeMo Learning Track](#11-physicsnemo-learning-track)
12. [BioNeMo Learning Track](#12-bionemo-learning-track)
13. [ODE Progression](#13-ode-progression)
14. [PDE Progression](#14-pde-progression)
15. [Inverse Problems and Parameter Estimation](#15-inverse-problems-and-parameter-estimation)
16. [Biological PINN Projects](#16-biological-pinn-projects)
17. [Advanced PINN Methods](#17-advanced-pinn-methods)
18. [Integrated PhysicsNeMo and BioNeMo Projects](#18-integrated-physicsnemo-and-bionemo-projects)
19. [Recommended Study Workflow](#19-recommended-study-workflow)
20. [What to Build First](#20-what-to-build-first)

---

# 1. Course Purpose

This course teaches you how to build Physics-Informed Neural Networks step by step using:

- Python
- PyTorch
- NVIDIA PhysicsNeMo
- NVIDIA BioNeMo concepts and biological model integrations

The course starts with simple mathematics and small neural networks and gradually progresses toward:

- first-order ODEs
- nonlinear ODEs
- second-order ODEs
- coupled ODE systems
- stiff systems
- chaotic systems
- PDEs
- coupled PDEs
- inverse problems
- parameter estimation
- biological systems
- pharmacokinetics
- reaction kinetics
- gene regulation
- diffusion in tissue
- tumor growth models
- molecularly conditioned PINNs
- neural operators

The main goal is not only to run code. The goal is to understand:

1. what each equation means;
2. why the neural network is structured in a particular way;
3. how automatic differentiation is used;
4. how physics becomes part of the loss function;
5. how to diagnose training failure;
6. how to improve accuracy;
7. how to estimate unknown physical or biological parameters.

---

# 2. PhysicsNeMo vs BioNeMo

## NVIDIA PhysicsNeMo

PhysicsNeMo is NVIDIA's scientific machine learning framework for physics-based and engineering applications.

We will use PhysicsNeMo for:

- Physics-Informed Neural Networks
- ODE solving
- PDE solving
- inverse problems
- parameter estimation
- geometry-based sampling
- neural operators
- scientific machine learning workflows

PhysicsNeMo is the main NVIDIA framework used in this course for PINNs.

## NVIDIA BioNeMo

BioNeMo is NVIDIA's framework for biomolecular and life-science AI.

BioNeMo is designed for areas such as:

- proteins
- molecules
- genes
- drug discovery
- molecular generation
- biological foundation models
- biological embeddings
- biomolecular representation learning

BioNeMo is not a general-purpose PINN solver.

Instead, it can be integrated with PINNs by providing:

- molecular representations
- protein representations
- learned biological features
- estimates of molecular properties
- parameters for mechanistic models
- conditioning variables for PINNs

## How the two frameworks fit together

A possible integrated workflow is:

```text
Protein or molecule
        ↓
BioNeMo representation
        ↓
Estimated biological or chemical properties
        ↓
PhysicsNeMo PINN
        ↓
ODE or PDE prediction
```

Another example is:

```text
Molecular structure
        ↓
BioNeMo embedding
        ↓
Predicted kinetic parameters
        ↓
Pharmacokinetic PINN
        ↓
Drug concentration over time
```

---

# 3. CPU-First Learning Strategy

The entire foundational course will be designed to run on CPU.

## CPU-supported components

| Component | CPU support |
|---|---:|
| Basic PyTorch neural networks | Yes |
| Basic PINNs | Yes |
| First-order ODE PINNs | Yes |
| Second-order ODE PINNs | Yes |
| Coupled ODE systems | Yes |
| Small PDE PINNs | Yes |
| Inverse parameter estimation | Yes |
| Adaptive sampling demonstrations | Yes |
| PhysicsNeMo symbolic examples | Usually yes for small educational problems |
| Biological ODE systems | Yes |
| Lightweight molecular descriptors | Yes |
| Small custom biological models | Yes |
| Large BioNeMo foundation model training | No, GPU required in practice |
| Large BioNeMo model fine-tuning | No, GPU required in practice |

## Course policy

Every required lesson will have a CPU-compatible version.

Optional GPU exercises will be clearly labeled.

A GPU will improve training speed but will not be required for the core course.

---

# 4. Complete Learning Roadmap

## Part I: Foundations

### Module 1: Python Fundamentals

You will learn:

- variables
- numbers
- lists
- dictionaries
- loops
- functions
- classes
- modules
- error handling

### Module 2: NumPy Fundamentals

You will learn:

- arrays
- shapes
- indexing
- broadcasting
- vectorized calculations
- numerical functions

### Module 3: PyTorch Fundamentals

You will learn:

- tensors
- tensor shapes
- CPU device selection
- gradients
- autograd
- layers
- neural networks
- loss functions
- optimizers

### Module 4: Differential Equation Foundations

You will learn:

- ODEs
- PDEs
- independent variables
- dependent variables
- initial conditions
- boundary conditions
- analytical solutions
- numerical solutions
- residuals

---

## Part II: PINNs from First Principles

### Module 5: First Neural Network

Supervised learning example:

\[
y = 2x + 1
\]

### Module 6: First PINN

Solve:

\[
\frac{du}{dt} + u = 0,
\qquad u(0)=1
\]

### Module 7: Nonlinear ODE PINN

Solve logistic growth:

\[
\frac{dN}{dt} = rN\left(1-\frac{N}{K}\right)
\]

### Module 8: Second-Order ODE PINN

Solve:

\[
m\frac{d^2x}{dt^2}+c\frac{dx}{dt}+kx=0
\]

### Module 9: Coupled ODE Systems

Examples:

- Lotka-Volterra
- SIR
- SEIR
- pharmacokinetics
- enzyme kinetics
- gene regulation

---

## Part III: NVIDIA PhysicsNeMo

### Module 10: PhysicsNeMo Fundamentals

You will learn:

- PhysicsNeMo package structure
- symbolic equations
- coordinate variables
- field variables
- residual definitions
- geometry
- collocation sampling
- automatic differentiation

### Module 11: PhysicsNeMo ODEs

You will solve:

- exponential decay
- logistic growth
- second-order oscillators
- coupled biological systems

### Module 12: PhysicsNeMo PDEs

You will solve:

- Poisson equation
- heat equation
- wave equation
- Burgers equation
- reaction-diffusion systems

### Module 13: PhysicsNeMo Inverse Problems

You will estimate:

- decay rates
- diffusion coefficients
- reaction rates
- source terms
- unknown initial conditions
- unknown boundary conditions

---

## Part IV: NVIDIA BioNeMo

### Module 14: Biological Data Foundations

You will learn:

- protein sequences
- amino-acid tokenization
- SMILES strings
- molecular descriptors
- gene-expression matrices
- biological embeddings

### Module 15: BioNeMo Concepts

You will learn:

- biomolecular foundation models
- pretrained biological models
- protein embeddings
- molecular embeddings
- biological conditioning variables

### Module 16: BioNeMo and PINN Integration

You will connect biological representations with mechanistic models.

Examples:

- molecule-conditioned pharmacokinetic PINNs
- protein-conditioned kinetic models
- learned reaction-rate estimation
- compound-specific diffusion models

---

## Part V: Advanced ODE Systems

### Module 17: Stiff Systems

Examples:

- Robertson kinetics
- chemical reaction networks
- multi-timescale biological systems

### Module 18: Chaotic Systems

Examples:

- Lorenz system
- nonlinear oscillators
- long-time instability

### Module 19: Time-Domain Decomposition

You will learn:

- time windows
- curriculum training
- causal PINNs
- sequential training

---

## Part VI: Advanced PDEs

### Module 20: Two-Dimensional PDEs

Examples:

- 2D Poisson
- 2D heat equation
- Helmholtz equation
- advection-diffusion

### Module 21: Fluid Mechanics

Examples:

- incompressible Navier-Stokes
- channel flow
- lid-driven cavity
- flow around an obstacle

### Module 22: Coupled Biological PDEs

Examples:

- reaction-diffusion
- tissue transport
- tumor growth
- nutrient diffusion
- drug penetration

---

## Part VII: Advanced PINN Techniques

### Module 23: Adaptive Sampling

You will learn:

- uniform random sampling
- Latin hypercube sampling
- Sobol sampling
- residual-based adaptive refinement
- boundary-focused sampling

### Module 24: Loss Balancing

You will learn:

- manual loss weights
- adaptive weights
- gradient balancing
- uncertainty weighting

### Module 25: Optimization

You will compare:

- Adam
- AdamW
- L-BFGS
- learning-rate schedules
- gradient clipping

### Module 26: Neural Operators

You will study:

- DeepONet
- Fourier Neural Operator
- physics-informed neural operators

---

# 5. Course Folder Structure

```text
nvidia_physicsnemo_bionemo_pinn_course/
│
├── 00_environment/
│   ├── 00_check_python.ipynb
│   ├── 01_check_pytorch_cpu.ipynb
│   ├── 02_check_physicsnemo.ipynb
│   └── 03_bionemo_overview.ipynb
│
├── 01_pytorch_foundations/
│   ├── 01_tensors.ipynb
│   ├── 02_autograd.ipynb
│   ├── 03_neural_networks.ipynb
│   └── 04_optimization.ipynb
│
├── 02_basic_odes/
│   ├── 01_exponential_decay.ipynb
│   ├── 02_logistic_growth.ipynb
│   ├── 03_second_order_ode.ipynb
│   └── 04_coupled_odes.ipynb
│
├── 03_physicsnemo/
│   ├── 01_symbolic_equations.ipynb
│   ├── 02_first_ode.ipynb
│   ├── 03_heat_equation.ipynb
│   ├── 04_burgers_equation.ipynb
│   └── 05_inverse_problem.ipynb
│
├── 04_bionemo_foundations/
│   ├── 01_biological_sequences.ipynb
│   ├── 02_smiles_and_molecules.ipynb
│   ├── 03_protein_embeddings.ipynb
│   ├── 04_molecular_features.ipynb
│   └── 05_bionemo_architecture.ipynb
│
├── 05_biological_pinns/
│   ├── 01_pharmacokinetics.ipynb
│   ├── 02_enzyme_kinetics.ipynb
│   ├── 03_gene_regulation.ipynb
│   ├── 04_reaction_diffusion.ipynb
│   └── 05_tumor_growth.ipynb
│
├── 06_inverse_biology/
│   ├── 01_estimate_elimination_rate.ipynb
│   ├── 02_estimate_diffusivity.ipynb
│   ├── 03_estimate_reaction_rates.ipynb
│   └── 04_unknown_source.ipynb
│
├── 07_advanced_methods/
│   ├── 01_adaptive_sampling.ipynb
│   ├── 02_loss_balancing.ipynb
│   ├── 03_domain_decomposition.ipynb
│   ├── 04_uncertainty.ipynb
│   └── 05_neural_operators.ipynb
│
└── 08_integrated_projects/
    ├── 01_molecule_conditioned_pinn.ipynb
    ├── 02_drug_tissue_transport.ipynb
    ├── 03_pk_parameter_estimation.ipynb
    └── 04_multiscale_biological_model.ipynb
```

---

# 6. Environment Setup

## Recommended Python version

Use Python 3.11.

## Create a virtual environment

### Linux or macOS

```bash
python3.11 -m venv nvidia-bio-pinn-cpu
source nvidia-bio-pinn-cpu/bin/activate
```

### Windows PowerShell

```powershell
py -3.11 -m venv nvidia-bio-pinn-cpu
nvidia-bio-pinn-cpu\Scripts\Activate.ps1
```

## Upgrade pip

```bash
python -m pip install --upgrade pip
```

## Install CPU PyTorch

```bash
pip install torch torchvision torchaudio
```

## Install scientific Python packages

```bash
pip install numpy scipy sympy pandas matplotlib scikit-learn jupyterlab
```

## Install PhysicsNeMo symbolic support

```bash
pip install "nvidia-physicsnemo[sym]"
```

Do not install CUDA-specific extras for the CPU-only course.

## Start JupyterLab

```bash
jupyter lab
```

---

## CPU environment verification

Create a file named `check_cpu_environment.py`.

```python
import platform
import sys

import numpy as np
import sympy as sp
import torch


def print_environment() -> None:
    print("=" * 60)
    print("CPU PINN environment")
    print("=" * 60)

    print(f"Python:       {sys.version}")
    print(f"Platform:     {platform.platform()}")
    print(f"Processor:    {platform.processor()}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"NumPy:        {np.__version__}")
    print(f"SymPy:        {sp.__version__}")

    print(f"CUDA present: {torch.cuda.is_available()}")

    device = torch.device("cpu")
    print(f"Course device: {device}")

    x = torch.tensor(
        [[1.0], [2.0], [3.0]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    y = x**2

    dy_dx = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
    )[0]

    print("\nAutograd test")
    print("x:")
    print(x)

    print("\ny = x²:")
    print(y)

    print("\ndy/dx:")
    print(dy_dx)

    expected = 2.0 * x

    if torch.allclose(dy_dx, expected):
        print("\nEnvironment test passed.")
    else:
        raise RuntimeError("Autograd test failed.")


if __name__ == "__main__":
    print_environment()
```

Run it:

```bash
python check_cpu_environment.py
```

Expected final message:

```text
Environment test passed.
```

---

# 7. Python and PyTorch Foundations

## Tensor example

```python
import torch

x = torch.tensor(
    [[1.0], [2.0], [3.0]],
    dtype=torch.float32,
)

print(x)
print(x.shape)
```

The shape is:

```text
(3, 1)
```

This means:

- three samples
- one feature per sample

## Automatic differentiation

```python
x = torch.tensor(
    [[1.0], [2.0], [3.0]],
    requires_grad=True,
)

y = x**2

first_derivative = torch.autograd.grad(
    outputs=y,
    inputs=x,
    grad_outputs=torch.ones_like(y),
    create_graph=True,
)[0]

print(first_derivative)
```

Because:

\[
y=x^2
\]

we expect:

\[
\frac{dy}{dx}=2x
\]

## Second derivative

```python
second_derivative = torch.autograd.grad(
    outputs=first_derivative,
    inputs=x,
    grad_outputs=torch.ones_like(first_derivative),
    create_graph=True,
)[0]

print(second_derivative)
```

Because:

\[
\frac{d^2y}{dx^2}=2
\]

this should return values close to 2.

---

# 8. How a PINN Works

A Physics-Informed Neural Network approximates the unknown solution of a differential equation.

Suppose the exact function is:

\[
u(t)
\]

The network approximation is:

\[
u_\theta(t)
\]

where \(\theta\) represents all network weights and biases.

## General loss function

A PINN often uses:

\[
\mathcal{L}
=
\lambda_r\mathcal{L}_{physics}
+
\lambda_{IC}\mathcal{L}_{IC}
+
\lambda_{BC}\mathcal{L}_{BC}
+
\lambda_d\mathcal{L}_{data}
\]

where:

- \(\mathcal{L}_{physics}\) measures equation violation;
- \(\mathcal{L}_{IC}\) measures initial-condition violation;
- \(\mathcal{L}_{BC}\) measures boundary-condition violation;
- \(\mathcal{L}_{data}\) measures mismatch with observations.

## Collocation points

Collocation points are input locations where the equation residual is evaluated.

They do not require known solution values.

For an ODE, these may be time points:

\[
t_1,t_2,\ldots,t_N
\]

For a PDE, these may be space-time points:

\[
(x_i,t_i)
\]

---

# 9. First PINN: Exponential Decay

We solve:

\[
\frac{du}{dt}+u=0,
\qquad u(0)=1
\]

The exact solution is:

\[
u(t)=e^{-t}
\]

## Step 1: Imports

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
```

## Step 2: Force CPU execution

```python
device = torch.device("cpu")
print("Using device:", device)
```

## Step 3: Reproducibility

```python
torch.manual_seed(42)
np.random.seed(42)
```

## Step 4: Create collocation points

```python
t_collocation = torch.linspace(
    0.0,
    5.0,
    100,
    device=device,
).reshape(-1, 1)

t_collocation.requires_grad_(True)
```

## Step 5: Define the neural network

```python
class PINN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.network(t)
```

## Step 6: Create the model

```python
model = PINN().to(device)
```

## Step 7: Derivative helper

```python
def derivative(
    output: torch.Tensor,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.autograd.grad(
        outputs=output,
        inputs=input_tensor,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]
```

## Step 8: Physics residual

```python
def physics_residual(
    model: nn.Module,
    t: torch.Tensor,
) -> torch.Tensor:
    u = model(t)
    du_dt = derivative(u, t)

    residual = du_dt + u
    return residual
```

## Step 9: Initial-condition loss

```python
t_initial = torch.tensor(
    [[0.0]],
    dtype=torch.float32,
    device=device,
)

u_initial = torch.tensor(
    [[1.0]],
    dtype=torch.float32,
    device=device,
)
```

## Step 10: Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)
```

## Step 11: Training loop

```python
loss_history = []

for epoch in range(5000):
    optimizer.zero_grad()

    residual = physics_residual(model, t_collocation)
    physics_loss = torch.mean(residual**2)

    predicted_initial = model(t_initial)
    initial_loss = torch.mean(
        (predicted_initial - u_initial) ** 2
    )

    total_loss = physics_loss + 10.0 * initial_loss

    total_loss.backward()
    optimizer.step()

    loss_history.append(total_loss.item())

    if epoch % 500 == 0:
        print(
            f"Epoch {epoch:5d} | "
            f"Total: {total_loss.item():.6e} | "
            f"Physics: {physics_loss.item():.6e} | "
            f"IC: {initial_loss.item():.6e}"
        )
```

## Step 12: Evaluate the model

```python
t_test = torch.linspace(
    0.0,
    5.0,
    200,
    device=device,
).reshape(-1, 1)

with torch.no_grad():
    u_predicted = model(t_test)

u_exact = torch.exp(-t_test)
```

## Step 13: Plot the result

```python
plt.figure(figsize=(8, 5))

plt.plot(
    t_test.cpu().numpy(),
    u_exact.cpu().numpy(),
    label="Exact",
)

plt.plot(
    t_test.cpu().numpy(),
    u_predicted.cpu().numpy(),
    "--",
    label="PINN",
)

plt.xlabel("t")
plt.ylabel("u(t)")
plt.title("Exponential Decay PINN")
plt.legend()
plt.grid(True)
plt.show()
```

## Step 14: Plot training loss

```python
plt.figure(figsize=(8, 5))
plt.semilogy(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()
```

## Step 15: Calculate error

```python
absolute_error = torch.abs(u_predicted - u_exact)
mean_absolute_error = absolute_error.mean().item()
max_absolute_error = absolute_error.max().item()

print("Mean absolute error:", mean_absolute_error)
print("Maximum absolute error:", max_absolute_error)
```

---

# 10. Biological Interpretation: Drug Elimination

The same equation can describe drug elimination.

\[
\frac{dC}{dt}+kC=0,
\qquad C(0)=C_0
\]

where:

- \(C(t)\) is drug concentration;
- \(k\) is the elimination-rate constant;
- \(C_0\) is the initial concentration.

The exact solution is:

\[
C(t)=C_0e^{-kt}
\]

For example:

\[
C_0=10,
\qquad k=0.5
\]

Then:

\[
C(t)=10e^{-0.5t}
\]

## Biological PINN residual

\[
r_\theta(t)
=
\frac{dC_\theta}{dt}
+
kC_\theta(t)
\]

## Modified residual function

```python
def drug_residual(
    model: nn.Module,
    t: torch.Tensor,
    elimination_rate: float,
) -> torch.Tensor:
    concentration = model(t)
    dconcentration_dt = derivative(concentration, t)

    return dconcentration_dt + elimination_rate * concentration
```

## Initial condition

```python
initial_concentration = 10.0
elimination_rate = 0.5
```

The initial-condition loss becomes:

```python
predicted_initial = model(t_initial)
initial_loss = torch.mean(
    (predicted_initial - initial_concentration) ** 2
)
```

---

# 11. PhysicsNeMo Learning Track

PhysicsNeMo will be introduced only after the manual PyTorch implementation is understood.

This prevents the framework from hiding the important mathematical steps.

## PhysicsNeMo topics

You will learn:

- symbolic variables
- symbolic differential equations
- PDE classes
- coordinate systems
- field variables
- residual generation
- geometry objects
- interior constraints
- boundary constraints
- initial constraints
- data constraints
- validation
- inference

## PhysicsNeMo lesson sequence

### Lesson 1: Symbolic equation definitions

You will express equations using symbolic mathematics.

### Lesson 2: First-order ODE

Rebuild exponential decay using PhysicsNeMo.

### Lesson 3: Nonlinear ODE

Rebuild logistic growth.

### Lesson 4: Second-order ODE

Build the mass-spring-damper problem.

### Lesson 5: Coupled ODE system

Build pharmacokinetic or epidemiological systems.

### Lesson 6: First PDE

Solve the one-dimensional heat equation.

### Lesson 7: Nonlinear PDE

Solve Burgers equation.

### Lesson 8: Inverse problem

Estimate an unknown coefficient.

---

# 12. BioNeMo Learning Track

BioNeMo will be introduced through biological representation learning and optional pretrained-model workflows.

## CPU-compatible BioNeMo foundations

The CPU track will include:

- amino-acid sequence processing
- protein tokenization
- SMILES processing
- molecular descriptors
- biological feature vectors
- small embedding models
- custom PyTorch biological encoders

## Optional GPU BioNeMo track

The optional GPU track may include:

- pretrained protein models
- pretrained molecular models
- BioNeMo inference
- model fine-tuning
- biological foundation-model embeddings

## CPU substitute

When a large BioNeMo model requires a GPU, we will use one of the following CPU substitutes:

- RDKit descriptors
- one-hot protein encoding
- small trainable sequence encoder
- pretrained lightweight public embedding
- manually generated molecular feature vector

The PINN integration logic remains the same.

---

# 13. ODE Progression

## ODE 1: Exponential decay

\[
\frac{du}{dt}+u=0
\]

## ODE 2: Logistic growth

\[
\frac{dN}{dt}=rN\left(1-\frac{N}{K}\right)
\]

Topics:

- nonlinear residuals
- carrying capacity
- long-term behavior
- extrapolation
- parameter sensitivity

## ODE 3: Harmonic oscillator

\[
\frac{d^2x}{dt^2}+\omega^2x=0
\]

## ODE 4: Damped oscillator

\[
m\frac{d^2x}{dt^2}+c\frac{dx}{dt}+kx=0
\]

## ODE 5: Lotka-Volterra

\[
\frac{dx}{dt}=\alpha x-\beta xy
\]

\[
\frac{dy}{dt}=\delta xy-\gamma y
\]

## ODE 6: SIR model

\[
\frac{dS}{dt}=-\beta\frac{SI}{N}
\]

\[
\frac{dI}{dt}=\beta\frac{SI}{N}-\gamma I
\]

\[
\frac{dR}{dt}=\gamma I
\]

## ODE 7: Two-compartment pharmacokinetics

\[
\frac{dC_1}{dt}
=
-k_{10}C_1-k_{12}C_1+k_{21}C_2
\]

\[
\frac{dC_2}{dt}
=
k_{12}C_1-k_{21}C_2
\]

## ODE 8: Michaelis-Menten kinetics

\[
\frac{dS}{dt}
=
-\frac{V_{max}S}{K_m+S}
\]

## ODE 9: Gene regulation

\[
\frac{dm}{dt}
=
\frac{\alpha}{1+(p/K)^n}
-
\gamma_m m
\]

\[
\frac{dp}{dt}
=
\beta m-\gamma_p p
\]

## ODE 10: Stiff reaction system

Example:

- Robertson chemical kinetics
- fast and slow reactions
- multiple time scales

---

# 14. PDE Progression

## PDE 1: Poisson equation

\[
\frac{d^2u}{dx^2}=f(x)
\]

## PDE 2: Heat equation

\[
\frac{\partial u}{\partial t}
=
\alpha\frac{\partial^2u}{\partial x^2}
\]

## PDE 3: Wave equation

\[
\frac{\partial^2u}{\partial t^2}
=
c^2\frac{\partial^2u}{\partial x^2}
\]

## PDE 4: Burgers equation

\[
u_t+uu_x-\nu u_{xx}=0
\]

## PDE 5: Reaction-diffusion

\[
\frac{\partial C}{\partial t}
=
D\frac{\partial^2C}{\partial x^2}
-kC
\]

## PDE 6: Fisher-KPP equation

\[
\frac{\partial u}{\partial t}
=
D\nabla^2u
+
ru(1-u)
\]

## PDE 7: Coupled tissue model

\[
\frac{\partial n}{\partial t}
=
D_n\nabla^2n
+
rn\left(1-\frac{n}{K}\right)
-
\gamma Cn
\]

\[
\frac{\partial C}{\partial t}
=
D_C\nabla^2C-k_CC
\]

## PDE 8: Navier-Stokes

The advanced course will include incompressible fluid equations.

---

# 15. Inverse Problems and Parameter Estimation

In a forward problem, the physical parameters are known.

In an inverse problem, one or more parameters are unknown and learned from data.

## Example: unknown decay rate

\[
\frac{du}{dt}=-ku
\]

The parameter \(k\) is unknown.

## Trainable parameter

```python
raw_k = torch.nn.Parameter(
    torch.tensor(0.0, dtype=torch.float32)
)
```

To force the learned value to stay positive:

```python
k = torch.nn.functional.softplus(raw_k)
```

## Optimizer with model and physical parameter

```python
optimizer = torch.optim.Adam(
    list(model.parameters()) + [raw_k],
    lr=1e-3,
)
```

## Data loss

Suppose observations are available:

```python
t_data = torch.tensor(
    [[0.0], [1.0], [2.0], [3.0]],
    dtype=torch.float32,
)

u_data = torch.tensor(
    [[1.0], [0.61], [0.37], [0.22]],
    dtype=torch.float32,
)
```

The data loss is:

```python
prediction_data = model(t_data)
data_loss = torch.mean((prediction_data - u_data) ** 2)
```

## Total inverse-problem loss

```python
total_loss = (
    physics_loss
    + 10.0 * initial_loss
    + 5.0 * data_loss
)
```

## Inverse problems covered

- unknown decay rate
- unknown logistic growth rate
- unknown carrying capacity
- unknown pharmacokinetic parameters
- unknown diffusion coefficient
- unknown wave speed
- unknown viscosity
- unknown reaction rates
- unknown source term
- unknown spatially varying coefficient

---

# 16. Biological PINN Projects

## Project 1: Drug elimination

Estimate the elimination rate from sparse concentration data.

## Project 2: Two-compartment pharmacokinetics

Estimate:

- central elimination
- intercompartmental exchange
- concentration dynamics

## Project 3: Michaelis-Menten kinetics

Estimate:

- \(V_{max}\)
- \(K_m\)

## Project 4: Gene regulation

Predict:

- messenger RNA
- protein concentration
- regulatory parameters

## Project 5: Drug diffusion in tissue

Estimate:

- diffusion coefficient
- elimination coefficient
- source location

## Project 6: Tumor growth

Model:

- tumor density
- nutrient concentration
- drug concentration
- proliferation
- treatment response

## Project 7: Epidemiological systems

Estimate:

- transmission rate
- recovery rate
- hidden infected population

## Project 8: Reaction network inference

Estimate unknown rates in a multi-reaction biological system.

---

# 17. Advanced PINN Methods

## Adaptive sampling

Training points are concentrated where residuals are large.

Procedure:

1. train an initial PINN;
2. evaluate residuals over a dense candidate set;
3. find high-residual regions;
4. add new collocation points;
5. continue training.

## Loss balancing

PINNs may fail when one loss term dominates.

We will study:

- static weights
- adaptive weights
- normalized gradients
- uncertainty weighting

## Input normalization

Inputs may be mapped to:

\[
[-1,1]
\]

This often improves optimization.

## Nondimensionalization

Physical quantities with very different scales can be transformed into dimensionless quantities.

## Hard constraints

Instead of penalizing an initial condition, it can be built into the network output.

For example:

\[
u_\theta(t)=1+tN_\theta(t)
\]

This automatically satisfies:

\[
u_\theta(0)=1
\]

## Fourier features

Fourier features help networks represent high-frequency behavior.

## Domain decomposition

Large spatial or temporal domains can be divided into subdomains.

## Neural operators

Neural operators learn mappings between functions rather than solving only one equation instance.

---

# 18. Integrated PhysicsNeMo and BioNeMo Projects

## Integrated Project 1: Molecule-conditioned pharmacokinetics

Inputs:

- molecular descriptor or BioNeMo embedding
- time

Outputs:

- drug concentration
- predicted kinetic parameters

Physics constraint:

\[
\frac{dC}{dt}+kC=0
\]

The value of \(k\) depends on the molecular representation.

## Integrated Project 2: Protein-conditioned enzyme kinetics

Inputs:

- protein representation
- substrate concentration
- time

Outputs:

- reaction dynamics
- estimated kinetic constants

## Integrated Project 3: Drug diffusion through tissue

Inputs:

- compound representation
- space
- time

Outputs:

- concentration field
- effective diffusion coefficient
- removal rate

## Integrated Project 4: Multi-compound comparison

The model compares several compounds using biological or chemical feature vectors.

## Integrated Project 5: Biological foundation model plus PINN

Optional GPU workflow:

```text
BioNeMo pretrained model
        ↓
Biomolecular embedding
        ↓
Conditioned PhysicsNeMo model
        ↓
Mechanistic prediction
```

CPU substitute:

```text
Small descriptor vector
        ↓
Conditioned PyTorch PINN
        ↓
Mechanistic prediction
```

---

# 19. Recommended Study Workflow

For every lesson, follow this sequence.

## Step 1: Read the equation

Identify:

- independent variables
- dependent variables
- parameters
- initial conditions
- boundary conditions

## Step 2: Derive the residual manually

Move all terms to one side.

## Step 3: Build the simplest network

Start with:

- small hidden layers
- Tanh activation
- CPU execution

## Step 4: Write the derivative code

Use PyTorch autograd.

## Step 5: Write each loss separately

Track:

- physics loss
- initial-condition loss
- boundary-condition loss
- data loss

## Step 6: Train and inspect

Plot:

- prediction
- exact or reference solution
- absolute error
- residual
- training loss

## Step 7: Diagnose problems

Check:

- collocation-point count
- learning rate
- loss weights
- network depth
- normalization
- training length
- residual distribution

## Step 8: Rebuild with PhysicsNeMo

Only after understanding the manual PyTorch implementation.

## Step 9: Extend the problem

Examples:

- add noise
- hide a parameter
- increase the time interval
- add a second state variable
- convert an ODE into a PDE

## Step 10: Complete an exercise independently

Each lesson should end with an exercise and a solution section.

---

# 20. What to Build First

The recommended first sequence is:

1. check the CPU environment;
2. learn tensors and shapes;
3. learn first and second derivatives with autograd;
4. train a simple supervised neural network;
5. solve exponential decay with a manual PyTorch PINN;
6. interpret the same model as drug elimination;
7. estimate the unknown decay or elimination rate;
8. rebuild the problem using PhysicsNeMo;
9. solve logistic growth;
10. solve a two-state biological ODE system;
11. solve the heat equation;
12. estimate an unknown diffusion coefficient;
13. introduce molecular descriptors;
14. condition a PINN on a biological feature vector;
15. optionally replace descriptors with BioNeMo embeddings on GPU.

---

# Final Course Outcome

After completing this course, you should be able to:

- explain how a PINN works;
- derive ODE and PDE residuals;
- build PINNs in PyTorch;
- build scientific models using PhysicsNeMo;
- understand where BioNeMo fits into biological modeling;
- run all foundational lessons on CPU;
- solve forward ODE and PDE problems;
- solve inverse problems;
- estimate unknown physical and biological parameters;
- diagnose common PINN failures;
- implement adaptive sampling;
- build coupled biological models;
- combine biological representations with mechanistic differential equations;
- progress toward neural operators and large-scale scientific machine learning.

---

## Suggested Next Lesson

The next lesson should be a complete notebook titled:

```text
01_first_cpu_pinn_exponential_decay.ipynb
```

It should contain:

- beginner-friendly explanations;
- tensor shape demonstrations;
- manual derivative calculations;
- full training code;
- plots;
- error metrics;
- exercises;
- solutions;
- a biological drug-elimination interpretation;
- a first inverse parameter-estimation extension.
