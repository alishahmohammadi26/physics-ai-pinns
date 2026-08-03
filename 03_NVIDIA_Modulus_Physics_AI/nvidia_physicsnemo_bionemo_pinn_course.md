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

$$y = 2x + 1$$

### Module 6: First PINN

Solve:

$$
\frac{du}{dt} + u = 0,
\qquad u(0)=1
$$

### Module 7: Nonlinear ODE PINN

Solve logistic growth:

$$
\frac{dN}{dt} = rN\left(1-\frac{N}{K}\right)
$$

### Module 8: Second-Order ODE PINN

Solve:

$$
m\frac{d^2x}{dt^2}+c\frac{dx}{dt}+kx=0
$$

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

$$y = x^2$$

we expect:

$$\frac{dy}{dx} = 2x$$

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
$$
\frac{d^2y}{dx^2}=2
$$

this should return values close to 2.

---

# 8. How a PINN Works

A Physics-Informed Neural Network approximates the unknown solution of a differential equation.

Suppose the exact function is:

$$
u(t)
$$

The network approximation is:

$$
u_\theta(t)
$$

where $\theta$ represents all network weights and biases.

## General loss function

A PINN often uses:

$$
\mathcal{L}
=
\lambda_r\mathcal{L}_{physics}
+
\lambda_{IC}\mathcal{L}_{IC}
+
\lambda_{BC}\mathcal{L}_{BC}
+
\lambda_d\mathcal{L}_{data}
$$

where:

- $\mathcal{L}_{physics}$ measures equation violation;
- $\mathcal{L}_{IC}$ measures initial-condition violation;
- $\mathcal{L}_{BC}$ measures boundary-condition violation;
- $\mathcal{L}_{data}$ measures mismatch with observations.

## Collocation points

Collocation points are input locations where the equation residual is evaluated.

They do not require known solution values.

For an ODE, these may be time points:

$$
t_1,t_2,\ldots,t_N
$$

For a PDE, these may be space-time points:

$$
(x_i,t_i)
$$

---

# 9. First PINN: Exponential Decay

We solve:

$$
\frac{du}{dt}+u=0,
\qquad u(0)=1
$$

The exact solution is:

$$
u(t)=e^{-t}
$$

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

$$
\frac{dC}{dt}+kC=0,
\qquad C(0)=C_0
$$

where:

- $C(t)$ is drug concentration;
- $k$ is the elimination-rate constant;
- $C_0$ is the initial concentration.

The exact solution is:

$$
C(t)=C_0e^{-kt}
$$

For example:

$$
C_0=10,
\qquad k=0.5
$$

Then:

$$
C(t)=10e^{-0.5t}
$$

## Biological PINN residual

$$
r_\theta(t)
=
\frac{dC_\theta}{dt}
+
kC_\theta(t)
$$

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

$$
\frac{du}{dt}+u=0
$$

## ODE 2: Logistic growth

$$
\frac{dN}{dt}=rN\left(1-\frac{N}{K}\right)
$$

Topics:

- nonlinear residuals
- carrying capacity
- long-term behavior
- extrapolation
- parameter sensitivity

## ODE 3: Harmonic oscillator

$$
\frac{d^2x}{dt^2}+\omega^2x=0
$$

## ODE 4: Damped oscillator

$$
m\frac{d^2x}{dt^2}+c\frac{dx}{dt}+kx=0
$$

## ODE 5: Lotka-Volterra

$$
\frac{dx}{dt}=\alpha x-\beta xy
$$

$$
\frac{dy}{dt}=\delta xy-\gamma y
$$

## ODE 6: SIR model

$$
\frac{dS}{dt}=-\beta\frac{SI}{N}
$$

$$
\frac{dI}{dt}=\beta\frac{SI}{N}-\gamma I
$$

$$
\frac{dR}{dt}=\gamma I
$$

## ODE 7: Two-compartment pharmacokinetics

$$
\frac{dC_1}{dt}
=
-k_{10}C_1-k_{12}C_1+k_{21}C_2
$$

$$
\frac{dC_2}{dt}
=
k_{12}C_1-k_{21}C_2
$$

## ODE 8: Michaelis-Menten kinetics

$$
\frac{dS}{dt}
=
-\frac{V_{max}S}{K_m+S}
$$

## ODE 9: Gene regulation

$$
\frac{dm}{dt}
=
\frac{\alpha}{1+(p/K)^n}
-
\gamma_m m
$$

$$
\frac{dp}{dt}
=
\beta m-\gamma_p p
$$

## ODE 10: Stiff reaction system

Example:

- Robertson chemical kinetics
- fast and slow reactions
- multiple time scales

---

# 14. PDE Progression

## PDE 1: Poisson equation

$$
\frac{d^2u}{dx^2}=f(x)
$$

## PDE 2: Heat equation

$$
\frac{\partial u}{\partial t}
=
\alpha\frac{\partial^2u}{\partial x^2}
$$

## PDE 3: Wave equation

$$
\frac{\partial^2u}{\partial t^2}
=
c^2\frac{\partial^2u}{\partial x^2}
$$

## PDE 4: Burgers equation

$$
u_t+uu_x-\nu u_{xx}=0
$$

## PDE 5: Reaction-diffusion

$$
\frac{\partial C}{\partial t}
=
D\frac{\partial^2C}{\partial x^2}
-kC
$$

## PDE 6: Fisher-KPP equation

$$
\frac{\partial u}{\partial t}
=
D\nabla^2u
+
ru(1-u)
$$

## PDE 7: Coupled tissue model

$$
\frac{\partial n}{\partial t}
=
D_n\nabla^2n
+
rn\left(1-\frac{n}{K}\right)
-
\gamma Cn
$$

$$
\frac{\partial C}{\partial t}
=
D_C\nabla^2C-k_CC
$$

## PDE 8: Navier-Stokes

The advanced course will include incompressible fluid equations.

---

# 15. Inverse Problems and Parameter Estimation

In a forward problem, the physical parameters are known.

In an inverse problem, one or more parameters are unknown and learned from data.

## Example: unknown decay rate

$$
\frac{du}{dt}=-ku
$$

The parameter $k$ is unknown.

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

- $V_{max}$
- $K_m$

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

$$
[-1,1]
$$

This often improves optimization.

## Nondimensionalization

Physical quantities with very different scales can be transformed into dimensionless quantities.

## Hard constraints

Instead of penalizing an initial condition, it can be built into the network output.

For example:

$$
u_\theta(t)=1+tN_\theta(t)
$$

This automatically satisfies:

$$
u_\theta(0)=1
$$

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

$$
\frac{dC}{dt}+kC=0
$$

The value of $k$ depends on the molecular representation.

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

---

# 21. Complete Code Laboratory

This section converts the roadmap above into executable reference programs. Every required example is written for CPU first. Where an NVIDIA BioNeMo workflow genuinely requires supported NVIDIA GPU hardware, the document provides both:

1. a runnable CPU educational substitute; and
2. a clearly marked optional GPU workflow.

> **Version note:** PhysicsNeMo 2.x uses ordinary PyTorch training loops and the upstreamed `physicsnemo.sym` package. Older tutorials based on `Domain`, `Solver`, `Constraint`, `Key`, or pre-built symbolic PDE classes may not match the current API.

## 21.1 Shared CPU utilities

Save the following as `pinn_utils.py`. All later examples reuse it.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

Tensor = torch.Tensor
DEVICE = torch.device("cpu")
DTYPE = torch.float32


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def grad(y: Tensor, x: Tensor) -> Tensor:
    """Return dy/dx while retaining the graph for higher derivatives."""
    return torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def mse(x: Tensor) -> Tensor:
    return torch.mean(x**2)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Sequence[int] = (64, 64, 64),
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = in_features
        for next_width in hidden:
            layers.extend([nn.Linear(width, next_width), activation()])
            width = next_width
        layers.append(nn.Linear(width, out_features))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class NormalizedMLP(MLP):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lower: Sequence[float],
        upper: Sequence[float],
        hidden: Sequence[int] = (64, 64, 64),
    ) -> None:
        super().__init__(in_features, out_features, hidden)
        self.register_buffer("lower", torch.tensor(lower, dtype=DTYPE))
        self.register_buffer("upper", torch.tensor(upper, dtype=DTYPE))

    def forward(self, x: Tensor) -> Tensor:
        z = 2.0 * (x - self.lower) / (self.upper - self.lower) - 1.0
        return self.net(z)


@dataclass
class TrainHistory:
    total: list[float]
    terms: dict[str, list[float]]


def train(
    model: nn.Module,
    loss_fn: Callable[[], tuple[Tensor, dict[str, Tensor]]],
    epochs: int = 5000,
    lr: float = 1e-3,
    extra_parameters: Iterable[nn.Parameter] = (),
    print_every: int = 500,
) -> TrainHistory:
    parameters = list(model.parameters()) + list(extra_parameters)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    history = TrainHistory(total=[], terms={})

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, terms = loss_fn()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {loss}")
        loss.backward()
        optimizer.step()

        history.total.append(float(loss.detach()))
        for name, value in terms.items():
            history.terms.setdefault(name, []).append(float(value.detach()))

        if epoch % print_every == 0 or epoch == epochs - 1:
            details = " | ".join(
                f"{name}={float(value.detach()):.3e}"
                for name, value in terms.items()
            )
            print(f"epoch={epoch:6d} | total={float(loss.detach()):.3e} | {details}")

    return history


def plot_history(history: TrainHistory, title: str = "Training loss") -> None:
    plt.figure(figsize=(8, 4.5))
    plt.semilogy(history.total, label="total")
    for name, values in history.terms.items():
        plt.semilogy(values, label=name, alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def relative_l2(prediction: Tensor, reference: Tensor) -> float:
    numerator = torch.linalg.vector_norm(prediction - reference)
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    return float((numerator / denominator).detach())
```

## 21.2 Supervised neural-network warm-up

This program learns the mapping $y=2x+1$ from data. It does not use a differential equation yet.

```python
import matplotlib.pyplot as plt
import torch
from torch import nn

from pinn_utils import DEVICE, MLP, seed_everything

seed_everything()
x = torch.linspace(-1.0, 1.0, 100, device=DEVICE).reshape(-1, 1)
y = 2.0 * x + 1.0

model = MLP(1, 1, hidden=(32, 32)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(2000):
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x)
    loss = nn.functional.mse_loss(prediction, y)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(epoch, float(loss.detach()))

with torch.no_grad():
    prediction = model(x)

plt.plot(x.numpy(), y.numpy(), label="exact")
plt.plot(x.numpy(), prediction.numpy(), "--", label="network")
plt.legend()
plt.grid(True)
plt.show()
```

---

# 22. Complete ODE Programs

## 22.1 Exponential decay

```python
import matplotlib.pyplot as plt
import torch

from pinn_utils import DEVICE, MLP, grad, mse, relative_l2, seed_everything, train

seed_everything()
model = MLP(1, 1).to(DEVICE)
t_f = torch.linspace(0.0, 5.0, 200, device=DEVICE).reshape(-1, 1)
t_f.requires_grad_(True)
t0 = torch.tensor([[0.0]], device=DEVICE)


def loss_fn():
    u = model(t_f)
    residual = grad(u, t_f) + u
    physics = mse(residual)
    initial = mse(model(t0) - 1.0)
    total = physics + 20.0 * initial
    return total, {"physics": physics, "initial": initial}


train(model, loss_fn, epochs=4000)

t = torch.linspace(0.0, 5.0, 400, device=DEVICE).reshape(-1, 1)
with torch.no_grad():
    prediction = model(t)
reference = torch.exp(-t)
print("relative L2:", relative_l2(prediction, reference))

plt.plot(t.numpy(), reference.numpy(), label="exact")
plt.plot(t.numpy(), prediction.numpy(), "--", label="PINN")
plt.legend(); plt.grid(True); plt.show()
```

## 22.2 Logistic growth

The hard output transform below guarantees $N(0)=N_0$ exactly.

```python
import matplotlib.pyplot as plt
import torch

from pinn_utils import DEVICE, MLP, grad, mse, seed_everything, train

seed_everything()
r, K, N0 = 1.0, 10.0, 0.5
base = MLP(1, 1).to(DEVICE)


def population(t):
    return N0 + t * base(t)


t_f = torch.linspace(0.0, 8.0, 250, device=DEVICE).reshape(-1, 1)
t_f.requires_grad_(True)


def loss_fn():
    N = population(t_f)
    residual = grad(N, t_f) - r * N * (1.0 - N / K)
    physics = mse(residual)
    positivity = mse(torch.relu(-N))
    return physics + positivity, {"physics": physics, "positivity": positivity}


train(base, loss_fn, epochs=6000)

t = torch.linspace(0.0, 8.0, 400).reshape(-1, 1)
with torch.no_grad():
    pred = population(t)
exact = K / (1.0 + ((K - N0) / N0) * torch.exp(-r * t))
plt.plot(t, exact, label="exact")
plt.plot(t, pred, "--", label="PINN")
plt.legend(); plt.grid(True); plt.show()
```

## 22.3 Damped harmonic oscillator

```python
import matplotlib.pyplot as plt
import torch

from pinn_utils import DEVICE, MLP, grad, mse, seed_everything, train

seed_everything()
m, c, k = 1.0, 0.4, 4.0
x0, v0 = 1.0, 0.0
model = MLP(1, 1, hidden=(64, 64, 64)).to(DEVICE)
t_f = torch.linspace(0.0, 10.0, 300, device=DEVICE).reshape(-1, 1)
t_f.requires_grad_(True)
t0 = torch.tensor([[0.0]], device=DEVICE, requires_grad=True)


def loss_fn():
    x = model(t_f)
    x_t = grad(x, t_f)
    x_tt = grad(x_t, t_f)
    residual = m * x_tt + c * x_t + k * x
    physics = mse(residual)

    x_initial = model(t0)
    v_initial = grad(x_initial, t0)
    initial = mse(x_initial - x0) + mse(v_initial - v0)
    return physics + 50.0 * initial, {"physics": physics, "initial": initial}


train(model, loss_fn, epochs=8000)

t = torch.linspace(0.0, 10.0, 500).reshape(-1, 1)
with torch.no_grad():
    pred = model(t)
plt.plot(t, pred)
plt.xlabel("t"); plt.ylabel("x(t)"); plt.grid(True); plt.show()
```

## 22.4 Lotka-Volterra system

```python
import matplotlib.pyplot as plt
import torch

from pinn_utils import DEVICE, MLP, grad, mse, seed_everything, train

seed_everything()
alpha, beta, delta, gamma = 1.5, 1.0, 0.75, 1.0
x0, y0 = 1.0, 1.0
model = MLP(1, 2, hidden=(96, 96, 96)).to(DEVICE)
t_f = torch.linspace(0.0, 12.0, 400, device=DEVICE).reshape(-1, 1)
t_f.requires_grad_(True)
t0 = torch.tensor([[0.0]], device=DEVICE)


def loss_fn():
    states = model(t_f)
    x, y = states[:, 0:1], states[:, 1:2]
    x_t, y_t = grad(x, t_f), grad(y, t_f)
    rx = x_t - (alpha * x - beta * x * y)
    ry = y_t - (delta * x * y - gamma * y)
    physics = mse(rx) + mse(ry)
    initial = mse(model(t0) - torch.tensor([[x0, y0]], device=DEVICE))
    positivity = mse(torch.relu(-states))
    return physics + 50.0 * initial + positivity, {
        "physics": physics, "initial": initial, "positivity": positivity
    }


train(model, loss_fn, epochs=12000, lr=5e-4)
t = torch.linspace(0.0, 12.0, 600).reshape(-1, 1)
with torch.no_grad():
    states = model(t)
plt.plot(t, states[:, 0], label="prey")
plt.plot(t, states[:, 1], label="predator")
plt.legend(); plt.grid(True); plt.show()
```

## 22.5 SIR epidemiological system

The output is normalized so $S+I+R=1$ by applying `softmax`.

```python
import matplotlib.pyplot as plt
import torch
from torch import nn

from pinn_utils import DEVICE, MLP, grad, mse, seed_everything, train

seed_everything()
beta, gamma = 0.8, 0.25
initial_state = torch.tensor([[0.99, 0.01, 0.0]], device=DEVICE)
base = MLP(1, 3, hidden=(96, 96, 96)).to(DEVICE)
t_f = torch.linspace(0.0, 40.0, 500, device=DEVICE).reshape(-1, 1)
t_f.requires_grad_(True)
t0 = torch.tensor([[0.0]], device=DEVICE)


def states(t):
    return nn.functional.softmax(base(t), dim=1)


def loss_fn():
    y = states(t_f)
    S, I, R = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    rS = grad(S, t_f) + beta * S * I
    rI = grad(I, t_f) - beta * S * I + gamma * I
    rR = grad(R, t_f) - gamma * I
    physics = mse(rS) + mse(rI) + mse(rR)
    initial = mse(states(t0) - initial_state)
    return physics + 100.0 * initial, {"physics": physics, "initial": initial}


train(base, loss_fn, epochs=12000, lr=5e-4)
t = torch.linspace(0.0, 40.0, 800).reshape(-1, 1)
with torch.no_grad():
    y = states(t)
for i, label in enumerate(["S", "I", "R"]):
    plt.plot(t, y[:, i], label=label)
plt.legend(); plt.grid(True); plt.show()
```

## 22.6 Two-compartment pharmacokinetics

```python
import matplotlib.pyplot as plt
import torch

from pinn_utils import DEVICE, MLP, grad, mse, seed_everything, train

seed_everything()
k10, k12, k21 = 0.25, 0.7, 0.4
initial = torch.tensor([[10.0, 0.0]], device=DEVICE)
model = MLP(1, 2, hidden=(96, 96, 96)).to(DEVICE)
t_f = torch.linspace(0.0, 24.0, 500, device=DEVICE).reshape(-1, 1)
t_f.requires_grad_(True)
t0 = torch.tensor([[0.0]], device=DEVICE)


def loss_fn():
    C = model(t_f)
    C1, C2 = C[:, 0:1], C[:, 1:2]
    r1 = grad(C1, t_f) + (k10 + k12) * C1 - k21 * C2
    r2 = grad(C2, t_f) - k12 * C1 + k21 * C2
    physics = mse(r1) + mse(r2)
    ic = mse(model(t0) - initial)
    positive = mse(torch.relu(-C))
    return physics + 100.0 * ic + positive, {
        "physics": physics, "initial": ic, "positivity": positive
    }


train(model, loss_fn, epochs=12000, lr=5e-4)
t = torch.linspace(0.0, 24.0, 700).reshape(-1, 1)
with torch.no_grad():
    C = model(t)
plt.plot(t, C[:, 0], label="central")
plt.plot(t, C[:, 1], label="peripheral")
plt.legend(); plt.grid(True); plt.show()
```

## 22.7 Michaelis-Menten substrate depletion

```python
import matplotlib.pyplot as plt
import torch

from pinn_utils import DEVICE, MLP, grad, mse, seed_everything, train

seed_everything()
Vmax, Km, S0 = 2.0, 1.5, 10.0
base = MLP(1, 1).to(DEVICE)
t_f = torch.linspace(0.0, 8.0, 300, device=DEVICE).reshape(-1, 1)
t_f.requires_grad_(True)


def substrate(t):
    return torch.nn.functional.softplus(S0 + t * base(t))


def loss_fn():
    S = substrate(t_f)
    residual = grad(S, t_f) + Vmax * S / (Km + S)
    physics = mse(residual)
    return physics, {"physics": physics}


train(base, loss_fn, epochs=8000)
t = torch.linspace(0.0, 8.0, 400).reshape(-1, 1)
with torch.no_grad():
    S = substrate(t)
plt.plot(t, S); plt.grid(True); plt.show()
```

## 22.8 Gene-regulation system

```python
import matplotlib.pyplot as plt
import torch

from pinn_utils import DEVICE, MLP, grad, mse, seed_everything, train

seed_everything()
alpha, K, hill_n = 4.0, 1.0, 2.0
gamma_m, beta, gamma_p = 0.7, 1.2, 0.4
initial = torch.tensor([[0.1, 0.1]], device=DEVICE)
model = MLP(1, 2, hidden=(96, 96, 96)).to(DEVICE)
t_f = torch.linspace(0.0, 20.0, 500, device=DEVICE).reshape(-1, 1)
t_f.requires_grad_(True)
t0 = torch.tensor([[0.0]], device=DEVICE)


def loss_fn():
    y = model(t_f)
    m, p = y[:, 0:1], y[:, 1:2]
    rm = grad(m, t_f) - alpha / (1.0 + (p / K) ** hill_n) + gamma_m * m
    rp = grad(p, t_f) - beta * m + gamma_p * p
    physics = mse(rm) + mse(rp)
    ic = mse(model(t0) - initial)
    positive = mse(torch.relu(-y))
    return physics + 50.0 * ic + positive, {
        "physics": physics, "initial": ic, "positivity": positive
    }


train(model, loss_fn, epochs=12000, lr=5e-4)
t = torch.linspace(0.0, 20.0, 600).reshape(-1, 1)
with torch.no_grad():
    y = model(t)
plt.plot(t, y[:, 0], label="mRNA")
plt.plot(t, y[:, 1], label="protein")
plt.legend(); plt.grid(True); plt.show()
```

## 22.9 Robertson stiff kinetics

A naive global PINN often struggles with this problem. The code uses logarithmically spaced collocation times and conservation loss. It is an advanced CPU example and may require tuning.

```python
import matplotlib.pyplot as plt
import torch
from torch import nn

from pinn_utils import DEVICE, MLP, grad, mse, seed_everything, train

seed_everything()
base = MLP(1, 3, hidden=(128, 128, 128, 128)).to(DEVICE)
log_t = torch.linspace(-6.0, 3.0, 1200, device=DEVICE).reshape(-1, 1)
log_t.requires_grad_(True)
t0 = torch.tensor([[-6.0]], device=DEVICE)
initial = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE)


def state(log_time):
    return nn.functional.softmax(base(log_time), dim=1)


def loss_fn():
    t = 10.0**log_t
    y = state(log_t)
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    scale = t * torch.log(torch.tensor(10.0, device=DEVICE))
    y1_t = grad(y1, log_t) / scale
    y2_t = grad(y2, log_t) / scale
    y3_t = grad(y3, log_t) / scale
    r1 = y1_t + 0.04 * y1 - 1.0e4 * y2 * y3
    r2 = y2_t - 0.04 * y1 + 1.0e4 * y2 * y3 + 3.0e7 * y2**2
    r3 = y3_t - 3.0e7 * y2**2
    physics = mse(r1) + mse(r2) + mse(r3)
    ic = mse(state(t0) - initial)
    return physics + 100.0 * ic, {"physics": physics, "initial": ic}


train(base, loss_fn, epochs=20000, lr=2e-4)
with torch.no_grad():
    y = state(log_t)
plt.semilogx((10.0**log_t).detach(), y[:, 0].detach(), label="y1")
plt.semilogx((10.0**log_t).detach(), y[:, 1].detach(), label="y2")
plt.semilogx((10.0**log_t).detach(), y[:, 2].detach(), label="y3")
plt.legend(); plt.grid(True); plt.show()
```

---

# 23. Complete PDE Programs

## 23.1 One-dimensional Poisson equation

We solve $u_{xx}=-\pi^2\sin(\pi x)$ on $x\in[0,1]$ with $u(0)=u(1)=0$. The exact solution is $u(x)=\sin(\pi x)$.

```python
import math
import matplotlib.pyplot as plt
import torch

from pinn_utils import DEVICE, MLP, grad, mse, relative_l2, seed_everything, train

seed_everything()
model = MLP(1, 1).to(DEVICE)
x_f = torch.linspace(0.0, 1.0, 250, device=DEVICE).reshape(-1, 1)
x_f.requires_grad_(True)
x_b = torch.tensor([[0.0], [1.0]], device=DEVICE)


def loss_fn():
    u = model(x_f)
    u_xx = grad(grad(u, x_f), x_f)
    forcing = -(math.pi**2) * torch.sin(math.pi * x_f)
    physics = mse(u_xx - forcing)
    boundary = mse(model(x_b))
    return physics + 50.0 * boundary, {"physics": physics, "boundary": boundary}


train(model, loss_fn, epochs=6000)
x = torch.linspace(0.0, 1.0, 500).reshape(-1, 1)
with torch.no_grad():
    pred = model(x)
exact = torch.sin(math.pi * x)
print("relative L2:", relative_l2(pred, exact))
plt.plot(x, exact, label="exact")
plt.plot(x, pred, "--", label="PINN")
plt.legend(); plt.grid(True); plt.show()
```

## 23.2 One-dimensional heat equation

We solve $u_t=\alpha u_{xx}$ with $u(x,0)=\sin(\pi x)$ and zero Dirichlet boundaries.

```python
import math
import matplotlib.pyplot as plt
import torch

from pinn_utils import DEVICE, NormalizedMLP, grad, mse, seed_everything, train

seed_everything()
alpha = 0.1
model = NormalizedMLP(2, 1, lower=(0.0, 0.0), upper=(1.0, 1.0)).to(DEVICE)

Nf = 3000
xt_f = torch.rand(Nf, 2, device=DEVICE)
xt_f.requires_grad_(True)

x0 = torch.rand(400, 1, device=DEVICE)
t0 = torch.zeros_like(x0)
xt0 = torch.cat([x0, t0], dim=1)
u0 = torch.sin(math.pi * x0)

tb = torch.rand(400, 1, device=DEVICE)
xb0 = torch.zeros_like(tb)
xb1 = torch.ones_like(tb)
left = torch.cat([xb0, tb], dim=1)
right = torch.cat([xb1, tb], dim=1)


def loss_fn():
    u = model(xt_f)
    du = grad(u, xt_f)
    u_x, u_t = du[:, 0:1], du[:, 1:2]
    u_xx = grad(u_x, xt_f)[:, 0:1]
    physics = mse(u_t - alpha * u_xx)
    initial = mse(model(xt0) - u0)
    boundary = mse(model(left)) + mse(model(right))
    total = physics + 20.0 * initial + 20.0 * boundary
    return total, {"physics": physics, "initial": initial, "boundary": boundary}


train(model, loss_fn, epochs=10000, lr=7e-4)

x = torch.linspace(0.0, 1.0, 160)
t = torch.linspace(0.0, 1.0, 120)
xx, tt = torch.meshgrid(x, t, indexing="ij")
points = torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1)
with torch.no_grad():
    U = model(points).reshape(xx.shape)
plt.contourf(tt, xx, U, levels=40)
plt.xlabel("t"); plt.ylabel("x"); plt.colorbar(label="u"); plt.show()
```

## 23.3 Wave equation

```python
import math
import matplotlib.pyplot as plt
import torch

from pinn_utils import DEVICE, NormalizedMLP, grad, mse, seed_everything, train

seed_everything()
c = 1.0
model = NormalizedMLP(2, 1, lower=(0.0, 0.0), upper=(1.0, 1.0), hidden=(96, 96, 96)).to(DEVICE)
xt_f = torch.rand(4000, 2, device=DEVICE, requires_grad=True)

x0 = torch.rand(500, 1, device=DEVICE)
xt0 = torch.cat([x0, torch.zeros_like(x0)], dim=1)
xt0.requires_grad_(True)

tb = torch.rand(500, 1, device=DEVICE)
left = torch.cat([torch.zeros_like(tb), tb], dim=1)
right = torch.cat([torch.ones_like(tb), tb], dim=1)


def loss_fn():
    u = model(xt_f)
    du = grad(u, xt_f)
    u_x, u_t = du[:, 0:1], du[:, 1:2]
    u_xx = grad(u_x, xt_f)[:, 0:1]
    u_tt = grad(u_t, xt_f)[:, 1:2]
    physics = mse(u_tt - c**2 * u_xx)

    u_initial = model(xt0)
    initial_grad = grad(u_initial, xt0)
    displacement = mse(u_initial - torch.sin(math.pi * x0))
    velocity = mse(initial_grad[:, 1:2])
    boundary = mse(model(left)) + mse(model(right))
    total = physics + 20.0 * (displacement + velocity + boundary)
    return total, {
        "physics": physics,
        "displacement": displacement,
        "velocity": velocity,
        "boundary": boundary,
    }


train(model, loss_fn, epochs=14000, lr=5e-4)
x = torch.linspace(0.0, 1.0, 200).reshape(-1, 1)
for time_value in [0.0, 0.25, 0.5, 0.75, 1.0]:
    points = torch.cat([x, torch.full_like(x, time_value)], dim=1)
    with torch.no_grad():
        plt.plot(x, model(points), label=f"t={time_value}")
plt.legend(); plt.grid(True); plt.show()
```

## 23.4 Burgers equation

This is a compact educational implementation. Shock-like regions may need more points, Fourier features, or adaptive sampling.

```python
import matplotlib.pyplot as plt
import torch

from pinn_utils import DEVICE, NormalizedMLP, grad, mse, seed_everything, train

seed_everything()
nu = 0.01 / torch.pi
model = NormalizedMLP(2, 1, lower=(-1.0, 0.0), upper=(1.0, 1.0), hidden=(128, 128, 128)).to(DEVICE)

x = 2.0 * torch.rand(6000, 1, device=DEVICE) - 1.0
t = torch.rand(6000, 1, device=DEVICE)
xt_f = torch.cat([x, t], dim=1).requires_grad_(True)

x0 = 2.0 * torch.rand(600, 1, device=DEVICE) - 1.0
initial_points = torch.cat([x0, torch.zeros_like(x0)], dim=1)
initial_values = -torch.sin(torch.pi * x0)

tb = torch.rand(600, 1, device=DEVICE)
left = torch.cat([-torch.ones_like(tb), tb], dim=1)
right = torch.cat([torch.ones_like(tb), tb], dim=1)


def loss_fn():
    u = model(xt_f)
    du = grad(u, xt_f)
    u_x, u_t = du[:, 0:1], du[:, 1:2]
    u_xx = grad(u_x, xt_f)[:, 0:1]
    physics = mse(u_t + u * u_x - nu * u_xx)
    initial = mse(model(initial_points) - initial_values)
    boundary = mse(model(left)) + mse(model(right))
    return physics + 20.0 * initial + 20.0 * boundary, {
        "physics": physics, "initial": initial, "boundary": boundary
    }


train(model, loss_fn, epochs=16000, lr=5e-4)

xg = torch.linspace(-1.0, 1.0, 200)
tg = torch.linspace(0.0, 1.0, 150)
xx, tt = torch.meshgrid(xg, tg, indexing="ij")
points = torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1)
with torch.no_grad():
    U = model(points).reshape(xx.shape)
plt.contourf(tt, xx, U, levels=50)
plt.xlabel("t"); plt.ylabel("x"); plt.colorbar(); plt.show()
```

## 23.5 Reaction-diffusion equation

```python
import math
import matplotlib.pyplot as plt
import torch

from pinn_utils import DEVICE, NormalizedMLP, grad, mse, seed_everything, train

seed_everything()
D, k = 0.05, 0.4
model = NormalizedMLP(2, 1, lower=(0.0, 0.0), upper=(1.0, 2.0)).to(DEVICE)
xt_f = torch.cat([
    torch.rand(4000, 1, device=DEVICE),
    2.0 * torch.rand(4000, 1, device=DEVICE),
], dim=1).requires_grad_(True)

x0 = torch.rand(500, 1, device=DEVICE)
initial_points = torch.cat([x0, torch.zeros_like(x0)], dim=1)
initial_values = torch.sin(math.pi * x0)

tb = 2.0 * torch.rand(500, 1, device=DEVICE)
left = torch.cat([torch.zeros_like(tb), tb], dim=1)
right = torch.cat([torch.ones_like(tb), tb], dim=1)


def loss_fn():
    C = model(xt_f)
    dC = grad(C, xt_f)
    C_x, C_t = dC[:, 0:1], dC[:, 1:2]
    C_xx = grad(C_x, xt_f)[:, 0:1]
    physics = mse(C_t - D * C_xx + k * C)
    initial = mse(model(initial_points) - initial_values)
    boundary = mse(model(left)) + mse(model(right))
    return physics + 20.0 * initial + 20.0 * boundary, {
        "physics": physics, "initial": initial, "boundary": boundary
    }


train(model, loss_fn, epochs=10000)
```

## 23.6 Fisher-KPP equation

```python
import torch
from torch import nn

from pinn_utils import DEVICE, NormalizedMLP, grad, mse, seed_everything, train

seed_everything()
D, r = 0.01, 2.0
base = NormalizedMLP(2, 1, lower=(-1.0, 0.0), upper=(1.0, 1.0), hidden=(96, 96, 96)).to(DEVICE)


def field(points):
    return torch.sigmoid(base(points))

x = 2.0 * torch.rand(5000, 1) - 1.0
t = torch.rand(5000, 1)
xt_f = torch.cat([x, t], dim=1).to(DEVICE).requires_grad_(True)

x0 = 2.0 * torch.rand(600, 1, device=DEVICE) - 1.0
xt0 = torch.cat([x0, torch.zeros_like(x0)], dim=1)
u0 = torch.exp(-40.0 * (x0 + 0.5) ** 2)


def loss_fn():
    u = field(xt_f)
    du = grad(u, xt_f)
    u_x, u_t = du[:, 0:1], du[:, 1:2]
    u_xx = grad(u_x, xt_f)[:, 0:1]
    physics = mse(u_t - D * u_xx - r * u * (1.0 - u))
    initial = mse(field(xt0) - u0)
    return physics + 20.0 * initial, {"physics": physics, "initial": initial}


train(base, loss_fn, epochs=12000)
```

## 23.7 Coupled tumor-drug PDE

This small one-dimensional model predicts tumor density $n(x,t)$ and drug concentration $C(x,t)$.

```python
import torch
from torch import nn

from pinn_utils import DEVICE, NormalizedMLP, grad, mse, seed_everything, train

seed_everything()
Dn, Dc = 0.002, 0.02
growth, capacity, kill, clearance = 1.0, 1.0, 1.2, 0.3
base = NormalizedMLP(2, 2, lower=(0.0, 0.0), upper=(1.0, 2.0), hidden=(128, 128, 128)).to(DEVICE)


def fields(points):
    raw = base(points)
    n = torch.sigmoid(raw[:, 0:1])
    C = nn.functional.softplus(raw[:, 1:2])
    return n, C

xt_f = torch.cat([torch.rand(7000, 1), 2.0 * torch.rand(7000, 1)], dim=1).to(DEVICE)
xt_f.requires_grad_(True)
x0 = torch.rand(700, 1, device=DEVICE)
xt0 = torch.cat([x0, torch.zeros_like(x0)], dim=1)
n0 = torch.exp(-80.0 * (x0 - 0.5) ** 2)
C0 = torch.zeros_like(x0)

tb = 2.0 * torch.rand(700, 1, device=DEVICE)
left = torch.cat([torch.zeros_like(tb), tb], dim=1)
right = torch.cat([torch.ones_like(tb), tb], dim=1)


def loss_fn():
    n, C = fields(xt_f)
    dn, dC = grad(n, xt_f), grad(C, xt_f)
    n_xx = grad(dn[:, 0:1], xt_f)[:, 0:1]
    C_xx = grad(dC[:, 0:1], xt_f)[:, 0:1]
    rn = dn[:, 1:2] - Dn * n_xx - growth * n * (1.0 - n / capacity) + kill * C * n
    rC = dC[:, 1:2] - Dc * C_xx + clearance * C
    physics = mse(rn) + mse(rC)

    n_initial, C_initial = fields(xt0)
    initial = mse(n_initial - n0) + mse(C_initial - C0)

    n_left, C_left = fields(left)
    n_right, C_right = fields(right)
    boundary = mse(n_left) + mse(n_right) + mse(C_left - 1.0) + mse(C_right)
    return physics + 20.0 * initial + 20.0 * boundary, {
        "physics": physics, "initial": initial, "boundary": boundary
    }


train(base, loss_fn, epochs=18000, lr=4e-4)
```

---

# 24. Inverse Problems and Parameter Estimation Code

## 24.1 Unknown exponential-decay rate

```python
import matplotlib.pyplot as plt
import torch
from torch import nn

from pinn_utils import DEVICE, MLP, grad, mse, seed_everything, train

seed_everything()
true_k = 0.7
t_data = torch.tensor([[0.0], [0.5], [1.0], [1.5], [2.5], [4.0]], device=DEVICE)
u_data = torch.exp(-true_k * t_data) + 0.01 * torch.randn_like(t_data)

model = MLP(1, 1).to(DEVICE)
raw_k = nn.Parameter(torch.tensor(0.0, device=DEVICE))
t_f = torch.linspace(0.0, 5.0, 300, device=DEVICE).reshape(-1, 1)
t_f.requires_grad_(True)
t0 = torch.tensor([[0.0]], device=DEVICE)


def loss_fn():
    k = nn.functional.softplus(raw_k)
    u = model(t_f)
    physics = mse(grad(u, t_f) + k * u)
    initial = mse(model(t0) - 1.0)
    data = mse(model(t_data) - u_data)
    total = physics + 20.0 * initial + 20.0 * data
    return total, {"physics": physics, "initial": initial, "data": data}


train(model, loss_fn, epochs=8000, extra_parameters=[raw_k])
estimated_k = float(nn.functional.softplus(raw_k).detach())
print("true k:", true_k)
print("estimated k:", estimated_k)
```

## 24.2 Joint estimation of logistic $r$ and $K$

```python
import torch
from torch import nn

from pinn_utils import DEVICE, MLP, grad, mse, seed_everything, train

seed_everything()
N0 = 0.5
true_r, true_K = 0.9, 12.0
t_data = torch.linspace(0.0, 7.0, 18, device=DEVICE).reshape(-1, 1)
N_data = true_K / (1.0 + ((true_K - N0) / N0) * torch.exp(-true_r * t_data))
N_data += 0.05 * torch.randn_like(N_data)

model = MLP(1, 1).to(DEVICE)
raw_r = nn.Parameter(torch.tensor(0.0, device=DEVICE))
raw_K = nn.Parameter(torch.tensor(2.0, device=DEVICE))
t_f = torch.linspace(0.0, 7.0, 400, device=DEVICE).reshape(-1, 1)
t_f.requires_grad_(True)


def population(t):
    return N0 + t * model(t)


def loss_fn():
    r = nn.functional.softplus(raw_r)
    K = N0 + nn.functional.softplus(raw_K)
    N = population(t_f)
    physics = mse(grad(N, t_f) - r * N * (1.0 - N / K))
    data = mse(population(t_data) - N_data)
    return physics + 20.0 * data, {"physics": physics, "data": data}


train(model, loss_fn, epochs=12000, lr=5e-4, extra_parameters=[raw_r, raw_K])
print("r:", float(nn.functional.softplus(raw_r).detach()))
print("K:", float((N0 + nn.functional.softplus(raw_K)).detach()))
```

## 24.3 Unknown heat diffusivity

```python
import math
import torch
from torch import nn

from pinn_utils import DEVICE, NormalizedMLP, grad, mse, seed_everything, train

seed_everything()
true_alpha = 0.15
model = NormalizedMLP(2, 1, lower=(0.0, 0.0), upper=(1.0, 1.0)).to(DEVICE)
raw_alpha = nn.Parameter(torch.tensor(-2.0, device=DEVICE))

xt_f = torch.rand(5000, 2, device=DEVICE).requires_grad_(True)
xt_data = torch.rand(120, 2, device=DEVICE)
x_d, t_d = xt_data[:, 0:1], xt_data[:, 1:2]
u_data = torch.exp(-(math.pi**2) * true_alpha * t_d) * torch.sin(math.pi * x_d)
u_data += 0.005 * torch.randn_like(u_data)

x0 = torch.rand(500, 1, device=DEVICE)
xt0 = torch.cat([x0, torch.zeros_like(x0)], dim=1)
u0 = torch.sin(math.pi * x0)

tb = torch.rand(500, 1, device=DEVICE)
left = torch.cat([torch.zeros_like(tb), tb], dim=1)
right = torch.cat([torch.ones_like(tb), tb], dim=1)


def loss_fn():
    alpha = nn.functional.softplus(raw_alpha)
    u = model(xt_f)
    du = grad(u, xt_f)
    u_xx = grad(du[:, 0:1], xt_f)[:, 0:1]
    physics = mse(du[:, 1:2] - alpha * u_xx)
    data = mse(model(xt_data) - u_data)
    initial = mse(model(xt0) - u0)
    boundary = mse(model(left)) + mse(model(right))
    total = physics + 20.0 * data + 10.0 * initial + 10.0 * boundary
    return total, {
        "physics": physics, "data": data, "initial": initial, "boundary": boundary
    }


train(model, loss_fn, epochs=14000, lr=5e-4, extra_parameters=[raw_alpha])
print("true alpha:", true_alpha)
print("estimated alpha:", float(nn.functional.softplus(raw_alpha).detach()))
```

## 24.4 Unknown source function

Two networks are trained: one predicts $u(x,t)$ and one predicts the source $f(x,t)$ in $u_t-Du_{xx}=f$.

```python
import torch

from pinn_utils import DEVICE, NormalizedMLP, grad, mse, seed_everything, train

seed_everything()
D = 0.05
u_model = NormalizedMLP(2, 1, lower=(0.0, 0.0), upper=(1.0, 1.0)).to(DEVICE)
f_model = NormalizedMLP(2, 1, lower=(0.0, 0.0), upper=(1.0, 1.0), hidden=(48, 48)).to(DEVICE)

xt_f = torch.rand(5000, 2, device=DEVICE).requires_grad_(True)
xt_data = torch.rand(250, 2, device=DEVICE)
x, t = xt_data[:, 0:1], xt_data[:, 1:2]
# Synthetic observed field generated from u=sin(pi*x)*(1+t)
u_data = torch.sin(torch.pi * x) * (1.0 + t)


def loss_fn():
    u = u_model(xt_f)
    du = grad(u, xt_f)
    u_xx = grad(du[:, 0:1], xt_f)[:, 0:1]
    source = f_model(xt_f)
    physics = mse(du[:, 1:2] - D * u_xx - source)
    data = mse(u_model(xt_data) - u_data)
    source_smoothness = mse(grad(source, xt_f))
    total = physics + 20.0 * data + 1e-4 * source_smoothness
    return total, {
        "physics": physics, "data": data, "source_smoothness": source_smoothness
    }


train(
    u_model,
    loss_fn,
    epochs=12000,
    lr=5e-4,
    extra_parameters=list(f_model.parameters()),
)
```

---

# 25. Advanced Training Methods

## 25.1 Residual-based adaptive refinement

```python
import torch

from pinn_utils import DEVICE, MLP, grad, mse, seed_everything

seed_everything()
model = MLP(1, 1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
t_f = torch.linspace(0.0, 5.0, 50, device=DEVICE).reshape(-1, 1)

for refinement_round in range(6):
    t_f = t_f.detach().requires_grad_(True)
    for epoch in range(1500):
        optimizer.zero_grad(set_to_none=True)
        u = model(t_f)
        residual = grad(u, t_f) + u
        initial = model(torch.tensor([[0.0]], device=DEVICE)) - 1.0
        loss = mse(residual) + 20.0 * mse(initial)
        loss.backward()
        optimizer.step()

    candidates = torch.linspace(0.0, 5.0, 2000, device=DEVICE).reshape(-1, 1)
    candidates.requires_grad_(True)
    residual_abs = torch.abs(grad(model(candidates), candidates) + model(candidates))
    indices = torch.topk(residual_abs.reshape(-1), k=50).indices
    new_points = candidates.detach()[indices]
    t_f = torch.cat([t_f.detach(), new_points], dim=0)
    print(f"round={refinement_round}, collocation points={len(t_f)}")
```

## 25.2 Adaptive loss weights

The following simple gradient-balancing example updates the initial-condition weight using gradient magnitudes.

```python
import torch

from pinn_utils import DEVICE, MLP, grad, mse, seed_everything

seed_everything()
model = MLP(1, 1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
t = torch.linspace(0.0, 5.0, 250, device=DEVICE).reshape(-1, 1).requires_grad_(True)
t0 = torch.tensor([[0.0]], device=DEVICE)
lambda_ic = 1.0

for epoch in range(5000):
    optimizer.zero_grad(set_to_none=True)
    u = model(t)
    physics_loss = mse(grad(u, t) + u)
    initial_loss = mse(model(t0) - 1.0)

    if epoch % 50 == 0:
        p_grads = torch.autograd.grad(
            physics_loss, model.parameters(), retain_graph=True, allow_unused=True
        )
        i_grads = torch.autograd.grad(
            initial_loss, model.parameters(), retain_graph=True, allow_unused=True
        )
        p_norm = sum(g.abs().mean() for g in p_grads if g is not None)
        i_norm = sum(g.abs().mean() for g in i_grads if g is not None).clamp_min(1e-12)
        target = float((p_norm / i_norm).detach())
        lambda_ic = 0.9 * lambda_ic + 0.1 * target

    loss = physics_loss + lambda_ic * initial_loss
    loss.backward()
    optimizer.step()
```

## 25.3 Adam followed by L-BFGS

```python
import torch

# Assume model and a closure-compatible loss_fn already exist.
adam = torch.optim.Adam(model.parameters(), lr=1e-3)
for _ in range(3000):
    adam.zero_grad(set_to_none=True)
    loss, _ = loss_fn()
    loss.backward()
    adam.step()

lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=500,
    history_size=50,
    line_search_fn="strong_wolfe",
)


def closure():
    lbfgs.zero_grad(set_to_none=True)
    loss, _ = loss_fn()
    loss.backward()
    return loss


lbfgs.step(closure)
```

## 25.4 Fourier-feature PINN

```python
import math
import torch
from torch import nn


class FourierFeatureMLP(nn.Module):
    def __init__(self, in_features=2, frequencies=32, out_features=1):
        super().__init__()
        B = 8.0 * torch.randn(in_features, frequencies)
        self.register_buffer("B", B)
        self.net = nn.Sequential(
            nn.Linear(2 * frequencies, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, out_features),
        )

    def forward(self, x):
        projection = 2.0 * math.pi * x @ self.B
        features = torch.cat([torch.sin(projection), torch.cos(projection)], dim=1)
        return self.net(features)
```

## 25.5 Time-domain decomposition

```python
import torch

from pinn_utils import DEVICE, MLP, grad, mse, seed_everything, train

seed_everything()
windows = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0)]
models = []
left_value = torch.tensor([[1.0]], device=DEVICE)

for left, right in windows:
    model = MLP(1, 1).to(DEVICE)
    t = torch.linspace(left, right, 200, device=DEVICE).reshape(-1, 1)
    t.requires_grad_(True)
    t_left = torch.tensor([[left]], device=DEVICE)

    def loss_fn():
        u = model(t)
        physics = mse(grad(u, t) + u)
        interface = mse(model(t_left) - left_value)
        return physics + 50.0 * interface, {
            "physics": physics, "interface": interface
        }

    train(model, loss_fn, epochs=3000)
    with torch.no_grad():
        left_value = model(torch.tensor([[right]], device=DEVICE)).detach()
    models.append(model)
```

---

# 26. Neural Operator Code

## 26.1 Minimal DeepONet for a family of ODE solutions

We learn the operator mapping an initial value $u_0$ to $u(t)=u_0e^{-t}$.

```python
import torch
from torch import nn

from pinn_utils import DEVICE, MLP, seed_everything

seed_everything()


class DeepONet(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.branch = MLP(1, latent, hidden=(64, 64))
        self.trunk = MLP(1, latent, hidden=(64, 64))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u0, t):
        return torch.sum(self.branch(u0) * self.trunk(t), dim=1, keepdim=True) + self.bias


model = DeepONet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(8000):
    u0 = 0.1 + 2.9 * torch.rand(512, 1, device=DEVICE)
    t = 5.0 * torch.rand(512, 1, device=DEVICE)
    target = u0 * torch.exp(-t)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(u0, t)
    loss = torch.mean((prediction - target) ** 2)
    loss.backward()
    optimizer.step()
    if epoch % 800 == 0:
        print(epoch, float(loss.detach()))
```

## 26.2 Physics-informed DeepONet

```python
for epoch in range(8000):
    u0 = 0.1 + 2.9 * torch.rand(512, 1, device=DEVICE)
    t = 5.0 * torch.rand(512, 1, device=DEVICE, requires_grad=True)
    prediction = model(u0, t)
    prediction_t = torch.autograd.grad(
        prediction, t, torch.ones_like(prediction), create_graph=True
    )[0]
    residual_loss = torch.mean((prediction_t + prediction) ** 2)
    initial_loss = torch.mean((model(u0, torch.zeros_like(t)) - u0) ** 2)
    loss = residual_loss + 20.0 * initial_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

---

# 27. Current NVIDIA PhysicsNeMo 2.x Code

## 27.1 Installation and verification

```bash
python -m venv physicsnemo-cpu
source physicsnemo-cpu/bin/activate
python -m pip install --upgrade pip
pip install torch numpy scipy sympy matplotlib
pip install "nvidia-physicsnemo[sym]"
python -c "import physicsnemo; print(physicsnemo.__version__)"
```

> PhysicsNeMo is optimized for NVIDIA GPUs, but standard PyTorch models and small symbolic examples can be developed on CPU when the installed dependencies support the host platform. Some NVIDIA examples, optimized kernels, meshes, or large architectures may require CUDA.

## 27.2 Verify a PhysicsNeMo model on CPU

```python
import torch
from physicsnemo.models.mlp.fully_connected import FullyConnected

model = FullyConnected(in_features=1, out_features=1).to("cpu")
x = torch.linspace(0.0, 1.0, 32).reshape(-1, 1)
y = model(x)
print(y.shape)
```

## 27.3 Define a custom symbolic diffusion equation

PhysicsNeMo 2.x expects custom PDEs to be defined inline with SymPy.

```python
from sympy import Function, Symbol
from physicsnemo.sym import PDE


class Diffusion1D(PDE):
    def __init__(self, diffusivity: float = 0.1):
        x = Symbol("x")
        u = Function("u")(x)
        self.equations = {
            "diffusion": -diffusivity * u.diff(x, 2),
        }


pde = Diffusion1D(0.1)
pde.pprint()
```

## 27.4 PhysicsInformer with automatic differentiation

The following example demonstrates the current integration pattern. A custom PyTorch network predicts $u(x)$, and `PhysicsInformer` calculates the symbolic second-derivative residual.

```python
import torch
from torch import nn
from physicsnemo.sym import PDE, PhysicsInformer
from sympy import Function, Symbol


class Poisson1D(PDE):
    def __init__(self):
        x = Symbol("x")
        u = Function("u")(x)
        self.equations = {"poisson": u.diff(x, 2)}


model = nn.Sequential(
    nn.Linear(1, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, 1),
).to("cpu")

physics = PhysicsInformer(
    required_outputs=["poisson"],
    equations=Poisson1D(),
    grad_method="autodiff",
    device="cpu",
)

x = torch.linspace(0.0, 1.0, 128).reshape(-1, 1)
x.requires_grad_(True)
u = model(x)
residuals = physics.forward({"coordinates": x, "u": u})
print(residuals["poisson"].shape)
```

## 27.5 PhysicsNeMo-informed PyTorch training loop

```python
import math
import torch

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x_boundary = torch.tensor([[0.0], [1.0]])

for epoch in range(5000):
    optimizer.zero_grad(set_to_none=True)
    x = torch.rand(256, 1, requires_grad=True)
    u = model(x)
    residual = physics.forward({"coordinates": x, "u": u})["poisson"]
    target_second_derivative = -(math.pi**2) * torch.sin(math.pi * x)
    physics_loss = torch.mean((residual - target_second_derivative) ** 2)
    boundary_loss = torch.mean(model(x_boundary) ** 2)
    loss = physics_loss + 20.0 * boundary_loss
    loss.backward()
    optimizer.step()
```

> If a particular installed PhysicsNeMo build reports that only spatial derivatives are handled automatically, calculate time derivatives with PyTorch autograd and include them in the dictionary passed to `PhysicsInformer`, following that version's API documentation.

---

# 28. BioNeMo: CPU and GPU Code Paths

## 28.1 Important support boundary

The official BioNeMo Framework is a GPU-oriented, Linux-based framework. A CPU-only machine should not be presented as a supported environment for full BioNeMo foundation-model training or fine-tuning. The CPU code below teaches the same integration architecture using lightweight encoders and descriptors. The optional GPU section shows how to prepare a supported BioNeMo environment.

## 28.2 CPU protein-sequence tokenizer

```python
import torch

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
VOCAB = {aa: index + 1 for index, aa in enumerate(AMINO_ACIDS)}
PAD = 0


def encode_protein(sequence: str, max_length: int = 128) -> torch.Tensor:
    sequence = sequence.upper().strip()
    ids = [VOCAB.get(aa, PAD) for aa in sequence[:max_length]]
    ids += [PAD] * (max_length - len(ids))
    return torch.tensor(ids, dtype=torch.long)


sequence = "MKTFFVLLL"
encoded = encode_protein(sequence)
print(encoded.shape)
```

## 28.3 CPU protein encoder

```python
import torch
from torch import nn


class ProteinEncoder(nn.Module):
    def __init__(self, vocab_size=21, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        _, hidden = self.gru(embedded)
        return hidden[-1]


encoder = ProteinEncoder()
batch = torch.stack([encode_protein("MKTFFV"), encode_protein("AGHIKLM")])
embedding = encoder(batch)
print(embedding.shape)
```

## 28.4 CPU SMILES descriptor

This is intentionally simple and dependency-free. It is not a chemically complete parser.

```python
import torch

SMILES_TOKENS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "=", "#", "(", ")"]


def simple_smiles_descriptor(smiles: str) -> torch.Tensor:
    features = [smiles.count(token) for token in SMILES_TOKENS]
    features.extend([
        len(smiles),
        smiles.count("1") + smiles.count("2") + smiles.count("3"),
        smiles.count("+") - smiles.count("-"),
    ])
    return torch.tensor(features, dtype=torch.float32)


print(simple_smiles_descriptor("CC(=O)OC1=CC=CC=C1C(=O)O"))
```

## 28.5 Better CPU descriptors with RDKit

```bash
pip install rdkit
```

```python
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch


def rdkit_descriptor(smiles: str) -> torch.Tensor:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    values = [
        Descriptors.MolWt(molecule),
        Descriptors.MolLogP(molecule),
        Descriptors.TPSA(molecule),
        Descriptors.NumHDonors(molecule),
        Descriptors.NumHAcceptors(molecule),
        Descriptors.NumRotatableBonds(molecule),
        Descriptors.RingCount(molecule),
    ]
    return torch.tensor(values, dtype=torch.float32)
```

## 28.6 Molecule-conditioned pharmacokinetic PINN

This model predicts a compound-specific elimination rate from a descriptor and uses the rate inside the ODE residual.

```python
import torch
from torch import nn

from pinn_utils import DEVICE, MLP, grad, mse, seed_everything

seed_everything()
descriptor_dim = 15
rate_model = MLP(descriptor_dim, 1, hidden=(64, 64)).to(DEVICE)
concentration_model = MLP(descriptor_dim + 1, 1, hidden=(96, 96, 96)).to(DEVICE)

# Synthetic batch of compounds.
descriptors = torch.randn(16, descriptor_dim, device=DEVICE)
true_rates = 0.1 + 0.8 * torch.sigmoid(descriptors[:, :1])

optimizer = torch.optim.Adam(
    list(rate_model.parameters()) + list(concentration_model.parameters()),
    lr=1e-3,
)

for epoch in range(8000):
    compound_index = torch.randint(0, len(descriptors), (256,), device=DEVICE)
    z = descriptors[compound_index]
    t = 8.0 * torch.rand(256, 1, device=DEVICE, requires_grad=True)
    inputs = torch.cat([z, t], dim=1)

    k_pred = nn.functional.softplus(rate_model(z))
    C = concentration_model(inputs)
    C_t = torch.autograd.grad(C, t, torch.ones_like(C), create_graph=True)[0]
    physics_loss = mse(C_t + k_pred * C)

    zero_t = torch.zeros_like(t)
    C0 = concentration_model(torch.cat([z, zero_t], dim=1))
    initial_loss = mse(C0 - 10.0)

    # Optional synthetic parameter supervision.
    parameter_loss = mse(k_pred - true_rates[compound_index])
    loss = physics_loss + 20.0 * initial_loss + parameter_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

## 28.7 Protein-conditioned enzyme kinetics

```python
import torch
from torch import nn

from pinn_utils import DEVICE, MLP, grad, mse

protein_encoder = ProteinEncoder().to(DEVICE)
kinetic_head = MLP(64, 2, hidden=(64, 64)).to(DEVICE)
substrate_model = MLP(64 + 1, 1, hidden=(96, 96, 96)).to(DEVICE)

protein_tokens = torch.stack([
    encode_protein("MKTFFVLLL"),
    encode_protein("AGHIKLMNPQ"),
]).to(DEVICE)

optimizer = torch.optim.Adam(
    list(protein_encoder.parameters())
    + list(kinetic_head.parameters())
    + list(substrate_model.parameters()),
    lr=1e-3,
)

for epoch in range(6000):
    index = torch.randint(0, len(protein_tokens), (128,), device=DEVICE)
    tokens = protein_tokens[index]
    embedding = protein_encoder(tokens)
    kinetic_raw = kinetic_head(embedding)
    Vmax = nn.functional.softplus(kinetic_raw[:, 0:1])
    Km = nn.functional.softplus(kinetic_raw[:, 1:2]) + 1e-4

    t = 5.0 * torch.rand(128, 1, device=DEVICE, requires_grad=True)
    S = nn.functional.softplus(substrate_model(torch.cat([embedding, t], dim=1)))
    S_t = torch.autograd.grad(S, t, torch.ones_like(S), create_graph=True)[0]
    residual = S_t + Vmax * S / (Km + S)
    loss = mse(residual)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

## 28.8 Optional supported BioNeMo GPU environment

The exact container tag changes over time. Select a current BioNeMo Framework image from NVIDIA NGC rather than copying an old tag blindly.

```bash
# Linux with NVIDIA Container Toolkit installed.
# Replace <CURRENT_TAG> with a tag from the NVIDIA NGC BioNeMo catalog.
docker pull nvcr.io/nvidia/clara/bionemo-framework:<CURRENT_TAG>

docker run --rm -it --gpus all \
  -v "$PWD":/workspace/project \
  -w /workspace/project \
  nvcr.io/nvidia/clara/bionemo-framework:<CURRENT_TAG>
```

Inside the container, verify the GPU:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

A general embedding integration pattern is:

```python
# Pseudocode because model-specific BioNeMo APIs, checkpoints, and package names vary.
# Use the README for the selected BioNeMo model package and release.
with torch.no_grad():
    biomolecular_embedding = bionemo_model.encode(batch_of_sequences_or_molecules)

prediction = conditioned_pinn(
    coordinates=space_time_coordinates,
    conditioning=biomolecular_embedding,
)
```

Do not claim this pseudocode is directly executable without selecting a specific BioNeMo model, checkpoint, release, and container.

---

# 29. GPU Scaling Patterns

## 29.1 Automatic CPU/GPU selection

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
```

## 29.2 Mixed precision on an NVIDIA GPU

```python
import torch

scaler = torch.amp.GradScaler("cuda")

for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        prediction = model(batch["inputs"].cuda())
        loss = loss_function(prediction, batch["targets"].cuda())
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

For second-derivative PINNs, mixed precision can reduce derivative accuracy. Compare against float32 before adopting it.

## 29.3 Multi-GPU DistributedDataParallel skeleton

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    model = MyModel().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    # Build a DistributedSampler and ordinary training loop here.
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

Run with:

```bash
torchrun --standalone --nproc_per_node=4 train.py
```

---

# 30. Testing and Validation

## 30.1 Derivative unit test

```python
import torch
from pinn_utils import grad

x = torch.linspace(-1.0, 1.0, 20).reshape(-1, 1).requires_grad_(True)
y = x**3
assert torch.allclose(grad(y, x), 3.0 * x**2, atol=1e-6)
assert torch.allclose(grad(grad(y, x), x), 6.0 * x, atol=1e-5)
print("Derivative tests passed")
```

## 30.2 Boundary-condition test

```python
with torch.no_grad():
    boundary_prediction = model(torch.tensor([[0.0], [1.0]]))
assert float(boundary_prediction.abs().max()) < 1e-2
```

## 30.3 Residual diagnostic

```python
x = torch.linspace(0.0, 1.0, 1000).reshape(-1, 1).requires_grad_(True)
u = model(x)
residual = grad(grad(u, x), x) + torch.pi**2 * torch.sin(torch.pi * x)
print("mean absolute residual:", float(residual.abs().mean()))
print("maximum absolute residual:", float(residual.abs().max()))
```

## 30.4 Save and reload a model

```python
import torch

torch.save({"model_state": model.state_dict()}, "pinn_checkpoint.pt")
checkpoint = torch.load("pinn_checkpoint.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state"])
model.eval()
```

---

# 31. Recommended Execution Order

Run the code in this order:

1. `pinn_utils.py`
2. supervised warm-up
3. exponential decay
4. inverse decay-rate estimation
5. logistic growth
6. damped oscillator
7. Lotka-Volterra or SIR
8. two-compartment pharmacokinetics
9. Poisson equation
10. heat equation
11. inverse diffusivity
12. wave equation
13. Burgers equation
14. reaction-diffusion
15. adaptive sampling
16. DeepONet
17. PhysicsNeMo 2.x symbolic examples
18. CPU protein and molecule encoders
19. molecule-conditioned PINN
20. optional BioNeMo GPU workflow

---

# 32. Reproducibility and Practical Warnings

- CPU execution is deliberately prioritized, but advanced PDE examples can take substantial time. Reduce collocation points and epochs while learning, then increase them for final experiments.
- A small loss does not guarantee a correct solution. Always compare against an analytical solution, a trusted numerical solver, conservation laws, held-out data, and residual plots.
- Inverse problems can be non-identifiable. Different parameter combinations may explain the same sparse observations.
- BioNeMo foundation-model training and fine-tuning should be treated as GPU workflows. The CPU encoders in this document are educational substitutes, not BioNeMo foundation models.
- PhysicsNeMo is undergoing active API evolution. The code in Section 27 follows the PhysicsNeMo 2.x `physicsnemo.sym` and explicit-PyTorch-loop direction; verify the installed release documentation when an import or argument changes.
- Never use all available future observations when evaluating genuine forecasting performance. Separate calibration, validation, and test periods.

---

# 33. Completion Checklist

You now have executable code for:

- PyTorch foundations;
- first-order ODEs;
- nonlinear ODEs;
- second-order ODEs;
- coupled ODE systems;
- stiff kinetics;
- Poisson, heat, wave, Burgers, reaction-diffusion, Fisher-KPP, and coupled biological PDEs;
- inverse scalar parameters;
- inverse PDE coefficients;
- unknown source functions;
- adaptive sampling;
- adaptive loss balancing;
- Adam and L-BFGS optimization;
- Fourier features;
- time-domain decomposition;
- DeepONet and physics-informed DeepONet;
- current PhysicsNeMo symbolic integration;
- CPU biological sequence and molecular representations;
- molecule- and protein-conditioned PINNs;
- optional supported BioNeMo GPU setup;
- GPU mixed precision and multi-GPU patterns;
- tests, diagnostics, and model persistence.

