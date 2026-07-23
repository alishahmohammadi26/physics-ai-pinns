## Introduction

### 1. Motivation: Combining Physical Knowledge with Machine Learning

Many systems in science and engineering are described by ordinary differential
equations (ODEs) or partial differential equations (PDEs). Examples include the
heat equation for thermal diffusion, the Navier–Stokes equations for fluid flow,
the wave equation for acoustics and structural vibration, and reaction–diffusion
equations in chemistry and biology. These equations encode prior knowledge about
how a system behaves through conservation laws, constitutive relations,
symmetries, and balances of mass, momentum, or energy.

Traditionally, differential equations are solved using numerical methods such as
finite differences, finite volumes, finite elements, or spectral methods. These
approaches have mature mathematical foundations and are generally the preferred
tools for accurate forward simulation. However, they usually require some form
of spatial or temporal discretization. Depending on the problem, they may also
require mesh generation, specialized numerical schemes, iterative solvers, and
problem-specific treatment of complicated geometries or boundary conditions.

Conventional numerical solvers may also become difficult to integrate with
sparse, irregular, or noisy experimental measurements. This difficulty is
particularly important in **inverse problems**, where physical parameters,
source terms, material properties, or hidden state variables must be inferred
from observations.

Purely data-driven neural networks approach the problem from the opposite
direction. Instead of starting with a governing equation, they learn a mapping
from inputs to outputs using labelled examples. However, a neural network trained
only on observations does not automatically know that its predictions should
obey physical laws. It may fit the available measurements while violating
conservation principles or producing physically inconsistent behavior between
measurement locations. It may also require a large labelled dataset, which is
problematic when experimental measurements or high-fidelity simulations are
expensive to obtain.

Physics-Informed Neural Networks, commonly abbreviated as **PINNs**, attempt to
connect these two approaches. A PINN represents an unknown physical field using
a neural network and incorporates the governing differential equations,
boundary conditions, initial conditions, and available observations into the
training objective. The model is therefore trained not only to reproduce data,
but also to generate a function that is approximately consistent with the
underlying mathematical model.

The use of neural networks to solve differential equations predates the term
*Physics-Informed Neural Network*. Lagaris, Likas, and Fotiadis developed neural
trial functions for solving ODEs and PDEs and constructed the trial functions so
that selected boundary or initial conditions were satisfied automatically [1].
Sirignano and Spiliopoulos later introduced the Deep Galerkin Method, in which
neural networks were trained by evaluating differential-equation residuals at
randomly sampled space–time points [2]. The modern PINN framework was
popularized by Raissi, Perdikaris, and Karniadakis, who demonstrated a common
neural-network formulation for forward problems, inverse problems, and
data-driven discovery of nonlinear PDEs [3].

### 2. Mathematical Formulation of a PINN

This section builds the PINN framework step by step — starting from a concrete
everyday example, then introducing each mathematical ingredient one at a time,
and finally assembling everything into the full formulation.

#### 2.1 Starting with a Simple Example: Heat in a Metal Rod

Imagine holding one end of a metal rod over a flame. Heat flows from the hot
end toward the cold end, and the temperature at every point along the rod
changes over time. The question a PINN tries to answer is:

> *Given the physics of heat flow, what is the temperature at every point along
> the rod, at every moment in time?*

Let $u(x, t)$ denote the temperature at position $x$ along the rod at time $t$.
This single function $u$ is what we want to find — the **unknown field**.

Physics tells us that heat spreads according to the **heat equation**:

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2},$$

which reads: *"the rate at which temperature changes in time equals the
diffusivity $\alpha$ times the curvature of the temperature profile in space."*
In plain terms, a sharp peak in temperature will spread and flatten over time.

This is the core idea behind all governing equations: they encode a physical
law as a relationship involving the unknown function and its derivatives.

#### 2.2 The Unknown Field

Generalizing from the rod example, a PINN targets an unknown physical quantity
— such as temperature, velocity, pressure, or concentration — that varies across
**space** and possibly **time**:

$$u(\mathbf{x}, t),$$

where:
- $\mathbf{x}$ is a point in the spatial domain $\Omega$ (e.g., a location along
  a rod, on a surface, or inside a 3-D volume); and
- $t \in [0, T]$ is time.

The domain $\Omega$ is simply the region of space where the physics applies —
for the rod, it is the interval from one end to the other.

#### 2.3 The Governing Equation: Writing Down the Physics

For the rod, the heat equation told us how $u$ must behave. In general, any
physical law that involves rates of change can be written as a **governing
equation** of the form:

$$\mathcal{N}\!\left[u(\mathbf{x},t);\,\boldsymbol{\lambda}\right] = s(\mathbf{x},t),
\qquad (\mathbf{x},t)\in\Omega\times(0,T].$$

This looks intimidating, but each piece has a clear meaning:

| Symbol | Plain meaning |
|--------|--------------|
| $\mathcal{N}[\cdot]$ | The physical law — an operator that takes derivatives of $u$ and combines them (e.g., the heat equation, Navier–Stokes, wave equation) |
| $\boldsymbol{\lambda}$ | Known physical constants — e.g., diffusivity $\alpha$, viscosity, wave speed |
| $s(\mathbf{x},t)$ | An external input to the system — e.g., a heater switching on at a specific location |

When there is no external source, $s = 0$ and the equation simply says
"the physics is self-contained." For the heat equation, $\mathcal{N}[u] =
\partial u/\partial t - \alpha\,\partial^2 u/\partial x^2$ and $s = 0$.

#### 2.4 Boundary and Initial Conditions: Pinning Down the Solution

The governing equation alone describes a *family* of possible solutions. To
select the one that matches a specific physical situation, we need two more
ingredients.

**Initial condition** — the state of the system at the start ($t = 0$):

$$u(\mathbf{x}, 0) = u_0(\mathbf{x}).$$

*For the rod: the temperature profile at the moment you first hold it over the
flame.*

**Boundary conditions** — what happens at the edges of the domain at all times:

$$\mathcal{B}[u](\mathbf{x},t) = g(\mathbf{x},t), \qquad \mathbf{x}\in\partial\Omega.$$

$\partial\Omega$ denotes the boundary (e.g., the two ends of the rod).
Common types include:

| Type | Physical meaning | Example |
|------|-----------------|---------|
| **Dirichlet** | Fix the value of $u$ at the boundary | "The left end is held at 100 °C" |
| **Neumann** | Fix the *flux* (rate of flow) at the boundary | "The right end is insulated — no heat escapes" |
| **Robin** | A mix of value and flux | Convective cooling at a surface |
| **Periodic** | Solution repeats across boundaries | A ring-shaped domain |

Together, the governing equation, initial condition, and boundary conditions
fully define the problem. A PINN must satisfy all three.

#### 2.5 The Neural Network as an Approximation

Solving the above problem analytically is often impossible for real-world
geometries, nonlinear equations, or when experimental data is mixed in. A PINN
sidesteps this by using a **neural network** as a flexible, trainable
approximation to the unknown field:

$$u(\mathbf{x}, t) \approx u_{\theta}(\mathbf{x}, t).$$

Here $\theta$ represents all the trainable parameters of the network — its
weights and biases. You can think of the network as a highly adjustable function
that takes coordinates $(x, t)$ as inputs and outputs an estimate of the
temperature (or whatever physical quantity) at that location and time.

The job of training is to find the values of $\theta$ that make
$u_{\theta}$ behave like a physically correct solution.

#### 2.6 Computing Derivatives via Automatic Differentiation

The governing equation involves derivatives of $u$ — such as
$\partial u/\partial t$ or $\partial^2 u/\partial x^2$. Since the network is
a smooth mathematical function, its derivatives can be computed **exactly**
(up to floating-point precision) using the chain rule, a technique called
**automatic differentiation**:

$$\frac{\partial u_{\theta}}{\partial t}, \qquad
\nabla u_{\theta}, \qquad
\nabla^2 u_{\theta}.$$

This is one of the key advantages of PINNs: every derivative that appears in
the physics equation can be evaluated automatically from the network itself,
without needing a computational grid or a finite-difference stencil.

> **Important distinction:** automatic differentiation gives exact derivatives
> of *whatever function the network currently represents*. It does not
> guarantee that function is the correct solution — that depends on whether
> training succeeds.

#### 2.7 The Residual: Measuring How Wrong the Network Is

With the network output and its derivatives in hand, we can check how well the
network satisfies the governing equation at any point. The **residual** is
simply the left-hand side minus the right-hand side of the physics equation:

$$r_{\theta}(\mathbf{x},t)
= \mathcal{N}\!\left[u_{\theta}(\mathbf{x},t);\,\boldsymbol{\lambda}\right]
- s(\mathbf{x},t).$$

*For the heat equation:* $r_\theta = \dfrac{\partial u_\theta}{\partial t} - \alpha\,\dfrac{\partial^2 u_\theta}{\partial x^2}$.

If the network were a perfect solution, this residual would be zero everywhere:

$$r_{\theta}(\mathbf{x},t) = 0.$$

In practice, training pushes the residual toward zero at a large set of
scattered interior points called **collocation points** (also called residual
points or physics points). Think of these as "spot checks" where we verify
that the physics law holds.

#### 2.8 The Loss Function: One Score to Minimize

The training objective combines four types of error into a single **loss**
$\mathcal{L}$ that the optimizer tries to minimize:

$$\mathcal{L}
= w_r\mathcal{L}_r
+ w_b\mathcal{L}_b
+ w_i\mathcal{L}_i
+ w_d\mathcal{L}_d.$$

Each term measures a different kind of mismatch, and the weights
$w_r, w_b, w_i, w_d$ control how much each one matters during training.

---

**Physics residual loss** $\mathcal{L}_r$ — *"Does the network satisfy the
governing equation at interior points?"*

$$\mathcal{L}_r
= \frac{1}{N_r}
\sum_{j=1}^{N_r}
\left| r_{\theta}(\mathbf{x}_j^r, t_j^r) \right|^2.$$

This is the average squared residual over $N_r$ collocation points scattered
inside the domain. It is the uniquely PINN ingredient — no labels are needed,
just the physics equation evaluated at sampled coordinates.

---

**Boundary condition loss** $\mathcal{L}_b$ — *"Does the network respect the
prescribed values or fluxes at the edges?"*

$$\mathcal{L}_b
= \frac{1}{N_b}
\sum_{j=1}^{N_b}
\left|
\mathcal{B}[u_{\theta}](\mathbf{x}_j^b, t_j^b)
- g(\mathbf{x}_j^b, t_j^b)
\right|^2.$$

For the rod: "Is the predicted temperature at the left end actually 100 °C?"

---

**Initial condition loss** $\mathcal{L}_i$ — *"Does the network match the
known starting state?"*

$$\mathcal{L}_i
= \frac{1}{N_i}
\sum_{j=1}^{N_i}
\left|
u_{\theta}(\mathbf{x}_j^i, 0)
- u_0(\mathbf{x}_j^i)
\right|^2.$$

For the rod: "Does the predicted temperature profile at $t=0$ match the
actual initial temperature distribution?"

---

**Data loss** $\mathcal{L}_d$ — *"Does the network match any real
measurements we have?"* (optional — used when experimental data is available)

$$\mathcal{L}_d
= \frac{1}{N_d}
\sum_{j=1}^{N_d}
\left|
u_{\theta}(\mathbf{x}_j^d, t_j^d)
- y_j
\right|^2,$$

where $y_j$ is a measured value at location $(\mathbf{x}_j^d, t_j^d)$.
This term bridges physics and experiment: you can feed in sparse sensor
readings and the network will try to be consistent with both the data *and*
the governing equation simultaneously.

---

#### 2.9 Putting It All Together

Training a PINN is ultimately an **optimization problem**: find the network
parameters $\theta$ that minimize the combined loss $\mathcal{L}$.

Each of the four loss components plays a distinct role:

| Loss term | What it enforces | Where the labels come from |
|-----------|-----------------|---------------------------|
| $\mathcal{L}_r$ | Governing equation (the physics) | No labels needed — just sampled coordinates |
| $\mathcal{L}_b$ | Boundary conditions | Known from the problem setup |
| $\mathcal{L}_i$ | Initial conditions | Known from the problem setup |
| $\mathcal{L}_d$ | Experimental observations | Real measurements (optional) |

Instead of solving a large algebraic system of equations on a mesh, the PINN
converts the entire problem — physics, boundary conditions, initial state, and
data — into a single differentiable objective that gradient descent can
minimize. The result is a neural network that, once trained, can be queried at
any coordinate to predict the physical field and its derivatives [3].

### 3. Why Include a Differential Equation in the Loss Function?

The primary motivation for including the governing differential equation in the
loss is that the equation provides meaningful supervision even at locations
where the value of the solution is unknown.

At an interior collocation point, the correct value of
$u(\mathbf{x},t)$ may not be available. However, if the mathematical model is
assumed to be valid, the desired value of the residual is known:

$$
r_{\theta}(\mathbf{x},t)=0.
$$

Each collocation point therefore provides a physics-based training signal
without requiring a labelled solution value. This is why PINNs are sometimes
described as unsupervised or self-supervised PDE solvers. More precisely, they
are **physics-supervised** models: supervision comes from the governing
equation, boundary and initial conditions, and any available observations.

#### 3.1 Restricting the Space of Possible Solutions

A sufficiently large neural network can represent many different functions.
Only a small subset of these functions will be consistent with a particular
differential equation.

Minimizing the residual loss discourages the network from exploring arbitrary
functions and pushes it toward the set of functions that approximately satisfy
the assumed physical model. In this sense, the differential equation acts as a
strong **inductive bias** or a form of structured regularization.

For example, suppose temperature measurements are available at only a few
locations. A purely data-driven network could fit those values while producing
many arbitrary shapes between the measurement points. Requiring the prediction
to satisfy the heat equation links its spatial curvature to its temporal rate of
change. The network is therefore encouraged to interpolate the observations in
a way that is compatible with thermal diffusion.

The governing equation does not merely add another empirical pattern for the
network to learn. It imposes a structured relationship among the output,
independent variables, and derivatives of the output.

#### 3.2 Providing Information in Unobserved Regions

The differential equation allows information to propagate from observed
regions to unobserved regions.

Suppose observations are available only near one part of the domain. A
data-only model has limited information about the field elsewhere. A PINN can
evaluate the physics residual throughout the domain, including locations where
no measurements exist. The equation therefore acts as a mechanism for
constraining predictions in regions that are not directly observed.

This property can reduce the amount of labelled data required compared with a
purely supervised neural network. However, it does not eliminate the need for
adequate collocation-point coverage. A PINN may still produce inaccurate
behavior in poorly sampled regions.

#### 3.3 Encouraging Physical Consistency

A model trained only on data may produce results that fit the observations but
violate conservation laws or known relationships among physical variables. A
physics residual penalizes such violations.

For example, in a fluid-flow problem, the loss may include:

- conservation of mass;
- conservation of momentum;
- constitutive relationships; and
- boundary conditions at solid walls, inlets, and outlets.

The resulting network is encouraged to produce velocity and pressure fields
that are mutually compatible with the assumed flow equations.

This physical consistency is particularly valuable when observations are noisy
or incomplete. The governing equation can prevent the model from following
every fluctuation in the data when those fluctuations are inconsistent with the
assumed dynamics.

However, the usefulness of the physics term depends on the correctness of the
mathematical model. If the governing equation omits important effects, contains
incorrect parameters, or is applied outside its valid regime, a strong physics
penalty can introduce systematic bias.

#### 3.4 Enabling Parameter Estimation

Because the PINN computational graph is differentiable, unknown physical
parameters can be optimized together with the neural-network weights.

Consider the diffusion equation

$$
\frac{\partial u}{\partial t}
-
\alpha\nabla^2u
=
0,
$$

where the diffusivity $\alpha$ is unknown.

The PINN residual becomes

$$
r_{\theta,\alpha}
=
\frac{\partial u_{\theta}}{\partial t}
-
\alpha\nabla^2u_{\theta}.
$$

Both $\theta$ and $\alpha$ can be treated as trainable variables. The
observations constrain the solution field, while the governing equation
constrains which values of $\alpha$ are physically compatible with that field.
This provides a natural formulation for inverse parameter estimation.

#### 3.5 End-to-End Differentiability

The neural-network representation, physical parameters, and residual
calculations belong to one differentiable computational graph. Gradients of the
complete objective can therefore be evaluated with respect to both network
parameters and unknown physical quantities.

This end-to-end differentiability is one of the reasons PINNs are attractive for
inverse problems, data assimilation, model calibration, and design
optimization.

It should nevertheless be noted that differentiability alone does not guarantee
that the resulting optimization problem is easy to solve. PINN objectives are
often highly nonconvex, poorly scaled, or ill-conditioned.

#### 3.6 Soft Constraints Rather Than Exact Enforcement

In a standard PINN, the differential equation is usually imposed as a **soft
constraint**. The optimizer attempts to reduce the residual, but the residual is
not guaranteed to be zero.

A small empirical residual means that the equation is approximately satisfied
at the selected collocation points. It does not automatically prove that:

- the residual is small everywhere;
- the network is close to the exact solution;
- the correct solution branch has been identified; or
- the governing equation is an accurate description of the real system.

The relationship between residual error and solution error depends on the
stability and well-posedness properties of the underlying differential
equation.

### 4. Why Boundary and Initial Conditions Are Essential

The governing equation alone generally does not determine a unique physical
solution. Boundary and initial conditions identify the particular solution
associated with the physical experiment or engineering configuration.

Consider the simple ODE

$$
\frac{du}{dx}-u=0.
$$

Its general solution is

$$
u(x)=Ce^x,
$$

where $C$ is an arbitrary constant.

Every value of $C$ produces a function that satisfies the differential
equation exactly. Therefore, minimizing only the differential-equation residual
cannot determine which solution is required.

The initial condition

$$
u(0)=1
$$

selects the unique solution

$$
u(x)=e^x.
$$

The same principle applies to PDEs. A governing equation may define a family of
possible solutions, while the boundary and initial conditions select the
solution relevant to a particular physical problem.

Boundary and initial conditions may specify quantities such as:

- an imposed wall temperature;
- an inlet velocity;
- a heat or mass flux;
- an initial displacement or velocity;
- an initial concentration distribution;
- periodic behavior; or
- continuity across an interface.

Without these conditions, the problem may be non-unique, underdetermined, or
physically meaningless. In some homogeneous problems, minimizing only the PDE
residual may even lead to a trivial solution such as

$$
u_{\theta}=0,
$$

because the zero function satisfies the governing equation but not the intended
boundary or initial data.

### 5. Soft and Hard Enforcement of Boundary Conditions

Boundary and initial conditions can be incorporated into PINNs through either
soft or hard enforcement.

#### 5.1 Soft Enforcement

In soft enforcement, violations of the conditions are added to the loss
function:

$$
\mathcal{L}
=
w_r\mathcal{L}_r
+
w_b\mathcal{L}_b
+
w_i\mathcal{L}_i.
$$

This approach is straightforward and flexible. It can accommodate:

- Dirichlet, Neumann, Robin, and mixed boundary conditions;
- noisy or uncertain boundary measurements;
- complicated systems with multiple physical variables; and
- conditions that are expected to hold only approximately.

However, soft enforcement does not guarantee that the boundary or initial
conditions will be satisfied exactly. The optimizer must balance the residual,
boundary, initial, and data objectives. Poor loss weighting may cause one
condition to be neglected while another is optimized.

#### 5.2 Hard Enforcement

In hard enforcement, the neural-network output is constructed so that selected
conditions are satisfied for every possible value of the trainable parameters.

For example, the initial condition

$$
u(\mathbf{x},0)=u_0(\mathbf{x})
$$

can be imposed using

$$
u_{\theta}(\mathbf{x},t)
=
u_0(\mathbf{x})
+
t\widetilde{u}_{\theta}(\mathbf{x},t).
$$

At $t=0$,

$$
u_{\theta}(\mathbf{x},0)
=
u_0(\mathbf{x}),
$$

regardless of the value produced by
$\widetilde{u}_{\theta}$.

Similarly, a Dirichlet boundary condition may be imposed using

$$
u_{\theta}(\mathbf{x},t)
=
g^*(\mathbf{x},t)
+
d(\mathbf{x})\widetilde{u}_{\theta}(\mathbf{x},t),
$$

where:

- $g^*$ is a function that satisfies the required boundary values; and
- $d(\mathbf{x})=0$ on the boundary.

The second term can modify the interior solution without changing the prescribed
boundary value.

This trial-function approach appeared in early neural methods for differential
equations [1]. More recent approaches use distance functions and
geometry-dependent constructions to impose Dirichlet, Neumann, and Robin
conditions on more complicated domains [4].

Hard enforcement has several potential advantages:

- boundary-condition error is eliminated for the enforced conditions;
- the optimization search space is reduced;
- competition between the PDE and boundary loss terms is reduced; and
- less tuning of boundary-loss weights may be required.

Its main disadvantage is that constructing a suitable trial function can be
difficult for complex geometries, moving boundaries, coupled multiphysics
systems, or uncertain boundary data.

### 6. What “Mesh-Free” Means

PINNs are often described as **mesh-free**, but this term must be interpreted
carefully.

In a finite-difference method, the domain is typically represented by a grid,
and derivatives are approximated using predefined stencils. In a conventional
finite-element method, the domain is divided into connected elements. The
method uses element connectivity, local basis functions, numerical quadrature,
and an assembly procedure to construct a global system of equations.

A standard PINN does not require:

- an element-connectivity table;
- a finite-difference stencil;
- a conventional finite-element mesh; or
- assembly of a global stiffness or residual matrix.

Instead, the governing-equation residual is evaluated at a collection of
coordinates. These residual points may be distributed irregularly and do not
need to be connected to one another.

Possible sampling strategies include:

- uniform grids;
- random uniform sampling;
- Latin hypercube sampling;
- Sobol or Halton low-discrepancy sequences;
- importance sampling;
- residual-based adaptive sampling; and
- repeated resampling during training.

The Deep Galerkin Method demonstrated this idea by training neural networks at
randomly sampled space–time points rather than relying on a fixed connected
mesh [2]. Later studies showed that the number and distribution of collocation
points can strongly influence PINN accuracy and that residual-based adaptive
sampling can improve performance by concentrating points in difficult regions
[5].

#### 6.1 Practical Advantages of Mesh-Free Collocation

Mesh-free collocation provides several forms of flexibility:

1. Interior points can be added, removed, or moved without rebuilding element
   connectivity.
2. Experimental measurements collected at irregular locations can be included
   directly.
3. More points can be placed near boundaries, interfaces, steep gradients, or
   regions of high residual.
4. Space and time can be treated together as neural-network inputs.
5. Once trained, the network can be evaluated at coordinates that were not used
   during training.

This flexibility can be useful for high-dimensional problems, moving
interfaces, irregular measurement locations, and inverse problems involving
sparse observations.

#### 6.2 Mesh-Free Does Not Mean Discretization-Free

A PINN is mesh-free in the sense that it does not require a connected
tessellation of the computational domain. It is not completely
discretization-free.

The continuous residual objective would ideally involve an integral such as

$$
\int_{\Omega\times[0,T]}
\left|
r_{\theta}(\mathbf{x},t)
\right|^2
\,d\mathbf{x}\,dt.
$$

In practice, this integral is approximated using a finite collection of
collocation points:

$$
\frac{1}{N_r}
\sum_{j=1}^{N_r}
\left|
r_{\theta}(\mathbf{x}_j^r,t_j^r)
\right|^2.
$$

Therefore, PINNs still rely on a discrete sampling of the domain. Their accuracy
can depend on:

- the number of collocation points;
- the distribution of those points;
- the sampling or quadrature rule;
- the treatment of the boundary;
- the frequency of resampling; and
- whether important localized features are adequately sampled.

Mesh-free also does not mean **geometry-free**. Complex geometries must still be
represented through boundary point clouds, signed-distance functions,
constructive geometry, coordinate mappings, or other geometric descriptions.

Finally, mesh-free does not imply that a PINN will automatically be faster or
more accurate than a mesh-based numerical solver. It describes how the domain
and residual are represented, not the overall computational efficiency of the
method.

### 7. Forward Problems, Inverse Problems, and Data Assimilation

PINNs are commonly used for three related classes of problems.

#### 7.1 Forward Problems

In a forward problem, the following quantities are assumed to be known:

- the governing equation;
- physical parameters;
- source or forcing terms;
- boundary conditions; and
- initial conditions.

The objective is to determine the solution field
$u(\mathbf{x},t)$.

A PINN can solve the problem using the governing equation and auxiliary
conditions without requiring a labelled full-field solution. However, this does
not necessarily mean that the PINN will be more computationally efficient than a
classical numerical solver.

#### 7.2 Inverse Problems

In an inverse problem, one or more parts of the physical model are unknown.
Examples include:

- scalar physical coefficients;
- spatially varying material properties;
- unknown source terms;
- unknown initial conditions;
- hidden state variables; or
- constitutive relationships.

The unknown quantities can be trained jointly with the network parameters.
Observations constrain the state, while the governing equation restricts which
states and parameter values are physically compatible with those observations
[3].

#### 7.3 Data Assimilation and Hidden-Field Reconstruction

In data assimilation, incomplete or noisy measurements are combined with a
physical model to reconstruct unobserved quantities.

An influential example is the hidden-fluid-mechanics framework, which combined
flow-visualization data with the Navier–Stokes equations to infer velocity and
pressure fields that were not directly measured [6].

This example illustrates one of the most attractive uses of PINNs: measurements,
hidden variables, unknown parameters, and governing equations can be combined in
a single differentiable optimization framework.

However, including physics does not make every inverse problem uniquely
solvable. Successful inference still depends on **identifiability**. The
measurements must contain sufficient information to distinguish the unknown
parameters or fields. Theoretical studies of PINNs for inverse problems connect
the reconstruction error to conditional stability properties of the underlying
inverse problem [7].

### 8. Principal Advantages of PINNs

#### 8.1 Reduced Dependence on Labelled Full-Field Data

The governing equation provides residual targets throughout the computational
domain. A dense labelled solution obtained from experiments or high-fidelity
simulations is therefore not always required.

This should be interpreted as **label efficiency**, not necessarily
computational efficiency. A PINN may use little labelled data while still
requiring many collocation points and a substantial amount of optimization.

#### 8.2 Integration of Data and Physical Models

PINNs naturally combine:

- governing equations;
- boundary and initial conditions;
- sparse observations;
- unknown parameters; and
- hidden physical fields.

This integration is especially valuable when neither data nor a mathematical
model is sufficient on its own.

#### 8.3 A Common Framework for Forward and Inverse Problems

The same general computational framework can be used to approximate a solution
field, estimate unknown coefficients, infer source terms, or reconstruct hidden
variables. The primary changes involve which quantities are treated as known,
observed, or trainable.

#### 8.4 Continuous Coordinate-Based Representation

A trained PINN defines a function

$$
u_{\theta}(\mathbf{x},t)
$$

that can be evaluated at arbitrary coordinates. Derived quantities such as
gradients, fluxes, stresses, or sensitivities can also be calculated through
automatic differentiation.

The term *continuous* refers to the form of the neural representation. It does
not guarantee that the prediction is accurate at every coordinate.

#### 8.5 Flexible Use of Irregular Measurements

Measurements do not need to lie on a regular computational grid. Data collected
at scattered spatial or temporal locations can be included directly in the loss
function.

#### 8.6 Differentiability with Respect to Physical Parameters

Unknown coefficients can be included directly in the computational graph and
estimated using gradient-based optimization. This can simplify the formulation
of some parameter-identification and model-calibration problems.

#### 8.7 Potential Reuse After Training

Once trained, a neural network can be inexpensive to evaluate at many query
points. This can be beneficial when repeated evaluation of one trained solution
is required.

However, a basic PINN usually learns one particular problem instance. It does
not automatically learn a reusable solution operator for arbitrary initial
conditions, forcing functions, material parameters, or geometries. Broader
generalization normally requires parameterized PINNs, transfer learning, or
neural-operator methods.

### 9. Limitations and Failure Modes

Despite their flexibility, PINNs are not universal replacements for conventional
numerical solvers.

#### 9.1 Multi-Objective Loss Imbalance

PINN training is inherently a multi-objective optimization problem. The PDE
residual, boundary conditions, initial conditions, interface conditions, and
data terms may have different:

- physical units;
- numerical magnitudes;
- derivative orders;
- gradient scales; and
- convergence rates.

A fixed weighted sum may therefore produce strongly imbalanced gradients. The
optimizer may reduce the PDE residual while neglecting the boundary conditions,
or fit the observations while leaving a large physics error.

Gradient-flow analyses have identified numerical stiffness and gradient
imbalance as important causes of difficult PINN training [8]. Neural tangent
kernel analyses have similarly shown that different components of a PINN loss
may converge at significantly different rates [9].

#### 9.2 Difficult Optimization Landscapes

A neural network may have enough representational capacity to approximate a
solution, but the optimizer may still fail to find that solution.

Studies of convection, reaction, and reaction–diffusion equations have shown
that the soft physics constraint can create poorly conditioned optimization
problems [10]. This distinction is important:

- **representation error** occurs when the neural-network class cannot express
  the desired solution;
- **optimization error** occurs when training fails to locate a good
  representation; and
- **sampling error** occurs when the collocation points do not adequately
  represent the continuous domain.

Increasing network size alone does not necessarily resolve optimization or
sampling errors.

#### 9.3 Spectral Bias and Multiscale Behavior

Standard fully connected neural networks tend to learn smooth,
low-frequency components before high-frequency components. This behavior is
often called **spectral bias**.

As a result, conventional PINNs may struggle with:

- rapidly oscillating waves;
- multiscale solutions;
- thin boundary layers;
- high-frequency forcing;
- sharp fronts; and
- localized solution features.

Fourier-feature networks and other multiscale representations have been proposed
to improve the learning of high-frequency solution components [11].

#### 9.4 High-Order Derivatives

Strong-form PINNs directly evaluate all derivatives appearing in the governing
equation. High-order derivatives can:

- increase computational cost;
- require smooth activation functions;
- amplify numerical and optimization difficulties; and
- make training more sensitive to network initialization.

Weak and variational formulations reduce the derivative order applied to the
neural-network approximation by transferring some derivatives to test functions
through integration by parts. Variational Physics-Informed Neural Networks
(VPINNs) were developed in part to exploit this idea [12].

#### 9.5 Discontinuities and Shocks

Standard PINNs usually employ smooth neural networks and pointwise
strong-form residuals. These assumptions are poorly matched to discontinuous
solutions such as shocks, contact discontinuities, and material interfaces.

A smooth network may smear a discontinuity or generate oscillations around it.
Specialized weak formulations, conservation-law treatments, domain
decomposition, adaptive sampling, or discontinuity-aware architectures may be
required.

#### 9.6 Sensitivity to Collocation-Point Sampling

A network can obtain a small residual at the training points while retaining a
large residual in unsampled regions.

Adding more points uniformly may be inefficient if the main errors are
concentrated in a small part of the domain. Residual-based adaptive sampling
attempts to address this problem by placing additional points in high-error
regions [5].

However, adaptive sampling is itself challenging because a low predicted
residual does not always indicate a low solution error, particularly for
unstable or poorly conditioned equations.

#### 9.7 Training Loss Is Not the Same as Solution Error

A low empirical PINN loss does not automatically imply that

$$
u_{\theta}\approx u,
$$

where $u$ is the exact solution.

The connection between residual loss and solution error depends on:

- well-posedness of the PDE;
- stability of the governing operator;
- the norm used in the loss;
- enforcement of boundary and initial conditions;
- sampling or quadrature accuracy;
- neural-network approximation capacity; and
- optimization accuracy.

Generalization-error estimates have been established for several classes of
forward PDE problems [13]. Convergence results have also been developed for
selected linear elliptic and parabolic equations [14]. More recent research has
continued to examine the stability and convergence of PINN approximations [15]
and to develop unified frameworks that separate approximation, optimization,
and quadrature errors [16].

These theoretical developments are important, but they do not yet provide a
universal convergence guarantee for every nonlinear PDE, neural architecture,
sampling strategy, and nonconvex optimization algorithm.

#### 9.8 Computational Cost

PINN training often requires:

- repeated evaluation of high-order derivatives;
- large numbers of collocation points;
- many optimization iterations;
- careful loss-weight tuning; and
- high-precision arithmetic for difficult problems.

For routine low-dimensional forward simulations, mature numerical methods may
be considerably faster and more accurate.

A systematic comparison between PINNs and the finite-element method found that
the tested PINNs did not outperform finite elements in solution time or accuracy
for the considered PDE problems, although the trained neural networks could be
fast to evaluate afterward [17].

A broader benchmark covering numerous PDEs and PINN variants likewise found
that performance is strongly problem-dependent. Important challenges included
multiscale behavior, nonlinear dynamics, complex geometries, loss imbalance, and
domain decomposition [18].

These findings do not imply that PINNs are ineffective. Instead, they show that
PINNs are most compelling when their specific advantages—such as data
assimilation, parameter inference, irregular observations, or differentiable
state reconstruction—are important to the application.

#### 9.9 Model Error

A PINN assumes that the governing equation included in the loss is an adequate
description of the physical system.

Real systems may contain:

- unmodeled forcing;
- uncertain constitutive relationships;
- unresolved scales;
- inaccurate boundary conditions;
- measurement bias; or
- parameters that vary in space or time.

If the physics model is incorrect, forcing the network to satisfy it too
strongly can reduce predictive accuracy. In such cases, the relative weights of
the physics and data terms reflect a trade-off between trusting the mathematical
model and trusting the observations.

### 10. Current Research Directions

Most advanced PINN research can be interpreted as addressing one or more of the
limitations described above.

#### 10.1 Adaptive Loss Weighting

Adaptive weighting methods attempt to balance the PDE, boundary, initial, and
data losses during training. Proposed approaches use gradient magnitudes,
learning rates, uncertainty parameters, neural tangent kernels, or
multi-objective optimization principles.

#### 10.2 Nondimensionalization and Equation Scaling

Variables and equations may differ by several orders of magnitude.
Nondimensionalization can place variables, derivatives, and residual terms on
more comparable scales and improve optimization.

#### 10.3 Curriculum and Sequential Training

For long-time integration or strongly nonlinear dynamics, training may proceed
from easier to harder tasks. Examples include:

- training on short time intervals before longer intervals;
- gradually increasing physical parameters;
- progressively adding higher-frequency features; and
- moving through the time domain sequentially.

#### 10.4 Adaptive Sampling

Residual points can be added in regions with high residuals, steep gradients,
interfaces, or large estimated errors. Sampling can also be changed dynamically
during optimization.

#### 10.5 Hard Constraint Enforcement

Distance functions, coordinate transformations, and geometry-aware trial
functions can enforce selected boundary and initial conditions exactly. This
reduces competition between the constraint terms in the loss.

#### 10.6 Weak and Variational Formulations

Weak-form PINNs and VPINNs replace pointwise strong-form residuals with
integrated residual statements. These formulations can reduce derivative order
and may be better suited to solutions with limited regularity.

#### 10.7 Domain Decomposition

A large domain can be divided into smaller subdomains, with a separate neural
network assigned to each subdomain.

Extended Physics-Informed Neural Networks (XPINNs) use space–time domain
decomposition and impose compatibility conditions at the interfaces [19].
Domain decomposition can provide:

- additional local representation capacity;
- parallelization opportunities;
- better treatment of localized behavior; and
- separate approximation strategies in different physical regions.

The hp-VPINN framework combines variational residuals, domain decomposition, and
polynomial test functions inspired by classical $hp$-finite-element methods
[20].

#### 10.8 Improved Architectures

Research has explored Fourier features, sinusoidal activations, residual
connections, multiplicative architectures, multiscale networks, and specialized
operator-based architectures. These modifications seek to address spectral
bias, gradient propagation, and multiscale dynamics.

#### 10.9 Hybrid Numerical and Neural Methods

An increasingly important direction is to combine PINNs with established
numerical methods rather than treating them as competing approaches.

Examples include:

- neural corrections to reduced-order models;
- PINNs coupled with finite-element solvers;
- neural constitutive models within numerical simulations;
- physics-informed surrogate models;
- learned preconditioners;
- differentiable numerical solvers; and
- neural estimation of unknown closure terms.

These hybrid methods can retain the robustness and conservation properties of
classical solvers while using neural networks for inference, acceleration, or
model correction.

### 11. Overall Perspective

PINNs are best understood as a **differentiable framework for
physics-constrained approximation and inference**, rather than as a universal
replacement for finite-difference, finite-volume, finite-element, or spectral
methods.

Their central idea is powerful: a governing differential equation provides
training information at locations where labelled solution values are
unavailable. Boundary and initial conditions select the physically relevant
solution, while observations can anchor the approximation and support parameter
identification.

The term *mesh-free* means that the residual can be evaluated at scattered
coordinates without constructing a connected computational mesh. It does not
mean that sampling, geometry representation, or numerical approximation is
eliminated.

PINNs are particularly attractive when:

- measurements are sparse, noisy, or irregular;
- unknown physical parameters must be inferred;
- hidden fields must be reconstructed;
- equations and observations must be combined;
- a differentiable representation of the state is required; or
- repeated evaluation of a trained approximation is useful.

At the same time, PINNs introduce a difficult optimization problem. Their
accuracy depends on the governing equation, boundary treatment, loss weighting,
sampling strategy, network architecture, parameter identifiability, numerical
precision, and optimizer.

Consequently, the main research question is no longer simply whether a neural
network can represent the solution of a differential equation. The more
important question is how to design the representation, physical constraints,
sampling strategy, and optimization procedure so that the correct solution can
be identified reliably and at a computational cost appropriate for the intended
application.


## 12. Notebook index

| # | Notebook | Level | What you build |
|---|---|---|---|
| 01 | [Intro to PINNs](notebooks/01_Beginner/01_intro_to_pinn.ipynb) | Beginner | Solve `dy/dx=-y` from scratch; autodiff; sparse-data variant |
| 02 | [Second-order ODEs](notebooks/01_Beginner/02_second_order_odes.ipynb) | Beginner | Damped oscillator; two ICs; hard-constraint ansatz |
| 03 | [Heat equation](notebooks/02_Intermediate/03_heat_equation.ipynb) | Intermediate | First PDE `(x,t)→u`; IC + BCs; space-time field |
| 04 | [Burgers' equation](notebooks/02_Intermediate/04_burgers_equation.ipynb) | Intermediate | Nonlinear PDE; Adam→L-BFGS; shock layer |
| 05 | [Reaction kinetics](notebooks/02_Intermediate/05_reaction_kinetics.ipynb) | Intermediate | Coupled ODE system A→B→C; mass conservation |
| 06 | [Wave equation](notebooks/02_Intermediate/06_wave_equation.ipynb) | Intermediate | Hyperbolic PDE; initial velocity condition |
| 07 | [Inverse parameter estimation](notebooks/03_Advanced/07_inverse_parameter_estimation.ipynb) | Advanced | Learn a diffusivity from sparse noisy data |
| 08 | [Adaptive sampling (RAR)](notebooks/03_Advanced/08_adaptive_sampling.ipynb) | Advanced | Boundary-layer problem; residual-based refinement |
| 09 | [DeepONet](notebooks/03_Advanced/09_deeponet.ipynb) | Advanced | Operator learning; branch/trunk from scratch |
| 10 | [Navier–Stokes (Taylor–Green)](notebooks/04_Research_Papers/10_navier_stokes_cylinder.ipynb) | Research | Infer pressure + viscosity from velocity data |
| 11 | [Physics-informed DeepONet](notebooks/04_Research_Papers/11_physics_informed_deeponet.ipynb) | Research | Operator learning with **zero** labelled data |
| 12 | [Architectures & training](notebooks/04_Research_Papers/12_architectures_and_training.ipynb) | Research | Fourier features vs spectral bias; PirateNets/PIKANs map |



## 13. Repository structure

```
PINN-Learning-Series/
├── README.md                         ← you are here
├── requirements.txt
├── notebooks/
│   ├── 01_Beginner/                  01–02
│   ├── 02_Intermediate/              03–06
│   ├── 03_Advanced/                  07–09
│   └── 04_Research_Papers/           10–12
├── articles/
│   ├── Beginner_Guides/
│   ├── Tutorials/
│   ├── Case_Studies/
│   └── Research_Summaries/
├── references/
│   ├── *.pdf                         provided seminal papers
│   └── Papers_and_Bibliography/      review, annotated bibliography, .bib
├── datasets/                         (optional) original benchmark data
└── src/                              shared helper utilities
```


## References

1. Lagaris, I. E., Likas, A., & Fotiadis, D. I. (1998). Artificial neural
   networks for solving ordinary and partial differential equations. *IEEE
   Transactions on Neural Networks, 9*(5), 987–1000.
   [doi:10.1109/72.712178](https://doi.org/10.1109/72.712178)

2. Sirignano, J., & Spiliopoulos, K. (2018). DGM: A deep learning algorithm for
   solving partial differential equations. *Journal of Computational Physics,
   375*, 1339–1364.
   [doi:10.1016/j.jcp.2018.08.029](https://doi.org/10.1016/j.jcp.2018.08.029)

3. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
   Physics-informed neural networks: A deep learning framework for solving
   forward and inverse problems involving nonlinear partial differential
   equations. *Journal of Computational Physics, 378*, 686–707.
   [doi:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

4. Sukumar, N., & Srivastava, A. (2022). Exact imposition of boundary
   conditions with distance functions in physics-informed deep neural networks.
   *Computer Methods in Applied Mechanics and Engineering, 389*, 114333.
   [doi:10.1016/j.cma.2021.114333](https://doi.org/10.1016/j.cma.2021.114333)

5. Wu, C., Zhu, M., Tan, Q., Kartha, Y., & Lu, L. (2023). A comprehensive
   study of non-adaptive and residual-based adaptive sampling for
   physics-informed neural networks. *Computer Methods in Applied Mechanics and
   Engineering, 403*, 115671.
   [doi:10.1016/j.cma.2022.115671](https://doi.org/10.1016/j.cma.2022.115671)

6. Raissi, M., Yazdani, A., & Karniadakis, G. E. (2020). Hidden fluid
   mechanics: Learning velocity and pressure fields from flow visualizations.
   *Science, 367*(6481), 1026–1030.
   [doi:10.1126/science.aaw4741](https://doi.org/10.1126/science.aaw4741)

7. Mishra, S., & Molinaro, R. (2022). Estimates on the generalization error of
   physics-informed neural networks for approximating a class of inverse
   problems for PDEs. *IMA Journal of Numerical Analysis, 42*(2), 981–1022.
   [doi:10.1093/imanum/drab032](https://doi.org/10.1093/imanum/drab032)

8. Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating
   gradient flow pathologies in physics-informed neural networks. *SIAM Journal
   on Scientific Computing, 43*(5), A3055–A3081.
   [doi:10.1137/20M1318043](https://doi.org/10.1137/20M1318043)

9. Wang, S., Yu, X., & Perdikaris, P. (2022). When and why PINNs fail to
   train: A neural tangent kernel perspective. *Journal of Computational
   Physics, 449*, 110768.
   [doi:10.1016/j.jcp.2021.110768](https://doi.org/10.1016/j.jcp.2021.110768)

10. Krishnapriyan, A. S., Gholami, A., Zhe, S., Kirby, R. M., & Mahoney,
    M. W. (2021). Characterizing possible failure modes in physics-informed
    neural networks. *Advances in Neural Information Processing Systems, 34*.

11. Wang, S., Wang, H., & Perdikaris, P. (2021). On the eigenvector bias of
    Fourier feature networks: From regression to solving multiscale PDEs with
    physics-informed neural networks. *Computer Methods in Applied Mechanics and
    Engineering, 384*, 113938.
    [doi:10.1016/j.cma.2021.113938](https://doi.org/10.1016/j.cma.2021.113938)

12. Kharazmi, E., Zhang, Z., & Karniadakis, G. E. (2019). Variational
    physics-informed neural networks for solving partial differential
    equations. *arXiv preprint arXiv:1912.00873*.
    [doi:10.48550/arXiv.1912.00873](https://doi.org/10.48550/arXiv.1912.00873)

13. Mishra, S., & Molinaro, R. (2023). Estimates on the generalization error
    of physics-informed neural networks for approximating PDEs. *IMA Journal of
    Numerical Analysis, 43*(1), 1–43.
    [doi:10.1093/imanum/drab093](https://doi.org/10.1093/imanum/drab093)

14. Shin, Y., Darbon, J., & Karniadakis, G. E. (2020). On the convergence of
    physics informed neural networks for linear second-order elliptic and
    parabolic type PDEs. *Communications in Computational Physics, 28*(5),
    2042–2074.
    [doi:10.4208/cicp.OA-2020-0193](https://doi.org/10.4208/cicp.OA-2020-0193)

15. Gazoulis, D., Gkanis, I., & Makridakis, C. G. (2025). On the stability
    and convergence of physics informed neural networks. *IMA Journal of
    Numerical Analysis*, draf090.
    [doi:10.1093/imanum/draf090](https://doi.org/10.1093/imanum/draf090)

16. Zeinhofer, M., Masri, R., & Mardal, K.-A. (2025). A unified framework for
    the error analysis of physics-informed neural networks. *IMA Journal of
    Numerical Analysis, 45*(5), 2988–3025.
    [doi:10.1093/imanum/drae081](https://doi.org/10.1093/imanum/drae081)

17. Grossmann, T. G., Komorowska, U. J., Latz, J., & Schönlieb, C.-B. (2024).
    Can physics-informed neural networks beat the finite element method? *IMA
    Journal of Applied Mathematics, 89*(1), 143–174.
    [doi:10.1093/imamat/hxae011](https://doi.org/10.1093/imamat/hxae011)

18. Hao, Z., Yao, J., Su, C., Su, H., Wang, Z., Lu, F., Xia, Z., Zhang, Y.,
    Liu, S., Lu, L., & Zhu, J. (2024). PINNacle: A comprehensive benchmark of
    physics-informed neural networks for solving PDEs. *Advances in Neural
    Information Processing Systems, 37*, 76721–76774.

19. Jagtap, A. D., & Karniadakis, G. E. (2020). Extended physics-informed
    neural networks (XPINNs): A generalized space–time domain decomposition
    based deep learning framework for nonlinear partial differential equations.
    *Communications in Computational Physics, 28*(5), 2002–2041.
    [doi:10.4208/cicp.OA-2020-0164](https://doi.org/10.4208/cicp.OA-2020-0164)

20. Kharazmi, E., Zhang, Z., & Karniadakis, G. E. (2021). hp-VPINNs:
    Variational physics-informed neural networks with domain decomposition.
    *Computer Methods in Applied Mechanics and Engineering, 374*, 113547.
    [doi:10.1016/j.cma.2020.113547](https://doi.org/10.1016/j.cma.2020.113547)