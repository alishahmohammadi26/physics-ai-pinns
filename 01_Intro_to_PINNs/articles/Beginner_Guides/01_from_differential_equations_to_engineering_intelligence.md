# Physics-Informed Neural Networks: From Differential Equations to Engineering Intelligence

*Why the future of scientific AI is not about replacing physics — but embedding it
directly into machine-learning models.*

**Companion notebook:** [`notebooks/01_Beginner/01_intro_to_pinn.ipynb`](../../notebooks/01_Beginner/01_intro_to_pinn.ipynb)

---

## Who this is for
Chemical and process engineers, pharmaceutical scientists, CFD modellers, data
scientists moving into scientific AI, graduate students, and anyone curious about
NVIDIA Modulus and scientific machine learning. No prior PINN knowledge is
assumed — only comfort with derivatives and a little Python.

---

## 1 · The problem: why traditional ML struggles in engineering

Large language models work because the internet supplies billions of text
examples, and language has enormous statistical redundancy. Engineering is the
opposite world. Ask why a model of reaction kinetics or turbulent flow is so much
harder than autocomplete and you arrive at three facts:

- **Engineering data is scarce and expensive.** A single high-fidelity CFD run or
  lab experiment can cost hours and money. There is no billion-example corpus.
- **Engineering systems obey laws.** Mass, momentum, and energy are conserved.
  These laws are known *exactly*, as differential equations.
- **Ignoring the laws is costly.** A purely data-driven model can fit its training
  points and still violate conservation, extrapolate nonsensically, and demand
  far more data than you can afford.

The central idea of this series: instead of hoping a network *learns* physics from
data, we **build the physics into the loss function** so the network cannot help
but respect it.

## 2 · What is a physics-informed neural network?

Consider the simplest possible example:

$$\frac{dy}{dx} = -y, \qquad y(0) = 1.$$

A traditional network would need a table of `(x, y)` pairs and would learn the
mapping `x → y`. A PINN needs **no solution data at all**. Instead we tell the
network:

> *Whatever function `y(x)` you represent, its derivative must equal `-y`
> everywhere, and it must pass through `y(0) = 1`.*

Concretely, we define two penalties:
- a **physics (residual) loss**: `mean( (dy/dx + y)^2 )` sampled at many points,
- a **boundary/initial-condition loss**: `(y(0) - 1)^2`.

Minimising their sum forces the network to become the solution `y = e^{-x}`. The
difference from a conventional network is one line of philosophy:

```
Traditional NN:   Data            → Loss
PINN:             Data + Physics  → Loss   (and often no data at all)
```

Because the constraint is the equation itself, **PINNs need no labelled outputs**.

## 3 · The magic: automatic differentiation

How can a network evaluate `dy/dx`, `d²y/dx²`, `∂u/∂x`, `∂²u/∂y²` — the very terms
the equation is made of? Not by finite differences. Modern frameworks (PyTorch,
TensorFlow, JAX) compute derivatives **exactly** through **automatic
differentiation** (autodiff): the same mechanism that backpropagates gradients to
train the weights can differentiate the network's *output* with respect to its
*input*.

This is usually the "aha" moment. The network is a differentiable function, so any
derivative appearing in your PDE is available exactly and cheaply. There is no
mesh, no stencil, no discretisation error in the derivatives themselves. In
PyTorch it is a single call to `torch.autograd.grad`, and you can nest it to get
second and higher derivatives.

## 4 · Building your first PINN

The companion notebook walks through six steps for `dy/dx = -y`, `y(0)=1`:

1. **Define the network** — a small fully-connected net with tanh activations
   (smooth, so it is infinitely differentiable).
2. **Define the domain** — sample "collocation" points where the equation will be
   enforced.
3. **Define the residual** — `dy/dx + y`, computed via autodiff.
4. **Define the boundary loss** — `(y(0) - 1)^2`.
5. **Train** — Adam to get close, optionally L-BFGS to finish.
6. **Plot** — compare against the analytical `e^{-x}`.

**How much data do you need? None.** The differential equation *is* the data.
That single sentence captures why PINNs are different.

## 5 · Why this is revolutionary

| Classical solver | Requirement |
|---|---|
| Euler, RK4 | a mesh / time grid |
| Finite element / volume | a mesh |
| **PINN** | **a network (mesh-free)** |

Key advantages of the PINN approach:
- **Mesh-free** — no grid generation, awkward for complex geometries in classical
  methods.
- **Continuous solution** — evaluate the answer anywhere, at any resolution, and
  differentiate it freely.
- **Works with sparse data** — measurements slot in as an extra loss term.
- **Solves inverse problems** — make an unknown coefficient a trainable variable
  and discover it (see the inverse case study).
- **Fuses measurements and physics** — ideal for digital twins.

## 6 · Where PINNs start becoming interesting

This first notebook is the door, not the room. The series continues with
second-order ODEs (oscillators), the heat equation, nonlinear Burgers' flow,
reaction kinetics `A → B → C`, parameter estimation from data, and finally the
Navier–Stokes equations — the entry point to CFD.

## 7 · Limitations (let's be honest)

Most introductions stop at the hype. This one won't. PINNs have real weaknesses:
- **Training instability** — the loss landscape can be treacherous.
- **Stiff and high-frequency problems** — plain PINNs learn low frequencies first
  (spectral bias) and can fail on sharp or fast features.
- **Slow convergence** — often many thousands of iterations.
- **Scaling** — high dimensions and long time horizons are hard.
- **Multi-physics coupling** — more equations, more competing loss terms.

The good news: each weakness has a research response — domain decomposition
(XPINN), Fourier features, adaptive sampling, causal training, neural operators
(DeepONet, FNO), and residual-adaptive architectures (PirateNets). The Advanced
and Research tracks of this series are devoted to them.

## 8 · Conclusion

The important idea is *not* that neural networks replace physics. It is that
**physics becomes part of the neural network**. For decades engineers described
systems with equations; for the last decade AI described systems with data. PINNs
are the convergence of those two traditions — and that convergence is reshaping
simulation, reaction engineering, CFD, digital twins, and scientific discovery.

---

**Next:** run [notebook 01](../../notebooks/01_Beginner/01_intro_to_pinn.ipynb),
then read the tutorial on
[automatic differentiation and residual losses](../Tutorials/automatic_differentiation_and_residual_losses.md).

**References:** Raissi, Perdikaris & Karniadakis (2019); Karniadakis et al. (2021).
Full details in the [bibliography](../../references/Papers_and_Bibliography/bibliography.md).
