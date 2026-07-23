# Physics-Informed Neural Networks — Literature Review (2017–2026)

This review traces the development of physics-informed neural networks (PINNs)
from the original formulation to the current research frontier. It is organised
first chronologically (to show how the field evolved) and then by theme (to help
you navigate by topic). For each significant paper you will find its key
contribution, its advantages, its limitations, and practical implementation
notes. Download links and full citations are in
[`bibliography.md`](bibliography.md) and [`references.bib`](references.bib).

> Scope note: PINN research has grown explosively — thousands of papers exist by
> 2026. This review is deliberately curated around *seminal*, *highly cited*, and
> *methodologically influential* works plus a representative sample of recent
> (2024–2026) directions. It is a map, not a census.

---

## 1 · Timeline at a glance

| Year | Milestone |
|---|---|
| 2017 | Raissi, Perdikaris & Karniadakis post the two-part "Physics Informed Deep Learning" preprints. |
| 2019 | The consolidated PINN paper appears in *J. Comput. Phys.* — the field's anchor. |
| 2020 | Domain decomposition (XPINN, cPINN); high-speed flows; *Hidden fluid mechanics* (Science). |
| 2021 | Operator learning goes mainstream (DeepONet in *Nat. Mach. Intell.*, FNO at ICLR); DeepXDE library; Nature Reviews Physics overview; first rigorous failure-mode analyses. |
| 2022 | Training theory matures: NTK perspective, causal training; comprehensive SciML review (Cuomo et al.). |
| 2023 | Adaptive sampling systematised; "expert's guide" to training crystallises best practice. |
| 2024 | Kolmogorov–Arnold Networks (KANs) ignite a new architecture wave; PirateNets stabilise very deep PINNs; Laplace neural operator. |
| 2025–26 | PIKANs and fair MLP-vs-KAN comparisons; conservation-preserving neural operators; a second generation of review articles consolidates the field. |

---

## 2 · The foundations (2017–2019)

### Raissi, Perdikaris & Karniadakis (2019) — *Physics-informed neural networks* (JCP)
**Contribution.** The paper that named and unified the approach. It shows a
single neural network can (a) solve a PDE forward by minimising the PDE residual
evaluated via automatic differentiation, and (b) solve inverse problems by making
unknown coefficients trainable. Burgers', Schrödinger, Allen–Cahn and
Navier–Stokes examples are demonstrated.
**Advantages.** Mesh-free; unifies forward and inverse problems in one framework;
needs little or no labelled data; trivially assimilates scattered measurements.
**Limitations.** Training can be slow and unstable; struggles with stiff/
high-frequency/multiscale problems; accuracy typically trails classical solvers
on smooth problems; no convergence guarantees.
**Implementation notes.** Tanh activations, Xavier init, Adam followed by L-BFGS.
This is reproduced in notebooks `01`, `04`, and `10`.

*(The 2017 arXiv preprints, Parts I and II, are the original announcement and are
worth reading for the cleanest statement of the idea.)*

---

## 3 · Scaling up: domain decomposition & applications (2020)

### Jagtap & Karniadakis (2020) — XPINN, and Jagtap, Kharazmi & Karniadakis (2020) — cPINN
**Contribution.** Split the domain into sub-domains, each with its own network,
stitched by interface conditions. cPINN enforces conservation across interfaces;
XPINN generalises to arbitrary space–time decomposition and irregular geometries.
**Advantages.** Parallelisable; each sub-net handles a simpler local problem;
better for large or multiscale domains.
**Limitations.** Interface loss terms add tuning burden; load balancing is
non-trivial; gains are problem-dependent.
**Implementation notes.** Conceptually covered in notebook `12`; a natural
extension exercise after the single-domain notebooks.

### Raissi, Yazdani & Karniadakis (2020) — *Hidden fluid mechanics* (Science)
**Contribution.** Recovers full velocity and **pressure** fields from only
scalar concentration snapshots (e.g. dye/scalar transport), by enforcing
Navier–Stokes. A flagship demonstration of inferring unmeasured physics.
**Advantages.** Extracts quantities that are hard or impossible to measure
directly; robust to noise.
**Limitations.** Expensive; sensitive to network/loss design; validated mostly on
canonical flows.
**Implementation notes.** The pressure-from-velocity idea is reproduced in
notebook `10` on the Taylor–Green vortex.

---

## 4 · Operator learning arrives (2021)

Operator learning shifts the target from *one solution* to the *solution
operator* — the mapping between function spaces — so a trained model generalises
across inputs and infers in a single forward pass.

### Lu et al. (2021) — DeepONet (*Nature Machine Intelligence*)
**Contribution.** A branch network (encodes the input function at fixed sensors)
and a trunk network (encodes the query location) combined by an inner product,
backed by a universal approximation theorem for operators.
**Advantages.** Amortised cost; flexible output querying; strong empirical
generalisation; extends naturally to physics-informed training.
**Limitations.** Fixed sensor grid for inputs; accuracy depends on training-data
coverage; large data needs for the vanilla (data-driven) version.
**Implementation notes.** Built from scratch in notebook `09`.

### Li et al. (2021) — Fourier Neural Operator (ICLR)
**Contribution.** Learns operators by parameterising integral kernels in Fourier
space, giving resolution-invariance and excellent performance on parametric PDEs
(notably turbulence and weather-scale problems).
**Advantages.** Discretisation-invariant; very fast inference; strong on periodic/
smooth-spectrum problems.
**Limitations.** Native FNO assumes regular grids and (quasi-)periodic domains;
complex geometries need extensions (Geo-FNO, U-FNO).
**Implementation notes.** Discussed in notebooks `09` and `12`; the FNO family is
the main alternative to DeepONet.

### Lu, Meng, Mao & Karniadakis (2021) — DeepXDE (*SIAM Review*)
**Contribution.** The reference open-source library for PINNs and DeepONet, with
clean abstractions for geometry, sampling, BC/IC handling, and residual-based
adaptive refinement (RAR).
**Advantages.** Dramatically lowers implementation effort; backend-agnostic.
**Limitations.** Abstraction can hide details useful for research-grade tuning.
**Implementation notes.** Recommended as the "production" counterpart to the
from-scratch PyTorch code in this series; RAR is reimplemented in notebook `08`.

### Karniadakis et al. (2021) — *Physics-informed machine learning* (Nature Reviews Physics)
**Contribution.** The definitive high-level overview: taxonomy of physics-
embedding strategies (observational, inductive, learning biases), applications,
and open challenges.
**Use.** The best single orientation piece for newcomers and for framing the
field in talks or papers.

---

## 5 · Understanding and fixing training (2021–2023)

The central practical lesson of this era: **PINNs are hard to optimise, and the
difficulty is structural, not incidental.**

### Krishnapriyan et al. (2021) — *Characterizing possible failure modes* (NeurIPS)
**Contribution.** Shows that even simple convection/reaction problems induce
pathological loss landscapes; proposes curriculum and sequence-to-sequence
strategies.
**Takeaway.** A "failed" PINN is often an optimisation failure, not a capacity
failure. Reproduce the pain in notebook `04` (low-viscosity Burgers).

### Wang, Yu & Perdikaris (2022) — NTK perspective (JCP)
**Contribution.** Uses the neural tangent kernel to explain why loss terms train
at different rates, motivating principled adaptive weighting.
**Takeaway.** Loss-term imbalance is diagnosable and fixable. See notebook `12`.

### Wang, Sankaran & Perdikaris (2022) — *Respecting causality* (arXiv)
**Contribution.** Time-dependent PINNs often violate temporal causality (fitting
late times before early ones); a causal weighting restores correct ordering.
**Takeaway.** Essential for oscillatory/transport problems (notebooks `02`, `06`).

### Wang, Sankaran, Wang & Perdikaris (2023) — *An expert's guide to training PINNs* (CMAME)
**Contribution.** Consolidates the whole toolkit — Fourier features, modified MLP,
adaptive weights, causal training — into a reproducible recipe with strong
benchmarks.
**Takeaway.** The single most practical reference for getting PINNs to work; it
underpins notebook `12`.

### Tancik et al. (2020) & Wang, Wang & Perdikaris (2021) — Fourier features
**Contribution.** Random Fourier feature embeddings counter spectral bias,
enabling high-frequency and multiscale solutions.
**Takeaway.** Often the highest-return single change; demonstrated in notebook `12`.

---

## 6 · Sampling, stiffness, and other refinements (2022–2023)

### Wu et al. (2023) — Adaptive sampling study (CMAME)
Systematic comparison of non-adaptive and residual-based adaptive sampling (RAR,
RAD, RAR-D). **Takeaway:** where you place collocation points matters as much as
how you train. Reimplemented in notebook `08`.

### Ji et al. (2021) — Stiff-PINN (*J. Phys. Chem. A*)
Applies quasi-steady-state assumptions to make stiff chemical kinetics tractable
for PINNs. **Takeaway:** stiffness needs problem-specific reformulation; relevant
to notebook `05`.

### Cuomo et al. (2022) — SciML review (*J. Sci. Comput.*)
A broad, well-organised survey of methods, variants, and applications through
2022 — a good bridge between the foundations and the modern literature.

---

## 7 · The current frontier (2024–2026)

### Wang, Li, Chen & Perdikaris (2024) — PirateNets
**Contribution.** Physics-informed initialisation plus adaptive residual
connections let *very deep* PINNs train stably, achieving state-of-the-art
accuracy on several benchmarks where plain deep MLPs diverge.
**Advantages.** Unlocks depth; strong accuracy gains.
**Limitations.** More moving parts; still an MLP-family method.
**Implementation notes.** Included in your references PDF; capstone read for
notebook `12`.

### Liu et al. (2024) — KAN: Kolmogorov–Arnold Networks
**Contribution.** Replaces fixed node activations with learnable spline edge
functions grounded in the Kolmogorov–Arnold representation theorem, offering
accuracy and interpretability advantages on structured scientific tasks.
**Advantages.** Interpretable; parameter-efficient on some problems; supports
symbolic-formula extraction.
**Limitations.** Slower per-parameter; benefits are task-dependent; a large,
sometimes noisy follow-up literature.

### Wang et al. (2025) — Kolmogorov–Arnold-Informed Neural Network (CMAME)
**Contribution.** A physics-informed framework built on KANs for forward and
inverse PDE problems (a.k.a. PIKAN), showing competitive or superior accuracy on
several benchmarks.
**Takeaway.** The KAN idea, made physics-informed. Read alongside Shukla et al.
(2024)'s FAIR MLP-vs-KAN comparison for a balanced view.

### Toscano et al. (2025) — *From PINNs to PIKANs*
A synthesis review charting the shift from MLP-based PINNs toward KAN-based ones,
plus recent training advances. The best recent orientation piece.

### Cao, Goswami & Karniadakis (2024) — Laplace Neural Operator
Extends operator learning to non-periodic and transient problems via a Laplace-
space parameterisation, addressing an FNO weakness.

### Second-generation reviews (2024–2026)
Farea et al. (2024), Zhao et al. (2024, complex fluids), Meng et al. (2025),
and Fan & Chen (2026) collectively map applications (fluids, biomedicine,
materials, geoscience, energy) and open problems (theory/convergence,
scalability, uncertainty quantification, standard benchmarks). Use these to find
the review closest to your application domain.

---

## 8 · Open problems as of 2026

1. **Theory.** Convergence and error bounds remain partial; when and why PINNs
   succeed is still being formalised.
2. **Optimisation.** Despite the modern toolkit, robust out-of-the-box training
   for stiff/multiscale/turbulent problems is not solved.
3. **Scalability.** High-dimensional and long-time-horizon problems strain both
   PINNs and operators.
4. **Uncertainty quantification.** Bayesian PINNs and ensemble methods exist but
   are not yet routine.
5. **Benchmarks.** The community still lacks universally accepted, standardised
   benchmark suites (an active 2025–26 effort).
6. **Architecture.** MLP vs KAN vs operator hybrids is an open, actively contested
   design space.

---

## 9 · How the notebooks map to this review

| Notebook | Papers it embodies |
|---|---|
| `01` intro | Raissi 2019; Karniadakis 2021 |
| `02` 2nd-order ODE | Raissi 2019; Wang 2022 (causality) |
| `03` heat | Raissi 2019; Lu 2021 (DeepXDE) |
| `04` Burgers | Raissi 2019; Krishnapriyan 2021 |
| `05` kinetics | Ji 2021 (Stiff-PINN) |
| `06` wave | Wang 2022 (NTK) |
| `07` inverse | Raissi 2019; Raissi 2020 (hidden physics) |
| `08` adaptive sampling | Lu 2021; Wu 2023 |
| `09` DeepONet | Lu 2021 |
| `10` Navier–Stokes | Raissi 2019; Raissi 2020; Cai 2021 |
| `11` PI-DeepONet | Wang 2021 (Sci. Adv.) |
| `12` architectures | Tancik 2020; Wang 2021/2023; Wang 2024 (PirateNets); Liu 2024 (KAN) |
