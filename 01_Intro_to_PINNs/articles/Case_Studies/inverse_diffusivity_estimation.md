# Case Study: Discovering a Diffusivity from Sparse, Noisy Data

**Notebook:** `03_Advanced/07_inverse_parameter_estimation`

## The scenario
You have 40 noisy temperature readings scattered in space and time from a heated
rod, but you do **not** know the material's thermal diffusivity `α`. Classical
inverse methods would need a bespoke optimisation loop wrapped around a solver.
A PINN needs one extra trainable variable.

## The setup
- Model outputs `u(x, t)`.
- `α` is a learnable parameter (we optimise `log α` to keep it positive).
- Loss = **data loss** (fit the 40 sensors) + **physics loss** (obey the heat
  equation with the *current* `α`).

## Why it works
With only 40 noisy points a plain regressor would overfit. The physics term
restricts the solution to functions that actually satisfy the heat equation, so
the network cannot fit the noise without breaking the PDE. `α` is pushed to the
value that makes data and physics mutually consistent — and converges to the true
0.4 within a couple of percent.

## Transferable lessons
- Make unknowns `nn.Parameter`s and let the optimiser handle them alongside the
  weights.
- Reparameterise to enforce constraints (positivity via `exp`).
- The physics loss is a *regulariser* that makes inversion robust to noise.

Extensions: multiple unknown parameters; a spatially varying `α(x)` represented by
a small sub-network.
