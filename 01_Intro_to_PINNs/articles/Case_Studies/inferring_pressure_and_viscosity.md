# Case Study: Inferring Pressure and Viscosity from Velocity Alone

**Notebook:** `04_Research_Papers/10_navier_stokes_cylinder`

## The scenario
In many flows you can measure velocity (e.g. from PIV or flow visualisation) but
**not** pressure, and you may not know the viscosity precisely. Raissi, Yazdani &
Karniadakis (2020) showed a PINN can reconstruct the *unmeasured* pressure field
and recover viscosity purely by enforcing the Navier–Stokes equations.

## The setup (reproduced on the exact Taylor–Green vortex)
- The network outputs a **stream function** `ψ` and pressure `p`; velocities come
  from `u = ψ_y`, `v = -ψ_x`, which makes the flow divergence-free by
  construction.
- Training data: sparse, noisy `(u, v)` samples only.
- The momentum equations form the physics loss; viscosity `ν` is a learnable
  parameter.

## The result
Pressure — never present in the data — emerges because it is the only field
consistent with the momentum balance and the observed velocities. Viscosity
converges to its true value. This is the essence of "hidden physics" discovery.

## Transferable lessons
- Use potential/stream-function formulations to satisfy constraints exactly.
- Unmeasured fields can be *inferred* when they are coupled to measured ones
  through the equations.
- The same code recovers the original cylinder-wake result by swapping in the
  authors' dataset.
