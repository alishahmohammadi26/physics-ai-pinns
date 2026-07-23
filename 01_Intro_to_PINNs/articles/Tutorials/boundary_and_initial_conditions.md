# Tutorial: Boundary and Initial Conditions

**Notebooks:** `02_second_order_odes`, `03_heat_equation`, `06_wave_equation`

Constraints enter a PINN in two ways.

## Soft constraints (penalty method)
Add a loss term for each condition, e.g. `(u(0) - u0)**2`. Simple and general, but
the condition is only *approximately* satisfied, and the term competes with the
PDE residual — so it often needs a weight:
```python
loss = loss_pde + w_ic * loss_ic + w_bc * loss_bc
```
Rules of thumb: weight ICs/BCs more heavily early (e.g. 10–50×); watch the
individual loss terms, not just the total.

## Order matters
A second-order equation needs **two** conditions. For an oscillator you must
impose both `u(0)` and `u'(0)`; for the wave equation both the initial
displacement and the initial velocity. Compute `u'(0)` with autograd, exactly as
you compute the residual.

## Hard constraints (exact by construction)
You can bake conditions into the architecture so they hold *exactly* and drop out
of the loss:
```python
# u(0)=u0, u'(0)=v0 automatically:
u = u0 + v0 * t + t**2 * N(t)
```
Hard constraints remove tuning and often speed up training, at the cost of a
problem-specific ansatz. See notebook `02` for a working comparison.
