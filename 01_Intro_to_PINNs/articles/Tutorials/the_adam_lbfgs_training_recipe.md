# Tutorial: The Adam → L-BFGS Training Recipe

**Notebooks:** `04_burgers_equation`, `06_wave_equation`

The most reliable habit in the PINN literature is a **two-stage** optimiser.

1. **Adam** (first-order, robust, stochastic-friendly) gets you into a good basin
   from a cold start.
2. **L-BFGS** (quasi-Newton, uses curvature) then drives the loss down several
   more orders of magnitude and sharpens fine features.

```python
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for _ in range(15000):
    opt.zero_grad(); loss = loss_fn(); loss.backward(); opt.step()

opt2 = torch.optim.LBFGS(model.parameters(), max_iter=3000,
                         line_search_fn="strong_wolfe")
def closure():
    opt2.zero_grad(); l = loss_fn(); l.backward(); return l
opt2.step(closure)
```

Why not L-BFGS alone? From a cold start it often diverges. Why not Adam alone? It
tends to leave the solution slightly blurry. Together they are the workhorse
combination. For the hardest problems, layer on Fourier features, adaptive loss
weights, and causal training (notebook `12`).
