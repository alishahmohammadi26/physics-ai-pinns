# Tutorial: Automatic Differentiation and Residual Losses

**Notebooks:** `01_intro_to_pinn`, `03_heat_equation`

## The one idea
A neural network is a differentiable function `N(x)`. Autodiff lets us compute
derivatives of the *output* with respect to the *input* — exactly, not by finite
differences. Every term in a differential equation is therefore directly
available.

## First derivative (PyTorch)
```python
x = torch.linspace(0, 1, 100).view(-1, 1).requires_grad_(True)
y = model(x)
dy_dx = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
```
`create_graph=True` is essential: it keeps the derivative itself differentiable,
so the physics loss can be back-propagated into the weights, and so you can take
higher derivatives.

## Second derivative
```python
d2y_dx2 = torch.autograd.grad(dy_dx, x, torch.ones_like(dy_dx), create_graph=True)[0]
```
Just differentiate again. Partial derivatives for PDEs work the same way — pass
the specific input column (`x` or `t`) you want to differentiate with respect to.

## From derivatives to a loss
For `dy/dx = -y`, the residual is `r(x) = dy/dx + y`, which should be zero
everywhere. The physics loss is its mean square over collocation points:
```python
loss_physics = torch.mean((dy_dx + y)**2)
```
Add the boundary loss and minimise. That is the entire PINN idea in three lines.

## Common pitfalls
- **Forgetting `requires_grad_(True)`** on the input → autograd can't differentiate.
- **Reusing a target tensor built from a grad-tracking input** across epochs →
  "backward through the graph a second time". `.detach()` your targets.
- **Using ReLU** → its second derivative is zero almost everywhere; prefer `tanh`.
