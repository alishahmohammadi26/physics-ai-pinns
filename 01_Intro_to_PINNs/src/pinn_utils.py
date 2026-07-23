"""
Shared helpers for the PINN Learning Series.

The notebooks are intentionally self-contained (so each reads top-to-bottom
without hidden dependencies). These utilities are provided for readers who want
to refactor the notebook code into reusable building blocks for their own
projects.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """Fully-connected tanh network with Xavier init — the standard PINN backbone."""

    def __init__(self, layers, activation=nn.Tanh):
        super().__init__()
        seq = []
        for i in range(len(layers) - 1):
            seq.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                seq.append(activation())
        self.net = nn.Sequential(*seq)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1) if len(inputs) > 1 else inputs[0]
        return self.net(x)


def grad(outputs, inputs):
    """First derivative d(outputs)/d(inputs) with graph retained (for higher orders)."""
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
    )[0]


def laplacian_1d(u, x):
    """u_xx for a scalar field u(x)."""
    return grad(grad(u, x), x)


def relative_l2(pred, true):
    """Relative L2 error, works with numpy arrays or torch tensors."""
    import numpy as np
    pred = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)
    true = true.detach().cpu().numpy() if hasattr(true, "detach") else np.asarray(true)
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def train_adam(loss_closure, params, epochs=5000, lr=1e-3, log_every=1000):
    """Minimal Adam loop. `loss_closure()` must return a scalar tensor."""
    opt = torch.optim.Adam(params, lr=lr)
    history = []
    for ep in range(epochs):
        opt.zero_grad()
        loss = loss_closure()
        loss.backward()
        opt.step()
        history.append(loss.item())
        if log_every and ep % log_every == 0:
            print(f"epoch {ep:6d} | loss {loss.item():.3e}")
    return history


def finish_lbfgs(loss_closure, params, max_iter=2000):
    """Second-stage L-BFGS polish — the standard PINN closer."""
    opt = torch.optim.LBFGS(params, max_iter=max_iter,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = loss_closure()
        loss.backward()
        return loss

    opt.step(closure)
    return loss_closure().item()
