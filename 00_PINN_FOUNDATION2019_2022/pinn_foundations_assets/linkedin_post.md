# LinkedIn Post — PINN Review 2019–2026

## Post Text

---

Seven years of Physics-Informed Neural Networks — distilled into one review.

I published a source-centred deep-dive covering all 34 landmark PINN studies from 2019 to 2026. Every section includes the original formulation, what the authors actually reported, where it breaks, and an executable reproduction with an animated diagnostic.

**What the review covers:**

Part I (2019–2021): the foundations
- Raissi, Perdikaris & Karniadakis — the composite-loss framework that started everything
- Adversarial, Bayesian and dropout-based uncertainty quantification (three very different approaches)
- Domain decomposition: PPINN, cPINN, XPINN — when and why partitioning helps
- DeepXDE adaptive refinement and variational residuals
- DeepONet: learning operators, not solutions
- Stiff-PINN: QSSA reduction for kinetics that defeat gradient descent
- NTK theory: the mathematical reason your PINN stops training

Part II (2022–2026): when and why PINNs fail
- Krishnapriyan et al. — characterizing failure modes systematically
- Gradient pathologies: loss blocks converge at incompatible rates
- Spectral bias and Fourier feature networks
- Sampling failures: uniform collocation misses high-residual regions
- Causality violations: future-leakage in time-dependent problems
- PINNacle: the first standardized PDE benchmark suite
- PirateNets: residual-adaptive architectures
- Spurious solutions: low-residual points that are physically wrong

The honest finding: PINNs are a constrained regression framework, not a universal PDE solver. Their real advantages are inverse problems, sparse-data fusion, and reusable operator surrogates. Everything else depends on whether you can make the composite objective trainable.

Read the full article (with all 34 animated diagnostics and code): [link in comments]

#PhysicsAI #ScientificML #PINNs #DeepLearning #ComputationalScience #AIforScience

---

## Comment (add link)

Full article with all 34 animated diagnostics, executable notebooks, and the complete paper-by-paper analysis:
https://alishahmohammadi22.github.io/blog/pinn-review-2019-2026.html

GitHub notebooks (6 organized by theme):
https://github.com/alishahmohammadi22/alishahmohammadi22.github.io

---

## Infographic for LinkedIn Post

See: `linkedin_infographic.svg` (in this same folder)

## GIF for LinkedIn Post

Use one of the animated diagnostics embedded in the blog article.
Recommended: the Burgers equation Picard-sweep animation (Paper 1 diagnostic) —
it's the most visually compelling and directly shows the PINN mechanism.

To extract it, run: `python3 extract_linkedin_gif.py`
