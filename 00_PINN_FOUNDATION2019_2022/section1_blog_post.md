# (B) §1 Introduction

## 1. Introduction

Scientific and engineering decisions often begin with a modeling choice rather than an algorithmic choice. A practitioner may have governing equations but incomplete boundary conditions, sparse measurements but trusted conservation laws, high-fidelity simulators but prohibitive repeated-query cost, or observations that are rich enough for prediction but too narrow for extrapolation. Classical mechanistic models includings, finite element, finite volume, finite difference, spectral, and related solvers, remain the reference standard when the governing equations, domains, coefficients, initial/boundary conditions, and numerical tolerances are sufficiently well specified. Yet the same sources of strength can become bottlenecks when data assimilation, uncertain parameters, high-dimensional design spaces, inverse inference, or repeated surrogate evaluations dominate the workflow[11](https://www.mdpi.com/2076-3417/15/14/8092)[10](https://www.mdpi.com/2673-2688/5/3/74).

Purely data-driven models occupy the opposite end of the spectrum. They can provide fast inference after training and can exploit large observational or simulation datasets, but their predictions need not satisfy conservation laws, constitutive relations, boundary conditions, or invariances unless those constraints are encoded by design. This limitation is especially consequential in regimes with sparse data, shifted operating conditions, or extrapolation outside the training distribution. Physics-informed learning was developed to occupy the middle ground: it uses domain knowledge not merely as post-hoc interpretation, but as part of the learning problem itself through residual losses, constrained architectures, structured data, or hybrid model components[10](https://www.mdpi.com/2673-2688/5/3/74)[9](https://www.mdpi.com/2076-3417/13/12/6892).

Physics-Informed Neural Networks (PINNs) are the most visible member of this broader physics-informed machine-learning family. In the original formulation, a neural network approximates a latent solution field while a companion residual network is obtained by differentiating the network output with respect to space, time, or parameters; training then balances data misfit with differential-equation, boundary-condition, and initial-condition residuals[1](https://arxiv.org/abs/1711.10561). Raissi, Perdikaris, and Karniadakis introduced this framework in a two-part 2017 arXiv treatise: Part I treated data-driven solutions of nonlinear PDEs, while Part II treated data-driven discovery of nonlinear PDEs[1](https://arxiv.org/abs/1711.10561)[2](https://link.springer.com/article/10.1007/s44379-025-00015-1). The 2019 Journal of Computational Physics paper established the widely cited peer-reviewed formulation for forward and inverse problems involving nonlinear PDEs[3](https://www.mindat.org/reference.php?id=15892303).

For practitioners, the appeal of PINNs is clearest in problem classes where **some physics is trusted but not everything is known**. Forward problems use the network as a mesh-free surrogate or solver constrained by equations and boundary/initial data; inverse problems additionally estimate hidden parameters, coefficients, constitutive terms, or latent state variables from observations. Reviews and primary studies document uses in sparse-observation inference, data assimilation, surrogate modeling, multi-fidelity learning, and multiphysics systems, but the evidence is strongest when claims are stated conditionally: PINNs can reduce data requirements and improve physical plausibility when the constraints are informative, correctly specified, and trainable; they do not automatically deliver robust identification or solver-grade reliability[8](https://arxiv.org/abs/1907.04502)[10](https://www.mdpi.com/2673-2688/5/3/74).

The field has also broadened beyond the original residual-minimization template. Methodological themes include adaptive loss weighting, residual/adaptive sampling, hard versus soft constraints, variational and conservative formulations, domain decomposition, Bayesian and ensemble uncertainty quantification, multi-fidelity learning, and connections to neural operators[2](https://link.springer.com/article/10.1007/s44379-025-00015-1)[9](https://www.mdpi.com/2076-3417/13/12/6892). Operator-learning methods such as DeepONet and FNO address a different repeated-query regime by learning mappings between input functions and solution functions, rather than solving a single instance; physics-informed neural operators and physics-informed DeepONets then add residual or physics constraints to operator learning[6](https://www.nature.com/articles/s42256-021-00302-5)[7](https://iclr.cc/virtual/2021/poster/3281)[9](https://www.mdpi.com/2076-3417/13/12/6892). Recent reviews also track Physics-Informed Kolmogorov–Arnold Network variants, but this literature is still emerging and should be treated as exploratory rather than settled practice[2](https://link.springer.com/article/10.1007/s44379-025-00015-1).

At the same time, a practitioner review must foreground limitations. PINNs are multi-objective optimization problems, and the different loss terms can produce gradient imbalance, stiffness, spectral/eigenvector bias, collocation sensitivity, and brittle hyperparameter dependence. Krishnapriyan et al. showed that standard soft-regularized PINNs can learn simple cases yet fail to capture relevant physical phenomena for more complex convection, reaction, and diffusion settings because the setup can make the loss landscape difficult to optimize[12](https://proceedings.neurips.cc/paper/2021/hash/df438e5206f31600e6ae4af72f2725f1-Abstract.html). Later reviews catalog related issues: high training cost, poor scalability, sensitivity to initialization and loss weighting, boundary-condition enforcement challenges, and limited robustness in complex physics or noisy data regimes[4](https://www.mdpi.com/2504-2289/6/4/140)[10](https://www.mdpi.com/2673-2688/5/3/74).

The most consequential limitation for inverse problems is identifiability. A small residual and a visually plausible state estimate do not, by themselves, prove that inferred parameters are unique, well-conditioned, or physically meaningful. Classical identifiability analysis distinguishes structural identifiability from practical identifiability, and Wieland et al. emphasize that Fisher Information Matrix–based approaches can have severe shortcomings for practical non-identifiability, recommending profile-likelihood reasoning as a complementary diagnostic[13](https://arxiv.org/abs/2102.05100). PINN-based inverse modeling therefore needs residual diagnostics, independent validation data, uncertainty quantification, sensitivity analysis, and identifiability checks rather than residual loss alone.

---

# (C) Table 1.1 — Prior-review positioning audit

**Operational definitions:** **● deep/central coverage**; **◐ partial, domain-specific, or mentioned but not operationalized**; **○ absent/minimal**. Ratings are evidence-based from opened review pages and abstracts, not designed to favor this manuscript.

| Review | Taxonomy | Theory | Applications | Paradigm comparison | Inverse problems | Identifiability/FIM | Code/resources | Industry/practitioner orientation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Karniadakis et al. 2021 | ● | ◐ | ● | ◐ | ◐ | ○ | ○ | ◐ |
| Cai et al. 2021 fluid mechanics | ◐ | ○ | ◐ | ◐ | ● | ○ | ○ | ○ |
| Cuomo et al. 2022 | ● | ◐ | ● | ◐ | ◐ | ○ | ○ | ◐ |
| Lawal et al. 2022 | ◐ | ○ | ◐ | ○ | ◐ | ○ | ○ | ○ |
| Pateras et al. 2023 | ● | ◐ | ● | ◐ | ◐ | ○ | ○ | ○ |
| Farea et al. 2024 | ◐ | ◐ | ● | ◐ | ◐ | ○ | ○ | ◐ |
| Toscano et al. 2025 | ● | ● | ● | ◐ | ◐ | ◐ | ● | ◐ |
| Ren et al. 2025 | ● | ● | ● | ◐ | ◐ | ○ | ◐ | ◐ |
| Present review, planned | ◐ | ◐ | ◐ | ● | ● | ● | ◐* | ● |

\*Resource integration is **author-specified/planned** unless repositories, notebooks, or checklists are released and cited.

---

![alt text](<Figure 1.png>)
*Figure 1: Practitioner decision framework contrasting mechanistic solvers, purely data-driven models, and PINN/PIML hybrids by the amount of trusted physics, data availability, uncertainty, and intended use case.*


### 1.3 Why PINNs exist: the practical gap

The PINN formulation was motivated by classes of problem in which classical numerical methods, or classical machine-learning methods, individually leave the practitioner without a good option. <Person>Raissi</Person>–<Person>Perdikaris</Person>–<Person>Karniadakis</Person>'s two-part 2017 preprints separated these into two archetypes that structure the rest of the field[11](https://scholar.google.com/citations?user=dCdmUaYAAAAJ&hl=en).

- **Forward problems.** The governing equations and their parameters are assumed known; the *state* is unknown and is sought at collocation points, possibly on complex geometries or in high dimensions[11](https://scholar.google.com/citations?user=dCdmUaYAAAAJ&hl=en)[2](https://arxiv.org/abs/1907.04502). Here, PINNs enter as an alternative *approximator* to FEM/FVM/spectral methods and are compared with them on accuracy and cost[1](https://academic.oup.com/imamat/article/89/1/143/7680268).
- **Inverse problems.** Some combination of parameters, coefficients, forcing terms, boundary conditions, latent fields or initial conditions must be *inferred* from indirect and typically sparse and noisy observations, subject to the governing physics[2](https://arxiv.org/abs/1907.04502). The value proposition of PINNs is qualitatively different in this regime: automatic differentiation over the same network that represents the state gives near-trivial access to gradients with respect to the unknown parameters, and the physics residual acts as a regulariser[2](https://arxiv.org/abs/1907.04502)[9](https://link.springer.com/article/10.1007/s10409-021-01148-1).

These two regimes are distinct because the *what-is-known* and *what-is-unknown* structure of the problem is different. The distinction propagates through every subsequent methodological choice — loss weighting, sampling, uncertainty quantification, identifiability diagnostics — and it is a central organising axis of this review.

<!-- Copilot-Researcher-Visualization -->
```html
<style>
        :root {
        --accent: #464feb;
        --timeline-ln: linear-gradient(to bottom, transparent 0%, #b0beff 15%, #b0beff 85%, transparent 100%);
        --timeline-border: #ffffff;
        --bg-card: #f5f7fa;
        --bg-hover: #ebefff;
        --text-title: #424242;
        --text-accent: var(--accent);
        --text-sub: #424242;
        --radius: 12px;
        --border: #e0e0e0;
        --shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
        --hover-shadow: 0 4px 14px rgba(39, 16, 16, 0.1);
        --font: "Segoe Sans", "Segoe UI", "Segoe UI Web (West European)", -apple-system, "system-ui", Roboto, "Helvetica Neue", sans-serif;
        --overflow-wrap: break-word;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --accent: #7385ff;
            --timeline-ln: linear-gradient(to bottom, transparent 0%, transparent 3%, #6264a7 30%, #6264a7 50%, transparent 97%, transparent 100%);
            --timeline-border: #424242;
            --bg-card: #1a1a1a;
            --bg-hover: #2a2a2a;
            --text-title: #ffffff;
            --text-sub: #ffffff;
            --shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
            --hover-shadow: 0 4px 14px rgba(0, 0, 0, 0.5);
            --border: #3d3d3d;
        }
    }

    @media (prefers-contrast: more),
    (forced-colors: active) {
        :root {
            --accent: ActiveText;
            --timeline-ln: ActiveText;
            --timeline-border: Canvas;
            --bg-card: Canvas;
            --bg-hover: Canvas;
            --text-title: CanvasText;
            --text-sub: CanvasText;
            --shadow: 0 2px 10px Canvas;
            --hover-shadow: 0 4px 14px Canvas;
            --border: ButtonBorder;
        }
    }

    .insights-container {
        display: grid;
        grid-template-columns: repeat(2,minmax(240px,1fr));
        padding: 0px 16px 0px 16px;
        gap: 16px;
        margin: 0 0;
        font-family: var(--font);
    }

    .insight-card:last-child:nth-child(odd){
        grid-column: 1 / -1;
    }

    .insight-card {
        background-color: var(--bg-card);
        border-radius: var(--radius);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        min-width: 220px;
        padding: 16px 20px 16px 20px;
    }

    .insight-card:hover {
        background-color: var(--bg-hover);
    }

    .insight-card h4 {
        margin: 0px 0px 8px 0px;
        font-size: 1.1rem;
        color: var(--text-accent);
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .insight-card .icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        font-size: 1.1rem;
        color: var(--text-accent);
    }

    .insight-card p {
        font-size: 0.92rem;
        color: var(--text-sub);
        line-height: 1.5;
        margin: 0px;
        overflow-wrap: var(--overflow-wrap);
    }

    .insight-card p b, .insight-card p strong {
        font-weight: 600;
    }

    .metrics-container {
        display:grid;
        grid-template-columns:repeat(2,minmax(210px,1fr));
        font-family: var(--font);
        padding: 0px 16px 0px 16px;
        gap: 16px;
    }

    .metric-card:last-child:nth-child(odd){
        grid-column:1 / -1; 
    }

    .metric-card {
        flex: 1 1 210px;
        padding: 16px;
        background-color: var(--bg-card);
        border-radius: var(--radius);
        border: 1px solid var(--border);
        text-align: center;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .metric-card:hover {
        background-color: var(--bg-hover);
    }

    .metric-card h4 {
        margin: 0px;
        font-size: 1rem;
        color: var(--text-title);
        font-weight: 600;
    }

    .metric-card .metric-card-value {
        margin: 0px;
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--text-accent);
    }

    .metric-card p {
        font-size: 0.85rem;
        color: var(--text-sub);
        line-height: 1.45;
        margin: 0;
        overflow-wrap: var(--overflow-wrap);
    }

    .timeline-container {
        position: relative;
        margin: 0 0 0 0;
        padding: 0px 16px 0px 56px;
        list-style: none;
        font-family: var(--font);
        font-size: 0.9rem;
        color: var(--text-sub);
        line-height: 1.4;
    }

    .timeline-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: calc(-40px + 56px);
        width: 2px;
        height: 100%;
        background: var(--timeline-ln);
    }

    .timeline-container > li {
        position: relative;
        margin-bottom: 16px;
        padding: 16px 20px 16px 20px;
        border-radius: var(--radius);
        background: var(--bg-card);
        border: 1px solid var(--border);
    }

    .timeline-container > li:last-child {
        margin-bottom: 0px;
    }

    .timeline-container > li:hover {
        background-color: var(--bg-hover);
    }

    .timeline-container > li::before {
        content: "";
        position: absolute;
        top: 18px;
        left: -40px;
        width: 14px;
        height: 14px;
        background: var(--accent);
        border: var(--timeline-border) 2px solid;
        border-radius: 50%;
        transform: translateX(-50%);
        box-shadow: 0px 0px 2px 0px #00000012, 0px 4px 8px 0px #00000014;
    }

    .timeline-container > li h4 {
        margin: 0 0 5px;
        font-size: 1rem;
        font-weight: 600;
        color: var(--accent);
    }

    .timeline-container > li h4 em {
        margin: 0 0 5px;
        font-size: 1rem;
        font-weight: 600;
        color: var(--accent);
        font-style: normal;
    }

    .timeline-container > li * {
        margin: 0;
        font-size: 0.9rem;
        color: var(--text-sub);
        line-height: 1.4;
    }

    .timeline-container > li * b, .timeline-container > li * strong {
        font-weight: 600;
    }
        @media (max-width:600px){
        .metrics-container,
        .insights-container{
            grid-template-columns:1fr;
      }
    }
</style>
<div class="insights-container">
 <div class="insight-card">
  <h4>Forward problem</h4>
  <p>Physics and parameters known; state unknown. PINNs compete directly with FEM/FVM/spectral solvers on accuracy, cost and mesh-generation burden.</p>
 </div>
 <div class="insight-card">
  <h4>Inverse problem</h4>
  <p>State and/or parameters unknown, with indirect noisy observations. PINNs act as a differentiable data-assimilation framework, but identifiability and conditioning become first-order concerns.</p>
 </div>
 <div class="insight-card">
  <h4>Practitioner's real question</h4>
  <p>Which paradigm (mechanistic, data-driven, PINN) and which PINN variant fit the data, physics, budget and required output?</p>
 </div>
</div>
```
[1](https://academic.oup.com/imamat/article/89/1/143/7680268) [2](https://arxiv.org/abs/1907.04502) 

Beyond these two archetypes, PINNs have been *studied* on state reconstruction and data assimilation with sparse observations[2](https://arxiv.org/abs/1907.04502)[10](https://www.nature.com/articles/s42254-021-00314-5); on surrogate modelling and repeated-query settings, where the training cost may be amortised over many evaluations[9](https://link.springer.com/article/10.1007/s10409-021-01148-1); on high-dimensional problems where classical spatial discretisation is limited by the curse of dimensionality[1](https://academic.oup.com/imamat/article/89/1/143/7680268); on partially known physics, where a UDE-style hybrid learns the missing terms; and on multiphysics problems where physical constraints across sub-problems are difficult to couple with purely data-driven approximators[9](https://link.springer.com/article/10.1007/s10409-021-01148-1). The evidence base is heterogeneous: it does *not* show that PINNs succeed equally in all of these categories, and the review develops this nuance in later sections.

### 1.4 Scale and evolution of the field

The modern PINN literature originated with the two-part 2017 arXiv preprints[11](https://scholar.google.com/citations?user=dCdmUaYAAAAJ&hl=en) and was consolidated in the 2019 <em>Journal of Computational Physics</em> article[10](https://www.nature.com/articles/s42254-021-00314-5). As a single indicator of scale, Google Scholar reported **28,581 citations** for the 2019 article on the day of writing (accessed 9 August 2026)[11](https://scholar.google.com/citations?user=dCdmUaYAAAAJ&hl=en); the 2021 <em>Nature Reviews Physics</em> review by <Person>Karniadakis</Person> et al. reported **8,009 citations** and 149k accesses on the publisher page[10](https://www.nature.com/articles/s42254-021-00314-5). These numbers are given for orientation only; citation databases differ and counts change quickly.

<!-- Copilot-Researcher-Visualization -->
```html
<style>
        :root {
        --accent: #464feb;
        --timeline-ln: linear-gradient(to bottom, transparent 0%, #b0beff 15%, #b0beff 85%, transparent 100%);
        --timeline-border: #ffffff;
        --bg-card: #f5f7fa;
        --bg-hover: #ebefff;
        --text-title: #424242;
        --text-accent: var(--accent);
        --text-sub: #424242;
        --radius: 12px;
        --border: #e0e0e0;
        --shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
        --hover-shadow: 0 4px 14px rgba(39, 16, 16, 0.1);
        --font: "Segoe Sans", "Segoe UI", "Segoe UI Web (West European)", -apple-system, "system-ui", Roboto, "Helvetica Neue", sans-serif;
        --overflow-wrap: break-word;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --accent: #7385ff;
            --timeline-ln: linear-gradient(to bottom, transparent 0%, transparent 3%, #6264a7 30%, #6264a7 50%, transparent 97%, transparent 100%);
            --timeline-border: #424242;
            --bg-card: #1a1a1a;
            --bg-hover: #2a2a2a;
            --text-title: #ffffff;
            --text-sub: #ffffff;
            --shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
            --hover-shadow: 0 4px 14px rgba(0, 0, 0, 0.5);
            --border: #3d3d3d;
        }
    }

    @media (prefers-contrast: more),
    (forced-colors: active) {
        :root {
            --accent: ActiveText;
            --timeline-ln: ActiveText;
            --timeline-border: Canvas;
            --bg-card: Canvas;
            --bg-hover: Canvas;
            --text-title: CanvasText;
            --text-sub: CanvasText;
            --shadow: 0 2px 10px Canvas;
            --hover-shadow: 0 4px 14px Canvas;
            --border: ButtonBorder;
        }
    }

    .insights-container {
        display: grid;
        grid-template-columns: repeat(2,minmax(240px,1fr));
        padding: 0px 16px 0px 16px;
        gap: 16px;
        margin: 0 0;
        font-family: var(--font);
    }

    .insight-card:last-child:nth-child(odd){
        grid-column: 1 / -1;
    }

    .insight-card {
        background-color: var(--bg-card);
        border-radius: var(--radius);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        min-width: 220px;
        padding: 16px 20px 16px 20px;
    }

    .insight-card:hover {
        background-color: var(--bg-hover);
    }

    .insight-card h4 {
        margin: 0px 0px 8px 0px;
        font-size: 1.1rem;
        color: var(--text-accent);
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .insight-card .icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        font-size: 1.1rem;
        color: var(--text-accent);
    }

    .insight-card p {
        font-size: 0.92rem;
        color: var(--text-sub);
        line-height: 1.5;
        margin: 0px;
        overflow-wrap: var(--overflow-wrap);
    }

    .insight-card p b, .insight-card p strong {
        font-weight: 600;
    }

    .metrics-container {
        display:grid;
        grid-template-columns:repeat(2,minmax(210px,1fr));
        font-family: var(--font);
        padding: 0px 16px 0px 16px;
        gap: 16px;
    }

    .metric-card:last-child:nth-child(odd){
        grid-column:1 / -1; 
    }

    .metric-card {
        flex: 1 1 210px;
        padding: 16px;
        background-color: var(--bg-card);
        border-radius: var(--radius);
        border: 1px solid var(--border);
        text-align: center;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .metric-card:hover {
        background-color: var(--bg-hover);
    }

    .metric-card h4 {
        margin: 0px;
        font-size: 1rem;
        color: var(--text-title);
        font-weight: 600;
    }

    .metric-card .metric-card-value {
        margin: 0px;
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--text-accent);
    }

    .metric-card p {
        font-size: 0.85rem;
        color: var(--text-sub);
        line-height: 1.45;
        margin: 0;
        overflow-wrap: var(--overflow-wrap);
    }

    .timeline-container {
        position: relative;
        margin: 0 0 0 0;
        padding: 0px 16px 0px 56px;
        list-style: none;
        font-family: var(--font);
        font-size: 0.9rem;
        color: var(--text-sub);
        line-height: 1.4;
    }

    .timeline-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: calc(-40px + 56px);
        width: 2px;
        height: 100%;
        background: var(--timeline-ln);
    }

    .timeline-container > li {
        position: relative;
        margin-bottom: 16px;
        padding: 16px 20px 16px 20px;
        border-radius: var(--radius);
        background: var(--bg-card);
        border: 1px solid var(--border);
    }

    .timeline-container > li:last-child {
        margin-bottom: 0px;
    }

    .timeline-container > li:hover {
        background-color: var(--bg-hover);
    }

    .timeline-container > li::before {
        content: "";
        position: absolute;
        top: 18px;
        left: -40px;
        width: 14px;
        height: 14px;
        background: var(--accent);
        border: var(--timeline-border) 2px solid;
        border-radius: 50%;
        transform: translateX(-50%);
        box-shadow: 0px 0px 2px 0px #00000012, 0px 4px 8px 0px #00000014;
    }

    .timeline-container > li h4 {
        margin: 0 0 5px;
        font-size: 1rem;
        font-weight: 600;
        color: var(--accent);
    }

    .timeline-container > li h4 em {
        margin: 0 0 5px;
        font-size: 1rem;
        font-weight: 600;
        color: var(--accent);
        font-style: normal;
    }

    .timeline-container > li * {
        margin: 0;
        font-size: 0.9rem;
        color: var(--text-sub);
        line-height: 1.4;
    }

    .timeline-container > li * b, .timeline-container > li * strong {
        font-weight: 600;
    }
        @media (max-width:600px){
        .metrics-container,
        .insights-container{
            grid-template-columns:1fr;
      }
    }
</style>
<ul class="timeline-container">
 <li>
  <h4>Nov 2017: Foundational preprints</h4>
  <p>Raissi–Perdikaris–Karniadakis, Physics-Informed Deep Learning Parts I and II (arXiv:1711.10561; arXiv:1711.10566).</p>
 </li>
 <li>
  <h4>2018–2019: Consolidation</h4>
  <p>Journal of Computational Physics article establishing the vanilla PINN loss for forward and inverse PDE problems; Sirignano & Spiliopoulos, DGM, complementary deep-learning solver.</p>
 </li>
 <li>
  <h4>2019–2021: Reproducible software</h4>
  <p>DeepXDE (SIAM Review, 2021; Lu et al.); NeuroDiffEq; industrial-scale framework NVIDIA Modulus / PhysicsNeMo, later open-sourced.</p>
 </li>
 <li>
  <h4>2020–2022: Variants and theory</h4>
  <p>Variational PINN (VPINN, hp-VPINN), conservative & extended PINNs (cPINN, XPINN), Bayesian PINN (B-PINN), self-adaptive weights, adaptive sampling.</p>
 </li>
 <li>
  <h4>2020–2022: Diagnosis of failure modes</h4>
  <p>Gradient pathologies (Wang, Teng, Perdikaris); NTK perspective on when PINNs fail to train; possible failure modes on stiff/multiscale PDEs (Krishnapriyan et al., NeurIPS 2021).</p>
 </li>
 <li>
  <h4>2023–2025: Training pipelines & new architectures</h4>
  <p>Causal training; "Expert's guide" to training PINNs; PirateNets; PIKAN/KAN-based physics-informed models (Liu et al., 2024/2025); PINN–operator hybrids.</p>
 </li>
 <li>
  <h4>2024–2025: Theoretical maturity</h4>
  <p>De Ryck & Mishra, Acta Numerica 33: comprehensive numerical analysis of PINNs and related models.</p>
 </li>
</ul>
```
[11](https://scholar.google.com/citations?user=dCdmUaYAAAAJ&hl=en)  [10](https://www.nature.com/articles/s42254-021-00314-5)  [2](https://arxiv.org/abs/1907.04502)              

Rather than reproducing a chronological catalogue, we organise the evolution in later sections around the recurring pattern **failure mode → diagnosis → methodological fix**. Adaptive loss-weighting responded to gradient pathologies; residual-based adaptive sampling responded to poor collocation coverage of sharp features; domain-decomposition (cPINN, XPINN, hp-VPINN) responded to scaling to complex geometries and multiscale behaviour; causal training responded to non-causal propagation of information in time-dependent PDEs; variational and conservative formulations responded to weak-regularity solutions and conservation-law violations; Bayesian PINNs responded to noise and uncertainty; multi-fidelity and neural-operator hybrids responded to repeated-query and generalisation requirements[10](https://www.nature.com/articles/s42254-021-00314-5); and PIKANs (2024–2025) explored alternative function approximators for potential representational and interpretability gains[4](https://arxiv.org/abs/2410.13228). We refrain from claiming a specific numerical count of "named variants": the reviews we audit (Cuomo, Toscano, Farea, Ren) each enumerate dozens, with substantial overlap in what they name[7](https://www.iris.unina.it/handle/11588/893872)[3](https://www.mdpi.com/2076-3417/15/14/8092).

---

# (H) References

1. Raissi, M.; Perdikaris, P.; Karniadakis, G.E. **Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations.** arXiv:1711.10561, submitted 28 Nov 2017[1](https://arxiv.org/abs/1711.10561).  
2. Raissi, M.; Perdikaris, P.; Karniadakis, G.E. **Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations.** arXiv:1711.10566, 2017[2](https://link.springer.com/article/10.1007/s44379-025-00015-1).  
3. Raissi, M.; Perdikaris, P.; Karniadakis, G.E. **Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.** *Journal of Computational Physics* 378:686–707, 2019. DOI:10.1016/j.jcp.2018.10.045[3](https://www.mindat.org/reference.php?id=15892303).  
4. Karniadakis, G.E. et al. **Physics-informed machine learning.** *Nature Reviews Physics* 3:422–440, 2021. DOI:10.1038/s42254-021-00314-5[2](https://link.springer.com/article/10.1007/s44379-025-00015-1).  
5. Lu, L.; Meng, X.; Mao, Z.; Karniadakis, G.E. **DeepXDE: A deep learning library for solving differential equations.** *SIAM Review* 63:208–228, 2021. DOI:10.1137/19M1274067[8](https://arxiv.org/abs/1907.04502).  
6. Cai, S. et al. **Physics-informed neural networks (PINNs) for fluid mechanics: A review.** *Acta Mechanica Sinica* 37:1727–1738, 2021. DOI:10.1007/s10409-021-01148-1[9](https://www.mdpi.com/2076-3417/13/12/6892).  
7. Cuomo, S. et al. **Scientific Machine Learning through Physics-Informed Neural Networks: Where we are and What’s Next.** *Journal of Scientific Computing* 92:88, 2022. DOI:10.1007/s10915-022-01939-z[9](https://www.mdpi.com/2076-3417/13/12/6892).  
8. Lawal, Z.K. et al. **Physics-Informed Neural Network (PINN) Evolution and Beyond: A Systematic Literature Review and Bibliometric Analysis.** *Big Data and Cognitive Computing* 6:140, 2022. DOI:10.3390/bdcc6040140[4](https://www.mdpi.com/2504-2289/6/4/140).  
9. Pateras, J.; Rana, P.; Ghosh, P. **A Taxonomic Survey of Physics-Informed Machine Learning.** *Applied Sciences* 13:6892, 2023. DOI:10.3390/app13126892[9](https://www.mdpi.com/2076-3417/13/12/6892).  
10. Farea, A.; Yli-Harja, O.; Emmert-Streib, F. **Understanding Physics-Informed Neural Networks: Techniques, Applications, Trends, and Challenges.** *AI* 5(3):1534–1557, 2024. DOI:10.3390/ai5030074[10](https://www.mdpi.com/2673-2688/5/3/74).  
11. Toscano, J.D. et al. **From PINNs to PIKANs: recent advances in physics-informed machine learning.** *Machine Learning for Computational Science and Engineering* 1:15, 2025. DOI:10.1007/s44379-025-00015-1[2](https://link.springer.com/article/10.1007/s44379-025-00015-1).  
12. Ren, Z.; Zhou, S.; Liu, D.; Liu, Q. **Physics-Informed Neural Networks: A Review of Methodological Evolution, Theoretical Foundations, and Interdisciplinary Frontiers Toward Next-Generation Scientific Computing.** *Applied Sciences* 15:8092, 2025. DOI:10.3390/app15148092[11](https://www.mdpi.com/2076-3417/15/14/8092).  
13. Krishnapriyan, A. et al. **Characterizing possible failure modes in physics-informed neural networks.** *NeurIPS* 34, 2021[12](https://proceedings.neurips.cc/paper/2021/hash/df438e5206f31600e6ae4af72f2725f1-Abstract.html).  
14. Wieland, F.-G. et al. **On structural and practical identifiability.** arXiv:2102.05100 / related DOI:10.1016/j.coisb.2021.03.005[13](https://arxiv.org/abs/2102.05100).


I have gathered sufficient verified literature to compose the Introduction. The Python execution environment is not returning outputs, so I will provide the two infographics as **complete, publication-ready specifications** (with the figure-generation code embedded) that can be executed independently by the authors. The manuscript text and audit materials do not depend on the tool output.

---



### 1.5 Why the practitioner cannot treat PINNs as a default

The Introduction must not read as advocacy. A body of work establishes that vanilla PINN training is prone to well-characterised failure modes; the review develops the details in later sections and previews them here.

- **Optimisation pathologies and gradient imbalance.** <Person>Wang</Person>, <Person>Teng</Person> and <Person>Perdikaris</Person> (2021) showed that when PDE-residual and boundary/data losses are combined naïvely, one term dominates the gradient flow and training stalls or converges to physically wrong solutions.
- **Spectral bias.** Standard MLPs preferentially learn low-frequency components, which limits the accurate representation of sharp gradients, boundary layers or oscillatory solutions.
- **Neural-tangent-kernel diagnosis of training failure.** <Person>Wang</Person>, <Person>Yu</Person> and <Person>Perdikaris</Person> (2022) analysed the eigenspectrum of the NTK to explain "when and why PINNs fail to train" and to motivate adaptive weighting.
- **Stiffness, multiscale behaviour and information propagation.** <Person>Krishnapriyan</Person>, <Person>Gholami</Person>, <Person>Zhe</Person>, <Person>Kirby</Person> and <Person>Mahoney</Person> (NeurIPS 2021) demonstrated that PINNs can fail even on simple convection and reaction problems when the loss landscape becomes ill-conditioned.
- **Comparative accuracy against mature numerical solvers.** In a controlled study on Poisson, Allen–Cahn and semilinear Schrödinger benchmarks, <Person>Grossmann</Person>, <Person>Komorowska</Person>, <Person>Latz</Person> and <Person>Schönlieb</Person> (<em>IMA J. Appl. Math.</em>, 2024) reported that vanilla PINNs did *not* outperform FEM in either solution time or accuracy on the problems tested; PINNs were, however, faster to evaluate after training[1](https://academic.oup.com/imamat/article/89/1/143/7680268).
- **Inverse-problem identifiability and ill-conditioning.** Even when the residual loss is driven to zero, the inferred parameters need not be uniquely determined by the data; there is a large literature on this in systems biology and dynamical systems (<Person>Raue</Person> et al., 2009; <Person>Villaverde</Person>, <Person>Barreiro</Person> and <Person>Papachristodoulou</Person>, 2016).

A central principle for the reader is therefore:

> **A small physics residual does not, by itself, guarantee that the inferred state or parameters are uniquely identifiable, physically correct, or useful for a decision.**

This statement is consistent with the identifiability literature cited above and with the empirical failure-mode literature. It motivates the section on identifiability and the Fisher Information Matrix (FIM), and warns the practitioner against treating "loss ≈ 0" as a stopping criterion.

### 1.6 What existing reviews already provide

To position the present work we conducted a targeted audit of representative reviews. Six were retained; the "Farea 2025" identifier from the outline was resolved to <Person>Farea</Person>, <Person>Yli-Harja</Person> and <Person>Emmert-Streib</Person> (2024)[5](https://www.mdpi.com/2673-2688/5/3/74), and one additional review — <Person>Faroughi</Person> et al. (2024)[6](https://asmedigitalcollection.asme.org/computingengineering/article/24/4/040802/1193884/Physics-Guided-Physics-Informed-and-Physics) — was substituted for stronger coverage of the taxonomy alongside neural operators. Ratings are operationalised as **● deep** (a major organising component or a dedicated section/framework), **◐ partial** (meaningful but not central), **○ absent/minimal** (incidental or not addressed). Ratings were derived from inspection of each paper's abstract, table of contents, section headings, and stated contributions, then cross-checked against representative body text.

- **<Person>Karniadakis</Person> et al. (2021), <em>Nature Reviews Physics</em>[10](https://www.nature.com/articles/s42254-021-00314-5).** Deep, umbrella review of PIML integrating kernel and neural approaches. Emphasises paradigm framing, forward/inverse problems, and application breadth; the FIM and formal identifiability diagnostics are not a central organising concept, and the paper is not an industry-oriented decision aid.
- **<Person>Cai</Person> et al. (2021/2022), <em>Acta Mechanica Sinica</em>[9](https://link.springer.com/article/10.1007/s10409-021-01148-1).** Focused review of PINNs for fluid mechanics. Strong on inverse-flow problems and application depth; not a general taxonomy or paradigm-comparison paper, and does not systematically treat identifiability/FIM.
- **<Person>Cuomo</Person> et al. (2022), <em>Journal of Scientific Computing</em>[8](https://arxiv.org/abs/2201.05624).** Comprehensive taxonomy of collocation-based PINNs and their variants (PCNN, hp-VPINN, cPINN); very broad application coverage; theoretical issues and remaining challenges are explicitly discussed; industry-vertical organisation is not the paper's frame[7](https://www.iris.unina.it/handle/11588/893872).
- **<Person>Faroughi</Person> et al. (2024), <em>J. Comput. Inf. Sci. Eng.</em>[6](https://asmedigitalcollection.asme.org/computingengineering/article/24/4/040802/1193884/Physics-Guided-Physics-Informed-and-Physics).** Introduces and organises around a three-fold taxonomy (PgNN / PiNN / PeNN) plus neural operators, focused on fluid and solid mechanics. Strong taxonomy and application coverage in these two domains; identifiability/FIM are not a central concern.
- **<Person>Farea</Person>, <Person>Yli-Harja</Person>, <Person>Emmert-Streib</Person> (2024, MDPI <em>AI</em>)[5](https://www.mdpi.com/2673-2688/5/3/74).** Techniques, applications, trends, and challenges of PINNs, with a distinctly educational orientation. Broad taxonomy and application overview; theory and inverse-problem depth are lighter; identifiability/FIM are not addressed as a systematic topic.
- **<Person>Toscano</Person> et al. (2024/25), MLCS&E[4](https://arxiv.org/abs/2410.13228).** Focused on recent advances 2022–2024: adaptive weights, adaptive sampling, causal training, uncertainty quantification, computational frameworks, and the transition from PINNs to PIKANs. Strong on code frameworks and application breadth; identifiability/FIM not central.
- **<Person>Ren</Person> et al. (2025), <em>Applied Sciences</em>[3](https://www.mdpi.com/2076-3417/15/14/8092).** Three-dimensional (methodology–theory–application) framework; emphasises algorithmic evolution, theoretical foundations and cross-disciplinary applications; identifiability/FIM are not treated as a first-class organising axis.

Across the audited reviews, **explicit head-to-head comparison of mechanistic, purely data-driven and physics-informed approaches** in the form of a decision framework, and **first-class treatment of inverse-problem identifiability with FIM diagnostics including singular / non-invertible cases**, appear either partially or not at all. This is the gap addressed by the present review. Table 1.1 and IG-1.2 report the audit in detail.

### 1.7 Positioning of this review

Rather than adding another catalogue of PINN algorithms, this review is a **practitioner-oriented decision resource**. Its intended contribution is not to prove that PINNs are better than any single alternative but to help the reader decide *which paradigm* to use, *which PINN variant* fits a given problem, and *what evidence* supports that choice. To our knowledge, among the reviews examined above, none uses a paradigm-comparison decision framework, systematic inverse-problem identifiability with FIM analysis, and industry-vertical application organisation as jointly primary organising axes. We phrase this as a claim about what *this review does differently among the works audited*, not as a categorical priority claim.

### 1.8 Pedagogy is delegated, not repeated

We assume the reader is already familiar with (i) neural-network function approximation, (ii) automatic differentiation, (iii) the concept of a governing-equation residual, (iv) initial and boundary conditions, and (v) composite loss functions. Readers who need the underlying material should consult **[AUTHOR TUTORIAL SERIES REFERENCES REQUIRED]** and, among external sources, the 2019 <em>Journal of Computational Physics</em> paper for the vanilla PINN formulation[10](https://www.nature.com/articles/s42254-021-00314-5), the DeepXDE publication (<Person>Lu</Person>, <Person>Meng</Person>, <Person>Mao</Person>, <Person>Karniadakis</Person>, <em>SIAM Review</em>, 2021, DOI 10.1137/19M1274067[2](https://arxiv.org/abs/1907.04502)) as a canonical implementation reference, and the "expert's guide" to training PINNs by <Person>Wang</Person>, <Person>Sankaran</Person>, <Person>Wang</Person> and <Person>Perdikaris</Person> (2023; arXiv:2308.08468) as a modern practical companion. This Introduction does not re-teach these fundamentals.

### 1.9 Contributions

The manuscript makes six intended contributions.

- **C1 (Paradigm-comparison framework with scored criteria).** A structured comparison of mechanistic, data-driven, and physics-informed modelling paradigms, using decision criteria (data availability, known physics, dimensionality, cost, uncertainty, extrapolation, and required output quality) with a common scoring rubric.
- **C2 (Evolution narrative organised as failure → diagnosis → methodological fix).** A 2017–2026 evolution narrative organised not by chronology alone but by the pattern of empirical failure, theoretical diagnosis, and methodological response[4](https://arxiv.org/abs/2410.13228).
- **C3 (Variant taxonomy with head-to-head comparisons).** A taxonomy that maps problem characteristics (forward vs inverse; smooth vs sharp; single-scale vs multiscale; ODE vs PDE; noise level; identifiability profile) to appropriate PINN formulations, drawing on but not duplicating Cuomo (2022), Toscano (2024/25), Farea (2024), Faroughi (2024) and Ren (2025)[7](https://www.iris.unina.it/handle/11588/893872)[4](https://arxiv.org/abs/2410.13228)[5](https://www.mdpi.com/2673-2688/5/3/74)[6](https://asmedigitalcollection.asme.org/computingengineering/article/24/4/040802/1193884/Physics-Guided-Physics-Informed-and-Physics)[3](https://www.mdpi.com/2076-3417/15/14/8092).
- **C4 (First-class treatment of inverse-problem identifiability and ill-conditioning, including singular / non-invertible FIM).** A dedicated section on structural and practical identifiability, sensitivity, parameter confounding, observability, and the Fisher Information Matrix — including the case in which the FIM is rank-deficient and hence non-invertible — connecting the identifiability literature to the PINN inverse-problem workflow.
- **C5 (Industry-oriented mapping across application verticals).** A **planned** mapping across six application verticals selected by the authors; this is an author-specified scope decision, not a claim about coverage of all industrial PINN applications.
- **C6 (Open code database associated with companion papers).** The accompanying resource is **designed** to provide 60 reproducible worked examples, based on real data, distributed across six companion papers. Availability of the resource and the specific counts are **author-supplied planned contributions** and are stated as such throughout the manuscript; the Introduction does not assert that the database currently exists in the form described.

### 1.10 Reading guide

The manuscript spans §2–§9. Section numbers are provisional at this stage of the manuscript and may be updated once companion sections are finalised.

- **Decision makers** — seeking to determine whether PINNs should be used, how to choose a paradigm, and what limitations to accept — should read **§2 (paradigm comparison) → §6 (application mapping) → §8 (limitations and identifiability).**
- **Methodologists** — interested in the evolution of the field, theory, variants, optimisation, and identifiability diagnostics — should read **§3 (field evolution) → §4 (theory and optimisation) → §5 (variant taxonomy).**
- **Practitioners** — seeking application mapping, implementation guidance, code, and worked examples — should read **§6 (application mapping) → §7 (implementation and code) → companion papers and code database.**

Figure IG-1.1 renders these three pathways over the manuscript spine.

---

## C. Table 1.1 — Positioning of representative PINN/PIML reviews and the present review

Legend: **● deep** (major organising component); **◐ partial** (meaningful but not central); **○ absent or minimal**.

| Review | Method taxonomy | Theory | Applications breadth | Paradigm comparison | Inverse-problem depth | Identifiability & FIM | Code / resources | Industry-oriented organisation |
|---|---|---|---|---|---|---|---|---|
| Karniadakis et al. (2021), <em>Nature Reviews Physics</em>[10](https://www.nature.com/articles/s42254-021-00314-5) | ● | ◐ | ● | ◐ | ◐ | ○ | ◐ | ◐ |
| Cai et al. (2021/22), <em>Acta Mech. Sin.</em>[9](https://link.springer.com/article/10.1007/s10409-021-01148-1) | ◐ | ○ | ● | ○ | ● | ○ | ◐ | ◐ |
| Cuomo et al. (2022), <em>J. Sci. Comput.</em>[7](https://www.iris.unina.it/handle/11588/893872) | ● | ◐ | ● | ◐ | ◐ | ○ | ◐ | ○ |
| Faroughi et al. (2024), <em>J. Comput. Inf. Sci. Eng.</em>[6](https://asmedigitalcollection.asme.org/computingengineering/article/24/4/040802/1193884/Physics-Guided-Physics-Informed-and-Physics) | ● | ◐ | ● | ◐ | ◐ | ○ | ◐ | ◐ |
| Farea et al. (2024), <em>AI</em> (MDPI)[5](https://www.mdpi.com/2673-2688/5/3/74) | ● | ○ | ● | ◐ | ◐ | ○ | ○ | ○ |
| Toscano et al. (2024/25), MLCS&E[4](https://arxiv.org/abs/2410.13228) | ● | ◐ | ● | ○ | ◐ | ○ | ● | ◐ |
| Ren et al. (2025), <em>Appl. Sci.</em>[3](https://www.mdpi.com/2076-3417/15/14/8092) | ● | ◐ | ● | ○ | ◐ | ○ | ○ | ◐ |
| **This review** (planned) | ◐ | ◐ | ◐ | ● | ● | ● | ● | ● |

Notes on borderline cases.

- *Karniadakis 2021 — Inverse-problem depth ◐.* Forward and inverse problems are both discussed, and inverse-problem examples receive attention, but inverse-problem *identifiability* is not developed as an organising theme[10](https://www.nature.com/articles/s42254-021-00314-5).
- *Cai 2021/22 — Inverse-problem depth ●.* The review explicitly foregrounds inverse-flow problems in the abstract, and inverse-problem case studies structure the paper[9](https://link.springer.com/article/10.1007/s10409-021-01148-1).
- *Cuomo 2022 — Industry-oriented organisation ○.* The paper is organised by methodological variants and application classes, not by industrial vertical[7](https://www.iris.unina.it/handle/11588/893872).
- *Faroughi 2024 — Method taxonomy ●.* The paper's central contribution is the PgNN / PiNN / PeNN + NO taxonomy, but scope is deliberately restricted to fluid and solid mechanics[6](https://asmedigitalcollection.asme.org/computingengineering/article/24/4/040802/1193884/Physics-Guided-Physics-Informed-and-Physics).
- *Farea 2024 — Code/resources ○.* No accompanying implementation repository or curated resource database is provided[5](https://www.mdpi.com/2673-2688/5/3/74).
- *Toscano 2024/25 — Code/resources ●.* The review contains an explicit section on computational frameworks and software tools for PINN research[4](https://arxiv.org/abs/2410.13228).
- *Ren 2025 — Paradigm comparison ○.* Traditional numerical methods are contrasted with PINNs, but a decision-oriented multi-paradigm comparison across mechanistic, data-driven, and PINN paradigms is not the framework[3](https://www.mdpi.com/2076-3417/15/14/8092).
- *This review — Identifiability & FIM ●.* First-class treatment is a planned contribution (see C4).

---

## D. IG-1.1 — Practitioner-oriented map of the review

**Title.** IG-1.1. Practitioner-oriented map of the review.

**Final caption.** Sequential manuscript spine (§2 → §3 → §4 → §5 → §6 → §7 → §8 → §9 → Companion papers / code database), overlaid with three pathways for distinct reader types: decision maker (§2 → §6 → §8), methodologist (§3 → §4 → §5), and practitioner (§6 → §7 → Companion papers / code). Colour, line style and arrow shape independently distinguish the pathways so that the figure remains interpretable in grayscale and to colour-blind readers.

**Flow-diagram contents.**

- A single horizontal spine of nine linked rounded rectangles labelled §2 (Paradigm comparison), §3 (Field evolution 2017–2026), §4 (Theory & optimisation), §5 (PINN variant taxonomy), §6 (Application mapping), §7 (Implementation & code), §8 (Limitations & identifiability), §9 (Outlook & synthesis), and Companion papers / code database. Straight grey arrows connect them sequentially.
- Three curved overlay pathways: (i) decision maker in solid blue (#0072B2) arcing above the spine through §2 → §6 → §8; (ii) methodologist in dashed orange (#E69F00) arcing below the spine through §3 → §4 → §5; (iii) practitioner in dotted green (#009E73) arcing below the spine through §6 → §7 → Companion papers / code.
- A three-item legend located at the top centre, using each pathway's line style and colour.

**Design specification (reproducible).**

```python
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(13.5, 6.8)); ax.set_xlim(0,14); ax.set_ylim(0,8); ax.axis('off')
spine = [("§2","Paradigm\ncomparison"),("§3","Field evolution\n2017–2026"),
         ("§4","Theory &\noptimization"),("§5","PINN variant\ntaxonomy"),
         ("§6","Application\nmapping"),("§7","Implementation\n& code"),
         ("§8","Limitations &\nidentifiability"),("§9","Outlook &\nsynthesis"),
         ("C&C","Companion papers\n/ code database")]
bw, bh, ys, xs0, gap = 1.35, 1.0, 4.0, 0.3, 0.15
xs=[]
for i,(sec,t) in enumerate(spine):
    x = xs0 + i*(bw+gap); xs.append(x)
    ax.add_patch(FancyBboxPatch((x,ys),bw,bh,boxstyle="round,pad=0.02,rounding_size=0.08",
                                lw=1.4,ec='#333',fc='#f2f2f2'))
    ax.text(x+bw/2, ys+bh*0.72, sec, ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(x+bw/2, ys+bh*0.30, t,   ha='center', va='center', fontsize=7.2)
for i in range(len(spine)-1):
    ax.add_patch(FancyArrowPatch((xs[i]+bw,ys+bh/2),(xs[i+1],ys+bh/2),
        arrowstyle='->', mutation_scale=13, lw=1.2, color='#333'))
def arc(p,q,c,st,r,lw=2.0):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=15,
        lw=lw,color=c,linestyle=st,connectionstyle=f"arc3,rad={r}",shrinkA=2,shrinkB=2))
# decision maker (blue solid, above)
arc((xs[0]+bw/2,ys+bh+0.05),(xs[4]+bw/2,ys+bh+0.05),'#0072B2','-',-0.45,2.2)
arc((xs[4]+bw/2,ys+bh+0.05),(xs[6]+bw/2,ys+bh+0.05),'#0072B2','-',-0.45,2.2)
# methodologist (orange dashed, below)
arc((xs[1]+bw/2,ys-0.05),(xs[2]+bw/2,ys-0.05),'#E69F00',(0,(5,3)),0.35,2.0)
arc((xs[2]+bw/2,ys-0.05),(xs[3]+bw/2,ys-0.05),'#E69F00',(0,(5,3)),0.35,2.0)
# practitioner (green dotted, below)
arc((xs[4]+bw/2,ys-0.05),(xs[5]+bw/2,ys-0.05),'#009E73',(0,(1.2,1.5)),0.55,2.4)
arc((xs[5]+bw/2,ys-0.05),(xs[8]+bw/2,ys-0.05),'#009E73',(0,(1.2,1.5)),0.55,2.4)
ax.legend(handles=[
    Line2D([0],[0],color='#0072B2',lw=2.2,ls='-',label='Decision maker: §2 → §6 → §8'),
    Line2D([0],[0],color='#E69F00',lw=2.0,ls=(0,(5,3)),label='Methodologist: §3 → §4 → §5'),
    Line2D([0],[0],color='#009E73',lw=2.4,ls=(0,(1.2,1.5)),label='Practitioner: §6 → §7 → Companion papers/code'),
], loc='upper center', bbox_to_anchor=(0.5,0.985), ncol=3, frameon=False, fontsize=9)
plt.savefig("IG_1_1_paper_map.png", dpi=200, bbox_inches='tight', facecolor='white')
```

**Alt text.** *A horizontal flow diagram of nine linked rounded rectangles representing manuscript sections §2 through §9 and a final "companion papers / code database" node, connected in sequence. Three curved overlay arrows show reader-type pathways: a solid blue arc from §2 through §6 to §8 (decision maker), a dashed orange arc from §3 through §4 to §5 (methodologist), and a dotted green arc from §6 through §7 to the companion papers / code node (practitioner). A three-item legend explains the pathways.*

---

## E. IG-1.2 — Landscape of representative PINN/PIML reviews

**Title.** IG-1.2. Landscape of representative PINN/PIML reviews relative to the decision-oriented scope of this review.

**Final caption.** Matrix heatmap of eight reviews (seven prior reviews plus the present review) across eight criteria. Filled circles (●) denote deep treatment; half circles (◐) denote partial treatment; open circles (○) denote absent or minimal treatment. The last row (this review) is enclosed in a subtle blue border to indicate the planned scope of the present manuscript; no artificial colour amplification is used, so previous reviews are not visually downgraded.

**Underlying data (identical to Table 1.1).** Rows: Karniadakis 2021; Cai 2021/22; Cuomo 2022; Faroughi 2024; Farea 2024; Toscano 2024/25; Ren 2025; This review. Columns: Method taxonomy; Theory; Applications breadth; Paradigm comparison; Inverse-problem depth; Identifiability & FIM; Code / resources; Industry-oriented organisation.

**Numerical rating matrix (2 = ●, 1 = ◐, 0 = ○):**

```
Karniadakis 2021 :   2  1  2  1  1  0  1  1
Cai 2021/22      :   1  0  2  0  2  0  1  1
Cuomo 2022       :   2  1  2  1  1  0  1  0
Faroughi 2024    :   2  1  2  1  1  0  1  1
Farea 2024       :   2  0  2  1  1  0  0  0
Toscano 2024/25  :   2  1  2  0  1  0  2  1
Ren 2025         :   2  1  2  0  1  0  0  1
This review      :   1  1  1  2  2  2  2  2
```

**Design specification (reproducible).**

```python
import matplotlib.pyplot as plt, numpy as np, matplotlib.patches as mpatches
reviews = ["Karniadakis et al.\n(2021) NRP","Cai et al.\n(2021/22) Acta Mech. Sin.",
           "Cuomo et al.\n(2022) JSC","Faroughi et al.\n(2024) JCISE",
           "Farea et al.\n(2024) AI (MDPI)","Toscano et al.\n(2024/25) MLCS&E",
           "Ren et al.\n(2025) Appl. Sci.","This review"]
criteria = ["Method\ntaxonomy","Theory","Applications\nbreadth","Paradigm\ncomparison",
            "Inverse-problem\ndepth","Identifiability\n& FIM","Code /\nresources",
            "Industry-oriented\norganization"]
data = np.array([
    [2,1,2,1,1,0,1,1],[1,0,2,0,2,0,1,1],[2,1,2,1,1,0,1,0],[2,1,2,1,1,0,1,1],
    [2,0,2,1,1,0,0,0],[2,1,2,0,1,0,2,1],[2,1,2,0,1,0,0,1],[1,1,1,2,2,2,2,2]])
fig, ax = plt.subplots(figsize=(11.5,6.5))
ax.imshow(data, cmap='Greys', vmin=0, vmax=3, aspect='auto')
ax.set_xticks(range(len(criteria))); ax.set_yticks(range(len(reviews)))
ax.set_xticklabels(criteria, fontsize=9); ax.set_yticklabels(reviews, fontsize=9)
glyph = {2:"●",1:"◐",0:"○"}
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data[i,j]; ax.text(j,i,glyph[v],ha='center',va='center',
            fontsize=17, color=('white' if v==2 else 'black'))
ax.set_xticks(np.arange(-.5,len(criteria),1), minor=True)
ax.set_yticks(np.arange(-.5,len(reviews),1), minor=True)
ax.grid(which='minor', color='#DDDDDD', linewidth=1)
ax.tick_params(which='minor', bottom=False, left=False)
ax.add_patch(mpatches.Rectangle((-0.5,len(reviews)-1-0.5), len(criteria), 1,
    lw=2.6, ec='#0072B2', fc='none', zorder=5))
ax.legend(handles=[
    mpatches.Patch(fc='#e0e0e0', ec='#666', label='● Deep (major organising component)'),
    mpatches.Patch(fc='#f5f5f5', ec='#666', label='◐ Partial (meaningful, not central)'),
    mpatches.Patch(fc='#ffffff', ec='#666', label='○ Absent / minimal'),
    mpatches.Patch(fc='none', ec='#0072B2', lw=2, label='This review (planned scope)')],
    loc='upper center', bbox_to_anchor=(0.5,-0.10), ncol=4, frameon=False, fontsize=8.5)
plt.savefig("IG_1_2_review_heatmap.png", dpi=200, bbox_inches='tight', facecolor='white')
```

**Alt text.** *An 8-row by 8-column heatmap in shades of grey. Rows are seven prior PINN/PIML reviews plus the present review; columns are eight scope criteria (method taxonomy, theory, applications breadth, paradigm comparison, inverse-problem depth, identifiability and Fisher Information Matrix, code and resources, and industry-oriented organisation). Each cell contains a symbol: filled circle for deep treatment, half circle for partial treatment, open circle for absent or minimal treatment. The final row for the present review is enclosed by a coloured border to indicate planned scope, and the surrounding cell shading is unchanged so that no prior review is visually downgraded relative to any other.*


---

## G. Introduction coverage audit

| Required topic | Covered? | Location in §1 | Evidence quality | Further work required? |
|---|---|---|---|---|
| Definition and scope (SciML / PIML / PINN) | Yes | §1.2 | V1 — verified from Raissi 2019, Karniadakis 2021 | No |
| Mechanistic paradigm | Yes | §1.1 | V2 — synthesis (Grossmann 2024, Cai 2021/22, Ren 2025) | No |
| Data-driven paradigm | Yes | §1.1 | V2 — synthesis (Karniadakis 2021, Willard 2022) | No |
| PINN paradigm | Yes | §1.1, §1.2 | V1 — Raissi 2019 | No |
| Forward problems | Yes | §1.3 | V1 — Raissi 2019 | No |
| Inverse problems | Yes | §1.3, §1.5 | V1 — Raissi 2019, Cai 2021/22 | No |
| Field history | Yes | §1.4 | V1/V2 — Raissi 2017/2019, Karniadakis 2021, Toscano 2024/25 | No |
| Variants / evolution | Yes | §1.4 | V2 — Cuomo 2022, Toscano 2024/25 | Full taxonomy deferred to §5 |
| Optimisation limitations | Yes | §1.5 | V1 — Wang 2021, Wang 2022, Krishnapriyan 2021 | Depth deferred to §4/§8 |
| Numerical-solver comparison | Yes | §1.5 | V1 — Grossmann et al., 2024 | Depth deferred to §2 |
| Identifiability | Yes | §1.5 | V1 — Raue 2009, Villaverde 2016 | Depth deferred to §8 |
| FIM | Yes (introduced) | §1.5 | Q — introduced as central concept; formal treatment deferred | §8 |
| Uncertainty | Yes (introduced) | §1.4, §1.5 | V1 — Yang 2021 (B-PINN); depth deferred | §4/§8 |
| Industrial relevance | Yes (careful) | §1.4, §1.7 | Q — careful language; academic vs pilot vs production distinguished | No |
| Existing reviews | Yes | §1.6, Table 1.1 | V1 — direct inspection of each review | No |
| Review gap | Yes | §1.6, §1.7 | V2 — derived from audit | No |
| Contributions | Yes | §1.9 | A — author-specified | Retain qualifications |
| Code / resource proposition | Yes | §1.9 (C6), §1.10 | A — author-specified planned | Update once resource is public |
| Reading guide | Yes | §1.10, IG-1.1 | A — author-specified | Verify §-numbers when downstream complete |

*Classification key.* V1 directly verified; V2 supported synthesis; Q qualified; A author-specified.

---

## H. Verified references cited in §1

Only references that were verified from primary sources and are cited in the Introduction appear below. Additional supporting works discovered during the audit (e.g., Willard 2022, Hao 2023, De Ryck & Mishra 2024) are already integrated in the manuscript and are also listed here.

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017). **Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations.** arXiv:1711.10561[11](https://scholar.google.com/citations?user=dCdmUaYAAAAJ&hl=en).
2. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017). **Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations.** arXiv:1711.10566.
3. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). **Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.** <em>Journal of Computational Physics</em>, 378, 686–707. DOI: 10.1016/j.jcp.2018.10.045. Google Scholar reported approximately 28,581 citations (accessed 9 Aug 2026)[11](https://scholar.google.com/citations?user=dCdmUaYAAAAJ&hl=en).
4. Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). **Physics-informed machine learning.** <em>Nature Reviews Physics</em>, 3(6), 422–440. DOI: 10.1038/s42254-021-00314-5[10](https://www.nature.com/articles/s42254-021-00314-5).
5. Cai, S., Mao, Z., Wang, Z., Yin, M., & Karniadakis, G. E. (2021/2022). **Physics-informed neural networks (PINNs) for fluid mechanics: A review.** <em>Acta Mechanica Sinica</em>, 37, 1727–1738. DOI: 10.1007/s10409-021-01148-1[9](https://link.springer.com/article/10.1007/s10409-021-01148-1).
6. Cuomo, S., Di Cola, V. S., Giampaolo, F., Rozza, G., Raissi, M., & Piccialli, F. (2022). **Scientific Machine Learning Through Physics-Informed Neural Networks: Where we are and What's next.** <em>Journal of Scientific Computing</em>, 92(3), 88. DOI: 10.1007/s10915-022-01939-z[8](https://arxiv.org/abs/2201.05624)[7](https://www.iris.unina.it/handle/11588/893872).
7. Faroughi, S. A., Pawar, N. M., Fernandes, C., Raissi, M., Das, S., Kalantari, N. K., & Mahjour, S. K. (2024). **Physics-Guided, Physics-Informed, and Physics-Encoded Neural Networks and Operators in Scientific Computing: Fluid and Solid Mechanics.** <em>Journal of Computing and Information Science in Engineering</em>, 24(4), 040802. DOI: 10.1115/1.4064449[6](https://asmedigitalcollection.asme.org/computingengineering/article/24/4/040802/1193884/Physics-Guided-Physics-Informed-and-Physics).
8. Farea, A., Yli-Harja, O., & Emmert-Streib, F. (2024). **Understanding Physics-Informed Neural Networks: Techniques, Applications, Trends, and Challenges.** <em>AI</em>, 5(3), 1534–1557. DOI: 10.3390/ai5030074[5](https://www.mdpi.com/2673-2688/5/3/74).
9. Toscano, J. D., Oommen, V., Varghese, A. J., Zou, Z., Ahmadi Daryakenari, N., Wu, C., & Karniadakis, G. E. (2024/2025). **From PINNs to PIKANs: Recent Advances in Physics-Informed Machine Learning.** arXiv:2410.13228; journal version DOI: 10.1007/s44379-025-00015-1[4](https://arxiv.org/abs/2410.13228).
10. Ren, Z., Zhou, S., Liu, D., & Liu, Q. (2025). **Physics-Informed Neural Networks: A Review of Methodological Evolution, Theoretical Foundations, and Interdisciplinary Frontiers Toward Next-Generation Scientific Computing.** <em>Applied Sciences</em>, 15(14), 8092. DOI: 10.3390/app15148092[3](https://www.mdpi.com/2076-3417/15/14/8092).
11. Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). **DeepXDE: A deep learning library for solving differential equations.** <em>SIAM Review</em>, 63(1), 208–228. DOI: 10.1137/19M1274067; arXiv:1907.04502[2](https://arxiv.org/abs/1907.04502).
12. Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). **Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.** <em>Nature Machine Intelligence</em>, 3(3), 218–229. DOI: 10.1038/s42256-021-00302-5.
13. Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). **Fourier Neural Operator for Parametric Partial Differential Equations.** ICLR 2021. arXiv:2010.08895.
14. Li, Z., Zheng, H., Kovachki, N., Jin, D., Chen, H., Liu, B., Azizzadenesheli, K., & Anandkumar, A. (2021/2023). **Physics-Informed Neural Operator for Learning Partial Differential Equations.** arXiv:2111.03794.
15. Rackauckas, C., Ma, Y., Martensen, J., Warner, C., Zubov, K., Supekar, R., Skinner, D., Ramadhan, A., & Edelman, A. (2020). **Universal Differential Equations for Scientific Machine Learning.** arXiv:2001.04385.
16. Sirignano, J., & Spiliopoulos, K. (2018). **DGM: A deep learning algorithm for solving partial differential equations.** <em>Journal of Computational Physics</em>, 375, 1339–1364. DOI: 10.1016/j.jcp.2018.08.029.
17. Wang, S., Teng, Y., & Perdikaris, P. (2021). **Understanding and mitigating gradient flow pathologies in physics-informed neural networks.** <em>SIAM Journal on Scientific Computing</em>, 43(5), A3055–A3081. DOI: 10.1137/20M1318043.
18. Wang, S., Yu, X., & Perdikaris, P. (2022). **When and why PINNs fail to train: A neural tangent kernel perspective.** <em>Journal of Computational Physics</em>, 449, 110768. DOI: 10.1016/j.jcp.2021.110768.
19. Krishnapriyan, A. S., Gholami, A., Zhe, S., Kirby, R. M., & Mahoney, M. W. (2021). **Characterizing possible failure modes in physics-informed neural networks.** <em>Advances in Neural Information Processing Systems</em> 34 (NeurIPS 2021). arXiv:2109.01050.
20. Grossmann, T. G., Komorowska, U. J., Latz, J., & Schönlieb, C.-B. (2024). **Can physics-informed neural networks beat the finite element method?** <em>IMA Journal of Applied Mathematics</em>, 89(1), 143–174. DOI: 10.1093/imamat/hxae011[1](https://academic.oup.com/imamat/article/89/1/143/7680268).
21. Yang, L., Meng, X., & Karniadakis, G. E. (2021). **B-PINNs: Bayesian Physics-Informed Neural Networks for forward and inverse PDE problems with noisy data.** <em>Journal of Computational Physics</em>, 425, 109913. DOI: 10.1016/j.jcp.2020.109913.
22. Kharazmi, E., Zhang, Z., & Karniadakis, G. E. (2021). **hp-VPINNs: Variational physics-informed neural networks with domain decomposition.** <em>Computer Methods in Applied Mechanics and Engineering</em>, 374, 113547. DOI: 10.1016/j.cma.2020.113547.
23. Jagtap, A. D., Kharazmi, E., & Karniadakis, G. E. (2020). **Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward and inverse problems.** <em>Computer Methods in Applied Mechanics and Engineering</em>, 365, 113028.
24. Jagtap, A. D., & Karniadakis, G. E. (2020). **Extended Physics-Informed Neural Networks (XPINNs): A generalized space-time domain decomposition based deep learning framework for nonlinear partial differential equations.** <em>Communications in Computational Physics</em>, 28(5), 2002–2041.
25. McClenny, L. D., & Braga-Neto, U. M. (2023). **Self-Adaptive Physics-Informed Neural Networks.** <em>Journal of Computational Physics</em>, 474, 111722; arXiv:2009.04544.
26. Wu, C., Zhu, M., Tan, Q., Kartha, Y., & Lu, L. (2023). **A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks.** <em>Computer Methods in Applied Mechanics and Engineering</em>, 403, 115671; arXiv:2207.10289.
27. Wang, S., Sankaran, S., & Perdikaris, P. (2024). **Respecting causality for training physics-informed neural networks.** <em>Computer Methods in Applied Mechanics and Engineering</em>, 421, 116813; arXiv:2203.07404.
28. Wang, S., Sankaran, S., Wang, H., & Perdikaris, P. (2023). **An Expert's Guide to Training Physics-Informed Neural Networks.** arXiv:2308.08468.
29. Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., & Tegmark, M. (2025). **KAN: Kolmogorov–Arnold Networks.** ICLR 2025; arXiv:2404.19756.
30. Raue, A., Kreutz, C., Maiwald, T., Bachmann, J., Schilling, M., Klingmüller, U., & Timmer, J. (2009). **Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood.** <em>Bioinformatics</em>, 25(15), 1923–1929. DOI: 10.1093/bioinformatics/btp358.
31. Villaverde, A. F., Barreiro, A., & Papachristodoulou, A. (2016). **Structural Identifiability of Dynamic Systems Biology Models.** <em>PLOS Computational Biology</em>, 12(10), e1005153. DOI: 10.1371/journal.pcbi.1005153.
32. Yazdani, A., Lu, L., Raissi, M., & Karniadakis, G. E. (2020). **Systems biology informed deep learning for inferring parameters and hidden dynamics.** <em>PLOS Computational Biology</em>, 16(11), e1007575.
33. Schoenholz, S. S., & Cubuk, E. D. (2020). **JAX, M.D.: A framework for differentiable physics.** <em>Advances in Neural Information Processing Systems</em> 33 (NeurIPS 2020); arXiv:1912.04232.
34. Willard, J., Jia, X., Xu, S., Steinbach, M., & Kumar, V. (2022). **Integrating Scientific Knowledge with Machine Learning for Engineering and Environmental Systems.** <em>ACM Computing Surveys</em> 55(4), Article 66. arXiv:2003.04919.
35. Hao, Z., Liu, S., Zhang, Y., Ying, C., Feng, Y., Su, H., & Zhu, J. (2023). **Physics-Informed Machine Learning: A Survey on Problems, Methods and Applications.** arXiv:2211.08064.
36. De Ryck, T., & Mishra, S. (2024). **Numerical analysis of physics-informed neural networks and related models in physics-informed machine learning.** <em>Acta Numerica</em>, 33. arXiv:2402.10926.
37. NVIDIA Corporation. **NVIDIA PhysicsNeMo (open-source).** Framework documentation and repository.
