# Module 02 · Solving ODEs with Neural Networks

### Parameter estimation and forward solutions for ordinary differential equations with Physics-Informed Neural Networks

This module is a set of self-contained notebooks that solve **real ODE problems
from the literature** with PINNs — with a deliberate emphasis on **parameter
estimation** (recovering rate constants, growth rates, transmission rates, and
other kinetic coefficients from data). Every problem is drawn from published work,
and every notebook uses either real measured data or data reconstructed from a
paper's published parameters (clearly labelled).

Each notebook is structured the same way: **describe the paper → why a PINN helps
here → our approach → runnable code → solve and estimate**. All code runs on CPU
and was verified end-to-end (the printed parameter estimates below come from full
training runs).

---

## The notebooks

| # | Notebook | Model | Estimated parameters | Data provenance |
|---|---|---|---|---|
| 01 | [Logistic population growth](notebooks/01_logistic_population_growth.ipynb) | single ODE | growth rate $r$, carrying capacity $K$ | **Real** U.S. Census 1790–1940 |
| 02 | [Consecutive reaction kinetics A→B→C](notebooks/02_consecutive_reaction_kinetics.ipynb) | 3-ODE system | rate constants $k_1,k_2$ | Reconstructed from published constants |
| 03 | [Lotka–Volterra predator–prey](notebooks/03_lotka_volterra_lynx_hare.ipynb) | 2-ODE system | $\alpha,\beta,\gamma,\delta$ | **Real** Hudson Bay lynx–hare 1900–1920 |
| 04 | [SIR epidemic model](notebooks/04_sir_epidemic_model.ipynb) | 3-ODE system | transmission $\beta$, recovery $\gamma$, $R_0$ | **Real** 1978 boarding-school flu outbreak |
| 05 | [Van der Pol oscillator](notebooks/05_van_der_pol_oscillator.ipynb) | 2nd-order ODE | nonlinear damping $\mu$ | Reconstructed from literature value |
| 06 | [Stiff kinetics + QSSA (enzyme)](notebooks/06_stiff_kinetics_qssa_enzyme.ipynb) | stiff system → reduced | $V_{\max}, K_m$ | Reconstructed (Michaelis–Menten) |
| 07 | [Bioprocess viable cell density](notebooks/07_bioprocess_viable_cell_density.ipynb) | 2-ODE Monod system | $\mu_{\max}, K_s, Y_{xs}$ | **Open** IndPenSim benchmark (+ runnable fallback) |

### Techniques covered
- Making unknown coefficients **trainable parameters** inside the PINN (with positivity reparameterisation via softplus).
- **Nested automatic differentiation** for second-order ODEs (notebook 05).
- **Two-phase / warm-start training** for coupled nonlinear inversions (notebooks 03, 07).
- **Adam → L-BFGS** two-stage optimisation throughout.
- Inferring **unobserved compartments** from a single measured signal (notebook 04).
- **Stiffness reduction** via the quasi-steady-state assumption (notebook 06).
- Non-dimensionalisation and residual scaling for stable training (all notebooks).

---

## Data provenance — read this

Two honest categories, both standard practice in the PINN literature:

**Real measured data (embedded in the notebook):**
- *Logistic* — U.S. decennial census counts (public domain).
- *Lotka–Volterra* — Hudson's Bay Company lynx & hare pelt records, 1900–1920 (Hewitt 1921).
- *SIR* — daily counts of boys confined to bed, 1978 influenza outbreak at an English boarding school (BMJ 1978).

**Reconstructed from a paper's published parameters** (the reference trajectory is
generated from the literature model + constants, then sampled with noise — exactly
how DeepXDE demonstrates the Lorenz inverse problem):
- *Consecutive kinetics, Van der Pol, Michaelis–Menten.*

**Open published dataset with a loader + fallback:**
- *Bioprocess (notebook 07)* uses the **IndPenSim** benchmark (Goldrick et al.).
  The notebook loads a real downloaded batch if you provide one
  (`INDPENSIM_CSV`), and otherwise runs on a batch generated from the same
  published Monod model so it works out-of-the-box. Download the real 100-batch
  data from the IndPenSim project (Mendeley Data / industrialpenicillinsimulation.com).

Nothing here requires you to download anything to run the notebooks; the only
optional download is the real IndPenSim batches for notebook 07.

---

## References

- Verhulst, P.-F. (1838). *Notice sur la loi que la population suit dans son accroissement.* Corresp. Math. Phys. 10:113–121.
- Pearl, R. & Reed, L. J. (1920). *On the rate of growth of the population of the United States since 1790.* PNAS 6(6):275–288. https://doi.org/10.1073/pnas.6.6.275
- Lotka, A. J. (1925). *Elements of Physical Biology.* Williams & Wilkins.
- Volterra, V. (1926). *Variazioni e fluttuazioni del numero d'individui in specie animali conviventi.*
- Hewitt, C. G. (1921). *The Conservation of the Wild Life of Canada* (lynx–hare pelt records).
- van der Pol, B. (1926). *On relaxation-oscillations.* Phil. Mag. 2(11):978–992. https://doi.org/10.1080/14786442608564127
- Michaelis, L. & Menten, M. L. (1913). *Die Kinetik der Invertinwirkung.* Biochem. Z. 49:333–369.
- Briggs, G. E. & Haldane, J. B. S. (1925). *A note on the kinetics of enzyme action.* Biochem. J. 19(2):338–339.
- "Influenza in a boarding school." (1978). *British Medical Journal* 1:587 — the 1978 outbreak dataset.
- Kermack, W. O. & McKendrick, A. G. (1927). *A contribution to the mathematical theory of epidemics.* Proc. R. Soc. A 115:700–721.
- Ji, W., Qiu, W., Shi, Z., Pan, S. & Deng, S. (2021). *Stiff-PINN: Physics-Informed Neural Network for Stiff Chemical Kinetics.* J. Phys. Chem. A 125(36):8098–8106. https://doi.org/10.1021/acs.jpca.1c05102
- Kharazmi, E., Cai, M., Zheng, X., Zhang, Z., Lin, G. & Karniadakis, G. E. (2021). *Identifiability and predictability of integer- and fractional-order epidemiological models using PINNs.* Nature Computational Science 1:744–753. https://doi.org/10.1038/s43588-021-00158-0
- Goldrick, S., Ştefan, A., Lovett, D., Montague, G. & Lennox, B. (2015). *The development of an industrial-scale fed-batch fermentation simulation (IndPenSim).* J. Biotechnol. 193:70–82. https://doi.org/10.1016/j.jbiotec.2014.10.029
- Goldrick, S. et al. (2019). *Modern day monitoring and control challenges outlined on an industrial-scale benchmark fermentation process.* Comput. Chem. Eng. 130:106471.
- Adebar, N. et al. (2025). *Physics-informed neural networks for biopharmaceutical cultivation processes.* Biotechnol. Bioeng. https://doi.org/10.1002/bit.28851
- Alam, M. et al. (2025). *PINN-guided modelling and multiobjective optimization of a mAb production process.* Can. J. Chem. Eng. https://doi.org/10.1002/cjce.25446
- Lu, L., Meng, X., Mao, Z. & Karniadakis, G. E. (2021). *DeepXDE.* SIAM Review 63(1):208–228 (Lorenz/inverse methodology reference).

---

## Setup

```bash
pip install torch numpy scipy matplotlib jupyter
jupyter lab
```

Open any notebook and run top to bottom. Each finishes in seconds to a couple of
minutes on CPU. This module is the ODE-focused companion to the main
**PINN Learning Series** repository.
