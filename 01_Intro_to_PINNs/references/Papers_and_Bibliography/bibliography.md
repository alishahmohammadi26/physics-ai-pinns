# Annotated Bibliography & Download Links

Links point to publisher pages / arXiv. Papers already provided in this repo's
`references/` folder are marked **[in repo]**. Machine-readable entries are in
[`references.bib`](references.bib); the narrative synthesis is in
[`literature_review_2017_2026.md`](literature_review_2017_2026.md).

## Foundational
- **Raissi, Perdikaris & Karniadakis (2019)** — *Physics-informed neural networks*, J. Comput. Phys. 378:686–707. **[in repo]** — https://doi.org/10.1016/j.jcp.2018.10.045
- Raissi, Perdikaris & Karniadakis (2017) — *Physics Informed Deep Learning (Part I)*, arXiv:1711.10561 — https://arxiv.org/abs/1711.10561
- Raissi, Perdikaris & Karniadakis (2017) — *Physics Informed Deep Learning (Part II)*, arXiv:1711.10566 — https://arxiv.org/abs/1711.10566
- **Karniadakis, Kevrekidis, Lu, Perdikaris, Wang & Yang (2021)** — *Physics-informed machine learning*, Nat. Rev. Phys. 3:422–440. **[in repo]** — https://doi.org/10.1038/s42254-021-00314-5

## Software / libraries
- **Lu, Meng, Mao & Karniadakis (2021)** — *DeepXDE*, SIAM Review 63(1):208–228. **[in repo]** — https://doi.org/10.1137/19M1274067 · code: https://github.com/lululxvi/deepxde

## Reviews & surveys
- **Cai, Mao, Wang, Yin & Karniadakis (2021/2022)** — *PINNs for fluid mechanics: A review*, Acta Mech. Sinica. **[in repo]** — https://doi.org/10.1007/s10409-021-01148-1
- **Cuomo, Di Cola, Giampaolo, Rozza, Raissi & Piccialli (2022)** — *Scientific ML through PINNs: Where we are and what's next*, J. Sci. Comput. 92:88. **[in repo]** — https://doi.org/10.1007/s10915-022-01939-z
- Farea, Yli-Harja & Emmert-Streib (2024) — *Understanding PINNs: techniques, applications, trends, challenges*, AI 5(3):1534–1557 — https://doi.org/10.3390/ai5030074
- Zhao, Zhang, Lou, Wang & Yang (2024) — *Advances in PINNs and applications in complex fluid dynamics*, Phys. Fluids 36(10) — https://doi.org/10.1063/5.0226562
- Meng, Griesemer, Cao, Seo & Liu (2025) — *When physics meets ML: a survey*, ML Comput. Sci. Eng. 1:20 — https://doi.org/10.1007/s44379-025-00016-0
- Toscano et al. (2025) — *From PINNs to PIKANs: recent advances*, ML Comput. Sci. Eng. 1:15 — https://doi.org/10.1007/s44379-025-00015-1
- Fan & Chen (2026) — *Embedding Physics into ML: PINNs as PDE forward solvers*, Tsinghua Sci. Tech. 31(3):1326–1364 — https://doi.org/10.26599/TST.2025.9010157

## Training, optimisation & failure modes
- Krishnapriyan, Gholami, Zhe, Kirby & Mahoney (2021) — *Characterizing possible failure modes in PINNs*, NeurIPS — https://arxiv.org/abs/2109.01050
- Wang, Yu & Perdikaris (2022) — *When and why PINNs fail to train: an NTK perspective*, J. Comput. Phys. 449:110768 — https://doi.org/10.1016/j.jcp.2021.110768
- Wang, Sankaran & Perdikaris (2022) — *Respecting causality is all you need for training PINNs*, arXiv:2203.07404 — https://arxiv.org/abs/2203.07404
- Wang, Sankaran, Wang & Perdikaris (2023) — *An expert's guide to training PINNs*, CMAME / arXiv:2308.08468 — https://arxiv.org/abs/2308.08468
- Tancik et al. (2020) — *Fourier features let networks learn high-frequency functions*, NeurIPS — https://arxiv.org/abs/2006.10739
- Wang, Wang & Perdikaris (2021) — *On the eigenvector bias of Fourier feature networks*, CMAME 384:113938 — https://doi.org/10.1016/j.cma.2021.113938
- **Wang, Li, Chen & Perdikaris (2024)** — *PirateNets: physics-informed deep learning with residual adaptive networks*, arXiv:2402.00326. **[in repo]** — https://arxiv.org/abs/2402.00326

## Domain decomposition
- Jagtap & Karniadakis (2020) — *Extended PINNs (XPINN)*, Commun. Comput. Phys. 28(5):2002–2041 — https://doi.org/10.4208/cicp.OA-2020-0164
- Jagtap, Kharazmi & Karniadakis (2020) — *Conservative PINNs (cPINN)*, CMAME 365:113028 — https://doi.org/10.1016/j.cma.2020.113028

## Adaptive sampling
- Wu, Zhu, Tan, Kartha & Lu (2023) — *Non-adaptive and residual-based adaptive sampling for PINNs*, CMAME 403:115671 — https://doi.org/10.1016/j.cma.2022.115671

## Stiff systems / kinetics
- Ji, Qiu, Shi, Pan & Deng (2021) — *Stiff-PINN for stiff chemical kinetics*, J. Phys. Chem. A 125(36):8098–8106 — https://doi.org/10.1021/acs.jpca.1c05102

## Applications (fluids / hidden physics)
- Raissi, Yazdani & Karniadakis (2020) — *Hidden fluid mechanics*, Science 367(6481):1026–1030 — https://doi.org/10.1126/science.aaw4741
- Mao, Jagtap & Karniadakis (2020) — *PINNs for high-speed flows*, CMAME 360:112789 — https://doi.org/10.1016/j.cma.2019.112789

## Neural operators
- Lu, Jin, Pang, Zhang & Karniadakis (2021) — *DeepONet*, Nat. Mach. Intell. 3:218–229 — https://doi.org/10.1038/s42256-021-00302-5
- Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart & Anandkumar (2021) — *Fourier Neural Operator*, ICLR / arXiv:2010.08895 — https://arxiv.org/abs/2010.08895
- Wang, Wang & Perdikaris (2021) — *Physics-informed DeepONets*, Sci. Adv. 7(40):eabi8605 — https://doi.org/10.1126/sciadv.abi8605
- Cao, Goswami & Karniadakis (2024) — *Laplace Neural Operator*, Nat. Mach. Intell. 6:631–640 — https://doi.org/10.1038/s42256-024-00844-4

## Kolmogorov–Arnold Networks / PIKANs
- Liu, Wang, Vaidya, Ruehle, Halverson, Soljačić, Hou & Tegmark (2024) — *KAN: Kolmogorov-Arnold Networks*, arXiv:2404.19756 — https://arxiv.org/abs/2404.19756
- Liu, Ma, Wang, Matusik & Tegmark (2024) — *KAN 2.0: KANs meet science*, arXiv:2408.10205 — https://arxiv.org/abs/2408.10205
- Wang, Sun, Bai, Anitescu, Eshaghi, Zhuang, Rabczuk & Liu (2025) — *Kolmogorov-Arnold-Informed neural network (PIKAN)*, CMAME 433:117518 — https://doi.org/10.1016/j.cma.2024.117518
- Shukla, Toscano, Wang, Zou & Karniadakis (2024) — *A comprehensive and FAIR comparison between MLP and KAN*, CMAME 431:117290 — https://doi.org/10.1016/j.cma.2024.117290

## Datasets referenced by notebooks
- **Cylinder wake** (`cylinder_nektar_wake.mat`) and **Burgers reference** (`burgers_shock.mat`) — from the original PINN repository — https://github.com/maziarraissi/PINNs
- Taylor–Green vortex (notebook 10) is generated analytically in-notebook; no download needed.
