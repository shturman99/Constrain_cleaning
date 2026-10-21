# Literature review: ML for constraint control in PDE solvers

First pass, 2026-10-15. Organized by the questions in `main.tex`.

Legend: **[Z]** already in Zotero (tag `constraint cleaning`) · **[S]** found via web search; link, title and authors verified against arXiv, but only the abstract read · unmarked = standard reference, not yet in Zotero. All links were resolved through Crossref (DOIs) or the arXiv API, not written from memory.

---

## 1. Classical constraint handling (the baselines)

### Divergence-free B in MHD
- Brackbill & Barnes (1980), [*The effect of nonzero ∇·B on the numerical solution of the MHD equations*](https://doi.org/10.1016/0021-9991%2880%2990079-0), *JCP* 35, 426: why ∇·B ≠ 0 matters (spurious force along B). The motivation paper.
- Evans & Hawley (1988), [*Simulation of MHD flows: a constrained transport method*](https://doi.org/10.1086/166684), *ApJ* 332, 659: constrained transport (CT) on a staggered grid.
- Powell et al. (1999), [*A solution-adaptive upwind scheme for ideal MHD*](https://doi.org/10.1006/jcph.1999.6299), *JCP* 154, 284: 8-wave formulation; advects the error instead of removing it.
- Tóth (2000), [*The ∇·B = 0 constraint in shock-capturing MHD codes*](https://doi.org/10.1006/jcph.2000.6519), *JCP* 161, 605: systematic comparison of ∇·B schemes (CT variants, projection, 8-wave). **The template for our benchmark design.**
- Dedner et al. (2002), [*Hyperbolic divergence cleaning for the MHD equations*](https://doi.org/10.1006/jcph.2001.6961), *JCP* 175, 645: hyperbolic/parabolic (GLM) cleaning. Baseline named in the project plan.
- **[Z]** Balsara (2001), [*Divergence-free adaptive mesh refinement for MHD*](https://doi.org/10.1006/jcph.2001.6917), *JCP*: divergence-free reconstruction for AMR.
- Pencil Code Collaboration (2021), [*The Pencil Code*](https://doi.org/10.21105/joss.02807), *JOSS* 6, 2807: evolves A, so ∇·B = 0 by construction. Relevant for the Stage 3 comparison.

### Incompressibility
- Chorin (1968), [*Numerical solution of the Navier–Stokes equations*](https://doi.org/10.1090/s0025-5718-1968-0242392-2), *Math. Comp.* 22, 745: projection method. What our Leray layer is, in discrete form.

### Constraints in numerical relativity
- Gundlach, Calabrese, Hinder & Martín-García (2005), [*Constraint damping in the Z4 formulation and harmonic gauge*](https://doi.org/10.1088/0264-9381/22/17/025), *CQG* 22, 3767. **The classical version of approach (2).**
- Pretorius (2005), [*Evolution of binary black-hole spacetimes*](https://doi.org/10.1103/PhysRevLett.95.121101), *PRL* 95, 121101: generalized harmonic + damping; first stable binary black hole evolution.
- Alic et al. (2012), [*Conformal and covariant formulation of the Z4 system with constraint-violation damping*](https://doi.org/10.1103/PhysRevD.85.064040) (CCZ4), *PRD* 85, 064040 ([arXiv:1106.2254](https://arxiv.org/abs/1106.2254)).
- Weyhausen, Bernuzzi & Hilditch (2012), [*Constraint damping for the Z4c formulation of general relativity*](https://doi.org/10.1103/PhysRevD.85.024038), *PRD* 85, 024038 ([arXiv:1107.5539](https://arxiv.org/abs/1107.5539)). Damping parameters are hand-tuned and can go unstable: exactly the gap ML could fill.

### Gauge theories
- Kogut & Susskind (1975), [*Hamiltonian formulation of Wilson's lattice gauge theories*](https://doi.org/10.1103/PhysRevD.11.395), *PRD* 11, 395: the Gauss law as a constraint on states.
- Ipp & Müller (2018), [*Implicit schemes for real-time lattice gauge theory*](https://doi.org/10.1140/epjc/s10052-018-6323-x), *EPJC* 78: Gauss constraint preserved even for large time steps. Classical baseline for Stage 2.

### Structure-preserving discretization (general theory)
- **[Z]** Arnold, Falk & Winther (2006), [*Finite element exterior calculus, homological techniques, and applications*](https://doi.org/10.1017/S0962492906210018), *Acta Numerica*. The mathematical framework behind "constraint-preserving parameterizations" (approach 3).

---

## 2. Neural surrogates for PDEs

- Li et al. (2020), [*Fourier neural operator for parametric PDEs*](https://arxiv.org/abs/2010.08895). Our current surrogate.
- Lu et al. (2021), [*Learning nonlinear operators via DeepONet*](https://doi.org/10.1038/s42256-021-00302-5), *Nat. Mach. Intell.* 3, 218.
- Li et al. (2021), [*Physics-informed neural operator for learning PDEs*](https://arxiv.org/abs/2111.03794) (PINO).
- Li et al. (2022), [*Learning chaotic dynamics in dissipative systems*](https://arxiv.org/abs/2106.06898), NeurIPS: source of the Kolmogorov datasets; argues for statistical, not pointwise, evaluation of long rollouts.
- **[Z]** Raissi, Perdikaris & Karniadakis (2019), [*Physics-informed neural networks*](https://doi.org/10.1016/j.jcp.2018.10.045), *JCP* 378.
- **[Z]** Cai et al. (2021), [*PINNs for fluid mechanics: a review*](https://doi.org/10.1007/s10409-021-01148-1), *Acta Mech. Sin.*
- Brandstetter, Worrall & Welling (2022), [*Message passing neural PDE solvers*](https://arxiv.org/abs/2202.03376), ICLR: pushforward trick for rollout stability.
- Lippe et al. (2023), [*PDE-Refiner: accurate long rollouts with neural PDE solvers*](https://arxiv.org/abs/2308.05732), NeurIPS: long-rollout stability via spectral refinement.
- **[Z]** Ohana, McCabe et al. (2024), [*The Well*](https://arxiv.org/abs/2412.00568): large simulation collection, includes MHD datasets. Possible ready-made training data.
- Takamoto et al. (2022), [*PDEBench*](https://arxiv.org/abs/2210.07182), NeurIPS Datasets & Benchmarks.

### MHD surrogates specifically
- **[Z]** Rosofsky & Huerta (2023), [*MHD with physics-informed neural operators*](https://doi.org/10.1088/2632-2153/ace30a), *MLST* ([arXiv:2302.08332](https://arxiv.org/abs/2302.08332)): 2D incompressible MHD; ∇·B as a soft loss, and an alternative that predicts A.
- [S] Kim et al. (2024), [*Neural operators learn the local physics of MHD*](https://arxiv.org/abs/2404.16015), *Comput. Fluids* (2025): flux neural operator; ∇·B penalty in the loss.
- [S] Duarte et al. (2025), [*Spectral learning of magnetized plasma dynamics: a neural operator application*](https://arxiv.org/abs/2507.01388): FNO on Orszag–Tang vortices; no constraint enforcement.
- [S] Kacmaz et al. (2025), [*Resolving turbulent MHD: a hybrid operator–diffusion framework*](https://arxiv.org/abs/2507.02106): PINO + diffusion corrector for 2D resistive MHD turbulence; predicts A; recovers fine scales.
- **[Z]** Huerta (2026), [*Magnetohydrodynamics simulations*](https://arxiv.org/abs/2605.19057), review chapter: useful overview of where the field is.

---

## 3. Hard constraint enforcement in networks — approach (1)

- Jiang et al. (2020), [*Enforcing physical constraints in CNNs through differentiable PDE layer*](http://cfd.me.berkeley.edu/wp-content/uploads/2022/02/enforcing_physical_constraints.pdf), ICLR DeepDiffEq workshop: spectral projection layer for divergence-free outputs. First version of what we implemented.
- **[Z]** Duruisseaux, Liu-Schiaffini, Berner & Anandkumar (2024), [*Towards enforcing hard physics constraints in operator learning frameworks*](https://openreview.net/forum?id=Zvxm14Rd1F), ICML AI4Science workshop: projection on neural operator outputs; reproduced in `experiments/hard_constraint_fno.py`.
- Négiar, Mahoney & Krishnapriyan (2023), [*Learning differentiable solvers for systems with hard constraints*](https://arxiv.org/abs/2207.08675), ICLR.
- Chalapathi, Du & Krishnapriyan (2024), [*Scaling physics-informed hard constraints with mixture-of-experts*](https://arxiv.org/abs/2402.13412), ICLR.
- **[Z]** Hansen et al. (2024), [*Learning physical models that can respect conservation laws*](https://doi.org/10.1016/j.physd.2023.133952) (ProbConserv), *Physica D* 457 ([arXiv:2302.11002](https://arxiv.org/abs/2302.11002)): conservation laws via a Bayesian update.
- Harder et al. (2022), [*Hard-constrained deep learning for climate downscaling*](https://arxiv.org/abs/2208.05424): constraint layers for mass/energy conservation.
- Min & Azizan (2024), [*HardNet: hard-constrained neural networks with universal approximation guarantees*](https://arxiv.org/abs/2410.10807): differentiable projection layer.
- Chen et al. (2024), [*PINNs with hard linear equality constraints*](https://doi.org/10.1016/j.compchemeng.2024.108764) (KKT-hPINN), *Comput. Chem. Eng.* 189: KKT-derived projection.
- [S] Singha (2025), [*Learning under laws: a constraint-projected neural PDE solver*](https://arxiv.org/abs/2511.03578): projection onto conservation, entropy and positivity sets for Burgers/Euler.
- [S] Jiang et al. (2026), [*Error-conditioned neural solvers*](https://arxiv.org/abs/2606.27354): residual-conditioned corrector with hard projection at every step. **Closest to approach (1); read first.**

## 4. Learned corrections inside solvers — approach (2)

- Bar-Sinai et al. (2019), [*Learning data-driven discretizations for PDEs*](https://doi.org/10.1073/pnas.1814058116), *PNAS* 116, 15344.
- Um et al. (2020), [*Solver-in-the-Loop*](https://arxiv.org/abs/2007.00016), NeurIPS: training through the solver is what makes corrections stable over long rollouts. **Core method for approach (2).**
- Kochkov et al. (2021), [*Machine learning–accelerated computational fluid dynamics*](https://doi.org/10.1073/pnas.2101784118), *PNAS* 118 ([arXiv:2102.01010](https://arxiv.org/abs/2102.01010)): network inside a differentiable JAX solver whose pressure projection keeps incompressibility exact.
- List, Chen & Thuerey (2022), [*Learned turbulence modelling with differentiable fluid solvers*](https://doi.org/10.1017/jfm.2022.738), *JFM* 949.
- [S] Wang et al. (2025), [*Accelerating a high-order energy-stable DG solver using auto-differentiation*](https://arxiv.org/abs/2512.09128): solver-in-the-loop for discontinuous Galerkin.

## 5. Constraint-preserving parameterizations — approach (3)

- Mohan et al. (2023), [*Embedding hard physical constraints in NN coarse-graining of 3D turbulence*](https://doi.org/10.1103/PhysRevFluids.8.014604), *PRFluids* 8, 014604: u = ∇×v.
- Richter-Powell, Lipman & Chen (2022), [*Neural conservation laws: a divergence-free perspective*](https://arxiv.org/abs/2210.01741), NeurIPS: divergence-free networks via differential forms.
- Xing et al. (2024), [*HelmFluid: learning Helmholtz dynamics*](https://arxiv.org/abs/2310.10565), ICML: learns Helmholtz potentials.
- Greydanus et al. (2019), [*Hamiltonian neural networks*](https://arxiv.org/abs/1906.01563); Cranmer et al. (2020), [*Lagrangian neural networks*](https://arxiv.org/abs/2003.04630): structure by construction for ODE-level constraints.
- [S] Shikhman et al. (2026), [*Cellular sheaf neural operators for structure-preserving surrogate modeling of constrained PDEs*](https://arxiv.org/abs/2606.00937): includes MHD divergence control.
- [S] He et al. (2026), [*Divergence-free linearized neural networks: integral representation and optimal approximation rates*](https://arxiv.org/abs/2603.28638).
- Related, conservation-law discovery (relevant if approach (3) needs to *find* the invariant structure of an unfamiliar system):
  - **[Z]** Liu & Tegmark (2021), [*Machine learning conservation laws from trajectories*](https://doi.org/10.1103/PhysRevLett.126.180604), *PRL* 126.
  - **[Z]** Zhu et al. (2023), [*ML of independent conservation laws through neural deflation*](https://doi.org/10.1103/PhysRevE.108.L022301), *PRE* 108.
  - **[Z]** Chen et al. (2025), [*Data-driven discovery of conservation laws from trajectories via neural deflation*](https://doi.org/10.1016/j.cnsns.2024.108563), *CNSNS*.

## 6. Nonlinear constraints: gauge theories and gravity (Stage 2)

### Lattice gauge
- Kanwar et al. (2020), [*Equivariant flow-based sampling for lattice gauge theory*](https://doi.org/10.1103/PhysRevLett.125.121601), *PRL* 125.
- Favoni et al. (2022), [*Lattice gauge equivariant CNNs*](https://doi.org/10.1103/PhysRevLett.128.032003) (L-CNN), *PRL* 128 ([arXiv:2012.12901](https://arxiv.org/abs/2012.12901)); Favoni et al. (2022), [*Applications of lattice gauge equivariant NNs*](https://arxiv.org/abs/2212.00832).
- Luo et al. (2021), [*Gauge equivariant neural networks for quantum lattice gauge theories*](https://doi.org/10.1103/PhysRevLett.127.276402), *PRL* 127, 276402: Gauss law as gauge invariance of the wavefunction.
- [S] Rayat et al. (2026), [*Gauge-equivariant graph neural networks for lattice gauge theories*](https://arxiv.org/abs/2604.20797): learned forces fed into real-time evolution in 3+1D. **Closest existing work to Stage 2; read first.**

### Numerical relativity
- [S] Zhou et al. (2026), [*Solving the Hamiltonian constraint equation with PINNs*](https://arxiv.org/abs/2607.06002): binary black hole initial data.
- [S] Shukla et al. (2026), [*Puncture evolution with PINNs*](https://arxiv.org/abs/2610.06776): Z4c variables in spherical symmetry.
- [S] Bayona et al. (2026), [*Solving Einstein's vacuum equations with PINNs*](https://arxiv.org/abs/2608.08846): boundary conditions and domain decomposition.
- [S] Carretero et al. (2026), [*Towards solving general relativity with PINNs*](https://arxiv.org/abs/2609.38331): Einstein–Klein–Gordon boson stars.
- [S] Cranganore et al. (2025), [*Einstein Fields: a neural perspective to computational GR*](https://arxiv.org/abs/2507.11589).

## 7. Evaluation practice

- McGreivy & Hakim (2024), [*Weak baselines and reporting biases lead to overoptimism in ML for fluid-related PDEs*](https://doi.org/10.1038/s42256-024-00897-5), *Nat. Mach. Intell.* 6, 1256. **Read before finalizing the experimental design.**
- **[Z]** Taira, Rigas & Fukami (2025), [*Machine learning in fluid dynamics: a critical assessment*](https://doi.org/10.1103/8t52-mtb9), *PRFluids* 10 ([arXiv:2508.13430](https://arxiv.org/abs/2508.13430)).

---

## Gaps (where the project could be new)

1. **Learned constraint damping is unexplored.** Searches for ML + constraint damping in numerical relativity or MHD found nothing; classical damping is hand-tuned and fragile (Z4c instabilities). Approach (2) looks open.
2. **MHD surrogates rarely use hard constraints and never compare against classical cleaning.** Existing work uses soft ∇·B penalties or predicts A. No paper found benchmarks a learned method against projection, Dedner and CT inside a time loop, as Tóth (2000) did for classical schemes.
3. **Constraint satisfied ≠ correct.** Hard-projection papers report constraint error plus short-horizon L2. None found asks which correction best recovers the true solution over long rollouts. Our 2D run already shows that projection removes only about 3% of the error.
4. **Nonlinear constraints in real-time evolution.** Gauge-equivariant networks enforce symmetry, not Gauss-law preservation under learned dynamics; NR + ML is almost entirely PINNs for initial data or toy evolutions. Wide open, and higher risk.

## Next to read (priority order)

1. [Tóth (2000)](https://doi.org/10.1006/jcph.2000.6519): benchmark design template.
2. [McGreivy & Hakim (2024)](https://doi.org/10.1038/s42256-024-00897-5): baseline and reporting pitfalls.
3. [Um et al. (2020)](https://arxiv.org/abs/2007.00016) + [Kochkov et al. (2021)](https://doi.org/10.1073/pnas.2101784118): solver-in-the-loop training.
4. [Jiang et al. (2026), Error-conditioned neural solvers](https://arxiv.org/abs/2606.27354): closest competitor to approach (1).
5. [Gundlach et al. (2005)](https://doi.org/10.1088/0264-9381/22/17/025): what a learned damping term would generalize.
6. [Rayat et al. (2026)](https://arxiv.org/abs/2604.20797): closest competitor for Stage 2.
