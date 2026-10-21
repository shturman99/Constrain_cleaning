---
marp: true
math: katex
paginate: true
style: |
  section { font-size: 26px; }
  section.small { font-size: 22px; }
  h1 { font-size: 40px; }
  table { font-size: 22px; margin: 0 auto; }
---

<!-- _paginate: false -->

# Solving PDEs with Constraints

### Can machine learning control constraint violation inside a solver's time loop?

Murman Gurgenidze

---

# The problem: constraints drift

Many evolved fields must also satisfy a constraint at all times:

| System | Evolution | Constraint |
|---|---|---|
| Incompressible flow | $\partial_t u + (u\cdot\nabla)u = -\nabla p/\rho + \nu\nabla^2 u + F$ | $\nabla\cdot u = 0$ |
| MHD / Maxwell | induction equation | $\nabla\cdot B = 0,\ \nabla\cdot E = \rho_e$ |
| Yang–Mills | gauge-field evolution | Gauss law (nonlinear) |
| Numerical relativity | 3+1 evolution | Hamiltonian & momentum (nonlinear) |

- The continuum dynamics preserve the constraint; **discretization error does not**
- Violations feed back: unphysical forces, unstable growing modes

---

# Classical remedies, and where they stop working

| Method | Idea |
|---|---|
| Constrained transport | discretize so $\nabla\cdot B = 0$ holds exactly |
| Projection | remove the violating part (Poisson solve / Leray) |
| Hyperbolic (Dedner) cleaning | propagate and damp the error away |
| Constraint damping | add terms that drive violation to zero |

- Work well for **linear** constraints with **known** structure
- Expensive, hand-tuned, or unavailable for **nonlinear** constraints and for field systems with no known constraint-preserving formulation

---

# Three ML approaches to compare

1. **Neural corrector + hard enforcement**
   network corrects the state, then an exact projection guarantees the constraint
2. **Learned constraint damping**
   damping terms trained end-to-end through a differentiable solver
3. **Learned constraint-preserving parameterizations**
   generalize the vector-potential idea ($B = \nabla\times A$) to systems where no potential is known

**Goal:** corrections that recover the *true* solution, not just ones that satisfy the constraint

---

<!-- _class: small -->

# Starting point: hard projection on a neural operator

Reproduced Duruisseaux et al. (2024): FNO time-stepper for 2D Kolmogorov flow (Re = 500, 64²), output projected in Fourier space

$$\hat u(k) \;\leftarrow\; \hat u(k) - k\,\frac{k\cdot\hat u(k)}{|k|^2}$$

| Method | Test rel. $L^2$ (×10⁻²) | RMS $\nabla\cdot u$ |
|---|---|---|
| Baseline FNO | 3.30 | $1.6\times10^{-1}$ |
| + projection at inference | 3.22 | $5.5\times10^{-6}$ |
| + projection fine-tuning | 3.26 | $5.5\times10^{-6}$ |
| trained with projection | **3.19** | $5.5\times10^{-6}$ |

Constraint fixed for free, but **~97% of the error is divergence-free**: projection cannot touch it

---

# Plan

**Stage 1: 2D MHD benchmark**
- Compare against exact projection and Dedner cleaning
- Measure constraint violation *and* accuracy of the physical fields, including long rollouts

**Stage 2: nonlinear constraints**
- Lattice Yang–Mills, simplified 3+1 gravity: no closed-form projector, where ML can offer a real advantage

**Stage 3: 3D decaying MHD**
- Validate against Pencil Code runs: energy decay, spectra, magnetic helicity

**Open:** experimental design, metrics for "true solution" vs "constraint satisfied"
