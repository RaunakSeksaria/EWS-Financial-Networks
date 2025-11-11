## Project Context (High-Level, Code-Agnostic)

### Goal
- Build a clean, from-scratch, reproducible notebook that simulates a toy financial network, sweeps a control parameter (“market coupling” ε), and visualizes the system’s transition (tipping) from a healthy state to a crashed state.
- Show two quantities on a single plot across ε:
  - Mean Health (system state, “blue line”)
  - Stability indicator Re(λ) of the system Jacobian (dominant eigenvalue, “green line”)
- Mark the tipping point (where Re(λ) crosses 0) and ensure the curves qualitatively match the patterns seen in recent tipping-point literature.

### Core Idea (What we’re modeling)
- A network of N firms with intrinsic recovery/growth (logistic-like) and pairwise coupling that transmits stress/shocks.
- The control parameter ε increases the strength of network coupling (“market stress”).
- For small ε, the network is healthy and stable; as ε increases, stability degrades until a critical transition occurs (tipping).

### Mathematical Model (Conceptual)
- State vector x (firm “health”/activity).
- Time evolution is an ODE: intrinsic dynamics (recovery/growth) minus network coupling.
- Network is a sparse scale-free-like graph (e.g., Barabási–Albert) used only to define adjacency/coupling structure.
- Stability is assessed via the Jacobian at the converged state; the dominant eigenvalue’s real part Re(λ) indicates local stability (negative = stable, zero = critical, positive = unstable).
- We sweep ε, numerically relax to a steady state per ε, then record:
  - Mean Health: average of the components of x
  - Stability: Re(λ) at that state

### Visualization (Conceptual)
- Single figure with two y-axes over the same x-axis (ε):
  - Left axis: Mean Health (blue)
  - Right axis: Re(λ) (green)
  - Vertical dashed line at the tipping point εc (Re(λ) = 0)

### What “Correct” Behavior Should Look Like
- As ε increases:
  - Mean Health starts high (healthy), gently drifts downward, then sharply declines near εc.
  - Re(λ) starts negative (stable), rises toward 0, crosses 0 at εc, and becomes positive (unstable).
- The ε at which Re(λ) crosses 0 is the local instability (tipping) threshold.

### References (Conceptual Background)
- Early predictor and tipping point literature:
  - Physical Review X (2024), “Early Predictor for the Onset of Critical Transitions in Networked Dynamical Systems.”  
    DOI: 10.1103/PhysRevX.14.031009  
    Link: https://journals.aps.org/prx/abstract/10.1103/PhysRevX.14.031009
- General critical transitions, local stability, and Jacobians in dynamical systems (standard texts).

### Assumptions and Simplifications
- The model is intentionally “toy”/phenomenological: it captures qualitative behavior (tipping) rather than fitting real financial series.
- We treat “health” as a nonnegative state; feasibility handling can be done either via constrained integration or post-processing (implementation choice).
- The stability indicator is local (linearization at the steady state); it aims to match the literature’s qualitative signals, not to predict real markets.

### Non-Goals
- No real data ingestion, calibration, or forecasting.
- No risk/portfolio optimization.
- No attempt to estimate real-world ε from market data.

### Deliverable Shape
- A single Jupyter notebook that:
  - Defines the network and ODE (at a high level as above).
  - Sweeps ε, relaxes to steady states, computes Mean Health and Re(λ).
  - Produces the twin-axis visualization with a clearly marked tipping point.
- Clear, minimal markdown cells explaining the story (no excessive code commentary).

### Quality/Validation Heuristics (Qualitative)
- Blue line (Mean Health) high → slow decline → sharp drop near εc.
- Green line (Re(λ)) negative → rises to 0 at εc → positive after εc.
- Tipping point visually and numerically aligned (Re(λ) ≈ 0 where the blue curve breaks).

### Implementation Notes for Whoever Builds It (No Code, Just Guidance)
- Prefer a sparse, scale-free-like network generator for realism and variability.
- Use a robust ODE integrator with steady-state relaxation per ε; warm-start each ε with the previous state.
- Compute the Jacobian at the relaxed state and take the dominant eigenvalue’s real part for stability.
- Keep reproducibility via fixed seeds and a simple parameter block (N, m, r, K, ε range/steps).

### Model Equations (Exact)
- State: x ∈ ℝ^N, where x_i ≥ 0 represents firm i’s health.
- Adjacency: A ∈ ℝ^{N×N} (e.g., unweighted Barabási–Albert graph; A_ij ≥ 0).
- Control parameter (market coupling): ε ∈ ℝ_+, swept over a range (e.g., ε ∈ [0, 0.1]).

Dynamics (ODE):
- dx/dt = r x ∘ (1 − x/K) − ε A x
  - r > 0: intrinsic growth/recovery rate (scalar).
  - K > 0: carrying capacity (scalar).
  - “∘” denotes elementwise product.
  - (A x)_i = ∑_j A_ij x_j.

Equilibrium and Metrics:
- Mean Health:  H(x) = (1/N) ∑_{i=1}^N x_i.
- Jacobian at equilibrium x:
  - J(x; ε) = diag(r (1 − 2 x / K)) − ε A.
- Stability indicator (dominant eigenvalue real part):
  - Re(λ_max(J(x; ε))).
- Tipping point:
  - The ε_c where Re(λ_max(J(x; ε_c))) = 0 (loss of local stability).


