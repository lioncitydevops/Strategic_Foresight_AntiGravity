# AETHERIA: Strategic Foresight Engine & Simulation Environment

A Python-based computational strategic foresight framework and interactive simulation environment implementing the complete methodology defined in:
> **"From Narrative to Model: Formalizing, Testing and Simulating Assigned Scenarios with Temporal Knowledge Graphs and Game-Theoretic Agents" (Sept 30, 2026)**

---

## 🌟 Core Architectural Commitments

1. **The owner decides what a scenario means:** The model formally translates each qualitative narrative $\Sigma_n = \langle \mathcal{D}_n, \mathcal{X}_n, \Theta_n, \mathcal{O}_n, \mathcal{C}_n \rangle$ with round-trip prose verification.
2. **Stipulation is not evidence:** Narrative statements about the future never update present beliefs. Premises are imposed as causal interventions $\text{do}(\mathcal{D}_n)$; end states are conditioned on $\mathcal{C}_n$ to perform backcasting.
3. **Findings, not verdicts:** Inconsistencies, hidden assumptions, and irrational deviations go back to decision makers as findings, never automated rejections.
4. **Every approximation reports its error:** LLM sensor extraction precision/recall/F1, PSRO NashConv exploitability lower bound, and KAN surrogate fidelity ($R^2$, action agreement, KL divergence) are explicitly calculated and displayed.
5. **Claims are tested out of sample:** Robustness matrix across scenarios without subjective scenario probabilities.

---

## 🏗️ Architecture & Modules

- **Sparse Temporal Knowledge Graph (TKG)** (`src/core/tkg.py`, `src/core/ontology.py`, `src/core/state.py`):
  - State $s_t = (\mathcal{G}_t, x_t)$ combining sparse weighted facts $(i, k, j, w)$ with continuous exogenous indicators $x_t \in \mathbb{R}^d$.
  - Fact-indexed storage running in $O(|\mathcal{F}_t| \ll N^2 M)$.
  - Typed graph edits for strategic players: tariffs, sanctions, trade pacts, and export controls.
- **Present-State Extraction (LLM Noisy Sensor)** (`src/extraction/observation_model.py`, `src/extraction/llm_sensor.py`):
  - Calibrated observation model:
    $$p(o_t \mid s_t) = \prod_{c \in o_t} p(c \mid s_t) \prod_{(i, k, j) \in \mathcal{C}_t \cap \mathcal{F}_t \text{ unclaimed}} (1 - r_k)$$
  - Enforces verbatim evidence span groundings and ontology schema constraints.
- **Scenario Compiler & 5-Stage Test Suite** (`src/compiler/compiler.py`, `src/compiler/tester.py`):
  - Compiles narratives into formal specification tuples $\Sigma_n = \langle \mathcal{D}_n, \mathcal{X}_n, \Theta_n, \mathcal{O}_n, \mathcal{C}_n \rangle$.
  - 5 Verification tests:
    1. *Consistency*: Constraint satisfiability check.
    2. *Plausibility*: $\rho_n = P(\mathcal{C}_n \mid \text{do}(\mathcal{D}_n), \mathcal{X}_n, \Theta_n)$ via Sequential Importance Sampling.
    3. *Backcast*: Posterior parameter shift $p(\theta, \xi, \psi \mid \text{do}(\mathcal{D}_n), \mathcal{C}_n)$ exposing hidden assumptions.
    4. *Strategic Coherence*: Deviation gain $G_{j,t} = \min_{\theta_j \in \Theta_n} [\max_{a'} Q_j(h_t, a'; \theta_j) - Q_j(h_t, a_{j,t}; \theta_j)]$.
    5. *Causal Claims*: Average Treatment Effect under $\text{do}(B)$ vs $\text{do}(\neg B)$.
- **Belief Tracking Particle Filter** (`src/monitoring/particle_filter.py`, `src/monitoring/signposts.py`):
  - Particle state $z_t = (s_t, \theta, \psi)$ with $K=150$ particles.
  - Bayesian Inverse Reinforcement Learning (BIRL) myopic logit choice update.
  - Effective sample size (ESS) monitoring with Gilks & Berzuini MCMC rejuvenation.
  - Bayes factor scenario tracking and 6–12 month diagnostic signpost triggers.
- **Game-Theoretic Actor Models (POSG, PSRO & QRE)** (`src/games/posg.py`, `src/games/psro.py`, `src/games/qre.py`):
  - Bayesian Partially Observable Stochastic Game for strategic actors (USA, CHN, EU).
  - Policy-Space Response Oracles (PSRO) expanding policy populations with lower-bound exploitability metric $\text{NashConv}(\pi)$.
  - Logit Quantal Response Equilibrium (QRE) tracing principal branch.
  - Interventional cross-actor relevance: $\Delta_{ij} = U_i(\pi) - U_i(\pi_i, \pi_j^{\text{base}}, \pi_{-ij})$.
- **Interpretable Policy Surrogate (KAN)** (`src/surrogate/kan.py`, `src/surrogate/symbolic_snapping.py`, `src/surrogate/concept_layer.py`):
  - Concept layer mapping to $p$ auditable variables + static time-to-horizon $\tau$.
  - Learnable univariate edge activations $\phi(x) = w_b \, \text{silu}(x) + w_s \sum c_m B_m(x)$.
  - $L_1$ edge activation regularization and pruning.
  - Symbolic snapping of surviving edges to closed-form elementary functions $(c, ax+b, ax^2+bx+c, \text{sigmoid}, \exp)$.
  - Fidelity auditing (In-distribution match, Off-distribution match, KL divergence).
- **Strategy Robustness Matrix & Verifiable Narrative Reporting** (`src/evaluation/robustness_matrix.py`, `src/evaluation/narrative_reporter.py`):
  - Compares candidate strategies across scenarios for Median, 10th-percentile Downside, and Minimax Regret.
  - Narrative reports where every single factual sentence strictly cites simulated events `[Sim:T<step>-Act<id>]`.
- **Modern Web Application Dashboard** (`web/index.html`, `web/styles.css`, `web/app.js`):
  - Glassmorphic dark mode UI.
  - Interactive SVG force/circular layout for TKG with timeline scrubber ($t \in [0, 6]$).
  - Scenario compiler workbench & test findings cards.
  - KAN neural-symbolic topology visualizer and explicit algebraic formula cards.
  - Strategy $\times$ Scenario robustness matrix and cited narrative viewer.

---

## 🚀 Getting Started

### 1. Run Verification Suite
```powershell
python -u tests/run_verification.py
```

### 2. Launch FastAPI Server & Web Visualizer
```powershell
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```
Open your browser at:
`http://127.0.0.1:8000/`
