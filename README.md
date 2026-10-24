# Cost-Weighted Correlated Multi-Armed Bandits

Work on cost-weighted, correlated fixed-confidence best-arm identification, done under Prof. Jayakrishnan Nair and Prof. N. Hemachandra (IIT Bombay).

The full write-up (lower bound, stopping rule, δ-PAC proof, asymptotic optimality and experiments) is in `Compilation_Document.pdf`.

## Problem

Two Gaussian arms with unit variance. At each round the learner chooses one of three actions:

| Action | Cost | Observation |
|---|---|---|
| Pull arm 1 | `c` | `X1 ~ N(mu1, 1)` |
| Pull arm 2 | `c` | `X2 ~ N(mu2, 1)` |
| Joint pull | `c12` | `(U, V) ~ N((mu1, mu2), [[1, rho], [rho, 1]])` |

The correlation `rho` is unknown. The goal is to identify the arm with the larger mean with error probability at most `delta`, using as little total cost as possible.

The lower bound on the expected cost is `kl(delta, 1 - delta) / I*(theta)`. For `c1 = c2 = c`, the optimal allocation is one of two vertices:

- `v_J = (0, 0, 1/c12)`: joint pulls only;
- `v_I = (1/(2c), 1/(2c), 0)`: alternating individual pulls.

The switch between them happens at the cost threshold `c12_th(delta_gap, rho) = 4 c K / delta_gap^2`, where `delta_gap = mu1 - mu2`.

The algorithm is a cost-weighted Track-and-Stop. At each round:

1. Compute the MLE `(mu1_hat, mu2_hat, rho_hat)` and the GLR statistic `Z_t`.
2. Stop when `Z_t > beta(t, delta)`.
3. Otherwise, make a joint pull if `N12(t) < sqrt(t + 1)` (forced exploration).
4. Otherwise, pull according to whichever vertex is better at the MLE. For `v_I`, alternate between arm 1 and arm 2.

## Files

| File | What it does | Writes |
|---|---|---|
| `Compilation_Document.pdf` | The paper. | — |
| `beta_threshold_numeric.m` | MATLAB: numerical study of the stopping threshold. Estimates `R_t = sup L_t / (mixture of L_t)` and `beta(t, delta) = log(R_t / delta)` for a uniform prior on the bounded set `[-B, B]^2 x [-1 + eps0, 1 - eps0]`. Part 1 simulates data from a fixed environment and fits the growth of `log R_t` in `log t`, comparing it with `1.5 log t` and `2 log t`. Part 2 uses constructed data `U = V = A` to show how `R_t` grows with `A` at fixed `t`. | Printed tables and one 6-panel figure |
| `compare_baselines.py` | Our algorithm against GK Track-and-Stop, KL-LUCB, Top-Two Thompson Sampling, Successive Elimination, and "joint pulls, `rho` ignored", in Regimes A–D (Experiment 3). | `baselines_results.csv`, `baselines_table.txt`, `fig_baselines.png` |
| `compare_c12_sweep.py` | Sweeps `c12` from 2 to past the threshold on `theta = (0.1, 0, 0.8)` and `(0.2, 0, 0.5)` (Experiment 2). Imports `compare_baselines.py`. | `c12_sweep_results.csv`, `c12_sweep_table.txt`, `fig_c12_sweep.png` |
| `plot_results.py` | Figures for Experiments 2 and 3 in absolute cost, as used in the paper. Reads the two CSV files above. | `fig_baselines_abs.{png,pdf}`, `fig_c12_sweep_abs.{png,pdf}` |
| `lower_bound_plots.py` | Lower-bound plots: solves the cubic `C(rho') = 0` for `rho'*`, computes `c12_th`, and plots the optimal allocation against `c12`, the threshold against the gap and against `rho`, and 3-D surfaces of `c12_th(delta_gap, rho)` and `rho'*(delta_gap, rho)`. | `fig_allocation_vs_c12`, `fig_threshold_slices`, `fig_c12_threshold_3d`, `fig_rho_star_3d` (`.png` and `.pdf`) |

## Requirements

Python 3.9 or later:

```bash
pip install numpy matplotlib
```

MATLAB for `beta_threshold_numeric.m` (no toolboxes needed).

## Running

```bash
# Experiment 3 (Regimes A-D): 1000 runs, delta = 0.1 and 0.01, all CPU cores
python compare_baselines.py
python compare_baselines.py --runs 300 --deltas 0.1 0.01 0.001 --workers 8   # other settings

# Experiment 2 (c12 sweep), same options
python compare_c12_sweep.py

# Paper figures for Experiments 2 and 3 (run the two scripts above first)
python plot_results.py

# Lower-bound plots
python lower_bound_plots.py
```

In MATLAB, run `beta_threshold_numeric` from the repo folder. The settings at the top of the file are `theta = (1, 0, 0.5)`, `B = 3`, `eps0 = 0.1`, `delta = 0.05`, horizon 80 and 150 runs. Its sampling rule forces joint pulls while `N12 < sqrt(t)` and otherwise alternates the two arms. It does not use the vertex rule, because it only studies the threshold.

Note that `beta_threshold_numeric.m` uses a uniform prior on a bounded parameter set. The threshold proved in the paper instead uses Cauchy priors on the means and a uniform prior on `rho`, with no bound on the means.

## Settings used in the paper

- **Costs:** individual pull cost `c = 1` everywhere.
- **Stopping threshold:** every algorithm uses `beta(t, delta) = log((log t + 1) / delta)`, the rate Garivier and Kaufmann (2016) use in their experiments. The paper proves δ-PAC for the larger threshold `log(1/delta) + (3/2) log t + C_t`. The experiments use the smaller rate, so that every algorithm is compared on the same footing.
- **Forced exploration:** only joint pulls are forced, `N12(t) < sqrt(t + 1)`. A second version uses `(t + 1)^(1/3)`. This matches Algorithm 1 and the proof in the paper.
- **Regimes A–D:**

  | Regime | `(mu1, mu2, rho)` | `c12` |
  |---|---|---|
  | A | (0.1, 0, 0.8) | 2 |
  | B | (0.2, 0, -0.5) | 2 |
  | C | (0.2, 0, 0) | 1.5 |
  | D | (0.2, 0, -0.5) | 1.5 |

## References

- A. Garivier and E. Kaufmann. Optimal best arm identification with fixed confidence. COLT 2016.
- E. Kaufmann and S. Kalyanakrishnan. Information complexity in bandit subset selection. COLT 2013.
- D. Russo. Simple Bayesian algorithms for best arm identification. COLT 2016.
- E. Even-Dar, S. Mannor and Y. Mansour. Action elimination and stopping conditions for the multi-armed bandit and reinforcement learning problems. JMLR 2006.
