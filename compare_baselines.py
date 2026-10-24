"""
Where does the cost-weighted correlated Track-and-Stop stand?
================================================================

Compares, on the same instances and with the same exploration rate,

  YOUR MODEL (joint pulls allowed, correlation modelled):
    ours_sqrt       Track-and-Stop with vertex comparator, forced joint pulls N12 < sqrt(t+1)
    ours_cbrt       same, forced joint pulls N12 < (t+1)^(1/3)

  EXISTING MODELS without joint pulls (two independent Gaussian arms, cost c per pull):
    gk_tas          Track-and-Stop, D-tracking + GLR (Garivier & Kaufmann 2016).
                    For two equal-variance Gaussian arms w* = (1/2, 1/2), so this is
                    alternating sampling. Chernoff-Racing coincides with it for K = 2.
    kl_lucb         KL-LUCB (Kaufmann & Kalyanakrishnan 2013). KL-Racing coincides with it for K = 2.
    ttts            Top-Two Thompson Sampling, beta = 1/2 (Russo 2016), with GLR stopping.
    se_hoeffding    Successive Elimination (Even-Dar, Mannor & Mansour 2006).

  EXISTING MODEL that allows joint pulls but ignores correlation:
    naive_joint     treats a joint pull as two independent pulls at cost c12; uses whichever
                    action is cheaper per sample (joint iff c12 < 2c) and an independence GLR.
                    Shows what happens when the correlation is not modelled.

Exploration rate for every algorithm: beta(t, delta) = log((log t + 1) / delta), the rate
Garivier & Kaufmann use in their experiments (Section 5).

Usage:
    python compare_baselines.py                    # 1000 runs, delta = 0.1 and 0.01
    python compare_baselines.py --runs 300 --workers 8 --deltas 0.1 0.01 0.001
Outputs: baselines_results.csv, baselines_table.txt, fig_baselines.png
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # one BLAS thread per worker, no oversubscription
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import csv
import time
from multiprocessing import Pool

import numpy as np

C_IND = 1.0  # cost of an individual pull

INSTANCES = {
    "A: rho=0.8, c12=2 (joint pulls help)":          ((0.1, 0.0, 0.8), 2.0),
    "B: rho=-0.5, c12=2 (joint pulls do not help)":  ((0.2, 0.0, -0.5), 2.0),
    "C: rho=0, c12=1.5 (cheap joint, uncorrelated)": ((0.2, 0.0, 0.0), 1.5),
    "D: rho=-0.5, c12=1.5 (cheap joint, neg. corr.)": ((0.2, 0.0, -0.5), 1.5),
}

ALGOS = ["ours_sqrt", "ours_cbrt", "gk_tas", "kl_lucb", "ttts", "se_hoeffding", "naive_joint"]
LABELS = {
    "ours_sqrt": "Ours: Track-and-Stop, joint pulls (forcing √t)",
    "ours_cbrt": "Ours: Track-and-Stop, joint pulls (forcing t^1/3)",
    "gk_tas": "GK Track-and-Stop / Chernoff-Racing (no joint)",
    "kl_lucb": "KL-LUCB / KL-Racing (no joint)",
    "ttts": "Top-Two Thompson Sampling + GLR (no joint)",
    "se_hoeffding": "Successive Elimination (no joint)",
    "naive_joint": "Joint pulls, correlation ignored",
}
T_MIN, N12_MIN = 10, 5   # short initialization before the first stopping test (joint model)


def beta(t, delta):
    return np.log((np.log(max(t, 1)) + 1.0) / delta)


# ===========================================================================
# Information constants
# ===========================================================================
_Z = np.linspace(-6.0, 6.0, 121)
RGRID = np.tanh(_Z)


def _d12_min(d, rho, r):
    om = 1.0 - r * r
    return 0.5 * np.log(om / (1.0 - rho * rho)) + (1.0 - rho * r) / om - 1.0 + d * d / (4.0 * (1.0 - r))


def joint_info(d, rho):
    v = _d12_min(d, rho, RGRID)
    i = int(np.argmin(v))
    zz = np.linspace(_Z[max(i - 1, 0)], _Z[min(i + 1, len(_Z) - 1)], 41)
    return float(np.min(_d12_min(d, rho, np.tanh(zz))))


def vertex_rates(mu1, mu2, rho, c, c12):
    d = mu1 - mu2
    return d * d / (8.0 * c), joint_info(d, rho) / c12


def lower_bounds(theta, c12, delta):
    """T*_c kl(delta, 1-delta) for the joint model and for the no-joint model."""
    kl = delta * np.log(delta / (1 - delta)) + (1 - delta) * np.log((1 - delta) / delta)
    Ii, Ij = vertex_rates(*theta, C_IND, c12)
    return kl / max(Ii, Ij), kl / Ii


# ===========================================================================
# Exact GLR for the joint model (means profiled in closed form, rho by grid)
# ===========================================================================
def _loglik(n1, s1, n2, s2, n, su, sv, suu, svv, suv, m1, m2, r):
    om = 1.0 - r * r
    ind = -0.5 * (n1 * m1 * m1 - 2 * m1 * s1) - 0.5 * (n2 * m2 * m2 - 2 * m2 * s2)
    Suu = suu - 2 * m1 * su + n * m1 * m1
    Svv = svv - 2 * m2 * sv + n * m2 * m2
    Suv = suv - m1 * sv - m2 * su + n * m1 * m2
    return ind - 0.5 * n * np.log(om) - (Suu - 2 * r * Suv + Svv) / (2 * om)


def _profiles(st, r):
    n1, s1, n2, s2, n, su, sv, suu, svv, suv = st
    om = 1.0 - r * r
    k = n / om
    A11, A22, A12 = n1 + k, n2 + k, -r * k
    b1 = s1 + (su - r * sv) / om
    b2 = s2 + (sv - r * su) / om
    det = A11 * A22 - A12 * A12
    m1 = (A22 * b1 - A12 * b2) / det
    m2 = (A11 * b2 - A12 * b1) / det
    mb = (s1 + s2 + (su + sv) / (1 + r)) / (n1 + n2 + 2 * n / (1 + r))
    return m1, m2, _loglik(*st, m1, m2, r), _loglik(*st, mb, mb, r)


def _refine_max(f, vals):
    i = int(np.argmax(vals))
    zz = np.linspace(_Z[max(i - 1, 0)], _Z[min(i + 1, len(_Z) - 1)], 25)
    v2 = f(np.tanh(zz))
    j = int(np.argmax(v2))
    return v2[j], np.tanh(zz[j])


def glr_joint(st):
    """Z_t, empirical leader, MLE (mu1, mu2, rho). Requires at least one joint pull."""
    _, _, lu, _ = _profiles(st, RGRID)
    LU, rhat = _refine_max(lambda r: _profiles(st, r)[2], lu)
    a1, a2, _, _ = _profiles(st, np.array([rhat]))
    leader = 1 if a1[0] > a2[0] else 2

    def falt(r):
        p1, p2, u, b = _profiles(st, r)
        inside = (p1 <= p2) if leader == 1 else (p1 >= p2)
        return np.where(inside, u, b)

    LA, _ = _refine_max(falt, falt(RGRID))
    return max(LU - LA, 0.0), leader, (a1[0], a2[0], rhat)


def glr_indep(n1, m1, n2, m2):
    """GLR for two independent unit-variance Gaussian arms."""
    return n1 * n2 / (n1 + n2) * (m1 - m2) ** 2 / 2.0


# ===========================================================================
# One run of one algorithm. Returns (cost, recommended arm, rounds).
# ===========================================================================
def run(algo, theta, c12, delta, seed, tmax=10**6):
    rng = np.random.default_rng(seed)
    mu1, mu2, rho = theta
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    c = C_IND

    # ---------------- models without joint pulls ----------------
    if algo in ("gk_tas", "kl_lucb", "ttts", "se_hoeffding"):
        n = [0, 0]; s = [0.0, 0.0]; cost = 0.0; t = 0
        mus = (mu1, mu2)

        def pull(a):
            nonlocal cost, t
            s[a] += mus[a] + rng.standard_normal(); n[a] += 1; cost += c; t += 1

        pull(0); pull(1)
        while t < tmax:
            m = (s[0] / n[0], s[1] / n[1])
            lead = 0 if m[0] > m[1] else 1
            if algo in ("gk_tas", "ttts"):
                if glr_indep(n[0], m[0], n[1], m[1]) > beta(t, delta):
                    return cost, lead + 1, t
                if algo == "gk_tas":
                    # D-tracking with w* = (1/2, 1/2); forced exploration is implied
                    pull(0 if n[0] <= n[1] else 1)
                else:
                    th = (m[0] + rng.standard_normal() / np.sqrt(n[0]),
                          m[1] + rng.standard_normal() / np.sqrt(n[1]))
                    top = 0 if th[0] > th[1] else 1
                    pull(top if rng.random() < 0.5 else 1 - top)
            elif algo == "kl_lucb":
                b = beta(t, delta)
                lo = m[lead] - np.sqrt(2 * b / n[lead])
                hi = m[1 - lead] + np.sqrt(2 * b / n[1 - lead])
                if lo > hi:
                    return cost, lead + 1, t
                pull(lead); pull(1 - lead)   # LUCB samples leader and challenger
            else:  # successive elimination, Hoeffding radius for 1-sub-Gaussian arms
                k = min(n)
                rad = np.sqrt(2 * np.log(8 * k * k / delta) / k)
                if abs(m[0] - m[1]) > 2 * rad:
                    return cost, lead + 1, t
                pull(0); pull(1)
        return cost, -1, t

    # ---------------- joint model ----------------
    n1 = n2 = n = 0
    s1 = s2 = su = sv = suu = svv = suv = 0.0
    cost = 0.0
    naive_joint_action = c12 < 2 * c
    for t in range(tmax):
        if algo == "naive_joint":
            # independence model: a joint pull = one sample of each arm
            N1, N2 = n1 + n, n2 + n
            if N1 >= 1 and N2 >= 1 and t >= T_MIN:
                M1, M2 = (s1 + su) / N1, (s2 + sv) / N2
                if glr_indep(N1, M1, N2, M2) > beta(t, delta):
                    return cost, (1 if M1 > M2 else 2), t
            act = 12 if naive_joint_action else (1 if n1 <= n2 else 2)
        else:
            est = None
            if n >= N12_MIN and t >= T_MIN:
                Z, leader, est = glr_joint((n1, s1, n2, s2, n, su, sv, suu, svv, suv))
                if Z > beta(t, delta):
                    return cost, leader, t
            force = np.sqrt(t + 1) if algo == "ours_sqrt" else (t + 1) ** (1.0 / 3.0)
            if n < max(force, N12_MIN) or est is None:
                act = 12
            else:
                Ii, Ij = vertex_rates(est[0], est[1], est[2], c, c12)
                act = 12 if Ij > Ii else (1 if n1 <= n2 else 2)
        if act == 1:
            x = mu1 + rng.standard_normal(); n1 += 1; s1 += x; cost += c
        elif act == 2:
            x = mu2 + rng.standard_normal(); n2 += 1; s2 += x; cost += c
        else:
            e = L @ rng.standard_normal(2)
            u, v = mu1 + e[0], mu2 + e[1]
            n += 1; su += u; sv += v; suu += u * u; svv += v * v; suv += u * v
            cost += c12
    return cost, -1, tmax


def _job(args):
    inst, algo, delta, seed = args
    theta, c12 = INSTANCES[inst]
    return inst, algo, delta, run(algo, theta, c12, delta, seed)


# ===========================================================================
# Main: run, tabulate, plot
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.1, 0.01])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="parallel processes (default: all cores)")
    args = ap.parse_args()

    jobs, seed = [], 0
    for inst in INSTANCES:
        for algo in ALGOS:
            for dl in args.deltas:
                for _ in range(args.runs):
                    jobs.append((inst, algo, dl, seed)); seed += 1
    print(f"{len(jobs)} runs on {args.workers} workers ...", flush=True)
    t0 = time.time()
    out = {}
    with Pool(args.workers) as pool:
        for i, (inst, algo, dl, r) in enumerate(pool.imap_unordered(_job, jobs, chunksize=20)):
            out.setdefault((inst, algo, dl), []).append(r)
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    rows, lines = [], []
    for dl in args.deltas:
        lines.append(f"=== delta = {dl:g}   (exploration rate log((log t + 1)/delta), {args.runs} runs) ===\n")
        for inst, (theta, c12) in INSTANCES.items():
            lb_j, lb_n = lower_bounds(theta, c12, dl)
            Ii, Ij = vertex_rates(*theta, C_IND, c12)
            lines.append(f"{inst}   (mu1, mu2, rho) = {theta}, c = {C_IND}, c12 = {c12}")
            lines.append(f"  optimal vertex: {'joint' if Ij > Ii else 'individual'};  lower bound "
                         f"T*_c kl(delta,1-delta): joint model {lb_j:.0f}, no-joint model {lb_n:.0f}")
            lines.append(f"  {'algorithm':52s} {'mean cost':>9s} {'±95%':>6s} {'error':>6s}")
            for algo in ALGOS:
                res = out[(inst, algo, dl)]
                cost = np.array([r[0] for r in res])
                err = np.mean([r[1] != 1 for r in res])
                hw = 1.96 * cost.std() / np.sqrt(len(cost))
                flag = "  <-- error above delta" if err > dl else ""
                lines.append(f"  {LABELS[algo]:52s} {cost.mean():9.0f} {hw:6.0f} {err:6.3f}{flag}")
                rows.append({"delta": dl, "instance": inst, "algorithm": algo, "mean_cost": cost.mean(),
                             "ci95": hw, "error_rate": err, "runs": len(cost),
                             "lower_bound_joint_model": lb_j, "lower_bound_no_joint_model": lb_n})
            lines.append("")
    text = "\n".join(lines)
    print(text)
    open("baselines_table.txt", "w").write(text)
    with open("baselines_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    plot(out, args.deltas[0])
    print(f"done in {time.time() - t0:.0f}s -> baselines_table.txt, baselines_results.csv, fig_baselines.png")


def plot(out, dl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ACCENT, NEUTRAL, WARN = "#2a78d6", "#a8a6a0", "#e34948"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (inst, (theta, c12)) in zip(axes.ravel(), INSTANCES.items()):
        lb_j, lb_n = lower_bounds(theta, c12, dl)
        names = ALGOS[::-1]
        means = [np.mean([r[0] for r in out[(inst, a, dl)]]) for a in names]
        errs = [np.mean([r[1] != 1 for r in out[(inst, a, dl)]]) for a in names]
        cols = [ACCENT if a.startswith("ours") else NEUTRAL for a in names]
        y = np.arange(len(names))
        ax.barh(y, means, color=cols, height=0.6)
        for yi, m, e in zip(y, means, errs):
            ax.text(m, yi, f"  {m:.0f}  (error {e:.3f})", va="center", fontsize=8,
                    color=WARN if e > dl else "#52514e")
        ax.axvline(lb_j, color="#0b0b0b", ls="--", lw=1)
        ax.axvline(lb_n, color="#0b0b0b", ls=":", lw=1)
        ax.text(lb_j, len(names) - 0.3, " lower bound, joint model", fontsize=7, va="bottom")
        ax.text(lb_n, len(names) + 0.15, " lower bound, no joint", fontsize=7, va="bottom")
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[a] for a in names], fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(min(min(means), lb_j) * 0.6, max(means) * 6)
        ax.set_ylim(-0.6, len(names) + 0.8)
        ax.set_title(inst, fontsize=10)
        ax.set_xlabel("expected cost at stopping (log scale)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"Expected stopping cost, δ = {dl:g} (blue = your algorithm; red text = error rate above δ)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("fig_baselines.png", dpi=150)


if __name__ == "__main__":
    main()
