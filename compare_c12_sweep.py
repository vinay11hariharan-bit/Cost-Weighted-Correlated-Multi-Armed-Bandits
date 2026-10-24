"""
Joint pulls that cost MORE than two individual pulls (c12 > 2c)
================================================================

Question: for c12 between 2c and the threshold c12*, joint pulls are still the
optimal action. Does the algorithm still beat the no-joint models there?

Threshold.  The joint vertex wins iff  J(theta)/c12 > d^2/(8c), i.e.

        c12 < c12* = 8 c J(theta) / d^2        (for small gaps, c12* ~ 2c/(1-rho))

and the asymptotic cost ratio (joint model / no-joint model) is simply

        T*_c(joint model) / T*_c(no joint) = min(c12 / c12*, 1).

So the predicted saving shrinks linearly in c12 and vanishes exactly at c12*.

What this script does, for each theta below:
  * computes c12*,
  * sweeps c12 = 2 + f (c12* - 2) for f in FRACS (all strictly between 2 and c12*),
    plus one control point above c12* (joint pulls no longer help),
  * runs ours_sqrt and ours_cbrt at every c12,
  * runs the best no-joint baselines (GK Track-and-Stop, TTTS) once per theta
    (they never use joint pulls, so their cost does not depend on c12).
  "naive_joint" is left out: for c12 > 2c it never uses joint pulls and is identical to GK.

Needs compare_baselines.py in the same folder.

Usage:
    python compare_c12_sweep.py                       # 1000 runs, delta = 0.1 and 0.01
    python compare_c12_sweep.py --runs 300 --workers 8
Outputs: c12_sweep_table.txt, c12_sweep_results.csv, fig_c12_sweep.png
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

import compare_baselines as B

THETAS = {
    "A: (0.1, 0, 0.8)": (0.1, 0.0, 0.8),   # c12* ~ 9.88
    "E: (0.2, 0, 0.5)": (0.2, 0.0, 0.5),   # c12* ~ 3.93
}
FRACS = [0.0, 0.25, 0.5, 0.75, 0.9]   # position between 2 and c12*  (0.0 -> c12 = 2, the old point)
CONTROL = 1.2                          # one point at 1.2 * c12*  (joint pulls should NOT be used)
OURS = ["ours_sqrt", "ours_cbrt"]
NOJOINT = ["gk_tas", "ttts"]


def c12_star(theta):
    d = theta[0] - theta[1]
    return 8.0 * B.C_IND * B.joint_info(d, theta[2]) / d ** 2


def c12_grid(theta):
    cs = c12_star(theta)
    return [round(2.0 + f * (cs - 2.0), 3) for f in FRACS] + [round(CONTROL * cs, 3)]


def _job(args):
    name, algo, c12, delta, seed = args
    return name, algo, c12, delta, B.run(algo, THETAS[name], c12, delta, seed)


def joint_fraction_hint(theta, c12):
    Ii, Ij = B.vertex_rates(*theta, B.C_IND, c12)
    return "joint" if Ij > Ii else "individual"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.1, 0.01])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="parallel processes (default: all cores)")
    args = ap.parse_args()

    jobs, seed = [], 10**7          # seeds disjoint from compare_baselines.py
    for name, theta in THETAS.items():
        for dl in args.deltas:
            for algo in NOJOINT:                       # c12 irrelevant for these
                for _ in range(args.runs):
                    jobs.append((name, algo, 2.0, dl, seed)); seed += 1
            for c12 in c12_grid(theta):
                for algo in OURS:
                    for _ in range(args.runs):
                        jobs.append((name, algo, c12, dl, seed)); seed += 1
    print(f"{len(jobs)} runs on {args.workers} workers ...", flush=True)
    t0 = time.time()
    out = {}
    with Pool(args.workers) as pool:
        for i, (name, algo, c12, dl, r) in enumerate(pool.imap_unordered(_job, jobs, chunksize=20)):
            out.setdefault((name, algo, c12, dl), []).append(r)
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    def stats(key):
        res = out[key]
        x = np.array([r[0] for r in res])
        return x.mean(), x.std() / np.sqrt(len(x)), np.mean([r[1] != 1 for r in res])

    rows, lines = [], []
    for dl in args.deltas:
        lines.append(f"=== delta = {dl:g}  ({args.runs} runs, exploration rate log((log t + 1)/delta)) ===\n")
        for name, theta in THETAS.items():
            cs = c12_star(theta)
            gk_m, gk_se, gk_e = stats((name, "gk_tas", 2.0, dl))
            tt_m, _, tt_e = stats((name, "ttts", 2.0, dl))
            lines.append(f"{name}   c = {B.C_IND}, threshold c12* = {cs:.2f}")
            lines.append(f"  no joint pulls:  GK Track-and-Stop {gk_m:.0f} (error {gk_e:.3f}),  "
                         f"TTTS {tt_m:.0f} (error {tt_e:.3f})")
            lines.append(f"  {'c12':>6s} {'vertex':>10s} {'theory':>7s} "
                         + "".join(f"{a + ' cost':>17s} {'ratio [95% CI]':>19s} {'error':>6s}" for a in OURS))
            for c12 in c12_grid(theta):
                theory = min(c12 / cs, 1.0)
                cells = ""
                for algo in OURS:
                    m, se, e = stats((name, algo, c12, dl))
                    r = m / gk_m
                    rse = r * np.sqrt((se / m) ** 2 + (gk_se / gk_m) ** 2)
                    cells += f"{m:17.0f} {r:6.2f} [{r - 1.96 * rse:4.2f},{r + 1.96 * rse:4.2f}] {e:6.3f}"
                    rows.append({"delta": dl, "theta": name, "c12": c12, "c12_star": cs,
                                 "algorithm": algo, "mean_cost": m, "ci95": 1.96 * se, "error_rate": e,
                                 "gk_tas_cost": gk_m, "ttts_cost": tt_m, "ratio_to_gk": r,
                                 "ratio_ci95": 1.96 * rse, "theory_ratio": theory})
                lines.append(f"  {c12:6.2f} {joint_fraction_hint(theta, c12):>10s} {theory:7.2f} {cells}")
            lines.append("  ratio = E[cost ours] / E[cost GK Track-and-Stop];  theory = min(c12/c12*, 1)\n")
    text = "\n".join(lines)
    print(text)
    open("c12_sweep_table.txt", "w").write(text)
    with open("c12_sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    plot(rows, args.deltas)
    print(f"done in {time.time() - t0:.0f}s -> c12_sweep_table.txt, c12_sweep_results.csv, fig_c12_sweep.png")


def plot(rows, deltas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = ["#eb6834", "#2a78d6", "#1f9e6e"]
    fig, axes = plt.subplots(1, len(THETAS), figsize=(6.2 * len(THETAS), 4.4))
    axes = np.atleast_1d(axes)
    for ax, (name, theta) in zip(axes, THETAS.items()):
        cs = c12_star(theta)
        xs = np.linspace(2.0, CONTROL * cs, 200)
        ax.plot(xs, np.minimum(xs / cs, 1.0), color="#0b0b0b", ls="--", lw=1.2,
                label="theory  min(c12/c12*, 1)")
        for k, dl in enumerate(deltas):
            pts = sorted((r["c12"], r["ratio_to_gk"], r["ratio_ci95"]) for r in rows
                         if r["theta"] == name and r["delta"] == dl and r["algorithm"] == "ours_sqrt")
            x, y, e = zip(*pts)
            ax.errorbar(x, y, yerr=e, color=COL[k % len(COL)], lw=2, marker="o", ms=5, capsize=0,
                        label=f"ours (forcing √t), δ = {dl:g}")
        ax.axhline(1.0, color="#9a9890", lw=0.8)
        ax.axvline(cs, color="#9a9890", lw=0.8, ls=":")
        ax.text(cs, 0.05, f" c12* = {cs:.2f}", fontsize=8, color="#52514e")
        ax.axvspan(2.0, cs, color="#2a78d6", alpha=0.05)
        ax.set_xlim(1.8, CONTROL * cs * 1.03)
        ax.set_ylim(0, 1.35)
        ax.set_title(f"{name}   (c = 1)", fontsize=10)
        ax.set_xlabel("joint pull cost c12   (shaded: 2 < c12 < c12*)")
        ax.grid(color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("E[cost, ours] / E[cost, GK Track-and-Stop]")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Joint pulls dearer than two individual pulls: the saving shrinks linearly and ends at c12*",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("fig_c12_sweep.png", dpi=150)


if __name__ == "__main__":
    main()
