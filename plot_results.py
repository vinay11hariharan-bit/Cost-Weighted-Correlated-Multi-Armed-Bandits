"""
Figures in absolute expected cost (no ratios).

    python plot_results.py
Reads   baselines_results.csv   (from compare_baselines.py)
        c12_sweep_results.csv   (from compare_c12_sweep.py)
Writes  fig_baselines_abs.png/.pdf, fig_c12_sweep_abs.png/.pdf
"""
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---- palette / style -------------------------------------------------------
OURS, OURS2 = "#2a78d6", "#86b6ef"      # ours: forcing sqrt(t), forcing t^(1/3)
OTHER = "#b9b7b0"                        # existing algorithms
GK = "#52514e"                           # best no-joint baseline (line chart)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0"
SURF = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.linewidth": 0.8, "savefig.facecolor": SURF, "figure.facecolor": SURF,
})
thousands = FuncFormatter(lambda v, _: f"{v:,.0f}")


def kl(d):
    return d * math.log(d / (1 - d)) + (1 - d) * math.log((1 - d) / d)


def clean(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ===========================================================================
# Figure 1: algorithm comparison, absolute cost, linear axis
# ===========================================================================
def fig_baselines():
    rows = list(csv.DictReader(open("baselines_results.csv")))
    deltas = sorted({float(r["delta"]) for r in rows}, reverse=True)
    insts = list(dict.fromkeys(r["instance"] for r in rows))
    algos = ["ours_sqrt", "ours_cbrt", "gk_tas", "ttts", "kl_lucb", "naive_joint", "se_hoeffding"]
    lab = {"ours_sqrt": "Ours, forcing √t", "ours_cbrt": "Ours, forcing t^1/3",
           "gk_tas": "GK Track-and-Stop", "ttts": "Top-Two Thompson", "kl_lucb": "KL-LUCB",
           "naive_joint": "Joint pulls, ρ ignored", "se_hoeffding": "Successive Elimination"}
    short = {i: i.split(" (")[0].replace("rho", "ρ").replace("c12", "c₁₂") for i in insts}
    sub = {i: i.split(" (")[1].rstrip(")") for i in insts}
    get = {(float(r["delta"]), r["instance"], r["algorithm"]): r for r in rows}

    fig, axes = plt.subplots(len(deltas), len(insts), figsize=(17, 3.3 * len(deltas) + 1.2),
                             sharey=True)
    for j, inst in enumerate(insts):
        # one x-range per column (shared by both deltas); SE is shown truncated
        xmax = max(float(get[(d, inst, a)]["mean_cost"]) + float(get[(d, inst, a)]["ci95"])
                   for d in deltas for a in algos if a != "se_hoeffding") / 0.80
        for i, d in enumerate(deltas):
            ax = axes[i, j]
            y = list(range(len(algos)))[::-1]
            for yi, a in zip(y, algos):
                r = get[(d, inst, a)]
                m, ci, err = float(r["mean_cost"]), float(r["ci95"]), float(r["error_rate"])
                col = OURS if a == "ours_sqrt" else OURS2 if a == "ours_cbrt" else OTHER
                shown = min(m, 0.84 * xmax)
                ax.barh(yi, shown, height=0.62, color=col, zorder=2)
                if m > shown:          # truncated bar: break mark + true value
                    for dx in (0.0, 0.025):
                        x0 = shown * 0.86 + dx * xmax
                        ax.plot([x0 - 0.008 * xmax, x0 + 0.008 * xmax], [yi - 0.42, yi + 0.42],
                                color=SURF, lw=2.2, zorder=3)
                    ax.text(shown + 0.015 * xmax, yi, f"{m:,.0f}  (axis cut)", va="center",
                            fontsize=8, color=INK2)
                else:
                    ax.errorbar(m, yi, xerr=ci, color=INK, lw=0.9, capsize=0, zorder=3)
                    note = ""
                    if a == "naive_joint" and m < float(r["lower_bound_joint_model"]):
                        note = "  below lower bound"
                    if err > d:
                        note += f"  error {err:.3f} > δ"
                    ax.text(m + ci + 0.015 * xmax, yi, f"{m:,.0f}{note}", va="center", fontsize=8, zorder=4,
                            bbox=dict(boxstyle="square,pad=0.15", fc=SURF, ec="none"),
                            color=INK if a.startswith("ours") else INK2,
                            fontweight="bold" if a == "ours_sqrt" else "normal")
            lbj = float(get[(d, inst, "ours_sqrt")]["lower_bound_joint_model"])
            lbn = float(get[(d, inst, "ours_sqrt")]["lower_bound_no_joint_model"])
            ax.axvline(lbj, color=INK, ls=(0, (4, 3)), lw=1.0, zorder=1)
            if abs(lbj - lbn) / lbn < 0.01:
                lbtxt = f"lower bound {lbj:,.0f} (joint pulls do not lower it)"
            else:
                ax.axvline(lbn, color=INK, ls=(0, (1, 2)), lw=1.0, zorder=1)
                lbtxt = f"lower bound {lbj:,.0f} with joint pulls, {lbn:,.0f} without"
            ax.text(0, 1.01, lbtxt, transform=ax.transAxes, fontsize=8, color=INK, va="bottom")
            ax.set_xlim(0, xmax)
            ax.set_ylim(-0.6, len(algos) - 0.4)
            ax.xaxis.set_major_formatter(thousands)
            clean(ax, "x")
            if i == 0:
                ax.set_title(f"{short[inst]}   ", loc="left", pad=40)
                ax.text(0, 1.10, sub[inst], transform=ax.transAxes, fontsize=8.5, color=INK2,
                        va="bottom")
            if i == len(deltas) - 1:
                ax.set_xlabel("expected cost at stopping")
            if j == 0:
                ax.set_yticks(y)
                ax.set_yticklabels([lab[a] for a in algos])
                ax.text(-0.62, 0.5, f"δ = {d:g}", transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=11, fontweight="bold", color=INK)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=OURS, label="Ours, forcing √t"),
                        Patch(color=OURS2, label="Ours, forcing t^1/3"),
                        Patch(color=OTHER, label="existing algorithms"),
                        Line2D([], [], color=INK, ls=(0, (4, 3)), lw=1, label="lower bound, joint pulls allowed"),
                        Line2D([], [], color=INK, ls=(0, (1, 2)), lw=1, label="lower bound, no joint pulls")],
               loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=5, frameon=False, fontsize=8.5)
    fig.suptitle("Expected cost to identify the best arm  (c = 1, 1000 runs, 95% CI; "
                 "every algorithm's error rate is below δ)", x=0.01, ha="left", fontsize=11.5,
                 color=INK)
    fig.tight_layout(rect=(0.0, 0, 1, 0.93), w_pad=2.0, h_pad=1.6)
    for ext in ("png", "pdf"):
        fig.savefig(f"fig_baselines_abs.{ext}", dpi=200)


# ===========================================================================
# Figure 2: joint pull cost sweep, absolute cost
# ===========================================================================
def fig_sweep():
    rows = list(csv.DictReader(open("c12_sweep_results.csv")))
    deltas = sorted({float(r["delta"]) for r in rows}, reverse=True)
    thetas = list(dict.fromkeys(r["theta"] for r in rows))
    gap = {t: float(t.split("(")[1].split(",")[0]) - float(t.split(",")[1]) for t in thetas}

    fig, axes = plt.subplots(len(deltas), len(thetas), figsize=(13, 4.0 * len(deltas) + 0.8))
    for i, d in enumerate(deltas):
        for j, th in enumerate(thetas):
            ax = axes[i, j]
            R = [r for r in rows if r["theta"] == th and float(r["delta"]) == d]
            cs = float(R[0]["c12_star"])
            gk, tt = float(R[0]["gk_tas_cost"]), float(R[0]["ttts_cost"])
            lb_no = kl(d) * 8.0 / gap[th] ** 2            # no-joint lower bound
            xs_all = sorted({float(r["c12"]) for r in R})
            xlo, xhi = 1.75, xs_all[-1] * 1.04

            ax.axvspan(2.0, cs, color=OURS, alpha=0.05, lw=0)
            ax.axvline(cs, color=MUTED, lw=0.8)
            ax.text(cs, 0.02, f"  c₁₂* = {cs:.2f}\n  joint pulls stop helping",
                    transform=ax.get_xaxis_transform(), fontsize=7.5, color=INK2, va="bottom")

            # lower bound with joint pulls: kl * min(c12/J, 8/d^2) = lb_no * min(c12/c12*, 1)
            xx = [xlo + k * (xhi - xlo) / 300 for k in range(301)]
            ax.plot(xx, [lb_no * min(x / cs, 1.0) for x in xx], color=INK, ls=(0, (4, 3)), lw=1.0)
            # best no-joint algorithm: constant in c12
            ax.axhline(gk, color=GK, lw=1.6)
            ax.text(xlo + 0.02 * (xhi - xlo), gk, f"GK Track-and-Stop, no joint pulls: {gk:,.0f}",
                    va="bottom", fontsize=8, color=INK2)
            ax.text(xhi, lb_no * 1.02, f"lower bound, no joint pulls: {lb_no:,.0f} ", ha="right", va="bottom",
                    fontsize=7.5, color=INK)

            for a, col, mk, lab in (("ours_cbrt", OURS2, "s", "Ours, forcing t^1/3"),
                                    ("ours_sqrt", OURS, "o", "Ours, forcing √t")):
                P = sorted((float(r["c12"]), float(r["mean_cost"]), float(r["ci95"]))
                           for r in R if r["algorithm"] == a)
                x, m, e = zip(*P)
                ax.errorbar(x, m, yerr=e, color=col, lw=2, marker=mk, ms=6, capsize=0,
                            mec=SURF, mew=1.5, zorder=4, label=lab)
                if a == "ours_sqrt":
                    inside = [(xi, mi) for xi, mi in zip(x, m) if xi < cs]
                    ax.fill_between([p[0] for p in inside], [p[1] for p in inside], gk,
                                    color=OURS, alpha=0.10, lw=0, zorder=1)
                    x0, m0 = x[0], m[0]
                    ax.annotate(f"{m0:,.0f}", (x0, m0), xytext=(8, -12), textcoords="offset points",
                                fontsize=8.5, fontweight="bold", color=INK, zorder=5,
                                bbox=dict(boxstyle="square,pad=0.15", fc=SURF, ec="none"))
                    ax.text(x0 + 0.06 * (xhi - xlo), (m0 + gk) / 2,
                            f"saves {gk - m0:,.0f}\nat c₁₂ = 2", fontsize=8, color=INK2, va="center")
            ax.plot([], [], color=INK, ls=(0, (4, 3)), lw=1.0, label="lower bound (joint pulls allowed)")
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(0, max(gk, tt) * 1.18)
            ax.yaxis.set_major_formatter(thousands)
            clean(ax, "y")
            ax.set_title(f"θ = {th.split(': ')[1]},  δ = {d:g}", loc="left")
            if i == len(deltas) - 1:
                ax.set_xlabel("joint pull cost c₁₂   (individual pull c = 1)")
            if j == 0:
                ax.set_ylabel("expected cost at stopping")
    h, l = axes[0, 0].get_legend_handles_labels()
    want = ["Ours, forcing √t", "Ours, forcing t^1/3", "lower bound (joint pulls allowed)"]
    fig.legend([h[l.index(w)] for w in want], want, loc="upper left",
               bbox_to_anchor=(0.01, 0.965), ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle("Joint pulls costing more than two individual pulls: expected cost vs c₁₂  "
                 "(1000 runs, 95% CI; shaded: 2 < c₁₂ < c₁₂*)", x=0.01, ha="left", fontsize=11.5,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.945), h_pad=2.0)
    for ext in ("png", "pdf"):
        fig.savefig(f"fig_c12_sweep_abs.{ext}", dpi=200)


if __name__ == "__main__":
    fig_baselines()
    fig_sweep()
    print("wrote fig_baselines_abs.png/.pdf and fig_c12_sweep_abs.png/.pdf")
