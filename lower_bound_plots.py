"""
Lower-bound plots (Section 3, Case II, symmetric individual costs c1 = c2 = c).
delta = mu1 - mu2 is the mean gap (as in Section 3), not the confidence level.

  f(x)   = delta^2/(2(1-x)) + log((1-x^2)/(1-rho^2)) + 2(x^2 - rho x)/(1-x^2)
  C(x)   = delta^2 (1+x)^2 + 4(x^2+1)(x - rho)          (f'(x) = 0  <=>  C(x) = 0)
  rho'*  = root of C in (-1, rho) minimizing f,   K = f(rho'*),   c12^th = 4 c K / delta^2
  Case I threshold: 2c/(1-rho)
"""
import numpy as np
import matplotlib.pyplot as plt

c = 1.0  # individual pull cost

def f(x, delta, rho):
    return delta**2/(2*(1 - x)) + np.log((1 - x**2)/(1 - rho**2)) + 2*(x**2 - rho*x)/(1 - x**2)

def rho_star(delta, rho):
    # C(x) = 4x^3 + (delta^2 - 4rho) x^2 + (2delta^2 + 4) x + (delta^2 - 4rho)
    r = np.roots([4.0, delta**2 - 4*rho, 2*delta**2 + 4, delta**2 - 4*rho])
    r = r[np.abs(r.imag) < 1e-10].real
    r = r[(r > -1) & (r < rho)]
    return r[np.argmin(f(r, delta, rho))]

def c12_th(delta, rho):
    return 4*c*f(rho_star(delta, rho), delta, rho)/delta**2

def save(fig, name):
    fig.savefig(name + ".pdf", bbox_inches="tight", pad_inches=0.4)
    fig.savefig(name + ".png", dpi=300, bbox_inches="tight", pad_inches=0.4)

# ======================================================================
# Fig 1: optimal cost fractions alpha vs c12 for a fixed theta
#        (alpha_a = c_a gamma_a; v_J -> alpha = (0, 0, 1), v_I -> alpha = (1/2, 1/2, 0))
# ======================================================================
mu1, mu2, rho = 1.0, 0.0, 0.4            # fixed environment
delta = mu1 - mu2
th, th1 = c12_th(delta, rho), 2*c/(1 - rho)
print(f"theta = ({mu1}, {mu2}, {rho}):  rho'* = {rho_star(delta, rho):.4f},  "
      f"c12^th = {th:.3f},  Case I threshold = {th1:.3f}")

cc = np.linspace(1.5, 5.5, 2001)
a12 = np.where(cc < th, 1.0, 0.0)
a1  = np.where(cc < th, 0.0, 0.5)

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(cc, a12, color="firebrick", lw=2.5, label=r"$\alpha_{1,2}$ (joint)")
ax.plot(cc, a1, color="dodgerblue", lw=2.5, ls="--", label=r"$\alpha_1 = \alpha_2$ (individual)")
ax.axvline(th, color="red", lw=2, label=rf"Case II threshold ($c_{{1,2}}^{{\mathrm{{th}}}} = {th:.2f}$)")
ax.axvline(th1, color="k", lw=2, ls=":", label=rf"Case I threshold ($2c/(1-\rho) = {th1:.2f}$)")
ax.set_xlim(cc[0], cc[-1]); ax.set_ylim(-0.05, 1.05)
ax.set_xlabel(r"Joint pull cost $c_{1,2}$", fontsize=13)
ax.set_ylabel(r"Optimal cost fractions $\alpha^\star$", fontsize=13)
ax.set_title(rf"$\theta = (\mu_1, \mu_2, \rho) = ({mu1:g}, {mu2:g}, {rho:g}),\ c = {c:g}$", fontsize=13)
ax.grid(alpha=0.3)
ax.legend(loc="center right", fontsize=10)
save(fig, "fig_allocation_vs_c12")

# ======================================================================
# Fig 2: c12^th slices -- (a) vs delta for fixed rho, (b) vs rho for fixed delta
# ======================================================================
fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 4.5))
d = np.linspace(0.1, 4, 400)
for r in [-0.5, 0.0, 0.4, 0.7]:
    axa.plot(d, [c12_th(x, r) for x in d], lw=2, label=rf"$\rho = {r}$")
axa.set_xlim(d[0], d[-1])
axa.set_xlabel(r"Mean gap $\delta = \mu_1 - \mu_2$", fontsize=13)
axa.set_ylabel(r"$c_{1,2}^{\mathrm{th}}$", fontsize=13)
axa.set_title(r"(a) $c_{1,2}^{\mathrm{th}}$ vs. $\delta$", fontsize=13)
axa.grid(alpha=0.3); axa.legend()

rr = np.linspace(-0.9, 0.95, 400)
for x in [0.5, 1.0, 2.0]:
    axb.plot(rr, [c12_th(x, r) for r in rr], lw=2, label=rf"$\delta = {x}$")
axb.plot(rr, 2*c/(1 - rr), "k--", lw=2, alpha=0.75, label=r"Case I: $2c/(1-\rho)$")
axb.set_xlim(rr[0], rr[-1]); axb.set_ylim(0, 15)
axb.set_xlabel(r"Correlation $\rho$", fontsize=13)
axb.set_ylabel(r"$c_{1,2}^{\mathrm{th}}$", fontsize=13)
axb.set_title(r"(b) $c_{1,2}^{\mathrm{th}}$ vs. $\rho$", fontsize=13)
axb.grid(alpha=0.3); axb.legend()
fig.tight_layout()
save(fig, "fig_threshold_slices")

# ======================================================================
# Fig 3 and Fig 4: 3-D surfaces of c12^th(delta, rho) and rho'*(delta, rho)
# ======================================================================
D, R = np.meshgrid(np.linspace(0.05, 4, 120), np.linspace(-0.9, 0.85, 120))
TH = np.vectorize(c12_th)(D, R)
RS = np.vectorize(rho_star)(D, R)

for Z, zl, title, name, levels in [
        (TH, r"$c_{1,2}^{\mathrm{th}}$", r"$c_{1,2}^{\mathrm{th}}(\delta, \rho)$", "fig_c12_threshold_3d",
         [1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 10, 12]),
        (RS, r"$\rho'^\star$",          r"$\rho'^\star(\delta, \rho)$",          "fig_rho_star_3d",
         np.round(np.arange(-0.8, 0.81, 0.1), 2))]:
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(projection="3d")
    s = ax.plot_surface(D, R, Z, cmap="turbo", linewidth=0, antialiased=True, alpha=0.85)
    ax.contour(D, R, Z, levels=levels, colors="k", linewidths=0.8,
               linestyles="solid")                                                  # contour lines on the surface
    ax.set_xlabel(r"$\delta$ (gap)", fontsize=13, labelpad=8)
    ax.set_ylabel(r"$\rho$ (correlation)", fontsize=13, labelpad=8)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(zl, fontsize=13, rotation=90, labelpad=10)
    ax.set_box_aspect(None, zoom=0.88)
    ax.view_init(elev=22, azim=-125)
    cb = fig.colorbar(s, ax=ax, shrink=0.55, pad=0.08)
    cb.set_label(zl, fontsize=13)
    ax.set_title(title, fontsize=14)
    save(fig, name)

plt.show()
