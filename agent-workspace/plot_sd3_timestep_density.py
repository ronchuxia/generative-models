"""SD3 training noise levels: uniform (rf), logit-normal(0, 1) (256^2 pretraining), logit-normal(0, 1) + shift 3 (1024^2).
Check: shift s moves logit(t) by log(s), so the shifted t is logit-normal(log s, 1).
Usage: python plot_sd3_timestep_density.py OUT_PNG
"""
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference palette slots 1-3
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": TEXT, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def logit_normal_pdf(t, m, s):
    return 1 / (s * math.sqrt(2 * math.pi)) / (t * (1 - t)) * np.exp(-(np.log(t / (1 - t)) - m) ** 2 / (2 * s ** 2))


# Numeric check of the shifted distribution
s = 3.0
rng = np.random.default_rng(0)
t = 1 / (1 + np.exp(-rng.normal(0, 1, 1_000_000)))  # logit-normal(0, 1)
t_shift = s * t / (1 + (s - 1) * t)
lt = np.log(t_shift / (1 - t_shift))
print(f"logit(shifted t): mean {lt.mean():.4f} (log 3 = {math.log(s):.4f}), std {lt.std():.4f}")
print(f"P(t > 0.5): logit-normal(0,1) {np.mean(t > 0.5):.3f}, shifted {np.mean(t_shift > 0.5):.3f}")

x = np.linspace(0.001, 0.999, 999)
fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
ax.plot(x, np.ones_like(x), color=SERIES[0], lw=1.0, label="uniform (rf)")
ax.plot(x, logit_normal_pdf(x, 0, 1), color=SERIES[1], lw=1.0, label="logit-normal(0, 1), 256² pretraining")
ax.plot(x, logit_normal_pdf(x, math.log(s), 1), color=SERIES[2], lw=1.0, label="logit-normal(0, 1) + shift 3, 1024²")
ax.set_xlabel("t   (SD3: 0 = data, 1 = pure noise; same as diffusers sigma)")
ax.set_ylabel("density of training t")
ax.set_title("SD3: which noise levels the model trains on", loc="left", color=TEXT)
ax.legend(frameon=False, labelcolor=TEXT, loc="upper left")
ax.set_xlim(0, 1)
ax.set_ylim(0, 2.6)
fig.tight_layout()
fig.savefig(sys.argv[1])
