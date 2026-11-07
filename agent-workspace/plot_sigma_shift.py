"""Plot the shifted sigma schedule of SD3 / FLUX (diffusers FlowMatchEulerDiscreteScheduler) and a demo of why the shift is needed.
Figure 1: sigma per step and step size per step, for shift 1, 1.5, 3. Base grid as in the FLUX pipeline: linspace(1, 1/N, N), shift, then append 0.
Figure 2: same sigma at low and high resolution. Noise: x = (1 - sigma) * image + sigma * eps (diffusers scale_noise).
  The error of the 64x64 estimate pool(x) / (1 - sigma) is measured. SD3 predicts sigma/(1-sigma)/sqrt(k), k = pixels per 64x64 pixel.
Usage: python plot_sigma_shift.py OUT_SCHEDULE_PNG OUT_DEMO_PNG
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import load_sample_image

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference palette slots 1-3
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": TEXT, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def shift(sigmas, s):
    return s * sigmas / (1 + (s - 1) * sigmas)


# Figure 1: the schedule
n_steps = 28
base = np.linspace(1, 1 / n_steps, n_steps)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
for color, s, label in zip(SERIES, [1.0, 1.5, 3.0], ["shift 1 (no shift)", "shift 1.5", "shift 3 (SD3, 1024²)"]):
    sig = np.append(shift(base, s), 0.0)
    axes[0].plot(range(n_steps + 1), sig, color=color, lw=1.0, marker="o", ms=4, label=label)
    axes[1].plot(range(1, n_steps + 1), -np.diff(sig), color=color, lw=1.0, marker="o", ms=4, label=label)
axes[0].set_xlabel("sampling step (28 steps, from noise to data)")
axes[0].set_ylabel("sigma (1 = pure noise, 0 = data)")
axes[0].set_title("Sigma at each step", loc="left", color=TEXT)
axes[0].legend(frameon=False, labelcolor=TEXT, loc="lower left")
axes[1].set_xlabel("sampling step")
axes[1].set_ylabel("step size  |Δsigma|")
axes[1].set_title("Step size: small near noise, large near data", loc="left", color=TEXT)
axes[1].legend(frameon=False, labelcolor=TEXT)
fig.tight_layout()
fig.savefig(sys.argv[1])

# Figure 2: why high resolution needs more noise
img = load_sample_image("china.jpg")[:, 120:547]  # 427 x 427 center crop
hi = np.asarray(Image.fromarray(img).resize((256, 256), Image.LANCZOS), dtype=np.float32) / 127.5 - 1
hi = torch.tensor(hi).permute(2, 0, 1)[None]  # (1, 3, 256, 256), values in [-1, 1]
lo = F.avg_pool2d(hi, 4)  # (1, 3, 64, 64), the same image at 16x fewer pixels
alpha = 4.0  # sqrt(256^2 / 64^2)
sigma_lo = 0.6
sigma_hi = shift(sigma_lo, alpha)
torch.manual_seed(0)


def noisy(x, sigma):
    return (1 - sigma) * x + sigma * torch.randn_like(x)


def error_64(x, sigma):
    """RMSE of the 64x64 estimate pool(x) / (1 - sigma) vs the clean 64x64 image."""
    est = F.avg_pool2d(x, x.shape[-1] // 64) / (1 - sigma)
    return (est - lo).pow(2).mean().sqrt().item()


panels = [
    ("clean, 256×256", hi, None),
    (f"64×64, sigma = {sigma_lo}", noisy(lo, sigma_lo), sigma_lo),
    (f"256×256, sigma = {sigma_lo}", noisy(hi, sigma_lo), sigma_lo),
    (f"256×256, sigma = {sigma_hi:.3f} (shift 4)", noisy(hi, sigma_hi), sigma_hi),
]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), dpi=150)
for ax, (title, x, sigma) in zip(axes, panels):
    ax.imshow(((x[0].permute(1, 2, 0).clamp(-1, 1) + 1) / 2).numpy(), interpolation="nearest")
    ax.set_title(title, color=TEXT, fontsize=10)
    if sigma is not None:
        e, pred = error_64(x, sigma), sigma / (1 - sigma) / (x.shape[-1] // 64)
        ax.set_xlabel(f"error of 64×64 estimate: {e:.3f}\n(SD3 formula: {pred:.3f})", color=TEXT2)
        print(f"{title:40s} measured {e:.3f}  formula {pred:.3f}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
fig.tight_layout()
fig.savefig(sys.argv[2])
