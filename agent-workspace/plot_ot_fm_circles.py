"""Plot the final points and radius histograms from diagnose_ot_fm_circles.py.
Usage: python plot_ot_fm_circles.py OUT_DIR OUT_PNG NAME [NAME ...]   (NAME is e.g. base_euler_100)
"""
import os
import sys
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir, out_png, names = sys.argv[1], sys.argv[2], ['ref'] + sys.argv[3:]
fig, axes = plt.subplots(2, len(names), figsize=(4 * len(names), 8))
ref_r = torch.load(os.path.join(out_dir, 'ref.pt')).norm(dim=1)
for i, name in enumerate(names):
    pts = torch.load(os.path.join(out_dir, f'{name}.pt'))[:5000]
    axes[0, i].scatter(pts[:, 0], pts[:, 1], s=0.3)
    axes[0, i].set_title('circles (target)' if name == 'ref' else name)
    axes[0, i].set_xlim(-1.4, 1.4)
    axes[0, i].set_ylim(-1.4, 1.4)
    axes[0, i].set_aspect('equal')
    r = torch.load(os.path.join(out_dir, f'{name}.pt')).norm(dim=1)
    axes[1, i].hist(ref_r, bins=100, range=(0.5, 1.3), alpha=0.5, label='target')
    axes[1, i].hist(r, bins=100, range=(0.5, 1.3), alpha=0.5, label='generated')
    axes[1, i].set_xlabel('radius')
    axes[1, i].legend()
fig.tight_layout()
fig.savefig(out_png)
