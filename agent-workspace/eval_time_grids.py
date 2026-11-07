"""Check: effect of the Euler time grid on a trained model (from diagnose_ot_fm_circles.py).
Separates two effects: (1) the same step budget moved between start and end, (2) extra small steps added at the end.
Usage: python eval_time_grids.py OUT_DIR NAME HIDDEN   (e.g. big_long_cosine 256)
"""
import os
import sys
import numpy as np
import torch
from generative_models.data import MoonsDataset, CirclesDataset
from generative_models.model import FlowMatchingMLP
from generative_models.de import NeuralODE, Euler

out_dir, name, h = sys.argv[1], sys.argv[2], int(sys.argv[3])
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FlowMatchingMLP(2, [h, h, h, 2]).to(device)
model.load_state_dict(torch.load(os.path.join(out_dir, f'{name}_model.pt')))
model.eval()

n = 20000
np.random.seed(1)
x0 = MoonsDataset(n_samples=n, noise=0.05).x.to(device)
ref = CirclesDataset(n_samples=n, noise=0.04).x.to(device)


def radius_w1(a, c):
    return (a.norm(dim=1).sort().values - c.norm(dim=1).sort().values).abs().mean().item()


def sliced_w1(a, c, k=256):
    d = torch.randn(2, k, generator=torch.Generator().manual_seed(0)).to(a.device)
    d = d / d.norm(dim=0)
    return ((a @ d).sort(0).values - (c @ d).sort(0).values).abs().mean().item()


s = torch.linspace(0, 1, 101)
grids = {
    'uniform 100': s,
    'small at end, 100 (budget moved)': 1 - (1 - s) ** 2,
    'small at start, 100 (budget moved)': s ** 2,
    'uniform 100 + last 10 steps split x10 (190)': torch.cat([s[:91], torch.linspace(0.9, 1, 101)[1:]]),
    'uniform 100 + first 10 steps split x10 (190)': torch.cat([torch.linspace(0, 0.1, 101), s[11:]]),
    'uniform 100 + middle 80 steps split x10 (820)': torch.cat([s[:10], torch.linspace(0.1, 0.9, 801), s[91:]]),
    'uniform 1000': torch.linspace(0, 1, 1001),
}
print(f"{'grid':48s} {'steps':>5s} {'first dt':>9s} {'last dt':>9s} {'radius W1':>10s} {'sliced W1':>10s}")
for g, t in grids.items():
    t = t.to(device)
    with torch.no_grad():
        x1 = Euler(ode=NeuralODE(model)).simulate_trajectory(x0, t[None, :].expand(n, -1))
    dt = t.diff()
    print(f"{g:48s} {len(dt):5d} {dt[0].item():9.5f} {dt[-1].item():9.5f} {radius_w1(x1, ref):10.4f} {sliced_w1(x1, ref):10.4f}")
