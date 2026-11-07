"""Check: at which t does the gap between the two rings appear in p_t (independent pairs, xt = t z + (1-t) x)?
Prints density at the gap (r = 0.9) divided by density at the inner peak (r = 0.8)."""
import numpy as np
import torch
from generative_models.data import MoonsDataset, CirclesDataset

np.random.seed(0)
n = 1000000
x = MoonsDataset(n_samples=n, noise=0.05).x
z = CirclesDataset(n_samples=n, noise=0.04).x[torch.randperm(n)]
for t in [0.5, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0]:
    r = (t * z + (1 - t) * x).norm(dim=1) / t   # divide by t so the rings stay at r = 0.8 and 1.0
    gap = ((r > 0.88) & (r < 0.92)).float().mean()
    peak = ((r > 0.78) & (r < 0.82)).float().mean()
    print(f"t = {t:.2f}  gap / peak = {(gap / peak).item():.2f}")
