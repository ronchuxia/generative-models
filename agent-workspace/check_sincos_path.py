import math
import torch
from generative_models.data import MoonsCirclesDataset

torch.manual_seed(0)
alpha = lambda t: torch.sin(0.5 * torch.pi * t)
beta = lambda t: torch.cos(0.5 * torch.pi * t)

# 1. Distance of the sin/cos path from the straight segment, for moons-circles pairs
ds = MoonsCirclesDataset(n_samples=10000, noise_moons=0.05, noise_circles=0.04)
x, z = ds.x0, ds.x1
t = torch.linspace(0, 1, 201)[None, :, None]
xt = alpha(t) * z[:, None] + beta(t) * x[:, None]                       # (b, l, 2)
seg = z[:, None] - x[:, None]
s = ((xt - x[:, None]) * seg).sum(-1, keepdim=True) / (seg**2).sum(-1, keepdim=True)
dist = (xt - (x[:, None] + s.clamp(0, 1) * seg)).norm(dim=-1).amax(1)    # max distance to segment
print(f"1. max distance to straight segment: mean {dist.mean():.3f}, median {dist.median():.3f}, pair length mean {seg.norm(dim=-1).mean():.3f}")
mid = alpha(torch.tensor(0.5)) * z + beta(torch.tensor(0.5)) * x
print(f"   path point at t=0.5 / straight midpoint: {(mid / ((x + z) / 2)).mean():.4f} (sqrt 2 = {math.sqrt(2):.4f})")

# 2. True velocity of the path vs. the trainer target z - x
tt = torch.rand(10000, 1)
h = 1e-4
v_fd = ((alpha(tt + h) - alpha(tt - h)) * z + (beta(tt + h) - beta(tt - h)) * x) / (2 * h)
v_formula = 0.5 * torch.pi * (beta(tt) * z - alpha(tt) * x)
print(f"2. |finite-diff velocity - (pi/2)(cos z - sin x)| max: {(v_fd - v_formula).abs().max():.2e}")
print(f"   |finite-diff velocity - (z - x)| mean: {(v_fd - (z - x)).norm(dim=-1).mean():.3f}")

# 3. 1D Gaussian example: x ~ N(0, 1), z ~ N(0, 4), independent.
#    Best model for target T at time t: E[T | xt = y] = k(t) y (linear, all Gaussian). The ODE dy/dt = k(t) y gives std(1) = exp(int k).
sx2, sz2 = 1.0, 4.0
def ks(t):
    a, b = math.sin(math.pi * t / 2), math.cos(math.pi * t / 2)
    var = a * a * sz2 + b * b * sx2
    k_trainer = (a * sz2 - b * sx2) / var                                   # target z - x
    k_true = (math.pi / 2) * (b * a * sz2 - a * b * sx2) / var             # target alpha' z + beta' x
    return k_trainer, k_true
n = 100000
ints = [sum(ks((i + 0.5) / n)[j] for i in range(n)) / n for j in (0, 1)]
print(f"3. final std, target std = {math.sqrt(sz2):.3f}: trainer target z - x -> {math.exp(ints[0]):.3f}, true velocity -> {math.exp(ints[1]):.3f}")
# Monte Carlo check of k(0.3) by least squares
t0 = 0.3
xg, zg = torch.randn(1_000_000), 2 * torch.randn(1_000_000)
y = math.sin(math.pi * t0 / 2) * zg + math.cos(math.pi * t0 / 2) * xg
print(f"   Monte Carlo k_trainer(0.3) = {((zg - xg) * y).mean() / (y * y).mean():.4f}, formula = {ks(t0)[0]:.4f}")
