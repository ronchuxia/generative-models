"""Check: OT flow matching (moons -> circles), model with t vs model without t (fake_t = 0).
1. Loss on fresh samples.
2. Binned check: no-t model ~= E[z - x | xt] (target averaged over t); with-t model varies with t inside a bin.
3. Loss gap vs E||u(xt, t) - v(xt)||^2.
4. Generation quality: Chamfer distance of generated points to circles.
5. Mean radius along the trajectory.
"""
import os
os.environ["WANDB_MODE"] = "disabled"

import torch
from generative_models.data import MoonsCirclesDataset, MoonsDataset, CirclesDataset
from generative_models.trainer import OTFlowMatchingTrainer
from generative_models.model import FlowMatchingMLP
from generative_models.de import NeuralODE, Euler


class FlowMatchingMLPNoT(FlowMatchingMLP):
    def forward(self, x, t):
        fake_t = torch.zeros_like(t)
        return self.mlp(torch.concat([x, fake_t], dim=1))


torch.manual_seed(0)
config = {'lr': 1e-3, 'batch_size': 1000, 'total_steps': 10000, 'log': False, 'project': 'x', 'name': 'x'}
dataset = MoonsCirclesDataset(n_samples=100000, noise_moons=0.05, noise_circles=0.04)
models = {'with_t': FlowMatchingMLP(2, [64, 64, 64, 2]), 'no_t': FlowMatchingMLPNoT(2, [64, 64, 64, 2])}
for m in models.values():
    trainer = OTFlowMatchingTrainer(m, dataset, config)
    trainer.train()
    m.eval()
device = trainer.device

# Fresh independent pairs, same as the trainer
n = 400000
x = MoonsDataset(n_samples=n, noise=0.05).x.to(device)
z = CirclesDataset(n_samples=n, noise=0.04).x.to(device)
t = torch.rand((n, 1), device=device)
xt = t * z + (1 - t) * x
target = z - x
with torch.no_grad():
    u = models['with_t'](xt, t)
    v = models['no_t'](xt, t)

print("\n[1] loss on fresh samples")
print("  with t:", ((u - target) ** 2).sum(1).mean().item())
print("  no t:  ", ((v - target) ** 2).sum(1).mean().item())

print("\n[2] binned check (40x40 grid, bins with >= 200 samples)")
k = 40
lo, hi = xt.min(0).values, xt.max(0).values
ij = ((xt - lo) / (hi - lo) * k).long().clamp(max=k - 1)
bin_id = ij[:, 0] * k + ij[:, 1]
count = torch.bincount(bin_id, minlength=k * k).float()
def bin_mean(a):
    s = torch.zeros((k * k, 2), device=device).index_add_(0, bin_id, a)
    return s / count[:, None].clamp(min=1)
keep = count >= 200
w = count[keep] / count[keep].sum()
mean_target, mean_u, mean_v = bin_mean(target)[keep], bin_mean(u)[keep], bin_mean(v)[keep]
var_u = (bin_mean(u ** 2) - bin_mean(u) ** 2)[keep].sum(1)
var_v = (bin_mean(v ** 2) - bin_mean(v) ** 2)[keep].sum(1)
print("  bins used:", keep.sum().item(), " samples covered:", count[keep].sum().item() / n)
print("  mean |target| per bin:                    ", (w * mean_target.norm(dim=1)).sum().item())
print("  mean |bin avg of v(xt) - bin avg target|:  ", (w * (mean_v - mean_target).norm(dim=1)).sum().item())
print("  mean |bin avg of u(xt,t) - bin avg target|:", (w * (mean_u - mean_target).norm(dim=1)).sum().item())
print("  spread of u(xt,t) inside a bin (std):      ", (w * var_u.clamp(min=0).sqrt()).sum().item())
print("  spread of v(xt) inside a bin (std):        ", (w * var_v.clamp(min=0).sqrt()).sum().item())

print("\n[3] loss gap vs E|u(xt,t) - v(xt)|^2")
print("  loss(no t) - loss(with t):", (((v - target) ** 2).sum(1) - ((u - target) ** 2).sum(1)).mean().item())
print("  E|u - v|^2:               ", ((u - v) ** 2).sum(1).mean().item())

print("\n[4] generation, 100 Euler steps from moons. Chamfer distance to circles (lower is better)")
b, l = 5000, 100
x0 = MoonsDataset(n_samples=b, noise=0.05).x.to(device)
ref = CirclesDataset(n_samples=b, noise=0.04).x.to(device)
ref2 = CirclesDataset(n_samples=b, noise=0.04).x.to(device)
def chamfer(a, c):
    d = torch.cdist(a, c)
    return (d.min(1).values.mean() + d.min(0).values.mean()).item() / 2
tt = torch.linspace(0, 1, l, device=device)[None, :].expand(b, l)
traj = {}
for name, m in models.items():
    with torch.no_grad():
        traj[name] = Euler(ode=NeuralODE(m)).simulate_trajectory(x0, tt, whole=True)
    print(f"  {name}: {chamfer(traj[name][:, -1], ref):.3f}")
print(f"  floor (circles vs circles):        {chamfer(ref2, ref):.3f}")
print(f"  no movement (moons vs circles):    {chamfer(x0, ref):.3f}")

print("\n[5] mean radius |x| along the trajectory (circles radii are 1.0 and 0.8)")
for name in models:
    print(f"  {name}:", [round(traj[name][:, i].norm(dim=1).mean().item(), 3) for i in [0, 20, 40, 50, 60, 80, 99]],
          " at t =", [round(tt[0, i].item(), 2) for i in [0, 20, 40, 50, 60, 80, 99]])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (title, pts) in zip(axes, [("circles (target)", ref), ("with t, t=1", traj['with_t'][:, -1]), ("no t, t=1", traj['no_t'][:, -1])]):
    pts = pts.cpu()
    ax.scatter(pts[:, 0], pts[:, 1], s=0.5)
    ax.set_title(title)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
fig.savefig(os.environ.get("OUT_PNG", "check_ot_fm_no_t.png"))
