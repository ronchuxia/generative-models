"""Check: training with shifted noise levels (SD3 shift) vs uniform noise levels with a loss weight.
Setup as GaussianOTFlowMatchingTrainer (noise -> moons): xt = t z + (1 - t) eps, target z - eps. sigma = 1 - t.
A: sigma = s u / (1 + (s - 1) u), u ~ U(0, 1), plain loss.
B: sigma ~ U(0, 1), loss * q(sigma), q(sigma) = s / (s - (s - 1) sigma)^2 = density of the shifted sigma.
Same expected loss and gradient. Compare the mean gradients and the variance of one batch gradient.
"""
import numpy as np
import torch
from generative_models.data import MoonsDataset
from generative_models.model import FlowMatchingMLP

torch.manual_seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
s, b, k = 3.0, 1000, 2000
data = MoonsDataset(n_samples=100000, noise=0.05).x.to(device)
model = FlowMatchingMLP(2, [64, 64, 64, 2]).to(device)


def batch_loss(mode):
    z = data[torch.randint(len(data), (b,), device=device)]
    eps = torch.randn_like(z)
    u = torch.rand((b, 1), device=device)
    if mode == 'A':
        sigma, w = s * u / (1 + (s - 1) * u), torch.ones_like(u)
    else:
        sigma, w = u, s / (s - (s - 1) * u) ** 2
    t = 1 - sigma
    xt = t * z + (1 - t) * eps
    return (w * (model(xt, t) - (z - eps)) ** 2).sum() / b


def grad_stats(mode):
    losses, grads = [], []
    for _ in range(k):
        model.zero_grad()
        loss = batch_loss(mode)
        loss.backward()
        losses.append(loss.item())
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
    g = torch.stack(grads)
    return np.mean(losses), np.std(losses) / np.sqrt(k), g.mean(0), g.var(0).sum().item()


def report(title):
    la, sea, ga, va = grad_stats('A')
    lb, seb, gb, vb = grad_stats('B')
    noise = np.sqrt((va + vb) / k)  # expected size of |mean A - mean B| if the true means are equal
    print(f"\n[{title}]")
    print(f"  mean loss               A: {la:.4f} +- {sea:.4f}   B: {lb:.4f} +- {seb:.4f}")
    print(f"  |mean grad A|: {ga.norm().item():.4f}   |mean grad A - mean grad B|: {(ga - gb).norm().item():.4f}   (noise level: {noise:.4f})")
    print(f"  cosine(mean grad A, mean grad B): {torch.nn.functional.cosine_similarity(ga, gb, dim=0).item():.4f}")
    print(f"  variance of one batch gradient  A: {va:.4f}   B: {vb:.4f}   ratio B/A: {vb / va:.2f}")


report("random init")
opt = torch.optim.AdamW(model.parameters(), 1e-3)
for _ in range(3000):
    opt.zero_grad()
    batch_loss('A').backward()
    opt.step()
report("after 3000 training steps with A")
