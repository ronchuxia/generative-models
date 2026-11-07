"""Diagnose why OT flow matching (moons -> circles) gives blurry circles.
Each experiment trains a model, then samples with 100 and 1000 uniform Euler steps, and 100 Euler steps that are smaller near t = 1.
Metrics vs a new circles sample (lower is better):
  radius W1: Wasserstein-1 distance of the radius distributions (ring sharpness).
  sliced W1: mean 1D Wasserstein-1 distance over random projections (whole 2D distribution).
Usage: [SEED=n] python diagnose_ot_fm_circles.py OUT_DIR EXP [EXP ...]   (SEED changes the training seed, default 0)
"""
import os
import sys
import json
os.environ["WANDB_MODE"] = "disabled"

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from generative_models.data import MoonsCirclesDataset, MoonsDataset, CirclesDataset
from generative_models.trainer import OTFlowMatchingTrainer
from generative_models.model import FlowMatchingMLP
from generative_models.de import NeuralODE, Euler


class CosineLR:
    def train_iter(self, batch, progress, step):
        if step == 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.total_steps)
        super().train_iter(batch, progress, step)
        self.scheduler.step()


class CosineLRTrainer(CosineLR, OTFlowMatchingTrainer):
    pass


class TSkewTrainer(OTFlowMatchingTrainer):
    """Same as OTFlowMatchingTrainer, but t = 1 - (1 - u)^2, so more samples near t = 1."""
    def train_iter(self, batch, progress, step):
        x, z = batch[0].to(self.device), batch[1].to(self.device)
        t = 1 - (1 - torch.rand((z.shape[0], 1), device=self.device)) ** 2
        xt = t * z + (1 - t) * x
        loss = torch.sum((self.model(xt, t) - (z - x)) ** 2) / z.shape[0]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class CosineTSkewTrainer(CosineLR, TSkewTrainer):
    pass


class MinibatchOTTrainer(OTFlowMatchingTrainer):
    """Pair x and z by exact OT inside each OT sub-batch of 250 (Hungarian on 1000 points takes 0.3 s)."""
    def train_iter(self, batch, progress, step):
        x, z = batch
        z = torch.cat([zc[linear_sum_assignment(torch.cdist(xc, zc).pow(2).numpy())[1]]
                       for xc, zc in zip(x.split(250), z.split(250))])
        super().train_iter((x, z), progress, step)


class CosineMinibatchOTTrainer(CosineLR, MinibatchOTTrainer):
    pass


EXPERIMENTS = {
    # name: (trainer class, hidden features, total steps)
    'base': (OTFlowMatchingTrainer, [64, 64, 64, 2], 10000),
    'cosine': (CosineLRTrainer, [64, 64, 64, 2], 10000),
    'long': (OTFlowMatchingTrainer, [64, 64, 64, 2], 50000),
    'big': (OTFlowMatchingTrainer, [256, 256, 256, 2], 10000),
    'long_cosine': (CosineLRTrainer, [64, 64, 64, 2], 50000),
    'big_cosine': (CosineLRTrainer, [256, 256, 256, 2], 10000),
    'big_long': (OTFlowMatchingTrainer, [256, 256, 256, 2], 50000),
    'big_long_cosine':(CosineLRTrainer, [256, 256, 256, 2], 50000),
    'ot': (MinibatchOTTrainer, [64, 64, 64, 2], 10000),
    'ot_cosine': (CosineMinibatchOTTrainer, [64, 64, 64, 2], 10000),
    'cosine_tskew': (CosineTSkewTrainer, [64, 64, 64, 2], 10000),
    'big_long_cosine_tskew': (CosineTSkewTrainer, [256, 256, 256, 2], 50000),
}


def radius_w1(a, c):
    return (a.norm(dim=1).sort().values - c.norm(dim=1).sort().values).abs().mean().item()


def sliced_w1(a, c, k=256):
    d = torch.randn(2, k, generator=torch.Generator().manual_seed(0)).to(a.device)
    d = d / d.norm(dim=0)
    return ((a @ d).sort(0).values - (c @ d).sort(0).values).abs().mean().item()


out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
n = 20000

seed = int(os.environ.get('SEED', 0))  # training seed; evaluation samples stay fixed
for name in sys.argv[2:]:
    trainer_cls, hidden, steps = EXPERIMENTS[name]
    tag = name if seed == 0 else f'{name}_seed{seed}'
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = {'lr': 1e-3, 'batch_size': 1000, 'total_steps': steps, 'log': False, 'project': 'x', 'name': name}
    dataset = MoonsCirclesDataset(n_samples=100000, noise_moons=0.05, noise_circles=0.04)
    model = FlowMatchingMLP(2, hidden)
    trainer = trainer_cls(model, dataset, config)
    trainer.train()
    model.eval()
    device = trainer.device
    torch.save(model.state_dict(), os.path.join(out_dir, f'{tag}_model.pt'))

    np.random.seed(1)
    x0 = MoonsDataset(n_samples=n, noise=0.05).x.to(device)
    ref = CirclesDataset(n_samples=n, noise=0.04).x.to(device)
    ref2 = CirclesDataset(n_samples=n, noise=0.04).x.to(device)

    metrics = {'floor': {'radius_w1': radius_w1(ref2, ref), 'sliced_w1': sliced_w1(ref2, ref)}}
    s = torch.linspace(0, 1, 101, device=device)
    grids = {
        'euler_100': s,
        'euler_1000': torch.linspace(0, 1, 1001, device=device),
        'euler_100_late': 1 - (1 - s) ** 2,   # smaller steps near t = 1
    }
    for g, t in grids.items():
        with torch.no_grad():
            x1 = Euler(ode=NeuralODE(model)).simulate_trajectory(x0, t[None, :].expand(n, -1))
        metrics[g] = {'radius_w1': radius_w1(x1, ref), 'sliced_w1': sliced_w1(x1, ref)}
        torch.save(x1.cpu(), os.path.join(out_dir, f'{tag}_{g}.pt'))
    torch.save(ref.cpu(), os.path.join(out_dir, 'ref.pt'))
    json.dump(metrics, open(os.path.join(out_dir, f'{tag}.json'), 'w'), indent=1)
    print(tag, json.dumps(metrics), flush=True)
