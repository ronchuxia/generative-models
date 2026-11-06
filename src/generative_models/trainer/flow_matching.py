from .trainer import Trainer
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm


class GaussianOTFlowMatchingTrainer(Trainer):
    def __init__(self, model: nn.Module, dataset: Dataset, config: dict) -> None:
        super().__init__(model, dataset, config)

    def train_iter(self, batch: tuple, progress: tqdm) -> None:
        """
        Train one iteration.

        Args:
            batch: (z,). A batch of data points z with shape (b, ...).
        """
        z = batch[0]
        b = z.shape[0]
        dim = len(z.shape)

        z = z.to(self.device)

        # Sample t
        t = torch.rand((b,), device=self.device).reshape((b,) + (1,) * (dim - 1))   # shape (b, 1, 1, ...)

        # Sample xt
        epsilon = torch.randn_like(z)   # shape (b, ...)
        xt = t * z + (1 - t) * epsilon  # shape (b, ...)

        # Predict ut_pred(xt)
        ut_pred = self.model(xt, t)

        # Compute ut_cond(xt|zt)
        ut_cond = z - epsilon

        # Compute loss
        loss = torch.sum((ut_pred - ut_cond)**2) / b

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log
        progress.set_postfix(loss=f"{loss.item():.2f}")