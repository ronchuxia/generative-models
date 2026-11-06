import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from abc import ABC, abstractmethod
from tqdm import tqdm
import wandb


class Trainer(ABC):
    def __init__(self, model: nn.Module, dataset: Dataset, config: dict) -> None:        
        self.model = model
        self.dataset = dataset
        self.config = config

        # Parse configs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = self.config['batch_size']
        self.lr = self.config['lr']
        self.total_steps = self.config['total_steps']
        self.log = self.config['log']
        self.project = self.config['project']
        self.name = self.config['name']

        # Prepare model, dataloader and optimizer
        self.model.to(self.device)

        self.dataloader = DataLoader(self.dataset, self.batch_size, shuffle=True)

        self.optimizer = AdamW(self.model.parameters(), self.lr)

        # Initialize wandb
        if self.log:
            wandb.init(project=self.project, name=self.name, config=self.config)

    def train(self) -> None:
        self.model.train()

        dataloader_iter = iter(self.dataloader)
        progress = tqdm(range(self.total_steps))
        for step in progress:
            # Get a batch of data
            try:
                batch = next(dataloader_iter)
            except:
                dataloader_iter = iter(self.dataloader)
                batch = next(dataloader_iter)

            self.train_iter(batch, progress, step)

    @abstractmethod
    def train_iter(self, batch: tuple, progress: tqdm, step: int) -> None:
        pass