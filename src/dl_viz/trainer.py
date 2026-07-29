from typing import Optional
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dl_viz.data import get_cifar10_loaders

class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_data: DataLoader,
            val_data: DataLoader,
            test_data: DataLoader,
            criterion: nn.CrossEntropyLoss,
            optimizer: optim.AdamW,
            snapshot_path: str,
            num_epochs: int,
            scheduler: Optional[optim.lr_scheduler._LRScheduler]
    ):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.scheduler = scheduler

    def _load_snapshot(self):
        pass

    def _save_snapshot(self):
        pass

    def _run_batch(self, batch):
        self.optimizer.zero_grad()

        features = batch[0].to(self.device)
        targets = batch[1].to(self.device).long()
        outputs = self.model(features)

        loss = self.criterion(outputs, targets)

        loss.backward()
        self.optimizer.step()

    def _run_epoch(self):
        train_epoch_loss = torch.tensor(0.0, device=self.device)
        train_epoch_preds = 0

        for batch in tqdm(self.train_data, desc="Training.....", total=len(self.train_data)):
            batch_loss = self._run_batch(batch)

        pass

    def _run_eval(self):
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(self.val_data, desc="Eval", total=len(self.val_data)):
                features = batch[0].to(self.device)
                targets = batch[1].to(self.device).long()
                outputs = self.model(features)

                loss = self.criterion(outputs, targets)

    def train(self):
        best_loss = self._run_eval()

        for epoch in range(1, self.num_epochs+1):
            self.model.train()

            self._run_epoch()