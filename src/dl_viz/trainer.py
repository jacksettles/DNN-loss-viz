from typing import Optional
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import pandas as pd
from datetime import datetime


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs_run = 0
        self.num_epochs = num_epochs
        self.snapshot_path = snapshot_path
        self.scheduler = scheduler
        self.run_name = datetime.now().strftime("model_%Y-%m-%d")

    def _load_snapshot(self):
        pass

    def _save_snapshot(self,
                       metrics: list[dict],
                       epoch: int):
        models_dir = self.snapshot_path / "models"
        metrics_dir = self.snapshot_path / "metrics"
        latest_dir = self.snapshot_path / "latest"

        models_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        latest_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / f"{self.run_name}.pt"
        metrics_path = metrics_dir / f"{self.run_name}.csv"

        latest_model_path = latest_dir / f"{self.run_name}.pt"
        latest_metrics_path = latest_dir / f"{self.run_name}.csv"

        metrics_df = pd.DataFrame(metrics)
        snapshot = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if self.scheduler is not None:
            snapshot["scheduler_state_dict"] = (
                self.scheduler.state_dict()
            )

        # Overwrites this run's existing checkpoint every epoch
        torch.save(snapshot, model_path)
        metrics_df.to_csv(metrics_path, index=False)

        # Remove files belonging to an older run
        for existing_file in latest_dir.iterdir():
            if existing_file.is_file():
                existing_file.unlink()

        # Save current run as latest
        torch.save(snapshot, latest_model_path)
        metrics_df.to_csv(latest_metrics_path, index=False)

    def _run_batch(self, batch):
        self.optimizer.zero_grad()

        features = batch[0].to(self.device)
        targets = batch[1].to(self.device).long()
        outputs = self.model(features)

        loss = self.criterion(outputs, targets)

        batch_size = targets.size(0)

        total_loss = loss.item() * batch_size
        correct = (outputs.argmax(dim=1) == targets).sum().item()
        total = batch_size

        loss.backward()
        self.optimizer.step()
        return total_loss, correct, total

    def _run_epoch(self, current_epoch):
        running_loss = torch.tensor(0.0, device=self.device)
        correct = 0
        total = 0

        for batch in tqdm(self.train_data, desc=f"Training Epoch {current_epoch}.....", total=len(self.train_data)):
            batch_loss, batch_correct, batch_total = self._run_batch(batch)

            running_loss += batch_loss
            correct += batch_correct
            total += batch_total

        return {
            "epoch": current_epoch,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "training_loss": (running_loss / total).item(),
            "training_accuracy": correct / total,
            "training_num_samples": total,
        }

    def _evaluate_loader(
        self,
        data: DataLoader,
        desc: str,
    ):
        self.model.eval()

        running_loss = torch.tensor(0.0, device=self.device)
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(data, desc=desc, total=len(data)):
                features = batch[0].to(self.device)
                targets = batch[1].to(self.device).long()

                outputs = self.model(features)
                loss = self.criterion(outputs, targets)

                batch_size = targets.size(0)

                running_loss += loss.item() * batch_size
                correct += (
                    outputs.argmax(dim=1) == targets
                ).sum().item()
                total += batch_size

        return {
            "loss": (running_loss / total).item(),
            "accuracy": correct / total,
            "num_samples": total,
        }

    def _run_eval(self):
        eval_data = self.val_data if self.val_data else self.test_data
        eval_data_name = "Validation" if self.val_data else "Test"

        metrics = self._evaluate_loader(
            data=eval_data,
            desc=f"Eval on {eval_data_name}",
        )

        return {
            "eval_loss": metrics["loss"],
            "eval_accuracy": metrics["accuracy"],
            "eval_num_samples": metrics["num_samples"],
        }

    def train(self):
        baseline_metrics = {
            "epoch": 0,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "training_loss": np.nan,
            "training_accuracy": np.nan,
            "training_num_samples": 0,
        }
        baseline_metrics = baseline_metrics | self._run_eval()
        baseline_metrics["stage"] = "pre_training"

        metric_dicts: list[dict] = [baseline_metrics]

        for epoch in range(1, self.num_epochs+1):
            self.model.train() # set model to train mode for grad tracking

            train_metrics = self._run_epoch(epoch)
            eval_metrics = self._run_eval()

            epoch_metrics = train_metrics | eval_metrics
            epoch_metrics['stage'] = "training"
            metric_dicts.append(epoch_metrics)

            self._save_snapshot(metric_dicts, epoch)