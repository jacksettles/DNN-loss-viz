from typing import Any

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

from dl_viz.data import get_cifar10_loaders
from dl_viz.configs import CNNConfig
from dl_viz.models import MiniCNN

DATA_REGISTRY = {
    "cifar10": get_cifar10_loaders,
}


MODEL_REGISTRY = {
    "MiniCNN": {
        "model_class": MiniCNN,
        "config_class": CNNConfig,
    },
}

LOSS_REGISTRY = {
    "cross_entropy_loss": nn.CrossEntropyLoss()
}

OPTIMIZER_REGISTRY = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "RMSprop": optim.RMSprop,
}

SCHEDULER_REGISTRY: dict[str, type[LRScheduler]] = {
    "StepLR": optim.lr_scheduler.StepLR,
    "MultiStepLR": optim.lr_scheduler.MultiStepLR,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "ExponentialLR": optim.lr_scheduler.ExponentialLR,
}