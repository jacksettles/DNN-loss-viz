import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Sequence

@dataclass
class CNNConfig():


class MiniCNN(nn.Module):
    def __init__(self, CNNConfig):
        super(MiniCNN, self).__init__()


    def forward(self):