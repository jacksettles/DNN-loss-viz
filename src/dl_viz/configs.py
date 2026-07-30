from dataclasses import dataclass, field
from typing import Sequence, Literal


ActivationName = Literal[
    "relu",
    "leaky_relu",
    "tanh",
    "sigmoid",
    "gelu",
]

@dataclass
class ConvLayerConfig:
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 0
    use_batch_norm: bool = True
    use_max_pool: bool = True
    pool_kernel_size: int = 2
    use_skip_connection: bool = True

@dataclass
class CNNConfig:
    in_channels: int = 3
    num_classes: int = 10

    conv_layers: list[ConvLayerConfig] = field(
        default_factory=lambda: [
            ConvLayerConfig(out_channels=32, use_skip_connection=False),
            ConvLayerConfig(out_channels=64, use_skip_connection=False),
            ConvLayerConfig(out_channels=128, use_skip_connection=False)
        ]
    )

    classifier_hidden_dims: list[int] = field(
        default_factory=lambda: [128]
    )

    hidden_dim: int = 128
    dropout: float = 0.0

    activation: ActivationName = "relu"
    classifier_activation: ActivationName | None = None
    leaky_relu_slope: float = 0.01