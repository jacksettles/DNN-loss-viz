from dataclasses import dataclass, field
from typing import Any, Literal  # CHANGED: removed Sequence, added Any


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

    padding: int = 1

    use_batch_norm: bool = True
    use_max_pool: bool = True
    pool_kernel_size: int = 2

    use_skip_connection: bool = False


@dataclass
class CNNConfig:
    in_channels: int = 3
    num_classes: int = 10

    conv_layers: list[ConvLayerConfig] = field(
        default_factory=lambda: [
            ConvLayerConfig(out_channels=32),
            ConvLayerConfig(out_channels=64),
            ConvLayerConfig(out_channels=128),
        ]
    )

    classifier_hidden_dims: list[int] = field(
        default_factory=lambda: [128]
    )

    dropout: float = 0.0

    activation: ActivationName = "relu"
    classifier_activation: ActivationName | None = None
    leaky_relu_slope: float = 0.01

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
    ) -> "CNNConfig":
        config = config.copy()

        if "conv_channels" in config:
            conv_channels = config.pop("conv_channels")
            conv_defaults = config.pop("conv_defaults", {})

            config["conv_layers"] = [
                ConvLayerConfig(
                    out_channels=out_channels,
                    **conv_defaults,
                )
                for out_channels in conv_channels
            ]

        elif "conv_layers" in config:
            config["conv_layers"] = [
                ConvLayerConfig(**layer_config)
                for layer_config in config["conv_layers"]
            ]

        return cls(**config)