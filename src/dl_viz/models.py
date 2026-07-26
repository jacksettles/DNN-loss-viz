import torch
import torch.nn as nn
from dl_viz.configs import CNNConfig, ActivationName


def build_activation(
    name: ActivationName,
    *,
    leaky_relu_slope: float = 0.01,
) -> nn.Module:
    activations: dict[str, nn.Module] = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(
            negative_slope=leaky_relu_slope
        ),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "gelu": nn.GELU(),
    }

    try:
        return activations[name]
    except KeyError as exc:
        valid_names = ", ".join(activations)
        raise ValueError(
            f"Unknown activation {name!r}. "
            f"Expected one of: {valid_names}."
        ) from exc


class MiniCNN(nn.Module):
    def __init__(self, config: CNNConfig) -> None:
        super(MiniCNN, self).__init__()

        self.config = config
        self.features = self._build_conv_layers(config)

        self.avg_pool = nn.AvgPool2d((1,1))
        final_channels = config.conv_layers[-1].out_channels

        self.classifier = self._build_classifier(
            input_dim=final_channels,
            config=config,
        )


    @staticmethod
    def _build_conv_layers(self, config: CNNConfig) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_channels = config.in_channels
        for layer_config in config.conv_layers:
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=layer_config.out_channels,
                    kernel_size=layer_config.kernel_size,
                    stride=layer_config.stride,
                    padding=layer_config.padding,
                    bias=not layer_config.use_batch_norm,
                )
            )

            if layer_config.use_batch_norm:
                layers.append(
                    nn.BatchNorm2d(layer_config.out_channels)
                )
            
            layers.append(
                build_activation(
                    name=config.activation,
                    leaky_relu_slope=config.leaky_relu_slope,
                )
            )

            if layer_config.use_max_pool:
                layers.append(
                    nn.MaxPool2d(kernel_size=layer_config.pool_kernel_size)
                )

            in_channels = layer_config.out_channels
        return nn.Sequential(*layers)
    
    @staticmethod
    def _build_classifier(self, 
                          input_dim: int,
                          config: CNNConfig) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Flatten()]

        activation_name = (
            config.classifier_activation
            if config.classifier_activation is not None
            else config.activation
        )

        current_dim = input_dim

        for hidden_dim in config.classifier_hidden_dims:
            layers.append(
                nn.Linear(
                    in_features=current_dim,
                    out_features=hidden_dim
                )
            )

            layers.append(
                build_activation(
                    activation_name,
                    leaky_relu_slope=config.leaky_relu_slope,
                )
            )

            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            
            current_dim = hidden_dim
        layers.append(
            nn.Linear(
                in_features=current_dim,
                out_features=config.num_classes
            )
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avg_pool(x)
        return self.classifier(x)