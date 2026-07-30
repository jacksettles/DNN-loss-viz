
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"

import yaml

import torch.optim as optim
import torch.nn as nn

from dl_viz.trainer import Trainer
from dl_viz.data import DataConfig, get_cifar10_loaders

import dl_viz.models as dvm
from dl_viz.configs import ConvLayerConfig, CNNConfig, ActivationName
from dl_viz.registries import DATA_REGISTRY, MODEL_REGISTRY, LOSS_REGISTRY, OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY


def load_configs(
    config_dir: Path = CONFIG_DIR,
) -> dict[str, dict]:
    configs: dict[str, dict] = {}

    for yaml_path in sorted(config_dir.glob("*.yaml")):
        with yaml_path.open("r", encoding="utf-8") as file:
            configs[yaml_path.stem] = yaml.safe_load(file) or {}
    return configs

def load_model(config: dict[str, Any]) -> nn.Module:
    model_config = CNNConfig(
        **config.get("model", {}).get("params", {})
    )

    model_name = model_config.get("name")

    if model_name is None:
        raise ValueError(
            "Model config must contain a 'name' field."
        )

    if model_name not in MODEL_REGISTRY:
        valid_models = ", ".join(MODEL_REGISTRY)
        raise ValueError(
            f"Unknown model {model_name!r}. "
            f"Expected one of: {valid_models}."
        )

    registry_entry = MODEL_REGISTRY[model_name]

    model_class = registry_entry["model_class"]
    config_class = registry_entry["config_class"]

    model_params = model_config.get("params", {})

    if hasattr(config_class, "from_dict"):
        model_config = config_class.from_dict(model_params)
    else:
        model_config = config_class(**model_params)

    return model_class(model_config)

def load_data(config: dict):
    raw_data_config = config.get("data", {})

    data_config = DataConfig(
        data_dir=DATA_DIR,
        batch_size=raw_data_config.get("batch_size", 128),
        num_workers=raw_data_config.get("num_workers", 2),
        pin_memory=raw_data_config.get("pin_memory", True),
        download=raw_data_config.get("download", True),
        use_augmentation=raw_data_config.get(
            "use_augmentation",
            True,
        ),
    )

    data_name = raw_data_config.get("dataset")
    if data_name is None:
        raise ValueError(
            "Data config must contain a 'dataset' field."
        )

    if data_name not in DATA_REGISTRY:
        valid_datasets = ", ".join(DATA_REGISTRY)
        raise ValueError(
            f"Unknown dataset {data_name!r}. "
            f"Expected one of: {valid_datasets}."
        )
    return DATA_REGISTRY[data_name](data_config)

def load_criterion(training_config: dict):
    criterion_name = training_config.get("criterion", {}).get("name", None)

    if criterion_name is None:
        raise ValueError(
            "Training config must contain a 'criterion' field"
        )
    
    if criterion_name not in LOSS_REGISTRY:
        valid_criteria = ", ".join(LOSS_REGISTRY)
        raise ValueError(
            f"Unknown criterion/loss function {criterion_name!r}. "
            f"Expected one of: {valid_criteria}"
        )

    return LOSS_REGISTRY[criterion_name]


def load_optimizer(training_config: dict, model: nn.Module):
    optim_name = training_config.get("optimizer", {}).get("name", None)
    optim_params = training_config.get("optimizer", {}).get("params", {})

    if optim_name is None:
        raise ValueError(
            "Training config must contain a 'criterion' field"
        )
    
    if optim_name not in OPTIMIZER_REGISTRY:
        valid_optims = ", ".join(OPTIMIZER_REGISTRY)
        raise ValueError(
            f"Unknown optimizer {optim_name!r}. "
            f"Expected one of: {valid_optims}"
        )

    return OPTIMIZER_REGISTRY[optim_name](model.parameters(), **optim_params)

def load_scheduler(
    training_config: dict[str, Any] | None,
    optimizer: optim.Optimizer,
) -> LRScheduler | None:
    scheduler_name = training_config.get("scheduler", {}).get("name", None)
    scheduler_params = training_config.get("scheduler", {}).get("params", {})

    if scheduler_name is None:
        raise ValueError(
            "Training config must contain a 'scheduler' field"
        )
    
    if scheduler_name not in SCHEDULER_REGISTRY:
        valid_schedulers = ", ".join(SCHEDULER_REGISTRY)
        raise ValueError(
            f"Unknown scheduler {scheduler_name!r}. "
            f"Expected one of: {valid_schedulers}"
        )

    return SCHEDULER_REGISTRY[scheduler_name](optimizer, **scheduler_params)


def load_runner_dict(
    config: dict[str, Any],
) -> dict[str, Any]:

    model = load_model(config)
    data = load_data(config)
    train_data = data["train_data"]
    val_data = data["val_data"]
    test_data = data["test_data"]

    training_config = config.get("training", {})

    criterion = load_criterion(training_config)
    optimizer = load_optimizer(training_config, model)

    scheduler = load_scheduler(training_config, optimizer)

    checkpoint_config = config.get("checkpointing", {})
    checkpoint_dir = PROJECT_ROOT / checkpoint_config.get("directory", "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = checkpoint_dir / "latest.pt"
    num_epochs = training_config.get("num_epochs", 10)
    return {
        "model": model,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "snapshot_path": snapshot_path,
        "num_epochs": num_epochs,
    }

def main(config):
    runner_dict = load_runner_dict(config)
    trainer = Trainer(**runner_dict)
    trainer.train(config)


if __name__ == "__main__":
    config = load_configs()
    main(config)