from __future__ import annotations

import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

import sys
import torch
from typing import Any, TYPE_CHECKING
from pathlib import Path
import torch.cuda as t_cuda

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"

import yaml

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader


from dl_viz.trainer import Trainer
from dl_viz.data import DataConfig
from dl_viz.registries import DATA_REGISTRY, MODEL_REGISTRY, LOSS_REGISTRY, OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY


def load_configs(
    config_dir: Path = CONFIG_DIR,
) -> dict[str, dict]:
    configs: dict[str, dict] = {}

    for yaml_path in sorted(config_dir.glob("*.yaml")):
        with yaml_path.open("r", encoding="utf-8") as file:
            file_config = yaml.safe_load(file) or {}
        configs.update(file_config)
    return configs

def load_model(
    config: dict[str, Any],
    experiment_id: str,
) -> nn.Module:
    logger.info("Loading model object...")

    experiments = config.get("experiments", {})
    models = config.get("models", {})

    if experiment_id not in experiments:
        valid_experiments = ", ".join(experiments)
        raise ValueError(
            f"Unknown experiment {experiment_id!r}. "
            f"Expected one of: {valid_experiments}."
        )

    experiment_config = experiments[experiment_id]

    model_id = experiment_config.get("model_id")

    if model_id is None:
        raise ValueError(
            f"Experiment {experiment_id!r} must contain "
            f"a 'model_id' field."
        )

    if model_id not in models:
        valid_model_ids = ", ".join(models)
        raise ValueError(
            f"Unknown model ID {model_id!r}. "
            f"Expected one of: {valid_model_ids}."
        )

    model_dict = models[model_id]

    model_name = model_dict.get("name")
    model_params = model_dict.get("params", {})

    if model_name is None:
        raise ValueError(
            f"Model {model_id!r} must contain a 'name' field."
        )

    if model_name not in MODEL_REGISTRY:
        valid_models = ", ".join(MODEL_REGISTRY)
        raise ValueError(
            f"Unknown model class {model_name!r}. "
            f"Expected one of: {valid_models}."
        )

    registry_entry = MODEL_REGISTRY[model_name]

    model_class = registry_entry["model_class"]
    config_class = registry_entry["config_class"]

    if hasattr(config_class, "from_dict"):
        model_config = config_class.from_dict(model_params)
    else:
        model_config = config_class(**model_params)

    logger.info(
        "Using experiment %s -> model %s (%s)",
        experiment_id,
        model_id,
        model_name,
    )

    return model_class(model_config)

def load_data(config: dict)-> dict[str, DataLoader | None]:
    logger.info("Loading data...")
    raw_data_config = config.get("data", {})

    pin_mem = t_cuda.is_available() and raw_data_config.get("pin_memory", True)
    data_config = DataConfig(
        data_dir=DATA_DIR,
        batch_size=raw_data_config.get("batch_size", 128),
        num_workers=raw_data_config.get("num_workers", 2),
        pin_memory=pin_mem,
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
    logger.info("Loading %s from %s", data_name, data_config.data_dir)
    return DATA_REGISTRY[data_name](data_config)

def load_criterion(
    training_config: dict[str, Any],
) -> nn.Module:
    logger.info("Loading criterion...")
    criterion_config = training_config.get("criterion", {})
    criterion_name = criterion_config.get("name")
    criterion_params = criterion_config.get("params", {})

    if criterion_name is None:
        raise ValueError(
            "Training config must contain a "
            "'training.criterion.name' field."
        )

    if criterion_name not in LOSS_REGISTRY:
        valid_criteria = ", ".join(LOSS_REGISTRY)
        raise ValueError(
            f"Unknown criterion/loss function {criterion_name!r}. "
            f"Expected one of: {valid_criteria}."
        )

    criterion_class = LOSS_REGISTRY[criterion_name]

    return criterion_class(**criterion_params)


def load_optimizer(
    training_config: dict[str, Any],
    model: nn.Module,
) -> Optimizer:
    logger.info("Loading optimizer...")
    optimizer_config = training_config.get("optimizer", {})
    optimizer_name = optimizer_config.get("name")
    optimizer_params = optimizer_config.get("params", {})

    if optimizer_name is None:
        raise ValueError(
            "Training config must contain a "
            "'training.optimizer.name' field."
        )

    if optimizer_name not in OPTIMIZER_REGISTRY:
        valid_optimizers = ", ".join(OPTIMIZER_REGISTRY)
        raise ValueError(
            f"Unknown optimizer {optimizer_name!r}. "
            f"Expected one of: {valid_optimizers}."
        )

    optimizer_class = OPTIMIZER_REGISTRY[optimizer_name]

    return optimizer_class(
        model.parameters(),
        **optimizer_params,
    )

def load_scheduler(
    training_config: dict[str, Any],
    optimizer: Optimizer,
) -> LRScheduler | None:
    logger.info("Loading LR scheduler...")
    scheduler_config = training_config.get("scheduler")

    if scheduler_config is None:
        return None

    scheduler_name = scheduler_config.get("name")
    scheduler_params = scheduler_config.get("params", {})

    if scheduler_name is None:
        return None

    if scheduler_name not in SCHEDULER_REGISTRY:
        valid_schedulers = ", ".join(SCHEDULER_REGISTRY)
        raise ValueError(
            f"Unknown scheduler {scheduler_name!r}. "
            f"Expected one of: {valid_schedulers}."
        )

    scheduler_class = SCHEDULER_REGISTRY[scheduler_name]

    return scheduler_class(
        optimizer,
        **scheduler_params,
    )


def load_runner_dict(
    config: dict[str, Any],
    experiment_id: str,
) -> dict[str, Any]:
    experiment_name = config.get("experiments", {}).get(experiment_id, {}).get("name")
    if experiment_name is None:
        raise ValueError(
            f"The value in the 'name' field for experiment_id: {experiment_id} came back as None",
            f"Please provide a value in the 'name' field for experiment_id: {experiment_id}"
        )
    
    model = load_model(
        config=config,
        experiment_id=experiment_id,
    )

    data = load_data(config)

    train_data = data["train_data"]
    val_data = data["val_data"]
    test_data = data["test_data"]

    training_config = config.get("training", {})

    criterion = load_criterion(training_config)
    optimizer = load_optimizer(training_config, model)
    scheduler = load_scheduler(
        training_config,
        optimizer,
    )

    checkpoint_config = config.get(
        "checkpointing",
        {},
    )

    checkpoint_dir = (
        PROJECT_ROOT
        / checkpoint_config.get(
            "directory",
            "checkpoints",
        )
    )

    num_epochs = training_config.get(
        "num_epochs",
        10,
    )

    return {
        "model": model,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "snapshot_path": checkpoint_dir,
        "num_epochs": num_epochs,
        "experiment_name": experiment_name,
    }

def main(
    config: dict[str, Any],
    experiment_id: str,
):
    logger.info("Loading experiment %s", experiment_id)

    runner_dict = load_runner_dict(
        config=config,
        experiment_id=experiment_id,
    )

    logger.info("Building trainer object...")
    trainer = Trainer(**runner_dict)

    logger.info("Commencing training...")
    trainer.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one or more training experiments."
    )

    parser.add_argument(
        "experiment_ids",
        nargs="+",
        help="Experiment IDs to run, or 'all' to run every configured experiment.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_configs()

    if "all" in args.experiment_ids and args.experiment_ids != ["all"]:
        raise ValueError(
            "'all' cannot be combined with individual experiment IDs."
        )

    if args.experiment_ids == ["all"]:
        experiment_ids = list(
            config.get("experiments", {}).keys()
        )
    else:
        experiment_ids = args.experiment_ids

    for experiment_id in experiment_ids:
        main(
            config=config,
            experiment_id=experiment_id,
        )