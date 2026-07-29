
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

import yaml

import torch.optim as optim

from dl_viz.trainer import Trainer
from dl_viz.data import DataConfig, get_cifar10_loaders
import dl_viz.models as dvm

def load_configs(
    config_dir: Path = CONFIG_DIR,
) -> dict[str, dict]:
    configs: dict[str, dict] = {}

    for yaml_path in sorted(config_dir.glob("*.yaml")):
        with yaml_path.open("r", encoding="utf-8") as file:
            configs[yaml_path.stem] = yaml.safe_load(file) or {}
    return configs


def load_runner_dict(args):

    model = dvm.MiniCNN()

    data_config = DataConfig()
    train_data, test_data = get_cifar10_loaders(data_config)
    

    return {
        "model": model,
        "train_data": train_data, 
        "val_data": None,
        "test_data": test_data,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "snapshot_path": snapshot_path,
        "num_epochs": num_epochs
    }

def main(config):
    runner_dict = load_runner_dict(args)
    trainer = Trainer(
        model = model,
        train_data = train_data,
        val_data = val_data,
        criterion = criterion,
        optimizer = optimizer,
        scheduler = scheduler,
        snapshot_path = args.save_path,
        num_epochs = args.num_epochs,
    )
    trainer.train(args)


if __name__ == "__main__":
    config = load_configs()
    main(config)