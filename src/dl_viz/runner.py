
import argparse
import pyyaml

import torch.optim as optim

from dl_viz.trainer import Trainer
from dl_viz.data import DataConfig, get_cifar10_loaders
import dl_viz.models as dvm

parser = argparse.ArgumentParser()

parser.add_argument()

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

def main(args):
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
    args = parser.parse_args()
    main(args)