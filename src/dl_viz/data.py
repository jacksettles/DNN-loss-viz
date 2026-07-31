from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class DataConfig:
    data_dir: str = "data"
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True
    download: bool = True
    use_augmentation: bool = True


def build_transforms(
    use_augmentation: bool = True,
) -> tuple[transforms.Compose, transforms.Compose]:
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cifar10_mean,
                std=cifar10_std,
            ),
        ]
    )

    if use_augmentation:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(
                    size=32,
                    padding=4,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=cifar10_mean,
                    std=cifar10_std,
                ),
            ]
        )
    else:
        train_transform = test_transform

    return train_transform, test_transform


def _get_cifar10_datasets(
    config: DataConfig,
) -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    train_transform, test_transform = build_transforms(
        use_augmentation=config.use_augmentation
    )

    train_dataset = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        transform=train_transform,
        download=config.download,
    )

    test_dataset = datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        transform=test_transform,
        download=config.download,
    )

    return train_dataset, test_dataset


def get_cifar10_loaders(
    config: DataConfig,
) -> dict[str, DataLoader | None]:
    train_dataset, test_dataset = _get_cifar10_datasets(config)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    return {
        "train_data": train_loader,
        "val_data": None,
        "test_data": test_loader
    }