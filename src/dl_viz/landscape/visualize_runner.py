from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dl_viz.landscape.visualizer import LossLandscapeVisualizer
from dl_viz.landscape.plotting import (
    plot_loss_surface_3d,
    plot_loss_surface_3d_interactive,
)
from dl_viz.runner import (
    PROJECT_ROOT,
    load_configs,
    load_model,
    load_data,
    load_criterion,
)


class VisualizationRunner:
    def __init__(
        self,
        model: nn.Module,
        data: DataLoader,
        criterion: nn.Module,
        checkpoint_path: Path,
        output_dir: Path,
        device: Optional[torch.device] = None,
    ):
        self.device = (
            device
            if device is not None
            else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        self.model = model
        self.data = data
        self.criterion = criterion
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir

    def load_checkpoint(self) -> None:
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)

    def run(
        self,
        alphas: list[float],
        betas: list[float],
        save_raw_results: bool = True,
        save_static_plot: bool = True,
        save_interactive_plot: bool = True,
    ) -> list[dict]:
        self.load_checkpoint()

        visualizer = LossLandscapeVisualizer(
            model=self.model,
            data=self.data,
            criterion=self.criterion,
            device=self.device,
        )

        landscape = visualizer.compute_2d_landscape(
            alphas=alphas,
            betas=betas,
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if save_raw_results:
            landscape_df = pd.DataFrame(landscape)

            landscape_df.to_csv(
                self.output_dir / "landscape.csv",
                index=False,
            )

        if save_interactive_plot:
            plot_loss_surface_3d_interactive(
                landscape=landscape,
                save_path=self.output_dir / "loss_surface.html",
            )

        if save_static_plot:
            plot_loss_surface_3d(
                landscape=landscape,
                save_path=self.output_dir / "loss_surface.png",
            )

        return landscape


def main():
    config = load_configs()

    landscape_config = config.get("landscape", {})

    experiment_id = landscape_config.get("experiment_id")

    if experiment_id is None:
        raise ValueError(
            "Landscape config must contain "
            "'landscape.experiment_id'."
        )

    checkpoint_path = landscape_config.get("checkpoint_path")

    if checkpoint_path is None:
        raise ValueError(
            "Landscape config must contain "
            "'landscape.checkpoint_path'."
        )

    checkpoint_path = PROJECT_ROOT / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    model = load_model(
        config=config,
        experiment_id=experiment_id,
    )

    data = load_data(config)

    data_split = landscape_config.get(
        "data_split",
        "test",
    )

    data_key = f"{data_split}_data"

    if data_key not in data:
        raise ValueError(
            f"Unknown landscape data split {data_split!r}."
        )

    landscape_data = data[data_key]

    if landscape_data is None:
        raise ValueError(
            f"Data split {data_split!r} is not available."
        )

    training_config = config.get("training", {})

    criterion = load_criterion(training_config)

    alphas = np.linspace(
        landscape_config.get("alpha_min", -1.0),
        landscape_config.get("alpha_max", 1.0),
        landscape_config.get("alpha_steps", 21),
    ).tolist()

    betas = np.linspace(
        landscape_config.get("beta_min", -1.0),
        landscape_config.get("beta_max", 1.0),
        landscape_config.get("beta_steps", 21),
    ).tolist()

    output_dir = (
        PROJECT_ROOT
        / landscape_config.get(
            "output_dir",
            f"checkpoints/landscapes/{experiment_id}",
        )
    )

    runner = VisualizationRunner(
        model=model,
        data=landscape_data,
        criterion=criterion,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )

    runner.run(
        alphas=alphas,
        betas=betas,
        save_raw_results=landscape_config.get(
            "save_raw_results",
            True,
        ),
        save_static_plot=landscape_config.get(
            "save_static_plot",
            True,
        ),
        save_interactive_plot=landscape_config.get(
            "save_interactive_plot",
            True,
        ),
    )


if __name__ == "__main__":
    main()