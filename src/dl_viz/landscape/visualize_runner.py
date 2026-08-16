from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dl_viz.landscape.visualizer import LossLandscapeVisualizer
from dl_viz.landscape.plotting import (
    plot_loss_surface_3d,
    plot_loss_surface_3d_interactive,
)

ALPHAS = [
    -1.0, -0.9, -0.8, -0.7, -0.6,
    -0.5, -0.4, -0.3, -0.2, -0.1,
     0.0,
     0.1,  0.2,  0.3,  0.4,  0.5,
     0.6,  0.7,  0.8,  0.9,  1.0,
]

BETAS = [
    -1.0, -0.9, -0.8, -0.7, -0.6,
    -0.5, -0.4, -0.3, -0.2, -0.1,
     0.0,
     0.1,  0.2,  0.3,  0.4,  0.5,
     0.6,  0.7,  0.8,  0.9,  1.0,
]


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
        alphas: list[float] = ALPHAS,
        betas: list[float] = BETAS,
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

        plot_loss_surface_3d_interactive(
            landscape=landscape,
            save_path=self.output_dir / "loss_surface.html",
        )

        plot_loss_surface_3d(
            landscape=landscape,
            save_path=self.output_dir / "loss_surface.png",
        )

        return landscape
    
def main():
    checkpoint_path = Path(
        "checkpoints/latest/model_2026-08-16.pt"
    )

    output_dir = Path(
        "checkpoints/landscapes/model_2026-08-16"
    )

    model = ...
    data = ...
    criterion = ...

    runner = VisualizationRunner(
        model=model,
        data=data,
        criterion=criterion,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )

    runner.run()


if __name__ == "__main__":
    main()