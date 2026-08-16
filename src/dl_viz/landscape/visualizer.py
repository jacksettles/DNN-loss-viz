import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dl_viz.landscape.directions import (
    create_random_direction,
    filter_normalize_direction,
)
from dl_viz.landscape.parameters import (
    get_parameter_state,
    set_parameter_state,
    apply_two_directions,
)


class LossLandscapeVisualizer:
    def __init__(
        self,
        model: nn.Module,
        data: DataLoader,
        criterion: nn.Module,
        device: torch.device | None = None,
    ):
        self.device = (
            device
            if device is not None
            else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        self.model = model.to(self.device)
        self.data = data
        self.criterion = criterion

    def _evaluate_model(
        self,
        desc: str = "Evaluating landscape",
    ) -> float:
        self.model.eval()

        running_loss = 0.0
        total = 0

        with torch.no_grad():
            for batch in tqdm(
                self.data,
                desc=desc,
                total=len(self.data),
            ):
                features = batch[0].to(self.device)
                targets = batch[1].to(self.device).long()

                outputs = self.model(features)
                loss = self.criterion(outputs, targets)

                batch_size = targets.size(0)

                running_loss += loss.item() * batch_size
                total += batch_size

        return running_loss / total

    def compute_2d_landscape(
        self,
        alphas: list[float],
        betas: list[float],
    ) -> list[dict]:
        base_state = get_parameter_state(self.model)

        direction_x = create_random_direction(self.model)
        direction_x = filter_normalize_direction(
            self.model,
            direction_x,
        )

        direction_y = create_random_direction(self.model)
        direction_y = filter_normalize_direction(
            self.model,
            direction_y,
        )

        landscape = []

        try:
            for alpha in alphas:
                for beta in betas:
                    apply_two_directions(
                        model=self.model,
                        base_state=base_state,
                        direction_x=direction_x,
                        direction_y=direction_y,
                        alpha=alpha,
                        beta=beta,
                    )

                    loss = self._evaluate_model(
                        desc=(
                            f"Landscape "
                            f"a={alpha:.2f}, "
                            f"b={beta:.2f}"
                        )
                    )

                    landscape.append({
                        "alpha": alpha,
                        "beta": beta,
                        "loss": loss,
                    })

        finally:
            set_parameter_state(
                model=self.model,
                state=base_state,
            )

        return landscape