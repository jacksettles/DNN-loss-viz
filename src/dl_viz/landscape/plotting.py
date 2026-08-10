from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np


def plot_loss_surface_3d(
    landscape: list[dict],
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    alphas = sorted(set(point["alpha"] for point in landscape))
    betas = sorted(set(point["beta"] for point in landscape))

    X, Y = np.meshgrid(alphas, betas)

    Z = np.zeros((len(betas), len(alphas)))

    for point in landscape:
        alpha_idx = alphas.index(point["alpha"])
        beta_idx = betas.index(point["beta"])

        Z[beta_idx, alpha_idx] = point["loss"]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X,
        Y,
        Z,
        cmap="viridis",
        edgecolor="none",
    )

    ax.set_xlabel("Alpha")
    ax.set_ylabel("Beta")
    ax.set_zlabel("Loss")
    ax.set_title("Loss Landscape")

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_loss_surface_3d_interactive(
    landscape: list[dict],
    save_path,
) -> None:
    alphas = sorted(set(point["alpha"] for point in landscape))
    betas = sorted(set(point["beta"] for point in landscape))

    Z = np.zeros((len(betas), len(alphas)))

    for point in landscape:
        alpha_idx = alphas.index(point["alpha"])
        beta_idx = betas.index(point["beta"])

        Z[beta_idx, alpha_idx] = point["loss"]

    fig = go.Figure(
        data=[
            go.Surface(
                x=alphas,
                y=betas,
                z=Z,
                colorscale="Viridis",
            )
        ]
    )

    fig.update_layout(
        title="Loss Landscape",
        scene={
            "xaxis_title": "Alpha",
            "yaxis_title": "Beta",
            "zaxis_title": "Loss",
        },
    )

    fig.write_html(save_path)