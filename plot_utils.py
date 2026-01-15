import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from generate_data import DatasetConfig, TrainingConfig, scale_inputs
from models.black_scholes import bs_call_price
from typing import Dict, List
from pathlib import Path


def plot_loss_histories(
    histories: Dict[str, Dict[str, List[float]]], output: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (name, history) in zip(axes, histories.items()):
        ax.plot(history["train_loss"], label="Train")
        ax.plot(history["val_loss"], label="Validation")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_yscale("log")
        ax.grid(True, which="both")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_call_curves(
    models: Dict[str, nn.Module],
    *,
    config: DatasetConfig,
    output: Path,
    device: torch.device,
) -> None:
    spot = torch.linspace(config.s_min, config.s_max, 400, device=device).unsqueeze(1)
    time = torch.zeros_like(spot)
    features = torch.cat([time, spot], dim=1)
    features_scaled = scale_inputs(
        features, t=config.t, s_min=config.s_min, s_max=config.s_max
    )

    true_prices = bs_call_price(
        spot, time, K=config.k, T=config.t, r=config.r, sigma=config.sigma
    )

    fig, axes = plt.subplots(1, len(models), figsize=(18, 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        model.eval()
        with torch.no_grad():
            preds = model(features_scaled).squeeze(1).cpu()
        ax.plot(
            spot.squeeze(1).cpu(),
            true_prices.squeeze(1).cpu(),
            label="Black–Scholes",
            linewidth=2,
        )
        ax.plot(spot.squeeze(1).cpu(), preds, label=name)
        ax.set_xlabel("S")
        ax.set_ylabel("Call price C(t=0, S)")
        ax.set_title(f"Courbe finale ({name})")
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
