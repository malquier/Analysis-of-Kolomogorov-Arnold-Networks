import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from kan import KAN
from models.black_scholes import bs_call_price
from models.mlp import MLP
from models.my_kan import KANNet
from generate_data import (
    DatasetConfig,
    TrainingConfig,
    BatchProvider,
    make_simulated_call_data,
    scale_inputs,
)
from compare_mlp_kan import compare_models
from plot_utils import plot_call_curves, plot_loss_histories


def main() -> None:
    dataset_config = DatasetConfig()
    training_config = TrainingConfig()

    torch.manual_seed(training_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = make_simulated_call_data(
        dataset_config.n_train,
        k=dataset_config.k,
        t=dataset_config.t,
        sigma=dataset_config.sigma,
        r=dataset_config.r,
        s_min=dataset_config.s_min,
        s_max=dataset_config.s_max,
        device=device,
    )
    X_val, y_val = make_simulated_call_data(
        dataset_config.n_val,
        k=dataset_config.k,
        t=dataset_config.t,
        sigma=dataset_config.sigma,
        r=dataset_config.r,
        s_min=dataset_config.s_min,
        s_max=dataset_config.s_max,
        device=device,
    )

    my_kan_config = {
        "dims": [2, 5, 5, 1],
        "p": 3,
        "n_intervals": 16,
        "domains": [(-1.0, 1.0)] * 3,
        "init_scale": 1e-2,
    }
    py_kan_config = {
        "width": [2, 8, 8, 1],
        "grid": 5,
        "k": 3,
        "seed": training_config.seed,
        "device": str(device),
    }
    mlp_config = {"in_dim": 2, "hidden": 256, "depth": 3, "act": "tanh"}

    my_kan, py_kan, mlp, histories, output_dir = compare_models(
        X_train,
        y_train,
        X_val,
        y_val,
        training_config=training_config,
        dataset_config=dataset_config,
        my_kan_config=my_kan_config,
        py_kan_config=py_kan_config,
        mlp_config=mlp_config,
    )

    plot_loss_histories(histories, output_dir / "losses.png")
    plot_call_curves(
        {"my_kan": my_kan, "py_kan": py_kan, "mlp": mlp},
        config=dataset_config,
        output=output_dir / "call_curves.png",
        device=device,
    )

    print(f"\nRésultats sauvegardés dans: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
