import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from kan import KAN
from models.black_scholes import bs_call_price
from models.mlp import MLP
from models.my_kan import KANNet
from generate_data import (
    DatasetConfig,
    HybridBatchProvider,
    PINNBatchProvider,
    TrainingConfig,
    BatchProvider,
    make_simulated_call_data,
    scale_inputs,
)
from get_data import OptionPricingDataset, get_train_test_datasets
from compare_mlp_kan import compare_models, plot_compare_models, train_supervised
from plot_utils import plot_call_curves, plot_loss_histories


def save_training_history(
    history: dict,
    *,
    out_dir: str = "resultats/training",
    name: str = "training_history",
):
    os.makedirs(out_dir, exist_ok=True)

    # --- sauvegarde torch ---
    torch_path = os.path.join(out_dir, f"{name}.pt")
    torch.save(history, torch_path)

    # --- sauvegarde json (cast en float) ---
    json_path = os.path.join(out_dir, f"{name}.json")
    history_json = {k: [float(v) for v in vals] for k, vals in history.items()}

    with open(json_path, "w") as f:
        json.dump(history_json, f, indent=2)

    print(f"[OK] History saved to:\n  - {torch_path}\n  - {json_path}")


def save_model(
    model,
    *,
    out_dir: str = "resultats/models",
    name: str = "model",
):
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"{name}.pt")
    torch.save(model.state_dict(), path)

    print(f"[OK] Model state_dict saved to: {path}")


def main() -> None:
    dataset_config = DatasetConfig()
    training_config = TrainingConfig(batch_size=256, epochs=200)

    torch.manual_seed(training_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- data simulée ---
    full_dataset = OptionPricingDataset(ticker="^STOXX50E", start_date="2015-01-01")

    # 2. Récupérer les 4 blocs séparés
    X_train, y_train, X_val, y_val = get_train_test_datasets(
        full_dataset, split_ratio=0.8
    )

    # --- supervised batches (tu gardes ton BatchProvider) ---
    data_train_batches = BatchProvider(
        X_train,
        y_train,
        training_config.batch_size,
        shuffle=True,
        seed=training_config.seed,
    ).batches
    data_val_batches = BatchProvider(
        X_val,
        y_val,
        training_config.batch_size,
        shuffle=False,
        seed=training_config.seed,
    ).batches

    # --- PINN batches (même nombre de batches que data) ---
    pinn_train = PINNBatchProvider(
        n_batches=len(data_train_batches),
        n_int=training_config.batch_size,
        n_bc=training_config.batch_size,
        S_max=dataset_config.s_max,
        T=dataset_config.t,
        K=dataset_config.k,
        device=device,
        seed=training_config.seed,
    )
    pinn_val = PINNBatchProvider(
        n_batches=len(data_val_batches),
        n_int=training_config.batch_size,
        n_bc=training_config.batch_size,
        S_max=dataset_config.s_max,
        T=dataset_config.t,
        K=dataset_config.k,
        device=device,
        seed=training_config.seed + 1,
    )

    train_batches = HybridBatchProvider(data_train_batches, pinn_train).batches
    val_batches = HybridBatchProvider(data_val_batches, pinn_val).batches

    # --- ton modèle ---
    py_kan = KAN(
        width=[2, 8, 8, 1], grid=5, k=3, seed=training_config.seed, device=str(device)
    )

    history = train_supervised(
        py_kan, train_batches, val_batches, training_config, device=device
    )

    save_training_history(history, out_dir="resultats/training", name="loss curves")
    save_model(py_kan, out_dir="resultats/modeles", name="py_kan")

    # plt.semilogy(history["train_data"], label="data")
    plt.semilogy(history["train_bs"], label="BS(PDE)")
    plt.semilogy(history["train_bc"], label="Boundaries Conditions")
    plt.semilogy(history["train_reg"], label="regularisation")
    plt.semilogy(history["train_total"], label="total")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
