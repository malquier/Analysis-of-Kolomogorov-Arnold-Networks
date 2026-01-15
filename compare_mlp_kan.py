import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from kan import KAN
from models.mlp import MLP
from models.my_kan import KANNet
from generate_data import TrainingConfig, DatasetConfig, BatchProvider, scale_inputs


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def train_supervised(
    model: nn.Module,
    train_batches: List[tuple[torch.Tensor, torch.Tensor]],
    val_batches: List[tuple[torch.Tensor, torch.Tensor]],
    config: TrainingConfig,
    *,
    device: torch.device,
) -> Dict[str, List[float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for xb, yb in train_batches:
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            batch_size = xb.shape[0]
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        train_loss = running_loss / n_samples
        history["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for xb, yb in val_batches:
                preds = model(xb)
                loss = loss_fn(preds, yb)
                batch_size = xb.shape[0]
                val_loss += loss.item() * batch_size
                val_samples += batch_size

        history["val_loss"].append(val_loss / val_samples)

        if (
            epoch == 1
            or epoch % max(1, config.epochs // 6) == 0
            or epoch == config.epochs
        ):
            print(
                f"Epoch {epoch:4d}/{config.epochs} | train_loss={train_loss:.3e} | "
                f"val_loss={history['val_loss'][-1]:.3e}"
            )

    return history


def compare_models(
    X_train,
    y_train,
    X_val,
    y_val,
    training_config=None,
    dataset_config=None,
    my_kan_config=None,
    py_kan_config=None,
    mlp_config=None,
):
    if training_config == None:
        training_config = TrainingConfig()

    if dataset_config == None:
        dataset_config = DatasetConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = scale_inputs(
        X_train,
        t=dataset_config.t,
        s_min=dataset_config.s_min,
        s_max=dataset_config.s_max,
    )
    X_val = scale_inputs(
        X_val,
        t=dataset_config.t,
        s_min=dataset_config.s_min,
        s_max=dataset_config.s_max,
    )

    train_batches = BatchProvider(
        X_train,
        y_train,
        training_config.batch_size,
        shuffle=False,
        seed=training_config.seed,
    ).batches
    val_batches = BatchProvider(
        X_val,
        y_val,
        training_config.batch_size,
        shuffle=False,
        seed=training_config.seed,
    ).batches

    if my_kan_config == None:
        my_kan_config = {
            "dims": [2, 5, 5, 1],
            "p": 3,
            "n_intervals": 16,
            "domains": [(-1.0, 1.0)] * 3,
            "init_scale": 1e-2,
        }

    if py_kan_config == None:

        py_kan_config = {
            "width": [2, 8, 8, 1],
            "grid": 5,
            "k": 3,
            "seed": training_config.seed,
            "device": str(device),
        }

    if mlp_config == None:
        mlp_config = {"in_dim": 2, "hidden": 128, "depth": 3, "act": "tanh"}

    my_kan = KANNet(
        dims=my_kan_config["dims"],
        p=my_kan_config["p"],
        n_intervals=my_kan_config["n_intervals"],
        domains=my_kan_config["domains"],
        init_scale=my_kan_config["init_scale"],
        device=device,
        dtype=torch.float32,
    )
    py_kan = KAN(
        width=py_kan_config["width"],
        grid=py_kan_config["grid"],
        k=py_kan_config["k"],
        seed=py_kan_config["seed"],
        device=py_kan_config["device"],
    )
    mlp = MLP(
        in_dim=mlp_config["in_dim"],
        hidden=mlp_config["hidden"],
        depth=mlp_config["depth"],
        out_dim=1,
        act=mlp_config["act"],
    )

    histories: Dict[str, Dict[str, List[float]]] = {}

    print("\n=== Entraînement My KAN ===")
    histories["my_kan"] = train_supervised(
        my_kan, train_batches, val_batches, training_config, device=device
    )

    print("\n=== Entraînement PyKAN ===")
    histories["py_kan"] = train_supervised(
        py_kan, train_batches, val_batches, training_config, device=device
    )

    print("\n=== Entraînement MLP ===")
    histories["mlp"] = train_supervised(
        mlp, train_batches, val_batches, training_config, device=device
    )

    output_dir = Path("resultats")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "modeles"
    model_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        {
            "dataset": asdict(dataset_config),
            "training": asdict(training_config),
            "models": {
                "my_kan": my_kan_config,
                "py_kan": py_kan_config,
                "mlp": mlp_config,
            },
        },
        output_dir / "parametres.json",
    )

    save_json(histories["my_kan"], output_dir / "my_kan_loss.json")
    save_json(histories["py_kan"], output_dir / "py_kan_loss.json")
    save_json(histories["mlp"], output_dir / "mlp_loss.json")

    torch.save(my_kan.state_dict(), model_dir / "my_kan_state.pt")
    torch.save(py_kan.state_dict(), model_dir / "py_kan_state.pt")
    torch.save(mlp.state_dict(), model_dir / "mlp_state.pt")

    return my_kan, py_kan, mlp, histories, output_dir
