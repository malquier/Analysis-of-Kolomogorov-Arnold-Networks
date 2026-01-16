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
from generate_data import (
    HybridBatchProvider,
    PINNBatchProvider,
    TrainingConfig,
    DatasetConfig,
    BatchProvider,
    scale_inputs,
)
from plot_utils import plot_call_curves, plot_loss_histories


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def safe_grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Gradient robuste pour PINN :
    - si outputs n'est pas dans le graphe -> renvoie 0 connecté
    - si grad est None -> renvoie 0 connecté
    Compatible dérivées secondes.
    """
    # outputs peut être un tensor sans grad_fn (ex: constant)
    if not outputs.requires_grad:
        return torch.zeros_like(inputs, requires_grad=True)

    grad = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,  # nécessaire pour d2/dx2
        retain_graph=True,  # utile quand on enchaîne plusieurs grads
        allow_unused=True,
    )[0]

    if grad is None:
        return torch.zeros_like(inputs, requires_grad=True)

    # Important: si grad ne requiert pas grad (constant), on renvoie un zero "second-derivative safe"
    if not grad.requires_grad:
        return torch.zeros_like(inputs, requires_grad=True)

    return grad


def train_supervised(
    model: nn.Module,
    train_batches: List[dict],
    val_batches: List[dict],
    config: TrainingConfig,
    *,
    device: torch.device,
) -> Dict[str, List[float]]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()

    histories = {
        "train_total": [],
        "train_data": [],
        "train_bs": [],
        "train_bc": [],
        "train_reg": [],
        "val_total": [],
    }

    for epoch in range(1, config.epochs + 1):
        model.train()

        sum_total = sum_data = sum_bs = sum_bc = sum_reg = 0.0
        n = 0

        for batch in train_batches:
            opt.zero_grad(set_to_none=True)

            # -------------------------
            # (A) Data loss (supervised)
            # -------------------------
            xb, yb = batch["data"]
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss_data = mse(pred, yb)

            # -------------------------
            # (B) BS PDE loss on interior points
            # x_int = [t, S]
            # -------------------------
            x_int = batch["interior"].to(device)
            x_int.requires_grad_(True)

            t = x_int[:, 0:1]
            S = x_int[:, 1:2]

            C = model(x_int)

            dC_dt = safe_grad(C, t)
            dC_dS = safe_grad(C, S)
            d2C_dS2 = safe_grad(dC_dS, S)

            # PDE residual (no dividend)
            bs_res = (
                dC_dt
                + 0.5 * (config.sigma**2) * (S**2) * d2C_dS2
                + config.r * S * dC_dS
                - config.r * C
            )
            loss_bs = torch.mean(bs_res**2)

            # -------------------------
            # (C) Boundary / terminal loss
            # -------------------------
            x_bc, y_bc = batch["boundary"]
            x_bc = x_bc.to(device)
            y_bc = y_bc.to(device)
            C_bc = model(x_bc)
            loss_bc = mse(C_bc, y_bc)

            # -------------------------
            # (D) Regularization
            # -------------------------
            reg = torch.zeros((), device=device)
            for p in model.parameters():
                reg = reg + (p**2).sum()
            loss_reg = config.lambda_reg * reg

            # -------------------------
            # Total
            # -------------------------
            loss = (
                loss_data
                + config.lambda_bs * loss_bs
                + config.lambda_bc * loss_bc
                + loss_reg
            )
            loss.backward()
            opt.step()

            bs = xb.shape[0]
            n += bs
            sum_total += loss.item() * bs
            sum_data += loss_data.item() * bs
            sum_bs += loss_bs.item() * bs
            sum_bc += loss_bc.item() * bs
            sum_reg += loss_reg.item() * bs

        histories["train_total"].append(sum_total / n)
        histories["train_data"].append(sum_data / n)
        histories["train_bs"].append(sum_bs / n)
        histories["train_bc"].append(sum_bc / n)
        histories["train_reg"].append(sum_reg / n)

        # ---- Validation: total (ici on fait simple: data+bc ; tu peux aussi ajouter bs pareil) ----
        model.eval()
        with torch.no_grad():
            v = 0.0
            m = 0
            for batch in val_batches:
                xb, yb = batch["data"]
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                v += mse(pred, yb).item()
                m += 1
            histories["val_total"].append(v / max(1, m))

        if (
            epoch == 1
            or epoch % max(1, config.epochs // 6) == 0
            or epoch == config.epochs
        ):
            print(
                f"Epoch {epoch:4d}/{config.epochs} | "
                f"total={histories['train_total'][-1]:.3e} | "
                f"data={histories['train_data'][-1]:.3e} | "
                f"bs={histories['train_bs'][-1]:.3e} | "
                f"bc={histories['train_bc'][-1]:.3e} | "
                f"reg={histories['train_reg'][-1]:.3e}"
            )

    return histories


def train_supervised_mse(
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

        for batch in train_batches:
            xb, yb = batch["data"]
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
            for batch in val_batches:
                xb, yb = batch["data"]
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
    full_dataset=None,
    my_kan_config=None,
    py_kan_config=None,
    mlp_config=None,
):
    if training_config == None:
        training_config = TrainingConfig()

    if full_dataset == None:
        dataset_config = DatasetConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = scale_inputs(
        X_train,
        t=full_dataset.maturity,
    )
    X_val = scale_inputs(
        X_val,
        t=full_dataset.maturity,
    )

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

    pinn_train = PINNBatchProvider(
        n_batches=len(data_train_batches),
        n_int=training_config.batch_size,
        n_bc=training_config.batch_size,
        S_max=full_dataset.s_max,
        T=full_dataset.maturity,
        K=full_dataset.strike,
        device=device,
        seed=training_config.seed,
    )
    pinn_val = PINNBatchProvider(
        n_batches=len(data_val_batches),
        n_int=training_config.batch_size,
        n_bc=training_config.batch_size,
        S_max=full_dataset.s_max,
        T=full_dataset.maturity,
        K=full_dataset.strike,
        device=device,
        seed=training_config.seed + 1,
    )

    train_batches = HybridBatchProvider(data_train_batches, pinn_train).batches
    val_batches = HybridBatchProvider(data_val_batches, pinn_val).batches

    if my_kan_config == None:
        my_kan_config = {
            "dims": [2, 5, 5, 1],
            "p": 3,
            "n_intervals": 3,
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


def plot_compare_models(
    X_train, y_train, X_val, y_val, full_dataset, training_config=None
):
    if training_config == None:
        training_config = TrainingConfig()

    torch.manual_seed(training_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # X_train, y_train = make_simulated_call_data(
    #     full_dataset.n_train,
    #     k=dataset_config.k,
    #     t=dataset_config.t,
    #     sigma=dataset_config.sigma,
    #     r=dataset_config.r,
    #     s_min=dataset_config.s_min,
    #     s_max=dataset_config.s_max,
    #     device=device,
    # )
    # X_val, y_val = make_simulated_call_data(
    #     dataset_config.n_val,
    #     k=dataset_config.k,
    #     t=dataset_config.t,
    #     sigma=dataset_config.sigma,
    #     r=dataset_config.r,
    #     s_min=dataset_config.s_min,
    #     s_max=dataset_config.s_max,
    #     device=device,
    # )

    my_kan_config = {
        "dims": [2, 5, 5, 1],
        "p": 3,
        "n_intervals": 32,
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
        full_dataset=full_dataset,
        my_kan_config=my_kan_config,
        py_kan_config=py_kan_config,
        mlp_config=mlp_config,
    )

    plot_loss_histories(histories, output_dir / "losses.png")
    plot_call_curves(
        {"my_kan": my_kan, "py_kan": py_kan, "mlp": mlp},
        config=full_dataset,
        output=output_dir / "call_curves.png",
        device=device,
    )

    print(f"\nRésultats sauvegardés dans: {output_dir.resolve()}")
