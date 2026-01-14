import os
import json
from typing import Any, Dict, Optional, Tuple, Union, Callable

import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_results(
    results_dir: str,
    kan_model: Optional[torch.nn.Module] = None,
    mlp_model: Optional[torch.nn.Module] = None,
    pykan_model: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu",
    # --- Call curve settings ---
    K: float = 1.0,
    T: float = 1.0,
    S_min: float = 0.0,
    S_max: float = 4.0,
    n_S: int = 400,
    # If your models expect 2D input [S,t], we evaluate at t=0 by default
    eval_t: float = 0.0,
    # --- Error curve settings ---
    # If you already computed per-epoch errors during training and saved them into history.json,
    # plot_results will use them. Otherwise it can compute errors vs BS closed form at the final epoch
    # or compute errors on-demand if you pass a callback.
    bs_closed_form_fn: Optional[
        Callable[[np.ndarray, float, float, float, float], np.ndarray]
    ] = None,
    r: float = 0.05,
    sigma: float = 0.2,
    # optional: if you saved checkpoints each epoch you can provide a format string
    # like "checkpoints/mlp_epoch{epoch}.pt". Otherwise error-vs-epoch uses history entries if present.
    ckpt_format: Optional[
        Dict[str, str]
    ] = None,  # {"mlp": "checkpoints/mlp_epoch{epoch}.pt", ...}
    save_dirname: str = "plots",
) -> Dict[str, str]:
    """
    From a training run folder (results/<run_name>/), generate 3 plots:
      1) Loss curves (train/val) for KAN, MLP, PyKAN
      2) Estimated call price curve (V(S, t=eval_t)) for each model + payoff + (optional) BS true
      3) Error vs epochs for each model (from history if available, else computed if checkpoints exist)

    Parameters
    ----------
    results_dir : str
        Path to a run directory, e.g. "results/test_run"
    kan_model, mlp_model, pykan_model :
        Model instances. Needed for plot #2 and for computing error if not stored.
        If you only want plot #1, you can omit these.
    bs_closed_form_fn :
        Optional function returning Black-Scholes call price for vector S:
            bs_closed_form_fn(S, t, K, r, sigma) -> price
        If provided, we will plot the analytic curve too, and use it to compute errors.
        Signature we accept here:
            fn(S: np.ndarray, t: float, K: float, r: float, sigma: float) -> np.ndarray
    ckpt_format :
        Optional checkpoint patterns (relative to results_dir). If provided, we can compute
        error-vs-epoch even if it wasn't logged, by loading each epoch checkpoint.

    Returns
    -------
    Dict[str, str]
        Paths to the saved figures.
    """
    device = torch.device(device)

    history_path = os.path.join(results_dir, "history.json")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"history.json not found in {results_dir}")

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    # Where to save plots
    out_plots = os.path.join(results_dir, save_dirname)
    os.makedirs(out_plots, exist_ok=True)

    # ---------- Helper to get loss arrays ----------
    def _get_losses(model_key: str) -> Tuple[np.ndarray, np.ndarray]:
        train = np.array(history.get(model_key, {}).get("train_loss", []), dtype=float)
        val = np.array(history.get(model_key, {}).get("val_loss", []), dtype=float)
        return train, val

    # ---------- Plot 1: Loss curves ----------
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    for key, label in [("kan", "KAN"), ("mlp", "MLP"), ("pykan", "PyKAN")]:
        tr, va = _get_losses(key)
        if tr.size > 0:
            ax1.plot(np.arange(1, tr.size + 1), tr, label=f"{label} train")
        if va.size > 0:
            ax1.plot(np.arange(1, va.size + 1), va, label=f"{label} val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")  # usually better for PINN/pricing
    ax1.set_title("Loss curves (train/val)")
    ax1.grid(True, which="both")
    ax1.legend()
    path1 = os.path.join(out_plots, "1_loss_curves.png")
    fig1.savefig(path1, dpi=160, bbox_inches="tight")
    plt.close(fig1)

    # ---------- Plot 2: Estimated call curve ----------
    # Only if models provided (at least one)
    path2 = ""
    if any(m is not None for m in [kan_model, mlp_model, pykan_model]):
        S = np.linspace(S_min, S_max, n_S)
        payoff = np.maximum(S - K, 0.0)

        # build torch input: [S, t]
        S_torch = torch.tensor(S, dtype=torch.float32, device=device).view(-1, 1)
        t_torch = torch.full_like(S_torch, float(eval_t))
        X = torch.cat([S_torch, t_torch], dim=1)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

        ax2.plot(S, payoff, label="Payoff max(S-K,0)")

        # optional BS
        if bs_closed_form_fn is not None:
            bs_true = bs_closed_form_fn(
                S, float(eval_t), float(K), float(r), float(sigma)
            )
            ax2.plot(S, bs_true, label="Black-Scholes (analytic)")

        @torch.no_grad()
        def _pred_curve(model) -> np.ndarray:
            model.eval()
            y = model(X).detach().cpu().numpy().reshape(-1)
            return y

        if kan_model is not None:
            kan_model = kan_model.to(device)
            ax2.plot(S, _pred_curve(kan_model), label="KAN estimate")

        if mlp_model is not None:
            mlp_model = mlp_model.to(device)
            ax2.plot(S, _pred_curve(mlp_model), label="MLP estimate")

        if pykan_model is not None:
            # try torch-like forward first
            try:
                if hasattr(pykan_model, "to"):
                    pykan_model = pykan_model.to(device)
                # callable model
                with torch.no_grad():
                    y = pykan_model(X).detach().cpu().numpy().reshape(-1)
                ax2.plot(S, y, label="PyKAN estimate")
            except Exception:
                # last resort: cannot evaluate
                pass

        ax2.set_xlabel("Underlying price S")
        ax2.set_ylabel("Call value V(S,t)")
        ax2.set_title(f"Estimated call curve at t={eval_t}")
        ax2.grid(True)
        ax2.legend()
        path2 = os.path.join(out_plots, "2_call_curve.png")
        fig2.savefig(path2, dpi=160, bbox_inches="tight")
        plt.close(fig2)

    # ---------- Plot 3: Error vs epochs ----------
    # Priority:
    #  (A) If history already contains "error" arrays -> plot those
    #  (B) Else if ckpt_format + bs_closed_form_fn provided -> compute error per epoch
    path3 = ""

    def _get_error_array(model_key: str) -> Optional[np.ndarray]:
        # You can store e.g. history["mlp"]["rel_l2"] or ["error"] during training
        d = history.get(model_key, {})
        for k in ["rel_l2", "rel_l1", "error", "val_error", "test_error"]:
            if k in d and isinstance(d[k], list) and len(d[k]) > 0:
                return np.array(d[k], dtype=float)
        return None

    errors = {}
    for key in ["kan", "mlp", "pykan"]:
        arr = _get_error_array(key)
        if arr is not None:
            errors[key] = arr

    can_compute_from_ckpt = (
        ckpt_format is not None
        and bs_closed_form_fn is not None
        and any(m is not None for m in [kan_model, mlp_model, pykan_model])
    )

    if len(errors) == 0 and can_compute_from_ckpt:
        # Compute relative L2 error at each epoch from checkpoints
        # relL2 = ||pred-true||2 / ||true||2
        epochs = len(history.get("mlp", {}).get("train_loss", []))
        if epochs == 0:
            epochs = len(history.get("kan", {}).get("train_loss", []))
        if epochs == 0:
            epochs = len(history.get("pykan", {}).get("train_loss", []))

        S = np.linspace(S_min, S_max, n_S)
        true = bs_closed_form_fn(S, float(eval_t), float(K), float(r), float(sigma))
        denom = np.linalg.norm(true) + 1e-12

        S_torch = torch.tensor(S, dtype=torch.float32, device=device).view(-1, 1)
        t_torch = torch.full_like(S_torch, float(eval_t))
        X = torch.cat([S_torch, t_torch], dim=1)

        def _rel_l2(pred: np.ndarray) -> float:
            return float(np.linalg.norm(pred - true) / denom)

        def _load_state(model: torch.nn.Module, ckpt_path: str):
            sd = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(sd)

        for key, model in [("kan", kan_model), ("mlp", mlp_model)]:
            if model is None or key not in ckpt_format:
                continue
            model = model.to(device)
            fmt = ckpt_format[key]
            arr = []
            for ep in range(1, epochs + 1):
                rel_path = fmt.format(epoch=ep)
                ckpt_path = os.path.join(results_dir, rel_path)
                if not os.path.exists(ckpt_path):
                    # stop at first missing
                    break
                _load_state(model, ckpt_path)
                with torch.no_grad():
                    pred = model(X).detach().cpu().numpy().reshape(-1)
                arr.append(_rel_l2(pred))
            if len(arr) > 0:
                errors[key] = np.array(arr, dtype=float)

        # PyKAN: only if torch-like + checkpoints saved as state_dict
        if (
            pykan_model is not None
            and "pykan" in ckpt_format
            and hasattr(pykan_model, "load_state_dict")
        ):
            pykan_model = (
                pykan_model.to(device) if hasattr(pykan_model, "to") else pykan_model
            )
            fmt = ckpt_format["pykan"]
            arr = []
            for ep in range(1, epochs + 1):
                ckpt_path = os.path.join(results_dir, fmt.format(epoch=ep))
                if not os.path.exists(ckpt_path):
                    break
                sd = torch.load(ckpt_path, map_location=device)
                try:
                    pykan_model.load_state_dict(sd)
                except Exception:
                    break
                with torch.no_grad():
                    pred = pykan_model(X).detach().cpu().numpy().reshape(-1)
                arr.append(_rel_l2(pred))
            if len(arr) > 0:
                errors["pykan"] = np.array(arr, dtype=float)

    if len(errors) > 0:
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        for key, label in [("kan", "KAN"), ("mlp", "MLP"), ("pykan", "PyKAN")]:
            if key in errors:
                arr = errors[key]
                ax3.plot(np.arange(1, arr.size + 1), arr, label=label)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Error")
        ax3.set_yscale("log")
        ax3.set_title("Error vs epochs")
        ax3.grid(True, which="both")
        ax3.legend()
        path3 = os.path.join(out_plots, "3_error_vs_epochs.png")
        fig3.savefig(path3, dpi=160, bbox_inches="tight")
        plt.close(fig3)

    return {
        "loss_curves": path1,
        "call_curve": path2,
        "error_vs_epochs": path3,
    }


if __name__ == "__main__":
    plot_results(
        results_dir="C:/Users/alqui/Analysis-of-Kolomogorov-Arnold-Networks/results/test_run"
    )
