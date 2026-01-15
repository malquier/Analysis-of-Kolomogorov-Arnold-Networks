import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np
from generate_data import DatasetConfig, TrainingConfig, scale_inputs
from models.black_scholes import bs_call_price
from typing import Dict, List, Callable, Tuple, Any
from models.spline import (
    uniform_clamped_knots_np,
    build_basis_matrix,
    fit_spline_ridge_dense,
)
from pathlib import Path
import os


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


def plot_spline_approximation(
    f: Callable[[np.ndarray], np.ndarray],
    domain: Tuple[float, float],
    parameters: Dict[str, Any],
) -> Dict[str, str]:
    """
    Approxime une fonction f sur un domaine par un spline B-spline (clamped) et sauvegarde des plots.

    Parameters
    ----------
    f : callable
        Fonction à approximer. Doit accepter un np.ndarray et retourner un np.ndarray (vectorisée).
        Exemple: lambda x: np.sin(np.pi * x**2)
    domain : (a, b)
        Domaine de x.
    parameters : dict
        Hyperparamètres. Valeurs par défaut ci-dessous.

        - p (int) : degré du spline (ex: 3)
        - n_intervals (int) : nb d'intervalles pour les knots (ex: 12)
        - n_samples (int) : nb de points (x_i) pour fitter (ex: 200)
        - noise_std (float) : bruit gaussien sur y (ex: 0.0)
        - seed (int) : seed RNG
        - diff_order (int) : ordre diff pour pénalité (ex: 2)
        - smoothness_path (list[float]) : liste de lambdas (ex: np.logspace(-8, -1, 25))
        - grid_size (int) : nb de points pour l'évaluation/plot (ex: 800)
        - out_dir (str) : dossier de sortie (ex: "plot/spline")
        - prefix (str) : préfixe fichiers (ex: "sin_pi_x2")

    Returns
    -------
    dict : mapping nom->chemin des fichiers produits (plots + data)
    """

    # -------------------------
    # Defaults
    # -------------------------
    a, b = float(domain[0]), float(domain[1])
    if not (b > a):
        raise ValueError("domain doit vérifier b > a")

    p: int = int(parameters.get("p", 3))
    n_intervals: int = int(parameters.get("n_intervals", 12))
    n_samples: int = int(parameters.get("n_samples", 200))
    noise_std: float = float(parameters.get("noise_std", 0.0))
    seed: int = int(parameters.get("seed", 0))
    diff_order: int = int(parameters.get("diff_order", 2))

    smoothness_path = parameters.get("smoothness_path", None)
    if smoothness_path is None:
        smoothness_path = np.logspace(-8, -1, 25)  # chemin de régularisation (λ)
    smoothness_path = np.asarray(smoothness_path, dtype=float)

    grid_size: int = int(parameters.get("grid_size", 800))
    out_dir: str = str(parameters.get("out_dir", "plot/spline"))
    prefix: str = str(parameters.get("prefix", "spline_approx"))

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # Data generation
    # -------------------------
    rng = np.random.default_rng(seed)
    x_train = rng.uniform(a, b, size=n_samples)
    x_train.sort()
    y_true = f(x_train)

    y_train = y_true.copy()
    if noise_std > 0:
        y_train = y_train + rng.normal(0.0, noise_std, size=n_samples)

    # -------------------------
    # Spline setup
    # -------------------------
    # open uniform clamped knot vector
    t = uniform_clamped_knots_np(a, b, n_intervals=n_intervals, p=p)

    # grid for plotting
    x_grid = np.linspace(a, b, grid_size)
    y_grid_true = f(x_grid)

    # Design matrix on grid -> gives ALL basis values at x_grid
    B_grid = build_basis_matrix(x_grid, t, p)  # (grid_size, K)
    K = B_grid.shape[1]

    # -------------------------
    # Fit along lambda path (weights evolution)
    # -------------------------
    C = np.zeros((len(smoothness_path), K), dtype=float)  # coefficients per lambda
    yhat_grid_path = np.zeros((len(smoothness_path), grid_size), dtype=float)

    for i, lmbd in enumerate(smoothness_path):
        c = fit_spline_ridge_dense(
            x_train,
            y_train,
            t,
            p,
            smoothness_coeff=float(lmbd),
            diff_order=diff_order,
        )
        C[i, :] = c
        yhat_grid_path[i, :] = B_grid @ c

    # choose a "best" lambda for display (ici: le plus petit -> quasi LS)
    idx_show = 0
    c_show = C[idx_show]
    yhat_show = yhat_grid_path[idx_show]

    # -------------------------
    # 1) Plot basis functions
    # -------------------------
    fig = plt.figure()
    for k in range(K):
        plt.plot(x_grid, B_grid[:, k])
    plt.title(f"B-spline bases (p={p}, K={K}, intervals={n_intervals})")
    plt.xlabel("x")
    plt.ylabel("B_k(x)")
    plt.tight_layout()
    path_bases = os.path.join(out_dir, f"{prefix}_bases.png")
    fig.savefig(path_bases, dpi=160)
    plt.close(fig)

    # -------------------------
    # 2) Plot approximation vs truth
    # -------------------------
    fig = plt.figure()
    plt.plot(x_grid, y_grid_true, label="f(x) (true)")
    plt.scatter(x_train, y_train, s=12, alpha=0.7, label="train samples")
    plt.plot(
        x_grid,
        yhat_show,
        label=f"spline approx (lambda={smoothness_path[idx_show]:.2e})",
    )
    plt.title("Spline approximation vs true function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    path_fit = os.path.join(out_dir, f"{prefix}_fit.png")
    fig.savefig(path_fit, dpi=160)
    plt.close(fig)

    # -------------------------
    # 3) Coefficients evolution (several lambdas)
    # -------------------------
    # pick a few lambdas to visualize (min, median, max)
    idxs = np.unique(np.array([0, len(smoothness_path) // 2, len(smoothness_path) - 1]))
    fig = plt.figure()
    for j in idxs:
        plt.plot(
            C[j, :],
            marker="o",
            markersize=3,
            linewidth=1,
            label=f"lambda={smoothness_path[j]:.2e}",
        )
    plt.title("Spline coefficients (weights) for a few lambdas")
    plt.xlabel("coefficient index k")
    plt.ylabel("c[k]")
    plt.legend()
    plt.tight_layout()
    path_coeffs = os.path.join(out_dir, f"{prefix}_coeffs_path.png")
    fig.savefig(path_coeffs, dpi=160)
    plt.close(fig)

    # -------------------------
    # 4) Heatmap coefficients vs lambda
    # -------------------------
    fig = plt.figure()
    # Use log10(lambda) on y-axis ticks for readability
    plt.imshow(C, aspect="auto", interpolation="nearest")
    plt.title("Coefficient heatmap: rows=lambda index, cols=coeff index")
    plt.xlabel("coeff index k")
    plt.ylabel("lambda index (increasing)")
    plt.tight_layout()
    path_heat = os.path.join(out_dir, f"{prefix}_coeffs_heatmap.png")
    fig.savefig(path_heat, dpi=160)
    plt.close(fig)

    # -------------------------
    # Save raw data for reuse
    # -------------------------
    path_data = os.path.join(out_dir, f"{prefix}_data.npz")
    np.savez(
        path_data,
        domain=np.array([a, b]),
        p=p,
        n_intervals=n_intervals,
        n_samples=n_samples,
        noise_std=noise_std,
        diff_order=diff_order,
        knot_vector=t,
        x_train=x_train,
        y_train=y_train,
        x_grid=x_grid,
        y_grid_true=y_grid_true,
        smoothness_path=smoothness_path,
        coeffs=C,
        yhat_grid_path=yhat_grid_path,
    )

    return {
        "bases_plot": path_bases,
        "fit_plot": path_fit,
        "coeffs_path_plot": path_coeffs,
        "coeffs_heatmap_plot": path_heat,
        "data_npz": path_data,
        "out_dir": out_dir,
    }


if __name__ == "__main__":
    f = lambda x: np.sin(np.pi * x**2)

    artifacts = plot_spline_approximation(
        f=f,
        domain=(0.0, 2.0),
        parameters={
            "p": 3,
            "n_intervals": 10,
            "n_samples": 250,
            "noise_std": 0.0,
            "seed": 42,
            "diff_order": 2,
            "smoothness_path": np.logspace(-10, -2, 30),
            "grid_size": 1000,
            "out_dir": "plot/spline",
            "prefix": "sin_pi_x2",
        },
    )

    print("Saved:")
    for k, v in artifacts.items():
        print(f" - {k}: {v}")
