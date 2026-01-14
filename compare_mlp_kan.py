import math
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ---- Ton KAN ----
from models.my_kan import KANNet


# -------------------------
# Black–Scholes closed-form
# -------------------------
def norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


@torch.no_grad()
def bs_call_closed_form(S, K, r, sigma, T, t):
    # S,t: (N,1)
    tau = (T - t).clamp_min(1e-12)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * torch.sqrt(tau))
    d2 = d1 - sigma * torch.sqrt(tau)
    return S * norm_cdf(d1) - K * torch.exp(-r * tau) * norm_cdf(d2)


# -------------------------
# Simple MLP baseline
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=64, depth=3, act="tanh"):
        super().__init__()
        layers = []
        d = in_dim
        activation = nn.Tanh() if act == "tanh" else nn.ReLU()
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(activation)
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------
# Utilities
# -------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_dataset(
    N: int,
    *,
    K=100.0,
    r=0.05,
    sigma=0.2,
    T=1.0,
    S_max=300.0,
    device="cpu",
):
    device = torch.device(device)

    # Uniform sampling
    t = torch.rand(N, 1, device=device) * T
    S = torch.rand(N, 1, device=device) * S_max
    X = torch.cat([t, S], dim=1)

    y = bs_call_closed_form(S, K, r, sigma, T, t)  # (N,1)
    return X, y


def scaler_fit(X: torch.Tensor, *, T: float, S_max: float):
    # map t in [0,T] -> [-1,1], S in [0,Smax] -> [-1,1]
    def transform(Xin):
        t = Xin[:, 0:1]
        S = Xin[:, 1:2]
        t_ = 2.0 * (t / T) - 1.0
        S_ = 2.0 * (S / S_max) - 1.0
        return torch.cat([t_, S_], dim=1)

    return transform


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    rel_sum = 0.0
    n = 0
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        pred = model(Xb)
        err = pred - yb
        mse_sum += torch.sum(err**2).item()
        mae_sum += torch.sum(torch.abs(err)).item()
        rel_sum += torch.sum(torch.abs(err) / (torch.abs(yb) + 1e-8)).item()
        n += Xb.shape[0]
    rmse = math.sqrt(mse_sum / n)
    mae = mae_sum / n
    rel = rel_sum / n
    return rmse, mae, rel


def train_supervised(
    model,
    train_loader,
    val_loader,
    *,
    epochs=200,
    lr=1e-3,
    device="cpu",
    print_every=50,
):
    device = torch.device(device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    hist = {
        "train_mse": [],
        "val_rmse": [],
        "val_mae": [],
        "val_rel": [],
    }

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        mse_acc = 0.0
        n = 0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            mse_acc += loss.item() * Xb.shape[0]
            n += Xb.shape[0]

        train_mse = mse_acc / n
        hist["train_mse"].append(train_mse)

        rmse, mae, rel = evaluate(model, val_loader, device)
        hist["val_rmse"].append(rmse)
        hist["val_mae"].append(mae)
        hist["val_rel"].append(rel)

        if ep == 1 or ep % print_every == 0 or ep == epochs:
            print(
                f"epoch {ep:4d}/{epochs} | train_mse={train_mse:.3e} | val_rmse={rmse:.3e} | val_mae={mae:.3e} | val_rel={rel:.3e}"
            )

    elapsed = time.time() - t0
    return hist, elapsed


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # -----------------------
    # Problem setup (call BS)
    # -----------------------
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    S_max = 300.0

    # -----------------------
    # Fast training settings
    # -----------------------
    # Ajuste si tu veux encore plus rapide:
    # - N_train: 20k à 80k
    # - epochs: 50 à 150
    # - batch: 1024 à 4096
    N_train = 50_000
    N_val = 10_000
    batch_train = 2048
    batch_val = 4096
    epochs = 120
    lr = 2e-3

    # -----------------------
    # Data
    # -----------------------
    Xtr, ytr = make_dataset(
        N_train, K=K, r=r, sigma=sigma, T=T, S_max=S_max, device=device
    )
    Xva, yva = make_dataset(
        N_val, K=K, r=r, sigma=sigma, T=T, S_max=S_max, device=device
    )

    # Normalize inputs to [-1,1]
    transform = scaler_fit(Xtr, T=T, S_max=S_max)
    Xtr_s = transform(Xtr)
    Xva_s = transform(Xva)

    train_loader = DataLoader(
        TensorDataset(Xtr_s, ytr), batch_size=batch_train, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(Xva_s, yva), batch_size=batch_val, shuffle=False
    )

    # -----------------------
    # Models
    # -----------------------
    # KAN: architecture "light" for speed
    kan = KANNet(
        dims=[2, 5, 5, 1],
        p=3,
        n_intervals=16,
        domains=[(-1.0, 1.0)] * 3,
        init_scale=1e-2,
        device=torch.device(device),
        dtype=torch.float32,
    )

    # MLP: comparable capacity but fast
    mlp = MLP(in_dim=2, hidden=48, depth=3, act="tanh")

    print("\n--- Model sizes ---")
    print("KAN params:", count_params(kan))
    print("MLP params:", count_params(mlp))

    # -----------------------
    # Train
    # -----------------------
    print("\n=== Training KAN ===")
    kan_hist, kan_time = train_supervised(
        kan,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        print_every=max(10, epochs // 6),
    )

    print("\n=== Training MLP ===")
    mlp_hist, mlp_time = train_supervised(
        mlp,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        print_every=max(10, epochs // 6),
    )

    # Final eval
    kan_rmse, kan_mae, kan_rel = evaluate(kan, val_loader, device)
    mlp_rmse, mlp_mae, mlp_rel = evaluate(mlp, val_loader, device)

    print("\n--- Final comparison (validation) ---")
    print(
        f"KAN: rmse={kan_rmse:.3e}, mae={kan_mae:.3e}, rel={kan_rel:.3e}, time={kan_time:.1f}s"
    )
    print(
        f"MLP: rmse={mlp_rmse:.3e}, mae={mlp_mae:.3e}, rel={mlp_rel:.3e}, time={mlp_time:.1f}s"
    )

    # -----------------------
    # Plots: convergence
    # -----------------------
    epochs_axis = list(range(1, epochs + 1))

    # 1) Train MSE
    plt.figure()
    plt.plot(epochs_axis, kan_hist["train_mse"], label="KAN train MSE")
    plt.plot(epochs_axis, mlp_hist["train_mse"], label="MLP train MSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE (log scale)")
    plt.title("Convergence: Train MSE")
    plt.legend()
    plt.tight_layout()

    # 2) Val RMSE
    plt.figure()
    plt.plot(epochs_axis, kan_hist["val_rmse"], label="KAN val RMSE")
    plt.plot(epochs_axis, mlp_hist["val_rmse"], label="MLP val RMSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Validation RMSE (log scale)")
    plt.title("Convergence: Validation RMSE")
    plt.legend()
    plt.tight_layout()

    # 3) Val relative error (optional, often very informative)
    plt.figure()
    plt.plot(epochs_axis, kan_hist["val_rel"], label="KAN val rel. err")
    plt.plot(epochs_axis, mlp_hist["val_rel"], label="MLP val rel. err")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Mean relative error (log scale)")
    plt.title("Convergence: Validation Relative Error")
    plt.legend()
    plt.tight_layout()

    # 4) Diagnostic curve at t=0 (pricing curve vs BS)
    with torch.no_grad():
        S = torch.linspace(0.0, S_max, 400, device=device).unsqueeze(1)
        t0 = torch.zeros_like(S)
        X = torch.cat([t0, S], dim=1)
        Xs = transform(X)

        true = bs_call_closed_form(S, K, r, sigma, T, t0)
        kan_pred = kan(Xs)
        mlp_pred = mlp(Xs)

        plt.figure()
        plt.plot(S.squeeze(1).cpu(), true.squeeze(1).cpu(), label="Black–Scholes")
        plt.plot(S.squeeze(1).cpu(), kan_pred.squeeze(1).cpu(), label="KAN")
        plt.plot(S.squeeze(1).cpu(), mlp_pred.squeeze(1).cpu(), label="MLP")
        plt.xlabel("S")
        plt.ylabel("Call price C(t=0, S)")
        plt.title("Pricing curve at t=0")
        plt.legend()
        plt.tight_layout()

        # Absolute error vs S at t=0
        plt.figure()
        plt.plot(
            S.squeeze(1).cpu(),
            (kan_pred - true).abs().squeeze(1).cpu(),
            label="KAN |err|",
        )
        plt.plot(
            S.squeeze(1).cpu(),
            (mlp_pred - true).abs().squeeze(1).cpu(),
            label="MLP |err|",
        )
        plt.yscale("log")
        plt.xlabel("S")
        plt.ylabel("Absolute error (log scale)")
        plt.title("Absolute error vs S at t=0")
        plt.legend()
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
