import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from models.py_kan import PyKANAdapter


# --- Black–Scholes call price (no dividends) in torch (vectorized) ---
def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    # Standard normal CDF using erf
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


@torch.no_grad()
def bs_call_price_torch(
    S: torch.Tensor, K: float, T: float, r: float, sigma: float, t: float = 0.0
) -> torch.Tensor:
    """
    Returns Black–Scholes European call price at time t (default t=0).
    S: shape (N,1) or (N,)
    """
    if S.dim() == 2:
        S_ = S.squeeze(1)
    else:
        S_ = S

    tau = torch.tensor(max(T - t, 1e-12), dtype=S_.dtype, device=S_.device)
    K_t = torch.tensor(K, dtype=S_.dtype, device=S_.device)
    r_t = torch.tensor(r, dtype=S_.dtype, device=S_.device)
    sig_t = torch.tensor(sigma, dtype=S_.dtype, device=S_.device)

    # avoid log(0)
    S_safe = torch.clamp(S_, min=1e-12)

    d1 = (torch.log(S_safe / K_t) + (r_t + 0.5 * sig_t**2) * tau) / (
        sig_t * torch.sqrt(tau)
    )
    d2 = d1 - sig_t * torch.sqrt(tau)

    call = S_ * _norm_cdf(d1) - K_t * torch.exp(-r_t * tau) * _norm_cdf(d2)
    return call.unsqueeze(1)


# --- Minimal MLP to fit call price as a function of S (supervised regression) ---
class SimpleMLP(nn.Module):
    def __init__(self, in_dim=1, hidden=128, depth=3, out_dim=1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def test():
    """
    - Génère un dataset (S -> prix call BS) en Black–Scholes
    - Entraîne un modèle torch "pykan-like" via PyKANAdapter (ici, un MLP torch)
    - Affiche un graphe : prix vrai vs prix estimé + payoff
    """
    # -------------------------
    # 1) Black–Scholes params
    # -------------------------
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2

    # -------------------------
    # 2) Dataset supervised (t = 0)
    # -------------------------
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Underlying prices
    S_min, S_max = 1e-6, 3.0 * K
    N_train, N_val = 3000, 600

    S_train = torch.rand(N_train, 1) * (S_max - S_min) + S_min
    S_val = torch.linspace(S_min, S_max, N_val).unsqueeze(1)

    y_train = bs_call_price_torch(S_train, K=K, T=T, r=r, sigma=sigma, t=0.0)
    y_val = bs_call_price_torch(S_val, K=K, T=T, r=r, sigma=sigma, t=0.0)

    # Optional normalization for easier training
    S_scale = S_max
    X_train = (S_train / S_scale).to(device)
    X_val = (S_val / S_scale).to(device)
    y_train = y_train.to(device)
    y_val = y_val.to(device)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=256, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=512, shuffle=False)

    # -------------------------
    # 3) "PyKAN" model (torch-like) + adapter
    # -------------------------
    # Ici on utilise un MLP torch comme stand-in "pykan torch-like"
    # Si tu as un vrai modèle pykan callable+parameters(), tu peux le mettre ici.
    model = SimpleMLP(in_dim=1, hidden=128, depth=3, out_dim=1).to(device)

    adapter = PyKANAdapter(model)  # uses forward() = model(x)

    # -------------------------
    # 4) Train loop
    # -------------------------
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(adapter.parameters(), lr=1e-3)

    epochs = 800
    train_losses, val_losses = [], []

    for ep in range(1, epochs + 1):
        # train
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = adapter.forward(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        train_loss = total / max(1, n)

        # val
        model.eval()
        with torch.no_grad():
            total, n = 0.0, 0
            for xb, yb in val_loader:
                pred = adapter.forward(xb)
                loss = loss_fn(pred, yb)
                total += float(loss.item()) * xb.size(0)
                n += xb.size(0)
            val_loss = total / max(1, n)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if ep % 100 == 0 or ep == 1:
            print(
                f"Epoch {ep:4d}/{epochs} | train={train_loss:.6e} | val={val_loss:.6e}"
            )

    # -------------------------
    # 5) Plot: fitted call curve + payoff + (optional) losses
    # -------------------------
    model.eval()
    with torch.no_grad():
        # Evaluate on a dense grid
        S_plot = torch.linspace(0.0, S_max, 600).unsqueeze(1)
        X_plot = (S_plot / S_scale).to(device)
        pred = adapter.forward(X_plot).detach().cpu().squeeze(1)
        true = (
            bs_call_price_torch(S_plot, K=K, T=T, r=r, sigma=sigma, t=0.0)
            .cpu()
            .squeeze(1)
        )
        payoff = torch.clamp(S_plot.squeeze(1) - K, min=0.0)

    plt.figure()
    plt.plot(S_plot.squeeze(1).numpy(), true.numpy(), label="Black–Scholes (vrai)")
    plt.plot(
        S_plot.squeeze(1).numpy(), pred.numpy(), label="Modèle entraîné (estimate)"
    )
    plt.plot(S_plot.squeeze(1).numpy(), payoff.numpy(), label="Payoff max(S-K,0)")
    plt.xlabel("Prix du sous-jacent S")
    plt.ylabel("Prix du call")
    plt.title("Call européen : vrai vs estimé")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label="train loss")
    plt.plot(range(1, epochs + 1), val_losses, label="val loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Évolution de la loss")
    plt.grid(True, which="both")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    test()
