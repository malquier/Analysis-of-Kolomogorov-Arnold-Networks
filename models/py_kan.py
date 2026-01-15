from dataclasses import dataclass
import matplotlib.pyplot as plt
from black_scholes import payoff_call, bs_call_price
import torch
import torch.nn as nn
from kan import KAN


def make_mapper(S_min: float, S_max: float, T: float):
    def st_to_x(S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xS = 2.0 * (S - S_min) / (S_max - S_min) - 1.0
        xt = 2.0 * t / T - 1.0
        return torch.cat([xS, xt], dim=1)

    def x_to_st(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xS = x[:, [0]]
        xt = x[:, [1]]
        S = 0.5 * (xS + 1.0) * (S_max - S_min) + S_min
        t = 0.5 * (xt + 1.0) * T
        return S, t

    return st_to_x, x_to_st


def train_pykan_call_with_loss_breakdown(
    # BS params
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.2,
    # domain
    S_min_mult: float = 0.05,
    S_max_mult: float = 3.0,
    # model
    width=(2, 8, 8, 1),
    grid: int = 5,
    k: int = 3,
    # training
    epochs: int = 2000,
    lr: float = 2e-3,
    # batch sizes for each term
    n_data: int = 2048,
    n_T: int = 512,  # terminal boundary
    n_S0: int = 512,  # S=0 boundary
    n_Smax: int = 512,  # S=Smax boundary
    # weights
    w_data: float = 1.0,
    w_bc: float = 5.0,
    w_reg: float = 1e-6,
    device: str | None = None,
    seed: int = 0,
):
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    S_min = S_min_mult * K
    S_max = S_max_mult * K

    st_to_x, _ = make_mapper(S_min, S_max, T)

    # pykan model (torch Module)
    model = KAN(width=list(width), grid=grid, k=k, seed=seed, device=str(device))
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # logs
    hist_total = []
    hist_data = []
    hist_bc = []
    hist_reg = []

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        # -------------------------
        # (1) Data loss: fit analytic BS call on interior points
        # sample S ~ Uniform[S_min,S_max], t ~ Uniform[0,T)
        # -------------------------
        S = torch.rand(n_data, 1, device=device) * (S_max - S_min) + S_min
        t = torch.rand(n_data, 1, device=device) * (T - 1e-6)  # avoid exactly T
        x = st_to_x(S, t)

        y_true = bs_call_price(S, t, K=K, T=T, r=r, sigma=sigma)
        y_pred = model(x)
        loss_data = mse(y_pred, y_true)

        # -------------------------
        # (2) Boundary losses
        #   a) terminal: t=T -> payoff
        #   b) S=0: C(0,t)=0
        #   c) S=Smax: C(Smax,t)≈Smax - K exp(-r(T-t))
        # -------------------------
        # a) terminal
        S_T = torch.rand(n_T, 1, device=device) * (S_max - S_min) + S_min
        t_T = torch.full_like(S_T, T)
        x_T = st_to_x(S_T, t_T)
        y_T = payoff_call(S_T, K=K)
        loss_T = mse(model(x_T), y_T)

        # b) S=0 boundary (use S_min≈0, but we can enforce at S=0 exactly too)
        t_0 = torch.rand(n_S0, 1, device=device) * T
        S_0 = torch.zeros_like(t_0)
        x_S0 = st_to_x(S_0 + 0.0, t_0)  # exact 0
        y_S0 = torch.zeros_like(t_0)
        loss_S0 = mse(model(x_S0), y_S0)

        # c) S=Smax boundary
        t_max = torch.rand(n_Smax, 1, device=device) * T
        S_hi = torch.full_like(t_max, S_max)
        x_Smax = st_to_x(S_hi, t_max)
        y_Smax = S_hi - K * torch.exp(-r * (T - t_max))
        loss_Smax = mse(model(x_Smax), y_Smax)

        loss_bc = loss_T + loss_S0 + loss_Smax

        # -------------------------
        # (3) Regularization (simple L2 on parameters)
        # -------------------------
        reg = torch.zeros((), device=device)
        for p in model.parameters():
            reg = reg + (p * p).sum()
        loss_reg = reg

        # total
        loss_total = w_data * loss_data + w_bc * loss_bc + w_reg * loss_reg
        loss_total.backward()
        opt.step()

        # log
        hist_total.append(float(loss_total.item()))
        hist_data.append(float(loss_data.item()))
        hist_bc.append(float(loss_bc.item()))
        hist_reg.append(float(loss_reg.item()))

        if ep % max(1, (epochs // 10)) == 0 or ep == 1:
            print(
                f"Epoch {ep:5d}/{epochs} | "
                f"total={hist_total[-1]:.3e} | data={hist_data[-1]:.3e} | "
                f"bc={hist_bc[-1]:.3e} | reg={hist_reg[-1]:.3e}"
            )

    return (
        model,
        (hist_total, hist_data, hist_bc, hist_reg),
        (S_min, S_max, K, T, r, sigma),
        st_to_x,
    )


@torch.no_grad()
def plot_loss_breakdown(hist_total, hist_data, hist_bc, hist_reg):
    plt.figure()
    plt.plot(hist_total, label="Total")
    plt.plot(hist_data, label="Data (BS fit)")
    plt.plot(hist_bc, label="Boundary (BC)")
    plt.plot(hist_reg, label="Regularization (L2 raw)")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss breakdown during training")
    plt.grid(True, which="both")
    plt.legend()


@torch.no_grad()
def plot_call_surface_3d(
    model, st_to_x, S_min, S_max, K, T, r, sigma, device, nS=70, nT=50, plot_true=False
):
    # grid in (S,t)
    Sg = torch.linspace(S_min, S_max, nS, device=device).view(-1, 1)
    tg = torch.linspace(0.0, T, nT, device=device).view(-1, 1)

    # mesh
    S_mesh = Sg.repeat(1, nT)  # (nS,nT)
    t_mesh = tg.t().repeat(nS, 1)  # (nS,nT)

    # flatten for model
    S_flat = S_mesh.reshape(-1, 1)
    t_flat = t_mesh.reshape(-1, 1)
    x = st_to_x(S_flat, t_flat)

    pred = model(x).reshape(nS, nT).detach().cpu().numpy()
    S_np = S_mesh.detach().cpu().numpy()
    t_np = t_mesh.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(S_np, t_np, pred, linewidth=0, antialiased=True)
    ax.set_xlabel("Spot S")
    ax.set_ylabel("Time t")
    ax.set_zlabel("Call C(S,t)")
    ax.set_title("PyKAN surface: C(S,t)")

    if plot_true:
        true = (
            bs_call_price(S_flat, t_flat, K=K, T=T, r=r, sigma=sigma)
            .reshape(nS, nT)
            .cpu()
            .numpy()
        )
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection="3d")
        ax2.plot_surface(S_np, t_np, true, linewidth=0, antialiased=True)
        ax2.set_xlabel("Spot S")
        ax2.set_ylabel("Time t")
        ax2.set_zlabel("Call C(S,t)")
        ax2.set_title("Black–Scholes surface: C(S,t)")


def main():
    model, (hist_total, hist_data, hist_bc, hist_reg), params, st_to_x = (
        train_pykan_call_with_loss_breakdown(
            # ---- choose BS params
            K=100.0,
            T=1.0,
            r=0.05,
            sigma=0.2,
            # ---- domain
            S_min_mult=0.05,
            S_max_mult=3.0,
            # ---- pykan model
            width=(2, 8, 8, 1),
            grid=5,
            k=3,
            # ---- training
            epochs=2000,
            lr=2e-3,
            # ---- samples
            n_data=2048,
            n_T=512,
            n_S0=512,
            n_Smax=512,
            # ---- weights
            w_data=1.0,
            w_bc=5.0,
            w_reg=1e-6,
            seed=0,
        )
    )

    (S_min, S_max, K, T, r, sigma) = params
    device = next(model.parameters()).device

    plot_loss_breakdown(hist_total, hist_data, hist_bc, hist_reg)
    plot_call_surface_3d(
        model,
        st_to_x,
        S_min,
        S_max,
        K,
        T,
        r,
        sigma,
        device=device,
        nS=70,
        nT=50,
        plot_true=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
