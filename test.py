import math
import torch
import torch.nn as nn

# Import ton KAN
from kan import KANNet


# -----------------------------
# 1) Black-Scholes closed-form (pour validation)
# -----------------------------
def norm_cdf(x: torch.Tensor) -> torch.Tensor:
    # CDF normale via erf (torch)
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


@torch.no_grad()
def bs_call_closed_form(S, K, r, sigma, T, t):
    # S, t tensors
    tau = (T - t).clamp_min(1e-12)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * torch.sqrt(tau))
    d2 = d1 - sigma * torch.sqrt(tau)
    return S * norm_cdf(d1) - K * torch.exp(-r * tau) * norm_cdf(d2)


# -----------------------------
# 2) Sampling points
# -----------------------------
def sample_collocation(n, T, S_max, device):
    t = torch.rand(n, 1, device=device) * T
    S = torch.rand(n, 1, device=device) * S_max
    x = torch.cat([t, S], dim=1)
    x.requires_grad_(True)
    return x


def sample_terminal(n, T, S_max, device):
    t = torch.full((n, 1), T, device=device)
    S = torch.rand(n, 1, device=device) * S_max
    x = torch.cat([t, S], dim=1)
    return x


def sample_boundary_S0(n, T, device):
    t = torch.rand(n, 1, device=device) * T
    S = torch.zeros(n, 1, device=device)
    x = torch.cat([t, S], dim=1)
    return x


def sample_boundary_Smax(n, T, S_max, device):
    t = torch.rand(n, 1, device=device) * T
    S = torch.full((n, 1), S_max, device=device)
    x = torch.cat([t, S], dim=1)
    return x


# -----------------------------
# 3) PDE residual computation
# -----------------------------
def pde_residual_black_scholes(model, x, r, sigma):
    """
    x: (N,2) with columns [t, S], requires_grad=True
    returns: residual tensor (N,1)
    """
    t = x[:, 0:1]
    S = x[:, 1:2]

    C = model(x)  # (N,1) ideally
    if C.ndim == 1:
        C = C.unsqueeze(1)

    # dC/dt and dC/dS
    grads = torch.autograd.grad(
        C,
        x,
        grad_outputs=torch.ones_like(C),
        create_graph=True,
        retain_graph=True,
    )[0]
    C_t = grads[:, 0:1]
    C_S = grads[:, 1:2]

    # d2C/dS2
    C_SS = torch.autograd.grad(
        C_S,
        x,
        grad_outputs=torch.ones_like(C_S),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1:2]

    # Black-Scholes PDE residual:
    # C_t + 0.5*sigma^2*S^2*C_SS + r*S*C_S - r*C = 0
    res = C_t + 0.5 * sigma**2 * (S**2) * C_SS + r * S * C_S - r * C
    return res


# -----------------------------
# 4) Training loop (PINN)
# -----------------------------
def train_kan_black_scholes_call(
    *,
    K=100.0,
    r=0.05,
    sigma=0.2,
    T=1.0,
    S_max=300.0,
    dims=(2, 32, 32, 1),
    p=3,
    n_intervals=20,
    lr=1e-3,
    steps=200,
    n_f=4096,  # collocation
    n_t=2048,  # terminal
    n_b=2048,  # boundary each
    w_pde=1.0,
    w_term=10.0,
    w_bc=1.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device)

    # Domain scaling:
    # KAN knots are typically on [-1,1], so it's smart to normalize inputs.
    # We'll map t in [0,T] -> [-1,1], S in [0,S_max] -> [-1,1].
    def scale_in(x):
        t = x[:, 0:1]
        S = x[:, 1:2]
        t_ = 2.0 * (t / T) - 1.0
        S_ = 2.0 * (S / S_max) - 1.0
        return torch.cat([t_, S_], dim=1)

    # Build model with per-layer domains [-1,1]
    domains = [(-1.0, 1.0)] * (len(dims) - 1)
    model = KANNet(
        dims=list(dims),
        p=p,
        n_intervals=n_intervals,
        domains=domains,
        init_scale=1e-2,
        device=device,
        dtype=torch.float32,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(1, steps + 1):
        model.train()
        opt.zero_grad()

        # --- collocation points for PDE
        xf = sample_collocation(n_f, T, S_max, device)
        # scale but KEEP grad flow: scaling is torch ops so autograd OK
        xf_s = scale_in(xf)
        xf_s.requires_grad_(True)

        # --- terminal points
        xt = sample_terminal(n_t, T, S_max, device)
        xt_s = scale_in(xt)

        # --- boundaries
        xb0 = sample_boundary_S0(n_b, T, device)
        xb0_s = scale_in(xb0)

        xbS = sample_boundary_Smax(n_b, T, S_max, device)
        xbS_s = scale_in(xbS)

        # PDE residual
        # Important: pde uses unscaled S in coefficient terms S^2, S, etc.
        # So we pass x in ORIGINAL scale to compute S, but model consumes scaled.
        # Easiest: define a wrapper model_scaled that expects original x.
        def model_on_original(x_orig):
            return model(scale_in(x_orig))

        res = pde_residual_black_scholes(model_on_original, xf, r, sigma)
        loss_pde = torch.mean(res**2)

        # Terminal payoff
        S_t = xt[:, 1:2]
        payoff = torch.clamp(S_t - K, min=0.0)
        C_T = model(xt_s)
        if C_T.ndim == 1:
            C_T = C_T.unsqueeze(1)
        loss_term = torch.mean((C_T - payoff) ** 2)

        # Boundary S=0: C=0
        C_0 = model(xb0_s)
        if C_0.ndim == 1:
            C_0 = C_0.unsqueeze(1)
        loss_bc0 = torch.mean(C_0**2)

        # Boundary S=Smax: C = Smax - K*exp(-r*(T-t))
        t_b = xbS[:, 0:1]
        bcS_target = S_max - K * torch.exp(-r * (T - t_b))
        C_Smax = model(xbS_s)
        if C_Smax.ndim == 1:
            C_Smax = C_Smax.unsqueeze(1)
        loss_bcS = torch.mean((C_Smax - bcS_target) ** 2)

        loss_bc = loss_bc0 + loss_bcS

        loss = w_pde * loss_pde + w_term * loss_term + w_bc * loss_bc
        loss.backward()
        opt.step()

        if step == 1 or step % 200 == 0:
            # quick validation at t=0 on random S
            with torch.no_grad():
                S_test = torch.linspace(0.0, S_max, 200, device=device).unsqueeze(1)
                t0 = torch.zeros_like(S_test)
                x_test = torch.cat([t0, S_test], dim=1)
                pred = model(scale_in(x_test))
                if pred.ndim == 1:
                    pred = pred.unsqueeze(1)
                true = bs_call_closed_form(S_test, K, r, sigma, T, t0)

                rmse = torch.sqrt(torch.mean((pred - true) ** 2)).item()

            print(
                f"step {step:5d} | "
                f"loss={loss.item():.3e} "
                f"(pde={loss_pde.item():.3e}, term={loss_term.item():.3e}, bc={loss_bc.item():.3e}) "
                f"| rmse(t=0)={rmse:.3e}"
            )

    return model


if __name__ == "__main__":
    model = train_kan_black_scholes_call(
        K=100.0,
        r=0.05,
        sigma=0.2,
        T=1.0,
        S_max=300.0,
        dims=(2, 32, 32, 1),
        p=3,
        n_intervals=24,
        lr=2e-3,
        steps=4000,
        n_f=4096,
        n_t=2048,
        n_b=2048,
        w_pde=1.0,
        w_term=20.0,
        w_bc=1.0,
    )
