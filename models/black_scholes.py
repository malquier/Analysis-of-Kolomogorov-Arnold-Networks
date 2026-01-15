import torch
import math
import matplotlib.pyplot as plt


# -------------------------
# Black–Scholes closed-form
# -------------------------
def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


@torch.no_grad()
def bs_call_price_torch(
    S: torch.Tensor,
    K: float,
    T: float,
    r: float,
    sigma: float,
    t: float = 0.0,
    sigma_obs=0.0,
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

    call = (S_ * _norm_cdf(d1) - K_t * torch.exp(-r_t * tau) * _norm_cdf(d2)) * (
        1 - torch.randn(1, S.shape[0]) * sigma_obs
    )

    return call.unsqueeze(1)


def bs_call_price(
    S: torch.Tensor, t: torch.Tensor, K: float, T: float, r: float, sigma: float
) -> torch.Tensor:
    """
    European call price at time t with maturity T.
    S: (N,1), t: (N,1) in real units
    returns: (N,1)
    """
    eps = 1e-12
    tau = torch.clamp(T - t, min=eps)  # time to maturity
    S_safe = torch.clamp(S, min=eps)

    d1 = (torch.log(S_safe / K) + (r + 0.5 * sigma**2) * tau) / (
        sigma * torch.sqrt(tau)
    )
    d2 = d1 - sigma * torch.sqrt(tau)

    return S * _norm_cdf(d1) - K * torch.exp(-r * tau) * _norm_cdf(d2)


def payoff_call(S: torch.Tensor, K: float) -> torch.Tensor:
    return torch.clamp(S - K, min=0.0)


def test():
    T = 1
    K = 160
    S = torch.linspace(50, 300, 100)
    r = 0.02
    sigma = 0.05

    payoff = bs_call_price_torch(S, K, T, r, sigma, sigma_obs=0.1)
    theory = (S - K).clamp(0)
    # print(payoff[-10:-1])
    plt.scatter(S, payoff, marker="x", color="r")
    plt.plot(S, theory)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    test()
