import torch
from models.black_scholes import bs_call_closed_form


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
