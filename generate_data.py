import torch
from models.black_scholes import bs_call_price
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class DatasetConfig:
    n_train: int = 50_000
    n_val: int = 10_000
    k: float = 120.0
    t: float = 1.0
    sigma: float = 0.2
    r: float = 0.05
    s_min: float = 50.0
    s_max: float = 200.0


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 120
    lr: float = 2e-3
    seed: int = 0
    r = 0.02
    sigma = 0.2

    lambda_bs: float = 1.0
    lambda_bc: float = 10.0
    lambda_reg: float = 1e-2


class BatchProvider:
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.batches = self._make_batches()

    def _make_batches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        n = self.X.shape[0]
        if self.shuffle:
            generator = torch.Generator(device=self.X.device)
            generator.manual_seed(self.seed)
            indices = torch.randperm(n, generator=generator, device=self.X.device)
        else:
            indices = torch.arange(n, device=self.X.device)

        batches = []
        for start in range(0, n, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            batches.append((self.X[batch_idx], self.y[batch_idx]))
        return batches


class PINNBatchProvider:
    def __init__(
        self,
        *,
        n_batches: int,
        n_int: int,
        n_bc: int,
        S_max: float,
        T: float,
        K: float,
        device: torch.device,
        seed: int = 0,
    ) -> None:
        self.n_batches = n_batches
        self.n_int = n_int
        self.n_bc = n_bc
        self.S_max = S_max
        self.T = T
        self.K = K
        self.device = device

        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

        self.batches = self._make_batches()

    def _make_batches(self):
        batches = []
        for _ in range(self.n_batches):
            # Interior: (t, S)
            S_int = (
                torch.rand(self.n_int, 1, generator=self.generator, device=self.device)
                * self.S_max
            ) / self.K
            t_int = (
                torch.rand(self.n_int, 1, generator=self.generator, device=self.device)
                * self.T
            )
            x_int = torch.cat([t_int, S_int], dim=1)  # [t, S]

            # Boundary terminal: t = T, C(S,T)=max(S-K,0)
            S_T = (
                torch.rand(self.n_bc, 1, generator=self.generator, device=self.device)
                * self.S_max
            ) / self.K
            t_T = torch.full_like(S_T, self.T)
            x_bc = torch.cat([t_T, S_T], dim=1)  # [t, S]
            y_bc = torch.relu(S_T - self.K)

            batches.append({"interior": x_int, "boundary": (x_bc, y_bc)})

        return batches


class HybridBatchProvider:
    def __init__(
        self,
        data_batches: list[tuple[torch.Tensor, torch.Tensor]],
        pinn_provider: PINNBatchProvider,
    ) -> None:
        if len(data_batches) != len(pinn_provider.batches):
            raise ValueError(
                "data_batches et pinn_batches doivent avoir le même nombre de batches"
            )
        self.batches = []
        for (xb, yb), pb in zip(data_batches, pinn_provider.batches):
            self.batches.append(
                {
                    "data": (xb, yb),
                    "interior": pb["interior"],
                    "boundary": pb["boundary"],
                }
            )


@torch.no_grad()
def make_simulated_call_data(
    n_samples: int,
    *,
    k: float,
    t: float,
    sigma: float,
    r: float,
    s_min: float,
    s_max: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    spot = torch.rand(n_samples, 1, device=device) * (s_max - s_min) + s_min
    time = torch.rand(n_samples, 1, device=device) * t
    features = torch.cat([time, spot], dim=1)
    targets = bs_call_price(spot, time, K=k, T=t, r=r, sigma=sigma)
    return features, targets


def scale_inputs(X: torch.Tensor, t: float) -> torch.Tensor:
    time = X[:, 1:2]
    spot = X[:, 0:1]
    s_min = min(spot)
    s_max = max(spot)
    time_scaled = 2.0 * (time / t) - 1.0
    spot_scaled = 2.0 * (spot - s_min) / (s_max - s_min) - 1.0
    return torch.cat([time_scaled, spot_scaled], dim=1)


def main():
    dataset_config = DatasetConfig(n_train=200, n_val=50)
    training_config = TrainingConfig()

    torch.manual_seed(training_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = make_simulated_call_data(
        dataset_config.n_train,
        k=dataset_config.k,
        t=dataset_config.t,
        sigma=dataset_config.sigma,
        r=dataset_config.r,
        s_min=dataset_config.s_min,
        s_max=dataset_config.s_max,
        device=device,
    )
    X_val, y_val = make_simulated_call_data(
        dataset_config.n_val,
        k=dataset_config.k,
        t=dataset_config.t,
        sigma=dataset_config.sigma,
        r=dataset_config.r,
        s_min=dataset_config.s_min,
        s_max=dataset_config.s_max,
        device=device,
    )

    plt.plot(X_train[:, 1], y_train, label="Training set", marker="x", linestyle="None")
    plt.plot(X_val[:, 1], y_val, label="Validation set", marker="o", linestyle="None")
    plt.title("Data simulating the price of a call")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot/data/dataset.png")
    plt.show()


if __name__ == "__main__":
    main()
