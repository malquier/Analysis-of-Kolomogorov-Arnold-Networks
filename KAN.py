from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spline import uniform_clamped_knots, bspline_basis_matrix


class KANLayer(nn.Module):
    """
    One KAN layer:
        y_j = sum_i phi_{j,i}(x_i)
    where each phi_{j,i} is a 1D spline:
        phi_{j,i}(u) = sum_k c_{j,i,k} B_k(u)

    This implementation:
      - uses a single shared knot vector `t` for all input dims in the layer
      - stores coefficients as nn.Parameter of shape (d_out, d_in, K)
      - computes the **full** basis vector in torch (differentiable w.r.t. x)
      - forward is vectorized over batch and edges for speed/clarity
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        t: torch.Tensor,
        p: int = 3,
        init_scale: float = 1e-2,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if d_in <= 0 or d_out <= 0:
            raise ValueError("d_in and d_out must be positive.")
        if p < 0:
            raise ValueError("p must be >= 0.")

        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.p = int(p)

        if not isinstance(t, torch.Tensor):
            raise TypeError("t must be a torch.Tensor (knot vector)")

        # store knots as a non-trainable buffer so it follows .to(device)
        t = t.to(device=device, dtype=dtype)
        self.register_buffer("t", t)

        K = int(self.t.numel() - self.p - 1)
        if K <= 0:
            raise ValueError(
                f"Invalid knot vector length for p={p}: len(t)={int(self.t.numel())}"
            )
        self.K = int(K)

        # Trainable spline coefficients per edge (j,i,:)
        coeffs = init_scale * torch.randn(
            self.d_out, self.d_in, self.K, device=device, dtype=dtype
        )
        self.coeffs = nn.Parameter(coeffs)

        # Optional pruning mask (non-trainable): 1 active, 0 pruned
        self.register_buffer(
            "mask", torch.ones(self.d_out, self.d_in, dtype=dtype, device=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (d_in,) or (B, d_in)
        returns y: shape (d_out,) or (B, d_out)
        """
        # Accept (d_in,) or (B, d_in)
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze_out = True
        elif x.ndim == 2:
            squeeze_out = False
        else:
            raise ValueError("x must be 1D or 2D tensor.")

        if x.shape[1] != self.d_in:
            raise ValueError(f"Expected x shape (B,{self.d_in}), got {tuple(x.shape)}")

        # Build basis matrix per input dimension: (B, d_in, K)
        B_list = [
            bspline_basis_matrix(x[:, i], self.t, self.p) for i in range(self.d_in)
        ]
        Bx = torch.stack(B_list, dim=1)  # (B, d_in, K)

        # Apply pruning mask on edges: (d_out, d_in, 1)
        C = self.coeffs * self.mask.unsqueeze(-1)

        # y[b, j] = sum_i sum_k Bx[b,i,k] * C[j,i,k]
        y = torch.einsum("bik,oik->bo", Bx, C)

        return y.squeeze(0) if squeeze_out else y

    # -------------------------
    # Regularization helpers
    # -------------------------
    def l1_coeffs(self) -> torch.Tensor:
        return torch.sum(torch.abs(self.coeffs) * self.mask.unsqueeze(-1))

    def smoothness_penalty(self, diff_order: int = 2) -> torch.Tensor:
        if diff_order not in (1, 2):
            raise ValueError("diff_order must be 1 or 2 (extend if needed).")

        C = self.coeffs * self.mask.unsqueeze(-1)

        if diff_order == 1:
            D = C[..., 1:] - C[..., :-1]
        else:
            if self.K < 3:
                return torch.zeros((), device=C.device, dtype=C.dtype)
            D = C[..., 2:] - 2.0 * C[..., 1:-1] + C[..., :-2]

        return torch.sum(D * D)

    @torch.no_grad()
    def prune_edges(self, threshold: float) -> None:
        if threshold < 0:
            raise ValueError("threshold must be >= 0.")

        norms = torch.linalg.norm(self.coeffs, dim=-1)
        to_prune = norms < threshold
        self.mask[to_prune] = 0.0
        self.coeffs[to_prune] = 0.0


class KANNet(nn.Module):
    def __init__(
        self,
        dims: Sequence[int],
        p: int = 3,
        n_intervals: int = 20,
        domains: Optional[Sequence[Tuple[float, float]]] = None,
        init_scale: float = 1e-2,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if len(dims) < 2:
            raise ValueError("dims must have at least 2 elements (input and output).")

        self.dims = list(map(int, dims))
        self.p = int(p)
        self.n_intervals = int(n_intervals)

        L = len(self.dims) - 1

        if domains is None:
            domains = [(-1.0, 1.0)] * L
        if len(domains) != L:
            raise ValueError(f"domains must have length {L} (one per layer).")

        layers: List[KANLayer] = []
        for l in range(L):
            a, b = domains[l]
            t = uniform_clamped_knots(
                float(a),
                float(b),
                n_intervals=self.n_intervals,
                p=self.p,
                device=device,
                dtype=dtype,
            )

            layer = KANLayer(
                d_in=self.dims[l],
                d_out=self.dims[l + 1],
                t=t,
                p=self.p,
                init_scale=init_scale,
                device=device,
                dtype=dtype,
            )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, return_activations: bool = False):
        h = x
        activations = [h] if return_activations else None

        for layer in self.layers:
            h = layer(h)
            if return_activations:
                activations.append(h)

        if return_activations:
            return h, activations
        return h

    def regularization(
        self,
        lambda_l1: float = 0.0,
        lambda_smooth: float = 0.0,
        diff_order: int = 2,
    ) -> torch.Tensor:
        reg = torch.zeros(
            (), device=self.layers[0].coeffs.device, dtype=self.layers[0].coeffs.dtype
        )

        if lambda_l1 != 0.0:
            for layer in self.layers:
                reg = reg + float(lambda_l1) * layer.l1_coeffs()

        if lambda_smooth != 0.0:
            for layer in self.layers:
                reg = reg + float(lambda_smooth) * layer.smoothness_penalty(
                    diff_order=diff_order
                )

        return reg

    @torch.no_grad()
    def prune(self, threshold: float) -> None:
        for layer in self.layers:
            layer.prune_edges(threshold)


def train_kan_from_dataset(
    model,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    X_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    batch_size: int = 64,
    epochs: int = 200,
    lr: float = 3e-2,
    shuffle: bool = True,
    lambda_l1: float = 0.0,
    lambda_smooth: float = 0.0,
    diff_order: int = 2,
    device: Optional[torch.device] = None,
    print_every: int = 20,
) -> dict[str, Any]:

    if device is None:
        device = next(model.parameters()).device

    model.to(device)

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    if y_train.ndim == 1:
        y_train = y_train.unsqueeze(1)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)

    val_loader = None
    if X_val is not None and y_val is not None:
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        if y_val.ndim == 1:
            y_val = y_val.unsqueeze(1)
        val_ds = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_mse": [],
        "train_reg": [],
        "val_loss": [],
        "val_mse": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        sum_loss = 0.0
        sum_mse = 0.0
        sum_reg = 0.0
        n_samples = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()

            y_hat = model(xb)
            mse = torch.mean((y_hat - yb) ** 2)

            reg = 0.0
            if (lambda_l1 != 0.0) or (lambda_smooth != 0.0):
                reg = model.regularization(
                    lambda_l1=lambda_l1,
                    lambda_smooth=lambda_smooth,
                    diff_order=diff_order,
                )

            loss = mse + reg
            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            sum_loss += float(loss.detach().item()) * bs
            sum_mse += float(mse.detach().item()) * bs
            sum_reg += (
                float(reg.detach().item()) * bs
                if isinstance(reg, torch.Tensor)
                else float(reg) * bs
            )
            n_samples += bs

        train_loss = sum_loss / n_samples
        train_mse = sum_mse / n_samples
        train_reg = sum_reg / n_samples

        history["train_loss"].append(train_loss)
        history["train_mse"].append(train_mse)
        history["train_reg"].append(train_reg)

        val_loss = None
        val_mse = None
        if val_loader is not None:
            model.eval()
            sum_vloss = 0.0
            sum_vmse = 0.0
            vn = 0

            with torch.no_grad():
                for xb, yb in val_loader:
                    y_hat = model(xb)
                    vmse = torch.mean((y_hat - yb) ** 2)

                    vreg = 0.0
                    if (lambda_l1 != 0.0) or (lambda_smooth != 0.0):
                        vreg = model.regularization(
                            lambda_l1=lambda_l1,
                            lambda_smooth=lambda_smooth,
                            diff_order=diff_order,
                        )

                    vloss = vmse + vreg

                    bs = xb.shape[0]
                    sum_vloss += float(vloss.item()) * bs
                    sum_vmse += float(vmse.item()) * bs
                    vn += bs

            val_loss = sum_vloss / vn
            val_mse = sum_vmse / vn
            history["val_loss"].append(val_loss)
            history["val_mse"].append(val_mse)

        if (epoch == 1) or (epoch % print_every == 0) or (epoch == epochs):
            if val_loader is None:
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"train_loss={train_loss:.6e} (mse={train_mse:.6e}, reg={train_reg:.6e})"
                )
            else:
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"train_loss={train_loss:.6e} (mse={train_mse:.6e}, reg={train_reg:.6e}) | "
                    f"val_loss={val_loss:.6e} (mse={val_mse:.6e})"
                )

    return history
