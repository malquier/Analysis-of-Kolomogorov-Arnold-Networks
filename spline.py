"""
spline.py (Torch)

Differentiable B-spline utilities implemented in PyTorch.

This module replaces the former NumPy-based implementation so that gradients
can flow through the B-spline basis computation w.r.t. inputs `x`.

Notes
-----
- B-spline basis functions are piecewise polynomials. They are differentiable
  almost everywhere in x (except exactly at knot locations).
- The *support selection* (whether x is inside a knot interval) uses boolean
  masks (comparisons). Gradients do not propagate through those comparisons,
  which is fine: the basis functions still depend smoothly on x inside each
  interval. For PINN/PDE use, you typically avoid evaluating exactly on knots.

The main function you will use is:
    bspline_basis_matrix(x, t, p)
which returns the full basis vector b(x) of length K = len(t) - p - 1.

We also keep:
    uniform_clamped_knots(a, b, n_intervals, p)
so existing code can keep importing it.
"""

from __future__ import annotations

from typing import Optional
import torch


def uniform_clamped_knots(
    a: float,
    b: float,
    n_intervals: int,
    p: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Open uniform clamped knot vector on [a,b] with n_intervals intervals."""
    if n_intervals < 1:
        raise ValueError("n_intervals must be >= 1")
    if not (b > a):
        raise ValueError("Require b > a")
    if p < 0:
        raise ValueError("p must be >= 0")

    h = (b - a) / float(n_intervals)

    # internal knots: a+h, ..., b-h  (n_intervals-1 points)
    if n_intervals == 1:
        internal = torch.empty((0,), device=device, dtype=dtype)
    else:
        internal = torch.linspace(
            a + h, b - h, n_intervals - 1, device=device, dtype=dtype
        )

    t = torch.cat(
        [
            torch.full((p + 1,), a, device=device, dtype=dtype),
            internal,
            torch.full((p + 1,), b, device=device, dtype=dtype),
        ],
        dim=0,
    )
    return t


def clamp_to_domain(x: torch.Tensor, t: torch.Tensor, p: int) -> torch.Tensor:
    """Clamp x into the valid spline domain [t[p], t[K]] with K=len(t)-p-1."""
    if p < 0:
        raise ValueError("p must be >= 0")
    K = t.numel() - p - 1
    if K <= 0:
        raise ValueError("Invalid knot vector length for given degree.")
    lo = t[p]
    hi = t[K]
    return torch.clamp(x, min=lo, max=hi)


def bspline_basis_matrix(x: torch.Tensor, t: torch.Tensor, p: int) -> torch.Tensor:
    """Compute the full B-spline basis matrix in torch (differentiable).

    Parameters
    ----------
    x : torch.Tensor
        Shape (B,) or (...) any shape. Will be flattened and restored.
    t : torch.Tensor
        Knot vector, shape (m,). Should be non-decreasing.
    p : int
        Spline degree.

    Returns
    -------
    B : torch.Tensor
        Basis values with shape (*x.shape, K) where K = len(t) - p - 1.

    Implementation
    --------------
    Uses Cox–de Boor recursion with vectorized torch ops.
    """
    if p < 0:
        raise ValueError("p must be >= 0")
    if t.ndim != 1:
        raise ValueError("t must be a 1D knot vector")

    orig_shape = x.shape
    x = x.reshape(-1)  # (B,)

    device = x.device
    dtype = x.dtype
    t = t.to(device=device, dtype=dtype)

    m = t.numel()
    K = m - p - 1
    if K <= 0:
        raise ValueError(f"Invalid knot vector length: len(t)={m}, p={p}")

    # Clamp to valid domain.
    x = clamp_to_domain(x, t, p)

    # Degree-0 bases: N_{i,0}(x) = 1 if t[i] <= x < t[i+1] else 0
    x_col = x.unsqueeze(1)  # (B,1)
    t0 = t[:-1].unsqueeze(0)  # (1, m-1)
    t1 = t[1:].unsqueeze(0)  # (1, m-1)
    N = ((x_col >= t0) & (x_col < t1)).to(dtype)  # (B, m-1)

    # Special case: x == t[-1] belongs to last interval
    at_end = torch.isclose(x, t[-1])
    if at_end.any():
        N[at_end] = 0.0
        N[at_end, -1] = 1.0

    # Cox–de Boor recursion
    for k in range(1, p + 1):
        # Build denominators for all i at once
        t_i = t[: -k - 1]  # (m-k-1,)
        t_ik = t[k:-1]  # (m-k-1,)
        t_i1 = t[1:-k]  # (m-k-1,)
        t_ik1 = t[k + 1 :]  # (m-k-1,)

        denom_left = (t_ik - t_i).unsqueeze(0)  # (1, m-k-1)
        denom_right = (t_ik1 - t_i1).unsqueeze(0)  # (1, m-k-1)

        # Previous N has shape (B, m-k)
        N_left = N[:, :-1]  # (B, m-k-1)
        N_right = N[:, 1:]  # (B, m-k-1)

        left_num = x_col - t_i.unsqueeze(0)  # (B, m-k-1)
        right_num = t_ik1.unsqueeze(0) - x_col  # (B, m-k-1)

        # IMPORTANT: torch.where evaluates both branches,
        # so we must avoid dividing by zero even if we mask later.
        mask_left = denom_left > 0
        mask_right = denom_right > 0

        denom_left_safe = torch.where(
            mask_left, denom_left, torch.ones_like(denom_left)
        )
        denom_right_safe = torch.where(
            mask_right, denom_right, torch.ones_like(denom_right)
        )

        left_term = (left_num / denom_left_safe) * N_left * mask_left.to(dtype)
        right_term = (right_num / denom_right_safe) * N_right * mask_right.to(dtype)

        N = left_term + right_term  # (B, m-k-1)

    # After recursion, N has shape (B, K)
    B = N.reshape(*orig_shape, K)
    return B
