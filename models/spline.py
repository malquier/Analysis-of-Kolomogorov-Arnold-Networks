from __future__ import annotations

from typing import Optional
import torch
import numpy as np


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


import numpy as np
from typing import Optional


def uniform_clamped_knots_np(
    a: float,
    b: float,
    n_intervals: int,
    p: int,
    *,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Open uniform clamped knot vector on [a,b] with n_intervals intervals (NumPy)."""
    if n_intervals < 1:
        raise ValueError("n_intervals must be >= 1")
    if not (b > a):
        raise ValueError("Require b > a")
    if p < 0:
        raise ValueError("p must be >= 0")

    h = (b - a) / float(n_intervals)

    # internal knots: a+h, ..., b-h  (n_intervals-1 points)
    if n_intervals == 1:
        internal = np.empty((0,), dtype=dtype)
    else:
        internal = np.linspace(a + h, b - h, n_intervals - 1, dtype=dtype)

    t = np.concatenate(
        [
            np.full((p + 1,), a, dtype=dtype),
            internal,
            np.full((p + 1,), b, dtype=dtype),
        ],
        axis=0,
    )
    return t


def fit_spline_ridge_dense(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: int,
    smoothness_coeff: float = 0.0,
    diff_order: int = 2,
) -> np.ndarray:
    """
    Fit d'un spline B-spline 1D par moindres carrés régularisés (ridge smoothing spline).

    On résout :
        min_c ||B c - y||^2 + λ ||D c||^2

    où :
        - B est la matrice de design B-spline
        - D est une matrice de différences finies d'ordre `diff_order`
        - λ = smoothness_coeff

    Parameters
    ----------
    x : ndarray, shape (n,)
        Points d'entrée.
    y : ndarray, shape (n,)
        Valeurs cibles.
    t : ndarray
        Vecteur de knots (clamped).
    p : int
        Degré du spline.
    smoothness_coeff : float
        Coefficient de régularisation λ.
    diff_order : int
        Ordre de la pénalité de différences (1 = pente, 2 = courbure).

    Returns
    -------
    c : ndarray, shape (K,)
        Coefficients du spline.
    """

    # -----------------------------
    # Design matrix B
    # -----------------------------
    B = build_basis_matrix(x, t, p)  # shape (n, K)
    n, K = B.shape

    # -----------------------------
    # Least squares part
    # -----------------------------
    BtB = B.T @ B
    Bty = B.T @ y

    # -----------------------------
    # Regularization (finite differences)
    # -----------------------------
    if smoothness_coeff > 0.0 and diff_order > 0:
        D = build_finite_difference(K, diff_order)  # shape (K-d, K)
        DtD = D.T @ D
        A = BtB + smoothness_coeff * DtD
    else:
        A = BtB

    # -----------------------------
    # Solve linear system
    # -----------------------------
    c = np.linalg.solve(A, Bty)

    return c


def build_finite_difference(K: int, order: int = 2) -> np.ndarray:

    if order < 1:
        raise ValueError("order must be >= 1")
    if order >= K:
        raise ValueError("order must be < K")

    D = np.zeros((K - order, K))

    if order == 1:
        for i in range(K - 1):
            D[i, i] = -1.0
            D[i, i + 1] = 1.0

    elif order == 2:
        for i in range(K - 2):
            D[i, i] = 1.0
            D[i, i + 1] = -2.0
            D[i, i + 2] = 1.0

    else:
        raise NotImplementedError("Only finite difference order 1 or 2 supported")

    return D


import numpy as np


def build_basis_matrix(
    x: np.ndarray,
    t: np.ndarray,
    p: int,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.shape[0]

    K = len(t) - p - 1
    B = np.zeros((n, K), dtype=float)

    for k in range(K):
        B[:, k] = bspline_basis(k, p, t, x)

    return B


def bspline_basis(k: int, p: int, t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Évalue la base B-spline B_{k,p}(x) via la récursion de Cox–de Boor.
    """
    if p == 0:
        return np.where((t[k] <= x) & (x < t[k + 1]), 1.0, 0.0)

    denom1 = t[k + p] - t[k]
    denom2 = t[k + p + 1] - t[k + 1]

    term1 = 0.0
    term2 = 0.0

    if denom1 > 0:
        term1 = (x - t[k]) / denom1 * bspline_basis(k, p - 1, t, x)

    if denom2 > 0:
        term2 = (t[k + p + 1] - x) / denom2 * bspline_basis(k + 1, p - 1, t, x)

    return term1 + term2
