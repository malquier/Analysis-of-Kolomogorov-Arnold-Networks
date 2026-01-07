import numpy as np


def find_span(t: np.ndarray, p: int, x: float) -> int:
    """
    Return s such that t[s] <= x < t[s+1]
    Convention: if x == t[K], return K-1
    where K = len(t) - p - 1
    """
    K = len(t) - p - 1

    if x >= t[K]:
        return K - 1
    if x <= t[p]:
        return p  # left boundary

    low, high = p, K
    mid = (low + high) // 2

    while not (t[mid] <= x < t[mid + 1]):
        if x < t[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2

    return mid


def basis_funs(t: np.ndarray, p: int, s: int, x: float) -> np.ndarray:
    """
    Compute the (p+1) nonzero B-spline basis functions at x:
    returns N[0..p] = [B_{s-p,p}(x), ..., B_{s,p}(x)]
    (Algorithm 'BasisFuns' from Piegl & Tiller)
    """
    N = np.zeros(p + 1, dtype=float)
    left = np.zeros(p + 1, dtype=float)
    right = np.zeros(p + 1, dtype=float)

    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = x - t[s + 1 - j]
        right[j] = t[s + j] - x
        saved = 0.0

        for r in range(0, j):
            denom = right[r + 1] + left[j - r]
            temp = 0.0 if denom == 0.0 else N[r] / denom
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp

        N[j] = saved

    return N


def uniform_clamped_knots(a: float, b: float, n_intervals: int, p: int) -> np.ndarray:
    """
    Open uniform clamped knot vector on [a,b] with n_intervals intervals.
    Internal knots are uniform; endpoints repeated p+1 times.
    """
    if n_intervals < 1:
        raise ValueError("n_intervals must be >= 1")
    if not (b > a):
        raise ValueError("Require b > a")

    h = (b - a) / n_intervals
    internal = np.array(
        [a + i * h for i in range(1, n_intervals)], dtype=float
    )  # k-1 points
    t = np.concatenate([np.full(p + 1, a), internal, np.full(p + 1, b)])
    return t


def build_design_matrix_dense(x: np.ndarray, t: np.ndarray, p: int) -> np.ndarray:
    """
    Dense design matrix B of shape (N, K) where
    B[n, i] = B_{i,p}(x[n]).
    Uses locality: only p+1 entries per row are nonzero.
    """
    x = np.asarray(x, dtype=float)
    K = len(t) - p - 1
    B = np.zeros((len(x), K), dtype=float)

    for n, xn in enumerate(x):
        s = find_span(t, p, float(xn))
        Nvals = basis_funs(t, p, s, float(xn))
        first = s - p
        B[n, first : first + p + 1] = Nvals

    return B


def fit_spline_ls(x: np.ndarray, y: np.ndarray, t: np.ndarray, p: int) -> np.ndarray:
    """
    Least squares fit: c = argmin ||B c - y||^2
    """
    B = build_design_matrix_dense(x, t, p)
    c, *_ = np.linalg.lstsq(B, y, rcond=None)
    return c


import numpy as np


def difference_matrix(K: int, order: int = 2) -> np.ndarray:
    """
    Build a finite-difference matrix D of shape (K-order, K) such that:
    - order=1: (Dc)[i] = c[i+1] - c[i]
    - order=2: (Dc)[i] = c[i+2] - 2c[i+1] + c[i]
    More generally, uses binomial coefficients with alternating signs.
    """
    if order < 0:
        raise ValueError("order must be >= 0")
    if order == 0:
        return np.eye(K)

    if K <= order:
        raise ValueError("Need K > order to build difference matrix.")

    # coefficients for forward differences: [(-1)^(order-j) * C(order, j)] for j=0..order
    # e.g. order=2 -> [1, -2, 1]
    coeffs = np.array(
        [((-1) ** (order - j)) * comb(order, j) for j in range(order + 1)], dtype=float
    )

    D = np.zeros((K - order, K), dtype=float)
    for i in range(K - order):
        D[i, i : i + order + 1] = coeffs
    return D


def comb(n: int, k: int) -> int:
    # small helper, exact integer binomial coefficient
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    num = 1
    den = 1
    for j in range(1, k + 1):
        num *= n - (k - j)
        den *= j
    return num // den


def fit_spline_ridge_dense(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: int,
    lmbd: float = 1e-3,
    diff_order: int = 2,
) -> np.ndarray:
    """
    Ridge-regularized spline fit:
        min_c ||B c - y||^2 + lmbd ||D c||^2

    Parameters
    ----------
    x, y : arrays of shape (N,)
    t    : knot vector (clamped), shape (m,)
    p    : spline degree
    lmbd : regularization strength (>=0)
    diff_order : order of finite differences in penalty (typically 2)

    Returns
    -------
    c : coefficients of shape (K,)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if lmbd < 0:
        raise ValueError("lmbd must be >= 0")

    B = build_design_matrix_dense(x, t, p)  # (N, K)
    N, K = B.shape

    # Unregularized case
    if lmbd == 0.0:
        c, *_ = np.linalg.lstsq(B, y, rcond=None)
        return c

    D = difference_matrix(K, order=diff_order)  # (K-diff_order, K)

    A = B.T @ B + lmbd * (D.T @ D)  # (K, K)
    b = B.T @ y  # (K,)

    # Solve (SPD in practice for lmbd>0) - use solve; fall back to lstsq if needed
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c, *_ = np.linalg.lstsq(A, b, rcond=None)

    return c


def eval_spline(x0: float, t: np.ndarray, p: int, c: np.ndarray) -> float:
    """
    Evaluate spline at x0 using only p+1 local basis functions.
    """
    K = len(t) - p - 1
    s = find_span(t, p, float(x0))
    Nvals = basis_funs(t, p, s, float(x0))
    first = s - p
    return float(np.dot(c[first : first + p + 1], Nvals))


def test():
    return ()


if __name__ == "__main__":
    test()
