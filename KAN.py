import matplotlib.pyplot as plt
from spline import *
from typing import Union

LayerParams = tuple[np.ndarray, int, np.ndarray]


"""

----------------------------- Construction of KAN forward -----------------------------

"""


def edge_spline_forward(u: float, t: np.ndarray, p: int, c_edge: np.ndarray) -> float:
    """
    Evaluate one spline edge φ(u) using local B-spline bases.

    Parameters
    ----------
    u      : scalar input
    t      : knot vector (clamped), shape (m,)
    p      : spline degree
    c_edge : coefficients, shape (K,) where K = m - p - 1

    Returns
    -------
    float : φ(u)
    """
    t = np.asarray(t, dtype=float)
    c_edge = np.asarray(c_edge, dtype=float)

    if not np.isfinite(u):
        raise ValueError(f"u must be finite, got {u}")

    if p < 0:
        raise ValueError("p must be >= 0")

    K = len(t) - p - 1
    if K <= 0:
        raise ValueError(
            f"Invalid knot vector length for degree p={p}: len(t)={len(t)}"
        )

    if len(c_edge) != K:
        raise ValueError(f"c_edge has wrong length: got {len(c_edge)}, expected {K}")

    # clamp u into valid domain [t[p], t[K]]
    u_clamped = min(max(float(u), float(t[p])), float(t[K]))

    # find span
    s = find_span(t, p, u_clamped)

    # basis values (p+1)
    N = basis_funs(t, p, s, u_clamped)

    first = s - p
    # dot product between active coeffs and active bases
    return float(np.dot(c_edge[first : first + p + 1], N))


def kan_layer_forward(
    u: np.ndarray, t: np.ndarray, p: int, c_edges: np.ndarray
) -> np.ndarray:
    """
    Forward pass of one KAN layer:
        y_j = sum_i phi_{j,i}(u_i)

    Parameters
    ----------
    u       : input vector, shape (d_in,)
    t       : knot vector, shape (m,)
    p       : spline degree
    c_edges : edge coeffs, shape (d_out, d_in, K), with K = len(t) - p - 1

    Returns
    -------
    y : output vector, shape (d_out,)
    """

    K = len(t) - p - 1
    d_in = u.shape[0]
    d_out = c_edges.shape[0]

    if c_edges.shape != (d_out, d_in, K):
        raise ValueError(
            f"c_edges has shape {c_edges.shape}, expected {(d_out, d_in, K)}"
        )

    y = np.zeros(d_out, dtype=float)

    for j in range(d_out):
        acc = 0.0
        for i in range(d_in):
            acc += edge_spline_forward(float(u[i]), t, p, c_edges[j, i])
        y[j] = acc

    return y


def kan_forward(
    x: np.ndarray, network_params: list[LayerParams], return_activations: bool = False
) -> Union[np.ndarray, tuple[np.ndarray, list[np.ndarray]]]:
    """
    Forward pass through a multi-layer KAN.

    Parameters
    ----------
    x : shape (d0,)
    network_params : list of layers (t, p, c_edges)
    return_activations : if True, also returns intermediate activations

    Returns
    -------
    y_hat : shape (dL,)
    (optional) activations : list of activations including input
    """
    h = np.asarray(x, dtype=float)

    activations = [h] if return_activations else None

    for t, p, c_edges in network_params:
        h = kan_layer_forward(h, t, p, c_edges)
        if return_activations:
            activations.append(h)

    if return_activations:
        return h, activations
    return h


"""

----------------------------- Loss functions -----------------------------

"""


def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    return float(np.mean((y - y_hat) ** 2))


def mse_loss(dataset, network_params: list[LayerParams]) -> float:
    """
    dataset: iterable of (x, y) where x is (d0,) and y is (dL,)
    """
    total = 0.0
    N = len(dataset)
    for x, y in dataset:
        y_hat = kan_forward(x, network_params)
        total += mse(y, y_hat)
    return total / N


"""

----------------------------- KAn for (1,1)-layer -----------------------------

"""


def predict_kan11(x: np.ndarray, t: np.ndarray, p: int, c: np.ndarray) -> np.ndarray:
    """
    Predict y_hat = B(x) @ c without building full B matrix explicitly.
    """
    y_estimated = np.zeros_like(x, dtype=float)
    for n, xn in enumerate(x):
        b = basis_vector(float(xn), t, p)
        y_estimated[n] = float(np.dot(b, c))
    return y_estimated


def train_kan11_gd(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: int,
    actualisation_rate: float = 1e-2,
    n_epochs: int = 200,
    ridge_lambda: float = 0.0,
    seed: int = 42,
):
    """
    Gradient descent training for KAN(1,1):
        y_hat_n = sum_k c_k B_k(x_n)
        L = mean((y_hat - y)^2) + ridge_lambda * ||c||^2

    Returns:
        c, losses
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    K = len(t) - p - 1
    c = 0.01 * rng.standard_normal(K)  # small init

    losses = []

    for _ in range(1, n_epochs + 1):
        # forward + gradient accumulation
        grad = np.zeros_like(c)
        loss = 0.0

        for xn, yn in zip(x, y):
            b = basis_vector(float(xn), t, p)  # (K,)
            y_estimated = float(np.dot(b, c))
            err = y_estimated - float(yn)
            loss += err * err
            # d/dc mean(err^2) -> (2/N) * err * b
            grad += err * b

        N = len(x)
        loss = loss / N
        grad = (2.0 / N) * grad

        # ridge
        if ridge_lambda > 0.0:
            loss += ridge_lambda * float(np.dot(c, c))
            grad += 2.0 * ridge_lambda * c

        # update
        c -= actualisation_rate * grad
        losses.append(loss)

    return c, np.array(losses)


def train_kan_1_1(
    x: np.ndarray,
    y: np.ndarray,
    p: int = 3,
    n_intervals: int = 20,
    t=None,
    ridge_lambda: float = 0.0,
) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Train a KAN(1,1) (single spline edge) with least squares / ridge.

    Parameters
    ----------
    x, y : arrays shape (N,)
    p : spline degree
    n_intervals : number of intervals for uniform clamped knots (ignored if t is provided)
    t : optional knot vector. If None, built from min/max of x.
    ridge_lambda : ridge regularization (>=0). 0 -> plain least squares.

    Returns
    -------
    (t, p, c_edge)
      t : knot vector
      p : degree
      c_edge : coefficients shape (K,)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if len(x) != len(y):
        raise ValueError("x and y must have same length.")
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda must be >= 0.")

    # Build knots if not provided
    if t is None:
        a, b = float(np.min(x)), float(np.max(x))
        t = uniform_clamped_knots(a, b, n_intervals, p)
    else:
        t = np.asarray(t, dtype=float)

    # Build design matrix
    B = build_design_matrix_dense(x, t, p)  # (N, K)
    _, K = B.shape

    # Solve for c
    if ridge_lambda == 0.0:
        c_edge, *_ = np.linalg.lstsq(B, y, rcond=None)
    else:
        A = B.T @ B + ridge_lambda * np.eye(K)
        bvec = B.T @ y
        c_edge = np.linalg.solve(A, bvec)

    return t, p, c_edge


def test_learning_evolution_kan11(
    n_train: int = 200,
    p: int = 5,
    n_intervals: int = 30,
    actualisation_rate: float = 5e-2,
    n_epochs: int = 200,
    ridge_lambda: float = 0,
    seed: int = 0,
):
    """
    Demo function showing learning evolution on a simple target function.
    Prints losses and coefficient snapshots.

    Returns:
        dict with knots t, final coeffs c, loss history, and a test RMSE.
    """
    rng = np.random.default_rng(seed)

    # 1) Create a toy regression dataset: y = sin(pi x) + noise
    a, b = -2.0, 2.0
    x_train = rng.uniform(a, b, size=n_train)
    y_true = np.sin(np.pi * x_train)
    y_train = y_true + 0.05 * rng.standard_normal(n_train)

    # 2) Build knots (fixed bases!)
    t = uniform_clamped_knots(a, b, n_intervals, p)

    # 3) Train with GD to show evolution
    c, losses = train_kan11_gd(
        x_train,
        y_train,
        t,
        p,
        actualisation_rate=actualisation_rate,
        n_epochs=n_epochs,
        ridge_lambda=ridge_lambda,
        seed=seed,
    )

    # 4) Show evolution snapshots
    print("=== KAN(1,1) learning evolution (GD on spline coeffs) ===")
    for ep in [1, 5, 10, 20, 50, 100, n_epochs]:
        if 1 <= ep <= n_epochs:
            print(f"Epoch {ep:4d} | loss = {losses[ep-1]:.6f}")

    # Show a few coefficients (beginning, middle, end indices)
    K = len(c)
    idxs = [0, 1, 2, K // 2, K - 3, K - 2, K - 1] if K >= 7 else list(range(K))
    print("\nSome learned coefficients c[k]:")
    for k in idxs:
        print(f"  c[{k:3d}] = {c[k]: .6f}")

    # 5) Evaluate on a test grid
    x_test = np.linspace(a, b, 400)
    y_test = np.sin(np.pi * x_test)
    y_estimated = predict_kan11(x_test, t, p, c)
    rmse = float(np.sqrt(np.mean((y_estimated - y_test) ** 2)))
    print(f"\nTest RMSE (vs noiseless sin(pi x)) = {rmse:.6f}")

    _, axs = plt.subplots(2, 1, figsize=(9, 7))

    # Plot the losses evolution
    axs[0].plot(losses)
    axs[0].set_xlabel("Number of data training")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True)

    # Plot the estimation result
    axs[1].plot(x_test, y_test, label="Real")
    axs[1].plot(x_test, y_estimated, label="Estimated")
    axs[1].grid(True)

    plt.show()

    return {
        "t": t,
        "p": p,
        "c": c,
        "losses": losses,
        "rmse_test": rmse,
        "x_test": x_test,
        "y_hat_test": y_estimated,
        "y_test": y_test,
    }


def test_kan_1_1():
    # Exemple: approx sin(pi x) sur [-10,10]
    x = np.linspace(-10, 10, 500)
    y = np.sin(np.pi * x)

    t, p, c = train_kan_1_1(x, y, p=3, n_intervals=20, ridge_lambda=1e-3)

    # Prédictions
    y_estimated = np.array([edge_spline_forward(float(xi), t, p, c) for xi in x])

    rmse = np.sqrt(np.mean((y_estimated - y) ** 2))
    print("RMSE:", rmse)

    plt.plot(x, y, label="Real")
    plt.plot(x, y_estimated, label="Estimated")
    plt.legend()
    plt.grid(True)
    plt.show()


def test():
    p = 3
    t = uniform_clamped_knots(-1.0, 1.0, n_intervals=5, p=p)
    K = len(t) - p - 1
    print(K)

    d_in, d_out = 3, 4
    u = np.array([0.1, -0.3, 0.7])
    c_edges = np.random.randn(d_out, d_in, K)

    y = kan_layer_forward(u, t, p, c_edges)
    print(y)  # (4,)
    print(c_edges.shape)


if __name__ == "__main__":
    out = test_learning_evolution_kan11()
