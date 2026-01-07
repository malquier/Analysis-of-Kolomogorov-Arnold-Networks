import numpy as np
from spline import *
import time
import matplotlib.pyplot as plt

import numpy as np


class TestSpline:
    """
    Lightweight test suite for a 1D B-spline approximation pipeline.

    Expected API functions:
      - uniform_clamped_knots(a: float, b: float, n_intervals: int, p: int) -> array-like
      - find_span(t, p, x) -> int
      - basis_funs(t, p, s, x) -> array-like of shape (p+1,)
      - build_design_matrix_dense(x, t, p) -> ndarray (N, K)   [optional but recommended]
      - fit_spline_ls(x, y, t, p) -> coeffs of shape (K,)
      - eval_spline(x0, t, p, c) -> float

    Notes:
      - If you don't have build_design_matrix_dense, LS tests can still run
        if fit_spline_ls exists. Some checks will be skipped.
    """

    def __init__(self, api: dict, verbose: bool = True):
        self.api = api
        self.verbose = verbose

        # Required
        self.uniform_clamped_knots = api.get("uniform_clamped_knots")
        self.find_span = api.get("find_span")
        self.basis_funs = api.get("basis_funs") or api.get("basis_functions")
        self.fit_spline_ls = api.get("fit_spline_ls")
        self.eval_spline = api.get("eval_spline")

        # Optional (but useful)
        self.build_design_matrix_dense = api.get("build_design_matrix_dense")

        self._check_required()

    # ---------- helpers ----------
    def _check_required(self):
        missing = []
        for name, fn in [
            ("uniform_clamped_knots", self.uniform_clamped_knots),
            ("find_span", self.find_span),
            ("basis_funs/basis_functions", self.basis_funs),
            ("fit_spline_ls", self.fit_spline_ls),
            ("eval_spline", self.eval_spline),
        ]:
            if fn is None:
                missing.append(name)
        if missing:
            raise ValueError(f"Missing required API functions: {missing}")

    def _assert(self, cond: bool, msg: str):
        if not cond:
            raise AssertionError(msg)

    def _allclose(self, a, b, tol=1e-10):
        return np.allclose(a, b, atol=tol, rtol=0)

    def _print(self, s: str):
        if self.verbose:
            print(s)

    # ---------- tests ----------
    def test_uniform_clamped_knots_structure(self):
        a, b = -10.0, 10.0
        p = 3
        k = 7  # n_intervals
        t = np.asarray(self.uniform_clamped_knots(a, b, k, p), dtype=float)

        expected_len = (p + 1) + (k - 1) + (p + 1)
        self._assert(
            len(t) == expected_len,
            f"Knots length mismatch: got {len(t)}, expected {expected_len}",
        )

        self._assert(np.all(np.diff(t) >= 0), "Knots must be nondecreasing.")

        self._assert(
            np.all(t[: p + 1] == a),
            f"Left boundary should repeat a={a} exactly p+1 times.",
        )
        self._assert(
            np.all(t[-(p + 1) :] == b),
            f"Right boundary should repeat b={b} exactly p+1 times.",
        )

        internal = t[p + 1 : -(p + 1)]
        if len(internal) > 1:
            diffs = np.diff(internal)
            self._assert(
                self._allclose(diffs, diffs[0], tol=1e-12),
                "Internal knots are not uniformly spaced.",
            )

    def test_find_span_basic(self):
        a, b = 0.0, 1.0
        p = 3
        k = 5
        t = np.asarray(self.uniform_clamped_knots(a, b, k, p), dtype=float)
        K = len(t) - p - 1

        xs = np.linspace(a, b, 101)
        for x in xs:
            s = self.find_span(t, p, float(x))
            self._assert(
                p <= s <= K - 1, f"Span s out of range: s={s}, expected in [{p},{K-1}]"
            )

            # Span property (except the right endpoint handled specially)
            if x < t[K]:
                self._assert(
                    t[s] <= x < t[s + 1],
                    f"Span condition violated at x={x}: "
                    f"t[s]={t[s]}, t[s+1]={t[s+1]}, s={s}",
                )
            else:
                self._assert(
                    s == K - 1, f"At x==t[K], span should be K-1. Got s={s}, K-1={K-1}"
                )

    def test_basis_partition_unity_and_nonnegativity(self):
        a, b = -2.0, 3.0
        p = 3
        k = 8
        t = np.asarray(self.uniform_clamped_knots(a, b, k, p), dtype=float)
        K = len(t) - p - 1

        xs = np.linspace(a, b, 200)
        for x in xs:
            s = self.find_span(t, p, float(x))
            N = np.asarray(self.basis_funs(t, p, s, float(x)), dtype=float)

            self._assert(
                N.shape == (p + 1,),
                f"basis_funs must return shape (p+1,), got {N.shape}",
            )

            self._assert(
                np.all(N >= -1e-12),
                f"Basis functions should be nonnegative (up to tol) at x={x}.",
            )

            self._assert(
                abs(N.sum() - 1.0) < 1e-10,
                f"Partition of unity violated at x={x}. Sum={N.sum()}",
            )

            first = s - p
            self._assert(
                0 <= first <= K - (p + 1),
                f"First active basis index out of bounds: first={first}, K={K}",
            )

    def test_fit_polynomial_accuracy(self):
        p = 3
        k = 25
        a, b = -1.0, 1.0
        x = np.linspace(a, b, 600)
        y = 1.0 + 2.0 * x - 0.5 * x**2 + 0.1 * x**3

        t = np.asarray(self.uniform_clamped_knots(a, b, k, p), dtype=float)
        c = np.asarray(self.fit_spline_ls(x, y, t, p), dtype=float)

        yhat = np.array([self.eval_spline(float(xi), t, p, c) for xi in x])
        rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))

        self._assert(
            rmse < 1e-3, f"Polynomial fit too inaccurate. RMSE={rmse} (expected < 1e-3)"
        )

    def test_fit_sine_accuracy(self):
        p = 3
        k = 40
        a, b = -10.0, 10.0
        x = np.linspace(a, b, 1200)
        y = np.sin(np.pi * x)

        t = np.asarray(self.uniform_clamped_knots(a, b, k, p), dtype=float)
        c = np.asarray(self.fit_spline_ls(x, y, t, p), dtype=float)

        x_test = np.linspace(a, b, 400)
        y_test = np.sin(np.pi * x_test)
        yhat = np.array([self.eval_spline(float(xi), t, p, c) for xi in x_test])

        rmse = float(np.sqrt(np.mean((yhat - y_test) ** 2)))
        self._assert(
            rmse < 5e-2, f"Sine fit too inaccurate. RMSE={rmse} (expected < 5e-2)"
        )

    def test_fit_log_and_sqrt(self):
        p = 3
        k = 30
        a, b = 0.1, 10.0
        x = np.linspace(a, b, 800)

        for func, name, thresh in [(np.log, "log", 1e-2), (np.sqrt, "sqrt", 1e-2)]:
            y = func(x)
            t = np.asarray(self.uniform_clamped_knots(a, b, k, p), dtype=float)
            c = np.asarray(self.fit_spline_ls(x, y, t, p), dtype=float)
            yhat = np.array([self.eval_spline(float(xi), t, p, c) for xi in x])
            rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
            self._assert(
                rmse < thresh,
                f"{name} fit too inaccurate. RMSE={rmse} (expected < {thresh})",
            )

    def test_design_matrix_locality(self):
        """
        Optional: checks that each row has at most p+1 nonzeros.
        Requires build_design_matrix_dense.
        """
        if self.build_design_matrix_dense is None:
            self._print(
                "ℹ️  Skipping test_design_matrix_locality (no build_design_matrix_dense provided)."
            )
            return

        p = 3
        k = 10
        a, b = -2.0, 2.0
        t = np.asarray(self.uniform_clamped_knots(a, b, k, p), dtype=float)

        x = np.linspace(a, b, 200)
        B = np.asarray(self.build_design_matrix_dense(x, t, p), dtype=float)

        # Count nonzeros per row with a tolerance
        nnz_per_row = np.sum(np.abs(B) > 1e-14, axis=1)
        self._assert(
            np.all(nnz_per_row <= (p + 1)),
            f"Locality violated: found rows with > p+1 nonzeros. "
            f"max nnz per row = {nnz_per_row.max()}",
        )

    # ---------- runner ----------
    def run_all(self):
        tests = [
            self.test_uniform_clamped_knots_structure,
            self.test_find_span_basic,
            self.test_basis_partition_unity_and_nonnegativity,
            self.test_design_matrix_locality,
            self.test_fit_polynomial_accuracy,
            self.test_fit_sine_accuracy,
            self.test_fit_log_and_sqrt,
        ]

        self._print("Running spline tests...")
        for test_fn in tests:
            name = test_fn.__name__
            test_fn()
            self._print(f"✅ {name} passed")
        self._print("✅ All spline tests passed!")


def test_spline():
    api = {
        "uniform_clamped_knots": uniform_clamped_knots,
        "find_span": find_span,
        "basis_funs": basis_funs,  # ou "basis_functions": ...
        "build_design_matrix_dense": build_design_matrix_dense,  # optionnel
        "fit_spline_ls": fit_spline_ls,
        "eval_spline": eval_spline,
    }

    tester = TestSpline(api)
    tester.run_all()


def f1(x, y):
    return np.exp(np.sin(np.pi * x) + y**2)


def f2(x, y, z):
    return f1(x, y) * np.log(z) + np.sqrt(z * (x**2))


def simulate_random(a: float, b: float, n: int) -> list:
    r = np.random.random(n)
    return list(map(lambda x: a + (b - a) * x, r))


def test_spline_plot():
    x = np.linspace(-10, 10, 100)
    y = np.sin(np.pi * x)
    p = 3
    n_intervals = 30

    t = uniform_clamped_knots(-10, 10, n_intervals, p)

    t0 = time.time()
    c_ls = fit_spline_ls(x, y, t, p)
    y_ls = [eval_spline(xi, t, p, c_ls) for xi in x]
    t1 = time.time()

    c_ridge = fit_spline_ridge_dense(x, y, t, p)
    y_ridge = [eval_spline(xi, t, p, c_ridge) for xi in x]
    t2 = time.time()

    print("Least squares time execution:", (t1 - t0))
    print("Ridge time execution:", (t2 - t1))

    plt.plot(x, y, label="Real")
    plt.plot(x, y_ls, label="Least Squares")
    plt.plot(x, y_ridge, label="Ridge")

    plt.legend()
    plt.grid(True)
    plt.show()


def test():
    n = 500
    x_min, x_max = -10, 10
    y_min, y_max = -10, 10
    z_min, z_max = 0, 10

    x = simulate_random(x_min, x_max, n)
    y = simulate_random(y_min, y_max, n)
    z = simulate_random(z_min, z_max, n)

    res = [f2(x, y, z) for (x, y, z) in zip(x, y, z)]
    data = list(zip(x, y, z, res))

    return ()


if __name__ == "__main__":
    test_spline()
