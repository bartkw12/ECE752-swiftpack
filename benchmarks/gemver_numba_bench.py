# gemver kernel in Polybench benchmarking
# Bart Kowal
# 400089782

import os

# Set single-threaded execution for external BLAS libraries to ensure reproducible benchmarks.
# These must be set before NumPy is imported to take effect.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
import matplotlib.pyplot as plt
from numba import njit
import numpy as np

# Default benchmark configuration
DEFAULT_N = 512
ALPHA = 1.5
BETA = 1.2

# Hardware used for the reported benchmark results
CPU_NAME = "AMD Ryzen 9 5900X 12-Core Processor"

# ---------------------------------------------------------
# Input Initialization
# ---------------------------------------------------------
def initialize_gemver(n):
    """
    Create and initialize the arrays used by the PolyBench GEMVER kernel.

    Returns fresh NumPy arrays. Although the baseline uses Python loops,
    NumPy arrays give every later implementation the same contiguous
    float64 data representation.
    """
    fn = float(n)

    A = np.zeros((n, n), dtype=np.float64)

    u1 = np.zeros(n, dtype=np.float64)
    v1 = np.zeros(n, dtype=np.float64)
    u2 = np.zeros(n, dtype=np.float64)
    v2 = np.zeros(n, dtype=np.float64)

    w = np.zeros(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    z = np.zeros(n, dtype=np.float64)

    for i in range(n):
        u1[i] = i
        u2[i] = ((i + 1) / fn) / 2.0
        v1[i] = ((i + 1) / fn) / 4.0
        v2[i] = ((i + 1) / fn) / 6.0
        y[i] = ((i + 1) / fn) / 8.0
        z[i] = ((i + 1) / fn) / 9.0

        # These are already zero, but the assignments document
        # the original PolyBench initialization.
        x[i] = 0.0
        w[i] = 0.0

        for j in range(n):
            A[i, j] = float((i * j) % n) / n

    return A, u1, v1, u2, v2, w, x, y, z


# ---------------------------------------------------------
# Baseline 0: Pure Python Naive GEMVER
# ---------------------------------------------------------
def gemver_python_0(
    alpha,
    beta,
    A,
    u1,
    v1,
    u2,
    v2,
    w,
    x,
    y,
    z,
):
    """
    Direct loop-based translation of the PolyBench GEMVER kernel.

    A, x, and w are modified in place.
    """
    n = A.shape[0]

    # Stage 1:
    # A = A + u1 * v1.T + u2 * v2.T
    for i in range(n):
        for j in range(n):
            A[i, j] = (
                A[i, j]
                + u1[i] * v1[j]
                + u2[i] * v2[j]
            )

    # Stage 2:
    # x = x + beta * A.T * y
    for i in range(n):
        for j in range(n):
            x[i] = x[i] + beta * A[j, i] * y[j]

    # Stage 3:
    # x = x + z
    for i in range(n):
        x[i] = x[i] + z[i]

    # Stage 4:
    # w = w + alpha * A * x
    for i in range(n):
        for j in range(n):
            w[i] = w[i] + alpha * A[i, j] * x[j]


# ---------------------------------------------------------
# Baseline 1: Naive gemver in Numba
# ---------------------------------------------------------
@njit
def gemver_numba_1(
    alpha,
    beta,
    A,
    u1,
    v1,
    u2,
    v2,
    w,
    x,
    y,
    z,
):
    """
    Direct loop-based translation of the PolyBench GEMVER kernel now using numba.

    A, x, and w are modified in place.
    """
    n = A.shape[0]

    # Stage 1:
    # A = A + u1 * v1.T + u2 * v2.T
    for i in range(n):
        for j in range(n):
            A[i, j] = (
                A[i, j]
                + u1[i] * v1[j]
                + u2[i] * v2[j]
            )

    # Stage 2:
    # x = x + beta * A.T * y
    for i in range(n):
        for j in range(n):
            x[i] = x[i] + beta * A[j, i] * y[j]

    # Stage 3:
    # x = x + z
    for i in range(n):
        x[i] = x[i] + z[i]

    # Stage 4:
    # w = w + alpha * A * x
    for i in range(n):
        for j in range(n):
            w[i] = w[i] + alpha * A[i, j] * x[j]


# ---------------------------------------------------------
# NumPy Reference Implementation
# ---------------------------------------------------------
def gemver_numpy_reference(
    alpha,
    beta,
    A,
    u1,
    v1,
    u2,
    v2,
    w,
    x,
    y,
    z,
):
    """
    NumPy implementation used to produce the expected result.

    Copies are made because GEMVER updates A, x, and w in place.
    """
    A_result = A.copy()
    x_result = x.copy()
    w_result = w.copy()


    A_result += np.outer(u1, v1) + np.outer(u2, v2)
    x_result += beta * np.dot(A_result.T, y)
    x_result += z
    w_result += alpha * np.dot(A_result, x_result)

    return A_result, x_result, w_result


# ---------------------------------------------------------
# Operation Count
# ---------------------------------------------------------
def gemver_flop_count(n):
    """
    Approximate floating-point operation count for GEMVER.

    Stage 1: 4 * N^2
    Stage 2: 3 * N^2
    Stage 3: N
    Stage 4: 3 * N^2

    Total: 10 * N^2 + N
    """
    return 10.0 * (n ** 2) + n


# ---------------------------------------------------------
# Execution and Benchmarking Routine
# ---------------------------------------------------------
def run_benchmark(vector_size=DEFAULT_N):
    (
        A_initial,
        u1,
        v1,
        u2,
        v2,
        w_initial,
        x_initial,
        y,
        z,
    ) = initialize_gemver(vector_size)

    # Compute the expected results outside the measured region.
    A_expected, x_expected, w_expected = gemver_numpy_reference(
        ALPHA,
        BETA,
        A_initial,
        u1,
        v1,
        u2,
        v2,
        w_initial,
        x_initial,
        y,
        z,
    )

    total_flops = gemver_flop_count(vector_size)

    def run_once(fn):
        """
        Run fn on fresh copies of A, x, and w (GEMVER updates them in place)
        and return the elapsed kernel time along with the result arrays.
        """
        A = A_initial.copy()
        x = x_initial.copy()
        w = w_initial.copy()

        start = time.perf_counter()
        fn(ALPHA, BETA, A, u1, v1, u2, v2, w, x, y, z)
        elapsed = time.perf_counter() - start

        return elapsed, A, x, w

    def measure(fn, warmup=True, reps=9):
        # Warm-up call triggers Numba JIT compilation, so it is not timed.
        if warmup:
            run_once(fn)

        # Correctness check against the NumPy reference results
        _, A, x, w = run_once(fn)
        is_correct = (
            np.allclose(A, A_expected, rtol=1e-5, atol=1e-5)
            and np.allclose(x, x_expected, rtol=1e-5, atol=1e-5)
            and np.allclose(w, w_expected, rtol=1e-5, atol=1e-5)
        )

        # Array copies happen outside the timed region of each repetition.
        elapsed = sum(run_once(fn)[0] for _ in range(reps)) / reps
        gflops = (total_flops / elapsed) / 1e9

        return gflops, elapsed, is_correct

    results = []

    # 0. Pure Python (GEMVER is O(N^2), so no size cap is needed for
    # moderate N, but very large N will still take a long time)
    gflops, elapsed, is_correct = measure(gemver_python_0, warmup=False, reps=1)
    results.append(("0_python_naive", gflops, elapsed, is_correct))

    # Baseline functions list
    functions = [
        ("1_numba_naive", gemver_numba_1),
    ]

    for name, fn in functions:
        gflops, elapsed, is_correct = measure(fn)
        results.append((name, gflops, elapsed, is_correct))

    return results


# ---------------------------------------------------------
# Formatting and Output Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            N_input = int(sys.argv[1])

            if N_input <= 0:
                raise ValueError

        except ValueError:
            print(
                "Invalid vector size. Using the default "
                f"N={DEFAULT_N}."
            )
            N_input = DEFAULT_N
    else:
        N_input = DEFAULT_N

    print(f"\nRunning GEMVER Benchmarks for N={N_input}...")
    print(f"CPU: {CPU_NAME}\n")

    benchmark_data = run_benchmark(N_input)

    py_elapsed = benchmark_data[0][2]  # Reference execution time for pure Python naive

    # Table Header Formatting
    header = (
        f"| {'Baseline Implementation':<25} "
        f"| {'GFLOP/s':>10} "
        f"| {'Time (s)':>12} "
        f"| {'Abs Speedup':>12} "
        f"| {'Rel Speedup':>12} "
        f"| {'Correct':<8} |"
    )

    divider = "-" * len(header)

    print(divider)
    print(header)
    print(divider)

    prev_elapsed = None

    for name, gflops, elapsed, is_correct in benchmark_data:
        abs_speedup = py_elapsed / elapsed
        rel_speedup = (prev_elapsed / elapsed) if prev_elapsed is not None else 1.0
        status = "PASS" if is_correct else "FAIL"

        print(
            f"| {name:<25} "
            f"| {gflops:10.4f} "
            f"| {elapsed:12.6f} "
            f"| {abs_speedup:11.2f}x "
            f"| {rel_speedup:11.2f}x "
            f"| {status:<8} |"
        )
        prev_elapsed = elapsed

    print(divider)

    # Plotting Output
    names = [row[0] for row in benchmark_data]
    gflops_vals = [row[1] for row in benchmark_data]

    plt.figure(figsize=(16, 6))
    bars = plt.bar(names, gflops_vals, color="skyblue", edgecolor="navy")

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (0.02 * max(gflops_vals)),
            f"{yval:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.ylabel("GFLOP/s (Higher is better)")
    plt.title(f"Numba GEMVER Benchmark Performance (N={N_input})\nCPU: {CPU_NAME}")
    plt.xticks(rotation=45, ha="right")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
