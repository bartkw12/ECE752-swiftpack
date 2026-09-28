# gemver kernel in Polybench benchmarking
# Bart Kowal
# 400089782

import sys
import time
 
import numpy as np

# Default benchmark configuration
DEFAULT_N = 256
ALPHA = 1.5
BETA = 1.2

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
    
    # Compute the expected result outside the measured region.
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
    
    # Give the baseline fresh arrays because GEMVER modifies them.
    A_result = A_initial.copy()
    x_result = x_initial.copy()
    w_result = w_initial.copy()
    
    start = time.perf_counter()
    
    gemver_python_0(
        ALPHA,
        BETA,
        A_result,
        u1,
        v1,
        u2,
        v2,
        w_result,
        x_result,
        y,
        z,
    )
    
    elapsed = time.perf_counter() - start
    
    # Check all three modified outputs.
    A_correct = np.allclose(
        A_result,
        A_expected,
        rtol=1e-5,
        atol=1e-5,
    )
    
    x_correct = np.allclose(
        x_result,
        x_expected,
        rtol=1e-5,
        atol=1e-5,
    )
    
    w_correct = np.allclose(
        w_result,
        w_expected,
        rtol=1e-5,
        atol=1e-5,
    )
    
    is_correct = A_correct and x_correct and w_correct
    
    total_flops = gemver_flop_count(vector_size)
    gflops = (total_flops / elapsed) / 1e9
    
    results = [
        (
            "0_python_naive",
            gflops,
            elapsed,
            is_correct,
        )
    ]
    
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
    
    print(
        f"\nRunning GEMVER Pure-Python Baseline "
        f"for N={N_input}...\n"
    )
    
    benchmark_data = run_benchmark(N_input)
    
    header = (
        f"| {'Baseline Implementation':<25} "
        f"| {'GFLOP/s':>10} "
        f"| {'Time (s)':>12} "
        f"| {'Correct':<8} |"
    )
    
    divider = "-" * len(header)
    
    print(divider)
    print(header)
    print(divider)
    
    for name, gflops, elapsed, is_correct in benchmark_data:
        status = "PASS" if is_correct else "FAIL"
        
        print(
            f"| {name:<25} "
            f"| {gflops:10.6f} "
            f"| {elapsed:12.6f} "
            f"| {status:<8} |"
        )
        
    print(divider)
