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


