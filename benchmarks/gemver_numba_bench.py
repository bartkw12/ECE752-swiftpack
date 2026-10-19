# GEMVER kernel benchmark
# Bart Kowal - 400089782
#
# GEMVER overview
# - Performs two rank-1 updates, A^T*y, vector addition, and A*x.
# - Used as a compact benchmark for matrix/vector workloads in numerical computing.
# - Advantage: simple O(N^2) loops with several legal optimization choices.
# - Limitation: low data reuse, so performance is often limited by memory movement.
# - Main challenge: stage 2 uses A^T and can read a row-major matrix inefficiently.
# - Main opportunities: stride-1 access, fastmath/SIMD, loop fusion, and safe parallelism.
#                   - stride-1 means walking through contiguous memory one element at a time.
#
# GEMVER equations
# - A = A + u1*v1^T + u2*v2^T
# - x = x + beta*A^T*y
# - x = x + z
# - w = w + alpha*A*x
#
# Correctness
# - A, x, and w are modified in place and checked against a NumPy reference.
# - allclose uses rtol=atol=1e-5 because optimized reductions may change rounding order.

import os

# Set single-threaded execution for external BLAS libraries to ensure reproducible benchmarks.
# These must be set before NumPy is imported to take effect.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
import matplotlib.pyplot as plt
from numba import njit, prange, get_num_threads, set_num_threads, threading_layer
import numpy as np

# Default benchmark configuration
DEFAULT_N = 512
ALPHA = 1.5
BETA = 1.2

# Block sizes for the tiled baselines (same as the matmul benchmark)
BLOCK_SIZE = 32
BLOCK_SIZE_L2 = 128
BLOCK_SIZE_L1 = 32

# Hardware used for the reported benchmark results
CPU_NAME = "AMD Ryzen 9 5900X 12-Core Processor @ 4.55 GHz"
# Per-core view: 32 KB L1d and 512 KB L2 per core; 32 MB L3 per 6-core CCD.
# (Chip totals are 384 KB L1d, 6 MB L2, 64 MB L3, but one core only sees its CCD's 32 MB L3.)
L1_cache = "32 KB per core"
L2_cache = "512 KB per core"
L3_cache = "32 MB per CCD (64 MB total)"

# ---------------------------------------------------------
# Input Initialization
# - Allocates A and all vectors as contiguous float64 NumPy arrays.
# - Uses deterministic PolyBench formulas, so every implementation gets identical data.
# - x and w start at zero because GEMVER accumulates into them.
# - Fresh copies are required for every run because A, x, and w are modified in place.
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
# Baseline 2: Loop Order Permutations
# - Tests ij/ji orders for stages 1, 2, and 4 (these are the only loops)
# - Best order is ij_ji_ij: stage 2 walks A[j, i] across a row with stride-1 access.
# - Stage 2 updates consecutive elements of x using one row of A.
# - The operations are independent, so LLVM can use SIMD to process multiple elements at once.
# - Poor ji orders traverse columns of row-major A and lose cache locality.
# ---------------------------------------------------------

# testing individual building blocks of GEMVER with different loop orders for stages 1, 2, and 4
# Stage 2 needs the updated A, and Stage 4 needs the completed x.
@njit
def _stage1_ij(A, u1, v1, u2, v2):
    # stage 1, with loop order i then j
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

@njit
def _stage1_ji(A, u1, v1, u2, v2):
    n = A.shape[0]
    for j in range(n):
        for i in range(n):
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

@njit
def _stage2_ij(beta, A, x, y):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            x[i] = x[i] + beta * A[j, i] * y[j]

@njit
def _stage2_ji(beta, A, x, y):
    n = A.shape[0]
    for j in range(n):
        for i in range(n):
            x[i] = x[i] + beta * A[j, i] * y[j]

@njit
def _stage3(x, z):
    for i in range(x.shape[0]):
        x[i] = x[i] + z[i]

@njit
def _stage4_ij(alpha, A, w, x):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            w[i] = w[i] + alpha * A[i, j] * x[j]

@njit
def _stage4_ji(alpha, A, w, x):
    n = A.shape[0]
    for j in range(n):
        for i in range(n):
            w[i] = w[i] + alpha * A[i, j] * x[j]

# put all individual stages together to form the full GEMVER kernel with different loop orders for stages 1, 2, and 4
# The original PolyBench loop organization:
@njit
def gemver_2_ij_ij_ij(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    _stage1_ij(A, u1, v1, u2, v2)
    _stage2_ij(beta, A, x, y)
    _stage3(x, z)
    _stage4_ij(alpha, A, w, x)

# The expected best order:
@njit
def gemver_2_ij_ji_ij(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    _stage1_ij(A, u1, v1, u2, v2)
    _stage2_ji(beta, A, x, y)
    _stage3(x, z)
    _stage4_ij(alpha, A, w, x)

# A mixed case:
@njit
def gemver_2_ji_ji_ji(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    _stage1_ji(A, u1, v1, u2, v2)
    _stage2_ji(beta, A, x, y)
    _stage3(x, z)
    _stage4_ji(alpha, A, w, x)

# The expected worst order:
@njit
def gemver_2_ji_ij_ji(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    _stage1_ji(A, u1, v1, u2, v2)
    _stage2_ij(beta, A, x, y)
    _stage3(x, z)
    _stage4_ji(alpha, A, w, x)


# ---------------------------------------------------------
# Baseline 3: Optimization Flags for Backend (ij_ji_ij order)
# ---------------------------------------------------------
# - Adds @njit(fastmath=True) to ij_ji_ij.
# - Lets LLVM reassociate floating-point reductions and use SIMD/FMA more freely.
# - Trade-off: tiny floating-point differences are possible.
# ---------------------------------------------------------
@njit(fastmath=True)
def gemver_opt_flags_3(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]

    for i in range(n):
        for j in range(n):
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

    for j in range(n):
        for i in range(n):
            x[i] = x[i] + beta * A[j, i] * y[j]

    for i in range(n):
        x[i] = x[i] + z[i]

    for i in range(n):
        for j in range(n):
            w[i] = w[i] + alpha * A[i, j] * x[j]


# ---------------------------------------------------------
# Baseline 4: Parallel Loop Versions
# - parallel_rows uses prange where each iteration owns a distinct output row/element.
# - Stage 2 stays in the strided order to avoid multiple threads writing the same x values.
# - parallel_inner starts a parallel region inside every outer j iteration.
# - legal parallelism can still lose when fork/join overhead or memory traffic dominates.
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def gemver_parallel_rows_4(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]

    # Each thread owns different rows of A.
    # prange gives each thread a contiguous chunk of rows -> good stride 1 access
    for i in prange(n):
        # prange tells Numba that different i iterations may run on different threads.
        for j in range(n):
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

    # Each thread owns a different element of x:
    # give up the best serial order to make straightforward row parallelism safe.
        # every thread would update the entire x vector (data race)
    for i in prange(n):
        for j in range(n):
            x[i] = x[i] + beta * A[j, i] * y[j]

    # Each thread updates a distinct x[i]
    for i in prange(n):
        x[i] = x[i] + z[i]

    # Each thread owns one output element w[i] and reads one row of A.
    for i in prange(n):
        for j in range(n):
            w[i] = w[i] + alpha * A[i, j] * x[j]

    # this provides:
        # no data race
        # stide 1 access


@njit(parallel=True, fastmath=True)
def gemver_parallel_inner_4(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]

    for i in prange(n):
        for j in range(n):
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

    # try to preserve stride 1 stage 2 order
    for j in range(n):
        for i in prange(n):
            # race-free because each i iteration owns a distinct element of x.
            x[i] = x[i] + beta * A[j, i] * y[j]

    for i in prange(n):
        x[i] = x[i] + z[i]

    for i in prange(n):
        for j in range(n):
            w[i] = w[i] + alpha * A[i, j] * x[j]

# cost of scheduling threads and synchronizing after every row becomes larger than the useful computation.
# little work -> introduces far too much parallel overhead.


# ---------------------------------------------------------
# Baseline 5: Blocked (Tiled) Parallel Code
# - Divides the matrix into 32x32 tiles and parallelizes independent blocks.
# - Stage 2 assigns each thread a separate x column block, avoiding a data race.

# - loop tiling changes iteration order to improve locality.
# - GEMVER has little reuse inside a tile, so blocking gives no major gain.
# ---------------------------------------------------------

# give each thread a block of x
# Thread 0 owns x[0:32], Thread 1 owns x[32:64]...
@njit(parallel=True, fastmath=True)
def gemver_blocked_parallel_5(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]
    bs = BLOCK_SIZE
    num_blocks = (n + bs - 1) // bs

    for b in prange(num_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)
        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            for i in range(i_block, i_end):
                for j in range(j_block, j_end):
                    A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

    for b in prange(num_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)
        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            for j in range(j_block, j_end):
                for i in range(i_block, i_end):
                    x[i] = x[i] + beta * A[j, i] * y[j]

    for i in prange(n):
        x[i] = x[i] + z[i]

    for b in prange(num_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)
        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            for i in range(i_block, i_end):
                for j in range(j_block, j_end):
                    w[i] = w[i] + alpha * A[i, j] * x[j]

# Blocking is most useful when data loaded into the cache is reused several times.
# A tile does not remain in the cache for significant repeated computation. 
# The extra block loops therefore add complexity without creating much additional reuse.


# ---------------------------------------------------------
# Baseline 6: Blocked with np.dot / np.outer
# - Replaces inner tile loops with NumPy/BLAS-style operations.
        # - NumPy vs. Numba-compiled scalar loops
# - np.outer handles rank-1 tile updates; np.dot handles matrix-vector tile work.
# - Non-contiguous transposed tiles trigger the Numba performance warning.
# - Small calls and temporary arrays offset the benefit of optimized library routines.
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def gemver_blocked_np_dot_6(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]
    bs = BLOCK_SIZE
    num_blocks = (n + bs - 1) // bs

    for b in prange(num_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)
        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            A[i_block:i_end, j_block:j_end] += (
                np.outer(u1[i_block:i_end], v1[j_block:j_end])
                + np.outer(u2[i_block:i_end], v2[j_block:j_end])
            )

    for b in prange(num_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)
        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            x[i_block:i_end] += beta * np.dot(
                A[j_block:j_end, i_block:i_end].T,
                y[j_block:j_end]
            )

    for i in prange(n):
        x[i] = x[i] + z[i]

    for b in prange(num_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)
        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            w[i_block:i_end] += alpha * np.dot(
                A[i_block:i_end, j_block:j_end],
                x[j_block:j_end]
            )

# potential disadvantage
# Each np.outer() creates a temporary tile. Baseline 5 updates A directly.
# operations are small, some inputs are non-contiguous, 
# and temporary arrays are created, so NumPy does not produce a major speedup.


# ---------------------------------------------------------
# Baseline 7: Blocked Temporary Copies
# - Copies A tiles and x/w segments into contiguous temporary arrays.
# - Updates each temporary, then writes the result back.
# - Result: 0.311 ms at N=512; slightly slower than Baseline 6 at N=4096.
# - Extra copies add memory traffic, while GEMVER does not reuse each tile enough to recover the cost.
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def gemver_blocked_temp_copy_7(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]
    bs = BLOCK_SIZE
    num_blocks = (n + bs - 1) // bs

    for b in prange(num_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)
        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            temp = A[i_block:i_end, j_block:j_end].copy()
            temp += (
                np.outer(u1[i_block:i_end], v1[j_block:j_end])
                + np.outer(u2[i_block:i_end], v2[j_block:j_end])
            )
            A[i_block:i_end, j_block:j_end] = temp

    for b in prange(num_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)
        temp_x = x[i_block:i_end].copy()
        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            temp_x += beta * np.dot(
                A[j_block:j_end, i_block:i_end].T,
                y[j_block:j_end]
            )
        x[i_block:i_end] = temp_x

    for i in prange(n):
        x[i] = x[i] + z[i]

    for b in prange(num_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)
        temp_w = w[i_block:i_end].copy()
        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            temp_w += alpha * np.dot(
                A[i_block:i_end, j_block:j_end],
                x[j_block:j_end]
            )
        w[i_block:i_end] = temp_w

# baseline 7:
# A tests whether copying a matrix tile into contiguous storage helps tile processing.
# x and w test whether keeping partial accumulations local reduces repeated loads and stores.
# Result: slower because extra copy traffic outweighs any locality benefit.


# ---------------------------------------------------------
# Baseline 8: Two-Level Blocking
# - Uses 128-wide outer blocks and 32-wide inner blocks for L2/L1 locality. Direct port from matmul.
# - Result: 0.515 ms at N=512, the slowest blocked version.
# - Only four outer blocks exist at N=512, so most of the 24 threads receive no work.
# - Multi-level cache blocking helps only when data is reused; GEMVER mostly streams A.
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def gemver_two_level_blocked_8(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]
    l2 = BLOCK_SIZE_L2
    l1 = BLOCK_SIZE_L1
    num_l2_blocks = (n + l2 - 1) // l2

    for b2 in prange(num_l2_blocks):
        i2_start = b2 * l2
        i2_end = min(i2_start + l2, n)
        for j2_start in range(0, n, l2):
            j2_end = min(j2_start + l2, n)
            for i1_start in range(i2_start, i2_end, l1):
                i1_end = min(i1_start + l1, i2_end)
                for j1_start in range(j2_start, j2_end, l1):
                    j1_end = min(j1_start + l1, j2_end)
                    A[i1_start:i1_end, j1_start:j1_end] += (
                        np.outer(u1[i1_start:i1_end], v1[j1_start:j1_end])
                        + np.outer(u2[i1_start:i1_end], v2[j1_start:j1_end])
                    )

    for b2 in prange(num_l2_blocks):
        i2_start = b2 * l2
        i2_end = min(i2_start + l2, n)
        temp_l2 = x[i2_start:i2_end].copy()
        for j2_start in range(0, n, l2):
            j2_end = min(j2_start + l2, n)
            for i1_start in range(i2_start, i2_end, l1):
                i1_end = min(i1_start + l1, i2_end)
                i1_rel_start = i1_start - i2_start
                i1_rel_end = i1_end - i2_start
                temp_l1 = temp_l2[i1_rel_start:i1_rel_end].copy()
                for j1_start in range(j2_start, j2_end, l1):
                    j1_end = min(j1_start + l1, j2_end)
                    temp_l1 += beta * np.dot(
                        A[j1_start:j1_end, i1_start:i1_end].T,
                        y[j1_start:j1_end]
                    )
                temp_l2[i1_rel_start:i1_rel_end] = temp_l1
        x[i2_start:i2_end] = temp_l2

    for i in prange(n):
        x[i] = x[i] + z[i]

    for b2 in prange(num_l2_blocks):
        i2_start = b2 * l2
        i2_end = min(i2_start + l2, n)
        temp_l2 = w[i2_start:i2_end].copy()
        for j2_start in range(0, n, l2):
            j2_end = min(j2_start + l2, n)
            for i1_start in range(i2_start, i2_end, l1):
                i1_end = min(i1_start + l1, i2_end)
                i1_rel_start = i1_start - i2_start
                i1_rel_end = i1_end - i2_start
                temp_l1 = temp_l2[i1_rel_start:i1_rel_end].copy()
                for j1_start in range(j2_start, j2_end, l1):
                    j1_end = min(j1_start + l1, j2_end)
                    temp_l1 += alpha * np.dot(
                        A[i1_start:i1_end, j1_start:j1_end],
                        x[j1_start:j1_end]
                    )
                temp_l2[i1_rel_start:i1_rel_end] = temp_l1
        w[i2_start:i2_end] = temp_l2


# ---------------------------------------------------------
# Baseline 9: Zero-Allocation Blocking
# - Removes np.outer, np.dot, and per-tile temporary arrays from blocked loops.
        # zero repeated per-tile allocation of large temporary arrays (inside main tile processing loops).
# - Uses one small accumulator per block/thread and plain compiled loops.
# - Result: 0.261 ms at N=512; close to the other blocked variants at N=4096.
# - Conclusion: allocation was not the main bottleneck; matrix traffic remained unchanged.
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def gemver_blocked_zero_alloc_9(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]
    bs = BLOCK_SIZE
    num_blocks = (n + bs - 1) // bs

    for b in prange(num_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)
        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            for i in range(i_block, i_end):
                a1 = u1[i]
                a2 = u2[i]
                for j in range(j_block, j_end):
                    A[i, j] = A[i, j] + a1 * v1[j] + a2 * v2[j]

    for b in prange(num_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)
        h = i_end - i_block

        acc_tile = np.empty(bs, dtype=A.dtype)
        acc = acc_tile[:h]
        for ii in range(h):
            acc[ii] = x[i_block + ii]

        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            for j in range(j_block, j_end):
                c = beta * y[j]
                for ii in range(h):
                    acc[ii] += c * A[j, i_block + ii]

        for ii in range(h):
            x[i_block + ii] = acc[ii]

    for i in prange(n):
        x[i] = x[i] + z[i]

    for b in prange(num_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)
        h = i_end - i_block

        acc_tile = np.empty(bs, dtype=A.dtype)
        acc = acc_tile[:h]
        acc.fill(0.0)

        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            for ii in range(h):
                s = 0.0
                for j in range(j_block, j_end):
                    s += A[i_block + ii, j] * x[j]
                acc[ii] += s

        for ii in range(h):
            w[i_block + ii] = w[i_block + ii] + alpha * acc[ii]


# ---------------------------------------------------------
# Baseline 10: NumPy Reference
# - Expresses GEMVER with np.outer and np.dot; external BLAS is forced to one thread.
# - Provides a concise reference implementation and expected output.
# - Result: 1.75 ms at N=512; 115 ms at N=4096.
# - Stage 1 creates large temporary matrices, which adds significant memory traffic.
# ---------------------------------------------------------

def gemver_numpy_10(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    A += np.outer(u1, v1) + np.outer(u2, v2)
    x += beta * np.dot(A.T, y)
    x += z
    w += alpha * np.dot(A, x)

# stages 2 and 4 are single BLAS gemv calls, which are well
# optimized (single-threaded here, see the *_NUM_THREADS settings). Stage 1
# builds two full N x N np.outer temporaries and then adds them, which is about
# 3x the memory traffic of the fused in-place loop. Serves as a ceiling for
# the serial Numba versions and a floor for the parallel ones.


# ---------------------------------------------------------
# Baseline 11: Fused Stages 1 and 2
# - Updates each row of A and immediately uses the new values to update x.
            # combined into one matrix traversal
# - Keeps row-major stride-1 access and enables fastmath/SIMD.
# - Stage 4 remains separate because it requires the completed x vector.
# - Result: 0.153 ms at N=512; 14.2 ms at N=4096.
# - loop fusion removes one full pass over A and reduces memory traffic.
# ---------------------------------------------------------

# Stage 4 cannot be fused because it needs the final x.
# Stage 4 accumulates in a scalar and applies alpha once per row.

@njit(fastmath=True)
def gemver_fused_11(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]

    # combined stage 1 and 2: update A and x in one pass
    for i in range(n):
        # values reused across the row
        a1 = u1[i]
        a2 = u2[i]
        c = beta * y[i]
        for j in range(n):
            # update one element of A and immediately use it to update x
            a = A[i, j] + a1 * v1[j] + a2 * v2[j]
            A[i, j] = a
            x[j] += c * a

    for i in range(n):
        x[i] = x[i] + z[i]

    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i, j] * x[j]
        w[i] = w[i] + alpha * s

# does not change the GEMVER mathematics
# changes when Stage 2 consumes each updated element of A, reducing memory traffic.


# ---------------------------------------------------------
# Baseline 12: Fused and Parallel
# - Splits rows among threads while preserving the fused stage 1+2 sweep.
# - Each thread writes to a private partial x vector to prevent a data race.
# - Partial vectors are reduced into x; stage 4 is parallelized by row.
# - Result: 0.140 ms at N=512; 14.2 ms at N=4096, fastest overall/tied at large N.
# - Limited scaling shows that extra cores cannot overcome memory-bandwidth limits.
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def gemver_fused_parallel_12(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]
    num_chunks = get_num_threads()
    chunk = (n + num_chunks - 1) // num_chunks
    partial = np.zeros((num_chunks, n), dtype=A.dtype)

    for t in prange(num_chunks):
        row_start = t * chunk
        row_end = min(row_start + chunk, n)
        for i in range(row_start, row_end):
            a1 = u1[i]
            a2 = u2[i]
            c = beta * y[i]
            for j in range(n):
                a = A[i, j] + a1 * v1[j] + a2 * v2[j]
                A[i, j] = a
                partial[t, j] += c * a

    for j in prange(n):
        s = x[j]
        for t in range(num_chunks):
            s += partial[t, j]
        x[j] = s + z[j]

    for i in prange(n):
        s = 0.0
        for j in range(n):
            s += A[i, j] * x[j]
        w[i] = w[i] + alpha * s


# ---------------------------------------------------------
# Removed after testing: Baselines 13 and 14
# - 13 pre-transposed A: building a contiguous copy of A^T costs a full strided
#   pass over A. Slower than naive Baseline 1 (154 ms vs 116 ms at N=4096).
#   Loop interchange (Baseline 2) gets stride-1 access for free.
# - 14 scalar accumulators (x[i]/w[i] kept in a local variable): no measurable
#   change. LLVM already keeps the running sum in a register.
# ---------------------------------------------------------


# ---------------------------------------------------------
# Baseline 15: Tiling + Register Microkernel
# - Builds on Baseline 9 (same 32x32 tiles, same parallel split, fastmath).
# - Microkernel = the small innermost piece of a tiled loop that works on a
#   tiny block (here 4 rows) entirely in CPU registers.
#     - Tiling (Baseline 9) keeps data in the L1 cache.
#     - The microkernel adds one more level: reuse inside registers.
# - Stage 4 (w = A*x): 4 rows at once (unroll i by 4, "jam" into the j loop)
#     - x[j] is loaded once and used 4 times.
#     - 4 independent sums (s0..s3) instead of 1 dependency chain.
# - Stage 2 (x += A^T*y): 4 rows of A per pass over the x tile
#     - each acc[ii] is loaded/stored once per 4 rows instead of every row.
# - Stage 1 unchanged: each A element is touched once and v1[j]/v2[j] already
#   sit in L1, so there is nothing to keep in registers.
# - Expect: small gain at N=512 (A in L3), ~no gain at N=4096 (DRAM-bound).
#   It reduces register/L1 traffic, not DRAM traffic.
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def gemver_tiled_microkernel_15(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]
    bs = BLOCK_SIZE
    num_blocks = (n + bs - 1) // bs

    # Stage 1: same as Baseline 9
    for b in prange(num_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)
        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            for i in range(i_block, i_end):
                a1 = u1[i]
                a2 = u2[i]
                for j in range(j_block, j_end):
                    A[i, j] = A[i, j] + a1 * v1[j] + a2 * v2[j]

    # Stage 2: each thread owns a 32-wide block of x (no data race)
    for b in prange(num_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)
        h = i_end - i_block

        acc_tile = np.empty(bs, dtype=A.dtype)
        acc = acc_tile[:h]
        for ii in range(h):
            acc[ii] = x[i_block + ii]

        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            j = j_block
            # microkernel: 4 rows of A update the x tile in one pass
            while j + 4 <= j_end:
                c0 = beta * y[j]
                c1 = beta * y[j + 1]
                c2 = beta * y[j + 2]
                c3 = beta * y[j + 3]
                for ii in range(h):
                    i = i_block + ii
                    acc[ii] += (c0 * A[j, i] + c1 * A[j + 1, i]
                                + c2 * A[j + 2, i] + c3 * A[j + 3, i])
                j += 4
            # leftover rows when the tile height is not a multiple of 4
            while j < j_end:
                c = beta * y[j]
                for ii in range(h):
                    acc[ii] += c * A[j, i_block + ii]
                j += 1

        for ii in range(h):
            x[i_block + ii] = acc[ii]

    for i in prange(n):
        x[i] = x[i] + z[i]

    # Stage 4: each thread owns a 32-row block of w
    for b in prange(num_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)
        h = i_end - i_block

        acc_tile = np.zeros(bs, dtype=A.dtype)
        acc = acc_tile[:h]

        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            ii = 0
            # microkernel: 4 rows x one 32-wide tile, 4 sums in registers
            while ii + 4 <= h:
                i = i_block + ii
                s0 = 0.0
                s1 = 0.0
                s2 = 0.0
                s3 = 0.0
                for j in range(j_block, j_end):
                    xj = x[j]          # loaded once, used 4 times
                    s0 += A[i, j] * xj
                    s1 += A[i + 1, j] * xj
                    s2 += A[i + 2, j] * xj
                    s3 += A[i + 3, j] * xj
                acc[ii] += s0
                acc[ii + 1] += s1
                acc[ii + 2] += s2
                acc[ii + 3] += s3
                ii += 4
            # leftover rows
            while ii < h:
                i = i_block + ii
                s = 0.0
                for j in range(j_block, j_end):
                    s += A[i, j] * x[j]
                acc[ii] += s
                ii += 1

        for ii in range(h):
            w[i_block + ii] = w[i_block + ii] + alpha * acc[ii]


# ---------------------------------------------------------
# Baseline 16: Hoisting and Common-Subexpression Elimination (CSE) check
# - Bentley rules from the lecture slides: hoisting (loop-invariant code
#   motion) and CSE.
# - Baseline 11 already hoists by hand: a1 = u1[i], a2 = u2[i], c = beta*y[i]
#   are computed once per row, and the new A value is reused via "a".
# - Baseline 16 is Baseline 11 with that hand-work UNDONE:
#     - u1[i], u2[i], beta*y[i] are written inside the inner j loop
#     - A[i, j] is read back from memory instead of reusing "a"
# - Question it answers: does the compiler (LLVM) do this for us?
#     - Same time as 11 -> yes, the compiler hoists it (check with "asm" mode).
#     - Slower than 11  -> no, the hand hoisting in 11 matters.
# ---------------------------------------------------------
@njit(fastmath=True)
def gemver_no_hoist_16(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]

    for i in range(n):
        for j in range(n):
            # loop-invariant values left inside the j loop on purpose
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]
            x[j] += beta * y[i] * A[i, j]

    for i in range(n):
        x[i] = x[i] + z[i]

    # stage 4 identical to Baseline 11, so only stage 1+2 differs
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i, j] * x[j]
        w[i] = w[i] + alpha * s


# ---------------------------------------------------------
# Baseline 17: Fused + Unroll-and-Jam (serial and parallel)
# - Builds on Baseline 11 (fused stages 1+2), the best serial version.
# - Unroll the OUTER i loop by 4 and "jam" the 4 copies into one j loop.
#   (The inner j loop is already unrolled/vectorized by LLVM with fastmath,
#   so unrolling it by hand would add nothing.)
# - Stage 1+2: each pass updates 4 rows of A, then x[j] once with all 4.
#     - x[j] load/store drops from once per row to once per 4 rows.
#     - v1[j] and v2[j] are loaded once for 4 rows.
# - Stage 4: same 4-row microkernel idea as Baseline 15 (x[j] loaded once).
# - A is still read/written the same number of times (3 passes), so DRAM
#   traffic does not change. Expect a gain at N=512, little at N=4096.
# - The parallel version gives each thread a chunk of rows and a private
#   partial x (same race-free scheme as Baseline 12).
# ---------------------------------------------------------
@njit(fastmath=True)
def _fused_rows_4(beta, A, u1, v1, u2, v2, y, xout, row_start, row_end):
    # Stage 1+2 for rows [row_start, row_end), accumulating A^T*y into xout.
    n = A.shape[1]
    i = row_start
    while i + 4 <= row_end:
        a10 = u1[i]
        a11 = u1[i + 1]
        a12 = u1[i + 2]
        a13 = u1[i + 3]
        a20 = u2[i]
        a21 = u2[i + 1]
        a22 = u2[i + 2]
        a23 = u2[i + 3]
        c0 = beta * y[i]
        c1 = beta * y[i + 1]
        c2 = beta * y[i + 2]
        c3 = beta * y[i + 3]
        for j in range(n):
            v1j = v1[j]
            v2j = v2[j]
            r0 = A[i, j] + a10 * v1j + a20 * v2j
            r1 = A[i + 1, j] + a11 * v1j + a21 * v2j
            r2 = A[i + 2, j] + a12 * v1j + a22 * v2j
            r3 = A[i + 3, j] + a13 * v1j + a23 * v2j
            A[i, j] = r0
            A[i + 1, j] = r1
            A[i + 2, j] = r2
            A[i + 3, j] = r3
            xout[j] += c0 * r0 + c1 * r1 + c2 * r2 + c3 * r3
        i += 4
    # leftover rows (n not a multiple of 4)
    while i < row_end:
        a1 = u1[i]
        a2 = u2[i]
        c = beta * y[i]
        for j in range(n):
            a = A[i, j] + a1 * v1[j] + a2 * v2[j]
            A[i, j] = a
            xout[j] += c * a
        i += 1


@njit(fastmath=True)
def _matvec_rows_4(alpha, A, x, w, row_start, row_end):
    # Stage 4 for rows [row_start, row_end): w += alpha * A * x, 4 rows at once.
    n = A.shape[1]
    i = row_start
    while i + 4 <= row_end:
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        for j in range(n):
            xj = x[j]
            s0 += A[i, j] * xj
            s1 += A[i + 1, j] * xj
            s2 += A[i + 2, j] * xj
            s3 += A[i + 3, j] * xj
        w[i] += alpha * s0
        w[i + 1] += alpha * s1
        w[i + 2] += alpha * s2
        w[i + 3] += alpha * s3
        i += 4
    while i < row_end:
        s = 0.0
        for j in range(n):
            s += A[i, j] * x[j]
        w[i] += alpha * s
        i += 1


@njit(fastmath=True)
def gemver_fused_unroll_jam_17(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]
    _fused_rows_4(beta, A, u1, v1, u2, v2, y, x, 0, n)
    for i in range(n):
        x[i] = x[i] + z[i]
    _matvec_rows_4(alpha, A, x, w, 0, n)


@njit(parallel=True, fastmath=True)
def gemver_fused_unroll_jam_parallel_17(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]
    num_chunks = get_num_threads()
    # chunk is a multiple of 4 so the unrolled path covers almost every row
    chunk = ((n + num_chunks - 1) // num_chunks + 3) // 4 * 4
    partial = np.zeros((num_chunks, n), dtype=A.dtype)

    for t in prange(num_chunks):
        row_start = min(t * chunk, n)
        row_end = min(row_start + chunk, n)
        _fused_rows_4(beta, A, u1, v1, u2, v2, y, partial[t], row_start, row_end)

    for j in prange(n):
        s = x[j]
        for t in range(num_chunks):
            s += partial[t, j]
        x[j] = s + z[j]

    for t in prange(num_chunks):
        row_start = min(t * chunk, n)
        row_end = min(row_start + chunk, n)
        _matvec_rows_4(alpha, A, x, w, row_start, row_end)


# ---------------------------------------------------------
# Memory Bandwidth Reference (not a GEMVER baseline)
# - Measures how fast this machine can stream data from DRAM.
# - Kernel a = a + s*b: reads a, reads b, writes a (3 x 8 bytes per element).
#     - Same read-modify-write pattern as GEMVER's update of A,
#       so no hidden write-allocate traffic.
# - Gives the ceiling used for the "% of peak" column.
# ---------------------------------------------------------
STREAM_N = 1 << 24          # 16M doubles = 128 MiB per array, far larger than L3

@njit(fastmath=True)
def _stream_update_serial(a, b, s):
    for i in range(a.shape[0]):
        a[i] = a[i] + s * b[i]

@njit(parallel=True, fastmath=True)
def _stream_update_parallel(a, b, s):
    for i in prange(a.shape[0]):
        a[i] = a[i] + s * b[i]

def measure_bandwidth(reps=10):
    """Return (serial GB/s, parallel GB/s) for the a = a + s*b kernel."""
    a = np.ones(STREAM_N)
    b = np.ones(STREAM_N)
    nbytes = 3 * 8 * STREAM_N
    out = []
    for fn in (_stream_update_serial, _stream_update_parallel):
        fn(a, b, 1e-9)          # compile + touch pages
        times = []
        for _ in range(reps):
            start = time.perf_counter()
            fn(a, b, 1e-9)
            times.append(time.perf_counter() - start)
        out.append(nbytes / min(times) / 1e9)
    return out[0], out[1]


# Passes over A (each pass = 8*N^2 bytes) used for the effective GB/s column.
# - Unfused: stage 1 read + stage 1 write + stage 2 read + stage 4 read = 4.
# - Fused (11, 12, 16, 17): stage 2 reuses A while it is in a register = 3.
def gemver_passes_over_A(name):
    fused = ("11_", "12_", "16_", "17_")
    return 3 if name.startswith(fused) else 4

# Multi-threaded baselines are compared with the parallel bandwidth peak,
# single-threaded ones with the serial peak (one core cannot use all of DRAM).
def gemver_is_parallel(name):
    parallel = ("4_", "5_", "6_", "7_", "8_", "9_", "12_", "15_", "17_fused_unroll_jam_par")
    return name.startswith(parallel)


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

    gemver_numpy_10(alpha, beta, A_result, u1, v1, u2, v2, w_result, x_result, y, z)

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

    def measure(fn, warmup=True, min_reps=9, min_time=0.3, max_reps=200):
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
        # Repeat until enough time has been measured, then take the median
        # (short N=512 runs are noisy; the median ignores outliers).
        times = []
        while len(times) < min_reps or (sum(times) < min_time and len(times) < max_reps):
            times.append(run_once(fn)[0])
        elapsed = float(np.median(times))
        gflops = (total_flops / elapsed) / 1e9

        return gflops, elapsed, is_correct

    results = []

    # 0. Pure Python (GEMVER is O(N^2), so no size cap is needed for
    # moderate N, but very large N will still take a long time)
    gflops, elapsed, is_correct = measure(gemver_python_0, warmup=False, min_reps=1, min_time=0.0)
    results.append(("0_python_naive", gflops, elapsed, is_correct))

    # Baseline functions list
    functions = [
        ("1_numba_naive", gemver_numba_1),
        ("2_order_ij_ij_ij", gemver_2_ij_ij_ij),
        ("2_order_ij_ji_ij", gemver_2_ij_ji_ij),
        ("2_order_ji_ji_ji", gemver_2_ji_ji_ji),
        ("2_order_ji_ij_ji", gemver_2_ji_ij_ji),
        ("3_fastmath_ij_ji_ij", gemver_opt_flags_3),
        ("4_parallel_rows", gemver_parallel_rows_4),
        ("4_parallel_inner", gemver_parallel_inner_4),
        ("5_blocked_parallel", gemver_blocked_parallel_5),
        ("6_blocked_np_dot", gemver_blocked_np_dot_6),
        ("7_blocked_temp_copy", gemver_blocked_temp_copy_7),
        ("8_two_level_blocked", gemver_two_level_blocked_8),
        ("9_blocked_zero_alloc", gemver_blocked_zero_alloc_9),
        ("10_numpy", gemver_numpy_10),
        ("11_fused_fastmath", gemver_fused_11),
        ("12_fused_parallel", gemver_fused_parallel_12),
        ("15_tiled_microkernel", gemver_tiled_microkernel_15),
        ("16_no_hoist", gemver_no_hoist_16),
        ("17_fused_unroll_jam", gemver_fused_unroll_jam_17),
        ("17_fused_unroll_jam_par", gemver_fused_unroll_jam_parallel_17),
    ]

    for name, fn in functions:
        gflops, elapsed, is_correct = measure(fn)
        results.append((name, gflops, elapsed, is_correct))

    return results


# ---------------------------------------------------------
# Thread-Count Sweep
# ---------------------------------------------------------
# Run with:  python gemver_numba_bench.py <N> threads
# - Runs the parallel baselines at 1..24 threads (5900X: 12 cores / 24 threads).
# - Compares each one with the best serial version (11_fused_fastmath).
# - Expectation: GEMVER is memory-bound, so threads help only until memory
#   bandwidth is used up. Curves should flatten after a few threads at N=4096.
# - Observed (5900X), N=4096:
#     - 12_fused_parallel peaks at 2-4 threads (~1.25x over its own 1 thread,
#       ~1.4x over serial 11), then drops back to ~1.0x at 24 threads.
#       Once memory is saturated, more threads only add contention.
#     - 4_parallel_rows scales best (~4.8x) only because its 1-thread run is
#       slow (strided stage 2 waits on cache misses). It never reaches serial 11.
#       Good scaling is not the same as good performance.
#     - 5_blocked_parallel flattens at ~2x by 6 threads.
# - Observed (5900X), N=1024 (A = 8 MB, fits in L3): under 1 ms per run and
#   too noisy (1.2x and 2.7x for the same setting), so no conclusion drawn.
THREAD_COUNTS = [1, 2, 4, 6, 8, 12, 16, 24]

def run_thread_sweep(vector_size=DEFAULT_N):
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
    max_threads = get_num_threads()
    thread_counts = [t for t in THREAD_COUNTS if t <= max_threads]

    def run_once(fn):
        A = A_initial.copy()
        x = x_initial.copy()
        w = w_initial.copy()

        start = time.perf_counter()
        fn(ALPHA, BETA, A, u1, v1, u2, v2, w, x, y, z)
        elapsed = time.perf_counter() - start

        return elapsed, A, x, w

    def measure(fn, min_reps=9, min_time=0.3, max_reps=200):
        # Warm-up call: compiles on first use and starts the thread pool.
        run_once(fn)

        # Repeat until enough time has been measured, then take the median,
        # which is less sensitive to outliers than the mean for short kernels.
        times = []
        while len(times) < min_reps or (sum(times) < min_time and len(times) < max_reps):
            elapsed, A, x, w = run_once(fn)
            times.append(elapsed)

        is_correct = (
            np.allclose(A, A_expected, rtol=1e-5, atol=1e-5)
            and np.allclose(x, x_expected, rtol=1e-5, atol=1e-5)
            and np.allclose(w, w_expected, rtol=1e-5, atol=1e-5)
        )

        elapsed = float(np.median(times))
        gflops = (total_flops / elapsed) / 1e9

        return gflops, elapsed, is_correct

    # Serial reference: the best single-threaded version
    serial = measure(gemver_fused_11)

    kernels = [
        ("4_parallel_rows", gemver_parallel_rows_4),
        ("5_blocked_parallel", gemver_blocked_parallel_5),
        ("12_fused_parallel", gemver_fused_parallel_12),
        ("17_fused_unroll_jam_par", gemver_fused_unroll_jam_parallel_17),
    ]

    results = []
    for name, fn in kernels:
        rows = []
        for t in thread_counts:
            set_num_threads(t)
            gflops, elapsed, is_correct = measure(fn)
            rows.append((t, gflops, elapsed, is_correct))
        results.append((name, rows))

    set_num_threads(max_threads)

    return thread_counts, serial, results


def report_thread_sweep(vector_size):
    print(f"\nRunning GEMVER Thread-Count Sweep for N={vector_size}...")
    print(f"CPU: {CPU_NAME}\n")

    thread_counts, serial, results = run_thread_sweep(vector_size)
    serial_gflops, serial_elapsed, serial_correct = serial

    print(f"Numba threading layer: {threading_layer()}\n")

    header = (
        f"| {'Baseline Implementation':<28} "
        f"| {'Threads':>7} "
        f"| {'GFLOP/s':>10} "
        f"| {'Time (s)':>12} "
        f"| {'vs 1 Thread':>12} "
        f"| {'vs Serial 11':>12} "
        f"| {'Correct':<8} |"
    )
    divider = "-" * len(header)

    print(divider)
    print(header)
    print(divider)

    status = "PASS" if serial_correct else "FAIL"
    print(
        f"| {'11_fused_fastmath (serial)':<28} "
        f"| {'-':>7} "
        f"| {serial_gflops:10.4f} "
        f"| {serial_elapsed:12.6f} "
        f"| {'-':>12} "
        f"| {1.0:11.2f}x "
        f"| {status:<8} |"
    )
    print(divider)

    for name, rows in results:
        one_thread_elapsed = rows[0][2]
        for t, gflops, elapsed, is_correct in rows:
            status = "PASS" if is_correct else "FAIL"
            print(
                f"| {name:<28} "
                f"| {t:>7} "
                f"| {gflops:10.4f} "
                f"| {elapsed:12.6f} "
                f"| {one_thread_elapsed / elapsed:11.2f}x "
                f"| {serial_elapsed / elapsed:11.2f}x "
                f"| {status:<8} |"
            )
        print(divider)

    # Plotting Output: throughput (left) and self-relative speedup (right)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5cd9"]
    markers = ["o", "s", "^", "D"]
    muted = "#898781"

    fig, (ax_gflops, ax_speedup) = plt.subplots(1, 2, figsize=(16, 6))

    max_speedup = 1.0
    for (name, rows), color, marker in zip(results, colors, markers):
        gflops_vals = [row[1] for row in rows]
        speedups = [rows[0][2] / row[2] for row in rows]
        max_speedup = max(max_speedup, max(speedups))

        ax_gflops.plot(thread_counts, gflops_vals, color=color, marker=marker,
                       linewidth=2, markersize=8, label=name)
        ax_speedup.plot(thread_counts, speedups, color=color, marker=marker,
                        linewidth=2, markersize=8, label=name)

    ax_gflops.axhline(serial_gflops, color=muted, linestyle="--", linewidth=1.5,
                      label="11_fused_fastmath (serial)")
    ax_gflops.set_ylim(bottom=0)
    ax_gflops.set_ylabel("GFLOP/s (Higher is better)")
    ax_gflops.set_title("Throughput")

    ax_speedup.plot(thread_counts, thread_counts, color=muted, linestyle="--",
                    linewidth=1.5, label="ideal (linear)")
    ax_speedup.set_ylim(0, max_speedup * 1.5)
    ax_speedup.set_ylabel("Speedup over the same kernel on 1 thread")
    ax_speedup.set_title("Scaling")

    for ax in (ax_gflops, ax_speedup):
        if 12 in thread_counts:
            ax.axvline(12, color=muted, linestyle=":", linewidth=1)
            ax.text(12, ax.get_ylim()[1], " 12 physical cores", color=muted,
                    ha="left", va="top", fontsize=8)
        ax.set_xticks(thread_counts)
        ax.set_xlabel("Numba threads")
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle(f"Numba GEMVER Thread-Count Sweep (N={vector_size})\nCPU: {CPU_NAME}")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# Assembly Check
# ---------------------------------------------------------
# Run with:  python gemver_numba_bench.py <N> asm
# - Prints instruction counts from the machine code Numba generated.
#     - "vector FMA (ymm)": packed fused multiply-adds on 4 doubles -> SIMD is used.
#     - "scalar add/FMA": one double at a time -> no SIMD.
#     - "loads of u1/u2/y in hot loop" is not counted directly; compare the
#       total size of 11 vs 16 instead (same size -> the compiler hoisted).
# - Full assembly is written to asm/<name>.s so it can be opened and shown.
ASM_KERNELS = [
    ("1_numba_naive", gemver_numba_1),
    ("3_fastmath_ij_ji_ij", gemver_opt_flags_3),
    ("11_fused_fastmath", gemver_fused_11),
    ("16_no_hoist", gemver_no_hoist_16),
    ("17_fused_unroll_jam", gemver_fused_unroll_jam_17),
]

def report_asm(vector_size):
    import re
    A, u1, v1, u2, v2, w, x, y, z = initialize_gemver(vector_size)
    os.makedirs("asm", exist_ok=True)
    print(f"\n{'Kernel':<24} | {'vector FMA (ymm)':>16} | {'scalar add/FMA':>14} | {'instructions':>12}")
    print("-" * 76)
    for name, fn in ASM_KERNELS:
        fn(ALPHA, BETA, A.copy(), u1, v1, u2, v2, w.copy(), x.copy(), y, z)  # compile
        asm = "\n".join(fn.inspect_asm().values())
        if name.startswith("17_"):
            # the work happens in the two helpers, so include them
            for helper in (_fused_rows_4, _matvec_rows_4):
                asm += "\n".join(helper.inspect_asm().values())
        with open(os.path.join("asm", name + ".s"), "w") as fh:
            fh.write(asm)
        code = [l for l in asm.splitlines() if l.startswith("\t") and not l.startswith("\t.")]
        vec_fma = sum(1 for l in code if re.search(r"vfn?madd\w*pd\s.*ymm", l))
        scalar = sum(1 for l in code if re.search(r"\b(vaddsd|vfn?madd\w*sd)\b", l))
        print(f"{name:<24} | {vec_fma:>16} | {scalar:>14} | {len(code):>12}")
    print("\nFull assembly written to ./asm/")


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

    # Optional second argument selects a sweep instead of the baseline table.
    if len(sys.argv) > 2 and sys.argv[2] == "threads":
        report_thread_sweep(N_input)
        sys.exit(0)
    if len(sys.argv) > 2 and sys.argv[2] == "asm":
        report_asm(N_input)
        sys.exit(0)

    print(f"\nRunning GEMVER Benchmarks for N={N_input}...")
    print(f"CPU: {CPU_NAME}")
    print(f"Numba threads: {get_num_threads()}\n")

    benchmark_data = run_benchmark(N_input)

    # The threading layer is only chosen once a parallel function has run.
    print(f"Numba threading layer: {threading_layer()}\n")

    bw_serial, bw_parallel = measure_bandwidth()
    a_bytes = 8.0 * N_input * N_input
    print(f"Measured memory bandwidth (a = a + s*b, 128 MiB arrays): "
          f"{bw_serial:.1f} GB/s serial (1 thread), {bw_parallel:.1f} GB/s parallel")
    print(f"A = {a_bytes / 1e6:.1f} MB "
          f"({'fits in' if a_bytes <= 32e6 else 'larger than'} the 32 MB L3)."
          f" GB/s = passes over A x size of A / time.")
    print("% of peak: serial baselines vs the serial number, parallel ones vs the parallel number.")
    if a_bytes <= 32e6:
        print("A is served from cache at this size, so '% of peak' can exceed 100%.")
    print()

    py_elapsed = benchmark_data[0][2]  # Reference execution time for pure Python naive

    # Table Header Formatting
    header = (
        f"| {'Baseline Implementation':<28} "
        f"| {'GFLOP/s':>10} "
        f"| {'Time (s)':>12} "
        f"| {'Abs Speedup':>12} "
        f"| {'Rel Speedup':>12} "
        f"| {'GB/s':>7} "
        f"| {'% of peak':>11} "
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
        gbps = gemver_passes_over_A(name) * a_bytes / elapsed / 1e9
        bw_peak = bw_parallel if gemver_is_parallel(name) else bw_serial

        print(
            f"| {name:<28} "
            f"| {gflops:10.4f} "
            f"| {elapsed:12.6f} "
            f"| {abs_speedup:11.2f}x "
            f"| {rel_speedup:11.2f}x "
            f"| {gbps:7.1f} "
            f"| {100 * gbps / bw_peak:10.0f}% "
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
    plt.title(f"Numba GEMVER Benchmark Performance (N={N_input})\nCPU: {CPU_NAME} | Numba threads: {get_num_threads()}")
    plt.xticks(rotation=45, ha="right")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
