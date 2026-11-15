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
import sys

# "blas_mt" on the command line: let BLAS use all its threads (row 18_blas_mt)
# - Must be decided here: OpenBLAS reads its thread count once, when NumPy/SciPy load.
BLAS_MT = "blas_mt" in sys.argv[1:]

# Set single-threaded execution for external BLAS libraries to ensure reproducible benchmarks.
# These must be set before NumPy is imported to take effect.
if not BLAS_MT:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

import time
import matplotlib

# Image path (*.png) on the command line: draw off-screen with Agg (no display needed,
# as on the professor's run_bench.sh server). Must be set before pyplot is imported.
if any(arg.lower().endswith(".png") for arg in sys.argv[1:]):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from numba import njit, prange, get_num_threads, set_num_threads, threading_layer
import numpy as np
import scipy
from scipy.linalg import blas

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
# - The serial kernel calls the helpers with the constant start row 0, which
#   makes LLVM pick a slower (scalar) path. Fixed in Baseline 17b below.
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
# Baseline 17b: Baseline 17 with the start row passed in at run time
# - Same code as 17. Only change: the start row is an argument (the benchmark
#   passes 0) instead of the constant 0 written in the kernel.
# - Why that matters (found while testing Baseline 19):
#     - The 4-row loop writes 4 rows of A and x. To use SIMD, LLVM must know
#       they do not overlap in memory, so it adds a runtime check:
#       overlap -> scalar loop, no overlap -> SIMD loop.
#     - Start row known (0): LLVM hoists the check out of the row loop and runs
#       it ONCE for all rows. Row i (over all i) and row i+1 (over all i) cover
#       almost the same memory -> check says "overlap" -> scalar loop, every time.
#     - Start row unknown: the check stays per 4-row block -> passes -> SIMD.
# - Proof (5900X, S1+S2 alone, N=512, A in cache):
#     - constant 0: 124 us; start row as argument: 41 us (3x faster)
#     - constant 0 with LLVM option -hoist-runtime-checks=false: 37 us
# - Expect: clear gain at N=512 (compute matters), smaller at N=4096
#   (waiting on DRAM hides most of the scalar slowdown).
# - Observed (5900X, 17 vs 17b, alternating runs):
#     - N=512:  0.21 -> 0.18 ms (1.2x)
#     - N=4096: 1.04x-1.13x (the main table's +-10% noise can hide it)
# - 17_par was never affected: its row bounds are computed at run time.
# - Lesson: more compile-time knowledge is not always faster.
# ---------------------------------------------------------
@njit(fastmath=True)
def gemver_fused_unroll_jam_17b(alpha, beta, A, u1, v1, u2, v2, w, x, y, z, row_start):
    n = A.shape[0]
    _fused_rows_4(beta, A, u1, v1, u2, v2, y, x, row_start, n)
    for i in range(n):
        x[i] = x[i] + z[i]
    _matvec_rows_4(alpha, A, x, w, row_start, n)


# ---------------------------------------------------------
# Baseline 18: Optimized library baseline (direct BLAS, in place)
# - The "np.dot of GEMVER": vendor BLAS for every stage (matmul compares to np.dot).
# - GEMVER itself is not a routine in OpenBLAS or MKL (only in the 2001 BLAS
#   Technical Forum standard), so it is built from dgemm + dgemv calls.
# - S1 = one rank-2 dgemm: A += [u1 u2] [v1 v2]^T -> 1 read + 1 write of A.
#     - Baseline 10 instead builds two N x N np.outer temporaries.
# - S2, S4 = dgemv; S3 = vector add. No temporaries the size of A.
# - Row-major A is passed as A.T, which BLAS sees as a column-major matrix
#   (same memory, no copy). A += U V^T then becomes A.T += V U^T.
# - Still 4 passes over A: separate library calls cannot fuse stages.
# - Expect: about Baseline 3 speed (also 4 passes); slower than fused 11/17.
# - If SciPy ever copied instead of writing in place, A/x/w would stay
#   unchanged and the correctness check would show FAIL.
# - "blas_mt" mode runs it again with all BLAS threads (row 18_blas_mt).
# ---------------------------------------------------------
def gemver_blas_18(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]
    At = A.T                                  # column-major view of A, no copy

    U = np.empty((n, 2), order="F")           # [u1 u2], column-major like BLAS wants
    U[:, 0] = u1
    U[:, 1] = u2
    V = np.empty((n, 2), order="F")           # [v1 v2]
    V[:, 0] = v1
    V[:, 1] = v2

    # S1: At = 1.0 * V U^T + 1.0 * At   (one rank-2 update, written into A)
    blas.dgemm(1.0, V, U, beta=1.0, c=At, trans_b=1, overwrite_c=1)
    # S2: x = beta * At y + 1.0 * x     (At y = A^T y)
    blas.dgemv(beta, At, y, beta=1.0, y=x, overwrite_y=1)
    # S3
    x += z
    # S4: w = alpha * At^T x + 1.0 * w  (trans=1: At^T = A)
    blas.dgemv(alpha, At, x, beta=1.0, y=w, trans=1, overwrite_y=1)


# ---------------------------------------------------------
# Baseline 19: Full fusion of all 4 stages by column panels (+ micro-kernel)
# - Baseline 11/17: S4 needs the FINAL x, so it waits for every row -> 3rd pass over A.
# - But column j's part of S4 only needs x[j], and x[j] is final once
#   column j has been seen by every row. So walk A in column panels:
#     1. pass 1 over the panel: S1 + S2 for columns [jb, je)   (A from DRAM)
#     2. S3 for the panel: x[jb:je] += z[jb:je]                 (x[jb:je] final)
#     3. pass 2 over the SAME panel: w += alpha * A[:, jb:je] x[jb:je]
#        (the panel was just touched -> it should come from L3, not DRAM)
# - DRAM traffic: read A once + write A once = 2 passes (11/17: 3).
#   2 is the minimum: A is an output, so it must be read and written once.
# - Legal: S2 uses only updated A in its panel; S4 uses only final x and A.
#   w is summed panel by panel -> different rounding order (fine at 1e-5).
# - Panel width PW: the panel (N x PW x 8 bytes) must stay in L3 until pass 2.
# - Three versions:
#     - 19_fused_panel:     1 row at a time (fusion only)
#     - 19_fused_panel_mk:  4-row micro-kernel from 17 inside each panel
#     - 19_fused_panel_par: threads split the ROWS of each panel
# - Row slices A[i, jb:je] + a local j index: this form vectorizes (AVX2 FMAs);
#   indexing A[i, j] with j in range(jb, je) did not (3x slower in a first test).
# - Observed (5900X), N=4096, median of interleaved runs (17 = 15.6 ms):
#     - 19_mk vs panel width: 512 -> 17.1 ms, 1024 -> 14.8, 2048 -> 13.8,
#       4096 (one panel, so S4 is NOT fused) -> 13.0. Narrower is slower.
#     - Why panels do not pay off here:
#         - pass 1 on 4 KB row pieces is slower than on full rows (11.0 vs 9.3 ms)
#         - pass 2 is not fully served from L3 (the 16.8 MB panel gets partly evicted)
#     - 19_fused_panel (no micro-kernel) at PW=512: 21.3 ms -> the micro-kernel matters.
#     - 19_fused_panel_par: 13.4 ms at 8-12 threads, 14.3 ms at 24
#       (17_par: 13.8 ms at 2 threads, 19.7 ms at 24).
#     - Tried and not kept: PW as a compile-time constant (closure in a factory).
#       Numba still calls the generic helpers (same asm); writing the loops inline
#       with a fixed trip count was slower (20.6 vs 17.1 ms).
# - Side finding: 19_mk with one panel does 17's work but is faster (13.0 vs 15.6 ms).
#   The cause was in 17 (constant start row -> scalar path); see Baseline 17b.
# ---------------------------------------------------------
L3_BYTES = 32 * 1024 * 1024     # one CCD's L3, the most one core can use
PANEL_WIDTH = 512               # default for N=4096 (4096 x 512 x 8 B = 16.8 MB)

def default_panel_width(n):
    # - Largest power of two with N x PW x 8 bytes <= L3 / 2, and PW <= N.
    # - Half the L3: leaves room for x, w and the other vectors.
    pw = 8
    while 2 * pw <= n and n * (2 * pw) * 8 <= L3_BYTES // 2:
        pw *= 2
    return pw


@njit(fastmath=True)
def _panel_s12_rows1(beta, A, u1, v1, u2, v2, y, xp, jb, je, row_start, row_end):
    # - S1+S2 on columns [jb, je) for rows [row_start, row_end), one row at a time
    # - xp: the x entries of this panel (x[jb:je], or a thread's private copy)
    m = je - jb
    v1p = v1[jb:je]
    v2p = v2[jb:je]
    for i in range(row_start, row_end):
        r = A[i, jb:je]
        a1 = u1[i]
        a2 = u2[i]
        c = beta * y[i]
        for j in range(m):
            a = r[j] + a1 * v1p[j] + a2 * v2p[j]
            r[j] = a
            xp[j] += c * a


@njit(fastmath=True)
def _panel_s4_rows1(alpha, A, x, w, jb, je, row_start, row_end):
    # - S4 part for columns [jb, je): w[i] += alpha * A[i, jb:je] . x[jb:je]
    m = je - jb
    xp = x[jb:je]
    for i in range(row_start, row_end):
        r = A[i, jb:je]
        s = 0.0
        for j in range(m):
            s += r[j] * xp[j]
        w[i] += alpha * s


@njit(fastmath=True)
def _panel_s12_rows4(beta, A, u1, v1, u2, v2, y, xp, jb, je, row_start, row_end):
    # - Micro-kernel: same as 17's _fused_rows_4, but only on columns [jb, je)
    #     - v1[j], v2[j] loaded once for 4 rows; xp[j] loaded/stored once per 4 rows
    m = je - jb
    v1p = v1[jb:je]
    v2p = v2[jb:je]
    i = row_start
    while i + 4 <= row_end:
        r0 = A[i, jb:je]
        r1 = A[i + 1, jb:je]
        r2 = A[i + 2, jb:je]
        r3 = A[i + 3, jb:je]
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
        for j in range(m):
            p = v1p[j]
            q = v2p[j]
            e0 = r0[j] + a10 * p + a20 * q
            e1 = r1[j] + a11 * p + a21 * q
            e2 = r2[j] + a12 * p + a22 * q
            e3 = r3[j] + a13 * p + a23 * q
            r0[j] = e0
            r1[j] = e1
            r2[j] = e2
            r3[j] = e3
            xp[j] += c0 * e0 + c1 * e1 + c2 * e2 + c3 * e3
        i += 4
    # leftover rows (fewer than 4)
    _panel_s12_rows1(beta, A, u1, v1, u2, v2, y, xp, jb, je, i, row_end)


@njit(fastmath=True)
def _panel_s4_rows4(alpha, A, x, w, jb, je, row_start, row_end):
    # - Micro-kernel: x[j] loaded once, used for 4 rows; 4 independent sums
    m = je - jb
    xp = x[jb:je]
    i = row_start
    while i + 4 <= row_end:
        r0 = A[i, jb:je]
        r1 = A[i + 1, jb:je]
        r2 = A[i + 2, jb:je]
        r3 = A[i + 3, jb:je]
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        for j in range(m):
            xj = xp[j]
            s0 += r0[j] * xj
            s1 += r1[j] * xj
            s2 += r2[j] * xj
            s3 += r3[j] * xj
        w[i] += alpha * s0
        w[i + 1] += alpha * s1
        w[i + 2] += alpha * s2
        w[i + 3] += alpha * s3
        i += 4
    _panel_s4_rows1(alpha, A, x, w, jb, je, i, row_end)


@njit(fastmath=True)
def gemver_fused_panel_19(alpha, beta, A, u1, v1, u2, v2, w, x, y, z, pw=PANEL_WIDTH):
    # - Fusion only (1-row helpers): isolates the effect of the panel order
    n = A.shape[0]
    for jb in range(0, n, pw):
        je = min(jb + pw, n)            # last panel = cleanup panel (n % pw columns)
        _panel_s12_rows1(beta, A, u1, v1, u2, v2, y, x[jb:je], jb, je, 0, n)
        for j in range(jb, je):
            x[j] = x[j] + z[j]
        _panel_s4_rows1(alpha, A, x, w, jb, je, 0, n)


@njit(fastmath=True)
def gemver_fused_panel_mk_19(alpha, beta, A, u1, v1, u2, v2, w, x, y, z, pw=PANEL_WIDTH):
    # - Fusion + 4-row micro-kernel (the professor's "micro-kernel on fused code")
    n = A.shape[0]
    for jb in range(0, n, pw):
        je = min(jb + pw, n)
        # pass 1: S1+S2 on the panel (A comes from DRAM)
        _panel_s12_rows4(beta, A, u1, v1, u2, v2, y, x[jb:je], jb, je, 0, n)
        # S3: x[jb:je] is now final
        for j in range(jb, je):
            x[j] = x[j] + z[j]
        # pass 2: S4 on the same panel (A should come from L3)
        _panel_s4_rows4(alpha, A, x, w, jb, je, 0, n)


@njit(parallel=True, fastmath=True)
def gemver_fused_panel_par_19(alpha, beta, A, u1, v1, u2, v2, w, x, y, z, pw=PANEL_WIDTH):
    # - Panels in order; threads split the ROWS of each panel
    #     - pass 1: all threads add into x[jb:je] -> race -> private partial x per
    #       thread (only pw long), summed after pass 1 (same idea as Baseline 12)
    #     - pass 2: each thread owns its rows of w -> no race on w
    #     - a thread re-reads in pass 2 the rows it wrote in pass 1 (its own cache)
    # - Other split tried (each thread takes whole column panels, private w):
    #   slower (best 14.6 ms vs 13.4 ms); narrow panels per thread = short row pieces.
    n = A.shape[0]
    num_chunks = get_num_threads()
    chunk = ((n + num_chunks - 1) // num_chunks + 3) // 4 * 4    # multiple of 4 rows
    partial = np.zeros((num_chunks, pw), dtype=A.dtype)

    for jb in range(0, n, pw):
        je = min(jb + pw, n)
        m = je - jb

        for t in prange(num_chunks):
            row_start = min(t * chunk, n)
            row_end = min(row_start + chunk, n)
            xp = partial[t, :m]
            xp[:] = 0.0
            _panel_s12_rows4(beta, A, u1, v1, u2, v2, y, xp, jb, je, row_start, row_end)

        # sum the partial x's, then S3 -> x[jb:je] final
        for j in range(m):
            s = x[jb + j] + z[jb + j]
            for t in range(num_chunks):
                s += partial[t, j]
            x[jb + j] = s

        for t in prange(num_chunks):
            row_start = min(t * chunk, n)
            row_end = min(row_start + chunk, n)
            _panel_s4_rows4(alpha, A, x, w, jb, je, row_start, row_end)


# ---------------------------------------------------------
# Memory Bandwidth Reference (not a GEMVER baseline)
# - Measures how fast this machine can stream data from DRAM.
# - Kernel a = a + s*b: reads a, reads b, writes a (3 x 8 bytes per element).
#     - Same read-modify-write pattern as GEMVER's update of A,
#       so no hidden write-allocate traffic.
# - Gives the ceiling used for the "% of peak" column.
# - The parallel kernel is run at several thread counts and the best is kept.
#     - On the 5900X, 24 threads got LESS bandwidth than 1 thread (27.6 vs 37.2 GB/s).
#     - Too many memory streams at once can make DRAM slower, so the peak is
#       usually reached with a few threads, not all 24.
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
    """
    Return (serial GB/s, {threads: GB/s}) for the a = a + s*b kernel.
    The parallel kernel is measured at each thread count up to the maximum.
    """
    a = np.ones(STREAM_N)
    b = np.ones(STREAM_N)
    nbytes = 3 * 8 * STREAM_N

    def best_gbps(fn):
        fn(a, b, 1e-9)          # compile + touch pages
        times = []
        for _ in range(reps):
            start = time.perf_counter()
            fn(a, b, 1e-9)
            times.append(time.perf_counter() - start)
        return nbytes / min(times) / 1e9

    serial = best_gbps(_stream_update_serial)

    max_threads = get_num_threads()
    by_threads = {}
    for t in [t for t in [1, 2, 4, 6, 8, 12, 16, 24] if t < max_threads] + [max_threads]:
        set_num_threads(t)
        by_threads[t] = best_gbps(_stream_update_parallel)
    set_num_threads(max_threads)

    return serial, by_threads


# Passes over A (each pass = 8*N^2 bytes) used for the effective GB/s column.
# - Unfused: stage 1 read + stage 1 write + stage 2 read + stage 4 read = 4.
# - Fused (11, 12, 16, 17, 17b): stage 2 reuses A while it is in a register = 3.
# - BLAS (18): dgemm read + write, then 2 dgemv reads = 4 (the default below).
# - Column panels (19): read + write from DRAM, pass 2 from L3 = 2
#   (the ideal; it holds only while the panel stays in L3).
def gemver_passes_over_A(name):
    if name.startswith("19_"):
        return 2
    fused = ("11_", "12_", "16_", "17_", "17b_")
    return 3 if name.startswith(fused) else 4


# Roofline hook (used by demos/roofline.py)
# - Bytes moved = passes over A x 8 bytes x N^2 (vectors are O(N), ignored).
# - Unfused baselines read/write A 4 times, fused ones 3 times, panels (19) 2 times.
def roofline_bytes(name, n):
    return gemver_passes_over_A(name) * 8.0 * n * n


# Which BLAS is really called (professor: "make sure the call is the right BLAS")
# - NumPy (Baseline 10) and SciPy (Baseline 18) can each bundle their own BLAS build.
# - Library name/version from show_config(); thread count from threadpoolctl when it
#   is installed, otherwise from the environment variables set at the top of this file.
def blas_info():
    builds = {}
    for label, lib in (("numpy", np), ("scipy", scipy)):
        try:
            cfg = lib.show_config(mode="dicts")["Build Dependencies"]["blas"]
            builds[label] = f"{cfg.get('name', '?')} {cfg.get('version', '')}".strip()
        except Exception:
            builds[label] = "unknown"

    try:
        from threadpoolctl import threadpool_info
        pools = [p for p in threadpool_info() if p.get("user_api") == "blas"]
        threads = ", ".join(f"{p.get('internal_api')} {p.get('version')}: {p.get('num_threads')} thr"
                            for p in pools) or "no BLAS pool found"
        threads = "threadpoolctl -> " + threads
    except ImportError:
        if BLAS_MT:
            threads = "library default (blas_mt: thread pins not set)"
        else:
            threads = f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}"

    return (f"BLAS: NumPy uses {builds['numpy']}; SciPy (Baseline 18) uses {builds['scipy']}\n"
            f"BLAS threads: {threads}")


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
# Rows kept in "blas_mt" mode (multi-threaded BLAS vs the best Numba versions)
BLAS_MT_ROWS = (
    "3_fastmath_ij_ji_ij",
    "11_fused_fastmath",
    "17_fused_unroll_jam",
    "17b_unroll_jam_runtime_start",
    "17_fused_unroll_jam_par",
    "18_blas",
    "19_fused_panel_mk",
    "19_fused_panel_par",
)

def run_benchmark(vector_size=DEFAULT_N, blas_mt=False, pw=None):
    # pw: panel width for Baseline 19 (None = default_panel_width(N), or --pw on the command line)
    if pw is None:
        pw = default_panel_width(vector_size)

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

    def run_once(fn, extra=()):
        """
        Run fn on fresh copies of A, x, and w (GEMVER updates them in place)
        and return the elapsed kernel time along with the result arrays.
        extra: arguments after z (17b's start row, Baseline 19's panel width).
        """
        A = A_initial.copy()
        x = x_initial.copy()
        w = w_initial.copy()

        start = time.perf_counter()
        fn(ALPHA, BETA, A, u1, v1, u2, v2, w, x, y, z, *extra)
        elapsed = time.perf_counter() - start

        return elapsed, A, x, w

    def measure(fn, extra=(), warmup=True, min_reps=9, min_time=0.3, max_reps=200):
        # Warm-up call triggers Numba JIT compilation, so it is not timed.
        if warmup:
            run_once(fn, extra)

        # Correctness check against the NumPy reference results
        _, A, x, w = run_once(fn, extra)
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
            times.append(run_once(fn, extra)[0])
        elapsed = float(np.median(times))
        gflops = (total_flops / elapsed) / 1e9

        return gflops, elapsed, is_correct

    results = []

    # 0. Pure Python (GEMVER is O(N^2), so no size cap is needed for
    # moderate N, but very large N will still take a long time)
    if not blas_mt:
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
        # third element: extra arguments after z (17b: start row 0, 19: panel width)
        ("17b_unroll_jam_runtime_start", gemver_fused_unroll_jam_17b, (0,)),
        ("17_fused_unroll_jam_par", gemver_fused_unroll_jam_parallel_17),
        ("18_blas", gemver_blas_18),
        ("19_fused_panel", gemver_fused_panel_19, (pw,)),
        ("19_fused_panel_mk", gemver_fused_panel_mk_19, (pw,)),
        ("19_fused_panel_par", gemver_fused_panel_par_19, (pw,)),
    ]

    # blas_mt mode: small table, and the BLAS row is renamed to show its threads
    if blas_mt:
        functions = [("18_blas_mt" if entry[0] == "18_blas" else entry[0],) + tuple(entry[1:])
                     for entry in functions if entry[0] in BLAS_MT_ROWS]

    for name, fn, *extra in functions:
        gflops, elapsed, is_correct = measure(fn, tuple(extra[0]) if extra else ())
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

def run_thread_sweep(vector_size=DEFAULT_N, pw=None):
    if pw is None:
        pw = default_panel_width(vector_size)

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

    def run_once(fn, extra=()):
        A = A_initial.copy()
        x = x_initial.copy()
        w = w_initial.copy()

        start = time.perf_counter()
        fn(ALPHA, BETA, A, u1, v1, u2, v2, w, x, y, z, *extra)
        elapsed = time.perf_counter() - start

        return elapsed, A, x, w

    def measure(fn, extra=(), min_reps=9, min_time=0.3, max_reps=200):
        # Warm-up call: compiles on first use and starts the thread pool.
        run_once(fn, extra)

        # Repeat until enough time has been measured, then take the median,
        # which is less sensitive to outliers than the mean for short kernels.
        times = []
        while len(times) < min_reps or (sum(times) < min_time and len(times) < max_reps):
            elapsed, A, x, w = run_once(fn, extra)
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
        ("19_fused_panel_par", gemver_fused_panel_par_19, (pw,)),
    ]

    results = []
    for name, fn, *extra in kernels:
        rows = []
        for t in thread_counts:
            set_num_threads(t)
            gflops, elapsed, is_correct = measure(fn, tuple(extra[0]) if extra else ())
            rows.append((t, gflops, elapsed, is_correct))
        results.append((name, rows))

    set_num_threads(max_threads)

    return thread_counts, serial, results


def report_thread_sweep(vector_size, pw=None, image_path=None):
    print(f"\nRunning GEMVER Thread-Count Sweep for N={vector_size}...")
    print(f"CPU: {CPU_NAME}\n")

    thread_counts, serial, results = run_thread_sweep(vector_size, pw)
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
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5cd9", "#d6336c"]
    markers = ["o", "s", "^", "D", "v"]
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
    save_or_show(image_path)


def save_or_show(image_path):
    # - With a .png path: save the PNG, no window (works on a server with no display).
    # - Without: show the window, as in the live walkthrough.
    if image_path:
        os.makedirs(os.path.dirname(image_path) or ".", exist_ok=True)
        plt.savefig(image_path, dpi=300)
        print(f"Plot saved to {image_path}")
    else:
        plt.show()


# ---------------------------------------------------------
# Per-Stage Breakdown
# ---------------------------------------------------------
# Run with:  python gemver_numba_bench.py <N> breakdown   (add "md" for a Markdown table)
# - Professor: "what is the breakdown of each step, use it for justification".
# - Times each stage separately: shows where the time goes and what each
#   optimization removed. One bar per version:
#     - 3 (unfused, fastmath):        S1 | S2 | S3 | S4
#     - 11 (S1+S2 fused):             S1+S2 | S3 | S4
#     - 17b (+ 4-row unroll-and-jam): S1+S2 | S3 | S4
#     - 19_mk (all 4 fused, panels):  one block (its stages are interleaved)
# - How: fresh copies of A, x, w for every run (as in run_benchmark), then the
#   stages IN ORDER with a timestamp between them. Versions alternate every
#   round; median of >= 9 rounds.
# - Why not time each stage alone on a fresh copy: right after the copy, A is
#   read at DRAM speed even at N=512 (23 GB/s, vs 84 GB/s once a stage has used
#   it), so S2 alone took 3x longer than S2 right after S1.
# - Check: a version's stage times should add up to its whole-kernel time
#   (the 4 calls from Python add ~1 us in total).
# - Expect at N=4096 (memory-bound, so time ~ passes over A):
#     - S1 reads + writes A (2 passes) ~ 50%; S2 and S4 read A once ~ 25% each;
#       S3 touches only vectors ~ 0%.
#     - Fusing S1+S2 removes S2's pass; the panels also remove S4's (ideal case).
# - Observed (5900X):
#     - N=4096: S1 48%, S2 27%, S4 25%, S3 0% -> time follows passes (2:1:1).
#       Fused S1+S2 (9.7 ms in 17b) costs about what S1 alone did (10.6 ms):
#       S2's pass is gone. S4 stays ~5.4 ms (its own pass) until panels.
#     - N=512: S1 63% (first pass after the copy, A from DRAM); S2 and S4 then
#       read A from cache at 80-95 GB/s, so fusion saves less at this size.
# - Output: table, results/breakdown_N<N>.csv, stacked bar PNG
#   (results/breakdown_N<N>.png, or the .png path given on the command line).
# ---------------------------------------------------------

# Baseline 3's loops, one stage each: the same Python code as Baseline 2's
# building blocks (orders ij, ji, -, ij), compiled with fastmath like Baseline 3.
_bd_s1 = njit(fastmath=True)(_stage1_ij.py_func)
_bd_s2 = njit(fastmath=True)(_stage2_ji.py_func)
_bd_s3 = njit(fastmath=True)(_stage3.py_func)
_bd_s4 = njit(fastmath=True)(_stage4_ij.py_func)


@njit(fastmath=True)
def _bd_fused12_11(beta, A, u1, v1, u2, v2, x, y):
    # - Baseline 11's fused S1+S2 loop, copied so it can be timed alone
    n = A.shape[0]
    for i in range(n):
        a1 = u1[i]
        a2 = u2[i]
        c = beta * y[i]
        for j in range(n):
            a = A[i, j] + a1 * v1[j] + a2 * v2[j]
            A[i, j] = a
            x[j] += c * a


@njit(fastmath=True)
def _bd_s4_11(alpha, A, w, x):
    # - Baseline 11's S4 loop (row sum in a scalar, alpha once per row)
    n = A.shape[0]
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i, j] * x[j]
        w[i] = w[i] + alpha * s


# - One color per stage, the same in every bar (fixed categorical slots).
# - Checked: touching segments stay distinct for normal and color-blind vision.
BREAKDOWN_COLORS = {
    "S1": "#2a78d6",            # blue
    "S2": "#eb6834",            # orange
    "S3": "#1baf7a",            # aqua
    "S4": "#6250d6",            # violet
    "S1+S2": "#eda100",         # yellow (labels on it use dark text)
    "S1+S2+S3+S4": "#008300",   # green
}
BREAKDOWN_LEGEND = {
    "S1": r"S1: $A$ += $u_1 v_1^T + u_2 v_2^T$",
    "S2": r"S2: $x$ += $\beta A^T y$",
    "S3": r"S3: $x$ += $z$",
    "S4": r"S4: $w$ += $\alpha A x$",
    "S1+S2": "S1+S2 fused (one pass)",
    "S1+S2+S3+S4": "S1-S4 fused by column panels",
}


def run_breakdown(n, pw=None):
    if pw is None:
        pw = default_panel_width(n)
    A0, u1, v1, u2, v2, w0, x0, y, z = initialize_gemver(n)
    a, b = ALPHA, BETA
    A_exp, x_exp, w_exp = gemver_numpy_reference(a, b, A0, u1, v1, u2, v2, w0, x0, y, z)

    # Each piece gets fresh (A, x, w) and runs one stage or one whole kernel.
    # 17b's helpers called from Python get (0, n) as run-time values -> same SIMD path as 17b.
    pieces = {
        "3:S1": lambda A, x, w: _bd_s1(A, u1, v1, u2, v2),
        "3:S2": lambda A, x, w: _bd_s2(b, A, x, y),
        "S3": lambda A, x, w: _bd_s3(x, z),
        "3:S4": lambda A, x, w: _bd_s4(a, A, w, x),
        "11:S1+S2": lambda A, x, w: _bd_fused12_11(b, A, u1, v1, u2, v2, x, y),
        "11:S4": lambda A, x, w: _bd_s4_11(a, A, w, x),
        "17b:S1+S2": lambda A, x, w: _fused_rows_4(b, A, u1, v1, u2, v2, y, x, 0, n),
        "17b:S4": lambda A, x, w: _matvec_rows_4(a, A, x, w, 0, n),
        "3:whole": lambda A, x, w: gemver_opt_flags_3(a, b, A, u1, v1, u2, v2, w, x, y, z),
        "11:whole": lambda A, x, w: gemver_fused_11(a, b, A, u1, v1, u2, v2, w, x, y, z),
        "17b:whole": lambda A, x, w: gemver_fused_unroll_jam_17b(a, b, A, u1, v1, u2, v2, w, x, y, z, 0),
        "19:whole": lambda A, x, w: gemver_fused_panel_mk_19(a, b, A, u1, v1, u2, v2, w, x, y, z, pw),
    }
    # (version, [(stage, piece, passes over A)], whole-kernel piece)
    versions = [
        ("3: unfused", [("S1", "3:S1", 2), ("S2", "3:S2", 1), ("S3", "S3", 0), ("S4", "3:S4", 1)], "3:whole"),
        ("11: S1+S2 fused", [("S1+S2", "11:S1+S2", 2), ("S3", "S3", 0), ("S4", "11:S4", 1)], "11:whole"),
        ("17b: + unroll-and-jam", [("S1+S2", "17b:S1+S2", 2), ("S3", "S3", 0), ("S4", "17b:S4", 1)], "17b:whole"),
        (f"19_mk: all fused (PW={pw})", [("S1+S2+S3+S4", "19:whole", 2)], "19:whole"),
    ]

    def run_chain(keys):
        # fresh A, x, w; run the pieces in order with a timestamp between them
        A = A0.copy()
        x = x0.copy()
        w = w0.copy()
        stamps = [time.perf_counter()]
        for k in keys:
            pieces[k](A, x, w)
            stamps.append(time.perf_counter())
        return np.diff(stamps), A, x, w

    # jobs: each version's stage chain, plus its whole kernel when it is split
    jobs = []
    for version, stages, whole in versions:
        jobs.append((version, "stages", [piece for _, piece, _ in stages]))
        if len(stages) > 1:
            jobs.append((version, "whole", [whole]))

    correct = {}
    for version, kind, keys in jobs:       # warm-up (compiles) + correctness of the chain
        _, A, x, w = run_chain(keys)
        if kind == "stages":
            correct[version] = (np.allclose(A, A_exp, rtol=1e-5, atol=1e-5)
                                and np.allclose(x, x_exp, rtol=1e-5, atol=1e-5)
                                and np.allclose(w, w_exp, rtol=1e-5, atol=1e-5))

    # first round also sets the number of rounds (about 8 s in total, 9 to 201)
    start = time.perf_counter()
    times = {(v, kind): [run_chain(keys)[0]] for v, kind, keys in jobs}
    rounds = int(min(201, max(9, 8.0 / (time.perf_counter() - start))))
    for r in range(1, rounds):
        for v, kind, keys in (jobs if r % 2 == 0 else jobs[::-1]):
            times[(v, kind)].append(run_chain(keys)[0])

    results = []
    for version, stages, _ in versions:
        stage_med = np.median(np.array(times[(version, "stages")]), axis=0)   # per stage
        whole = times.get((version, "whole"))
        results.append({
            "version": version,
            "stages": [(stage, float(t), passes) for (stage, _, passes), t in zip(stages, stage_med)],
            "whole": float(np.median([t[0] for t in whole])) if whole else float(stage_med.sum()),
            "correct": correct[version],
        })
    return results, rounds, pw


def stage_bytes(passes, n):
    # A stages: passes x 8N^2. S3 (no pass over A): reads x, z and writes x = 3 x 8N.
    return passes * 8.0 * n * n if passes else 3 * 8.0 * n


def fmt_ms(ms):
    return f"{ms:.1f}" if ms >= 10 else f"{ms:.2f}" if ms >= 1 else f"{ms:.3f}"


def report_breakdown(n, pw=None, image_path=None, markdown=False):
    print(f"\nRunning GEMVER Per-Stage Breakdown for N={n}...")
    print(f"CPU: {CPU_NAME}")
    results, rounds, pw = run_breakdown(n, pw)
    print(f"Median of {rounds} rounds (fresh A, x, w every run; stages timed in order). "
          f"Baseline 19 panel width: {pw}\n")

    # rows shared by the table and the CSV
    rows = []
    for r in results:
        total = sum(t for _, t, _ in r["stages"])
        for stage, t, passes in r["stages"]:
            rows.append((r["version"], stage, t, 100 * t / total, passes,
                         stage_bytes(passes, n) / t / 1e9, total, r["whole"]))

    if markdown:
        print("| Version | Stage | Time (ms) | % of total | Passes over A | GB/s |")
        print("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for version, stage, t, pct, passes, gbps, _, _ in rows:
            print(f"| {version} | {stage} | {t * 1e3:.3f} | {pct:.0f}% | {passes} | {gbps:.1f} |")
        print()

    header = (f"| {'Version':<26} | {'Stage':<13} | {'Time (ms)':>10} | {'% of total':>10} "
              f"| {'Passes':>6} | {'GB/s':>7} |")
    divider = "-" * len(header)
    print(divider)
    print(header)
    print(divider)
    for r in results:
        total = sum(t for _, t, _ in r["stages"])
        for stage, t, passes in r["stages"]:
            gbps = stage_bytes(passes, n) / t / 1e9
            print(f"| {r['version']:<26} | {stage:<13} | {t * 1e3:10.3f} | {100 * t / total:9.0f}% "
                  f"| {passes:>6} | {gbps:7.1f} |")
        passes_total = sum(p for _, _, p in r["stages"])
        print(f"| {'':<26} | {'sum of stages':<13} | {total * 1e3:10.3f} | {100:9.0f}% "
              f"| {passes_total:>6} | {'':>7} |")
        print(f"| {'':<26} | {'whole kernel':<13} | {r['whole'] * 1e3:10.3f} | {'':>10} "
              f"| {'':>6} | {'':>7} |")
        print(divider)
    print("Stages run in order vs NumPy reference: " + ", ".join(
        f"{r['version'].split(':')[0]} {'PASS' if r['correct'] else 'FAIL'}" for r in results))
    print("GB/s = passes x size of A / time (S3: its three vectors).")

    csv_path = os.path.join(RESULTS_DIR, f"breakdown_N{n}.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    import csv
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(("n", "version", "stage", "time_s", "pct_of_version", "passes_over_A",
                         "gbps", "version_sum_s", "whole_kernel_s"))
        for version, stage, t, pct, passes, gbps, total, whole in rows:
            writer.writerow((n, version, stage, f"{t:.9f}", f"{pct:.1f}", passes, f"{gbps:.2f}",
                             f"{total:.9f}", f"{whole:.9f}"))
    print(f"CSV written to {csv_path}")

    plot_breakdown(results, n, rounds, image_path or os.path.join(RESULTS_DIR, f"breakdown_N{n}.png"),
                   show=image_path is None)


def plot_breakdown(results, n, rounds, path, show):
    # Horizontal stacked bars: one bar per version, one segment per stage.
    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    grid, axis_color = "#e1e0d9", "#c3c2b7"

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=surface)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.72, bottom=0.13)
    ax.set_facecolor(surface)

    totals = [sum(t for _, t, _ in r["stages"]) * 1e3 for r in results]
    longest = max(totals)
    ax.set_xlim(0, longest * 1.22)
    ax.set_ylim(len(results) - 0.5, -0.5)          # first version on top

    labels = []
    for row, r in enumerate(results):
        left = 0.0
        for stage, t, _ in r["stages"]:
            width = t * 1e3
            ax.barh(row, width, left=left, height=0.5, color=BREAKDOWN_COLORS[stage],
                    edgecolor=surface, linewidth=1.5)      # surface-colored gap between segments
            labels.append((row, left, width, f"{stage}  {fmt_ms(width)} ms",
                           ink if stage == "S1+S2" else "#ffffff"))
            left += width
        # bar tip: sum of the stages; the whole kernel timed in one call, when it is split
        tip = f"{fmt_ms(left)} ms"
        if len(r["stages"]) > 1:
            tip += f"\nwhole kernel {fmt_ms(r['whole'] * 1e3)} ms"
        ax.text(left + 0.01 * longest, row, tip, va="center", ha="left", fontsize=9.5,
                color=ink, linespacing=1.4)

    # label inside a segment only where it fits with padding (S3 is usually too thin)
    renderer = fig.canvas.get_renderer()
    for row, left, width, text, color in labels:
        label = ax.text(left + width / 2, row, text, ha="center", va="center", fontsize=9, color=color)
        x0 = ax.transData.transform((left, row))[0]
        x1 = ax.transData.transform((left + width, row))[0]
        if label.get_window_extent(renderer).width + 12 > x1 - x0:
            label.remove()

    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([f"{r['version']}\n{sum(p for _, _, p in r['stages'])} passes over A"
                        for r in results], color=ink2, fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=muted)
    ax.xaxis.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(axis_color)
    ax.set_xlabel("Time per GEMVER call (ms), lower is better", color=ink2)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=BREAKDOWN_LEGEND[s]) for s, c in BREAKDOWN_COLORS.items()]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3,
              frameon=False, fontsize=9, labelcolor=ink2)
    fig.text(0.2, 0.95, f"Where GEMVER's time goes, stage by stage (N={n})",
             fontsize=13, color=ink, ha="left")
    fig.text(0.2, 0.905, f"Stages timed in order inside each run, median of {rounds} rounds | CPU: {CPU_NAME}",
             fontsize=9, color=muted, ha="left")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=300, facecolor=surface)
    print(f"Plot saved to {path}")
    if show:
        plt.show()


# ---------------------------------------------------------
# Scalability over N
# ---------------------------------------------------------
# Run with:  python gemver_numba_bench.py scale        (add "md" for Markdown tables)
#            python gemver_numba_bench.py 2048 scale   (only sizes up to 2048: quicker)
# - Professor (to classmates): "test scalability across data sizes",
#   "how scalable is the optimization".
# - Kernels: 1, 2 (best order), 3, 10, 11, 12, 17, 17b, 18, 19_mk, 19_par.
#   Baseline 0 (pure Python) only up to N=512 (about 22 s per call at N=4096).
# - Timing is "warm": the same arrays are passed again and again, so A stays in
#   the cache whenever it fits (the matmul benchmark times its kernels this way).
#     - The first call runs on fresh copies and is checked against NumPy.
#     - The timed calls repeat the same work on the updated arrays; median of >= 9.
# - Why not a fresh copy before each call (as in the main table):
#     - For A between 2 and 32 MB, how much of A the copy leaves in the cache
#       depends on Windows' memory state, so the time jumps between runs
#       (17b at N=512: 15 or 23-28 GFLOP/s; warm: 44 on every run).
#     - Once A is larger than the L3, both ways give the same time.
# - 19 uses the default panel width at each N (L3/2 rule). Up to N=1024 that is
#   one panel (PW = N), so S4 is not fused there (3 passes, like 17).
# - Where A lives (A = 8 N^2 bytes; one core sees 512 KB of L2 and 32 MB of L3):
#     - L2: N <= 256      - L3: N <= 2048      - DRAM: N > 2048
#     - Threads on both CCDs see 64 MB of L3 (N <= 2896).
# - DRAM roofs: if every pass over A comes from DRAM at the measured bandwidth,
#   GFLOP/s <= 10 N^2 / (passes x 8 N^2 / BW) = 1.25 x BW / passes (same for every N).
# - Expect: fastest while A is in cache, a drop once A leaves the L3 (N > 2048),
#   then flat lines under the DRAM roofs. Fused versions (3 passes) keep their
#   lead over the unfused ones (4 passes) after the drop.
# - Observed (5900X, roofs at 37.7 GB/s):
#     - Serial, A in cache (N <= 1536): 17b and 19_mk 37-46 GFLOP/s, 3 about 30.
#     - N=2048 (A = 33.6 MB, just over the L3): serial speed halves (17b 45 -> 22).
#     - DRAM: 17b 12-14.5 GFLOP/s (86% of the 3-pass roof at N=4096);
#       3 about 9 (77% of the 4-pass roof).
#     - Parallel 12 / 19_par: 74-159 GFLOP/s while A fits in both CCDs' L3
#       (N <= 2048), 11-14 in DRAM. Threads pay off in cache, hardly in DRAM.
#     - Fusion keeps its lead after the drop: 17b 1.35-1.6x over 3 for N >= 3072
#       (11: 1.15-1.3x). BLAS (18) stays at about 0.7x of 3.
#     - 19_mk falls behind as N grows: PW shrinks to 256 at N >= 6144 (short rows).
#     - 17 vs 17b: 2.6x slower in cache (N=1024), only 1.1x in DRAM.
# - Output: tables (GFLOP/s, speedup vs 3), results/scale.csv, results/scale.png.
# ---------------------------------------------------------
SCALE_SIZES = (256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
SCALE_PYTHON_MAX_N = 512
SCALE_ROWS = (
    ("0_python_naive", gemver_python_0),
    ("1_numba_naive", gemver_numba_1),
    ("2_order_ij_ji_ij", gemver_2_ij_ji_ij),
    ("3_fastmath_ij_ji_ij", gemver_opt_flags_3),
    ("10_numpy", gemver_numpy_10),
    ("11_fused_fastmath", gemver_fused_11),
    ("12_fused_parallel", gemver_fused_parallel_12),
    ("17_fused_unroll_jam", gemver_fused_unroll_jam_17),
    ("17b_unroll_jam_runtime_start", gemver_fused_unroll_jam_17b),
    ("18_blas", gemver_blas_18),
    ("19_fused_panel_mk", gemver_fused_panel_mk_19),
    ("19_fused_panel_par", gemver_fused_panel_par_19),
)
L2_BYTES = 512 * 1024           # one core's L2

# Chart: one fixed color + marker per kernel (8 slots, in order, never cycled).
# - 1 and 2 are gray context lines; 0 (two points) and 17 (= 17b before the fix)
#   are only in the tables and the CSV.
SCALE_STYLE = {
    "3_fastmath_ij_ji_ij": ("#2a78d6", "o", "3: unfused + fastmath"),
    "10_numpy": ("#eb6834", "s", "10: NumPy"),
    "11_fused_fastmath": ("#1baf7a", "^", "11: S1+S2 fused"),
    "12_fused_parallel": ("#eda100", "D", "12: fused + parallel"),
    "17b_unroll_jam_runtime_start": ("#e87ba4", "v", "17b: fused + unroll-and-jam"),
    "18_blas": ("#008300", "P", "18: BLAS (dgemm + 2 dgemv)"),
    "19_fused_panel_mk": ("#6250d6", "X", "19_mk: column panels + micro-kernel"),
    "19_fused_panel_par": ("#e34948", "h", "19_par: column panels, parallel"),
}
SCALE_CONTEXT = {
    "1_numba_naive": (":", "1: naive Numba"),
    "2_order_ij_ji_ij": ("--", "2: best loop order"),
}


def cache_level(n):
    # Smallest cache level that holds A (one core's view)
    a_bytes = 8.0 * n * n
    return "L2" if a_bytes <= L2_BYTES else "L3" if a_bytes <= L3_BYTES else "DRAM"


def scale_roof(passes, bw):
    # GFLOP/s when every pass over A runs at bandwidth bw (GB/s): 10 N^2 FLOPs per passes x 8 N^2 bytes
    return 10.0 / (8.0 * passes) * bw


def free_ram_bytes():
    # Free physical memory in bytes (no extra package needed); None if unknown.
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                    (f, ctypes.c_ulonglong) for f in ("total", "avail", "total_page", "avail_page",
                                                      "total_virtual", "avail_virtual", "avail_ext")]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.avail
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, AttributeError, ValueError):
        pass
    return None


def run_scale(sizes):
    # Per N: NumPy check on fresh copies, then warm timing. Returns (rows, sizes that ran).
    rows, ran = [], []
    for n in sizes:
        # - RAM check: A, the reference A, the working copy and NumPy's 3 temporaries
        #   (Baseline 10) are each 8 N^2 bytes; 7 of them leaves some room.
        need, free = 7 * 8.0 * n * n, free_ram_bytes()
        if free is not None and need > free:
            print(f"N={n:5d}: skipped, needs about {need / 1e9:.1f} GB of RAM ({free / 1e9:.1f} GB free)")
            continue
        A0, u1, v1, u2, v2, w0, x0, y, z = initialize_gemver(n)
        A_exp, x_exp, w_exp = gemver_numpy_reference(ALPHA, BETA, A0, u1, v1, u2, v2, w0, x0, y, z)
        flops = gemver_flop_count(n)
        pw = default_panel_width(n)
        start = time.perf_counter()
        count = 0
        for name, fn in SCALE_ROWS:
            python = name == "0_python_naive"
            if python and n > SCALE_PYTHON_MAX_N:
                continue
            # arguments after z: 17b's start row, Baseline 19's panel width
            extra = (0,) if name.startswith("17b_") else (pw,) if name.startswith("19_") else ()
            A, x, w = A0.copy(), x0.copy(), w0.copy()
            fn(ALPHA, BETA, A, u1, v1, u2, v2, w, x, y, z, *extra)     # compiles; result checked
            correct = (np.allclose(A, A_exp, rtol=1e-5, atol=1e-5)
                       and np.allclose(x, x_exp, rtol=1e-5, atol=1e-5)
                       and np.allclose(w, w_exp, rtol=1e-5, atol=1e-5))
            # warm timing: same arrays again (same work, A in cache when it fits)
            times = []
            while len(times) < (1 if python else 9) or (not python and sum(times) < 0.3 and len(times) < 200):
                t0 = time.perf_counter()
                fn(ALPHA, BETA, A, u1, v1, u2, v2, w, x, y, z, *extra)
                times.append(time.perf_counter() - t0)
            elapsed = float(np.median(times))
            rows.append({"n": n, "name": name, "gflops": flops / elapsed / 1e9, "time_s": elapsed,
                         "passes": gemver_passes_over_A(name), "pw": pw if name.startswith("19_") else "",
                         "correct": correct})
            count += 1
        del A, x, w
        note = "" if n <= SCALE_PYTHON_MAX_N else "  (Baseline 0 skipped: pure Python, too slow)"
        print(f"N={n:5d}  A = {8.0 * n * n / 1e6:6.1f} MB ({cache_level(n):>4})  PW = {pw:4d}  "
              f"{count:2d} kernels in {time.perf_counter() - start:5.1f} s{note}", flush=True)
        ran.append(n)
    return rows, ran


def report_scale(sizes, image_path=None, markdown=False, stem=None):
    # stem: output path without extension (default results/scale)
    stem = stem or os.path.join(RESULTS_DIR, "scale")
    print(f"\nRunning GEMVER Scalability Sweep, N = {', '.join(str(n) for n in sizes)}...")
    print(f"CPU: {CPU_NAME}")
    print(f"Numba threads: {get_num_threads()} (used by 12 and 19_par)")
    print("Warm timing: the same arrays are called repeatedly (A stays in cache when it fits); "
          "the first call is checked against NumPy.")
    print(f"Baseline 0 (pure Python) only up to N={SCALE_PYTHON_MAX_N}; "
          f"Baseline 19 uses the default panel width at each N.\n")

    start = time.perf_counter()
    rows, ran = run_scale(sizes)
    if not ran:
        print("Nothing was run.")
        return
    print(f"\nSweep took {time.perf_counter() - start:.0f} s. Numba threading layer: {threading_layer()}")

    bw_serial, bw_by_threads = measure_bandwidth()
    bw_peak = max(bw_serial, max(bw_by_threads.values()))
    print(f"Measured memory bandwidth (peak of the a = a + s*b runs): {bw_peak:.1f} GB/s")
    print("DRAM roofs (every pass over A from DRAM): " + ", ".join(
        f"{p} passes {scale_roof(p, bw_peak):.1f}" for p in (4, 3, 2)) + " GFLOP/s\n")

    # - Per row: effective GB/s, % of peak, speedup over Baseline 3 at the same N
    base = {r["n"]: r["time_s"] for r in rows if r["name"] == "3_fastmath_ij_ji_ij"}
    for r in rows:
        r["gbps"] = r["passes"] * 8.0 * r["n"] ** 2 / r["time_s"] / 1e9
        r["pct_peak"] = 100 * r["gbps"] / bw_peak
        r["vs3"] = base[r["n"]] / r["time_s"] if r["n"] in base else float("nan")
    by_name = {}
    for r in rows:
        by_name.setdefault(r["name"], {})[r["n"]] = r
    names = [name for name, _ in SCALE_ROWS if name in by_name]

    def table(cell, markdown_style):
        # rows = kernels, columns = N; "-" where a kernel was not run
        head = ["Baseline"] + [str(n) for n in ran] + ["Correct"]
        info = [["A size (MB)"] + [f"{8.0 * n * n / 1e6:.1f}" for n in ran] + [""],
                ["A fits in"] + [cache_level(n) for n in ran] + [""]]
        body = []
        for name in names:
            cells = [cell(by_name[name][n]) if n in by_name[name] else "-" for n in ran]
            bad = [str(n) for n, r in by_name[name].items() if not r["correct"]]
            body.append([name] + cells + ["PASS" if not bad else "FAIL at " + ",".join(bad)])
        if markdown_style:
            lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(":---" for _ in head) + " |"]
            lines += ["| " + " | ".join(row) + " |" for row in info + body]
            return "\n".join(lines)
        widths = [max([28] + [len(row[0]) for row in body])] + [7] * len(ran) + [8]
        fmt = lambda row: "| " + " | ".join(
            (c.ljust(w) if k in (0, len(row) - 1) else c.rjust(w)) for k, (c, w) in enumerate(zip(row, widths))) + " |"
        divider = "-" * len(fmt(head))
        return "\n".join([divider, fmt(head)] + [fmt(row) for row in info] + [divider]
                         + [fmt(row) for row in body] + [divider])

    gflops_table = table(lambda r: f"{r['gflops']:.2f}" if r["gflops"] >= 0.1 else f"{r['gflops']:.3f}",
                         markdown)
    speedup_table = table(lambda r: f"{r['vs3']:.2f}x", markdown)
    print("GFLOP/s (higher is better)")
    print(gflops_table)
    print("\nSpeedup over Baseline 3 at the same N (does the lead hold as N grows?)")
    print(speedup_table)

    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    if markdown:
        with open(stem + ".md", "w", encoding="utf-8") as fh:
            fh.write("GFLOP/s (higher is better)\n\n" + gflops_table + "\n\n"
                     "Speedup over Baseline 3 at the same N\n\n" + speedup_table + "\n")
        print(f"\nMarkdown tables written to {stem}.md")

    import csv
    with open(stem + ".csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(("n", "a_mb", "a_fits_in", "baseline", "gflops", "time_s", "passes_over_A",
                         "gbps", "pct_of_peak", "speedup_vs_3", "panel_width", "correct"))
        for r in rows:
            writer.writerow((r["n"], f"{8.0 * r['n'] ** 2 / 1e6:.2f}", cache_level(r["n"]), r["name"],
                             f"{r['gflops']:.4f}", f"{r['time_s']:.9f}", r["passes"], f"{r['gbps']:.2f}",
                             f"{r['pct_peak']:.1f}", f"{r['vs3']:.3f}", r["pw"],
                             "PASS" if r["correct"] else "FAIL"))
    print(f"CSV written to {stem}.csv")

    plot_scale(by_name, ran, bw_peak, image_path or stem + ".png", show=image_path is None)


def plot_scale(by_name, sizes, bw_peak, path, show):
    # Line chart: GFLOP/s vs N (log x), cache levels shaded, DRAM roofs dashed.
    from matplotlib.ticker import NullLocator
    from matplotlib.transforms import blended_transform_factory
    surface, ink, ink2, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    grid, axis_color = "#e1e0d9", "#c3c2b7"

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=surface)
    fig.subplots_adjust(left=0.075, right=0.845, top=0.74, bottom=0.14)
    ax.set_facecolor(surface)
    ax.set_xscale("log", base=2)
    lo, hi = min(sizes) / 1.6, max(sizes) * 1.25
    ax.set_xlim(lo, hi)

    # - Shaded bands: where A lives. Boundary = N at which A equals the cache size.
    n_l2, n_l3 = (L2_BYTES / 8) ** 0.5, (L3_BYTES / 8) ** 0.5
    bands = [(lo, n_l2, "#eeede8", "A in L2", "(0.5 MB)"),
             (n_l2, n_l3, "#f5f4f0", "A in L3", "(32 MB)"),
             (n_l3, hi, surface, "A in DRAM", "(beyond 32 MB)")]
    renderer = fig.canvas.get_renderer()
    for x0, x1, color, label, sub in bands:
        x0, x1 = max(x0, lo), min(x1, hi)
        if x1 <= x0:
            continue
        ax.axvspan(x0, x1, color=color, zorder=0, linewidth=0)
        text = ax.text((x0 * x1) ** 0.5, 0.985, f"{label}\n{sub}", transform=blended_transform_factory(
            ax.transData, ax.transAxes), ha="center", va="top", fontsize=9, color=ink2, linespacing=1.3)
        left, right = ax.transData.transform((x0, 0))[0], ax.transData.transform((x1, 0))[0]
        if text.get_window_extent(renderer).width + 8 > right - left:
            text.remove()                          # band too narrow for its label
    for boundary in (n_l2, n_l3):
        if lo < boundary < hi:
            ax.axvline(boundary, color=axis_color, linewidth=0.8, zorder=1)
    # - Threads on both CCDs share 2 x 32 MB of L3: the parallel kernels drop later
    n_l3_both = (2 * L3_BYTES / 8) ** 0.5
    if lo < n_l3_both < hi:
        ax.axvline(n_l3_both, color=axis_color, linewidth=0.8, linestyle=(0, (1, 2)), zorder=1)
        ax.text(n_l3_both * 1.03, 0.84, "64 MB: L3 of both CCDs\n(limit for the parallel kernels)",
                transform=blended_transform_factory(ax.transData, ax.transAxes),
                ha="left", va="top", fontsize=8, color=muted, linespacing=1.3)

    # - Context lines first (gray), then the colored kernels on top
    for name, (style, label) in SCALE_CONTEXT.items():
        if name in by_name:
            ns = sorted(by_name[name])
            ax.plot(ns, [by_name[name][n]["gflops"] for n in ns], color=muted, linestyle=style,
                    linewidth=1.4, marker="o", markersize=3.5, label=label, zorder=2)
    plotted = [by_name[name][n]["gflops"] for name in list(SCALE_STYLE) + list(SCALE_CONTEXT)
               if name in by_name for n in by_name[name]]
    for name, (color, marker, label) in SCALE_STYLE.items():
        if name in by_name:
            ns = sorted(by_name[name])
            ax.plot(ns, [by_name[name][n]["gflops"] for n in ns], color=color, linewidth=1.8,
                    marker=marker, markersize=8, markeredgecolor=surface, markeredgewidth=1.2,
                    label=label, zorder=3)

    # - DRAM roofs: only meaningful where A comes from DRAM; labels in the right margin
    roof_x0 = max(n_l3, lo)
    to_margin = blended_transform_factory(ax.transAxes, ax.transData)
    for passes in (4, 3, 2):
        g = scale_roof(passes, bw_peak)
        plotted.append(g)
        if roof_x0 < hi:
            ax.hlines(g, roof_x0, hi, colors=ink2, linestyles=(0, (5, 3)), linewidth=1.1, zorder=2)
            ax.text(1.012, g, f"{passes}-pass roof  {g:.1f}", transform=to_margin,
                    ha="left", va="center", fontsize=8.5, color=ink2)

    # - Log y: from about 1 (naive) to about 150 (parallel, in cache); equal gaps = equal ratios
    ax.set_yscale("log")
    ax.set_ylim(min(plotted) / 1.5, max(plotted) * 2.2)    # head room for the band labels
    ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500) if min(plotted) / 1.5 <= t <= max(plotted) * 2.2]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n}\n{8.0 * n * n / 1e6:.1f} MB" if 8.0 * n * n < 10e6 else
                        f"{n}\n{8.0 * n * n / 1e6:.0f} MB" for n in sizes], fontsize=9, color=ink2)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", colors=muted, length=0)
    ax.yaxis.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(axis_color)
    ax.set_xlabel("Matrix size N (second line: size of A)", color=ink2)
    ax.set_ylabel("GFLOP/s (log scale, higher is better)", color=ink2)

    handles, labels = ax.get_legend_handles_labels()
    n_context = sum(1 for name in SCALE_CONTEXT if name in by_name)      # gray lines go last
    order = list(range(n_context, len(labels))) + list(range(n_context))
    ax.legend([handles[k] for k in order], [labels[k] for k in order], loc="lower left",
              bbox_to_anchor=(0, 1.02), ncol=4, frameon=False, fontsize=9, labelcolor=ink2,
              handlelength=2.6, columnspacing=1.6)
    fig.text(0.075, 0.955, f"GEMVER throughput as the matrix grows (N={sizes[0]} to {sizes[-1]})",
             fontsize=13, color=ink, ha="left")
    fig.text(0.075, 0.918, f"Warm timing (A stays in cache when it fits), median of >= 9 calls. "
             f"Dashed: DRAM roofs at {bw_peak:.1f} GB/s | CPU: {CPU_NAME}", fontsize=9, color=muted, ha="left")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=300, facecolor=surface)
    print(f"Plot saved to {path}")
    if show:
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
# - Third element: helpers where the work happens (their assembly is included).
ASM_KERNELS = [
    ("1_numba_naive", gemver_numba_1),
    ("3_fastmath_ij_ji_ij", gemver_opt_flags_3),
    ("11_fused_fastmath", gemver_fused_11),
    ("16_no_hoist", gemver_no_hoist_16),
    ("17_fused_unroll_jam", gemver_fused_unroll_jam_17, (_fused_rows_4, _matvec_rows_4)),
    ("19_fused_panel", gemver_fused_panel_19, (_panel_s12_rows1, _panel_s4_rows1)),
    ("19_fused_panel_mk", gemver_fused_panel_mk_19, (_panel_s12_rows4, _panel_s4_rows4)),
]

def report_asm(vector_size):
    import re
    A, u1, v1, u2, v2, w, x, y, z = initialize_gemver(vector_size)
    os.makedirs("asm", exist_ok=True)
    print(f"\n{'Kernel':<24} | {'vector FMA (ymm)':>16} | {'scalar add/FMA':>14} | {'instructions':>12}")
    print("-" * 76)
    for name, fn, *helpers in ASM_KERNELS:
        fn(ALPHA, BETA, A.copy(), u1, v1, u2, v2, w.copy(), x.copy(), y, z)  # compile
        asm = "\n".join(fn.inspect_asm().values())
        for helper in (helpers[0] if helpers else ()):
            asm += "\n".join(helper.inspect_asm().values())
        with open(os.path.join("asm", name + ".s"), "w") as fh:
            fh.write(asm)
        code = [l for l in asm.splitlines() if l.startswith("\t") and not l.startswith("\t.")]
        vec_fma = sum(1 for l in code if re.search(r"vfn?madd\w*pd\s.*ymm", l))
        scalar = sum(1 for l in code if re.search(r"\b(vaddsd|vfn?madd\w*sd)\b", l))
        print(f"{name:<24} | {vec_fma:>16} | {scalar:>14} | {len(code):>12}")
    print("\nFull assembly written to ./asm/")


# ---------------------------------------------------------
# Result Rows, Markdown Table and CSV
# ---------------------------------------------------------
# - One row per baseline with every column. The terminal table, the Markdown
#   table ("md" mode) and the CSV are all printed from the same rows.
# - Markdown: same layout as doc/benchmarks/matmul.md, so the doc table can be
#   copied straight from the benchmark output.
# - CSV: always written to results/gemver_results_N<N>.csv (slides, charts).
RESULT_COLUMNS = ("Baseline Implementation", "GFLOP/s", "Time (ms)", "Abs Speedup",
                  "Rel Speedup", "Passes over A", "GB/s", "% of peak", "Correct")
CSV_COLUMNS = ("n", "baseline", "gflops", "time_s", "abs_speedup", "rel_speedup",
               "passes_over_A", "gbps", "pct_of_peak", "correct")

def result_rows(benchmark_data, n, bw_peak):
    a_bytes = 8.0 * n * n
    ref_elapsed = benchmark_data[0][2]      # Abs Speedup: vs the first row (pure Python)
    rows = []
    prev_elapsed = None
    for name, gflops, elapsed, is_correct in benchmark_data:
        passes = gemver_passes_over_A(name)
        gbps = passes * a_bytes / elapsed / 1e9
        rows.append({
            "name": name,
            "gflops": gflops,
            "time_s": elapsed,
            "abs_speedup": ref_elapsed / elapsed,
            "rel_speedup": prev_elapsed / elapsed if prev_elapsed is not None else 1.0,
            "passes": passes,
            "gbps": gbps,
            "pct_peak": 100 * gbps / bw_peak,
            "correct": "PASS" if is_correct else "FAIL",
        })
        prev_elapsed = elapsed
    return rows


def print_box_table(rows):
    header = (
        f"| {'Baseline Implementation':<28} "
        f"| {'GFLOP/s':>10} "
        f"| {'Time (s)':>12} "
        f"| {'Abs Speedup':>12} "
        f"| {'Rel Speedup':>12} "
        f"| {'Passes':>6} "
        f"| {'GB/s':>7} "
        f"| {'% of peak':>11} "
        f"| {'Correct':<8} |"
    )
    divider = "-" * len(header)
    print(divider)
    print(header)
    print(divider)
    for r in rows:
        print(
            f"| {r['name']:<28} "
            f"| {r['gflops']:10.4f} "
            f"| {r['time_s']:12.6f} "
            f"| {r['abs_speedup']:11.2f}x "
            f"| {r['rel_speedup']:11.2f}x "
            f"| {r['passes']:>6} "
            f"| {r['gbps']:7.1f} "
            f"| {r['pct_peak']:10.0f}% "
            f"| {r['correct']:<8} |"
        )
    print(divider)


def markdown_table(rows):
    lines = [
        "| " + " | ".join(RESULT_COLUMNS) + " |",
        "| " + " | ".join(":---" for _ in RESULT_COLUMNS) + " |",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']:<28} "
            f"| {r['gflops']:10.3f} "
            f"| {r['time_s'] * 1e3:10.3f} "
            f"| {r['abs_speedup']:10.2f}x "
            f"| {r['rel_speedup']:10.2f}x "
            f"| {r['passes']:>13} "
            f"| {r['gbps']:7.1f} "
            f"| {r['pct_peak']:8.0f}% "
            f"| {r['correct']:<7} |"
        )
    return "\n".join(lines)


def write_results_csv(rows, n, path):
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for r in rows:
            writer.writerow([
                n, r["name"], f"{r['gflops']:.4f}", f"{r['time_s']:.9f}",
                f"{r['abs_speedup']:.3f}", f"{r['rel_speedup']:.3f}", r["passes"],
                f"{r['gbps']:.2f}", f"{r['pct_peak']:.1f}", r["correct"],
            ])
    print(f"CSV written to {path}")


# ---------------------------------------------------------
# Formatting and Output Execution
# ---------------------------------------------------------
# Command line: python gemver_numba_bench.py [N] [mode words] [image.png] [--pw W]
# - N: matrix size (default 512).
# - Mode words (any order): threads, asm, blas_mt, md, breakdown, scale.
#     - md: print the results as a Markdown table (matmul.md format)
#       and save it as results/gemver_results_N<N>.md.
#     - breakdown: time per stage instead of the baseline table.
#     - scale: sweep N = 256 ... 8192 (an N given with it = largest size).
# - image.png: save the plot there instead of opening a window
#   (upstream run_bench.sh: "512 doc/gemver_numba_bench_512.png").
# - --pw W: panel width for Baseline 19 (default: default_panel_width(N)).
MODES = ("threads", "asm", "blas_mt", "md", "breakdown", "scale")
RESULTS_DIR = "results"

def parse_args(argv):
    n = DEFAULT_N
    n_given = False         # scale mode: an N on the command line caps the sizes
    modes = set()
    image_path = None
    pw = None
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg == "--pw" or arg.startswith("--pw="):
            value = arg.split("=", 1)[1] if "=" in arg else (args.pop(0) if args else "")
            if value.isdigit() and int(value) > 0:
                pw = int(value)
            else:
                print(f"Ignoring --pw '{value}': expected a positive integer.")
        elif arg.lower().endswith(".png"):
            image_path = arg
        elif arg in MODES:
            modes.add(arg)
        else:
            try:
                value = int(arg)
            except ValueError:
                print(f"Ignoring unknown argument '{arg}'. "
                      f"Expected N, a .png path, or one of: {', '.join(MODES)}.")
                continue
            if value > 0:
                n = value
                n_given = True
            else:
                print(f"Invalid vector size. Using the default N={DEFAULT_N}.")
    return n, modes, image_path, pw, n_given


if __name__ == "__main__":
    N_input, modes, image_path, pw_input, n_given = parse_args(sys.argv[1:])
    pw = pw_input if pw_input is not None else default_panel_width(N_input)

    # scale: every size in SCALE_SIZES (or up to the N given); CSV/PNG in results/
    if "scale" in modes:
        sizes = [n for n in SCALE_SIZES if n <= N_input] if n_given else list(SCALE_SIZES)
        if pw_input is not None:
            print("scale: --pw ignored (Baseline 19 uses the default panel width at each N).")
        report_scale(sizes or [N_input], image_path, markdown="md" in modes,
                     stem=os.path.join(RESULTS_DIR, f"scale_to{N_input}" if n_given else "scale"))
        sys.exit(0)

    # A mode word selects a sweep instead of the baseline table.
    if "threads" in modes:
        report_thread_sweep(N_input, pw, image_path)
        sys.exit(0)
    if "asm" in modes:
        report_asm(N_input)
        sys.exit(0)
    if "breakdown" in modes:
        report_breakdown(N_input, pw, image_path, markdown="md" in modes)
        sys.exit(0)

    print(f"\nRunning GEMVER Benchmarks for N={N_input}"
          f"{' (blas_mt: multi-threaded BLAS, short table)' if BLAS_MT else ''}...")
    print(f"CPU: {CPU_NAME}")
    print(f"Numba threads: {get_num_threads()}")
    print(blas_info())
    print(f"Baseline 19 panel width: {pw} columns "
          f"({'--pw' if pw_input is not None else 'default'}; "
          f"panel = N x PW x 8 B = {N_input * pw * 8 / 1e6:.1f} MB, L3 = {L3_BYTES / 1e6:.1f} MB)\n")

    benchmark_data = run_benchmark(N_input, blas_mt=BLAS_MT, pw=pw)

    # The threading layer is only chosen once a parallel function has run.
    print(f"Numba threading layer: {threading_layer()}\n")

    bw_serial, bw_by_threads = measure_bandwidth()
    best_t = max(bw_by_threads, key=bw_by_threads.get)
    bw_peak = max(bw_serial, bw_by_threads[best_t])
    a_bytes = 8.0 * N_input * N_input
    print("Measured memory bandwidth (a = a + s*b, 128 MiB arrays):")
    print(f"  serial kernel: {bw_serial:.1f} GB/s")
    print("  parallel kernel: " + ", ".join(
        f"{t} thr {g:.1f}" for t, g in bw_by_threads.items()) + " GB/s")
    print(f"  peak used for '% of peak': {bw_peak:.1f} GB/s")
    print(f"A = {a_bytes / 1e6:.1f} MB "
          f"({'fits in' if a_bytes <= 32e6 else 'larger than'} the 32 MB L3)."
          f" GB/s = passes over A x size of A / time.")
    if a_bytes <= 32e6:
        print("A is served from cache at this size, so '% of peak' can exceed 100%.")
    print()

    if BLAS_MT:
        print(f"(blas_mt: no Python row, so Abs Speedup is relative to {benchmark_data[0][0]}.)\n")

    rows = result_rows(benchmark_data, N_input, bw_peak)

    # File names: results/gemver_results_N<N>[_blas_mt][_pw<W>].csv / .md
    stem = os.path.join(RESULTS_DIR, f"gemver_results_N{N_input}"
                        f"{'_blas_mt' if BLAS_MT else ''}"
                        f"{f'_pw{pw}' if pw_input is not None else ''}")

    if "md" in modes:
        table = markdown_table(rows)
        print(table)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(stem + ".md", "w", encoding="utf-8") as fh:
            fh.write(table + "\n")
        print(f"\nMarkdown table written to {stem}.md")
    else:
        print_box_table(rows)

    write_results_csv(rows, N_input, stem + ".csv")

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

    # Optional image path (same as upstream matmul / run_bench.sh)
    save_or_show(image_path)
