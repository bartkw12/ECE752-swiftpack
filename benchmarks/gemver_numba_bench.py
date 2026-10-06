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
L1_cache = "768 KB"
L2_cache = "6.0 MB"
L3_cache = "64.0 MB"

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
# ---------------------------------------------------------
# parallel_rows: prange over the outer i loop of every stage. Stage 2 is kept
# in the original (column-reading) order because then each thread owns its
# own x[i]. The tempting alternative (prange over j in the stride-1 ji order)
# is a DATA RACE: every thread would update all of x at the same time.
#
# parallel_inner: stage 2 in the stride-1 ji order with prange over the INNER
# i loop, inside a serial j loop (the same as matmul's parallel_k).
#
# Expectation / Why: parallel_rows should beat the serial versions (12 cores),
# but its stage 2 is still strided. parallel_inner starts a new parallel region
# for every row j (N of them), each doing only N multiply-adds, so thread
# synchronization overhead dominates and it is likely slower than serial.
#
# Observed (5900X): parallel_inner is ~10x slower than serial at N=512, as
# expected. parallel_rows beats the naive serial code but NOT the serial
# fastmath version (Baseline 3). At N=512 the whole kernel takes ~0.1 ms, so
# the four parallel regions' fork/join cost is significant. At N=4096, A (128 MB)
# lives in DRAM, and one core with stride-1 access already uses most of the
# memory bandwidth. More threads cannot fetch A faster, and the strided stage 2
# wastes the bandwidth that is available.
@njit(parallel=True, fastmath=True)
def gemver_parallel_rows_4(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]

    for i in prange(n):
        for j in range(n):
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

    for i in prange(n):
        for j in range(n):
            x[i] = x[i] + beta * A[j, i] * y[j]

    for i in prange(n):
        x[i] = x[i] + z[i]

    for i in prange(n):
        for j in range(n):
            w[i] = w[i] + alpha * A[i, j] * x[j]

@njit(parallel=True, fastmath=True)
def gemver_parallel_inner_4(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]

    for i in prange(n):
        for j in range(n):
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

    for j in range(n):
        for i in prange(n):
            x[i] = x[i] + beta * A[j, i] * y[j]

    for i in prange(n):
        x[i] = x[i] + z[i]

    for i in prange(n):
        for j in range(n):
            w[i] = w[i] + alpha * A[i, j] * x[j]


# ---------------------------------------------------------
# Baseline 5: Blocked (Tiled) Parallel Code
# ---------------------------------------------------------
# Stages 1 and 4: prange over row blocks, bs x bs tiles.
# Stage 2: prange over COLUMN blocks. Each thread walks every row j but only
# its own strip of columns, so the reads are stride-1 and each thread owns
# its own slice of x (no race).
#
# Expectation / Why: in matmul, tiling keeps A/B/C tiles in cache and reuses
# them ~bs times. GEMVER has no reuse within a stage, so the tiles
# themselves buy nothing. The only gain over Baseline 4 comes from the race-free
# stride-1 parallel stage 2. Expect roughly Baseline 4, not a matmul-sized jump.
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


# ---------------------------------------------------------
# Baseline 6: Blocked Parallel using np.dot / np.outer for Sub-blocks
# ---------------------------------------------------------
# Expectation / Why: in matmul, np.dot on a tile is a Level-3 BLAS call (gemm)
# doing bs^3 FLOPs on bs^2 data, so the call overhead pays for itself. Here
# each tile becomes a Level-2 call (gemv) doing only bs^2 FLOPs on bs^2 data.
# The tiles are not contiguous (Numba copies them for BLAS), and np.outer / the
# np.dot result allocate a new temporary for every tile. Expect this to be
# SLOWER than the plain loops in Baseline 5.
#
# Observed (5900X): about the same as Baseline 5, not clearly slower. The
# per-tile overhead is real, but it is hidden behind the memory traffic of A,
# which is the same for every blocked variant. Unlike matmul, BLAS gives no
# speedup here either.
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


# ---------------------------------------------------------
# Baseline 7: Blocked Temp Copy-In / Copy-Out
# ---------------------------------------------------------
# Stage 1 copies each A tile into a temp, updates it, and writes it back.
# Stages 2 and 4 keep the accumulator segment (x or w) in a temp and add
# every j-block into it before writing it back.
#
# Expectation / Why: in matmul the C tile is updated n / bs times, so a
# private, contiguous copy pays off. In GEMVER each A tile is updated exactly
# once, so its copy is pure extra memory traffic, and the x/w segments are
# only bs doubles that already stay in L1. Expect no gain over Baseline 6.
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


# ---------------------------------------------------------
# Baseline 8: Two-Level Blocked Parallel with Temp Copy & np.dot
# ---------------------------------------------------------
# Direct port of matmul Baseline 8: L2-sized outer blocks, L1-sized inner
# blocks, temp copies of the accumulator segment at both levels.
#
# Expectation / Why: multi-level blocking keeps a working set resident at each
# cache level so it can be reused. GEMVER streams each element of A once per
# stage, so there is nothing to keep resident. This only adds loop and BLAS call
# overhead: expect equal to or slower than Baseline 7.
#
# Observed (5900X): the slowest blocked version at N=512, and roughly equal to
# the others at N=4096. At N=512 the reason is parallelism, not caching: prange
# runs over L2 blocks, and n / 128 = 4 blocks means only 4 of the 24 threads
# get any work. Large blocks reduce the available parallelism.
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
# Baseline 9: Zero Allocation Blocked GEMVER
# ---------------------------------------------------------
# Same blocking as Baselines 6/7, but with no np.outer, no np.dot and no
# temporary arrays inside the tile loops. Each prange iteration allocates one
# bs-length accumulator once and reuses it for every j-block.
#
# Expectation / Why: this isolates what Baselines 6-8 were paying for:
# per-tile allocations and small BLAS calls. It should be much faster than
# 6-8, but it cannot beat Baseline 5 by much, because the memory traffic
# over A is unchanged.
#
# Observed (5900X): only faster than Baseline 8, and within noise of 5-7.
# Allocation and BLAS call overhead were not the bottleneck. Every blocked
# variant moves the same bytes of A, and that is what sets the time.
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
# Baseline 10: Reference NumPy (BLAS)
# ---------------------------------------------------------
# Expectation / Why: stages 2 and 4 are single BLAS gemv calls, which are well
# optimized (single-threaded here, see the *_NUM_THREADS settings). Stage 1
# builds two full N x N np.outer temporaries and then adds them, which is about
# 3x the memory traffic of the fused in-place loop. Serves as a ceiling for
# the serial Numba versions and a floor for the parallel ones.
#
# Observed (5900X): SLOWER than even the naive Numba loop (~1.4 GFLOP/s at
# both N). The BLAS calls are fast, but stage 1 allocates and fills two N x N
# temporaries plus their sum (3 x 128 MB at N=4096, with the page faults of
# fresh memory). For a memory-bound kernel, extra temporaries cost more than
# BLAS saves.
def gemver_numpy_10(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    A += np.outer(u1, v1) + np.outer(u2, v2)
    x += beta * np.dot(A.T, y)
    x += z
    w += alpha * np.dot(A, x)


# ---------------------------------------------------------
# Baseline 11: Fused Stage 1 + Stage 2 with fastmath (GEMVER-specific)
# ---------------------------------------------------------
# The row-major version of PolyBench's Pluto fusion (kernel_pluto): while each
# row of A is being updated (stage 1), use it immediately to update x
# (stage 2, ji order). Stage 4 cannot be fused because it needs the final x.
# Stage 4 accumulates in a scalar and applies alpha once per row.
#
# Expectation / Why: A now streams through the memory hierarchy 3 times
# (read+write in the fused sweep, read in stage 4) instead of 4. At N=512 A
# (2 MB) sits in L3, so the gain is small; at large N (A in DRAM) it should
# approach the ~25% traffic reduction.
#
# Observed (5900X): within noise of Baseline 3 at N=512, and ~15% faster at
# N=4096 (16.2 ms vs 18.9 ms). Fusion only pays once A no longer fits in cache.
@njit(fastmath=True)
def gemver_fused_11(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]

    for i in range(n):
        a1 = u1[i]
        a2 = u2[i]
        c = beta * y[i]
        for j in range(n):
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


# ---------------------------------------------------------
# Baseline 12: Fused + Parallel (GEMVER-specific)
# ---------------------------------------------------------
# Baseline 11 with the fused sweep split into one chunk of rows per thread.
# Every thread's rows contribute to ALL of x, so each thread accumulates into
# its own private row of `partial`, and the partials are then added together
# (fused with stage 3). Stage 4 is a parallel loop over rows.
#
# Expectation / Why: combines stride-1 access, SIMD (fastmath), fewer passes
# over A (fusion) and all cores. Expected to be the fastest Numba version; at
# large N it should approach the DRAM bandwidth limit.
#
# Observed (5900X): fastest at N=4096 (13.5 ms, ~1.4x over serial Baseline 3),
# but tied with 3/11 at N=512. 24 threads give only ~1.4x because the kernel
# already runs close to the dual-channel DDR4 bandwidth (~30 GB/s effective).
# For a memory-bound kernel, the ceiling is bandwidth, not core count.
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
# Baseline 13: Pre-Transposed A (GEMVER-specific)
# ---------------------------------------------------------
# Stage 2 reads A^T, i.e. columns of row-major A. Instead of interchanging the
# loops, build a contiguous transpose AT after stage 1 (A changes there) and
# run stage 2 in its ORIGINAL loop order on AT, which is now stride-1. The
# copy is inside the timed kernel, because it is part of the cost.
#
# Expectation / Why: building AT is itself one strided pass over A (the same
# access pattern it is trying to avoid) plus N^2 extra bytes written, and the
# kernel then still reads AT once. Loop interchange (2_order_ij_ji_ij) gets
# the same stride-1 access with no copy. Expect faster than naive (the strided
# pass is now a simple copy, not a dependent reduction) but slower than
# 2_order_ij_ji_ij / Baseline 3.
#
# Observed (5900X): SLOWER than even the naive Baseline 1 (1.3 ms vs 0.9 ms at
# N=512, 154 ms vs 116 ms at N=4096), and fastmath barely helps. Compared with
# Baseline 3, the fastmath variant spends ~120 ms extra at N=4096, i.e. the
# transposed copy alone costs about as much as the whole naive kernel. The
# improved locality of stage 2 does not come close to paying for building AT.
@njit
def gemver_pretranspose_13(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]

    for i in range(n):
        for j in range(n):
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

    AT = np.ascontiguousarray(A.T)

    for i in range(n):
        for j in range(n):
            x[i] = x[i] + beta * AT[i, j] * y[j]

    for i in range(n):
        x[i] = x[i] + z[i]

    for i in range(n):
        for j in range(n):
            w[i] = w[i] + alpha * A[i, j] * x[j]

@njit(fastmath=True)
def gemver_pretranspose_fastmath_13(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]

    for i in range(n):
        for j in range(n):
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

    AT = np.ascontiguousarray(A.T)

    for i in range(n):
        for j in range(n):
            x[i] = x[i] + beta * AT[i, j] * y[j]

    for i in range(n):
        x[i] = x[i] + z[i]

    for i in range(n):
        for j in range(n):
            w[i] = w[i] + alpha * A[i, j] * x[j]


# ---------------------------------------------------------
# Baseline 14: Local Scalar Accumulators (GEMVER-specific)
# ---------------------------------------------------------
# In the dot-product loops, keep x[i] / w[i] in a local variable and store it
# once after the inner loop, instead of loading and storing the array element
# on every iteration. The arithmetic is otherwise unchanged.
#   scalar_acc_naive:    original order, scalars in stage 2 and stage 4
#   scalar_acc_ij_ji_ij: best order, scalar in stage 4 only (stage 2 is an
#                        axpy over x there, so there is nothing to accumulate)
#   scalar_acc_fastmath: scalar_acc_ij_ji_ij with fastmath
#
# Expectation / Why: Numba cannot prove that w (or x) does not overlap A, so
# "w[i] = w[i] + ..." may store and reload w[i] every iteration, which puts a
# store-to-load forward on top of the add in the dependency chain. A local
# scalar stays in a register. Expect a gain over Baseline 1 and over
# 2_order_ij_ji_ij (no fastmath), and little or none over Baseline 3, where
# the loop is already vectorized.
#
# Observed (5900X): no measurable change in any of the three variants (all
# within ~5% of Baseline 1, 2_order_ij_ji_ij and Baseline 3 respectively, at
# both N=512 and N=4096). The expected store/reload penalty is not there, so
# the compiler evidently already keeps the running sum in a register. The
# limits remain the ones found earlier: the add dependency chain without
# fastmath, and memory traffic with it.
@njit
def gemver_scalar_acc_naive_14(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    n = A.shape[0]

    for i in range(n):
        for j in range(n):
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

    for i in range(n):
        s = x[i]
        for j in range(n):
            s = s + beta * A[j, i] * y[j]
        x[i] = s

    for i in range(n):
        x[i] = x[i] + z[i]

    for i in range(n):
        s = w[i]
        for j in range(n):
            s = s + alpha * A[i, j] * x[j]
        w[i] = s

@njit
def gemver_scalar_acc_ij_ji_ij_14(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
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
        s = w[i]
        for j in range(n):
            s = s + alpha * A[i, j] * x[j]
        w[i] = s

@njit(fastmath=True)
def gemver_scalar_acc_fastmath_14(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
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
        s = w[i]
        for j in range(n):
            s = s + alpha * A[i, j] * x[j]
        w[i] = s


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
        ("13_pretranspose", gemver_pretranspose_13),
        ("13_pretranspose_fastmath", gemver_pretranspose_fastmath_13),
        ("14_scalar_acc_naive", gemver_scalar_acc_naive_14),
        ("14_scalar_acc_ij_ji_ij", gemver_scalar_acc_ij_ji_ij_14),
        ("14_scalar_acc_fastmath", gemver_scalar_acc_fastmath_14),
    ]

    for name, fn in functions:
        gflops, elapsed, is_correct = measure(fn)
        results.append((name, gflops, elapsed, is_correct))

    return results


# ---------------------------------------------------------
# Thread-Count Sweep
# ---------------------------------------------------------
# Run with:  python gemver_numba_bench.py <N> threads
#
# The 5900X has 12 physical cores / 24 logical threads. The sweep runs the
# parallel baselines at each thread count and compares them with the best
# serial version (11_fused_fastmath).
#
# Expectation / Why: GEMVER is memory-bound, so extra threads only help until
# the memory system is saturated. Expect the curves to flatten after a few
# threads at large N (A in DRAM), no gain from 24 logical threads over 12
# physical cores, and better scaling when A fits in L3.
#
# Observed (5900X), N=4096:
#   12_fused_parallel peaks at 2-4 threads (only ~1.25x over its own 1-thread
#   run, ~1.4x over serial Baseline 11) and then DROPS back to ~1.0x at 24
#   threads: once memory is saturated, more threads only add contention.
#   4_parallel_rows scales best (~4.8x) but only because its 1-thread run is
#   slow: its strided stage 2 waits on cache misses, and several threads can
#   wait at once. It flattens at 12 threads (24 adds nothing) and never reaches
#   serial Baseline 11. Good scaling is not the same as good performance.
#   5_blocked_parallel flattens at ~2x by 6 threads.
# Observed (5900X), N=1024 (A = 8 MB, fits in L3): the kernel takes under 1 ms
#   and the results are not stable between runs (12_fused_parallel at 12 threads
#   measured both 1.2x and 2.7x), so no scaling conclusion is drawn at this size.
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
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    markers = ["o", "s", "^"]
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

    print(f"\nRunning GEMVER Benchmarks for N={N_input}...")
    print(f"CPU: {CPU_NAME}")
    print(f"Numba threads: {get_num_threads()}\n")

    benchmark_data = run_benchmark(N_input)

    # The threading layer is only chosen once a parallel function has run.
    print(f"Numba threading layer: {threading_layer()}\n")

    py_elapsed = benchmark_data[0][2]  # Reference execution time for pure Python naive

    # Table Header Formatting
    header = (
        f"| {'Baseline Implementation':<28} "
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
            f"| {name:<28} "
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
    plt.title(f"Numba GEMVER Benchmark Performance (N={N_input})\nCPU: {CPU_NAME} | Numba threads: {get_num_threads()}")
    plt.xticks(rotation=45, ha="right")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
