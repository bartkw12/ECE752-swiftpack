/*
 * GEMVER from PolyBench/C 4.2.1, with a timing driver for gemver_numba_bench.py
 * ("polybench_c" mode: the optimized C reference row).
 *
 * - init_array() and kernel_gemver() are copied unchanged from PolyBench/C 4.2.1
 *   (linear-algebra/blas/gemver/gemver.c): same data, same loop order.
 * - The 5 macros below give the same code as polybench.h's defaults
 *   (no padding, no restrict, array sizes fixed at compile time, loop bound n).
 * - main() is new: it times the kernel the same way as the Python rows.
 *     - first call on fresh arrays; A, x, w are written to argv[1] so the Python
 *       script can check them against NumPy
 *     - then the same arrays again ("warm"): median of >= 9 calls and >= 0.3 s
 *     - PolyBench's own main() flushes a 32 MB buffer and times one call.
 * - Build (N is a compile-time constant, as in PolyBench):
 *     gcc -O3 -march=native -DN=4096 gemver_polybench.c -o gemver_c
 *     gcc -O3 -march=native -ffast-math -DPOLYBENCH_USE_RESTRICT -DN=4096 ...
 *     cl /O2 /arch:AVX2 /DN=4096 gemver_polybench.c
 * - Run:  ./gemver_c out.bin  ->  prints "N calls median_seconds checksum(w)"
 *
 * ---------------------------------------------------------------------------
 * PolyBench/C license (kept as its terms require):
 *
 * OHIO STATE UNIVERSITY SOFTWARE DISTRIBUTION LICENSE
 *
 * PolyBench/C, a collection of benchmarks containing static control
 * parts (the "Software")
 * Copyright (c) 2010-2016, Ohio State University. All rights reserved.
 *
 * Redistribution and use of the Software in source and binary forms, with or
 * without modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the capitalized paragraph below.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the capitalized paragraph below in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of Ohio State University, or its faculty, staff or students may
 *    not be used to endorse or promote products derived from the Software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE HAS BEEN APPROVED FOR PUBLIC RELEASE, UNLIMITED
 * DISTRIBUTION.  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT ANY
 * EXPRESS, IMPLIED OR STATUTORY WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, WARRANTIES OF ACCURACY, COMPLETENESS, NONINFRINGEMENT,
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * ACCESS OR USE OF THE SOFTWARE IS ENTIRELY AT THE USER'S RISK.  IN NO
 * EVENT SHALL OHIO STATE UNIVERSITY OR ITS FACULTY, STAFF OR STUDENTS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  THE SOFTWARE USER SHALL
 * INDEMNIFY, DEFEND AND HOLD HARMLESS OHIO STATE UNIVERSITY AND ITS
 * FACULTY, STAFF AND STUDENTS FROM ANY AND ALL CLAIMS, ACTIONS, DAMAGES,
 * LOSSES, LIABILITIES, COSTS AND EXPENSES, INCLUDING ATTORNEYS' FEES AND
 * COURT COSTS, DIRECTLY OR INDIRECTLY ARISING OUT OF OR IN CONNECTION
 * WITH ACCESS OR USE OF THE SOFTWARE.
 * ---------------------------------------------------------------------------
 */
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#  include <malloc.h>
/* the two timer functions from windows.h (that header breaks when N is a macro) */
__declspec(dllimport) int __stdcall QueryPerformanceCounter(long long *count);
__declspec(dllimport) int __stdcall QueryPerformanceFrequency(long long *freq);
#else
#  include <time.h>
#endif

#ifndef N
#  define N 2000                 /* PolyBench's LARGE_DATASET */
#endif

/* Same expansion as polybench.h with its defaults: POLYBENCH_PADDING_FACTOR 0,
   no POLYBENCH_USE_C99_PROTO (sizes = N), no POLYBENCH_USE_SCALAR_LB (bound = n).
   -DPOLYBENCH_USE_RESTRICT (a PolyBench option) marks the arrays restrict:
   "they do not overlap", so the compiler may keep x[i] and w[i] in registers. */
#ifdef POLYBENCH_USE_RESTRICT
#  define POLYBENCH_RESTRICT restrict
#else
#  define POLYBENCH_RESTRICT
#endif
#define DATA_TYPE double
#define POLYBENCH_1D(var, dim1, ddim1) var[POLYBENCH_RESTRICT dim1]
#define POLYBENCH_2D(var, dim1, dim2, ddim1, ddim2) var[POLYBENCH_RESTRICT dim1][dim2]
#define POLYBENCH_LOOP_BOUND(x, y) y
#define _PB_N POLYBENCH_LOOP_BOUND(N,n)


/* ======== From PolyBench/C 4.2.1 gemver.c (unchanged) ======== */

/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE *alpha,
		 DATA_TYPE *beta,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(u1,N,n),
		 DATA_TYPE POLYBENCH_1D(v1,N,n),
		 DATA_TYPE POLYBENCH_1D(u2,N,n),
		 DATA_TYPE POLYBENCH_1D(v2,N,n),
		 DATA_TYPE POLYBENCH_1D(w,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;

  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      u1[i] = i;
      u2[i] = ((i+1)/fn)/2.0;
      v1[i] = ((i+1)/fn)/4.0;
      v2[i] = ((i+1)/fn)/6.0;
      y[i] = ((i+1)/fn)/8.0;
      z[i] = ((i+1)/fn)/9.0;
      x[i] = 0.0;
      w[i] = 0.0;
      for (j = 0; j < n; j++)
        A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemver(int n,
		   DATA_TYPE alpha,
		   DATA_TYPE beta,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(u1,N,n),
		   DATA_TYPE POLYBENCH_1D(v1,N,n),
		   DATA_TYPE POLYBENCH_1D(u2,N,n),
		   DATA_TYPE POLYBENCH_1D(v2,N,n),
		   DATA_TYPE POLYBENCH_1D(w,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n),
		   DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

#pragma scop

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < _PB_N; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      w[i] = w[i] +  alpha * A[i][j] * x[j];

#pragma endscop
}

/* ======== End of the PolyBench/C code ======== */


/* Timing driver (new) */

static double now_seconds(void)
{
#ifdef _WIN32
  long long freq, count;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&count);
  return (double)count / (double)freq;
#else
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
#endif
}

/* 4096-byte aligned, like polybench_alloc_data() */
static void *alloc_data(size_t bytes)
{
  void *p = NULL;
#ifdef _WIN32
  p = _aligned_malloc(bytes, 4096);
#else
  if (posix_memalign(&p, 4096, bytes) != 0)
    p = NULL;
#endif
  if (!p) {
    fprintf(stderr, "cannot allocate %lu bytes\n", (unsigned long)bytes);
    exit(1);
  }
  return p;
}

static void free_data(void *p)
{
#ifdef _WIN32
  _aligned_free(p);
#else
  free(p);
#endif
}

static int compare_doubles(const void *a, const void *b)
{
  double da = *(const double *)a, db = *(const double *)b;
  return (da > db) - (da < db);
}

#define MIN_CALLS 9
#define MAX_CALLS 200
#define MIN_TIME 0.3

int main(int argc, char **argv)
{
  int n = N;
  DATA_TYPE alpha, beta;
  DATA_TYPE (*A)[N][N] = alloc_data(sizeof(*A));
  DATA_TYPE (*u1)[N] = alloc_data(sizeof(*u1));
  DATA_TYPE (*v1)[N] = alloc_data(sizeof(*v1));
  DATA_TYPE (*u2)[N] = alloc_data(sizeof(*u2));
  DATA_TYPE (*v2)[N] = alloc_data(sizeof(*v2));
  DATA_TYPE (*w)[N] = alloc_data(sizeof(*w));
  DATA_TYPE (*x)[N] = alloc_data(sizeof(*x));
  DATA_TYPE (*y)[N] = alloc_data(sizeof(*y));
  DATA_TYPE (*z)[N] = alloc_data(sizeof(*z));
  double times[MAX_CALLS], total = 0.0, median, checksum = 0.0;
  int calls = 0, i;

  init_array(n, &alpha, &beta, *A, *u1, *v1, *u2, *v2, *w, *x, *y, *z);

  /* first call: its result is checked against NumPy by the Python script */
  kernel_gemver(n, alpha, beta, *A, *u1, *v1, *u2, *v2, *w, *x, *y, *z);
  if (argc > 1) {
    FILE *out = fopen(argv[1], "wb");
    if (!out) {
      fprintf(stderr, "cannot open %s\n", argv[1]);
      return 1;
    }
    fwrite(*A, sizeof(DATA_TYPE), (size_t)N * N, out);
    fwrite(*x, sizeof(DATA_TYPE), N, out);
    fwrite(*w, sizeof(DATA_TYPE), N, out);
    fclose(out);
  }

  /* warm timing: the same arrays again, median of >= 9 calls and >= 0.3 s */
  while (calls < MIN_CALLS || (total < MIN_TIME && calls < MAX_CALLS)) {
    double start = now_seconds();
    kernel_gemver(n, alpha, beta, *A, *u1, *v1, *u2, *v2, *w, *x, *y, *z);
    times[calls] = now_seconds() - start;
    total += times[calls++];
  }
  qsort(times, calls, sizeof(double), compare_doubles);
  median = calls % 2 ? times[calls / 2] : 0.5 * (times[calls / 2 - 1] + times[calls / 2]);

  for (i = 0; i < n; i++)          /* uses w, so the calls cannot be removed */
    checksum += (*w)[i];
  printf("%d %d %.9e %.6e\n", N, calls, median, checksum);

  free_data(A); free_data(u1); free_data(v1); free_data(u2); free_data(v2);
  free_data(w); free_data(x); free_data(y); free_data(z);
  return 0;
}
