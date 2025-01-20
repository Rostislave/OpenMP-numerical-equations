#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))
#define N (2*2*2*2*2*2*2*2*2*2*2*2*2*2+10000)
#define TASK_SIZE 1024

double maxeps = 0.1e-7;
int itmax = 10;
double eps;
double A[N][N], B[N][N];

void init();
void relax(double src[N][N], double dest[N][N]);
void verify(double src[N][N]);

int main() {
    double time_start, time_end;
    time_start = omp_get_wtime();
    int it;
    init();

    for (it = 1; it <= itmax; it++) {
        eps = 0.0;

        if (it % 2 == 1) {
            relax(B, A);
        } else {
            relax(A, B);
        }

        printf("it=%4d   eps=%f\n", it, eps);
        if (eps < maxeps) {
            break;
        }
    }

    if ((it - 1) % 2 == 1) {
        verify(A);
    } else {
        verify(B);
    }
    time_end = omp_get_wtime();
    printf("Time = %f sec\n", time_end - time_start);
    return 0;
}

void init() {
    for (int i = 0; i < N; i++) {
        A[0][i] = A[N-1][i] = 0.0;
        A[i][0] = A[i][N-1] = 0.0;

        B[0][i] = B[N-1][i] = 0.0;
        B[i][0] = B[i][N-1] = 0.0;
    }

    for (int i = 1; i <= N-2; i++) {
        for (int j = 1; j <= N-2; j++) {
            A[i][j] = 1.0 + i + j;
            B[i][j] = 1.0 + i + j;
        }
    }
}

void process_block(int bi, int bj, double src[N][N], double dest[N][N], double *local_eps) {
    double block_eps = 0.0;

    for (int i = bi; i < bi + TASK_SIZE && i < N-1; i++) {
        for (int j = bj; j < bj + TASK_SIZE && j < N-1; j++) {
            double new_val = (src[i-1][j] + src[i+1][j] +
                              src[i][j-1] + src[i][j+1]) / 4.0;

            double diff = fabs(src[i][j] - new_val);
            block_eps = Max(block_eps, diff);

            dest[i][j] = new_val;
        }
    }

    #pragma omp critical
    {
        *local_eps = Max(*local_eps, block_eps);
    }
}

void relax(double src[N][N], double dest[N][N]) {
    double local_eps = 0.0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int bi = 1; bi < N-1; bi += TASK_SIZE) {
                for (int bj = 1; bj < N-1; bj += TASK_SIZE) {
                    #pragma omp task firstprivate(bi, bj) shared(local_eps, src, dest)
                    {
                        process_block(bi, bj, src, dest, &local_eps);
                    }
                }
            }
        }
    }

    eps = local_eps;
}