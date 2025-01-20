#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))
#define N (2*2*2*2*2*2*2*2*2*2*2*2*2*2+10000)

double maxeps = 0.1e-7;
int itmax = 10;
double eps;
double A[N][N], B[N][N];
int i, j;

void init();
void relax(double src[N][N], double dest[N][N]);
void verify(double src[N][N]);

int main() {
    int it;

    #pragma omp parallel private(it)
    {
        init();
        for (it = 1; it <= itmax; it++) {
            #pragma omp barrier
            #pragma omp master
                eps = 0.0;

            if (it % 2 == 1) {
                relax(B, A);
            } else {
                relax(A, B);
            }
            #pragma omp single
            {
                printf("it=%4d   eps=%f\n", it, eps);
            }
            if (eps < maxeps) {
                break;
            }
        }

        if ((it - 1) % 2 == 1) {
            verify(A);
        } else {
            verify(B);
        }
    }
    return 0;
}

void init() {
    #pragma omp for schedule(static) private(i)
    for (int i = 0; i < N; i++) {
        A[0][i] = A[N-1][i] = 0.0;
        A[i][0] = A[i][N-1] = 0.0;

        B[0][i] = B[N-1][i] = 0.0;
        B[i][0] = B[i][N-1] = 0.0;
    }

    #pragma omp for schedule(static) private(i, j)
    for (int i = 1; i <= N-2; i++) {
        for (int j = 1; j <= N-2; j++) {
            A[i][j] = 1.0 + i + j;
            B[i][j] = 1.0 + i + j;
        }
    }
}

void relax(double src[N][N], double dest[N][N]) {

    #pragma omp for reduction(max:eps) private(i, j) schedule(static)
    for (int i = 1; i <= N-2; i++) {
        for (int j = 1; j <= N-2; j++) {
            double new_val = (src[i-1][j] + src[i+1][j] +
                              src[i][j-1] + src[i][j+1]) / 4.0;

            double diff = fabs(src[i][j] - new_val);
            eps = Max(eps, diff);

            dest[i][j] = new_val;
        }
    }
}

void verify(double src[N][N]) {
    double s = 0.0;

    #pragma omp parallel for reduction(+:s) private(i, j) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            s += src[i][j] * (i + 1) * (j + 1);
        }
    }
    #pragma omp master
    {
        s /= (N * N);
        printf("  S = %f\n", s);
    }

}