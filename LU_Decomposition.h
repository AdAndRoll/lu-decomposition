#include <omp.h>
#include <vector>
#include <algorithm>
#include <cstring>

void LU_Decomposition(double *A, double *L, double *U, int n) {

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        memset(&L[i * n], 0, n * sizeof(double));
        memset(&U[i * n], 0, n * sizeof(double));
        L[i * n + i] = 1.0;
    }

    const int BLOCK_SIZE = 128; 

    #pragma omp parallel
    #pragma omp single
    {
        for (int k = 0; k < n; k += BLOCK_SIZE) {
            int k_end = std::min(k + BLOCK_SIZE, n);

            #pragma omp task depend(inout: A[k*n:BLOCK_SIZE*n]) shared(A, L, U)
            {
                for (int i = k; i < k_end; i++) {
                    int i_off = i * n;
                    for (int j = i; j < k_end; j++) {
                        double sum = 0;
                        for (int m = k; m < i; m++) sum += A[i_off + m] * A[m * n + j];
                        A[i_off + j] -= sum;
                        U[i_off + j] = A[i_off + j];
                    }
                    double diag_val = A[i_off + i];
                    for (int j = i + 1; j < k_end; j++) {
                        int j_off = j * n;
                        double sum = 0;
                        for (int m = k; m < i; m++) sum += A[j_off + m] * A[m * n + i];
                        A[j_off + i] = (A[j_off + i] - sum) / diag_val;
                        L[j_off + i] = A[j_off + i];
                    }
                }
            }

            for (int b = k + BLOCK_SIZE; b < n; b += BLOCK_SIZE) {
                int b_end = std::min(b + BLOCK_SIZE, n);

                #pragma omp task depend(in: A[k*n:BLOCK_SIZE*n]) depend(inout: A[k*n:BLOCK_SIZE*n]) shared(A, U)
                {
                    for (int i = k; i < k_end; i++) {
                        int i_off = i * n;
                        for (int col = b; col < b_end; col++) {
                            double sum = 0;
                            for (int m = k; m < i; m++) sum += A[i_off + m] * A[m * n + col];
                            A[i_off + col] -= sum;
                            U[i_off + col] = A[i_off + col];
                        }
                    }
                }

                #pragma omp task depend(in: A[k*n:BLOCK_SIZE*n]) depend(inout: A[b*n:BLOCK_SIZE*n]) shared(A, L)
                {
                    for (int row = b; row < b_end; row++) {
                        int r_off = row * n;
                        for (int col = k; col < k_end; col++) {
                            double sum = 0;
                            for (int m = k; m < col; m++) sum += A[r_off + m] * A[m * n + col];
                            A[r_off + col] = (A[r_off + col] - sum) / A[col * n + col];
                            L[r_off + col] = A[r_off + col];
                        }
                    }
                }
            }
            
            #pragma omp taskwait 

            for (int i = k + BLOCK_SIZE; i < n; i += BLOCK_SIZE) {
                int i_end = std::min(i + BLOCK_SIZE, n);
                for (int j = k + BLOCK_SIZE; j < n; j += BLOCK_SIZE) {
                    int j_end = std::min(j + BLOCK_SIZE, n);

                    #pragma omp task depend(in: A[i*n:BLOCK_SIZE*n], A[k*n:BLOCK_SIZE*n]) depend(inout: A[i*n:BLOCK_SIZE*n]) shared(A)
                    {
                        for (int row = i; row < i_end; row++) {
                            int r_off = row * n;
                            for (int m = k; m < k_end; m++) {
                                double l_val = A[r_off + m];
                                int m_off = m * n;
                                #pragma omp simd
                                for (int col = j; col < j_end; col++) {
                                    A[r_off + col] -= l_val * A[m_off + col];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}