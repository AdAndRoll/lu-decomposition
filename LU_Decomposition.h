#include <algorithm>
#include <cmath>
#include <omp.h>
#include <cstring>

void LU_Decomposition(double* A, double* L, double* U, int n)
{
    if (!A || !L || !U || n <= 0)
        return;

    const int b_size = 64;
    const double precision = 1e-15;

    for (int k_outer = 0; k_outer < n; k_outer += b_size)
    {
        int curr_b = std::min(b_size, n - k_outer);
        int b_limit = k_outer + curr_b;

        for (int k = k_outer; k < b_limit; ++k)
        {
            double* base_row = A + k * n;
            
            if (std::abs(base_row[k]) < precision)
            {
                base_row[k] = (base_row[k] < 0.0) ? -precision : precision;
            }

            double inv_pivot = 1.0 / base_row[k];

            for (int i = k + 1; i < b_limit; ++i)
            {
                double* target_row = A + i * n;
                double acc = target_row[k];
                for (int p = k_outer; p < k; ++p)
                {
                    acc -= target_row[p] * A[p * n + k];
                }
                target_row[k] = acc * inv_pivot;
            }

            for (int j = k + 1; j < b_limit; ++j)
            {
                double acc = base_row[j];
                for (int p = k_outer; p < k; ++p)
                {
                    acc -= base_row[p] * A[p * n + j];
                }
                base_row[j] = acc;
            }
        }

        if (b_limit >= n) continue;

        #pragma omp parallel
        {
            #pragma omp for schedule(static) nowait
            for (int j = b_limit; j < n; ++j)
            {
                for (int i = k_outer; i < b_limit; ++i)
                {
                    double* r_i = A + i * n;
                    double dot = r_i[j];
                    for (int p = k_outer; p < i; ++p)
                    {
                        dot -= r_i[p] * A[p * n + j];
                    }
                    r_i[j] = dot;
                }
            }

            #pragma omp for schedule(static)
            for (int i = b_limit; i < n; ++i)
            {
                double* r_i = A + i * n;
                for (int j = k_outer; j < b_limit; ++j)
                {
                    double dot = r_i[j];
                    for (int p = k_outer; p < j; ++p)
                    {
                        dot -= r_i[p] * A[p * n + j];
                    }
                    r_i[j] = dot / A[j * n + j];
                }
            }

            #pragma omp for schedule(static)
            for (int i = b_limit; i < n; ++i)
            {
                double* r_i = A + i * n;
                for (int p = k_outer; p < b_limit; ++p)
                {
                    double* r_p = A + p * n;
                    double multiplier = r_i[p];
                    int j = b_limit;
                    for (; j <= n - 4; j += 4)
                    {
                        r_i[j] -= multiplier * r_p[j];
                        r_i[j + 1] -= multiplier * r_p[j + 1];
                        r_i[j + 2] -= multiplier * r_p[j + 2];
                        r_i[j + 3] -= multiplier * r_p[j + 3];
                    }
                    for (; j < n; ++j)
                    {
                        r_i[j] -= multiplier * r_p[j];
                    }
                }
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
    {
        double* src = A + i * n;
        double* dst_l = L + i * n;
        double* dst_u = U + i * n;
        for (int j = 0; j < n; ++j)
        {
            if (i > j)
            {
                dst_l[j] = src[j];
                dst_u[j] = 0.0;
            }
            else if (i == j)
            {
                dst_l[j] = 1.0;
                dst_u[j] = src[j];
            }
            else
            {
                dst_l[j] = 0.0;
                dst_u[j] = src[j];
            }
        }
    }
}