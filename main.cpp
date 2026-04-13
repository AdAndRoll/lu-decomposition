#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include "LU_Decomposition.h"
// Вставляем функцию LU_Decomposition здесь или линкуем объектный файл

void verify_result(double *A_orig, double *L, double *U, int n) {
    double diff_norm_sq = 0;
    double a_norm_sq = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double lu_val = 0;
            for (int k = 0; k < n; k++) {
                lu_val += L[i * n + k] * U[k * n + j];
            }
            
            double diff = lu_val - A_orig[i * n + j];
            diff_norm_sq += diff * diff;
            a_norm_sq += A_orig[i * n + j] * A_orig[i * n + j];
        }
    }

    double relative_error = std::sqrt(diff_norm_sq) / std::sqrt(a_norm_sq);
    std::cout << "Relative Error: " << relative_error << std::endl;
    
    if (relative_error < 0.01) {
        std::cout << "Status: SUCCESS" << std::endl;
    } else {
        std::cout << "Status: FAILED" << std::endl;
    }
}

int main() {
    int n = 3000; // Можно увеличить до 3000 для финального теста
    std::cout << "Matrix size: " << n << "x" << n << std::endl;

    // Выделение памяти (используем векторы для автоматического управления)
    std::vector<double> A(n * n);
    std::vector<double> A_copy(n * n);
    std::vector<double> L(n * n);
    std::vector<double> U(n * n);

    // Инициализация матрицы случайными числами
    // Важно: делаем матрицу диагонально доминирующей, чтобы избежать деления на 0
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = dis(gen);
            if (i == j) A[i * n + j] += n; // Гарантируем стабильность
            A_copy[i * n + j] = A[i * n + j];
        }
    }

    // Замер времени
    auto start = std::chrono::high_resolution_clock::now();
    
    LU_Decomposition(A.data(), L.data(), U.data(), n);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Time: " << diff.count() << " seconds" << std::endl;
    
    // Вычисление GFLOPS (приблизительно 2/3 * n^3 операций)
    double gflops = (2.0 * n * n * n) / (3.0 * diff.count() * 1e9);
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    // Проверка (только для небольших n, так как проверка занимает O(n^3))
    if (n <= 1000) {
        verify_result(A_copy.data(), L.data(), U.data(), n);
    }

    return 0;
}