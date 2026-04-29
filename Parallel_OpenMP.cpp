#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <random>
#include <iomanip>
#include <sstream>  
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// Функция для генерации случайной матрицы
void initMatrix(const string& filename, int size) {
    ofstream out(filename);
    if (!out.is_open()) {
        cerr << "Ошибка открытия файла: " << filename << endl;
        return;
    }

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(0.0, 1.0);

    out << size << "\n";
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            out << fixed << setprecision(6) << dist(gen);
            if (j < size - 1) out << " ";
        }
        out << "\n";
    }
    out.close();
}

// Чтение матрицы из файла
vector<double> loadMatrix(const string& filename, int& size) {
    ifstream in(filename);
    if (!in.is_open()) {
        cerr << "Ошибка открытия файла: " << filename << endl;
        return {};
    }

    in >> size;
    vector<double> mat(size * size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            in >> mat[i * size + j];
        }
    }
    in.close();
    return mat;
}

// Умножение матриц с OpenMP
vector<double> multiplyMatrices(const vector<double>& A, const vector<double>& B,
    int size, int threadCount) {
    vector<double> C(size * size, 0.0);

#ifdef _OPENMP
    omp_set_num_threads(threadCount);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double sum = 0.0;
            for (int k = 0; k < size; ++k) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
#else
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double sum = 0.0;
            for (int k = 0; k < size; ++k) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
#endif

    return C;
}

// Сохранение результата
void saveResult(const string& filename, const vector<double>& C, int size) {
    ofstream out(filename);
    if (!out.is_open()) {
        cerr << "Ошибка открытия файла: " << filename << endl;
        return;
    }

    out << size << "\n";
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            out << fixed << setprecision(6) << C[i * size + j];
            if (j < size - 1) out << " ";
        }
        out << "\n";
    }
    out.close();
}

int main() {
    setlocale(LC_ALL, "Russian");

    cout << "=== Лабораторная работа: Параллельное умножение матриц ===" << endl;
    cout << "OpenMP версия" << endl;

#ifdef _OPENMP
    int maxThreads = omp_get_max_threads();
    cout << "Максимальное количество потоков: " << maxThreads << endl;
#else
    cout << "OpenMP не поддерживается" << endl;
    return 1;
#endif

    // Размеры матриц (до 1600)
    vector<int> matrixSizes = { 100, 250, 500, 800, 1100, 1400, 1600 };
    vector<int> threadCounts = { 1, 2, 4 };

    ofstream results("experiment_results.txt");
    if (!results.is_open()) {
        cerr << "Не удалось создать файл результатов" << endl;
        return 1;
    }

    cout << "\nЗапуск экспериментов...\n" << endl;

    for (int size : matrixSizes) {
        cout << "Размер матрицы: " << size << "x" << size << endl;

        // Генерация входных матриц
        initMatrix("matrix_a.txt", size);
        initMatrix("matrix_b.txt", size);

        int loadedSize;
        vector<double> A = loadMatrix("matrix_a.txt", loadedSize);
        vector<double> B = loadMatrix("matrix_b.txt", loadedSize);

        if (A.empty() || B.empty()) {
            cerr << "Ошибка загрузки матриц для размера " << size << endl;
            continue;
        }

        for (int threads : threadCounts) {
            if (threads > maxThreads) continue;

            auto start = chrono::high_resolution_clock::now();
            vector<double> C = multiplyMatrices(A, B, size, threads);
            auto end = chrono::high_resolution_clock::now();

            double elapsed = chrono::duration<double>(end - start).count();

            cout << "  Потоков: " << threads << " -> Время: " << fixed << setprecision(4) << elapsed << " сек" << endl;
            results << size << " " << threads << " " << elapsed << "\n";

            // Сохраняем результат только для максимального размера
            if (size == 1600) {
                saveResult("result_" + to_string(threads) + ".txt", C, size);
            }
        }
        cout << endl;
    }

    results.close();

    cout << "\nЭксперименты завершены!" << endl;
    cout << "Результаты сохранены в experiment_results.txt" << endl;

    return 0;
}