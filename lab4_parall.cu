#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <random>
#include <cuda_runtime.h>

using namespace std;

// Оптимизированное CUDA ядро с использованием разделяемой памяти (shared memory)
__global__ void matrixMulKernelShared(const double* A, const double* B, double* C, int n) {
    __shared__ double sharedA[32][32];
    __shared__ double sharedB[32][32];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * 32 + ty;
    int col = bx * 32 + tx;

    double sum = 0.0;
    for (int k = 0; k < n; k += 32) {
        if (row < n && k + tx < n)
            sharedA[ty][tx] = A[row * n + k + tx];
        else
            sharedA[ty][tx] = 0.0;

        if (col < n && k + ty < n)
            sharedB[ty][tx] = B[(k + ty) * n + col];
        else
            sharedB[ty][tx] = 0.0;
        __syncthreads();

        for (int i = 0; i < 32; i++) {
            sum += sharedA[ty][i] * sharedB[i][tx];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Простое ядро для маленьких матриц (базовое)
__global__ void matrixMulKernelSimple(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Функции для работы с матрицами 
void generateMatrix(const string& filename, int n) {
    ofstream file(filename);
    mt19937 gen(42); // Фиксированное зерно
    uniform_real_distribution<> dis(1.0, 10.0);

    file << n << "\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << dis(gen);
            if (j < n - 1) file << " ";
        }
        file << "\n";
    }
    file.close();
}

vector<vector<double>> readMatrix(const string& filename, int& n) {
    ifstream file(filename);
    file >> n;
    vector<vector<double>> matrix(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            file >> matrix[i][j];
    file.close();
    return matrix;
}

vector<vector<double>> multiplySequential(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

void writeResult(const string& filename, const vector<vector<double>>& C) {
    ofstream file(filename);
    int n = C.size();
    file << n << "\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << C[i][j];
            if (j < n - 1) file << " ";
        }
        file << "\n";
    }
    file.close();
}

int main() {
    int sizes[] = { 100, 200, 300, 400, 600, 800, 1000, 1200 };
    int num_sizes = 8;

    // Конфигурации блоков для GTX 1660 Ti
    struct BlockConfig {
        int x;
        int y;
        int threads;
        const char* name;
    };

    BlockConfig blockConfigs[] = {
        {8, 8, 64, "8x8 (64 threads)"},
        {16, 16, 256, "16x16 (256 threads)"},
        {32, 16, 512, "32x16 (512 threads)"},
        {32, 32, 1024, "32x32 (1024 threads)"}
    };
    int numConfigs = 4;

    // Проверка CUDA
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    if (deviceCount == 0) {
        cerr << "No CUDA devices found!" << endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    cout << "========================================" << endl;
    cout << "CUDA Matrix Multiplication Benchmark" << endl;
    cout << "========================================" << endl;
    cout << "GPU: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "CUDA Cores: " << prop.multiProcessorCount * 64 << " (approx)" << endl;
    cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
    cout << "Max block dimensions: " << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << endl;
    cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << endl;
    cout << "========================================" << endl << endl;

    // Очистка файла результатов
    ofstream results("cuda_results.txt");
    results << "N,BlockSize,Threads,Time(ms),Speedup,GFLOPS" << endl;
    results.close();

    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];
        cout << "=== Testing N=" << n << " ===" << endl;

        generateMatrix("mat_a.txt", n);
        generateMatrix("mat_b.txt", n);

        int n1, n2;
        auto A = readMatrix("mat_a.txt", n1);
        auto B = readMatrix("mat_b.txt", n2);

        // Последовательное умножение (для сравнения)
        cout << "  Sequential... ";
        auto start_seq = chrono::high_resolution_clock::now();
        auto C_seq = multiplySequential(A, B);
        auto end_seq = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> seq_time = end_seq - start_seq;
        cout << seq_time.count() << " ms" << endl;

        // Подготовка данных для CUDA
        int size = n * n;
        size_t bytes = size * sizeof(double);

        // Уплощение матриц
        vector<double> A_flat(size), B_flat(size);
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                A_flat[row * n + col] = A[row][col];
                B_flat[row * n + col] = B[row][col];
            }
        }

        // Выделение памяти на GPU
        double* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);

        cudaMemcpy(d_A, A_flat.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B_flat.data(), bytes, cudaMemcpyHostToDevice);

        // Тестирование разных конфигураций
        for (int cfg = 0; cfg < numConfigs; cfg++) {
            int blockX = blockConfigs[cfg].x;
            int blockY = blockConfigs[cfg].y;

            dim3 blockDim(blockX, blockY);
            dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                (n + blockDim.y - 1) / blockDim.y);

            // Выбор ядра (для маленьких матриц - простое, для больших - с shared memory)
            auto kernel = (n <= 400) ? matrixMulKernelSimple : matrixMulKernelShared;
            size_t sharedMemSize = (n <= 400) ? 0 : (blockX * blockY * 2 * sizeof(double));

            // Прогрев (3 раза)
            for (int warm = 0; warm < 3; warm++) {
                if (n <= 400) {
                    matrixMulKernelSimple << <gridDim, blockDim >> > (d_A, d_B, d_C, n);
                }
                else {
                    matrixMulKernelShared << <gridDim, blockDim, sharedMemSize >> > (d_A, d_B, d_C, n);
                }
            }
            cudaDeviceSynchronize();

            // Измерение времени
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            if (n <= 400) {
                matrixMulKernelSimple << <gridDim, blockDim >> > (d_A, d_B, d_C, n);
            }
            else {
                matrixMulKernelShared << <gridDim, blockDim, sharedMemSize >> > (d_A, d_B, d_C, n);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            // Проверка ошибок
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                cout << "  " << blockConfigs[cfg].name << ": ERROR - " << cudaGetErrorString(err) << endl;
                continue;
            }

            double speedup = seq_time.count() / elapsed_ms;
            long long flops = 2LL * n * n * n;
            double gflops = flops / (elapsed_ms * 1e6);

            cout << "  " << blockConfigs[cfg].name << ": " << elapsed_ms << " ms, "
                << "x" << speedup << " speedup, " << gflops << " GFLOPS" << endl;

            // Запись результатов
            ofstream results("cuda_results.txt", ios::app);
            results << n << "," << blockConfigs[cfg].name << "," << blockConfigs[cfg].threads << ","
                << elapsed_ms << "," << speedup << "," << gflops << endl;
            results.close();
        }

        // Копирование результата для верификации
        vector<double> C_flat(size);
        cudaMemcpy(C_flat.data(), d_C, bytes, cudaMemcpyDeviceToHost);

        vector<vector<double>> C(n, vector<double>(n));
        for (int row = 0; row < n; row++)
            for (int col = 0; col < n; col++)
                C[row][col] = C_flat[row * n + col];

        // Верификация
        bool correct = true;
        double max_diff = 0.0;
        for (int row = 0; row < n && correct; row++) {
            for (int col = 0; col < n; col++) {
                double diff = abs(C[row][col] - C_seq[row][col]);
                if (diff > 1e-5) {
                    correct = false;
                    max_diff = diff;
                    break;
                }
            }
        }

        if (correct) {
            cout << "  Verification: PASSED" << endl;
        }
        else {
            cout << "  Verification: FAILED (max diff: " << max_diff << ")" << endl;
        }
        cout << endl;

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    cout << "========================================" << endl;
    cout << "All tests completed!" << endl;
    cout << "Results saved to cuda_results.txt" << endl;
    cout << "========================================" << endl;

    return 0;
}