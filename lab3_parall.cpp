#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <random>
#include <mpi.h>
#include <string>
#include <algorithm>
#include <cmath>

using namespace std;

void generateMatrix(const string& filename, int n) {
    ofstream file(filename);
    mt19937 gen(42); 
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

void writeResult(const string& filename, const vector<vector<double>>& C, int n) {
    ofstream file(filename);
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

void multiplyBlock(const vector<double>& rowA, const vector<vector<double>>& B,
    vector<double>& rowC, int n, int blockSize) {
    fill(rowC.begin(), rowC.end(), 0.0);

    for (int jj = 0; jj < n; jj += blockSize) {
        int jmax = min(jj + blockSize, n);
        for (int kk = 0; kk < n; kk += blockSize) {
            int kmax = min(kk + blockSize, n);
            for (int j = jj; j < jmax; j++) {
                double sum = 0.0;
                for (int k = kk; k < kmax; k++) {
                    sum += rowA[k] * B[k][j];
                }
                rowC[j] += sum;
            }
        }
    }
}

void runTest(int n, int num_procs, int rank, MPI_Comm comm, int warmup) {
    if (rank == 0) {
        cout << "  N=" << n << " [" << num_procs << "p] ... " << flush;

        static vector<int> generatedSizes;
        if (find(generatedSizes.begin(), generatedSizes.end(), n) == generatedSizes.end()) {
            generateMatrix("mat_a.txt", n);
            generateMatrix("mat_b.txt", n);
            generatedSizes.push_back(n);
        }
    }

    MPI_Barrier(comm);

    int matrix_size;
    vector<vector<double>> A, B;

    if (rank == 0) {
        int dummy;
        A = readMatrix("mat_a.txt", matrix_size);
        B = readMatrix("mat_b.txt", dummy);
    }

    MPI_Bcast(&matrix_size, 1, MPI_INT, 0, comm);
    if (matrix_size != n) {
        if (rank == 0) cerr << "Ошибка размера матрицы!" << endl;
        MPI_Abort(comm, 1);
    }

    if (rank != 0) {
        B.resize(n, vector<double>(n));
    }
    for (int row = 0; row < n; row++) {
        MPI_Bcast(B[row].data(), n, MPI_DOUBLE, 0, comm);
    }

    int rows_per_proc = n / num_procs;
    int remainder = n % num_procs;
    int start_row = rank * rows_per_proc + min(rank, remainder);
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    vector<int> send_counts(num_procs, 0);
    vector<int> displs(num_procs, 0);

    if (rank == 0) {
        int offset = 0;
        for (int p = 0; p < num_procs; p++) {
            send_counts[p] = (rows_per_proc + (p < remainder ? 1 : 0)) * n;
            displs[p] = offset;
            offset += send_counts[p];
        }
    }

    vector<double> flat_A;
    if (rank == 0) {
        flat_A.resize(n * n);
        for (int i = 0; i < n; i++)
            copy(A[i].begin(), A[i].end(), flat_A.begin() + i * n);
    }

    vector<double> flat_local(local_rows * n);
    MPI_Scatterv(rank == 0 ? flat_A.data() : nullptr,
        send_counts.data(), displs.data(), MPI_DOUBLE,
        flat_local.data(), local_rows * n, MPI_DOUBLE,
        0, comm);

    vector<vector<double>> local_A(local_rows, vector<double>(n));
    for (int i = 0; i < local_rows; i++)
        copy(flat_local.begin() + i * n, flat_local.begin() + (i + 1) * n, local_A[i].begin());

    MPI_Barrier(comm);

    if (warmup == 1) {
        vector<vector<double>> dummy_local(local_rows, vector<double>(n, 0.0));
        for (int i = 0; i < min(local_rows, 10); i++) {
            for (int j = 0; j < min(n, 100); j++) {
                for (int k = 0; k < min(n, 100); k++) {
                    dummy_local[i][j] += local_A[i][k] * B[k][j];
                }
            }
        }
    }

    MPI_Barrier(comm);
    double start_time = MPI_Wtime();

    vector<vector<double>> local_C(local_rows, vector<double>(n, 0.0));
    for (int i = 0; i < local_rows; i++) {
        for (int k = 0; k < n; k++) {
            double aik = local_A[i][k];
            for (int j = 0; j < n; j++) {
                local_C[i][j] += aik * B[k][j];
            }
        }
    }

    double end_time = MPI_Wtime();
    double parallel_time = end_time - start_time;

    vector<double> flat_result;
    if (rank == 0) {
        flat_result.resize(n * n);
    }

    vector<double> flat_local_C(local_rows * n);
    for (int i = 0; i < local_rows; i++)
        copy(local_C[i].begin(), local_C[i].end(), flat_local_C.begin() + i * n);

    MPI_Gatherv(flat_local_C.data(), local_rows * n, MPI_DOUBLE,
        rank == 0 ? flat_result.data() : nullptr,
        send_counts.data(), displs.data(), MPI_DOUBLE,
        0, comm);

    if (rank == 0) {
        vector<vector<double>> C(n, vector<double>(n));
        for (int i = 0; i < n; i++)
            copy(flat_result.begin() + i * n, flat_result.begin() + (i + 1) * n, C[i].begin());

        string filename = "res_" + to_string(n) + "_" + to_string(num_procs) + "p.txt";
        writeResult(filename, C, n);

        long long flops = 2LL * n * n * n;
        double gflops = flops / (parallel_time * 1e9);

        ofstream results("benchmark.csv", ios::app);
        results << n << "," << num_procs << "," << parallel_time << "," << flops << "," << gflops << "\n";
        results.close();

        cout << parallel_time << " s (" << gflops << " GFLOPS)" << endl;
    }
}
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> sizes = { 100, 200, 300, 400, 600, 800, 1000, 1200 };

    if (rank == 0) {
        cout << "==================================================" << endl;
        cout << "MPI Matrix Multiplication Benchmark" << endl;
        cout << "CPU: AMD A10-6800K (4 cores, no HT)" << endl;
        cout << "==================================================" << endl;
        cout << "Testing with " << size << " process(es)" << endl;

        ofstream results("benchmark.csv", ios::app);
        results << "N,Processes,Time(s),Flops,GFLOPS\n";
        results.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Прогрев
    runTest(100, size, rank, MPI_COMM_WORLD, 1);

    // Основные тесты
    for (int n : sizes) {
        if (n == 100) continue;
        runTest(n, size, rank, MPI_COMM_WORLD, 0);
    }

    if (rank == 0) {
        cout << "\n==================================================" << endl;
        cout << "Benchmark complete!" << endl;
        cout << "Results saved to benchmark.csv" << endl;
        cout << "==================================================" << endl;
    }

    MPI_Finalize();
    return 0;
}