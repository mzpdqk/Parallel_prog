#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <clocale>
#include <string>
using namespace std;
using namespace chrono;


vector<vector<double>> readMatrix(const string& filename, int& n) {
    ifstream file(filename);
    file >> n;

    vector<vector<double>> matrix(n, vector<double>(n));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            file >> matrix[i][j];

    return matrix;
}

void writeMatrix(const string& filename, const vector<vector<double>>& matrix) {
    ofstream file(filename);
    int n = matrix.size();

    file << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            file << matrix[i][j] << " ";
        file << endl;
    }
}

vector<vector<double>> multiply(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    int n
) {
    vector<vector<double>> C(n, vector<double>(n, 0));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

int main() {
    setlocale(LC_ALL, "Russian");

    ofstream dataFile("data.txt");
    if (!dataFile.is_open()) {
        cout << "Ошибка: не удалось открыть файл data.txt для записи" << endl;
        return 1;
    }

    vector<int> sizes = { 100, 200, 300, 400, 500 };

    for (int n : sizes) {
        string fileA = "matrixA_" + to_string(n) + ".txt";
        string fileB = "matrixB_" + to_string(n) + ".txt";

        int n1, n2;
        auto A = readMatrix(fileA, n1);
        auto B = readMatrix(fileB, n2);

        if (n1 != n2) {
            cout << "Ошибка: размеры матриц в файлах " << fileA << " и " << fileB << " не совпадают!" << endl;
            dataFile << "Ошибка: размеры матриц в файлах " << fileA << " и " << fileB << " не совпадают!" << endl;
            continue;
        }

        auto start = high_resolution_clock::now();
        auto C = multiply(A, B, n);
        auto end = high_resolution_clock::now();

        double time = duration<double>(end - start).count();

        string resultFile = "result_" + to_string(n) + ".txt";
        writeMatrix(resultFile, C);

        long long operations = (long long)n * n * n * 2;

        cout << "Размер: " << n << "x" << n << endl;
        cout << "Время выполнения: " << time << " сек" << endl;
        cout << "Объем задачи (операций): " << operations << endl;
        cout << "Результат записан в: " << resultFile << endl;
        cout << "----------------------------------------" << endl;
        dataFile << n << " " << time << " " << operations << endl;
    }

    dataFile.close();
    cout << "\nРезультаты сохранены в data.txt" << endl;

    return 0;
}