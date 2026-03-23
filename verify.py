import numpy as np

def read_matrix(filename):
    with open(filename) as f:
        n = int(f.readline())
        data = []
        for _ in range(n):
            data.append(list(map(float, f.readline().split())))
    return np.array(data)

for n in [100, 200, 300, 400, 500]:
    A = read_matrix(f"matrixA_{n}.txt")
    B = read_matrix(f"matrixB_{n}.txt")
    C_cpp = read_matrix(f"result_{n}.txt")
    
    C_py = A @ B
    
    if np.allclose(C_cpp, C_py):
        print(f"n={n}: Результат верен")
    else:
        print(f"n={n}: Ошибка в вычислениях")