import numpy as np

def generate_matrices_simple():
    """
    Простая генерация матриц для размеров 100, 200, 300, 400, 500
    """
    sizes = [100, 200, 300, 400, 500]
    
    for size in sizes:
        A = np.random.randint(-10, 10, size=(size, size))
        with open(f'matrixA_{size}.txt', 'w') as f:
            f.write(f"{size}\n")
            for row in A:
                f.write(' '.join(map(str, row)) + '\n')
        
        B = np.random.randint(-10, 10, size=(size, size))
        with open(f'matrixB_{size}.txt', 'w') as f:
            f.write(f"{size}\n")
            for row in B:
                f.write(' '.join(map(str, row)) + '\n')
        
        print(f"Сгенерированы матрицы {size}x{size}")

if __name__ == "__main__":
    generate_matrices_simple()
    print("Готово!")