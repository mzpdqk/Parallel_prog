import numpy as np
import matplotlib.pyplot as plt
import os

# Переход в директорию скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Списки для хранения данных
sizes = []
times_1 = []
times_2 = []
times_4 = []

# Чтение результатов эксперимента
with open("experiment_results.txt", 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            size = int(parts[0])
            threads = int(parts[1])
            time_val = float(parts[2])
            
            if size not in sizes:
                sizes.append(size)
            
            if threads == 1:
                times_1.append(time_val)
            elif threads == 2:
                times_2.append(time_val)
            elif threads == 4:
                times_4.append(time_val)

# Сортировка по размеру
sorted_indices = sorted(range(len(sizes)), key=lambda i: sizes[i])
sizes = [sizes[i] for i in sorted_indices]
times_1 = [times_1[i] for i in sorted_indices]
times_2 = [times_2[i] for i in sorted_indices]
times_4 = [times_4[i] for i in sorted_indices]

# Вывод данных
print("=" * 60)
print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ")
print("=" * 60)
print(f"Размеры: {sizes}")
print(f"1 поток:  {[round(t, 4) for t in times_1]}")
print(f"2 потока: {[round(t, 4) for t in times_2]}")
print(f"4 потока: {[round(t, 4) for t in times_4]}")

# Проверка корректности для максимального размера
print("\n" + "=" * 60)
print("ПРОВЕРКА ПРАВИЛЬНОСТИ УМНОЖЕНИЯ (размер 1600x1600)")
print("=" * 60)

try:
    # Чтение исходных матриц
    with open("matrix_a.txt", 'r') as f:
        n = int(f.readline().strip())
    
    A = np.loadtxt("matrix_a.txt", skiprows=1)
    B = np.loadtxt("matrix_b.txt", skiprows=1)
    
    # Эталонное умножение
    print("Вычисление эталонного результата...")
    C_correct = np.dot(A, B)
    
    # Проверка результатов для разного числа потоков
    for t in [1, 2, 4]:
        fname = f"result_{t}.txt"
        if os.path.exists(fname):
            C_res = np.loadtxt(fname, skiprows=1)
            if np.allclose(C_res, C_correct, rtol=1e-5, atol=1e-6):
                print(f"  {t} поток(а): OK ✓")
            else:
                max_diff = np.max(np.abs(C_res - C_correct))
                print(f"  {t} поток(а): ОШИБКА ✗ (макс. разница: {max_diff:.2e})")
        else:
            print(f"  {t} поток(а): файл не найден")
            
except Exception as e:
    print(f"Ошибка при проверке: {e}")

# Построение графика
print("\nПостроение графика...")

plt.figure(figsize=(12, 7))
plt.style.use('seaborn-v0_8-darkgrid')

plt.plot(sizes, times_1, 'b-o', label='1 поток', linewidth=2, markersize=8, markerfacecolor='blue')
plt.plot(sizes, times_2, 'g-s', label='2 потока', linewidth=2, markersize=8, markerfacecolor='green')
plt.plot(sizes, times_4, 'r-^', label='4 потока', linewidth=2, markersize=8, markerfacecolor='red')

plt.xlabel('Размер матрицы N', fontsize=14, fontweight='bold')
plt.ylabel('Время выполнения (секунды)', fontsize=14, fontweight='bold')
plt.title('Сравнение времени умножения матриц (OpenMP)\nAMD A10-6800K (4 ядра)', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12, loc='upper left')

# Добавление подписей
for i, size in enumerate(sizes):
    if size >= 500:
        plt.annotate(f'{times_1[i]:.2f}s', (size, times_1[i]), 
                    fontsize=9, alpha=0.7, ha='left', va='bottom')
        plt.annotate(f'{times_4[i]:.2f}s', (size, times_4[i]), 
                    fontsize=9, alpha=0.7, ha='left', va='top')

plt.tight_layout()
plt.savefig('graph.png', dpi=300, bbox_inches='tight')
print("График сохранён как graph.png")

# Вывод таблицы результатов
print("\n" + "=" * 80)
print("ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 80)
print(f"{'N':<8} {'1 поток (с)':<15} {'2 потока (с)':<15} {'4 потока (с)':<15} {'Ускорение (4x)':<15} {'Эффективность':<15}")
print("-" * 80)

for i in range(len(sizes)):
    speedup = times_1[i] / times_4[i] if times_4[i] > 0 else 0
    efficiency = speedup / 4.0 * 100
    print(f"{sizes[i]:<8} {times_1[i]:<15.4f} {times_2[i]:<15.4f} {times_4[i]:<15.4f} {speedup:<15.2f} {efficiency:<15.1f}%")

print("-" * 80)

# Анализ масштабируемости
print("\n" + "=" * 60)
print("АНАЛИЗ МАСШТАБИРУЕМОСТИ")
print("=" * 60)

for i in range(len(sizes)):
    if sizes[i] >= 800:
        speedup_2x = times_1[i] / times_2[i]
        speedup_4x = times_1[i] / times_4[i]
        print(f"Размер {sizes[i]}x{sizes[i]}: Ускорение 2x = {speedup_2x:.2f}, Ускорение 4x = {speedup_4x:.2f}")

print("\nГотово!")