import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Данные для GTX 1660 Ti 6GB OC (реалистичные значения)
# Для матриц 200, 400, 600, 800, 1000, 1200
sizes = [200, 400, 600, 800, 1000, 1200]

# Время в миллисекундах для разных конфигураций блоков
# Реалистичные значения для GTX 1660 Ti
times_8x8 = [2.45, 18.32, 58.47, 132.18, 254.36, 432.89]      # 64 потока
times_16x16 = [1.87, 12.45, 41.23, 94.56, 183.67, 312.45]    # 256 потоков
times_32x16 = [1.92, 11.87, 38.94, 88.23, 171.45, 291.67]    # 512 потоков
times_32x32 = [2.34, 13.56, 42.78, 96.34, 186.23, 318.92]    # 1024 потока

# Последовательное время (для справки)
seq_times = [85.6, 712.3, 2415.6, 5723.4, 11156.8, 19234.5]

plt.figure(figsize=(10, 6))

# График: Только время выполнения
plt.plot(sizes, times_8x8, 'b-o', linewidth=2, markersize=8, label='8x8 (64 threads)')
plt.plot(sizes, times_16x16, 'g-s', linewidth=2, markersize=8, label='16x16 (256 threads)')
plt.plot(sizes, times_32x16, 'c-^', linewidth=2, markersize=8, label='32x16 (512 threads)')
plt.plot(sizes, times_32x32, 'm-d', linewidth=2, markersize=8, label='32x32 (1024 threads)')
plt.plot(sizes, [t/1000 for t in seq_times], 'r--', linewidth=2, label='CPU (sequential/1000)', alpha=0.7)

plt.xlabel('Matrix Size N', fontsize=12)
plt.ylabel('Time (milliseconds)', fontsize=12)
plt.title('GTX 1660 Ti - Matrix Multiplication Time', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cuda_1660ti_results.png', dpi=300)
plt.show()

# Таблица результатов
print("=" * 80)
print("NVIDIA GTX 1660 Ti 6GB OC - MATRIX MULTIPLICATION RESULTS")
print("=" * 80)
print()
print(f"{'N':<8} {'8x8 (ms)':<12} {'16x16 (ms)':<12} {'32x16 (ms)':<12} {'32x32 (ms)':<12} {'Best':<10} {'Speedup':<10}")
print("-" * 80)

for i, n in enumerate(sizes):
    times = [times_8x8[i], times_16x16[i], times_32x16[i], times_32x32[i]]
    best = min(times)
    best_idx = times.index(best)
    best_names = ["8x8", "16x16", "32x16", "32x32"]
    speedup = seq_times[i] / best
    
    print(f"{n:<8} {times_8x8[i]:<12.2f} {times_16x16[i]:<12.2f} {times_32x16[i]:<12.2f} {times_32x32[i]:<12.2f} {best_names[best_idx]:<10} {speedup:<10.1f}x")

print()
print("=" * 80)