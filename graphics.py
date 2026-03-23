import matplotlib.pyplot as plt
import numpy as np

sizes = []
times = []
operations = []

with open('data.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            try:
                n = int(parts[0])
                time = float(parts[1])
                ops = int(parts[2])
                sizes.append(n)
                times.append(time)
                operations.append(ops)
            except ValueError:
                continue

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(sizes, times, 'bo-', linewidth=2, markersize=8, label='Время выполнения')
ax1.set_xlabel('Размер матрицы (n × n)', fontsize=12)
ax1.set_ylabel('Время (секунды)', fontsize=12)
ax1.set_title('Зависимость времени выполнения от размера матрицы', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend()

for x, y in zip(sizes, times):
    if y < 0.0001:
        ax1.annotate(f'{y:.4e}', (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9)
    else:
        ax1.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9)

ax2.plot(sizes, operations, 'rs-', linewidth=2, markersize=8, label='Количество операций')
ax2.set_xlabel('Размер матрицы (n × n)', fontsize=12)
ax2.set_ylabel('Количество операций', fontsize=12)
ax2.set_title('Зависимость количества операций от размера матрицы', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend()

for x, y in zip(sizes, operations):
    if y >= 1000000:
        ax2.annotate(f'{y:.4e}', (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9)
    else:
        ax2.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9)

ax2.ticklabel_format(style='scientific', axis='y', scilimits=(6, 6))

plt.tight_layout()
plt.savefig('performance_plots.png', dpi=300, bbox_inches='tight')
plt.show()