import matplotlib.pyplot as plt

# Данные (время в секундах)
sizes = [100, 200, 400, 500, 1000, 2000, 4000]

# Время для разного количества ядер
time_1 = [0.018, 0.067, 0.414, 0.817, 6.447, 54.978, 426]
time_2 = [0.011, 0.041, 0.214, 0.415, 3.233, 26.463, 248.83]
time_4 = [0.007, 0.027, 0.113, 0.219, 1.637, 14.635, 128.19]
time_8 = [0.004, 0.017, 0.061, 0.118, 0.901, 7.526, 64.63]
time_10 = [0.004, 0.015, 0.050, 0.096, 0.733, 6.031, 52.10]
time_20 = [0.004, 0.011, 0.030, 0.052, 0.403, 3.121, 26.49]
time_40 = [0.002, 0.006, 0.015, 0.032, 0.221, 1.608, 13.79]

# График
plt.figure(figsize=(12, 6))

plt.plot(sizes, time_1, 'o-', label='1 ядро', linewidth=2, markersize=8)
plt.plot(sizes, time_2, 'o-', label='2 ядра', linewidth=2, markersize=8)
plt.plot(sizes, time_4, 'o-', label='4 ядра', linewidth=2, markersize=8)
plt.plot(sizes, time_8, 'o-', label='8 ядер', linewidth=2, markersize=8)
plt.plot(sizes, time_10, 'o-', label='10 ядер', linewidth=2, markersize=8)
plt.plot(sizes, time_20, 'o-', label='20 ядер', linewidth=2, markersize=8)
plt.plot(sizes, time_40, 'o-', label='40 ядер', linewidth=2, markersize=8)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Размер матрицы', fontsize=12)
plt.ylabel('Время выполнения (секунды)', fontsize=12)
plt.title('Зависимость времени выполнения от размера матрицы', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.savefig('graphic.jpg', dpi=150, bbox_inches='tight')
plt.show()

print("График сохранен как graphic.jpg")
