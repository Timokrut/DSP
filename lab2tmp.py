import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Параметры
# ==============================
T = 4
l = T/2
N = 5  # количество членов ряда
x = np.linspace(-1, 3, 5000)  # один период

# ==============================
# Исходная функция
# ==============================
def f(x):
    x = np.array(x)
    y = np.zeros_like(x)
    # участок [-1, 0]
    mask1 = (x >= -1) & (x <= 0)
    y[mask1] = x[mask1] + 1
    # участок [0, 3]
    mask2 = (x >= 0) & (x <= 3)
    y[mask2] = 1 - x[mask2] / 3
    return y

# ==============================
# РУЧНЫЕ КОЭФФИЦИЕНТЫ
# ==============================
def a_n_manual(n):
    a = np.pi * n / 2
    return (8 / (3 * np.pi**2 * n**2)) * (1 - np.cos(a)**3)

def b_n_manual(n):
    a = np.pi * n / 2
    return (8 / (3 * np.pi**2 * n**2)) * (np.sin(a)**3)

a0_manual = 1  # найдено аналитически

# Строим ряд РУЧНО
S_manual = (a0_manual / 2) * np.ones_like(x)

for n in range(1, N + 1):
    S_manual += a_n_manual(n) * np.cos(np.pi * n * x / 2) \
              + b_n_manual(n) * np.sin(np.pi * n * x / 2)

# ==============================
# АВТОМАТИЧЕСКОЕ ЧИСЛЕННОЕ ВЫЧИСЛЕНИЕ КОЭФФИЦИЕНТОВ
# ==============================
def compute_fourier_coeffs(n_max):
    xs = np.linspace(-1, 3, 20001)
    fx = f(xs)

    a0 = (1/l) * np.trapezoid(fx, xs)

    a_coeffs = []
    b_coeffs = []

    for n in range(1, n_max + 1):
        w = np.pi * n / 2
        a_n = (1/l) * np.trapezoid(fx * np.cos(w * xs), xs)
        b_n = (1/l) * np.trapezoid(fx * np.sin(w * xs), xs)
        a_coeffs.append(a_n)
        b_coeffs.append(b_n)

    return a0, a_coeffs, b_coeffs

print("Считаем автоматические коэффициенты...")
a0_auto, a_auto, b_auto = compute_fourier_coeffs(N)

# Строим ряд с автоматическими коэффициентами
S_auto = (a0_auto / 2) * np.ones_like(x)

for n in range(1, N + 1):
    w = np.pi * n / 2
    S_auto += a_auto[n-1] * np.cos(w * x) + b_auto[n-1] * np.sin(w * x)

# ==============================
# ГРАФИКИ
# ==============================
plt.figure(figsize=(10, 6))

plt.plot(x, f(x), label="Исходная f(x)", color="black", linewidth=3)
plt.plot(x, S_manual, label=f"Ручные коэф. (N={N})", linestyle="--", color="red", linewidth=2)
plt.plot(x, S_auto, label=f"Авто коэф. (N={N})", linestyle="-.", color="blue", linewidth=2)

plt.grid(True)
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Сравнение ряда Фурье для треугольного сигнала")
plt.show()