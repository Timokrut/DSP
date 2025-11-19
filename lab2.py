import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Исходный сигнал
# -------------------------------

def f(x):
    return np.where(x < 0, x + 1, 1 - x/3)

# -------------------------------
# 2. Аналитические коэффициенты
# -------------------------------

def a_n(n):
    a = (np.pi * n) / 2
    if n == 0:
        return 1
    return 0.5 * ((1 - np.cos(a)) / (a**2) + (1 - np.cos(3*a)) / (3 * a**2))


def b_n(n):
    a = (np.pi * n) / 2
    return 0.5 * ((np.sin(a) / (a**2)) - (np.sin(3*a) / (3 * a**2)))

# -------------------------------
# 3. Ряд Фурье c аналитическими коэффициентами
# -------------------------------

def fourier_analytic(x, N):
    s = a_n(0) / 2
    for n in range(1, N+1):
        a = a_n(n)
        b = b_n(n)
        s += a * np.cos(np.pi * n * x / 2) + b * np.sin(np.pi * n * x / 2)
    return s

# -------------------------------
# 4. Численные коэффициенты (интегралы)
# -------------------------------

def numeric_coeffs(N):
    X = np.linspace(-1, 3, 5000)
    Y = f(X)

    a = []
    b = []

    for n in range(N+1):
        cos_term = np.cos(np.pi * n * X / 2)
        sin_term = np.sin(np.pi * n * X / 2)

        a_n_num = (1/2) * np.trapz(Y * cos_term, X)
        b_n_num = (1/2) * np.trapz(Y * sin_term, X)

        a.append(a_n_num)
        b.append(b_n_num)

    return np.array(a), np.array(b)

def fourier_numeric(x, a, b):
    s = a[0]/2
    for n in range(1, len(a)):
        s += a[n] * np.cos(np.pi * n * x / 2) + b[n] * np.sin(np.pi * n * x / 2)
    return s

# -------------------------------
# Plot
# -------------------------------

N = 50  # число гармоник

x = np.linspace(-1, 3, 2000)
y = f(x)

# аналитический ряд
y_an = fourier_analytic(x, N)

# численный ряд
a_num, b_num = numeric_coeffs(N)
y_num = fourier_numeric(x, a_num, b_num)

# -------------------------------
# Графики
# -------------------------------

plt.figure(figsize=(12, 8))

plt.plot(x, y, label="Исходный сигнал", linewidth=2)
plt.plot(x, y_an, label="Ряд Фурье (аналитические коэффициенты)")
plt.plot(x, y_num, label="Ряд Фурье (численные коэффициенты)", linestyle="--")

plt.grid(True)
plt.legend()
plt.title(f"Фурье-анализ (N={N})")
plt.show()
