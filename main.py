import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, fftfreq
from scipy.signal import spectrogram

# -------------------- Общие параметры --------------------
N = 5000           # число отсчетов
fs = 1000.0        # частота дискретизации
t = np.linspace(0, N/fs, N, endpoint=False)  # время 0 ... 5 с
T = 2.0            # длительность
r = 0.5 * T        # по условию варианта
alpha = 2.5

# -------------------- Определение сигнала --------------------
def triangular_pulse(t, T, r):
    s = np.zeros_like(t)
    mask = (t >= 0) & (t <= r)   # треугольник только на [0, r]
    s[mask] = 1.0 - np.abs((2*t[mask] - r) / r)
    return s

s_t = triangular_pulse(t, T, r)

# -------------------- Figure 1: временная область --------------------
plt.figure(figsize=(10,4))
plt.plot(t, s_t, linewidth=1)
plt.title('Figure 1: Треугольный импульс s(t)')
plt.xlabel('Время, с')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.xlim(0, 2.2)
plt.tight_layout()
plt.show()

# -------------------- Figure 2: Амплитудный и фазовый спектр --------------------
S = fft(s_t)
S_shift = fftshift(S)
freqs = fftshift(fftfreq(N, d=1.0/fs))
amp = np.abs(S_shift) / N
phase = np.angle(S_shift)

plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(freqs, amp, linewidth=1)
plt.title('Figure 2a: Амплитудный спектр |S(f)|')
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда (норм.)')
plt.grid(True)
plt.xlim(-100, 100)

plt.subplot(2,1,2)
plt.plot(freqs, phase, linewidth=1)
plt.title('Figure 2b: Фазовый спектр arg(S(f))')
plt.xlabel('Частота, Гц')
plt.ylabel('Фаза, рад')
plt.grid(True)
plt.xlim(-100, 100)
plt.tight_layout()
plt.show()

# -------------------- Figure 3: спектр в бинах и в Гц --------------------
# S = fft(s_t)  # БПФ без сдвига
# amp_bins = np.abs(S) / N  # Амплитуды для несдвинутого БПФ
# bins = np.arange(N)       # Ось бинов (0, 1, 2, ..., N-1)

# # Для графика в Гц используем несдвинутые частоты и несдвинутые амплитуды
# freqs = fftfreq(N, d=1.0/fs)  # Несдвинутые частоты
# amp_hz = np.abs(S) / N        # Амплитуды для несдвинутых частот

# plt.figure(figsize=(12, 4))

# # График 3a: Амплитудный спектр в бинах
# plt.subplot(1, 2, 1)
# plt.plot(bins, amp_bins, linewidth=1)
# plt.title('Figure 3a: Амплитудный спектр (бины)')
# plt.xlabel('Бины')
# plt.ylabel('Амплитуда (норм.)')
# plt.grid(True)
# plt.xlim(0, 500)

# # График 3b: Амплитудный спектр в Гц (только положительные частоты)
# plt.subplot(1, 2, 2)
# # Берем только положительные частоты
# positive_mask = freqs >= 0
# plt.plot(freqs[positive_mask], amp_hz[positive_mask], linewidth=1)
# plt.title('Figure 3b: Амплитудный спектр (Гц)')
# plt.xlabel('Частота, Гц')
# plt.ylabel('Амплитуда (норм.)')
# plt.grid(True)
# plt.xlim(0, 200)
# plt.tight_layout()
# plt.show()

# amp_bins = np.abs(S) / N
# bins = np.arange(N)
# freqs = fftfreq(N, d=1.0/fs)
# amp_hz = np.abs(S_shift) / N

amp_bins = np.abs(S) / N
bins = np.arange(N)

# СДВИГАЕМ и частоты, и амплитуды
freqs_shift = fftshift(fftfreq(N, d=1.0/fs))
amp_hz_shift = np.abs(fftshift(S)) / N

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(bins, amp_bins, linewidth=1)
plt.title('Figure 3a: Амплитудный спектр (бины)')
plt.xlabel('Бины')
plt.ylabel('Амплитуда (норм.)')
plt.grid(True)
plt.xlim(0, 200)

plt.subplot(1,2,2)
plt.plot(freqs_shift, amp_hz_shift, linewidth=1)
plt.title('Figure 3b: Амплитудный спектр (Гц)')
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда (норм.)')
plt.grid(True)
plt.xlim(0, 200)
plt.tight_layout()
plt.show()

# # -------------------- Figure 4: Zero-padding --------------------
# a) +24 нуля
s_24 = np.concatenate([s_t, np.zeros(24)])
N24 = len(s_24)
S24 = fftshift(fft(s_24))
f24 = fftshift(fftfreq(N24, d=1.0/fs))
amp24 = np.abs(S24) / N
phase24 = np.angle(S24)

# б) +1024 нулей
s_1024 = np.concatenate([s_t, np.zeros(1024)])
N1024 = len(s_1024)
S1024 = fftshift(fft(s_1024))
f1024 = fftshift(fftfreq(N1024, d=1.0/fs))
amp1024 = np.abs(S1024) / (N1024)
phase1024 = np.angle(S1024)

plt.figure(figsize=(12,8))

# Амплитудный спектр (24 нуля)
plt.subplot(2,2,1)
plt.plot(f24, amp24, linewidth=1)
plt.title('Figure 4a: Амплитудный спектр (24 нуля)')
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда (норм.)')
plt.grid(True)
plt.xlim(0, 100)

# Фазовый спектр (24 нуля)
plt.subplot(2,2,2)
plt.plot(f24, phase24, linewidth=1)
plt.title('Figure 4b: Фазовый спектр (24 нуля)')
plt.xlabel('Частота, Гц')
plt.ylabel('Фаза, рад')
plt.grid(True)
plt.xlim(0, 200)

# Амплитудный спектр (1024 нуля)
plt.subplot(2,2,3)
plt.plot(f1024, amp1024, linewidth=1)
plt.title('Figure 4c: Амплитудный спектр (1024 нуля)')
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда (норм.)')
plt.grid(True)
plt.xlim(0, 100)

# Фазовый спектр (1024 нуля)
plt.subplot(2,2,4)
plt.plot(f1024, phase1024, linewidth=1)
plt.title('Figure 4d: Фазовый спектр (1024 нуля)')
plt.xlabel('Частота, Гц')
plt.ylabel('Фаза, рад')
plt.grid(True)
plt.xlim(0, 200)

plt.tight_layout()
plt.show()

# -------------------- Figure 5: восстановление сигнала --------------------

s_rec = ifft(S).real

# Графики
plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(t, s_t, label="Исходный")
plt.xlim(0, 2.2)
plt.grid(True)
plt.legend()

plt.subplot(2,1,2)
plt.plot(t, s_rec, label="Восстановленный", linestyle="--")
plt.xlim(0, 2.2)
plt.grid(True)
plt.legend()

plt.show()

print("Макс. ошибка:", np.max(np.abs(s_t - s_rec)))

# -------------------- Figure 6: спектрограммы --------------------
nperseg = 20
noverlap = nperseg // 2

f_box, tt_box, Sxx_box = spectrogram(s_t, fs=fs, window='boxcar', nperseg=nperseg, noverlap=noverlap)
f_hann, tt_hann, Sxx_hann = spectrogram(s_t, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)

plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.pcolormesh(tt_box, f_box, 10*np.log10(Sxx_box+1e-12), shading='gouraud')
plt.title('Figure 6a: Спектрограмма (boxcar)')
plt.ylabel('Частота, Гц')
plt.ylim(0, 200)
plt.colorbar(label='Мощность, dB')

plt.subplot(2,1,2)
plt.pcolormesh(tt_hann, f_hann, 10*np.log10(Sxx_hann+1e-12), shading='gouraud')
plt.title('Figure 6b: Спектрограмма (hann)')
plt.xlabel('Время, с')
plt.ylabel('Частота, Гц')
plt.ylim(0, 200)
plt.colorbar(label='Мощность, dB')
plt.tight_layout()
plt.show()

