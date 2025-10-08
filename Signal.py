import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, get_window

# -----------------------------
# Параметры сигнала
T = 2.0          # полная длительность
tau = 0.5 * T    # = 1
alpha = 2.5
N = 5000         # количество отсчетов

# Временная сетка (равномерная от 0 до T, N отсчетов)
t = np.linspace(0, T, N)
dt = t[1] - t[0]
fs = 1 / dt      # частота дискретизации

# Определим сигнал
s = np.zeros_like(t)
mask = (t >= 0) & (t <= tau)
arg = ((t[mask] / tau) - 0.5)
s[mask] = np.exp(-2 * alpha * (arg**2)) * np.cos(10 * alpha * arg)

# -----------------------------
# 2) Временная форма сигнала
plt.figure(1)
plt.plot(t, s)
plt.xlabel("t, с")
plt.ylabel("s(t)")
plt.title("Figure 1: Временная форма сигнала")
plt.grid(True)

# -----------------------------
# 3) ДПФ (FFT)
S = np.fft.fft(s)
magS = np.abs(S)
phaseS = np.angle(S)
bins = np.arange(N)

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(bins, magS)
plt.xlabel("Bin")
plt.ylabel("|S[k]|")
plt.title("Figure 2a: Амплитудный спектр (в bin)")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(bins, phaseS)
plt.xlabel("Bin")
plt.ylabel("arg(S[k]), рад")
plt.title("Figure 2b: Фазовый спектр (в bin)")
plt.grid(True)

# -----------------------------
# 4) Отображение спектра в bin и в Гц
fs_alt = 1000  # условие: 1000 отсчетов = 1 с
freq_actual = np.arange(N) * (fs / N)
freq_alt = np.arange(N) * (fs_alt / N)

plt.figure(3)
plt.subplot(2, 1, 1)
plt.plot(freq_actual, magS)
plt.xlabel("f, Гц (реальный fs)")
plt.ylabel("|S(f)|")
plt.title("Figure 3a: Амплитудный спектр (fs=N/T)")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(freq_alt, magS)
plt.xlabel("f, Гц (fs=1000)")
plt.ylabel("|S(f)|")
plt.title("Figure 3b: Амплитудный спектр (fs=1000)")
plt.grid(True)

# -----------------------------
# 5) Дополнение нулями
# a) 24 нуля
s_zpad24 = np.concatenate([s, np.zeros(24)])
S_zpad24 = np.fft.fft(s_zpad24)
bins24 = np.arange(len(s_zpad24))

# b) 1024 нуля
s_zpad1024 = np.concatenate([s, np.zeros(1024)])
S_zpad1024 = np.fft.fft(s_zpad1024)
bins1024 = np.arange(len(s_zpad1024))

plt.figure(4)
plt.subplot(2, 1, 1)
plt.plot(bins24, np.abs(S_zpad24))
plt.xlabel("Bin")
plt.ylabel("|S[k]|")
plt.title("Амплитудный спектр (24 нуля)")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(bins1024, np.abs(S_zpad1024))
plt.xlabel("Bin")
plt.ylabel("|S[k]|")
plt.title("Амплитудный спектр (1024 нуля)")
plt.grid(True)

# -----------------------------
# 6) ОБПФ (IFFT)
s_rec = np.fft.ifft(S)
err = np.max(np.abs(np.real(s_rec) - s))
print(f"Макс. ошибка реконструкции через IFFT: {err:.2e}")

plt.figure(5)
plt.plot(t, np.real(s_rec), '-', label="реконструированный")
plt.plot(t, s, '--', label="исходный")
plt.xlabel("t, с")
plt.title("Figure 5: Сравнение исходного и IFFT сигнала")
plt.legend()
plt.grid(True)

# -----------------------------
# 7) Спектрограмма
win_len = 20
noverlap = win_len // 2
nfft = 128

# a) прямоугольное окно
f_rect, t_rect, Sxx_rect = spectrogram(s, fs=fs, window='boxcar',
                                       nperseg=win_len, noverlap=noverlap, nfft=nfft)
# b) окно Ханна
f_hann, t_hann, Sxx_hann = spectrogram(s, fs=fs, window=get_window('hann', win_len),
                                       nperseg=win_len, noverlap=noverlap, nfft=nfft)

plt.figure(6)
plt.subplot(2, 1, 1)
plt.pcolormesh(t_rect, f_rect, 10 * np.log10(Sxx_rect), shading='gouraud')
plt.title("Спектрограмма (прямоугольное окно)")
plt.ylabel("Частота [Гц]")
plt.xlabel("Время [с]")
plt.colorbar(label="дБ")

plt.subplot(2, 1, 2)
plt.pcolormesh(t_hann, f_hann, 10 * np.log10(Sxx_hann), shading='gouraud')
plt.title("Спектрограмма (окно Ханна)")
plt.ylabel("Частота [Гц]")
plt.xlabel("Время [с]")
plt.colorbar(label="дБ")

plt.tight_layout()
plt.show()
