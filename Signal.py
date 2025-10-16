import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram 

# -----------------------------
# Параметры сигнала
T = 2.0         
tau = 0.5 * T    
alpha = 2.5
N = 5000      

t = np.linspace(0, T, N)
dt = t[1] - t[0]
fs = 1 / dt  
print(fs)

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
# 3) ДПФ (FFT) и спектр в Гц
S = np.fft.fft(s)
magS = np.abs(S)
phaseS = np.angle(S)

freq = np.fft.fftfreq(N, d=1/fs)
S_shifted = np.fft.fftshift(S)
magS_shifted = np.abs(S_shifted)
phaseS_shifted = np.angle(S_shifted)
freq_shifted = np.fft.fftshift(freq)

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(freq_shifted, magS_shifted)
plt.xlabel("f, Гц")
plt.ylabel("|S(f)|")
plt.title("Figure 2a: Амплитудный спектр, Гц")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(freq_shifted, phaseS_shifted)
plt.xlabel("f, Гц")
plt.ylabel("arg(S(f)), рад")
plt.title("Figure 2b: Фазовый спектр, Гц")
plt.grid(True)

# -----------------------------
# 4) Отображение спектра в bin и в Гц
bins = np.arange(N)
magS = np.abs(S)

fs_alt = 1000.0
freq_alt = np.fft.fftfreq(N, d=1.0/fs_alt)
freq_alt_shifted = np.fft.fftshift(freq_alt)

plt.figure(3)
plt.subplot(2,1,1)
plt.plot(bins, magS)
plt.xlabel("Bin (f)")
plt.ylabel("|S[f]|")
plt.title("Figure 3a: Амплитудный спектр в bin")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(freq_alt_shifted, magS_shifted)
plt.xlabel("f, Гц")
plt.ylabel("|S(f)|")
plt.title("Figure 3b: Амплитудный спектр в Гц")
plt.grid(True)

# =========================
# 5) Дополнение нулями (24 и 1024)
# =========================
# a) 24 нуля
s_zpad24 = np.concatenate([s, np.zeros(24)])
S_zpad24 = np.fft.fft(s_zpad24)
freq24 = np.fft.fftfreq(len(s_zpad24), d=1/fs)
S_zpad24_shifted = np.fft.fftshift(S_zpad24)
freq24_shifted = np.fft.fftshift(freq24)
bins24 = np.arange(len(s_zpad24))

plt.figure(4)
plt.subplot(2, 2, 1)
plt.plot(freq24_shifted, np.abs(S_zpad24_shifted))
plt.xlabel("f, Гц")
plt.ylabel("|S(f)|")
plt.title("Амплитудный спектр 24 нуля, Гц")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(freq24_shifted, np.angle(S_zpad24_shifted))
plt.xlabel("f, Гц")
plt.ylabel("arg(S(f)), рад")
plt.title("Фазовый спектр 24 нуля, Гц")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(bins24, np.abs(S_zpad24))
plt.xlabel("Bin (f)")
plt.ylabel("|S[f]|")
plt.title("Амплитудный спектр 24 нуля, Bin")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(bins24, np.angle(S_zpad24))
plt.xlabel("Bin (f)")
plt.ylabel("arg(S[f]), рад")
plt.title("Фазовый спектр 24 нуля, Bin")
plt.grid(True)

plt.tight_layout()

# b) 1024 нуля
s_zpad1024 = np.concatenate([s, np.zeros(1024)])
S_zpad1024 = np.fft.fft(s_zpad1024)
freq1024 = np.fft.fftfreq(len(s_zpad1024), d=1/fs)
S_zpad1024_shifted = np.fft.fftshift(S_zpad1024)
freq1024_shifted = np.fft.fftshift(freq1024)
bins1024 = np.arange(len(s_zpad1024))

plt.figure(5)
plt.subplot(2, 2, 1)
plt.plot(freq1024_shifted, np.abs(S_zpad1024_shifted))
plt.xlabel("f, Гц")
plt.ylabel("|S(f)|")
plt.title("Амплитудный спектр 1024 нуля, Гц")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(freq1024_shifted, np.angle(S_zpad1024_shifted))
plt.xlabel("f, Гц")
plt.ylabel("arg(S(f)), рад")
plt.title("Фазовый спектр 1024 нуля, Гц")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(bins1024, np.abs(S_zpad1024))
plt.xlabel("Bin (k)")
plt.ylabel("|S[k]|")
plt.title("Амплитудный спектр 1024 нуля, Bin")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(bins1024, np.angle(S_zpad1024))
plt.xlabel("Bin (k)")
plt.ylabel("arg(S[k]), рад")
plt.title("Фазовый спектр 1024 нуля, Bin")
plt.grid(True)

plt.tight_layout()

# -----------------------------
# 6) ОБПФ (IFFT)
s_rec = np.fft.ifft(S)
err = np.max(np.abs(np.real(s_rec) - s))

plt.figure(6)
plt.plot(t, np.real(s_rec), '-', label="реконструированный")
plt.plot(t, s, '--', label="исходный")
plt.xlabel("t, с")
plt.title("Figure 5: Сравнение исходного и IFFT сигнала")
plt.legend()
plt.grid(True)

# -----------------------------
# 7) Спектрограмма
nperseg = 20
noverlap = nperseg // 2

f_box, tt_box, Sxx_box = spectrogram(s, fs=fs, window='boxcar', nperseg=nperseg, noverlap=noverlap)
f_hann, tt_hann, Sxx_hann = spectrogram(s, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)

plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.pcolormesh(tt_box, f_box, 10*np.log10(Sxx_box+1e-12), shading='gouraud')
plt.title('Спектрограмма (прямоугольное окно)')
plt.xlabel('Время, с')
plt.ylabel('Частота, Гц')
plt.xlim(0, 2)
plt.colorbar(label='Амплитуда')

plt.subplot(2,1,2)
plt.pcolormesh(tt_hann, f_hann, 10*np.log10(Sxx_hann+1e-12), shading='gouraud')
plt.title('Спектрограмма (окно Ханна)')
plt.xlabel('Время, с')
plt.ylabel('Частота, Гц')
plt.xlim(0, 2)
plt.colorbar(label='Амплитуда')
plt.tight_layout()
plt.show()

