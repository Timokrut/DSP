import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, fftfreq
from scipy.signal import spectrogram

class Signal:
    def __init__(self, T, r, a, N=5000, fs=1000):
        self.T = T   # длительность сигнала   
        self.r = r   # tau  
        self.a = a   # alpha 
        self.N = N   # число отсчетов    
        self.fs = fs # частота дискретизации 

        # Создание сигнала
        self.t = np.linspace(0, T, N)
        self.s = np.zeros(N)
        mask = (self.t > 0) & (self.t <= self.r)
        print(mask)
        t_valid = self.t[mask]
        print(t_valid)

        self.s[mask] = np.exp(-2 * (a * (1/t_valid - 0.5))**2) * np.cos(10 * a * (1/t_valid - 0.5))

    def task2(self):
        S = fft(self.s)
        S_shift = fftshift(S)
        freqs = fftshift(fftfreq(self.N, d=1.0/self.fs))
        amp = np.abs(S_shift) / self.N
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

    def task1(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.t, self.s, 'b-', linewidth=1.5)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Время t (сек)')
        plt.ylabel('Амплитуда s(t)')
        plt.title('Сигнал s(t) = exp(-2(a(1/t - 0.5))²)cos(10a(1/t - 0.5))')
        plt.xlim(0, self.T)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    T = 2
    r = 0.5 * T 
    a = 2.5
    s = Signal(T, r, a)
    s.task2()
