import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import tkinter as tk
from tkinter import filedialog

# 打开文件选择对话框来选择wav文件
root = tk.Tk()
root.withdraw()  # 隐藏主窗口
filename = filedialog.askopenfilename(title="选择一个WAV文件", filetypes=(("WAV files", "*.wav"), ("All files", "*.*")))

# 如果没有选择文件，则退出
if not filename:
    print("未选择文件，程序退出。")
    exit()

# 读取选定的音频文件
audio_data, fs = sf.read(filename)

# 归一化音频信号，使其幅度范围在[-1, 1]
audio_data = audio_data / np.max(np.abs(audio_data))
print(f"音频文件的最大幅度: {np.max(np.abs(audio_data))}")

# 创建Butterworth带通滤波器
lowcut = 200.0
highcut = 2000.0
order = 3
b, a = signal.butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')

# 应用滤波器
filtered_data = signal.filtfilt(b, a, audio_data)

# 归一化滤波后的信号
filtered_data = filtered_data / np.max(np.abs(filtered_data))

# 计算FFT
N = len(audio_data)
frequencies = fftfreq(N, 1/fs)
fft_audio = fft(audio_data)
fft_filtered = fft(filtered_data)

# 归一化FFT幅度
fft_audio = np.abs(fft_audio) / N
fft_filtered = np.abs(fft_filtered) / N

# 创建图形
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# 时域图像 - 原始信号
axs[0].plot(np.arange(N) / fs, audio_data, label="Original Signal")
axs[0].set_title("Time Domain - Original Signal")
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel("Amplitude")
axs[0].legend()

# 时域图像 - 滤波后信号
axs[1].plot(np.arange(N) / fs, filtered_data, label="Filtered Signal", color="orange")
axs[1].set_title("Time Domain - Filtered Signal")
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("Amplitude")
axs[1].legend()

# 频域图像 - FFT
axs[2].plot(frequencies[:N//2], fft_audio[:N//2], label="Original Signal FFT")
axs[2].plot(frequencies[:N//2], fft_filtered[:N//2], label="Filtered Signal FFT", color="orange")
axs[2].set_title("Frequency Domain - FFT")
axs[2].set_xlabel("Frequency [Hz]")
axs[2].set_ylabel("Amplitude")
axs[2].legend()

# 显示图形
plt.tight_layout()
plt.show()
