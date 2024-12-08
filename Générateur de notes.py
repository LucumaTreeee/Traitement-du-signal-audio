import numpy as np
from scipy.io.wavfile import write

# 参数设置
fs = 44100  # 采样率
duration = 1.0  # 持续时间（秒）
freq = 261.63  # C4 的频率

# 生成信号
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
signal = 0.5 * np.sin(2 * np.pi * freq * t)  # 振幅为 0.5 的正弦波

# 将信号归一化到 16-bit PCM 格式
signal_int16 = np.int16(signal * 32767)  # 32767 是 16 位音频的最大值

# 保存为 WAV 文件
output_filename = "C4_piano.wav"
write(output_filename, fs, signal_int16)

print(f"WAV 文件已保存为: {output_filename}")
