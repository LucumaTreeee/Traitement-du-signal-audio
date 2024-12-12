import matplotlib.pyplot as plt

# 读取 selected_audio_info.txt 文件
def read_audio_file(file_path):
    with open(file_path, 'r') as file:
        # 假设文件中的每一行是一个数值，表示音频信号的幅度
        data = [float(line.strip()) for line in file.readlines()]
    return data

# 绘制音频信号图像
def plot_audio_signal(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title('Audio Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

# 文件路径
file_path = 'selected_audio_info.txt'

# 读取数据并绘制图像
audio_data = read_audio_file(file_path)
plot_audio_signal(audio_data)
