import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import subprocess
import soundfile as sf  # 用于播放音频
import sounddevice as sd
from scipy.io.wavfile import write  # 用于保存音频文件
import pandas as pd

# 初始化全局变量
audio_data = None
sample_rate = None
selected_region = None
canvas = None
note_de_musique_click_count = 0  # 跟踪 Note de musique 按钮的点击次数

# 读取音频文件并打印其基本信息
def analyze_wav(filename):
    y, sr = librosa.load(filename, sr=None)  # y是音频数据，sr是采样率
    info_text = (
        f"Audio file: {filename}\n"
        f"Sample Rate: {sr} Hz\n"
        f"Total Samples: {len(y)} samples\n"
        f"Duration: {len(y) / sr:.2f} seconds"
    )
    return y, sr, info_text

# 绘制时域图
def plot_waveform(y, sr, frame, title="Waveform of the Audio", highlight=None):
    global canvas
    for widget in frame.winfo_children():
        widget.destroy()  # 清空之前的内容

    fig, ax = plt.subplots(figsize=(8, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.5, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")

    # 如果有高亮区域，绘制高亮区域
    if highlight:
        ax.axvspan(highlight[0], highlight[1], color='red', alpha=0.3)

    # 嵌入图像到 Tkinter 窗口
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# 打开文件对话框，选择文件
def open_wav_file():
    global audio_data, sample_rate, selected_region
    file_path = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])

    if file_path:
        try:
            y, sr, info_text = analyze_wav(file_path)
            info_label.config(text=info_text)

            # 绘制新的时域图
            plot_waveform(y, sr, fig_frame)

            audio_data, sample_rate = y, sr
            selected_region = None
        except Exception as e:
            messagebox.showerror("Error", f"Error processing the file: {e}")

# 捕获鼠标点击的起止位置
def start_decoupe():
    global selected_region
    if selected_region and len(selected_region) == 2:
        show_selected_region()
        return

    selected_region = []

    def onclick(event):
        if len(selected_region) < 2:
            selected_region.append(event.xdata)
            if len(selected_region) == 2:
                # 更新高亮区域并显示时间范围
                selected_region.sort()  # 确保顺序
                start_time, end_time = selected_region
                plot_waveform(audio_data, sample_rate, fig_frame, highlight=selected_region)
                region_label.config(text=f"Selected Region: {start_time:.2f}s to {end_time:.2f}s")
                fig.canvas.mpl_disconnect(cid)

    # 启用 Matplotlib 的交互功能
    fig = plt.gcf()  # 获取当前 Matplotlib 图像
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

def show_selected_region():
    global selected_region
    if selected_region and len(selected_region) == 2:
        start_time, end_time = selected_region

        # 转换时间范围到采样点范围
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # 提取选中的音频数据
        selected_audio = audio_data[start_sample:end_sample]

        # 绘制选定区域的时域图
        plot_waveform(selected_audio, sample_rate, fig_frame, title="Selected Region")
        region_label.config(text=f"Displaying Region: {start_time:.2f}s to {end_time:.2f}s")

        # 自动保存选中的音频信息到文件
        save_path = "selected_audio_info.txt"  # 自动保存的文件名
        try:
            # 保存选中的音频数据到文件，不写表头
            np.savetxt(save_path, selected_audio, fmt="%.6f")  # 只写振幅数据
            messagebox.showinfo("Success", f"Audio information automatically saved to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")
    else:
        messagebox.showwarning("Warning", "No region selected. Please use Decoupe first.")

# 读取并绘制 FFT 结果
def read_fft_result(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            real, imag = map(float, line.strip().split())
            magnitude = np.sqrt(real**2 + imag**2)
            data.append((real, imag, magnitude))
    return data

def plot_positive_frequencies_and_magnitudes(data, frame):
    global canvas
    frequencies = []
    magnitudes = []
    for i, (real, imag, magnitude) in enumerate(data):
        if magnitude > 0:
            frequencies.append(i)  # 假设频率是按顺序排列的
            magnitudes.append(magnitude)

    # 清空之前的内容
    for widget in frame.winfo_children():
        widget.destroy()

    # 绘制图像
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frequencies, magnitudes, label="Magnitude")
    ax.set_title("Positive Frequencies and Magnitudes")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.legend()
    ax.grid(True)

    # 添加鼠标点击事件
    def onclick(event):
        if event.inaxes == ax:
            x_value = event.xdata
            print(f"Clicked at x={x_value:.2f} Hz")
            messagebox.showinfo("X Value", f"Clicked at x={x_value:.2f} Hz")

    fig.canvas.mpl_connect('button_press_event', onclick)

    # 嵌入图像到 Tkinter 窗口
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def plot_fft_result():
    try:
        fft_data = read_fft_result('fft_result.txt')
        plot_positive_frequencies_and_magnitudes(fft_data, fig_frame)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load FFT result: {e}")

def plot_note_de_musique():
    global note_de_musique_click_count
    note_de_musique_click_count += 1

    if note_de_musique_click_count == 1:
        try:
            fft_data = read_fft_result('fft_result.txt')
            frequencies = []
            magnitudes = []
            for i, (real, imag, magnitude) in enumerate(fft_data):
                if magnitude > 0:
                    frequencies.append(i)  # 假设频率是按顺序排列的
                    magnitudes.append(magnitude)

            # 清空之前的内容
            for widget in fig_frame.winfo_children():
                widget.destroy()

            # 绘制图像
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(frequencies, magnitudes, label="Magnitude")
            ax.set_title("Positive Frequencies and Magnitudes")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")
            ax.legend()
            ax.grid(True)

            # 设置 x 轴的频率限制
            ax.set_xlim([16, 2093])

            # 添加鼠标点击事件
            def onclick(event):
                if event.inaxes == ax:
                    x_value = event.xdata
                    print(f"Clicked at x={x_value:.2f} Hz")
                    messagebox.showinfo("X Value", f"Clicked at x={x_value:.2f} Hz")

            fig.canvas.mpl_connect('button_press_event', onclick)

            # 嵌入图像到 Tkinter 窗口
            canvas = FigureCanvasTkAgg(fig, master=fig_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # 显示音符频率表
            show_note_frequency_table()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load FFT result: {e}")

def show_note_frequency_table():
    # 根据 A4 的频率，计算其他音符的频率
    A4_freq = 440.0  # A4的频率是 440 Hz
    semitone_ratio = 2 ** (1/12)  # 每个半音的频率比

    # 创建音符的名称（C0到C7）
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octaves = list(range(8))  # 从C0到C7
    note_frequencies = []

    # 计算从 C0 到 C7 的所有音符频率
    for octave in octaves:
        for i, note in enumerate(notes):
            note_name = f"{note}{octave}"
            semitone_offset = (octave - 4) * 12 + i - 9  # A4 对应的索引为 9，其他音符相对于 A4 的半音偏移
            frequency = A4_freq * (semitone_ratio ** semitone_offset)  # 计算该音符的频率
            note_frequencies.append([note_name, frequency])

    # 使用 pandas 创建 DataFrame
    df = pd.DataFrame(note_frequencies, columns=["Note", "Frequency (Hz)"])

    # 创建表格框架
    table_frame = ttk.Frame(root)
    table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # 创建 Treeview 小部件显示 DataFrame 中的数据
    tree = ttk.Treeview(table_frame, columns=["Note", "Frequency (Hz)"], show="headings")

    # 设置列标题
    tree.heading("Note", text="Note")
    tree.heading("Frequency (Hz)", text="Frequency (Hz)")

    # 插入数据
    for index, row in df.iterrows():
        tree.insert("", "end", values=(row["Note"], f"{row['Frequency (Hz)']:.2f}"))

    # 显示滚动条
    scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    tree.pack(fill=tk.BOTH, expand=True)

# 读取并绘制反傅里叶变换结果
def plot_inverse_fft_result():
    try:
        inverse_fft_data = np.loadtxt('inverse_fft_result.txt')
        plot_result(inverse_fft_data, "Inverse FFT Result", "Time (seconds)", "Amplitude", fig_frame)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load inverse FFT result: {e}")

# 绘制图像的通用函数
def plot_result(data, title, xlabel, ylabel, frame):
    global canvas
    for widget in frame.winfo_children():
        widget.destroy()  # 清空之前的内容

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 添加鼠标点击事件
    def onclick(event):
        if event.inaxes == ax:
            x_value = event.xdata
            print(f"Clicked at x={x_value:.2f}")
            messagebox.showinfo("X Value", f"Clicked at x={x_value:.2f}")

    fig.canvas.mpl_connect('button_press_event', onclick)

    # 嵌入图像到 Tkinter 窗口
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# 播放反傅里叶变换后的音频
def play_inverse_fft_audio():
    try:
        # 假设反傅里叶变换后的音频数据保存在 'inverse_fft_result.txt' 文件中
        file_path = 'inverse_fft_result.txt'
        audio_data = np.loadtxt(file_path)
        
        # 检查音频数据的有效性
        if len(audio_data) > 0 and audio_data.max() != audio_data.min():
            # 放大音频数据，确保它足够大
            audio_data *= 10000  # 你可以调整这个值，直到能听到声音

            # 确保音频数据在 [-1, 1] 范围内
            audio_data = np.clip(audio_data, -1, 1)

            # 播放音频
            sd.play(audio_data, sample_rate)
            sd.wait()  # 等待音频播放完成
            print("音频播放完成。")
        else:
            print("音频数据无效，无法播放。")
    except Exception as e:
        print(f"播放音频时发生错误: {e}")

# 运行 Traitement_son.cpp 的程序
def run_traitement_son():
    try:
        # 保存采样率到文件
        with open("sample_rate.txt", "w") as f:
            f.write(str(sample_rate))
        
        # 运行外部的 C++ 程序 Traitement_son（假设已经编译成可执行文件）
        result = subprocess.run(
            ['./Traitement_son'],  # 调用的可执行文件路径
            check=True,  # 确保外部程序成功运行，否则抛出异常
            text=True,  # 自动将输出转换为字符串
            capture_output=True,  # 捕获程序输出
            encoding='utf-8'  # 设置编码为 utf-8，避免 GBK 解码问题
        )
        messagebox.showinfo("Success", "Traitement_son 程序成功执行！")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"调用 Traitement_son 程序失败: {e}")
    except FileNotFoundError:
        messagebox.showerror("Error", "Traitement_son 程序文件未找到，请确认程序路径是否正确。")
    except Exception as e:
        messagebox.showerror("Error", f"发生错误: {e}")

        
# 显示 selected_audio_info.txt 的时域图像
def show_signal_selected():
    try:
        # 读取 selected_audio_info.txt 文件中的数据
        data = np.loadtxt('selected_audio_info.txt')
        sample_rate = 44100  # 假设采样率为 44100 Hz

        # 绘制时域图像
        plot_waveform(data, sample_rate, fig_frame, title="Selected Audio Signal")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load selected audio info: {e}")

# 创建主界面
root = tk.Tk()
root.title("Audio Analysis and Processing")
root.geometry("1200x800")

# 创建按钮框架
button_frame = tk.Frame(root)
button_frame.pack(side=tk.LEFT, padx=20)

# 创建按钮
# 创建按钮（法语名称）
btn_open = tk.Button(button_frame, text="Ouvrir un fichier WAV", command=open_wav_file, height=2, width=30)
btn_open.pack(pady=10)

btn_decoupe = tk.Button(button_frame, text="Découper", command=start_decoupe, height=2, width=30)
btn_decoupe.pack(pady=10)

btn_traitement = tk.Button(button_frame, text="Traitement", command=run_traitement_son, height=2, width=30)
btn_traitement.pack(pady=10)

btn_show_signal_selected = tk.Button(button_frame, text="Afficher le signal sélectionné", command=show_signal_selected, height=2, width=30)
btn_show_signal_selected.pack(pady=10)

btn_fft_result = tk.Button(button_frame, text="Afficher le résultat FFT", command=plot_fft_result, height=2, width=30)
btn_fft_result.pack(pady=10)

btn_note_de_musique = tk.Button(button_frame, text="Note de musique", command=plot_note_de_musique, height=2, width=30)
btn_note_de_musique.pack(pady=10)

btn_inverse_fft_result = tk.Button(button_frame, text="Afficher le résultat FFT inverse", command=plot_inverse_fft_result, height=2, width=30)
btn_inverse_fft_result.pack(pady=10)

btn_lance = tk.Button(button_frame, text="Lancer l'audio", command=play_inverse_fft_audio, height=2, width=30)
btn_lance.pack(pady=10)


info_label = tk.Label(root, text="WAV File Info will be displayed here.", justify="left")
info_label.pack(pady=10)

region_label = tk.Label(root, text="Selected Region: None", justify="left", fg="blue")
region_label.pack(pady=5)

fig_frame = tk.Frame(root)
fig_frame.pack(fill=tk.BOTH, expand=True)

root.mainloop()