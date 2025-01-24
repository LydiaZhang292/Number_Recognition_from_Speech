import os
import numpy as np
from tqdm import tqdm
import wave
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import librosa

# 设置采样率和窗口参数
sample_rate = 44100
win_len = 1024
win_shift = int(win_len / 3)
win_type = 'hanning'

def read_data(directory):
    """读取指定目录中的音频数据及其标签"""
    audios = []
    labels = []

    # 遍历目录中的文件
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):  # 只处理.wav文件
                label = int(file.split('_')[0])  # 提取文件名中的标签
                labels.append(label)
                
                file_path = os.path.join(root, file)
                
                # 使用librosa加载音频文件
                audio_data, sr = librosa.load(file_path, sr=44100)
                if sr != sample_rate:
                    # 重新采样到指定的采样率
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sample_rate)
                # 剪除音频开始和结束的静音部分
                audio_data, _ = librosa.effects.trim(audio_data, top_db=20)

                n_frames = len(audio_data)
                audios.append((audio_data, n_frames))

    labels = np.array(labels)
    return audios, labels

def framing(wave_data, nframes):
    """将音频数据分帧"""
    win_num = int(np.ceil(1 + (nframes - win_len) / win_shift))
    frames = np.zeros([win_num, win_len])

    for i in range(win_num):
        if i * win_shift + win_len <= nframes:
            frames[i, :] = wave_data[i * win_shift:i * win_shift + win_len]
        else:
            frames[i, :nframes - i * win_shift] = wave_data[i * win_shift:nframes]

    window = np.ones(win_len)

    # 应用窗口函数
    if win_type == 'hanning':
        window = np.hanning(win_len)
    elif win_type == 'hamming':
        window = np.hamming(win_len)

    for i in range(win_num):
        frames[i, :] = window * frames[i, :]

    return frames

def pre_emphasis(signal, a=0.97):
    """对信号进行预加重处理"""
    return np.append(signal[0], signal[1:] - a * signal[:-1])

def preprocess(signals):
    """对信号进行预处理，包括预加重和分帧"""
    all_frames = []
    for wave_data, nframes in signals:
        emphasized_signal = pre_emphasis(wave_data)
        frames = framing(emphasized_signal, nframes)
        all_frames.append(frames)
    return all_frames

def compute_fft(frames):
    """计算帧的快速傅里叶变换"""
    return [np.fft.rfft(frame, n=frame.shape[1], axis=1) for frame in frames]

def compute_spectral_energy(fft_frames):
    """计算频谱能量"""
    return [np.abs(frame) ** 2 for frame in fft_frames]

def mel_filter_bank(num_filters, fft_size, sample_rate):
    """生成梅尔滤波器组"""
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin_points = np.floor((fft_size // 2 + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((num_filters, int(fft_size // 2 + 1)))
    for m in range(1, num_filters + 1):
        f_m_minus = int(bin_points[m - 1])
        f_m = int(bin_points[m])
        f_m_plus = int(bin_points[m + 1])

        for k in range(f_m_minus, f_m):
            filters[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
        for k in range(f_m, f_m_plus):
            filters[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])

    return filters

def compute_mfcc(spectral_energy, sample_rate, num_filters=60, num_ceps=16):
    """计算MFCC特征"""
    mfccs = []
    for energy in spectral_energy:
        fft_size = (energy.shape[-1] - 1) * 2
        mel_filters = mel_filter_bank(num_filters, fft_size, sample_rate)
        mel_energy = np.tensordot(energy, mel_filters.T, axes=([-1], [0]))
        mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)
        log_mel_energy = np.log(mel_energy)
        mfcc = dct(log_mel_energy, type=2, axis=-1, norm='ortho')[..., :num_ceps]
        mfccs.append(mfcc)
    return mfccs

def euclidean_distance(x, y):
    """计算欧氏距离"""
    return (1 / x.size) * np.sqrt(np.sum((x - y) ** 2))

def dtw(s1, s2):
    """动态时间规整算法"""
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n, m), np.inf)
    dtw_matrix[0, 0] = euclidean_distance(s1[0], s2[0])

    for i in range(1, n):
        dtw_matrix[i, 0] = dtw_matrix[i-1, 0] + euclidean_distance(s1[i], s2[0])
    for j in range(1, m):
        dtw_matrix[0, j] = dtw_matrix[0, j-1] + euclidean_distance(s1[0], s2[j])

    for i in range(1, n):
        for j in range(1, m):
            cost = euclidean_distance(s1[i], s2[j])
            # 计算最小路径
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # 插入
                                          dtw_matrix[i, j-1],    # 删除
                                          dtw_matrix[i-1, j-1])  # 匹配

    return dtw_matrix

def plot_dtw(s1, s2, dtw_matrix, path):
    """可视化DTW成本矩阵及路径"""
    plt.figure(figsize=(10, 8))
    plt.imshow(dtw_matrix[1:, 1:], origin='lower', cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title('DTW Cost Matrix')
    plt.xlabel('Sequence 2')
    plt.ylabel('Sequence 1')
    plt.plot(path[:, 1], path[:, 0], 'r')  # 绘制路径
    plt.show()

def compute_dtw_path(dtw_matrix):
    """计算DTW路径"""
    n, m = dtw_matrix.shape
    path = []
    i, j = n - 1, m - 1
    path.append((i, j))

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            steps = np.array([dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1]])
            step_index = np.argmin(steps)
            if step_index == 0:
                i, j = i-1, j-1
            elif step_index == 1:
                i -= 1
            else:
                j -= 1
        path.append((i, j))

    path.reverse()
    return np.array(path)

def compute_dtw_distance(templates, tests):
    """计算DTW距离矩阵"""
    num_templates = len(templates)
    num_tests = len(tests)
    distance_matrix = np.zeros((num_templates, num_tests))

    for i in tqdm(range(num_templates)):
        for j in range(num_tests):
            s1 = templates[i]
            s2 = tests[j]
            
            # 计算DTW
            dtw_matrix = dtw(s1, s2)
            
            # 计算DTW距离
            distance = dtw_matrix[-1, -1]
            distance_matrix[i, j] = distance
    
    return distance_matrix

def plot_distance_matrix(distance_matrix):
    """绘制距离矩阵的热图"""
    plt.figure(figsize=(10, 8))
    plt.imshow(distance_matrix, interpolation='nearest', cmap='hot', aspect='auto')
    plt.title("DTW Distance Matrix Heatmap")
    plt.xlabel("Test Samples")
    plt.ylabel("Template Samples")
    plt.colorbar(label='DTW Distance')
    plt.show()

def calculate_accuracy(distance_matrix, template_labels, test_labels):
    """计算分类准确率"""
    num_tests = distance_matrix.shape[1]
    correct_predictions = 0

    for j in range(num_tests):
        # 找到每个测试样本的最小距离索引
        min_index = np.argmin(distance_matrix[:, j])
        predicted_label = template_labels[min_index]
        
        # 检查预测标签是否与实际标签匹配
        if predicted_label == test_labels[j]:
            correct_predictions += 1

    # 计算准确率
    accuracy = correct_predictions / num_tests
    print(f"Classification Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def train_test(templates_directory, test_directories):
    """训练和测试模型"""
    templates_audios, templates_labels = read_data(templates_directory)
    
    # 初始化测试音频和标签的列表
    test_audios = []
    test_labels = []
    
    # 从所有测试目录读取数据
    for test_directory in test_directories:
        audios, labels = read_data(test_directory)
        test_audios.extend(audios)
        test_labels.extend(labels)
    
    # 处理并计算
    templates_frames = preprocess(templates_audios)
    test_frames = preprocess(test_audios)
    X_temp = compute_fft(templates_frames)
    X_test = compute_fft(test_frames)
    E_temp = compute_spectral_energy(X_temp)
    E_test = compute_spectral_energy(X_test)
    mfcc_temp = compute_mfcc(E_temp, sample_rate)
    mfcc_test = compute_mfcc(E_test, sample_rate)
    distance_matrix = compute_dtw_distance(mfcc_temp, mfcc_test)
    acc = calculate_accuracy(distance_matrix, templates_labels, test_labels)
    return acc

def evaluate_user_data(templates_directory, test_directory):
    """评估用户数据"""
    user_ids = os.listdir(test_directory)
    overall_accuracies = []

    for user_id in tqdm(user_ids):
        user_dir = os.path.join(test_directory, user_id)
        
        if not os.path.isdir(user_dir):
            continue

        # 列出用户的所有号码文件夹
        number_folders = [f for f in os.listdir(user_dir) if os.path.isdir(os.path.join(user_dir, f))]

        template_dir = templates_directory

        # 创建测试目录列表，排除模板
        test_dirs = [os.path.join(user_dir, num) for num in number_folders]

        # 运行train_test函数并记录准确率
        accuracy = train_test(template_dir, test_dirs)
        print(f"User {user_id} test acc: {accuracy * 100:.2f}%")
        overall_accuracies.append(accuracy)

    if overall_accuracies:
        average_accuracy = np.mean(overall_accuracies)
        print(f"Average Accuracy across all users: {average_accuracy * 100:.2f}%")
    else:
        print("No valid data to evaluate.")

# 示例用法
templates_directory = 'dtw/template'
test_directory = 'dtw/test'
evaluate_user_data(templates_directory, test_directory)
