import os
import numpy as np
from tqdm import tqdm
import wave
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import sounddevice as sd

sample_rate = 44100
record_duration = 3  # Record duration in seconds
win_len = 300
win_shift = int(win_len / 3)
win_type = 'hanning'
sigma = 1

def record_audio():
    """Record audio from the microphone."""
    messagebox.showinfo("Recording", "Recording will start for 3 seconds.")
    audio = sd.rec(int(record_duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    return audio.flatten()

def save_audio(audio_data, filename="recorded_audio.wav"):
    """Save the recorded audio to a .wav file.只能录一次，路径没有自动化"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


def read_data(directory):
    audios = []
    labels = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                label = int(file.split('_')[0])
                labels.append(label)
                
                file_path = os.path.join(root, file)
                with wave.open(file_path, 'rb') as signal:
                    n_channels, sampwidth, framerate, n_frames, comptype, compname = signal.getparams()
                    frames = signal.readframes(n_frames)
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                    audios.append((audio_data, n_frames))

    labels = np.array(labels)
    return audios, labels

def framing(wave_data, nframes):
    win_num = int(np.ceil(1 + (nframes - win_len) / win_shift))
    frames = np.zeros([win_num, win_len])

    for i in range(win_num):
        if i * win_shift + win_len <= nframes:
            frames[i, :] = wave_data[i * win_shift:i * win_shift + win_len]
        else:
            frames[i, :nframes - i * win_shift] = wave_data[i * win_shift:nframes]

    window = np.ones(win_len)

    if win_type == 'hanning':
        window = np.hanning(win_len)
    elif win_type == 'hamming':
        window = np.hamming(win_len)

    for i in range(win_num):
        frames[i, :] = window * frames[i, :]

    return frames

def pre_emphasis(signal, a=0.97):
    return np.append(signal[0], signal[1:] - a * signal[:-1])

def preprocess(signals):
    all_frames = []
    print('Feature extracting and end_point_detecting ...')
    for wave_data, nframes in tqdm(signals, total=len(signals)):
        emphasized_signal = pre_emphasis(wave_data)
        frames = framing(emphasized_signal, nframes)
        all_frames.append(frames)
    return all_frames

def compute_fft(frames):
    return [np.fft.rfft(frame, n=frame.shape[1], axis=1) for frame in frames]

def compute_spectral_energy(fft_frames):
    return [np.abs(frame) ** 2 for frame in fft_frames]

def mel_filter_bank(num_filters, fft_size, sample_rate):
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

def compute_mfcc(spectral_energy, sample_rate, num_filters=40, num_ceps=12):
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
    return (1 / x.size) * np.sqrt(np.sum((x - y) ** 2))

def dtw(s1, s2):
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n, m), np.inf)
    dtw_matrix[0, 0] = euclidean_distance(s1[0], s2[0])

    for i in range(1, n):
        dtw_matrix[i, 0] = dtw_matrix[i-1, 0] + euclidean_distance(s1[i], s2[0])
    for j in range(1, m):
        dtw_matrix[0, j] = dtw_matrix[0, j-1] + euclidean_distance(s1[0], s2[j])

    for i in tqdm(range(1, n)):
        for j in range(1, m):
            cost = euclidean_distance(s1[i], s2[j])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # Insertion
                                          dtw_matrix[i, j-1],    # Deletion
                                          dtw_matrix[i-1, j-1])  # Match

    return dtw_matrix

def plot_dtw(s1, s2, dtw_matrix, path):
    plt.figure(figsize=(10, 8))
    plt.imshow(dtw_matrix[1:, 1:], origin='lower', cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title('DTW Cost Matrix')
    plt.xlabel('Sequence 2')
    plt.ylabel('Sequence 1')
    plt.plot(path[:, 1], path[:, 0], 'r')  # Plot the path
    plt.show()

def compute_dtw_path(dtw_matrix):
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
    num_templates = len(templates)
    num_tests = len(tests)
    distance_matrix = np.zeros((num_templates, num_tests))

    for i in tqdm(range(num_templates)):
        for j in range(num_tests):
            s1 = templates[i]
            s2 = tests[j]
            
            # Compute DTW
            dtw_matrix = dtw(s1, s2)
            
            # Calculate DTW distance
            distance = dtw_matrix[-1, -1]
            distance_matrix[i, j] = distance
    
    return distance_matrix


def match_audio():
    """Process the recorded audio and match it to the templates."""
    # Step 1: Record audio
    recorded_audio = record_audio()

    # Step 2: Save the audio to a file
    save_audio(recorded_audio)

    # Step 3: Preprocess the recorded audio
    audio_frames = preprocess([(recorded_audio, len(recorded_audio))])
    audio_fft = compute_fft(audio_frames)
    audio_energy = compute_spectral_energy(audio_fft)
    audio_mfcc = compute_mfcc(audio_energy, sample_rate)

    # Step 4: Load and preprocess templates
    templates_directory = 'exp2/dtw/template/'  # Change this to your template path
    templates_audios, templates_labels = read_data(templates_directory)
    templates_frames = preprocess(templates_audios)
    templates_fft = compute_fft(templates_frames)
    templates_energy = compute_spectral_energy(templates_fft)
    templates_mfcc = compute_mfcc(templates_energy, sample_rate)

    # Step 5: Compute DTW distances
    distance_matrix = compute_dtw_distance(templates_mfcc, [audio_mfcc])

    # Step 6: Find the best match
    best_match_index = np.argmin(distance_matrix, axis=0)[0]
    best_match_label = templates_labels[best_match_index]

    # Step 7: Display the result
    result_message = f"The recorded audio matches with template label: {best_match_label}"
    messagebox.showinfo("Matching Result", result_message)


def main():
    root = tk.Tk()
    root.title("Audio Matching")

    # Title label
    title_label = tk.Label(root, text="Audio Matching System", font=("Arial", 16))
    title_label.pack(pady=20)

    # Record and Match Button
    match_button = tk.Button(root, text="Record and Match", font=("Arial", 14), command=match_audio)
    match_button.pack(pady=20)

    # Exit Button
    exit_button = tk.Button(root, text="Exit", font=("Arial", 14), command=root.quit)
    exit_button.pack(pady=20)

    # Run the main loop
    root.mainloop()

if __name__ == '__main__':
    main()
