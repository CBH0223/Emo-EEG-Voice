import numpy as np
import mne
import pickle as pkl
import os
from scipy.signal.windows import hann
from scipy.io import loadmat


folder_path = '/Users/chenxin/Desktop/MDD/data/EEG/EEG-MODMA'


n_vids = 28
freq = 250
nsec = 30
nchn = 25
freq_bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 47)]

def _get_relative_psd(relative_energy_graph, freq_bands, sample_freq, stft_n=256):
    start_index = int(np.floor(freq_bands[0] / sample_freq * stft_n))
    end_index = int(np.floor(freq_bands[1] / sample_freq * stft_n))
    psd = np.mean(relative_energy_graph[:, start_index - 1:end_index] ** 2, axis=1)
    return psd

def extract_psd_feature(data, window_size, freq_bands, stft_n=256):
    sample_freq = freq
    if len(data.shape) > 2:
        data = np.squeeze(data)
    n_channels, n_samples = data.shape
    point_per_window = int(sample_freq * window_size)
    window_num = int(n_samples // point_per_window)
    psd_feature = np.zeros((window_num, len(freq_bands), n_channels))

    for window_index in range(window_num):
        start_index, end_index = point_per_window * window_index, point_per_window * (window_index + 1)
        window_data = data[:, start_index:end_index]
        hdata = window_data * hann(point_per_window)
        fft_data = np.fft.fft(hdata, n=stft_n)
        energy_graph = np.abs(fft_data[:, 0: int(stft_n / 2)])
        relative_energy_graph = energy_graph / np.sum(energy_graph)

        for band_index, band in enumerate(freq_bands):
            band_relative_psd = _get_relative_psd(relative_energy_graph, band, sample_freq, stft_n)
            psd_feature[window_index, band_index, :] = band_relative_psd

    return psd_feature


psd_features = []


for file_name in os.listdir(folder_path):
    if file_name.endswith('.mat'):
        file_path = os.path.join(folder_path, file_name)
        print(file_name)
        mat_content = loadmat(file_path, struct_as_record=False, squeeze_me=True)
        EEG1 = mat_content['EEG1']
        data = EEG1.data
        sampling_rate = EEG1.srate
        data_s = data[0:25, 0:7500]
        psd_data = extract_psd_feature(data_s, 1, freq_bands)
        psd_features.append(psd_data)

n_files = len(psd_features)
window_num, n_bands, n_channels = psd_features[0].shape

psd_tensor = np.zeros((n_files, window_num, n_bands, n_channels))

for i, psd_data in enumerate(psd_features):
    psd_tensor[i, :, :, :] = psd_data


np.save('psd_tensor.npy', psd_tensor)


freq_bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 47)]
freq = 250
nsec = 30


folder_path = '/Users/chenxin/Desktop/MDD/data/EEG/EEG-MODMA'


nchn = 25  
subs_de = np.zeros((nchn, nsec, len(freq_bands), len(os.listdir(folder_path))))


for idx, file_name in enumerate(os.listdir(folder_path)):
    if file_name.endswith('.mat'):
        file_path = os.path.join(folder_path, file_name)
        print(file_name)
        mat_content = loadmat(file_path, struct_as_record=False, squeeze_me=True)
        EEG1 = mat_content['EEG1']
        data = EEG1.data
        sampling_rate = EEG1.srate
        data_s = data[0:25, 0:7500].astype(np.float64)  
        for i in range(len(freq_bands)):
            low_freq = freq_bands[i][0]
            high_freq = freq_bands[i][1]
            data_video_filt = mne.filter.filter_data(data_s, freq, l_freq=low_freq, h_freq=high_freq)
            data_video_filt = data_video_filt.reshape(nchn, -1, freq)
            de_one = 0.5 * np.log(2 * np.pi * np.exp(1) * (np.var(data_video_filt, 2)))
            subs_de[:, :, i, idx] = de_one


np.save('de_tensor.npy', subs_de)
