import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, iirnotch, hilbert
from PyEMD import EMD

CIAOOOOOOOOOOOOOOOOOOOOOO
# Definiamo le bande di frequenza
FREQ_BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta_Low": (12, 16),
    "Beta_High": (16, 24),
    "Gamma": (24, 40)
}

### FILTRAGGIO DEL SEGNALE ###
def butter_bandpass(lowcut, highcut, fs, order=5):
    """Crea un filtro Butterworth passa-banda"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def notch_filter(data, freq=50.0, fs=128.0, quality=30):
    """Applica un filtro Notch per rimuovere il rumore a 50Hz"""
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    b, a = iirnotch(w0, quality)
    return filtfilt(b, a, data)

def apply_bandpass(signal, lowcut, highcut, fs=128.0, order=5):
    """Applica un filtro Butterworth passa-banda"""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal)

def preprocess_eeg_filtering(file_path, num_channels, num_samples_per_channel, fs=128.0):
    """
    Applica il filtraggio EEG (Notch + 6 Bande di frequenza).

    :param file_path: Path del dataset CSV
    :param num_channels: Numero di canali EEG
    :param num_samples_per_channel: Numero di campioni per canale EEG
    :param fs: Frequenza di campionamento EEG (default: 128 Hz)
    :return: Nome del file filtrato
    """
    df = pd.read_csv(file_path)
    base_name = os.path.splitext(file_path)[0]

    labels = df.iloc[:, 0].values
    eeg_data = df.iloc[:, 1:].values

    num_samples = eeg_data.shape[0]
    expected_columns = num_channels * num_samples_per_channel

    if eeg_data.shape[1] != expected_columns:
        raise ValueError(f"Errore dimensioni: atteso {expected_columns} colonne, trovato {eeg_data.shape[1]}.")

    eeg_data_filtered = np.zeros((num_samples, num_channels * num_samples_per_channel * 6))

    for i in range(num_samples):
        eeg_signals = eeg_data[i].reshape(num_channels, num_samples_per_channel)
        filtered_signals = []
        for ch in range(num_channels):
            notch_filtered_signal = notch_filter(eeg_signals[ch, :], freq=50.0, fs=fs)
            channel_bands = [apply_bandpass(notch_filtered_signal, low, high, fs) for low, high in FREQ_BANDS.values()]
            filtered_signals.append(np.concatenate(channel_bands))
        eeg_data_filtered[i] = np.concatenate(filtered_signals)

    df_filtered = pd.DataFrame(np.column_stack((labels, eeg_data_filtered)))
    output_file = f"{base_name}_filtered.csv"
    df_filtered.to_csv(output_file, index=False, header=False)
    print(f"Dataset filtrato salvato in: {output_file}")
    
    return output_file


### EMPIRICAL MODE DECOMPOSITION (EMD) ###
def apply_emd(signal, max_imfs=10):
    """Applica EMD a un segnale EEG"""
    emd = EMD()
    imfs = emd(signal, max_imf=max_imfs)
    print(f"Numero di IMF generate: {imfs.shape[0]}")  # 👈 Aggiunto per il debug
    return imfs


def extract_hht_features(imfs):
    """Estrazione delle feature HHT (IA, IP, IF)"""
    ia_features, ip_features, if_features = [], [], []
    for imf in imfs:
        analytic_signal = hilbert(imf)
        ia = np.abs(analytic_signal)  # Ampiezza Istantanea
        ip = np.unwrap(np.angle(analytic_signal))  # Fase Istantanea
        if_ = np.diff(ip)  # Frequenza Istantanea (derivata della fase)
        if_ = np.append(if_, if_[-1])  # Padding

        ia_features.append(ia)
        ip_features.append(ip)
        if_features.append(if_)

    return np.array(ia_features), np.array(ip_features), np.array(if_features)

def preprocess_emd(file_path, num_channels, num_samples_per_channel, max_imfs=10):
    """
    Applica EMD e salva il dataset finale con feature IA, IP, IF.

    :param file_path: Path del dataset filtrato
    :param num_channels: Numero di canali EEG
    :param num_samples_per_channel: Numero di campioni per canale EEG
    :param max_imfs: Numero massimo di IMF
    """
    df = pd.read_csv(file_path)
    base_name = os.path.splitext(file_path)[0]

    labels = df.iloc[:, 0].values
    eeg_data = df.iloc[:, 1:].values

    num_samples = eeg_data.shape[0]
    expected_columns = num_channels * num_samples_per_channel * 6
    if eeg_data.shape[1] != expected_columns:
        raise ValueError(f"Errore dimensioni: atteso {expected_columns} colonne, trovato {eeg_data.shape[1]}.")

    num_features_per_signal = max_imfs * 3
    feature_matrix = np.zeros((num_samples, num_channels * num_features_per_signal))

    for i in range(num_samples):
        eeg_signals = eeg_data[i].reshape(num_channels, 6, num_samples_per_channel)
        feature_vector = []
        for ch in range(num_channels):
            for band in range(6):
                imfs = apply_emd(eeg_signals[ch, band, :], max_imfs=max_imfs)
                ia, ip, if_ = extract_hht_features(imfs)
                feature_vector.extend(np.mean(ia, axis=1))
                feature_vector.extend(np.mean(ip, axis=1))
                feature_vector.extend(np.mean(if_, axis=1))

        feature_matrix[i] = feature_vector

    df_emd = pd.DataFrame(np.column_stack((labels, feature_matrix)))
    output_file = f"{base_name}_preproc.csv"
    df_emd.to_csv(output_file, index=False, header=False)
    print(f"Dataset finale salvato in: {output_file}")

    return output_file


### MAIN ###
def main():
    file_path = r"C:\Users\Mauro\Desktop\Mauro\Universita\AI\Progetto\Dataset\test_lines2_3.csv"
    num_channels = 14
    num_samples_per_channel = 256
    fs = 128.0  
    max_imfs = 10

    # 1. Filtraggio + Decomposizione in Bande
    filtered_file = preprocess_eeg_filtering(file_path, num_channels, num_samples_per_channel, fs)

    # 2. EMD + Estrazione Feature
    final_file = preprocess_emd(filtered_file, num_channels, num_samples_per_channel, max_imfs)

if __name__ == "__main__":
    main()
