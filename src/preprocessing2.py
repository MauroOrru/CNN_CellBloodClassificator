import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, iirnotch, hilbert
from PyEMD import EMD

### **FILTRAGGIO DEL SEGNALE** ###
def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    high = cutoff / nyquist
    b, a = butter(order, high, btype='high')
    return b, a

def apply_highpass(signal, cutoff, fs=128.0, order=5):
    """Applica un filtro High-Pass (HPF) per rimuovere rumore a bassa frequenza"""
    b, a = butter_highpass(cutoff, fs, order)
    return filtfilt(b, a, signal)

def notch_filter(data, freq=50.0, fs=128.0, quality=30):
    """Applica un filtro Notch per rimuovere il rumore a 50Hz"""
    w0 = freq / (0.5 * fs)
    b, a = iirnotch(w0, quality)
    return filtfilt(b, a, data)

def preprocess_eeg_filtering(file_path, num_channels, num_samples_per_channel, fs=128.0):
    """Applica il filtraggio EEG: rimozione DC offset, filtro Notch e HPF a 0.3 Hz"""
    print(f"📂 Caricamento del dataset da: {file_path}")
    df = pd.read_csv(file_path, delimiter=',', skip_blank_lines=False, encoding='utf-8')
    base_name = os.path.splitext(file_path)[0]

    labels = df.iloc[:, 0].values
    eeg_data = df.iloc[:, 1:].values

    num_samples = eeg_data.shape[0]
    expected_columns = num_channels * num_samples_per_channel

    if eeg_data.shape[1] != expected_columns:
        raise ValueError(f"Errore dimensioni: atteso {expected_columns} colonne, trovato {eeg_data.shape[1]}.")

    eeg_data_filtered = np.zeros((num_samples, num_channels * num_samples_per_channel))

    for i in range(num_samples):
        print(f"🧐 [Filtraggio] Elaborazione esempio {i+1}/{num_samples} con label {labels[i]}")

        eeg_signals = eeg_data[i].reshape(num_channels, num_samples_per_channel)
        
        # **Rimozione DC offset**
        eeg_signals -= np.mean(eeg_signals, axis=1, keepdims=True)

        filtered_signals = []
        for ch in range(num_channels):
            filtered_signal = apply_highpass(eeg_signals[ch, :], cutoff=0.3, fs=fs)  # HPF a 0.3Hz
            filtered_signal = notch_filter(filtered_signal, freq=50.0, fs=fs)  # Notch a 50Hz
            filtered_signals.append(filtered_signal)

        eeg_data_filtered[i] = np.concatenate(filtered_signals)

    df_filtered = pd.DataFrame(np.column_stack((labels, eeg_data_filtered)))
    output_file = f"{base_name}_filtered.csv"
    df_filtered.to_csv(output_file, index=False, header=False, na_rep='0', encoding='utf-8')

    print(f"✅ Filtraggio completato! Dataset salvato in: {output_file}")

    return output_file


### **EMPIRICAL MODE DECOMPOSITION (EMD) + HILBERT TRANSFORM** ###
def apply_emd(signal, max_imfs=4):
    """Applica EMD e restituisce al massimo `max_imfs` IMF"""
    emd = EMD()
    imfs = emd(signal)

    if imfs.shape[0] > max_imfs:
        imfs = imfs[:max_imfs, :]  # Limitiamo il numero di IMF

    return imfs

def extract_hht_features(imfs, num_samples_per_channel, max_imfs):
    """Estrazione delle feature IA, IP, IF con zero padding per ogni IMF"""
    n_imfs_computed = imfs.shape[0]

    IA_padded = np.zeros((max_imfs, num_samples_per_channel))
    IP_padded = np.zeros((max_imfs, num_samples_per_channel))
    IF_padded = np.zeros((max_imfs, num_samples_per_channel))

    for j in range(n_imfs_computed):
        analytic_signal = hilbert(imfs[j, :])
        ia = np.abs(analytic_signal)
        ip = np.unwrap(np.angle(analytic_signal))
        if_ = np.gradient(ip)  

        # **Overshoot Correction (Stabilizzazione Numerica)**
        if len(ia) > 1:
            ia[0], ia[-1] = ia[1], ia[-2]
            ip[0], ip[-1] = ip[1], ip[-2]
            if_[0], if_[-1] = if_[1], if_[-2]

        # **Padding separato per ogni feature**
        ia_padded = np.pad(ia, (0, num_samples_per_channel - len(ia)), mode='constant')[:num_samples_per_channel]
        ip_padded = np.pad(ip, (0, num_samples_per_channel - len(ip)), mode='constant')[:num_samples_per_channel]
        if_padded = np.pad(if_, (0, num_samples_per_channel - len(if_)), mode='constant')[:num_samples_per_channel]

        IA_padded[j, :] = ia_padded
        IP_padded[j, :] = ip_padded
        IF_padded[j, :] = if_padded

    return IA_padded, IP_padded, IF_padded

def preprocess_emd(file_path, num_channels, num_samples_per_channel, max_imfs=4):
    """Applica EMD e Hilbert Transform dopo il filtraggio EEG"""
    print(f"📂 Caricamento del dataset filtrato per EMD da: {file_path}")
    df = pd.read_csv(file_path, delimiter=',', skip_blank_lines=False, encoding='utf-8')
    base_name = os.path.splitext(file_path)[0]

    labels = df.iloc[:, 0].values
    eeg_data = df.iloc[:, 1:].values

    num_samples = eeg_data.shape[0]

    feature_matrix = []

    for i in range(num_samples):
        print(f"🧐 [EMD] Elaborazione esempio {i+1}/{num_samples} con label {labels[i]}")
        eeg_signals = eeg_data[i].reshape(num_channels, num_samples_per_channel)
        feature_vector = [labels[i]]  # Iniziamo con la label
        
        for ch in range(num_channels):
            imfs = apply_emd(eeg_signals[ch, :], max_imfs=max_imfs)

            print(f"🔍 Canale {ch+1}: {imfs.shape[0]} IMF generate (Attese: {max_imfs})")

            ia, ip, if_ = extract_hht_features(imfs, num_samples_per_channel, max_imfs)

            feature_vector.extend(ia.flatten(order='F'))  
            feature_vector.extend(ip.flatten(order='F'))  
            feature_vector.extend(if_.flatten(order='F'))

        feature_matrix.append(feature_vector)

    df_emd = pd.DataFrame(feature_matrix)
    output_file = f"{base_name}_preproc.csv"
    df_emd.to_csv(output_file, index=False, header=False)

    print(f"✅ EMD completata! Dataset salvato in: {output_file}")

    return output_file


### **MAIN** ###
def main():
    file_path = r"C:\Users\Mauro\Desktop\Mauro\Universita\AI\Progetto\Dataset\test_lines2_3d.csv"
    num_channels = 14
    num_samples_per_channel = 256
    fs = 128.0  
    max_imfs = 6 

    filtered_file = preprocess_eeg_filtering(file_path, num_channels, num_samples_per_channel, fs)
    final_file = preprocess_emd(filtered_file, num_channels, num_samples_per_channel, max_imfs)

if __name__ == "__main__":
    main()
