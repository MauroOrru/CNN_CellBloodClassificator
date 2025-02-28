import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, iirnotch, hilbert
from PyEMD import EMD

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
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def notch_filter(data, freq=50.0, fs=128.0, quality=30):
    w0 = freq / (0.5 * fs)
    b, a = iirnotch(w0, quality)
    return filtfilt(b, a, data)

def apply_bandpass(signal, lowcut, highcut, fs=128.0, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal)

def preprocess_eeg_filtering(file_path, num_channels, num_samples_per_channel, fs=128.0):
    """
    Applica il filtraggio EEG (Rimozione DC Offset + Notch + 6 Bande di frequenza).
    """
    df = pd.read_csv(file_path, delimiter=',', skip_blank_lines=False, encoding='utf-8')

    base_name = os.path.splitext(file_path)[0]

    labels = df.iloc[:, 0].values
    eeg_data = df.iloc[:, 1:].values

    print(f"⚡ Numero di esempi letti dal CSV originale: {df.shape[0]}")
    # print(f"⚡ Label degli esempi originali: {labels}")

    num_samples = eeg_data.shape[0]
    expected_columns = num_channels * num_samples_per_channel

    if eeg_data.shape[1] != expected_columns:
        raise ValueError(f"Errore dimensioni: atteso {expected_columns} colonne, trovato {eeg_data.shape[1]}.")

    eeg_data_filtered = np.zeros((num_samples, num_channels * num_samples_per_channel * 6))

    for i in range(num_samples):
        print(f"🧐 Sto elaborando l'esempio {i} con label {labels[i]}")  # Debug

        eeg_signals = eeg_data[i].reshape(num_channels, num_samples_per_channel)

        # **RIMOZIONE DELLA COMPONENTE CONTINUA (DC OFFSET)**
        eeg_signals -= np.mean(eeg_signals, axis=1, keepdims=True)

        filtered_signals = []
        for ch in range(num_channels):
            # **Filtro Notch dopo la rimozione del DC offset**
            notch_filtered_signal = notch_filter(eeg_signals[ch, :], freq=50.0, fs=fs)
            
            # Filtraggio nelle 6 bande di frequenza
            channel_bands = [apply_bandpass(notch_filtered_signal, low, high, fs) for low, high in FREQ_BANDS.values()]
            filtered_signals.append(np.concatenate(channel_bands))

        eeg_data_filtered[i] = np.concatenate(filtered_signals)

    df_filtered = pd.DataFrame(np.column_stack((labels, eeg_data_filtered)))

    print(f"⚡ Numero di esempi nel dataset filtrato PRIMA del salvataggio: {df_filtered.shape[0]}")
    # print(f"⚡ Label PRIMA del salvataggio: {df_filtered.iloc[:, 0].values}")

    output_file = f"{base_name}_filtered.csv"

    df_filtered.to_csv(output_file, index=False, header=False, na_rep='0', encoding='utf-8')

    df_check = pd.read_csv(output_file, header=None)
    print(f"🔍 Esempi salvati nel CSV filtrato: {df_check.shape[0]}")
    print(f"🔍 Prime righe del CSV filtrato:\n{df_check.head()}")

    return output_file


### EMPIRICAL MODE DECOMPOSITION (EMD) ###
def apply_emd(signal, max_imfs=10):
    """Applica EMD a un segnale EEG e garantisce un numero fisso di IMF."""
    emd = EMD()
    imfs = emd(signal, max_imf=max_imfs)

    num_imfs = imfs.shape[0]
    if num_imfs < max_imfs:
        padding = np.zeros((max_imfs - num_imfs, imfs.shape[1]))
        imfs = np.vstack((imfs, padding))
    elif num_imfs > max_imfs:
        imfs = imfs[:max_imfs, :]

    return imfs

def extract_hht_features(imfs):
    """Estrazione delle feature HHT (IA, IP, IF)"""
    ia_features, ip_features, if_features = [], [], []
    for imf in imfs:
        analytic_signal = hilbert(imf)
        ia = np.abs(analytic_signal)  
        ip = np.unwrap(np.angle(analytic_signal))  
        if_ = np.diff(ip)  
        if_ = np.append(if_, if_[-1])  

        ia_features.append(ia)
        ip_features.append(ip)
        if_features.append(if_)

    return np.array(ia_features), np.array(ip_features), np.array(if_features)

def preprocess_emd(file_path, num_channels, num_samples_per_channel, max_imfs=10):
    """
    Applica EMD e salva il dataset finale con feature IA, IP, IF senza ridurre la dimensione con la media.
    """
    df = pd.read_csv(file_path, delimiter=',', skip_blank_lines=False, encoding='utf-8')

    base_name = os.path.splitext(file_path)[0]

    labels = df.iloc[:, 0].values
    eeg_data = df.iloc[:, 1:].values

    print(f"⚡ Numero di esempi nel CSV filtrato prima dell'EMD: {df.shape[0]}")
    print(f"⚡ Label degli esempi nel CSV filtrato: {labels}")

    num_samples = eeg_data.shape[0]
    num_features_per_signal = num_channels * len(FREQ_BANDS) * max_imfs * eeg_data.shape[1]  

    feature_matrix = np.zeros((num_samples, num_features_per_signal))

    for i in range(num_samples):
        print(f"🧐 EMD - Elaborazione esempio {i} con label {labels[i]}")  # Debug

        eeg_signals = eeg_data[i].reshape(num_channels, 6, num_samples_per_channel)
        feature_vector = []
        
        for ch in range(num_channels):
            for band in range(6):
                imfs = apply_emd(eeg_signals[ch, band, :], max_imfs=max_imfs)

                if imfs.shape[0] == 0:
                    print(f"⚠ Nessuna IMF per esempio {i}, assegno feature di default.")
                    feature_vector.extend(np.zeros(max_imfs * eeg_data.shape[1] * 3))  # 🟢 Ora non lo salta!
                    continue

                ia, ip, if_ = extract_hht_features(imfs)

                # 🟢 Conserviamo tutti i dati senza riduzione
                feature_vector.extend(ia.flatten())  
                feature_vector.extend(ip.flatten())  
                feature_vector.extend(if_.flatten())

        # 🟢 Debug: Stampiamo la dimensione del feature_vector
        print(f"🔍 Debug Esempio {i} - Lunghezza feature_vector = {len(feature_vector)}, Atteso = {num_features_per_signal}")

        # 🟢 Padding o Troncamento per adattare la dimensione del feature_vector
        expected_length = num_features_per_signal

        if len(feature_vector) < expected_length:
            feature_vector = np.pad(feature_vector, (0, expected_length - len(feature_vector)), mode='constant')
        elif len(feature_vector) > expected_length:
            feature_vector = feature_vector[:expected_length]

        feature_matrix[i] = feature_vector

    df_emd = pd.DataFrame(np.column_stack((labels, feature_matrix)))
    output_file = f"{base_name}_preproc.csv"
    df_emd.to_csv(output_file, index=False, header=False)

    print(f"⚡ Dataset finale salvato: {df_emd.shape}")
    print(f"⚡ Label finali nel dataset preproc: {df_emd.iloc[:, 0].values}")

    return output_file





### MAIN ###
def main():
    file_path = r"C:\Users\Mauro\Desktop\Mauro\Universita\AI\Progetto\Dataset\test_lines2_3.csv"
    num_channels = 14
    num_samples_per_channel = 256
    fs = 128.0  
    max_imfs = 10

    filtered_file = preprocess_eeg_filtering(file_path, num_channels, num_samples_per_channel, fs)
    final_file = preprocess_emd(filtered_file, num_channels, num_samples_per_channel, max_imfs)

if __name__ == "__main__":
    main()
