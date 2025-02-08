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
    Applica il filtraggio EEG (Notch + 6 Bande di frequenza).
    
    Parametri:
      file_path: percorso al file CSV contenente il dataset EEG.
      num_channels: numero di canali EEG presenti nel dataset.
      num_samples_per_channel: numero di campioni per ciascun canale.
      fs: frequenza di campionamento del segnale EEG (default 128 Hz).
    """
    # Legge il file CSV e ne crea un DataFrame (una tabella) usando pandas.
    df = pd.read_csv(file_path)
    
    # Estrae il nome base del file (senza estensione) per poterlo usare per il file di output.
    base_name = os.path.splitext(file_path)[0]

    # Estrae la prima colonna del DataFrame, che contiene le label, e la trasforma in un array.
    labels = df.iloc[:, 0].values

    # Estrae le colonne successive (dati EEG) e le salva in una matrice (array) NumPy.
    eeg_data = df.iloc[:, 1:].values

    # Stampa la forma (dimensioni) del dataset originale.
    print(f"⚡ Dataset originale: {df.shape}")
    
    # Calcola il numero di esempi (righe) presenti nel dataset di dati EEG.
    num_samples = eeg_data.shape[0]
    
    # Calcola il numero atteso di colonne, ovvero il prodotto tra il numero di canali e il numero di campioni per canale.
    expected_columns = num_channels * num_samples_per_channel

    # Se il numero reale di colonne dei dati EEG non corrisponde a quello atteso, solleva un errore.
    if eeg_data.shape[1] != expected_columns:
        raise ValueError(f"Errore dimensioni: atteso {expected_columns} colonne, trovato {eeg_data.shape[1]}.")

    # Inizializza un array NumPy per contenere i dati EEG filtrati.
    # La dimensione è: (numero esempi, numero di canali * campioni per canale * 6),
    # poiché per ogni canale verranno calcolate 6 bande di frequenza.
    eeg_data_filtered = np.zeros((num_samples, num_channels * num_samples_per_channel * 6))

    # Ciclo che processa ogni esempio (cioè ogni riga del dataset EEG).
    for i in range(num_samples):
        # Per l'esempio i-esimo, si "riempone" un vettore piatto di dati in una matrice:
        # si trasforma in una matrice di forma (num_channels, num_samples_per_channel)
        eeg_signals = eeg_data[i].reshape(num_channels, num_samples_per_channel)
        
        # Lista per raccogliere i segnali filtrati per ciascun canale.
        filtered_signals = []
        
        # Ciclo per processare ogni canale (riga della matrice eeg_signals).
        for ch in range(num_channels):
            # Applica il filtro "Notch" per rimuovere il rumore a 50 Hz al segnale del canale ch.
            notch_filtered_signal = notch_filter(eeg_signals[ch, :], freq=50.0, fs=fs)
            
            # Per il segnale già filtrato con il filtro Notch, applica i filtri passa-banda per ogni banda di frequenza.
            # FREQ_BANDS è un dizionario con le coppie (low, high) per ciascuna banda (es. Delta, Theta, etc.).
            # Usa una list comprehension per iterare su tutti i valori del dizionario e applicare il filtro.
            channel_bands = [apply_bandpass(notch_filtered_signal, low, high, fs) for low, high in FREQ_BANDS.values()]
            
            # Concatena i segnali filtrati di tutte le bande per il canale corrente in un unico vettore,
            # e lo aggiunge alla lista dei segnali filtrati.
            filtered_signals.append(np.concatenate(channel_bands))
        
        # Per l'esempio attuale, concatena i vettori di tutti i canali in un unico vettore piatto
        # e lo inserisce nell'array dei dati filtrati alla riga corrispondente.
        eeg_data_filtered[i] = np.concatenate(filtered_signals)

    # Combina le label originali e i dati EEG filtrati in un nuovo DataFrame.
    df_filtered = pd.DataFrame(np.column_stack((labels, eeg_data_filtered)))
    
    # Prepara il nome del file di output aggiungendo "_filtered" al nome base.
    output_file = f"{base_name}_filtered.csv"
    
    # Salva il nuovo DataFrame in un file CSV.
    # index=False significa che non si salva l'indice delle righe.
    # header=False evita di scrivere i nomi delle colonne.
    # na_rep='0' sostituisce i valori mancanti con '0'.
    df_filtered.to_csv(output_file, index=False, header=False, na_rep='0')

    # Stampa la dimensione del dataset filtrato per confermare l'operazione.
    print(f"⚡ Dataset filtrato salvato: {df_filtered.shape}")
    
    # Ritorna il percorso del file di output.
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

    print(f"IMF finali: {imfs.shape[0]} (dovrebbe essere {max_imfs})")
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
    Applica EMD e salva il dataset finale con feature IA, IP, IF.
    """
    df = pd.read_csv(file_path, delimiter=',', skip_blank_lines=False)
    base_name = os.path.splitext(file_path)[0]

    labels = df.iloc[:, 0].values
    eeg_data = df.iloc[:, 1:].values

    print(f"⚡ Dataset filtrato caricato per EMD: {df.shape}")

    num_samples = eeg_data.shape[0]
    num_features_per_signal = num_channels * len(FREQ_BANDS) * max_imfs * 3  

    feature_matrix = np.zeros((num_samples, num_features_per_signal))

    for i in range(num_samples):
        eeg_signals = eeg_data[i].reshape(num_channels, 6, num_samples_per_channel)
        feature_vector = []
        
        for ch in range(num_channels):
            for band in range(6):
                imfs = apply_emd(eeg_signals[ch, band, :], max_imfs=max_imfs)

                if imfs.shape[0] == 0:
                    print(f"⚠ Nessuna IMF per esempio {i}, assegno feature di default.")
                    feature_vector.extend(np.zeros(num_features_per_signal // num_channels))
                    continue

                ia, ip, if_ = extract_hht_features(imfs)
                feature_vector.extend(np.mean(ia, axis=1))
                feature_vector.extend(np.mean(ip, axis=1))
                feature_vector.extend(np.mean(if_, axis=1))

        feature_matrix[i] = feature_vector

    df_emd = pd.DataFrame(np.column_stack((labels, feature_matrix)))
    output_file = f"{base_name}_preproc.csv"
    df_emd.to_csv(output_file, index=False, header=False)

    print(f"⚡ Dataset finale salvato: {df_emd.shape}")

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
