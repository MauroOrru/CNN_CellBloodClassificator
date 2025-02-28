import torch
import matplotlib.pyplot as plt

def generate_dataset(input_path, output_path="dataset_raw.txt",
                     start_eeg=788, num_channels=64, sample_per_channel=400):
    """
    Legge il file di testo presente in input_path; per ogni riga:
      - Si assume che la riga sia composta da:
          col0: TRAIN/TEST
          col1: id
          col2: label (classe)
          col3..786: 784 elementi dell'immagine MNIST
          col787: timestemp
          col[start_eeg].. : segnale EEG (num_channels canali per sample_per_channel campioni)
      - Viene creato un nuovo file in output_path in cui per ogni riga
        sono salvati solo la label e il segnale EEG.
        
    Parametri:
      input_path: percorso del file di input
      output_path: percorso del file di output
      start_eeg: indice di partenza nel file per il segnale EEG (default 788)
      num_channels: numero di canali EEG (default 64)
      sample_per_channel: numero di campioni per canale (default 400)
    """
    end_idx = start_eeg + num_channels * sample_per_channel
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            tokens = line.strip().split(",")
            # Verifica che la riga contenga almeno il numero minimo di token
            if len(tokens) < end_idx + 1:
                continue
            label = tokens[2]  # terza colonna = label
            eeg = tokens[start_eeg:end_idx]  # estrai i campioni EEG calcolati
            new_line = label + " " + " ".join(eeg) + "\n"
            fout.write(new_line)

def plot_example(channel_num: int, example_num: int, dataset_path="dataset_raw.txt",
                 num_channels=64, sample_per_channel=400):
    """
    Plotta il segnale EEG del canale specificato per l'esempio indicato.
    
    Parametri:
      - channel_num: numero del canale da plottare (in base a num_channels)
      - example_num: numero dell'esempio (riga nel file, 1-based)
      - dataset_path: percorso del file di testo contenente i dati (default "dataset_raw.txt")
      - num_channels: numero di canali totali presenti nel dataset
      - sample_per_channel: numero di campioni per canale
    
    Ogni riga del file deve avere la struttura:
      label segnale_EEG
    dove segnale_EEG sono num_channels * sample_per_channel valori separati da spazi.
    """
    if channel_num < 1 or channel_num > num_channels:
        print("Il numero del canale deve essere compreso tra 1 e", num_channels)
        return

    with open(dataset_path, "r") as f:
        lines = f.readlines()

    if example_num < 1 or example_num > len(lines):
        print("Il numero dell'esempio è fuori range.")
        return

    line = lines[example_num - 1].strip()
    tokens = line.split()
    # La riga deve avere almeno 1 token (label) + num_channels * sample_per_channel
    if len(tokens) < 1 + num_channels * sample_per_channel:
        print("La riga non contiene il numero atteso di campioni EEG.")
        return

    label = tokens[0]
    eeg_tokens = tokens[1:]
    
    start_idx = (channel_num - 1) * sample_per_channel
    end_idx = start_idx + sample_per_channel
    channel_data_tokens = eeg_tokens[start_idx:end_idx]

    try:
        channel_data = [float(x) for x in channel_data_tokens]
    except ValueError:
        print("Errore nella conversione dei dati del segnale EEG in float.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(channel_data)
    plt.xlabel("Campione")
    plt.ylabel("Valore EEG")
    plt.title(f"Esempio {example_num} - Canale {channel_num} - Label: {label}")
    plt.tight_layout()
    plt.show()

# Esempio d'uso:
if __name__ == "__main__":
    input_path = r"C:\Users\Mauro\Desktop\Mauro\Universita\AI\Progetto\EP1.01.txt"
    output_path = r"C:\Users\Mauro\Desktop\Mauro\Universita\AI\Progetto\dataset_epoc_raw.txt"
    
    # Parametri personalizzati: ad esempio 3 canali da 512 campioni, a partire dalla colonna 788
    generate_dataset(input_path, output_path, start_eeg=788, num_channels=3, sample_per_channel=512)
    
    # Plotta il canale 2 dell'esempio 5 usando gli stessi parametri
    plot_example(channel_num=2, example_num=5, dataset_path=output_path, num_channels=3, sample_per_channel=512)
