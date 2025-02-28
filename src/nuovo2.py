import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from torch.utils.data import DataLoader, Dataset, random_split

# ----------------------------------------------------------
# 1) Dataset class per file .mat (v7.3) usando h5py
#    Si assume che il file abbia forma (N, 64513):
#      - colonna 0  => etichette (labels)
#      - colonne 1..64512 => 21*170 features
#    Le features vengono reshapeate in (21, 170) e poi trasposte in (170, 21)
# ----------------------------------------------------------
class EEGMatDataset(Dataset):
    def __init__(self, mat_path, variable_name='df_emd'):
        """
        Args:
            mat_path      : Path al file .mat
            variable_name : Nome della variabile all'interno del file, es. 'df_emd'.
        
        Il file si assume abbia dimensione (N, 64513):
          - colonna 0 => label
          - colonne 1..64512 => 21*170 features
        Vengono reshapeate in (21, 170) e trasposte in (170, 21)
        """
        super().__init__()
        self.mat_path = mat_path
        self.variable_name = variable_name

        # Lettura dal file .mat (HDF5 se v7.3)
        with h5py.File(self.mat_path, 'r') as f:
            data = f[self.variable_name][:]
            data = np.transpose(data)
        
        # Estrazione delle label e delle features
        self.labels = data[:, 0].astype(np.int64)
        self.features = data[:, 1:].astype(np.float32)
        self.features = self.features.reshape((-1, 21, 170))
        self.features = self.features.transpose((0, 2, 1))  # Risultato: (N, 170, 21)

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        x = self.features[idx]  # shape: (170, 21)
        y = self.labels[idx]
        return x, y


# ----------------------------------------------------------
# 2) Positional Encoding per il Transformer
# ----------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Args:
            d_model : dimensione dell'embedding
            dropout : dropout rate
            max_len : lunghezza massima della sequenza
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Creazione della matrice di positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # Gestione della dimensione dispari
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # forma: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor di forma (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ----------------------------------------------------------
# 3) Modello Ibrido CNN + Transformer per EEG
# ----------------------------------------------------------
class HybridEEGNetTransformer(nn.Module):
    def __init__(self, input_channels=21, seq_len=170, cnn_channels=64, d_model=64,
                 nhead=8, num_layers=3, num_classes=2, dropout=0.1):
        """
        Args:
            input_channels: numero di canali in ingresso (es. 21)
            seq_len       : lunghezza della sequenza (es. 170)
            cnn_channels  : numero di canali in uscita dai layer CNN
            d_model       : dimensione dell'embedding per il Transformer
            nhead         : numero di teste nel Transformer
            num_layers    : numero di layer del Transformer Encoder
            num_classes   : numero di classi per la classificazione
            dropout       : dropout rate
        """
        super(HybridEEGNetTransformer, self).__init__()
        
        # --- Front-end CNN ---
        # Ingresso: (batch, seq_len, input_channels) -> lo trasformiamo in (batch, input_channels, seq_len)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=cnn_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.conv2 = nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        self.relu = nn.ReLU()
        
        # Se cnn_channels e d_model sono diversi, proiettiamo le feature
        self.proj = nn.Linear(cnn_channels, d_model) if cnn_channels != d_model else nn.Identity()
        
        # --- Codifica Posizionale ---
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len)
        
        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- Classificatore ---
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        """
        x: Tensor di forma (batch, seq_len, input_channels)  (es. (batch, 170, 21))
        """
        # Applicazione del front-end CNN:
        # Trasposizione per la CNN: (batch, input_channels, seq_len)
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        # Ripristino della forma: (batch, seq_len, cnn_channels)
        x = x.transpose(1, 2)
        # Proiezione a d_model se necessario
        x = self.proj(x)
        
        # Aggiunta della codifica posizionale
        x = self.pos_encoder(x)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)
        
        # Pooling: media lungo la dimensione temporale
        x = torch.mean(x, dim=1)
        
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


# ----------------------------------------------------------
# 4) Funzioni di Training e Valutazione
# ----------------------------------------------------------
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)  # logits: (batch, num_classes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += batch_size

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += batch_size

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


# ----------------------------------------------------------
# 5) Funzione Main
# ----------------------------------------------------------
def main():
    # Specifica il percorso del file .mat e il nome della variabile (modifica secondo le tue necessità)
    mat_file = r"C:\Users\Mauro\Desktop\Mauro\Universita\AI\Progetto\Dataset\pat_aa.mat"
    variable_name = "ds"  # da cambiare se necessario

    batch_size = 200
    learning_rate = 1e-4
    num_epochs = 50
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("CUDA disponibile:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Nome GPU:", torch.cuda.get_device_name(0))
    else:
        print("Nessuna GPU trovata")

    # Caricamento del dataset
    full_dataset = EEGMatDataset(mat_file, variable_name=variable_name)
    print('Dataset caricato correttamente.')

    # Split del dataset in training (80%) e test (20%)
    n_samples = len(full_dataset)
    n_train = int(0.8 * n_samples)
    n_test = n_samples - n_train
    train_dataset, test_dataset = random_split(full_dataset, [n_train, n_test])

    # Creazione dei DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Inizializzazione del modello, loss e ottimizzatore
    model = HybridEEGNetTransformer(input_channels=21, seq_len=170, cnn_channels=64,
                                    d_model=64, nhead=8, num_layers=3,
                                    num_classes=num_classes, dropout=0.1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Ciclo di training
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Test  Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # Salvataggio del modello
    torch.save(model.state_dict(), "hybrid_eeg_transformer.pth")
    print("✅ Modello salvato come 'hybrid_eeg_transformer.pth'")


if __name__ == "__main__":
    main()
