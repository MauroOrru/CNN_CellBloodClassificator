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
#      - colonne 1..64512 => features (che poi vengono reshape in (21, 170) 
#         e trasposte in (170, 21) per avere sequence length=170 e features=21)
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
        Vengono quindi reshapeate in (21, 170) e trasposte in (170, 21)
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
            max_len : lunghezza massima della sequenza per il calcolo delle posizioni
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Creazione della matrice di positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # Se d_model è dispari, gestiamo l'ultima colonna separatamente
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # forma: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor di input di forma (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ----------------------------------------------------------
# 3) Modello Transformer per EEG
# ----------------------------------------------------------
class EEGTransformer(nn.Module):
    def __init__(self, input_dim=21, d_model=64, nhead=8, num_layers=3, num_classes=2, dropout=0.1):
        """
        Args:
            input_dim  : dimensione in ingresso (numero di canali, es. 21)
            d_model    : dimensione dell'embedding del Transformer
            nhead      : numero di teste di attenzione
            num_layers : numero di layer del TransformerEncoder
            num_classes: numero di classi (es. 2 per classificazione binaria)
            dropout    : dropout rate
        """
        super(EEGTransformer, self).__init__()
        
        # Proiezione iniziale: da input_dim a d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Codifica posizionale
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Dropout finale e layer di classificazione
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        """
        x: tensor di forma (batch, seq_len, input_dim) -> (batch, 170, 21)
        """
        # Proiezione iniziale
        x = self.input_proj(x)  # -> (batch, seq_len, d_model)
        
        # Aggiunta della codifica posizionale
        x = self.pos_encoder(x)
        
        # Passaggio attraverso il Transformer Encoder
        x = self.transformer_encoder(x)  # -> (batch, seq_len, d_model)
        
        # Pooling: media lungo la dimensione della sequenza
        x = torch.mean(x, dim=1)  # -> (batch, d_model)
        
        x = self.dropout(x)
        logits = self.fc(x)  # -> (batch, num_classes)
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
        outputs = model(features)          # logits: (batch, num_classes)
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
    # Path al file .mat (modifica il percorso e il nome della variabile secondo le tue necessità)
    mat_file = r"C:\Users\Mauro\Desktop\Mauro\Universita\AI\Progetto\Dataset\pat_aa.mat"
    variable_name = "ds"  # Cambia in base al nome della variabile nel .mat

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

    # Inizializzazione del modello, criterio di loss e ottimizzatore
    model = EEGTransformer(input_dim=21, d_model=64, nhead=8, num_layers=3, num_classes=num_classes, dropout=0.1).to(device)
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
    torch.save(model.state_dict(), "eeg_transformer.pth")
    print("✅ Modello salvato come 'eeg_transformer.pth'")


if __name__ == "__main__":
    main()
