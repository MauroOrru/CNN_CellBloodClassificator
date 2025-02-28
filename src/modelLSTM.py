import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
from torch.utils.data import DataLoader, Dataset, random_split

# ----------------------------------------------------------
# 1) Dataset class per file .mat (v7.3)
#    Assumiamo:
#      - Il file .mat ha shape (N, 64513)
#      - Colonna 0: etichette
#      - Colonne 1..64512: features (che poi verranno reshape in 252 x 256)
# ----------------------------------------------------------
class EEGMatDataset(Dataset):
    def __init__(self, mat_path, variable_name='df_emd'):
        """
        Args:
            mat_path      : Path al file .mat
            variable_name : Nome della variabile (es. 'df_emd')
        
        Si assume che i dati abbiano shape (N, 64513):
          - colonna 0: etichette
          - colonne [1..64512]: features da reshappare in (252, 256)
        """
        super().__init__()
        self.mat_path = mat_path
        self.variable_name = variable_name

        # Apertura file .mat (formato HDF5 per v7.3)
        with h5py.File(self.mat_path, 'r') as f:
            data = f[self.variable_name][:]  # shape (N, 64513)
            # Alcuni file .mat salvati in HDF5 vanno trasposti
            data = np.transpose(data)
            
        # Estrai etichette e features
        self.labels = data[:, 0].astype(np.int64)
        self.features = data[:, 1:].astype(np.float32)
        # Reshape in (N, 252, 256) 
        self.features = self.features.reshape((-1, 252, 256))
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        x = self.features[idx]  # shape (252, 256)
        y = self.labels[idx]
        return x, y

# ----------------------------------------------------------
# 2) Definiamo un blocco Residuale per la branch CNN
# ----------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm1d(out_channels)
        
        # Se il numero di canali o la risoluzione cambia, usiamo un downsample
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ----------------------------------------------------------
# 3) Positional Encoding per la branch Transformer
# ----------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Aggiunge codifica posizionale (sinusoidale) agli embeddings
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

# ----------------------------------------------------------
# 4) Modello Ibrido: CNN + Transformer
# ----------------------------------------------------------
class HybridEEGNet(nn.Module):
    def __init__(self, num_classes=10):
        super(HybridEEGNet, self).__init__()
        # --- Branch CNN ---
        # Input atteso: (batch, 252, 256) trattando 252 come "canali"
        # Conv iniziale per ridurre dimensionalità spaziale
        self.cnn_conv = nn.Conv1d(in_channels=252, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.cnn_bn   = nn.BatchNorm1d(64)
        self.cnn_relu = nn.ReLU(inplace=True)
        # Due blocchi residuali per aumentare i canali e ridurre la lunghezza temporale
        self.resblock1 = ResidualBlock(64, 128, stride=2)   # da (batch, 64, 128) -> (batch, 128, 64)
        self.resblock2 = ResidualBlock(128, 256, stride=2)  # da (batch, 128, 64) -> (batch, 256, 32)
        # Global Average Pooling lungo la dimensione temporale
        self.cnn_gap = nn.AdaptiveAvgPool1d(1)  # output: (batch, 256, 1)

        # --- Branch Transformer ---
        # In questa branch consideriamo i 256 timestep come sequenza di token
        # e ogni token ha 252 features (che proiettiamo in d_model)
        self.transformer_input_proj = nn.Linear(252, 256)  # proiezione: 252 -> 256
        self.pos_encoder = PositionalEncoding(d_model=256, max_len=256)
        # Transformer Encoder: usiamo 4 layer, 8 teste, feedforward=512, dropout=0.1
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        # Dopo il transformer applicheremo una media sui 256 timestep

        # --- Fusion & Classification ---
        # Combiniamo le due branch (ognuna produce un vettore di 256 dimensioni)
        self.fc1 = nn.Linear(256 + 256, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # x shape: (batch, 252, 256)
        # --- CNN branch ---
        cnn_out = self.cnn_conv(x)       # -> (batch, 64, 256/2 = 128)
        cnn_out = self.cnn_bn(cnn_out)
        cnn_out = self.cnn_relu(cnn_out)
        cnn_out = self.resblock1(cnn_out)  # -> (batch, 128, 64)
        cnn_out = self.resblock2(cnn_out)  # -> (batch, 256, 32)
        cnn_out = self.cnn_gap(cnn_out)    # -> (batch, 256, 1)
        cnn_out = cnn_out.squeeze(-1)      # -> (batch, 256)

        # --- Transformer branch ---
        # Per il Transformer trattiamo i 256 timestep come sequenza: (batch, 256, 252)
        transformer_in = x.permute(0, 2, 1)  # cambia shape da (batch, 252, 256) a (batch, 256, 252)
        transformer_in = self.transformer_input_proj(transformer_in)  # -> (batch, 256, 256)
        transformer_in = self.pos_encoder(transformer_in)             # aggiunge positional encoding
        transformer_out = self.transformer_encoder(transformer_in)      # -> (batch, 256, 256)
        transformer_out = transformer_out.mean(dim=1)  # media sui timestep -> (batch, 256)

        # --- Fusion ---
        fused = torch.cat([cnn_out, transformer_out], dim=1)  # (batch, 512)
        fused = self.fc1(fused)   # (batch, 256)
        fused = self.cnn_relu(fused)
        fused = self.dropout(fused)
        logits = self.fc2(fused)  # (batch, num_classes)
        return logits

# ----------------------------------------------------------
# 5) Funzioni di Training e Valutazione
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
# 6) Main: setup, training, valutazione e salvataggio del modello
# ----------------------------------------------------------
def main():
    # Specifica il path al tuo file .mat
    mat_file = r"C:\Users\Mauro\Desktop\Mauro\Universita\AI\Progetto\Dataset\test_filtered_preproc.mat"
    variable_name = "df_emd"

    # Parametri di training
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 50  # puoi aumentare il numero di epoche per un training più approfondito
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("CUDA disponibile:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    
    # Carica l'intero dataset
    full_dataset = EEGMatDataset(mat_file, variable_name=variable_name)
    print('Dataset caricato: {} campioni.'.format(len(full_dataset)))
    
    # Suddividi in training/test (es. 80/20)
    n_samples = len(full_dataset)
    n_train = int(0.8 * n_samples)
    n_test = n_samples - n_train
    train_dataset, test_dataset = random_split(full_dataset, [n_train, n_test])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Istanzia il modello, la loss e l'ottimizzatore
    model = HybridEEGNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Loop di training
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")
    
    # Salva il modello
    torch.save(model.state_dict(), "hybrid_eegnet.pth")
    print("✅ Modello salvato come 'hybrid_eegnet.pth'")

if __name__ == "__main__":
    main()
