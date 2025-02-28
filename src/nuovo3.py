import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from torch.utils.data import DataLoader, Dataset, random_split

# ----------------------------------------------------------
# 1) Dataset class per file .mat (v7.3) usando h5py
#    Si assume che il file abbia forma (N, ...):
#      - colonna 0  => label
#      - colonne 1.. => features
#    Le features vengono reshapeate in (118, 875) e poi trasposte in (875, 118)
#    Risultato: (N, 875, 118) con sequenza di lunghezza 875 e 118 canali.
# ----------------------------------------------------------
class EEGMatDataset(Dataset):
    def __init__(self, mat_path, variable_name='df_emd'):
        """
        Args:
            mat_path      : Path al file .mat
            variable_name : Nome della variabile all'interno del file, es. 'df_emd'.
        
        Il file si assume abbia forma (N, ...):
          - colonna 0 => label
          - colonne 1.. => features
        Le features vengono reshapeate in (118, 875) e trasposte in (875, 118)
        """
        super().__init__()
        self.mat_path = mat_path
        self.variable_name = variable_name

        with h5py.File(self.mat_path, 'r') as f:
            data = f[self.variable_name][:]
            data = np.transpose(data)
        
        self.labels = data[:, 0].astype(np.int64)
        self.features = data[:, 1:].astype(np.float32)
        self.features = self.features.reshape((-1, 118, 875))
        self.features = self.features.transpose((0, 2, 1))  # Ora shape: (N, 875, 118)

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        x = self.features[idx]  # shape: (875, 118)
        y = self.labels[idx]
        return x, y


# ----------------------------------------------------------
# 2) Positional Encoding per Transformer/Conformer
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
        
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # Gestione di d_model dispari
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: Tensor di forma (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ----------------------------------------------------------
# 3) Moduli di base per il Conformer
# ----------------------------------------------------------

# 3.1) FeedForward Module
class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        """
        Args:
            d_model         : dimensione dell'embedding
            expansion_factor: fattore di espansione (solitamente 4)
            dropout         : dropout rate
        """
        super(FeedForwardModule, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# 3.2) Multi-Head Self-Attention Module
class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        """
        Args:
            d_model : dimensione dell'embedding
            nhead   : numero di teste
            dropout : dropout rate
        """
        super(MultiHeadSelfAttentionModule, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                         dropout=dropout, batch_first=True)
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        out, _ = self.mha(x, x, x)
        return out


# 3.3) Convolution Module (ispirato al Conformer)
class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size=15, dropout=0.1):
        """
        Args:
            d_model    : dimensione dell'embedding
            kernel_size: dimensione del kernel per la depthwise convolution
            dropout    : dropout rate
        """
        super(ConvolutionModule, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)  # Divide i canali per 2 e applica un gating mechanism
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                        padding=kernel_size // 2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()  # Swish
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # -> (batch, d_model, seq_len)
        x = self.pointwise_conv1(x)  # -> (batch, 2*d_model, seq_len)
        x = self.glu(x)              # -> (batch, d_model, seq_len)
        x = self.depthwise_conv(x)   # -> (batch, d_model, seq_len)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)  # -> (batch, d_model, seq_len)
        x = x.transpose(1, 2)        # -> (batch, seq_len, d_model)
        return self.dropout(x)


# 3.4) Blocco Conformer
class ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead, conv_kernel_size=15, dropout=0.1,
                 expansion_factor=4):
        """
        Args:
            d_model         : dimensione dell'embedding
            nhead           : numero di teste per l'attenzione
            conv_kernel_size: kernel size per il modulo convolution
            dropout         : dropout rate
            expansion_factor: fattore di espansione per i moduli feed-forward
        """
        super(ConformerBlock, self).__init__()
        self.ffn1 = FeedForwardModule(d_model, expansion_factor, dropout)
        self.ffn2 = FeedForwardModule(d_model, expansion_factor, dropout)
        self.mha = MultiHeadSelfAttentionModule(d_model, nhead, dropout)
        self.conv_module = ConvolutionModule(d_model, kernel_size=conv_kernel_size, dropout=dropout)
        
        self.norm_ffn1 = nn.LayerNorm(d_model)
        self.norm_mha  = nn.LayerNorm(d_model)
        self.norm_conv = nn.LayerNorm(d_model)
        self.norm_ffn2 = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Feed Forward Module (primo) con residuo a metà peso
        x = x + 0.5 * self.dropout(self.ffn1(self.norm_ffn1(x)))
        # Multi-Head Self-Attention con residuo
        x = x + self.dropout(self.mha(self.norm_mha(x)))
        # Convolution Module con residuo
        x = x + self.dropout(self.conv_module(self.norm_conv(x)))
        # Feed Forward Module (secondo) con residuo a metà peso
        x = x + 0.5 * self.dropout(self.ffn2(self.norm_ffn2(x)))
        # Normalizzazione finale
        x = self.final_norm(x)
        return x


# ----------------------------------------------------------
# 4) Modello ConformerEEG
# ----------------------------------------------------------
class ConformerEEG(nn.Module):
    def __init__(self, input_dim=118, seq_len=875, cnn_embed_dim=64, d_model=128,
                 nhead=8, num_blocks=2, num_classes=2, conv_kernel_size=15,
                 dropout=0.1, expansion_factor=4):
        """
        Args:
            input_dim     : numero di canali in ingresso (118)
            seq_len       : lunghezza della sequenza (875)
            cnn_embed_dim : dimensione intermedia del front-end CNN
            d_model       : dimensione dell'embedding per i blocchi Conformer
            nhead         : numero di teste per l'attenzione
            num_blocks    : numero di blocchi Conformer (modificato a 2 per ridurre la memoria)
            num_classes   : numero di classi per la classificazione
            conv_kernel_size: kernel size per il modulo convolution
            dropout       : dropout rate
            expansion_factor: fattore di espansione per i moduli feed-forward
        """
        super(ConformerEEG, self).__init__()
        
        # Front-end CNN per estrarre feature locali dal segnale EEG.
        # Input atteso: (batch, seq_len, input_dim) -> (batch, 875, 118)
        # Per la CNN trasponiamo in (batch, input_dim, seq_len)
        self.cnn_frontend = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_embed_dim,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_embed_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_embed_dim, out_channels=cnn_embed_dim,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_embed_dim),
            nn.ReLU()
        )
        # Proiezione per ottenere dimensione d_model (se necessario)
        self.proj = nn.Linear(cnn_embed_dim, d_model) if cnn_embed_dim != d_model else nn.Identity()
        
        # Codifica posizionale: max_len deve essere almeno pari a seq_len (875)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len)
        
        # Stack di blocchi Conformer
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, nhead, conv_kernel_size, dropout, expansion_factor)
            for _ in range(num_blocks)
        ])
        
        # Pooling globale e testa di classificazione
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) => (batch, 875, 118)
        x = x.transpose(1, 2)  # -> (batch, input_dim, seq_len)
        x = self.cnn_frontend(x)  # -> (batch, cnn_embed_dim, seq_len)
        x = x.transpose(1, 2)      # -> (batch, seq_len, cnn_embed_dim)
        x = self.proj(x)           # -> (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        
        for block in self.conformer_blocks:
            x = block(x)
        
        # Pooling globale lungo la dimensione temporale
        x = x.transpose(1, 2)              # (batch, d_model, seq_len)
        x = self.global_pool(x)            # (batch, d_model, 1)
        x = x.squeeze(-1)                  # (batch, d_model)
        logits = self.fc(x)                # (batch, num_classes)
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
# 6) Funzione Main
# ----------------------------------------------------------
def main():
    # Specifica il percorso del file .mat e il nome della variabile (modifica se necessario)
    mat_file = r"C:\Users\Mauro\Desktop\Mauro\Universita\AI\Progetto\Dataset\pat_aa2.mat"
    variable_name = "ds"  # Aggiorna se necessario
    
    batch_size = 50       # Batch size ridotto per contenere il consumo di memoria
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
    print("Dataset caricato correttamente.")
    
    # Split del dataset: 80% train, 20% test
    n_samples = len(full_dataset)
    n_train = int(0.8 * n_samples)
    n_test = n_samples - n_train
    train_dataset, test_dataset = random_split(full_dataset, [n_train, n_test])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Inizializzazione del modello con parametri aggiornati
    model = ConformerEEG(
        input_dim=118,    # aggiornato da 21 a 118
        seq_len=875,      # aggiornato da 170 a 875
        cnn_embed_dim=64,
        d_model=128,
        nhead=8,
        num_blocks=2,     # ridotto da 4 per ridurre il consumo di memoria
        num_classes=num_classes,
        conv_kernel_size=15,
        dropout=0.1,
        expansion_factor=4
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Ciclo di training
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc   = evaluate_model(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Test  Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    # Salvataggio del modello
    torch.save(model.state_dict(), "conformer_eeg.pth")
    print("✅ Modello salvato come 'conformer_eeg.pth'")


if __name__ == "__main__":
    main()
