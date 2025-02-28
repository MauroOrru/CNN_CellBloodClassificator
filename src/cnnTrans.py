import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import h5py
from torch.utils.data import DataLoader, Dataset, random_split
import logging

# Configurazione del logging per messaggi di debug
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
DEBUG = True  # Imposta a True per abilitare i messaggi di debug

# ================================================================
# 1) Dataset: Lettura dei dati EEG da file .mat (v7.3)
#    - Il file .mat ha shape (N, 64513):
#         • colonna 0: etichette
#         • colonne 1..64512: features da rimodellare in (252, 256)
# ================================================================
class EEGMatDataset(Dataset):
    def __init__(self, mat_path, variable_name='df_emd'):
        """
        Args:
            mat_path      : percorso del file .mat
            variable_name : nome della variabile (es. 'df_emd')
        """
        super(EEGMatDataset, self).__init__()
        self.mat_path = mat_path
        self.variable_name = variable_name
        
        with h5py.File(self.mat_path, 'r') as f:
            data = f[self.variable_name][:]
            data = np.transpose(data)
        self.labels = data[:, 0].astype(np.int64)
        self.features = data[:, 1:].astype(np.float32)
        # Rimodella in (N, 252, 256) poiché 14*6*3 = 252
        self.features = self.features.reshape((-1, 252, 256))
        if DEBUG:
            logging.debug(f"Dataset caricato: {self.features.shape[0]} campioni, ogni campione {self.features.shape[1:]}")

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        x = self.features[idx]  # shape: (252, 256)
        y = self.labels[idx]
        return x, y

# ================================================================
# 2) Positional Encoding per i Transformer
# ================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        if DEBUG:
            logging.debug(f"[PositionalEncoding] Input shape: {x.shape}")
        out = x + self.pe[:, :x.size(1), :]
        if DEBUG:
            logging.debug(f"[PositionalEncoding] Output shape: {out.shape}")
        return out

# ================================================================
# 3) Residual Block con Squeeze-and-Excitation (SE)
# ================================================================
class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, reduction=8):
        super(ResidualSEBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size),
                               stride=(1, stride), padding=(0, padding), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size),
                               stride=1, padding=(0, padding), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(out_channels, out_channels // reduction, bias=False)
        self.fc2 = nn.Linear(out_channels // reduction, out_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, stride), bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        if DEBUG:
            logging.debug(f"[ResidualSEBlock] Input shape: {x.shape}")
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        b, c, _, _ = out.size()
        w = self.global_pool(out).view(b, c)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w).view(b, c, 1, 1)
        out = out * w + self.shortcut(x)
        out = self.relu(out)
        if DEBUG:
            logging.debug(f"[ResidualSEBlock] Output shape: {out.shape}")
        return out

# ================================================================
# 4) UltraCNNBranch (versione ridotta)
# ================================================================
class UltraCNNBranch(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, num_blocks=4, out_dim=256):
        super(UltraCNNBranch, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, base_channels, kernel_size=(1, 7),
                                stride=(1, 2), padding=(0, 3), bias=False))
        layers.append(nn.BatchNorm2d(base_channels))
        layers.append(nn.ReLU(inplace=True))
        channels = base_channels
        for i in range(num_blocks):
            out_channels = channels * 2 if i % 2 == 0 else channels
            stride = 2 if i % 3 == 0 else 1
            layers.append(ResidualSEBlock(channels, out_channels, kernel_size=3, stride=stride, reduction=8))
            channels = out_channels
            if DEBUG:
                logging.debug(f"[UltraCNNBranch] Dopo blocco {i+1}: {channels} canali")
        self.cnn = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels, out_dim)
    
    def forward(self, x):
        if DEBUG:
            logging.debug(f"[UltraCNNBranch] Input shape: {x.shape}")
        out = self.cnn(x)
        if DEBUG:
            logging.debug(f"[UltraCNNBranch] Shape dopo CNN: {out.shape}")
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        if DEBUG:
            logging.debug(f"[UltraCNNBranch] Shape dopo pooling: {out.shape}")
        out = self.fc(out)
        if DEBUG:
            logging.debug(f"[UltraCNNBranch] Output shape: {out.shape}")
        return out

# ================================================================
# 5) UltraTransformerBranch (versione ridotta)
# ================================================================
class UltraTransformerBranch(nn.Module):
    def __init__(self, input_dim=252, seq_len=256, embed_dim=256, num_heads=4, num_layers=4, out_dim=256, dropout=0.1):
        super(UltraTransformerBranch, self).__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, out_dim)
    
    def forward(self, x):
        if DEBUG:
            logging.debug(f"[UltraTransformerBranch] Input shape: {x.shape}")
        x = x.permute(0, 2, 1)  # (batch, 256, 252)
        if DEBUG:
            logging.debug(f"[UltraTransformerBranch] Dopo permute: {x.shape}")
        x = self.input_proj(x)  # (batch, 256, embed_dim)
        if DEBUG:
            logging.debug(f"[UltraTransformerBranch] Dopo input_proj: {x.shape}")
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, 256, embed_dim)
        if DEBUG:
            logging.debug(f"[UltraTransformerBranch] Dopo transformer: {x.shape}")
        x = x.mean(dim=1)  # pooling medio sui timestep
        if DEBUG:
            logging.debug(f"[UltraTransformerBranch] Dopo pooling: {x.shape}")
        x = self.fc(x)
        if DEBUG:
            logging.debug(f"[UltraTransformerBranch] Output shape: {x.shape}")
        return x

# ================================================================
# 6) AttentionModule (invariato)
# ================================================================
class AttentionModule(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionModule, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        attn_weights = self.attn(x)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        if DEBUG:
            logging.debug(f"[AttentionModule] Attn weights shape: {attn_weights.shape}")
        weighted = x * attn_weights
        output = weighted.sum(dim=1)
        if DEBUG:
            logging.debug(f"[AttentionModule] Output shape: {output.shape}")
        return output

# ================================================================
# 7) UltraLSTMBranch (versione ridotta)
# ================================================================
class UltraLSTMBranch(nn.Module):
    def __init__(self, input_dim=252, hidden_dim=128, num_layers=2, bidirectional=True, dropout=0.3, out_dim=256):
        super(UltraLSTMBranch, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, 
                            bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.attention = AttentionModule(hidden_dim * 2 if bidirectional else hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, out_dim)
    
    def forward(self, x):
        if DEBUG:
            logging.debug(f"[UltraLSTMBranch] Input shape: {x.shape}")
        x = x.permute(0, 2, 1)  # (batch, 256, 252)
        if DEBUG:
            logging.debug(f"[UltraLSTMBranch] Dopo permute: {x.shape}")
        lstm_out, _ = self.lstm(x)
        if DEBUG:
            logging.debug(f"[UltraLSTMBranch] LSTM output shape: {lstm_out.shape}")
        attn_out = self.attention(lstm_out)
        if DEBUG:
            logging.debug(f"[UltraLSTMBranch] Attention output shape: {attn_out.shape}")
        out = self.fc(attn_out)
        if DEBUG:
            logging.debug(f"[UltraLSTMBranch] Output shape: {out.shape}")
        return out

# ================================================================
# 8) UltraMLPBranch (versione ridotta)
# ================================================================
class UltraMLPBranch(nn.Module):
    def __init__(self, input_shape=(252, 256), out_dim=256):
        super(UltraMLPBranch, self).__init__()
        self.flatten_dim = input_shape[0] * input_shape[1]
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, out_dim)
    
    def forward(self, x):
        if DEBUG:
            logging.debug(f"[UltraMLPBranch] Input shape: {x.shape}")
        x = x.view(x.size(0), -1)
        if DEBUG:
            logging.debug(f"[UltraMLPBranch] After flatten: {x.shape}")
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if DEBUG:
            logging.debug(f"[UltraMLPBranch] Output shape: {x.shape}")
        return x

# ================================================================
# 9) FusionModule (versione ridotta)
# ================================================================
class FusionModule(nn.Module):
    def __init__(self, input_dims, fusion_dim=512, num_heads=4, out_dim=512):
        super(FusionModule, self).__init__()
        total_dim = sum(input_dims)  # Es: 256*4 = 1024
        self.fc_pre = nn.Linear(total_dim, fusion_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.attention = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, batch_first=True)
        self.fc_post = nn.Linear(fusion_dim, out_dim)
    
    def forward(self, x):
        if DEBUG:
            logging.debug(f"[FusionModule] Input shape: {x.shape}")
        fusion = self.fc_pre(x)
        fusion = self.relu(fusion)
        fusion = self.dropout(fusion)
        fusion = fusion.unsqueeze(1)  # (batch, 1, fusion_dim)
        attn_output, _ = self.attention(fusion, fusion, fusion)
        attn_output = attn_output.squeeze(1)  # (batch, fusion_dim)
        out = self.fc_post(attn_output)
        if DEBUG:
            logging.debug(f"[FusionModule] Output shape: {out.shape}")
        return out

# ================================================================
# 10) UltraEEGNet (versione ridotta)
# ================================================================
class UltraEEGNet(nn.Module):
    def __init__(self, num_classes=10):
        super(UltraEEGNet, self).__init__()
        # Ogni ramo restituisce un vettore a 256 dimensioni
        self.cnn_branch = UltraCNNBranch(in_channels=1, base_channels=16, num_blocks=4, out_dim=256)
        self.transformer_branch = UltraTransformerBranch(input_dim=252, seq_len=256,
                                                         embed_dim=256, num_heads=4, num_layers=4,
                                                         out_dim=256, dropout=0.1)
        self.lstm_branch = UltraLSTMBranch(input_dim=252, hidden_dim=128, num_layers=2,
                                           bidirectional=True, dropout=0.3, out_dim=256)
        self.mlp_branch = UltraMLPBranch(input_shape=(252, 256), out_dim=256)
        
        # Concatenazione dei 4 rami: 256*4 = 1024
        self.fusion = FusionModule(input_dims=[256, 256, 256, 256], fusion_dim=512,
                                     num_heads=4, out_dim=512)
        
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        if DEBUG:
            logging.debug(f"[UltraEEGNet] Input shape: {x.shape}")
        cnn_input = x.unsqueeze(1)  # (batch, 1, 252, 256)
        if DEBUG:
            logging.debug(f"[UltraEEGNet] cnn_input shape: {cnn_input.shape}")
        out_cnn = self.cnn_branch(cnn_input)
        out_trans = self.transformer_branch(x)
        out_lstm = self.lstm_branch(x)
        out_mlp = self.mlp_branch(x)
        if DEBUG:
            logging.debug(f"[UltraEEGNet] Branch outputs: CNN {out_cnn.shape}, Transformer {out_trans.shape}, LSTM {out_lstm.shape}, MLP {out_mlp.shape}")
        fused_input = torch.cat([out_cnn, out_trans, out_lstm, out_mlp], dim=1)
        if DEBUG:
            logging.debug(f"[UltraEEGNet] Fused input shape: {fused_input.shape}")
        fusion_out = self.fusion(fused_input)
        x = self.fc1(fusion_out)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        if DEBUG:
            logging.debug(f"[UltraEEGNet] Logits shape: {logits.shape}")
        return logits

# ================================================================
# 11) Funzioni di Training ed Evaluation
# ================================================================
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size
        
        if DEBUG and batch_idx % 10 == 0:
            logging.debug(f"[Train] Batch {batch_idx}: Loss={loss.item():.4f}, Batch Acc={(preds == labels).float().mean().item():.4f}")
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(test_loader):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size
            
            if DEBUG and batch_idx % 10 == 0:
                logging.debug(f"[Eval] Batch {batch_idx}: Loss={loss.item():.4f}")
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy

# ================================================================
# 12) Main: Setup, Training e Salvataggio del Modello
# ================================================================
def main():
    mat_file = r"C:\Users\Mauro\Desktop\Mauro\Universita\AI\Progetto\Dataset\test_filtered_preproc.mat"
    variable_name = "df_emd"
    
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 100  # Puoi ridurre questo valore se vuoi test rapidi
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f"Dispositivo utilizzato: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    full_dataset = EEGMatDataset(mat_file, variable_name=variable_name)
    logging.info(f"Numero di campioni nel dataset: {len(full_dataset)}")
    
    n_samples = len(full_dataset)
    n_train = int(0.8 * n_samples)
    n_test = n_samples - n_train
    train_dataset, test_dataset = random_split(full_dataset, [n_train, n_test])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = UltraEEGNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs} in corso...")
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        logging.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    torch.save(model.state_dict(), "ultra_eegnet_small.pth")
    logging.info("✅ Modello salvato come 'ultra_eegnet_small.pth'")

if __name__ == "__main__":
    main()
