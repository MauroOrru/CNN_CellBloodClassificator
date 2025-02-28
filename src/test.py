import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# **Caricare il dataset**
def load_dataset(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def split_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    # Estrae le etichette dal terzo campo (colonna con indice 2)
    y = df.iloc[:, 2].values  # Label (0-9 e -1)
    
    # Definizione degli indici per il segnale EEG:
    start_idx = 788                      # Inizio segnale EEG
    n_channels = 64
    n_samples = 400
    end_idx = start_idx + n_channels * n_samples  # Fine segnale EEG

    # Estrae la porzione di dati corrispondente al segnale EEG ed effettua il reshape in (N, 64, 400)
    X = df.iloc[:, start_idx:end_idx].values.reshape(-1, n_channels, n_samples)

    # Rimuove le righe in cui la label è -1
    valid_mask = y != -1
    X = X[valid_mask]
    y = y[valid_mask]

    # Divisione train/test con stratificazione sulle etichette
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Converte i dati in float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Converte le etichette in interi
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    return (X_train, y_train), (X_test, y_test)

# **Creazione del modello CNN in PyTorch**
class BrainDigiCNN(nn.Module):
    def __init__(self, num_classes=11):
        super(BrainDigiCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=7, padding="same")
        self.bn1 = nn.BatchNorm1d(256)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=7, padding="same")
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=7, padding="same")
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=7, padding="same")
        self.bn4 = nn.BatchNorm1d(32)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        # Layer fully connected: la dimensione di input verrà ora fissata a 32x25=800 grazie all'adaptive pooling nel forward
        self.fc1 = nn.Linear(32 * 25, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        # Adattiamo la dimensione lungo la dimensione spaziale a 25
        x = nn.functional.adaptive_avg_pool1d(x, 25)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# **Funzione di training**
def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # **Valutazione sul test set**
        model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_test += (predicted == labels).sum().item()
                total_test += labels.size(0)

        test_loss /= total_test
        test_acc = correct_test / total_test

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")

# **Main**
if __name__ == "__main__":
    dataset_csv_path = r"C:\Users\Mauro\Desktop\Mauro\Universita\AI\Progetto\dataset.csv"  
    (X_train, y_train), (X_test, y_test) = split_dataset(load_dataset(dataset_csv_path))
    
    # Converto in tensori PyTorch
    X_train_tensor = torch.tensor(X_train).permute(0, 2, 1)  # (N, 1, 400)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test).permute(0, 2, 1)    # (M, 1, 400)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # **Dataset e DataLoader**
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # --------------------------- PLOTTING ---------------------------
    # Seleziono ad esempio il primo campione del training set e del test set
    sample_idx = 0
    train_sample = X_train_tensor[sample_idx].squeeze().cpu().numpy()  # Rimuove la dimensione del canale
    test_sample = X_test_tensor[sample_idx].squeeze().cpu().numpy()

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(train_sample)
    plt.title("Campione dal Training Set")
    plt.xlabel("Tempo")
    plt.ylabel("Ampiezza")

    plt.subplot(2, 1, 2)
    plt.plot(test_sample)
    plt.title("Campione dal Test Set")
    plt.xlabel("Tempo")
    plt.ylabel("Ampiezza")

    plt.tight_layout()
    plt.show()
    # -----------------------------------------------------------------

    """
    # **Crea il modello**
    num_classes = 11
    model = BrainDigiCNN(num_classes=num_classes)

    # **Ottimizzatore e Loss**
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # **Addestramento**
    num_epochs = 50
    train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=num_epochs)

    # **Salva il modello**
    torch.save(model.state_dict(), "brain_digi_cnn.pth")
    print("Modello salvato!")
    """

