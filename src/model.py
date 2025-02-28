import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import DataLoader, Dataset, random_split

# ----------------------------------------------------------
# 1) Dataset class for .mat files (v7.3) using h5py
#    We assume:
#      - .mat has shape (N, 64513)
#      - 1st col => labels
#      - next 64512 => 252*256 features
# ----------------------------------------------------------
class EEGMatDataset(Dataset):
    def __init__(self, mat_path, variable_name='df_emd'):
        """
        Args:
            mat_path      : Path to the .mat file
            variable_name : The variable inside the .mat,
                            e.g. 'df_emd'.
        
        The data is assumed to be (N, 64513):
          - column 0 => labels
          - columns [1..64512] => 252*256 flatten
        We'll reshape them as (252, 256).
        """
        super().__init__()
        self.mat_path = mat_path
        self.variable_name = variable_name

        # Read from .mat (HDF5 format if v7.3)
        with h5py.File(self.mat_path, 'r') as f:
            data = f[self.variable_name][:]  # shape (N, 64513)
            data= np.transpose(data)
        # Extract labels and features
        self.labels = data[:, 0].astype(np.int64)
        self.features = data[:, 1:].astype(np.float32)

        # Reshape (N, 64512) -> (N, 252, 256)
        self.features = self.features.reshape((-1, 252, 256))
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        x = self.features[idx]  # shape (252, 256)
        y = self.labels[idx]
        return x, y


# ----------------------------------------------------------
# 2) Refined 1D CNN:
#    - Conv -> BN -> ReLU -> Pool, repeated 4 times
#    - Flatten -> Dense(128) -> ReLU -> Dense(64) -> ReLU -> Dense(10)
#    - Output: logits (no softmax)
#
#    Using kernel_size=7 + padding=3 => length is unchanged by conv
#    so each pool(2) halves the length:
#       256 -> pool -> 128 -> pool -> 64 -> pool -> 32 -> pool -> 16
# ----------------------------------------------------------
class BrainDigiCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(BrainDigiCNN, self).__init__()

        # Because input has shape (252 channels, 256 timesteps),
        # the in_channels for conv1 is 252, out_channels=256, k=7, pad=3
        self.conv1 = nn.Conv1d(in_channels=252, out_channels=256,
                               kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(256)

        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128,
                               kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64,
                               kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=32,
                               kernel_size=7, padding=3)
        self.bn4 = nn.BatchNorm1d(32)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()

        # Flatten
        self.flatten = nn.Flatten()

        # After 4 pools, 256 -> 16 in length
        # out_channels last conv = 32 => final shape: (32, 16) => 512
        self.fc1 = nn.Linear(32 * 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """
        x shape: (batch_size, 252, 256)
        We'll treat dimension 252 as 'channels' and 256 as 'sequence length'
        """
        # block 1: conv1 -> bn1 -> relu -> pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # length halved from 256 to 128

        # block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)  # length halved from 128 to 64

        # block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)  # length halved from 64 to 32

        # block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)  # length halved from 32 to 16

        # flatten: (32, 16) => 512
        x = self.flatten(x)
        # dense layers
        x = self.relu(self.fc1(x))  # => (128,)
        x = self.relu(self.fc2(x))  # => (64,)

        # final logits
        x = self.fc3(x)            # => (num_classes)
        return x  # raw logits (no softmax)


# ----------------------------------------------------------
# 3) Training and Evaluation (using CrossEntropyLoss on logits)
# ----------------------------------------------------------
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)          # shape (batch_size, num_classes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate stats
        batch_size_ = features.size(0)
        total_loss += loss.item() * batch_size_
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += batch_size_

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
            batch_size_ = features.size(0)
            total_loss += loss.item() * batch_size_
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += batch_size_

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy

# ----------------------------------------------------------
# 4) Main
# ----------------------------------------------------------
def main():
    # Path to your .mat file (v7.3 format)
    mat_file = r"C:\Users\Mauro\Desktop\Mauro\Universita\AI\Progetto\Dataset\train_filtered_preproc.mat"
    variable_name = "df_emd"

    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 20
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("CUDA disponibile:", torch.cuda.is_available())
    print("Nome GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nessuna GPU trovata")
    # Load the entire dataset
    full_dataset = EEGMatDataset(mat_file, variable_name=variable_name)


    
    # Split into train/test (80/20 for example)
    n_samples = len(full_dataset)
    n_train = int(0.8 * n_samples)
    n_test = n_samples - n_train
    # 
    print('Ho caricato il dataset GALLO\n')
    train_dataset, test_dataset = random_split(full_dataset, [n_train, n_test])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = BrainDigiCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  # expects raw logits
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Test  Loss: {test_loss:.4f}  | Test  Acc: {test_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), "brain_digi_cnn.pth")
    print("✅ Model saved as 'brain_digi_cnn.pth'")

if __name__ == "__main__":
    main()
