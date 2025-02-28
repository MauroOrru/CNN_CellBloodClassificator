import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import timm  # Per caricare modelli pre-addestrati (ResNet, ResNeXt, ecc.)

# ---- PARAMETRI ----
DATASET_PATH = r"C:\Users\Mauro\Desktop\Mauro\Universita\AI\DATASETS\LEUCEMIA\PKG-AML-Cytomorphology_LMU"
TARGET_IMAGES_PER_CLASS = 400
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 400
TEST_SPLIT = 0.2

MODEL_NAME = "resnet18"  # Cambia in "resnext50_32x4d" se vuoi provare ResNeXt
NUM_WORKERS = 0  # Aumenta se il tuo sistema supporta data loading in parallelo

# ---- DATA AUGMENTATION ----
def get_augmentations():
    """
    Restituisce una Compose di Albumentations da applicare come data augmentation.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=180, p=0.5),
        A.RandomBrightnessContrast(p=0.1),
        A.Resize(IMG_SIZE, IMG_SIZE),  # Assicuriamoci che l'immagine sia 400x400
        ToTensorV2()
    ])

# ---- DATASET CON BILANCIAMENTO SOLO PER CLASSI < TARGET_IMAGES_PER_CLASS ----
class BloodCellDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_count=400):
        """
        root_dir: cartella principale che contiene le sottocartelle per ciascuna classe.
        transform: pipeline di trasformazioni/augmentations (Albumentations).
        target_count: numero minimo di immagini per classe (oversampling se < target_count).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_count = target_count

        # Carica le immagini suddivise per classe in un dizionario {class_name: [liste di path]}
        self.class_images = self._load_images_per_class()
        self.classes = list(self.class_images.keys())
        self.num_classes = len(self.classes)

        if not self.class_images:
            raise ValueError(f"Errore: Nessuna immagine trovata in {root_dir}")

        self.image_paths = []
        self.labels = []

        # Oversampling (se una classe ha meno di target_count immagini)
        for class_idx, class_name in enumerate(self.classes):
            class_imgs = self.class_images[class_name]
            if len(class_imgs) < self.target_count:
                class_imgs.extend(random.choices(class_imgs, k=self.target_count - len(class_imgs)))

            # Aggiungi tutti i path (anche quelli oversamplati) alla lista globale
            for img_path in class_imgs:
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

        self.total_images = len(self.image_paths)

    def _load_images_per_class(self):
        class_images = {}
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                images = [
                    os.path.join(class_path, img)
                    for img in os.listdir(class_path)
                    if img.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))
                ]
                if images:
                    class_images[class_name] = images
        return class_images

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            # Se per qualche ragione l'immagine non si legge, creiamo un placeholder nero
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            # Albumentations richiede img come dict { "image": ... }
            transformed = self.transform(image=img)
            img = transformed["image"]
        else:
            # In caso di fallback, ridimensioniamo e convertiamo in tensor
            # (ma idealmente useremmo sempre Albumentations)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = T.ToTensor()(img)

        return img.float(), label


# ---- MODELLO (RESNET o RESNEXT) ----
class CustomModel(nn.Module):
    def __init__(self, num_classes, model_name="resnet18"):
        super(CustomModel, self).__init__()
        # Crea un modello timm (pretrained=True carica i pesi ImageNet)
        self.model = timm.create_model(model_name, pretrained=False)

        # Identifichiamo l'ultimo layer FC (varia a seconda del modello)
        if hasattr(self.model, 'fc'):  # ad es. ResNet standard
            in_feats = self.model.fc.in_features
            self.model.fc = nn.Linear(in_feats, num_classes)
        elif hasattr(self.model, 'classifier'):  # per alcuni modelli timm
            in_feats = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_feats, num_classes)
        else:
            raise ValueError("Modello TIMM non supportato in automatico. Serve personalizzazione.")

    def forward(self, x):
        return self.model(x)

# ---- FUNZIONI DI TRAIN E VALIDAZIONE ----
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        # Calcolo accuracy
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    # Altre metriche
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss, epoch_acc, precision, recall, f1, all_preds, all_labels


def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def plot_normalized_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # Normalizzazione per riga
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.show()


def main():
    # ---- CREAZIONE DATASET ----
    transform = get_augmentations()
    dataset = BloodCellDataset(DATASET_PATH, transform=transform, target_count=TARGET_IMAGES_PER_CLASS)
    num_classes = dataset.num_classes

    # Suddivisione train/test
    train_size = int((1 - TEST_SPLIT) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Numero di classi: {num_classes}")
    print(f"Train set: {train_size} immagini, Test set: {test_size} immagini")

    # ---- INIZIALIZZA MODELLO, LOSS, OPTIM ----
    model = CustomModel(num_classes=num_classes, model_name=MODEL_NAME).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Liste per salvare metriche
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # --- Train ---
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # --- Validation/Test ---
        val_loss, val_acc, precision, recall, f1, preds, labels = validate(model, test_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

    # ---- AL TERMINE, MOSTRA CONFUSION MATRIX SUL TEST SET FINALE ----
    # (usiamo le ultime preds e labels)
    class_names = list(dataset.class_images.keys())  # Recuperiamo i nomi delle cartelle
    plot_confusion_matrix(labels, preds, class_names)
    plot_normalized_confusion_matrix(labels, preds, class_names)

    # ---- GRAFICI DI LOSS E ACCURACY ----
    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs_range, train_accuracies, label='Train Acc')
    plt.plot(epochs_range, val_accuracies, label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ---- SALVA IL MODELLO ----
    torch.save(model.state_dict(), f"{MODEL_NAME}_leucemia_model.pth")
    print("Modello salvato correttamente.")


if __name__ == '__main__':
    main()
