import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNeXt50_32X4D_Weights
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from tqdm import tqdm

# ---- PARAMETRI ----
DATASET_PATH = r"C:\Users\Mauro\Desktop\Mauro\Universita\AI\DATASETS\LEUCEMIA\PKG-AML-Cytomorphology_LMU"
TARGET_IMAGES_PER_CLASS = 1000  # Numero di immagini desiderato per classe
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "resnext50_weights.pth"  # Percorso per salvare i pesi

# ---- DATA AUGMENTATION ----
def get_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=360, p=0.8),
        A.RandomBrightnessContrast(p=0.2),
        A.ElasticTransform(p=0.2, alpha=1, sigma=50),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=0, p=0.5),
        A.GaussNoise(p=0.2),
        ToTensorV2()
    ])

# ---- CUSTOM DATASET ----
class BloodCellDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False, target_count=TARGET_IMAGES_PER_CLASS):
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.target_count = target_count
        self.class_images = self._load_images_per_class()
        self.classes = list(self.class_images.keys())
        self.total_images = sum(len(imgs) for imgs in self.class_images.values())

        if self.total_images == 0:
            raise ValueError(f"Errore: Nessuna immagine trovata nel dataset {root_dir}")

        print(f"Dataset caricato con successo: {self.total_images} immagini totali.")
        for cls, imgs in self.class_images.items():
            print(f"Classe {cls}: {len(imgs)} immagini")

    def _load_images_per_class(self):
        class_images = {}
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
                if images:
                    class_images[class_name] = images
                else:
                    print(f"⚠️ Attenzione: Nessuna immagine trovata per la classe {class_name}!")
        return class_images

    def __len__(self):
        return self.total_images

    def __getitem__(self, index):
        class_name = random.choice(self.classes)
        images = self.class_images[class_name]
        
        img_path = random.choice(images)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"❌ Errore nel caricamento dell'immagine: {img_path}")
            img = np.zeros((400, 400, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"] if self.transform else transforms.ToTensor()(img)
        img = img.to(torch.float32)  # FIX: Convertire in FloatTensor

        label = self.classes.index(class_name)
        return img, label

# ---- CREAZIONE DATASET ----
transform = get_augmentations()
dataset = BloodCellDataset(DATASET_PATH, transform=transform, augment=True)

if len(dataset) == 0:
    raise ValueError("Errore: il dataset è vuoto! Controlla il percorso e il formato delle immagini.")

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ---- RESNEXT-50 MODELLO (EVITA DOWNLOAD CONTINUO) ----
def get_resnext_model(num_classes, model_path=MODEL_PATH):
    model = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)

    # Se il modello è già salvato, carica i pesi
    if os.path.exists(model_path):
        print("🔄 Caricamento dei pesi salvati...")
        model.load_state_dict(torch.load(model_path))
    else:
        print("⬇️ Scaricamento e salvataggio dei pesi del modello...")
        torch.save(model.state_dict(), model_path)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

model = get_resnext_model(num_classes=len(dataset.classes))

# ---- PERDITA E OTTIMIZZATORE ----
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---- TRAINING LOOP ----
def train_model(model, train_loader, epochs, criterion, optimizer):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=total_loss/total, acc=100.*correct/total)

if __name__ == '__main__':
    train_model(model, train_loader, EPOCHS, criterion, optimizer)
