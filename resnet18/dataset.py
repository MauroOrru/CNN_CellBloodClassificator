# dataset.py

import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset

class BloodCellDataset(Dataset):
    """
    Dataset 'puro': raccoglie i path di immagini da valid_folders,
    SENZA fare oversampling o augmentation di default.
    """

    def __init__(self, root_dir, valid_folders=None, img_size=400, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.valid_folders = valid_folders if valid_folders else []
        self.img_size = img_size
        self.transform = transform  # Se vuoi una transform "base", di default None

        # Carichiamo i path divisi per classe
        self.class_images = self._load_images_per_class()
        self.classes = list(self.class_images.keys())
        self.num_classes = len(self.classes)

        if not self.class_images:
            raise ValueError(f"Nessuna immagine trovata in {self.valid_folders}")

        self.image_paths = []
        self.labels = []

        # NESSUN oversampling qui: semplicemente inseriamo i file originali
        for class_idx, class_name in enumerate(self.classes):
            for img_path in self.class_images[class_name]:
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

        self.total_images = len(self.image_paths)

    def _load_images_per_class(self):
        class_images = {}
        for class_name in os.listdir(self.root_dir):
            if self.valid_folders and class_name not in self.valid_folders:
                continue
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                imgs = [
                    os.path.join(class_path, f)
                    for f in os.listdir(class_path)
                    if f.lower().endswith(('.png','.jpg','.jpeg','.tiff'))
                ]
                if imgs:
                    class_images[class_name] = imgs
        return class_images

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Leggiamo con cv2
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.img_size,self.img_size,3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Se transform è None, facciamo un fallback base
        if self.transform:
            # Albumentations si aspetta un dict {"image": ...}
            augmented = self.transform(image=img)
            img = augmented["image"]  # Tensore
        else:
            # Convertiamo in Tensore PyTorch
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = T.ToTensor()(img)

        return img, label
