import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset

class BloodCellDataset(Dataset):
    def __init__(self, root_dir, valid_folders=None, img_size=400, transform=None):
        super().__init__()
        self.root_dir = root_dir  # Main path to the dataset 
        self.valid_folders = valid_folders if valid_folders else [] # List of valid folders (if None, all folders are valid)
        self.img_size = img_size # Image size
        self.transform = transform  

        self.class_images = self._load_images_per_class()
        self.classes = list(self.class_images.keys()) # List of classes
        self.num_classes = len(self.classes) # Number of classes

        # Check if there are images in the folders
        if not self.class_images:
            raise ValueError(f"No images founded in {self.valid_folders}")

        self.image_paths = [] # List of image paths
        self.labels = [] # List of labels

        for class_idx, class_name in enumerate(self.classes):
            for img_path in self.class_images[class_name]:
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

        self.total_images = len(self.image_paths)

    # Load images per class. Return a dictionary with the class name as key and a list of images as value
    def _load_images_per_class(self):
        class_images = {} # Empty dictionary
        for class_name in os.listdir(self.root_dir): # List of folders in the root_dir
            if self.valid_folders and class_name not in self.valid_folders: # If valid_folders is not None, we check if the folder is in the list
                continue # Skip the folder
            class_path = os.path.join(self.root_dir, class_name) # Path to the folder
            if os.path.isdir(class_path):
                imgs = [
                    # List comprehension to get all images in the folder
                    os.path.join(class_path, f) 
                    for f in os.listdir(class_path) 
                    if f.lower().endswith(('.png','.jpg','.jpeg','.tiff'))
                ]
                if imgs:
                    class_images[class_name] = imgs # Add the list of images to the dictionary
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
