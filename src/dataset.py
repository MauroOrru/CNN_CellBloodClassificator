import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset

class BloodCellDataset(Dataset):
    def __init__(self, root_dir, valid_folders=None, img_size=400, transform=None):
        super().__init__() 
        self.root_dir = root_dir  
        self.valid_folders = valid_folders if valid_folders else [] 
        self.img_size = img_size 
        self.transform = transform  
        
        self.class_images = self._load_images_per_class() # key: labels, value: list of paths of the images for that label
        self.classes = list(self.class_images.keys()) # List of the labels
        self.num_classes = len(self.classes) # Number of classes

        if not self.class_images: 
            raise ValueError(f"No images founded in {self.valid_folders}") 

        self.image_paths = [] # Contains all the paths of the generated dataset
        self.labels = []      # Contains the labels (enumerated) for each image in self.image_paths

        for class_idx, class_name in enumerate(self.classes):
            for img_path in self.class_images[class_name]: 
                self.image_paths.append(img_path)  
                self.labels.append(class_idx) 

        self.total_images = len(self.image_paths) # Total number of images in the dataset

    # Returns a dictionary whose keys are the classes and the values are the paths of the examples belonging to those classes
    def _load_images_per_class(self):
        class_images = {} 
        for class_name in os.listdir(self.root_dir): 
            if self.valid_folders and class_name not in self.valid_folders: 
                continue 
            class_path = os.path.join(self.root_dir, class_name) # Generation of the class path
            if os.path.isdir(class_path): 
                imgs = [] # List of the paths of the images of the specific class
                for f in os.listdir(class_path): 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')): # Check if the file is an image
                        full_path = os.path.join(class_path, f) # Path of the image
                        imgs.append(full_path) 
                if imgs:
                    class_images[class_name] = imgs 
        return class_images

    def __len__(self):
        return self.total_images
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx] # Retrieve the image path
        label = self.labels[idx] # Retrieve the image label

        img = cv2.imread(img_path, cv2.IMREAD_COLOR) # Read the image with OpenCV
        if img is None: # If the image was not read correctly
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8) # Create a black image
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the image from BGR to RGB

        if self.transform:
            augmented = self.transform(image=img) # Apply transformations to the image. Returns a dictionary with key "image" and value the transformed image
            img = augmented["image"]  # Retrieve the transformed image from the dictionary "augmented"
        else:
            # Convert to PyTorch Tensor
            img = cv2.resize(img, (self.img_size, self.img_size)) # Resize the image to the specified size 
            img = T.ToTensor()(img) # Convert the image to a PyTorch tensor (in our case 3,400,400) where each pixel is a float between 0 and 1

        return img, label
