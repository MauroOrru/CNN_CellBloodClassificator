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
        
        self.class_images = self._load_images_per_class() # key: labels, value: list of paths of the images
        self.classes = list(self.class_images.keys()) # List of the classes
        self.num_classes = len(self.classes) 

        if not self.class_images: 
            raise ValueError(f"No images founded in {self.valid_folders}") 

        self.image_paths = [] # Lista contentente tutti i path del dataset valido
        self.labels = [] # Lista delle label (enumerate) per ogni immagine in self.image_paths

        for class_idx, class_name in enumerate(self.classes):
            for img_path in self.class_images[class_name]: # Itero su tutti i path delle immagini per la classe corrente
                self.image_paths.append(img_path)  # Aggiungo il path dell'immagine alla lista
                self.labels.append(class_idx) # Aggiungo la label (enumerata) alla lista

        self.total_images = len(self.image_paths)

    # Crea un dizionaro i cui le chiavi sono le classi e i valori i path degli esempi appartenenti a tali classi
    def _load_images_per_class(self):
        class_images = {} # Dizionario con i path delle immagini per ogni classe
        for class_name in os.listdir(self.root_dir): # Itero su tutte le cartelle nella root_dir che sono rinominate come le label
            if self.valid_folders and class_name not in self.valid_folders: # Se ho specificato delle cartelle valide e la cartella non è "valida"
                continue # Salto alla prossima iterazione
            class_path = os.path.join(self.root_dir, class_name) # Class path è il path della cartella della classe valida
            if os.path.isdir(class_path): # Fa un check se il path è una cartella
                imgs = [] 
                for f in os.listdir(class_path): # Itero su tutti i file nella cartella corrente
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')): # Se il file è un'immagine
                        full_path = os.path.join(class_path, f) # Costruisco il path completo dell'immagine
                        imgs.append(full_path) # Append il path dell'immagine alla lista
                if imgs:
                    class_images[class_name] = imgs # Add the list of images to the dictionary
        return class_images

    # Metodo che restituisce il numero di esempi (immagini p.e.) nel dataset
    def __len__(self):
        return self.total_images
    
    # Metodo che restituisce l'immagine e la label per un determinato indice idx
    def __getitem__(self, idx):
        img_path = self.image_paths[idx] # Recupero il path dell'immagine
        label = self.labels[idx] # Recupero la label dell'immagine

        # Leggiamo con cv2
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) # Leggo l'immagine con OpenCV
        if img is None: # Se l'immagine non è stata letta correttamente
            img = np.zeros((self.img_size,self.img_size,3), dtype=np.uint8) # Creo un'immagine nera
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converto l'immagine da BGR a RGB

        if self.transform:
            augmented = self.transform(image=img) # Applichiamo le trasformazioni all'immagine. Restituisce un dizionario con chiave "image" e valore l'immagine trasformata
            img = augmented["image"]  # Recupero l'immagine trasformata dal dizionario "augmented"
        else:
            # Convertiamo in Tensore PyTorch
            img = cv2.resize(img, (self.img_size, self.img_size)) # Ridimensiono l'immagine alla dimensione specificata
            img = T.ToTensor()(img) # Converto l'immagine in un tensore PyTorch (nel nostro caso 3,400,400) in cui ogni pixel è un float tra 0 e 1

        return img, label
