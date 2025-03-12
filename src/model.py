# model.py

import torch.nn as nn
import timm

class CustomModel(nn.Module):
    """
    Modello generico basato su timm, con ultimo layer personalizzato 
    in base al numero di classi.
    model_name es: 'resnet18', 'resnext50_32x4d', ecc.
    """
    def __init__(self, num_classes, model_name="resnet18", pretrained=False):
        super(CustomModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

        # Controlliamo se il modello ha attributo fc o classifier (dipende dal modello timm)
        if hasattr(self.model, 'fc'):
            in_feats = self.model.fc.in_features
            self.model.fc = nn.Linear(in_feats, num_classes)
        elif hasattr(self.model, 'classifier'):
            in_feats = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_feats, num_classes)
        else:
            raise ValueError("Modello TIMM non supportato.")

    def forward(self, x):
        return self.model(x)
