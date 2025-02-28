import torch.nn as nn
import timm
from resnext18_custom import ResNeXt18  # Importa il modello custom

class CustomModel(nn.Module):
    """
    Modello generico che supporta sia TIMM che il modello custom ResNeXt-18.
    """
    def __init__(self, num_classes, model_name="resnet18", pretrained=False):
        super(CustomModel, self).__init__()
        self.model = ResNeXt18(num_classes=num_classes, cardinality=16, width=2)

    def forward(self, x):
        return self.model(x)
