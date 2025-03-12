import torch.nn as nn
import timm

class CustomModel(nn.Module):
    def __init__(self, num_classes, model_name="resnet18", pretrained=False):
        super(CustomModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

        # Check if the model has attribute fc or classifier (depends on the timm model)
        if hasattr(self.model, 'fc'):
            in_feats = self.model.fc.in_features
            self.model.fc = nn.Linear(in_feats, num_classes)
        elif hasattr(self.model, 'classifier'):
            in_feats = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_feats, num_classes)
        else:
            raise ValueError("Unsupported TIMM model.")

    def forward(self, x):
        return self.model(x)
