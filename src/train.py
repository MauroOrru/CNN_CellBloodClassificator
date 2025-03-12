import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score

def train_one_epoch(model, dataloader, criterion, optimizer, device):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for images, labels in tqdm(dataloader, desc="Training", leave=False):
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * images.size(0)

    _, preds = torch.max(outputs, dim=1)
    correct += (preds == labels).sum().item()
    total += labels.size(0)

  epoch_loss = running_loss / total
  epoch_acc = correct / total
  return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
  model.eval()
  running_loss = 0.0
  correct = 0
  total = 0

  all_preds = []
  all_labels = []

  with torch.no_grad():
    for images, labels in tqdm(dataloader, desc="Validation", leave=False):
      images = images.to(device)
      labels = labels.to(device)

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

  # Calculate global metrics
  precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
  recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
  f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

  return epoch_loss, epoch_acc, precision, recall, f1, all_preds, all_labels
