import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

DATA_DIR = "data/raw"
MODEL_PATH = "model/pest_model.pth"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes

# Manual validation split (robust)
indices = np.arange(len(dataset))
np.random.shuffle(indices)

val_size = int(0.2 * len(dataset))
val_indices = indices[:val_size]

val_dataset = Subset(dataset, val_indices)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Validation samples:", len(val_dataset))
print("Classes:", class_names)

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
