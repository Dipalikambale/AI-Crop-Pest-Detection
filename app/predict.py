import torch
from torchvision import models, transforms
from PIL import Image
import json
from torch import nn

MODEL_PATH = "model/pest_model.pth"
MAPPING_PATH = "rules/pesticide_mapping.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mapping
with open(MAPPING_PATH, "r") as f:
    mapping = json.load(f)

class_names = list(mapping.keys())

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)

    label = class_names[idx.item()]
    result = mapping[label]

    return {
        "class": label,
        "confidence": round(conf.item() * 100, 2),
        "disease": result["disease"],
        "recommended_chemical": result["recommended_chemical"],
        "note": result["note"]
    }

























