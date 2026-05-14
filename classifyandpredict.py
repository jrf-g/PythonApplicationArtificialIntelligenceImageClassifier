import torch
from torchvision import models, transforms, datasets
from PIL import Image
import json
import os
from tqdm import tqdm
import numpy as np

# -----------------------------
# 1. Define paths and transforms
# -----------------------------
DATA_DIR = "samples/"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 2. Load class names from JSON
# -----------------------------
json_path = os.path.join(DATA_DIR, "classnames.json")
with open(json_path, "r") as f:
    class_names = json.load(f)

print("Classes:", class_names)

# -----------------------------
# 3. Load dataset for evaluation
# -----------------------------
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataset.classes = class_names
dataset.class_to_idx = {name: i for i, name in enumerate(class_names)}

loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# -----------------------------
# 4. Load model
# -----------------------------
model = models.mobilenet_v2(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, len(class_names))
model.load_state_dict(torch.load("image_classifier_mobilenetv2_cpu.pth", map_location="cpu"))
model.eval()

# -----------------------------
# 5. Prediction function
# -----------------------------
def predict_image(path):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        _, idx = torch.max(out, 1)

    return class_names[idx.item()]

# -----------------------------
# 6. Evaluate entire dataset
# -----------------------------
print("\nEvaluating dataset...\n")

all_preds = []
all_labels = []

for img, label in tqdm(loader, desc="Predicting", ncols=80):
    with torch.no_grad():
        out = model(img)
        _, pred = torch.max(out, 1)

    all_preds.append(pred.item())
    all_labels.append(label.item())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# -----------------------------
# 7. Overall accuracy
# -----------------------------
accuracy = (all_preds == all_labels).mean() * 100
print(f"\nOverall Accuracy: {accuracy:.2f}%\n")

# -----------------------------
# 8. Per-class accuracy
# -----------------------------
print("Per-Class Accuracy:")
for i, name in enumerate(class_names):
    idx = (all_labels == i)
    if idx.sum() == 0:
        print(f"  {name}: No samples")
        continue
    acc = (all_preds[idx] == all_labels[idx]).mean() * 100
    print(f"  {name}: {acc:.2f}%")

# -----------------------------
# 9. Confusion Matrix
# -----------------------------
print("\nConfusion Matrix:")
cm = np.zeros((len(class_names), len(class_names)), dtype=int)

for t, p in zip(all_labels, all_preds):
    cm[t, p] += 1

# Pretty print
header = "Pred → " + "  ".join([f"{c[:6]:>6}" for c in class_names])
print(header)
print("-" * len(header))

for i, row in enumerate(cm):
    row_str = "  ".join([f"{v:6d}" for v in row])
    print(f"{class_names[i][:6]:>6} | {row_str}")

# -----------------------------
# 10. Test single image
# -----------------------------
print("\nSingle image test:")
print(predict_image("classify.png"))