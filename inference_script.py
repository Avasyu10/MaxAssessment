import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def run_inference(npz_path, model_path="best_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(npz_path)
    x = data["x"]
    y = data["y"].flatten()

    if x.ndim == 4:
        x = np.mean(x, axis=-1)

    x = x.astype(np.float32) / 255.0

    x = torch.tensor(x).unsqueeze(1).to(device) 
    y = torch.tensor(y).to(device)

    model = CNN(num_classes=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
  
    correct, total = 0, 0
    with torch.no_grad():
        outputs = model(x)
        preds = outputs.argmax(dim=1)
        correct = (preds == y).sum().item()
        total = y.size(0)

    accuracy = correct / total
    return accuracy

acc = run_inference("new_test_file.npz", "best_model.pth")
print("Test Accuracy:", acc)


