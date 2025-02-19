import torch
from model import MusicGenreCNN
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import GTZANDataset

MODEL_PATH = "../models/music_genre_cnn.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MusicGenreCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

def evaluate(model, dataloader):
    model.eval()
    predictions, actual = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            actual.extend(labels.cpu().numpy())

    accuracy = accuracy_score(actual, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")

dataset = GTZANDataset(root_dir="data/raw", transform=transforms.Resize((128, 128)))
test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
evaluate(model, test_loader)
