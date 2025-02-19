import torch
import torch.nn as nn
from torchvision import transforms
from model import MusicGenreCNN
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import GTZANDataset

batch_size = 16
learning_rate = 0.001
num_epochs = 20

dataset = GTZANDataset(root_dir="data/raw", transform=transforms.Resize((128, 128)))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
model = MusicGenreCNN().to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Training complete!")
torch.save(model.state_dict(), "models/music_genre_cnn.pth")
