import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from convert import audio_to_mel_spectrogram

# Define genre mapping
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']
genre_to_idx = {genre: i for i, genre in enumerate(GENRES)}

class GTZANDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = []

        for genre in os.listdir(root_dir):
            genre_path = os.path.join(root_dir, genre)
            if os.path.isdir(genre_path):
                for file in os.listdir(genre_path):
                    if file.endswith('.wav'):
                        self.file_list.append((os.path.join(genre_path, file), genre_to_idx[genre]))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        mel_spec = audio_to_mel_spectrogram(file_path)

        mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32)

        if mel_spec_tensor.dim() == 1:
            mel_spec_tensor = mel_spec_tensor.unsqueeze(0)

        resize_transform = transforms.Resize((128, 128))
        mel_spec_tensor = resize_transform(mel_spec_tensor.unsqueeze(0))

        return mel_spec_tensor, torch.tensor(label, dtype=torch.long)