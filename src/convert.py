import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

RAW_DATA_DIR = "data/raw/"
SPEC_DATA_DIR = "data/spectrograms/"

os.makedirs(SPEC_DATA_DIR, exist_ok=True)

def audio_to_mel_spectrogram(file_path, sr=22050, n_mels=128, hop_length=512):
    y, sr = librosa.load(file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db

def generate_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path, sr=22050)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(5, 5))
    librosa.display.specshow(log_spectrogram, sr=sr, cmap="magma")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def process_dataset():
    for genre in os.listdir(RAW_DATA_DIR):
        genre_path = os.path.join(RAW_DATA_DIR, genre)
        genre_output_path = os.path.join(SPEC_DATA_DIR, genre)

        if not os.path.isdir(genre_path):
            continue

        os.makedirs(genre_output_path, exist_ok=True)

        for filename in os.listdir(genre_path):
            if filename.endswith(".wav"):
                input_path = os.path.join(genre_path, filename)
                output_filename = filename.replace(".wav", ".png")
                output_path = os.path.join(genre_output_path, output_filename)

                generate_spectrogram(input_path, output_path)
        print(f"✔ Processed: {genre_output_path}")

if __name__ == "__main__":
    process_dataset()