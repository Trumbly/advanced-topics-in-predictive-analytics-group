import librosa
import numpy as np
import pandas as pd
import os, json
from tqdm import tqdm

DATASET_PATH = os.path.expanduser("~/birdclef-2026")
OUTPUT_DIR = "data/spectrograms"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("data/data_summary.json") as f:
    summary = json.load(f)

train_df = pd.read_csv(f"{DATASET_PATH}/train.csv")
species_list = summary["species_list"]
num_species = len(species_list)
species_to_idx = {s: i for i, s in enumerate(species_list)}

print(f"Preprocessing {len(train_df)} audio files -> spectrograms...")
print(f"Output: {OUTPUT_DIR}/")

spectrograms = []
labels = []
failed = 0

for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
    audio_path = f"{DATASET_PATH}/train_audio/{row['filename']}"
    try:
        audio, sr = librosa.load(audio_path, sr=32000, duration=10.0, mono=True)
        target_len = 32000 * 5
        if len(audio) > target_len:
            start = (len(audio) - target_len) // 2
            audio = audio[start:start + target_len]
        else:
            audio = np.pad(audio, (0, max(0, target_len - len(audio))))

        mel = librosa.feature.melspectrogram(y=audio, sr=32000, n_mels=128, fmin=20, fmax=16000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_min, mel_max = mel_db.min(), mel_db.max()
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)
        mel_resized = np.array(mel_norm, dtype=np.float32)[:, :256]
        if mel_resized.shape[1] < 256:
            mel_resized = np.pad(mel_resized, ((0,0),(0,256-mel_resized.shape[1])))

        label_vec = np.zeros(num_species, dtype=np.float32)
        if row['primary_label'] in species_to_idx:
            label_vec[species_to_idx[row['primary_label']]] = 1.0

        spectrograms.append(mel_resized)
        labels.append(label_vec)

    except Exception as e:
        failed += 1
        continue

print(f"\nDone! {len(spectrograms)} processed, {failed} failed")
print("Saving numpy arrays (this may take a moment)...")

X = np.array(spectrograms, dtype=np.float32)
y = np.array(labels, dtype=np.float32)
print(f"X shape: {X.shape}  (samples x mel_bins x time_steps)")
print(f"y shape: {y.shape}  (samples x species)")

np.save("data/spectrograms.npy", X)
np.save("data/labels.npy", y)
print("\nSaved: data/spectrograms.npy and data/labels.npy")
print("Preprocessing complete! You can now run the agent.")
