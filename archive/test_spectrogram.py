import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import json, os

with open("data/data_summary.json") as f:
    summary = json.load(f)

audio_path = summary["example_file"]
print(f"Loading: {audio_path}")

audio, sr = librosa.load(audio_path, sr=32000, duration=5)
print(f"Audio loaded: {len(audio)} samples at {sr}Hz")

mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmin=20, fmax=16000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", fmin=20, fmax=16000)
plt.colorbar(format="%+2.0f dB")
species = os.path.basename(os.path.dirname(audio_path))
plt.title(f"Mel Spectrogram — Species: {species}")
plt.tight_layout()
plt.savefig("data/test_spectrogram.png", dpi=150)
print("Spectrogram saved to data/test_spectrogram.png")
print("Run: open data/test_spectrogram.png")
