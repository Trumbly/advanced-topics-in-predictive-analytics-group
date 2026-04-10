import pandas as pd
import os, json

DATASET_PATH = os.path.expanduser("~/birdclef-2026")

print("=" * 60)
print("BIRDCLEF 2026 - DATASET EXPLORATION")
print("=" * 60)

train_df = pd.read_csv(f"{DATASET_PATH}/train.csv")
print(f"\ntrain.csv shape: {train_df.shape}")
print(f"Columns: {list(train_df.columns)}")
print(f"\nFirst row:")
print(train_df.iloc[0])

species_list = sorted(train_df["primary_label"].unique().tolist())
print(f"\nTotal unique species: {len(species_list)}")
print(f"First 10: {species_list[:10]}")

train_audio_path = f"{DATASET_PATH}/train_audio"
total_files = sum(len(files) for _, _, files in os.walk(train_audio_path))
print(f"\nTotal audio files: {total_files}")

counts = train_df["primary_label"].value_counts()
print(f"\nMost common: {counts.head(5).to_dict()}")
print(f"Least common: {counts.tail(5).to_dict()}")

first_species = species_list[0]
species_folder = f"{train_audio_path}/{first_species}"
first_file = os.listdir(species_folder)[0]
example_path = f"{species_folder}/{first_file}"
print(f"\nExample audio file: {example_path}")

os.makedirs("data", exist_ok=True)
summary = {
    "num_train": len(train_df),
    "num_species": len(species_list),
    "species_list": species_list,
    "total_audio_files": total_files,
    "dataset_path": DATASET_PATH,
    "train_audio_path": train_audio_path,
    "example_file": example_path,
    "columns": list(train_df.columns)
}
with open("data/data_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\ndata/data_summary.json saved!")
print("=" * 60)
