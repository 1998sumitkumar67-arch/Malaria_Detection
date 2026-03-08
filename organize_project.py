import os
import shutil

# Root directory
root = r"C:\Work\Malaria_Detection"

# New folder structure
folders = [
    "data/raw",
    "data/processed",
    "data/split",
    "src",
    "models",
    "outputs/plots",
    "outputs/logs",
    "notebooks"
]

# Create directories
for folder in folders:
    path = os.path.join(root, folder)
    os.makedirs(path, exist_ok=True)

print("Folder structure created.")

# Files to move into src
files_to_src = [
    "DatasetSpliting.py",
    "train_model.py"
]

for file in files_to_src:
    src_path = os.path.join(root, file)
    dst_path = os.path.join(root, "src", file)

    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f"Moved {file} → src/")

# Move archive dataset into raw
archive_path = os.path.join(root, "archive")
raw_path = os.path.join(root, "data", "raw")

if os.path.exists(archive_path):
    shutil.move(archive_path, raw_path)
    print("Moved archive → data/raw/")

# Move dataset split folder
dataset_path = os.path.join(root, "dataset")
split_path = os.path.join(root, "data", "split")

if os.path.exists(dataset_path):
    shutil.move(dataset_path, split_path)
    print("Moved dataset → data/split/")

print("Project successfully organized!")