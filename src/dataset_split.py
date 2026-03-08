import os
import random
import shutil

# dataset location
source_dir = r"C:\Work\Malaria_Detection\archive\cell_images"

# output folder
output_dir = r"C:\Work\Malaria_Detection\dataset"

classes = ["Parasitized", "Uninfected"]

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

for cls in classes:

    class_path = os.path.join(source_dir, cls)
    images = os.listdir(class_path)
    random.shuffle(images)

    train_split = int(len(images) * train_ratio)
    val_split = int(len(images) * (train_ratio + val_ratio))

    train_images = images[:train_split]
    val_images = images[train_split:val_split]
    test_images = images[val_split:]

    for dataset_type, dataset_images in zip(
        ["train", "validation", "test"],
        [train_images, val_images, test_images]
    ):

        path = os.path.join(output_dir, dataset_type, cls)
        os.makedirs(path, exist_ok=True)

        for img in dataset_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(path, img)
            shutil.copy(src, dst)

print("Dataset successfully split!")