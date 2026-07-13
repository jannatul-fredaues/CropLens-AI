import os, random, shutil

dataset_dir = "dataset_clean"
output_dir = "dataset_split"

splits = ["train","val","test"]
for s in splits:
    os.makedirs(os.path.join(output_dir, s), exist_ok=True)

for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    images = os.listdir(class_path)
    random.shuffle(images)

    n = len(images)
    train_end = int(0.7*n)
    val_end = int(0.85*n)

    split_map = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in split_map.items():
        dest_class = os.path.join(output_dir, split, class_name)
        os.makedirs(dest_class, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(class_path, f), os.path.join(dest_class, f))

print("Dataset split completed!")
