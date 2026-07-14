import os
import shutil
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, img_to_array, load_img, array_to_img
)

# ---------------- CONFIG ----------------
INPUT_DIR = "/content/drive/MyDrive/Croplens/Dataset"              # original dataset, organized by class folders
OUTPUT_DIR = "/content/drive/MyDrive/CropLens/Augmented_Dataset"    # where augmented images will be saved
IMAGES_PER_ORIGINAL = 4             # how many augmented copies to generate per image
TARGET_SIZE = (224, 224)            # resize target
# -----------------------------------------

# Safe augmentation ranges — mild enough to avoid distortion/black borders
datagen = ImageDataGenerator(
    rotation_range=15,        # small rotation only
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=0.15,          # mild zoom, avoids cropping out the flower
    brightness_range=[0.85, 1.15],
    horizontal_flip=True,
    fill_mode="reflect",      # avoids black corners from rotation/shift
)


def augment_class_folder(class_name):
    src_folder = os.path.join(INPUT_DIR, class_name)
    dst_folder = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(dst_folder, exist_ok=True)

    image_files = [
        f for f in os.listdir(src_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for filename in image_files:
        img_path = os.path.join(src_folder, filename)
        name, ext = os.path.splitext(filename)

        # Load and resize original, then save a copy into the output folder
        img = load_img(img_path, target_size=TARGET_SIZE)
        img.save(os.path.join(dst_folder, filename))

        # Prepare for augmentation
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Generate augmented versions
        i = 0
        for batch in datagen.flow(x, batch_size=1):
            aug_img = array_to_img(batch[0])
            aug_filename = f"{name}_aug{i}{ext}"
            aug_img.save(os.path.join(dst_folder, aug_filename))
            i += 1
            if i >= IMAGES_PER_ORIGINAL:
                break

    print(f"Done: {class_name} -> {len(image_files)} originals, "
          f"{len(image_files) * IMAGES_PER_ORIGINAL} augmented images created.")


if __name__ == "__main__":
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)  # start clean each run
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_DIR):
        print(f"CRITICAL ERROR: Input directory '{INPUT_DIR}' does not exist just before listing.")
        print("Please ensure your Google Drive is mounted and the path is correct.")
        class_names = [] # Ensure no attempt to list a non-existent directory
    else:
        try:
            class_names = [
                d for d in os.listdir(INPUT_DIR)
                if os.path.isdir(os.path.join(INPUT_DIR, d))
            ]
        except FileNotFoundError:
            print(f"ERROR: The input directory '{INPUT_DIR}' was not found during os.listdir.")
            print("This might be a transient Google Drive issue. Please try re-running this cell.")
            class_names = []
        except Exception as e:
            print(f"An unexpected error occurred while listing {INPUT_DIR}: {e}")
            class_names = []

    if not class_names:
        print("No classes found in the input directory. Augmentation cannot proceed.")
    else:
        for class_name in class_names:
            augment_class_folder(class_name)

        print("\nAugmentation complete. Augmented dataset saved to:", OUTPUT_DIR)