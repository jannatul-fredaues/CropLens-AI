import os
from rembg import remove
from PIL import Image

input_folder = "dataset/sample_images"
output_folder = "results/after"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.lower().endswith((".png",".jpg",".jpeg")):

        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file.split('.')[0] + ".png")

        with Image.open(input_path) as img:
            output = remove(img)
            output.save(output_path)

print("Background removal completed.")
