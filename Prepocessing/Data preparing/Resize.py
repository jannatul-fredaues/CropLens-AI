from PIL import Image
import os

input_folder = "/content/sample_data/Black"
output_folder = "/content/sample_data/Black_resized"
os.makedirs(output_folder, exist_ok=True)

SIZE = (224, 224)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        path = os.path.join(input_folder, filename)
        img = Image.open(path)
        img = img.resize(SIZE)
        img.save(os.path.join(output_folder, filename))

print("Resizing completed!")
