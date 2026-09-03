import os
import shutil
from pathlib import Path

# ==========================================================
# SOURCE DATASET
# ==========================================================

SOURCE_DATASET = r"C:\Users\shiva\Desktop\Waste Classification\dataset"

# ==========================================================
# OUTPUT DATASET
# ==========================================================

OUTPUT_DATASET = r"C:\Users\shiva\Desktop\Waste Classification\prepared_dataset"

# ==========================================================
# Allowed Image Extensions
# ==========================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

# ==========================================================
# Main Categories
# ==========================================================

MAIN_CLASSES = [
    "Hazardous",
    "Non-Recyclable",
    "Organic",
    "Recyclable"
]

# ==========================================================
# Create Output Folder
# ==========================================================

os.makedirs(OUTPUT_DATASET, exist_ok=True)

# ==========================================================
# Start Processing
# ==========================================================

print("=" * 60)
print("Preparing Dataset...")
print("=" * 60)

total_images = 0

for main_class in MAIN_CLASSES:

    source_path = os.path.join(SOURCE_DATASET, main_class)

    destination_path = os.path.join(OUTPUT_DATASET, main_class)

    os.makedirs(destination_path, exist_ok=True)

    copied = 0

    # Walk through every folder inside the class
    for root, dirs, files in os.walk(source_path):

        for file in files:

            if file.lower().endswith(IMAGE_EXTENSIONS):

                source_file = os.path.join(root, file)

                # Prevent duplicate filenames
                filename = Path(file).stem
                extension = Path(file).suffix

                destination_file = os.path.join(
                    destination_path,
                    file
                )

                counter = 1

                while os.path.exists(destination_file):

                    destination_file = os.path.join(
                        destination_path,
                        f"{filename}_{counter}{extension}"
                    )

                    counter += 1

                shutil.copy2(source_file, destination_file)

                copied += 1
                total_images += 1

    print(f"{main_class:20} : {copied} images")

print("=" * 60)
print(f"Total Images Copied : {total_images}")
print("=" * 60)

print("\nDataset Prepared Successfully!")

print("\nOutput Folder:")
print(OUTPUT_DATASET)