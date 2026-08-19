from pathlib import Path
from PIL import Image
import hashlib
from collections import Counter
import numpy as np
import os

# Analyze original dataset
print("Original Dataset: ")

train_orig = Path("original_data/train")
test_orig = Path("original_data/test")

# Find all image files
train_images = list(train_orig.rglob("*.jpg"))
test_images = list(test_orig.rglob("*.jpg"))

print(f"# Original training images: {len(train_images)}")
print(f"# Original test images: {len(test_images)}")

# Analyze the first image
img = Image.open(train_images[0])

print(f"First image: {train_images[0]}")
print(f"Image dimensions: {img.size}")
print(f"Image mode: {img.mode}")
print(f"Image format: {img.format}")

# Check for class imbalance
train_classes = Counter(path.parent.name for path in train_images)
test_classes = Counter(path.parent.name for path in test_images)

print("Training class counts:")
for class_name, count in train_classes.items():
    print(f"{class_name}: {count}")

print("Test class counts:")
for class_name, count in test_classes.items():
    print(f"{class_name}: {count}")

# Check for all images for consistency

# # Check all training image dimensions
# train_sizes = Counter()
# for path in train_images:
#     with Image.open(path) as img:
#         train_sizes[img.size] += 1

# print("Training image dimensions:")
# for size, count in train_sizes.items():
#     print(f"{size}: {count}")

# # Check all test image dimensions
# test_sizes = Counter()
# for path in test_images:
#     with Image.open(path) as img:
#         test_sizes[img.size] += 1

# print("Test image dimensions:")
# for size, count in test_sizes.items():
#     print(f"{size}: {count}")

# # Check if any images are non-grayscale
# dataset_path = "original_data"

# non_grayscale = []
# total = 0

# for root, _, files in os.walk(dataset_path):
#     for file in files:
#         if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
#             path = os.path.join(root, file)
#             total += 1

#             with Image.open(path) as img:
#                 if img.mode != "L":
#                     non_grayscale.append((path, img.mode))

# print(f"Total images checked: {total}")
# print(f"Non-grayscale images: {len(non_grayscale)}")

# if non_grayscale:
#     print("\nNon-grayscale images:")
#     for path, mode in non_grayscale[:20]:
#         print(f"{mode}: {path}")
# else:
#     print("✓ ALL IMAGES ARE GRAYSCALE")



# # Verify that the original Kaggle dataset matches the Mendeley dataset

# mendeley = Path("mendeley_data")
# kaggle_original = Path("original_data")

# # Returns all images in a dataset
# def get_images(dataset_path):
#     return list(dataset_path.rglob("*.jpg"))

# # Create a hash for each image for comparison
# def image_hash(image_path):
#     with Image.open(image_path) as img:
#         img = img.convert("L")
#         return hashlib.sha256(img.tobytes()).hexdigest()

# # Count images based on their parent folder
# def class_counts(image_paths):
#     return Counter(path.parent.name for path in image_paths)

# # Load data
# mendeley_images = get_images(mendeley)
# kaggle_images = get_images(kaggle_original)

# print("\n=== DATASET CORRESPONDENCE: MENDELEY vs. KAGGLE ===")

# print(f"Mendeley images: {len(mendeley_images)}")
# print(f"Kaggle original images: {len(kaggle_images)}")

# # Comapre class distributions
# mendeley_classes = class_counts(mendeley_images)
# kaggle_classes = class_counts(kaggle_images)

# print("\nClass distributions:")

# print("\nMendeley:")
# for class_name, count in sorted(mendeley_classes.items()):
#     print(f"  {class_name}: {count}")

# print("\nKaggle:")
# for class_name, count in sorted(kaggle_classes.items()):
#     print(f"  {class_name}: {count}")

# # Compare image content
# print("\nComparing image contents...")

# mendeley_hashes = {}

# for image_path in mendeley_images:
#     h = image_hash(image_path)
#     mendeley_hashes[h] = image_path

# matching_images = 0
# kaggle_only_images = []

# for image_path in kaggle_images:
#     h = image_hash(image_path)

#     if h in mendeley_hashes:
#         matching_images += 1
#     else:
#         kaggle_only_images.append(image_path)

# # Results
# print("\n=== RESULTS ===")

# print(f"Mendeley images: {len(mendeley_images)}")
# print(f"Kaggle original images: {len(kaggle_images)}")
# print(f"Matching images: {matching_images}")
# print(f"Kaggle-only images: {len(kaggle_only_images)}")

# if (
#     len(mendeley_images) == len(kaggle_images)
#     and matching_images == len(mendeley_images)
# ):
#     print("\nKaggle original dataset matches the Mendeley dataset.")
# else:
#     print("\nThe datasets are not completely identical.")

#     if kaggle_only_images:
#         print("\nExamples of Kaggle images without a Mendeley match:")
#         for image_path in kaggle_only_images[:10]:
#             print(f"  {image_path}")