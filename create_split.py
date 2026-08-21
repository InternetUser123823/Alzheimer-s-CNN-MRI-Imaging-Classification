import os
import shutil
import random

# Settings
SOURCE_DIR = "original_data2"
OUTPUT_DIR = "original_data_independent_split"
TRAIN_RATIO = 0.80
SEED = 42

random.seed(SEED)

# Classes are the subfolder names
classes = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]

# Create output directories
for split in ["train", "test"]:
    for class_name in classes:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, class_name),
            exist_ok=True
        )

# Split each class 80/20
for class_name in classes:

    source_class_dir = os.path.join(SOURCE_DIR, class_name)

    # Get image files only
    images = [
        f for f in os.listdir(source_class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Shuffle using the fixed seed
    random.shuffle(images)

    # Calculate training size
    train_size = int(len(images) * TRAIN_RATIO)

    train_images = images[:train_size]
    test_images = images[train_size:]

    # Copy training images
    for image in train_images:
        source = os.path.join(source_class_dir, image)
        destination = os.path.join(
            OUTPUT_DIR, "train", class_name, image
        )
        shutil.copy2(source, destination)

    # Copy test images
    for image in test_images:
        source = os.path.join(source_class_dir, image)
        destination = os.path.join(
            OUTPUT_DIR, "test", class_name, image
        )
        shutil.copy2(source, destination)

    print(
        f"{class_name}: "
        f"{len(train_images)} train, "
        f"{len(test_images)} test"
    )

print("\nSplit complete!")