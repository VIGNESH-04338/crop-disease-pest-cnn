from pathlib import Path
import tensorflow as tf

DATA_DIR = Path("dataset/images/PlantVillage")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

print("Checking dataset path...")

if not DATA_DIR.exists():
    raise SystemExit(f"Dataset not found: {DATA_DIR.resolve()}")

print("Dataset path:", DATA_DIR.resolve())

class_folders = [
folder for folder in DATA_DIR.iterdir()
if folder.is_dir()
]

print("Number of categories:", len(class_folders))
print("\nLoading a small batch of images...")

dataset = tf.keras.utils.image_dataset_from_directory(
DATA_DIR,
image_size=IMG_SIZE,
batch_size=BATCH_SIZE,
shuffle=True,
seed=42
)

images, labels = next(iter(dataset))

print("\nTEST RESULTS")
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Number of classes:", len(dataset.class_names))
print("First five class names:", dataset.class_names[:5])
print("\nDataset test completed successfully!")
