from pathlib import Path
import json
import tensorflow as tf

MODEL_PATH = Path("models/crop_classifier.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")

print("Loading trained model...")

if not MODEL_PATH.exists():
    raise SystemExit("Trained model file not found.")

if not CLASS_NAMES_PATH.exists():
    raise SystemExit("Class names file not found.")

model = tf.keras.models.load_model(MODEL_PATH)

class_names = json.loads(
CLASS_NAMES_PATH.read_text(encoding="utf-8")
)

print("Model loaded successfully!")
print("Number of classes:", len(class_names))

print("\nModel input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

print("\nModel verification completed successfully!")
