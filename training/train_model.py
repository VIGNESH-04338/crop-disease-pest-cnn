from pathlib import Path
import json
import argparse
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "dataset" / "images" / "PlantVillage"
MODEL_DIR = ROOT / "models"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def main():
    parser = argparse.ArgumentParser(
        description="Train a CNN for crop disease classification."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Dataset folder not found: {args.data}")

    print("Loading dataset from:", args.data)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
        label_mode="int"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
        label_mode="int"
    )

    class_names = train_ds.class_names

    if len(class_names) < 2:
        raise SystemExit("At least two class folders are required.")

    print(f"\nNumber of classes: {len(class_names)}")
    print("Classes:", class_names)

    # Prefetch batches without caching the complete dataset in RAM.
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.shuffle(
        1000, seed=SEED, reshuffle_each_iteration=True
    ).prefetch(buffer_size=autotune)

    val_ds = val_ds.prefetch(buffer_size=autotune)

    # Data augmentation
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
    ], name="data_augmentation")

    # CNN backbone with transfer learning
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    outputs = tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_DIR / "crop_classifier.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger(
            ROOT / "training_history.csv"
        )
    ]

    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Save class labels
    class_names_path = MODEL_DIR / "class_names.json"
    class_names_path.write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8"
    )

    loss, accuracy = model.evaluate(val_ds, verbose=1)

    print(f"\nValidation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.2%}")
    print(f"Model saved to: {MODEL_DIR / 'crop_classifier.keras'}")
    print(f"Class names saved to: {class_names_path}")


if __name__ == "__main__":
    main()
