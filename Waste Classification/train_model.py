import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam


# ==========================================================
# Paths
# ==========================================================
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "prepared_dataset"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

# ==========================================================
# Settings
# ==========================================================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.20
SEED = 42
LEARNING_RATE = 0.0001


def set_seed(seed):
    """Set random seed for repeatable train/validation split."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def validate_dataset():
    """Check that prepared_dataset exists."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"\nDataset folder not found:\n{DATASET_DIR}\n\n"
            "Make sure prepared_dataset is inside your project folder."
        )

    class_folders = [folder for folder in DATASET_DIR.iterdir() if folder.is_dir()]

    if len(class_folders) < 2:
        raise ValueError(
            "prepared_dataset must contain at least two class folders."
        )


def load_datasets():
    """
    Load images and automatically create:
    80% training data
    20% validation data
    """

    common_settings = {
        "directory": str(DATASET_DIR),
        "labels": "inferred",
        "label_mode": "categorical",
        "validation_split": VALIDATION_SPLIT,
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
    }

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        subset="training",
        shuffle=True,
        **common_settings,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        subset="validation",
        shuffle=False,
        **common_settings,
    )

    class_names = train_dataset.class_names

    print("\nClass Names:")
    for index, class_name in enumerate(class_names):
        print(f"{index}: {class_name}")

    # Improves training speed
    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(autotune)
    validation_dataset = validation_dataset.prefetch(autotune)

    return train_dataset, validation_dataset, class_names


def build_model(num_classes):
    """Create EfficientNetB0 transfer-learning model."""

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )

    # EfficientNetB0 includes its own normalization layer.
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )

    # Freeze pretrained EfficientNet feature extractor
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))

    x = data_augmentation(inputs)
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.40)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_graphs(history):
    """Save accuracy and loss graphs."""

    # Accuracy graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy",
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(RESULTS_DIR / "accuracy.png", dpi=150)
    plt.close()

    # Loss graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss",
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
    )

    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(RESULTS_DIR / "loss.png", dpi=150)
    plt.close()


def main():
    set_seed(SEED)
    validate_dataset()

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Waste Classification Model Training")
    print("=" * 60)

    print(f"\nDataset Path: {DATASET_DIR}")
    print("Training Split: 80%")
    print("Validation Split: 20%")

    # Load dataset
    train_dataset, validation_dataset, class_names = load_datasets()

    # Save class names for predict.py later
    with open(MODELS_DIR / "class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    print("\nClass names saved successfully.")

    # Build model
    model = build_model(len(class_names))

    print("\nModel Summary:\n")
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(
        filepath=str(MODELS_DIR / "best_model.keras"),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=0.0000001,
        verbose=1,
    )

    callbacks = [
        checkpoint,
        early_stop,
        reduce_lr,
    ]

    # Train model
    print("\nTraining Started...\n")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Evaluate model
    print("\nEvaluating Model...\n")

    loss, accuracy = model.evaluate(validation_dataset, verbose=0)

    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Validation Loss: {loss:.4f}")

    # Save final model
    model.save(MODELS_DIR / "final_waste_model.keras")

    # Save training history
    history_dataframe = pd.DataFrame(history.history)

    history_dataframe.to_csv(
        RESULTS_DIR / "training_history.csv",
        index=False,
    )

    # Save graphs
    save_graphs(history)

    print("\n" + "=" * 60)
    print("Training Completed Successfully!")
    print("=" * 60)

    print(f"\nBest Model:  {MODELS_DIR / 'best_model.keras'}")
    print(f"Final Model: {MODELS_DIR / 'final_waste_model.keras'}")
    print(f"Class Names: {MODELS_DIR / 'class_names.json'}")
    print(f"Results:     {RESULTS_DIR}")


if __name__ == "__main__":
    main()