import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from recommendations import display_recommendation


# ==========================================================
# Paths
# ==========================================================
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "best_model.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "models" / "class_names.json"

IMAGE_SIZE = (224, 224)


def load_class_names():
    """Load saved class names."""

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found:\n{CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_trained_model():
    """Load the trained model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}\n\n"
            "Run train_model.py first."
        )

    return tf.keras.models.load_model(MODEL_PATH)


def predict_image(image_path, model, class_names):
    """Predict the waste category for one image."""

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found:\n{image_path}")

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
    )

    image_array = tf.keras.utils.img_to_array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    return predicted_class, confidence, predictions


def main():
    parser = argparse.ArgumentParser(
        description="Predict waste category from an image."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path of the image to classify.",
    )

    arguments = parser.parse_args()

    print("\nLoading model...")
    model = load_trained_model()

    print("Loading class names...")
    class_names = load_class_names()

    predicted_class, confidence, all_predictions = predict_image(
        arguments.image,
        model,
        class_names,
    )

    print("\n" + "=" * 55)
    print("WASTE CLASSIFICATION RESULT")
    print("=" * 55)

    print(f"\nImage: {arguments.image}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    print("\nAll Class Probabilities:")

    for class_name, probability in zip(class_names, all_predictions):
        print(f"{class_name:20}: {probability * 100:.2f}%")

    # Display disposal advice for the predicted category
    display_recommendation(predicted_class)

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()