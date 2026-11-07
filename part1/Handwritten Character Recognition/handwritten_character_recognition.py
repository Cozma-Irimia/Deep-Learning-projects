"""Recognize handwritten digits with a small convolutional network.

Resource used for the model:
https://keras.io/examples/vision/mnist_convnet/

The network, the [0, 1] pixel scaling, and the training setup follow that
Keras MNIST convnet example. OpenCV is added in front of the network so a
digit can be read from an image file, resized to 28x28, and scaled the same
way before predict() is called.

The reference example trains for 15 epochs. EPOCHS below is shorter so the
script finishes on CPU; the architecture is unchanged. Set EPOCHS = 15 to
match the example exactly.
"""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from keras import layers

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

# Model / data parameters from the Keras example.
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
BATCH_SIZE = 128
EPOCHS = 5


def load_and_scale_mnist():
    """Load MNIST and scale pixels from 0..255 into 0..1.

    The extra axis makes each image a rank-3 tensor (height, width, channel),
    which is what Conv2D expects.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # One-hot labels, because the loss below is categorical_crossentropy.
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    return (x_train, y_train), (x_test, y_test)


def build_model():
    """The Keras example convnet: two conv/pool blocks, then a softmax head."""
    model = keras.Sequential(
        [
            keras.Input(shape=INPUT_SHAPE),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def preprocess_digit_image(image):
    """Turn one image array into a single-example batch for the network.

    OpenCV reads files as BGR or grayscale. The model was trained on MNIST,
    so the image is converted to one gray channel, resized to 28x28, lightly
    blurred, and divided by 255. Output shape is (1, 28, 28, 1), values in
    [0, 1], with bright strokes on a dark background — the MNIST convention.
    """
    if image is None:
        raise ValueError("OpenCV could not read the digit image.")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    normalized = blurred.astype("float32") / 255.0
    return normalized.reshape(1, 28, 28, 1)


def save_digit_grid(images, labels, filename, title):
    """Save a small grid. `labels` are integer class ids."""
    count = len(images)
    cols = 5
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    fig.suptitle(title)
    for ax, image, label in zip(np.ravel(axes), images, labels):
        ax.imshow(np.squeeze(image), cmap="gray")
        ax.set_title(str(int(label)))
        ax.axis("off")
    for ax in np.ravel(axes)[count:]:
        ax.axis("off")
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def predict_image_file(model, path):
    """Read a PNG with OpenCV and return the digit plus the class probabilities."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    batch = preprocess_digit_image(image)
    probabilities = model.predict(batch, verbose=0)[0]
    digit = int(np.argmax(probabilities))
    return digit, probabilities


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(42)

    (x_train, y_train), (x_test, y_test) = load_and_scale_mnist()

    model = build_model()
    model.summary()
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    model_path = OUTPUT_DIR / "mnist_cnn.keras"
    model.save(model_path)
    print(f"Saved {model_path}")

    # Integer labels for the figures. y_test is still one-hot.
    true_digits = np.argmax(y_test, axis=1)

    preview = 10
    save_digit_grid(
        x_test[:preview],
        true_digits[:preview],
        "sample_digits.png",
        "First 10 MNIST test digits",
    )

    # Write those digits to PNG files and read them back through OpenCV,
    # which is the path a scanned or uploaded character would take.
    file_predictions = []
    for index in range(preview):
        png_path = OUTPUT_DIR / f"digit_{index}.png"
        pixels = (np.squeeze(x_test[index]) * 255).astype("uint8")
        cv2.imwrite(str(png_path), pixels)
        predicted, probabilities = predict_image_file(model, png_path)
        confidence = float(probabilities[predicted])
        actual = int(true_digits[index])
        file_predictions.append((index, actual, predicted, confidence))
        print(
            f"digit_{index}.png: predicted {predicted} "
            f"({confidence:.1%}), true {actual}"
        )

    # Direct batch prediction on the same ten tensors, for the comparison plot.
    batch_probabilities = model.predict(x_test[:preview], verbose=0)
    batch_digits = np.argmax(batch_probabilities, axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    fig.suptitle("OpenCV file path: prediction and confidence")
    for ax, (index, actual, predicted, confidence) in zip(axes.ravel(), file_predictions):
        image = cv2.imread(str(OUTPUT_DIR / f"digit_{index}.png"), cv2.IMREAD_GRAYSCALE)
        ax.imshow(image, cmap="gray")
        color = "#1a9850" if predicted == actual else "#d73027"
        ax.set_title(f"pred {predicted} ({confidence:.0%})\ntrue {actual}", color=color, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    figure_path = OUTPUT_DIR / "predictions.png"
    fig.savefig(figure_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {figure_path}")

    report = [
        "Handwritten digit recognition — MNIST convnet",
        f"epochs={EPOCHS}",
        f"test_loss={score[0]:.6f}",
        f"test_accuracy={score[1]:.6f}",
        "",
        "OpenCV file predictions (index, true, predicted, confidence):",
    ]
    for index, actual, predicted, confidence in file_predictions:
        report.append(f"{index},{actual},{predicted},{confidence:.4f}")
    report.append("")
    report.append(
        "Direct tensor predictions: " + " ".join(str(int(d)) for d in batch_digits)
    )
    (OUTPUT_DIR / "evaluation.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {OUTPUT_DIR / 'evaluation.txt'}")


if __name__ == "__main__":
    main()
