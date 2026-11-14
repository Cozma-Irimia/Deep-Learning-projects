"""Recognize the emotion on a face.

Resource:
FER2013, from the Kaggle facial-expression challenge
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

FER2013 faces are 48x48 grayscale crops labeled with seven emotions:
Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The network is
trained with categorical cross-entropy. At inference, OpenCV's YuNet DNN
finds the face, the crop is resized to 48x48, and the CNN returns the emotion
with its confidence.
"""

import csv
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

CSV_URL = "https://huggingface.co/datasets/DerrickUnleashed/FER-2013/resolve/main/train.csv"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# FER2013 integer labels, in the order published with the challenge.
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
IMG_SIZE = 48
EPOCHS = 12
BATCH_SIZE = 128


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def ensure_csv():
    path = DATA_DIR / "train.csv"
    if path.exists():
        return path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {CSV_URL}")
    keras.utils.get_file(origin=CSV_URL, fname=path.name, cache_dir=DATA_DIR, cache_subdir=".")
    if not path.exists():
        found = list(DATA_DIR.rglob("train.csv"))
        if not found:
            raise FileNotFoundError("train.csv was not downloaded")
        path = found[0]
    return path


def load_fer(csv_path):
    """Read emotion,pixels rows into a uint8 array of shape (N, 48, 48, 1)."""
    cache_path = DATA_DIR / "fer2013.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        print(f"Loaded cache {cache_path} with {len(cached['labels'])} faces")
        return cached["images"], cached["labels"]

    images = []
    labels = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header[0] != "emotion":
            raise ValueError(f"Unexpected FER header: {header}")
        for row in reader:
            if len(row) < 2:
                continue
            pixels = np.fromstring(row[1], dtype=np.uint8, sep=" ")
            if pixels.size != IMG_SIZE * IMG_SIZE:
                continue
            images.append(pixels)
            labels.append(int(row[0]))
            if len(labels) % 5000 == 0:
                print(f"  parsed {len(labels)} faces")

    image_array = np.array(images, dtype=np.uint8).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    label_array = np.array(labels, dtype=np.int64)
    np.savez_compressed(cache_path, images=image_array, labels=label_array)
    print(f"Saved {cache_path}")
    return image_array, label_array


def build_model(num_classes):
    """Grayscale CNN. Softmax + categorical cross-entropy, as the brief asks."""
    model = keras.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    epochs_range = range(1, len(history.history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs_range, history.history["accuracy"], label="Training")
    axes[0].plot(epochs_range, history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(epochs_range, history.history["loss"], label="Training")
    axes[1].plot(epochs_range, history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].legend()
    fig.tight_layout()
    save_figure(fig, "training_history.png")


def plot_confusion(y_true, y_pred):
    matrix = np.zeros((len(EMOTIONS), len(EMOTIONS)), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("FER2013 confusion matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "confusion_matrix.png")


def plot_grid(images, y_true, probabilities):
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    fig.suptitle("Held-out faces")
    for ax, index in zip(axes.ravel(), range(12)):
        predicted = int(np.argmax(probabilities[index]))
        confidence = float(probabilities[index, predicted])
        actual = int(y_true[index])
        ax.imshow(images[index, :, :, 0], cmap="gray")
        color = "#1a9850" if predicted == actual else "#d73027"
        ax.set_title(f"{EMOTIONS[predicted]} {confidence:.0%}\n({EMOTIONS[actual]})", color=color, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, "predictions.png")


def ensure_yunet():
    path = DATA_DIR / "face_detection_yunet_2023mar.onnx"
    if path.exists():
        return path
    print(f"Downloading {YUNET_URL}")
    keras.utils.get_file(origin=YUNET_URL, fname=path.name, cache_dir=DATA_DIR, cache_subdir=".")
    if not path.exists():
        found = list(DATA_DIR.rglob(path.name))
        if not found:
            raise FileNotFoundError(path.name)
        return found[0]
    return path


def face_on_canvas(gray_face):
    """Place an upscaled FER crop on a larger frame so YuNet has a face to find."""
    canvas = np.full((480, 480, 3), 32, dtype=np.uint8)
    color = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)
    big = cv2.resize(color, (220, 220), interpolation=cv2.INTER_CUBIC)
    canvas[130:350, 130:350] = big
    return canvas


def detect_and_classify(canvas, detector, model):
    """Find a face, crop it, and return the emotion probabilities for that crop."""
    detector.setInputSize((canvas.shape[1], canvas.shape[0]))
    _, faces = detector.detect(canvas)
    if faces is None:
        return None

    x, y, box_w, box_h = [int(value) for value in faces[0, :4]]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(canvas.shape[1], x + box_w)
    y1 = min(canvas.shape[0], y + box_h)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = cv2.cvtColor(canvas[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    batch = resized.astype("float32").reshape(1, IMG_SIZE, IMG_SIZE, 1)
    probabilities = model.predict(batch, verbose=0)[0]
    return (x0, y0, x1, y1), probabilities


def plot_detected_faces(images, model):
    """Run the full detect-then-classify path on a handful of test faces."""
    detector = cv2.FaceDetectorYN.create(str(ensure_yunet()), "", (480, 480), 0.5, 0.3, 5000)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    fig.suptitle("YuNet face box, then the emotion classifier")
    found = 0
    for image in images:
        if found == 6:
            break
        canvas = face_on_canvas(image[:, :, 0])
        detection = detect_and_classify(canvas, detector, model)
        if detection is None:
            continue
        (x0, y0, x1, y1), probabilities = detection
        emotion = int(np.argmax(probabilities))
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        ax = axes.ravel()[found]
        ax.imshow(rgb)
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#1a9850", linewidth=2)
        ax.add_patch(rect)
        ax.set_title(f"{EMOTIONS[emotion]} {probabilities[emotion]:.0%}", fontsize=10)
        ax.axis("off")
        found += 1
        print(
            f"Detected face -> {EMOTIONS[emotion]} "
            f"({probabilities[emotion]:.1%})"
        )
    for ax in axes.ravel()[found:]:
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, "face_detections.png")
    return found


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(42)

    images, labels = load_fer(ensure_csv())
    print("Class counts:")
    for index, name in enumerate(EMOTIONS):
        print(f"  {name:10s} {int((labels == index).sum())}")

    x_train, x_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    y_train_cat = keras.utils.to_categorical(y_train, len(EMOTIONS))
    y_test_cat = keras.utils.to_categorical(y_test, len(EMOTIONS))

    model = build_model(len(EMOTIONS))
    model.summary()
    history = model.fit(
        x_train,
        y_train_cat,
        validation_data=(x_test, y_test_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    plot_history(history)

    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    probabilities = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    plot_confusion(y_test, predicted)
    plot_grid(x_test, y_test, probabilities)

    print("\nDirect predictions on the first test faces:")
    for index in range(5):
        emotion = int(predicted[index])
        print(
            f"  actual {EMOTIONS[int(y_test[index])]:10s} "
            f"predicted {EMOTIONS[emotion]:10s} {probabilities[index, emotion]:.1%}"
        )

    detected = plot_detected_faces(x_test[:30], model)
    print(f"YuNet found and labeled {detected} faces")

    model_path = OUTPUT_DIR / "fer2013_cnn.keras"
    model.save(model_path)
    print(f"Saved {model_path}")

    text = [
        "Facial emotion recognition — FER2013",
        f"test_loss={test_loss:.4f}",
        f"test_accuracy={test_acc:.4f}",
        "loss=categorical_crossentropy",
        f"yunet_detections_drawn={detected}",
    ]
    (OUTPUT_DIR / "evaluation.txt").write_text("\n".join(text), encoding="utf-8")
    print(f"Wrote {OUTPUT_DIR / 'evaluation.txt'}")


if __name__ == "__main__":
    main()
