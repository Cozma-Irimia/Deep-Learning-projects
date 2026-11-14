"""Classify German traffic signs and report a coarse road-sign category.

Resource:
German Traffic Sign Recognition Benchmark (GTSRB)
https://benchmark.ini.rub.de/

Every image is resized, then contrast-normalized with CLAHE on the lightness
channel, then scaled to [0, 1] inside the network. A CNN predicts one of the
43 GTSRB classes. The assignment categories (speed limit, stop, yield, turn)
are read off that class name.

The training archive is the official Final Training Images zip. Twenty percent
of those images are held out for evaluation.
"""

import os
import zipfile
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

ARCHIVE_URL = (
    "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
    "GTSRB_Final_Training_Images.zip"
)

IMG_SIZE = 32
EPOCHS = 10
BATCH_SIZE = 64

# Official class order, folder 00000 through 00042.
CLASS_NAMES = [
    "Speed limit 20 km/h",
    "Speed limit 30 km/h",
    "Speed limit 50 km/h",
    "Speed limit 60 km/h",
    "Speed limit 70 km/h",
    "Speed limit 80 km/h",
    "End of speed limit 80 km/h",
    "Speed limit 100 km/h",
    "Speed limit 120 km/h",
    "No passing",
    "No passing for vehicles over 3.5 tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 tons",
]


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def coarse_category(class_name):
    """Map a GTSRB name onto the categories named in the assignment."""
    lowered = class_name.lower()
    if lowered.startswith("speed limit"):
        return "speed limit"
    if class_name == "Stop":
        return "stop"
    if class_name == "Yield":
        return "yield"
    if any(word in lowered for word in ("turn", "keep", "ahead", "roundabout", "straight")):
        return "turn"
    return "other"


def ensure_class_folders():
    known = DATA_DIR / "GTSRB" / "Final_Training" / "Images"
    if known.is_dir():
        return known

    archive_path = DATA_DIR / "GTSRB_Final_Training_Images.zip"
    if not archive_path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {ARCHIVE_URL}")
        keras.utils.get_file(
            origin=ARCHIVE_URL,
            fname=archive_path.name,
            cache_dir=DATA_DIR,
            cache_subdir=".",
        )
    print(f"Extracting {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(DATA_DIR)
    if not known.is_dir():
        raise FileNotFoundError(f"GTSRB class folders were not found at {known}")
    return known


# One CLAHE object is reused for every sign. Creating it per image is slow.
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def contrast_normalize(bgr):
    """CLAHE on the lightness channel. This is the contrast-normalization step."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = CLAHE.apply(lightness)
    merged = cv2.merge((lightness, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def load_images(class_root):
    """Resize every PPM sign to 32x32 and cache the result as a numpy archive."""
    cache_path = DATA_DIR / "gtsrb_32.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        print(f"Loaded cache {cache_path} with {len(cached['labels'])} images")
        return cached["images"], cached["labels"]

    images = []
    labels = []
    folders = sorted(path for path in class_root.iterdir() if path.is_dir())
    for folder in folders:
        class_id = int(folder.name)
        for image_path in sorted(folder.glob("*.ppm")):
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                continue
            resized = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            images.append(contrast_normalize(resized))
            labels.append(class_id)
        print(f"  class {class_id:02d}: {len(labels)} images so far")

    image_array = np.stack(images).astype("uint8")
    label_array = np.array(labels, dtype=np.int64)
    np.savez_compressed(cache_path, images=image_array, labels=label_array)
    print(f"Saved {cache_path}")
    return image_array, label_array


def build_model(num_classes):
    model = keras.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, 3, activation="relu"),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.4),
            layers.Dense(256, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
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


def plot_predictions(images, y_true, probabilities):
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    fig.suptitle("Held-out signs: prediction and confidence")
    chosen = np.linspace(0, len(images) - 1, num=12, dtype=int)
    for ax, index in zip(axes.ravel(), chosen):
        predicted = int(np.argmax(probabilities[index]))
        confidence = float(probabilities[index, predicted])
        actual = int(y_true[index])
        ax.imshow(images[index])
        color = "#1a9850" if predicted == actual else "#d73027"
        ax.set_title(
            f"{coarse_category(CLASS_NAMES[predicted])}\n{confidence:.0%}",
            color=color,
            fontsize=9,
        )
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, "predictions.png")


def category_accuracy(y_true, y_pred):
    lines = []
    for category in ("speed limit", "stop", "yield", "turn", "other"):
        mask = np.array([coarse_category(CLASS_NAMES[label]) == category for label in y_true])
        if not mask.any():
            continue
        accuracy = float((y_pred[mask] == y_true[mask]).mean())
        count = int(mask.sum())
        line = f"{category}: accuracy {accuracy:.3f} on {count} images"
        print(line)
        lines.append(line)
    return lines


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(42)

    images, labels = load_images(ensure_class_folders())
    print(f"Dataset: {images.shape[0]} images, {len(CLASS_NAMES)} classes")

    x_train, x_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    model = build_model(len(CLASS_NAMES))
    model.summary()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    plot_history(history)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    probabilities = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    predicted = np.argmax(probabilities, axis=1)

    # Show a few concrete decisions, including the category the brief asks for.
    print("\nSample predictions:")
    report_lines = []
    for index in (0, 1, 2, 3, 4):
        class_id = int(predicted[index])
        name = CLASS_NAMES[class_id]
        category = coarse_category(name)
        confidence = float(probabilities[index, class_id])
        actual = CLASS_NAMES[int(y_test[index])]
        line = (
            f"predicted {name} [{category}] {confidence:.1%}, actual {actual}"
        )
        print(line)
        report_lines.append(line)

    plot_predictions(x_test, y_test, probabilities)
    group_lines = category_accuracy(y_test, predicted)

    model_path = OUTPUT_DIR / "gtsrb_cnn.keras"
    model.save(model_path)
    print(f"Saved {model_path}")

    text = [
        "Traffic sign recognition — GTSRB",
        f"test_loss={test_loss:.4f}",
        f"test_accuracy={test_acc:.4f}",
        "",
        *report_lines,
        "",
        *group_lines,
    ]
    (OUTPUT_DIR / "evaluation.txt").write_text("\n".join(text), encoding="utf-8")
    print(f"Wrote {OUTPUT_DIR / 'evaluation.txt'}")


if __name__ == "__main__":
    main()
