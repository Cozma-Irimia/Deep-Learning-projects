"""Detect pneumonia on a chest X-ray.

Resource:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The images are the Kermany pediatric chest X-ray set published on that page:
Normal or Pneumonia. Loading, resizing, rescaling to [0, 1], random flip,
rotation, and zoom, then a convolutional network, follow the TensorFlow
image-classification tutorial. OpenCV reads one X-ray the way an uploaded
file would arrive. The printed line is the predicted class and its confidence.

The published validation folder has only 16 images, so validation is a
stratified 15% of the training files. The published test folder is held out.
This is a teaching model, not a diagnosis.
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
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
ZIP_PATH = DATA_DIR / "chest-xray-pneumonia.zip"

# Kaggle page for this project. The zip is the dataset archive from that page.
KAGGLE_SLUG = "paultimothymooney/chest-xray-pneumonia"

FOLDERS = {"NORMAL": "Normal", "PNEUMONIA": "Pneumonia"}
CLASS_NAMES = ["Normal", "Pneumonia"]
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 15


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def download_zip():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 100_000_000:
        return ZIP_PATH
    print(f"Downloading {KAGGLE_SLUG}")
    import kaggle

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(KAGGLE_SLUG, path=str(DATA_DIR), unzip=False, quiet=False)
    downloaded = next(DATA_DIR.glob("*.zip"), None)
    if downloaded is None:
        raise FileNotFoundError(f"No zip landed in {DATA_DIR}")
    if downloaded != ZIP_PATH:
        downloaded.replace(ZIP_PATH)
    return ZIP_PATH


def find_dataset_root():
    """Locate train/NORMAL after the Kaggle zip is extracted."""
    for train_dir in DATA_DIR.rglob("train"):
        if (train_dir / "NORMAL").is_dir() and (train_dir / "PNEUMONIA").is_dir():
            root = train_dir.parent
            if (root / "test" / "NORMAL").is_dir():
                return root
    return None


def ensure_images():
    root = find_dataset_root()
    if root is not None:
        return root
    archive = download_zip()
    print(f"Extracting {archive.name}")
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(DATA_DIR)
    root = find_dataset_root()
    if root is None:
        raise FileNotFoundError("Could not find train/NORMAL after extraction")
    return root


def list_images(split_dir):
    paths = []
    labels = []
    for folder, name in FOLDERS.items():
        class_index = CLASS_NAMES.index(name)
        for path in sorted((split_dir / folder).iterdir()):
            if path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                paths.append(str(path))
                labels.append(class_index)
    return np.asarray(paths), np.asarray(labels, dtype=np.int32)


def make_dataset(paths, labels, shuffle):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)

    def load(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, IMG_SIZE)
        image.set_shape(IMG_SIZE + (3,))
        return image, label

    return dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def plot_samples(dataset):
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("Training chest X-rays")
    for images, labels in dataset.take(1):
        for index in range(min(9, len(images))):
            ax = plt.subplot(3, 3, index + 1)
            ax.imshow(images[index].numpy().astype("uint8"))
            ax.set_title(CLASS_NAMES[int(labels[index])])
            ax.axis("off")
        break
    save_figure(fig, "sample_images.png")


def build_model():
    """Tutorial CNN, with global pooling so the head stays small."""
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    epochs = range(1, len(history.history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history.history["accuracy"], label="Training accuracy")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend(loc="lower right")
    axes[1].plot(epochs, history.history["loss"], label="Training loss")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation loss")
    axes[1].set_title("Loss")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, "training_history.png")


def plot_confusion(labels, predicted):
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for actual, guess in zip(labels, predicted):
        matrix[int(actual), int(guess)] += 1
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Test confusion matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout()
    save_figure(fig, "confusion_matrix.png")
    return matrix


def preprocess_with_opencv(path):
    """Read one uploaded X-ray with OpenCV and match the network input."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32)


def result_sentence(class_name, confidence):
    if class_name == "Pneumonia":
        return f"Pneumonia detected ({confidence:.1%} confidence)"
    return f"Normal chest X-ray ({confidence:.1%} confidence)"


def plot_predictions(paths, labels, model):
    """Two test files per class, scored through the OpenCV path."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    fig.suptitle("Test X-rays")
    lines = []
    chosen = []
    for class_index in range(len(CLASS_NAMES)):
        matches = np.where(labels == class_index)[0]
        chosen.extend(int(index) for index in matches[:2])
    for axis, index in zip(axes.ravel(), chosen):
        class_index = int(labels[index])
        path = Path(paths[index])
        image = preprocess_with_opencv(path)
        probabilities = model.predict(image[None, ...], verbose=0)[0]
        guess = int(np.argmax(probabilities))
        confidence = float(probabilities[guess])
        correct = guess == class_index
        axis.imshow(image.astype("uint8"))
        axis.set_title(
            f"{CLASS_NAMES[guess]} {confidence:.0%}\n(actual {CLASS_NAMES[class_index]})",
            color="#1a9850" if correct else "#d73027",
            fontsize=10,
        )
        axis.axis("off")
        sentence = result_sentence(CLASS_NAMES[guess], confidence)
        line = f"actual {CLASS_NAMES[class_index]:10s} {path.name}: {sentence}"
        print(line)
        lines.append(line)
    fig.tight_layout()
    save_figure(fig, "predictions.png")
    return lines


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(42)

    root = ensure_images()
    train_paths, train_labels = list_images(root / "train")
    test_paths, test_labels = list_images(root / "test")
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=0.15,
        stratify=train_labels,
        random_state=42,
    )
    for name, labels, title in (
        (CLASS_NAMES, train_labels, "train"),
        (CLASS_NAMES, val_labels, "val"),
        (CLASS_NAMES, test_labels, "test"),
    ):
        counts = " ".join(f"{name} {int(np.sum(labels == index))}" for index, name in enumerate(CLASS_NAMES))
        print(f"{title:5s} {counts}")

    train_ds = make_dataset(train_paths, train_labels, shuffle=True)
    val_ds = make_dataset(val_paths, val_labels, shuffle=False)
    test_ds = make_dataset(test_paths, test_labels, shuffle=False)
    plot_samples(train_ds)

    model = build_model()
    model.summary()
    # Pneumonia is about three quarters of the training files. Without a
    # weight the network can score well by calling every X-ray pneumonia.
    counts = np.bincount(train_labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    class_weight = {
        index: float(len(train_labels) / (len(CLASS_NAMES) * count))
        for index, count in enumerate(counts)
    }
    print("Class weights:", {CLASS_NAMES[index]: round(weight, 3) for index, weight in class_weight.items()})
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    plot_history(history)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")

    predicted = []
    actual = []
    for images, labels in test_ds:
        probabilities = model.predict(images, verbose=0)
        predicted.append(np.argmax(probabilities, axis=1))
        actual.append(labels.numpy())
    predicted = np.concatenate(predicted)
    actual = np.concatenate(actual)
    matrix = plot_confusion(actual, predicted)

    pneumonia_support = matrix[1].sum()
    predicted_pneumonia = matrix[:, 1].sum()
    recall = matrix[1, 1] / pneumonia_support if pneumonia_support else 0.0
    precision = matrix[1, 1] / predicted_pneumonia if predicted_pneumonia else 0.0
    print(f"Pneumonia precision: {precision:.4f}")
    print(f"Pneumonia recall: {recall:.4f}")

    print("\nSample decisions:")
    decision_lines = plot_predictions(test_paths, test_labels, model)
    lines = [
        "Pneumonia detection — Kermany chest X-rays, convolutional classifier",
        "This is a teaching model, not a medical diagnosis.",
        f"test_loss={test_loss:.4f}",
        f"test_accuracy={test_accuracy:.4f}",
        f"pneumonia_precision={precision:.4f}",
        f"pneumonia_recall={recall:.4f}",
        "",
        *decision_lines,
        "",
        "confusion matrix rows=actual, columns=predicted " + ", ".join(CLASS_NAMES),
        np.array2string(matrix),
    ]
    (OUTPUT_DIR / "evaluation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    model_path = OUTPUT_DIR / "pneumonia_cnn.keras"
    model.save(model_path)
    print(f"Saved {model_path}")
    print(f"Wrote {OUTPUT_DIR / 'evaluation.txt'}")


if __name__ == "__main__":
    main()
