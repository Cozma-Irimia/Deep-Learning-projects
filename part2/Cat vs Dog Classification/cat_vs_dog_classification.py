"""Classify a photo as a cat or a dog.

Resource:
https://www.tensorflow.org/tutorials/images/classification

The loading pipeline, the three-block CNN, the Rescaling layer, dropout, and
the softmax confidence score follow that tutorial. Rotation and zoom come
from the same data-augmentation section. Shear is added on top because the
assignment asks for it; Keras applies it with RandomShear, in the same
preprocessing-layer style as RandomRotation and RandomZoom.

An uploaded file is read and resized with OpenCV, then scored by the network.
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
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

DATASET_URL = "https://download.mlcc.google.com/mledu-datasets/cats_and_dogs_filtered.zip"

# The tutorial uses 180px images. 128px keeps the same model trainable on CPU.
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
# The classification tutorial trains for 15 epochs. Accuracy was still
# rising at epoch 8 on CPU, so the tutorial length is used here as well.
EPOCHS = 15


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def ensure_dataset():
    """Download and unzip the filtered cats-and-dogs set once."""
    root = DATA_DIR / "cats_and_dogs_filtered"
    if (root / "train").is_dir() and (root / "validation").is_dir():
        return root

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / "cats_and_dogs_filtered.zip"
    if not archive_path.exists():
        print(f"Downloading {DATASET_URL}")
        keras.utils.get_file(origin=DATASET_URL, fname=archive_path.name, cache_dir=DATA_DIR, cache_subdir=".")
        # get_file may nest the download; fall back to a direct copy name.
        if not archive_path.exists():
            downloaded = list(DATA_DIR.rglob("cats_and_dogs_filtered.zip"))
            if not downloaded:
                raise FileNotFoundError("cats_and_dogs_filtered.zip was not downloaded")
            archive_path = downloaded[0]

    print(f"Extracting {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(DATA_DIR)
    if not (root / "train").is_dir():
        raise FileNotFoundError(f"Expected training images under {root}")
    return root


def load_splits(dataset_root):
    """Directory folders become the class names: cats, dogs."""
    train_ds = keras.utils.image_dataset_from_directory(
        dataset_root / "train",
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        seed=42,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        dataset_root / "validation",
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        seed=42,
    )
    class_names = train_ds.class_names
    print("Classes:", class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=42).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    return train_ds, val_ds, class_names


def plot_samples(dataset, class_names):
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("Training images")
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            ax.imshow(images[i].numpy().astype("uint8"))
            ax.set_title(class_names[int(labels[i])])
            ax.axis("off")
        break
    save_figure(fig, "sample_images.png")


def build_augmentation():
    """Flip, rotation, zoom, and shear. Active only while training."""
    return keras.Sequential(
        [
            layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomShear(x_factor=0.2, y_factor=0.2),
        ],
        name="data_augmentation",
    )


def plot_augmentation(dataset, augmentation):
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("One photo after random rotation, zoom, and shear")
    for images, _ in dataset.take(1):
        for i in range(9):
            # training=True forces the random layers to transform the batch.
            augmented = augmentation(images, training=True)
            ax = plt.subplot(3, 3, i + 1)
            ax.imshow(np.clip(augmented[0].numpy(), 0, 255).astype("uint8"))
            ax.axis("off")
        break
    save_figure(fig, "augmentation_examples.png")


def build_model(augmentation, num_classes):
    """Tutorial CNN: three conv/pool blocks, dropout, then a logit head.

    The last layer has no activation. SparseCategoricalCrossentropy(from_logits)
    is the tutorial loss. Softmax is applied later, only to report confidence.
    """
    model = keras.Sequential(
        [
            augmentation,
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, name="outputs"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs_range, acc, label="Training accuracy")
    axes[0].plot(epochs_range, val_acc, label="Validation accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend(loc="lower right")
    axes[1].plot(epochs_range, loss, label="Training loss")
    axes[1].plot(epochs_range, val_loss, label="Validation loss")
    axes[1].set_title("Loss")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, "training_history.png")


def predict_uploaded_image(model, path, class_names):
    """OpenCV resize + the network. Returns the RGB image and class probabilities."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype("float32"), 0)
    logits = model.predict(batch, verbose=0)[0]
    probabilities = tf.nn.softmax(logits).numpy()
    return resized, probabilities


def pretty_name(class_name):
    """Directory names are plural. The assignment asks for Cat or Dog."""
    return {"cats": "Cat", "dogs": "Dog"}.get(class_name, class_name)


def describe(probabilities, class_names):
    winner = int(np.argmax(probabilities))
    parts = ", ".join(
        f"{pretty_name(name)} {probabilities[index]:.1%}" for index, name in enumerate(class_names)
    )
    sentence = (
        f"This image most likely belongs to {pretty_name(class_names[winner])} "
        f"with a {100 * probabilities[winner]:.2f} percent confidence."
    )
    return winner, sentence, parts


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(42)

    dataset_root = ensure_dataset()
    train_ds, val_ds, class_names = load_splits(dataset_root)
    plot_samples(train_ds, class_names)

    augmentation = build_augmentation()
    plot_augmentation(train_ds, augmentation)

    model = build_model(augmentation, len(class_names))
    model.summary()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)
    plot_history(history)

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")

    model_path = OUTPUT_DIR / "cat_dog_cnn.keras"
    model.save(model_path)
    print(f"Saved {model_path}")

    # Score real files through OpenCV, which is the uploaded-image path.
    samples = []
    for class_name in class_names:
        found = sorted((dataset_root / "validation" / class_name).glob("*.jpg"))
        samples.extend(found[:3])

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    fig.suptitle("Validation photos scored as an upload")
    lines = [
        "Cat vs dog classification",
        f"validation_loss={val_loss:.4f}",
        f"validation_accuracy={val_acc:.4f}",
        "",
    ]
    for ax, path in zip(axes.ravel(), samples):
        image, probabilities = predict_uploaded_image(model, path, class_names)
        winner, sentence, parts = describe(probabilities, class_names)
        print(f"{path.name}: {sentence}")
        print(f"  {parts}")
        ax.imshow(image)
        ax.set_title(
            f"{pretty_name(class_names[winner])} {probabilities[winner]:.0%}",
            fontsize=10,
        )
        ax.axis("off")
        lines.append(f"{path.name}: {sentence} [{parts}]")
    fig.tight_layout()
    save_figure(fig, "predictions.png")

    (OUTPUT_DIR / "evaluation.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_DIR / 'evaluation.txt'}")


if __name__ == "__main__":
    main()
