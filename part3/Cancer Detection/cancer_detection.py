"""Detect breast cancer on ultrasound images with transfer learning.

Resource:
https://www.tensorflow.org/tutorials/images/transfer_learning

The pipeline follows that tutorial: random flip and rotation, MobileNetV2
pre-trained on ImageNet, a frozen feature extractor, then fine-tuning of the
top layers. Batch-normalization stays in inference mode, as the tutorial
requires.

The images are the BUSI breast ultrasound set (Al-Dhabyani et al., 2020):
normal tissue, a benign lesion, or a malignant lesion. A malignant prediction
is reported as cancer detected. OpenCV loads a single uploaded scan, resizes
it to the network input, and the same model scores it. This is a teaching
model, not a diagnosis.
"""

import os
import shutil
import urllib.request
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_ROOT = DATA_DIR / "images"
SPLIT_ROOT = DATA_DIR / "split"

# Hugging Face mirror of the BUSI release. The Cairo University zip is behind
# a browser security check, so this parquet copy is the reliable download.
PARQUET_URL = (
    "https://huggingface.co/datasets/MedOtter/BUSI/resolve/main/"
    "data/train-00000-of-00001.parquet"
)

CLASS_NAMES = ["benign", "malignant", "normal"]
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
BASE_LEARNING_RATE = 1e-4
# The tutorial trains the new head for 10 epochs, then fine-tunes for 10.
# Five fine-tuning epochs are enough here: BUSI has only 780 images.
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 8
# Ultrasound is far from ImageNet, but unfreezing too much of MobileNet
# overfits these 780 images. Leave the early layers frozen.
FINE_TUNE_AT = 130


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def download_parquet():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "busi.parquet"
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    print(f"Downloading {PARQUET_URL}")
    request = urllib.request.Request(PARQUET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=180) as response, path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return path


def image_bytes(value):
    """Hugging Face stores each image as a struct with a raw byte payload."""
    if isinstance(value, dict):
        return value["bytes"]
    return value


def export_images(parquet_path):
    """Write one PNG per ultrasound, leaving the segmentation masks unused."""
    if all((IMAGE_ROOT / name).is_dir() and any((IMAGE_ROOT / name).glob("*.png")) for name in CLASS_NAMES):
        return IMAGE_ROOT

    frame = pd.read_parquet(parquet_path)
    for name in CLASS_NAMES:
        (IMAGE_ROOT / name).mkdir(parents=True, exist_ok=True)

    for row in frame.itertuples(index=False):
        label = str(row.class_label)
        if label not in CLASS_NAMES:
            continue
        stem = "".join(char if char.isalnum() or char in " _-" else "_" for char in str(row.image_id))
        (IMAGE_ROOT / label / f"{stem}.png").write_bytes(image_bytes(row.image))
    return IMAGE_ROOT


def stratified_split(image_root):
    """70% train, 15% validation, 15% test, kept separate for each class.

    The tutorial cuts the test set from the validation batches. BUSI is small
    and unbalanced, so each class is split on its own instead.
    """
    train_dir = SPLIT_ROOT / "train"
    if train_dir.is_dir() and any(train_dir.glob("*/*.png")):
        return SPLIT_ROOT / "train", SPLIT_ROOT / "validation", SPLIT_ROOT / "test"

    rng = np.random.default_rng(42)
    for name in CLASS_NAMES:
        paths = sorted((image_root / name).glob("*.png"))
        order = rng.permutation(len(paths))
        paths = [paths[index] for index in order]
        n_test = max(1, int(round(len(paths) * 0.15)))
        n_val = max(1, int(round(len(paths) * 0.15)))
        groups = {
            "test": paths[:n_test],
            "validation": paths[n_test : n_test + n_val],
            "train": paths[n_test + n_val :],
        }
        for split, files in groups.items():
            destination = SPLIT_ROOT / split / name
            destination.mkdir(parents=True, exist_ok=True)
            for source in files:
                target = destination / source.name
                if not target.exists():
                    shutil.copy2(source, target)
    return SPLIT_ROOT / "train", SPLIT_ROOT / "validation", SPLIT_ROOT / "test"


def make_dataset(directory, shuffle):
    dataset = keras.utils.image_dataset_from_directory(
        directory,
        shuffle=shuffle,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode="int",
    )
    # prefetch() returns a wrapper that no longer exposes class_names.
    return dataset.prefetch(tf.data.AUTOTUNE), list(dataset.class_names)


def plot_samples(dataset, class_names):
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("BUSI training images")
    for images, labels in dataset.take(1):
        for index in range(min(9, len(images))):
            ax = plt.subplot(3, 3, index + 1)
            ax.imshow(images[index].numpy().astype("uint8"))
            ax.set_title(class_names[int(labels[index])])
            ax.axis("off")
        break
    save_figure(fig, "sample_images.png")


def plot_history(feature_history, fine_history):
    accuracy = feature_history.history["accuracy"] + fine_history.history["accuracy"]
    val_accuracy = feature_history.history["val_accuracy"] + fine_history.history["val_accuracy"]
    loss = feature_history.history["loss"] + fine_history.history["loss"]
    val_loss = feature_history.history["val_loss"] + fine_history.history["val_loss"]
    epochs_range = range(len(accuracy))
    boundary = len(feature_history.history["accuracy"]) - 1

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(epochs_range, accuracy, label="Training accuracy")
    axes[0].plot(epochs_range, val_accuracy, label="Validation accuracy")
    axes[0].axvline(boundary, color="gray", linestyle="--", label="Start fine-tuning")
    axes[0].set_title("Accuracy")
    axes[0].legend(loc="lower right")
    axes[1].plot(epochs_range, loss, label="Training loss")
    axes[1].plot(epochs_range, val_loss, label="Validation loss")
    axes[1].axvline(boundary, color="gray", linestyle="--", label="Start fine-tuning")
    axes[1].set_title("Loss")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, "training_history.png")


def build_model(class_count):
    """Frozen MobileNetV2 plus a new softmax head. The tutorial's binary head
    is a single sigmoid; BUSI has three clinical classes, so the head has
    one output per class.
    """
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
        ],
        name="data_augmentation",
    )
    preprocess_input = keras.applications.mobilenet_v2.preprocess_input
    base_model = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    # training=False keeps every BatchNormalization layer in inference mode.
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(class_count, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model, base_model


def compile_for_feature_extraction(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def fine_tune(model, base_model):
    """Unfreeze the top of MobileNetV2 and continue with a smaller step size."""
    base_model.trainable = True
    print("Number of layers in the base model:", len(base_model.layers))
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=BASE_LEARNING_RATE / 50),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def collect_predictions(dataset, model):
    images = []
    labels = []
    probabilities = []
    for image_batch, label_batch in dataset:
        batch_probabilities = model.predict(image_batch, verbose=0)
        images.append(image_batch.numpy())
        labels.append(label_batch.numpy())
        probabilities.append(batch_probabilities)
    return (
        np.concatenate(images),
        np.concatenate(labels),
        np.concatenate(probabilities),
    )


def preprocess_with_opencv(path):
    """Read one uploaded ultrasound with OpenCV.

    Training batches are resized by Keras. This is the same 160x160 RGB
    input for a file chosen outside the dataset pipeline. The model still
    maps pixel values from [0, 255] to [-1, 1].
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32)


def predict_uploaded_scans(model, class_names, test_dir):
    """Score one held-out file per class through the OpenCV path."""
    lines = []
    for name in class_names:
        path = next((test_dir / name).glob("*.png"))
        image = preprocess_with_opencv(path)
        probabilities = model.predict(image[None, ...], verbose=0)[0]
        class_index = int(np.argmax(probabilities))
        sentence = cancer_sentence(class_names[class_index], float(probabilities[class_index]))
        line = f"opencv {name:10s} {path.name}: {sentence}"
        print(line)
        lines.append(line)
    return lines


def cancer_sentence(class_name, confidence):
    if class_name == "malignant":
        return f"Cancer detected: malignant ({confidence:.1%} confidence)"
    return f"No cancer detected: {class_name} ({confidence:.1%} confidence)"


def plot_predictions(images, labels, probabilities, class_names):
    """Show two test images from each class, not only the first batch."""
    predicted = np.argmax(probabilities, axis=1)
    chosen = []
    for class_index in range(len(class_names)):
        matches = np.where(labels == class_index)[0]
        chosen.extend(matches[:2])
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    fig.suptitle("Test ultrasounds")
    for ax, index in zip(axes.ravel(), chosen):
        class_index = int(predicted[index])
        confidence = float(probabilities[index, class_index])
        correct = class_index == int(labels[index])
        ax.imshow(images[index].astype("uint8"))
        ax.set_title(
            f"{class_names[class_index]} {confidence:.0%}\n(actual {class_names[int(labels[index])]})",
            color="#1a9850" if correct else "#d73027",
            fontsize=9,
        )
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, "predictions.png")


def plot_confusion(labels, predicted, class_names):
    matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    for actual, guess in zip(labels, predicted):
        matrix[int(actual), int(guess)] += 1
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Test confusion matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout()
    save_figure(fig, "confusion_matrix.png")
    return matrix


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(42)

    image_root = export_images(download_parquet())
    train_dir, val_dir, test_dir = stratified_split(image_root)
    for name in CLASS_NAMES:
        print(
            f"{name:10s} train {len(list((train_dir / name).glob('*.png'))):3d}"
            f"  val {len(list((val_dir / name).glob('*.png'))):3d}"
            f"  test {len(list((test_dir / name).glob('*.png'))):3d}"
        )

    train_ds, class_names = make_dataset(train_dir, shuffle=True)
    val_ds, _ = make_dataset(val_dir, shuffle=False)
    test_ds, _ = make_dataset(test_dir, shuffle=False)
    print("Classes:", class_names)
    plot_samples(train_ds, class_names)

    # Stop when validation accuracy stalls and restore those weights.
    # Fine-tuning a large network on 780 images overfits if it runs too long.
    def stop_when_validation_stalls():
        return keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        )

    model, base_model = build_model(len(class_names))
    compile_for_feature_extraction(model)
    model.summary()
    feature_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=[stop_when_validation_stalls()],
        verbose=2,
    )

    _, feature_accuracy = model.evaluate(val_ds, verbose=0)
    feature_weights = OUTPUT_DIR / "feature_extractor.weights.h5"
    model.save_weights(feature_weights)

    fine_tune(model, base_model)
    fine_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=len(feature_history.epoch) + FINE_TUNE_EPOCHS,
        initial_epoch=len(feature_history.epoch),
        callbacks=[stop_when_validation_stalls()],
        verbose=2,
    )
    plot_history(feature_history, fine_history)
    _, fine_accuracy = model.evaluate(val_ds, verbose=0)
    if fine_accuracy < feature_accuracy:
        print(
            "Fine-tuning lowered validation accuracy "
            f"({fine_accuracy:.3f} < {feature_accuracy:.3f}). "
            "Restoring the frozen-feature model."
        )
        model.load_weights(feature_weights)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")

    images, labels, probabilities = collect_predictions(test_ds, model)
    predicted = np.argmax(probabilities, axis=1)
    plot_predictions(images, labels, probabilities, class_names)
    matrix = plot_confusion(labels, predicted, class_names)

    print("\nSample decisions:")
    lines = [
        "Cancer detection — BUSI breast ultrasound, MobileNetV2 transfer learning",
        "This is a teaching model, not a medical diagnosis.",
        f"test_loss={test_loss:.4f}",
        f"test_accuracy={test_accuracy:.4f}",
        "",
    ]
    shown = []
    for class_index in range(len(class_names)):
        matches = np.where(labels == class_index)[0]
        shown.extend(matches[:2])
    for index in shown:
        class_index = int(predicted[index])
        confidence = float(probabilities[index, class_index])
        sentence = cancer_sentence(class_names[class_index], confidence)
        actual = class_names[int(labels[index])]
        line = f"actual {actual:10s} {sentence}"
        print(line)
        lines.append(line)
    lines.append("")
    lines.append("confusion matrix rows=actual, columns=predicted " + ", ".join(class_names))
    lines.append(np.array2string(matrix))
    lines.append("")
    lines.append("OpenCV single-image predictions:")
    lines.extend(predict_uploaded_scans(model, class_names, test_dir))
    (OUTPUT_DIR / "evaluation.txt").write_text("\n".join(lines), encoding="utf-8")

    model_path = OUTPUT_DIR / "cancer_mobilenetv2.keras"
    model.save(model_path)
    print(f"Saved {model_path}")
    print(f"Wrote {OUTPUT_DIR / 'evaluation.txt'}")


if __name__ == "__main__":
    main()
