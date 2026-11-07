"""Classify clothing images with Fashion-MNIST.

Resource used for the dataset, the 0..1 scaling, and the prediction plots:
https://www.tensorflow.org/tutorials/keras/classification

That tutorial builds a dense network (Flatten, Dense 128, Dense 10). This
script keeps the same data pipeline, class names, logit head, and
probability plots, and inserts the convolutional stack the assignment asks
for: Conv2D, max pooling, then a dense classifier.

The tutorial trains for 10 epochs. EPOCHS below is a CPU-friendly default
with the same optimizer and loss. Set EPOCHS = 10 to match the tutorial.
"""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

# Class names are not stored in the dataset. This is the tutorial's list.
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

EPOCHS = 5
BATCH_SIZE = 128


def load_fashion_mnist():
    """60,000 training images and 10,000 test images, each 28x28 grayscale."""
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    return (train_images, train_labels), (test_images, test_labels)


def preprocess(train_images, test_images):
    """Scale pixels to [0, 1] and add the channel axis required by Conv2D.

    Training and test images are divided by the same constant, 255, as in
    the TensorFlow classification tutorial.
    """
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
    return train_images, test_images


def build_model():
    """Conv / pool feature extractor followed by a dense classifier.

    The last layer has no activation. Its raw scores are logits, and
    SparseCategoricalCrossentropy(from_logits=True) applies the log-softmax
    in a numerically stable way. A separate Softmax layer is attached later
    only when we need probabilities for the plots.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_training_samples(images, labels):
    """First 25 training articles, with the class name under each image."""
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Fashion-MNIST training samples")
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.squeeze(images[i]), cmap=plt.cm.binary)
        plt.xlabel(CLASS_NAMES[int(labels[i])])
    fig.tight_layout()
    save_figure(fig, "sample_images.png")


def plot_image(ax, index, predictions_array, true_label, images):
    """One test image labeled with the predicted class and its confidence."""
    true = int(true_label[index])
    image = np.squeeze(images[index])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image, cmap=plt.cm.binary)

    predicted = int(np.argmax(predictions_array))
    color = "tab:blue" if predicted == true else "tab:red"
    ax.set_xlabel(
        "{} {:2.0f}% ({})".format(
            CLASS_NAMES[predicted],
            100 * np.max(predictions_array),
            CLASS_NAMES[true],
        ),
        color=color,
        fontsize=8,
    )


def plot_value_array(ax, index, predictions_array, true_label):
    """Bar chart of the 10 class probabilities. Blue is the true class."""
    true = int(true_label[index])
    ax.grid(False)
    ax.set_xticks(range(10))
    ax.set_yticks([])
    bars = ax.bar(range(10), predictions_array, color="#777777")
    ax.set_ylim([0, 1])
    predicted = int(np.argmax(predictions_array))
    bars[predicted].set_color("tab:red")
    bars[true].set_color("tab:blue")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(42)
    print("TensorFlow", tf.__version__)

    (train_images, train_labels), (test_images, test_labels) = load_fashion_mnist()
    print("Training images:", train_images.shape, "labels:", len(train_labels))
    print("Test images:", test_images.shape, "labels:", len(test_labels))

    # Look at the raw range before scaling. Pixels are stored as 0..255.
    print(
        "Pixel range before scaling: "
        f"{int(train_images.min())}..{int(train_images.max())}"
    )
    train_images, test_images = preprocess(train_images, test_images)
    plot_training_samples(train_images, train_labels)

    model = build_model()
    model.summary()
    model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("\nTest accuracy:", test_acc)

    model_path = OUTPUT_DIR / "fashion_cnn.keras"
    model.save(model_path)
    print(f"Saved {model_path}")

    # Softmax turns logits into probabilities that sum to 1.
    probability_model = tf.keras.Sequential(
        [model, tf.keras.layers.Softmax()]
    )
    predictions = probability_model.predict(test_images, verbose=0)

    print("First test image probabilities:")
    print(np.array2string(predictions[0], precision=4, suppress_small=True))
    print(
        "Predicted:",
        CLASS_NAMES[int(np.argmax(predictions[0]))],
        "| actual:",
        CLASS_NAMES[int(test_labels[0])],
    )

    # A grid of test images next to their probability bars.
    # Correct predictions are blue and incorrect ones are red, as in the tutorial.
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    # constrained layout keeps the long class-name captions inside the figure.
    fig, axes = plt.subplots(
        num_rows,
        num_cols * 2,
        figsize=(14, 12),
        layout="constrained",
    )
    fig.suptitle("Test predictions (blue = correct, red = incorrect)")
    for i in range(num_images):
        row, col = divmod(i, num_cols)
        plot_image(axes[row, col * 2], i, predictions[i], test_labels, test_images)
        plot_value_array(axes[row, col * 2 + 1], i, predictions[i], test_labels)
    save_figure(fig, "predictions.png")

    # Single-image prediction. Keras expects a batch, so the image is wrapped
    # in a leading axis of length 1.
    image = test_images[1]
    print("Single image shape:", image.shape)
    batched = np.expand_dims(image, 0)
    print("Batch shape:", batched.shape)
    single = probability_model.predict(batched, verbose=0)
    predicted_index = int(np.argmax(single[0]))
    actual_index = int(test_labels[1])
    print(
        f"Single-image prediction: {CLASS_NAMES[predicted_index]} "
        f"({single[0][predicted_index]:.1%}), "
        f"actual {CLASS_NAMES[actual_index]}"
    )

    fig, ax = plt.subplots(figsize=(8, 3))
    plot_value_array(ax, 1, single[0], test_labels)
    ax.set_xticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_title(
        f"Predicted {CLASS_NAMES[predicted_index]} "
        f"(actual {CLASS_NAMES[actual_index]})"
    )
    fig.tight_layout()
    save_figure(fig, "single_prediction.png")

    report = [
        "Clothing image classification — Fashion-MNIST CNN",
        f"epochs={EPOCHS}",
        f"test_loss={test_loss:.6f}",
        f"test_accuracy={test_acc:.6f}",
        f"first_image_predicted={CLASS_NAMES[int(np.argmax(predictions[0]))]}",
        f"first_image_actual={CLASS_NAMES[int(test_labels[0])]}",
        f"single_image_predicted={CLASS_NAMES[predicted_index]}",
        f"single_image_actual={CLASS_NAMES[actual_index]}",
        f"single_image_confidence={float(single[0][predicted_index]):.4f}",
    ]
    (OUTPUT_DIR / "evaluation.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {OUTPUT_DIR / 'evaluation.txt'}")


if __name__ == "__main__":
    main()
