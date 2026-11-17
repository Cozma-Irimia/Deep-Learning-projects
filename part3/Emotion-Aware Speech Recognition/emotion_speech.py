"""Recognize emotion in a speaking voice from MFCC features.

Resources:
https://librosa.org/doc/latest/index.html
https://zenodo.org/records/1188976

Librosa loads each clip and computes mel-frequency cepstral coefficients,
the usual compact picture of a voice. A small convolutional network then
classifies the five emotions named for this project: Neutral, Happy, Sad,
Angry, and Fear. The clips are the speech portion of RAVDESS (Livingstone
and Russo, 2018), record 1188976 on Zenodo. The Zenodo zip host refused
this network, so the same Actor folders are read from a Hugging Face
parquet mirror. Actors held out of training are the test voices, so the
score is not inflated by hearing the same person in both splits.
"""

import io
import os
import urllib.request
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
AUDIO_ROOT = DATA_DIR / "ravdess"
CACHE_PATH = DATA_DIR / "ravdess_mfcc.npz"
SHARD_NAMES = [f"ravdess_emo_{index:04d}.parquet" for index in range(6)]
SHARD_URL = (
    "https://huggingface.co/datasets/TwinkStart/RAVDESS/resolve/"
    "refs%2Fconvert%2Fparquet/default/ravdess_emo/{name}"
)

# Calm, disgust, and surprise are in RAVDESS too. This project keeps the
# five emotions requested on the assignment card.
EMOTION_NAMES = {
    "neutral": "Neutral",
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "fearful": "Fear",
}
CLASS_NAMES = ["Neutral", "Happy", "Sad", "Angry", "Fear"]
# Actors 22-24 are unseen at training time. Actor 21 is validation only.
TEST_ACTORS = {22, 23, 24}
VAL_ACTORS = {21}

SAMPLE_RATE = 48000
N_MFCC = 40
# About 3.5 seconds at librosa's default hop of 512 samples.
N_FRAMES = 328
BATCH_SIZE = 32
EPOCHS = 40


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def download_shards():
    """Fetch the speech mirror. Each shard is one slice of the 1,440 clips."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in SHARD_NAMES:
        destination = DATA_DIR / name
        source_name = name.replace("ravdess_emo_", "")
        if destination.exists() and destination.stat().st_size > 40_000_000:
            paths.append(destination)
            continue
        url = SHARD_URL.format(name=source_name)
        print(f"Downloading {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        print(f"  {destination.stat().st_size // (1024 * 1024)} MB", flush=True)
        paths.append(destination)
    return paths


def actor_id(wav_path):
    """RAVDESS filenames end with the actor number."""
    return int(Path(wav_path).stem.split("-")[-1])


def read_audio(payload):
    """Decode a WAV payload to mono float audio at the training sample rate."""
    audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    if audio.size == 0:
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    return audio


def fit_frames(coeffs):
    """Center-crop or pad a coefficient matrix to a fixed width."""
    frames = coeffs.shape[1]
    if frames == N_FRAMES:
        return coeffs
    if frames > N_FRAMES:
        start = (frames - N_FRAMES) // 2
        return coeffs[:, start : start + N_FRAMES]
    pad = N_FRAMES - frames
    left = pad // 2
    return np.pad(coeffs, ((0, 0), (left, pad - left)))


def compute_mfcc(audio):
    coeffs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    return fit_frames(coeffs).astype(np.float32)


def load_features():
    if CACHE_PATH.exists():
        cache = np.load(CACHE_PATH, allow_pickle=True)
        return cache["features"], cache["labels"], cache["actors"], list(cache["paths"])

    features = []
    labels = []
    actors = []
    paths = []
    kept = 0
    for shard in download_shards():
        print(f"Reading {shard.name}")
        for row in pq.read_table(shard, columns=["emotion", "WavPath", "audio"]).to_pylist():
            emotion = EMOTION_NAMES.get(str(row["emotion"]).lower())
            if emotion is None:
                continue
            relative = str(row["WavPath"]).replace("\\", "/")
            audio = read_audio(row["audio"]["bytes"])
            destination = AUDIO_ROOT / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            sf.write(destination, audio, SAMPLE_RATE)
            features.append(compute_mfcc(audio))
            labels.append(CLASS_NAMES.index(emotion))
            actors.append(actor_id(relative))
            paths.append(relative)
            kept += 1
            if kept % 100 == 0:
                print(f"  {kept} clips", flush=True)
    print(f"Kept {kept} clips")
    features = np.stack(features)
    labels = np.asarray(labels, dtype=np.int32)
    actors = np.asarray(actors, dtype=np.int32)
    np.savez(CACHE_PATH, features=features, labels=labels, actors=actors, paths=np.array(paths))
    print(f"Saved {CACHE_PATH}")
    return features, labels, actors, paths


def split_by_actor(actors):
    """Keep every clip from an actor on one side of the split."""
    actors = np.asarray(actors)
    train = ~np.isin(actors, list(TEST_ACTORS | VAL_ACTORS))
    validation = np.isin(actors, list(VAL_ACTORS))
    test = np.isin(actors, list(TEST_ACTORS))
    return train, validation, test


def add_deltas(features):
    """Stack each MFCC with its first and second differences.

    Librosa's delta is the usual extra view of how the voice changes,
    and the three planes become the channels of the classifier.
    """
    channels = []
    for coeffs in features:
        first = librosa.feature.delta(coeffs)
        second = librosa.feature.delta(coeffs, order=2)
        channels.append(np.stack([coeffs, first, second], axis=-1))
    return np.stack(channels).astype(np.float32)


def standardize(train, validation, test):
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-6)
    return (train - mean) / std, (validation - mean) / std, (test - mean) / std


def build_model():
    """CNN over MFCC, delta, and delta-delta. Noise is active only while training.

    Pooling is heavier along time so the dense layer stays small. A flat
    50k-wide vector overfit these few hundred clips.
    """
    inputs = keras.Input(shape=(N_MFCC, N_FRAMES, 3))
    x = layers.GaussianNoise(0.1)(inputs)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 4))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 4))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 4))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def plot_samples(features, labels, paths):
    """One training clip per emotion: waveform beside its MFCC picture."""
    fig, axes = plt.subplots(len(CLASS_NAMES), 2, figsize=(10, 11), layout="constrained")
    fig.suptitle("RAVDESS speech features")
    for class_index, name in enumerate(CLASS_NAMES):
        match = np.where(labels == class_index)[0][0]
        audio, _ = librosa.load(AUDIO_ROOT / paths[match], sr=SAMPLE_RATE, mono=True)
        times = np.arange(len(audio)) / SAMPLE_RATE
        axes[class_index, 0].plot(times, audio, color="#4c78a8", linewidth=0.6)
        axes[class_index, 0].set_ylabel(name)
        axes[class_index, 0].set_yticks([])
        # Skip the first coefficient. It is loud energy and hides the rest.
        image = librosa.display.specshow(
            features[match][1:],
            sr=SAMPLE_RATE,
            x_axis="time",
            ax=axes[class_index, 1],
            cmap="magma",
        )
        axes[class_index, 1].set_ylabel("MFCC")
        if class_index < len(CLASS_NAMES) - 1:
            axes[class_index, 0].set_xlabel("")
            axes[class_index, 1].set_xlabel("")
    axes[-1, 0].set_xlabel("Time (s)")
    fig.colorbar(image, ax=axes[:, 1], fraction=0.046, pad=0.04)
    save_figure(fig, "sample_features.png")


def plot_history(history):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    epochs = range(1, len(history.history["accuracy"]) + 1)
    axes[0].plot(epochs, history.history["accuracy"], label="Training accuracy")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend(loc="lower right")
    axes[1].plot(epochs, history.history["loss"], label="Training loss")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, "training_history.png")


def plot_confusion(labels, predicted):
    matrix = confusion_matrix(labels, predicted, labels=list(range(len(CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30, ha="right")
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


def example_index(labels, predicted, probabilities, class_index):
    """Prefer a correct held-out clip, and among those the most confident."""
    matches = np.where(labels == class_index)[0]
    correct = matches[predicted[matches] == class_index]
    pool = correct if len(correct) else matches
    confidences = probabilities[pool, predicted[pool]]
    return int(pool[int(np.argmax(confidences))])


def plot_predictions(features, labels, probabilities):
    predicted = np.argmax(probabilities, axis=1)
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(14, 3.2))
    fig.suptitle("Held-out voices")
    for axis, class_index in zip(axes, range(len(CLASS_NAMES))):
        index = example_index(labels, predicted, probabilities, class_index)
        guess = int(predicted[index])
        confidence = float(probabilities[index, guess])
        axis.imshow(features[index], aspect="auto", origin="lower", cmap="magma")
        correct = guess == class_index
        axis.set_title(
            f"{CLASS_NAMES[guess]} {confidence:.0%}\n(actual {CLASS_NAMES[class_index]})",
            color="#1a9850" if correct else "#d73027",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    save_figure(fig, "predictions.png")


def emotion_sentence(class_name, confidence):
    return f"Emotion detected: {class_name} ({confidence:.1%} confidence)"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(42)

    features, labels, actors, paths = load_features()
    train_mask, val_mask, test_mask = split_by_actor(actors)
    for name in CLASS_NAMES:
        class_index = CLASS_NAMES.index(name)
        print(
            f"{name:8s} train {np.sum((labels == class_index) & train_mask):3d}"
            f"  val {np.sum((labels == class_index) & val_mask):3d}"
            f"  test {np.sum((labels == class_index) & test_mask):3d}"
        )

    plot_samples(features[train_mask], labels[train_mask], [paths[i] for i in np.where(train_mask)[0]])

    features = add_deltas(features)
    x_train, x_val, x_test = standardize(
        features[train_mask], features[val_mask], features[test_mask]
    )
    y_train = labels[train_mask]
    y_val = labels[val_mask]
    y_test = labels[test_mask]

    model = build_model()
    model.summary()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    plot_history(history)

    probabilities = model.predict(x_test, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(predicted == y_test))
    majority = float(np.mean(y_test == np.bincount(y_train).argmax()))
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Majority baseline: {majority:.4f}")

    matrix = plot_confusion(y_test, predicted)
    plot_predictions(x_test[..., 0], y_test, probabilities)

    lines = [
        "Emotion-aware speech — RAVDESS, librosa MFCC, convolutional classifier",
        "Test actors 22-24 were not heard during training. Actor 21 is validation only.",
        f"test_accuracy={accuracy:.4f}",
        f"majority_baseline={majority:.4f}",
        "",
        "Most confident correct clip of each emotion:",
    ]
    print("\nSample decisions:")
    for class_index, name in enumerate(CLASS_NAMES):
        index = example_index(y_test, predicted, probabilities, class_index)
        guess = int(predicted[index])
        confidence = float(probabilities[index, guess])
        sentence = emotion_sentence(CLASS_NAMES[guess], confidence)
        line = f"actual {name:8s} {sentence}"
        print(line)
        lines.append(line)
    lines.append("")
    lines.append("per-class recall on held-out actors:")
    for class_index, name in enumerate(CLASS_NAMES):
        support = matrix[class_index].sum()
        recall = matrix[class_index, class_index] / support if support else 0.0
        line = f"{name:8s} {recall:.3f} ({support} clips)"
        print(line)
        lines.append(line)
    lines.append("")
    lines.append("confusion matrix rows=actual, columns=predicted " + ", ".join(CLASS_NAMES))
    lines.append(np.array2string(matrix))
    (OUTPUT_DIR / "evaluation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    model_path = OUTPUT_DIR / "emotion_cnn.keras"
    model.save(model_path)
    print(f"Saved {model_path}")
    print(f"Wrote {OUTPUT_DIR / 'evaluation.txt'}")


if __name__ == "__main__":
    main()
