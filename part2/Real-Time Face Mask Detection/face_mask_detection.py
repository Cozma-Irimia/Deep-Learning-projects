"""Detect faces in a video stream and label each one MASK or NO MASK.

Resource:
https://docs.opencv.org/5.0/

Frames come from cv2.VideoCapture (a webcam index or a video file). Each
face is localized with OpenCV's YuNet DNN detector, FaceDetectorYN. This
OpenCV 5 build does not ship the older Haar CascadeClassifier or the Caffe
SSD loader; YuNet is the face detector documented for this version.

The crop is resized, scaled to [0, 1] by a Rescaling layer, and classified
by a small CNN. MASK is drawn in green and NO MASK in red.

Run a webcam with:
    python face_mask_detection.py --source 0
"""

import argparse
import json
import os
import urllib.request
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

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 8


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _has_mask_folders(path):
    return (path / "with_mask").is_dir() and (path / "without_mask").is_dir()


def download_mask_dataset():
    """Fetch the Prajna Bhandary with_mask / without_mask images from GitHub."""
    destination = DATA_DIR / "mask_photos"
    for folder in ("with_mask", "without_mask"):
        target = destination / folder
        target.mkdir(parents=True, exist_ok=True)
        api = (
            "https://api.github.com/repos/prajnasb/observations/contents/"
            f"experiements/data/{folder}"
        )
        request = urllib.request.Request(api, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            entries = json.loads(response.read().decode("utf-8"))
        print(f"Downloading {len(entries)} images into {target}")
        for entry in entries:
            if entry.get("type") != "file":
                continue
            file_path = target / entry["name"]
            if not file_path.exists():
                urllib.request.urlretrieve(entry["download_url"], file_path)
    return destination


def find_image_root():
    """Locate with_mask/ and without_mask/, downloading them on a fresh checkout."""
    candidates = [
        DATA_DIR / "observations" / "experiements" / "data",
        DATA_DIR / "mask_photos",
        DATA_DIR,
    ]
    for candidate in candidates:
        if _has_mask_folders(candidate):
            return candidate
    return download_mask_dataset()


def load_splits(image_root):
    train_ds = keras.utils.image_dataset_from_directory(
        image_root,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        image_root,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )
    class_names = list(train_ds.class_names)
    print("Classes:", class_names)
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(500, seed=42).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    return train_ds, val_ds, class_names


def build_model(num_classes):
    """CNN classifier. Logits out, softmax only when a confidence is needed."""
    model = keras.Sequential(
        [
            layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
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


def ensure_yunet():
    """Download the YuNet ONNX weights used by cv2.FaceDetectorYN."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "face_detection_yunet_2023mar.onnx"
    if not path.exists():
        print(f"Downloading {YUNET_URL}")
        keras.utils.get_file(origin=YUNET_URL, fname=path.name, cache_dir=DATA_DIR, cache_subdir=".")
        if not path.exists():
            found = list(DATA_DIR.rglob(path.name))
            if not found:
                raise FileNotFoundError(path.name)
            path = found[0]
    return path


def make_detector(model_path, frame_size):
    # config is unused for ONNX. input_size is (width, height).
    return cv2.FaceDetectorYN.create(str(model_path), "", frame_size, 0.6, 0.3, 5000)


def display_name(class_name):
    if class_name == "with_mask":
        return "MASK"
    if class_name == "without_mask":
        return "NO MASK"
    return class_name


def box_color(class_name):
    """Green when a mask is worn, red otherwise. Colors are BGR for OpenCV."""
    if class_name == "with_mask":
        return (0, 200, 0)
    return (0, 0, 220)


def classify_crop(model, class_names, bgr_crop):
    """Resize one face crop and return the class name plus its probability."""
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype("float32"), 0)
    logits = model.predict(batch, verbose=0)[0]
    probabilities = tf.nn.softmax(logits).numpy()
    index = int(np.argmax(probabilities))
    return class_names[index], float(probabilities[index])


def annotate_frame(frame, detector, model, class_names):
    """Draw a MASK / NO MASK box around every detected face."""
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    annotated = frame.copy()

    if faces is None:
        # A tight portrait may already be the face. Classify the whole frame.
        class_name, confidence = classify_crop(model, class_names, frame)
        _draw_label(annotated, class_name, confidence, 0, 0, width - 1, height - 1)
        return annotated

    for face in faces:
        x, y, box_w, box_h = [int(value) for value in face[:4]]
        x0 = max(0, int(x - 0.15 * box_w))
        y0 = max(0, int(y - 0.15 * box_h))
        x1 = min(width, int(x + box_w * 1.15))
        y1 = min(height, int(y + box_h * 1.15))
        if x1 <= x0 or y1 <= y0:
            continue
        class_name, confidence = classify_crop(model, class_names, frame[y0:y1, x0:x1])
        _draw_label(annotated, class_name, confidence, x0, y0, x1, y1)
    return annotated


def _draw_label(image, class_name, confidence, x0, y0, x1, y1):
    color = box_color(class_name)
    text = f"{display_name(class_name)} {confidence:.0%}"
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    cv2.rectangle(image, (x0, max(0, y0 - 24)), (x0 + 8 * len(text), y0), color, -1)
    cv2.putText(
        image,
        text,
        (x0 + 2, max(16, y0 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def letterbox(frame, size=480):
    """Put a photo on a square canvas so every demo frame has the same shape."""
    height, width = frame.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
    canvas = np.full((size, size, 3), 30, dtype=np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def demo_on_samples(image_root, detector, model, class_names):
    """Annotate a few dataset photos and write a short video from those frames."""
    paths = []
    for folder in ("with_mask", "without_mask"):
        paths.extend(sorted((image_root / folder).glob("*.jpg"))[:4])

    frames = []
    for path in paths:
        photo = cv2.imread(str(path))
        if photo is None:
            continue
        canvas = letterbox(photo)
        frames.append(annotate_frame(canvas, detector, model, class_names))

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("MASK (green) and NO MASK (red)")
    for ax, frame in zip(axes.ravel(), frames):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.axis("off")
    for ax in axes.ravel()[len(frames) :]:
        ax.axis("off")
    save_figure(fig, "sample_detections.png")

    video_path = OUTPUT_DIR / "sample_detections.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        4,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    for frame in frames:
        for _ in range(8):
            writer.write(frame)
    writer.release()
    print(f"Saved {video_path}")
    return frames


def run_capture(source, detector, model, class_names):
    """Read a webcam or a video file until the stream ends or the user presses q."""
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"Could not open source {source!r}.")
        return False

    video_path = OUTPUT_DIR / "live_capture.avi"
    writer = None
    window_ok = True
    print("Streaming. Press q in the preview window to stop.")
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        annotated = annotate_frame(frame, detector, model, class_names)
        if writer is None:
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                20,
                (annotated.shape[1], annotated.shape[0]),
            )
        writer.write(annotated)
        if window_ok:
            try:
                cv2.imshow("Face mask detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                window_ok = False
                print("No display is available. Frames are still being saved.")
    capture.release()
    if writer is not None:
        writer.release()
        print(f"Saved {video_path}")
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    return True


def main():
    parser = argparse.ArgumentParser(description="Train a mask CNN and run it on video.")
    parser.add_argument(
        "--source",
        default="",
        help="Webcam index such as 0, or a path to a video file. "
        "Leave empty to annotate sample photos.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(42)

    image_root = find_image_root()
    train_ds, val_ds, class_names = load_splits(image_root)
    model = build_model(len(class_names))
    model.summary()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)
    plot_history(history)

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")
    model_path = OUTPUT_DIR / "mask_cnn.keras"
    model.save(model_path)
    print(f"Saved {model_path}")

    detector = make_detector(ensure_yunet(), (320, 320))
    demo_on_samples(image_root, detector, model, class_names)

    lines = [
        "Face mask detection",
        f"validation_loss={val_loss:.4f}",
        f"validation_accuracy={val_acc:.4f}",
        "labels: MASK is green, NO MASK is red",
        "face detector: OpenCV FaceDetectorYN (YuNet DNN)",
    ]
    (OUTPUT_DIR / "evaluation.txt").write_text("\n".join(lines), encoding="utf-8")

    if args.source != "":
        source = int(args.source) if args.source.isdigit() else args.source
        opened = run_capture(source, detector, model, class_names)
        if not opened:
            print("Falling back to the sample-image demo, which was already saved.")


if __name__ == "__main__":
    main()
