"""Turn spoken audio into text with the SpeechRecognition library.

Resource:
https://pypi.org/project/SpeechRecognition/

The recognizer follows that guide: calibrate for room noise, record from a
microphone or from a WAV file, then send the clip to Google speech-to-text.
Before recognition, a spectral gate lowers steady background noise and a
short-time energy / zero-crossing pass extracts simple audio features.
A small stop-word filter then keeps the content words of the transcript.

Each recognized phrase is printed as soon as it is ready. Pass --mic to
listen on the default microphone; without it, the script transcribes the
sample clips in data/ so the run is reproducible.
"""

import argparse
import re
import urllib.request
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import speech_recognition as sr
from scipy.io import wavfile
from scipy.signal import istft, stft

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

# Clips used by the SpeechRecognition walkthrough: a clean Harvard sentence
# list, and the same list recorded over a jackhammer.
SAMPLE_URLS = {
    "harvard.wav": (
        "https://raw.githubusercontent.com/realpython/"
        "python-speech-recognition/master/audio_files/harvard.wav"
    ),
    "jackhammer.wav": (
        "https://raw.githubusercontent.com/realpython/"
        "python-speech-recognition/master/audio_files/jackhammer.wav"
    ),
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "it", "my", "be", "as", "at", "by", "from",
}


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def download_samples():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, url in SAMPLE_URLS.items():
        path = DATA_DIR / name
        if not path.exists() or path.stat().st_size < 10_000:
            print(f"Downloading {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                path.write_bytes(response.read())
        paths.append(path)
    return paths


def load_mono(path):
    """Read a WAV file as a float waveform in [-1, 1]."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        sample_rate, samples = wavfile.read(path)
    samples = np.asarray(samples)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32)
    peak = float(np.max(np.abs(samples))) or 1.0
    if peak > 1.0:
        samples = samples / peak
    return sample_rate, samples


def write_wav(path, sample_rate, samples):
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    wavfile.write(path, sample_rate, pcm)


def reduce_noise(samples, sample_rate):
    """Attenuate steady noise with a spectral gate.

    The quietest frames estimate the noise floor. Each frequency bin is then
    turned down in proportion to how much of it is noise. A floor keeps the
    speech from being punched into silence, and the result is not
    re-amplified, so leftover noise stays quieter than the voice.
    """
    _, _, spectrum = stft(samples, fs=sample_rate, nperseg=1024, noverlap=768)
    magnitude = np.abs(spectrum)
    energy = np.mean(magnitude**2, axis=0)
    quiet = energy <= np.quantile(energy, 0.20)
    if not np.any(quiet):
        quiet = np.zeros(energy.shape, dtype=bool)
        quiet[0] = True
    noise_floor = np.median(magnitude[:, quiet], axis=1, keepdims=True)
    gain = np.clip(1.0 - 0.85 * noise_floor / (magnitude + 1e-8), 0.25, 1.0)
    _, restored = istft(
        magnitude * gain * np.exp(1j * np.angle(spectrum)),
        fs=sample_rate,
        nperseg=1024,
        noverlap=768,
    )
    restored = restored[: len(samples)].astype(np.float32)
    return np.clip(restored, -1.0, 1.0)


def frame_features(samples, sample_rate, frame_ms=25, hop_ms=10):
    """Short-time energy and zero-crossing rate, one pair per frame."""
    frame = max(1, int(sample_rate * frame_ms / 1000))
    hop = max(1, int(sample_rate * hop_ms / 1000))
    energy = []
    crossings = []
    for start in range(0, len(samples) - frame, hop):
        chunk = samples[start : start + frame]
        energy.append(np.sqrt(np.mean(chunk**2)))
        audible = chunk.copy()
        audible[np.abs(audible) < 1e-3] = 0
        signs = np.sign(audible)
        signs[signs == 0] = 1
        crossings.append(np.mean(np.abs(np.diff(signs)) > 0))
    return np.asarray(energy), np.asarray(crossings)


def plot_audio(name, sample_rate, original, cleaned, energy, crossings):
    time = np.arange(len(original)) / sample_rate
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=False)
    axes[0].plot(time, original, color="#4c78a8", linewidth=0.6, label="Original")
    axes[0].plot(np.arange(len(cleaned)) / sample_rate, cleaned, color="#f58518", linewidth=0.6, alpha=0.8, label="Noise reduced")
    axes[0].set_title(name)
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")
    axes[1].specgram(cleaned, NFFT=512, Fs=sample_rate, noverlap=256, cmap="magma")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title("Spectrogram after noise reduction")
    feature_time = np.linspace(0, len(cleaned) / sample_rate, len(energy))
    axes[2].plot(feature_time, energy, label="Short-time energy")
    axes[2].plot(feature_time, crossings, label="Zero-crossing rate")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Frame features")
    axes[2].legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, f"{Path(name).stem}_features.png")


def content_words(text):
    """Drop function words so the transcript leaves the content terms."""
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def transcribe(recognizer, audio):
    """Return the best transcript, or a short reason when nothing comes back."""
    try:
        transcript, confidence = recognizer.recognize_google(
            audio, language="en-US", with_confidence=True
        )
    except sr.UnknownValueError:
        return "(no speech recognized)"
    except sr.RequestError as error:
        return f"(service unavailable: {error})"
    transcript = transcript.strip()
    if not transcript:
        return "(no speech recognized)"
    return f"{transcript} ({confidence:.0%} confidence)"


def recognize_file(recognizer, path):
    """Transcribe a whole clip in one request."""
    with sr.AudioFile(str(path)) as source:
        audio = recognizer.record(source)
    text = transcribe(recognizer, audio)
    print(text, flush=True)
    return text


def recognize_phrases(recognizer, path):
    """Print each phrase as the pause after it is reached.

    ``listen`` waits for speech, then stops on silence, which is the same
    loop the library uses for a live microphone. A fresh energy threshold
    keeps a loud clip from deafening the next one.
    """
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = False
    lines = []
    index = 0
    with sr.AudioFile(str(path)) as source:
        while index < 12:
            audio = recognizer.listen(source, phrase_time_limit=12)
            heard = len(audio.frame_data) / (audio.sample_width * max(audio.sample_rate, 1))
            if heard < 0.25:
                break
            index += 1
            text = transcribe(recognizer, audio)
            line = f"[{index}] {text}"
            print(line, flush=True)
            lines.append(line)
            if text == "(no speech recognized)":
                break
    return lines


def recognize_microphone(recognizer, turns=3):
    """Listen on the default microphone and print each phrase as it ends."""
    try:
        microphone = sr.Microphone()
    except (AttributeError, OSError) as error:
        print(f"Microphone unavailable ({error}).")
        return []
    lines = []
    with microphone as source:
        print("Calibrating for room noise...", flush=True)
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        for turn in range(1, turns + 1):
            print(f"Listening ({turn}/{turns}). Speak, then pause.", flush=True)
            try:
                audio = recognizer.listen(source, timeout=6, phrase_time_limit=6)
            except sr.WaitTimeoutError:
                print(f"[{turn}] (no speech heard)", flush=True)
                lines.append(f"[{turn}] (no speech heard)")
                continue
            text = transcribe(recognizer, audio)
            line = f"[{turn}] {text}"
            print(line, flush=True)
            lines.append(line)
    return lines


def prepare_clip(path):
    sample_rate, original = load_mono(path)
    cleaned = reduce_noise(original, sample_rate)
    energy, crossings = frame_features(cleaned, sample_rate)
    plot_audio(path.name, sample_rate, original, cleaned, energy, crossings)
    cleaned_path = OUTPUT_DIR / f"{path.stem}_cleaned.wav"
    write_wav(cleaned_path, sample_rate, cleaned)
    print(f"Saved {cleaned_path}")
    return cleaned_path


def main():
    parser = argparse.ArgumentParser(description="Transcribe speech to text.")
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Listen on the default microphone instead of the sample files.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recognizer = sr.Recognizer()
    report = ["Speech recognition — SpeechRecognition + Google speech-to-text", ""]

    if args.mic:
        print("Live microphone. Each phrase is printed when you pause.")
        lines = recognize_microphone(recognizer)
        report.append("Microphone:")
        report.extend(lines or ["(no microphone input)"])
    else:
        samples = download_samples()
        full_transcripts = []
        for path in samples:
            print(f"\n{path.name}")
            cleaned_path = prepare_clip(path)
            report.append(path.name)
            report.append("original:")
            print("original:")
            full_text = recognize_file(recognizer, path)
            report.append(full_text)
            if "no speech recognized" not in full_text and "service unavailable" not in full_text:
                full_transcripts.append(full_text)
            report.append("noise-reduced:")
            print("noise-reduced:")
            report.extend(recognize_phrases(recognizer, cleaned_path))
            report.append("")

        transcript = " ".join(full_transcripts)
        words = content_words(re.sub(r"\(\d+% confidence\)", "", transcript))
        keyword_line = "Content words: " + (", ".join(words) if words else "(none)")
        print(keyword_line)
        report.append(keyword_line)

    report_path = OUTPUT_DIR / "transcript.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
