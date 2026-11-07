"""Central tendency analysis and preparation of tabular data for deep learning.

Resources used by this project:
- Statistical Data Analysis with Pandas and NumPy
  https://pandas.pydata.org/docs/user_guide/index.html
- Handling Outliers and Data Normalization
  https://scikit-learn.org/stable/modules/preprocessing.html

The script builds a reproducible clinical-style table, then:
1. Summarizes each feature with the mean, median, and mode.
2. Plots the distribution and reports skewness.
3. Fills missing values and clips outliers with the IQR rule and with z-scores.
4. Scales the cleaned matrix so it can be fed to a neural network.
"""

from pathlib import Path

import matplotlib

# Save figures from a script. No interactive window is required.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Paths stay next to this file so the project folder is self-contained.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

FEATURES = [
    "age",
    "glucose",
    "bmi",
    "blood_pressure",
    "cholesterol",
    "heart_rate",
]

# Continuous measurements almost never repeat exactly, so the mode is taken
# after rounding each feature to a precision that matches how it is recorded.
MODE_DECIMALS = {
    "age": 0,
    "glucose": 0,
    "bmi": 1,
    "blood_pressure": 0,
    "cholesterol": 0,
    "heart_rate": 0,
}


def feature_label(name):
    """Turn a column name into a short chart title."""
    return name.replace("_", " ").title()


def describe_skew(value):
    """Map a skewness coefficient to a plain-language shape label.

    The usual teaching cutoffs are +/- 0.5. Between them the feature is
    close enough to symmetric that the mean and the median nearly agree.
    """
    if value > 0.5:
        return "right-skewed"
    if value < -0.5:
        return "left-skewed"
    return "approximately symmetric"


def build_dataset(n_rows=1500, seed=42):
    """Create a fixed lab-style table with missing cells and extreme values.

    Glucose and cholesterol are drawn from lognormal distributions, so their
    raw histograms are right-skewed. A handful of impossible readings are then
    written on top, which gives the outlier step something real to catch.
    """
    rng = np.random.default_rng(seed)

    frame = pd.DataFrame(
        {
            "age": rng.normal(46, 14, n_rows).clip(18, 85),
            # log(100) is about 4.6, so this cluster sits near a normal fasting glucose.
            "glucose": rng.lognormal(4.62, 0.16, n_rows),
            "bmi": rng.normal(27.5, 4.5, n_rows).clip(16, 45),
            "blood_pressure": rng.normal(122, 12, n_rows).clip(90, 160),
            # Centered near 190 mg/dL, with a long right tail.
            "cholesterol": rng.lognormal(5.25, 0.18, n_rows),
            "heart_rate": rng.normal(74, 8, n_rows).clip(52, 110),
        }
    )

    # Plant extreme readings before hiding any cells, so the outliers survive.
    outlier_idx = rng.choice(n_rows, size=18, replace=False)
    frame.loc[outlier_idx[:8], "glucose"] = rng.uniform(280, 420, size=8)
    frame.loc[outlier_idx[8:14], "cholesterol"] = rng.uniform(450, 650, size=6)
    frame.loc[outlier_idx[14:], "blood_pressure"] = rng.uniform(200, 240, size=4)

    missing_rates = {
        "glucose": 0.04,
        "bmi": 0.03,
        "blood_pressure": 0.05,
        "cholesterol": 0.04,
        "heart_rate": 0.02,
    }
    available = np.setdiff1d(np.arange(n_rows), outlier_idx)
    for column, rate in missing_rates.items():
        chosen = rng.choice(available, size=int(rate * n_rows), replace=False)
        frame.loc[chosen, column] = np.nan

    return frame


def rounded_mode(series, decimals):
    """Return the most common value after rounding, or NaN for an empty series."""
    rounded = series.dropna().round(decimals)
    modes = rounded.mode()
    if modes.empty:
        return np.nan
    return float(modes.iloc[0])


def central_tendency_table(frame):
    """Mean, median, and mode for every feature, plus the skewness coefficient."""
    rows = []
    for column in FEATURES:
        series = frame[column]
        skewness = float(series.skew())
        rows.append(
            {
                "feature": column,
                "count": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "mean": series.mean(),
                "median": series.median(),
                "mode": rounded_mode(series, MODE_DECIMALS[column]),
                "std": series.std(),
                "skewness": skewness,
                "shape": describe_skew(skewness),
            }
        )
    return pd.DataFrame(rows)


def iqr_bounds(series, k=1.5):
    """Tukey fences. Points outside [Q1 - k*IQR, Q3 + k*IQR] are outliers."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    spread = q3 - q1
    return q1 - k * spread, q3 + k * spread


def zscore_bounds(series, threshold=3.0):
    """Mean +/- threshold standard deviations.

    Population standard deviation (ddof=0) matches the z-score used by
    sklearn.preprocessing.scale.
    """
    center = series.mean()
    scale = series.std(ddof=0)
    return center - threshold * scale, center + threshold * scale


def clip_column(series, lower, upper):
    """Pull every value back inside [lower, upper] without dropping the row."""
    return series.clip(lower=lower, upper=upper)


def outlier_report(frame):
    """Count how many imputed values fall outside each rule."""
    rows = []
    for column in FEATURES:
        series = frame[column]
        iqr_low, iqr_high = iqr_bounds(series)
        z_low, z_high = zscore_bounds(series)
        rows.append(
            {
                "feature": column,
                "iqr_low": iqr_low,
                "iqr_high": iqr_high,
                "iqr_outliers": int(((series < iqr_low) | (series > iqr_high)).sum()),
                "z_low": z_low,
                "z_high": z_high,
                "zscore_outliers": int(((series < z_low) | (series > z_high)).sum()),
            }
        )
    return pd.DataFrame(rows)


def clip_frame(frame, bound_fn):
    """Return a copy with every feature clipped by bound_fn(series)."""
    clipped = frame.copy()
    for column in FEATURES:
        lower, upper = bound_fn(clipped[column])
        clipped[column] = clip_column(clipped[column], lower, upper)
    return clipped


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_histograms(frame, filename, title):
    """One histogram per feature, with the mean and the median drawn on top."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle(title)
    for ax, column in zip(axes.ravel(), FEATURES):
        series = frame[column].dropna()
        sns.histplot(series, kde=True, ax=ax, color="#2c7fb8")
        ax.axvline(series.mean(), color="#d95f0e", linestyle="--", label="mean")
        ax.axvline(series.median(), color="#1a9850", linestyle="-", label="median")
        skewness = series.skew()
        ax.set_title(f"{feature_label(column)}  skew={skewness:.2f}")
        ax.set_xlabel(feature_label(column))
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, filename)


def plot_boxplots(before, after, filename):
    """Side-by-side boxplots show which tails the clipping step removed."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle("Outlier clipping: imputed values vs IQR and z-score clipped values")
    for ax, column in zip(axes.ravel(), FEATURES):
        chart = pd.DataFrame(
            {
                "imputed": before[column],
                "clipped": after[column],
            }
        )
        sns.boxplot(data=chart, ax=ax, palette=["#9ecae1", "#31a354"])
        ax.set_title(feature_label(column))
    fig.tight_layout()
    save_figure(fig, filename)


def prepare_for_deep_learning(cleaned):
    """Scale cleaned features and write a train/validation split.

    MinMaxScaler maps every column to [0, 1], which is the usual input range
    for a network with bounded activations. StandardScaler is also fit so the
    z-score form from the preprocessing guide is available for comparison.
    There is no target column: this project only prepares the feature matrix.
    """
    minmax = MinMaxScaler()
    standard = StandardScaler()
    scaled_01 = minmax.fit_transform(cleaned[FEATURES])
    scaled_z = standard.fit_transform(cleaned[FEATURES])

    indices = np.arange(len(cleaned))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    np.savez(
        OUTPUT_DIR / "dl_ready.npz",
        X_train=scaled_01[train_idx],
        X_val=scaled_01[val_idx],
        X_train_standard=scaled_z[train_idx],
        X_val_standard=scaled_z[val_idx],
        feature_names=np.array(FEATURES),
    )

    preview = pd.DataFrame(scaled_01, columns=FEATURES)
    preview.to_csv(OUTPUT_DIR / "dl_ready_minmax.csv", index=False)

    scaler_stats = pd.DataFrame(
        {
            "feature": FEATURES,
            "minmax_data_min": minmax.data_min_,
            "minmax_data_max": minmax.data_max_,
            "standard_mean": standard.mean_,
            "standard_scale": standard.scale_,
        }
    )
    scaler_stats.to_csv(OUTPUT_DIR / "scaler_statistics.csv", index=False)
    return preview, scaler_stats


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    raw = build_dataset()
    raw.to_csv(DATA_DIR / "measurements.csv", index=False)
    print(f"Raw table: {len(raw)} rows, {raw[FEATURES].isna().sum().sum()} missing cells")

    # --- 1. Central tendency on the raw measurements ---------------------
    raw_summary = central_tendency_table(raw)
    raw_summary.to_csv(OUTPUT_DIR / "central_tendency_raw.csv", index=False)
    print("\nCentral tendency (raw data, missing values ignored):")
    print(raw_summary.round(3).to_string(index=False))
    print(
        "\nWhere the mean sits above the median, the right tail is pulling it. "
        "The median is the more resistant center for those features."
    )

    plot_histograms(
        raw,
        "histograms_raw.png",
        "Raw distributions (missing values dropped only for the plot)",
    )

    # --- 2. Missing values ------------------------------------------------
    # The median ignores the extreme readings that would drag a mean imputation.
    imputed = raw.copy()
    imputed[FEATURES] = imputed[FEATURES].fillna(imputed[FEATURES].median())
    print("\nMissing cells after median imputation:", int(imputed[FEATURES].isna().sum().sum()))

    # --- 3. Outliers: IQR fences and z-score clipping ---------------------
    report = outlier_report(imputed)
    report.to_csv(OUTPUT_DIR / "outlier_report.csv", index=False)
    print("\nOutliers detected on the imputed table:")
    print(report.round(3).to_string(index=False))

    # IQR clipping first. It does not assume a symmetric distribution, so it
    # is the right tool for the skewed glucose and cholesterol columns.
    iqr_clipped = clip_frame(imputed, iqr_bounds)

    # Z-score clipping second, using the mean and scale of the IQR-clipped
    # data. Doing it in this order keeps a few extreme points from inflating
    # the standard deviation and hiding the rest of the tail.
    cleaned = clip_frame(iqr_clipped, zscore_bounds)

    plot_histograms(
        cleaned,
        "histograms_cleaned.png",
        "Distributions after imputation and clipping",
    )
    plot_boxplots(imputed, cleaned, "boxplots_raw_vs_cleaned.png")

    cleaned_summary = central_tendency_table(cleaned)
    cleaned_summary.to_csv(OUTPUT_DIR / "central_tendency_cleaned.csv", index=False)
    print("\nCentral tendency after cleaning:")
    print(cleaned_summary.round(3).to_string(index=False))

    # --- 4. Normalize and export a matrix a network can train on ---------
    preview, scaler_stats = prepare_for_deep_learning(cleaned)
    print("\nScaler ranges learned from the cleaned table:")
    print(scaler_stats.round(3).to_string(index=False))
    print("\nFirst five rows scaled to [0, 1] (MinMaxScaler):")
    print(preview.head().round(3).to_string(index=False))
    print(f"\nDeep-learning arrays written to {OUTPUT_DIR / 'dl_ready.npz'}")
    print("X_train / X_val are MinMax scaled. X_train_standard / X_val_standard are z-scored.")


if __name__ == "__main__":
    main()
