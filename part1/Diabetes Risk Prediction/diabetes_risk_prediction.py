"""Predict diabetes risk from the Pima Indians Diabetes dataset.

Resource used for the metrics in this project:
https://scikit-learn.org/stable/modules/model_evaluation.html

The brief asks for a Diabetic / Non-Diabetic decision from glucose, insulin,
BMI, and age. Those four columns are part of the Pima table, together with
pregnancies, blood pressure, skin thickness, and the diabetes pedigree
function. All eight clinical features are kept, because each one is part of
the published dataset and improves the classifier.

Zero was used as a sentinel for "not recorded" in glucose, blood pressure,
skin thickness, insulin, and BMI. Those zeros are treated as missing values.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "diabetes.csv"
OUTPUT_DIR = ROOT / "outputs"

# Column order of the Pima Indians Diabetes dataset.
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"

# A recorded zero is not a possible measurement for these fields.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# The brief calls these out by name, so the charts focus on them.
FOCUS_FEATURES = ["Glucose", "Insulin", "BMI", "Age"]

CLASS_NAMES = ["Non-Diabetic", "Diabetic"]


def load_dataset(path):
    """Load the CSV and replace sentinel zeros with NaN."""
    frame = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame = frame[required].copy()
    frame[ZERO_AS_MISSING] = frame[ZERO_AS_MISSING].replace(0, np.nan)
    return frame


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_correlation(frame):
    """Pearson correlations, including each feature against the diagnosis.

    pandas computes each pair from the rows where both values are present,
    so the many missing insulin readings are left out of that pair instead
    of being replaced by a constant.
    """
    corr = frame[FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Feature correlation (Pima Indians Diabetes)")
    fig.tight_layout()
    save_figure(fig, "correlation_heatmap.png")
    return corr


def plot_focus_distributions(frame):
    """Compare glucose, insulin, BMI, and age for the two classes."""
    chart = frame.copy()
    chart["Status"] = chart[TARGET].map({0: "Non-Diabetic", 1: "Diabetic"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Class distributions for the features named in the brief")
    for ax, column in zip(axes.ravel(), FOCUS_FEATURES):
        sns.histplot(
            data=chart,
            x=column,
            hue="Status",
            kde=True,
            stat="density",
            common_norm=False,
            element="step",
            ax=ax,
        )
        ax.set_title(column)
    fig.tight_layout()
    save_figure(fig, "feature_distributions.png")


def build_pipeline():
    """Impute, standardize, then fit a logistic regression.

    Median imputation is fit on the training fold only, inside the pipeline,
    so the test fold cannot leak into the fill values. StandardScaler puts
    every feature on a common scale, which logistic regression needs because
    its coefficients are tied to the size of each input.
    class_weight='balanced' counters the 2:1 majority of non-diabetic rows.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate(pipeline, x_test, y_test):
    """Accuracy, precision, recall, F1, ROC AUC, and the confusion matrix.

    Definitions follow the scikit-learn model evaluation guide. The positive
    class is Diabetic (label 1).
    """
    predicted = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)
    classes = list(pipeline.named_steps["classifier"].classes_)
    diabetic_col = classes.index(1)
    diabetic_probability = probabilities[:, diabetic_col]

    metrics = {
        "accuracy": accuracy_score(y_test, predicted),
        "precision": precision_score(y_test, predicted, pos_label=1),
        "recall": recall_score(y_test, predicted, pos_label=1),
        "f1": f1_score(y_test, predicted, pos_label=1),
        "roc_auc": roc_auc_score(y_test, diabetic_probability),
    }
    report = classification_report(
        y_test,
        predicted,
        target_names=CLASS_NAMES,
        digits=3,
    )
    matrix = confusion_matrix(y_test, predicted)
    return predicted, diabetic_probability, metrics, report, matrix


def plot_evaluation(y_test, predicted, diabetic_probability):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predicted,
        display_labels=CLASS_NAMES,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    save_figure(fig, "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, diabetic_probability, name="Logistic regression", ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "roc_curve.png")


def describe_prediction(pipeline, row):
    """Return the class name and both class probabilities for one patient."""
    values = [float(row[column]) for column in FEATURES]
    frame = pd.DataFrame([values], columns=FEATURES)
    probabilities = pipeline.predict_proba(frame)[0]
    classes = list(pipeline.named_steps["classifier"].classes_)
    diabetic_p = float(probabilities[classes.index(1)])
    non_diabetic_p = float(probabilities[classes.index(0)])
    label = "Diabetic" if diabetic_p >= 0.5 else "Non-Diabetic"
    return label, non_diabetic_p, diabetic_p


def coefficient_table(pipeline):
    """Coefficients are on the standardized scale: change per 1 standard deviation."""
    coefficients = pipeline.named_steps["classifier"].coef_[0]
    table = pd.DataFrame({"feature": FEATURES, "coefficient": coefficients})
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    data = load_dataset(DATA_PATH)
    print(f"Loaded {len(data)} patients from {DATA_PATH}")
    print("Class counts:")
    print(data[TARGET].map({0: "Non-Diabetic", 1: "Diabetic"}).value_counts().to_string())
    print("\nMissing values after treating sentinel zeros as NaN:")
    print(data[FEATURES].isna().sum().to_string())

    # Charts use the observed measurements. Median imputation happens later,
    # inside the training pipeline, and is fit on the training fold only.
    correlation = plot_correlation(data)
    plot_focus_distributions(data)
    print("\nCorrelation with the diagnosis (1 = Diabetic):")
    print(correlation[TARGET].sort_values(ascending=False).round(3).to_string())

    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[TARGET],
        test_size=0.2,
        random_state=42,
        stratify=data[TARGET],
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    predicted, diabetic_probability, metrics, report, matrix = evaluate(pipeline, x_test, y_test)
    plot_evaluation(y_test, predicted, diabetic_probability)

    majority = int(y_train.value_counts().idxmax())
    baseline_accuracy = float((y_test == majority).mean())

    print("\nMajority-class baseline accuracy: {:.3f}".format(baseline_accuracy))
    print("Logistic regression on the held-out 20%:")
    for name, value in metrics.items():
        print(f"  {name:10s} {value:.3f}")
    print("\n" + report)
    print("Confusion matrix [rows = true, columns = predicted]:")
    print(pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES))

    coefficients = coefficient_table(pipeline)
    print("\nStandardized coefficients (larger magnitude = stronger association):")
    print(coefficients.round(3).to_string(index=False))

    # Two hand-built patients. Unspecified fields sit at the training median,
    # so the contrast comes from glucose, insulin, BMI, and age.
    typical = x_train.median(numeric_only=True)
    low_risk = typical.copy()
    low_risk["Glucose"] = 90
    low_risk["Insulin"] = 80
    low_risk["BMI"] = 22
    low_risk["Age"] = 25

    high_risk = typical.copy()
    high_risk["Glucose"] = 170
    high_risk["Insulin"] = 250
    high_risk["BMI"] = 35
    high_risk["Age"] = 52

    # Also score the first five real patients in the test fold.
    examples = [
        ("low_risk_profile", low_risk),
        ("high_risk_profile", high_risk),
    ]
    prediction_rows = []
    print("\nPredictions:")
    for name, row in examples:
        label, p_non, p_yes = describe_prediction(pipeline, row)
        print(
            f"  {name}: {label} "
            f"(Non-Diabetic {p_non:.1%}, Diabetic {p_yes:.1%})"
        )
        prediction_rows.append(
            {
                "case": name,
                "prediction": label,
                "probability_non_diabetic": p_non,
                "probability_diabetic": p_yes,
                **{column: float(row[column]) for column in FEATURES},
            }
        )

    for position in range(5):
        row = x_test.iloc[position]
        actual = int(y_test.iloc[position])
        label, p_non, p_yes = describe_prediction(pipeline, row)
        actual_name = CLASS_NAMES[actual]
        print(
            f"  test[{position}]: predicted {label}, actual {actual_name} "
            f"(Diabetic {p_yes:.1%})"
        )
        prediction_rows.append(
            {
                "case": f"test_{position}",
                "prediction": label,
                "actual": actual_name,
                "probability_non_diabetic": p_non,
                "probability_diabetic": p_yes,
                **{column: float(row[column]) for column in FEATURES},
            }
        )

    pd.DataFrame(prediction_rows).to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    lines = [
        "Diabetes risk prediction — held-out evaluation",
        f"baseline_accuracy={baseline_accuracy:.4f}",
    ]
    lines.extend(f"{name}={value:.4f}" for name, value in metrics.items())
    lines.append("")
    lines.append(report)
    (OUTPUT_DIR / "evaluation.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUTPUT_DIR / 'evaluation.txt'}")


if __name__ == "__main__":
    main()
