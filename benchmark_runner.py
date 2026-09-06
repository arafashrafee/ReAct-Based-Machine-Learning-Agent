"""
CSE445 Machine Learning Assignment #3
Benchmark runner: 3 algorithms across 2 datasets with cross-validation.

Algorithms:
- Logistic Regression
- Decision Tree
- Random Forest

Datasets:
- Iris
- Wine

The script reports mean CV accuracy and standard deviation, then prints
a Markdown comparison table and identifies the best result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


@dataclass
class BenchmarkResult:
    dataset: str
    algorithm: str
    mean_accuracy: float
    std_accuracy: float


def get_dataset(dataset_name: str):
    """Return X, y for a supported dataset."""
    if dataset_name == "iris":
        data = load_iris()
    elif dataset_name == "wine":
        data = load_wine()
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Use 'iris' or 'wine'."
        )

    return data.data, data.target


def get_models():
    """Return the three required benchmark models."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }


def run_benchmark(
    datasets: List[str] | None = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Run stratified k-fold cross-validation."""
    if datasets is None:
        datasets = ["iris", "wine"]

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    results: List[BenchmarkResult] = []

    for dataset_name in datasets:
        X, y = get_dataset(dataset_name)

        for algorithm_name, model in get_models().items():
            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring="accuracy",
                n_jobs=None,
            )

            results.append(
                BenchmarkResult(
                    dataset=dataset_name,
                    algorithm=algorithm_name,
                    mean_accuracy=float(np.mean(scores)),
                    std_accuracy=float(np.std(scores, ddof=1)),
                )
            )

    return pd.DataFrame(
        [
            {
                "Dataset": result.dataset,
                "Algorithm": result.algorithm,
                "Mean CV Accuracy": result.mean_accuracy,
                "Std. Deviation": result.std_accuracy,
            }
            for result in results
        ]
    )


def main():
    print("=" * 70)
    print("CSE445 ML ASSIGNMENT #3 — BENCHMARK")
    print("=" * 70)
    print("Datasets: Iris, Wine")
    print("Algorithms: Logistic Regression, Decision Tree, Random Forest")
    print("Cross-validation: Stratified 5-Fold")
    print("Random state: 42")
    print()

    df = run_benchmark()

    # Keep the numerical output easy to read while preserving full precision
    # in the underlying DataFrame.
    display_df = df.copy()
    display_df["Mean CV Accuracy"] = display_df["Mean CV Accuracy"].map(
        lambda x: f"{x:.4f}"
    )
    display_df["Std. Deviation"] = display_df["Std. Deviation"].map(
        lambda x: f"{x:.4f}"
    )

    print("RESULTS")
    print("-" * 70)
    print(display_df.to_string(index=False))

    print("\nMARKDOWN COMPARISON TABLE")
    print("-" * 70)
    print(display_df.to_markdown(index=False))

    best_idx = df["Mean CV Accuracy"].idxmax()
    best = df.loc[best_idx]

    print("\nBEST OVERALL CV RESULT")
    print("-" * 70)
    print(
        f"{best['Algorithm']} on {best['Dataset']} | "
        f"Mean CV Accuracy = {best['Mean CV Accuracy']:.4f} | "
        f"Std. Deviation = {best['Std. Deviation']:.4f}"
    )

    # Also provide the best model separately for each dataset.
    print("\nBEST MODEL PER DATASET")
    print("-" * 70)
    for dataset_name in df["Dataset"].unique():
        subset = df[df["Dataset"] == dataset_name]
        row = subset.loc[subset["Mean CV Accuracy"].idxmax()]
        print(
            f"{dataset_name}: {row['Algorithm']} | "
            f"Mean CV Accuracy = {row['Mean CV Accuracy']:.4f} | "
            f"Std. Deviation = {row['Std. Deviation']:.4f}"
        )


if __name__ == "__main__":
    main()
