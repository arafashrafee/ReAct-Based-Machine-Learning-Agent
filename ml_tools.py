# ml_tools.py
# CSE445 Assignment #3 - Autonomous Local LLM ML Agent
# Mac-native version (Ollama on macOS, MPS acceleration instead of WSL2/CUDA)

import json
import math

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim

DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


def _get_device() -> torch.device:
    """Prefer Apple Silicon MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Task 1: baseline tools
# ---------------------------------------------------------------------------

def load_dataset_summary(dataset_name: str) -> str:
    """Loads a standard benchmark dataset and returns summary statistics."""
    name = dataset_name.lower().strip()
    if name not in DATASETS:
        return json.dumps({"error": f"Unknown dataset '{name}'. Options: {list(DATASETS.keys())}"})

    data = DATASETS[name]()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    summary = {
        "dataset": name,
        "n_samples": df.shape[0],
        "n_features": len(data.feature_names),
        "feature_names": list(data.feature_names),
        "classes": [str(c) for c in np.unique(data.target)],
        "missing_values": int(df.isnull().sum().sum()),
    }
    return json.dumps(summary)


def train_sklearn_model(dataset_name: str, model_type: str, test_size: float = 0.2) -> str:
    """Trains a Scikit-Learn model (decision_tree, logistic_regression, random_forest)."""
    name = dataset_name.lower().strip()
    if name not in DATASETS:
        return json.dumps({"error": f"Dataset '{name}' not found."})

    data = DATASETS[name]()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=42, stratify=data.target
    )

    model_type = model_type.lower().strip()
    if model_type == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    elif model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        # NOTE: SVC is deliberately unsupported here so the agent's
        # self-healing logic has a real failure to catch and redirect
        # to tune_hyperparameters(model_type="svc").
        return json.dumps({
            "error": f"Unsupported model '{model_type}' for train_sklearn_model. "
                     f"Supported: decision_tree, logistic_regression, random_forest. "
                     f"For 'svc', use tune_hyperparameters instead."
        })

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cv_scores = cross_val_score(clf, data.data, data.target, cv=5)

    return json.dumps({
        "model": model_type,
        "dataset": name,
        "test_accuracy": round(float(acc), 4),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
    })


def train_pytorch_mlp(dataset_name: str, hidden_dim: int = 32, epochs: int = 50, lr: float = 0.01) -> str:
    """Trains a simple PyTorch MLP; uses MPS on Apple Silicon when available."""
    name = dataset_name.lower().strip()
    if name not in DATASETS:
        return json.dumps({"error": f"Dataset '{name}' not found."})

    data = DATASETS[name]()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-7
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    device = _get_device()
    num_features = X_train.shape[1]
    num_classes = len(np.unique(data.target))

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_test, dtype=torch.long, device=device)

    model = nn.Sequential(
        nn.Linear(num_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    final_loss = None
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    with torch.no_grad():
        test_out = model(X_val_t)
        test_preds = torch.argmax(test_out, dim=1)
        acc = (test_preds == y_val_t).float().mean().item()

    return json.dumps({
        "framework": "PyTorch",
        "dataset": name,
        "device": str(device),
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "final_loss": round(float(final_loss), 4),
        "test_accuracy": round(float(acc), 4),
    })


# ---------------------------------------------------------------------------
# Task 2: advanced tools
# ---------------------------------------------------------------------------

def tune_hyperparameters(dataset_name: str, model_type: str = "svc") -> str:
    """GridSearchCV tuning for 'svc' (StandardScaler+SVC pipeline) or 'decision_tree'."""
    name = dataset_name.lower().strip()
    if name not in DATASETS:
        return json.dumps({"error": f"Dataset '{name}' not found."})

    data = DATASETS[name]()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    model_type = model_type.lower().strip()
    if model_type == "svc":
        pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
        param_grid = {
            "svc__C": [0.1, 1, 10],
            "svc__kernel": ["linear", "rbf"],
            "svc__gamma": ["scale", "auto"],
        }
        estimator = pipe
    elif model_type == "decision_tree":
        estimator = DecisionTreeClassifier(random_state=42)
        param_grid = {
            "max_depth": [2, 4, 6, None],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        }
    else:
        return json.dumps({"error": f"tune_hyperparameters does not support model_type='{model_type}'."})

    grid = GridSearchCV(estimator, param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)
    preds = grid.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return json.dumps({
        "model": model_type,
        "dataset": name,
        "best_params": grid.best_params_,
        "cv_best_accuracy": round(float(grid.best_score_), 4),
        "test_accuracy": round(float(acc), 4),
    })


def reduce_features(dataset_name: str, method: str = "pca", n_components: int = 2) -> str:
    """Dimensionality reduction via 'pca', or feature selection via 'sfs'/'sequential_feature_selection'."""
    name = dataset_name.lower().strip()
    if name not in DATASETS:
        return json.dumps({"error": f"Dataset '{name}' not found."})

    # Deliberately raise so the self-healing controller has something real to catch.
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")

    data = DATASETS[name]()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    method = method.lower().strip()
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    if method == "pca":
        pca = PCA(n_components=n_components, random_state=42).fit(X_train_s)
        X_train_p, X_test_p = pca.transform(X_train_s), pca.transform(X_test_s)
        clf = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_p, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_p))
        return json.dumps({
            "method": "pca",
            "dataset": name,
            "n_components": n_components,
            "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            "total_explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
            "test_accuracy": round(float(acc), 4),
        })

    elif method in ("sfs", "sequential_feature_selection"):
        base_clf = LogisticRegression(max_iter=1000, random_state=42)
        sfs = SequentialFeatureSelector(
            base_clf, n_features_to_select=n_components, direction="forward", cv=5
        )
        sfs.fit(X_train_s, y_train)
        selected_mask = sfs.get_support()
        selected_features = [f for f, keep in zip(data.feature_names, selected_mask) if keep]

        clf = LogisticRegression(max_iter=1000, random_state=42).fit(sfs.transform(X_train_s), y_train)
        acc = accuracy_score(y_test, clf.predict(sfs.transform(X_test_s)))

        return json.dumps({
            "method": "sfs",
            "dataset": name,
            "n_components": n_components,
            "selected_features": selected_features,
            "test_accuracy": round(float(acc), 4),
        })

    else:
        return json.dumps({"error": f"Unsupported method '{method}'."})


class AdvancedMLP(nn.Module):
    """MLP with BatchNorm + Dropout regularization."""

    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_advanced_pytorch_classifier(
    dataset_name: str,
    hidden_dim: int = 64,
    epochs: int = 50,
    lr: float = 0.01,
    dropout: float = 0.2,
) -> str:
    """Regularized PyTorch MLP: Dropout + BatchNorm + StepLR scheduler. Uses MPS on Mac."""
    name = dataset_name.lower().strip()
    if name not in DATASETS:
        return json.dumps({"error": f"Dataset '{name}' not found."})

    # Parameter validation -> intentionally raises so the controller's
    # heal_tool_call() can catch and self-correct these exact cases.
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
    if epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {epochs}")
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if not (0 <= dropout < 1):
        raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}")

    data = DATASETS[name]()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-7
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    device = _get_device()
    num_features = X_train.shape[1]
    num_classes = len(np.unique(data.target))

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_test, dtype=torch.long, device=device)

    model = AdvancedMLP(num_features, hidden_dim, num_classes, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)

    model.train()
    final_loss = None
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss = loss.item()

    model.eval()
    with torch.no_grad():
        test_out = model(X_val_t)
        test_preds = torch.argmax(test_out, dim=1)
        acc = (test_preds == y_val_t).float().mean().item()

    loss_is_nan = final_loss is None or math.isnan(final_loss)

    return json.dumps({
        "framework": "PyTorch-Advanced",
        "dataset": name,
        "device": str(device),
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "lr": lr,
        "dropout": dropout,
        "final_loss": "nan" if loss_is_nan else round(float(final_loss), 4),
        "test_accuracy": round(float(acc), 4),
    })


# ---------------------------------------------------------------------------
# Tool registry consumed by react_agent.py
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS = {
    "load_dataset_summary": load_dataset_summary,
    "train_sklearn_model": train_sklearn_model,
    "train_pytorch_mlp": train_pytorch_mlp,
    "tune_hyperparameters": tune_hyperparameters,
    "reduce_features": reduce_features,
    "train_advanced_pytorch_classifier": train_advanced_pytorch_classifier,
}
