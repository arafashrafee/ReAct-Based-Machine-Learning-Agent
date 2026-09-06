import json
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim


DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


def load_dataset_summary(dataset_name: str) -> str:
    name = dataset_name.lower().strip()

    if name not in DATASETS:
        return json.dumps({
            "error": f"Unknown dataset '{name}'. Options: {list(DATASETS.keys())}"
        })

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

    return json.dumps(summary, indent=2)


def train_sklearn_model(dataset_name: str, model_type: str, test_size: float = 0.2) -> str:
    name = dataset_name.lower().strip()

    if name not in DATASETS:
        return json.dumps({"error": f"Dataset '{name}' not found."})

    data = DATASETS[name]()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=42,
        stratify=data.target,
    )

    model_type = model_type.lower().strip()

    if model_type == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    elif model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        return json.dumps({
            "error": f"Unsupported model '{model_type}'. Options: decision_tree, logistic_regression, random_forest"
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
    }, indent=2)


def train_pytorch_mlp(dataset_name: str, hidden_dim: int = 32, epochs: int = 50, lr: float = 0.01) -> str:
    name = dataset_name.lower().strip()

    if name not in DATASETS:
        return json.dumps({"error": f"Dataset '{name}' not found."})

    data = DATASETS[name]()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-7
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    num_features = X_train.shape[1]
    num_classes = len(np.unique(data.target))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_test, dtype=torch.long).to(device)

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

        if torch.isnan(loss):
            return json.dumps({
                "error": "NaN loss detected during PyTorch training.",
                "dataset": name,
                "hidden_dim": hidden_dim,
                "epochs": epochs,
                "lr": lr,
            })

        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    with torch.no_grad():
        test_out = model(X_val_t)
        test_preds = torch.argmax(test_out, dim=1)
        acc = (test_preds == y_val_t).float().mean().item()

    return json.dumps({
        "framework": "PyTorch",
        "device": str(device),
        "dataset": name,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "learning_rate": lr,
        "final_loss": round(float(final_loss), 4),
        "test_accuracy": round(float(acc), 4),
    }, indent=2)


AVAILABLE_TOOLS = {
    "load_dataset_summary": load_dataset_summary,
    "train_sklearn_model": train_sklearn_model,
    "train_pytorch_mlp": train_pytorch_mlp,
}


from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier


def tune_hyperparameters(dataset_name: str, model_type: str = "svc") -> str:
    name = dataset_name.lower().strip()

    if name not in DATASETS:
        return json.dumps({"error": f"Dataset '{name}' not found."})

    data = DATASETS[name]()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    model_type = model_type.lower().strip()

    if model_type == "svc":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC()),
        ])
        param_grid = {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"],
        }
    elif model_type == "decision_tree":
        pipeline = Pipeline([
            ("model", DecisionTreeClassifier(random_state=42)),
        ])
        param_grid = {
            "model__max_depth": [2, 3, 4, 5, None],
            "model__min_samples_split": [2, 4, 8],
            "model__criterion": ["gini", "entropy"],
        }
    else:
        return json.dumps({
            "error": "Unsupported model_type. Options: svc, decision_tree"
        })

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    test_accuracy = search.score(X_test, y_test)

    return json.dumps({
        "tool": "tune_hyperparameters",
        "dataset": name,
        "model_type": model_type,
        "best_params": search.best_params_,
        "cv_best_accuracy": round(float(search.best_score_), 4),
        "test_accuracy": round(float(test_accuracy), 4),
    }, indent=2)


def reduce_features(dataset_name: str, method: str = "pca", n_components: int = 2) -> str:
    name = dataset_name.lower().strip()

    if name not in DATASETS:
        return json.dumps({"error": f"Dataset '{name}' not found."})

    data = DATASETS[name]()
    method = method.lower().strip()

    if n_components < 1:
        return json.dumps({"error": "n_components must be at least 1."})

    max_features = data.data.shape[1]
    n_components = min(n_components, max_features)

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    if method == "pca":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("model", LogisticRegression(max_iter=3000, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)
        test_accuracy = pipeline.score(X_test, y_test)
        explained_variance = pipeline.named_steps["pca"].explained_variance_ratio_

        return json.dumps({
            "tool": "reduce_features",
            "dataset": name,
            "method": "pca",
            "n_components": n_components,
            "explained_variance_ratio": [round(float(v), 4) for v in explained_variance],
            "total_explained_variance": round(float(explained_variance.sum()), 4),
            "test_accuracy": round(float(test_accuracy), 4),
        }, indent=2)

    if method in {"sfs", "sequential_feature_selection"}:
        estimator = LogisticRegression(max_iter=3000, random_state=42)
        selector = SequentialFeatureSelector(
            estimator,
            n_features_to_select=n_components,
            direction="forward",
            cv=5,
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", selector),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        test_accuracy = pipeline.score(X_test, y_test)

        mask = pipeline.named_steps["selector"].get_support()
        selected_features = [
            feature for feature, keep in zip(data.feature_names, mask) if keep
        ]

        return json.dumps({
            "tool": "reduce_features",
            "dataset": name,
            "method": "sequential_feature_selection",
            "n_selected_features": len(selected_features),
            "selected_features": selected_features,
            "test_accuracy": round(float(test_accuracy), 4),
        }, indent=2)

    return json.dumps({
        "error": "Unsupported method. Options: pca, sfs, sequential_feature_selection"
    })


class AdvancedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


def train_advanced_pytorch_classifier(
    dataset_name: str,
    hidden_dim: int = 64,
    epochs: int = 80,
    lr: float = 0.01,
    dropout: float = 0.2,
    scheduler_step_size: int = 30,
    scheduler_gamma: float = 0.5,
) -> str:
    name = dataset_name.lower().strip()

    if name not in DATASETS:
        return json.dumps({"error": f"Dataset '{name}' not found."})

    if hidden_dim <= 0:
        return json.dumps({"error": "hidden_dim must be positive."})

    if epochs <= 0:
        return json.dumps({"error": "epochs must be positive."})

    if lr <= 0:
        return json.dumps({"error": "lr must be positive."})

    if not 0 <= dropout < 1:
        return json.dumps({"error": "dropout must be in the range [0, 1)."})

    data = DATASETS[name]()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-7
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    num_features = X_train.shape[1]
    num_classes = len(np.unique(data.target))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_test, dtype=torch.long).to(device)

    model = AdvancedMLP(num_features, hidden_dim, num_classes, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma,
    )

    final_loss = None

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_t)
        loss = criterion(output, y_t)

        if torch.isnan(loss):
            return json.dumps({
                "error": "NaN loss detected during advanced PyTorch training.",
                "dataset": name,
                "hidden_dim": hidden_dim,
                "epochs": epochs,
                "lr": lr,
                "dropout": dropout,
            })

        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        test_output = model(X_val_t)
        test_predictions = torch.argmax(test_output, dim=1)
        test_accuracy = (test_predictions == y_val_t).float().mean().item()

    return json.dumps({
        "tool": "train_advanced_pytorch_classifier",
        "framework": "PyTorch",
        "device": str(device),
        "dataset": name,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "learning_rate": lr,
        "dropout": dropout,
        "scheduler": {
            "type": "StepLR",
            "step_size": scheduler_step_size,
            "gamma": scheduler_gamma,
        },
        "final_loss": round(float(final_loss), 4),
        "test_accuracy": round(float(test_accuracy), 4),
    }, indent=2)


AVAILABLE_TOOLS.update({
    "tune_hyperparameters": tune_hyperparameters,
    "reduce_features": reduce_features,
    "train_advanced_pytorch_classifier": train_advanced_pytorch_classifier,
})
