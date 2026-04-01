import csv
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Tabular", "anemia.csv")
EXPERIMENT_LOG = os.path.join(os.path.dirname(__file__), "results", "experiments.csv")
COMPARISON_CSV = os.path.join(os.path.dirname(__file__), "results", "model_comparison.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "..", "models", "saved_models", "tabular_rf.pkl")


def load_tabular(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def preprocess_tabular(df: pd.DataFrame) -> pd.DataFrame:
    # Keep rows with age in 6..60 months to match target population
    df = df[df["Age(Months)"].between(6, 60)]
    # Drop columns that leak label information (HB_LEVEL) and identifiers
    df = df.drop(columns=[c for c in ["HB_LEVEL", "IMAGE_ID", "REMARK"] if c in df.columns])
    # Encode gender
    df["GENDER"] = df["GENDER"].str.lower().map({"male": 0, "female": 1}).fillna(0).astype(int)
    # Ensure Severity is categorical target
    df = df.rename(columns={"Severity": "target"})
    df["target"] = df["target"].astype(str)
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    # Use simple features: Age and Gender and one-hot of REGION if available
    cols = ["Age(Months)", "GENDER"]
    if "REGION" in df.columns:
        df = pd.concat([df, pd.get_dummies(df["REGION"], prefix="REG")], axis=1)
        region_cols = [c for c in df.columns if c.startswith("REG_")]
        cols += region_cols
    X = df[cols].fillna(0).to_numpy(dtype="float32")
    y = df["target"].to_numpy()
    return X, y, cols


def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    # stratified split to avoid label distribution shift
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # split train into train/val
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction, random_state=random_state, stratify=y_trainval
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def balance_training(X_train, y_train, use_smote: bool = True, random_state: int = 42):
    if use_smote and SMOTE is not None:
        sm = SMOTE(random_state=random_state)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        return X_res, y_res
    # fallback: return original arrays
    return X_train, y_train


def tune_and_train(X_train, y_train, use_class_weight: bool = False):
    clf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight=("balanced" if use_class_weight else None))
    param_grid = {"n_estimators": [50, 100], "max_depth": [None, 10, 20]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(clf, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs


def evaluate_model(model, X, y):
    preds = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "f1_macro": float(f1_score(y, preds, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
    }


def log_experiment(record: dict):
    os.makedirs(os.path.dirname(EXPERIMENT_LOG), exist_ok=True)
    write_header = not os.path.exists(EXPERIMENT_LOG)
    with open(EXPERIMENT_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def save_comparison(results: dict):
    os.makedirs(os.path.dirname(COMPARISON_CSV), exist_ok=True)
    df = pd.DataFrame([results])
    df.to_csv(COMPARISON_CSV, index=False)


def run(full_path: str | None = None, use_smote: bool = True, use_class_weight: bool = False):
    df = load_tabular(full_path or DATA_PATH)
    df = preprocess_tabular(df)
    X, y, feat_cols = build_feature_matrix(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    X_train_bal, y_train_bal = balance_training(X_train, y_train, use_smote=use_smote)

    gs = tune_and_train(X_train_bal, y_train_bal, use_class_weight=use_class_weight)

    best = gs.best_estimator_
    # evaluate
    train_metrics = evaluate_model(best, X_train, y_train)
    val_metrics = evaluate_model(best, X_val, y_val)
    test_metrics = evaluate_model(best, X_test, y_test)

    # baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    baseline_metrics = evaluate_model(dummy, X_test, y_test)

    # persist model
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(best, MODEL_OUT)

    # log experiment
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "params": str(gs.best_params_),
        "cv_best_score": float(gs.best_score_),
        "train_f1_macro": train_metrics["f1_macro"],
        "val_f1_macro": val_metrics["f1_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "baseline_test_f1_macro": baseline_metrics["f1_macro"],
    }
    log_experiment(record)

    comparison = {
        "model": "random_forest",
        "train_f1_macro": train_metrics["f1_macro"],
        "val_f1_macro": val_metrics["f1_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "baseline_test_f1_macro": baseline_metrics["f1_macro"],
    }
    save_comparison(comparison)

    return {"best_params": gs.best_params_, "train": train_metrics, "val": val_metrics, "test": test_metrics}


if __name__ == "__main__":
    out = run()
    print(out)
