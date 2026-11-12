import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.utils.common import resolve

COLUMNS = [
    "timestamp", "approach", "run", "preprocessing", "vectorizer", "model",
    "f1", "accuracy", "precision", "recall", "roc_auc", "f1_tuned_thr",
    "train_time_s", "infer_time_s", "n_train", "n_val", "extra",
]


def compute_metrics(y_true, scores, threshold=0.5):
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)
    return {
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, scores) if len(set(y_true)) > 1 else float("nan"),
    }


def best_threshold(y_true, scores):
    grid = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 181)))
    f1s = [f1_score(y_true, (scores >= t).astype(int)) for t in grid]
    i = int(np.argmax(f1s))
    return float(grid[i]), float(f1s[i])


def evaluate(y_true, scores, threshold=0.5):
    m = compute_metrics(y_true, scores, threshold)
    m["tuned_thr"], m["f1_tuned_thr"] = best_threshold(np.asarray(y_true), np.asarray(scores))
    return m


def log_result(cfg, row):
    path = resolve(cfg["results"]["metrics_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {c: row.get(c) for c in COLUMNS} | {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    row["extra"] = json.dumps(row["extra"] or {}, ensure_ascii=False)
    for k in ("f1", "accuracy", "precision", "recall", "roc_auc", "f1_tuned_thr"):
        row[k] = round(float(row[k]), 4)
    for k in ("train_time_s", "infer_time_s"):
        row[k] = round(float(row[k]), 2)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
    return row


def save_submission(cfg, ids, preds, name):
    out = resolve(cfg["results"]["submissions_dir"]) / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "target": np.asarray(preds).astype(int)}).to_csv(out, index=False)
    return out


def save_artifact(cfg, obj, approach, name):
    import joblib

    out = resolve(cfg["paths"]["models_dir"]) / approach / f"{name}.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, out)
    return out


def safe_name(run):
    return run.replace("|", "__").replace("/", "-")
