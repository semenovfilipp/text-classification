import pandas as pd

from src.utils.common import resolve

COLS = ["approach", "preprocessing", "vectorizer", "model", "f1", "accuracy", "precision", "recall", "roc_auc", "f1_tuned_thr", "train_time_s", "infer_time_s"]


def to_markdown(df):
    df = df[COLS]
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def build_report(cfg, top=20):
    df = pd.read_csv(resolve(cfg["results"]["metrics_path"]))
    df = df.sort_values("timestamp").drop_duplicates("run", keep="last")
    best = df.sort_values("f1", ascending=False).groupby("approach").head(1)
    parts = ["## Лучшая модель каждого подхода", "", to_markdown(best), ""]
    for approach, g in df.groupby("approach", sort=False):
        parts += [f"## {approach}: top-{top} по F1", "", to_markdown(g.sort_values("f1", ascending=False).head(top)), ""]
    out = resolve(cfg["results"]["metrics_path"]).with_name("metrics_table.md")
    out.write_text("\n".join(parts), encoding="utf-8")
    return out, best
