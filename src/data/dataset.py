import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.common import resolve


def load_raw(cfg):
    raw = resolve(cfg["data"]["raw_dir"])
    return pd.read_csv(raw / "train.csv"), pd.read_csv(raw / "test.csv")


def split_path(cfg):
    d = cfg["data"]
    return resolve(d["processed_dir"]) / f"{d['version']}_train_split.csv"


def resolve_duplicates(df):
    rate = df.groupby("text")["target"].mean()
    df = df[~df["text"].isin(rate.index[rate == 0.5])].copy()
    df["target"] = df["text"].map((rate > 0.5).astype(int))
    return df.drop_duplicates("text").reset_index(drop=True)


def build_split(cfg):
    d = cfg["data"]
    train, _ = load_raw(cfg)
    if d.get("dedup", True):
        train = resolve_duplicates(train)
    _, va_idx = train_test_split(train.index, test_size=d["val_size"], stratify=train["target"], random_state=cfg["seed"])
    train["fold"] = "train"
    train.loc[va_idx, "fold"] = "val"
    out = split_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(out, index=False)
    return train


def _prepare(df, use_keyword):
    df = df.copy()
    df["keyword"] = df["keyword"].fillna("").str.replace("%20", " ", regex=False)
    if use_keyword:
        df["text"] = (df["keyword"] + " " + df["text"]).str.strip()
    return df


def get_data(cfg, use_keyword=False):
    path = split_path(cfg)
    df = pd.read_csv(path) if path.exists() else build_split(cfg)
    _, test = load_raw(cfg)
    df, test = _prepare(df, use_keyword), _prepare(test, use_keyword)
    train = df[df.fold == "train"].reset_index(drop=True)
    val = df[df.fold == "val"].reset_index(drop=True)
    n = cfg.get("debug_subsample")
    if n:
        train = train.sample(min(n, len(train)), random_state=cfg["seed"]).reset_index(drop=True)
        val = val.sample(min(n, len(val)), random_state=cfg["seed"]).reset_index(drop=True)
        test = test.head(n)
    return train, val, test


def variant_texts(cfg, name, opts, frames, force=False):
    from src.data.preprocess import preprocess

    path = resolve(cfg["data"]["processed_dir"]) / f"{cfg['data']['version']}_{name}.csv"
    if path.exists() and not force and not cfg.get("debug_subsample"):
        df = pd.read_csv(path, keep_default_na=False)
        return tuple(df[df.split == s]["text"].tolist() for s in ("train", "val", "test"))
    out = []
    for f in frames:
        texts = f["text"]
        if opts.get("use_keyword"):
            texts = (f["keyword"] + " " + texts).str.strip()
        out.append(preprocess(texts.tolist(), opts))
    if not cfg.get("debug_subsample"):
        rows = [pd.DataFrame({"id": f["id"], "split": s, "text": t}) for f, s, t in zip(frames, ("train", "val", "test"), out)]
        pd.concat(rows).to_csv(path, index=False)
    return tuple(out)
