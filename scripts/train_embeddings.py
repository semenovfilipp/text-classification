import time

import click

from src.data.dataset import get_data
from src.data.preprocess import light_preprocess
from src.features.embeddings import encode
from src.models.classic import build_model, predict_scores
from src.utils.common import get_device, load_config, resolve, set_seed
from src.utils.metrics import evaluate, log_result, safe_name, save_artifact, save_submission


def run(cfg):
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    best = None
    for v_name, pp in cfg["preprocessing"].items():
        train, val, test = get_data(cfg, use_keyword=pp.get("use_keyword", False))
        texts = {s: light_preprocess(df["text"].tolist(), enabled=pp.get("light_clean", True)) for s, df in (("train", train), ("val", val), ("test", test))}
        for e_name in cfg["run_encoders"]:
            enc = cfg["encoders"][e_name]
            emb, enc_time = {}, {}
            for split, tx in texts.items():
                cache = None if cfg.get("debug_subsample") else resolve(cfg["embeddings_dir"]) / f"{e_name}_{v_name}_{split}.npy"
                emb[split], enc_time[split] = encode(enc, tx, device, cache)
            for m_name in cfg["run_models"]:
                model = build_model(cfg["models"][m_name], cfg["seed"])
                t0 = time.perf_counter()
                model.fit(emb["train"], train["target"])
                fit_time = time.perf_counter() - t0
                t0 = time.perf_counter()
                scores = predict_scores(model, emb["val"])
                head_time = time.perf_counter() - t0
                m = evaluate(val["target"], scores)
                name = f"{e_name}|{v_name}|{m_name}"
                row = log_result(cfg, {
                    "approach": "embeddings", "run": name, "preprocessing": v_name, "vectorizer": enc["model_name"], "model": m_name,
                    **m, "train_time_s": enc_time["train"] + fit_time, "infer_time_s": enc_time["val"] + head_time,
                    "n_train": len(train), "n_val": len(val),
                    "extra": {"dim": int(emb["train"].shape[1]), "device": device, "encode_train_s": round(enc_time["train"], 2)},
                })
                click.echo(f"{name:55s} f1={row['f1']:.4f} acc={row['accuracy']:.4f} auc={row['roc_auc']:.4f}")
                if best is None or m["f1"] > best["f1"]:
                    best = {"f1": m["f1"], "name": name, "model": model, "x_te": emb["test"], "ids": test["id"]}
    click.echo(f"best: {best['name']} f1={best['f1']:.4f}")
    name = safe_name(best["name"])
    if cfg.get("save_model"):
        click.echo(f"model: {save_artifact(cfg, best['model'], 'embeddings', name)}")
    if cfg.get("save_submission"):
        click.echo(f"submission: {save_submission(cfg, best['ids'], predict_scores(best['model'], best['x_te']) >= 0.5, 'embeddings__' + name)}")


@click.command()
@click.option("--config", default="configs/embeddings.yaml", show_default=True)
@click.option("--encoders", default=None, help="Список энкодеров через запятую, переопределяет run_encoders")
@click.option("--debug", type=int, default=None, help="Прогон на подвыборке из N объектов")
def main(config, encoders, debug):
    cfg = load_config(config)
    if encoders:
        cfg["run_encoders"] = encoders.split(",")
    if debug:
        cfg.update(debug_subsample=debug, save_model=False, save_submission=False)
        cfg["results"]["metrics_path"] = "results/metrics_debug.csv"
    run(cfg)


if __name__ == "__main__":
    main()
