import itertools
import time

import click

from src.data.dataset import get_data, variant_texts
from src.features.vectorizers import build_vectorizer
from src.models.classic import build_model, predict_scores
from src.utils.common import load_config, set_seed
from src.utils.metrics import evaluate, log_result, safe_name, save_artifact, save_submission


def as_list(value, pool):
    if value == "all":
        return list(pool)
    return value if isinstance(value, list) else [value]


def run(cfg):
    set_seed(cfg["seed"])
    frames = get_data(cfg)
    train, val, test = frames
    best = None
    for exp in cfg["experiments"]:
        grid = itertools.product(*(as_list(exp[k], cfg[p]) for k, p in (("preprocessing", "preprocessing"), ("vectorizer", "vectorizers"), ("model", "models"))))
        for p_name, v_name, m_name in grid:
            x_tr, x_va, x_te = variant_texts(cfg, p_name, cfg["preprocessing"][p_name] or {}, frames)
            vec = build_vectorizer(cfg["vectorizers"][v_name])
            model = build_model(cfg["models"][m_name], cfg["seed"])
            t0 = time.perf_counter()
            model.fit(vec.fit_transform(x_tr), train["target"])
            train_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            scores = predict_scores(model, vec.transform(x_va))
            infer_time = time.perf_counter() - t0
            thr = 0.0 if cfg["models"][m_name]["type"] == "linear_svc" else 0.5
            m = evaluate(val["target"], scores, thr)
            name = f"{exp['name']}|{p_name}|{v_name}|{m_name}"
            row = log_result(cfg, {
                "approach": "classic", "run": name, "preprocessing": p_name, "vectorizer": v_name, "model": m_name,
                **m, "train_time_s": train_time, "infer_time_s": infer_time, "n_train": len(train), "n_val": len(val),
            })
            click.echo(f"{name:65s} f1={row['f1']:.4f} acc={row['accuracy']:.4f} auc={row['roc_auc']:.4f} t={row['train_time_s']}s")
            if best is None or m["f1"] > best["f1"]:
                best = {"f1": m["f1"], "name": name, "vec": vec, "model": model, "x_te": x_te, "thr": thr}
    click.echo(f"best: {best['name']} f1={best['f1']:.4f}")
    name = safe_name(best["name"])
    if cfg.get("save_model"):
        click.echo(f"model: {save_artifact(cfg, {'vectorizer': best['vec'], 'model': best['model'], 'threshold': best['thr']}, 'classic', name)}")
    if cfg.get("save_submission"):
        preds = predict_scores(best["model"], best["vec"].transform(best["x_te"])) >= best["thr"]
        click.echo(f"submission: {save_submission(cfg, test['id'], preds, 'classic__' + name)}")


@click.command()
@click.option("--config", default="configs/classic.yaml", show_default=True)
@click.option("--debug", type=int, default=None, help="Прогон на подвыборке из N объектов")
def main(config, debug):
    cfg = load_config(config)
    if debug:
        cfg.update(debug_subsample=debug, save_model=False, save_submission=False)
        cfg["results"]["metrics_path"] = "results/metrics_debug.csv"
    run(cfg)


if __name__ == "__main__":
    main()
