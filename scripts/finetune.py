import os
import time

import click
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data.dataset import get_data
from src.data.preprocess import light_preprocess
from src.models.transformer import TextDataset, hf_metrics, positive_proba
from src.utils.common import load_config, resolve, set_seed
from src.utils.metrics import evaluate, log_result, save_submission


def run(cfg):
    set_seed(cfg["seed"])
    ft, pp = cfg["finetune"], cfg.get("preprocessing", {})
    short = ft["model_name"].split("/")[-1]
    train, val, test = get_data(cfg, use_keyword=pp.get("use_keyword", False))
    tok = AutoTokenizer.from_pretrained(ft["model_name"], **ft.get("tokenizer_kwargs", {}))

    def dataset(df, labels=True):
        texts = light_preprocess(df["text"].tolist(), enabled=pp.get("light_clean", True), **pp.get("clean_kwargs", {}))
        return TextDataset(tok(texts, truncation=True, max_length=ft["max_length"]), df["target"].values if labels else None)

    ds_train, ds_val, ds_test = dataset(train), dataset(val), dataset(test, labels=False)
    model = AutoModelForSequenceClassification.from_pretrained(ft["model_name"], num_labels=2)
    cuda = torch.cuda.is_available()
    run_dir = resolve(cfg["paths"]["models_dir"]) / "finetune" / short
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(resolve(cfg["paths"]["logs_dir"]) / "tensorboard" / short)
    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        learning_rate=ft["lr"],
        num_train_epochs=ft["epochs"],
        per_device_train_batch_size=ft["batch_size"],
        per_device_eval_batch_size=ft["batch_size"] * 2,
        warmup_steps=ft["warmup_ratio"],
        weight_decay=ft["weight_decay"],
        lr_scheduler_type=ft["scheduler"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=ft["logging_steps"],
        fp16=cuda and ft.get("fp16", True),
        seed=cfg["seed"],
        report_to=cfg.get("report_to", ["tensorboard"]),
        use_cpu=not cuda,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=hf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=ft["patience"])],
    )
    t0 = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores = positive_proba(trainer.predict(ds_val).predictions)
    infer_time = time.perf_counter() - t0
    m = evaluate(val["target"], scores)
    name = f"{ft['model_name']}|lr{ft['lr']}|ep{ft['epochs']}|len{ft['max_length']}"
    row = log_result(cfg, {
        "approach": "finetune", "run": name, "preprocessing": "light_clean" if pp.get("light_clean", True) else "tokenizer_norm",
        "vectorizer": "-", "model": ft["model_name"], **m, "train_time_s": train_time, "infer_time_s": infer_time,
        "n_train": len(train), "n_val": len(val),
        "extra": {"device": "cuda" if cuda else "cpu", "best_ckpt": trainer.state.best_model_checkpoint, "tuned_thr": round(m["tuned_thr"], 3)},
    })
    click.echo(f"{name} f1={row['f1']:.4f} acc={row['accuracy']:.4f} auc={row['roc_auc']:.4f} train={row['train_time_s']}s")
    if cfg.get("save_model"):
        trainer.save_model(str(run_dir / "best"))
        tok.save_pretrained(str(run_dir / "best"))
        click.echo(f"model: {run_dir / 'best'}")
    if cfg.get("save_submission"):
        test_scores = positive_proba(trainer.predict(ds_test).predictions)
        click.echo(f"submission: {save_submission(cfg, test['id'], test_scores >= 0.5, 'finetune__' + short)}")


@click.command()
@click.option("--config", default="configs/finetune.yaml", show_default=True)
@click.option("--debug", type=int, default=None, help="Прогон на подвыборке из N объектов, 1 эпоха")
def main(config, debug):
    cfg = load_config(config)
    if debug:
        cfg.update(debug_subsample=debug, save_model=False, save_submission=False, report_to="none")
        cfg["finetune"]["epochs"] = 1
        cfg["results"]["metrics_path"] = "results/metrics_debug.csv"
    run(cfg)


if __name__ == "__main__":
    main()
