import hashlib
from pathlib import Path

import click
import pandas as pd

from src.data.dataset import get_data, split_path
from src.data.preprocess import light_preprocess, preprocess
from src.utils.common import load_config, resolve


def require(condition, message):
    if not condition:
        raise click.ClickException(message)


@click.command()
@click.option("--config", default="configs/classic.yaml", show_default=True)
@click.option("--submissions", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", default="results/validation_audit.md", show_default=True)
def main(config, submissions, output):
    cfg = load_config(config)
    require(split_path(cfg).exists(), "Run preprocessing before auditing the split")
    train, val, test = get_data(cfg)
    metrics = pd.read_csv(resolve(cfg["results"]["metrics_path"]))
    require(not metrics.empty, "Metrics are empty")
    require(not metrics.isna().any().any(), "Metrics contain missing values")
    require(not metrics["run"].duplicated().any(), "Metrics contain duplicate runs")
    require(metrics["n_train"].eq(len(train)).all(), "Training sizes differ from the split")
    require(metrics["n_val"].eq(len(val)).all(), "Validation sizes differ from the split")
    for column in ("f1", "accuracy", "precision", "recall", "roc_auc", "f1_tuned_thr"):
        require(metrics[column].between(0, 1).all(), f"Invalid {column}")
    for column in ("train_time_s", "infer_time_s"):
        require(metrics[column].ge(0).all(), f"Negative {column}")

    lines = [
        "# Проверка результатов v1", "",
        f"Проверено {len(metrics)} экспериментов: " + ", ".join(
            f"{name}: {count}" for name, count in metrics.groupby("approach").size().items()
        ) + ". Пропусков и повторов run нет; значения метрик и размеры выборок корректны.", "",
        f"Train: {len(train)}, val: {len(val)}, test: {len(test)}. Seed: {cfg['seed']}.", "",
        "## Пересечения train / val", "",
        "Сравниваются тексты без keyword. Строки val считаются с учётом повторов; "
        "совпадения после нормализации не равнозначны пересечению исходных текстов.", "",
        "| Представление | Общих уникальных текстов | Строк val с совпадением в train |",
        "|---|---:|---:|",
    ]
    variants = {
        "Исходный текст": (train["text"].tolist(), val["text"].tolist()),
        "Лёгкая очистка URL и упоминаний": (
            light_preprocess(train["text"].tolist()), light_preprocess(val["text"].tolist())
        ),
        "Classic clean": (
            preprocess(train["text"].tolist(), cfg["preprocessing"]["clean"]),
            preprocess(val["text"].tolist(), cfg["preprocessing"]["clean"]),
        ),
    }
    for name, (train_texts, val_texts) in variants.items():
        shared = set(train_texts) & set(val_texts)
        if name == "Исходный текст":
            require(not shared, "Raw texts overlap between train and val")
        lines.append(f"| {name} | {len(shared)} | {sum(t in shared for t in val_texts)} |")
    lines += [
        "", "Исходные дубликаты исключены, но замена ссылок и упоминаний объединяет "
        "варианты твитов, оставшиеся по разные стороны сплита. Поэтому метрики v1 "
        "описывают этот фиксированный сплит и могут быть оптимистичными. Строгое "
        "сравнение требует отдельного группового сплита и повторного обучения; "
        "результаты такого прогона нельзя смешивать с v1.", "",
        "## Порог классификации", "",
        f"В сохранённом прогоне у {int((metrics['f1_tuned_thr'] < metrics['f1']).sum())} "
        "экспериментов f1_tuned_thr ниже F1 при стандартном пороге: историческая "
        "сетка квантилей не включала стандартный порог. Исходные значения сохранены. "
        "В коде поиска порога это исправлено для будущих запусков. "
        "Основная таблица сравнивает F1 при стандартном пороге. Подбор порога "
        "и оценка на одной val дают оптимистичную оценку.", "",
        "## Сабмиты", "",
    ]
    if submissions:
        files = sorted(submissions.glob("*.csv"))
        require(bool(files), "No submission CSV files found")
        for file in files:
            submission = pd.read_csv(file)
            require(list(submission.columns) == ["id", "target"], f"Invalid columns: {file.name}")
            require(submission["id"].equals(test["id"]), f"Test IDs differ: {file.name}")
            require(not submission["id"].duplicated().any(), f"Duplicate IDs: {file.name}")
            require(submission["target"].isin([0, 1]).all(), f"Invalid targets: {file.name}")
            lines.append(f"- `{file.name}`: {len(submission)} строк, ID и метки корректны.")
        lines += ["", "Проверен формат файлов. Меток test и оценок Kaggle в архиве нет; "
                  "качество на test не проверялось."]
    else:
        lines.append("Не проверялись: каталог не передан через --submissions.")
    lines += ["", "## Контрольные суммы SHA-256 (UTF-8, окончания строк LF)", ""]
    for path in (resolve(cfg["results"]["metrics_path"]), split_path(cfg)):
        content = path.read_text(encoding="utf-8").encode("utf-8")
        lines.append(f"- `{path.name}`: `{hashlib.sha256(content).hexdigest()}`")
    lines += ["", "Проверка не изменяет сплит, модели или сохранённые метрики.", ""]
    path = resolve(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    click.echo(f"Audit passed: {path}")


if __name__ == "__main__":
    main()
