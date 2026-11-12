import click

from src.data.dataset import build_split, get_data, variant_texts
from src.utils.common import load_config


@click.command()
@click.option("--config", default="configs/classic.yaml", show_default=True, help="Конфиг с секцией preprocessing")
def main(config):
    cfg = load_config(config)
    df = build_split(cfg)
    click.echo(f"split: {df['fold'].value_counts().to_dict()}, positive rate {df.groupby('fold')['target'].mean().round(3).to_dict()}")
    frames = get_data(cfg)
    for name, opts in cfg.get("preprocessing", {}).items():
        x_tr, _, _ = variant_texts(cfg, name, opts or {}, frames, force=True)
        click.echo(f"{name:22s} {x_tr[0][:90]}")


if __name__ == "__main__":
    main()
