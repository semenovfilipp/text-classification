import click

from src.utils.common import load_config
from src.utils.report import build_report, to_markdown


@click.command()
@click.option("--config", default="configs/data.yaml", show_default=True)
@click.option("--top", default=20, show_default=True, help="Сколько лучших запусков показывать по каждому подходу")
def main(config, top):
    path, best = build_report(load_config(config), top)
    click.echo(to_markdown(best.sort_values("f1", ascending=False)))
    click.echo(f"\nfull table: {path}")


if __name__ == "__main__":
    main()
