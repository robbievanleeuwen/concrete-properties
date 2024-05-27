"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """concreteproperties."""


if __name__ == "__main__":
    main(prog_name="concreteproperties")  # pragma: no cover
