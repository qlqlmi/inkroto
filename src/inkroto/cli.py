"""Command line interface."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

import typer

from inkroto.config import ConfigError, load_config
from inkroto.io import load_image, next_numbered_path, save_debug_sheet
from inkroto.render import render as render_video

app = typer.Typer(add_completion=False, no_args_is_help=True)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
SIDECAR_SUFFIX = ".yaml"


@app.callback()
def main() -> None:
    """Turn a still photo into a short ink-style rotoscope video."""


@app.command()
def render(
    input: Path = typer.Argument(
        ..., exists=True, dir_okay=False,
        help="Input photo, or a video or gif to use as the motion source.",
    ),
    output: Path = typer.Option(..., "-o", "--output", help="Output .mp4 path; a number is added (out_001.mp4, out_002.mp4, ...) so nothing is overwritten."),
    config: Optional[Path] = typer.Option(
        None, "--config", exists=True, dir_okay=False, help="YAML merged over the defaults."
    ),
    set_: Optional[list[str]] = typer.Option(
        None, "--set", help="Override a config value, e.g. --set output.fps=12. Repeatable."
    ),
    cache_dir: Path = typer.Option(Path(".cache"), "--cache-dir", help="Analysis cache folder."),
    debug: Optional[Path] = typer.Option(
        None, "--debug", file_okay=False,
        help="Save a sheet of every stage of the first drawing, plus the first, middle, and last frames, here.",
    ),
) -> None:
    """Render INPUT (a photo, video, or gif) to an ink-style video."""
    from inkroto.analyze import MissingMLDependencyError

    try:
        cfg = load_config(config, set_)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        out = render_video(
            input, next_numbered_path(output), cfg, cache_dir=cache_dir, debug_dir=debug
        )
    except MissingMLDependencyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"wrote {out}")


@app.command()
def analyze(
    input: Path = typer.Argument(..., exists=True, dir_okay=False, help="Input photo."),
    output: Path = typer.Option(..., "-o", "--output", help="Debug sheet .png path."),
    config: Optional[Path] = typer.Option(
        None, "--config", exists=True, dir_okay=False, help="YAML merged over the defaults."
    ),
    set_: Optional[list[str]] = typer.Option(
        None, "--set", help="Override a config value, e.g. --set output.width=640. Repeatable."
    ),
    cache_dir: Path = typer.Option(Path(".cache"), "--cache-dir", help="Analysis cache folder."),
) -> None:
    """Write a debug sheet with INPUT, its depth (bright = near), subject matte, and faces."""
    from inkroto import analyze as analysis

    try:
        cfg = load_config(config, set_)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    image = load_image(input, cfg.output.width)
    try:
        fc = cfg.stylize.faces
        result = analysis.analyze(image, cache_dir, faces=fc.enabled, face_min_score=fc.min_score)
    except analysis.MissingMLDependencyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    images = {"image": image, "depth (bright = near)": result.depth, "matte": result.matte}
    if result.faces is not None:
        images["faces"] = result.faces
    save_debug_sheet(images, output, cols=len(images))
    typer.echo(f"wrote {output.with_suffix('.png')}")


def find_images(folder: Path) -> list[Path]:
    """jpg, jpeg, and png files in `folder` (not recursive), sorted by name."""
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder of photos."),
    output_dir: Path = typer.Option(..., "-o", "--output", help="Folder for the clips."),
    config: Optional[Path] = typer.Option(
        None, "--config", exists=True, dir_okay=False, help="YAML merged over the defaults."
    ),
    set_: Optional[list[str]] = typer.Option(
        None, "--set", help="Override a config value for every image. Repeatable."
    ),
    cache_dir: Path = typer.Option(Path(".cache"), "--cache-dir", help="Analysis cache folder."),
    debug: Optional[Path] = typer.Option(
        None, "--debug", file_okay=False, help="Save debug sheets and frames for every image here."
    ),
) -> None:
    """Render every jpg, jpeg, and png in INPUT_DIR into OUTPUT_DIR.

    A `name.yaml` next to `name.jpg` is merged on top of the config for that image
    (before --set). A failing image is reported and the batch carries on.
    """
    from inkroto.analyze import MissingMLDependencyError

    try:
        load_config(config, set_)  # fail fast on a bad shared config
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    images = find_images(input_dir)
    if not images:
        typer.echo(f"no jpg, jpeg, or png files in {input_dir}", err=True)
        raise typer.Exit(1)

    stems = [p.stem.lower() for p in images]
    failed: list[Path] = []
    for n, image in enumerate(images, start=1):
        # photo.jpg and photo.png in one folder become photo_jpg and photo_png.
        name = image.stem
        if stems.count(image.stem.lower()) > 1:
            name = f"{image.stem}_{image.suffix.lstrip('.').lower()}"
        sidecar = image.with_suffix(SIDECAR_SUFFIX)
        layers = [p for p in (config, sidecar if sidecar.is_file() else None) if p is not None]
        label = f"[{n}/{len(images)}] {image.name}" + (f" (+ {sidecar.name})" if sidecar.is_file() else "")
        typer.echo(label)
        try:
            cfg = load_config(layers, set_)
            out = render_video(
                image, next_numbered_path(output_dir / f"{name}.mp4"), cfg,
                cache_dir=cache_dir, debug_dir=debug,
            )
            typer.echo(f"  wrote {out}")
        except MissingMLDependencyError as exc:  # every image would fail the same way
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        except Exception as exc:  # noqa: BLE001 - one bad image must not stop the batch
            failed.append(image)
            typer.echo(f"  FAILED {image.name}: {type(exc).__name__}: {exc}", err=True)
            typer.echo(traceback.format_exc(), err=True)

    done = len(images) - len(failed)
    typer.echo(f"done: {done} rendered, {len(failed)} failed")
    if failed:
        typer.echo("failed: " + ", ".join(p.name for p in failed), err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
