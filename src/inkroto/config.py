"""Config dataclasses and YAML loading.

The dataclasses mirror configs/default.yaml exactly. They have no defaults on
purpose: the YAML file is the single source of truth for every tunable number.
"""

from __future__ import annotations

import copy
import dataclasses
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Optional, Union

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Raised for unknown keys, missing keys, bad values, or malformed overrides."""


@dataclass
class OutputConfig:
    fps: int
    width: int
    duration_s: float


@dataclass
class MotionConfig:
    backend: str
    pan: list[float]
    zoom: float
    focus: float
    clip_path: Optional[str]
    depth_smoothing: float
    matte_threshold: float
    dilate_px: int
    inpaint_radius: float
    depth_blur: float


@dataclass
class CadenceConfig:
    hold: int


@dataclass
class ContrastConfig:
    enabled: bool
    clip_limit: float
    tiles: int


@dataclass
class RelightConfig:
    enabled: bool
    direction: list[float]
    strength: float
    normal_blur: float
    normal_z: float


@dataclass
class HazeConfig:
    enabled: bool
    start: float
    density: float
    value: float


@dataclass
class SimplifyConfig:
    method: str
    radius: int
    sigma_color: float


@dataclass
class TonesConfig:
    percentiles: list[float]
    mid_value: float
    jitter: float


@dataclass
class LinesConfig:
    mode: str
    source: str
    sigma: float
    k: float
    p: float
    epsilon: float
    edge_threshold: float
    denoise: float
    denoise_px: float
    phi: float
    wobble_px: float
    wobble_scale: float


@dataclass
class HatchConfig:
    enabled: bool
    spacing: float
    angle: float
    thickness: float
    jitter_px: float
    thickness_jitter: float
    noise_scale: float
    segment: list[float]
    gap: list[float]


@dataclass
class CleanupConfig:
    min_area: int
    morph: int


@dataclass
class FacesConfig:
    enabled: bool
    simplify_radius: int
    two_tone: bool
    min_score: float


@dataclass
class PencilConfig:
    line_value: float
    passes: int
    pass_offset_px: float
    pressure: float
    pressure_scale: float
    break_rate: float
    break_scale: float
    shade_values: list[float]
    paper_tooth: float
    paper_scale: float
    paper_grain: float
    smudge: float
    paper_tint: list[float]


@dataclass
class StylizeConfig:
    medium: str
    contrast: ContrastConfig
    relight: RelightConfig
    haze: HazeConfig
    simplify: SimplifyConfig
    tones: TonesConfig
    lines: LinesConfig
    hatch: HatchConfig
    cleanup: CleanupConfig
    faces: FacesConfig
    pencil: PencilConfig


@dataclass
class ColorConfig:
    enabled: bool
    hue_ranges: list[list[int]]
    min_sat: float
    min_val: float
    strength: float
    morph: int
    min_area: int
    feather: float
    accent_saturation: float
    accent_values: list[float]
    accent_split: float


@dataclass
class SmokeConfig:
    enabled: bool
    style: str
    layers: int
    opacity: float
    speed: float
    depths: list[float]
    octaves: int
    base_cells: int
    scale_step: float
    speed_step: float
    direction: list[float]
    softness: float
    density: list[float]
    value: float


@dataclass
class EmbersConfig:
    enabled: bool
    count: int
    speed: list[float]
    sway: list[float]
    sway_hz: list[float]
    size: list[float]
    life: list[float]
    brightness: float
    color: list[float]
    glow: float
    glow_size: float
    trail: int
    trail_s: float
    rim: float
    rim_size: float


@dataclass
class AshConfig:
    enabled: bool
    count: int
    speed: list[float]
    sway: list[float]
    sway_hz: list[float]
    size: list[float]
    life: list[float]
    value: float
    opacity: float


@dataclass
class AtmosphereConfig:
    smoke: SmokeConfig
    embers: EmbersConfig
    ash: AshConfig


@dataclass
class FilmConfig:
    grain: float
    vignette: float
    vignette_power: float
    flicker: float
    flicker_frames: int
    weave_px: float
    weave_frames: int


@dataclass
class Config:
    seed: int
    output: OutputConfig
    motion: MotionConfig
    cadence: CadenceConfig
    stylize: StylizeConfig
    color: ColorConfig
    atmosphere: AtmosphereConfig
    film: FilmConfig


def load_config(
    path: str | Path | Sequence[str | Path] | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Load the default config, merge user YAML files on top in order, then apply
    overrides.

    `path` is one YAML file or a list of them; later files win. Overrides are
    "dotted.key=value" strings; the value is parsed as YAML, so
    `motion.pan=[0.05,0]` and `stylize.hatch.enabled=false` both work.
    """
    data = _read_yaml(DEFAULT_CONFIG_PATH)
    paths = [] if path is None else [path] if isinstance(path, (str, Path)) else list(path)
    for user_path in paths:
        try:
            _merge(data, _read_yaml(Path(user_path)), prefix="")
        except ConfigError as exc:
            raise ConfigError(f"{user_path}: {exc}") from exc
    for item in overrides or []:
        _apply_override(data, item)
    return _build(Config, data, prefix="")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a mapping at the top level")
    return data


def _merge(base: dict[str, Any], user: dict[str, Any], prefix: str) -> None:
    for key, value in user.items():
        dotted = f"{prefix}{key}"
        if key not in base:
            raise ConfigError(_unknown_key_message(dotted, base, prefix))
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ConfigError(f"config key '{dotted}' is a section and needs a mapping")
            _merge(base[key], value, prefix=f"{dotted}.")
        else:
            base[key] = copy.deepcopy(value)


def _apply_override(data: dict[str, Any], item: str) -> None:
    if "=" not in item:
        raise ConfigError(f"override '{item}' must look like dotted.key=value")
    dotted, raw = item.split("=", 1)
    dotted = dotted.strip()
    parts = dotted.split(".")
    node = data
    for depth, key in enumerate(parts):
        prefix = "".join(f"{p}." for p in parts[:depth])
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(_unknown_key_message(dotted, node, prefix))
        if depth == len(parts) - 1:
            if isinstance(node[key], dict):
                raise ConfigError(f"override '{dotted}' names a section; set one of its keys")
            try:
                node[key] = yaml.safe_load(raw)
            except yaml.YAMLError as exc:
                raise ConfigError(f"override '{item}': cannot parse value: {exc}") from exc
        else:
            node = node[key]


def _unknown_key_message(dotted: str, node: Any, prefix: str) -> str:
    valid = ", ".join(f"{prefix}{k}" for k in node) if isinstance(node, dict) else "(none)"
    return f"unknown config key '{dotted}'. Valid keys here: {valid}"


def _build(cls: type, data: Any, prefix: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"config section '{prefix.rstrip('.') or '<root>'}' must be a mapping")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = [k for k in data if k not in hints]
    if unknown:
        raise ConfigError(_unknown_key_message(f"{prefix}{unknown[0]}", hints, prefix))
    missing = [n for n in names if n not in data]
    if missing:
        raise ConfigError(f"missing config key '{prefix}{missing[0]}'")
    kwargs = {}
    for name in names:
        hint = hints[name]
        dotted = f"{prefix}{name}"
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, data[name], prefix=f"{dotted}.")
        else:
            kwargs[name] = _coerce(data[name], hint, dotted)
    return cls(**kwargs)


def _coerce(value: Any, hint: Any, key: str) -> Any:
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        return _coerce(value, inner[0], key)

    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"config key '{key}' expects a list, got {value!r}")
        return [_coerce(v, args[0], f"{key}[{i}]") for i, v in enumerate(value)]

    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"config key '{key}' expects true/false, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"config key '{key}' expects an integer, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"config key '{key}' expects a number, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"config key '{key}' expects a string, got {value!r}")
        return value
    raise ConfigError(f"config key '{key}' has unsupported type {hint!r}")


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Plain nested dict of a config, in the same layout as default.yaml."""
    return dataclasses.asdict(cfg)
