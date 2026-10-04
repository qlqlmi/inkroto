"""Image and video read/write, and debug sheets.

This is the only module that converts to or from BGR or uint8.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import subprocess
import tempfile
from pathlib import Path
from types import TracebackType

import cv2
import imageio_ffmpeg
import numpy as np
import yaml

UINT8_MAX = 255.0
UINT16_MAX = 65535.0

# Output files are numbered name_001.mp4, name_002.mp4, ...
OUTPUT_NUMBER_DIGITS = 3

# Inputs treated as motion clips rather than photos.
VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".gif")
# Read files in chunks of this many bytes when hashing.
HASH_CHUNK = 1 << 20

# Encoder settings for VideoWriter.
VIDEO_CODEC = "libx264"
VIDEO_CRF = 16
VIDEO_PIX_FMT = "yuv420p"

# Debug sheet layout.
SHEET_TILE_WIDTH = 320
SHEET_LABEL_HEIGHT = 22
SHEET_PAD = 4
SHEET_FONT_SCALE = 0.5
SHEET_BG = 0.15
SHEET_TEXT = 1.0


def _even(n: int) -> int:
    return max(2, n - n % 2)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * UINT8_MAX + 0.5, 0, UINT8_MAX).astype(np.uint8)


def to_bgr_uint8(rgb: np.ndarray) -> np.ndarray:
    """RGB float32 0..1 to the BGR uint8 layout OpenCV detectors expect."""
    return cv2.cvtColor(to_uint8(rgb), cv2.COLOR_RGB2BGR)


def from_uint8(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / np.float32(UINT8_MAX)


def to_uint16(img: np.ndarray) -> np.ndarray:
    return np.clip(img * UINT16_MAX + 0.5, 0, UINT16_MAX).astype(np.uint16)


def from_uint16(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / np.float32(UINT16_MAX)


def as_rgb(img: np.ndarray) -> np.ndarray:
    """Return an (H, W, 3) float32 image from grayscale (H, W) or RGB (H, W, 3)."""
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        return np.repeat(img[:, :, None], 3, axis=2)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError(f"expected (H, W) or (H, W, 3) image, got shape {img.shape}")


def load_image(path: str | Path, width: int) -> np.ndarray:
    """Load an image as RGB float32 0..1, resized to `width` with even dimensions."""
    path = Path(path)
    buf = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not read image: {path}")
    src_h, src_w = bgr.shape[:2]
    out_w = _even(width)
    out_h = _even(int(round(src_h * out_w / src_w)))
    interp = cv2.INTER_AREA if out_w < src_w else cv2.INTER_CUBIC
    bgr = cv2.resize(bgr, (out_w, out_h), interpolation=interp)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return from_uint8(rgb)


def save_image(path: str | Path, img: np.ndarray) -> None:
    """Save a grayscale or RGB float32 0..1 image. Format comes from the suffix."""
    path = Path(path)
    bgr = cv2.cvtColor(to_uint8(as_rgb(img)), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(path.suffix or ".png", bgr)
    if not ok:
        raise ValueError(f"could not encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(path)


def next_numbered_path(path: str | Path) -> Path:
    """Next free numbered variant of `path`: out.mp4 -> out_001.mp4, out_002.mp4, ...

    The number is one more than the highest already in the folder, so earlier
    renders are never overwritten. A path that already ends in a number, such as
    out_004.mp4, counts in the same series as out.mp4.
    """
    path = Path(path)
    numbered = re.fullmatch(rf"(.*)_(\d{{{OUTPUT_NUMBER_DIGITS},}})", path.stem)
    base = numbered.group(1) if numbered else path.stem
    pattern = re.compile(
        rf"{re.escape(base)}_(\d{{{OUTPUT_NUMBER_DIGITS},}}){re.escape(path.suffix)}"
    )
    siblings = path.parent.iterdir() if path.parent.is_dir() else []
    used = [int(m.group(1)) for p in siblings if (m := pattern.fullmatch(p.name))]
    number = max(used, default=0) + 1
    return path.with_name(f"{base}_{number:0{OUTPUT_NUMBER_DIGITS}d}{path.suffix}")


def save_yaml(data: dict, path: str | Path) -> None:
    """Write a mapping as YAML, keeping key order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=None)


class VideoWriter:
    """Pipe raw RGB frames into ffmpeg (libx264, crf 16, yuv420p).

    `size` is (width, height); both must be even.
    """

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]):
        self.path = Path(path)
        self.fps = fps
        self.width, self.height = size
        if self.width % 2 or self.height % 2:
            raise ValueError(f"video size must be even, got {size}")
        self.frames_written = 0
        self._proc: subprocess.Popen | None = None
        self._stderr = None

    def __enter__(self) -> VideoWriter:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close(abort=exc_type is not None)

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-c:v", VIDEO_CODEC,
            "-crf", str(VIDEO_CRF),
            "-pix_fmt", VIDEO_PIX_FMT,
            "-movflags", "+faststart",
            str(self.path),
        ]
        self._stderr = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr
        )

    def write(self, frame: np.ndarray) -> None:
        """Write one RGB (H, W, 3) or grayscale (H, W) float32 0..1 frame."""
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("VideoWriter is not open")
        rgb = as_rgb(frame)
        if rgb.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"frame shape {rgb.shape[:2]} does not match video size "
                f"{(self.height, self.width)}"
            )
        try:
            self._proc.stdin.write(np.ascontiguousarray(to_uint8(rgb)).tobytes())
        except BrokenPipeError:
            self.close(abort=True)
            raise RuntimeError(f"ffmpeg exited early: {self._error_text()}") from None
        self.frames_written += 1

    def close(self, abort: bool = False) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        code = proc.wait()
        message = self._error_text()
        self._stderr.close()
        if code != 0 and not abort:
            raise RuntimeError(f"ffmpeg failed with exit code {code}: {message}")

    def _error_text(self) -> str:
        if self._stderr is None or self._stderr.closed:
            return ""
        self._stderr.seek(0)
        return self._stderr.read().decode(errors="replace").strip()


def read_video(path: str | Path) -> tuple[list[np.ndarray], float]:
    """Read every frame of a video as RGB float32 0..1. Returns (frames, fps)."""
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    meta = next(reader)
    width, height = meta["size"]
    frames = [
        from_uint8(np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3))
        for raw in reader
    ]
    return frames, float(meta["fps"])


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def file_sha1(path: str | Path) -> str:
    """sha1 of a file's bytes."""
    h = hashlib.sha1()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


class FrameStack:
    """Decoded video frames kept as uint8 to save memory; indexing returns
    RGB float32 0..1."""

    def __init__(self, frames: np.ndarray):
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError(f"expected (N, H, W, 3) uint8 frames, got {frames.shape} {frames.dtype}")
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int) -> np.ndarray:
        return from_uint8(self._frames[index])

    @property
    def size(self) -> tuple[int, int]:
        """(width, height)."""
        return self._frames.shape[2], self._frames.shape[1]


def _quiet_ffmpeg_reader(path: str, output_params: list[str] | None = None):
    # imageio_ffmpeg warns whenever output size differs from the source; that is
    # the point of resizing here.
    logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
    return imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24", output_params=output_params or [])


def probe_video(path: str | Path) -> tuple[tuple[int, int], float]:
    """((width, height), fps) of a video or gif without decoding it."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"video not found: {path}")
    reader = _quiet_ffmpeg_reader(str(path))
    try:
        meta = next(reader)
    finally:
        reader.close()
    return tuple(meta["source_size"]), float(meta["fps"])


def nearest_source_indices(n_out: int, src_fps: float, out_fps: float) -> list[int]:
    """Source frame nearest in time to each output frame k (at k / out_fps)."""
    return [int(math.floor(k * src_fps / out_fps + 0.5)) for k in range(n_out)]


def read_video_resampled(
    path: str | Path, fps: float, width: int, max_frames: int
) -> FrameStack:
    """Decode a video or gif and resample it to `fps` by nearest source frame,
    resized to `width` with even dimensions, keeping at most `max_frames` frames
    (fewer when the clip is shorter).

    Decoding is forced to a constant rate at the source fps first, so a gif's
    per-frame delays are honoured.
    """
    (src_w, src_h), src_fps = probe_video(path)
    if src_fps <= 0:
        raise ValueError(f"could not read a frame rate from {path}")
    out_w = _even(width)
    out_h = _even(int(round(src_h * out_w / src_w)))
    flags = "area" if out_w < src_w else "bicubic"
    wanted = nearest_source_indices(max_frames, src_fps, fps)
    reader = _quiet_ffmpeg_reader(
        str(path),
        ["-vf", f"scale={out_w}:{out_h}:flags={flags}", "-fps_mode", "cfr", "-r", f"{src_fps}"],
    )
    frames: list[np.ndarray] = []
    try:
        meta = next(reader)
        if tuple(meta["size"]) != (out_w, out_h):
            raise RuntimeError(f"ffmpeg returned {meta['size']}, expected {(out_w, out_h)}")
        k, index, last = 0, -1, None
        for index, raw in enumerate(reader):
            last = np.frombuffer(raw, dtype=np.uint8).reshape(out_h, out_w, 3)
            while k < len(wanted) and wanted[k] == index:  # a frame may repeat
                frames.append(last)
                k += 1
            if k == len(wanted):
                break
        # Output frames still inside the clip whose nearest frame would be one
        # past the end take the last frame.
        clip_end_s = (index + 1) / src_fps
        while last is not None and k < len(wanted) and k / fps < clip_end_s:
            frames.append(last)
            k += 1
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"no frames could be read from {path}")
    return FrameStack(np.stack(frames))


def save_debug_sheet(
    images: dict[str, np.ndarray], path: str | Path, cols: int | None = None
) -> None:
    """Tile labelled grayscale or RGB images into one PNG.

    `cols` sets the number of columns; by default the grid is roughly square.
    """
    if not images:
        raise ValueError("save_debug_sheet needs at least one image")
    bg = int(round(SHEET_BG * UINT8_MAX))
    text = int(round(SHEET_TEXT * UINT8_MAX))
    tiles = []
    for label, img in images.items():
        rgb = to_uint8(as_rgb(img))
        h, w = rgb.shape[:2]
        tile_h = max(1, int(round(h * SHEET_TILE_WIDTH / w)))
        body = cv2.resize(rgb, (SHEET_TILE_WIDTH, tile_h), interpolation=cv2.INTER_AREA)
        header = np.full((SHEET_LABEL_HEIGHT, SHEET_TILE_WIDTH, 3), bg, np.uint8)
        cv2.putText(
            header,
            str(label),
            (SHEET_PAD, SHEET_LABEL_HEIGHT - SHEET_PAD * 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            SHEET_FONT_SCALE,
            (text, text, text),
            1,
            cv2.LINE_AA,
        )
        tiles.append(np.vstack([header, body]))

    if cols is None:
        cols = math.ceil(math.sqrt(len(tiles)))
    cols = max(1, min(cols, len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    cell_h = max(t.shape[0] for t in tiles)
    cell_w = SHEET_TILE_WIDTH
    sheet = np.full(
        (rows * cell_h + (rows + 1) * SHEET_PAD, cols * cell_w + (cols + 1) * SHEET_PAD, 3),
        bg,
        np.uint8,
    )
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = SHEET_PAD + r * (cell_h + SHEET_PAD)
        x = SHEET_PAD + c * (cell_w + SHEET_PAD)
        sheet[y : y + tile.shape[0], x : x + cell_w] = tile
    save_image(Path(path).with_suffix(".png"), from_uint8(sheet))
