# inkroto

Turn a still photo into a short black-and-white ink-style video with a
hand-drawn, rotoscoped feel: flat tones, hard directional shadows, black
contour lines, scratchy hatching in the midtones, drawings held for two frames,
depth haze, drifting smoke, and a few flat color accents.

```
photo ─ load ─ analyze (depth, matte) ─ motion (parallax camera move)
      ─ group frames into held drawings
      ─ per drawing: average, stylize, color
      ─ per frame:   atmosphere, film finish
      ─ write mp4
```

## Install

Python 3.11 and a working `pip`. ffmpeg comes bundled through `imageio-ffmpeg`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[ml,dev]"
```

The `ml` extra (torch, transformers, rembg, onnxruntime, pillow) is needed for
depth and matte estimation. Face detection uses OpenCV's YuNet model, which is
downloaded (about 230 KB, checksum-verified) into the cache folder the first
time faces are enabled; without network access, rendering continues without
face handling. The first run downloads Depth Anything V2 Small and
the u2net matte model; after that, results are cached per image in `.cache/`.
Without the extra, `render`, `analyze`, and `batch` stop with a message telling
you to run `pip install -e ".[ml]"`.

Run the tests (offline, no GPU, real-model tests skipped):

```bash
pytest
```

Run the slow tests that load the real models:

```bash
pytest -m slow
```

## Commands

Every command takes `--config FILE` (YAML merged over the defaults) and any
number of `--set dotted.key=value` overrides. Values are parsed as YAML, so
`--set motion.pan=[0.05,0]` and `--set stylize.hatch.enabled=false` both work.
Unknown keys are rejected with the list of valid ones.

### `inkroto render`

```bash
inkroto render photo.jpg -o out.mp4
inkroto render clip.mp4 -o out.mp4
```

INPUT can be a photo, or a video or gif (`.mp4 .mov .m4v .mkv .webm .avi .gif`).
A photo gets a parallax camera move built from its depth map. A video or gif
is used as the motion source instead (the clip backend): its frames are
resampled to `output.fps` by nearest source frame, resized to `output.width`,
and trimmed to `output.duration_s` (a shorter clip keeps its own length). Depth
and matte are estimated only for the frame each drawing is made from, cached
per video and frame, smoothed over time, and normalized with the first
drawing's range so haze does not pump. No parallax move is added on top.

A good source clip is a few seconds of slow, subtle motion with a mostly static
camera, for example from an image-to-video model fed your photo. If tone
boundaries crawl on a moving subject, raise `cadence.hold` to 3 or
`stylize.simplify.radius`.

Writes `out_001.mp4`; the next run writes `out_002.mp4`, so nothing is ever
overwritten. Next to each video it writes the fully resolved config, e.g.
`out_001.config.yaml`, which you can pass back with `--config` to reproduce it.

- `--debug DIR` saves `out_001_stages.png` (every stage of the first drawing:
  input, depth, matte, luminance, relight, haze, simplify, tones, cleanup,
  hatch, lines, drawing, color) plus the first, middle, and last finished
  frames (with atmosphere and film) as PNGs.
- `--cache-dir DIR` moves the analysis cache (default `.cache`).

### Pencil look

```bash
inkroto render photo.jpg -o out.mp4 --config configs/pencil.yaml
```

`stylize.medium: pencil` renders the same drawing in graphite instead of ink:
gray lines sketched in several slightly offset passes, with pressure variation
and small gaps; hatched shading and no solid black; and a paper grain that
breaks up the graphite and stays still while the drawing redraws. The
[`configs/pencil.yaml`](configs/pencil.yaml) preset also fades distance to
white paper, turns color accents and smoke off, and lightens the film finish.
Combine it with `--set` as usual, e.g. `--set stylize.pencil.passes=2`.

### `inkroto analyze`

```bash
inkroto analyze photo.jpg -o sheet.png
```

Writes the photo, its depth map, its subject matte, and its face mask side by
side. Use it to check face detection: the detector can mistake face-like
objects (a spoked wheel, a drawn face) for faces and can miss small or blurry
ones. Raise `stylize.faces.min_score` to drop false faces, or set
`stylize.faces.enabled=false` for that image (a `name.yaml` sidecar in batch). Near
objects must be bright in the depth panel; if they are dark, flip
`DEPTH_MODEL_INVERTED` in `src/inkroto/analyze.py`.

### `inkroto batch`

```bash
inkroto batch photos/ -o clips/
```

Renders every `.jpg`, `.jpeg`, and `.png` in `photos/` (not recursive) into
`clips/`. Config layers, later wins:

1. `configs/default.yaml`
2. `--config FILE`
3. `photos/name.yaml`, if it sits next to `photos/name.jpg`
4. `--set` overrides

A failing image is reported with its error and the batch carries on; the exit
code is 1 if any image failed. Two images with the same name but different
extensions (`shot.jpg`, `shot.png`) become `shot_jpg_001.mp4` and
`shot_png_001.mp4`. `--debug DIR` works as in `render`, for every image.

## Config keys

All tunable numbers live in [`configs/default.yaml`](configs/default.yaml) and
the matching dataclasses in `src/inkroto/config.py`. Depth is 0 (far) to 1
(near); sizes in px are at the output width.

### Top level and output

| Key | Default | Meaning |
|---|---|---|
| `seed` | 1234 | All randomness derives from this; same seed, byte-identical output |
| `output.fps` | 24 | Frames per second |
| `output.width` | 1280 | Output width; height follows the photo's aspect, both even |
| `output.duration_s` | 4.0 | Clip length in seconds |

### `motion`: camera move

| Key | Default | Meaning |
|---|---|---|
| `backend` | parallax | `parallax` (camera move on a photo) or `clip` (motion from a video or gif) |
| `pan` | [0.03, 0.0] | Total travel as a fraction of width, height |
| `zoom` | 1.06 | End zoom (start is 1.0) |
| `focus` | 0.5 | Depth that stays still; nearer moves one way, farther the other |
| `clip_path` | null | Video or gif for the clip backend; set automatically when INPUT is one |
| `depth_smoothing` | 0.5 | Clip backend: weight of each new drawing's depth in a running average (1 = no smoothing) |
| `matte_threshold` | 0.5 | Matte above this is the subject when building the background plate |
| `dilate_px` | 9 | Grow the subject mask before inpainting the background |
| `inpaint_radius` | 5 | Inpaint radius at half resolution |
| `depth_blur` | 4.0 | Blur (px) on the depth that drives the warp, to avoid tears |

### `cadence`

| Key | Default | Meaning |
|---|---|---|
| `hold` | 2 | Output frames per drawing ("on twos") |

### `stylize`: turning a frame into a drawing

Stages run in this order: luminance, contrast, relight, haze, simplify, tones,
cleanup, hatch; lines are traced separately and laid on top.

| Key | Default | Meaning |
|---|---|---|
| `medium` | ink | `ink` (black ink, flat tones) or `pencil` (graphite on paper, see `stylize.pencil`) |
| `contrast.enabled` | false | Local contrast (CLAHE); helps flat, murky photos |
| `contrast.clip_limit` | 2.0 | Strength; higher = more local contrast and noise |
| `contrast.tiles` | 8 | Grid size; smaller = more local |
| `relight.enabled` | true | Directional shading from the depth map |
| `relight.direction` | [-0.6, -0.5, 0.6] | Light direction (x right, y down, z toward viewer) |
| `relight.strength` | 0.4 | 0 = no shading |
| `relight.normal_blur` | 3.0 | Blur on depth before taking normals |
| `relight.normal_z` | 1.0 | Smaller = steeper, harder shading |
| `haze.enabled` | true | Far things fade toward gray |
| `haze.start` | 0.4 | Depth below which haze begins |
| `haze.density` | 0.5 | How far the farthest pixels move toward `value` |
| `haze.value` | 0.75 | Gray that far objects fade toward |
| `simplify.method` | kuwahara | `kuwahara`, `meanshift`, or `bilateral` |
| `simplify.radius` | 3 | Larger = flatter regions, less detail |
| `simplify.sigma_color` | 0.1 | Color range for meanshift and bilateral |
| `tones.percentiles` | [5, 50] | Black below the first, white above the second, one gray between |
| `tones.mid_value` | 0.55 | The mid tone (replaced by hatching when hatch is on) |
| `tones.jitter` | 0.015 | Per-drawing threshold wobble, so held drawings differ slightly |
| `lines.mode` | edges | `edges`: a line wherever there is enough edge contrast, at any brightness; `level`: classic XDoG, which misses edges in bright areas and inks flat dark areas |
| `lines.source` | relit | Where lines are traced: `relit` (before haze), `hazed`, or `simplified` |
| `lines.sigma` | 0.8 | Line scale; smaller = finer lines |
| `lines.k`, `lines.p`, `lines.phi` | 1.6, 20, 40 | XDoG shape: blur ratio, sharpening, line softness |
| `lines.edge_threshold` | 0.10 | Edges mode: minimum edge contrast that gets a line; lower = more lines (and more texture) |
| `lines.epsilon` | 0.04 | Level mode: XDoG threshold; flat areas darker than this also ink |
| `lines.denoise`, `lines.denoise_px` | 0.06, 2.0 | Edge-preserving smoothing before tracing, so film grain and noise are not drawn (0 = off) |
| `lines.wobble_px` | 1.2 | Max hand-drawn displacement of lines (0 = off) |
| `lines.wobble_scale` | 32 | Size of the wobble's bends; smaller = shakier |
| `hatch.enabled` | true | Replace the flat mid tone with ink strokes |
| `hatch.spacing` | 9 | Px between strokes; wider keeps small shapes readable |
| `hatch.angle` | 35 | Stroke angle; the darker half of the midtones also gets angle + 90 |
| `hatch.thickness` | 0.9 | Stroke thickness |
| `hatch.jitter_px` | 1.0 | How far strokes wander sideways |
| `hatch.thickness_jitter` | 0.4 | Relative thickness variation |
| `hatch.noise_scale` | 40 | Size of the unevenness |
| `hatch.segment`, `hatch.gap` | [20, 80], [2, 10] | Stroke segment and gap length ranges |
| `cleanup.min_area` | 24 | Black or white specks smaller than this take the surrounding tone |
| `cleanup.morph` | 0 | Open/close kernel; above 1 also erases features thinner than this |
| `faces.enabled` | true | Detected faces read as clean light shapes: stronger simplify, two tones, no hatching; lines unchanged |
| `faces.simplify_radius` | 7 | Simplify radius inside faces |
| `faces.two_tone` | true | Collapse the mid tone into white inside faces |
| `faces.min_score` | 0.75 | Detector confidence needed to count as a face; raise it if objects are mistaken for faces |
| `pencil.line_value` | 0.15 | Darkest graphite line (ink is 0) |
| `pencil.passes`, `pencil.pass_offset_px` | 3, 0.8 | Sketchy overdraw: how many passes, and how far they drift apart |
| `pencil.pressure`, `pencil.pressure_scale` | 0.35, 48 | Line darkness variation along strokes, and its size in px |
| `pencil.break_rate`, `pencil.break_scale` | 0.15, 14 | Fraction of line length left as gaps, and their size in px |
| `pencil.shade_values` | [0.4, 0.9] | Graphite value of shadows and of the mid tone (hatching adds the rest) |
| `pencil.paper_tooth` | 0.6 | How much the paper grain breaks up graphite |
| `pencil.paper_scale`, `pencil.paper_grain` | 1.5, 0.05 | Grain size, and how visible it is on blank paper |
| `pencil.smudge` | 0.8 | Px blur that softens the shading |
| `pencil.paper_tint` | [0.97, 0.96, 0.92] | Page color |

### `color`: flat accents

| Key | Default | Meaning |
|---|---|---|
| `enabled` | true | Off = pure grayscale |
| `hue_ranges` | [[0, 12], [168, 180], [12, 28]] | Hues kept, OpenCV scale 0..180 (reds and oranges) |
| `min_sat`, `min_val` | 0.55, 0.35 | Only saturated, not-too-dark pixels |
| `strength` | 0.9 | How far masked pixels move from ink toward paint |
| `morph`, `min_area`, `feather` | 3, 64, 1.5 | Mask cleanup: speck removal, smallest island, edge blur |
| `accent_saturation` | 0.9 | Paint saturation |
| `accent_values`, `accent_split` | [0.55, 0.9], 0.5 | The two paint brightness levels and where they split |

### `atmosphere`: over the drawing, every frame

| Key | Default | Meaning |
|---|---|---|
| `smoke.enabled` | true | Drifting smoke; darkens paper toward gray, ink stays black |
| `smoke.style` | soft | `soft` wisps or `flat` hard-edged flat-tone shapes |
| `smoke.layers` | 3 | Number of smoke layers |
| `smoke.opacity` | 0.4 | Per layer |
| `smoke.speed` | 0.002 | Drift per frame as a fraction of the frame (farthest layer) |
| `smoke.depths` | [0.1, 0.6] | Depth of the farthest and nearest layer; only scene farther than a layer gets its smoke |
| `smoke.octaves`, `smoke.base_cells` | 4, 3 | Noise detail and wisp size |
| `smoke.scale_step`, `smoke.speed_step` | 1.5, 1.5 | Nearer layers have bigger wisps and move faster |
| `smoke.direction` | [1.0, -0.25] | Drift direction |
| `smoke.softness` | 0.1 | Soft edge where smoke meets nearer scene |
| `smoke.density` | [0.52, 0.68] | Noise range mapped from clear to full smoke |
| `smoke.value` | 0.72 | Gray the smoke darkens paper toward |
| `embers.enabled` | false | Rising orange sparks (white with an ink rim when color is off) |
| `embers.count`, `speed`, `sway`, `sway_hz`, `size`, `life` | | Particle count, rise speed, sideways sway, dot size, lifetime |
| `embers.brightness`, `glow`, `glow_size` | 1.0, 0.5, 3.0 | Core opacity and glow |
| `embers.trail`, `trail_s` | 4, 0.012 | Motion streak samples and spacing |
| `embers.rim`, `rim_size` | 1.5, 1.6 | Ink rim around white sparks |
| `ash.enabled` | false | Slowly falling gray flakes |
| `ash.count`, `speed`, `sway`, `sway_hz`, `size`, `life`, `value`, `opacity` | | As for embers, plus gray value and opacity |

### `film`: the last step before writing

| Key | Default | Meaning |
|---|---|---|
| `grain` | 0.06 | Grain at mid gray; fades toward pure black and white |
| `vignette`, `vignette_power` | 0.35, 2.0 | Corner darkening and its falloff |
| `flicker`, `flicker_frames` | 0.03, 6 | Brightness flicker amplitude and speed |
| `weave_px`, `weave_frames` | 0.4, 3 | Sub-pixel frame weave amplitude and speed |

Set any strength to 0 to turn that part off.

## Conventions

These hold across the codebase:

- Python 3.11, src layout, package `inkroto`.
- Images in memory are RGB, float32, 0..1, shape (H, W, 3). Conversion to or
  from BGR or integer types happens only in `io.py`.
- Grayscale is float32 0..1, shape (H, W): 0 is black ink, 1 is white paper.
- Depth is float32 0..1, (H, W): 1 is near the camera, 0 is far.
- Mattes are float32 0..1, (H, W): 1 is subject.
- Output width and height are always even.
- All randomness comes from `numpy.random.Generator` objects derived from the
  seed; no global random state. The same seed gives byte-identical output.
- Every tunable number lives in `configs/default.yaml` and the config
  dataclasses; no magic numbers inside functions.
- Heavy ML imports (torch, transformers, rembg) are lazy and live only in
  `analyze.py`.
- Each stage is a pure function: arrays and config in, arrays out. File I/O
  happens only in `io.py`, `analyze.py` (cache), and `render.py`.
- Tests run offline and without a GPU, use synthetic images from
  `tests/conftest.py`, and fake the depth and matte estimators. Tests that load
  real models are marked `slow` and skipped by default.

## Layout

```
src/inkroto/cli.py           command line (typer)
src/inkroto/config.py        dataclasses and YAML loading
src/inkroto/io.py            image, video, YAML read/write, debug sheets
src/inkroto/analyze.py       depth, matte, cache
src/inkroto/motion/          base.py (Shot), parallax.py, clip.py, backend registry
src/inkroto/cadence.py       frame grouping and per-drawing seeds
src/inkroto/stylize/         contrast, relight, haze, simplify, tones, cleanup,
                             hatch, lines, pencil, noise, pipeline
src/inkroto/composite/       color.py, atmosphere.py, film.py
src/inkroto/render.py        orchestration
configs/default.yaml
configs/pencil.yaml          pencil preset
tests/
```
