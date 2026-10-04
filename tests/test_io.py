import imageio_ffmpeg
import numpy as np
import pytest

from inkroto.io import VideoWriter, load_image, next_numbered_path, read_video, save_debug_sheet


@pytest.mark.parametrize("width", [160, 321, 640])
def test_load_image_even_float(photo_path, width):
    img = load_image(photo_path, width)
    h, w, c = img.shape
    assert c == 3
    assert w % 2 == 0 and h % 2 == 0
    assert abs(w - width) <= 1
    assert img.dtype == np.float32
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert abs(h / w - 240 / 320) < 0.02


def test_video_writer_frame_count(tmp_path, photo):
    path = tmp_path / "out.mp4"
    gray = photo.mean(axis=2)
    n = 10
    with VideoWriter(path, 24, (320, 240)) as writer:
        for i in range(n):
            writer.write(photo if i % 2 else gray)
    assert writer.frames_written == n
    assert path.is_file() and path.stat().st_size > 0

    frames, fps = read_video(path)
    assert len(frames) == n
    assert fps == pytest.approx(24)
    assert frames[0].shape == (240, 320, 3)
    assert frames[0].dtype == np.float32
    n_frames, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
    assert n_frames == n


def test_video_writer_rejects_wrong_size(tmp_path):
    with VideoWriter(tmp_path / "bad.mp4", 24, (64, 48)) as writer:
        with pytest.raises(ValueError):
            writer.write(np.zeros((40, 64), np.float32))
        writer.write(np.zeros((48, 64), np.float32))


def test_video_writer_rejects_odd_size(tmp_path):
    with pytest.raises(ValueError):
        VideoWriter(tmp_path / "odd.mp4", 24, (65, 48))


def test_debug_sheet(tmp_path, photo, fake_depth, fake_matte):
    path = tmp_path / "sheet.png"
    save_debug_sheet({"photo": photo, "depth": fake_depth, "matte": fake_matte}, path)
    sheet = load_image(path, 200)
    assert sheet.ndim == 3


def test_next_numbered_path(tmp_path):
    out = tmp_path / "out.mp4"
    assert next_numbered_path(out) == tmp_path / "out_001.mp4"
    (tmp_path / "out.mp4").touch()
    (tmp_path / "out_001.mp4").touch()
    (tmp_path / "out_007.mp4").touch()
    (tmp_path / "out_002.png").touch()       # different suffix
    (tmp_path / "outtake_050.mp4").touch()   # different name
    assert next_numbered_path(out) == tmp_path / "out_008.mp4"
    # A numbered path continues the same series instead of nesting numbers.
    assert next_numbered_path(tmp_path / "out_003.mp4") == tmp_path / "out_008.mp4"
    # Names with glob characters and a missing folder are fine.
    assert next_numbered_path(tmp_path / "a[1]*.mp4") == tmp_path / "a[1]*_001.mp4"
    assert next_numbered_path(tmp_path / "new" / "x.mp4") == tmp_path / "new" / "x_001.mp4"
