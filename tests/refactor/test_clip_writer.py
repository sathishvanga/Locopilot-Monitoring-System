"""T2 verification: clip_writer round-trip.

Builds a 5-frame 32x32 BGR clip in memory, calls save_video_clip,
asserts the file exists and is nonzero. If ffprobe is available on
PATH (or at /usr/bin/ffprobe), additionally asserts ffprobe reports
>=5 frames. The plan documents that on dev Macs without ffprobe in
/usr/bin we may relax to file-exists/size only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest


def _find_ffprobe() -> str | None:
    if os.path.exists('/usr/bin/ffprobe'):
        return '/usr/bin/ffprobe'
    return shutil.which('ffprobe')


def test_save_video_clip_writes_5_frame_mp4():
    from app.core.media.clip_writer import save_video_clip

    frames = [
        np.full((32, 32, 3), fill_value=i * 40, dtype=np.uint8)
        for i in range(5)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, 'test_clip.mp4')

        save_video_clip(frames, out_path, fps=1.0)

        assert os.path.exists(out_path), f"clip not written to {out_path}"
        assert os.path.getsize(out_path) > 0, "clip file is empty"

        ffprobe = _find_ffprobe()
        if ffprobe is None:
            pytest.skip("ffprobe not available; skipping frame-count assertion")

        # ffprobe -count_frames reports the exact number of decoded frames.
        result = subprocess.run(
            [
                ffprobe, '-v', 'error', '-count_frames',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_read_frames',
                '-of', 'default=nokey=1:noprint_wrappers=1',
                out_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"ffprobe failed: {result.stderr}"
        nb_frames = int(result.stdout.strip())
        assert nb_frames >= 5, f"expected >=5 frames, got {nb_frames}"
