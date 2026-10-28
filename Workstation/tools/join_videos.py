#!/usr/bin/env python3
"""All cases in one video: results/video/case<c>/video3d_case<c>.mp4 played one after the other (same size and codec,
joined without re-encoding), cases in the order 0, 1.1, 1.2, 2.1, 2.2, 3.1, 3.2 (missing ones are skipped).
usage: join_videos.py [DIR] [--out VIDEO.mp4]
"""
import argparse, os, subprocess, tempfile
import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser(); ap.add_argument("dir", nargs="?", default=os.path.join(ROOT, "results", "video")); ap.add_argument("--out")
a = ap.parse_args()
vids = [os.path.abspath(os.path.join(a.dir, f"case{c}", f"video3d_case{c}.mp4")) for c in ("0", "1.1", "1.2", "2.1", "2.2", "3.1", "3.2")]
vids = [v for v in vids if os.path.exists(v)]
out = a.out or os.path.join(a.dir, "video3d_all_cases.mp4")
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    for v in vids: f.write("file '" + v.replace("'", "'\\''") + "'\n")
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", f.name, "-c", "copy",
                out], check=True)
os.unlink(f.name)
print("written", out, "from", len(vids), "cases")
