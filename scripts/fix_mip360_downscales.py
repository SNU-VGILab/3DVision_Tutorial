#!/usr/bin/env python3
"""Normalize mip-NeRF 360 down-scaled images to floor() dimensions.

Nerfstudio's colmap dataparser derives each camera's (width, height) from the
full-resolution COLMAP intrinsics divided by ``--downscale-factor`` using the
default ``--downscale-rounding-mode floor``. It then loads pixels from the
matching ``images_<factor>/`` folder. The *official* mip-NeRF 360 release ships
pre-downscaled folders whose sizes were produced with inconsistent rounding
(some scenes round up, some down, and width/height can differ within a scene),
so the loaded image is occasionally 1 px larger than the camera grid. That
mismatch makes the ray generator index one past the end:

    IndexError: index 1296 is out of bounds for dimension 1 with size 1296

No single ``--downscale-rounding-mode`` fixes every scene (bicycle/room/stump
need ``ceil``, garden needs ``round``, ...), so instead we regenerate the
down-scaled folders ourselves at exactly ``floor(orig / factor)`` — which is
what the dataparser assumes by default. After this, every scene trains with the
plain ``--downscale-factor 4`` command and no extra flags.

Run after the mip-NeRF 360 download. Idempotent: re-running just overwrites the
folders with the same floor-sized images.
"""
import os
import sys

from PIL import Image

BASE = sys.argv[1] if len(sys.argv) > 1 else "data/mipnerf360"
FACTORS = (4, 8)  # the notebook uses 4; 8 is offered as a "faster" option
EXTS = (".jpg", ".jpeg", ".png")


def main() -> None:
    if not os.path.isdir(BASE):
        print(f"[fix_mip360_downscales] {BASE} not found, skipping.")
        return
    for scene in sorted(os.listdir(BASE)):
        src = os.path.join(BASE, scene, "images")
        if not os.path.isdir(src):
            continue
        files = sorted(f for f in os.listdir(src) if f.lower().endswith(EXTS))
        if not files:
            continue
        w0, h0 = Image.open(os.path.join(src, files[0])).size
        for factor in FACTORS:
            dst = os.path.join(BASE, scene, f"images_{factor}")
            os.makedirs(dst, exist_ok=True)
            tw, th = w0 // factor, h0 // factor
            for f in files:
                img = Image.open(os.path.join(src, f)).resize((tw, th), Image.LANCZOS)
                img.save(os.path.join(dst, f), quality=95)
        print(f"  {scene}: images_4={w0 // 4}x{h0 // 4}  images_8={w0 // 8}x{h0 // 8}")
    print("[fix_mip360_downscales] done.")


if __name__ == "__main__":
    main()
