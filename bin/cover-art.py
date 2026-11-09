"""Make, next to every post's cover.png:

- cover-art.png, the plate that the post's header shows beside its title: the cover's picture without the cover's own
  title, centred on a 5:4 plate with the cover's background; the page draws the dot grid behind it, so the box can
  grow with a long title while the picture is never cropped;
- cover-light.png and cover-art-light.png, the same two pictures on a light ground for the light theme.

The covers are drawn from a few flat colours, a dark background, neutral greys and the site's red, so the light
version is exact rather than an inversion: every pixel is split into its share of grey and its share of red over the
background, and rebuilt from paper, ink and red with the same shares, which keeps anti-aliased edges clean.

    uv run --with pillow --with numpy python bin/cover-art.py                 # every cover under assets/posts/
    uv run --with pillow --with numpy python bin/cover-art.py assets/posts/<slug>/cover.png
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, PAD = 1200, 960, 0.11
PAPER = np.array([246, 243, 236], float)  # a plate on the paper page
INK = np.array([29, 36, 48], float)  # the course's ink
RED_LIGHT = np.array([214, 48, 48], float)  # the site's red, a touch deeper on a light ground
RED_DARK = np.array([255, 59, 59], float)  # the covers' red


def background(cover):
    return cover.getpixel((cover.width - 4, cover.height - 4))


def plate(path):
    cover = Image.open(path).convert("RGB")
    bg = background(cover)
    if max(bg) > 60:  # not one of the site's dark covers: leave it to the fallback
        return None
    panel = cover.crop((cover.width // 2, 0, cover.width, cover.height))
    marks = panel.convert("L").point(lambda v: 255 if v > 70 else 0)
    box = marks.getbbox()
    if not box:
        return None
    motif = panel.crop(box)
    out = Image.new("RGB", (WIDTH, HEIGHT), bg)
    scale = min(WIDTH * (1 - 2 * PAD) / motif.width, HEIGHT * (1 - 2 * PAD) / motif.height, 1.6)
    motif = motif.resize((round(motif.width * scale), round(motif.height * scale)), Image.LANCZOS)
    out.paste(motif, ((WIDTH - motif.width) // 2, (HEIGHT - motif.height) // 2))
    return out


def lighten(image):
    img = np.asarray(image.convert("RGB")).astype(float)
    bg = img[-3, -3]
    red = np.clip((img[..., 0] - img[..., 1]) / (RED_DARK[0] - RED_DARK[1]), 0, 1)  # share of red
    grey = img.mean(axis=2) - red * (RED_DARK.mean() - bg.mean())
    light = np.clip((grey - bg.mean()) / (235 - bg.mean()), 0, 1)  # share of light grey over the background
    out = PAPER + light[..., None] * (INK - PAPER)
    out = out * (1 - red[..., None]) + red[..., None] * RED_LIGHT
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main():
    covers = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "assets/posts").glob("**/cover.png"))
    for cover in covers:
        art = plate(cover)
        if art is None:
            print("skipped", cover)
            continue
        art.save(cover.with_name("cover-art.png"), optimize=True)
        lighten(art).save(cover.with_name("cover-art-light.png"), optimize=True)
        lighten(Image.open(cover)).save(cover.with_name("cover-light.png"), optimize=True)
        print("made", cover.with_name("cover-art.png").relative_to(ROOT) if cover.is_absolute() else cover.with_name("cover-art.png"))


if __name__ == "__main__":
    main()
