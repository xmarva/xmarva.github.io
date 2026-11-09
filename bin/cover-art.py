"""Make the plate that a post's header shows beside its title: the picture of the post's cover, without the cover's
own title, centred on a 5:4 plate with the cover's background and dot grid. Saved next to the cover as cover-art.png;
the post layout uses it when it exists and falls back to the cover otherwise.

    uv run --with pillow python bin/cover-art.py                 # every cover under assets/posts/
    uv run --with pillow python bin/cover-art.py assets/posts/<slug>/cover.png
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, PAD = 1200, 960, 0.11


def background(cover):
    return cover.getpixel((cover.width - 4, cover.height - 4))


def dots(cover, bg):
    colors = cover.crop((cover.width // 2, 0, cover.width, cover.height)).getcolors(1 << 20) or []
    near = [c for c in colors if c[1] != bg and max(abs(a - b) for a, b in zip(c[1], bg)) < 30]
    return max(near)[1] if near else bg


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
    draw = ImageDraw.Draw(out)
    grid, step = dots(cover, bg), 48
    for y in range(step // 2, HEIGHT, step):
        for x in range(step // 2, WIDTH, step):
            draw.ellipse((x - 2.4, y - 2.4, x + 2.4, y + 2.4), fill=grid)
    scale = min(WIDTH * (1 - 2 * PAD) / motif.width, HEIGHT * (1 - 2 * PAD) / motif.height, 1.6)
    motif = motif.resize((round(motif.width * scale), round(motif.height * scale)), Image.LANCZOS)
    out.paste(motif, ((WIDTH - motif.width) // 2, (HEIGHT - motif.height) // 2))
    return out


def main():
    covers = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "assets/posts").glob("**/cover.png"))
    for cover in covers:
        art = plate(cover)
        if art is None:
            print("skipped", cover)
            continue
        art.save(cover.with_name("cover-art.png"), optimize=True)
        print("made", cover.with_name("cover-art.png").relative_to(ROOT) if cover.is_absolute() else cover.with_name("cover-art.png"))


if __name__ == "__main__":
    main()
