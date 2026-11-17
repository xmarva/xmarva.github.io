"""Draw the link-preview pictures of the teaching pages, in the style of the lesson covers: the dark ground with its
dot grid, the course's title on the left and its newest covers fanned out on the right, as on the page itself.

- assets/img/teaching/card-<slug>.png for each shown course of _data/courses.yml (the course page's og:image,
  picked up by _plugins/course-pages.rb);
- assets/img/teaching/card-teaching.png for /teaching/, with the newest cover of each course.

    uv run --with pillow --with pyyaml python bin/course-card.py

The fonts are the covers' IBM Plex, from the course repository next to this one (--fonts to point elsewhere). Run it
again when a course gets a new lesson or a new title.
"""

import argparse
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT.parent.parent / "teaching" / "transformers-from-scratch" / "src" / "tfs" / "viz" / "fonts"
W, H = 2400, 1260
BG, DOT, INK, MUTED, FAINT, RED, LINE = "#161618", "#232327", "#F2F3F5", "#9DA3AE", "#6E7380", "#FF3636", "#4A4D55"
WORDS = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}


def font(folder, name, size):
    return ImageFont.truetype(str(folder / f"IBMPlex{name}.ttf"), size)


def wrap(draw, text, face, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=face) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line]


def rounded(image, radius):
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, image.width - 1, image.height - 1), radius, fill=255)
    image = image.convert("RGBA")
    image.putalpha(mask)
    return image


def fan(canvas, covers, right, centre_y, width):
    """The covers as on the page: each one back is shifted up and left by a tenth, turned 4° more and darker."""
    for i in reversed(range(len(covers))):
        cover = Image.open(covers[i]).convert("RGB")
        cover = cover.resize((width, round(width * cover.height / cover.width)), Image.LANCZOS)
        cover = ImageEnhance.Brightness(cover).enhance(1 - 0.12 * i)
        card = rounded(cover, 22)
        turned = card.rotate(4 * i, resample=Image.BICUBIC, expand=True)
        cx = right - width / 2 - 0.10 * width * i
        cy = centre_y - 0.10 * card.height * i
        x, y = round(cx - turned.width / 2), round(cy - turned.height / 2)
        # the shadow is blurred on a margin of its own, or the blur stops at the card's edge
        m = 90
        alpha = Image.new("L", (turned.width + 2 * m, turned.height + 2 * m), 0)
        alpha.paste(turned.getchannel("A").point(lambda a: a * 0.6), (m, m))
        shadow = Image.new("RGBA", alpha.size, (0, 0, 0, 0))
        shadow.putalpha(alpha.filter(ImageFilter.GaussianBlur(30)))
        canvas.alpha_composite(shadow, (x - m, y - m + 26))
        canvas.alpha_composite(turned, (x, y))


def card(folder, tab, title, line, covers, out):
    canvas = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    for x in range(14, W, 28):
        for y in range(14, H, 28):
            draw.ellipse((x - 1.8, y - 1.8, x + 1.8, y + 1.8), fill=DOT)
    fan(canvas, covers, right=W - 100, centre_y=H / 2 + 50, width=900)

    # the text sits on a ground of its own, so that the fan never runs under it
    draw = ImageDraw.Draw(canvas)
    mono, sans, bold = font(folder, "Mono-Medium", 30), font(folder, "Sans-Regular", 46), font(folder, "Sans-SemiBold", 100)
    x0, width = 130, 900
    tab_w = draw.textlength(tab, font=mono) + 88
    draw.rounded_rectangle((x0, 150, x0 + tab_w, 214), 14, fill="#1E1F23", outline=LINE, width=3)
    draw.ellipse((x0 + 28, 172, x0 + 48, 192), fill=RED)
    draw.text((x0 + 64, 182), tab, font=mono, fill="#C9CCD3", anchor="lm")
    y = 300
    for row in wrap(draw, title, bold, width):
        draw.text((x0, y), row, font=bold, fill=INK)
        y += 118
    y += 34
    for row in wrap(draw, line, sans, width - 60):
        draw.text((x0, y), row, font=sans, fill=MUTED)
        y += 62
    draw.rectangle((x0, H - 150, x0 + 56, H - 144), fill=RED)
    draw.text((x0, H - 116), "blog.korolevalab.com/teaching", font=font(folder, "Mono-Regular", 32), fill=FAINT)
    canvas.convert("RGB").save(out, optimize=True)
    print(out.relative_to(ROOT), f"{out.stat().st_size / 1e6:.2f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fonts", type=Path, default=FONTS)
    args = parser.parse_args()
    if not (args.fonts / "IBMPlexSans-SemiBold.ttf").exists():
        raise SystemExit(f"no IBM Plex in {args.fonts}: pass --fonts")
    courses = [c for c in yaml.safe_load((ROOT / "_data" / "courses.yml").read_text()) if c.get("show")]
    newest = []
    for number, course in enumerate(courses, 1):
        published = [lesson for lesson in course["lessons"] if lesson.get("url")]
        covers = [ROOT / lesson["cover"] for lesson in reversed(published)][:3]
        newest.append(covers[0])
        n = len(course["lessons"])
        if course.get("price"):
            line = f"{n} lessons with notebooks on Kaggle and Colab, {course['price']} {course['currency']}"
        else:
            line = f"{n} free lessons with notebooks on Kaggle and Colab"
        card(args.fonts, f"course {number} of {len(courses)}", course["title"], line, covers,
             ROOT / "assets" / "img" / "teaching" / f"card-{course['slug']}.png")
    lessons = sum(len(course["lessons"]) for course in courses)
    card(args.fonts, "teaching", "Courses on Transformers",
         f"{WORDS.get(len(courses), len(courses))} free courses, {lessons} lessons, in PyTorch from scratch",
         newest[::-1][:3], ROOT / "assets" / "img" / "teaching" / "card-teaching.png")


if __name__ == "__main__":
    main()
