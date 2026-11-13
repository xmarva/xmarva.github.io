#!/usr/bin/env python3
"""Flag generated-sounding words, constructions and structure in the journal's texts (see docs/editorial-guide.md).

    python bin/style-lint.py _posts/2026-11-13-some-post.md
    python bin/style-lint.py ../transformers-from-scratch/architectures/a13_vlm/lesson.py
    python bin/style-lint.py --strict file.md      # exit status 1 if anything is flagged

Works on Markdown posts (front matter, prose; code blocks, HTML lines and the reference list are skipped) and on
jupytext lessons (only the markdown cells are read). English and Russian rules both apply. A flag is a candidate
for the editor, not a verdict: some uses are legitimate.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

WORDS_EN = {
    "filler-verb": r"\b(delve|delves|delving|dive into|deep dive|unpack|leverag(e|es|ing)|harness(es|ing)?|unlock(s|ing)?|"
                   r"unleash(es|ing)?|empower(s|ing)?|elevat(e|es|ing)|foster(s|ing)?|bolster(s|ing)?|showcas(e|es|ing))\b",
    "praise-adjective": r"\b(crucial|vital|pivotal|paramount|seamless(ly)?|cutting-edge|groundbreaking|revolutionary|"
                        r"game[- ]changer|game[- ]changing|transformative|robust(ly)?|powerful)\b",
    "vague-noun": r"\b(landscape|realm|tapestry|journey|paradigm shift|ever-evolving|fast-paced)\b",
    "complexity-word": r"\b(intricate|nuanced|multifaceted|holistic)\b",
    "testament": r"\b(a testament to|stands as|serves as a reminder|sheds? light on|paves? the way)\b",
    "empty-adverb": r"\b(it'?s worth noting|it is worth noting|notably|importantly|interestingly|essentially|"
                    r"fundamentally|ultimately|at its core)\b",
    "vague-quantity": r"\b(a myriad of|myriad|a plethora of|plethora|a wide range of)\b",
    "era-opening": r"\b(in today'?s (fast-paced )?world|in the era of|in the age of)\b",
}
CONSTRUCTIONS_EN = {
    "contrast-reveal": r"\b(not (just|only|merely) [^.;]{1,60}[,;—-] ?(it'?s|but)|isn'?t [^.]{1,40}\. it'?s|"
                       r"is not [^.]{1,40}\. it is)\b",
    "rhetorical-qa": r"(^|[.!]\s)(the (result|answer|catch|twist|kicker|problem|truth|verdict)|why does (this|it) "
                     r"matter|so what)\?",
    "signposting": r"\b(let'?s (dive|take a (closer )?look|break (it|this) down|explore|unpack)|in this (post|article|"
                   r"section),? (we|i)('ll| will)|here'?s the thing|in conclusion|to sum up|key takeaways|buckle up|"
                   r"without further ado)\b",
    "staged-suspense": r"(\bspoiler:|\bplot twist|^enter [A-Z]|\bthis is where [^.]{1,40} comes in\b)",
    "hedge-stack": r"\b(may potentially|could potentially|might possibly|could arguably|studies (show|suggest)|"
                   r"experts (say|agree|believe))\b",
    "audience-symmetry": r"\bwhether you'?re an? [^,.]{1,40} or an?\b",
    "imagine-opening": r"^(imagine|picture this)\b",
}
WORDS_RU = {
    "kantselyarit": r"(являет(ся|ются|ясь)|представляет собой|\bданн(ый|ая|ое|ые|ого|ой|ом|ых|ому|ую)\b|"
                    r"осуществл\w+|\bв рамках\b|на сегодняшний день|вышеуказанн\w+|\bпроизвод(ит|ят|ить) \w+ние)",
    "ru-key-role": r"(играет (ключевую|важную|решающую) роль|неотъемлем\w+|ключев(ой|ая|ое|ые) (роль|значение))",
    "ru-empty-intro": r"(стоит отметить|важно отметить|следует (отметить|подчеркнуть)|необходимо понимать|"
                      r"нельзя не отметить)",
    "ru-signposting": r"(давайте (разберёмся|разберемся|рассмотрим|погрузимся)|погрузимся в|разложим по полочкам|"
                      r"в заключение|подводя итог|резюмируя)",
    "ru-hype": r"(в современном мире|в эпоху \w+|новые горизонты|настоящий прорыв|уникальн\w+|инновацион\w+|"
               r"революцион\w+|широкий спектр|многогранн\w+)",
    "ru-reveal": r"(не просто [^.;]{1,60}, (а|это)|ответ прост|секрет в том|самое интересное|спойлер|"
                 r"это не [^.]{1,30}, это)",
    "ru-verbal-noun": r"\b(проведение|осуществление|выполнение|реализация) [а-яё]+(ия|ения|ания)\b",
}
TITLE_PATTERNS = {
    "colon-title": r":",
    "participle-title": r",\s*(explained|measured|revisited|demystified|decoded|unpacked|reimagined)\b",
    "slogan-title": r"\b(one [a-z]+, (two|three) [a-z]+|the (demo|magic|secret|truth) (of|behind)?)\b",
}


def prose_lines(path: Path):
    """(line number, text, kind) for the prose of a post or a lesson; kind is title, heading or body."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if path.suffix == ".py":  # jupytext: markdown cells only
        in_md = False
        for i, line in enumerate(lines, 1):
            if line.startswith("# %%"):
                in_md = "[markdown]" in line
                continue
            if in_md and line.startswith("#"):
                text = line[2:] if line.startswith("# ") else line[1:]
                if text.lstrip().startswith("<") or text.startswith("- ") and "](" in text and "http" in text:
                    continue
                kind = "heading" if re.match(r"#+ ", text) else "body"
                yield i, text, kind
        return
    in_front, in_code, refs = False, False, False
    for i, line in enumerate(lines, 1):
        if i == 1 and line.strip() == "---":
            in_front = True
            continue
        if in_front:
            if line.strip() == "---":
                in_front = False
            elif re.match(r"(title|description|seo_description):", line):
                yield i, line.split(":", 1)[1].strip().strip('"'), "title"
            continue
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or line.lstrip().startswith(("<", "{%", "|")):
            continue
        if re.match(r"#+ (references|bibliography|sources)", line, re.I):
            refs = True
        if refs:
            continue
        kind = "heading" if re.match(r"#+ ", line) else "body"
        yield i, line, kind


def lint(path: Path):
    flags = []
    paragraph, para_start = [], None
    in_steps = last_was_item = False

    def flush():
        text = " ".join(paragraph)
        sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ])", text) if s.strip()]
        if len(sentences) > 5:
            flags.append((para_start, "long-paragraph", f"{len(sentences)} sentences"))
        if text.count("—") > 1:
            flags.append((para_start, "em-dashes", f"{text.count('—')} em dashes in one paragraph"))

    for i, text, kind in prose_lines(path):
        low = text.lower()
        if kind in ("title", "heading"):
            plain = re.sub(r"^#+ ", "", text)
            for rule, rx in TITLE_PATTERNS.items():
                if re.search(rx, plain, re.I) and not (rule == "colon-title" and kind == "title" and len(plain) > 90):
                    flags.append((i, rule, plain))
        if kind == "body":
            stripped = text.strip()
            if not stripped:
                if paragraph:
                    flush()
                paragraph, para_start = [], None
                last_was_item = False
                continue
            is_item = bool(re.match(r"^(\d+\.|[-*]) ", stripped))
            continues_item = not is_item and last_was_item  # a wrapped line of the item above
            if "what you'll do" in low:
                in_steps = True  # the lesson's list of steps is the one list the guide allows
            elif is_item:
                if not in_steps and path.suffix == ".md":
                    flags.append((i, "list-in-body", stripped[:80]))
            elif not continues_item:
                in_steps = False
                if paragraph == []:
                    para_start = i
                paragraph.append(stripped)
            last_was_item = is_item or continues_item
            if re.search(r"(?<!^> )(?<!^)\*\*[^*]+\*\*", stripped) and not re.match(r"^>\s*\*\*", stripped):
                flags.append((i, "bold-in-sentence", re.search(r"\*\*[^*]+\*\*", stripped).group(0)))
            if "!" in re.sub(r"!\[|\]\(|https?://\S+", "", stripped):
                flags.append((i, "exclamation", stripped[:80]))
        for rules in (WORDS_EN, CONSTRUCTIONS_EN, WORDS_RU):
            for rule, rx in rules.items():
                for m in re.finditer(rx, text, re.I | re.M):
                    flags.append((i, rule, m.group(0)))
    if paragraph:
        flush()
    return sorted(flags)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--strict", action="store_true", help="exit with status 1 if anything is flagged")
    args = parser.parse_args()
    total = 0
    for path in args.files:
        flags = lint(path)
        total += len(flags)
        for line, rule, excerpt in flags:
            print(f"{path}:{line}: [{rule}] {excerpt}")
        print(f"{path}: {len(flags)} flags", file=sys.stderr)
    return 1 if args.strict and total else 0


if __name__ == "__main__":
    sys.exit(main())
