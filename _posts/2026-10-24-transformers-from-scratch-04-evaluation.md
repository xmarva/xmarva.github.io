---
layout: post
title: "Evaluating machine translation with chrF and COMET"
date: 2026-10-24 13:00:00
description: "Evaluate English–Russian translation with BLEU, chrF and COMET written from scratch: multiple references, agreement with professional translators, significance tests and a contaminated test set."
tags: [transformers-from-scratch, machine-translation, evaluation, bleu, comet]
categories: machine-learning
thumbnail: /assets/posts/tfs/04/cover.png
og_image: https://xmarva.github.io/assets/posts/tfs/04/cover.png
stylesheets: [/assets/css/tfs-course.css]
toc:
  beginning: true
---
{% raw %}

<p class="tfs-links"><strong>Transformers: from scratch to production</strong> · lesson 4 of 9 · <a href="/teaching/">all lessons</a> · <a href="https://www.kaggle.com/code/qmarva/transformers-from-scratch-04-bleu-chrf-comet">run on Kaggle</a> · <a href="https://colab.research.google.com/github/xmarva/transformers-from-scratch/blob/main/notebooks/colab/04_evaluation.ipynb">open in Colab</a> · <a href="https://github.com/xmarva/transformers-from-scratch">code on GitHub</a></p>

At WMT24, the main machine translation competition, Claude 3.5 Sonnet won 9 of the 11 language pairs. The verdict came from people who read the translations.

The automatic ranking had put a different system first in every pair, Unbabel's Tower 70B, and the organizers explained why: Tower chose among its candidate translations with COMET, the same metric the ranking used. A system had learned to please the judge.

For most of its history machine translation was judged by a much cruder metric. Of 769 MT papers published from
2010 to 2020, 98.8% reported BLEU and 74.3% reported nothing else
<sup class="cite">[<a href="#ref-1" id="cite-1" data-ref="1" aria-label="Reference 1: Marie, B., Fujita, A., Rubino, R. (2021). Scientific Credibility of Machine Translation Research: A Meta-Evaluation of 769 Papers. ACL.">1</a>]</sup>. In 2022 the organizers of WMT's metrics task
titled their report *Stop Using BLEU*. One sentence from our own test set shows the problem:

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/04/hero-light.png" width="760" alt="opus-mt-en-ru translates &#x27;Are you coming alone?&#x27; as Ты идёшь одна? The test set has 15 Russian translations of the sentence, each a test pair of its own. Scored against each of them, the same correct translation gets sentence BLEU between 16 and 35" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/04/hero-dark.png" width="760" alt="opus-mt-en-ru translates &#x27;Are you coming alone?&#x27; as Ты идёшь одна? The test set has 15 Russian translations of the sentence, each a test pair of its own. Scored against each of them, the same correct translation gets sentence BLEU between 16 and 35" loading="lazy">

The translation is correct. So are all fifteen references: Russian has to decide what English leaves open (ты or вы, a man or a woman, on foot or by car), and Tatoeba's volunteers wrote down every combination.

Our test set keeps each one as a separate pair. So this one translation is graded fifteen times, and BLEU calls it mediocre fifteen times over.

In this lesson you build BLEU, chrF and COMET from scratch, check each against its reference implementation, and then check the metrics themselves. Then they answer the questions lesson 3 left open: does beam search translate better than greedy decoding, and how far is our twenty-minute model from opus-mt-en-ru?

On the way we find out that the competitor has seen a quarter of the exam, and whether it helped.

**What you'll do**

1. Reproduce a published score of opus-mt-en-ru, and see why published scores are so often incomparable.
2. Write BLEU and chrF, and see what tokenization, the letter ё and multiple references do to them.
3. Take COMET apart: write its forward pass from the checkpoint and check it against the official package.
4. Test the metrics against professional translators and against translations broken on purpose.
5. Compare our model with opus-mt-en-ru with confidence intervals, find the test pairs opus-mt saw in training,
   and look at where each model fails.

**What you need**: a GPU. On Kaggle, *Settings → Accelerator → GPU*; on Colab, *Runtime → Change runtime type → T4 GPU*.

The notebook downloads the data and tokenizer of lesson 1, the model of lesson 3, opus-mt-en-ru and the COMET checkpoint (2.3 GB). On a T4 it runs in about twenty minutes.

## Setup

One small helper package, `tfs`, draws the course's plots and token chips and knows where the notebook runs
(Kaggle, Colab or your own machine). Everything the lesson teaches is written out in the notebook itself.

```python
import importlib.util
import subprocess
import sys

TFS = "git+https://github.com/xmarva/transformers-from-scratch@main"
if importlib.util.find_spec("tfs") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", TFS])

from tfs import env
from tfs.viz import table, token_chips, use_course_style

env.quiet()  # no progress bars or environment warnings in the outputs
SEED = 42
env.seed_everything(SEED)
COLORS = use_course_style()
WORK_DIR = env.work_dir()
```

```python
import io
import json
import math
import os
import re
import time
import unicodedata
import urllib.request
import zipfile
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file
from tfs import decoding
from tfs.model import Transformer
from tfs.viz import heatmap
from tokenizers import Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# sacreBLEU, the reference implementation of BLEU and chrF
if importlib.util.find_spec("sacrebleu") is None:
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    subprocess.check_call(pip + ["sacrebleu==2.6.0"])
import sacrebleu

SMOKE = env.smoke()  # CI runs the lesson on a small slice; you evaluate in full
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda" or SMOKE, "This lesson runs large models: switch to a GPU."
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
```

<pre class="tfs-printed" markdown="0"><code>GPU: Tesla T4</code></pre>

The test set of lesson 1, from the Hugging Face Hub:

```python
path = hf_hub_download("xmarva/tfs-tatoeba-en-ru", "data/test.parquet", repo_type="dataset")
test = pd.read_parquet(path)

HERO = "Are you coming alone?"
if SMOKE:  # 150 English sentences with all their translations, and the one from the picture
    chosen = set(test.en.drop_duplicates().sample(150, random_state=SEED)) | {HERO}
    test = test[test.en.isin(chosen)].reset_index(drop=True)
print("test pairs:", len(test), "distinct English sentences:", test.en.nunique())
```

<pre class="tfs-printed" markdown="0"><code>test pairs: 36385 distinct English sentences: 29243</code></pre>

## 1. A score from a model card

opus-mt-en-ru, the translator lesson 3 put next to ours, ends its model card on the Hub with a table of scores.
Read it straight from the card:

```python
card = hf_hub_download("Helsinki-NLP/opus-mt-en-ru", "README.md")
rows = []
with open(card, encoding="utf-8") as f:
    for line in f:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[1].replace(".", "").isdigit():  # a row of scores
            scores_here = {"BLEU": float(cells[1]), "chrF": float(cells[2])}
            rows.append({"test set": cells[0], **scores_here})
table(pd.DataFrame(rows), digits=3)
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">test set</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">BLEU</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">chrF</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">newstest2012.en.ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">31.100</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.581</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">newstest2013.en.ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">23.500</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.513</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">newstest2015-enru.en.ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">27.500</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.564</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">newstest2016-enru.en.ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">26.400</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.548</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">newstest2017-enru.en.ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">29.100</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.572</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">newstest2018-enru.en.ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">25.400</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.554</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">newstest2019-enru.en.ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">27.100</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.533</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Tatoeba.en.ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">48.400</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.669</td></tr></tbody></table></div>

Two metrics, eight test sets. The last row is the one closest to our data, and it is not our test set. The card
was written in February 2020, before the Tatoeba Translation Challenge
<sup class="cite">[<a href="#ref-2" id="cite-2" data-ref="2" aria-label="Reference 2: Tiedemann, J. (2020). The Tatoeba Translation Challenge – Realistic Data Sets for Low Resource and Multilingual MT. WMT.">2</a>]</sup> published the test set we use. Its *Tatoeba* is a
sample of 5,000 pairs that the model's training recipe set aside from the Tatoeba corpus.

The OPUS-MT team keeps a leaderboard that re-scores old models on new test sets. Here is the same model on three
releases of the challenge's English–Russian test set:

```python
LEADERBOARD = (
    "https://raw.githubusercontent.com/Helsinki-NLP/OPUS-MT-leaderboard/"
    "dcc11795b3653d36865af674ffa784808d721a12/models/OPUS-MT-models/en-ru/"
)
scores_file = urllib.request.urlopen(LEADERBOARD + "opus-2020-02-11.scores.txt")
rows = []
for line in scores_file.read().decode().splitlines():
    pair, test_set, chrf, bleu, release, sentences, words = line.split("\t")
    if test_set.startswith("tatoeba"):
        scores_here = {"BLEU": float(bleu), "chrF": float(chrf)}
        rows.append({"test set": test_set, "sentences": int(sentences), **scores_here})
published = pd.DataFrame(rows)
table(published, digits=3)
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">test set</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">sentences</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">BLEU</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">chrF</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">tatoeba-test-v2020-07-28</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">10,000</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">48.800</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.671</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">tatoeba-test-v2021-03-30</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">15,118</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">47.600</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.667</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">tatoeba-test-v2021-08-07</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">19,425</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">46.800</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.661</td></tr></tbody></table></div>

Same model, same weights, three numbers, all of them called Tatoeba. The test set grew with every release, and
ours, v2023-09-26, is the newest. It contains the v2021-08-07 release entirely, which you can check: download
that release, normalize it the way lesson 1 normalized ours, and look for every pair in our test set.

The normalization is lesson 1's, unchanged: Unicode NFKC, no stress marks, straight quotes, one kind of dash, one
space between words.

```python
QUOTES = {"«": '"', "»": '"', "„": '"', "“": '"', "”": '"', "‘": "'", "’": "'", "–": "—"}


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    kept = [char for char in text if unicodedata.category(char) != "Mn"]  # no stress marks
    text = "".join(kept)
    for fancy, plain in QUOTES.items():
        text = text.replace(fancy, plain)
    text = re.sub(r"\s+-\s+", " — ", text)  # " - " is a dash
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

```python
TESTSETS = (
    "https://raw.githubusercontent.com/Helsinki-NLP/OPUS-MT-testsets/"
    "ac0f19c0be3cb4e0d9450fcb2d7a9847c4bdd180/testsets/eng-rus/"
)
v2021_en = urllib.request.urlopen(TESTSETS + "tatoeba-test-v2021-08-07.eng").read().decode()
v2021_ru = urllib.request.urlopen(TESTSETS + "tatoeba-test-v2021-08-07.rus").read().decode()
v2021 = pd.DataFrame({"en": v2021_en.splitlines(), "ru": v2021_ru.splitlines()})

our_pairs = set(zip(test.en, test.ru))
found = 0
for en, ru in zip(v2021.en, v2021.ru):
    if (normalize(en), normalize(ru)) in our_pairs:
        found += 1
print("v2021-08-07 pairs:", len(v2021), "of them in our test set:", found)
```

<pre class="tfs-printed" markdown="0"><code>v2021-08-07 pairs: 19425 of them in our test set: 19425</code></pre>

Every pair is there. So we can reproduce the leaderboard's number ourselves, which is the first thing to do with
any metric before trusting it: translate those sentences with opus-mt-en-ru as it is published on the Hub, and
score them with sacreBLEU, the standard implementation of BLEU and chrF.

opus-mt-en-ru's own decoding settings (four beams, from lesson 3) are kept. Translating sorted by length puts
sentences of similar length into a batch, and a cache makes sure no sentence is translated twice: the same
English sentence appears in several test pairs.

```python
opus_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
opus = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru").to(device).eval()
opus_cache = {}


@torch.no_grad()
def translate_opus(sentences, batch_size=64):
    todo = [s for s in dict.fromkeys(sentences) if s not in opus_cache]
    todo.sort(key=len)  # similar lengths in one batch: little padding
    for start in range(0, len(todo), batch_size):
        chunk = todo[start : start + batch_size]
        batch = opus_tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        output = opus.generate(**batch)  # its generation_config: 4 beams
        texts = opus_tokenizer.batch_decode(output, skip_special_tokens=True)
        for sentence, text in zip(chunk, texts):
            opus_cache[sentence] = text
    return [opus_cache[s] for s in sentences]
```

```python
if SMOKE:
    v2021 = v2021.head(200)

start = time.perf_counter()
v2021_hyps = translate_opus(list(v2021.en))
print("translated", len(v2021), "sentences in", round(time.perf_counter() - start), "s")

bleu = sacrebleu.BLEU()
our_bleu = bleu.corpus_score(v2021_hyps, [list(v2021.ru)])
our_chrf = sacrebleu.corpus_chrf(v2021_hyps, [list(v2021.ru)])
row = published[published["test set"] == "tatoeba-test-v2021-08-07"].iloc[0]
print("leaderboard:  BLEU", row.BLEU, " chrF", row.chrF)
our_chrf_share = round(our_chrf.score / 100, 4)  # the leaderboard writes chrF from 0 to 1
print("we measure:   BLEU", round(our_bleu.score, 1), " chrF", our_chrf_share)
print("signature:   ", bleu.get_signature())
```

<pre class="tfs-printed" markdown="0"><code>translated 19425 sentences in 128 s
leaderboard:  BLEU 46.8  chrF 0.66066
we measure:   BLEU 46.8  chrF 0.6608
signature:    nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0</code></pre>

BLEU agrees with the leaderboard to the decimal, chrF to within a hair. Same test set, same weights, same metric and the same settings, which is what the signature line records: the numbers agree.

Change any one of the four and they stop agreeing. The card's 48.4 and the leaderboard's numbers describe the same weights on different test sets, and nothing on the card warns you that its Tatoeba isn't the Tatoeba everyone else uses.

> A score is comparable only with scores of the same test set, the same references and the same metric settings.

## 2. The plan

Now the real work: our model against opus-mt-en-ru on our whole test set.

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/04/plan-light.png" width="760" alt="The plan of the lesson: the test pairs are translated by three systems, our model with greedy decoding, our model with beam search and opus-mt-en-ru; the translations are scored by BLEU, chrF and COMET against the references; the metrics are checked against professional translators&#x27; judgments and against translations broken on purpose, and the differences between systems get confidence intervals from the bootstrap" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/04/plan-dark.png" width="760" alt="The plan of the lesson: the test pairs are translated by three systems, our model with greedy decoding, our model with beam search and opus-mt-en-ru; the translations are scored by BLEU, chrF and COMET against the references; the metrics are checked against professional translators&#x27; judgments and against translations broken on purpose, and the differences between systems get confidence intervals from the bootstrap" loading="lazy">

Three translators, three metrics. BLEU counts shared words, chrF shared characters, and COMET is a neural network
trained to predict what a human would say. The second half of the lesson checks the metrics themselves.

## 3. Translating the test set

Our model and tokenizer come from the Hub, where lessons 1 and 3 published them:

```python
tokenizer_path = hf_hub_download("xmarva/tfs-bpe-en-ru-16k", "tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)
MODEL_REPO = "xmarva/tfs-transformer-en-ru"
with open(hf_hub_download(MODEL_REPO, "config.json")) as f:
    CONFIG = json.load(f)
model = Transformer(**CONFIG)
model.load_state_dict(load_file(hf_hub_download(MODEL_REPO, "model.safetensors")))
model = model.to(device).eval()
print(sum(p.numel() for p in model.parameters()), "parameters")
```

<pre class="tfs-printed" markdown="0"><code>9625600 parameters</code></pre>

Lesson 3's beam search translated one sentence at a time, which would take the better part of an hour on the whole test set.

The course's `tfs.decoding` holds the same two procedures, greedy and beam search, step for step, with one change: at every step the live beams of many sentences go through the decoder together. A batch must not change a single translation. Check it on lesson 3's sentences, each translated alone and then all together:

```python
PROBES = [
    "The researcher said she was exhausted.",
    "The teacher said she was tired.",
    "My neighbour said he was busy.",
    "Tom forgot to lock the door.",
    "I can't believe you did that.",
    HERO,
]
alone = []
for sentence in PROBES:
    alone.append(decoding.beam_search(model, tokenizer, [sentence])[0])
together = decoding.beam_search(model, tokenizer, PROBES)
assert together == alone
table(pd.DataFrame({"English": PROBES, "ours, beam of 4": together}))
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">English</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">ours, beam of 4</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">The researcher said she was exhausted.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Исследователи сказали, что она измотанная.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">The teacher said she was tired.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Учитель сказал, что она устала.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">My neighbour said he was busy.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мой сосед сказал, что занят.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Tom forgot to lock the door.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Том забыл запереть дверь.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">I can&#x27;t believe you did that.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Поверить не могу, что ты это сделал.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Are you coming alone?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Ты придёшь один?</td></tr></tbody></table></div>

Now every distinct English sentence of the test set, three times. Each translator sees a sentence once, however
many references it has.

```python
sentences = list(dict.fromkeys(test.en))

start = time.perf_counter()
greedy = decoding.greedy(model, tokenizer, sentences)
print("ours, greedy:", round(time.perf_counter() - start), "s")

start = time.perf_counter()
beam = decoding.beam_search(model, tokenizer, sentences)
print("ours, beam of 4:", round(time.perf_counter() - start), "s")

start = time.perf_counter()
opus_raw = translate_opus(sentences)
print("opus-mt-en-ru:", round(time.perf_counter() - start), "s")
print(len(sentences), "sentences each")
```

<pre class="tfs-printed" markdown="0"><code>ours, greedy: 41 s
ours, beam of 4: 72 s
opus-mt-en-ru: 120 s
29243 sentences each</code></pre>

One step before scoring. Lesson 1 normalized our references: straight quotes instead of «ёлочки», one kind of dash.

Our model learned from normalized text and writes the same way. opus-mt-en-ru learned from raw text and writes «ёлочки», which a metric would count as wrong characters against our references. A fair comparison passes every translation through the normalization the references went through:

```python
opus_normalized = [normalize(text) for text in opus_raw]
changed = []
for raw, clean in zip(opus_raw, opus_normalized):
    if raw != clean:
        changed.append({"opus-mt-en-ru writes": raw, "normalized": clean})
print("translations changed by the normalization:", len(changed))
table(pd.DataFrame(changed).head(4))
```

<pre class="tfs-printed" markdown="0"><code>translations changed by the normalization: 918</code></pre>

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">opus-mt-en-ru writes</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">normalized</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">2013 год - это год, который я никогда не забуду.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">2013 год — это год, который я никогда не забуду.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">2013 год - это год, который я никогда не забуду.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">2013 год — это год, который я никогда не забуду.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Пчела - это не птица.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Пчела — это не птица.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">По словам Юджина Вигнера, лауреата Нобелевской премии в области физики, сознательное осознание людьми может действительно «проникать» — каким-то необъяснимым образом — и изменить физическое состояние системы.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">По словам Юджина Вигнера, лауреата Нобелевской премии в области физики, сознательное осознание людьми может действительно &quot;проникать&quot; — каким-то необъяснимым образом — и изменить физическое состояние системы.</td></tr></tbody></table></div>

```python
SYSTEMS = ["ours, greedy", "ours, beam", "opus-mt-en-ru"]
outputs = {
    "ours, greedy": dict(zip(sentences, greedy)),
    "ours, beam": dict(zip(sentences, beam)),
    "opus-mt-en-ru": dict(zip(sentences, opus_normalized)),
}
for system in SYSTEMS:
    test[system] = test.en.map(outputs[system])  # a translation for every test pair
table(test.sample(6, random_state=SEED)[["en", "ru", *SYSTEMS]])
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">en</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">ru</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">ours, greedy</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">ours, beam</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">opus-mt-en-ru</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">The novel has been translated into German.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Роман переведён на немецкий язык.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Роман был переведён на немецкий роман.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Роман был переведён на немецкий роман.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Роман был переведен на немецкий язык.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">I wonder why you never told me about it.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Интересно, почему ты никогда мне об этом не рассказывал.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Интересно, почему ты никогда мне об этом не рассказывал.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Интересно, почему ты никогда мне об этом не рассказывал.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Интересно, почему ты никогда не говорил мне об этом.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Arguing is no way to go about it.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Спорить здесь не вариант.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">И продолжаться не сподвигается.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Спасание — не способа пойти об этом.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Спорить — это не способ.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Please tell me I&#x27;m right.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Скажи мне, пожалуйста, что я прав.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Пожалуйста, скажи мне, что я прав.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Пожалуйста, скажи мне, что я прав.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Пожалуйста, скажи мне, что я прав.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">I watch TV all the time.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Я всё время смотрю телевизор.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Я постоянно смотрю телевизор.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Я постоянно смотрю телевизор.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Я все время смотрю телевизор.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Tell Tom that I may be late.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Скажи Тому, что я могу опоздать.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Скажи Тому, что я, возможно, опоздаю.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Скажи Тому, что я, возможно, опоздаю.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Скажи Тому, что я, возможно, опоздаю.</td></tr></tbody></table></div>

## 4. BLEU

**BLEU**<sup class="cite">[<a href="#ref-3" id="cite-3" data-ref="3" aria-label="Reference 3: Papineni, K., Roukos, S., Ward, T., Zhu, W.-J. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation. ACL.">3</a>]</sup> asks a blunt question: how many of the
translation's word sequences also occur in the reference? It counts **n-grams**, runs of n words, for n from 1
to 4. Shared single words reward the right vocabulary; shared runs of four reward the right order as well.

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/04/bleu-ngrams-light.png" width="760" alt="BLEU on one sentence: our model translates Tell Tom that I may be late. as Скажи Тому, что я, возможно, опоздаю., the reference says может быть instead of возможно; the single different word breaks n-grams of every length, 9 of 10 single words match but only 3 of 7 four-word runs, and with the brevity penalty BLEU is 60 for a correct translation" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/04/bleu-ngrams-dark.png" width="760" alt="BLEU on one sentence: our model translates Tell Tom that I may be late. as Скажи Тому, что я, возможно, опоздаю., the reference says может быть instead of возможно; the single different word breaks n-grams of every length, 9 of 10 single words match but only 3 of 7 four-word runs, and with the brevity penalty BLEU is 60 for a correct translation" loading="lazy">

Two details keep the counting honest. A word the translation repeats is credited at most as many times as the reference has it; that's **clipping**, and it is why printing «Том» ten times doesn't score.

And precision alone would reward a translation that says only what it is sure of: «Том.» has every word in the reference. So BLEU also multiplies by a **brevity penalty** when the translations are shorter than the references.

$$p_n = \frac{\sum_{s} \sum_{g \in G_n(h_s)} \min\big(c_{h_s}(g),\, c_{r_s}(g)\big)}{\sum_{s} \sum_{g \in
G_n(h_s)} c_{h_s}(g)}, \qquad \mathrm{BP} = \begin{cases} 1 & c > r \\ e^{\,1 - r/c} & c \le r \end{cases},
\qquad \mathrm{BLEU} = \mathrm{BP} \cdot \exp\Big(\frac{1}{4} \sum_{n=1}^{4} \log p_n\Big).$$

$$G_n(h_s)$$ is the set of distinct n-grams in the translation $$h_s$$ of test sentence $$s$$. $$c_{h_s}(g)$$ counts how
often n-gram $$g$$ occurs in the translation and $$c_{r_s}(g)$$ how often in the reference, so the minimum is the
clipping. $$p_n$$ is the share of the translation's n-grams that the reference confirms, and both sums run over the
whole test set before dividing.

$$c$$ and $$r$$ are the total lengths, in words, of all translations and all references. The four precisions are
combined by a geometric mean, which is unforgiving: a zero at any $$n$$ zeroes the whole score. Reported BLEU is
this number times 100.

Notice what BLEU is: a property of a whole test set. Papineni et al. write that it "only needs to match human
judgment when averaged over a test corpus; scores on individual sentences will often vary from human judgments".
The picture above, one sentence at a time, is for understanding, not for judging.

Words first. Where does «забыл,» end and the comma begin? sacreBLEU's default tokenizer, called **13a**, splits
punctuation off the words. We borrow it, so that our BLEU and sacreBLEU's see the same words and can be compared
exactly:

```python
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

tokenize_13a = Tokenizer13a()
print(tokenize_13a("Том забыл, чтобы я забыл дверь.").split())
```

<pre class="tfs-printed" markdown="0"><code>[&#x27;Том&#x27;, &#x27;забыл&#x27;, &#x27;,&#x27;, &#x27;чтобы&#x27;, &#x27;я&#x27;, &#x27;забыл&#x27;, &#x27;дверь&#x27;, &#x27;.&#x27;]</code></pre>

The counting itself. `bleu_stats` returns everything BLEU needs from one sentence: matched and total n-grams for
every $$n$$, and the two lengths. It already accepts several references, which section 6 needs: an n-gram is
credited up to its largest count in any one reference, and the reference length is the one closest to the
translation's.

```python
def ngram_counts(items, n):
    counts = Counter()
    for i in range(len(items) - n + 1):
        counts[tuple(items[i : i + n])] += 1
    return counts


def bleu_stats(hypothesis, references, max_n=4):
    words = tokenize_13a(hypothesis).split()
    reference_words = [tokenize_13a(r).split() for r in references]
    matched, total = [], []
    for n in range(1, max_n + 1):
        most = Counter()  # the largest count of each n-gram in any one reference
        for ref in reference_words:
            for gram, count in ngram_counts(ref, n).items():
                most[gram] = max(most[gram], count)
        clipped = 0
        for gram, count in ngram_counts(words, n).items():
            clipped += min(count, most[gram])
        matched.append(clipped)
        total.append(max(len(words) - n + 1, 0))
    lengths = [len(ref) for ref in reference_words]
    # the closest reference length; on a tie, the shorter one
    closest = min(lengths, key=lambda length: (abs(length - len(words)), length))
    return matched + total + [len(words), closest]
```

And the formula, applied to the sums over a test set:

```python
def bleu_from_stats(stats, max_n=4):
    stats = np.asarray(stats, dtype=float)
    if stats.ndim == 2:
        stats = stats.sum(axis=0)  # one row per sentence: sum over the test set
    matched, total = stats[:max_n], stats[max_n : 2 * max_n]
    c, r = stats[2 * max_n], stats[2 * max_n + 1]
    if matched.min() == 0:
        return 0.0
    log_precision = np.log(matched / total).mean()
    brevity = 1.0 if c > r else math.exp(1 - r / c)
    return 100 * brevity * math.exp(log_precision)
```

A test set of three sentences, small enough to check by hand. The first is what our model wrote in lesson 3 after two epochs of training; its second «забыл» is clipped.

The second differs from its reference in one letter. The third is the sentence from the picture.

```python
tiny = [
    ("Том забыл, чтобы я забыл дверь.", "Том забыл запереть дверь."),
    ("Ты еще слишком молода.", "Ты ещё слишком молода."),
    ("Скажи Тому, что я, возможно, опоздаю.", "Скажи Тому, что я, может быть, опоздаю."),
]
tiny_stats = []
for hypothesis, reference in tiny:
    stats = bleu_stats(hypothesis, [reference])
    tiny_stats.append(stats)
    print(hypothesis, "| matched", stats[:4], "of", stats[4:8], "| lengths", stats[8:])

ours = bleu_from_stats(tiny_stats)
theirs = sacrebleu.corpus_bleu([h for h, _ in tiny], [[r for _, r in tiny]]).score
assert abs(ours - theirs) < 1e-9
print("BLEU of the three:", round(ours, 2), "= sacreBLEU's", round(theirs, 2))
```

<pre class="tfs-printed" markdown="0"><code>Том забыл, чтобы я забыл дверь. | matched [4, 2, 0, 0] of [8, 7, 6, 5] | lengths [8, 5]
Ты еще слишком молода. | matched [4, 2, 1, 0] of [5, 4, 3, 2] | lengths [5, 5]
Скажи Тому, что я, возможно, опоздаю. | matched [9, 7, 5, 3] of [10, 9, 8, 7] | lengths [10, 11]
BLEU of the three: 41.87 = sacreBLEU&#x27;s 41.87</code></pre>

The clipped «забыл» counts once. The missing ё costs the second sentence a whole word and every n-gram around
it, and the synonym does the same to the third. Two of the three translations are correct, and the test set as a
whole gets the score above.

Now the test set. Every sentence's statistics are computed once and kept: section 10 resamples them a thousand
times.

```python
bleu_rows = {}
scores = []
for system in SYSTEMS:
    stats = []
    for hypothesis, reference in zip(test[system], test.ru):
        stats.append(bleu_stats(hypothesis, [reference]))
    bleu_rows[system] = np.array(stats)
    ours = bleu_from_stats(bleu_rows[system])
    theirs = sacrebleu.corpus_bleu(list(test[system]), [list(test.ru)]).score
    assert abs(ours - theirs) < 1e-6
    scores.append({"system": system, "BLEU": ours})
table(pd.DataFrame(scores), digits=1)
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">system</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">BLEU</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ours, greedy</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">39.6</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ours, beam</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">40.8</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">opus-mt-en-ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">45.4</td></tr></tbody></table></div>

Our functions and sacreBLEU agree to the last digit on every system. opus-mt-en-ru leads by several points, and
beam search buys about a point over greedy decoding. Whether a point is a lot is section 10's question. First,
the ways these numbers move while the translations stay exactly the same.

Where does each system sit on the brevity penalty? The ratio $$c/r$$ of the total lengths, on the penalty's curve:

```python
ratios = np.linspace(0.9, 1.06, 200)  # around 1, where the three systems are
penalty = np.where(ratios > 1, 1.0, np.exp(1 - 1 / ratios))

fig, ax = plt.subplots(figsize=(6.4, 3))
ax.plot(ratios, penalty, color=COLORS["muted"])
for system, color in zip(SYSTEMS, [COLORS["enc"], COLORS["dec"], COLORS["data"]]):
    totals = bleu_rows[system].sum(axis=0)
    ratio = totals[8] / totals[9]  # translation words / reference words
    brevity = 1.0 if ratio > 1 else math.exp(1 - 1 / ratio)
    ax.plot(ratio, brevity, "o", color=color, label=f"{system}: c/r = {ratio:.3f}")
ax.set_xlabel("c / r, total length of the translations over the references")
ax.set_ylabel("brevity penalty")
ax.set_title("The brevity penalty")
ax.legend(fontsize=9)
plt.show()
```

<img class="tfs-output" src="/assets/posts/tfs/04/output-01.png" alt="The brevity penalty" loading="lazy">

Greedy decoding writes a hair more words than the references and pays nothing. Beam search is the one that
comes out short and pays: lesson 3 warned that beam search prefers short translations, and its length penalty
makes up for that only partly. Part of beam search's gain in precision goes back in the brevity penalty.

### Tokenization, and why a BLEU score needs a signature

The 13a tokenizer goes back to `mteval-v13a.pl`, the Perl script WMT scored with for years. Its rules split ASCII punctuation and nothing else; a comment in sacreBLEU's source says it was written "assuming Western languages".

Russian «ёлочки» and the ellipsis «…» stay glued to their words. The `intl` tokenizer splits all Unicode punctuation. Compare the two on one of opus-mt-en-ru's raw translations:

```python
glued = None
for text in opus_raw:
    if "«" in text and len(text) < 60:
        glued = text
        break
if glued is None:  # the smoke slice may have no such sentence
    glued = "Он сказал: «Я не знаю…»"

tokenize_intl = sacrebleu.BLEU(tokenize="intl").tokenizer
rows = [tokenize_13a(glued).split(), tokenize_intl(glued).split()]
token_chips(rows, labels=["13a", "intl"], mark_spaces=False)
plt.show()
```

<img class="tfs-output" src="/assets/posts/tfs/04/output-02.png" alt="Output of the code above" loading="lazy">

Different words, different n-grams, different BLEU. Here is BLEU of the same three systems, same translations,
same references, under the tokenizers sacreBLEU offers and with or without lowercasing:

```python
rows = []
for tokenize in ["13a", "intl", "char", "none"]:
    for lowercase in [False, True]:
        metric = sacrebleu.BLEU(tokenize=tokenize, lowercase=lowercase)
        row = {"tokenizer": tokenize, "lowercase": lowercase}
        for system in SYSTEMS:
            row[system] = metric.corpus_score(list(test[system]), [list(test.ru)]).score
        rows.append(row)
table(pd.DataFrame(rows), digits=1)
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">tokenizer</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">lowercase</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">ours, greedy</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">ours, beam</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">opus-mt-en-ru</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">13a</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">False</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">39.6</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">40.8</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">45.4</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">13a</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">True</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">40.3</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">41.5</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">46.2</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">intl</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">False</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">39.8</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">41.1</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">45.6</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">intl</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">True</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">40.5</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">41.8</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">46.4</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">char</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">False</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">63.8</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">64.2</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">70.2</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">char</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">True</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">64.3</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">64.8</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">70.8</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">none</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">False</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">33.6</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">35.0</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">39.1</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">none</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">True</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">34.2</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">35.6</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">39.8</td></tr></tbody></table></div>

Without tokenization (`none`), «забыл,» and «забыл» are different words and every score drops by several points. Counting characters as words (`char`) inflates them by more than twenty.

Lowercasing adds a little; `intl` instead of `13a` adds less for our normalized text, because lesson 1 already turned «ёлочки» into straight quotes, which 13a can split. The ranking of the three systems never changes. The numbers do, by more than the gap between beam search and greedy.

That is why [Post (2018)](https://aclanthology.org/W18-6319/) wrote sacreBLEU and asked everyone to report its
**signature** with every score. He measured spreads of up to 1.8 BLEU between configurations in common use, and
1.3 for English–Russian, on the same translations. A score without its signature can't be compared with
anything:

```python
metric = sacrebleu.BLEU()
metric.corpus_score(list(test["ours, beam"]), [list(test.ru)])  # a signature needs a score
print(metric.get_signature())
```

<pre class="tfs-printed" markdown="0"><code>nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0</code></pre>

`nrefs:1`: one reference per sentence. `case:mixed`: case matters. `eff:no`: no "effective order" trick for sentences too short to have 4-grams.

`tok:13a`: the tokenizer. `smooth:exp`: what to do with a zero count. `version`: the package version, because implementations change too.

Even with the signature, the comparison breaks if the texts differ. Our references were normalized in lesson 1,
so we normalized opus-mt-en-ru's output too. Without it, its score drops for nothing but the shape of its quote
marks:

```python
raw_by_pair = test.en.map(dict(zip(sentences, opus_raw)))
raw_bleu = sacrebleu.corpus_bleu(list(raw_by_pair), [list(test.ru)]).score
print("opus-mt-en-ru, raw:       ", round(raw_bleu, 2))
print("opus-mt-en-ru, normalized:", round(bleu_from_stats(bleu_rows["opus-mt-en-ru"]), 2))
```

<pre class="tfs-printed" markdown="0"><code>opus-mt-en-ru, raw:        44.47
opus-mt-en-ru, normalized: 45.39</code></pre>

### The letter ё

Russian print usually writes ё as е: «все» for «всё», «еще» for «ещё». Both are correct spelling; ё is obligatory only where the word would be ambiguous.

Tatoeba's volunteers often write it. How often, and how often do the translators?

```python
def has_yo(text):
    return "ё" in text or "Ё" in text


def without_yo(text):
    return text.replace("ё", "е").replace("Ё", "Е")


print(f"references with ё: {test.ru.map(has_yo).mean():.1%}")
rows = []
for system in SYSTEMS:
    plain = [without_yo(text) for text in test[system]]
    references = [without_yo(text) for text in test.ru]
    rows.append(
        {
            "system": system,
            "writes ё": f"{test[system].map(has_yo).mean():.1%}",
            "BLEU": bleu_from_stats(bleu_rows[system]),
            "BLEU, ё as е": sacrebleu.corpus_bleu(plain, [references]).score,
        }
    )
table(pd.DataFrame(rows), digits=3)
```

<pre class="tfs-printed" markdown="0"><code>references with ё: 12.5%</code></pre>

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">system</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">writes ё</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">BLEU</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">BLEU, ё as е</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ours, greedy</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">13.3%</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">39.648</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">39.864</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ours, beam</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">13.0%</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">40.840</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">41.053</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">opus-mt-en-ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">3.8%</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">45.386</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">46.799</td></tr></tbody></table></div>

Our model learned from Tatoeba and writes ё about as often as the references do. opus-mt-en-ru learned from all of OPUS and mostly doesn't.

With ё and е made the same, opus-mt-en-ru gains more than a point of BLEU and our model almost nothing: a share of the gap between them is typography, and it hides how far ahead opus-mt-en-ru really is. One letter, used the way Russian print uses it, and BLEU calls it a mistranslation.

## 5. chrF

Russian builds words from parts: a stem, then suffixes and endings for case, number, gender. Lesson 1 counted what this does to a vocabulary.

For BLEU it means that a translation with the right word in the wrong form scores as if the word were missing: «сказал» is not «сказала», «Тому» is not «Том». A metric for Russian should give partial credit for a partly right word.

**chrF**<sup class="cite">[<a href="#ref-4" id="cite-4" data-ref="4" aria-label="Reference 4: Popović, M. (2015). chrF: character n-gram F-score for automatic MT evaluation. WMT.">4</a>]</sup> does it by counting characters instead of words.
It takes the character n-grams of the translation and the reference, for n from 1 to 6, with the spaces removed,
and computes how many are shared:

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/04/chrf-word-light.png" width="760" alt="chrF on the word сказал against the reference сказала: for n from 1 to 6, every character n-gram of the translation is found in the reference, so precision is 1; the reference has one n-gram more at every n, the ones with its last letter, so recall falls as n grows. Averaged, precision 1.00 and recall 0.73 give chrF 78, while a word-level metric sees one wrong word" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/04/chrf-word-dark.png" width="760" alt="chrF on the word сказал against the reference сказала: for n from 1 to 6, every character n-gram of the translation is found in the reference, so precision is 1; the reference has one n-gram more at every n, the ones with its last letter, so recall falls as n grows. Averaged, precision 1.00 and recall 0.73 give chrF 78, while a word-level metric sees one wrong word" loading="lazy">

Unlike BLEU, chrF also measures **recall**: the share of the reference's n-grams that the translation
reproduces. A translation that leaves things out loses recall, so chrF needs no brevity penalty.

$$\mathrm{chrP} = \frac{1}{N} \sum_{n=1}^{N} \frac{m_n}{h_n}, \qquad \mathrm{chrR} = \frac{1}{N} \sum_{n=1}^{N}
\frac{m_n}{r_n}, \qquad \mathrm{chrF}_\beta = (1 + \beta^2)\, \frac{\mathrm{chrP} \cdot
\mathrm{chrR}}{\beta^2\, \mathrm{chrP} + \mathrm{chrR}}.$$

$$h_n$$ and $$r_n$$ count the character n-grams of the translations and the references, $$m_n$$ the shared ones
(clipped, as in BLEU), each summed over the test set. Precision and recall are averaged over $$N = 6$$ orders and
combined into an F-score in which recall counts $$\beta$$ times as much as precision.

sacreBLEU uses $$\beta = 2$$, the value [Popović (2016)](https://aclanthology.org/W16-2341/) found to agree with
human judgments best. An order with no n-grams at all, say 6-grams of a three-letter answer, is left out of the
averages.

The code follows the formula. For several references, sacreBLEU's chrF keeps, per sentence, the reference that
gives the best score; `chrf_stats` does the same:

```python
def char_ngram_counts(text, max_n):
    chars = "".join(text.split())  # spaces removed
    return [ngram_counts(chars, n) for n in range(1, max_n + 1)]


def chrf_from_stats(stats, max_n=6, beta=2):
    stats = np.asarray(stats, dtype=float)
    if stats.ndim == 2:
        stats = stats.sum(axis=0)  # one row per sentence: sum over the test set
    precisions, recalls = [], []
    for n in range(max_n):
        h, r, m = stats[3 * n : 3 * n + 3]
        if h > 0 and r > 0:  # orders without n-grams are left out
            precisions.append(m / h)
            recalls.append(m / r)
    if not precisions:
        return 0.0
    p, r = np.mean(precisions), np.mean(recalls)
    if p + r == 0:
        return 0.0
    return 100 * (1 + beta**2) * p * r / (beta**2 * p + r)
```

```python
def chrf_stats(hypothesis, references, max_n=6):
    hyp_counts = char_ngram_counts(hypothesis, max_n)
    best, best_score = None, -1.0
    for reference in references:
        ref_counts = char_ngram_counts(reference, max_n)
        stats = []
        for n in range(max_n):
            shared = 0
            for gram, count in hyp_counts[n].items():
                shared += min(count, ref_counts[n][gram])
            ref_total = sum(ref_counts[n].values())
            # an order the reference is too short to have isn't held against the translation
            hyp_total = sum(hyp_counts[n].values()) if ref_total > 0 else 0
            stats += [hyp_total, ref_total, shared]
        score = chrf_from_stats(stats, max_n)
        if score > best_score:  # keep the reference that suits this translation best
            best, best_score = stats, score
    return best
```

The word from the picture, then lesson 3's sentence about the researcher, where two Russian words lose their
feminine endings:

```python
pairs = [
    ("сказал", "сказала"),
    ("Исследователь сказал, что она устала.", "Исследовательница сказала, что она устала."),
]
for hypothesis, reference in pairs:
    ours = chrf_from_stats(chrf_stats(hypothesis, [reference]))
    theirs = sacrebleu.sentence_chrf(hypothesis, [reference]).score
    assert abs(ours - theirs) < 1e-9
    bleu = sacrebleu.sentence_bleu(hypothesis, [reference]).score
    print(f"{hypothesis} | chrF {ours:.1f}, sentence BLEU {bleu:.1f}")
```

<pre class="tfs-printed" markdown="0"><code>сказал | chrF 77.6, sentence BLEU 0.0
Исследователь сказал, что она устала. | chrF 73.4, sentence BLEU 61.5</code></pre>

For the single word, chrF gives most of the credit and BLEU none at all. In the researcher sentence, the mistake lesson 3's models really made, both metrics are generous and chrF more so: the two wrong words differ from the right ones only in their endings.

That is chrF's weakness as much as its strength. Those endings are the whole difference between a woman talking about herself and a man talking about somebody else.

And the test set, checked against sacreBLEU again:

```python
chrf_rows = {}
for row in scores:
    system = row["system"]
    stats = []
    for hypothesis, reference in zip(test[system], test.ru):
        stats.append(chrf_stats(hypothesis, [reference]))
    chrf_rows[system] = np.array(stats)
    row["chrF"] = chrf_from_stats(chrf_rows[system])
    theirs = sacrebleu.corpus_chrf(list(test[system]), [list(test.ru)]).score
    assert abs(row["chrF"] - theirs) < 1e-6
    # chrF++: words and word pairs added to the characters (below)
    chrf_plus = sacrebleu.CHRF(word_order=2)
    row["chrF++"] = chrf_plus.corpus_score(list(test[system]), [list(test.ru)]).score
table(pd.DataFrame(scores), digits=1)
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">system</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">BLEU</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">chrF</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">chrF++</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ours, greedy</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">39.6</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">58.4</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">57.9</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ours, beam</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">40.8</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">59.3</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">58.7</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">opus-mt-en-ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">45.4</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">65.1</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">64.2</td></tr></tbody></table></div>

Exact agreement again. chrF puts the systems in the same order as BLEU, with larger numbers: in characters,
every translation shares more with its reference than in whole words.

The last column is **chrF++**<sup class="cite">[<a href="#ref-5" id="cite-5" data-ref="5" aria-label="Reference 5: Popović, 2017. aclanthology.org.">5</a>]</sup>, `word_order=2` in
sacreBLEU: chrF with single words and word pairs counted alongside the characters. It changes little here. In
the paper, adding single words helped for Russian but adding word pairs didn't, and Popović suggested Russian's
rich morphology and free word order as the reason.

## 6. Several references per sentence

Back to the picture at the top. Tatoeba keeps every translation volunteers have added, so many English sentences
in our test set come with several references. How many?

```python
# all the references of each English sentence
references = test.groupby("en", sort=False).ru.apply(list)
counts = references.apply(len).value_counts().sort_index()

fig, ax = plt.subplots(figsize=(6.4, 2.8))
ax.bar(counts.index, counts.values, color=COLORS["data"])
ax.set_yscale("log")  # the many sentences with one reference next to the few with fifteen
ax.set_xlabel("references per English sentence")
ax.set_ylabel("sentences")
ax.set_title("Most sentences have one reference; some have many")
plt.show()

several = (references.apply(len) > 1).sum()
print("sentences with several references:", several, "of", len(references))
```

<img class="tfs-output" src="/assets/posts/tfs/04/output-03.png" alt="Most sentences have one reference; some have many" loading="lazy">

<pre class="tfs-printed" markdown="0"><code>sentences with several references: 4934 of 29243</code></pre>

The record holders differ in exactly what English leaves open:

```python
most = references.apply(len).idxmax()  # the first sentence with the most references
print(most)
rows = []
for k in range(0, len(references[most]), 3):
    rows.append(references[most][k : k + 3])
token_chips(rows, role="dec", mark_spaces=False)
plt.show()
```

<pre class="tfs-printed" markdown="0"><code>Are you coming alone?</code></pre>

<img class="tfs-output" src="/assets/posts/tfs/04/output-04.png" alt="Output of the code above" loading="lazy">

Scoring each pair separately, as the test set comes, a system is graded against one reference per pair, and an
English sentence with fifteen references is translated once but graded fifteen times. The Tatoeba Challenge
knows this: its README says the data "implicitly creates multi-reference test sets but with the wrong
normalization".

The fix is to score each distinct English sentence once, against all of its references. Both metrics already allow it: BLEU credits an n-gram up to its largest count in any one reference and takes the closest reference length; chrF takes the best-matching reference.

Our functions take a list of references; sacreBLEU takes "reference streams", the k-th reference of every sentence in stream k, with `None` where a sentence has fewer.

```python
streams = []
for k in range(references.apply(len).max()):
    stream = []
    for refs in references:
        stream.append(refs[k] if k < len(refs) else None)
    streams.append(stream)

rows = []
for system in SYSTEMS:
    hypotheses = [outputs[system][sentence] for sentence in references.index]
    multi_bleu = [bleu_stats(h, refs) for h, refs in zip(hypotheses, references)]
    multi_chrf = [chrf_stats(h, refs) for h, refs in zip(hypotheses, references)]
    bleu_all = bleu_from_stats(multi_bleu)
    chrf_all = chrf_from_stats(multi_chrf)
    assert abs(bleu_all - sacrebleu.corpus_bleu(hypotheses, streams).score) < 1e-6
    assert abs(chrf_all - sacrebleu.corpus_chrf(hypotheses, streams).score) < 1e-6
    rows.append(
        {
            "system": system,
            "BLEU, first reference": sacrebleu.corpus_bleu(hypotheses, streams[:1]).score,
            "BLEU, all references": bleu_all,
            "chrF, first reference": sacrebleu.corpus_chrf(hypotheses, streams[:1]).score,
            "chrF, all references": chrf_all,
        }
    )
table(pd.DataFrame(rows), digits=1)
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">system</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">BLEU, first reference</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">BLEU, all references</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">chrF, first reference</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">chrF, all references</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ours, greedy</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">39.4</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">42.6</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">58.0</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">60.0</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ours, beam</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">40.7</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">43.8</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">59.0</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">60.9</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">opus-mt-en-ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">46.4</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">49.4</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">65.8</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">67.8</td></tr></tbody></table></div>

Our functions agree with sacreBLEU on several references too. With all references, every system gains a few points of BLEU and chrF, because a translation now only has to match one of the ways to say it.

The ranking stays the same and so do the gaps: on this test set the extra references are generous to everybody. That's the good news.

The bad news is the picture at the top: a sentence-level score against one reference can be badly wrong, and averaging over a large test set is what keeps the corpus score sane.

## 7. COMET

BLEU and chrF compare strings. Neither knows that «возможно» and «может быть» mean the same thing, or that
dropping «не» reverses a sentence while costing two letters. **COMET**<sup class="cite">[<a href="#ref-6" id="cite-6" data-ref="6" aria-label="Reference 6: Rei, R., Stewart, C., Farinha, A. C., Lavie, A. (2020). COMET: A Neural Framework for MT Evaluation. EMNLP.">6</a>]</sup> gives up on counting. It is a neural network trained to
predict the score a human annotator would give a translation, from the source sentence, the translation and the
reference together.

We use `Unbabel/wmt22-comet-da`, the reference-based COMET model Unbabel released in 2022. Here is all of it:

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/04/comet-light.png" width="760" alt="COMET: the source sentence, the translation and the reference are each read by the same XLM-R large encoder; a weighted mix of its 25 layers is averaged over the tokens into one vector of 1024 numbers per sentence, s, h and r; six combinations of them, h, r, the products h times r and h times s and the distances between them, make 6144 features, and a small feed-forward network turns them into one score. It was trained to predict human scores of translations" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/04/comet-dark.png" width="760" alt="COMET: the source sentence, the translation and the reference are each read by the same XLM-R large encoder; a weighted mix of its 25 layers is averaged over the tokens into one vector of 1024 numbers per sentence, s, h and r; six combinations of them, h, r, the products h times r and h times s and the distances between them, make 6144 features, and a small feed-forward network turns them into one score. It was trained to predict human scores of translations" loading="lazy">

The encoder is **XLM-R large**<sup class="cite">[<a href="#ref-7" id="cite-7" data-ref="7" aria-label="Reference 7: Conneau, A. et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. ACL. XLM-R.">7</a>]</sup>, a Transformer encoder like the one in lesson 2, 24 layers deep, pretrained on text in a hundred languages. It reads each of the three sentences separately and turns each into a vector.

The small network on top looks at the translation's vector $$h$$ next to the reference's $$r$$ and the source's $$s$$: their products and their distances measure, dimension by dimension, how close the translation's meaning is to each.

It was trained on the **direct assessments** (DA) of WMT 2017 to 2020: translations that annotators scored on a
scale from 0 to 100. COMET learned to predict those scores, so it learned whatever those annotators valued. How
well it agrees with careful professional judgment is a question section 8 puts to it.

The official package is `unbabel-comet`. We won't install it. Its current release pins numpy below 2.0, which would break this notebook's environment, and in a fresh environment of its own it failed twice before scoring a sentence: at import, for want of a module that recent `setuptools` no longer ships, and then in its data loader, with current PyTorch.

Writing COMET's forward pass ourselves is shorter than fixing that, and it shows what the metric is. Start with its settings, a small file on the Hub:

```python
COMET_REPO = "Unbabel/wmt22-comet-da"
with open(hf_hub_download(COMET_REPO, "hparams.yaml")) as f:
    print(f.read())
```

<pre class="tfs-printed" markdown="0"><code>activations: Tanh
batch_size: 4
class_identifier: regression_metric
dropout: 0.1
encoder_learning_rate: 1.0e-06
encoder_model: XLM-RoBERTa
final_activation: null
hidden_sizes:
- 3072
- 1024
keep_embeddings_frozen: true
layer: mix
layer_norm: false
layer_transformation: sparsemax
layerwise_decay: 0.95
learning_rate: 1.5e-05
loss: mse
nr_frozen_epochs: 0.3
optimizer: AdamW
pool: avg
pretrained_model: xlm-roberta-large
train_data:
- data/1720-da.csv
validation_data:
- data/wmt-ende-newstest2021.csv
- data/wmt-enru-newstest2021.csv
- data/wmt-zhen-newstest2021.csv</code></pre>

Read the file as a recipe. `pretrained_model: xlm-roberta-large` is the encoder. `layer: mix` and
`layer_transformation: sparsemax` say which of its layers COMET reads, `pool: avg` how a sentence becomes one
vector, `hidden_sizes` and `activations` describe the small network on top, and `train_data` names the WMT
direct assessments of 2017–2020.

The checkpoint is a 2.3 GB file of tensors. `weights_only=True` makes `torch.load` refuse anything that isn't a
tensor or a plain value, so a downloaded checkpoint can't run code on your machine. Everything that isn't part of
the encoder:

```python
checkpoint_path = hf_hub_download(COMET_REPO, "checkpoints/model.ckpt")
weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"]

rows = []
for name, tensor in weights.items():
    if not name.startswith("encoder.") and "scalar_parameters" not in name:
        rows.append({"tensor": name, "shape": str(tuple(tensor.shape))})
layer_scores = []
for name in weights:
    if "scalar_parameters" in name:
        layer_scores.append(weights[name])
shape = f"{len(layer_scores)} × (1,)"
rows.append({"tensor": "layerwise_attention.scalar_parameters.*", "shape": shape})
table(pd.DataFrame(rows))
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">tensor</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">shape</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">layerwise_attention.gamma</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">(1,)</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">layerwise_attention.dropout_mask</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">(25,)</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">layerwise_attention.dropout_fill</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">(25,)</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">estimator.ff.0.weight</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">(3072, 6144)</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">estimator.ff.0.bias</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">(3072,)</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">estimator.ff.3.weight</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">(1024, 3072)</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">estimator.ff.3.bias</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">(1024,)</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">estimator.ff.6.weight</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">(1, 1024)</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">estimator.ff.6.bias</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">(1,)</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">layerwise_attention.scalar_parameters.*</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">25 × (1,)</td></tr></tbody></table></div>

Next to the encoder there are only two small pieces: the 25 scores that mix XLM-R's layers, with a scale
`gamma`, and the three linear layers of the regressor. The `dropout_mask` and `dropout_fill` tensors are used in
training only.

### The encoder

The checkpoint holds XLM-R's weights after COMET's training, so we build an empty XLM-R large from its
configuration file and fill it with them. The tokenizer is XLM-R's own; COMET doesn't change it.

```python
from transformers import AutoConfig, XLMRobertaModel

XLMR = "FacebookAI/xlm-roberta-large"
xlmr_tokenizer = AutoTokenizer.from_pretrained(XLMR)
encoder = XLMRobertaModel(AutoConfig.from_pretrained(XLMR), add_pooling_layer=False)

encoder_weights = {}
for name, tensor in weights.items():
    if name.startswith("encoder.model."):
        encoder_weights[name.removeprefix("encoder.model.")] = tensor
missing, unexpected = encoder.load_state_dict(encoder_weights, strict=False)
print("missing:", missing, " unexpected:", unexpected)
encoder = encoder.to(device).eval()
print(sum(p.numel() for p in encoder.parameters()), "parameters in the encoder")
```

<pre class="tfs-printed" markdown="0"><code>missing: []  unexpected: [&#x27;embeddings.position_ids&#x27;]
558840832 parameters in the encoder</code></pre>

Nothing is missing. The one unexpected tensor, `position_ids`, is a table of positions 0, 1, 2, ... that newer
versions of `transformers` compute instead of storing.

### A mix of layers, and sparsemax

`layer: mix` means COMET doesn't read XLM-R's last layer only. It takes all 25 outputs, the embeddings and each
of the 24 layers, and adds them up with learned weights. Lower layers know more about words, higher ones more
about meaning in context, and training decided how much of each to use.

The 25 learned scores become weights through **sparsemax**<sup class="cite">[<a href="#ref-8" id="cite-8" data-ref="8" aria-label="Reference 8: Martins, A., Astudillo, R. (2016). From Softmax to Sparsemax. ICML.">8</a>]</sup> rather than softmax. Like softmax it turns any scores into non-negative
weights that sum to 1. Unlike softmax it can give a weight of exactly zero:

$$\operatorname{sparsemax}(\mathbf{z})_i = \max(z_i - \tau,\ 0).$$

$$\tau$$ is the one threshold for which the weights sum to 1: everything below it is cut to zero, everything above
it keeps its distance to it. To find $$\tau$$, sort the scores in decreasing order, $$z_{(1)} \ge z_{(2)} \ge
\dots$$; the survivors are the first $$k$$, for the largest $$k$$ with $$1 + k\, z_{(k)} > \sum_{j \le k} z_{(j)}$$, and
then $$\tau = \big(\sum_{j \le k} z_{(j)} - 1\big) / k$$.

```python
def sparsemax(z):
    z_sorted, _ = z.sort(descending=True)
    k = torch.arange(1, len(z) + 1, dtype=z.dtype)
    cumulative = z_sorted.cumsum(dim=0)
    survivors = (1 + k * z_sorted > cumulative).sum()  # how many weights stay above zero
    tau = (cumulative[survivors - 1] - 1) / survivors
    return (z - tau).clamp(min=0)
```

The sorting rule is not obvious, so check it against the definition itself: search for the $$\tau$$ at which the
weights sum to 1 by bisection, which needs no cleverness at all, and compare.

```python
def sparsemax_by_search(z, steps=100):
    low, high = z.min().item() - 1, z.max().item()
    for _ in range(steps):
        tau = (low + high) / 2
        if (z - tau).clamp(min=0).sum() > 1:  # threshold too low: the sum is over 1
            low = tau
        else:
            high = tau
    return (z - tau).clamp(min=0)


for z in [torch.tensor([2.0, 1.0, -1.0]), torch.tensor([0.5, 0.3, 0.1])]:
    assert torch.allclose(sparsemax(z), sparsemax_by_search(z), atol=1e-6)
    shown_sparsemax = [round(x, 3) for x in sparsemax(z).tolist()]
    shown_softmax = [round(x, 3) for x in z.softmax(dim=0).tolist()]
    shown_z = [round(x, 1) for x in z.tolist()]
    print(shown_z, "→ sparsemax", shown_sparsemax, " softmax", shown_softmax)
```

<pre class="tfs-printed" markdown="0"><code>[2.0, 1.0, -1.0] → sparsemax [1.0, 0.0, 0.0]  softmax [0.705, 0.259, 0.035]
[0.5, 0.3, 0.1] → sparsemax [0.533, 0.333, 0.133]  softmax [0.402, 0.329, 0.269]</code></pre>

The first example gives all the weight to the largest score: the others are too far behind. The second keeps all
three. Now COMET's 25 layers:

```python
layer_scores = torch.cat(layer_scores)
layer_weights = sparsemax(layer_scores)
assert torch.allclose(layer_weights, sparsemax_by_search(layer_scores), atol=1e-5)
gamma = weights["layerwise_attention.gamma"]

fig, ax = plt.subplots(figsize=(6.4, 2.8))
ax.bar(range(len(layer_weights)), layer_weights, color=COLORS["enc"])
for layer, weight in enumerate(layer_weights.tolist()):
    if weight == 0:
        ax.text(layer, 0.002, "0", ha="center", fontsize=8, color=COLORS["dec"])
ax.set_xlabel("XLM-R output (0: the embeddings, 24: the last layer)")
ax.set_ylabel("weight")
ax.set_title("How much of each layer COMET uses")
plt.show()
print("layers with weight exactly 0:", (layer_weights == 0).nonzero().flatten().tolist())
print("the mix is multiplied by gamma =", round(gamma.item(), 3))
```

<img class="tfs-output" src="/assets/posts/tfs/04/output-05.png" alt="How much of each layer COMET uses" loading="lazy">

<pre class="tfs-printed" markdown="0"><code>layers with weight exactly 0: [8, 9]
the mix is multiplied by gamma = 0.836</code></pre>

COMET listens most to the embeddings and the lowest layers, ignores a few in the middle completely, and spreads
the rest over the upper half. The lowest layers carry the words themselves, the upper ones their meaning in
context; a translation metric needs both.

### From sentences to a score

A sentence's vector is the mix of layers, averaged over its tokens (padding excluded). The three vectors go into
the features of the picture, and the features into a feed-forward network of three linear layers with $$\tanh$$
between them. The checkpoint numbers its layers with the dropout in between (0, 3, 6), ours without (0, 2, 4).

```python
@torch.no_grad()
def embed(texts):
    batch = xlmr_tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=510).to(device)
    layers = encoder(**batch, output_hidden_states=True).hidden_states  # 25 × (B, T, 1024)
    mixed = 0
    for weight, layer in zip(layer_weights.tolist(), layers):
        mixed = mixed + weight * layer
    mixed = gamma.item() * mixed
    mask = batch["attention_mask"].unsqueeze(-1).float()
    return (mixed * mask).sum(dim=1) / mask.sum(dim=1)  # the mean over real tokens


estimator = torch.nn.Sequential(
    torch.nn.Linear(6144, 3072),
    torch.nn.Tanh(),
    torch.nn.Linear(3072, 1024),
    torch.nn.Tanh(),
    torch.nn.Linear(1024, 1),
)
for ours, theirs in [(0, 0), (2, 3), (4, 6)]:
    estimator[ours].weight.data = weights[f"estimator.ff.{theirs}.weight"]
    estimator[ours].bias.data = weights[f"estimator.ff.{theirs}.bias"]
estimator = estimator.to(device).eval()
del weights, encoder_weights  # 2.3 GB of tensors we no longer need
```

A test set has the same source sentence and the same reference for every system, so each distinct text is
embedded once and kept: `embed_all` fills a dictionary of vectors, and `comet` only looks them up.

```python
embeddings = {}


def embed_all(texts, batch_size=64):
    todo = [text for text in dict.fromkeys(texts) if text not in embeddings]
    todo.sort(key=len)  # similar lengths in one batch: little padding
    for start in range(0, len(todo), batch_size):
        chunk = todo[start : start + batch_size]
        for text, vector in zip(chunk, embed(chunk).cpu()):
            embeddings[text] = vector


@torch.no_grad()
def comet(sources, translations, references, batch_size=512):
    embed_all(sources + translations + references)
    scores = []
    for start in range(0, len(translations), batch_size):
        end = start + batch_size
        s = torch.stack([embeddings[t] for t in sources[start:end]]).to(device)
        h = torch.stack([embeddings[t] for t in translations[start:end]]).to(device)
        r = torch.stack([embeddings[t] for t in references[start:end]]).to(device)
        features = torch.cat([h, r, h * r, (h - r).abs(), h * s, (h - s).abs()], dim=1)
        scores += estimator(features).squeeze(1).tolist()
    return scores
```

Does it compute what the official package computes? These five triples were scored with `unbabel-comet` 2.2.7,
in an environment of its own, and the scores are copied here. Ours must agree to the fourth decimal:

```python
PACKAGE_SCORES = [  # (source, translation, reference, score from unbabel-comet 2.2.7)
    ("The researcher said she was exhausted.", "Исследователь сказал, что она устала.",
     "Исследовательница сказала, что вымоталась.", 0.873125),
    ("The researcher said she was exhausted.", "Исследовательница сказала, что устала.",
     "Исследовательница сказала, что вымоталась.", 0.906672),
    ("Tom forgot to lock the door.", "Том забыл, чтобы я забыл дверь.",
     "Том забыл запереть дверь.", 0.459986),
    ("Tom forgot to lock the door.", "", "Том забыл запереть дверь.", 0.347070),
    ("Tom forgot to lock the door.", "Tom forgot to lock the door.",
     "Том забыл запереть дверь.", 0.614086),
]
fixture_sources = [row[0] for row in PACKAGE_SCORES]
fixture_translations = [row[1] for row in PACKAGE_SCORES]
fixture_references = [row[2] for row in PACKAGE_SCORES]
ours = comet(fixture_sources, fixture_translations, fixture_references)
for row, mine in zip(PACKAGE_SCORES, ours):
    theirs = row[3]
    assert abs(mine - theirs) < 1e-4
    print(f"{mine:.4f} = {theirs:.4f}  {row[1]!r}")
```

<pre class="tfs-printed" markdown="0"><code>0.8731 = 0.8731  &#x27;Исследователь сказал, что она устала.&#x27;
0.9067 = 0.9067  &#x27;Исследовательница сказала, что устала.&#x27;
0.4600 = 0.4600  &#x27;Том забыл, чтобы я забыл дверь.&#x27;
0.3471 = 0.3471  &#x27;&#x27;
0.6141 = 0.6141  &#x27;Tom forgot to lock the door.&#x27;</code></pre>

Our forward pass gives the package's scores. They are worth reading. Lesson 3's masculine translation of the researcher sentence scores within a few hundredths of a correct one: the wrong gender of two words barely registers.

And the English sentence, copied untranslated, beats the broken Russian of the second epoch. COMET's scores have no fixed scale either: nothing forces them into $$[0, 1]$$, and an empty translation still gets about a third.

Now the test set. Every distinct text is embedded once, so the three systems cost little more than one:

```python
start = time.perf_counter()
for row in scores:
    system = row["system"]
    test[f"COMET {system}"] = comet(list(test.en), list(test[system]), list(test.ru))
    row["COMET"] = test[f"COMET {system}"].mean()  # the system's score: the mean over pairs
print("embedded", len(embeddings), "texts in", round(time.perf_counter() - start), "s")
table(pd.DataFrame(scores), digits=3)
```

<pre class="tfs-printed" markdown="0"><code>embedded 110313 texts in 364 s</code></pre>

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">system</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">BLEU</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">chrF</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">chrF++</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">COMET</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ours, greedy</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">39.648</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">58.416</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">57.868</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.862</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ours, beam</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">40.840</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">59.267</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">58.706</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.875</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">opus-mt-en-ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">45.386</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">65.093</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">64.150</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.932</td></tr></tbody></table></div>

Same order as before: opus-mt-en-ru, then beam search, then greedy. On COMET's scale the gaps look small, but
COMET's scores cluster near the top: an average difference of a few hundredths separates a good translator from
a mediocre one. The numbers mean something only next to each other, never alone.

And the sentence from the top of the lesson. opus-mt-en-ru's translation against each of its fifteen references,
by all three metrics:

```python
hero_refs = references[HERO]
hero_translation = outputs["opus-mt-en-ru"][HERO]
hero_comet = comet([HERO] * len(hero_refs), [hero_translation] * len(hero_refs), hero_refs)
rows = []
for reference, comet_score in zip(hero_refs, hero_comet):
    rows.append(
        {
            "reference": reference,
            "BLEU": sacrebleu.sentence_bleu(hero_translation, [reference]).score,
            "chrF": sacrebleu.sentence_chrf(hero_translation, [reference]).score,
            "COMET": comet_score,
        }
    )
print("opus-mt-en-ru:", hero_translation)
table(pd.DataFrame(rows).sort_values("COMET", ascending=False), digits=2)
```

<pre class="tfs-printed" markdown="0"><code>opus-mt-en-ru: Ты идёшь одна?</code></pre>

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">reference</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">BLEU</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">chrF</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">COMET</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Ты придёшь одна?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">35.36</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">70.55</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.95</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Ты одна придёшь?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">21.02</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">38.93</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.94</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Ты придёшь один?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">19.00</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">45.30</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.94</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Ты один придёшь?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">19.00</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">30.61</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.93</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Вы придёте одна?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">31.95</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">29.60</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.92</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Вы одна придёте?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">19.00</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">23.53</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.92</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Ты приедешь одна?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">35.36</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">42.37</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.92</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Ты одна приедешь?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">21.02</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">23.10</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.92</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Вы придёте один?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">15.97</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">15.21</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.91</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Вы один придёте?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">15.97</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">15.21</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.90</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Ты один приедешь?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">19.00</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">15.31</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.90</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Вы приедете одна?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">31.95</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">22.80</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.89</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Вы одна приедете?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">19.00</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">17.14</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.88</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Вы приедете один?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">15.97</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">9.35</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.86</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Вы один приедете?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">15.97</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">9.35</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.85</td></tr></tbody></table></div>

BLEU and chrF jump around with the wording of each reference; COMET calls the translation good against all
fifteen. It still prefers some references to others, the ones closest in wording, but the spread is under a
tenth, not the whole scale. That's the difference between comparing strings and comparing meanings.

## 8. Metrics against professional translators

A metric is a model of human judgment, and it is tested like any model: on data it hasn't seen.

The most careful human judgments available are **MQM** scores<sup class="cite">[<a href="#ref-9" id="cite-9" data-ref="9" aria-label="Reference 9: Freitag, M. et al. (2021). Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation. TACL. MQM.">9</a>]</sup>: professional translators mark every error in a translation, classify it (accuracy, fluency, terminology...) and rate its severity, and the score is minus the weighted sum of the errors. Zero means flawless.

For the WMT22 metrics task<sup class="cite">[<a href="#ref-10" id="cite-10" data-ref="10" aria-label="Reference 10: Freitag, M. et al. (2022). Results of WMT22 Metrics Shared Task: Stop Using BLEU – Neural Metrics Are Better and More Robust. WMT.">10</a>]</sup>, professional
translators scored English–Russian this way, segment by segment: news, social media, e-commerce, conversations.
The released scores cover 15 systems on the same segments, and they are newer than COMET's training data, which
ends in 2020:

```python
MQM_REPO = "RicardoRei/wmt-mqm-human-evaluation"
mqm_path = hf_hub_download(MQM_REPO, "train.csv", repo_type="dataset")
mqm = pd.read_csv(mqm_path)
mqm = mqm[(mqm.lp == "en-ru") & (mqm.year == 2022)].reset_index(drop=True)
if SMOKE:
    kept = set(mqm.src.drop_duplicates().head(20))
    mqm = mqm[mqm.src.isin(kept)].reset_index(drop=True)
print("segments:", len(mqm), "systems:", mqm.system.nunique())
table(mqm[["src", "mt", "score", "system"]].sample(4, random_state=SEED))
```

<pre class="tfs-printed" markdown="0"><code>segments: 19725 systems: 15</code></pre>

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">src</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">mt</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">score</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">system</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">He might left it there.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Он мог оставить его там.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-0.00</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">JDExploreAcademy</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Repairing your #PRS_ORG# account on eReader</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Восстановление учетной записи #PRS_ORG# на eReader</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-0.00</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Online-A</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">We are still looking ways to improve our services and this will be noted as a feedback to one of our valued customer.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мы все еще ищем способы улучшения наших услуг, и это будет воспринято как подарок одному из наших уважаемых клиентов.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-6.00</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">QUARTZ_TuneReranking</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">De Klerk died on November 11 aged 85 following a battle with cancer.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Де Клерк умер 11 ноября в возрасте 85 лет после борьбы с раком.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-0.00</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">PROMT</td></tr></tbody></table></div>

Score every system with our three metrics, BLEU and chrF over each system's segments, COMET as the mean of its
segments' scores, and put them next to the translators' average:

```python
start = time.perf_counter()
mqm["COMET"] = comet(list(mqm.src), list(mqm.mt), list(mqm.ref))
print("COMET on", len(mqm), "segments in", round(time.perf_counter() - start), "s")

rows = []
for system, part in mqm.groupby("system"):
    rows.append(
        {
            "system": system,
            "MQM": part.score.mean(),
            "BLEU": sacrebleu.corpus_bleu(list(part.mt), [list(part.ref)]).score,
            "chrF": sacrebleu.corpus_chrf(list(part.mt), [list(part.ref)]).score,
            "COMET": part.COMET.mean(),
        }
    )
judged = pd.DataFrame(rows).sort_values("MQM", ascending=False).reset_index(drop=True)
for metric in ["BLEU", "chrF", "COMET"]:
    judged[f"{metric} rank"] = judged[metric].rank(ascending=False).astype(int)
table(judged, digits=2)
```

<pre class="tfs-printed" markdown="0"><code>COMET on 19725 segments in 116 s</code></pre>

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">system</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">MQM</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">BLEU</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">chrF</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">COMET</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">BLEU rank</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">chrF rank</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">COMET rank</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Online-W</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-1.37</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">32.91</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">58.73</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.90</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">8</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">6</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">bleu_bestmbr</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-1.85</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">36.30</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">60.31</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.89</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">3</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Online-B</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-1.94</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">35.66</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">60.29</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.89</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">4</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Online-G</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-2.03</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">33.39</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">59.93</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.89</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">5</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">3</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">5</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">JDExploreAcademy</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-2.08</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">33.21</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">58.64</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.88</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">7</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">7</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">7</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">comet_bestmbr</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-2.09</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">29.46</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">56.27</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.90</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">13</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">12</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Lan-Bridge</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-2.34</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">33.34</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">59.60</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.88</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">6</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">4</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">8</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Online-Y</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-2.55</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">34.35</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">59.18</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.88</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">3</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">5</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">6</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Online-A</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-2.85</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">33.55</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">58.44</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.87</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">4</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">8</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">9</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">PROMT</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-2.94</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">31.29</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">56.47</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.86</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">10</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">10</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">10</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">HuaweiTSC</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-3.40</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">31.50</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">56.32</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.85</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">9</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">11</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">14</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">SRPOL</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-3.68</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">30.90</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">56.68</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.86</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">11</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">9</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">12</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">eTranslation</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-3.79</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">29.76</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">55.89</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.85</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">12</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">13</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">13</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">QUARTZ_TuneReranking</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-4.06</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">25.38</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">51.71</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.86</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">15</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">15</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">11</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">M2M100_1.2B-B4</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">-4.56</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">28.60</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">53.18</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.84</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">14</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">14</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">15</td></tr></tbody></table></div>

Look at the top of the table. The translators' favourite, `Online-W`, a commercial online system, is far from first by BLEU.

First by BLEU is `bleu_bestmbr`, a system the organizers added that picks, among many candidate translations, the one with the best BLEU against the others. First by COMET is `comet_bestmbr`, the same trick with an older COMET. Each metric crowns the system built to please it.

The same in pictures, every dot a system:

```python
LABELLED = ["Online-W", "bleu_bestmbr", "comet_bestmbr", "M2M100_1.2B-B4"]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, metric in zip(axes, ["BLEU", "chrF", "COMET"]):
    ax.scatter(judged.MQM, judged[metric], color=COLORS["muted"], s=24)
    for _, row in judged[judged.system.isin(LABELLED)].iterrows():
        ax.scatter(row.MQM, row[metric], color=COLORS["dec"], s=30)
        point = (row.MQM, row[metric])
        ax.annotate(row.system, point, xytext=(4, 4), textcoords="offset points",
                    fontsize=8)
    ax.set_xlabel("MQM, translators (0 = no errors)")
    ax.set_title(metric)
plt.tight_layout()
plt.show()
```

<img class="tfs-output" src="/assets/posts/tfs/04/output-06.png" alt="Output of the code above" loading="lazy">

One number summarizes the picture: **pairwise accuracy**<sup class="cite">[<a href="#ref-11" id="cite-11" data-ref="11" aria-label="Reference 11: Kocmi, T. et al. (2021). To Ship or Not to Ship: An Extensive Evaluation of Automatic Metrics for Machine Translation. WMT. Pairwise accuracy.">11</a>]</sup>. Take every pair of systems and ask whether the metric orders
them the way the translators did. With 15 systems there are 105 pairs.

```python
rows = []
for metric in ["BLEU", "chrF", "COMET"]:
    agree, pairs = 0, 0
    for i in range(len(judged)):
        for j in range(i + 1, len(judged)):
            pairs += 1
            human = judged.MQM[i] - judged.MQM[j]
            machine = judged[metric][i] - judged[metric][j]
            if human * machine > 0:  # same sign: same order
                agree += 1
    rows.append({"metric": metric, "pairs in the translators' order": f"{agree} of {pairs}",
                 "accuracy": agree / pairs})
table(pd.DataFrame(rows), digits=3)
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">metric</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">pairs in the translators&#x27; order</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">accuracy</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">BLEU</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">82 of 105</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.781</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">chrF</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">89 of 105</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.848</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">COMET</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">93 of 105</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.886</td></tr></tbody></table></div>

COMET orders the most pairs the way the translators did, chrF fewer, BLEU fewest. That is the finding of the
WMT22 metrics task in miniature, where BLEU came last of 13 reference-based metrics, and of Kocmi et al., who
checked 3,344 system pairs and concluded: "Do not use BLEU, it is inferior to other metrics, and it has been
overused." Neither COMET nor anything else gets all pairs right.

## 9. Translations broken on purpose

Correlation over 15 systems says how a metric ranks whole systems. It doesn't say *what* the metric notices. For
that, make translations whose quality we know.

The test set provides the good ones: for a sentence with two different references, the second is a correct
translation that happens not to be the first. Score it against the first. The bad ones we make: take the first
reference itself and break it in one small, known way, then score the broken copy against the original.

A metric that understands translation must prefer the other correct translation to a broken copy of the
reference, even though the copy shares almost every word with it. Five ways to break a sentence:

```python
def breakages(reference, english):
    broken = {}
    if re.search(r"(?i)(?<![а-яё])не\s", reference):  # «не» as a word: drop it
        without = re.sub(r"(?i)(?<![а-яё])не\s", "", reference, count=1)
        broken["«не» dropped"] = without[:1].upper() + without[1:]
    if "Том" in reference:  # Tom is a different person now, in any case form
        broken["Tom becomes John"] = reference.replace("Том", "Джон")
    words = reference.rstrip(".!?").split()
    if len(words) >= 4:
        ending = reference[len(reference.rstrip(".!?")) :]
        broken["last word dropped"] = " ".join(words[:-1]) + ending
    broken["left in English"] = english
    if has_yo(reference):  # not an error at all: the control
        broken["ё written as е"] = without_yo(reference)
    return broken


print(breakages("Том не знает, где Мэри живёт.", "Tom doesn't know where Mary lives."))
```

<pre class="tfs-printed" markdown="0"><code>{&#x27;«не» dropped&#x27;: &#x27;Том знает, где Мэри живёт.&#x27;, &#x27;Tom becomes John&#x27;: &#x27;Джон не знает, где Мэри живёт.&#x27;, &#x27;last word dropped&#x27;: &#x27;Том не знает, где Мэри.&#x27;, &#x27;left in English&#x27;: &quot;Tom doesn&#x27;t know where Mary lives.&quot;, &#x27;ё written as е&#x27;: &#x27;Том не знает, где Мэри живет.&#x27;}</code></pre>

The last one is a control: «ё written as е» is the same translation in different typography. There both
translations are correct, so neither preference is wrong; what matters is how much a metric charges for the
change of letters, which the second table shows.

```python
rows = []
for english, refs in references.items():
    distinct = list(dict.fromkeys(refs))
    if len(distinct) < 2:
        continue
    first, other = distinct[0], distinct[1]
    for kind, broken in breakages(first, english).items():
        row = {"en": english, "reference": first, "other": other, "kind": kind}
        rows.append({**row, "broken": broken})
stress = pd.DataFrame(rows)
print(stress.kind.value_counts().to_string())
```

<pre class="tfs-printed" markdown="0"><code>kind
left in English      4934
last word dropped    4217
«не» dropped         1063
Tom becomes John      763
ё written as е        663</code></pre>

```python
stress["COMET other"] = comet(list(stress.en), list(stress.other), list(stress.reference))
stress["COMET broken"] = comet(list(stress.en), list(stress.broken), list(stress.reference))
for metric, score in [("BLEU", sacrebleu.sentence_bleu), ("chrF", sacrebleu.sentence_chrf)]:
    other, broken = [], []
    for row in stress.itertuples():
        other.append(score(row.other, [row.reference]).score)
        broken.append(score(row.broken, [row.reference]).score)
    stress[f"{metric} other"] = other
    stress[f"{metric} broken"] = broken

KINDS = ["«не» dropped", "Tom becomes John", "last word dropped", "left in English",
         "ё written as е"]  # the last one is no error at all
METRICS = ["BLEU", "chrF", "COMET"]
preferred = np.zeros((len(KINDS), len(METRICS)))
for i, kind in enumerate(KINDS):
    part = stress[stress.kind == kind]
    for j, metric in enumerate(METRICS):
        preferred[i, j] = (part[f"{metric} other"] > part[f"{metric} broken"]).mean()

heatmap(preferred, rows=KINDS, cols=METRICS, annotate=True, fmt="{:.0%}", vmax=1, cell=0.6,
        title="How often the metric prefers the other correct translation")
plt.show()
```

<img class="tfs-output" src="/assets/posts/tfs/04/output-07.png" alt="How often the metric prefers the other correct translation" loading="lazy">

Read the rows one by one. Untranslated English: everyone catches it, it shares no words with the reference. A dropped last word: BLEU and chrF still prefer the damaged copy most of the time, COMET mostly doesn't.

A dropped «не», which reverses the meaning: BLEU and chrF prefer the reversed sentence to a correct paraphrase most of the time, while COMET is right in most cases, though far from all.

Tom becoming John fools all three. For BLEU and chrF that's expected, one word in many. For COMET it is a known
blind spot: [Amrhein and Sennrich (2022)](https://aclanthology.org/2022.aacl-main.83/) found COMET models "not
sensitive enough to discrepancies in numbers and named entities".

The averages tell the same story from another side: what each kind of damage costs, in each metric's own units.

```python
rows = []
for kind in KINDS:
    part = stress[stress.kind == kind]
    row = {"translation": f"the reference, {kind}", "sentences": len(part)}
    for metric in METRICS:
        row[metric] = part[f"{metric} broken"].mean()
    rows.append(row)
others = stress.drop_duplicates("en")
row = {"translation": "another correct translation", "sentences": len(others)}
for metric in METRICS:
    row[metric] = others[f"{metric} other"].mean()
rows.append(row)
table(pd.DataFrame(rows), digits=2)
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">translation</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">sentences</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">BLEU</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">chrF</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">COMET</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">the reference, «не» dropped</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1,063</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">60.22</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">83.88</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.92</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">the reference, Tom becomes John</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">763</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">66.16</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">81.75</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.97</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">the reference, last word dropped</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">4,217</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">64.47</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">72.88</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.80</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">the reference, left in English</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">4,934</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">6.90</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.82</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.62</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">the reference, ё written as е</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">663</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">53.20</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">82.76</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.97</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">another correct translation</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">4,934</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">42.17</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">66.76</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.95</td></tr></tbody></table></div>

By BLEU, a sentence missing its «не» is much better than a correct translation worded differently, and ё written as е costs more than a reversed meaning. chrF is milder about the letters and just as blind to the meaning.

COMET hardly charges for ё at all, which is right; its numbers sit in a narrow band, but their order is mostly the order a reader would choose, names aside.

> Every metric is blind to something. Before trusting one, find out what.

## 10. The paired bootstrap: which differences are real

Back to our systems. Every score above was measured on one test set, and a test set is a sample: with other sentences, the numbers would come out somewhat different.

Is beam search really better than greedy, or did it get lucky with these sentences? The test set can answer that about itself.

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/04/bootstrap-light.png" width="760" alt="The paired bootstrap on a toy test set of 8 sentences: draw 8 sentences with replacement, so that some appear twice and others not at all, score both systems on the same draw and record the difference; on the whole toy set A wins, but on some draws B does. Repeated a thousand times, the differences form a distribution, and the share of draws where A beats B says how sure we can be" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/04/bootstrap-dark.png" width="760" alt="The paired bootstrap on a toy test set of 8 sentences: draw 8 sentences with replacement, so that some appear twice and others not at all, score both systems on the same draw and record the difference; on the whole toy set A wins, but on some draws B does. Repeated a thousand times, the differences form a distribution, and the share of draws where A beats B says how sure we can be" loading="lazy">

**Paired bootstrap resampling**<sup class="cite">[<a href="#ref-12" id="cite-12" data-ref="12" aria-label="Reference 12: Koehn, P. (2004). Statistical Significance Tests for Machine Translation Evaluation. EMNLP. The paired bootstrap.">12</a>]</sup> builds new test sets from the one we have: draw as many sentences as it has, with replacement, so that some come twice and some not at all. Score both systems on the same draw and note the difference.

A thousand draws give a thousand differences. In Koehn's words: "If, say, one system outperforms the other system 95% of the time, we draw the conclusion that it is better with 95% statistical significance."

The statistics we kept per sentence make it cheap. A draw is a list of sentence numbers; BLEU and chrF of the
draw come from the sums of their rows, COMET is the mean of its sentences' scores.

```python
def corpus_score(metric, system, rows):
    if metric == "BLEU":
        return bleu_from_stats(bleu_rows[system][rows].sum(axis=0))
    if metric == "chrF":
        return chrf_from_stats(chrf_rows[system][rows].sum(axis=0))
    return test[f"COMET {system}"].to_numpy()[rows].mean()


def paired_bootstrap(a, b, metric, subset=None, draws=1000):
    pool = np.arange(len(test)) if subset is None else np.flatnonzero(subset)
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(draws):
        rows = rng.choice(pool, size=len(pool), replace=True)  # the same draw for both
        differences.append(corpus_score(metric, a, rows) - corpus_score(metric, b, rows))
    return np.array(differences)
```

```python
COMPARISONS = [("ours, beam", "ours, greedy"), ("opus-mt-en-ru", "ours, beam")]
rows, all_differences = [], {}
for a, b in COMPARISONS:
    for metric in METRICS:
        differences = paired_bootstrap(a, b, metric)
        all_differences[(a, b, metric)] = differences
        everything = np.arange(len(test))
        whole = corpus_score(metric, a, everything) - corpus_score(metric, b, everything)
        low, high = np.percentile(differences, [2.5, 97.5])
        wins = f"{(differences > 0).mean():.1%}"
        rows.append({"A − B": f"{a} − {b}", "metric": metric, "on the test set": whole,
                     "95% of draws from": low, "to": high, "A wins in": wins})
table(pd.DataFrame(rows), digits=3)
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">A − B</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">metric</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">on the test set</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">95% of draws from</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">to</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">A wins in</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ours, beam − ours, greedy</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">BLEU</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1.191</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.968</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1.389</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">100.0%</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ours, beam − ours, greedy</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">chrF</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.852</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.739</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.952</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">100.0%</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ours, beam − ours, greedy</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">COMET</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.014</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.013</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.014</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">100.0%</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">opus-mt-en-ru − ours, beam</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">BLEU</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">4.547</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">4.177</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">4.906</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">100.0%</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">opus-mt-en-ru − ours, beam</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">chrF</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">5.826</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">5.520</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">6.131</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">100.0%</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">opus-mt-en-ru − ours, beam</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">COMET</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.057</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.055</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.058</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">100.0%</td></tr></tbody></table></div>

On the whole test set none of these differences is in doubt: beam search beats greedy, and opus-mt-en-ru beats
both, in every one of a thousand draws, by every metric. Tens of thousands of test sentences make even a point
of BLEU certain.

```python
fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
for ax, metric in zip(axes, METRICS):
    differences = all_differences[("ours, beam", "ours, greedy", metric)]
    ax.hist(differences, bins=40, color=COLORS["enc"])
    ax.axvline(0, color=COLORS["dec"], linewidth=1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))  # a few readable ticks
    ax.set_title(f"{metric}: beam − greedy", fontsize=10)
    ax.set_yticks([])
axes[0].set_ylabel("draws")
plt.tight_layout()
plt.show()
```

<img class="tfs-output" src="/assets/posts/tfs/04/output-08.png" alt="Output of the code above" loading="lazy">

None of the thousand differences crosses zero. That's a luxury: test sets in papers are often a thousand or
two sentences. Draw smaller test sets from ours and ask the bootstrap again, how often beam search wins:

```python
rng = np.random.default_rng(SEED)
rows = []
for size in [100, 300, 1_000, 3_000]:
    if size > len(test) // 2:
        break
    small = np.zeros(len(test), dtype=bool)
    small[rng.choice(len(test), size=size, replace=False)] = True  # a random small test set
    row = {"test sentences": size}
    for metric in METRICS:
        differences = paired_bootstrap("ours, beam", "ours, greedy", metric, subset=small)
        row[f"{metric}: beam wins in"] = f"{(differences > 0).mean():.0%}"
    rows.append(row)
table(pd.DataFrame(rows))
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">test sentences</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">BLEU: beam wins in</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">chrF: beam wins in</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">COMET: beam wins in</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">100</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">64%</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">87%</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">98%</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">300</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">92%</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">98%</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">100%</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1,000</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">99%</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">100%</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">100%</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">3,000</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">99%</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">100%</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">100%</td></tr></tbody></table></div>

On a hundred sentences COMET is already all but certain that beam search is better, while BLEU, measuring the same translations, is not much surer than a coin toss. It takes BLEU a few hundred sentences more to catch up.

A sentence gives BLEU a handful of n-gram matches, a noisy signal, and noise takes more sentences to average out. With a small test set, the metric decides what you can see at all.

> Report a difference with its bootstrap interval, or don't report it.

## 11. What opus-mt-en-ru saw in training

One more thing to check about our comparison: whether the test set is a test for both systems. Lesson 1 removed from our training data every pair that shares a sentence with the test set.

Nobody did that for opus-mt-en-ru. The Tatoeba Challenge says so itself, in its rules: "current OPUS-MT models can not be used as they contain Tatoeba data that may overlap with the test data in this release!"

The model card says it was trained on "opus", and the OPUS-MT training recipe of the time took every corpus OPUS had for the language pair. It held back 10,000 Tatoeba pairs for validation and testing and appended the rest of Tatoeba to the training data.

The Tatoeba release OPUS had then was published in July 2019. Download it, 15 MB, and look for our test pairs in it:

```python
OPUS_TATOEBA = "https://object.pouta.csc.fi/OPUS-Tatoeba/v20190709/moses/en-ru.txt.zip"
archive = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(OPUS_TATOEBA).read()))
opus_en = archive.read("Tatoeba.en-ru.en").decode().splitlines()
opus_ru = archive.read("Tatoeba.en-ru.ru").decode().splitlines()
print("pairs in OPUS Tatoeba v20190709:", len(opus_en))

opus_pairs = set()
for en, ru in zip(opus_en, opus_ru):
    opus_pairs.add((normalize(en), normalize(ru)))  # normalized like our test set
seen = []
for en, ru in zip(test.en, test.ru):
    seen.append((en, ru) in opus_pairs)
test["seen by opus-mt"] = seen
print(f"test pairs in it: {test['seen by opus-mt'].sum()} of {len(test)}")
```

<pre class="tfs-printed" markdown="0"><code>pairs in OPUS Tatoeba v20190709: 487196
test pairs in it: 9714 of 36385</code></pre>

About a quarter of our test set was public, as training-ready pairs, half a year before opus-mt-en-ru was
trained, by a recipe that put all such pairs into training except a held-out ten thousand.

Did it help? Score both parts separately. Our model is the control: it saw none of these pairs, so whatever
makes the "seen" part easier for it is the part, not memory.

```python
rows = []
for system in SYSTEMS:
    for flag, part_name in [(True, "seen by opus-mt"), (False, "not seen")]:
        rows_of_part = np.flatnonzero(test["seen by opus-mt"].to_numpy() == flag)
        part = test.iloc[rows_of_part]
        rows.append(
            {
                "system": system,
                "part": part_name,
                "pairs": len(part),
                "BLEU": corpus_score("BLEU", system, rows_of_part),
                "chrF": corpus_score("chrF", system, rows_of_part),
                "COMET": corpus_score("COMET", system, rows_of_part),
                "same as reference": (part[system] == part.ru).mean(),
            }
        )
seen_scores = pd.DataFrame(rows)
table(seen_scores, digits=3)
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">system</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">part</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">pairs</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">BLEU</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">chrF</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">COMET</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">same as reference</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ours, greedy</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">seen by opus-mt</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">9,714</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">47.893</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">64.540</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.898</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.263</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ours, greedy</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">not seen</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">26,671</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">36.998</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">56.492</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.848</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.175</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ours, beam</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">seen by opus-mt</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">9,714</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">48.794</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">65.236</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.908</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.273</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ours, beam</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">not seen</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">26,671</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">38.262</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">57.386</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.863</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.185</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">opus-mt-en-ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">seen by opus-mt</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">9,714</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">49.162</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">67.182</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.940</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.261</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">opus-mt-en-ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">not seen</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">26,671</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">44.188</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">64.439</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.929</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.208</td></tr></tbody></table></div>

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for ax, metric in zip(axes, ["BLEU", "COMET"]):
    x = np.arange(len(SYSTEMS))
    for offset, (part_name, color) in zip([-0.2, 0.2], [("seen by opus-mt", COLORS["dec"]),
                                                       ("not seen", COLORS["muted"])]):
        values = seen_scores[seen_scores.part == part_name][metric].to_numpy()
        drawn = ax.bar(x + offset, values, width=0.38, color=color, label=part_name)
        label_format = "%.1f" if metric == "BLEU" else "%.3f"
        ax.bar_label(drawn, fmt=label_format, fontsize=8, padding=2)
    ax.set_xticks(x, SYSTEMS)
    ax.set_title(metric)
    ax.set_ylim(0, seen_scores[metric].max() * 1.35)  # room for the legend
axes[0].legend(fontsize=9, ncol=2, loc="upper left")
plt.tight_layout()
plt.show()

profile = pd.DataFrame({
    "seen by opus-mt": test["seen by opus-mt"],
    "English words": test.en.str.split().apply(len),
    "share with Tom": test.en.str.contains(r"\bTom\b"),
})
table(profile.groupby("seen by opus-mt").mean().reset_index(), digits=3)
```

<img class="tfs-output" src="/assets/posts/tfs/04/output-09.png" alt="Output of the code above" loading="lazy">

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">seen by opus-mt</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">English words</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">share with Tom</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">False</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">6.991</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.168</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">True</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">6.166</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.228</td></tr></tbody></table></div>

Both models do better on the seen part, and ours gains more than opus-mt-en-ru, although it saw none of it. The seen part is simply easier: the older part of Tatoeba, with shorter sentences and more Tom.

Against that baseline, opus-mt-en-ru's memory doesn't show. It reproduces the reference word for word a little more often on the seen pairs, but no more than our model does.

So the competitor did see a quarter of the exam, and it didn't help. Sometimes the scandal turns out to be
nothing, and the only way to know is to measure it.

A leak is still a leak, though, so the comparison that counts is on the pairs opus-mt can't have seen, with the
bootstrap of section 10:

```python
unseen = ~test["seen by opus-mt"].to_numpy()
rows = []
for metric in METRICS:
    differences = paired_bootstrap("opus-mt-en-ru", "ours, beam", metric, subset=unseen)
    low, high = np.percentile(differences, [2.5, 97.5])
    rows.append({"metric": metric, "opus-mt − ours, beam": differences.mean(),
                 "95% of draws from": low, "to": high})
table(pd.DataFrame(rows), digits=3)
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">metric</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">opus-mt − ours, beam</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">95% of draws from</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">to</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">BLEU</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">5.925</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">5.489</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">6.377</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">chrF</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">7.053</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">6.705</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">7.414</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">COMET</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.066</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.064</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.067</td></tr></tbody></table></div>

On the clean part opus-mt-en-ru's lead is larger than on the whole test set, because the easier part it had
seen was the part where our model is closest to it.

## 12. Where each model goes wrong

A single number hides where the errors are. COMET scores every sentence, so we can sort the test set by
anything and watch the scores. First by length, in tokens of the English sentence:

```python
encoded = tokenizer.encode_batch(list(test.en))
test["English tokens"] = [len(encoding.ids) for encoding in encoded]
edges = [0, 5, 8, 11, 15, 20, 30, 1000]
test["length"] = pd.cut(test["English tokens"], edges)
by_length = test.groupby("length", observed=True)[[f"COMET {s}" for s in SYSTEMS]].mean()
counts = test.length.value_counts().reindex(by_length.index)

fig, ax = plt.subplots(figsize=(6.4, 3.2))
labels = []
for interval in by_length.index:
    if interval.right == edges[-1]:
        labels.append(f"over {interval.left}")
    else:
        labels.append(f"{interval.left + 1}–{interval.right}")
for system, color in zip(SYSTEMS, [COLORS["enc"], COLORS["dec"], COLORS["data"]]):
    ax.plot(labels, by_length[f"COMET {system}"], marker="o", color=color, label=system)
ax.set_xlabel("length of the English sentence, tokens")
ax.set_ylabel("COMET, mean")
ax.set_title("Long sentences")
ax.legend(fontsize=9)
plt.show()
print("pairs per length:", dict(zip(labels, counts.tolist())))
```

<img class="tfs-output" src="/assets/posts/tfs/04/output-10.png" alt="Long sentences" loading="lazy">

<pre class="tfs-printed" markdown="0"><code>pairs per length: {&#x27;1–5&#x27;: 3520, &#x27;6–8&#x27;: 15656, &#x27;9–11&#x27;: 10614, &#x27;12–15&#x27;: 4383, &#x27;16–20&#x27;: 1196, &#x27;21–30&#x27;: 576, &#x27;over 30&#x27;: 440}</code></pre>

opus-mt-en-ru declines slowly with length. Our model is close to it on short sentences and falls apart on long ones: it was trained on Tatoeba, where long sentences are rare (lesson 3 even dropped the longest), and it has never had to keep a long sentence together.

Few test sentences are long, so this hardly moves the average, which is exactly why averages need breaking down.

What do the failures look like? Our model's lowest-scoring translations:

```python
distinct = test.drop_duplicates("en")
worst = distinct.nsmallest(6, "COMET ours, beam")
table(worst[["en", "ours, beam", "COMET ours, beam"]])
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">en</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">ours, beam</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">COMET ours, beam</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">AlphaZero is a chess player that developed its creative style solely by playing games against itself.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Дельльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльльль</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.16</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">In the starting position of the game, White has the rooks in a1 and h1, the knights in b1 and g1, the bishops in c1 and f1, the queen in d1, and the king in e1. Its eight pawns occupy the second row of the board, from a2 to h2.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Начинаю ставить партию в паузу, у Уайру и в печь в столети и в столочке отры.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.24</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Avoid exposure to direct sunlight.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Извия экспортировку экспортивность.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.24</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">A state that entrusts its protection to mercenaries will sooner or later perish, as did ancient Rome, whose inhabitants only wanted to eat and entertain themselves, and left the protection of their native land to the Germanic tribes. The latter betrayed their employers as soon as it became profitable. A real, strong enemy retreats only when all the inhabitants of the country are ready to stand up against it.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Государство, которое доверие при пристуют ко власти, или времён при при при при при при при при при изучении своих обязательствах стабибибибибиле при при при при при при при при при при при при при при при при при при при</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.25</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Australia is now considering removing humpback whales from the endangered species list because of their growing numbers.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Австралия сейчас рассматривает колосс-похимирные шапки с корпуса из-за того, что у неё реализа того, чтобы выбрасывать чисровые чисровые чистые числы из-за того, чтобы вычивать чи</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.25</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">White: rook on a1, pawn on b6, king on c8. Black: pawns on a7 and b7, king on a8, bishop on b8. White puts the opponent in zugzwang by playing the rook to a6, after which Black has only two options: take the rook on a6 or move his bishop to any other square on the diagonal b8-h2. In the first case, White advances his pawn from b6 to b7 and checkmates. In the second situation, the white rook takes the black pawn on a7, with the same result.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Бесчастный поломанный паузу, находящийся из бровку, а не полому.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.25</td></tr></tbody></table></div>

Chess positions, a paragraph about ancient Rome, whales on the endangered list: long or specialised sentences,
nothing like Tatoeba's everyday ones. On some the decoder degenerates into repeating a syllable or a word, a
failure beam search can't fix, because the model itself gives the repetition a high probability.

Second, words our model has rarely or never seen. For every test sentence, find its rarest English word and
how often that word occurs in our training data:

```python
path = hf_hub_download("xmarva/tfs-tatoeba-en-ru", "data/train.parquet",
                       repo_type="dataset")
train_en = pd.read_parquet(path, columns=["en"]).en
word_counts = Counter()
for sentence in train_en:
    word_counts.update(re.findall(r"[a-z']+", sentence.lower()))

rarest = []
for sentence in test.en:
    counts_here = [word_counts[w] for w in re.findall(r"[a-z']+", sentence.lower())]
    rarest.append(min(counts_here) if counts_here else 0)
RARITY = ["0", "1–2", "3–9", "10–99", "100–999", "1000+"]
test["rarest word"] = pd.cut(rarest, [-1, 0, 2, 9, 99, 999, 10**9], labels=RARITY)
rows = []
for label, part in test.groupby("rarest word", observed=True):
    rows.append({"rarest word, times in training": label, "pairs": len(part),
                 "COMET ours, beam": part["COMET ours, beam"].mean(),
                 "COMET opus-mt-en-ru": part["COMET opus-mt-en-ru"].mean()})
table(pd.DataFrame(rows), digits=3)
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">rarest word, times in training</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">pairs</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">COMET ours, beam</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">COMET opus-mt-en-ru</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">0</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1,617</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.569</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.852</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">1–2</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1,198</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.630</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.879</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">3–9</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2,259</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.727</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.900</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">10–99</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">8,884</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.866</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.929</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">100–999</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">15,357</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.923</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.943</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">1000+</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">7,070</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.943</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.950</td></tr></tbody></table></div>

With every word familiar, our model is nearly as good as opus-mt-en-ru. With a word it has seen once or never,
it guesses, and COMET drops far below opus-mt-en-ru, which was trained on thousands of times more text.

And the other way round: sentences where our model beats opus-mt-en-ru by the widest margin.

```python
distinct = test.drop_duplicates("en").copy()
distinct["ours − opus-mt"] = distinct["COMET ours, beam"] - distinct["COMET opus-mt-en-ru"]
ours_higher = (distinct["ours − opus-mt"] > 0).mean()
print(f"ours scores higher on {ours_higher:.0%} of the sentences")
table(distinct.nlargest(6, "ours − opus-mt")[["en", "ours, beam", "opus-mt-en-ru"]])
```

<pre class="tfs-printed" markdown="0"><code>ours scores higher on 25% of the sentences</code></pre>

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">en</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">ours, beam</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">opus-mt-en-ru</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">I shook my head no.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Я покачал головой.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Я ударился головой, нет.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">What&#x27;s your name and address?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Как твоё имя и адрес?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Как вас зовут и как вас зовут?</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">How often should I get an oil change?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Как часто мне следует поменять масло?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Как часто я должен получать мелочь?</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Mary tried to compose herself.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мэри пыталась успокоиться.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мэри пыталась сочинить себя.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">We&#x27;re up to our ears in debt.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мы по уши в долгах.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мы в долгу перед нашими ушами.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Tom and I are bosom buddies.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мы с Томом приятели.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мы с Томом дружим с грудями.</td></tr></tbody></table></div>

Our model wins on about a quarter of the sentences, and look at where it wins most: idioms. opus-mt-en-ru translates *bosom buddies* and *up to our ears* word by word, and gives Mary *обезболивающие* (painkilling) eyes for hazel ones.

Our model learned these phrases from Tatoeba, where they come up again and again. A small model trained on the right data beats a bigger one trained on everything, on the data's home ground.

## 13. Save and publish

Lesson 5 changes the model and needs a baseline to beat: these translations and their scores. They go to the
Hub as a small dataset, every test pair with the three translations, the sentence-level COMET of each and the
flag of section 11.

```python
COLUMNS = ["en", "ru", *SYSTEMS, *[f"COMET {s}" for s in SYSTEMS], "seen by opus-mt"]
RESULTS_DIR = WORK_DIR / "translations"
RESULTS_DIR.mkdir(exist_ok=True)
test[COLUMNS].to_parquet(RESULTS_DIR / "test.parquet", index=False)

reloaded = pd.read_parquet(RESULTS_DIR / "test.parquet")
assert reloaded["ours, beam"].tolist() == test["ours, beam"].tolist()
print("saved", len(reloaded), "rows:", list(reloaded.columns))
```

<pre class="tfs-printed" markdown="0"><code>saved 36385 rows: [&#x27;en&#x27;, &#x27;ru&#x27;, &#x27;ours, greedy&#x27;, &#x27;ours, beam&#x27;, &#x27;opus-mt-en-ru&#x27;, &#x27;COMET ours, greedy&#x27;, &#x27;COMET ours, beam&#x27;, &#x27;COMET opus-mt-en-ru&#x27;, &#x27;seen by opus-mt&#x27;]</code></pre>

```python
summary = pd.DataFrame(scores).round(3).to_markdown(index=False)
DATA_CARD = f"""---
license: cc-by-2.0
language: [en, ru]
tags: [translation, evaluation, comet]
---
# Translations of the course's English–Russian test set, with COMET scores

Made in lesson 4 of the course "Transformers: from scratch to production":
https://github.com/xmarva/transformers-from-scratch

- test pairs: xmarva/tfs-tatoeba-en-ru (the Tatoeba Challenge test set, v2023-09-26)
- "ours, greedy", "ours, beam": xmarva/tfs-transformer-en-ru, greedy and 4 beams
- "opus-mt-en-ru": Helsinki-NLP/opus-mt-en-ru with its 4 beams, normalized
- COMET: Unbabel/wmt22-comet-da, sentence level
- "seen by opus-mt": the pair is in OPUS Tatoeba v20190709 (July 2019)

{summary}

BLEU and chrF are sacreBLEU's defaults (tok 13a, chrF2), one reference per pair.
"""

HUB_USER = os.environ.get("TFS_HUB_USER")
hf_token = env.secret("HF_TOKEN")

if HUB_USER and hf_token and not SMOKE:
    api = HfApi(token=hf_token)
    data_repo = f"{HUB_USER}/tfs-translations-en-ru"
    api.create_repo(data_repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(RESULTS_DIR), path_in_repo="data", repo_id=data_repo,
                      repo_type="dataset")
    card = DATA_CARD.encode()
    api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=data_repo,
                    repo_type="dataset")
    print("published", data_repo)
else:
    print("not publishing; lesson 5 downloads the course's copy:")
    print("xmarva/tfs-translations-en-ru")
```

<pre class="tfs-printed" markdown="0"><code>not publishing; lesson 5 downloads the course&#x27;s copy:
xmarva/tfs-translations-en-ru</code></pre>

## References

1. {:#ref-1} Marie, B., Fujita, A., Rubino, R. (2021). [Scientific Credibility of Machine Translation Research: A Meta-Evaluation of 769 Papers](https://aclanthology.org/2021.acl-long.566/). ACL. <a class="ref-back" href="#cite-1" aria-label="Back to the text">&#8617;</a>
2. {:#ref-2} Tiedemann, J. (2020). [The Tatoeba Translation Challenge – Realistic Data Sets for Low Resource and Multilingual MT](https://aclanthology.org/2020.wmt-1.139/). WMT. <a class="ref-back" href="#cite-2" aria-label="Back to the text">&#8617;</a>
3. {:#ref-3} Papineni, K., Roukos, S., Ward, T., Zhu, W.-J. (2002). [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/). ACL. <a class="ref-back" href="#cite-3" aria-label="Back to the text">&#8617;</a>
4. {:#ref-4} Popović, M. (2015). [chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/). WMT. <a class="ref-back" href="#cite-4" aria-label="Back to the text">&#8617;</a>
5. {:#ref-5} Popović, 2017. [aclanthology.org](https://aclanthology.org/W17-4770/). <a class="ref-back" href="#cite-5" aria-label="Back to the text">&#8617;</a>
6. {:#ref-6} Rei, R., Stewart, C., Farinha, A. C., Lavie, A. (2020). [COMET: A Neural Framework for MT Evaluation](https://aclanthology.org/2020.emnlp-main.213/). EMNLP. <a class="ref-back" href="#cite-6" aria-label="Back to the text">&#8617;</a>
7. {:#ref-7} Conneau, A. et al. (2020). [Unsupervised Cross-lingual Representation Learning at Scale](https://aclanthology.org/2020.acl-main.747/). ACL. *XLM-R.* <a class="ref-back" href="#cite-7" aria-label="Back to the text">&#8617;</a>
8. {:#ref-8} Martins, A., Astudillo, R. (2016). [From Softmax to Sparsemax](https://arxiv.org/abs/1602.02068). ICML. <a class="ref-back" href="#cite-8" aria-label="Back to the text">&#8617;</a>
9. {:#ref-9} Freitag, M. et al. (2021). [Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation](https://aclanthology.org/2021.tacl-1.87/). TACL. *MQM.* <a class="ref-back" href="#cite-9" aria-label="Back to the text">&#8617;</a>
10. {:#ref-10} Freitag, M. et al. (2022). [Results of WMT22 Metrics Shared Task: Stop Using BLEU – Neural Metrics Are Better and More Robust](https://aclanthology.org/2022.wmt-1.2/). WMT. <a class="ref-back" href="#cite-10" aria-label="Back to the text">&#8617;</a>
11. {:#ref-11} Kocmi, T. et al. (2021). [To Ship or Not to Ship: An Extensive Evaluation of Automatic Metrics for Machine Translation](https://aclanthology.org/2021.wmt-1.57/). WMT. *Pairwise accuracy.* <a class="ref-back" href="#cite-11" aria-label="Back to the text">&#8617;</a>
12. {:#ref-12} Koehn, P. (2004). [Statistical Significance Tests for Machine Translation Evaluation](https://aclanthology.org/W04-3250/). EMNLP. *The paired bootstrap.* <a class="ref-back" href="#cite-12" aria-label="Back to the text">&#8617;</a>
13. {:#ref-13} Post, M. (2018). [A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/). WMT. *sacreBLEU and its signatures.*
14. {:#ref-14} Popović, M. (2016). [chrF deconstructed: β parameters and n-gram weights](https://aclanthology.org/W16-2341/). WMT. *Why β = 2.*
15. {:#ref-15} Rei, R. et al. (2022). [COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task](https://aclanthology.org/2022.wmt-1.52/). WMT. *The wmt22-comet-da model.*
16. {:#ref-16} Kocmi, T. et al. (2024). [Findings of the WMT24 General Machine Translation Shared Task: The LLM Era Is Here but MT Is Not Solved Yet](https://aclanthology.org/2024.wmt-1.1/). WMT.
17. {:#ref-17} Amrhein, C., Sennrich, R. (2022). [Identifying Weaknesses in Machine Translation Metrics Through Minimum Bayes Risk Decoding: A Case Study for COMET](https://aclanthology.org/2022.aacl-main.83/). AACL. *COMET and named entities.*
18. {:#ref-18} Zouhar, V. et al. (2024). [Pitfalls and Outlooks in Using COMET](https://aclanthology.org/2024.wmt-1.121/). WMT.
19. {:#ref-19} Tiedemann, J., Thottingal, S. (2020). [OPUS-MT: Building open translation services for the World](https://aclanthology.org/2020.eamt-1.61/). EAMT.
{: .tfs-refs}



<p class="tfs-neighbours"><a href="/blog/2026/transformers-from-scratch-03-training/">← Lesson 3: Training & decoding</a> <a href="/blog/2026/transformers-from-scratch-05-llama/">Lesson 5: What Llama changed →</a></p>

{% endraw %}
