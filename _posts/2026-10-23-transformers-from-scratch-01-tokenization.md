---
layout: post
title: "How language models read: tokenization and BPE from scratch"
date: 2026-10-23 10:00:00
description: "What GPT-4o actually receives when you type, why Russian costs more tokens than English, and byte-pair encoding written from scratch."
tags: [transformers-from-scratch, tokenization, bpe, nlp, llm]
categories: machine-learning
thumbnail: /assets/posts/tfs/01/cover.png
og_image: https://xmarva.github.io/assets/posts/tfs/01/cover.png
stylesheets: [/assets/css/tfs-course.css]
toc:
  beginning: true
---
{% raw %}

<p class="tfs-links"><strong>Transformers: from scratch to production</strong> · lesson 1 of 10 · <a href="/teaching/">all lessons</a> · <a href="https://www.kaggle.com/code/qmarva/transformers-from-scratch-01-tokenization">run on Kaggle</a> · <a href="https://colab.research.google.com/github/xmarva/transformers-from-scratch/blob/main/notebooks/colab/01_tokenization.ipynb">open in Colab</a> · <a href="https://github.com/xmarva/transformers-from-scratch">code on GitHub</a></p>

Chatbots used to be famously bad at one question: *how many r's are in "strawberry"?* It looks like a test of
attention to detail. Mostly it tests something the model cannot see. Here is what GPT-4o actually receives
when you ask it:

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/01/hero-light.png" width="760" alt="The question as GPT-4o receives it: eight tokens, each an integer id; the word &#x27; strawberry&#x27; with its leading space is a single token" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/01/hero-dark.png" width="760" alt="The question as GPT-4o receives it: eight tokens, each an integer id; the word &#x27; strawberry&#x27; with its leading space is a single token" loading="lazy">

Eight integers. The word <code markdown="0">&nbsp;strawberry</code>, together with the space in front of it, arrives as one of them, and its letters are not in the input at all.

Whatever the model knows about how that integer is spelled, it had to pick up indirectly, from the rare places in its training text where someone spelled the word out.

The program that turns text into integers is the **tokenizer**. It is built before the model, from a large pile of text, and then frozen: every model you have heard of reads the world through a fixed list of pieces chosen that way.

The choice of pieces decides what a request costs, which languages a model handles well, and, as you just saw, what it can count.

This course builds a Transformer that translates English into Russian, and in this lesson you build the
tokenizer it will read and write with. Russian is a good language to learn this on: as you are about to see,
tokenizers trained mostly on English handle it badly, and the reasons are instructive.

**What you'll do**

1. Take apart the tokenizers of GPT-2, GPT-4, Llama 3, DeepSeek-V3 and GPT-4o.
2. Prepare an English–Russian corpus with training, validation and test splits that don't leak into each other.
3. Write byte-pair encoding (BPE), the algorithm behind all of those tokenizers, from scratch.
4. Train the course's tokenizer with Hugging Face `tokenizers` and put it next to GPT-4o's.

**What you need**: Python and some pandas. No previous lessons. A free CPU runtime is enough; the notebook runs
in about ten minutes.

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
import collections
import csv
import json
import os
import re
import time
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from tokenizers import Regex, Tokenizer, decoders, models, normalizers
from tokenizers import pre_tokenizers, trainers
```

## 1. Inside the tokenizers of real models

OpenAI publishes its tokenizers in the `tiktoken` library. `Xenova/gpt-4o` on the Hugging Face Hub is a port of
GPT-4o's tokenizer to the `tokenizers` library, which this course uses throughout; it produces the same tokens.
Loading it downloads one JSON file of about 10 MB: the whole tokenizer, rules and vocabulary.

```python
gpt4o = Tokenizer.from_pretrained("Xenova/gpt-4o")

question = "How many r's are in strawberry?"
print(gpt4o.encode(question).ids)
```

<pre class="tfs-printed" markdown="0"><code>[5299, 1991, 428, 885, 553, 306, 101830, 30]</code></pre>

The same eight numbers as in the picture above. Each one is a position in GPT-4o's vocabulary, and `decode` turns positions back into text.

Decoding the ids one at a time shows the piece behind each of them. The function below does that for a few texts and draws the pieces as chips, with the id under each chip:

```python
def show_tokens(tokenizer, texts):
    rows, row_ids = [], []
    for text in texts:
        # add_special_tokens=False: some tokenizers (Llama's) add a "start of text" token
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        rows.append([tokenizer.decode([token_id]) for token_id in ids])
        row_ids.append(ids)
    token_chips(rows, ids=row_ids, labels=[repr(text) for text in texts])
    plt.show()


spellings = ["strawberry", " strawberry", " Strawberry", " STRAWBERRY", " strawberries"]
show_tokens(gpt4o, spellings)
```

<img class="tfs-output" src="/assets/posts/tfs/01/output-01.png" alt="Output of the code above" loading="lazy">

Five spellings of one word, five unrelated sequences of numbers (`␣` marks a space inside a token). In the middle of a sentence the word comes with a space in front of it and is a single token.

At the very start of a message there is no space, and the same word becomes three tokens, `st`, `raw` and `berry`. The capitalized word has a token of its own, the word in capitals falls apart into four pieces, and the plural is yet another token.

Nothing in these numbers says that they are the same word. The model works it out from how the tokens are used, and it learns how each of them is spelled the same indirect way.

Tokenizers keep the space attached to the word that follows it because that saves a token per word; the price is that a word has a different token at the start of a text than in the middle.

### The same sentence in two languages

API prices, context windows and the speed of generation are all measured in tokens. So, does a sentence cost
the same in every language? Take one sentence and its Russian translation:

> The researcher said she was exhausted.
>
> Исследовательница сказала, что вымоталась.

and count their tokens with the tokenizers of five models released between 2019 and 2024. All of them are on
the Hugging Face Hub. Meta's own Llama repositories require signing in, so Llama 3's tokenizer comes from a
public copy of Llama 3.2, which uses the same tokenizer.

```python
LLM_TOKENIZERS = {
    "GPT-2 (2019)": "openai-community/gpt2",
    "GPT-4 (2023)": "Xenova/gpt-4",
    "Llama 3 (2024)": "unsloth/Llama-3.2-1B",
    "DeepSeek-V3 (2024)": "deepseek-ai/DeepSeek-V3",
    "GPT-4o (2024)": "Xenova/gpt-4o",
}

llm_tokenizers = {}
for name, repo in LLM_TOKENIZERS.items():
    llm_tokenizers[name] = Tokenizer.from_pretrained(repo)

EN_EXAMPLE = "The researcher said she was exhausted."
RU_EXAMPLE = "Исследовательница сказала, что вымоталась."


def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


rows = []
for name, tokenizer in llm_tokenizers.items():
    rows.append(
        {
            "model": name,
            "vocabulary": tokenizer.get_vocab_size(),
            "English": count_tokens(tokenizer, EN_EXAMPLE),
            "Russian": count_tokens(tokenizer, RU_EXAMPLE),
        }
    )
example_counts = pd.DataFrame(rows)
table(example_counts)
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">model</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">vocabulary</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">English</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">Russian</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">GPT-2 (2019)</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">50,257</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">7</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">46</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">GPT-4 (2023)</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">100,263</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">7</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">19</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Llama 3 (2024)</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">128,256</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">7</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">14</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">DeepSeek-V3 (2024)</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">128,815</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">7</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">14</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">GPT-4o (2024)</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">200,000</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">7</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">11</td></tr></tbody></table></div>

The same numbers as a chart. We will draw this chart again at the end of the lesson, with one more tokenizer in it, so it gets a function.

English is teal and Russian is rose here and in every plot of the course: they are the colors of the encoder, which reads English, and of the decoder, which writes Russian.

```python
def plot_token_counts(counts, title):
    fig, ax = plt.subplots(figsize=(6.4, 0.9 + 0.6 * len(counts)))
    rows = np.arange(len(counts))
    bar = 0.38
    english = ax.barh(rows - 0.2, counts.English, bar, color=COLORS["enc"], label="English")
    russian = ax.barh(rows + 0.2, counts.Russian, bar, color=COLORS["dec"], label="Russian")
    ax.bar_label(english, fmt="{:.3g}", padding=4, fontsize=9, color=COLORS["muted"])
    ax.bar_label(russian, fmt="{:.3g}", padding=4, fontsize=9, color=COLORS["muted"])

    labels = []
    for model, vocabulary in zip(counts.model, counts.vocabulary):
        labels.append(f"{model} · {vocabulary // 1000}k")
    ax.set_yticks(rows, labels)
    ax.invert_yaxis()  # the first model on top

    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("tokens")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.show()


plot_token_counts(example_counts, "One sentence, two languages, five tokenizers")
```

<img class="tfs-output" src="/assets/posts/tfs/01/output-02.png" alt="Output of the code above" loading="lazy">

Every tokenizer spends about the same on the English sentence. The Russian one costs several times more in GPT-2, and the newer tokenizers narrow the gap.

Note the vocabulary sizes next to the names: the gap closed as the vocabularies grew from about fifty thousand pieces to two hundred thousand, enough room for frequent pieces of many languages. Even in GPT-4o, though, the Russian sentence is more expensive than the English one.

The extra cost has a name, the **token tax**: [Petrov et al. (2023)](https://arxiv.org/abs/2305.15425) measured it across hundreds of languages and found that the same text can take up to 15 times as many tokens as in English.

Where does GPT-2's bill come from? Look at the pieces it cuts the first Russian word into:

```python
show_tokens(llm_tokenizers["GPT-2 (2019)"], ["Исследовательница"])
```

<img class="tfs-output" src="/assets/posts/tfs/01/output-03.png" alt="Output of the code above" loading="lazy">

Most letters are tokens of their own, and some, like the first one, are cut in half: each `�` is a fragment of a letter that is not valid text on its own. That is not a display bug.

GPT-2 tokenizes **bytes**. It writes text in UTF-8, where every Cyrillic letter takes two bytes, and starts from the 256 possible bytes as its alphabet.

GPT-2 was trained on WebText, a crawl from which, as the GPT-2 paper notes, OpenAI deliberately removed non-English pages, and its merges rarely glue Russian bytes into anything longer than a letter.

GPT-4o works on bytes too, but among its 200 000 pieces there are plenty of Russian letters, syllables and
words. Here is a word from the Russian sentence and one more:

```python
show_tokens(gpt4o, [" вымоталась", " недопонимание"])
```

<img class="tfs-output" src="/assets/posts/tfs/01/output-04.png" alt="Output of the code above" loading="lazy">

Fewer tokens, but look at where the cuts fall in the second word, *недопонимание*, "misunderstanding". It is built as `недо` (under-) + `понима` (the stem of *понимать*, to understand) + `ние` (which turns a verb into a noun), and GPT-4o's pieces cut across all three.

That isn't a mistake: a tokenizer learns statistics, not grammar, and it learns the statistics of the text it was built from.

By the end of this lesson you will have a tokenizer of your own, trained in a couple of minutes on
English–Russian sentences, with sixteen thousand pieces instead of two hundred thousand. On the Russian sentences
of our data it will need fewer tokens than GPT-4o.

Every tokenizer you've seen so far, and ours too, turns text into numbers in the same four stages:

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/01/pipeline-light.png" width="760" alt="Raw text is normalized, split into words, merged into subword tokens by BPE and mapped to integer ids that the model reads" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/01/pipeline-dark.png" width="760" alt="Raw text is normalized, split into words, merged into subword tokens by BPE and mapped to integer ids that the model reads" loading="lazy">

We build them in this order. First, though, the tokenizer needs text to learn from.

## 2. The data

[Tatoeba](https://tatoeba.org) ("for example" in Japanese) is a collection of example sentences and their
translations, started by Trang Ho in 2006 and written by volunteers ever since. The sentences are short and
everyday, which is what a small translator can learn from, and they are licensed under CC BY 2.0 FR. We use two
pieces of it:

- **training data**: the English–Russian pairs prepared by the sentence-transformers project;
- **test data**: the English–Russian test set of the
  [Tatoeba Translation Challenge](https://arxiv.org/abs/2010.06354) (Tiedemann, 2020), a public benchmark. In
  lesson 4 we compare our model with published systems on it.

Both downloads are pinned to an exact revision, so the numbers in this notebook don't change when the sources
do.

```python
TRAIN_REVISION = "cec1343ab5a7a8befe99af4a2d0ca847b6c84743"

dataset = load_dataset(
    "sentence-transformers/parallel-sentences-tatoeba",
    "en-ru",
    split="train",
    revision=TRAIN_REVISION,
)
train_raw = dataset.to_pandas()
train_raw = train_raw.rename(columns={"english": "en", "non_english": "ru"})
```

The test set is a plain text file. Each line holds four fields separated by tabs: the two language codes, the
English sentence and its Russian translation.

```python
TEST_URL = (
    "https://raw.githubusercontent.com/Helsinki-NLP/Tatoeba-Challenge/"
    "5509af44a74016748c26f41ab164b64f48445f3f/data/test/eng-rus/test.txt"
)

test_raw = pd.read_csv(
    TEST_URL,
    sep="\t",
    names=["src_lang", "tgt_lang", "en", "ru"],
    quoting=csv.QUOTE_NONE,  # quote marks belong to the sentences, not to the file format
    dtype=str,
    keep_default_na=False,  # a sentence like "NA" must stay a sentence
)
test_raw = test_raw[["en", "ru"]]

print("training pairs:", len(train_raw))
print("test pairs:", len(test_raw))
table(train_raw.sample(6, random_state=SEED))
```

<pre class="tfs-printed" markdown="0"><code>training pairs: 541666
test pairs: 36385</code></pre>

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">en</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">ru</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Tom says that he knows you.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Том говорит, что знает тебя.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Do you want us to help?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Хотите, мы поможем?</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Mary said that she&#x27;d come.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мэри сказала, что придёт.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Do you think Cookie would be a good name for a dog?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Как ты думаешь, Куки - хорошая кличка для собаки?</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">A lot of people think Buenos Aires is located in Brazil.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Многие думают, что Буэнос-Айрес находится в Бразилии.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">What you&#x27;re saying now has nothing to do with the question.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">То, что ты сейчас говоришь, не имеет отношения к вопросу.</td></tr></tbody></table></div>

Before cleaning anything, read some data; every corpus has habits. Tatoeba's most famous one is a man called Tom.

Prolific contributors use him (and Mary) as the default name in example sentences, so often that in 2019 the project [discussed banning them](https://blog.tatoeba.org/2019/08/should-we-stop-sentences-with-tom-and.html) and decided not to. He is everywhere:

```python
for name in ["Tom", "Mary"]:
    share = train_raw.en.str.contains(rf"\b{name}\b").mean()
    print(f"{name} appears in {share:.1%} of the English sentences")
```

<pre class="tfs-printed" markdown="0"><code>Tom appears in 22.8% of the English sentences
Mary appears in 4.5% of the English sentences</code></pre>

This matters more than it seems. Russian declines names: Tom is *Том*, *Тома*, *Тому*, *Томом*, *о Томе* depending on his role in the sentence, so our translator will get plenty of practice with Russian cases on one name.

It will also be much better at sentences about Tom than about anyone else. A model learns the distribution of its data, quirks included.

### One sentence, many translations

A second property of the data matters for everything that follows: one English sentence often comes with
several Russian translations, and one Russian sentence can translate several English ones. Count the
translations of every English sentence:

```python
translations_per_sentence = train_raw.groupby("en").size()

print("distinct English sentences:", len(translations_per_sentence))
print("translations per sentence, on average:", round(translations_per_sentence.mean(), 2))
```

<pre class="tfs-printed" markdown="0"><code>distinct English sentences: 378850
translations per sentence, on average: 1.43</code></pre>

And look at the record holder:

```python
champion = translations_per_sentence.idxmax()
its_translations = list(train_raw[train_raw.en == champion].ru)
print(repr(champion), "has", len(its_translations), "translations")

rows = []
for start in range(0, len(its_translations), 5):
    rows.append(its_translations[start : start + 5])
token_chips(rows, role="dec", mark_spaces=False)
plt.show()
```

<pre class="tfs-printed" markdown="0"><code>&quot;He&#x27;s stupid.&quot; has 25 translations</code></pre>

<img class="tfs-output" src="/assets/posts/tfs/01/output-05.png" alt="Output of the code above" loading="lazy">

Volunteers have been creative. None of these translations is wrong; they differ in register, imagery and
rudeness. Keep this in mind for lesson 4: a metric that compares the model's output with a single reference
translation punishes perfectly good answers.

For now the consequence is about splitting the data, and we come back to it after cleaning.

### Normalization

The same text can be typed in different ways. Count the quote marks and dashes in the corpus:

```python
char_counts = collections.Counter("".join(train_raw.en) + "".join(train_raw.ru))

rows = []
for char in "\"«»„“”'‘’-–—…":
    if char_counts[char] > 0:
        name = unicodedata.name(char)
        rows.append({"character": char, "Unicode name": name, "count": char_counts[char]})
table(pd.DataFrame(rows))
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">character</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">Unicode name</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">count</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">&quot;</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">QUOTATION MARK</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">17,843</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">«</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">LEFT-POINTING DOUBLE ANGLE QUOTATION MARK</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2,211</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">»</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2,210</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">„</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">DOUBLE LOW-9 QUOTATION MARK</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">6</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">“</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">LEFT DOUBLE QUOTATION MARK</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">83</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">”</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">RIGHT DOUBLE QUOTATION MARK</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">78</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">&#x27;</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">APOSTROPHE</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">191,728</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">‘</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">LEFT SINGLE QUOTATION MARK</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">33</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">’</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">RIGHT SINGLE QUOTATION MARK</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">292</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">-</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">HYPHEN-MINUS</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">32,740</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">–</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">EN DASH</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">383</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">—</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">EM DASH</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">5,224</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">…</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">HORIZONTAL ELLIPSIS</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">26</td></tr></tbody></table></div>

Some variants are a matter of style (`«ёлочки»` versus `"straight quotes"`). Others are invisible: the letter `й`
can be stored as one code point, or as `и` followed by a combining breve, a mark that the font draws on top of
the previous letter.

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/01/normalization-light.png" width="760" alt="The letter й stored two ways: one code point U+0439, or и U+0438 followed by a combining breve U+0306; NFKC turns the second into the first" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/01/normalization-dark.png" width="760" alt="The letter й stored two ways: one code point U+0439, or и U+0438 followed by a combining breve U+0306; NFKC turns the second into the first" loading="lazy">

Every reader sees the same letter. Python sees two different strings:

```python
one_code_point = "\u0439"
two_code_points = "\u0438\u0306"

print(one_code_point, two_code_points)
print(one_code_point == two_code_points)
print(len(one_code_point), len(two_code_points))
```

<pre class="tfs-printed" markdown="0"><code>й й
False
1 2</code></pre>

A tokenizer that doesn't know this would learn two tokens for one letter, and the model would have to learn
twice that they mean the same. Unicode defines **normal forms** for exactly this. Normal form **NFKC** composes
letters with their accents, and also maps "compatibility" characters to plain ones: subscript digits,
non-breaking spaces, the one-character ellipsis `…`.

```python
print(unicodedata.normalize("NFKC", two_code_points) == one_code_point)
print(unicodedata.normalize("NFKC", "H₂O"))
nbsp = "\u00a0"  # a non-breaking space
print(repr(unicodedata.normalize("NFKC", "two" + nbsp + "words")))
```

<pre class="tfs-printed" markdown="0"><code>True
H2O
&#x27;two words&#x27;</code></pre>

On top of NFKC we make four choices of our own. They are written down because the model will inherit them:
whatever we erase here, it will never be able to read or write.

1. Drop the combining marks that survive NFKC. Tatoeba marks stress in some Russian sentences (`То́му`), which
   is a pronunciation hint, not spelling.
2. Use straight quotes and apostrophes, and one kind of dash, `—`.
3. Read a hyphen surrounded by spaces as a dash.
4. Collapse runs of whitespace into one space.

```python
QUOTES = {"«": '"', "»": '"', "„": '"', "“": '"', "”": '"', "‘": "'", "’": "'", "–": "—"}


def normalize(text):
    text = unicodedata.normalize("NFKC", text)

    # Unicode category "Mn" is a combining mark, such as a stress accent
    kept = [char for char in text if unicodedata.category(char) != "Mn"]
    text = "".join(kept)

    for fancy, plain in QUOTES.items():
        text = text.replace(fancy, plain)

    text = re.sub(r"\s+-\s+", " — ", text)  # " - " is a dash
    text = re.sub(r"\s+", " ", text)  # one space between words
    return text.strip()
```

Try it on a few sentences before trusting it with half a million:

```python
examples = [
    "Он сказал: «Я не знаю».",
    "То́му было нечего пить.",
    "Wait a second - what?",
    "It’s 5 o’clock…",
]

rows = []
for before in examples:
    rows.append({"before": before, "after": normalize(before)})
table(pd.DataFrame(rows))
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">before</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">after</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Он сказал: «Я не знаю».</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Он сказал: &quot;Я не знаю&quot;.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">То́му было нечего пить.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Тому было нечего пить.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Wait a second - what?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Wait a second — what?</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">It’s 5 o’clock…</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">It&#x27;s 5 o&#x27;clock...</td></tr></tbody></table></div>

Now normalize both sides of both datasets, and count how many training sentences actually changed:

```python
train = train_raw.copy()
train["en"] = train_raw.en.map(normalize)
train["ru"] = train_raw.ru.map(normalize)

test = test_raw.copy()
test["en"] = test_raw.en.map(normalize)
test["ru"] = test_raw.ru.map(normalize)

changed = (train.en != train_raw.en).sum() + (train.ru != train_raw.ru).sum()
print("changed:", changed, "of", 2 * len(train), "training sentences")
```

<pre class="tfs-printed" markdown="0"><code>changed: 10225 of 1083332 training sentences</code></pre>

### Filters, and one we don't apply

A pair is dropped when it is clearly broken: Cyrillic on the English side, no Cyrillic on the Russian side, or
more than 40 words, which in Tatoeba means a paragraph that slipped in among the sentences. Each check below gives
a column of `True`/`False`, one value per pair:

```python
CYRILLIC = r"[а-яё]"


def word_count(sentences):
    return sentences.str.split().str.len()


cyrillic_in_english = train.en.str.contains(CYRILLIC, case=False)
no_cyrillic_in_russian = ~train.ru.str.contains(CYRILLIC, case=False)
too_long = (word_count(train.en) > 40) | (word_count(train.ru) > 40)

print("Cyrillic on the English side:", cyrillic_in_english.sum())
print("no Cyrillic on the Russian side:", no_cyrillic_in_russian.sum())
print("longer than 40 words:", too_long.sum())
```

<pre class="tfs-printed" markdown="0"><code>Cyrillic on the English side: 22
no Cyrillic on the Russian side: 0
longer than 40 words: 243</code></pre>

What does Cyrillic on the English side look like?

```python
table(train[cyrillic_in_english].head(2))
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">en</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">ru</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">You have added a comment, not a translation. To add a translation, click on the &quot;あ→а&quot; icon above the sentence.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Вы добавили комментарий, а не перевод. Чтобы добавить перевод, щёлкните по значку &quot;あ→а&quot; над предложением.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">I sent her birthday present bу post.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Я послал ей подарок ко дню рождения по почте.</td></tr></tbody></table></div>

The first pair is Tatoeba's own interface text that slipped into the data: the icon `あ→а` ends in a Cyrillic `а`. The second is a typo nobody can see: the `у` in `bу post` is the Cyrillic letter, which looks exactly like the Latin `y`.

A model would learn these lookalikes as different symbols, so we drop all the broken pairs, and then the duplicates: after normalization some pairs became identical.

```python
broken = cyrillic_in_english | no_cyrillic_in_russian | too_long
train = train[~broken]

size_before = len(train)
train = train.drop_duplicates()
print("duplicates removed:", size_before - len(train))
```

<pre class="tfs-printed" markdown="0"><code>duplicates removed: 66</code></pre>

Translation pipelines often also drop pairs whose lengths differ too much, on the theory that a translation
three times longer than its source must be misaligned. Before applying a recipe, look at what it would remove:

```python
length_ratio = train.en.str.len() / train.ru.str.len()
flagged = train[(length_ratio > 3) | (length_ratio < 1 / 3)]

print("the length-ratio filter would remove", len(flagged), "pairs, for example:")
table(flagged.sample(min(6, len(flagged)), random_state=SEED))
```

<pre class="tfs-printed" markdown="0"><code>the length-ratio filter would remove 73 pairs, for example:</code></pre>

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">en</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">ru</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">At ease.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Чувствуйте себя непринуждённо.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">The volume is over there.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Том там.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Is she sleeping?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Спит?</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">You took the words right out of my mouth.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">С языка снял.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Is it too small?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мал?</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">I&#x27;m the one who&#x27;s right.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Прав я.</td></tr></tbody></table></div>

Most of them are good translations: idioms and terse colloquial answers, exactly what makes a translator sound
human. The filter was designed for noisy corpora crawled from the web; Tatoeba is written by people, so we keep
these pairs.

> A cleaning step is a hypothesis about your data. Test it by reading what it removes.

### The test set must stay unseen

The Tatoeba test set was cut from the same pool of sentences as our training data. If a test sentence also appears in training, the model can recite the answer and the test score stops measuring translation.

This problem, **contamination**, is one that large-model papers take seriously: the GPT-4 technical report has a whole appendix of contamination checks for its benchmarks.

Comparing exact strings is not enough: `Хватит!` and `хватит.` are the same sentence for our purposes. We
compare **fingerprints** instead: lowercase, without punctuation and spaces, and with `ё` read as `е` (Russian
texts use the two letters interchangeably).

```python
def fingerprint(text):
    text = text.casefold()  # lowercase that also handles special cases like ß
    text = text.replace("ё", "е")
    return re.sub(r"[\W_]+", "", text)  # drop everything that is not a letter or a digit


print(fingerprint("Хватит!"), fingerprint("хватит."), fingerprint("Ты безнадёжна!"))
```

<pre class="tfs-printed" markdown="0"><code>хватит хватит тыбезнадежна</code></pre>

A training pair leaks when its English or its Russian sentence has the fingerprint of a test sentence. Compare
the two ways of checking:

```python
test_en_prints = set(test.en.map(fingerprint))
test_ru_prints = set(test.ru.map(fingerprint))

exact_en = train.en.isin(set(test.en))
exact_ru = train.ru.isin(set(test.ru))
print("pairs that share a sentence with the test set")
print("  exact strings:", (exact_en | exact_ru).sum())

printed_en = train.en.map(fingerprint).isin(test_en_prints)
printed_ru = train.ru.map(fingerprint).isin(test_ru_prints)
leaks = printed_en | printed_ru
print("  fingerprints:", leaks.sum())

train = train[~leaks]
print("training pairs left:", len(train))
```

<pre class="tfs-printed" markdown="0"><code>pairs that share a sentence with the test set
  exact strings: 21990
  fingerprints: 22873
training pairs left: 518463</code></pre>

### A validation split that doesn't leak

We also need a validation set: data the model never trains on, used to choose settings and to stop training at
the right moment. The usual move is to hold out a random 2% of the pairs. Here that would be a mistake, and the
picture shows why.

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/01/leakage-light.png" width="760" alt="A random split puts the pair &#x27;Cut it out!&#x27; / &#x27;Прекрати!&#x27; into validation while both sentences also appear in training pairs; splitting by connected component keeps the whole group on one side" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/01/leakage-dark.png" width="760" alt="A random split puts the pair &#x27;Cut it out!&#x27; / &#x27;Прекрати!&#x27; into validation while both sentences also appear in training pairs; splitting by connected component keeps the whole group on one side" loading="lazy">

These are real pairs from our data. Think of every sentence as a node and every translation pair as an edge.

A random split cuts edges independently, so a validation pair like `Cut it out!` → `Прекрати!` still has both of its sentences in training pairs. The model has seen the source and has learned to produce the target, and validation would reward memory.

The fix is to split whole **connected components**: groups of sentences linked by translation, directly or through each other. A classic data structure finds them, **union–find**. Every sentence starts as a group of its own.

Every pair says "these two sentences belong together", and we merge their groups. To tell whether two sentences are already in one group, each group has a representative, its *root*: every sentence keeps a link to a "parent" sentence, and following the parents leads to the root.

```python
class UnionFind:
    def __init__(self):
        self.parent = {}  # sentence -> its parent; a root is its own parent

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x  # a sentence seen for the first time is a group of one

        root = x
        while self.parent[root] != root:
            root = self.parent[root]

        # path compression: point everything on the way straight at the root
        while self.parent[x] != root:
            next_x = self.parent[x]
            self.parent[x] = root
            x = next_x
        return root

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_a] = root_b  # the two groups become one
```

The second loop in `find` is what keeps it fast. Without it, a chain of parents can grow long, and every lookup
walks the whole chain. With it, a lookup shortens the chain for everyone after it:

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/01/union-find-light.png" width="820" alt="Path compression: find(&#x27;Cut it out!&#x27;) walks from Cut it out! through Перестань! and Прекрати! to the root Прекрати это!; afterwards both Cut it out! and Перестань! point at the root directly" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/01/union-find-dark.png" width="820" alt="Path compression: find(&#x27;Cut it out!&#x27;) walks from Cut it out! through Перестань! and Прекрати! to the root Прекрати это!; afterwards both Cut it out! and Перестань! point at the root directly" loading="lazy">

That is the actual state of the structure halfway through the example below. A sentence is stored together
with its language, `("en", "Stop it.")`, because the same string can occur on both sides (names, numbers). Run
it on the pairs from the pictures, plus one unrelated pair:

```python
example_pairs = [
    ("Cut it out!", "Перестань!"),
    ("Stop it.", "Перестань!"),
    ("Cut it out!", "Прекрати!"),
    ("Stop that!", "Прекрати!"),
    ("Stop it.", "Прекрати это!"),
    ("Stop that!", "Прекрати это!"),
    ("Cut it out!", "Хватит!"),
    ("Enough!", "Хватит!"),
    ("I'm hungry.", "Я голоден."),
]

groups = UnionFind()
for en, ru in example_pairs:
    groups.union(("en", en), ("ru", ru))

rows = []
for en, ru in example_pairs:
    root_language, root_sentence = groups.find(("en", en))
    rows.append({"English": en, "Russian": ru, "group (its root)": root_sentence})
table(pd.DataFrame(rows))
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">English</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">Russian</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">group (its root)</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Cut it out!</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Перестань!</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Хватит!</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Stop it.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Перестань!</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Хватит!</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Cut it out!</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Прекрати!</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Хватит!</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Stop that!</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Прекрати!</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Хватит!</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Stop it.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Прекрати это!</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Хватит!</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Stop that!</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Прекрати это!</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Хватит!</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Cut it out!</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Хватит!</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Хватит!</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Enough!</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Хватит!</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Хватит!</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">I&#x27;m hungry.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Я голоден.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Я голоден.</td></tr></tbody></table></div>

All eight pairs from the picture end up in one group, and the unrelated pair forms its own.

With path compression alone, $$m$$ operations on $$n$$ sentences cost $$O(m \log n)$$ in the worst case. Also
attaching the smaller group under the larger one (union by rank) gives the famous near-linear bound
$$O(m\,\alpha(n))$$, where $$\alpha$$ is the inverse Ackermann function, which stays below 5 for any $$n$$ that fits
in our universe<sup class="cite">[<a href="#ref-1" id="cite-1" data-ref="1" aria-label="Reference 1: Tarjan, R. E. (1975). Efficiency of a good but not linear set union algorithm. Journal of the ACM, 22(2).">1</a>]</sup>. For our data compression alone is
plenty. Now the real thing: every training pair, and a group label for each.

```python
start = time.perf_counter()

groups = UnionFind()
for en, ru in zip(train.en, train.ru):
    groups.union(("en", en), ("ru", ru))

labels = []
for en in train.en:
    root = groups.find(("en", en))
    labels.append(str(root))  # a pair is labeled with the root of its group
train = train.assign(group=labels)
print("done in", round(time.perf_counter() - start, 1), "s")

group_sizes = train.groupby("group").size()
print("groups:", len(group_sizes))
print("groups with more than one pair:", (group_sizes > 1).sum())
print("pairs in the largest group:", group_sizes.max())
```

<pre class="tfs-printed" markdown="0"><code>done in 1.7 s
groups: 324245
groups with more than one pair: 106467
pairs in the largest group: 84</code></pre>

The largest group is a whole family of ways to tell someone to stop, or to go away. A few of its English and
Russian sentences:

```python
largest = train[train.group == group_sizes.idxmax()]
some_english = sorted(set(largest.en))[:12]
some_russian = sorted(set(largest.ru))[:12]

token_chips([some_english], role="enc", mark_spaces=False)
token_chips([some_russian], role="dec", mark_spaces=False)
plt.show()
```

<img class="tfs-output" src="/assets/posts/tfs/01/output-06.png" alt="Output of the code above" loading="lazy">

<img class="tfs-output" src="/assets/posts/tfs/01/output-07.png" alt="Output of the code above" loading="lazy">

Now compare the two ways of splitting. `leak_rate` measures the share of validation pairs whose English or
Russian sentence also occurs somewhere in training:

```python
def leak_rate(train_part, valid_part):
    seen_english = set(train_part.en)
    seen_russian = set(train_part.ru)
    leaked = valid_part.en.isin(seen_english) | valid_part.ru.isin(seen_russian)
    return leaked.mean()
```

First the random split: every pair goes to validation with probability 2%.

```python
VAL_SHARE = 0.02
rng = np.random.default_rng(SEED)

random_valid = rng.random(len(train)) < VAL_SHARE
random_leak = leak_rate(train[~random_valid], train[random_valid])
print(f"random split: {random_leak:.1%} of validation pairs leak")
```

<pre class="tfs-printed" markdown="0"><code>random split: 57.2% of validation pairs leak</code></pre>

Then the split by groups: we pick 2% of the groups, and their pairs, all of them, go to validation.

```python
all_groups = train.group.unique()
n_valid_groups = int(VAL_SHARE * len(all_groups))
valid_groups = rng.choice(all_groups, size=n_valid_groups, replace=False)
in_valid = train.group.isin(set(valid_groups))

valid = train[in_valid].drop(columns="group")
train = train[~in_valid].drop(columns="group")

group_leak = leak_rate(train, valid)
print(f"split by groups: {group_leak:.1%} of validation pairs leak")
assert group_leak == 0  # stops the notebook if a single pair leaks

print("train:", len(train), "validation:", len(valid), "test:", len(test))
```

<pre class="tfs-printed" markdown="0"><code>split by groups: 0.0% of validation pairs leak
train: 508093 validation: 10370 test: 36385</code></pre>

> Split by whatever must not be shared. Here that's sentences, so we split by groups of linked sentences; for
> medical images it would be patients, for code it would be repositories.

The splits are final. We save them; at the end of the lesson they are published, so that lesson 2 can start
from exactly these sentences.

```python
DATA_DIR = WORK_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

splits = {"train": train, "validation": valid, "test": test}
for name, frame in splits.items():
    frame.reset_index(drop=True).to_parquet(DATA_DIR / f"{name}.parquet")

print(sorted(path.name for path in DATA_DIR.iterdir()))
```

<pre class="tfs-printed" markdown="0"><code>[&#x27;test.parquet&#x27;, &#x27;train.parquet&#x27;, &#x27;validation.parquet&#x27;]</code></pre>

## 3. Why not words, and why not characters

Two designs come to mind before BPE does: one token per word, or one token per character. Both have been used,
and both fail in instructive ways.

A word tokenizer gives each distinct word an id. Its problem is that the number of distinct words never stops
growing. Watch it grow as we read through the training data, recording the number of distinct words after every
thousand words read:

```python
WORD = re.compile(r"\w+")  # a run of letters or digits


def vocabulary_growth(sentences, every=1_000):
    seen = set()
    words_read = 0
    n_points, v_points = [], []
    for sentence in sentences:
        for word in WORD.findall(sentence.lower()):
            seen.add(word)
            words_read += 1
            if words_read % every == 0:
                n_points.append(words_read)
                v_points.append(len(seen))
    return np.array(n_points), np.array(v_points)


shuffled = train.sample(frac=1, random_state=SEED)
growth = {"en": vocabulary_growth(shuffled.en), "ru": vocabulary_growth(shuffled.ru)}
```

Empirically, vocabulary growth follows **Heaps' law**:

$$V(N) = K N^{\beta}, \qquad 0 < \beta < 1,$$

where $$N$$ is the number of words read, $$V(N)$$ the number of distinct words among them, and $$K$$, $$\beta$$
constants of the language and the text. $$\beta$$ is the interesting one: it says how fast new word forms keep
appearing. With $$\beta = 0.5$$, reading four times more text doubles the vocabulary.

Taking logarithms turns the law into a straight line, $$\log V = \log K + \beta \log N$$, so we can estimate
$$\beta$$ as the slope of a least-squares line through the points in log–log coordinates; `np.polyfit(x, y, 1)`
returns the slope and the intercept.

```python
fig, ax = plt.subplots()

for lang, color in [("en", COLORS["enc"]), ("ru", COLORS["dec"])]:
    n, v = growth[lang]
    beta, log_k = np.polyfit(np.log(n), np.log(v), 1)
    label = f"{lang}: β = {beta:.2f}, {v[-1]:,} distinct words"
    ax.loglog(n, v, color=color, label=label)
    fitted = np.exp(log_k) * n**beta  # the fitted law, dashed
    ax.loglog(n, fitted, color=color, linestyle=(0, (2, 3)), linewidth=1.2)

ax.set_xlabel("words read, N")
ax.set_ylabel("distinct words, V(N)")
ax.set_title("Russian keeps producing new word forms")
ax.legend(loc="upper left")
plt.show()
```

<img class="tfs-output" src="/assets/posts/tfs/01/output-08.png" alt="Russian keeps producing new word forms" loading="lazy">

Russian grows faster and ends with several times more distinct words, although the Russian side has fewer words
in total. The reason is morphology: `понимать, понимаю, понимаешь, понимание, недопонимание, понимания, …` are
all different strings to a word tokenizer, while English mostly reuses `understand`.

The other side of the same coin is the long tail. Count how many distinct words occur exactly once in all of the
training data (linguists call them *hapax legomena*, "said once"):

```python
word_freqs = {}
for lang in ["en", "ru"]:
    counts = collections.Counter()
    for sentence in train[lang]:
        counts.update(WORD.findall(sentence.lower()))
    word_freqs[lang] = counts

    seen_once = 0
    for count in counts.values():
        if count == 1:
            seen_once += 1
    print(f"{lang}: {seen_once / len(counts):.0%} of distinct words occur once")
```

<pre class="tfs-printed" markdown="0"><code>en: 33% of distinct words occur once
ru: 42% of distinct words occur once</code></pre>

A large share of the vocabulary is words the model would see once in all of training, too rarely to learn
anything about them. A real vocabulary has to stop somewhere anyway. Keep the 30 000 most frequent training
words and count how many validation words fall outside, the **out-of-vocabulary (OOV)** rate:

```python
def oov_rate(counts, valid_sentences, size=30_000):
    vocabulary = set()
    for word, _ in counts.most_common(size):
        vocabulary.add(word)

    total, unknown = 0, 0
    for sentence in valid_sentences:
        for word in WORD.findall(sentence.lower()):
            total += 1
            if word not in vocabulary:
                unknown += 1
    return unknown / total


for lang in ["en", "ru"]:
    rate = oov_rate(word_freqs[lang], valid[lang])
    print(f"{lang}: {rate:.2%} of validation words are unknown")
```

<pre class="tfs-printed" markdown="0"><code>en: 0.35% of validation words are unknown
ru: 3.33% of validation words are unknown</code></pre>

Every unknown word becomes the same placeholder, and the model can neither read nor write it. For Russian that
happens an order of magnitude more often than for English, and those are exactly the rare inflected forms a
translator must produce.

The opposite extreme, one token per character, has no unknown words at all, but the sequences get long:

```python
for lang in ["en", "ru"]:
    chars = valid[lang].str.len().mean()
    words = word_count(valid[lang]).mean()
    print(f"{lang}: {chars:.1f} characters, {words:.1f} words per sentence;")
    print(f"    {chars / words:.1f} times longer in characters")
```

<pre class="tfs-printed" markdown="0"><code>en: 31.0 characters, 6.2 words per sentence;
    5.0 times longer in characters
ru: 31.0 characters, 5.4 words per sentence;
    5.7 times longer in characters</code></pre>

Length is expensive. In lesson 2 you will see that self-attention compares every position with every other, so
its cost grows with the square of the sequence length $$T$$:

$$\text{cost} \propto T^2 \quad\Rightarrow\quad \frac{\text{cost}_{\text{chars}}}{\text{cost}_{\text{words}}} =
\left(\frac{T_{\text{chars}}}{T_{\text{words}}}\right)^2.$$

Sequences $$k$$ times longer make attention $$k^2$$ times more expensive: with the ratios above, a character model
pays for attention dozens of times more than a word model, and it has to learn spelling before it can learn
meaning.

**Subword** tokenization is the compromise: frequent words stay whole, rare words are split into frequent
pieces, and every string can still be written with the basic alphabet. Byte-pair encoding is the most widely
used way to find those pieces.

## 4. BPE from scratch

BPE was invented in 1994 by Philip Gage as a way to compress files: find the most common pair of adjacent bytes, replace it everywhere with a byte value the file doesn't use, repeat.

In 2016 [Sennrich, Haddow and Birch](https://arxiv.org/abs/1508.07909) noticed that the same procedure, run on the letters of words, finds good subword units for translation. Today it builds the tokenizers of GPT-2, GPT-4o, Llama 3 and DeepSeek-V3, all of which you took apart in section 1.

The idea fits in one picture:

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/01/bpe-loop-light.png" width="560" alt="The BPE training loop: count adjacent symbol pairs, take the most frequent, merge it everywhere, add the new symbol to the vocabulary, repeat until the vocabulary is full" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/01/bpe-loop-dark.png" width="560" alt="The BPE training loop: count adjacent symbol pairs, take the most frequent, merge it everywhere, add the new symbol to the vocabulary, repeat until the vocabulary is full" loading="lazy">

Start with every word spelled out as single characters. Then repeat: find the pair of adjacent symbols that occurs most often in the corpus, merge it into one new symbol everywhere, and add that symbol to the vocabulary.

Frequent words end up as one symbol, because merging their pieces paid off again and again; rare words stay split into pieces that other words share.

Formally, let $$f(w)$$ be the frequency of word $$w$$ in the corpus and $$n_{ab}(w)$$ the number of times symbol $$a$$ is
immediately followed by symbol $$b$$ in the current spelling of $$w$$. The frequency of the pair is

$$c(a, b) = \sum_{w} f(w)\, n_{ab}(w),$$

and each step merges

$$(a^*, b^*) = \operatorname*{arg\,max}_{(a, b)} c(a, b).$$

If two pairs are equally frequent we take the one that compares greater as a pair of Python strings, that is, by Unicode code points.

Any fixed rule works; what matters is that it is deterministic, so the same corpus always yields the same tokenizer. After $$M$$ merges the vocabulary holds the initial alphabet $$\Sigma$$ plus one symbol per merge: $$\lvert\mathcal{V}\rvert = \lvert\Sigma\rvert + M$$.

### Pre-tokenization

Merges never cross word boundaries, so we first split text into words and punctuation. A word keeps the space in front of it, written as `▁` (the convention of SentencePiece, [Kudo and Richardson, 2018](https://arxiv.org/abs/1808.06226)).

It's the same trick you saw in GPT-4o's `␣strawberry`: with the spaces inside the tokens, decoding is plain concatenation, and the model can tell the beginning of a word from the middle of one.

The regular expression has two alternatives: <code markdown="0">&nbsp;?\w+</code> is an optional space followed by letters or digits, and
<code markdown="0">&nbsp;?[^\w\s]+</code> is an optional space followed by punctuation (anything that is neither a word character nor
whitespace).

```python
PRE_TOKEN = re.compile(r" ?\w+| ?[^\w\s]+")


def pretokenize(text):
    pieces = PRE_TOKEN.findall(" " + text)  # the leading space marks the first word too
    return [piece.replace(" ", "▁") for piece in pieces]


print(pretokenize(RU_EXAMPLE))
print(pretokenize("Let's go — now!"))
```

<pre class="tfs-printed" markdown="0"><code>[&#x27;▁Исследовательница&#x27;, &#x27;▁сказала&#x27;, &#x27;,&#x27;, &#x27;▁что&#x27;, &#x27;▁вымоталась&#x27;, &#x27;.&#x27;]
[&#x27;▁Let&#x27;, &quot;&#x27;&quot;, &#x27;s&#x27;, &#x27;▁go&#x27;, &#x27;▁—&#x27;, &#x27;▁now&#x27;, &#x27;!&#x27;]</code></pre>

### Counting pairs on a toy corpus

Before touching half a million sentences, run every step on three Russian words whose counts you can check by
hand. A word is stored as a tuple of its symbols, with its frequency:

```python
toy = {
    tuple("▁понимание"): 5,
    tuple("▁недопонимание"): 3,
    tuple("▁понимать"): 4,
}


def pair_counts(words):
    counts = collections.Counter()
    for word, freq in words.items():
        for left, right in zip(word, word[1:]):  # every adjacent pair in the word
            counts[left, right] += freq
    return counts


pair_counts(toy).most_common(5)
```

<pre class="tfs-printed" markdown="0"><code>[((&#x27;н&#x27;, &#x27;и&#x27;), 20),
 ((&#x27;п&#x27;, &#x27;о&#x27;), 12),
 ((&#x27;о&#x27;, &#x27;н&#x27;), 12),
 ((&#x27;и&#x27;, &#x27;м&#x27;), 12),
 ((&#x27;м&#x27;, &#x27;а&#x27;), 12)]</code></pre>

`н` + `и` wins: it occurs twice in `понимание` (five times), twice in `недопонимание` (three times) and once in
`понимать` (four times), which gives $$2 \cdot 5 + 2 \cdot 3 + 1 \cdot 4 = 20$$, the formula for $$c(a, b)$$ above.

Picking the winner, with our rule for ties:

```python
def most_frequent_pair(counts):
    best_pair, best_count = None, 0
    for pair, count in counts.items():
        if count > best_count or (count == best_count and pair > best_pair):
            best_pair, best_count = pair, count
    return best_pair


most_frequent_pair(pair_counts(toy))
```

<pre class="tfs-printed" markdown="0"><code>(&#x27;н&#x27;, &#x27;и&#x27;)</code></pre>

Merging a pair rewrites a word, scanning it from left to right:

```python
def merge_word(word, pair):
    merged = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            merged.append(word[i] + word[i + 1])  # two symbols become one
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


before = tuple("▁недопонимание")
after = merge_word(before, ("н", "и"))
token_chips([before, after], labels=["before", "after"])
plt.show()
```

<img class="tfs-output" src="/assets/posts/tfs/01/output-09.png" alt="Output of the code above" loading="lazy">

### The straightforward trainer

Training is these three steps in a loop:

```python
def train_bpe_simple(words, n_merges):
    merges = []
    for _ in range(n_merges):
        counts = pair_counts(words)
        if not counts:  # every word is a single symbol, nothing left to merge
            break
        pair = most_frequent_pair(counts)
        merges.append(pair)

        merged_words = {}
        for word, freq in words.items():
            merged_words[merge_word(word, pair)] = freq
        words = merged_words
    return merges


train_bpe_simple(toy, 6)
```

<pre class="tfs-printed" markdown="0"><code>[(&#x27;н&#x27;, &#x27;и&#x27;),
 (&#x27;п&#x27;, &#x27;о&#x27;),
 (&#x27;по&#x27;, &#x27;ни&#x27;),
 (&#x27;пони&#x27;, &#x27;м&#x27;),
 (&#x27;поним&#x27;, &#x27;а&#x27;),
 (&#x27;▁&#x27;, &#x27;понима&#x27;)]</code></pre>

Six merges on the toy corpus, each one built on the previous ones: `ни`, then `по`, then `по` + `ни` = `пони`,
and so on until the whole stem `▁понима`. Now a real sample: we count the words of 40 000 sentences, English and
Russian together, and run 300 merges.

```python
def word_counts(sentences):
    counts = collections.Counter()
    for sentence in sentences:
        counts.update(pretokenize(sentence))

    words = {}
    for word, freq in counts.items():
        words[tuple(word)] = freq
    return words


SAMPLE_SENTENCES = 40_000
SAMPLE_MERGES = 300

sample = pd.concat([train.en, train.ru]).sample(SAMPLE_SENTENCES, random_state=SEED)
sample_words = word_counts(sample)
print("distinct words in the sample:", len(sample_words))

start = time.perf_counter()
simple_merges = train_bpe_simple(sample_words, SAMPLE_MERGES)
simple_seconds = time.perf_counter() - start
print(len(simple_merges), "merges in", round(simple_seconds, 1), "s")
```

<pre class="tfs-printed" markdown="0"><code>distinct words in the sample: 23926
300 merges in 12.0 s</code></pre>

The first merges are the most frequent pieces of both languages, interleaved:

```python
first_30 = ["".join(pair) for pair in simple_merges[:30]]
token_chips(
    [first_30[0:10], first_30[10:20], first_30[20:30]],
    labels=["merges 1–10", "11–20", "21–30"],
)
plt.show()
```

<img class="tfs-output" src="/assets/posts/tfs/01/output-10.png" alt="Output of the code above" loading="lazy">

Word starts like `▁t`, `▁н`, `▁п` come first, then frequent syllables and whole short words. `▁T` seems to appear
twice in the second row: one is the Latin `T`, the other the Cyrillic `Т` that starts `Том`. They look identical
and are different characters, which is why we never compare text by how it looks.

The trainer is correct but slow. Every merge recounts all pairs in all words, so the cost is $$O(M \cdot L)$$,
where $$L$$ is the total number of symbols in all distinct words. Both factors grow for the real run: the full
training set has several times more distinct words than this sample, and we want thousands of merges, not
hundreds.

### The incremental trainer

Most of that recounting is wasted, because a merge changes very little. Take one word of the toy corpus and
compare its pairs before and after merging `н` + `и`:

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/01/bpe-update-light.png" width="760" alt="Merging н + и in ▁понимать, which occurs 4 times: the pairs (о, н), (н, и), (и, м) lose 4 each, the new pairs (о, ни), (ни, м) gain 4 each, and all other pairs keep their counts" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/01/bpe-update-dark.png" width="760" alt="Merging н + и in ▁понимать, which occurs 4 times: the pairs (о, н), (н, и), (и, м) lose 4 each, the new pairs (о, ни), (ни, м) gain 4 each, and all other pairs keep their counts" loading="lazy">

Only the pairs that touch the merge change, and only in words that contain the pair. So instead of recounting,
we keep two tables between steps:

- `pair_freq`: how often each pair occurs in the corpus, $$c(a, b)$$;
- `words_with_pair`: for each pair, which words contain it.

After a merge we visit only the affected words. For each of them, `remove_pairs` takes all its pairs out of the tables, the word is rewritten, and `add_pairs` puts its new pairs back.

Removing and re-adding the whole word is simpler than computing the exact difference, and the result is the same: the pairs that didn't change go out and come back with the same count.

```python
def train_bpe(word_freqs, n_merges):
    words = [list(word) for word in word_freqs]  # current spelling of every word
    freqs = list(word_freqs.values())
    pair_freq = collections.Counter()
    words_with_pair = collections.defaultdict(set)  # pair -> indices of words with it

    def add_pairs(i):
        for pair in zip(words[i], words[i][1:]):
            pair_freq[pair] += freqs[i]
            words_with_pair[pair].add(i)

    def remove_pairs(i):
        for pair in zip(words[i], words[i][1:]):
            pair_freq[pair] -= freqs[i]
            if pair_freq[pair] == 0:
                del pair_freq[pair]
            words_with_pair[pair].discard(i)

    for i in range(len(words)):
        add_pairs(i)

    merges = []
    for _ in range(n_merges):
        if not pair_freq:
            break
        pair = most_frequent_pair(pair_freq)
        merges.append(pair)
        for i in list(words_with_pair[pair]):  # only the words that contain the pair change
            remove_pairs(i)
            words[i] = list(merge_word(words[i], pair))
            add_pairs(i)

    final_spelling = {}
    for word, spelling in zip(word_freqs, words):
        final_spelling[word] = tuple(spelling)
    return merges, final_spelling
```

A faster algorithm is only useful if it computes the same thing. Same input, same number of merges:

```python
start = time.perf_counter()
fast_merges, _ = train_bpe(sample_words, SAMPLE_MERGES)
fast_seconds = time.perf_counter() - start

assert fast_merges == simple_merges
print("identical merges")
print("simple:", round(simple_seconds, 1), "s, incremental:", round(fast_seconds, 2), "s")
print(round(simple_seconds / fast_seconds), "times faster")
```

<pre class="tfs-printed" markdown="0"><code>identical merges
simple: 12.0 s, incremental: 0.46 s
26 times faster</code></pre>

Now the full training set, both languages together, and 8 000 merges. One shared vocabulary for English and
Russian lets the model share one embedding table between encoder and decoder later on, and the two languages
have more in common than their alphabets suggest: digits, punctuation, names, borrowed words.

```python
N_MERGES = 8_000
all_words = word_counts(pd.concat([train.en, train.ru]))

start = time.perf_counter()
merges, segmentation = train_bpe(all_words, N_MERGES)
seconds = round(time.perf_counter() - start)
print(len(merges), "merges over", len(all_words), "distinct words in", seconds, "s")

alphabet = set()
for word in all_words:
    alphabet.update(word)
vocabulary_size = len(alphabet) + len(merges)
print("vocabulary:", len(alphabet), "characters +", len(merges), "merges")
print("          =", vocabulary_size, "symbols")
```

<pre class="tfs-printed" markdown="0"><code>8000 merges over 119567 distinct words in 15 s
vocabulary: 242 characters + 8000 merges
          = 8242 symbols</code></pre>

### Encoding: replay the merges in order

To tokenize new text we don't count anything. We apply the learned merges in the order they were learned: in each word, find the adjacent pair that was merged earliest during training, its **rank**, merge it, and repeat until no adjacent pair is in the merge table.

Ranks matter because merges build on each other: `▁недо` exists only because `▁н`, `▁не` and `до` were merged first.

```python
ranks = {}
for rank, pair in enumerate(merges):
    ranks[pair] = rank  # pair -> when it was merged: 0, 1, 2, ...


def encode_word(word, ranks, steps=None):
    symbols = list(word)
    while True:
        # find the adjacent pair with the lowest rank
        best_i, best_rank = None, None
        for i in range(len(symbols) - 1):
            rank = ranks.get((symbols[i], symbols[i + 1]))
            if rank is not None and (best_rank is None or rank < best_rank):
                best_i, best_rank = i, rank

        if best_i is None:  # no pair left that training ever merged
            return symbols

        symbols[best_i : best_i + 2] = [symbols[best_i] + symbols[best_i + 1]]
        if steps is not None:
            steps.append((list(symbols), best_i))  # remember every step, to draw it below


steps = []
encode_word("▁недопонимание", ranks, steps)
rows = [symbols for symbols, _ in steps]
token_chips(rows, strong=lambda row, col, _: col == steps[row][1], size=11)
plt.show()
```

<img class="tfs-output" src="/assets/posts/tfs/01/output-11.png" alt="Output of the code above" loading="lazy">

Each row is one merge; the highlighted symbol is the one just created. Replaying merges by rank should reproduce
exactly the segmentation that training arrived at. Rather than believe it, check it for every word in the
training data:

```python
mismatches = []
for word, trained in segmentation.items():
    if encode_word("".join(word), ranks) != list(trained):
        mismatches.append(word)

assert not mismatches, mismatches[:5]
print("encoding reproduces the training segmentation for all", len(segmentation), "words")
```

<pre class="tfs-printed" markdown="0"><code>encoding reproduces the training segmentation for all 119567 words</code></pre>

Encoding a sentence is pre-tokenization plus `encode_word`, and decoding is concatenation with `▁` turned back
into spaces. Nothing is lost on the way; the round trip over the whole validation set confirms it:

```python
def encode(text):
    tokens = []
    for word in pretokenize(text):
        tokens.extend(encode_word(word, ranks))
    return tokens


def decode(tokens):
    text = "".join(tokens).replace("▁", " ")
    return text[1:]  # drop the space we added in front of the first word


for sentence in pd.concat([valid.en, valid.ru]):
    assert decode(encode(sentence)) == sentence

print(encode(RU_EXAMPLE))
```

<pre class="tfs-printed" markdown="0"><code>[&#x27;▁И&#x27;, &#x27;сс&#x27;, &#x27;ледо&#x27;, &#x27;ва&#x27;, &#x27;тель&#x27;, &#x27;ница&#x27;, &#x27;▁сказала&#x27;, &#x27;,&#x27;, &#x27;▁что&#x27;, &#x27;▁вы&#x27;, &#x27;мо&#x27;, &#x27;т&#x27;, &#x27;алась&#x27;, &#x27;.&#x27;]</code></pre>

### What the merges learned

Nobody told BPE about prefixes or suffixes, yet some appear on their own, because a frequent morpheme is a
frequent string:

```python
words = [
    "▁misunderstanding",
    "▁недопонимание",
    "▁исследовательница",
    "▁вымоталась",
    "▁researcher",
    "▁untranslatable",
]
segmented = [encode_word(word, ranks) for word in words]
token_chips(segmented, labels=[word[1:] for word in words], size=11)
plt.show()
```

<img class="tfs-output" src="/assets/posts/tfs/01/output-12.png" alt="Output of the code above" loading="lazy">

Some splits look like a linguist's: `недо`, or the feminine suffix `ница`, the piece that turns *исследователь* (a researcher) into a woman researcher. Others don't (`st` + `and`).

BPE optimizes frequency, not meaning. The model will learn what the pieces mean from context; what the tokenizer controls is that frequent things are short and every rare thing can still be spelled.

## 5. The production tokenizer

Our implementation is readable and correct. It would also be slow at serving time and easy to break: the normalization lives in one function, the pre-tokenization in another, and nothing forces a server to call them the same way as training did.

Hugging Face [`tokenizers`](https://huggingface.co/docs/tokenizers), written in Rust, packs the whole pipeline into one object that is saved as one file:

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/01/hf-tokenizer-light.png" width="760" alt="A Hugging Face tokenizer holds the normalizer, the pre-tokenizer and the BPE model for encoding, and the decoder for decoding; each stage replaces one function from the lesson, and all are saved in tokenizer.json" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/01/hf-tokenizer-dark.png" width="760" alt="A Hugging Face tokenizer holds the normalizer, the pre-tokenizer and the BPE model for encoding, and the decoder for decoding; each stage replaces one function from the lesson, and all are saved in tokenizer.json" loading="lazy">

The tokenizers of section 1 are such files; you downloaded five of them. The rule we follow:

> The tokenizer owns its normalization. Then training, evaluation and the deployed service can't normalize
> differently, because there is no separate step to forget.

First the normalizer, rebuilt from the same choices as `normalize`, step for step:

```python
hf_normalizer = normalizers.Sequence(
    [
        normalizers.NFKC(),
        normalizers.Replace(Regex(r"\p{Mn}"), ""),  # combining marks
        *[normalizers.Replace(fancy, plain) for fancy, plain in QUOTES.items()],
        normalizers.Replace(Regex(r"\s+-\s+"), " — "),
        normalizers.Replace(Regex(r"\s+"), " "),
        normalizers.Strip(),
    ]
)

print(hf_normalizer.normalize_str("Он сказал:   «Я не знаю» - и ушёл…"))
```

<pre class="tfs-printed" markdown="0"><code>Он сказал: &quot;Я не знаю&quot; — и ушёл...</code></pre>

"Rebuilt from the same choices" is a claim, so check it on every raw sentence we have:

```python
raw_sentences = pd.concat([train_raw.en, train_raw.ru, test_raw.en, test_raw.ru])

different = []
for sentence in raw_sentences:
    if hf_normalizer.normalize_str(sentence) != normalize(sentence):
        different.append(sentence)

assert not different, different[:5]
print("identical normalization on all", len(raw_sentences), "raw sentences")
```

<pre class="tfs-printed" markdown="0"><code>identical normalization on all 1156102 raw sentences</code></pre>

Then the pre-tokenizer: the same regular expression splits words and punctuation, and `Metaspace` turns the
spaces into `▁`. Regular-expression engines don't always agree on what a "word character" is (Python's and the
Rust library's disagree on combining marks and subscript digits), which is one more reason the normalizer
removed those first. Check this one too:

```python
hf_pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Split(Regex(r" ?\w+| ?[^\w\s]+"), behavior="isolated"),
        pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never", split=False),
    ]
)


def hf_pretokenize(text):
    pieces = hf_pre_tokenizer.pre_tokenize_str(" " + text)  # (piece, position) pairs
    return [piece for piece, _ in pieces]


normalized_sentences = pd.concat([train.en, train.ru, valid.en, valid.ru, test.en, test.ru])
for sentence in normalized_sentences:
    assert hf_pretokenize(sentence) == pretokenize(sentence), sentence
print("identical pre-tokenization on all", len(normalized_sentences), "sentences")
```

<pre class="tfs-printed" markdown="0"><code>identical pre-tokenization on all 1109696 sentences</code></pre>

The tokenizer needs four special tokens, which the model will use from lesson 2 on: `<pad>` fills shorter
sentences in a batch, `<unk>` stands for a character never seen in training, `<bos>` and `<eos>` mark the
beginning and end of a sentence. They take ids 0 to 3.

`<unk>` is the price of starting from characters rather than bytes, as GPT-2 does: a character that never
occurred in training (an emoji, a Chinese name) has no symbol. For a translator between two languages that is a
fair trade; the tokens stay readable, and no letter gets cut in half.

```python
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def build_tokenizer(vocab_size):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([hf_normalizer, normalizers.Prepend(" ")])
    tokenizer.pre_tokenizer = hf_pre_tokenizer
    tokenizer.decoder = decoders.Metaspace(
        replacement="▁", prepend_scheme="always", split=False
    )

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,  # ignore pairs seen only once
        show_progress=False,
    )
    tokenizer.train_from_iterator(list(train.en) + list(train.ru), trainer)
    return tokenizer


VOCAB_SIZE = 16_000

start = time.perf_counter()
tokenizer = build_tokenizer(VOCAB_SIZE)
seconds = round(time.perf_counter() - start, 1)
print("vocabulary of", tokenizer.get_vocab_size(), "trained in", seconds, "s")
print(tokenizer.encode(RU_EXAMPLE).tokens)
```

<pre class="tfs-printed" markdown="0"><code>vocabulary of 16000 trained in 19.3 s
[&#x27;▁И&#x27;, &#x27;сс&#x27;, &#x27;ледо&#x27;, &#x27;ватель&#x27;, &#x27;ница&#x27;, &#x27;▁сказала&#x27;, &#x27;,&#x27;, &#x27;▁что&#x27;, &#x27;▁вы&#x27;, &#x27;мот&#x27;, &#x27;алась&#x27;, &#x27;.&#x27;]</code></pre>

Compare the timings: the Rust trainer builds twice as many symbols in a fraction of the time our Python one needed for the same corpus.

Both see exactly the same words, since we checked normalization and pre-tokenization, so their merges should agree except where the two implementations make different choices: Hugging Face breaks ties between equally frequent pairs differently, and skips pairs seen only once (`min_frequency=2`). The saved tokenizer lists its merges in order; compare the first thousand with ours:

```python
saved = json.loads(tokenizer.to_str())

hf_merges = []
for merge in saved["model"]["merges"]:
    if isinstance(merge, str):  # older versions of tokenizers store a merge as "a b"
        merge = merge.split(" ")
    hf_merges.append(tuple(merge))

shared = set(hf_merges[:1000]) & set(merges[:1000])
print(len(shared), "of the first 1000 merges are the same in both")

first_difference = 0
while hf_merges[first_difference] == merges[first_difference]:
    first_difference += 1
print("the lists first differ at merge", first_difference + 1)
print("Hugging Face merged", hf_merges[first_difference])
print("and we merged", merges[first_difference])
```

<pre class="tfs-printed" markdown="0"><code>998 of the first 1000 merges are the same in both
the lists first differ at merge 339
Hugging Face merged (&#x27;р&#x27;, &#x27;ы&#x27;)
and we merged (&#x27;▁го&#x27;, &#x27;вори&#x27;)</code></pre>

Almost all of the first thousand merges are the same. Why does the first difference happen? Rebuild the corpus
as both trainers saw it at that moment: replaying the merges that came before it, with our `encode_word`, gives
every word's spelling at that step, and `pair_counts` gives the frequencies.

```python
earlier_ranks = {}
for rank, pair in enumerate(merges[:first_difference]):
    earlier_ranks[pair] = rank

corpus_at_that_step = {}
for word, freq in all_words.items():
    spelling = tuple(encode_word("".join(word), earlier_ranks))
    corpus_at_that_step[spelling] = freq

counts_then = pair_counts(corpus_at_that_step)
theirs, ours = hf_merges[first_difference], merges[first_difference]
print("Hugging Face's pair:", theirs, "occurred", counts_then[theirs], "times")
print("our pair:", ours, "occurred", counts_then[ours], "times")
```

<pre class="tfs-printed" markdown="0"><code>Hugging Face&#x27;s pair: (&#x27;р&#x27;, &#x27;ы&#x27;) occurred 11268 times
our pair: (&#x27;▁го&#x27;, &#x27;вори&#x27;) occurred 11268 times</code></pre>

A tie: both pairs were equally frequent, and the two implementations break ties differently. Same algorithm,
same data, a different rule for a coin toss.

The tokenizer is now self-contained. Feed it raw, unnormalized text, and decoding returns the normalized
sentence:

```python
messy = "  Он сказал:   «Я не  знаю» - и ушёл…"
ids = tokenizer.encode(messy).ids

print(ids)
print(repr(tokenizer.decode(ids)))
assert tokenizer.decode(ids) == normalize(messy)
assert [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] == [0, 1, 2, 3]
```

<pre class="tfs-printed" markdown="0"><code>[466, 534, 29, 619, 185, 269, 664, 5, 557, 318, 2916, 4093]
&#x27;Он сказал: &quot;Я не знаю&quot; — и ушёл...&#x27;</code></pre>

## 6. How many tokens?

Why 16 000 symbols, and not 8 000 or the 200 000 of GPT-4o? A bigger vocabulary makes sequences shorter, but
it isn't free. The first layer of the model is an **embedding table**, one learned vector per token:

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/01/embedding-light.png" width="760" alt="The embedding table has one row per token and d_model columns; a token&#x27;s id selects its row, a vector of d_model learned numbers; the table holds |V| times d_model parameters" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/01/embedding-dark.png" width="760" alt="The embedding table has one row per token and d_model columns; a token&#x27;s id selects its row, a vector of d_model learned numbers; the table holds |V| times d_model parameters" loading="lazy">

$$\text{embedding parameters} = \lvert\mathcal{V}\rvert \cdot d_{\text{model}},$$

and the last layer is the same size again, because at every output position the model scores every token in the vocabulary. Worse, a rare token gets few training examples, so its row is learned poorly.

The most famous case is <code markdown="0">&nbsp;SolidGoldMagikarp</code>: a Reddit username that was frequent enough in the text GPT-2's tokenizer was built from to become a token of its own, but that, it seems, almost never occurred in the text the models were then trained on.

In 2023, [Rumbelow and Watkins](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation) found that GPT-3 models, which share that tokenizer, couldn't even repeat the word back and answered with unrelated ones. The token is still there:

```python
show_tokens(llm_tokenizers["GPT-2 (2019)"], [" SolidGoldMagikarp"])
```

<img class="tfs-output" src="/assets/posts/tfs/01/output-13.png" alt="Output of the code above" loading="lazy">

So we measure both sides: the average number of tokens per validation sentence for several vocabulary sizes,
and the size of the embedding table for the model of lesson 2, whose vectors have $$d_{\text{model}} = 256$$
numbers.

```python
def tokens_per_sentence(tokenizer, sentences):
    encodings = tokenizer.encode_batch(list(sentences), add_special_tokens=False)
    lengths = [len(encoding.ids) for encoding in encodings]
    return np.mean(lengths)


D_MODEL = 256
VOCAB_SIZES = [2_000, 4_000, 8_000, 16_000, 32_000]

rows = []
for size in VOCAB_SIZES:
    candidate = build_tokenizer(size)
    rows.append(
        {
            "vocabulary": size,
            "embedding parameters": size * D_MODEL,
            "English tokens": tokens_per_sentence(candidate, valid.en),
            "Russian tokens": tokens_per_sentence(candidate, valid.ru),
        }
    )

sweep = pd.DataFrame(rows)
table(sweep)
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">vocabulary</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">embedding parameters</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">English tokens</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">Russian tokens</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2,000</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">512,000</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">10.93</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">11.89</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">4,000</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1,024,000</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">9.71</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">10.11</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">8,000</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2,048,000</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">8.96</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">8.80</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">16,000</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">4,096,000</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">8.52</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">7.97</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">32,000</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">8,192,000</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">8.27</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">7.43</td></tr></tbody></table></div>

```python
fig, ax = plt.subplots()
sizes = sweep.vocabulary
ax.plot(sizes, sweep["English tokens"], marker="o", color=COLORS["enc"], label="English")
ax.plot(sizes, sweep["Russian tokens"], marker="o", color=COLORS["dec"], label="Russian")
ax.axvline(VOCAB_SIZE, color=COLORS["muted"], linestyle=(0, (3, 3)), linewidth=1)

ax.set_xscale("log", base=2)
tick_labels = [f"{size // 1000}k" for size in sweep.vocabulary]
ax.set_xticks(sweep.vocabulary, tick_labels)
ax.set_xlabel("vocabulary size")
ax.set_ylabel("tokens per sentence")
ax.set_title("Diminishing returns of a bigger vocabulary")
ax.legend()
plt.show()
```

<img class="tfs-output" src="/assets/posts/tfs/01/output-14.png" alt="Diminishing returns of a bigger vocabulary" loading="lazy">

Each doubling of the vocabulary saves fewer tokens than the previous one, while the embedding table doubles every time. We keep 16 000 symbols.

Compare the last two rows of the table: going to 32 000 would save less than a token per sentence in either language, and double an embedding table that already holds millions of parameters, a sizeable share of a small translation model.

Models with billions of parameters can afford 200 000 tokens, because next to the rest of the model even a huge embedding table is small. Ours can't.

### The token tax, revisited

Back to where we started, with our tokenizer next to the five from section 1. This time we count tokens on the
whole validation set rather than on one sentence:

```python
contenders = dict(llm_tokenizers)
contenders["Ours (this lesson)"] = tokenizer

rows = []
for name, candidate in contenders.items():
    rows.append(
        {
            "model": name,
            "vocabulary": candidate.get_vocab_size(),
            "English": tokens_per_sentence(candidate, valid.en),
            "Russian": tokens_per_sentence(candidate, valid.ru),
        }
    )
validation_counts = pd.DataFrame(rows)
plot_token_counts(validation_counts, "Average tokens per validation sentence")
```

<img class="tfs-output" src="/assets/posts/tfs/01/output-15.png" alt="Output of the code above" loading="lazy">

On Russian, our tokenizer needs fewer tokens than every model in the chart, GPT-4o included, with a vocabulary a fraction of the size. That's not because we out-engineered OpenAI.

Our tokenizer was trained on exactly this kind of text in exactly these two languages; theirs has to serve code, math and dozens of languages at once.

On English we're slightly worse than all of them, which is the honest price of sharing 16 000 symbols between two languages. A tokenizer is a decision about what to be good at, and for a translator that reads English and writes Russian this one is right.

And the word from section 1, cut by GPT-4o and by ours:

```python
gpt4o_ids = gpt4o.encode(" недопонимание").ids
gpt4o_pieces = [gpt4o.decode([token_id]) for token_id in gpt4o_ids]
our_pieces = tokenizer.encode("недопонимание").tokens

token_chips([gpt4o_pieces, our_pieces], labels=["GPT-4o", "ours"])
plt.show()
```

<img class="tfs-output" src="/assets/posts/tfs/01/output-16.png" alt="Output of the code above" loading="lazy">

### Save and publish

The tokenizer is one JSON file. Saving it and loading it back must give the same tokens:

```python
TOKENIZER_PATH = WORK_DIR / "tokenizer.json"
tokenizer.save(str(TOKENIZER_PATH))

reloaded = Tokenizer.from_file(str(TOKENIZER_PATH))
assert reloaded.encode(RU_EXAMPLE).ids == tokenizer.encode(RU_EXAMPLE).ids
```

The next lessons download the tokenizer and the three splits from the Hugging Face Hub, so that every lesson can start on its own. The course's copies are published from this notebook.

To publish your own, set the environment variable `TFS_HUB_USER` to your Hugging Face username and add a write token as the `HF_TOKEN` secret (Kaggle: *Add-ons → Secrets*; Colab: the key icon in the sidebar).

Every repository on the Hub needs a card that says what it holds, where it comes from and how it was made. The
dataset's card:

```python
DATASET_CARD = f"""---
license: cc-by-2.0
language: [en, ru]
task_categories: [translation]
---
# Tatoeba English–Russian, cleaned and split without leaks

Built in lesson 1 of the course "Transformers: from scratch to production":
https://github.com/xmarva/transformers-from-scratch

- train and validation: sentence-transformers/parallel-sentences-tatoeba (en-ru),
  revision `{TRAIN_REVISION[:7]}`
- test: Tatoeba Translation Challenge eng-rus test set (Tiedemann, 2020)
- NFKC normalization, combining marks removed, straight quotes, one dash
- training pairs sharing a sentence fingerprint with the test set removed
- validation split by connected components of the translation graph,
  so no sentence occurs in both

Sentences from [Tatoeba](https://tatoeba.org), CC BY 2.0 FR.

| split | pairs |
|---|---|
| train | {len(train)} |
| validation | {len(valid)} |
| test | {len(test)} |
"""
```

And the tokenizer's, with the three lines it takes to use it:

````python
TOKENIZER_CARD = f"""---
license: cc-by-2.0
language: [en, ru]
tags: [tokenizer, bpe]
datasets: [xmarva/tfs-tatoeba-en-ru]
---
# BPE tokenizer for English and Russian, {tokenizer.get_vocab_size()} tokens

Built in lesson 1 of the course "Transformers: from scratch to production":
https://github.com/xmarva/transformers-from-scratch

- byte-pair encoding over characters, trained with Hugging Face `tokenizers` on the training
  split of xmarva/tfs-tatoeba-en-ru, both languages together
- owns its normalization: NFKC, combining marks removed, straight quotes, one dash
- words keep the space in front of them as `▁`
- special tokens `<pad>`, `<unk>`, `<bos>`, `<eos>` with ids 0 to 3

```python
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

path = hf_hub_download("xmarva/tfs-bpe-en-ru-16k", "tokenizer.json")
tokenizer = Tokenizer.from_file(path)
```

Trained on sentences from [Tatoeba](https://tatoeba.org), CC BY 2.0 FR.
"""
````

```python
HUB_USER = os.environ.get("TFS_HUB_USER")
hf_token = env.secret("HF_TOKEN")

if HUB_USER and hf_token:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    data_repo = f"{HUB_USER}/tfs-tatoeba-en-ru"
    tokenizer_repo = f"{HUB_USER}/tfs-bpe-en-ru-16k"

    api.create_repo(data_repo, repo_type="dataset", exist_ok=True)
    for name in splits:
        api.upload_file(
            path_or_fileobj=str(DATA_DIR / f"{name}.parquet"),
            path_in_repo=f"data/{name}.parquet",
            repo_id=data_repo,
            repo_type="dataset",
        )
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode(),
        path_in_repo="README.md",
        repo_id=data_repo,
        repo_type="dataset",
    )

    api.create_repo(tokenizer_repo, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(TOKENIZER_PATH),
        path_in_repo="tokenizer.json",
        repo_id=tokenizer_repo,
    )
    api.upload_file(
        path_or_fileobj=TOKENIZER_CARD.encode(),
        path_in_repo="README.md",
        repo_id=tokenizer_repo,
    )
    print("published", data_repo, "and", tokenizer_repo)
else:
    print("not publishing; lesson 2 downloads the course's copies:")
    print("xmarva/tfs-tatoeba-en-ru and xmarva/tfs-bpe-en-ru-16k")
```

<pre class="tfs-printed" markdown="0"><code>not publishing; lesson 2 downloads the course&#x27;s copies:
xmarva/tfs-tatoeba-en-ru and xmarva/tfs-bpe-en-ru-16k</code></pre>

## References

1. {:#ref-1} Tarjan, R. E. (1975). [Efficiency of a good but not linear set union algorithm](https://doi.org/10.1145/321879.321884). *Journal of the ACM*, 22(2). <a class="ref-back" href="#cite-1" aria-label="Back to the text">&#8617;</a>
2. {:#ref-2} Gage, P. (1994). A new algorithm for data compression. *The C Users Journal*, 12(2).
3. {:#ref-3} Sennrich, R., Haddow, B., Birch, A. (2016). [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909). ACL.
4. {:#ref-4} Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). OpenAI. *GPT-2 and its byte-level BPE.*
5. {:#ref-5} Kudo, T., Richardson, J. (2018). [SentencePiece: A simple and language independent subword tokenizer and detokenizer](https://arxiv.org/abs/1808.06226). EMNLP.
6. {:#ref-6} Petrov, A., La Malfa, E., Torr, P., Bibi, A. (2023). [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425). NeurIPS.
7. {:#ref-7} Rumbelow, J., Watkins, M. (2023). [SolidGoldMagikarp (plus, prompt generation)](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation). LessWrong.
8. {:#ref-8} OpenAI (2023). [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774). *Appendix C on contamination.*
9. {:#ref-9} Tiedemann, J. (2020). [The Tatoeba Translation Challenge](https://arxiv.org/abs/2010.06354). WMT.
10. {:#ref-10} Heaps, H. S. (1978). *Information Retrieval: Computational and Theoretical Aspects*. Academic Press.
11. {:#ref-11} [Unicode Standard Annex #15: Unicode Normalization Forms](https://unicode.org/reports/tr15/).
12. {:#ref-12} [tiktoken](https://github.com/openai/tiktoken), OpenAI's tokenizers, and the [Hugging Face Tokenizers documentation](https://huggingface.co/docs/tokenizers).
{: .tfs-refs}



<p class="tfs-neighbours"><a href="/blog/2026/transformers-from-scratch-02-transformer/">Lesson 2: The Transformer →</a></p>

{% endraw %}
