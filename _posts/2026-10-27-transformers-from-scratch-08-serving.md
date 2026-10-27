---
layout: post
title: "Serving a Transformer with FastAPI and Gradio: latency under load"
date: 2026-10-27 17:00:00
description: "Put a Transformer behind an HTTP API and measure it the way users meet it: a KV cache in beam search, int8 quantization that doesn't pay off, FastAPI and asyncio, open-loop load tests, queueing theory checked against the measurements, dynamic batching, and a Gradio page."
tags: [transformers-from-scratch, fastapi, inference, load-testing, queueing-theory]
categories: machine-learning
thumbnail: /assets/posts/tfs/08/cover.png
og_image: https://xmarva.github.io/assets/posts/tfs/08/cover.png
stylesheets: [/assets/css/tfs-course.css]
toc:
  beginning: true
---
{% raw %}

<p class="tfs-links"><strong>Transformers: from scratch to production</strong> · lesson 8 of 10 · <a href="/teaching/">all lessons</a> · <a href="https://www.kaggle.com/code/qmarva/transformers-from-scratch-08-serving-under-load">run on Kaggle</a> · <a href="https://colab.research.google.com/github/xmarva/transformers-from-scratch/blob/main/notebooks/colab/08_serving.ipynb">open in Colab</a> · <a href="https://github.com/xmarva/transformers-from-scratch">code on GitHub</a></p>

In November 2006 Greg Linden, who had built Amazon's recommendations, gave a talk at Stanford. One slide said
"+100 ms → −1% sales @ Amazon" and, next to it, "+500 ms → −20% traffic @ Google"
<sup class="cite">[<a href="#ref-1" id="cite-1" data-ref="1" aria-label="Reference 1: Linden, G. (2006). Make Data Useful, a talk at Stanford. The slide with Amazon&#x27;s 100 ms and Google&#x27;s 500 ms.">1</a>]</sup>. The two lines have
been quoted ever since, mostly without the footnotes: the Google experiment also showed thirty results instead
of ten, so it measured a heavier page, not just a slower one. When Google ran a clean experiment three years
later, adding 100 to 400 ms to search cost 0.2–0.6% of searches
<sup class="cite">[<a href="#ref-2" id="cite-2" data-ref="2" aria-label="Reference 2: Brutlag, J. (2009). Speed Matters for Google Web Search. Google.">2</a>]</sup>. The effect is smaller than the
legend, but it is there.

The other half of the story is what happens when many people ask at once. A request that is fast alone can wait
behind others, and users feel the waiting as much as the computing. The slow requests also matter more than their
share suggests. Dean and Barroso's paper on Google's serving points out that if answering a user takes a hundred
servers, and each is slow one time in a hundred, 63% of users wait for a slow one
<sup class="cite">[<a href="#ref-3" id="cite-3" data-ref="3" aria-label="Reference 3: Dean, J., Barroso, L. A. (2013). The Tail at Scale. Communications of the ACM 56(2).">3</a>]</sup>. That's why this lesson measures the slowest
requests, not only the average.

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/08/hero-light.png" width="760" alt="The share of slow requests against the number of servers a request waits for, when each server is slow one time in a hundred: 1% for one server, about 10% for ten, 63% for a hundred. The slowest part of a system decides how often its users wait" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/08/hero-dark.png" width="760" alt="The share of slow requests against the number of servers a request waits for, when each server is slow one time in a hundred: 1% for one server, about 10% for ten, 63% for a hundred. The slowest part of a system decides how often its users wait" loading="lazy">

This lesson puts perevod behind an HTTP API, and then measures it the way users will meet it: under load. On
the way it makes the model several times faster without changing a single translation, tries a famous speed-up
that doesn't pay off here, and finds out why a queue, not the model, decides most of the latency.

**What you'll do**

1. Time a real translator and ours, and find where a translation spends its time.
2. Bring lesson 5's KV cache into perevod's beam search, and check it changes no translation.
3. Try int8 quantization, measure that it doesn't help this model, and see why.
4. Build a FastAPI service, load it the way users do, and compare the curves with queueing theory.
5. Add dynamic batching, a Gradio page, and release perevod 0.2.0.

**What you need**: a CPU, as in lesson 7; Kaggle's free CPU sessions will do. The lesson downloads opus-mt (300 MB) for a comparison and lesson 7's model (40 MB), and runs in about a quarter of an hour, two thirds of it in load tests that run in real time.

On your own machine you can also open the service in a browser; on Kaggle and Colab the notebook talks to it from Python.

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
import asyncio
import math
import os
import random
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

for package in ["sacrebleu", "httpx"]:  # Kaggle has them, Colab may not
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
import httpx
import sacrebleu

print("CPU cores Python sees:", os.cpu_count())
print("threads PyTorch uses:", torch.get_num_threads())
```

<pre class="tfs-printed" markdown="0"><code>CPU cores Python sees: 4
threads PyTorch uses: 2</code></pre>

## 1. How long a translation takes

Start with a real model on this machine. opus-mt-en-ru, the baseline of the course, is a Marian Transformer with
six encoder and six decoder layers, trained on millions of sentence pairs. Here it translates sentences one at a
time, as a service would when requests arrive one by one, with the beam search it ships with:

```python
from transformers import MarianMTModel, MarianTokenizer

OPUS = "Helsinki-NLP/opus-mt-en-ru"
opus_tokenizer = MarianTokenizer.from_pretrained(OPUS)
opus = MarianMTModel.from_pretrained(OPUS).eval()

DATA_REVISION = "540be5b93bf77cd0846b02d55f2558a5596887a9"
test_path = hf_hub_download("xmarva/tfs-tatoeba-en-ru", "data/test.parquet",
                            repo_type="dataset", revision=DATA_REVISION)
test = pd.read_parquet(test_path)
probe = list(test.sample(30, random_state=SEED).en)


@torch.inference_mode()
def opus_translate(sentence):
    batch = opus_tokenizer([sentence], return_tensors="pt")
    output = opus.generate(**batch, num_beams=4, max_new_tokens=128)
    return opus_tokenizer.decode(output[0], skip_special_tokens=True)


def latencies(translate, sentences):
    times = []
    for sentence in sentences:
        start = time.perf_counter()
        translate(sentence)
        times.append(time.perf_counter() - start)
    return np.array(times)


opus_times = latencies(opus_translate, probe)
n_opus = sum(parameter.numel() for parameter in opus.parameters())
print(f"opus-mt: {n_opus / 1e6:.0f}M parameters, "
      f"median {np.median(opus_times) * 1000:.0f} ms per sentence")
```

<pre class="tfs-printed" markdown="0"><code>opus-mt: 77M parameters, median 264 ms per sentence</code></pre>

And perevod, the package of lesson 7. Its source lives in the course repository; we take the release we made,
by its tag, and only the `perevod/` folder of it, so the clone stays small. That folder is where this lesson
works:

```python
REPO = WORK_DIR / "course"
shutil.rmtree(REPO, ignore_errors=True)
URL = "https://github.com/xmarva/transformers-from-scratch"
QUIET = ["-c", "advice.detachedHead=false"]
subprocess.run(["git", *QUIET, "clone", "-q", "--depth", "1", "--branch", "perevod-v0.1.0",
                "--filter=blob:none", "--sparse", URL, str(REPO)], check=True)
subprocess.run(["git", "sparse-checkout", "set", "perevod"], cwd=REPO, check=True)
os.chdir(REPO)


def write(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text.lstrip("\n"), encoding="utf-8")


!{sys.executable} -m pip install -q ./perevod
importlib.invalidate_caches()

import perevod
from perevod import Translator

translator = Translator.from_pretrained(device="cpu")
n_ours = sum(parameter.numel() for parameter in translator.model.parameters())
ours_times = latencies(translator, probe)
print(f"perevod {perevod.__version__}: {n_ours / 1e6:.1f}M parameters, "
      f"median {np.median(ours_times) * 1000:.0f} ms")
```

<pre class="tfs-printed" markdown="0"><code>  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Building wheel for perevod (pyproject.toml) ... done
perevod 0.1.0: 9.6M parameters, median 73 ms</code></pre>

Ours is eight times smaller and three and a half times faster. Not eight: opus-mt's `generate` keeps a KV cache, the trick of lesson 5, and perevod 0.1.0 doesn't; section 3 adds it.

A sentence in less than a tenth of a second sounds like the end of the story. It isn't, for two reasons.

The time per sentence depends on its length, and the next section shows how badly. And a service doesn't translate one sentence: it translates whatever arrives, whenever it arrives, and section 7 shows what that does.

## 2. Where a translation spends its time

A translation is two very different computations. The encoder reads the English sentence once, all positions in parallel.

The decoder writes the Russian one token at a time, and every token is another pass through the decoder. Time both, on one long sentence, token by token:

```python
from perevod import decoding

model, tokenizer = translator.model, translator.tokenizer
LONG = ("The European Central Bank raised interest rates by a quarter of a percentage "
        "point on Thursday, citing persistent inflation in the services sector.")
src = torch.tensor([decoding.source_ids(tokenizer, LONG, model.config.max_len)])
pad_id, bos, eos = decoding.special_ids(tokenizer)

with torch.inference_mode():
    start = time.perf_counter()
    memory = model.encode(src)
    encode_ms = (time.perf_counter() - start) * 1000
    written = torch.full((1, 1), bos)
    plain_steps = []
    for _ in range(100):  # past the end of the translation too: only the clock matters here
        start = time.perf_counter()
        next_id = model.decode(written, memory, src)[:, -1].argmax(dim=-1)
        plain_steps.append((time.perf_counter() - start) * 1000)
        written = torch.cat([written, next_id[:, None]], dim=1)
print(f"encoder: {encode_ms:.1f} ms for {src.shape[1]} tokens")
print(f"decoder step 1: {plain_steps[0]:.1f} ms, step 100: {plain_steps[99]:.1f} ms")
```

<pre class="tfs-printed" markdown="0"><code>encoder: 6.6 ms for 39 tokens
decoder step 1: 5.4 ms, step 100: 16.5 ms</code></pre>

The encoder reads all the English tokens in about the time of one decoder step, and a translation takes one decoder step per Russian token: nearly all the time goes to the decoder. And its steps get slower as the translation grows.

The decoder is called with everything written so far, so every step recomputes all the earlier positions to get the scores for the next one. Step $$t$$ processes $$t$$ positions, and a translation of $$T$$ tokens processes

$$1 + 2 + \dots + T = \frac{T(T+1)}{2}$$

positions in all: twice as long a sentence means four times the positions. The clock shows less than that here. Step 100 handles a hundred times the positions of step 1 and takes only a few times as long, because one short sequence leaves the CPU mostly idle, and the fixed cost of a call dominates.

Beam search runs four hypotheses of every sentence of a batch at once, and there the positions dominate, as the next section measures.

Lesson 5 met this and fixed it for greedy decoding with a **KV cache**. perevod 0.1.0 ships the uncached code, because lesson 7 moved the simplest decoding
into the package. The package's default, beam search, never had a cache at all.

## 3. The KV cache, in the package

A refresher from lesson 5, in one sentence: at step $$t$$ the new token's query needs the keys and values of all
earlier positions, and those never change once computed, so we keep them per layer and compute only the new
token's. The cross-attention's keys and values come from the source alone and are computed once per sentence.

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/08/kv-step-light.png" width="760" alt="Decoding step 5 without and with a cache. Without one, the decoder runs on all five tokens and recomputes the keys and values of the first four. With one, it runs on the fifth token only, appends its key and value to the cache, and attends to all five" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/08/kv-step-dark.png" width="760" alt="Decoding step 5 without and with a cache. Without one, the decoder runs on all five tokens and recomputes the keys and values of the first four. With one, it runs on the fifth token only, appends its key and value to the cache, and attends to all five" loading="lazy">

Beam search adds one complication lesson 5 didn't have. After every step, the beams are re-chosen: a hypothesis can produce two survivors, and another none.

Each surviving beam continues the history of its parent, so its row in the cache must be its parent's row. Before the next step, the cache is reordered by the parents:

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/08/beam-reorder-light.png" width="760" alt="Beam search with a cache, beam size three. After a step, the three best continuations come from rows 0, 0 and 2 of the cache: row 1&#x27;s hypothesis died and row 0&#x27;s had two children. The cache is gathered with the index [0, 0, 2] so that each new beam&#x27;s row holds its parent&#x27;s keys and values" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/08/beam-reorder-dark.png" width="760" alt="Beam search with a cache, beam size three. After a step, the three best continuations come from rows 0, 0 and 2 of the cache: row 1&#x27;s hypothesis died and row 0&#x27;s had two children. The cache is gathered with the index [0, 0, 2] so that each new beam&#x27;s row holds its parent&#x27;s keys and values" loading="lazy">

The new `decoding.py` of perevod. The helpers at the top are lesson 7's:

```python
%%writefile perevod/src/perevod/decoding.py
# Greedy decoding and beam search of lesson 3, batched as in lesson 4, with lesson 5's KV
# cache: every step feeds the decoder only the newest token. A source longer than the
# model's positions is cut to fit, instead of failing inside the model.

import torch
import torch.nn.functional as F

from .model import apply_rope


def special_ids(tokenizer):
    pad_id = tokenizer.token_to_id("<pad>")
    return pad_id, tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")


def source_ids(tokenizer, sentence, limit):
    return tokenizer.encode(sentence).ids[:limit]


def pad(sequences, pad_id, device):
    longest = max(len(ids) for ids in sequences)
    batch = torch.full((len(sequences), longest), pad_id, device=device)
    for row, ids in enumerate(sequences):
        batch[row, : len(ids)] = torch.tensor(ids, device=device)
    return batch


def batches(sources, batch_size):
    # sentences in batches of similar length, so that little of a batch is padding
    order = sorted(range(len(sources)), key=lambda i: len(sources[i]))
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]


def length_penalty(length, alpha=0.6):
    # Wu et al. (2016): ((5 + |y|) / 6) ** alpha
    return ((5 + length) / 6) ** alpha
```

<pre class="tfs-printed" markdown="0"><code>Overwriting perevod/src/perevod/decoding.py</code></pre>

`attend` is one attention sublayer for the newest token. It works on the model's own `Attention` modules, so the weights are the model's; only the order of operations is new.

For self-attention it rotates the new query and key with RoPE and appends the key and value to the cache. It needs no causal mask: the only query is the newest token, and everything in the cache is its past.

For cross-attention the cache is computed once from the source, and marked `"fixed"` so that nothing is appended to it:

```python
%%writefile -a perevod/src/perevod/decoding.py


def attend(attention, x, cache, mask, angles):
    # one attention sublayer for the newest token; the cache holds everything before it
    q = attention.split_heads(attention.w_q(x))
    if angles is not None:  # RoPE: self-attention rotates queries and keys by position
        q = apply_rope(q, angles)
    if "fixed" not in cache:  # self-attention: add the new token's key and value
        k = attention.split_heads(attention.w_k(x))
        v = attention.split_heads(attention.w_v(x))
        if angles is not None:
            k = apply_rope(k, angles)
        if "k" in cache:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        cache["k"], cache["v"] = k, v
    out = F.scaled_dot_product_attention(q, cache["k"], cache["v"], attn_mask=mask)
    batch, _, length, _ = out.shape
    return attention.w_o(out.transpose(1, 2).reshape(batch, length, -1))
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/decoding.py</code></pre>

`decode_step` runs the decoder of `model.py` for one token, in the same order: pre-norm, self-attention,
cross-attention, feed-forward. It repeats the model's arithmetic outside the model, which is a debt: if
`model.py` changes, this has to change with it. The golden test of lesson 7 fails if the two drift apart.

```python
%%writefile -a perevod/src/perevod/decoding.py


def new_caches(model, memory):
    # per decoder layer: an empty self-attention cache, and the cross-attention's keys and
    # values, which depend only on the source and are computed once
    caches = []
    for layer in model.decoder:
        attention = layer.cross_attention
        k = attention.split_heads(attention.w_k(memory))
        v = attention.split_heads(attention.w_v(memory))
        caches.append({"self": {}, "cross": {"fixed": True, "k": k, "v": v}})
    return caches


def decode_step(model, token, memory_mask, caches, position):
    # the decoder of model.py, in the same order, for one new token; dropout is off anyway
    x = model.embed(token)
    angles = model.angles[position : position + 1]
    for layer, cache in zip(model.decoder, caches, strict=True):
        x = x + attend(layer.self_attention, layer.norm1(x), cache["self"], None, angles)
        x = x + attend(layer.cross_attention, layer.norm2(x), cache["cross"], memory_mask,
                       None)
        x = x + layer.feed_forward(layer.norm3(x))
    return (model.decoder_norm(x) @ model.embedding.weight.T)[:, -1]
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/decoding.py</code></pre>

`reorder` moves the cached keys and values along with their hypotheses, as in the figure above. It handles both
kinds of cache, so the source's keys and values follow their sentence too.

```python
%%writefile -a perevod/src/perevod/decoding.py


def reorder(caches, rows):
    # the cached keys and values follow their hypotheses: row r of the new caches is row
    # rows[r] of the old ones
    for cache in caches:
        for part in cache.values():
            for key in ("k", "v"):
                if key in part:
                    part[key] = part[key][rows]
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/decoding.py</code></pre>

Greedy decoding with the cache is lesson 5's `greedy_cached`, batched:

```python
%%writefile -a perevod/src/perevod/decoding.py


@torch.inference_mode()
def greedy(model, tokenizer, sentences, max_tokens=128, batch_size=256):
    model.eval()
    device = next(model.parameters()).device
    pad_id, bos, eos = special_ids(tokenizer)
    sources = []
    for sentence in sentences:
        sources.append(source_ids(tokenizer, sentence, model.config.max_len))
    max_tokens = min(max_tokens, model.config.max_len - 1)
    translations = [None] * len(sentences)
    for indices in batches(sources, batch_size):
        src = pad([sources[i] for i in indices], pad_id, device)
        memory_mask = (src != pad_id)[:, None, None, :]
        caches = new_caches(model, model.encode(src))
        token = torch.full((len(indices), 1), bos, device=device)
        written = []
        done = torch.zeros(len(indices), dtype=torch.bool, device=device)
        for position in range(max_tokens):
            scores = decode_step(model, token, memory_mask, caches, position)
            next_ids = scores.argmax(dim=-1)
            next_ids[done] = pad_id  # a finished sentence only collects padding
            written.append(next_ids)
            done |= next_ids == eos
            token = next_ids[:, None]
            if done.all():
                break
        written = torch.stack(written, dim=1)
        for row, i in enumerate(indices):
            ids = []
            for token_id in written[row].tolist():
                if token_id in (eos, pad_id):
                    break
                ids.append(token_id)
            translations[i] = tokenizer.decode(ids)
    return translations
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/decoding.py</code></pre>

Beam search is lesson 3's, step for step, with two changes: a hypothesis remembers its row in the cache, and the
cache is reordered before each step.

```python
%%writefile -a perevod/src/perevod/decoding.py


@torch.inference_mode()
def beam_search(model, tokenizer, sentences, beam_size=4, alpha=0.6, max_tokens=128,
                batch_size=64):
    model.eval()
    device = next(model.parameters()).device
    pad_id, bos, eos = special_ids(tokenizer)
    sources = []
    for sentence in sentences:
        sources.append(source_ids(tokenizer, sentence, model.config.max_len))
    max_tokens = min(max_tokens, model.config.max_len - 1)
    translations = [None] * len(sentences)
    for indices in batches(sources, batch_size):
        src = pad([sources[i] for i in indices], pad_id, device)
        memory_mask = (src != pad_id)[:, None, None, :]
        caches = new_caches(model, model.encode(src))
        # per sentence: (tokens so far, total log-probability, its row in the caches)
        beams = []
        for k in range(len(indices)):
            beams.append([([bos], 0.0, k)])
        finished = [[] for _ in indices]
        active = [True for _ in indices]
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/decoding.py</code></pre>

Each step does three things. It collects the cache rows of the live hypotheses of every sentence still being translated, and reorders the caches to them.

On the first step that turns one row per sentence into one row per hypothesis; later, sentences that are done drop out, and their rows with them. Then one decoder step runs for all those hypotheses at once, and each gets its `beam_size` best next tokens.

Finally, for each sentence, every hypothesis is extended by each of its tokens, the candidates are sorted, those that end with `<eos>` are set aside as finished, and the best `beam_size` others survive. A sentence stops when it has `beam_size` finished hypotheses.

```python
%%writefile -a perevod/src/perevod/decoding.py

        for position in range(max_tokens):
            rows, owners, last = [], [], []
            for k in range(len(indices)):
                if active[k]:
                    for tokens, _, row in beams[k]:
                        rows.append(row)
                        owners.append(k)
                        last.append(tokens[-1])
            if not rows:
                break
            reorder(caches, torch.tensor(rows, device=device))
            token = torch.tensor(last, device=device)[:, None]
            owner = torch.tensor(owners, device=device)
            scores = decode_step(model, token, memory_mask[owner], caches, position)
            top = scores.float().log_softmax(dim=-1).topk(beam_size)
            values, ids = top.values.tolist(), top.indices.tolist()

            row = 0
            for k in range(len(indices)):
                if not active[k]:
                    continue
                candidates = []
                for tokens, total, _ in beams[k]:
                    for log_p, token_id in zip(values[row], ids[row], strict=True):
                        candidates.append((tokens + [token_id], total + log_p, row))
                    row += 1
                candidates.sort(key=lambda candidate: candidate[1], reverse=True)
                beams[k] = []
                for tokens, total, parent in candidates:
                    if tokens[-1] == eos:
                        finished[k].append((tokens, total))
                    else:
                        beams[k].append((tokens, total, parent))
                    if len(beams[k]) == beam_size:
                        break
                if len(finished[k]) >= beam_size:
                    active[k] = False
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/decoding.py</code></pre>

At the end each sentence takes its best hypothesis by the length-normalized score, as in lesson 3:

```python
%%writefile -a perevod/src/perevod/decoding.py

        for k, i in enumerate(indices):

            def normalized(candidate):
                tokens, total = candidate[0], candidate[1]
                return total / length_penalty(len(tokens) - 1, alpha)  # without <bos>

            best = max(finished[k] or beams[k], key=normalized)
            translations[i] = tokenizer.decode(best[0][1:])  # decode drops <eos>
    return translations
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/decoding.py</code></pre>

The file now holds the new code, and the installed package still has the old one. That's useful for a moment:
the old module is already imported as `decoding`, and the new file can be loaded next to it under another name.

```python
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = decoding  # perevod 0.1.0, as installed
new = load_module("perevod.decoding_new", "perevod/src/perevod/decoding.py")
print("old:", old.__file__)
print("new:", new.__file__)
```

<pre class="tfs-printed" markdown="0"><code>old: /usr/local/lib/python3.13/dist-packages/perevod/decoding.py
new: /kaggle/working/tfs/course/perevod/src/perevod/decoding.py</code></pre>

The cache must not change a single translation. On 500 sentences of the test set and 30 of the news test set of
lesson 7, with beam search, the old code against the new:

```python
from sacrebleu.utils import get_source_file

tatoeba = test.sample(500, random_state=SEED)
news_file = Path(get_source_file("wmt20", "en-ru"))
news = random.Random(SEED).sample(news_file.read_text(encoding="utf-8").splitlines(), 30)

rows = []
for name, sentences in [("Tatoeba", list(tatoeba.en)), ("WMT20 news", news)]:
    start = time.perf_counter()
    before = old.beam_search(model, tokenizer, sentences)
    middle = time.perf_counter()
    after = new.beam_search(model, tokenizer, sentences)
    end = time.perf_counter()
    assert after == before, name
    rows.append({"test set": name, "sentences": len(sentences),
                 "without the cache, s": middle - start, "with it, s": end - middle,
                 "faster": (middle - start) / (end - middle)})
speed = pd.DataFrame(rows)
display(table(speed))
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">test set</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">sentences</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">without the cache, s</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">with it, s</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">faster</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Tatoeba</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">500</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">22.56</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">4.96</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">4.54</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">WMT20 news</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">30</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">68.38</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">3.77</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">18.16</td></tr></tbody></table></div>

The same translations, word for word, and faster, by more on the news. That's the $$T^2$$ at work: Tatoeba's
sentences are short, and a few steps of recomputation are cheap; a news sentence runs for dozens of steps, and the
recomputation dominates. The step times of section 2 show it directly:

```python
with torch.inference_mode():
    memory_mask = (src != pad_id)[:, None, None, :]
    caches = new.new_caches(model, model.encode(src))
    token = torch.full((1, 1), bos)
    cached_steps = []
    for position in range(100):
        start = time.perf_counter()
        scores = new.decode_step(model, token, memory_mask, caches, position)
        next_id = scores.argmax(dim=-1)
        cached_steps.append((time.perf_counter() - start) * 1000)
        token = next_id[:, None]

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(range(1, 101), plain_steps, color=COLORS["dec"], label="recompute everything")
ax.plot(range(1, 101), cached_steps, color=COLORS["enc"], label="with the KV cache")
ax.set_xlabel("decoding step")
ax.set_ylabel("ms per step")
ax.set_ylim(bottom=0)
ax.set_title("One decoding step, on a CPU")
ax.legend(frameon=False)
plt.show()
```

<img class="tfs-output" src="/assets/posts/tfs/08/output-01.png" alt="One decoding step, on a CPU" loading="lazy">

Without the cache a step gets slower as the translation grows; with it, every step costs about the same.

## 4. int8 quantization, measured

The next speed-up everyone reaches for is **quantization**: store weights as 8-bit integers instead of 32-bit
floats, and multiply in integers. Each weight matrix gets a scale $$s$$ so that $$w \approx s \cdot q$$, with $$q$$ an
integer from −128 to 127. PyTorch's dynamic quantization does it in one call: the weights of every `nn.Linear`
become int8 once, and at every call the input is measured and scaled the same way, the multiplication runs in
integers, and the result is scaled back
<sup class="cite">[<a href="#ref-4" id="cite-4" data-ref="4" aria-label="Reference 4: PyTorch: dynamic quantization, CPU threading.">4</a>]</sup>.

```python
from torch.ao.quantization import quantize_dynamic

engines = torch.backends.quantized.supported_engines
if torch.backends.quantized.engine == "none" and "qnnpack" in engines:  # ARM, e.g. a Mac
    torch.backends.quantized.engine = "qnnpack"
quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
sentences = list(tatoeba.en)
start = time.perf_counter()
fp32 = new.beam_search(model, tokenizer, sentences)
fp32_seconds = time.perf_counter() - start
start = time.perf_counter()
int8 = new.beam_search(quantized, tokenizer, sentences)
int8_seconds = time.perf_counter() - start
changed = sum(a != b for a, b in zip(fp32, int8))
fp32_chrf = sacrebleu.corpus_chrf(fp32, [list(tatoeba.ru)]).score
int8_chrf = sacrebleu.corpus_chrf(int8, [list(tatoeba.ru)]).score
print(f"float32: {fp32_seconds:.1f} s, chrF {fp32_chrf:.1f}")
print(f"int8:    {int8_seconds:.1f} s, chrF {int8_chrf:.1f}, {changed} of {len(sentences)} "
      "translations changed")
```

<pre class="tfs-printed" markdown="0"><code>float32: 4.9 s, chrF 61.0
int8:    4.8 s, chrF 59.9, 178 of 500 translations changed</code></pre>

About as fast as before, and worse. The speed-up that int8 promises comes from large matrix multiplications,
where moving 8-bit numbers instead of 32-bit ones saves memory traffic. Count where the work of one decoding step
goes in our model, layer by layer:

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/08/step-flops-light.png" width="760" alt="Multiply-adds of one decoding step of the published model: the output projection onto 16,000 tokens is the largest single part, larger than all attention projections together; dynamic quantization converts only the nn.Linear layers and leaves the output projection, which reuses the embedding matrix, in float32" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/08/step-flops-dark.png" width="760" alt="Multiply-adds of one decoding step of the published model: the output projection onto 16,000 tokens is the largest single part, larger than all attention projections together; dynamic quantization converts only the nn.Linear layers and leaves the output projection, which reuses the embedding matrix, in float32" loading="lazy">

The output projection onto the 16,000 tokens of the vocabulary does 59% of the multiply-adds of a step, and it isn't an `nn.Linear`: the model reuses the embedding matrix there, so quantization never touches it.

Even if int8 made every `nn.Linear` free, more than half of the work would stay in float32. What quantization does touch are 256-wide layers, too small for integer arithmetic to win back the cost of measuring and scaling the input at every call.

On a model this size, on this CPU, quantization costs quality and buys nothing, so perevod 0.2.0 doesn't use it. (PyTorch is moving quantization to its `torchao` library, and the cell prints a deprecation warning saying so; the arithmetic won't change with the API.)

## 5. Batches and threads

A CPU does more work per second on bigger matrices. The fixed cost of every call, and reading the weights from memory, are paid once per batch instead of once per sentence. Here is perevod's throughput with batches of different sizes, on 256 sentences.

`beam_search` groups sentences of similar length into a batch, so that none waits long for a much longer neighbour; these numbers are the best case:

```python
sentences = list(test.sample(256, random_state=SEED + 1).en)
rows = []
for batch_size in [1, 2, 4, 8, 16, 32, 64]:
    start = time.perf_counter()
    new.beam_search(model, tokenizer, sentences, batch_size=batch_size)
    seconds = time.perf_counter() - start
    rows.append({"batch": batch_size, "sentences per second": len(sentences) / seconds,
                 "ms per batch": 1000 * seconds / math.ceil(len(sentences) / batch_size)})
by_batch = pd.DataFrame(rows)
display(table(by_batch, digits=1))
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">batch</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">sentences per second</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">ms per batch</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">18.4</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">54.2</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">28.3</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">70.7</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">4</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">43.0</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">93.0</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">8</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">56.3</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">142.2</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">16</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">72.2</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">221.6</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">32</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">82.6</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">387.4</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">64</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">83.4</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">767.2</td></tr></tbody></table></div>

From one sentence to sixteen, the throughput grows four times, from 18 sentences a second to 72, and a batch of sixteen takes four times as long as a single sentence, not sixteen times.

That's the fact the second half of this lesson is built on: one translation of sixteen sentences costs about as much as four translations of one. Past sixteen the gain flattens.

A batch of 64 gives 83 sentences a second instead of 72 and takes three quarters of a second, which every sentence in it waits for. The server's batches will stop at sixteen.

One more knob: threads. PyTorch splits a matrix multiplication between CPU threads, and more isn't always better:

```python
threads_before = torch.get_num_threads()
rows = []
for threads in [1, 2, 4]:
    torch.set_num_threads(threads)
    start = time.perf_counter()
    new.beam_search(model, tokenizer, sentences[:128], batch_size=16)
    rows.append({"threads": threads, "seconds": time.perf_counter() - start})
torch.set_num_threads(threads_before)
display(table(pd.DataFrame(rows)))
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">threads</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">seconds</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">3.34</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2.08</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">4</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2.15</td></tr></tbody></table></div>

Kaggle's free CPU session shows Python four cores, but they are two physical cores running two hardware threads
each, and two threads that share a core mostly take turns. PyTorch starts one thread per physical core, two here,
and four buy nothing over two. The default is right on this machine and can be wrong on another: a container run
with `--cpus=2` still shows the process every core of its host, and PyTorch starts a thread per physical core of
the host. Check the number on the machine you deploy to, and set it if it's wrong: `torch.set_num_threads(n)` in
the code, or the `OMP_NUM_THREADS` environment variable before the process starts
<sup class="cite">[<a href="#ref-4" id="cite-5" data-ref="4" aria-label="Reference 4: PyTorch: dynamic quantization, CPU threading.">4</a>]</sup>.

## 6. A FastAPI service

A command line loads the model for every call, and lesson 7 measured that the loading costs more than translating
a sentence. A **service** loads it once and answers requests over HTTP for as long as it runs.

Two programs make one in Python. A server, here [uvicorn](https://www.uvicorn.org), listens on a port, speaks HTTP and hands every request to the app. The app, written with [FastAPI](https://fastapi.tiangolo.com), decides what to answer.

(The interface between the two is called ASGI; you'll meet the name in error messages.) In the app, a **handler** is the function called for one address.

Ours are a POST to `/translate` with a JSON body like `{"text": "..."}`, and a GET to `/health` that tells whoever asks that the service is up.

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/08/service-light.png" width="760" alt="How a request travels through perevod&#x27;s service: an HTTP request reaches uvicorn and FastAPI, which validates it and puts it into a queue; a batcher takes the requests that arrived within a few milliseconds, translates them as one batch in a worker thread, and answers each request with its own translation" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/08/service-dark.png" width="760" alt="How a request travels through perevod&#x27;s service: an HTTP request reaches uvicorn and FastAPI, which validates it and puts it into a queue; a batcher takes the requests that arrived within a few milliseconds, translates them as one batch in a worker thread, and answers each request with its own translation" loading="lazy">

FastAPI builds on pydantic, which turns a class with typed fields into a validator: a request body that doesn't fit is refused, with an explanation, before our code runs.

The model reads at most 256 tokens, about a thousand characters of English, so the API refuses longer texts instead of cutting them silently. A real service would split a long text into sentences and translate them one by one.

```python
%%writefile perevod/src/perevod/server.py
import argparse
import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .translator import MODEL, Translator


class TranslateRequest(BaseModel):
    # the model reads at most 256 tokens, about a thousand characters of English
    text: str = Field(max_length=1000, examples=["Tom is afraid of the dark."])


class TranslateResponse(BaseModel):
    translation: str
```

<pre class="tfs-printed" markdown="0"><code>Writing perevod/src/perevod/server.py</code></pre>

Many requests can be open at once, and the model should see them one batch at a time. Between the two sits a
queue, and the queue lives in asyncio, the part of Python that uvicorn runs on. It needs a paragraph of its own.

uvicorn runs the app on an **event loop**: one thread that juggles all the open requests. A function defined with `async def` is a **coroutine**, and at every `await` (on the network, a queue, a timer) it pauses and lets the loop run something else. That's how one process holds hundreds of requests without hundreds of threads.

The flip side is that code which computes without awaiting holds the whole loop: a translation of 60 ms would freeze every other request for 60 ms. So the model runs in another thread, through `asyncio.to_thread`, and the loop waits for it as it waits for the network.

Last, a **future** is a box for a result that doesn't exist yet: `await answer` pauses until someone calls `answer.set_result(...)`, and then returns that value.

Now the **batcher**. Every request puts its text and a future into the queue and waits for the future. The
batcher takes requests out of the queue: the first one, whenever it comes, then whatever else arrives within
`max_wait` seconds (`wait_for` gives up when the time left runs out), up to `max_batch` requests. With
`max_batch=1` it is a plain queue, one request at a time. NVIDIA's Triton inference server batches with the same
two settings
<sup class="cite">[<a href="#ref-5" id="cite-6" data-ref="5" aria-label="Reference 5: NVIDIA Triton: dynamic batcher.">5</a>]</sup>.

```python
%%writefile -a perevod/src/perevod/server.py


class Batcher:
    """Translates requests that arrive close together as one batch.

    A batch closes when it holds max_batch requests or max_wait seconds after its first one.
    """

    def __init__(self, translator, max_batch=16, max_wait=0.005):
        self.translator = translator
        self.max_batch, self.max_wait = max_batch, max_wait
        self.queue = asyncio.Queue()
        self.batch_sizes = []

    async def translate(self, text):
        answer = asyncio.get_running_loop().create_future()
        await self.queue.put((text, answer))
        return await answer  # until run() gives the future its result
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/server.py</code></pre>

```python
%%writefile -a perevod/src/perevod/server.py

    async def collect(self):
        batch = [await self.queue.get()]  # waits for the first request, however long
        deadline = time.monotonic() + self.max_wait
        while len(batch) < self.max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self.queue.get(), remaining))
            except asyncio.TimeoutError:
                break
        return batch
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/server.py</code></pre>

The batcher translates each batch in a worker thread and gives every request its own translation. If the translation fails, every request of the batch gets the error instead of waiting forever.

The `done()` checks cover a quieter case: a caller that stopped waiting, say on a timeout, leaves a cancelled future behind. Setting a result on it raises an error, and without the check that error would end `run()`; every later request would then wait forever, while `/health` went on answering "ok".

```python
%%writefile -a perevod/src/perevod/server.py

    async def run(self):
        while True:
            batch = await self.collect()
            self.batch_sizes.append(len(batch))
            texts = [text for text, _ in batch]
            try:  # in a thread, so that the event loop keeps serving requests meanwhile
                translations = await asyncio.to_thread(self.translator.translate, texts)
            except Exception as error:  # noqa: BLE001, every request of the batch gets it
                for _, answer in batch:
                    if not answer.done():
                        answer.set_exception(error)
                continue
            for (_, answer), translation in zip(batch, translations, strict=True):
                if not answer.done():  # done already if its caller stopped waiting
                    answer.set_result(translation)
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/server.py</code></pre>

The app itself. `lifespan` runs when the server starts and when it stops: it loads the model once and starts the batcher as a background task next to the handlers (`create_task`), and cancels it at the end.

`app.state` is where the handlers find the batcher. The `demo` switch mounts the Gradio page of section 9.

```python
%%writefile -a perevod/src/perevod/server.py


def create_app(model=MODEL, revision=None, max_batch=16, max_wait=0.005, demo=False):
    @asynccontextmanager
    async def lifespan(app):
        # before yield: when the server starts; after it: when it stops
        translator = Translator.from_pretrained(model, revision)
        app.state.batcher = Batcher(translator, max_batch, max_wait)
        worker = asyncio.create_task(app.state.batcher.run())  # runs next to the handlers
        yield
        worker.cancel()

    app = FastAPI(title="perevod", version=__version__, lifespan=lifespan)

    @app.post("/translate")
    async def translate(request: TranslateRequest) -> TranslateResponse:
        translation = await app.state.batcher.translate(request.text)
        return TranslateResponse(translation=translation)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__}

    if demo:
        import gradio as gr  # only the demo extra installs it

        from .demo import build_demo

        app = gr.mount_gradio_app(app, build_demo(app), path="/")
    return app
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/server.py</code></pre>

And a command, `perevod-server`, with the batcher's settings as options:

```python
%%writefile -a perevod/src/perevod/server.py


def main(argv=None):
    parser = argparse.ArgumentParser(prog="perevod-server",
                                     description="Serve perevod over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-batch", type=int, default=16, help="1 turns batching off")
    parser.add_argument("--max-wait-ms", type=float, default=5.0)
    parser.add_argument("--demo", action="store_true", help="serve a Gradio page at /")
    args = parser.parse_args(argv)

    import uvicorn  # only the serve extra installs it; the tests don't need it

    app = create_app(max_batch=args.max_batch, max_wait=args.max_wait_ms / 1000,
                     demo=args.demo)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/server.py</code></pre>

The package grows a new optional dependency, the server's: `pip install "perevod[serve]"` installs FastAPI and
uvicorn, and a plain `pip install perevod` stays as light as it was. The server gets a command of its own,
`perevod-server`.

The version becomes 0.2.0, and from now on it is written in one place. Lesson 7 wrote it twice, in `pyproject.toml` and in `__init__.py`, and its version test compared the command's output with the second copy, so the two could drift apart at the first release without a test noticing.

Now `pyproject.toml` declares the version `dynamic`, and hatchling reads it from `__init__.py` when it builds the wheel:

```python
write("perevod/pyproject.toml", """
[project]
name = "perevod"
dynamic = ["version"]  # read from src/perevod/__init__.py, see [tool.hatch.version]
description = "English-to-Russian translation with a small Transformer built from scratch"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "tokenizers>=0.15",
    "huggingface_hub>=0.20",
    "safetensors>=0.4",
]

[project.optional-dependencies]
serve = ["fastapi>=0.110", "uvicorn>=0.29"]
demo = ["fastapi>=0.110", "uvicorn>=0.29", "gradio>=4.40"]
test = ["pytest>=8", "ruff>=0.6", "sacrebleu>=2.4", "pandas>=2.0", "pyarrow>=14",
        "fastapi>=0.110", "httpx>=0.27"]

[project.scripts]
perevod = "perevod.cli:main"
perevod-server = "perevod.server:main"

[project.urls]
Source = "https://github.com/xmarva/transformers-from-scratch/tree/main/perevod"
Model = "https://huggingface.co/xmarva/perevod-en-ru"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.version]
path = "src/perevod/__init__.py"  # the one place the version is written

[tool.pytest.ini_options]
markers = ["slow: downloads the published model and translates real sentences"]

[tool.ruff]
line-length = 92  # what a notebook cell shows: lesson 7 writes these files in its cells
""")
```

```python
%%writefile perevod/src/perevod/__init__.py
__version__ = "0.2.0"


def __getattr__(name):
    # `import perevod` stays light, so that --help and --version answer at once;
    # PyTorch is loaded only when someone asks for the translator
    if name == "Translator":
        from .translator import Translator

        return Translator
    raise AttributeError(f"module 'perevod' has no attribute {name!r}")
```

<pre class="tfs-printed" markdown="0"><code>Overwriting perevod/src/perevod/__init__.py</code></pre>

From here on we change the package as we go, so it's installed in **editable** mode: pip puts a pointer to `perevod/src` into site-packages instead of a copy, and every process that imports perevod gets the current files.

That's the right install for working on a package. The CI job that installs the built wheel is what checks the copy users get, and section 10 makes it check more.

```python
!{sys.executable} -m pip install -q -e "./perevod[serve]"
```

<pre class="tfs-printed" markdown="0"><code>  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Installing backend dependencies ... done
  Preparing editable metadata (pyproject.toml) ... done
  Building editable for perevod (pyproject.toml) ... done</code></pre>

A service runs in its own process. `start_server` starts one, sends what it prints to a log file, and waits
until `/health` answers; if the process exits first, the error shows the end of its log. Then a sentence, first
with batching turned off:

```python
def start_server(port, max_batch, max_wait_ms=5, demo=False):
    command = [sys.executable, "-m", "perevod.server", "--port", str(port),
               "--max-batch", str(max_batch), "--max-wait-ms", str(max_wait_ms)]
    if demo:
        command.append("--demo")
    log = Path(f"server-{port}.log")
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
    for _ in range(180):
        if process.poll() is not None:  # it exited: a port in use, a missing package...
            break
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                return process
        except httpx.HTTPError:
            pass
        time.sleep(1)
    process.kill()
    raise RuntimeError("the server did not start:\n" + log.read_text()[-2000:])


server = start_server(8001, max_batch=1)
URL = "http://127.0.0.1:8001/translate"
answer = httpx.post(URL, json={"text": "Nobody understands me."})
print(answer.status_code, answer.json())
refused = httpx.post(URL, json={"text": "word " * 2000})
print(refused.status_code, refused.json()["detail"][0]["msg"])
```

<pre class="tfs-printed" markdown="0"><code>200 {&#x27;translation&#x27;: &#x27;Никто меня не понимает.&#x27;}
422 String should have at most 1000 characters</code></pre>

A translation, and for the text that is too long a 422, HTTP's code for "I understood the request, but its content is invalid", with a readable reason. Our code never saw that request: FastAPI refused it from the declaration of `TranslateRequest`.

The same declarations describe the API at `/docs`, an interactive page where anyone can try requests (on your own machine, open http://127.0.0.1:8001/docs in a browser), and at `/openapi.json`, a machine-readable description from which clients in other languages can be generated:

```python
spec = httpx.get("http://127.0.0.1:8001/openapi.json").json()
for path, operations in spec["paths"].items():
    for method, operation in operations.items():
        print(method.upper(), path, "-", operation["summary"])
```

<pre class="tfs-printed" markdown="0"><code>POST /translate - Translate
GET /health - Health</code></pre>

## 7. Load testing without fooling yourself

Now the question this lesson is about: how fast does the service answer when many people use it? The obvious
way to find out is a loop that sends a request, waits for the answer, records the time, and sends the next one,
with a few such loops in parallel. Tools built around "virtual users" that wait for their answers, like Locust or
JMeter's usual thread groups, work that way, and Gil Tene named what's wrong with it: **coordinated omission**
<sup class="cite">[<a href="#ref-6" id="cite-7" data-ref="6" aria-label="Reference 6: Tene, G. (2013). Coordinated Omission, on the mechanical-sympathy list; and the talk How NOT to Measure Latency (2015).">6</a>]</sup>.

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/08/omission-light.png" width="760" alt="A server stalls for one second. Users arriving at their own pace keep sending during the stall, and a quarter of their requests are slow. A closed-loop tester waits for its answer and sends nothing during the stall, so it records one slow request in thirty-two and reports a service that is almost never slow" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/08/omission-dark.png" width="760" alt="A server stalls for one second. Users arriving at their own pace keep sending during the stall, and a quarter of their requests are slow. A closed-loop tester waits for its answer and sends nothing during the stall, so it records one slow request in thirty-two and reports a service that is almost never slow" loading="lazy">

When the server stalls, a tester that waits for each answer stops sending, so it records one slow request where real users, who don't coordinate with the server, would have produced dozens.

The fix is an **open-loop** test: decide in advance when every request is sent, independently of the answers, and measure each latency from the moment the request was *supposed* to go.

[wrk2](https://github.com/giltene/wrk2), which Tene wrote for exactly this, and the arrival-rate executors of [k6](https://grafana.com/docs/k6/latest/using-k6/scenarios/concepts/open-vs-closed/) work that way. Users arriving independently at an average rate $$\lambda$$ are well described by a **Poisson process**, whose gaps between arrivals are exponentially distributed with mean $$1/\lambda$$:

```python
def arrival_times(rate, seconds, seed):
    rng = random.Random(seed)
    times, moment = [], rng.expovariate(rate)
    while moment < seconds:
        times.append(moment)
        moment += rng.expovariate(rate)  # exponential gaps: a Poisson process
    return times


async def load_test(url, rate, seconds, sentences, seed=0):
    # no connection limit: with httpx's default of 100, requests would queue in the
    # client and the test would blame the server; no timeout: a slow answer is a
    # measurement, not an error
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        start = time.perf_counter()

        async def one(k, at):
            await asyncio.sleep(max(0.0, start + at - time.perf_counter()))
            response = await client.post(url, json={"text": sentences[k % len(sentences)]})
            response.raise_for_status()
            return time.perf_counter() - (start + at)  # from when it was due, not sent

        tasks = []
        for k, at in enumerate(arrival_times(rate, seconds, seed)):
            tasks.append(one(k, at))
        return np.array(await asyncio.gather(*tasks))
```

Each request is a coroutine that sleeps until its moment, then sends. If the server is slow, the coroutines don't wait for each other, just like users.

`gather` runs them all together and returns their results in order. (A notebook can `await` at the top level; in a script you'd write `asyncio.run(load_test(...))`.)

Two shortcuts make this a lesson's load test rather than a release's. The load generator shares the server's two
cores, where a real test runs it from another machine. And `raise_for_status` stops the test at the first error,
where a real one counts errors next to latencies: a server that answers "500" quickly has excellent latency.

First, the **service time**: how long the server takes per request when no other request is waiting. For this one measurement the closed loop is exactly the right tool: send a request, wait for the answer, send the next.

There's never a queue, so every time it records is a service time. Five hundred sentences of the traffic the load test will send, in the same order:

```python
traffic = list(test.sample(2_000, random_state=SEED + 2).en)
service_times, answers = [], []
with httpx.Client(timeout=None) as client:
    for sentence in traffic[:500]:
        start = time.perf_counter()
        response = client.post(URL, json={"text": sentence})
        service_times.append(time.perf_counter() - start)
        answers.append(response.json()["translation"])
service_times = np.array(service_times)
service = service_times.mean()
mu = 1 / service
median, p99 = np.percentile(service_times, [50, 99]) * 1000
print(f"service time: mean {service * 1000:.0f} ms, median {median:.0f} ms, "
      f"99th percentile {p99:.0f} ms")
print(f"one request at a time, at most {mu:.1f} requests per second")
slowest = service_times.argmax()
print(f"\nthe slowest, {service_times[slowest] * 1000:.0f} ms:")
print(traffic[slowest])
print(answers[slowest])
```

<pre class="tfs-printed" markdown="0"><code>service time: mean 60 ms, median 50 ms, 99th percentile 261 ms
one request at a time, at most 16.7 requests per second

the slowest, 777 ms:
I have not read it, but someone who has read the book &quot;On the origin of species&quot; by Charles Darwin guarantees that he claims that all living beings — plants, insects, reptiles, birds, dinosaurs, whales, man — descend from a common ancestor, from which they evolved, diversifying according to a law he calls &quot;natural selection&quot;.
Руководителями, которые утверждают, что он утверждает, что он утверждает, что он заявляет, что он утверждает, что он говорит, что не умеет читать все, которые предвзяваются в том, что он говорит, что он говорит, что всё равно что предлагал в том, что он говорит, что он говорит, что всё, что подслушивает, что он говорит, что он говорит, что всё, по происхождению, из которых предшествует, что он намеревает, что он не читает в том, что он утверждает, что он говорит, что он</code></pre>

Half the requests take about 50 ms, the mean is 60, and one in a hundred takes more than a quarter of a second.

The slowest took 777 ms, fifteen times the median, and its translation shows why: on a long sentence the model fell into a loop, "он утверждает, что он утверждает…", and wrote until it reached the limit of 128 tokens, each token one more step of the decoder. Here the slowest requests are the model's failures.

A server can bound them by capping the output at a multiple of the input's length. perevod doesn't, because the cap would change translations, and this release promises to change none.

A service that answers one request at a time has a capacity of $$\mu = 1/S$$ requests per second, where $$S$$ is the
mean service time. The load test runs at fractions of it, from a fifth to almost all, thirty seconds each.

It reports two **percentiles** of the latency. The median, p50, is the latency half of the requests beat. The 90th percentile, p90, is the one nine requests in ten beat: one user in ten waits longer.

Dean and Barroso's "slow one time in a hundred" is the 99th percentile, but thirty seconds hold a few hundred requests at most, and their 99th percentile would rest on the slowest two or three. A load test before a release runs for many minutes and reports the 99th percentile and beyond.

```python
LOADS = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95]
one_at_a_time = {}
for load in LOADS:
    rate = load * mu
    one_at_a_time[rate] = await load_test(URL, rate, 30, traffic, seed=int(100 * load))
    p50, p90 = np.percentile(one_at_a_time[rate], [50, 90]) * 1000
    print(f"{rate:5.1f} requests/s, {load:.0%} of capacity: "
          f"p50 {p50:6.0f} ms, p90 {p90:6.0f} ms")
```

<pre class="tfs-printed" markdown="0"><code>  3.3 requests/s, 20% of capacity: p50     59 ms, p90    121 ms
  6.7 requests/s, 40% of capacity: p50     67 ms, p90    147 ms
 10.0 requests/s, 60% of capacity: p50     83 ms, p90    249 ms
 11.7 requests/s, 70% of capacity: p50     99 ms, p90    440 ms
 13.3 requests/s, 80% of capacity: p50    262 ms, p90   1527 ms
 15.0 requests/s, 90% of capacity: p50    192 ms, p90    719 ms
 15.8 requests/s, 95% of capacity: p50    366 ms, p90    874 ms</code></pre>

At a fifth of its capacity the server answers in about its service time: the median request takes 59 ms. The tail moves first.

At 60% the median has grown by 40% and the 90th percentile has doubled; at 95% the median is seven times the median service time, and one request in ten waits nearly a second, though the model works exactly as fast as before. The time went into the queue.

One point is out of line: the test at 80% load came out worse than the one at 90%, with a median of a quarter of
a second and one request in ten waiting a second and a half. Keep it in mind; the next pages explain it.

One check before any theory. This runs on a shared machine, and a cloud CPU can get slower or faster while you measure, when its neighbours start or stop working.

If it did, the service time we measured belongs to another machine than the load tests, and nothing below would add up. The first hundred sentences, one at a time, once more:

```python
again = []
with httpx.Client(timeout=None) as client:
    for sentence in traffic[:100]:
        start = time.perf_counter()
        client.post(URL, json={"text": sentence}).raise_for_status()
        again.append(time.perf_counter() - start)
before, after = 1000 * service_times[:100].mean(), 1000 * np.mean(again)
print(f"the same 100 sentences: {before:.0f} ms before the load tests, "
      f"{after:.0f} ms after")
```

<pre class="tfs-printed" markdown="0"><code>the same 100 sentences: 55 ms before the load tests, 54 ms after</code></pre>

The same speed before and after: the service time we measured is the one the load tests met. On one of the runs that went into writing this lesson it wasn't.

The machine took half again as long per sentence while the service time was measured and was fast again by the load tests, and the predictions near capacity came out three to five times too high. Measure the service time again after a load test; it costs a few seconds.

Queueing theory predicts this curve, and the prediction is worth having before you buy servers. Its formulas are simplest for the mean, so from here on the comparison is about mean latency.

The simplest model is the **M/M/1 queue** (Harchol-Balter, 2013): Poisson arrivals at rate $$\lambda$$, one server, exponentially distributed service times with mean $$S = 1/\mu$$.

The **utilization** $$\rho = \lambda / \mu$$ is the fraction of time the server is busy; it has to stay below 1, or the queue grows without end. The mean time a request spends in the system, waiting plus being served, is then

$$\mathbb{E}[T] = \frac{1}{\mu - \lambda} = \frac{S}{1 - \rho}.$$

The factor $$1/(1 - \rho)$$ is the whole story: at $$\rho = 0.5$$ a request takes twice its service time on average, at $$0.9$$ ten times, at $$0.99$$ a hundred. The last 10% of capacity costs more latency than the first 90%.

That's why services are planned to stay well below full at their busiest hour, often around 60–70% busy, where waiting adds one and a half to two and a half service times and a burst of traffic doesn't push them over the edge.

Real service times are rarely exponential. Kingman's approximation keeps the factor and corrects for how spread
out the service times are<sup class="cite">[<a href="#ref-7" id="cite-8" data-ref="7" aria-label="Reference 7: Kingman, J. F. C. (1961). The single server queue in heavy traffic. Proceedings of the Cambridge Philosophical Society 57(4).">7</a>]</sup>:

$$\mathbb{E}[T] \approx S + \frac{\rho}{1 - \rho} \cdot \frac{c_a^2 + c_s^2}{2} \cdot S,$$

where $$c_a$$ and $$c_s$$ are the coefficients of variation (standard deviation over mean) of the gaps between arrivals and of the service times.

Poisson arrivals have $$c_a = 1$$, and then the formula is exact, not approximate: it's the Pollaczek–Khinchine formula for a single server with any distribution of service times. Exponential service has $$c_s = 1$$ too, and then the formula is M/M/1's. Ours, from the five hundred service times:

```python
c_s = service_times.std() / service_times.mean()
print(f"coefficient of variation of the service time: {c_s:.2f}")
rows = []
for rate, times in one_at_a_time.items():
    rho = rate / mu
    kingman = service + rho / (1 - rho) * (1 + c_s**2) / 2 * service
    rows.append({"load, %": round(100 * rho), "measured mean, ms": 1000 * times.mean(),
                 "M/M/1, ms": 1000 * service / (1 - rho), "Kingman, ms": 1000 * kingman})
theory = pd.DataFrame(rows)
display(table(theory, digits=0))
```

<pre class="tfs-printed" markdown="0"><code>coefficient of variation of the service time: 0.91</code></pre>

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">load, %</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">measured mean, ms</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">M/M/1, ms</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">Kingman, ms</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">20</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">72</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">75</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">74</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">40</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">88</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">100</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">96</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">60</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">122</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">150</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">142</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">70</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">189</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">200</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">188</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">80</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">532</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">300</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">279</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">90</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">311</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">601</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">552</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">95</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">439</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1,201</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1,098</td></tr></tbody></table></div>

Both formulas describe a queue in its **steady state**, after it has run long enough to forget how it started.
Our tests start from an empty queue, and near capacity a queue forgets slowly. For M/M/1 the time it takes, the
relaxation time, is $$S/(1 - \sqrt{\rho})^2$$, which grows like $$1/(1 - \rho)^2$$
<sup class="cite">[<a href="#ref-8" id="cite-9" data-ref="8" aria-label="Reference 8: Morse, P. M. (1955). Stochastic Properties of Waiting Lines. Operations Research 3(3). The relaxation time of a queue.">8</a>]</sup>: about 90 service times at 80% load, 380 at 90% and 1,560
at 95%. A test that starts empty and ends before its queue has settled measures less than the steady state.

And a thirty-second test is one draw of a random process. A burst of arrivals early on leaves a backlog that a
server with little spare capacity clears slowly, and the mean of the whole test goes up with it. How much do such
tests vary? The queue can tell us itself. A single server that answers in order follows **Lindley's recursion**
<sup class="cite">[<a href="#ref-9" id="cite-10" data-ref="9" aria-label="Reference 9: Lindley, D. V. (1952). The theory of queues with a single server. Proceedings of the Cambridge Philosophical Society 48(2).">9</a>]</sup>: a request starts when it arrives or when the
previous one ends, whichever is later. Run it two ways. First, replay each of our tests exactly: the arrival
times it had, and for request $$k$$ the service time we measured for sentence $$k$$, the sentence it sent. Second,
two hundred times on fresh arrivals, with service times drawn at random from the five hundred, which shows how
much thirty-second tests at that load vary:

```python
def simulated_mean(arrivals, service_times):
    done, spent = 0.0, []
    for arrived, service_time in zip(arrivals, service_times, strict=True):
        done = max(arrived, done) + service_time  # on arrival, or when the last one ends
        spent.append(done - arrived)
    return np.mean(spent)


def simulate(arrivals):
    return simulated_mean(arrivals, rng.choice(service_times, len(arrivals)))


for load in [0.8, 0.9, 0.95]:
    print(f"relaxation time at {load:.0%} load: {service / (1 - load**0.5) ** 2:.0f} s")
rng = np.random.default_rng(SEED)
rows = []
for load, (rate, times) in zip(LOADS, one_at_a_time.items(), strict=True):
    own = arrival_times(rate, 30, seed=int(100 * load))  # the arrivals this test had
    sent = np.arange(len(own)) % len(service_times)  # request k sent sentence k
    replay = simulated_mean(own, service_times[sent])
    others = [simulate(arrival_times(rate, 30, seed=1_000 + k)) for k in range(200)]
    low, high = np.percentile(others, [10, 90])
    rows.append({"load, %": round(100 * load), "measured mean, ms": 1000 * times.mean(),
                 "replay, ms": 1000 * replay,
                 "8 in 10 tests: from": 1000 * low, "to": 1000 * high})
simulated = pd.DataFrame(rows)
display(table(simulated, digits=0))
```

<pre class="tfs-printed" markdown="0"><code>relaxation time at 80% load: 5 s
relaxation time at 90% load: 23 s
relaxation time at 95% load: 94 s</code></pre>

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">load, %</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">measured mean, ms</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">replay, ms</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">8 in 10 tests: from</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">to</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">20</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">72</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">67</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">59</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">96</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">40</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">88</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">83</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">73</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">128</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">60</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">122</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">117</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">98</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">198</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">70</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">189</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">182</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">118</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">259</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">80</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">532</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">514</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">161</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">427</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">90</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">311</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">305</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">221</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">903</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">95</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">439</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">460</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">261</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1,243</td></tr></tbody></table></div>

```python
rhos = np.linspace(0.05, 0.97, 100)
fig, ax = plt.subplots(figsize=(8, 3.8))
measured = [1000 * times.mean() for times in one_at_a_time.values()]
ax.plot(rhos, 1000 * (service + rhos / (1 - rhos) * (1 + c_s**2) / 2 * service),
        color=COLORS["att"], label="Kingman's approximation")
ax.plot(rhos, 1000 * service / (1 - rhos), color=COLORS["muted"], linestyle=(0, (3, 3)),
        label="M/M/1")
ax.vlines(LOADS, simulated["8 in 10 tests: from"], simulated["to"],
          color=COLORS["dec"], alpha=0.3, linewidth=7,
          label="8 in 10 simulated 30-second tests")
ax.plot(LOADS, simulated["replay, ms"], "x", color=COLORS["dec"], markersize=8,
        label="replay of the test")
ax.plot(LOADS, measured, "o", color=COLORS["dec"], label="measured mean")
ax.set_xlabel("utilization ρ = λ / μ")
ax.set_ylabel("mean latency, ms")
ax.set_ylim(0, 1.15 * max(max(measured), simulated["to"].max()))
ax.set_title("Latency against load, one request at a time")
ax.legend(frameon=False)
plt.show()
```

<img class="tfs-output" src="/assets/posts/tfs/08/output-02.png" alt="Latency against load, one request at a time" loading="lazy">

Our service times are almost as spread out as an exponential distribution's, $$c_s = 0.91$$, because of the few
sentences the model loops on, so Kingman's curve and M/M/1's nearly coincide. The measured means follow them up
to 70% of capacity. Above that, tests scatter widely, as the bands show, and the 80% point lies outside even its
band.

The replays settle it. Given the arrival times each test had and the service time measured for each sentence it sent, a dozen lines of arithmetic with no server in them reproduce every measured mean to within 7%, the 80% test included.

In that test a burst of arrivals, a quarter more than the server could handle, came between the 18th and the 21st second, together with two of the slowest sentences of the traffic, the looping Darwin among them. The queue they built took most of the remaining test to clear.

In the 90% test the same two sentences came just before a quieter stretch, and their queue was gone three seconds later. So this server is a queue and nothing more, and a thirty-second test is one throw of its dice.

Two practical rules follow. Below about 70% of capacity a short test is enough. Near capacity a single test can land almost anywhere: at 95%, eight thirty-second tests in ten fall between a quarter and a little more than the steady-state mean of Kingman's formula.

That's why the measured 95% point sits so far under the curve: thirty seconds is a third of its relaxation time of 94 seconds, and a test that starts from an empty queue doesn't reach the steady state in that time. Near capacity, run long tests, or many, before you believe a number.

## 8. Dynamic batching

Section 5 showed that one batch of sixteen costs a few times one sentence, not sixteen times. Requests waiting in
a queue are already a batch; the batcher of section 6 only takes them together: the first request, whatever else
arrives within a few milliseconds, all translated at once.

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/08/batching-light.png" width="760" alt="Requests arriving over time, one at a time and with dynamic batching. One at a time, each request waits for all those before it. With dynamic batching, the requests that arrive while a batch is being translated go into the next batch together, so the queue empties in a few batches instead of many single translations" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/08/batching-dark.png" width="760" alt="Requests arriving over time, one at a time and with dynamic batching. One at a time, each request waits for all those before it. With dynamic batching, the requests that arrive while a batch is being translated go into the next batch together, so the queue empties in a few batches instead of many single translations" loading="lazy">

The cost is a wait of at most `max_wait` for the first request of a batch when the server is idle; the gain is that under load the server translates many requests at the price of a few.

Section 5 also gives a forecast. Its batches of sixteen ran at about four times the throughput of single sentences.

The server will do worse: it batches whatever arrives, short and long sentences together, and every request in a batch waits for the longest. Two to four times the capacity is a fair guess.

The same server with batches of up to sixteen, at the same loads with the same arrivals, and at loads it couldn't
take before:

```python
server.terminate()
server.wait()
server = start_server(8002, max_batch=16, max_wait_ms=5)
URL = "http://127.0.0.1:8002/translate"
batched = {}
for load in LOADS + [1.5, 2, 2.5, 3]:
    rate = load * mu
    batched[rate] = await load_test(URL, rate, 30, traffic, seed=int(100 * load))
    p50, p90 = np.percentile(batched[rate], [50, 90]) * 1000
    print(f"{rate:5.1f} requests/s, {load:.0%} of the capacity one at a time: "
          f"p50 {p50:6.0f} ms, p90 {p90:6.0f} ms")
```

<pre class="tfs-printed" markdown="0"><code>  3.3 requests/s, 20% of the capacity one at a time: p50     61 ms, p90    131 ms
  6.7 requests/s, 40% of the capacity one at a time: p50     69 ms, p90    138 ms
 10.0 requests/s, 60% of the capacity one at a time: p50     77 ms, p90    164 ms
 11.7 requests/s, 70% of the capacity one at a time: p50     88 ms, p90    216 ms
 13.3 requests/s, 80% of the capacity one at a time: p50    103 ms, p90    310 ms
 15.0 requests/s, 90% of the capacity one at a time: p50    104 ms, p90    279 ms
 15.8 requests/s, 95% of the capacity one at a time: p50    104 ms, p90    228 ms
 25.0 requests/s, 150% of the capacity one at a time: p50    162 ms, p90    439 ms
 33.3 requests/s, 200% of the capacity one at a time: p50    265 ms, p90    867 ms
 41.6 requests/s, 250% of the capacity one at a time: p50    573 ms, p90   1449 ms
 50.0 requests/s, 300% of the capacity one at a time: p50   1392 ms, p90   2948 ms</code></pre>

```python
fig, ax = plt.subplots(figsize=(8, 3.8))
for runs, colour, name in [(one_at_a_time, COLORS["dec"], "one at a time"),
                           (batched, COLORS["enc"], "dynamic batching, up to 16")]:
    rates = list(runs)
    p50 = [1000 * np.percentile(runs[rate], 50) for rate in rates]
    p90 = [1000 * np.percentile(runs[rate], 90) for rate in rates]
    ax.plot(rates, p50, "o-", color=colour, label=f"{name}: median")
    ax.plot(rates, p90, "o--", color=colour, alpha=0.6, label=f"{name}: 90th percentile")
ax.axvline(mu, color=COLORS["muted"], linewidth=1, linestyle=(0, (3, 3)))
ax.text(mu, ax.get_ylim()[1] * 0.92, " capacity one at a time", color=COLORS["muted"],
        fontsize=9)
ax.set_yscale("log")
ax.set_xlabel("requests per second")
ax.set_ylabel("latency, ms (log scale)")
ax.set_title("The same server, without and with dynamic batching")
ax.legend(frameon=False, fontsize=8)
plt.show()
```

<img class="tfs-output" src="/assets/posts/tfs/08/output-03.png" alt="The same server, without and with dynamic batching" loading="lazy">

At the loads the server could already take, batching changes little at the bottom and a lot near the top.

At 20% of the old capacity the two servers answer alike, the batching one a couple of milliseconds slower, the price of `max_wait`. At 95% the median falls from 366 ms to 104 and the 90th percentile from 874 ms to 228.

And the server now takes loads it couldn't take at all: at one and a half times the old capacity the median is 162 ms, at twice 265 ms. At two and a half times the latency climbs, and at three the queue grows for as long as the test lasts.

The capacity grew about two and a half times, inside the forecast and short of section 5's fourfold, because a batch waits for its longest sentence.

Two settings remain to choose, and the measurements above say how. `max_wait` is added latency when the server is idle, and it hardly matters under load, because the queue fills batches while the model works: keep it a small fraction of the service time.

`max_batch` belongs where section 5's throughput flattens. Past that point a bigger batch adds little throughput, and every request in it waits for all of it.

One thing this server doesn't do. Its queue has no limit, so beyond capacity the latency grows for as long as the
overload lasts, and the last points of the plot measure the length of the test more than the server. A
production service caps its queue and answers 503, "try later", to the requests it can't take.

Large language model servers take the same idea further. A request to an LLM generates hundreds of tokens, and waiting for a whole batch to finish would hold short answers behind long ones, so vLLM and its predecessors batch at the level of single decoding steps: a finished sequence leaves the batch after its last token and a new one joins at the next step<sup class="cite">[<a href="#ref-10" id="cite-11" data-ref="10" aria-label="Reference 10: Yu, G.-I. et al. (2022). Orca: A Distributed Serving System for Transformer-Based Generative Models. OSDI. Iteration-level scheduling.">10</a>, <a href="#ref-11" id="cite-12" data-ref="11" aria-label="Reference 11: Kwon, W. et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP. vLLM.">11</a>]</sup>.

Our sentences are short enough that batching whole requests is the right trade.

## 9. A Gradio page

An API is for programs. For people, a page with a text box helps more. [Gradio](https://www.gradio.app/docs)
builds one from a few lines, and it can live inside the same FastAPI app, so the page and the API share one model
and one queue:

```python
%%writefile perevod/src/perevod/demo.py
import gradio as gr

EXAMPLES = ["My brother lives in a small town by the sea.",
            "Every morning I drink tea and read the news.",
            "The cat is sleeping on the windowsill.", "Tom is afraid of the dark."]
ABOUT = """# perevod
English to Russian with a small Transformer built from scratch in the course
[Transformers: from scratch to production][course]. It was trained on short everyday
sentences; its [model card][card] says where it fails.

[course]: https://github.com/xmarva/transformers-from-scratch
[card]: https://huggingface.co/xmarva/perevod-en-ru"""


def build_demo(app):
    async def translate(text):
        return await app.state.batcher.translate(text)  # through the same queue as the API

    with gr.Blocks(title="perevod") as demo:
        gr.Markdown(ABOUT)
        source = gr.Textbox(label="English", lines=3)
        target = gr.Textbox(label="Russian", lines=3)
        button = gr.Button("Translate", variant="primary")
        # no limit: Gradio would otherwise send its visitors one at a time
        button.click(translate, source, target, concurrency_limit=None)
        gr.Examples(EXAMPLES, source)
    return demo
```

<pre class="tfs-printed" markdown="0"><code>Writing perevod/src/perevod/demo.py</code></pre>

The page translates through `app.state.batcher`, so people on the page and programs calling the API wait in the same queue and fill the same batches.

One argument makes that true: Gradio sends a page's clicks through a queue of its own, one at a time by default, and `concurrency_limit=None` lets them all through to ours.

Gradio is a heavy dependency, so it is an extra of its own, `demo`, and the page appears only with `--demo`. Here is what it looks like:

<img class="tfs-figure" src="/assets/posts/tfs/08/gradio.png" width="700" alt="The Gradio page of perevod: a short description with links to the course and the model card, an English text box with &#x27;My brother lives in a small town by the sea.&#x27;, the Russian box below it with &#x27;Мой брат живёт в маленьком городке у моря.&#x27;, a Translate button and four example sentences" loading="lazy">

```python
server.terminate()
server.wait()
!{sys.executable} -m pip install -q -e "./perevod[demo]"
server = start_server(8003, max_batch=16, demo=True)
page = httpx.get("http://127.0.0.1:8003/")
answer = httpx.post("http://127.0.0.1:8003/translate", json={"text": "Do you dream?"})
server.terminate()
server.wait()
print("the page:", page.status_code, "| the API:", answer.json())
```

<pre class="tfs-printed" markdown="0"><code>  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Installing backend dependencies ... done
  Preparing editable metadata (pyproject.toml) ... done
  Building editable for perevod (pyproject.toml) ... done
ERROR: pip&#x27;s dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
dopamine-rl 4.1.2 requires gym&lt;=0.25.2, but you have gym 0.26.2 which is incompatible.
the page: 200 | the API: {&#x27;translation&#x27;: &#x27;Ты видишь сны?&#x27;}</code></pre>

Both answer. On Kaggle, pip also prints an error about `dopamine-rl` and `gym`, two packages of Kaggle's own
environment that disagree with each other; it has nothing to do with ours.

## 10. Tests and the release of 0.2.0

The server gets tests of its own, on lesson 7's tiny model. pytest runs the app in-process: FastAPI's `TestClient` calls it without a network and runs its lifespan, so the batcher is real.

The last test matters most: requests that end up in one batch must each get their own translation back. It's lesson 7's "translations come back in the wrong order" one level up, where request A would get request B's translation.

`TestClient` sends one request and waits for its answer, so it can never put two requests in one batch. That test sends all four at once from an asynchronous HTTPX client instead.

`ASGITransport` lets HTTPX call the app directly, without uvicorn or a network, but it doesn't run the lifespan, so the test enters it itself, and `asyncio.run` gives the test an event loop of its own. Then it checks that the batcher really saw one batch of four.

```python
%%writefile perevod/tests/test_server.py
import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from perevod.server import create_app


@pytest.fixture
def client(tiny_dir):
    app = create_app(str(tiny_dir), max_wait=0.05)
    with TestClient(app) as client:  # entering it runs the lifespan: model and batcher
        yield client


def test_the_health_check(client):
    assert client.get("/health").json()["status"] == "ok"


def test_the_api_translates_like_the_package(client, translator, pairs):
    english = next(iter(pairs))
    response = client.post("/translate", json={"text": english})
    assert response.json() == {"translation": translator.translate(english)}


def test_a_text_too_long_is_refused(client):
    assert client.post("/translate", json={"text": "tom " * 2000}).status_code == 422


def test_requests_in_one_batch_get_their_own_translations(tiny_dir, pairs):
    app = create_app(str(tiny_dir), max_wait=0.2)

    async def ask_all():
        async with app.router.lifespan_context(app):  # the model and the batcher
            transport = httpx.ASGITransport(app=app)
            base_url = "http://perevod"
            async with httpx.AsyncClient(transport=transport, base_url=base_url) as client:
                requests = []
                for english in pairs:
                    requests.append(client.post("/translate", json={"text": english}))
                return await asyncio.gather(*requests)

    responses = asyncio.run(ask_all())
    assert app.state.batcher.batch_sizes == [len(pairs)]  # one batch for all of them
    for response, russian in zip(responses, pairs.values(), strict=True):
        assert response.json()["translation"] == russian
```

<pre class="tfs-printed" markdown="0"><code>Writing perevod/tests/test_server.py</code></pre>

Rereading lesson 7 for this release turned up two gaps in its tests, and 0.2.0 closes them.

Lesson 7 planted "dropout left on" to see which tests notice, and the test that the same input gives the same translation was one of them.

Then section 11 made the tiny model learn its four pairs by heart, and the model became too sure of itself for dropout to change a word: that test stopped noticing, as lesson 7's second grid shows.

A test that compares outputs can lose its power this way when the model changes; the direct check can't. The file with one more test at the end:

```python
%%writefile perevod/tests/test_translator.py
import pytest


def test_an_empty_input_has_an_empty_translation(translator):
    assert translator.translate(["", "   ", "tom reads"])[:2] == ["", ""]


def test_a_string_gives_a_string_and_a_list_gives_a_list(translator):
    assert isinstance(translator.translate("tom reads"), str)
    assert isinstance(translator.translate(["tom reads"]), list)


def test_spaces_and_line_breaks_change_nothing(translator):
    expected = translator.translate("tom reads a book")
    assert translator.translate("tom  reads\na book") == expected


def test_a_long_input_is_cut_with_a_warning(translator):
    long_text = " ".join(["tom reads a book ."] * 10)  # 50 tokens, the model has 32 places
    with pytest.warns(UserWarning, match="cut"):
        translation = translator.translate(long_text)
    assert isinstance(translation, str)


def test_the_translator_switches_dropout_off(translator):
    assert not translator.model.training
```

<pre class="tfs-printed" markdown="0"><code>Overwriting perevod/tests/test_translator.py</code></pre>

The second gap is in CI. The wheel job ran only `perevod --version`, and thanks to lesson 7's light `import perevod` that command loads neither PyTorch nor the model code: a wheel without `decoding.py` would have passed.

The job now installs the wheel with its test extra and runs the fast tests against it, from a folder outside the source tree. And the quality job now starts the moment someone adds the label `model` to a pull request, not at the next push:

```python
write(".github/workflows/perevod.yml", """
name: perevod

on:
  push:
    branches: [main]
    paths: ["perevod/**", ".github/workflows/perevod.yml"]
  pull_request:
    types: [opened, synchronize, reopened, labeled]  # labeled: see the quality job
    paths: ["perevod/**", ".github/workflows/perevod.yml"]
  schedule:
    - cron: "0 6 * * 1"  # Mondays: the published model against the golden file

defaults:
  run:
    working-directory: perevod  # the package lives in a folder of the course repository

env:
  # PyTorch's CPU build, without gigabytes of CUDA libraries
  PIP_EXTRA_INDEX_URL: https://download.pytorch.org/whl/cpu

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ["3.10", "3.13"]  # the oldest Python the package promises, and the newest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: ${{ matrix.python }}
      - run: pip install -e ".[test]"
      - run: ruff check .
      - run: pytest -q -m "not slow"

  wheel:
    # tests import the package from its folder; this job checks what users install
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.12"
      - run: pip wheel --no-deps --wheel-dir dist .
      - run: pip install "$(ls dist/*.whl)[test]"
      - name: The installed command works outside the source tree
        working-directory: /tmp
        run: perevod --version
      - name: The fast tests pass against the installed wheel, outside the source tree
        working-directory: /tmp
        run: pytest -q -p no:cacheprovider -m "not slow" "$GITHUB_WORKSPACE/perevod/tests"

  quality:
    # slow, downloads the model: on main, on Mondays, on pull requests labelled "model"
    if: >-
      github.event_name != 'pull_request' ||
      contains(github.event.pull_request.labels.*.name, 'model')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.12"
      - uses: actions/cache@v6
        with:
          path: ~/.cache/huggingface
          key: huggingface-${{ hashFiles('perevod/src/perevod/translator.py') }}
      - run: pip install -e ".[test]"
      - run: pytest -q -m slow
""")
```

```python
!{sys.executable} -m pip install -q -e "./perevod[test,serve]"
!cd perevod && {sys.executable} -m pytest -q -p no:cacheprovider --color=no
```

<pre class="tfs-printed" markdown="0"><code>  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Installing backend dependencies ... done
  Preparing editable metadata (pyproject.toml) ... done
  Building editable for perevod (pyproject.toml) ... done
..........................                                               [100%]
26 passed in 12.65s</code></pre>

All of them, the slow ones included. The golden test of lesson 7 is the statement this release rests on: the cached beam search, the new decoding step and the reordered caches produce exactly what 0.1.0 produced on the twelve golden sentences, the chrF test keeps its floor, and section 3 compared 530 more sentences word for word.

That makes 0.2.0 a **minor** release in lesson 7's terms: new features (the server, the page, the speed), and nothing that changes a translation or breaks a call that worked in 0.1.0.

The release is a commit and a tag, `perevod-v0.2.0`, in the course repository. On your own machine it installs
and starts like this (on Linux without a GPU, add `--extra-index-url https://download.pytorch.org/whl/cpu` to the
first command, or pip downloads gigabytes of CUDA libraries with PyTorch):

```bash
pip install "perevod[demo] @ git+https://github.com/xmarva/transformers-from-scratch@perevod-v0.2.0#subdirectory=perevod"
perevod-server --port 8000 --demo
```

Then open http://127.0.0.1:8000 for the page and http://127.0.0.1:8000/docs for the API, or ask from a terminal:

```bash
curl -X POST http://127.0.0.1:8000/translate -H "Content-Type: application/json" -d '{"text": "Hello!"}'
```

## 11. The same for your own model

Nothing in the method depends on perevod. Measure the service time first, one request at a time, back to back, on inputs like the ones your users send, and compute the capacity $$\mu = 1/S$$.

Then run open-loop tests at a range of loads, from a fifth of $$\mu$$ to almost all of it; `arrival_times` and `load_test` above work against any HTTP endpoint.

Near capacity, run each load for many relaxation times of the queue, minutes rather than seconds, leave out the first stretch while the queue fills up, and count errors along with latencies. Read the median and the 90th percentile, and the 99th only from thousands of requests.

Then put the means next to Kingman's curve. If they follow it, you understand your server and can predict what more traffic will do to it.

If they don't, something besides the queue is at work: another process taking the CPU, or service times measured on the wrong inputs. Plan for the busiest hour at 60–70% of capacity; if that's not enough, batch, and measure again.

## References

1. {:#ref-1} Linden, G. (2006). [Make Data Useful](https://glinden.blogspot.com/2006/12/slides-from-my-talk-at-stanford.html), a talk at Stanford. *The slide with Amazon's 100 ms and Google's 500 ms.* <a class="ref-back" href="#cite-1" aria-label="Back to the text">&#8617;</a>
2. {:#ref-2} Brutlag, J. (2009). [Speed Matters for Google Web Search](https://services.google.com/fh/files/blogs/google_delayexp.pdf). Google. <a class="ref-back" href="#cite-2" aria-label="Back to the text">&#8617;</a>
3. {:#ref-3} Dean, J., Barroso, L. A. (2013). [The Tail at Scale](https://doi.org/10.1145/2408776.2408794). Communications of the ACM 56(2). <a class="ref-back" href="#cite-3" aria-label="Back to the text">&#8617;</a>
4. {:#ref-4} PyTorch: [dynamic quantization](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html), [CPU threading](https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html). <a class="ref-back" href="#cite-4" aria-label="Back to the text">&#8617;</a>
5. {:#ref-5} [NVIDIA Triton: dynamic batcher](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html). <a class="ref-back" href="#cite-6" aria-label="Back to the text">&#8617;</a>
6. {:#ref-6} Tene, G. (2013). [Coordinated Omission](https://groups.google.com/g/mechanical-sympathy/c/icNZJejUHfE/m/BfDekfBEs_sJ), on the mechanical-sympathy list; and the talk [How NOT to Measure Latency](https://www.youtube.com/watch?v=lJ8ydIuPFeU) (2015). <a class="ref-back" href="#cite-7" aria-label="Back to the text">&#8617;</a>
7. {:#ref-7} Kingman, J. F. C. (1961). [The single server queue in heavy traffic](https://doi.org/10.1017/S0305004100036094). Proceedings of the Cambridge Philosophical Society 57(4). <a class="ref-back" href="#cite-8" aria-label="Back to the text">&#8617;</a>
8. {:#ref-8} Morse, P. M. (1955). [Stochastic Properties of Waiting Lines](https://doi.org/10.1287/opre.3.3.255). Operations Research 3(3). *The relaxation time of a queue.* <a class="ref-back" href="#cite-9" aria-label="Back to the text">&#8617;</a>
9. {:#ref-9} Lindley, D. V. (1952). [The theory of queues with a single server](https://doi.org/10.1017/S0305004100027638). Proceedings of the Cambridge Philosophical Society 48(2). <a class="ref-back" href="#cite-10" aria-label="Back to the text">&#8617;</a>
10. {:#ref-10} Yu, G.-I. et al. (2022). [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu). OSDI. *Iteration-level scheduling.* <a class="ref-back" href="#cite-11" aria-label="Back to the text">&#8617;</a>
11. {:#ref-11} Kwon, W. et al. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). SOSP. *vLLM.* <a class="ref-back" href="#cite-12" aria-label="Back to the text">&#8617;</a>
12. {:#ref-12} Open-loop load generators: [wrk2](https://github.com/giltene/wrk2); k6, [open and closed models](https://grafana.com/docs/k6/latest/using-k6/scenarios/concepts/open-vs-closed/).
13. {:#ref-13} Harchol-Balter, M. (2013). *Performance Modeling and Design of Computer Systems: Queueing Theory in Action*. Cambridge University Press. *The M/M/1 queue.*
14. {:#ref-14} [FastAPI](https://fastapi.tiangolo.com), [Uvicorn](https://www.uvicorn.org), [Gradio](https://www.gradio.app/docs), [HTTPX](https://www.python-httpx.org).
{: .tfs-refs}





{% endraw %}
