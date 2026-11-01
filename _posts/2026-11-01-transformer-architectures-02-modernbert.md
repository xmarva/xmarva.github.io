---
layout: post
title: "From RoBERTa to ModernBERT: what seven years did to BERT"
date: 2026-11-01 11:00:00
description: "What seven years did to BERT: RoBERTa's training recipe, ELECTRA's replaced-token detection, DeBERTa's disentangled attention and ModernBERT's LLM-era block, each written in PyTorch and checked against the real weights, and the four compared on triaging GitHub issues of projects they never saw."
tags: [transformer-architectures, roberta, electra, deberta, modernbert, nlp]
categories: machine-learning
thumbnail: /assets/posts/arch/02/cover.png
og_image: https://xmarva.github.io/assets/posts/arch/02/cover.png
stylesheets: [/assets/css/tfs-course.css]
toc:
  beginning: true
---
{% raw %}

<p class="tfs-links"><strong>Transformer architectures: from BERT to diffusion</strong> · lesson 2 of 15 · <a href="/teaching/">all lessons</a> · <a href="https://www.kaggle.com/code/qmarva/transformer-architectures-02-roberta-modernbert">run on Kaggle</a> · <a href="https://colab.research.google.com/github/xmarva/transformers-from-scratch/blob/main/notebooks/colab/a02_modernbert.ipynb">open in Colab</a> · <a href="https://github.com/xmarva/transformers-from-scratch">code on GitHub</a></p>

Nine months after BERT, a team at Facebook AI and the University of Washington trained it again. They kept the architecture, every layer and every head, and changed only how it was trained: longer, on more data, with bigger batches and without one of its two training games.

The abstract of their paper says what came out: BERT "was significantly undertrained, and can match or exceed the performance of every model published after it" <sup class="cite">[<a href="#ref-1" id="cite-1" data-ref="1" aria-label="Reference 1: Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., &amp; Stoyanov, V. (2019). RoBERTa: a robustly optimized BERT pretraining approach. arXiv:1907.11692.">1</a>]</sup>.

That sentence sets the tone for the next five years of encoders. ELECTRA changed the training task, DeBERTa changed
how attention sees positions, and only in December 2024 did ModernBERT rebuild the block itself, out of the parts
that large language models had made standard in the meantime.

<img class="tfs-figure tfs-light" src="/assets/posts/arch/02/hero-light.png" width="760" alt="Four encoders after BERT, each with the one thing it changed: RoBERTa in 2019 the training recipe, ELECTRA in 2020 the training task, DeBERTa in 2020 the attention, ModernBERT in 2024 the whole block" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/arch/02/hero-dark.png" width="760" alt="Four encoders after BERT, each with the one thing it changed: RoBERTa in 2019 the training recipe, ELECTRA in 2020 the training task, DeBERTa in 2020 the attention, ModernBERT in 2024 the whole block" loading="lazy">

In this lesson you will write each change yourself, check it against the real weights, and then put the four
generations to work on a job engineers know well: sorting GitHub issues into bugs, feature requests, questions and
documentation, for projects the models have never seen.

**What you'll do**

1. Load RoBERTa into the BERT of lesson 1 with two small changes, and see what byte-level BPE does with code.
2. Write ELECTRA's replaced-token detection and race it against masked language modelling on a T4.
3. Write DeBERTa's disentangled attention and learn why its check needs float64.
4. Write ModernBERT: rotary positions, GeGLU, no biases, and layers that alternate local and global attention.
5. Fine-tune BERT, RoBERTa, DeBERTa-v3 and ModernBERT to triage issues of unseen repositories, and measure what
   reading the whole issue is worth.

**What you need**: a GPU. On Kaggle, *Settings → Accelerator → GPU*; on Colab, *Runtime → Change runtime type →
T4 GPU*. The notebook downloads four pre-trained encoders and the issues of thirteen open-source projects; on a T4
it runs in about an hour.

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
import ast
import json
import math
import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from IPython.display import display
from safetensors.torch import load_file
from tfs.viz import heatmap
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# the reference implementations, for the checks
from transformers import DebertaV2Model, ModernBertModel, RobertaModel

# mixed precision skips a step when a gradient overflows; PyTorch reports it as a warning
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step")
SMOKE = env.smoke()  # CI runs the lesson on a small slice; you run it in full
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda" or SMOKE, "This lesson fine-tunes several encoders: use a GPU."
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
```

<pre class="tfs-printed" markdown="0"><code>GPU: Tesla T4</code></pre>

## 1. RoBERTa: the same BERT, trained properly

RoBERTa's authors went through BERT's training choices one at a time, and most of the gain came from the plainest ones. BERT trained with batches of 256 sequences; RoBERTa with 8,000. BERT read 16 GB of text; RoBERTa "over 160GB", for longer.

The cleverer-sounding changes mattered less. BERT masked each sentence once, when the data was prepared; RoBERTa draws new masks every time a sequence is used, as the `mask_tokens` of lesson 1 does, and found this "comparable or slightly better".

Dropping BERT's second game, telling a real next sentence from a random one, "matches or slightly improves" the results, so the inputs became whole stretches of text instead of pairs.

The architecture is BERT's. So our BERT from lesson 1 should take RoBERTa's weights, and it does, with two changes
that are left-overs of the code RoBERTa was trained with. Here is lesson 1's model again, in one cell:

```python
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config["hidden_size"]
        self.words = nn.Embedding(config["vocab_size"], d)
        self.positions = nn.Embedding(config["max_position_embeddings"], d)
        self.segments = nn.Embedding(config["type_vocab_size"], d)
        self.norm = nn.LayerNorm(d, eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        # RoBERTa numbers positions from pad_token_id + 1: fairseq kept 0 and 1 for padding
        self.first = config["pad_token_id"] + 1 if config["model_type"] == "roberta" else 0

    def forward(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device) + self.first
        segments = torch.zeros_like(ids)  # one sentence per input
        x = self.words(ids) + self.positions(positions) + self.segments(segments)
        return self.dropout(self.norm(x))


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, eps = config["hidden_size"], config["layer_norm_eps"]
        self.heads = config["num_attention_heads"]
        self.q, self.k, self.v, self.out = (nn.Linear(d, d) for _ in range(4))
        self.norm1 = nn.LayerNorm(d, eps=eps)
        self.up = nn.Linear(d, config["intermediate_size"])
        self.down = nn.Linear(config["intermediate_size"], d)
        self.norm2 = nn.LayerNorm(d, eps=eps)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def split(self, x):
        batch, length, d = x.shape
        return x.view(batch, length, self.heads, d // self.heads).transpose(1, 2)

    def forward(self, x, padding):
        q, k, v = self.split(self.q(x)), self.split(self.k(x)), self.split(self.v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        lowest = torch.finfo(scores.dtype).min  # not -inf: see section 4
        scores = scores.masked_fill(padding[:, None, None, :], lowest)
        y = (self.dropout(scores.softmax(dim=-1)) @ v).transpose(1, 2).reshape(x.shape)
        x = self.norm1(x + self.dropout(self.out(y)))
        return self.norm2(x + self.dropout(self.down(F.gelu(self.up(x)))))


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        n_layers = config["num_hidden_layers"]
        self.layers = nn.ModuleList(Layer(config) for _ in range(n_layers))

    def forward(self, ids, padding):
        x = self.embeddings(ids)
        for layer in self.layers:
            x = layer(x, padding)
        return x
```

The second left-over is in the embedding table's shape: RoBERTa has one segment, not two, since it never packs two
sentences for the abandoned game, and 514 positions, two of them never used. The names of its tensors differ from
Google's too; they follow PyTorch's conventions, `weight` and `bias`, because RoBERTa was trained in PyTorch.

```python
def roberta_names(config):
    names = {
        "embeddings.words.weight": "roberta.embeddings.word_embeddings.weight",
        "embeddings.positions.weight": "roberta.embeddings.position_embeddings.weight",
        "embeddings.segments.weight": "roberta.embeddings.token_type_embeddings.weight",
        "embeddings.norm.weight": "roberta.embeddings.LayerNorm.weight",
        "embeddings.norm.bias": "roberta.embeddings.LayerNorm.bias",
    }
    parts = {"q": "attention.self.query", "k": "attention.self.key",
             "v": "attention.self.value", "out": "attention.output.dense",
             "norm1": "attention.output.LayerNorm", "up": "intermediate.dense",
             "down": "output.dense", "norm2": "output.LayerNorm"}
    for i in range(config["num_hidden_layers"]):
        for ours, theirs in parts.items():
            for p in ("weight", "bias"):
                names[f"layers.{i}.{ours}.{p}"] = f"roberta.encoder.layer.{i}.{theirs}.{p}"
    return names


ROBERTA = "FacebookAI/roberta-base"
with open(hf_hub_download(ROBERTA, "config.json")) as f:
    roberta_config = json.load(f)
weights = load_file(hf_hub_download(ROBERTA, "model.safetensors"))
roberta = Encoder(roberta_config)
state = {ours: weights[theirs] for ours, theirs in roberta_names(roberta_config).items()}
roberta.load_state_dict(state, strict=True)
roberta = roberta.to(device).eval()

roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA)
reference = RobertaModel.from_pretrained(ROBERTA, add_pooling_layer=False).to(device).eval()
texts = ["Segfault in torch.compile() with dynamic shapes 🚀", "Docs typo"]
check = roberta_tokenizer(texts, padding=True, return_tensors="pt").to(device)
padding = check["attention_mask"] == 0
with torch.no_grad():
    theirs = reference(**check).last_hidden_state
    ours = roberta(check["input_ids"], padding)
gap = (ours - theirs)[~padding].abs().max().item()
print(f"largest difference from Hugging Face's RoBERTa: {gap:.1e}")
assert gap < 1e-3
del reference
```

<pre class="tfs-printed" markdown="0"><code>largest difference from Hugging Face&#x27;s RoBERTa: 5.7e-06</code></pre>

RoBERTa's other visible change is its tokenizer. BERT's WordPiece splits text into characters it knows, and a character it doesn't know becomes `[UNK]`, the same token for an emoji, a Chinese name and a Greek letter.

RoBERTa borrowed GPT-2's **byte-level BPE**: the text is first turned into bytes, and any byte sequence can be spelled out with 256 byte tokens, so nothing is ever unknown.

The `Ġ` you'll see marks a token that starts with a space, and pairs like `Ã¯` are the two bytes of one character, *ï*, printed one by one.

```python
bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
samples = ["x = torch.compile(model)  # 🚀", "naïve café in Zürich, Ελλάδα"]
rows, labels = [], []
for text in samples:
    for name, tok in [("WordPiece", bert_tokenizer), ("byte BPE", roberta_tokenizer)]:
        rows.append(tok.tokenize(text))
        labels.append(name)
token_chips(rows, labels=labels, max_width=640,
            strong=lambda row, col, token: token == "[UNK]")
for name, tok in [("WordPiece", bert_tokenizer), ("byte BPE", roberta_tokenizer)]:
    unknown = sum(t == tok.unk_token for text in samples for t in tok.tokenize(text))
    print(f"{name}: {unknown} unknown tokens")
```

<pre class="tfs-printed" markdown="0"><code>WordPiece: 1 unknown tokens
byte BPE: 0 unknown tokens</code></pre>

<img class="tfs-output" src="/assets/posts/arch/02/output-01.png" alt="Output of the code above" loading="lazy">

WordPiece also lowercased everything and stripped the accents; byte BPE keeps the text exactly, and turns what it
hasn't seen often into more, smaller pieces, sometimes single bytes of a multi-byte character. For text that
mixes prose, code, stack traces and emoji, which is what GitHub issues are, that matters.

## 2. ELECTRA: learning from every token

Masked language modelling has a cost that lesson 1 mentioned in passing: the model learns only from the 15% of tokens it hides. ELECTRA, from Stanford and Google, turned the game around <sup class="cite">[<a href="#ref-2" id="cite-2" data-ref="2" aria-label="Reference 2: Clark, K., Luong, M.-T., Le, Q. V., &amp; Manning, C. D. (2020). ELECTRA: pre-training text encoders as discriminators rather than generators. ICLR 2020.">2</a>]</sup>. A small masked language model, the **generator**, fills the masked positions with its own guesses.

The model we actually want, the **discriminator**, sees the result and says for *every* token whether it is the original or a replacement. This is **replaced token detection**, and its headline result was a small model trained "on one GPU for 4 days that outperforms GPT (trained using 30x more compute)".

<img class="tfs-figure tfs-light" src="/assets/posts/arch/02/electra-light.png" width="760" alt="Replaced token detection. A sentence has some tokens masked; a small generator fills each mask with a plausible word, sometimes the right one; the discriminator reads the result and labels every token as original or replaced, so it learns from all of them" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/arch/02/electra-dark.png" width="760" alt="Replaced token detection. A sentence has some tokens masked; a small generator fills each mask with a plausible word, sometimes the right one; the discriminator reads the result and labels every token as original or replaced, so it learns from all of them" loading="lazy">

The generator is trained as an ordinary masked language model, "rather than being trained adversarially to fool the discriminator", and it is small: the paper recommends a quarter to half of the discriminator's size.

Its guesses are good enough to be hard to spot, and when it happens to guess the right word, that token counts as original. After pre-training the generator is thrown away.

To see what this buys we'll pre-train two small encoders from scratch, one with each task, for the same number of minutes on the same GPU, and then compare them on the job of section 5.

The text they learn from is GitHub issues themselves, which we load now. The dataset holds the issues of seventeen large projects with their labels <sup class="cite">[<a href="#ref-3" id="cite-3" data-ref="3" aria-label="Reference 3: open-index (2026). Open GitHub issues. Hugging Face dataset, ODC-By.">3</a>]</sup>; every project labels its issues its own way, so we map the labels to four kinds.

```python
ISSUES = "open-index/open-github-issues"
TRAIN_REPOS = ["python/cpython", "rust-lang/rust", "microsoft/TypeScript",
               "kubernetes/kubernetes", "pingcap/tidb", "swiftlang/swift",
               "ClickHouse/ClickHouse", "golang/go"]
TEST_REPOS = ["moby/moby", "etcd-io/etcd", "facebook/react", "redis/redis", "vuejs/core"]
KINDS = ["bug", "feature", "question", "docs"]
RULES = [("docs", r"\bdoc(s|umentation)?\b"), ("question", r"\bquestion\b|\bsupport\b"),
         ("feature", r"feature|enhancement|suggestion|proposal|improvement"),
         ("bug", r"\bbug\b|crash|unexpected behaviour")]
SKIP = r"not a bug|working as intended|duplicate|spam|question-answered"


def kind_of(labels):
    found = set()
    for label in labels:
        low = label.lower()
        if re.search(SKIP, low):
            return None
        for name, pattern in RULES:
            if re.search(pattern, low):
                found.add(name)
    return found.pop() if len(found) == 1 else None  # one kind, or the issue is left out


def load_issues(repo):
    path = hf_hub_download(ISSUES, f"data/issues/{repo}/0.parquet", repo_type="dataset")
    frame = pd.read_parquet(path, columns=["is_pull_request", "title", "body", "labels"])
    frame = frame[frame["is_pull_request"].astype(str) == "False"]
    if SMOKE:
        frame = frame.head(3000)
    labels = []
    for x in frame["labels"]:  # a list, the string of a list, or "null" for no labels
        if isinstance(x, str):
            labels.append(ast.literal_eval(x) if x.startswith("[") else [])
        else:
            labels.append(list(x) if x is not None else [])
    frame = frame.assign(kind=[kind_of(ls) for ls in labels], repo=repo)
    frame = frame[frame["kind"].notna()]
    text = frame["title"].fillna("") + "\n\n" + frame["body"].fillna("")
    return frame.assign(text=text)[["repo", "kind", "text"]]


issues = pd.concat([load_issues(r) for r in TRAIN_REPOS + TEST_REPOS], ignore_index=True)
table(pd.crosstab(issues["repo"], issues["kind"], margins=True).reset_index())
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">repo</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">bug</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">docs</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">feature</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">question</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">All</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ClickHouse/ClickHouse</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">8,429</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">456</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">4,570</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">3,362</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">16,817</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">etcd-io/etcd</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">850</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">183</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">690</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">449</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2,172</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">facebook/react</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1,417</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">579</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">501</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2,497</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">golang/go</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">3,137</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">7,899</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">46</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">11,082</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">kubernetes/kubernetes</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">14,636</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1,034</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">6,025</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2,316</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">24,011</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">microsoft/TypeScript</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">11,119</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">316</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">6,702</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">3,315</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">21,452</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">moby/moby</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">3,054</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">506</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2,952</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1,160</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">7,672</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">pingcap/tidb</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">11,258</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">11</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">6,956</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">972</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">19,197</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">python/cpython</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">27,473</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">6,944</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">13,204</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">47,621</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">redis/redis</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">189</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">11</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">200</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">90</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">490</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">rust-lang/rust</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">19,965</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">918</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">6,934</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">27,817</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">swiftlang/swift</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">14,161</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">30</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1,469</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">23</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">15,683</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">vuejs/core</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">957</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">9</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">411</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1,377</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">All</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">113,508</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">13,555</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">58,591</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">12,234</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">197,888</td></tr></tbody></table></div>

The projects in `TEST_REPOS` are kept for the end: neither pre-training nor fine-tuning will see a single issue of
theirs. Both pre-training runs read the issues of the training projects, cut into pieces of 128 tokens.

```python
train_text = issues.loc[issues["repo"].isin(TRAIN_REPOS), "text"]
n_sample = min(len(train_text), 400 if SMOKE else 40000)
sample = train_text.sample(n=n_sample, random_state=SEED)
encoded = bert_tokenizer(list(sample), add_special_tokens=False, truncation=True,
                         max_length=2048)["input_ids"]
cls, sep = bert_tokenizer.cls_token_id, bert_tokenizer.sep_token_id
pieces = []
for ids in encoded:
    for start in range(0, max(1, len(ids) - 32), 126):
        piece = [cls] + ids[start:start + 126] + [sep]
        pieces.append(piece + [0] * (128 - len(piece)))
corpus = torch.tensor(pieces)
real_tokens = corpus.ne(0).sum().item()
print(f"{len(corpus):,} pieces of 128 tokens, {real_tokens / 1e6:.1f}M real tokens")
```

<pre class="tfs-printed" markdown="0"><code>221,157 pieces of 128 tokens, 27.0M real tokens</code></pre>

Both encoders are the same small BERT, 4 layers and 256 wide, initialised as BERT was. The masked language model
gets lesson 1's head. ELECTRA's generator is a one-layer encoder that shares the discriminator's word embeddings,
a quarter of its depth; its head predicts words, and the discriminator's head predicts one number per token,
*replaced or not*.

```python
TINY = dict(vocab_size=len(bert_tokenizer), hidden_size=256, num_hidden_layers=4,
            num_attention_heads=4, intermediate_size=1024, max_position_embeddings=512,
            type_vocab_size=2, layer_norm_eps=1e-12, hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1, model_type="bert", pad_token_id=0)


def init_like_bert(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class WordHead(nn.Module):  # lesson 1's masked LM head: a score for every word
    def __init__(self, d, words):
        super().__init__()
        self.dense, self.norm, self.words = nn.Linear(d, d), nn.LayerNorm(d), words
        self.bias = nn.Parameter(torch.zeros(words.num_embeddings))

    def forward(self, h):
        return self.norm(F.gelu(self.dense(h))) @ self.words.weight.T + self.bias


class ReplacedHead(nn.Module):  # ELECTRA's discriminator: one score per token
    def __init__(self, d):
        super().__init__()
        self.dense, self.score = nn.Linear(d, d), nn.Linear(d, 1)

    def forward(self, h):
        return self.score(F.gelu(self.dense(h))).squeeze(-1)


def mask_tokens(ids, generator, rate=0.15):
    special = (ids == cls) | (ids == sep) | (ids == 0)
    chosen = (torch.rand(ids.shape, generator=generator) < rate) & ~special
    roll = torch.rand(ids.shape, generator=generator)
    masked = ids.clone()
    masked[chosen & (roll < 0.8)] = bert_tokenizer.mask_token_id
    swap = chosen & (roll >= 0.8) & (roll < 0.9)
    masked[swap] = torch.randint(len(bert_tokenizer), ids.shape, generator=generator)[swap]
    return masked, chosen
```

One training step of each. The masked language model computes its loss at the chosen positions only. ELECTRA runs the generator on the masked text, samples a word for every chosen position from the generator's own predictions, and asks the discriminator to find the replacements among all real tokens.

The paper weighs the discriminator's loss fifty times more than the generator's, since a binary decision per token gives much smaller losses than a choice among 30,000 words.

```python
def mlm_step(models, ids, generator):
    encoder, head = models
    masked, chosen = mask_tokens(ids, generator)
    logits = head(encoder(masked.to(device), (ids == 0).to(device)))
    return F.cross_entropy(logits[chosen.to(device)].float(), ids[chosen].to(device))


def electra_step(models, ids, generator):
    encoder, head, gen, gen_head = models
    masked, chosen = mask_tokens(ids, generator)
    padding = (ids == 0).to(device)
    ids, masked, chosen = ids.to(device), masked.to(device), chosen.to(device)
    gen_logits = gen_head(gen(masked, padding))[chosen].float()
    gen_loss = F.cross_entropy(gen_logits, ids[chosen])
    with torch.no_grad():
        guesses = torch.distributions.Categorical(logits=gen_logits).sample()
    corrupted = ids.clone()
    corrupted[chosen] = guesses
    replaced = (corrupted != ids).float()  # a lucky guess counts as original
    scores = head(encoder(corrupted, padding)).float()
    disc_loss = F.binary_cross_entropy_with_logits(scores[~padding], replaced[~padding])
    return gen_loss + 50 * disc_loss


def pretrain(task, minutes):
    torch.manual_seed(SEED)
    encoder = Encoder(TINY)
    if task == "mlm":
        models = [encoder, WordHead(256, encoder.embeddings.words)]
    else:
        generator_config = dict(TINY, num_hidden_layers=1)
        gen = Encoder(generator_config)
        gen.embeddings.words = encoder.embeddings.words  # shared word embeddings
        models = [encoder, ReplacedHead(256), gen, WordHead(256, encoder.embeddings.words)]
    for model in models:
        model.apply(init_like_bert)
    models = [m.to(device).train() for m in models]
    params = {id(p): p for m in models for p in m.parameters()}.values()
    optimizer = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.01)
    step_fn = mlm_step if task == "mlm" else electra_step
    generator = torch.Generator().manual_seed(SEED)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")  # float16 gradients in range
    steps, start = 0, time.time()
    while time.time() - start < minutes * 60:
        for group in optimizer.param_groups:
            group["lr"] = 5e-4 * min(1.0, (steps + 1) / 500)  # warm-up, then constant
        batch = corpus[torch.randint(len(corpus), (128,), generator=generator)]
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            loss = step_fn(models, batch, generator)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        steps += 1
    return encoder.eval(), steps


MINUTES = 0.3 if SMOKE else 8
mlm_encoder, mlm_steps = pretrain("mlm", MINUTES)
electra_encoder, electra_steps = pretrain("electra", MINUTES)
print(f"{MINUTES} minutes each: masked LM made {mlm_steps:,} steps, "
      f"ELECTRA {electra_steps:,}")
```

<pre class="tfs-printed" markdown="0"><code>8 minutes each: masked LM made 2,661 steps, ELECTRA 2,143</code></pre>

ELECTRA's step is more expensive, since it runs two encoders, so in the same minutes it makes fewer steps. What each pre-trained encoder learned shows in two ways.

First, without any training of the encoder: its vectors for whole issues, frozen, and a linear classifier on top that tells bugs from feature requests, questions and documentation.

Second, the way encoders are used in practice: fine-tuned, with only a hundred labelled issues of each kind, since pre-training matters most when labels are scarce. A third encoder starts from random weights. All the labelled issues come from the training projects, and the test issues from the five unseen ones.

```python
def balanced(frame, per_kind, seed):
    parts = []
    for kind in KINDS:
        rows = frame[frame["kind"] == kind]
        parts.append(rows.sample(n=min(per_kind, len(rows)), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


seen = issues[issues["repo"].isin(TRAIN_REPOS)]
unseen = issues[issues["repo"].isin(TEST_REPOS)]
probe_train = balanced(seen, 16 if SMOKE else 1000, SEED)
few_train = balanced(seen, 8 if SMOKE else 100, SEED + 2)
small_test = balanced(unseen, 8 if SMOKE else 300, SEED)
print(f"{len(probe_train)} issues for the probe, {len(few_train)} for fine-tuning, "
      f"{len(small_test)} test issues from unseen projects")


def macro_f1(truth, predicted):  # F1 of each kind, averaged: every kind counts the same
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    scores = []
    for c in range(len(KINDS)):
        hits = np.sum((predicted == c) & (truth == c))
        precision = hits / max(1, np.sum(predicted == c))
        recall = hits / max(1, np.sum(truth == c))
        scores.append(0.0 if hits == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def confusion(truth, predicted):  # rows: the actual kind, columns: what the model said
    counts = np.zeros((len(KINDS), len(KINDS)))
    for t, p in zip(truth, predicted, strict=True):
        counts[t, p] += 1
    return counts / np.maximum(1, counts.sum(axis=1, keepdims=True))


def batches(frame, size, shuffle):
    order = np.random.permutation(len(frame)) if shuffle else np.arange(len(frame))
    for k in range(0, len(frame), size):
        rows = frame.iloc[order[k:k + size]]
        enc = bert_tokenizer(list(rows["text"]), truncation=True, max_length=128,
                             padding=True, return_tensors="pt")
        labels = torch.tensor([KINDS.index(x) for x in rows["kind"]])
        yield enc["input_ids"].to(device), labels.to(device)


def pooled(encoder, ids):  # the mean of the last layer over real tokens
    real = (ids != 0).unsqueeze(-1).float()
    return (encoder(ids, ids == 0) * real).sum(1) / real.sum(1)


@torch.no_grad()
def frozen_features(encoder, frame):
    encoder.eval()
    xs, ys = [], []
    for ids, labels in batches(frame, 64, shuffle=False):
        xs.append(pooled(encoder, ids))
        ys.append(labels)
    return torch.cat(xs), torch.cat(ys)


def probe_f1(encoder):
    x, y = frozen_features(encoder, probe_train)
    x_test, y_test = frozen_features(encoder, small_test)
    mean, std = x.mean(0), x.std(0) + 1e-6
    torch.manual_seed(SEED)
    linear = nn.Linear(x.shape[1], len(KINDS)).to(device)
    optimizer = torch.optim.Adam(linear.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(300):  # the whole set at once: it is small
        loss = F.cross_entropy(linear((x - mean) / std), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    predicted = linear((x_test - mean) / std).argmax(-1)
    return macro_f1(y_test.tolist(), predicted.tolist())


def fine_tuned_f1(encoder, epochs=8):
    encoder = encoder.train()
    torch.manual_seed(SEED)
    head = nn.Linear(256, len(KINDS)).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    for _ in range(epochs):
        for ids, labels in batches(few_train, 32, shuffle=True):
            loss = F.cross_entropy(head(pooled(encoder, ids)), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    encoder.eval()
    predicted, truth = [], []
    with torch.no_grad():
        for ids, labels in batches(small_test, 64, shuffle=False):
            predicted += head(pooled(encoder, ids)).argmax(-1).tolist()
            truth += labels.tolist()
    return macro_f1(truth, predicted)


untrained = Encoder(TINY)
untrained.apply(init_like_bert)
untrained = untrained.to(device)
race = []
for name, steps, encoder in [("none, random weights", 0, untrained),
                             ("masked LM", mlm_steps, mlm_encoder),
                             ("ELECTRA", electra_steps, electra_encoder)]:
    probe = probe_f1(encoder)  # before fine-tuning changes the weights
    fine = fine_tuned_f1(encoder)
    race.append({"pre-training": name, "steps": steps, "frozen + linear probe": probe,
                 "fine-tuned on 400 issues": fine})
table(pd.DataFrame(race))
```

<pre class="tfs-printed" markdown="0"><code>4000 issues for the probe, 400 for fine-tuning, 1200 test issues from unseen projects</code></pre>

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">pre-training</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">steps</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">frozen + linear probe</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">fine-tuned on 400 issues</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">none, random weights</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.47</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.30</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">masked LM</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2,661</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.45</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.49</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ELECTRA</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2,143</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.47</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.32</td></tr></tbody></table></div>

The probe column first. Averaged vectors of an untrained encoder are a bag of random word vectors, and a linear
classifier on top already does far better than chance: which words an issue contains says a lot about its kind. Eight
minutes of pre-training didn't make either encoder's frozen vectors better than that.

Fine-tuning on few labels is where pre-training shows. The masked language model starts far ahead of random weights, the effect of lesson 1 at a tiny scale. ELECTRA, in our race, did not: its discriminator fine-tuned no better than random weights.

The paper's smallest ELECTRA trained for four days on a GPU, some seven hundred times our budget, with a generator narrower than its discriminator, while ours is one layer of the same width; its advantage is measured there.

A race this short says what happens in the first eight minutes, and in them the simpler task won. It is RoBERTa's lesson again: an idea is only as good as the recipe it is trained with, and ours is a sketch.

## 3. DeBERTa: positions as vectors of their own

BERT adds a position vector to each word vector at the input, and from then on a token is one vector that mixes what the word is with where it stands.

DeBERTa, from Microsoft, keeps them apart <sup class="cite">[<a href="#ref-4" id="cite-4" data-ref="4" aria-label="Reference 4: He, P., Liu, X., Gao, J., &amp; Chen, W. (2020). DeBERTa: decoding-enhanced BERT with disentangled attention. ICLR 2021.">4</a>]</sup>. Positions are relative, the distance from one token to another, and each layer's attention score between tokens $$i$$ and $$j$$ is the sum of three terms:

$$
A_{ij} = \underbrace{q_i^{c} \cdot k_j^{c}}_{\text{content to content}}
       + \underbrace{q_i^{c} \cdot k_{\delta(i,j)}^{r}}_{\text{content to position}}
       + \underbrace{k_j^{c} \cdot q_{\delta(j,i)}^{r}}_{\text{position to content}},
\qquad \text{scaled by } \frac{1}{\sqrt{3d}} .
$$

Here $$q^c, k^c$$ are the usual queries and keys of the tokens, and $$q^r, k^r$$ are queries and keys computed, with the
same projection matrices, from a table of relative-position embeddings; $$\delta(i, j)$$ is the distance from $$i$$ to
$$j$$, put into a bucket. The second term asks "what does this word look for at that distance", the third "what
does that word look for at this distance". The scale has 3 instead of 1 because three terms are summed.

<img class="tfs-figure tfs-light" src="/assets/posts/arch/02/deberta-light.png" width="760" alt="DeBERTa&#x27;s attention score between two tokens as the sum of three terms: content to content, as in BERT; content to position, the query of the first token against the embedding of their relative distance; and position to content, the key of the second token against the distance seen from the other side" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/arch/02/deberta-dark.png" width="760" alt="DeBERTa&#x27;s attention score between two tokens as the sum of three terms: content to content, as in BERT; content to position, the query of the first token against the embedding of their relative distance; and position to content, the key of the second token against the distance seen from the other side" loading="lazy">

It worked: at the end of 2020 a 1.5-billion-parameter DeBERTa became the first model to score above the human
baseline on SuperGLUE, a benchmark built to be hard for models, 89.9 against 89.8. Microsoft's own announcement
added that the model "is by no means reaching the human-level intelligence of NLU"
<sup class="cite">[<a href="#ref-5" id="cite-5" data-ref="5" aria-label="Reference 5: Microsoft Research, 2021. microsoft.com.">5</a>]</sup>.

DeBERTa-v3 sorts distances into 256 buckets: exact up to 128 tokens in either direction, and logarithmically
coarser beyond, so a model trained on 512-token inputs can still tell near from far in longer ones:

```python
def log_buckets(relative, buckets=256, max_position=512):
    mid = buckets // 2
    sign = torch.sign(relative)
    near = (relative < mid) & (relative > -mid)
    size = torch.where(near, torch.tensor(mid - 1), relative.abs())
    scale = math.log((max_position - 1) / mid)
    far = torch.ceil(torch.log(size / mid) / scale * (mid - 1)) + mid
    return torch.where(size <= mid, relative, (far * sign).long())


distance = torch.arange(-700, 701)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(distance, log_buckets(distance), color=COLORS["att"])
ax.set_title("DeBERTa-v3's bucket for each relative distance")
ax.set_xlabel("distance from the query to the key, tokens")
ax.set_ylabel("bucket")
plt.show()
```

<img class="tfs-output" src="/assets/posts/arch/02/output-02.png" alt="DeBERTa-v3" loading="lazy">

The rest of a DeBERTa-v3 layer is BERT's: post-norm, GELU. One more detail of v3: there are no absolute positions
anywhere, not even at the input.

```python
class DebertaLayer(Layer):
    def forward(self, x, padding, rel, relative, span=256):
        q, k, v = self.split(self.q(x)), self.split(self.k(x)), self.split(self.v(x))
        pq, pk = self.split(self.q(rel)), self.split(self.k(rel))  # same projections
        n, heads = x.shape[1], self.heads
        shape = (x.shape[0], heads, n, n)
        c2p_index = (relative + span).clamp(0, 2 * span - 1).expand(shape)
        p2c_index = (-relative + span).clamp(0, 2 * span - 1).expand(shape)
        content = q @ k.transpose(-2, -1)
        c2p = torch.gather(q @ pk.transpose(-2, -1), -1, c2p_index)
        p2c = torch.gather(k @ pq.transpose(-2, -1), -1, p2c_index).transpose(-1, -2)
        scores = (content + c2p + p2c) / math.sqrt(3 * q.shape[-1])
        lowest = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(padding[:, None, None, :], lowest)
        y = (self.dropout(scores.softmax(dim=-1)) @ v).transpose(1, 2).reshape(x.shape)
        x = self.norm1(x + self.dropout(self.out(y)))
        return self.norm2(x + self.dropout(self.down(F.gelu(self.up(x)))))


class Deberta(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, eps = config["hidden_size"], config["layer_norm_eps"]
        self.span = config["position_buckets"]
        self.words = nn.Embedding(config["vocab_size"], d)
        self.norm = nn.LayerNorm(d, eps=eps)
        self.distances = nn.Embedding(2 * self.span, d)
        self.distance_norm = nn.LayerNorm(d, eps=eps)
        n_layers = config["num_hidden_layers"]
        self.layers = nn.ModuleList(DebertaLayer(config) for _ in range(n_layers))

    def forward(self, ids, padding):
        x = self.norm(self.words(ids)) * (~padding)[..., None]
        positions = torch.arange(ids.shape[1], device=ids.device)
        relative = log_buckets(positions[:, None] - positions[None, :], self.span)
        rel = self.distance_norm(self.distances.weight)[None]
        for layer in self.layers:
            x = layer(x, padding, rel, relative, self.span)
        return x
```

Microsoft published DeBERTa-v3's weights as a PyTorch pickle rather than safetensors; `weights_only=True` loads the
tensors and refuses anything else a pickle could carry.

```python
DEBERTA = "microsoft/deberta-v3-base"
with open(hf_hub_download(DEBERTA, "config.json")) as f:
    deberta_config = json.load(f)
deberta_file = hf_hub_download(DEBERTA, "pytorch_model.bin")
deberta_weights = torch.load(deberta_file, map_location="cpu", weights_only=True)
names = {"words.weight": "deberta.embeddings.word_embeddings.weight",
         "norm.weight": "deberta.embeddings.LayerNorm.weight",
         "norm.bias": "deberta.embeddings.LayerNorm.bias",
         "distances.weight": "deberta.encoder.rel_embeddings.weight",
         "distance_norm.weight": "deberta.encoder.LayerNorm.weight",
         "distance_norm.bias": "deberta.encoder.LayerNorm.bias"}
parts = {"q": "attention.self.query_proj", "k": "attention.self.key_proj",
         "v": "attention.self.value_proj", "out": "attention.output.dense",
         "norm1": "attention.output.LayerNorm", "up": "intermediate.dense",
         "down": "output.dense", "norm2": "output.LayerNorm"}
for i in range(deberta_config["num_hidden_layers"]):
    for ours, theirs in parts.items():
        for p in ("weight", "bias"):
            names[f"layers.{i}.{ours}.{p}"] = f"deberta.encoder.layer.{i}.{theirs}.{p}"
deberta = Deberta(deberta_config)
deberta.load_state_dict({o: deberta_weights[t] for o, t in names.items()}, strict=True)
deberta.eval()

deberta_tokenizer = AutoTokenizer.from_pretrained(DEBERTA)
reference = DebertaV2Model.from_pretrained(DEBERTA).eval()
trace = "Traceback (most recent call last): File 'train.py', line 12 " * 25
check = deberta_tokenizer([trace], return_tensors="pt")
for dtype in (torch.float32, torch.float64):
    with torch.no_grad():
        theirs = reference.to(dtype)(**check).last_hidden_state
        ours = deberta.to(dtype)(check["input_ids"], check["attention_mask"] == 0)
    print(f"{str(dtype)[6:]}: largest difference {(ours - theirs).abs().max().item():.1e}, "
          f"largest activation {theirs.abs().max().item():.1f}")
del reference, deberta, deberta_weights
```

<pre class="tfs-printed" markdown="0"><code>float32: largest difference 1.7e-05, largest activation 9.5
float64: largest difference 3.1e-07, largest activation 9.5</code></pre>

In float32 the two implementations agree to four or five decimal places, the rounding of summing the same numbers
in a different order; in float64, to seven. Our forty lines are DeBERTa's attention.

The explicit `.to(dtype)` in that loop is there for a reason we found the hard way while writing this lesson. Microsoft stored DeBERTa-v3's weights in float16, and recent versions of `transformers` load a model in the precision of its checkpoint.

Compared without the cast, the reference quietly runs in half precision, and the two models disagree in the second decimal place, which looks exactly like a bug in the attention. Running both in float64 is the quickest way to tell a bug from rounding: a bug survives it, rounding doesn't.

## 4. ModernBERT: an encoder rebuilt from LLM parts

For five years after DeBERTa, decoder-only models got all the attention and nearly all the improvements. ModernBERT, from Answer.AI and LightOn, took those improvements back into an encoder <sup class="cite">[<a href="#ref-6" id="cite-6" data-ref="6" aria-label="Reference 6: Warner, B., Chaffin, A., Clavié, B., Weller, O., Hallström, O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Cooper, N., Adams, G., Howard, J., &amp; Poli, I. (2024). Smarter, better, faster, longer: a modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. arXiv:2412.13663.">6</a>]</sup>.

Every piece of its block has appeared in this course before, mostly in lesson 5 of the first one: pre-normalisation, rotary positions (RoPE) instead of learned ones, a gated feed-forward layer (GeGLU, the GELU sibling of Llama's SwiGLU), and no bias terms except in the last layer that scores the vocabulary.

One piece is new: most layers attend only to a window of 128 tokens around each token, and every third layer attends globally. It was trained on 2 trillion tokens of text and code, with 30% of tokens masked rather than 15%, and its authors call it "a major Pareto improvement over older encoders": better and faster at once.

<img class="tfs-figure tfs-light" src="/assets/posts/arch/02/modernbert-light.png" width="760" alt="ModernBERT-base&#x27;s 22 layers: every third layer, starting with the first, attends globally with rotary positions of base 160,000; the others attend to a window of 128 tokens with rotary base 10,000. Each layer is pre-norm, without biases, with a GeGLU feed-forward network" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/arch/02/modernbert-dark.png" width="760" alt="ModernBERT-base&#x27;s 22 layers: every third layer, starting with the first, attends globally with rotary positions of base 160,000; the others attend to a window of 128 tokens with rotary base 10,000. Each layer is pre-norm, without biases, with a GeGLU feed-forward network" loading="lazy">

```python
MODERNBERT = "answerdotai/ModernBERT-base"
with open(hf_hub_download(MODERNBERT, "config.json")) as f:
    modern_config = json.load(f)


def rotate(x, theta):
    d = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    angles = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)[:, None] * inv
    cos, sin = torch.cat([angles.cos()] * 2, -1), torch.cat([angles.sin()] * 2, -1)
    half = d // 2
    return x * cos + torch.cat([-x[..., half:], x[..., :half]], -1) * sin


class ModernLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        d, eps = config["hidden_size"], config["norm_eps"]
        self.heads = config["num_attention_heads"]
        self.is_global = index % config["global_attn_every_n_layers"] == 0
        self.theta = config["global_rope_theta" if self.is_global else "local_rope_theta"]
        self.half_window = config["local_attention"] // 2
        # the first layer reads the embeddings' norm directly, so it has none of its own
        if index > 0:
            self.attn_norm = nn.LayerNorm(d, eps=eps, bias=False)
        else:
            self.attn_norm = nn.Identity()
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.mlp_norm = nn.LayerNorm(d, eps=eps, bias=False)
        self.up = nn.Linear(d, 2 * config["intermediate_size"], bias=False)
        self.down = nn.Linear(config["intermediate_size"], d, bias=False)

    def attention(self, x, padding):
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = rotate(qkv[0], self.theta), rotate(qkv[1], self.theta), qkv[2]
        hidden = padding[:, None, None, :]
        if not self.is_global:
            position = torch.arange(n, device=x.device)
            distance = (position[:, None] - position[None, :]).abs()
            hidden = hidden | (distance > self.half_window)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        weights = scores.masked_fill(hidden, torch.finfo(scores.dtype).min).softmax(-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, n, d))

    def forward(self, x, padding):
        x = x + self.attention(self.attn_norm(x), padding)
        h, gate = self.up(self.mlp_norm(x)).chunk(2, dim=-1)  # GeGLU
        return x + self.down(F.gelu(h) * gate)


class ModernBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, eps = config["hidden_size"], config["norm_eps"]
        self.words = nn.Embedding(config["vocab_size"], d)
        self.norm = nn.LayerNorm(d, eps=eps, bias=False)
        n_layers = config["num_hidden_layers"]
        self.layers = nn.ModuleList(ModernLayer(config, i) for i in range(n_layers))
        self.final_norm = nn.LayerNorm(d, eps=eps, bias=False)

    def forward(self, ids, padding):
        x = self.norm(self.words(ids))
        for layer in self.layers:
            x = layer(x, padding)
        return self.final_norm(x)
```

A padded position far from any real token has every key hidden in a local layer, both by padding and by distance.
A softmax over nothing but $$-\infty$$ is `nan`, and `nan` times a zero weight is still `nan`, which would leak into
the real tokens through the next layer. Filling hidden scores with the most negative finite number instead of
$$-\infty$$, as above, keeps such rows harmless.

```python
weights = load_file(hf_hub_download(MODERNBERT, "model.safetensors"))
names = {"words.weight": "model.embeddings.tok_embeddings.weight",
         "norm.weight": "model.embeddings.norm.weight",
         "final_norm.weight": "model.final_norm.weight"}
for i in range(modern_config["num_hidden_layers"]):
    for ours, theirs in [("qkv", "attn.Wqkv"), ("out", "attn.Wo"), ("up", "mlp.Wi"),
                         ("down", "mlp.Wo"), ("mlp_norm", "mlp_norm")]:
        names[f"layers.{i}.{ours}.weight"] = f"model.layers.{i}.{theirs}.weight"
    if i > 0:
        names[f"layers.{i}.attn_norm.weight"] = f"model.layers.{i}.attn_norm.weight"
modern = ModernBert(modern_config)
modern.load_state_dict({o: weights[t] for o, t in names.items()}, strict=True)
modern = modern.to(device).eval()

modern_tokenizer = AutoTokenizer.from_pretrained(MODERNBERT)
reference = ModernBertModel.from_pretrained(MODERNBERT, attn_implementation="eager")
reference = reference.to(device).eval()
check = modern_tokenizer([trace, "Docs typo"], padding=True, return_tensors="pt").to(device)
padding = check["attention_mask"] == 0
with torch.no_grad():
    theirs = reference(**check).last_hidden_state
    ours = modern(check["input_ids"], padding)
gap = (ours - theirs)[~padding].abs().max().item()
length = check["input_ids"].shape[1]
print(f"{length} tokens, largest difference from Hugging Face's ModernBERT: {gap:.1e}")
assert gap < 1e-3
del reference
```

<pre class="tfs-printed" markdown="0"><code>403 tokens, largest difference from Hugging Face&#x27;s ModernBERT: 1.6e-04</code></pre>

What does the window buy? In a global layer each of $$n$$ tokens attends to all $$n$$; in a local layer to at most 129.
Counted over ModernBERT-base's 22 layers:

```python
rows = []
for n in [512, 2048, 8192]:
    pairs = sum(n * n if layer.is_global else n * min(n, 2 * layer.half_window + 1)
                for layer in modern.layers)
    rows.append({"tokens": n, "attended pairs": f"{pairs / 1e6:,.0f}M",
                 "share of an all-global model": f"{pairs / (22 * n * n):.0%}"})
table(pd.DataFrame(rows))
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">tokens</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">attended pairs</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">share of an all-global model</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">512</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">3M</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">52%</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2,048</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">37M</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">40%</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">8,192</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">552M</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">37%</td></tr></tbody></table></div>

The local layers get almost free as inputs grow, and the cost settles at that of the eight global layers: a little over a third of an all-global model at 8,192 tokens, the length ModernBERT was trained for.

On GPUs with FlashAttention the saving is real time; the T4 has no kernel for windows, so here the mask only hides scores that are still computed.

## 5. Triage for projects the model has never seen

Every large project gets more issues than its maintainers can read the day they arrive, and the first thing anyone does with a new issue is decide what it is: a bug, a feature request, a question, or a problem with the documentation.

A model that does this well saves real time, and the honest test of one is a project it was not trained on, since that is where a new tool would be used.

The four generations get the same job: 6,000 labelled issues from the eight training projects, 1,500 of each kind,
and the issues of the five unseen projects as the test, 400 of each kind. Each model is the pre-trained encoder with
a classification head, fine-tuned for two epochs with the same recipe, reading the first 256 tokens of each issue.

```python
train_set = balanced(seen, 8 if SMOKE else 1500, SEED + 1)
test_set = balanced(unseen, 4 if SMOKE else 400, SEED + 1)
TOKENS, EPOCHS = (64, 1) if SMOKE else (256, 2)  # CI checks the code on a CPU, briefly
lengths = [len(modern_tokenizer(t)["input_ids"]) for t in test_set["text"]]
print(f"test issues longer than 256 tokens: {np.mean(np.array(lengths) > 256):.0%}, "
      f"longer than 1,024: {np.mean(np.array(lengths) > 1024):.0%}")
example = test_set[test_set["kind"] == "bug"].iloc[0]
print(example["repo"], "|", example["kind"], "\n", example["text"][:700])
```

<pre class="tfs-printed" markdown="0"><code>test issues longer than 256 tokens: 56%, longer than 1,024: 21%
moby/moby | bug 
 docker push fails when DNS `AAAA` response with 0 answers beats `A` response with 1 answer

I&#x27;ve been experiencing an issue with docker push for a local, ipv4 only registry. Every once in a while, it&#x27;ll fail, with `dial tcp: lookup registry.weave.local: no such host`, only to succeed after.

I&#x27;ve run wireshark and have confirmed that Docker is issuing an `A` and `AAAA` DNS query simultaneously, and when the `AAAA` response comes back first, I get the error. Most of the time, the `A` response comes back first, and it succeeds.

Also, I&#x27;ve confirmed that AAAA is responding correctly with &quot;No answer&quot;.

![Screen shot from wireshark showing the proper AAAA response](https://files.gitter.im/docker</code></pre>

```python
ENCODERS = {"BERT": "google-bert/bert-base-uncased", "RoBERTa": ROBERTA,
            "DeBERTa-v3": DEBERTA, "ModernBERT": MODERNBERT}


def fine_tune(name, max_length, label, epochs=EPOCHS, lr=3e-5, batch_size=16):
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(ENCODERS[name])
    model = AutoModelForSequenceClassification.from_pretrained(ENCODERS[name],
                                                               num_labels=len(KINDS))
    model = model.float().to(device)  # some checkpoints are stored in float16
    steps = epochs * math.ceil(len(train_set) / batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup = max(1, steps // 10)

    def rate(step):
        return min((step + 1) / warmup, max(0.0, (steps - step) / max(1, steps - warmup)))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    def batches(frame, size, shuffle):
        order = np.random.permutation(len(frame)) if shuffle else np.arange(len(frame))
        for k in range(0, len(frame), size):
            rows = frame.iloc[order[k:k + size]]
            enc = tokenizer(list(rows["text"]), truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt").to(device)
            labels = torch.tensor([KINDS.index(x) for x in rows["kind"]], device=device)
            yield enc, labels

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for enc, labels in batches(train_set, batch_size, shuffle=True):
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                loss = F.cross_entropy(model(**enc).logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
    train_minutes = (time.time() - t0) / 60
    model.eval()
    predicted, truth = [], []
    t0 = time.time()
    half = torch.autocast(device, dtype=torch.float16, enabled=device == "cuda")
    with torch.no_grad(), half:
        for enc, labels in batches(test_set, 32, shuffle=False):
            predicted += model(**enc).logits.argmax(-1).tolist()
            truth += labels.tolist()
    per_second = len(test_set) / (time.time() - t0)
    result = {"model": label, "tokens": max_length,
              "macro F1": macro_f1(truth, predicted),
              "train, min": train_minutes, "issues/s": per_second}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return result, confusion(truth, predicted)


results, confusions = [], {}
for name in ENCODERS:
    result, confusions[name] = fine_tune(name, TOKENS, label=name)
    results.append(result)
    print(f"{name}: macro F1 {result['macro F1']:.3f}", flush=True)
table(pd.DataFrame(results))
```

<pre class="tfs-printed" markdown="0"><code>BERT: macro F1 0.703
RoBERTa: macro F1 0.717
DeBERTa-v3: macro F1 0.728
ModernBERT: macro F1 0.730</code></pre>

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">model</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">tokens</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">macro F1</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">train, min</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">issues/s</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">BERT</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">256</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.70</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2.85</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">223.35</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">RoBERTa</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">256</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.72</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2.92</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">220.92</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">DeBERTa-v3</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">256</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.73</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">5.16</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">119.90</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ModernBERT</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">256</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.73</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">4.93</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">113.90</td></tr></tbody></table></div>

The table needs a second look before any conclusion: each number comes from one run, one random seed, and a test set of 1,600 issues, so differences of a point or two are within the noise.

With that in mind, the order is the order of history. RoBERTa gains on BERT with exactly the same block, from its training alone; DeBERTa-v3 and ModernBERT add a little more and tie with each other.

On this GPU they also cost the same, about half the speed of BERT; ModernBERT's speed comes from FlashAttention and unpadding, which a T4 can't use.

Now the thing only ModernBERT can do here. BERT, RoBERTa and DeBERTa-v3 were trained on 512 tokens at most, while
ModernBERT was trained on 8,192. Issues are long, with logs and stack traces, so we let it read four times more:

```python
LONG = f"ModernBERT, {4 * TOKENS:,} tokens"
long_result, confusions[LONG] = fine_tune("ModernBERT", 4 * TOKENS, LONG, batch_size=8)
results.append(long_result)
display(table(pd.DataFrame(results)))

fig, ax = plt.subplots(figsize=(6.4, 3.8))
for row in results:
    color = COLORS["att"] if row["model"].startswith("ModernBERT") else COLORS["enc"]
    ax.scatter(row["issues/s"], row["macro F1"], s=60, color=color)
    ax.annotate(row["model"], (row["issues/s"], row["macro F1"]), xytext=(6, 4),
                textcoords="offset points", fontsize=9)
ax.set_title("Triage of unseen projects: quality against speed")
ax.set_xlabel("issues classified per second on this GPU")
ax.set_ylabel("macro F1")
plt.show()
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">model</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">tokens</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">macro F1</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">train, min</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">issues/s</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">BERT</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">256</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.70</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2.85</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">223.35</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">RoBERTa</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">256</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.72</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2.92</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">220.92</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">DeBERTa-v3</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">256</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.73</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">5.16</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">119.90</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ModernBERT</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">256</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.73</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">4.93</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">113.90</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">ModernBERT, 1,024 tokens</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1,024</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0.73</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">19.64</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">26.31</td></tr></tbody></table></div>

<img class="tfs-output" src="/assets/posts/arch/02/output-03.png" alt="Triage of unseen projects: quality against speed" loading="lazy">

```python
best = max(results, key=lambda row: row["macro F1"])["model"]
heatmap(confusions[best], rows=KINDS, cols=KINDS, annotate=True, fmt="{:.2f}", cell=0.6,
        title=f"{best}: what each kind of issue was taken for", xlabel="predicted",
        ylabel="actual", role="enc")
plt.show()
```

<img class="tfs-output" src="/assets/posts/arch/02/output-04.png" alt="Output of the code above" loading="lazy">

Reading four times more of each issue changed nothing. ModernBERT trained four times longer and classified four times slower for the same score: the title and the first paragraph decide what an issue is, and the stack trace below them adds nothing to that decision.

Long context earns its price where the answer is spread through a document, like the personal data of lesson 1's bank documents, not here.

The errors that remain gather in one place: almost a third of the bugs were taken for questions. A bug report often
is a question, "is this expected?", and the projects' own labels don't agree on where one ends and the other
begins.

## 6. Which one to use today

Encoders are not a museum piece. On the day this lesson was written, the four most downloaded models on the
Hugging Face Hub were small BERT-style encoders that turn text into vectors for search, the first of them more than
a hundred million times in a month<sup class="cite">[<a href="#ref-7" id="cite-7" data-ref="7" aria-label="Reference 7: Hugging Face, 2026. huggingface.co.">7</a>]</sup>. The next
lesson is about exactly those.

For a new project that needs an encoder, ModernBERT is a good default: it reads long inputs, it was trained on code as well as prose, and it is fast where FlashAttention is available.

DeBERTa-v3 remains a strong choice for short English text when the last point of accuracy matters more than speed, and years of Kaggle competitions have worn smooth every way of fine-tuning it.

RoBERTa and BERT are worth knowing because half the models on the Hub are their descendants, fine-tuned for one task or another, and an existing fine-tuned model often beats a better base trained by you on less data.

The lesson's other result is about method rather than models. RoBERTa won without touching the architecture,
ELECTRA without touching it either, and our tiny race showed that the training task alone moves the result. When a
paper reports a new architecture beating an old one, the first question is whether the old one was trained as well.

ModernBERT's passport, and the family tree with this lesson's branch:

<img class="tfs-figure tfs-light" src="/assets/posts/arch/02/passport-light.png" width="760" alt="ModernBERT&#x27;s passport: published December 2024 by Answer.AI and LightOn; encoder; masked language model with 30% of tokens masked, on 2 trillion tokens of text and code; attention alternating global and local windows of 128 tokens; rotary positions, 8,192 tokens; pre-norm LayerNorm, no biases except the vocabulary decoder; GeGLU; base 149 million parameters and 22 layers, large 395 million and 28 layers" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/arch/02/passport-dark.png" width="760" alt="ModernBERT&#x27;s passport: published December 2024 by Answer.AI and LightOn; encoder; masked language model with 30% of tokens masked, on 2 trillion tokens of text and code; attention alternating global and local windows of 128 tokens; rotary positions, 8,192 tokens; pre-norm LayerNorm, no biases except the vocabulary decoder; GeGLU; base 149 million parameters and 22 layers, large 395 million and 28 layers" loading="lazy">

<img class="tfs-figure tfs-light" src="/assets/posts/arch/02/family-light.png" width="760" alt="The family of the Transformer as this course covers it, with lesson 2, RoBERTa to ModernBERT, highlighted and lesson 1, BERT, done" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/arch/02/family-dark.png" width="760" alt="The family of the Transformer as this course covers it, with lesson 2, RoBERTa to ModernBERT, highlighted and lesson 1, BERT, done" loading="lazy">

## References

1. {:#ref-1} Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). [RoBERTa: a robustly optimized BERT pretraining approach](https://arxiv.org/abs/1907.11692). arXiv:1907.11692. <a class="ref-back" href="#cite-1" aria-label="Back to the text">&#8617;</a>
2. {:#ref-2} Clark, K., Luong, M.-T., Le, Q. V., & Manning, C. D. (2020). [ELECTRA: pre-training text encoders as discriminators rather than generators](https://arxiv.org/abs/2003.10555). *ICLR 2020*. <a class="ref-back" href="#cite-2" aria-label="Back to the text">&#8617;</a>
3. {:#ref-3} open-index (2026). [Open GitHub issues](https://huggingface.co/datasets/open-index/open-github-issues). Hugging Face dataset, ODC-By. <a class="ref-back" href="#cite-3" aria-label="Back to the text">&#8617;</a>
4. {:#ref-4} He, P., Liu, X., Gao, J., & Chen, W. (2020). [DeBERTa: decoding-enhanced BERT with disentangled attention](https://arxiv.org/abs/2006.03654). *ICLR 2021*. <a class="ref-back" href="#cite-4" aria-label="Back to the text">&#8617;</a>
5. {:#ref-5} Microsoft Research, 2021. [microsoft.com](https://www.microsoft.com/en-us/research/blog/microsoft-deberta-surpasses-human-performance-on-the-superglue-benchmark/). <a class="ref-back" href="#cite-5" aria-label="Back to the text">&#8617;</a>
6. {:#ref-6} Warner, B., Chaffin, A., Clavié, B., Weller, O., Hallström, O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Cooper, N., Adams, G., Howard, J., & Poli, I. (2024). [Smarter, better, faster, longer: a modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference](https://arxiv.org/abs/2412.13663). arXiv:2412.13663. <a class="ref-back" href="#cite-6" aria-label="Back to the text">&#8617;</a>
7. {:#ref-7} Hugging Face, 2026. [huggingface.co](https://huggingface.co/models?sort=downloads). <a class="ref-back" href="#cite-7" aria-label="Back to the text">&#8617;</a>
8. {:#ref-8} Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). [BERT: pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805). *NAACL 2019*.
9. {:#ref-9} He, P., Gao, J., & Chen, W. (2021). [DeBERTaV3: improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing](https://arxiv.org/abs/2111.09543). *ICLR 2023*.
10. {:#ref-10} Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). [RoFormer: enhanced transformer with rotary position embedding](https://arxiv.org/abs/2104.09864). arXiv:2104.09864.
11. {:#ref-11} Shazeer, N. (2020). [GLU variants improve Transformer](https://arxiv.org/abs/2002.05202). arXiv:2002.05202.
{: .tfs-refs}





{% endraw %}
