---
layout: post
title: "Your notebook is not a product: a Python package with tests and CI"
date: 2026-10-27 16:00:00
description: "Turn a model from a notebook into a Python package anyone can install: the src layout, pyproject.toml, a command line, four kinds of tests checked by mutation testing, GitHub Actions, and a model card that measures what the model can't do."
tags: [transformers-from-scratch, python-packaging, pytest, github-actions, mlops]
categories: machine-learning
thumbnail: /assets/posts/tfs/07/cover.png
og_image: https://xmarva.github.io/assets/posts/tfs/07/cover.png
stylesheets: [/assets/css/tfs-course.css]
toc:
  beginning: true
---
{% raw %}

<p class="tfs-links"><strong>Transformers: from scratch to production</strong> · lesson 7 of 10 · <a href="/teaching/">all lessons</a> · <a href="https://www.kaggle.com/code/qmarva/transformers-from-scratch-07-package-tests-ci">run on Kaggle</a> · <a href="https://colab.research.google.com/github/xmarva/transformers-from-scratch/blob/main/notebooks/colab/07_package.ipynb">open in Colab</a> · <a href="https://github.com/xmarva/transformers-from-scratch">code on GitHub</a></p>

In 2019 a team at the Fluminense Federal University in Brazil collected over a million Jupyter notebooks from GitHub and ran them. Of 863,878 notebooks they could try, 24% ran to the end without an error, and 4% gave the results their authors had saved<sup class="cite">[<a href="#ref-1" id="cite-1" data-ref="1" aria-label="Reference 1: Pimentel, J. F., Murta, L., Braganholo, V., Freire, J. (2019). A Large-scale Study about Quality and Reproducibility of Jupyter Notebooks. MSR. 1,159,166 notebooks; 24.11% ran without errors, 4.03% reproduced.">1</a>]</sup>.

The rest needed a file that wasn't there, a library version that no longer exists, or cells run in an order nobody wrote down.

Our six lessons are notebooks too. They run, because the course's CI runs them on every change, but try to *use* what they built.

To translate one sentence with lesson 6's model you need the model's class from lessons 2 and 5, the beam search of lesson 3, the right tokenizer, the right config, and the cells that glue them together, in the right order.

To translate a file, you copy all of it into a script. To find out that a library update broke it, you wait for someone to complain.

(The course's `tfs` package, which the lessons install, holds the parts but isn't a product either: it carries every experimental switch and the plotting, and has no loader, no command and no description of what the model can and can't do.)

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/07/hero-light.png" width="760" alt="The package perevod by lines of code: the model and its decoding are less than half of the lines; the rest is the interface people use, the tests, and the packaging and CI around them" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/07/hero-dark.png" width="760" alt="The package perevod by lines of code: the model and its decoding are less than half of the lines; the rest is the interface people use, the tests, and the packaging and CI around them" loading="lazy">

This lesson turns the model into **perevod**, a package anyone can install with one command and call from Python or from a terminal, with tests that run on every change and a model card that says honestly what the model can't do.

The picture is the package by lines of code, counted from the finished files: the model and its decoding are the smaller part. [Sculley et al. (2015)](https://proceedings.neurips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) drew the same thing for whole machine learning systems, with the model as "the small black box in the middle".

**What you'll do**

1. Open a real package and a real model repository and see what makes them installable and findable.
2. Move the model into a package with the src layout, a `pyproject.toml`, configs that refuse bad values, a
   two-line Python API and a command line.
3. Write tests of four kinds, plant ten bugs to see which tests catch which, and fix the test the bugs expose.
4. Set up CI on GitHub Actions and run its wheel check here, step by step.
5. Measure what the model can't do and write it into a model card.

**What you need**: a CPU; this lesson trains nothing beyond a twenty-word toy model for the tests, so Kaggle's free CPU sessions will do and no GPU quota is spent.

It downloads the model of lesson 6 (40 MB) and runs in about half an hour, most of it in section 11, where the test suite runs twenty times over.

A [Hugging Face](https://huggingface.co) token as the secret `HF_TOKEN` is optional: with it, and your user name in the environment variable `TFS_HUB_USER`, the last section publishes your copy of the model and its card.

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
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

if importlib.util.find_spec("sacrebleu") is None:  # Kaggle has it, Colab may not
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sacrebleu"])
import sacrebleu
```

## 1. What an installable package looks like

Start with something that works. In lesson 4 you installed sacreBLEU with `pip install sacrebleu` and then typed `sacrebleu` in a terminal. Both steps hide machinery worth seeing once.

What pip downloads is a **wheel**: a zip archive with a fixed layout, named after the package, its version and the platforms it runs on. Here is sacreBLEU's, straight from PyPI:

```python
SACREBLEU_VERSION = "2.5.1"
PYPI_URL = f"https://pypi.org/pypi/sacrebleu/{SACREBLEU_VERSION}/json"
with urllib.request.urlopen(PYPI_URL) as f:
    release = json.load(f)
for item in release["urls"]:
    if item["filename"].endswith(".whl"):
        wheel_url, wheel_name = item["url"], item["filename"]
wheel_path = WORK_DIR / wheel_name
urllib.request.urlretrieve(wheel_url, wheel_path)

archive = zipfile.ZipFile(wheel_path)
size = wheel_path.stat().st_size / 1000
print(wheel_name, f"{size:.0f} kB,", len(archive.namelist()), "files")
for name in archive.namelist():
    if ".dist-info/" in name:
        print(name)
```

<pre class="tfs-printed" markdown="0"><code>sacrebleu-2.5.1-py3-none-any.whl 104 kB, 41 files
sacrebleu-2.5.1.dist-info/LICENSE.txt
sacrebleu-2.5.1.dist-info/METADATA
sacrebleu-2.5.1.dist-info/WHEEL
sacrebleu-2.5.1.dist-info/entry_points.txt
sacrebleu-2.5.1.dist-info/top_level.txt
sacrebleu-2.5.1.dist-info/RECORD</code></pre>

The name says `py3-none-any`: any Python 3, no compiled code, any platform. Besides the code, the archive holds
one folder of files *about* the code. `METADATA` is what pip reads to install the dependencies:

```python
dist_info = next(name for name in archive.namelist() if ".dist-info/" in name).split("/")[0]
metadata = archive.read(f"{dist_info}/METADATA").decode()
for line in metadata.splitlines():
    if line.startswith(("Name:", "Version:", "Requires-Python:", "Requires-Dist:")):
        print(line)
print()
print(archive.read(f"{dist_info}/entry_points.txt").decode())
```

<pre class="tfs-printed" markdown="0"><code>Name: sacrebleu
Version: 2.5.1
Requires-Python: &gt;=3.8
Requires-Dist: portalocker
Requires-Dist: regex
Requires-Dist: tabulate&gt;=0.8.9
Requires-Dist: numpy&gt;=1.17
Requires-Dist: colorama
Requires-Dist: lxml
Requires-Dist: wheel; extra == &quot;dev&quot;
Requires-Dist: pytest; extra == &quot;dev&quot;
Requires-Dist: mypy; extra == &quot;dev&quot;
Requires-Dist: types-tabulate; extra == &quot;dev&quot;
Requires-Dist: lxml-stubs; extra == &quot;dev&quot;
Requires-Dist: setuptools; extra == &quot;dev&quot;
Requires-Dist: mecab-python3&lt;2.0.0,&gt;=1.0.9; extra == &quot;ja&quot;
Requires-Dist: ipadic&lt;2.0,&gt;=1.0; extra == &quot;ja&quot;
Requires-Dist: mecab-ko&lt;=1.0.1,&gt;=1.0.0; extra == &quot;ko&quot;
Requires-Dist: mecab-ko-dic&lt;2.0,&gt;=1.0; extra == &quot;ko&quot;

[console_scripts]
sacrebleu = sacrebleu.sacrebleu:main</code></pre>

And `entry_points.txt` is where the `sacrebleu` command comes from. Nobody wrote a script called `sacrebleu`:
pip read the line `sacrebleu = sacrebleu.sacrebleu:main` and generated a tiny program that imports that function
and calls it. `RECORD` lists every file with its hash, so an uninstall knows what to remove and a check knows
when a file was changed.

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/07/wheel-light.png" width="760" alt="Inside the sacreBLEU wheel: a zip archive with the package&#x27;s Python files and a dist-info folder (METADATA, entry_points.txt, RECORD, WHEEL and others). pip copies the package into site-packages, installs the dependencies listed in METADATA, and writes a sacrebleu command from entry_points.txt" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/07/wheel-dark.png" width="760" alt="Inside the sacreBLEU wheel: a zip archive with the package&#x27;s Python files and a dist-info folder (METADATA, entry_points.txt, RECORD, WHEEL and others). pip copies the package into site-packages, installs the dependencies listed in METADATA, and writes a sacrebleu command from entry_points.txt" loading="lazy">

A model has the same problem in a different shape: the weights alone are useless without the code that runs
them, the tokenizer, and the settings. On the Hugging Face Hub, the opus-mt model we have been chasing since
lesson 4 keeps all of it in one repository, and so do ours:

```python
def repo_files(repo_id):
    rows = []
    for item in HfApi().model_info(repo_id, files_metadata=True).siblings:
        if not item.rfilename.startswith("."):
            rows.append({"repository": repo_id, "file": item.rfilename,
                         "kB": round(item.size / 1000)})
    return rows


rows = repo_files("Helsinki-NLP/opus-mt-en-ru")
rows += repo_files("xmarva/tfs-transformer-en-ru-tuned")
display(table(pd.DataFrame(rows)))
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">repository</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">file</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:right;">kB</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Helsinki-NLP/opus-mt-en-ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">README.md</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Helsinki-NLP/opus-mt-en-ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">config.json</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Helsinki-NLP/opus-mt-en-ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">generation_config.json</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Helsinki-NLP/opus-mt-en-ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">pytorch_model.bin</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">306,992</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Helsinki-NLP/opus-mt-en-ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">rust_model.ot</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">563,075</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Helsinki-NLP/opus-mt-en-ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">source.spm</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">803</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Helsinki-NLP/opus-mt-en-ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">target.spm</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">1,080</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Helsinki-NLP/opus-mt-en-ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">tf_model.h5</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">307,476</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Helsinki-NLP/opus-mt-en-ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">tokenizer_config.json</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Helsinki-NLP/opus-mt-en-ru</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">vocab.json</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">2,602</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">xmarva/tfs-transformer-en-ru-tuned</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">README.md</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">xmarva/tfs-transformer-en-ru-tuned</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">config.json</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">xmarva/tfs-transformer-en-ru-tuned</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">model.safetensors</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">38,453</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">xmarva/tfs-transformer-en-ru-tuned</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">training.json</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:right;font-variant-numeric:tabular-nums;max-width:460px;white-space:normal;">0</td></tr></tbody></table></div>

Two differences matter for this lesson. The opus-mt repository is self-contained: tokenizer files, the model's config and its generation settings (`generation_config.json`) sit next to the weights.

Ours keeps the tokenizer in another repository, and nothing in it says which. And then there are the two `README.md` files. On the Hub the README is the **model card**, the page people read before trusting a model:

```python
for repo_id in ["Helsinki-NLP/opus-mt-en-ru", "xmarva/tfs-transformer-en-ru-tuned"]:
    card_text = Path(hf_hub_download(repo_id, "README.md")).read_text(encoding="utf-8")
    print(f"--- {repo_id}: {len(card_text.splitlines())} lines")
    print("\n".join(card_text.splitlines()[:14]))
    print()
```

<pre class="tfs-printed" markdown="0"><code>--- Helsinki-NLP/opus-mt-en-ru: 32 lines
---
tags:
- translation
license: apache-2.0
---

### opus-mt-en-ru

* source languages: en
* target languages: ru
*  OPUS readme: [en-ru](https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/models/en-ru/README.md)

*  dataset: opus
* model: transformer-align

--- xmarva/tfs-transformer-en-ru-tuned: 14 lines
---
license: cc-by-2.0
language: [en, ru]
tags: [translation, transformer]
datasets: [xmarva/tfs-tatoeba-en-ru]
---
# A small English→Russian Transformer, tuned

Built in lesson 6 of the course &quot;Transformers: from scratch to production&quot;:
https://github.com/xmarva/transformers-from-scratch

- lesson 5&#x27;s model (pre-norm, RMSNorm, SwiGLU, RoPE), hyperparameters from a search
- the training hyperparameters: training.json
- best validation cross-entropy: 0.992</code></pre>

The block between the `---` lines is YAML for machines: the Hub builds its search filters (language, license, task) from it.

Below it, opus-mt lists its training data and a table of scores, ours a validation loss. Neither says what the model is for or where it fails, and those are the questions a user has. Section 13 answers them for ours, with numbers.

## 2. The plan

Everything the notebooks did by hand gets a permanent place:

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/07/plan-light.png" width="760" alt="Where the parts of the course&#x27;s notebooks go: the model and the decoding become model.py and decoding.py; config dictionaries become config.py; downloading and loading become translator.py; running a cell becomes the perevod command; asserts become tests; running the notebook becomes a CI workflow; and a three-line README becomes a model card" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/07/plan-dark.png" width="760" alt="Where the parts of the course&#x27;s notebooks go: the model and the decoding become model.py and decoding.py; config dictionaries become config.py; downloading and loading become translator.py; running a cell becomes the perevod command; asserts become tests; running the notebook becomes a CI workflow; and a three-line README becomes a model card" loading="lazy">

The package is called **perevod**, Russian for "translation", which also has the merit of being free on PyPI. It lives in a folder of the course's repository, [`perevod/`](https://github.com/xmarva/transformers-from-scratch/tree/main/perevod), and the next two lessons install it from there to serve and deploy it.

This notebook writes every file of it again, so you can read every one of them. We work in a folder laid out like the course repository: the package in `perevod/`, CI in `.github/`.

```python
REPO = WORK_DIR / "course"
shutil.rmtree(REPO, ignore_errors=True)
for folder in ["perevod/src/perevod", "perevod/tests", ".github/workflows"]:
    (REPO / folder).mkdir(parents=True)
os.chdir(REPO)


def write(path, text):
    Path(path).write_text(text.lstrip("\n"), encoding="utf-8")


print("working in", Path.cwd())
```

<pre class="tfs-printed" markdown="0"><code>working in /kaggle/working/tfs/course</code></pre>

Python files are written with `%%writefile`, which saves a cell into a file instead of running it; `-a` appends
to the file, so a long module can arrive in several short cells. The few files that aren't Python go through the
small `write` function above.

## 3. The src layout

The first decision is where the code goes. There are two common layouts. In the **flat layout** the package folder sits at the top of the project, `perevod/perevod/`.

In the **src layout** it sits one level down, `perevod/src/perevod/`. The extra folder looks like bureaucracy, and it is there to stop one specific accident.

Python looks for modules in the folders of `sys.path`, in order, and the current folder comes first. Run the tests with `python -m pytest` from the project folder of a flat layout, and `import perevod` finds the source folder right there, never the copy pip installed.

Your tests pass; the package your users install may be missing a file, a module, or a dependency, and nothing tells you. The [Python Packaging User Guide](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) puts it as "the src layout helps prevent accidental usage of the in-development copy of the code".

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/07/import-path-light.png" width="760" alt="How Python finds import perevod when pytest runs in the project folder. The current folder comes first in the search path. In a flat layout the folder holds perevod/ itself, so the tests import the source and never the installed copy. In the src layout the source sits in src/, not on the path, so the tests import what pip installed" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/07/import-path-dark.png" width="760" alt="How Python finds import perevod when pytest runs in the project folder. The current folder comes first in the search path. In a flat layout the folder holds perevod/ itself, so the tests import the source and never the installed copy. In the src layout the source sits in src/, not on the path, so the tests import what pip installed" loading="lazy">

The current folder is the first thing Python checks in a `python -c` command, as the empty string at the start
of `sys.path`; `python -m`, which this lesson uses to run pytest, puts the current folder first as well:

```python
first = subprocess.run([sys.executable, "-c", "import sys; print(repr(sys.path[0]))"],
                       capture_output=True, text=True).stdout.strip()
print("sys.path[0] when Python runs a command:", first)
```

<pre class="tfs-printed" markdown="0"><code>sys.path[0] when Python runs a command: &#x27;&#x27;</code></pre>

The empty string means "the current folder". With the src layout the price is one habit: install the package
before running its tests, as users will. We'll do exactly that.

## 4. pyproject.toml: what the package is and what it needs

One file describes the package to every tool: pip, the build backend, pytest, ruff. Its format is standardized
([PEP 621](https://peps.python.org/pep-0621/) for the `[project]` table, now kept as the
[pyproject.toml specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/)), so
the same file works whatever builds the package.

```python
write("perevod/pyproject.toml", """
[project]
name = "perevod"
version = "0.1.0"
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
test = ["pytest>=8", "ruff>=0.6", "sacrebleu>=2.4", "pandas>=2.0", "pyarrow>=14"]

[project.scripts]
perevod = "perevod.cli:main"

[project.urls]
Source = "https://github.com/xmarva/transformers-from-scratch/tree/main/perevod"
Model = "https://huggingface.co/xmarva/perevod-en-ru"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
markers = ["slow: downloads the published model and translates real sentences"]

[tool.ruff]
line-length = 92  # what a notebook cell shows: lesson 7 writes these files in its cells
""")
```

Going through it table by table.

`[project]` is the metadata that ends up in the wheel's `METADATA`, the file we just read in sacreBLEU's. `requires-python` is a promise: CI will check the oldest version we claim.

The dependencies have lower bounds and no upper ones. That's deliberate: perevod is a library, other people install it next to their own packages, and a pin like `torch==2.5.1` would make it impossible to install alongside anything that needs another version.

An application (the server of lesson 8) is the opposite case: it pins everything, in a lock file, so that it runs the same tomorrow.

`[project.optional-dependencies]` adds what only developers need, installed with `pip install -e ".[test]"` from the project folder (section 9 explains the `-e`). `[project.scripts]` is our `entry_points.txt`: it asks pip to create a command `perevod` that calls `main` in `perevod/cli.py`, the same mechanism that gave us `sacrebleu`.

`[build-system]` names the tool that turns the folder into a wheel; hatchling finds `src/perevod` without being told. The last two tables configure pytest and ruff, so nobody has to remember their options.

And the README, which PyPI or GitHub show on the package's page:

````python
write("perevod/README.md", """
# perevod

English-to-Russian translation with the small Transformer built in the course
[Transformers: from scratch to production][course]. The model and its card are on the
Hugging Face Hub: [xmarva/perevod-en-ru](https://huggingface.co/xmarva/perevod-en-ru).

```bash
REPO=https://github.com/xmarva/transformers-from-scratch
pip install "perevod @ git+$REPO#subdirectory=perevod"
perevod "I think, therefore I am."
perevod --beam-size 1 < sentences.txt > translations.txt
```

```python
from perevod import Translator

translator = Translator.from_pretrained()
translator(["The doctor will see you now.", "Tom is reading."])
```

The model was trained on short everyday sentences from Tatoeba. Read its model card before
trusting it with anything else.

Tests: `pip install -e ".[test]"`, then `pytest -m "not slow"` (seconds, offline) or
`pytest` (downloads the model).

[course]: https://github.com/xmarva/transformers-from-scratch
""")
````

## 5. The model code, cut down to what we ship

The model is the code of lessons 2 and 5, and the notebooks' version of it has three switches: the 2017 design or pre-norm with LayerNorm or with RMSNorm, ReLU or SwiGLU, sinusoids or RoPE. Those switches were for the experiments of lesson 5.

A package ships one model, and every switch it keeps is a path that users can take and the tests have to cover. So perevod keeps the design the published model uses and nothing else: the module shrinks by almost half, and section 6 shows a bug the switches used to hide.

You have read this code before, so here it is in five cells without commentary beyond the comments. Rotary
positions first, as in lesson 5:

```python
%%writefile perevod/src/perevod/model.py
# The encoder-decoder Transformer of lessons 2 and 5, with only the design the published
# model uses: pre-norm with RMSNorm, SwiGLU, RoPE. The parameter names are the lessons', so
# lesson 6's weights load as they are.

import math

import torch
import torch.nn.functional as F
from torch import nn


def rope_angles(max_len, d_k, base=10_000):
    theta = base ** (-torch.arange(0, d_k, 2) / d_k)
    positions = torch.arange(max_len)
    return positions[:, None] * theta[None, :]


def apply_rope(x, angles):
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.empty_like(x)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated
```

<pre class="tfs-printed" markdown="0"><code>Writing perevod/src/perevod/model.py</code></pre>

```python
%%writefile -a perevod/src/perevod/model.py


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        x32 = x.float()
        mean_square = x32.pow(2).mean(dim=-1, keepdim=True)
        normed = x32 * torch.rsqrt(mean_square + self.eps)
        return (normed * self.weight).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        width = 8 * round(2 * d_ff / 3 / 8)  # three matrices with the weights of two
        self.w_gate = nn.Linear(d_model, width, bias=False)
        self.w_up = nn.Linear(d_model, width, bias=False)
        self.w_down = nn.Linear(width, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/model.py</code></pre>

Attention keeps PyTorch's fused kernel, `scaled_dot_product_attention`, which lesson 5 checked against our own
implementation. And the blocks, now always pre-norm:

```python
%%writefile -a perevod/src/perevod/model.py


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x_query, x_key_value, mask, angles=None):
        q = self.split_heads(self.w_q(x_query))
        k = self.split_heads(self.w_k(x_key_value))
        v = self.split_heads(self.w_v(x_key_value))
        if angles is not None:  # RoPE in self-attention; cross-attention has no positions
            q, k = apply_rope(q, angles), apply_rope(k, angles)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        batch, _, length, _ = out.shape
        return self.w_o(out.transpose(1, 2).reshape(batch, length, -1))
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/model.py</code></pre>

```python
%%writefile -a perevod/src/perevod/model.py


class Block(nn.Module):
    def __init__(self, config, cross):
        super().__init__()
        self.self_attention = Attention(config.d_model, config.n_heads)
        self.cross_attention = Attention(config.d_model, config.n_heads) if cross else None
        self.feed_forward = SwiGLU(config.d_model, config.d_ff)
        self.norm1, self.norm2 = RMSNorm(config.d_model), RMSNorm(config.d_model)
        if cross:
            self.norm3 = RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask, angles, memory=None, memory_mask=None):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attention(h, h, mask, angles))
        if self.cross_attention is None:
            return x + self.dropout(self.feed_forward(self.norm2(x)))
        h = self.norm2(x)
        x = x + self.dropout(self.cross_attention(h, memory, memory_mask))
        return x + self.dropout(self.feed_forward(self.norm3(x)))
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/model.py</code></pre>

```python
%%writefile -a perevod/src/perevod/model.py


class Seq2Seq(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        d_k = config.d_model // config.n_heads
        self.register_buffer("angles", rope_angles(config.max_len, d_k))
        self.encoder = nn.ModuleList([Block(config, False) for _ in range(config.n_layers)])
        self.decoder = nn.ModuleList([Block(config, True) for _ in range(config.n_layers)])
        self.encoder_norm = RMSNorm(config.d_model)
        self.decoder_norm = RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def embed(self, ids):
        return self.dropout(self.embedding(ids) * math.sqrt(self.config.d_model))

    def encode(self, src):
        mask = (src != self.config.pad_id)[:, None, None, :]
        x = self.embed(src)
        for layer in self.encoder:
            x = layer(x, mask, self.angles[: src.shape[1]])
        return self.encoder_norm(x)

    def decode(self, tgt, memory, src):
        length = tgt.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=tgt.device).tril()
        mask = causal[None, None] & (tgt != self.config.pad_id)[:, None, None, :]
        memory_mask = (src != self.config.pad_id)[:, None, None, :]
        x = self.embed(tgt)
        for layer in self.decoder:
            x = layer(x, mask, self.angles[:length], memory, memory_mask)
        return self.decoder_norm(x) @ self.embedding.weight.T

    def forward(self, src, tgt):
        return self.decode(tgt, self.encode(src), src)
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/model.py</code></pre>

The names of the parameters are untouched (`w_q`, `norm1`, `encoder_norm`), because they are the keys of the
weights lesson 6 published; rename one, and `load_state_dict` refuses the file.

Decoding is lesson 3's greedy search and beam search, batched as in lesson 4, with one change: a source longer
than the model's 256 positions is cut to fit. RoPE has angles for 256 positions only, and in the notebooks a
longer sentence crashed deep inside the model. A notebook can crash; a package called from someone's script
should not.

```python
%%writefile perevod/src/perevod/decoding.py
# Greedy decoding and beam search of lesson 3, batched as in lesson 4, moved here from the
# course's notebooks. One change: a source longer than the model's positions is cut to fit,
# instead of failing inside the model.

import torch


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

<pre class="tfs-printed" markdown="0"><code>Writing perevod/src/perevod/decoding.py</code></pre>

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
        memory = model.encode(src)
        written = torch.full((len(indices), 1), bos, device=device)
        done = torch.zeros(len(indices), dtype=torch.bool, device=device)
        for _ in range(max_tokens):
            next_ids = model.decode(written, memory, src)[:, -1].argmax(dim=-1)
            next_ids[done] = pad_id  # a finished sentence only collects padding
            written = torch.cat([written, next_ids[:, None]], dim=1)
            done |= next_ids == eos
            if done.all():
                break
        for row, i in enumerate(indices):
            ids = []
            for token_id in written[row, 1:].tolist():
                if token_id in (eos, pad_id):
                    break
                ids.append(token_id)
            translations[i] = tokenizer.decode(ids)
    return translations
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/decoding.py</code></pre>

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
        memory = model.encode(src)
        # per sentence: a list of (tokens so far, total log-probability)
        beams = [[([bos], 0.0)] for _ in indices]
        finished = [[] for _ in indices]
        active = [True for _ in indices]

        for _ in range(max_tokens):
            rows, owners = [], []
            for k in range(len(indices)):
                if active[k]:
                    for tokens, _ in beams[k]:
                        rows.append(tokens)
                        owners.append(k)
            if not rows:
                break
            owner = torch.tensor(owners, device=device)
            tgt = torch.tensor(rows, device=device)  # all live beams have the same length
            scores = model.decode(tgt, memory[owner], src[owner])[:, -1]
            top = scores.float().log_softmax(dim=-1).topk(beam_size)
            values, ids = top.values.tolist(), top.indices.tolist()

            row = 0
            for k in range(len(indices)):
                if not active[k]:
                    continue
                candidates = []
                for tokens, total in beams[k]:
                    for log_p, token_id in zip(values[row], ids[row], strict=True):
                        candidates.append((tokens + [token_id], total + log_p))
                    row += 1
                candidates.sort(key=lambda candidate: candidate[1], reverse=True)
                beams[k] = []
                for tokens, total in candidates:
                    if tokens[-1] == eos:
                        finished[k].append((tokens, total))
                    else:
                        beams[k].append((tokens, total))
                    if len(beams[k]) == beam_size:
                        break
                if len(finished[k]) >= beam_size:
                    active[k] = False

        for k, i in enumerate(indices):

            def normalized(candidate):
                tokens, total = candidate
                return total / length_penalty(len(tokens) - 1, alpha)  # without <bos>

            best_tokens, _ = max(finished[k] or beams[k], key=normalized)
            translations[i] = tokenizer.decode(best_tokens[1:])  # decode drops <eos>
    return translations
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/decoding.py</code></pre>

## 6. A config that refuses bad values

Lesson 6 saved the model's settings as a plain dictionary in `config.json`, and the notebooks built the model
with `Seq2Seq(**config)`. Here is what happens when someone writes `"RoPE"` where the code expects `"rope"`, with
the course's model class from lesson 5 and lesson 6's weights:

```python
from safetensors.torch import load_file
from tfs import decoding as notebook_decoding
from tfs.modern import Seq2Seq as NotebookSeq2Seq
from tokenizers import Tokenizer

LESSON6 = "xmarva/tfs-transformer-en-ru-tuned"
with open(hf_hub_download(LESSON6, "config.json")) as f:
    lesson6_config = json.load(f)
weights = load_file(hf_hub_download(LESSON6, "model.safetensors"))
TOKENIZER = hf_hub_download("xmarva/tfs-bpe-en-ru-16k", "tokenizer.json")
tokenizer = Tokenizer.from_file(TOKENIZER)

typo_config = dict(lesson6_config, setup={"norm": "rms", "ffn": "swiglu", "pos": "RoPE"})
typo_model = NotebookSeq2Seq(**typo_config)
print(typo_model.load_state_dict(weights))
```

<pre class="tfs-printed" markdown="0"><code>&lt;All keys matched successfully&gt;</code></pre>

All keys matched. The model was built, the weights loaded, nothing complained. Now let it translate, next to the
same weights with the config as it was saved:

```python
notebook_model = NotebookSeq2Seq(**lesson6_config)
notebook_model.load_state_dict(weights)
probe = ["Do you dream in color?", "Mary has two brothers and a sister.",
         "We don't remember days, we remember moments."]
display(table(pd.DataFrame({
    "English": probe,
    '"pos": "rope"': notebook_decoding.beam_search(notebook_model.eval(), tokenizer, probe),
    '"pos": "RoPE"': notebook_decoding.beam_search(typo_model.eval(), tokenizer, probe),
})))
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">English</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">&quot;pos&quot;: &quot;rope&quot;</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">&quot;pos&quot;: &quot;RoPE&quot;</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Do you dream in color?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Ты видишь сны в цвету?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Вы в цветуете? В цветужете во сне? В цветужете во сне?</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Mary has two brothers and a sister.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">У Мэри два брата и сестра.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">У Мэри два брата и две сестры.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">We don&#x27;t remember days, we remember moments.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мы не помним несколько дней, а помним мы помним.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мы не помним дни, как только дни.</td></tr></tbody></table></div>

The notebooks' model checks `pos == "rope"` to decide whether to rotate queries and keys, and `"RoPE"` is not `"rope"`. So it runs with no idea of word order at all.

Some of its sentences fall apart; others are fluent Russian that says something else, like the sister who became two. That's the worst kind of failure: no error, plausible output.

Removing the switches removes this bug, and the config adds a second line of defence. A package checks its config when it reads it, and stops with a message in the terms the user wrote.

A frozen dataclass does that in a few lines: it lists the allowed fields, refuses unknown ones, and checks the values in `__post_init__`, which runs right after the fields are set:

```python
%%writefile perevod/src/perevod/config.py
from dataclasses import asdict, dataclass, fields

DESIGN = {"norm": "rms", "ffn": "swiglu", "pos": "rope"}  # the one design perevod runs


def from_dict(cls, data):
    known = []
    for field in fields(cls):
        known.append(field.name)
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {unknown}, known are {known}")
    return cls(**data)


@dataclass(frozen=True)
class ModelConfig:
    setup: dict
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024
    n_layers: int = 3
    dropout: float = 0.1
    pad_id: int = 0
    max_len: int = 256

    def __post_init__(self):
        problems = []
        if self.setup != DESIGN:
            problems.append(f"setup is {self.setup}, but perevod runs {DESIGN} only")
        if self.d_model % self.n_heads != 0:
            problems.append(f"{self.n_heads} heads don't split d_model={self.d_model}")
        elif self.d_model // self.n_heads % 2 != 0:
            problems.append("RoPE rotates pairs of numbers: d_model / n_heads must be even")
        if not 0 <= self.dropout < 1:
            problems.append(f"dropout={self.dropout} is not a probability below 1")
        if problems:
            raise ValueError("ModelConfig: " + "; ".join(problems))

    @classmethod
    def from_dict(cls, data):
        return from_dict(cls, data)

    def to_dict(self):
        return asdict(self)
```

<pre class="tfs-printed" markdown="0"><code>Writing perevod/src/perevod/config.py</code></pre>

The settings of decoding go into a config of their own, like Hugging Face's `generation_config.json`. They describe
how to *use* the model, not what it is, and they change without retraining anything:

```python
%%writefile -a perevod/src/perevod/config.py
@dataclass(frozen=True)
class GenerationConfig:
    beam_size: int = 4
    alpha: float = 0.6
    max_tokens: int = 128

    def __post_init__(self):
        if self.beam_size < 1:
            raise ValueError(f"GenerationConfig: beam_size={self.beam_size} is below 1")

    @classmethod
    def from_dict(cls, data):
        return from_dict(cls, data)

    def to_dict(self):
        return asdict(self)
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/config.py</code></pre>

`frozen=True` makes a config read-only after creation, so nobody changes `n_heads` on a model that is already
built. The package isn't installed yet, but `config.py` imports nothing of its own, so `runpy` can run the file
directly. The same mistake through the new class, and two more:

```python
import runpy

ModelConfig = runpy.run_path("perevod/src/perevod/config.py")["ModelConfig"]
for bad in [typo_config, dict(lesson6_config, n_head=4), dict(lesson6_config, d_model=250)]:
    try:
        ModelConfig.from_dict(bad)
    except ValueError as error:
        print(error)
```

<pre class="tfs-printed" markdown="0"><code>ModelConfig: setup is {&#x27;norm&#x27;: &#x27;rms&#x27;, &#x27;ffn&#x27;: &#x27;swiglu&#x27;, &#x27;pos&#x27;: &#x27;RoPE&#x27;}, but perevod runs {&#x27;norm&#x27;: &#x27;rms&#x27;, &#x27;ffn&#x27;: &#x27;swiglu&#x27;, &#x27;pos&#x27;: &#x27;rope&#x27;} only
ModelConfig: unknown keys [&#x27;n_head&#x27;], known are [&#x27;setup&#x27;, &#x27;vocab_size&#x27;, &#x27;d_model&#x27;, &#x27;n_heads&#x27;, &#x27;d_ff&#x27;, &#x27;n_layers&#x27;, &#x27;dropout&#x27;, &#x27;pad_id&#x27;, &#x27;max_len&#x27;]
ModelConfig: 4 heads don&#x27;t split d_model=250</code></pre>

A capital letter, a missing letter in a key, a width that doesn't split into heads: three messages that point at
the line to fix.

## 7. The public API: two lines for the user

Everything a user of the package should need is this:

```python
translator = Translator.from_pretrained()
translator("I think, therefore I am.")
```

Behind those two lines are the decisions this section makes. First, where the model comes from. The package
downloads it from a repository on the Hub, and it asks for a specific **revision**, a commit of that repository.
Without a revision you get whatever is on `main` today, and the day someone uploads better weights, every user
of the package silently starts getting different translations. With one, a new model reaches users only with a
new version of the package, tested against it. Hugging Face wants the full 40-character hash for that
<sup class="cite">[<a href="#ref-2" id="cite-2" data-ref="2" aria-label="Reference 2: Hugging Face: model cards and their metadata, downloading at a revision.">2</a>]</sup>.

```python
%%writefile perevod/src/perevod/translator.py
import json
import warnings
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

from . import decoding
from .config import GenerationConfig, ModelConfig
from .model import Seq2Seq

MODEL = "xmarva/perevod-en-ru"
REVISION = "4abc0d63e862ca99d2bd99cd1007e4a9fe47c358"  # tested with this commit
FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json"]


class Translator:
    """Translates English into Russian with the course's Transformer.

    >>> translator = Translator.from_pretrained()
    >>> translator("I think, therefore I am.")
    """

    def __init__(self, model, tokenizer, generation):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.generation = generation
```

<pre class="tfs-printed" markdown="0"><code>Writing perevod/src/perevod/translator.py</code></pre>

The model repository `xmarva/perevod-en-ru` is lesson 6's model packed the way opus-mt packs its own: config,
generation settings, weights and tokenizer in one place, at one commit. Section 14 shows how it was made.

Loading accepts a repository or a local folder. The folder matters more than it seems: it lets tests build a toy
model on disk and load it through exactly the code users run, with no network at all.

```python
%%writefile -a perevod/src/perevod/translator.py

    @classmethod
    def from_pretrained(cls, model=MODEL, revision=None, device=None):
        """Load a model from a Hugging Face repository or from a local directory.

        The default model comes at the revision this version of perevod was tested with.
        """
        if model == MODEL and revision is None:
            revision = REVISION
        path = Path(model)
        if not path.is_dir():
            path = Path(snapshot_download(model, revision=revision, allow_patterns=FILES))
        config = ModelConfig.from_dict(json.loads((path / "config.json").read_text()))
        settings = json.loads((path / "generation_config.json").read_text())
        network = Seq2Seq(config)
        network.load_state_dict(load_file(path / "model.safetensors"))
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        return cls(network.to(device), tokenizer, GenerationConfig.from_dict(settings))
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/translator.py</code></pre>

`model.eval()` turns dropout off. Forgetting it is the most common bug in inference code, and it makes a model
translate the same sentence differently every time; section 11 plants exactly this bug to see who notices.

Now the translation itself. It accepts a string or a list and returns the same, and it has an opinion about one
special input. Watch what the notebooks' decoding does with an empty string, first alone:

```python
try:
    notebook_decoding.beam_search(notebook_model, tokenizer, [""])
except RuntimeError as error:
    print("RuntimeError:", str(error)[:90])
```

<pre class="tfs-printed" markdown="0"><code>RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1] because the unspecified dimensio</code></pre>

A tensor with zero tokens breaks a reshape deep inside attention. And in a batch, next to a real sentence, so
that padding gives it some length:

```python
pair = ["", "Nobody understands me."]
print(notebook_decoding.beam_search(notebook_model, tokenizer, pair))
```

<pre class="tfs-printed" markdown="0"><code>[&#x27;Точно так вышло, что всё равно что есть.&#x27;, &#x27;Никто меня не понимает.&#x27;]</code></pre>

Now it doesn't crash: asked to translate nothing, the model writes a sentence. It has never seen an empty source, and a decoder always produces *something*; machine translation people call this kind of output a hallucination.

So the same input crashes or invents text depending on its neighbours in a batch, which is about the worst contract a function can have. The package answers an empty input with an empty translation without asking the model, and collapses runs of spaces and line breaks, which the model hasn't seen either:

```python
%%writefile -a perevod/src/perevod/translator.py

    def translate(self, texts, beam_size=None):
        """Translate a string or a list of strings; beam_size=1 is greedy decoding."""
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        cleaned = []
        for text in texts:
            cleaned.append(self.clean(text))
        # an empty input has an empty translation: the model never sees it
        to_translate = [text for text in cleaned if text]
        done = iter(self.decode(to_translate, beam_size or self.generation.beam_size))
        translations = []
        for text in cleaned:
            translations.append(next(done) if text else "")
        return translations[0] if single else translations

    __call__ = translate

    def clean(self, text):
        text = " ".join(text.split())  # line breaks and runs of spaces become one space
        limit = self.model.config.max_len
        if len(self.tokenizer.encode(text).ids) > limit:
            warnings.warn(f"input longer than {limit} tokens is cut to fit", stacklevel=3)
        return text

    def decode(self, sentences, beam_size):
        settings = self.generation
        if beam_size == 1:
            return decoding.greedy(self.model, self.tokenizer, sentences,
                                   settings.max_tokens)
        return decoding.beam_search(self.model, self.tokenizer, sentences, beam_size,
                                    settings.alpha, settings.max_tokens)
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/src/perevod/translator.py</code></pre>

A long input gets a warning rather than an error. Cutting a sentence silently would be worse than crashing: the
user would get half a translation and no idea why.

The last Python file of the package is its `__init__.py`. It sets the version and makes `from perevod import
Translator<code markdown="0">&nbsp;work,&nbsp;but&nbsp;without&nbsp;importing&nbsp;PyTorch:&nbsp;the&nbsp;module-level&nbsp;</code>__getattr__`
<sup class="cite">[<a href="#ref-3" id="cite-3" data-ref="3" aria-label="Reference 3: PEP 621 (project metadata), PEP 517 (build backends), PEP 562 (module getattr).">3</a>]</sup> is called only when someone asks for a name the module doesn't
have, so `Translator`, and PyTorch with it, load on first use. Section 9 measures why that's worth the trick.

```python
%%writefile perevod/src/perevod/__init__.py
__version__ = "0.1.0"


def __getattr__(name):
    # `import perevod` stays light, so that --help and --version answer at once;
    # PyTorch is loaded only when someone asks for the translator
    if name == "Translator":
        from .translator import Translator

        return Translator
    raise AttributeError(f"module 'perevod' has no attribute {name!r}")
```

<pre class="tfs-printed" markdown="0"><code>Writing perevod/src/perevod/__init__.py</code></pre>

## 8. A command line

A function is for Python programmers. A command is for everyone else, and for every other program: shell scripts, cron jobs, pipelines. Ours follows the oldest conventions of the terminal.

Each argument is a sentence; with no arguments, it reads lines from standard input and writes one translation per line, so `perevod < in.txt > out.txt` keeps line 17 of the output aligned with line 17 of the input, even when line 17 is empty.

```python
%%writefile perevod/src/perevod/cli.py
import argparse
import sys

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="perevod",
        description="Translate English into Russian, one line per argument, or, "
        "without arguments, one line per line of the standard input.",
    )
    parser.add_argument("text", nargs="*", help="sentences to translate")
    parser.add_argument("--beam-size", type=int, help="1 is greedy decoding")
    parser.add_argument("--model", help="a Hugging Face repository or a local directory")
    parser.add_argument("--revision", help="a branch, tag or commit of the repository")
    parser.add_argument("--device", help="cpu or cuda (default: cuda when there is one)")
    parser.add_argument("--version", action="version", version=f"perevod {__version__}")
    args = parser.parse_args(argv)

    texts = args.text
    if not texts:
        texts = [line.rstrip("\n") for line in sys.stdin]

    from .translator import MODEL, Translator  # PyTorch loads only when there is work to do

    translator = Translator.from_pretrained(args.model or MODEL, args.revision, args.device)
    for translation in translator.translate(texts, beam_size=args.beam_size):
        print(translation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

<pre class="tfs-printed" markdown="0"><code>Writing perevod/src/perevod/cli.py</code></pre>

`argparse` is in the standard library, generates `--help` from the definitions, and checks the types. Notice where the translator is imported: inside `main`, after the arguments are parsed.

Together with the lazy `__init__.py`, that keeps PyTorch out of `perevod --help` and `perevod --version`. The next section measures what that saves.

## 9. Installing it, and the first translation

The files are in place, so the package can be installed like any other. Not in editable mode, though. `pip install -e` installs a pointer to the source folder instead of a copy, so edits take effect without reinstalling; that's the right install while you work on a package, and CI's test job uses it.

But with an editable install, what pip installed *is* the source folder, so a file missing from the wheel goes unnoticed; only a job that installs the built wheel can see that. Here we want the notebook to use what a user would get, a copy.

```python
!{sys.executable} -m pip install -q ./perevod
importlib.invalidate_caches()  # this Python started before perevod existed

import perevod
from perevod import Translator

print("perevod", perevod.__version__, "from", Path(perevod.__file__).parent)
```

<pre class="tfs-printed" markdown="0"><code>  Installing build dependencies ... [?25l[?25hdone
  Getting requirements to build wheel ... [?25l[?25hdone
  Preparing metadata (pyproject.toml) ... [?25l[?25hdone
  Building wheel for perevod (pyproject.toml) ... [?25l[?25hdone
perevod 0.1.0 from /usr/local/lib/python3.13/dist-packages/perevod</code></pre>

The path shows where Python found it: site-packages, not our folder. Now the two lines:

```python
translator = Translator.from_pretrained(device="cpu")
translator("I think, therefore I am.")
```

<pre class="tfs-printed" markdown="0"><code>&#x27;Я думаю, поэтому я есть.&#x27;</code></pre>

Before anything else, one check that matters more than any test we'll write: the package must translate exactly
as the notebooks did. Moving code is a change like any other, and a refactoring that changes the output is a
bug. Lesson 6's own code path, on a few sentences, against the package:

```python
sentences = ["Do you dream in color?", "Nobody understands me.",
             "I've been learning Russian for three years.", "Where can I buy a dictionary?",
             "Tom is reading a book about the brain."]
from_notebooks = notebook_decoding.beam_search(notebook_model, tokenizer, sentences)
from_package = translator(sentences)
display(table(pd.DataFrame({"English": sentences, "perevod": from_package})))
assert from_package == from_notebooks
print("the package translates exactly like the notebook code")
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">English</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">perevod</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Do you dream in color?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Ты видишь сны в цвету?</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Nobody understands me.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Никто меня не понимает.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">I&#x27;ve been learning Russian for three years.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Я уже три года учу русский язык.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Where can I buy a dictionary?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Где я могу купить словарь?</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Tom is reading a book about the brain.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Том читает книгу о мозге.</td></tr></tbody></table></div>

<pre class="tfs-printed" markdown="0"><code>the package translates exactly like the notebook code</code></pre>

Now the command. Each line below is a fresh Python process, the way a terminal runs a command, timed at its
fastest of three runs:

```python
def seconds(command, repeats=3):
    fastest = None
    for _ in range(repeats):
        start = time.perf_counter()
        subprocess.run(command, capture_output=True, check=True)
        took = time.perf_counter() - start
        if fastest is None or took < fastest:
            fastest = took
    return fastest


PEREVOD = shutil.which("perevod")
LOAD = "from perevod import Translator; Translator.from_pretrained(device='cpu')"
startup = {
    "import perevod": seconds([sys.executable, "-c", "import perevod"]),
    "perevod --version": seconds([PEREVOD, "--version"]),
    "import torch": seconds([sys.executable, "-c", "import torch"]),
    "import torch, load the model": seconds([sys.executable, "-c", LOAD]),
    "perevod 'Nobody understands me.'": seconds([PEREVOD, "Nobody understands me."]),
}
for command, took in startup.items():
    print(f"{took:.2f} s  {command}")
```

<pre class="tfs-printed" markdown="0"><code>0.09 s  import perevod
0.09 s  perevod --version
2.24 s  import torch
2.97 s  import torch, load the model
3.01 s  perevod &#x27;Nobody understands me.&#x27;</code></pre>

```python
fig, ax = plt.subplots(figsize=(8, 3))
bars = ax.barh(list(startup), list(startup.values()), color=[
    COLORS["data"], COLORS["data"], COLORS["muted"], COLORS["muted"], COLORS["enc"]])
ax.bar_label(bars, fmt="{:.2f} s", padding=4, fontsize=9, color=COLORS["muted"])
ax.invert_yaxis()
ax.set_xlabel("seconds, fastest of three runs")
ax.set_title("Where a command spends its time")
plt.show()
```

<img class="tfs-output" src="/assets/posts/tfs/07/output-01.png" alt="Where a command spends its time" loading="lazy">

Importing PyTorch is the expensive part on its own. `perevod --version` and `perevod --help` don't need it, so they answer in about the time it takes Python to start.

A translation pays for PyTorch, then for loading the model, and the difference between the last two bars is all the translating itself costs. Every process pays the first two again, which is the argument for the next lesson's server: it loads the model once and keeps it.

The command at work, on a file with an empty line in the middle:

```python
write("sentences.txt", """
Memory is the diary we all carry about with us.

Mary has two brothers and a sister.
""")
!perevod < sentences.txt > translations.txt
!paste -d '|' sentences.txt translations.txt
```

<pre class="tfs-printed" markdown="0"><code>Memory is the diary we all carry about with us.|Матери — это дневник, который мы все берём с собой.
|
Mary has two brothers and a sister.|У Мэри два брата и сестра.</code></pre>

Three lines in, three lines out, the empty one still empty.

## 10. Tests

Every lesson of this course checks itself with `assert`s: attention against PyTorch's, batched decoding against
one sentence at a time. Those checks run once, when someone runs the notebook. A test is the same check, kept in
a file and run on every change, by a program that complains when it fails.

The hard part with a model is deciding what to assert. "The translation is good" can't be a test. What can be
is narrower, and comes in four kinds:

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/07/test-kinds-light.png" width="760" alt="Four kinds of tests for a model package. Tests of the config and glue need no model and run in milliseconds. Invariant tests need only a tiny random model. Golden tests run the published model on a dozen sentences and catch any change in its output. A quality floor scores the model on a test set and catches a model that got worse on average" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/07/test-kinds-dark.png" width="760" alt="Four kinds of tests for a model package. Tests of the config and glue need no model and run in milliseconds. Invariant tests need only a tiny random model. Golden tests run the published model on a dozen sentences and catch any change in its output. A quality floor scores the model on a test set and catches a model that got worse on average" loading="lazy">

The first two kinds need no real model, and that's their strength: they run in milliseconds, offline, on every push.

The trick that makes the second kind possible is a model that is tiny and random: twenty words, sixteen numbers per vector, weights from a fixed seed. It translates nonsense, and that's fine, because what these tests check must hold for *any* weights.

pytest's rules are few. It collects every file named `test_*.py`, runs every function in it whose name starts with `test_`, and reports a failure when an `assert` is false or an exception escapes.

A test asks for what it needs by naming it as an argument, and pytest looks for a **fixture** of that name, a function marked with `@pytest.fixture`, in the test's file or in `conftest.py`, calls it, and passes the result in.

`scope="session"` builds a fixture once for the whole run instead of once per test, and `tmp_path_factory` is one of pytest's own fixtures, which hands out temporary folders. The tiny model, as a fixture:

```python
%%writefile perevod/tests/conftest.py
import json

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from perevod import Translator
from perevod.config import GenerationConfig, ModelConfig
from perevod.model import Seq2Seq

# a model small enough to build in a millisecond: twenty words, random weights. It
# translates nonsense, which is fine: these tests check what must hold for any weights
SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]
WORDS = SPECIAL + ["i", "think", "therefore", "am", ".", ",", "tom", "reads", "a", "book",
                   "?", "я", "думаю", "том", "читает", "книгу"]
```

<pre class="tfs-printed" markdown="0"><code>Writing perevod/tests/conftest.py</code></pre>

```python
%%writefile -a perevod/tests/conftest.py


def tiny_tokenizer():
    vocab = {}
    for token_id, word in enumerate(WORDS):
        vocab[word] = token_id
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.add_special_tokens(SPECIAL)  # so that decoding leaves them out
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/tests/conftest.py</code></pre>

```python
%%writefile -a perevod/tests/conftest.py


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-model")
    tiny_tokenizer().save(str(path / "tokenizer.json"))
    config = ModelConfig(setup={"norm": "rms", "ffn": "swiglu", "pos": "rope"},
                         vocab_size=len(WORDS), d_model=16, n_heads=2, d_ff=32, n_layers=2,
                         dropout=0.1, max_len=32)
    torch.manual_seed(0)
    save_file(Seq2Seq(config).state_dict(), path / "model.safetensors")
    (path / "config.json").write_text(json.dumps(config.to_dict()))
    generation = GenerationConfig(beam_size=3, max_tokens=10)
    (path / "generation_config.json").write_text(json.dumps(generation.to_dict()))
    return path


@pytest.fixture
def translator(tiny_dir):
    return Translator.from_pretrained(str(tiny_dir), device="cpu")
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/tests/conftest.py</code></pre>

The config tests are the simplest: each checks that one kind of bad value is refused, and says which in its
name. A good test name reads as a sentence about the code, because a failing test is first seen as just its
name, in red.

```python
%%writefile perevod/tests/test_config.py
import pytest

from perevod.config import GenerationConfig, ModelConfig

LLAMA_STYLE = {"norm": "rms", "ffn": "swiglu", "pos": "rope"}


def test_a_typo_in_a_key_is_an_error():
    with pytest.raises(ValueError, match="n_head"):
        ModelConfig.from_dict({"setup": LLAMA_STYLE, "vocab_size": 100, "n_head": 4})


def test_another_design_is_an_error():
    with pytest.raises(ValueError, match="setup"):
        ModelConfig(setup={**LLAMA_STYLE, "norm": "layer"}, vocab_size=100)


def test_the_heads_must_split_the_width():
    with pytest.raises(ValueError, match="heads"):
        ModelConfig(setup=LLAMA_STYLE, vocab_size=100, d_model=250, n_heads=4)


def test_a_beam_needs_at_least_one_hypothesis():
    with pytest.raises(ValueError, match="beam_size"):
        GenerationConfig(beam_size=0)


def test_a_config_survives_a_round_trip():
    config = ModelConfig(setup=LLAMA_STYLE, vocab_size=100)
    assert ModelConfig.from_dict(config.to_dict()) == config
```

<pre class="tfs-printed" markdown="0"><code>Writing perevod/tests/test_config.py</code></pre>

The invariants of the model are the properties lesson 2 built it to have. A position must not see the tokens
after it: change the last two tokens of the target, and the scores of the first three positions must not move.
And padding must change nothing: the same sentence with two `<pad>` tokens added must get the same scores.

```python
%%writefile perevod/tests/test_model.py
import torch

from perevod.config import ModelConfig
from perevod.model import Seq2Seq


def tiny_model():
    torch.manual_seed(0)
    config = ModelConfig(setup={"norm": "rms", "ffn": "swiglu", "pos": "rope"},
                         vocab_size=20, d_model=16, n_heads=2, d_ff=32, n_layers=2,
                         max_len=32)
    return Seq2Seq(config).eval()


def test_a_position_does_not_see_the_tokens_after_it():
    model = tiny_model()
    src = torch.tensor([[5, 6, 7, 8]])
    tgt = torch.tensor([[2, 9, 10, 11, 12]])
    changed = tgt.clone()
    changed[0, 3:] = torch.tensor([13, 14])
    before, after = model(src, tgt), model(src, changed)
    assert torch.allclose(before[0, :3], after[0, :3], atol=1e-6)
    assert not torch.allclose(before[0, 3:], after[0, 3:])


def test_padding_the_source_changes_nothing():
    model = tiny_model()
    src = torch.tensor([[5, 6, 7]])
    padded = torch.tensor([[5, 6, 7, 0, 0]])
    tgt = torch.tensor([[2, 9, 10]])
    assert torch.allclose(model(src, tgt), model(padded, tgt), atol=1e-5)
```

<pre class="tfs-printed" markdown="0"><code>Writing perevod/tests/test_model.py</code></pre>

Decoding has invariants of its own. A beam of one hypothesis is greedy decoding by definition.

A batch must translate every sentence exactly as it would alone; lesson 4 checked that once, now it's checked always. And the same input must give the same output, which is the test that notices a model left in training mode.

```python
%%writefile perevod/tests/test_decoding.py
from perevod import decoding

SENTENCES = ["i think therefore i am .", "tom reads a book", "i am ?", "tom"]


def test_a_beam_of_one_is_greedy_decoding(translator):
    model, tokenizer = translator.model, translator.tokenizer
    greedy = decoding.greedy(model, tokenizer, SENTENCES, max_tokens=10)
    beam = decoding.beam_search(model, tokenizer, SENTENCES, beam_size=1, max_tokens=10)
    assert beam == greedy


def test_a_batch_translates_like_one_sentence_at_a_time(translator):
    alone = []
    for sentence in SENTENCES:
        alone.append(translator.translate(sentence))
    assert translator.translate(SENTENCES) == alone


def test_the_same_input_gives_the_same_translation(translator):
    assert translator.translate(SENTENCES) == translator.translate(SENTENCES)
```

<pre class="tfs-printed" markdown="0"><code>Writing perevod/tests/test_decoding.py</code></pre>

The translator's tests are its promises from section 7, one test each:

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
```

<pre class="tfs-printed" markdown="0"><code>Writing perevod/tests/test_translator.py</code></pre>

The command line is tested two ways: in a fresh process, as a user runs it, and by calling `main` directly with
pytest's `capsys`, which captures what it prints, and `monkeypatch`, which replaces the standard input for the
duration of one test. The second test keeps the promise of section 7: importing perevod must not import PyTorch.

```python
%%writefile perevod/tests/test_cli.py
import io
import subprocess
import sys

from perevod import __version__
from perevod.cli import main


def run_python(*args):
    command = [sys.executable, *args]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_the_version():
    assert run_python("-m", "perevod.cli", "--version") == f"perevod {__version__}"


def test_importing_the_package_does_not_load_pytorch():
    assert run_python("-c", "import sys, perevod; print('torch' in sys.modules)") == "False"


def test_each_argument_gives_one_line(tiny_dir, capsys):
    main(["--model", str(tiny_dir), "--device", "cpu", "tom reads", "i think"])
    assert len(capsys.readouterr().out.splitlines()) == 2


def test_one_line_in_one_line_out(tiny_dir, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("tom reads\n\ni think\n"))
    main(["--model", str(tiny_dir), "--device", "cpu"])
    lines = capsys.readouterr().out.splitlines()
    # an empty line stays empty: line n of the output is the translation of line n
    assert len(lines) == 3
    assert lines[1] == ""
```

<pre class="tfs-printed" markdown="0"><code>Writing perevod/tests/test_cli.py</code></pre>

The fast tests, all of them:

```python
!cd perevod && {sys.executable} -m pytest -q -m "not slow" -p no:cacheprovider --color=no
```

<pre class="tfs-printed" markdown="0"><code>..................                                                       [100%]
18 passed in 0.80s</code></pre>

Under a second, and none of them uses the real model. The last two kinds do.

A **golden test** writes down what the published model translates today and fails when that changes. It doesn't claim the translations are right; some of ours are wrong, and the golden file keeps the mistakes too.

What it catches is *change*: a library update, a refactoring, a new tokenizer version. When a change is intended, you update the file in the same commit, and the diff shows every translation that moved. We record it now:

```python
GOLDEN = ["I think, therefore I am.", "Do you dream in color?",
          "Tom is reading a book about the brain.", "My doctor said she was tired.",
          "The researcher said she was exhausted.",
          "Language is a dialect with an army and navy.",
          "I've been learning Russian for three years.",
          "We don't remember days, we remember moments.",
          "Mary has two brothers and a sister.", "Where can I buy a dictionary?",
          "Memory is the diary we all carry about with us.", "Nobody understands me."]
golden = {}
for english, russian in zip(GOLDEN, translator(GOLDEN), strict=True):
    golden[english] = russian
Path("perevod/tests/golden.json").write_text(
    json.dumps(golden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
display(table(pd.DataFrame({"English": list(golden), "perevod": list(golden.values())})))
```

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">English</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">perevod</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">I think, therefore I am.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Я думаю, поэтому я есть.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Do you dream in color?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Ты видишь сны в цвету?</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Tom is reading a book about the brain.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Том читает книгу о мозге.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">My doctor said she was tired.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мой врач сказал, что она устала.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">The researcher said she was exhausted.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Исследование сказала, что она изнурена.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Language is a dialect with an army and navy.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Язык — это диалект с армией и флотом.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">I&#x27;ve been learning Russian for three years.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Я уже три года учу русский язык.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">We don&#x27;t remember days, we remember moments.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мы не помним несколько дней, а помним мы помним.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Mary has two brothers and a sister.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">У Мэри два брата и сестра.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Where can I buy a dictionary?</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Где я могу купить словарь?</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Memory is the diary we all carry about with us.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Матери — это дневник, который мы все берём с собой.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Nobody understands me.</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Никто меня не понимает.</td></tr></tbody></table></div>

Read the right column honestly: memory became "Матери" ("mothers"), and the doctor who said *she* was tired is
masculine in Russian. Section 13 measures both kinds of mistake.

Outside a notebook, the cell that records the file becomes a small script next to the tests, run by hand when a
change is intended. Record it on the kind of machine CI uses: floating-point results differ slightly between
CPUs, GPUs and math libraries, and a beam search near a tie can pick a different word.

A **quality floor** translates 500 sentences of the test set and fails if chrF drops below a line drawn a few points under today's score: not a check of every sentence, but of the average. Both tests are marked `slow`: they download the model, so the fast run skips them and CI runs them on its own schedule.

`pytestmark` marks every test of a file at once, `-m "not slow"` deselects them, and the marker is registered in `pyproject.toml` so that pytest knows it and warns about a misspelt one.

```python
%%writefile perevod/tests/test_quality.py
import json
from pathlib import Path

import pandas as pd
import pytest
import sacrebleu
from huggingface_hub import hf_hub_download

from perevod import Translator

# these tests download the published model: `pytest -m "not slow"` skips them
pytestmark = pytest.mark.slow

GOLDEN = Path(__file__).parent / "golden.json"
DATA = "xmarva/tfs-tatoeba-en-ru"
DATA_REVISION = "540be5b93bf77cd0846b02d55f2558a5596887a9"
CHRF_FLOOR = 57.0  # three points under what the published model scores on this sample


@pytest.fixture(scope="module")
def published():
    return Translator.from_pretrained(device="cpu")


def test_the_published_model_translates_as_it_did(published):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert published.translate(list(golden)) == list(golden.values())


def test_chrf_on_a_sample_of_the_test_set(published):
    path = hf_hub_download(DATA, "data/test.parquet", repo_type="dataset",
                           revision=DATA_REVISION)
    test = pd.read_parquet(path).sample(500, random_state=0)
    chrf = sacrebleu.corpus_chrf(published.translate(list(test.en)), [list(test.ru)]).score
    assert chrf >= CHRF_FLOOR
```

<pre class="tfs-printed" markdown="0"><code>Writing perevod/tests/test_quality.py</code></pre>

```python
!cd perevod && {sys.executable} -m pytest -q -p no:cacheprovider --color=no
```

<pre class="tfs-printed" markdown="0"><code>....................                                                     [100%]
20 passed in 47.99s</code></pre>

## 11. Do the tests catch bugs?

Twenty passing tests prove that the code does what the tests check, and nothing about what they don't. To measure a test suite, break the code on purpose and count how often the tests notice.

That's **mutation testing**, from [DeMillo, Lipton and Sayward (1978)](https://doi.org/10.1109/C-M.1978.218136): each deliberate bug is a mutant, and a test that fails because of it *kills* it. Tools such as mutmut generate mutants by the thousand, flipping operators and constants.

Ours are written by hand, because the bugs that matter in model code, like a forgotten `model.eval()`, aren't operator flips, and each run of our suite takes over a minute.

Here are ten mutants, each a bug that happens in real code: a mask off by one position, a forgotten
`model.eval()`, a missing scale, the length penalty multiplied instead of divided, translations returned in the
order of the sorted batch instead of the order of the input. Each is a replacement in one or two files:

```python
ENCODER_MASK = "        mask = (src != self.config.pad_id)[:, None, None, :]"
MUTANTS = {
    "the decoder sees the next token": [("model.py", ".tril()", ".tril(1)")],
    "the encoder ignores padding": [
        ("model.py", ENCODER_MASK, "        mask = None")],
    "embeddings are not scaled": [("model.py", " * math.sqrt(self.config.d_model)", "")],
    "positions are forgotten": [("model.py", "if angles is not None:", "if False:")],
    "length penalty inverted": [
        ("decoding.py", "total / length_penalty", "total * length_penalty")],
    "dropout left on": [
        ("translator.py", "self.model = model.eval()", "self.model = model"),
        ("decoding.py", "    model.eval()\n", "")],
    "empty input reaches the model": [
        ("translator.py", "[text for text in cleaned if text]", "cleaned"),
        ("translator.py", 'next(done) if text else ""', "next(done)")],
    "no warning for a long input": [("translator.py", "> limit:", "> 10 * limit:")],
    "translations come back in the wrong order": [
        ("decoding.py", "translations[i] = tokenizer.decode(best",
         "translations[k] = tokenizer.decode(best")],
    "input lines keep their newline": [("cli.py", 'line.rstrip("\\n")', "line")],
}
```

For each mutant: a fresh copy of the package, the replacements, and the whole test suite run against the copy.
The copy is used through `PYTHONPATH`, which puts its `src/` first on the search path: exactly the shortcut the
src layout keeps us from taking by accident, taken here on purpose.

```python
def run_tests(project):
    environment = dict(os.environ, PYTHONPATH=str(project / "src"))
    command = [sys.executable, "-m", "pytest", "-q", "-rf", "--tb=no", "--color=no",
               "-p", "no:cacheprovider"]
    result = subprocess.run(command, cwd=project, env=environment, capture_output=True,
                            text=True)
    failed = []
    for line in result.stdout.splitlines():
        if line.startswith("FAILED"):
            failed.append(line.split()[1].split("::")[1])
    return failed


def mutate(name, edits):
    project = WORK_DIR / "mutants" / name.replace(" ", "-")
    shutil.rmtree(project, ignore_errors=True)
    shutil.copytree("perevod", project)
    for module, old, new in edits:
        path = project / "src" / "perevod" / module
        source = path.read_text(encoding="utf-8")
        assert old in source, (name, old)  # the bug must really be planted
        path.write_text(source.replace(old, new), encoding="utf-8")
    return project
```

```python
def run_mutants():
    killed_by = {}
    start = time.perf_counter()
    for name, edits in MUTANTS.items():
        killed_by[name] = run_tests(mutate(name, edits))
        print(f"{name}: {len(killed_by[name])} failing tests")
    print(f"{time.perf_counter() - start:.0f} s for {len(MUTANTS)} mutants")
    return killed_by


first_run = run_mutants()
```

<pre class="tfs-printed" markdown="0"><code>the decoder sees the next token: 2 failing tests
the encoder ignores padding: 2 failing tests
embeddings are not scaled: 2 failing tests
positions are forgotten: 2 failing tests
length penalty inverted: 1 failing tests
dropout left on: 4 failing tests
empty input reaches the model: 0 failing tests
no warning for a long input: 1 failing tests
translations come back in the wrong order: 2 failing tests
input lines keep their newline: 0 failing tests
668 s for 10 mutants</code></pre>

Which test killed which mutant, as a grid:

```python
def show_grid(killed_by, title):
    test_names = []
    for path in sorted(Path("perevod/tests").glob("test_*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                test_names.append(node.name)
    catching = []
    for test in test_names:
        for failed in killed_by.values():
            if test in failed and test not in catching:
                catching.append(test)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for row, mutant in enumerate(MUTANTS):
        for col, test in enumerate(catching):
            colour = COLORS["dec"] if test in killed_by[mutant] else COLORS["grid"]
            ax.add_patch(plt.Rectangle((col, row), 0.92, 0.88, color=colour))
    ax.set_xlim(0, len(catching))
    ax.set_ylim(len(MUTANTS), 0)
    labels = [test.removeprefix("test_").replace("_", " ") for test in catching]
    ax.set_xticks([k + 0.46 for k in range(len(catching))], labels, rotation=60,
                  ha="right", fontsize=8)
    ax.set_yticks([k + 0.44 for k in range(len(MUTANTS))], list(MUTANTS), fontsize=9)
    ax.tick_params(length=0)
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title)
    plt.show()


show_grid(first_run, "Ten planted bugs and the tests that failed because of them")
```

<img class="tfs-output" src="/assets/posts/tfs/07/output-02.png" alt="Output of the code above" loading="lazy">

Read the grid column by column. The invariant tests, with their random twenty-word model, catch the structural bugs: a mask that lets the decoder peek, an encoder that attends to padding, dropout left on. They can't see the rest.

A missing $$\sqrt{d_{\text{model}}}$$ or lost positions don't break any property that holds for random weights; they only break a model that *learned* with them, and only the two tests that run the real model notice.

The inverted length penalty changes which hypothesis the beam search picks, so the golden test fails, but the
average over 500 sentences moves too little to cross the chrF floor. A floor a few points under today's score
catches large drops, not small ones; the golden test catches every change, including the small ones nobody would
see in a score.

The last mutant survives everything, correctly: the translator collapses whitespace, so a line that keeps its
newline is translated exactly like one that doesn't. A surviving mutant isn't always a missing test; sometimes
it's a change that changes nothing, an **equivalent mutant** in the field's terms.

Two other rows should worry us. The empty input that reaches the model survived everything too, although `test_an_empty_input_has_an_empty_translation` was written for exactly that bug.

And translations that come back in the wrong order passed `test_a_batch_translates_like_one_sentence_at_a_time`, which exists to catch them; only the slow tests noticed. Here is the tiny model the fast tests run, built the way `conftest.py` builds it, on the four sentences of the batch test:

```python
import torch
from perevod.config import GenerationConfig, ModelConfig
from perevod.model import Seq2Seq

conftest = runpy.run_path("perevod/tests/conftest.py")
tiny_config = ModelConfig(setup={"norm": "rms", "ffn": "swiglu", "pos": "rope"},
                          vocab_size=len(conftest["WORDS"]), d_model=16, n_heads=2, d_ff=32,
                          n_layers=2, dropout=0.1, max_len=32)
torch.manual_seed(0)
tiny = Translator(Seq2Seq(tiny_config), conftest["tiny_tokenizer"](),
                  GenerationConfig(beam_size=3, max_tokens=10))
print(tiny(["i think therefore i am .", "tom reads a book", "i am ?", "tom"]))
```

<pre class="tfs-printed" markdown="0"><code>[&#x27;&#x27;, &#x27;&#x27;, &#x27;&#x27;, &#x27;&#x27;]</code></pre>

Every sentence becomes an empty string: the random decoder writes only special tokens, which decoding drops. That explains both rows.

An empty input gets an empty translation whether the model sees it or not, and four identical answers can come back in any order. When every answer is the same, these two tests can't tell a right one from a wrong one.

The fix costs about ten seconds per run: let the tiny model learn something. Four sentence pairs made of its twenty
words, learned by heart in two hundred steps of Adam, lesson 3's training loop without the schedule. The second
version of `conftest.py`:

```python
%%writefile perevod/tests/conftest.py
import json

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from perevod import Translator
from perevod.config import GenerationConfig, ModelConfig
from perevod.decoding import pad
from perevod.model import Seq2Seq

# a model small enough to train in a second: twenty words, four sentence pairs learned by
# heart, so that different sentences get different translations
SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]
WORDS = SPECIAL + ["i", "think", "therefore", "am", ".", ",", "tom", "reads", "a", "book",
                   "?", "я", "думаю", "том", "читает", "книгу"]
PAIRS = {"i think therefore i am .": "я думаю .", "tom reads a book": "том читает книгу",
         "i am ?": "я ?", "tom": "том"}
```

<pre class="tfs-printed" markdown="0"><code>Overwriting perevod/tests/conftest.py</code></pre>

```python
%%writefile -a perevod/tests/conftest.py


def tiny_tokenizer():
    vocab = {}
    for token_id, word in enumerate(WORDS):
        vocab[word] = token_id
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.add_special_tokens(SPECIAL)  # so that decoding leaves them out
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


def learn_pairs(model, tokenizer, steps=200):
    bos, eos = tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")
    sources, targets = [], []
    for english, russian in PAIRS.items():
        sources.append(tokenizer.encode(english).ids)
        targets.append([bos] + tokenizer.encode(russian).ids + [eos])
    src, tgt = pad(sources, 0, "cpu"), pad(targets, 0, "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        logits = model(src, tgt[:, :-1])
        loss = F.cross_entropy(logits.transpose(1, 2), tgt[:, 1:], ignore_index=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/tests/conftest.py</code></pre>

```python
%%writefile -a perevod/tests/conftest.py


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-model")
    tokenizer = tiny_tokenizer()
    tokenizer.save(str(path / "tokenizer.json"))
    config = ModelConfig(setup={"norm": "rms", "ffn": "swiglu", "pos": "rope"},
                         vocab_size=len(WORDS), d_model=16, n_heads=2, d_ff=32, n_layers=2,
                         dropout=0.1, max_len=32)
    torch.manual_seed(0)
    model = Seq2Seq(config)
    learn_pairs(model, tokenizer)
    save_file(model.state_dict(), path / "model.safetensors")
    (path / "config.json").write_text(json.dumps(config.to_dict()))
    generation = GenerationConfig(beam_size=3, max_tokens=10)
    (path / "generation_config.json").write_text(json.dumps(generation.to_dict()))
    return path


@pytest.fixture
def pairs():
    return PAIRS


@pytest.fixture
def translator(tiny_dir):
    return Translator.from_pretrained(str(tiny_dir), device="cpu")
```

<pre class="tfs-printed" markdown="0"><code>Appending to perevod/tests/conftest.py</code></pre>

The decoding tests change in two ways. They run on the four learned sentences, and the batch test first checks
that those translate differently, so that it can't pass vacuously again. And there is a new test: the tiny model
must translate what it learned, a golden test in miniature that needs no download.

```python
%%writefile perevod/tests/test_decoding.py
from perevod import decoding


def test_a_beam_of_one_is_greedy_decoding(translator, pairs):
    model, tokenizer = translator.model, translator.tokenizer
    greedy = decoding.greedy(model, tokenizer, list(pairs), max_tokens=10)
    beam = decoding.beam_search(model, tokenizer, list(pairs), beam_size=1, max_tokens=10)
    assert beam == greedy


def test_a_batch_translates_like_one_sentence_at_a_time(translator, pairs):
    alone = []
    for sentence in pairs:
        alone.append(translator.translate(sentence))
    assert len(set(alone)) == len(alone)  # otherwise the order of the results can't be seen
    assert translator.translate(list(pairs)) == alone


def test_the_same_input_gives_the_same_translation(translator, pairs):
    assert translator.translate(list(pairs)) == translator.translate(list(pairs))


def test_the_tiny_model_translates_what_it_learned(translator, pairs):
    assert translator.translate(list(pairs)) == list(pairs.values())
```

<pre class="tfs-printed" markdown="0"><code>Overwriting perevod/tests/test_decoding.py</code></pre>

```python
!cd perevod && {sys.executable} -m pytest -q -m "not slow" -p no:cacheprovider --color=no
```

<pre class="tfs-printed" markdown="0"><code>...................                                                      [100%]
19 passed, 2 deselected in 12.99s</code></pre>

Still a matter of seconds, training included. And the same ten mutants against the improved tests:

```python
second_run = run_mutants()
show_grid(second_run, "The same bugs against the improved tests")
```

<pre class="tfs-printed" markdown="0"><code>the decoder sees the next token: 4 failing tests
the encoder ignores padding: 2 failing tests
embeddings are not scaled: 2 failing tests
positions are forgotten: 3 failing tests
length penalty inverted: 1 failing tests
dropout left on: 3 failing tests
empty input reaches the model: 2 failing tests
no warning for a long input: 1 failing tests
translations come back in the wrong order: 5 failing tests
input lines keep their newline: 0 failing tests
744 s for 10 mutants</code></pre>

<img class="tfs-output" src="/assets/posts/tfs/07/output-03.png" alt="Output of the code above" loading="lazy">

Both worrying rows are fixed. The empty input now fails its own test and the command line's, because a model that has learned something writes a sentence where there should be none. Translations in the wrong order fail three fast tests instead of none.

Lost positions, which needed the real model before, are caught offline now: the tiny model learned its sentences with RoPE and can't reproduce them without it. And the decoder that peeks at the next token, which only the causality test caught offline, now fails three fast tests instead of one.

Two things didn't move, and that's information too. The missing $$\sqrt{d_{\text{model}}}$$ still escapes
everything but the real model: the tiny model learned its four sentences so firmly that rescaled embeddings
don't change them. And the length penalty still needs the golden test, because four short sentences give a
beam search nothing to choose between.

And one row got worse. "Dropout left on" was caught by four tests and now by three: the column of `test_the_same_input_gives_the_same_translation` is gone from the grid, because it no longer kills anything.

A model that has learned four sentences by heart is too sure of itself for dropout to change a word, so the same input does give the same translation, dropout or not.

A test that compares outputs can lose its power when the model under it changes, and nothing announces it. Lesson 8 adds the direct check, that the translator's model is not in training mode.

That's the division of labour the four kinds of tests promised: the fast ones catch what breaks the structure,
the slow ones what breaks the quality. Mutation testing is how you find out which is which, and which test only
looked like it was working.

## 12. Continuous integration

Tests that someone has to remember to run will stop being run. **Continuous integration** (CI) runs them
automatically, on a clean machine, on every change. GitHub Actions is free for public repositories
<sup class="cite">[<a href="#ref-4" id="cite-4" data-ref="4" aria-label="Reference 4: GitHub: Actions billing for public repositories, matrix jobs.">4</a>]</sup>; a workflow is a YAML
file in `.github/workflows/`:

```python
write(".github/workflows/perevod.yml", """
name: perevod

on:
  push:
    branches: [main]
    paths: ["perevod/**", ".github/workflows/perevod.yml"]
  pull_request:
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
      - run: pip install dist/*.whl
      - name: The installed command works outside the source tree
        working-directory: /tmp
        run: perevod --version

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

Three jobs, each answering one question.

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/07/ci-light.png" width="760" alt="The CI workflow of perevod. A push to main, a pull request or the Monday schedule starts it. The test job runs on Python 3.10 and 3.13: install, ruff, the fast tests. The wheel job builds the wheel, installs it and runs the command outside the source folder. The quality job, on main, on Mondays and on pull requests labelled model, downloads the published model and runs the slow tests" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/07/ci-dark.png" width="760" alt="The CI workflow of perevod. A push to main, a pull request or the Monday schedule starts it. The test job runs on Python 3.10 and 3.13: install, ruff, the fast tests. The wheel job builds the wheel, installs it and runs the command outside the source folder. The quality job, on main, on Mondays and on pull requests labelled model, downloads the published model and runs the slow tests" loading="lazy">

`test` asks "does the code work?" on the oldest Python we promise and the newest, through a **matrix**: one job definition, run once per value.

Before the tests it runs ruff, a linter: it reads the code without running it and flags unused imports, undefined names and style problems, and `ruff check .` fails the job on any of them.

The extra index installs PyTorch's CPU build: the default wheel for Linux pulls in the CUDA libraries, about 3 GB of downloads on a machine that has no GPU.

`wheel` asks "does the thing users install work?" It builds the wheel, installs it, and runs the installed command from `/tmp`, far from the source folder, so it catches a wheel that doesn't build or install and a command that isn't there. It catches less than it looks like it does.

Thanks to the light `import perevod` of section 7, `perevod --version` loads only `__init__.py` and `cli.py`: a model file missing from the wheel, or PyTorch missing from the dependencies, would pass. Lesson 8 closes the gap by running the tests against the installed wheel.

`quality` asks "does the published model still translate as it did?" It runs the slow tests: after every merge, every Monday whether anything changed or not, and on pull requests that someone labels `model` (as written, the job sees the label at the next push to the pull request; lesson 8 makes it start the moment the label is added).

The Monday run is for the changes that come from outside: a new release of PyTorch or `tokenizers` arrives without a commit of ours, and the golden test is how we find out it translates differently.

The model downloads are cached between runs, under a key made from `translator.py`, the file that names the model's revision, so a new revision starts a fresh cache.

The `wheel` job, done here, step for step. The wheel first, and what's inside, the same way we opened
sacreBLEU's:

```python
!{sys.executable} -m pip wheel -q --no-deps --wheel-dir dist ./perevod
our_wheel = sorted(Path("dist").glob("perevod-*.whl"))[-1]
print(our_wheel.name)
for name in zipfile.ZipFile(our_wheel).namelist():
    print("  ", name)
```

<pre class="tfs-printed" markdown="0"><code>  Installing build dependencies ... [?25l[?25hdone
  Getting requirements to build wheel ... [?25l[?25hdone
  Preparing metadata (pyproject.toml) ... [?25l[?25hdone
  Building wheel for perevod (pyproject.toml) ... [?25l[?25hdone
perevod-0.1.0-py3-none-any.whl
   perevod/__init__.py
   perevod/cli.py
   perevod/config.py
   perevod/decoding.py
   perevod/model.py
   perevod/translator.py
   perevod-0.1.0.dist-info/METADATA
   perevod-0.1.0.dist-info/WHEEL
   perevod-0.1.0.dist-info/entry_points.txt
   perevod-0.1.0.dist-info/RECORD</code></pre>

The code, the metadata, and in `entry_points.txt` our command. Now the wheel installed into an empty folder,
without the dependencies (the notebook has them already), and perevod imported from `/tmp`:

```python
target = WORK_DIR / "clean-install"
shutil.rmtree(target, ignore_errors=True)
!{sys.executable} -m pip install -q --no-deps --target {target} {our_wheel}
check = subprocess.run([sys.executable, "-c", "import perevod; print(perevod.__file__)"],
                       cwd="/tmp", env=dict(os.environ, PYTHONPATH=str(target)),
                       capture_output=True, text=True, check=True)
print(check.stdout.strip())
```

<pre class="tfs-printed" markdown="0"><code>/kaggle/working/tfs/clean-install/perevod/__init__.py</code></pre>

The package that runs comes from the wheel, not from our folder: whatever the tests passed against, this is what
a user gets. The workflow's real runs are on the course repository's
[Actions page](https://github.com/xmarva/transformers-from-scratch/actions/workflows/perevod.yml); a red cross on
a pull request there means one of these three jobs found something.

## 13. A model card

[Mitchell et al. (2019)](https://doi.org/10.1145/3287560.3287596) proposed that every model ship with a short
document: what it is, what it's for and not for, what data it saw, how it was evaluated, and how its quality
differs across the situations it will meet. They called it a **model card**, and the Hub made it the README of
every model repository.

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/07/card-light.png" width="760" alt="A model card on the Hugging Face Hub has two parts. A YAML header that machines read: language, license, datasets, tags, and model-index with the metrics, which the Hub shows as an evaluation table. And a text for people: what the model is, what it is for and not for, the data, the evaluation and the limitations" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/07/card-dark.png" width="760" alt="A model card on the Hugging Face Hub has two parts. A YAML header that machines read: language, license, datasets, tags, and model-index with the metrics, which the Hub shows as an evaluation table. And a text for people: what the model is, what it is for and not for, the data, the evaluation and the limitations" loading="lazy">

A card is only as good as its numbers, and the interesting numbers are the uncomfortable ones. We measure four
things. First, the score on our own test set, which is the number every card has:

```python
DATA_REVISION = "540be5b93bf77cd0846b02d55f2558a5596887a9"  # the test set of lessons 1-6
test_path = hf_hub_download("xmarva/tfs-tatoeba-en-ru", "data/test.parquet",
                            repo_type="dataset", revision=DATA_REVISION)
tatoeba = pd.read_parquet(test_path).sample(2_000, random_state=SEED)
start = time.perf_counter()
tatoeba["perevod"] = translator(list(tatoeba.en))
tatoeba_seconds = time.perf_counter() - start
chrf = sacrebleu.CHRF()
bleu = sacrebleu.BLEU()
tatoeba_chrf = chrf.corpus_score(list(tatoeba.perevod), [list(tatoeba.ru)])
tatoeba_bleu = bleu.corpus_score(list(tatoeba.perevod), [list(tatoeba.ru)])
print(f"Tatoeba test, {len(tatoeba)} sentences: chrF {tatoeba_chrf.score:.1f}, "
      f"BLEU {tatoeba_bleu.score:.1f}")
print(f"{len(tatoeba) / tatoeba_seconds:.0f} sentences per second on this CPU, beam of 4")
print(chrf.get_signature())
```

<pre class="tfs-printed" markdown="0"><code>Tatoeba test, 2000 sentences: chrF 61.1, BLEU 42.6
14 sentences per second on this CPU, beam of 4
nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0</code></pre>

Second, the score on text the model wasn't trained for. Tatoeba sentences are short and everyday; the news
test set of [WMT20](https://www.statmt.org/wmt20/translation-task.html) is newspaper prose, with long sentences,
names and numbers. sacreBLEU downloads it by name. Two hundred sentences, because each one is long:

```python
from sacrebleu.utils import get_reference_files, get_source_file

source_file = get_source_file("wmt20", "en-ru")
reference_file = get_reference_files("wmt20", "en-ru")[0]
news = pd.DataFrame({
    "en": Path(source_file).read_text(encoding="utf-8").splitlines(),
    "ru": Path(reference_file).read_text(encoding="utf-8").splitlines(),
})
news = news.sample(200, random_state=SEED)
news["perevod"] = translator(list(news.en))
news_chrf = chrf.corpus_score(list(news.perevod), [list(news.ru)])
print(f"WMT20 news, {len(news)} sentences: chrF {news_chrf.score:.1f}")
```

<pre class="tfs-printed" markdown="0"><code>WMT20 news, 200 sentences: chrF 25.8</code></pre>

Third, where inside its own domain the model gets worse. Score the Tatoeba sample by the length of the English
sentence:

```python
tatoeba["words"] = tatoeba.en.str.split().str.len()
tatoeba["length"] = pd.cut(tatoeba.words, [0, 5, 8, 12, 100],
                           labels=["1–5 words", "6–8", "9–12", "13 and more"])
by_length = []
for length, group in tatoeba.groupby("length", observed=True):
    score = chrf.corpus_score(list(group.perevod), [list(group.ru)]).score
    by_length.append({"length": str(length), "sentences": len(group), "chrF": score})
by_length = pd.DataFrame(by_length)

fig, ax = plt.subplots(figsize=(8, 3))
labels = list(by_length.length) + ["WMT20 news"]
scores = list(by_length.chrF) + [news_chrf.score]
bars = ax.bar(labels, scores, color=[COLORS["data"]] * len(by_length) + [COLORS["dec"]])
ax.bar_label(bars, fmt="{:.1f}", padding=3, fontsize=9, color=COLORS["muted"])
ax.set_ylabel("chrF")
ax.set_title("chrF by sentence length on Tatoeba, and on news")
plt.show()
```

<img class="tfs-output" src="/assets/posts/tfs/07/output-04.png" alt="chrF by sentence length on Tatoeba, and on news" loading="lazy">

Up to twelve words the score stays between 61 and 65; at thirteen words and more it drops to about 41, on 81
sentences. That's what the card's "long sentences" rests on.

Fourth, a known weak spot of machine translation. English says "my doctor" without a gender, Russian can't: врач is masculine, and a woman who is a doctor needs the verb to say so, "сказал**а**".

The pronoun in the second half of the sentence is the only clue. [Stanovsky et al. (2019)](https://doi.org/10.18653/v1/P19-1164) built a benchmark of such sentences; Google Translate chose the right Russian gender in 37.7% of them. Our probe is small and blunt: one sentence, twelve professions, both pronouns.

```python
PROFESSIONS = ["doctor", "teacher", "lawyer", "engineer", "nurse", "scientist",
               "programmer", "translator", "neighbor", "boss", "director", "student"]
rows = []
for profession in PROFESSIONS:
    for pronoun in ["she", "he"]:
        sentence = f"My {profession} said {pronoun} was tired."
        translation = translator(sentence)
        rows.append({"profession": profession, "pronoun": pronoun, "perevod": translation})
gender = pd.DataFrame(rows)
gender["feminine verb"] = gender.perevod.str.contains("сказала")
for pronoun in ["she", "he"]:
    share = gender[gender.pronoun == pronoun]["feminine verb"].mean()
    print(f"'{pronoun} was tired': feminine 'сказала' in {share:.0%} of translations")
display(table(gender[gender.pronoun == "she"][["profession", "perevod"]]))
```

<pre class="tfs-printed" markdown="0"><code>&#x27;she was tired&#x27;: feminine &#x27;сказала&#x27; in 8% of translations
&#x27;he was tired&#x27;: feminine &#x27;сказала&#x27; in 17% of translations</code></pre>

<div class="tfs-table-wrap"><table class="tfs-table" style="border-collapse:collapse;font-family:'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;font-size:13.5px;line-height:1.45;margin:4px 0 8px;"><thead><tr><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">profession</th><th style="padding:7px 14px;border-bottom:2px solid #B9C0CB;font-weight:600;color:#5B6575;font-size:12px;letter-spacing:0.02em;text-align:left;">perevod</th></tr></thead><tbody><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">doctor</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мой врач сказал, что она устала.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">teacher</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мой учитель сказал, что она устала.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">lawyer</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мой адвокат сказал, что она устала.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">engineer</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мой инженер сказал, что она устала.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">nurse</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Моя медсестра сказала, что устала.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">scientist</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мой учёный сказал, что она устала.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">programmer</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мой программист сказал, что она устала.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">translator</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мой переводчик сказал, что она устала.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">neighbor</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мой сосед сказал, что она устала.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">boss</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мой начальник сказал, что она устала.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">director</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мой директор сказал, что она устала.</td></tr><tr><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">student</td><td style="padding:7px 14px;border-bottom:1px solid #E6E9EE;vertical-align:top;text-align:left;max-width:460px;white-space:normal;">Мой студент сказал, что она устала.</td></tr></tbody></table></div>

The pronoun doesn't steer the verb: with "he", the verb is feminine in two translations of twelve, more often than with "she". The verb follows the profession instead, masculine for a doctor or an engineer whatever the sentence says, feminine for a nurse, because медсестра is a feminine noun.

The Russian is grammatical, and that's the trouble: "Мой врач сказал, что она устала" means that a male doctor said some other woman was tired, a reading the English hardly suggests.

The likely cause is the data, where male doctors are probably mentioned more often than female ones; Stanovsky et al. found the same pattern in commercial systems.

Now the card. The header is YAML for the Hub; `model-index` holds the scores in a format the Hub turns into a
table on the model's page<sup class="cite">[<a href="#ref-2" id="cite-5" data-ref="2" aria-label="Reference 2: Hugging Face: model cards and their metadata, downloading at a revision.">2</a>]</sup>. The text follows Mitchell et
al.'s sections, with the measurements filled in by the code:

```python
she = gender[gender.pronoun == "she"]["feminine verb"].mean()
n_parameters = 0
for parameter in translator.model.parameters():
    n_parameters += parameter.numel()
length_lines = []
for row in by_length.itertuples():
    length_lines.append(f"| {row.length} | {row.sentences} | {row.chrF:.1f} |")
length_table = "\n".join(length_lines)

CARD = f"""---
language: [en, ru]
license: cc-by-2.0
library_name: perevod
pipeline_tag: translation
tags: [translation, transformer, from-scratch]
datasets: [xmarva/tfs-tatoeba-en-ru]
model-index:
- name: perevod-en-ru
  results:
  - task: {{type: translation, name: English to Russian}}
    dataset: {{type: tatoeba, name: Tatoeba test, sample of {len(tatoeba)} pairs}}
    metrics:
    - {{type: chrf, value: {tatoeba_chrf.score:.1f}, name: chrF}}
    - {{type: bleu, value: {tatoeba_bleu.score:.1f}, name: BLEU}}
  - task: {{type: translation, name: English to Russian}}
    dataset: {{type: wmt20, name: WMT20 news, sample of {len(news)} sentences}}
    metrics:
    - {{type: chrf, value: {news_chrf.score:.1f}, name: chrF}}
---
"""
```

```python
CARD += f"""
# perevod-en-ru: a small English→Russian Transformer

## Model details

An encoder–decoder Transformer with {n_parameters:,} parameters, built from scratch in
the course [Transformers: from scratch to production][course]: lesson 5's design
(pre-norm with RMSNorm, SwiGLU, RoPE) with lesson 6's tuned hyperparameters. Run it with
the package [perevod][package]: `perevod "I think, therefore I am."`

## Intended use

Short, everyday English sentences, and learning how translation models work. It is a course
model: small enough to train on a free GPU and read in full.

## Out of scope

Anything that matters. News, documents, legal or medical text, long sentences: see the
evaluation.

## Training data

About half a million English–Russian sentence pairs from [Tatoeba](https://tatoeba.org)
(dataset `xmarva/tfs-tatoeba-en-ru`): short sentences written by volunteers for language
learners.

## Evaluation

chrF and BLEU with sacreBLEU (`{chrf.get_signature()}`), beam search with 4 hypotheses.

| test set | chrF |
|---|---|
| Tatoeba, {len(tatoeba)} sentences | {tatoeba_chrf.score:.1f} |
| WMT20 news, {len(news)} sentences | {news_chrf.score:.1f} |

By the length of the English sentence, on the Tatoeba sample:

| length | sentences | chrF |
|---|---|---|
{length_table}

## Limitations and biases

The model's quality falls quickly outside short everyday sentences: on news it loses
more than half of its chrF and often writes fluent Russian unrelated to the source.

Gender: in "My doctor said she was tired" and eleven variants with other professions, the
Russian verb is feminine in {she:.0%} of translations; the model follows the profession's
stereotypical gender, not the pronoun.

An empty input makes the model write a sentence; the package returns an empty translation
instead. Inputs longer than 256 tokens are cut, with a warning.

[course]: https://github.com/xmarva/transformers-from-scratch
[package]: https://github.com/xmarva/transformers-from-scratch/tree/main/perevod
"""
Path("perevod-en-ru-README.md").write_text(CARD, encoding="utf-8")
print(CARD)
```

<pre class="tfs-printed" markdown="0"><code>---
language: [en, ru]
license: cc-by-2.0
library_name: perevod
pipeline_tag: translation
tags: [translation, transformer, from-scratch]
datasets: [xmarva/tfs-tatoeba-en-ru]
model-index:
- name: perevod-en-ru
  results:
  - task: {type: translation, name: English to Russian}
    dataset: {type: tatoeba, name: Tatoeba test, sample of 2000 pairs}
    metrics:
    - {type: chrf, value: 61.1, name: chrF}
    - {type: bleu, value: 42.6, name: BLEU}
  - task: {type: translation, name: English to Russian}
    dataset: {type: wmt20, name: WMT20 news, sample of 200 sentences}
    metrics:
    - {type: chrf, value: 25.8, name: chrF}
---

# perevod-en-ru: a small English→Russian Transformer

## Model details

An encoder–decoder Transformer with 9,602,304 parameters, built from scratch in
the course [Transformers: from scratch to production][course]: lesson 5&#x27;s design
(pre-norm with RMSNorm, SwiGLU, RoPE) with lesson 6&#x27;s tuned hyperparameters. Run it with
the package [perevod][package]: `perevod &quot;I think, therefore I am.&quot;`

## Intended use

Short, everyday English sentences, and learning how translation models work. It is a course
model: small enough to train on a free GPU and read in full.

## Out of scope

Anything that matters. News, documents, legal or medical text, long sentences: see the
evaluation.

## Training data

About half a million English–Russian sentence pairs from [Tatoeba](https://tatoeba.org)
(dataset `xmarva/tfs-tatoeba-en-ru`): short sentences written by volunteers for language
learners.

## Evaluation

chrF and BLEU with sacreBLEU (`nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`), beam search with 4 hypotheses.

| test set | chrF |
|---|---|
| Tatoeba, 2000 sentences | 61.1 |
| WMT20 news, 200 sentences | 25.8 |

By the length of the English sentence, on the Tatoeba sample:

| length | sentences | chrF |
|---|---|---|
| 1–5 words | 792 | 63.8 |
| 6–8 | 881 | 65.3 |
| 9–12 | 246 | 61.5 |
| 13 and more | 81 | 40.7 |

## Limitations and biases

The model&#x27;s quality falls quickly outside short everyday sentences: on news it loses
more than half of its chrF and often writes fluent Russian unrelated to the source.

Gender: in &quot;My doctor said she was tired&quot; and eleven variants with other professions, the
Russian verb is feminine in 8% of translations; the model follows the profession&#x27;s
stereotypical gender, not the pronoun.

An empty input makes the model write a sentence; the package returns an empty translation
instead. Inputs longer than 256 tokens are cut, with a warning.

[course]: https://github.com/xmarva/transformers-from-scratch
[package]: https://github.com/xmarva/transformers-from-scratch/tree/main/perevod</code></pre>

Notice what the card does not say: that the model is good. It says where it's usable and where it isn't, with the
measurement next to each claim, so a reader can decide for their own case. That's the idea of Mitchell et al.:
the question isn't whether a model is good, but what it is good for.

The card's license is CC BY 2.0, not the package's MIT: the weights follow the license of the Tatoeba sentences
they were trained on, and the code that runs them has its own. A model and the package around it are licensed
separately.

## 14. Release

A release ties three things to one name: the code, the model, and the card that describes them. The model side
is the repository `xmarva/perevod-en-ru`: lesson 6's weights with everything needed to run them, so that one
revision of one repository is the whole model. With a token, the notebook publishes it to your account:

```python
from perevod.config import GenerationConfig, ModelConfig

release = WORK_DIR / "perevod-en-ru"
release.mkdir(exist_ok=True)
shutil.copy(hf_hub_download(LESSON6, "model.safetensors"), release / "model.safetensors")
shutil.copy(TOKENIZER, release / "tokenizer.json")
release_config = ModelConfig.from_dict(lesson6_config)
(release / "config.json").write_text(json.dumps(release_config.to_dict(), indent=2) + "\n")
(release / "generation_config.json").write_text(
    json.dumps(GenerationConfig().to_dict(), indent=2) + "\n")
(release / "README.md").write_text(CARD, encoding="utf-8")
print(sorted(path.name for path in release.iterdir()))

HUB_USER = os.environ.get("TFS_HUB_USER")
hf_token = env.secret("HF_TOKEN")
if HUB_USER and hf_token:
    api = HfApi(token=hf_token)
    repo_id = f"{HUB_USER}/perevod-en-ru"
    api.create_repo(repo_id, exist_ok=True)
    commit = api.upload_folder(folder_path=str(release), repo_id=repo_id)
    print("published", commit.oid)
else:
    print("not publishing; the course's copy is xmarva/perevod-en-ru")
```

<pre class="tfs-printed" markdown="0"><code>[&#x27;README.md&#x27;, &#x27;config.json&#x27;, &#x27;generation_config.json&#x27;, &#x27;model.safetensors&#x27;, &#x27;tokenizer.json&#x27;]
not publishing; the course&#x27;s copy is xmarva/perevod-en-ru</code></pre>

The package side is a version number. perevod follows [semantic versioning](https://semver.org/spec/v2.0.0.html): MAJOR.MINOR.PATCH, where a bug fix raises the patch, a new feature the minor version, and a change that breaks what users rely on the major one.

For a translator, the question is what "breaks" means, and our answer is opinionated: a new default model is a major release. Nothing in the API changes, but anyone who stored translations, or tuned a pipeline around them, gets different output, and that's what breaking means to them.

<img class="tfs-figure tfs-light" src="/assets/posts/tfs/07/versions-light.png" width="760" alt="Two version numbers move separately. The package version follows semantic versioning: a bug fix is a patch, a new option a minor release, and a new default model, which changes the translations users get, a major one. The model is pinned by the commit of its repository, so a corrected model card changes no one&#x27;s translations" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/tfs/07/versions-dark.png" width="760" alt="Two version numbers move separately. The package version follows semantic versioning: a bug fix is a patch, a new option a minor release, and a new default model, which changes the translations users get, a major one. The model is pinned by the commit of its repository, so a corrected model card changes no one&#x27;s translations" loading="lazy">

That's why the two numbers move separately. The card on the Hub can be corrected tomorrow with a new commit, and nobody's translations change, because perevod 0.1.0 asks for the commit it was tested with.

While the version starts with 0, SemVer allows anything to change at any time; we will still follow the rules above, because users don't read the fine print of version numbers.

In practice a release is a short routine, in this order. Upload the model and note the commit the upload returns.

Put that commit into `REVISION` in `translator.py` (and your repository into `MODEL`, if you publish your own copy), and record the golden file with that model. Raise the version, commit, then tag the commit and push the tag: `git tag perevod-v0.1.0 && git push origin perevod-v0.1.0`.

One weak spot of ours: the version is written in two places, `pyproject.toml` and `__init__.py`, and the version test compares the command with one of them, so nothing checks that the two agree. Lesson 8 moves the version to one place.

perevod isn't on PyPI: the name is free there, but a course package shouldn't take it. Publishing your own takes
two more steps. `python -m build` makes the wheel and the source archive, and `twine upload dist/*`, or PyPI's
trusted publishing from a GitHub workflow, uploads them.

The next lessons install exactly this version from the course repository, by its tag:

```bash
pip install "perevod @ git+https://github.com/xmarva/transformers-from-scratch@perevod-v0.1.0#subdirectory=perevod"
```

Lesson 8 puts it behind an HTTP API and measures how fast it answers when many people ask at once, and the start-up
times of section 9 already tell part of that story.

## References

1. {:#ref-1} Pimentel, J. F., Murta, L., Braganholo, V., Freire, J. (2019). [A Large-scale Study about Quality and Reproducibility of Jupyter Notebooks](https://doi.org/10.1109/MSR.2019.00077). MSR. *1,159,166 notebooks; 24.11% ran without errors, 4.03% reproduced.* <a class="ref-back" href="#cite-1" aria-label="Back to the text">&#8617;</a>
2. {:#ref-2} Hugging Face: [model cards and their metadata](https://huggingface.co/docs/hub/model-cards), [downloading at a revision](https://huggingface.co/docs/huggingface_hub/en/guides/download). <a class="ref-back" href="#cite-2" aria-label="Back to the text">&#8617;</a>
3. {:#ref-3} [PEP 621](https://peps.python.org/pep-0621/) (project metadata), [PEP 517](https://peps.python.org/pep-0517/) (build backends), [PEP 562](https://peps.python.org/pep-0562/) (module `__getattr__`). <a class="ref-back" href="#cite-3" aria-label="Back to the text">&#8617;</a>
4. {:#ref-4} GitHub: [Actions billing for public repositories](https://docs.github.com/en/billing/concepts/product-billing/github-actions), [matrix jobs](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/run-job-variations). <a class="ref-back" href="#cite-4" aria-label="Back to the text">&#8617;</a>
5. {:#ref-5} Sculley, D. et al. (2015). [Hidden Technical Debt in Machine Learning Systems](https://proceedings.neurips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html). NeurIPS.
6. {:#ref-6} Mitchell, M. et al. (2019). [Model Cards for Model Reporting](https://doi.org/10.1145/3287560.3287596). FAT*.
7. {:#ref-7} Stanovsky, G., Smith, N. A., Zettlemoyer, L. (2019). [Evaluating Gender Bias in Machine Translation](https://doi.org/10.18653/v1/P19-1164). ACL.
8. {:#ref-8} DeMillo, R. A., Lipton, R. J., Sayward, F. G. (1978). [Hints on Test Data Selection: Help for the Practicing Programmer](https://doi.org/10.1109/C-M.1978.218136). IEEE Computer. *The idea of mutation testing.*
9. {:#ref-9} Barrault, L. et al. (2020). [Findings of the 2020 Conference on Machine Translation (WMT20)](https://aclanthology.org/2020.wmt-1.1/). *The news test set.*
10. {:#ref-10} Python Packaging User Guide: [src layout vs flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/), [the pyproject.toml specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/), [entry points](https://packaging.python.org/en/latest/specifications/entry-points/).
11. {:#ref-11} [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
12. {:#ref-12} [pytest documentation](https://docs.pytest.org/): fixtures, `capsys`, `monkeypatch`, markers.
{: .tfs-refs}





{% endraw %}
