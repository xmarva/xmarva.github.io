---
layout: post
title: Understanding Byte-Pair Encoding Algorithm
date: 2025-04-01 15:00:00
description: Implement one of the most popular tokenization algorithms and learn how to use ready-made solutions.
tags: tokenization, bpe, algorithms, nlp
categories: featured-posts
featured: false
---

# Tokenization

[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/qmarva/1-bpe-tokenization-algorithm-eng?scriptVersionId=231677033) [![Colab](https://img.shields.io/badge/Colab-F9AB00?style=flat-square&logo=google-colab&logoColor=white)](https://colab.research.google.com/drive/1lmfuMdC8v-lXL_MuyC0uBewdLLCTQzCO?usp=sharing)

Tokenization is a fundamental stage in natural language processing, the task of which is to split text into meaningful units (tokens).

These units can be words, parts of words, or even characters. Historically, simple methods were used: splitting by spaces, regular expressions for extracting words and punctuation, manual rules for handling abbreviations. However, such approaches scaled poorly for languages with agglutinative morphology (e.g., Russian or Finnish) and complex word combinations.

Traditional tokenization methods like space splitting or manual rules often prove ineffective in real-world scenarios: they struggle with typos, rare words, and multilingual texts. For example, words like "gooood" or mixed languages in a single sentence can break a classical tokenizer.

In modern NLP, subword tokenization algorithms like [BPE (Byte Pair Encoding)](https://arxiv.org/pdf/1508.07909) dominate, balancing the semantic integrity of tokens with efficient vocabulary usage. In this notebook, we will examine the BPE algorithm in detail and learn to work with tokenizers from the Hugging Face library.

First, we will import all libraries and functions needed for this notebook.

```python
import re
import spacy
import numpy as np
import pandas as pd
from datasets import load_dataset

from dataclasses import dataclass
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import chain
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
```

## Loading Data

For demonstration, we will load the parallel English-Russian [Tatoeba](https://arxiv.org/abs/1812.10464) corpus from Artetxe et al. (2019) via the [Hugging Face Datasets](http://huggingface.co/docs/datasets/loading) library.

[Tatoeba](https://tatoeba.org/en/sentences/index) is a free collection of translated example sentences for language learners, available in over 400 languages. Its name comes from the Japanese phrase «tatoeba» (例えば), meaning "for example." It is written and maintained by a community of volunteers through open collaboration. Individual contributors are known as Tatoebans.

We will use only the English and Russian subsets. All examples in this dataset are short everyday phrases: "Let's try something." → "Давайте что-нибудь попробуем!".

This format is convenient for training transformers, which work with sequences of limited length. In this notebook, we will not delve into transformer architecture but focus on text data preprocessing.

```python
def load_translation_dataset():
    print("Loading Tatoeba en-ru...")
    try:
        dataset = load_dataset("Helsinki-NLP/tatoeba", lang1="en", lang2="ru", trust_remote_code=True)

    except Exception as e:
        print(f"Error while loading dataset: {e}")
        raise

    print("\nDataset structure:")
    print(dataset)

    print("\nData sample:")
    for i in range(2):
        print(f"EN: {dataset['train'][i]['translation']['en']}")
        print(f"RU: {dataset['train'][i]['translation']['ru']}\n")

    return dataset
    
    dataset = load_translation_dataset()
```

```python
dataset = load_translation_dataset()
```

## Data Analysis

Let's take a quick look at the dataset to understand what we're dealing with. We won't dive deep into data analysis methods but will examine basic statistics.

The `analyze_dataset` function shows that the average length of English sentences is 7.2 words, Russian — 6.2. The maximum lengths (30 and 28 words) indicate the presence of outliers that may require truncation.

The histograms show right-skewed distributions: most sentences are shorter than 15 words. These observations influence model hyperparameter choices, e.g., `max_length=64` provides padding headroom even if actual sequences are shorter.

```python
def analyze_dataset(dataset, n_samples: int = 1000):
    samples = dataset['train'].select(range(n_samples))

    en_lengths = [len(s['translation']['en'].split()) for s in samples]
    ru_lengths = [len(s['translation']['ru'].split()) for s in samples]

    print(f"Analysis based on first {n_samples} samples:")
    print(f"\nEnglish sentences:")
    print(f"Average length: {np.mean(en_lengths):.1f} words")
    print(f"Max length: {max(en_lengths)} words")
    print(f"Min length: {min(en_lengths)} words")

    print(f"\nRussian sentences:")
    print(f"Average length: {np.mean(ru_lengths):.1f} words")
    print(f"Max length: {max(ru_lengths)} words")
    print(f"Min length: {min(ru_lengths)} words")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(en_lengths, bins=30)
    plt.title('English Sentence Lengths')
    plt.xlabel('Words')

    plt.subplot(1, 2, 2)
    sns.histplot(ru_lengths, bins=30)
    plt.title('Russian Sentence Lengths')
    plt.xlabel('Words')

    plt.tight_layout()
    plt.show()

    return max(max(en_lengths), max(ru_lengths))
```

```python
max_sentence_length = analyze_dataset(dataset)
```

## Simple Tokenizer

Now we will write a `BaseTokenizer` class for text preprocessing, building a token vocabulary, and collecting token frequency statistics. This class will serve as the foundation for more complex tokenizers and provide a common structure for processing text data.

We declare the class using the [@dataclass](https://docs.python.org/3/library/dataclasses.html) decorator to auto-generate the constructor. Parameters we need: `language` (text language), `vocab_size` (max vocabulary size), `min_freq` (minimum frequency for including a token in the vocabulary), and `special_tokens` (list of special tokens).

If `special_tokens` are not specified, default values are used: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`.

```python
@dataclass
class BaseTokenizer:
    language: str
    vocab_size: int
    min_freq: int = 2
    special_tokens: List[str] = None
```

The `__post_init__` method is called immediately after object initialization. Here, we initialize the `token2id` and `id2token` dictionaries that map tokens to their numeric IDs. Special tokens must be added to the vocabulary first. For example, `<PAD>` gets ID 0, `<UNK>` — 1, etc.

```python
def __post_init__(self):
    self.special_tokens = self.special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    self.token2id = {token: idx for idx, token in enumerate(self.special_tokens)}
    self.id2token = {idx: token for idx, token in enumerate(self.special_tokens)}
```

The `preprocess_text` method is used to preprocess text. We will convert text to lowercase and split it into tokens using a regular expression.

The pattern `r"\w+[\w']*|['’][a-z]+|[^\w\s]"` captures:
- Words with apostrophes (e.g., `don't` → `["don't"]`).
- Contractions starting with an apostrophe (e.g., `'s` → `["'s"]`).
- Individual punctuation marks (e.g., `"!"` → `["!"]`).

Note that the regex may not cover all edge cases (e.g., emojis or compound symbols), requiring modification for specific tasks.

```python
def preprocess_text(self, text: str) -> List[str]:
    tokens = re.findall(r"\w+[\w']*|['’][a-z]+|[^\w\s]", text.lower())
    return tokens
```

The `get_stats` method collects token frequency statistics. For each text in the `examples` list, the `preprocess_text` function is called, then the `Counter` is updated.

For example, the text `"Hello, world!"` returns a counter with keys `["hello", ",", "world", "!"]` and their frequencies. This method is used during tokenizer training to select tokens for the vocabulary based on `min_freq` and `vocab_size`.

```python
def get_stats(self, examples: List[str]) -> Counter:
    counter = Counter()
    for text in examples:
        tokens = self.preprocess_text(text)
        counter.update(tokens)
    return counter
```

Below, we consolidate all code into a single class:

```python
@dataclass
class BaseTokenizer:
    language: str
    vocab_size: int
    min_freq: int = 2
    special_tokens: List[str] = None

    def __post_init__(self):
        self.special_tokens = self.special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.token2id = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.id2token = {idx: token for idx, token in enumerate(self.special_tokens)}

    def preprocess_text(self, text: str) -> List[str]:
        tokens = re.findall(r"\w+[\w']*|['’][a-z]+|[^\w\s]", text.lower())
        return tokens

    def get_stats(self, examples: List[str]) -> Counter:
        counter = Counter()
        for text in examples:
            tokens = self.preprocess_text(text)
            counter.update(tokens)
        return counter
```

```python
en_tokenizer = BaseTokenizer(language='en', vocab_size=32000)
ru_tokenizer = BaseTokenizer(language='ru', vocab_size=32000)
```

In reality, this basic approach has many drawbacks. For example, converting text to lowercase may lose case information. Additionally, this tokenization ignores word morphology, leading to issues with rare words or homonyms.

Let's write an `analyze_token_statistics` function to count unique tokens and their frequencies.

```python
def analyze_token_statistics(dataset, tokenizer: BaseTokenizer, n_samples: int = 1000):
    samples = dataset['train'].select(range(n_samples))
    texts = [s['translation'][tokenizer.language] for s in samples]

    stats = tokenizer.get_stats(texts)

    print(f"\nToken statistics for {tokenizer.language}:")
    print(f"Total unique tokens: {len(stats)}")
    print("\nTop 10 most frequent tokens:")
    for token, count in stats.most_common(10):
        print(f"{token}: {count}")

    return stats
```

```python
en_stats = analyze_token_statistics(dataset, en_tokenizer)
ru_stats = analyze_token_statistics(dataset, ru_tokenizer)
```

The difference in token counts for English (1337) and Russian (2065) stems from language features: Russian has richer morphology (endings, prefixes) and more word forms. The dominance of punctuation (. and , in the top) suggests the need for their pre-filtering or separate handling.

Interestingly, the `"` token appears more frequently in English (146 times) — likely due to translation specifics in Tatoeba.

Critically, this approach does not split words into subword units, leaving rare words intact and inflating vocabulary size. For comparison, we will explore the BPE tokenizer in subsequent experiments.

## BPE Tokenization Algorithm

Now let's examine how the **BPE (Byte Pair Encoding)** tokenizer works. The core idea is to iteratively merge the most frequent character or token pairs, gradually forming a subword vocabulary. This efficiently handles rare and complex words by splitting them into known components.

### BPETokenizer Class

First, declare the class with the `@dataclass` decorator. Since it inherits from `BaseTokenizer`, it already includes parameters `language`, `vocab_size`, `min_freq`, and `special_tokens`.

```python
@dataclass
class BPETokenizer(BaseTokenizer):
```

### Initialization

The `__post_init__` method is called after object creation. Here we:
1. Call the parent `__post_init__` to initialize base structures like `token2id`.
2. Add a `merges` dictionary to store character pairs and their merged versions (e.g., `('h', 'e')` → `'he'`).
3. Initialize `vocab` with special tokens.

```python
def __post_init__(self):
    super().__post_init__()
    self.merges = {}
    self.vocab = set(self.special_tokens)
```

### Generating Character Pairs

The `get_pairs` method splits a word into consecutive character pairs. For example, the word `['h', 'e', 'l', 'l', 'o']` returns pairs `[('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o')]`. These pairs are analyzed during training to find the most frequent combinations.

```python
def get_pairs(self, word: List[str]) -> List[Tuple[str, str]]:
    return [(word[i], word[i+1]) for i in range(len(word)-1)]
```

---

### Training the Tokenizer

The `train` method is the core of BPE. It has several stages:

**1. Collect Initial Statistics:**
- Split each token into characters and count character sequence frequencies. For example, token `"hello"` becomes `['h', 'e', 'l', 'l', 'o']`, and its frequency increments the counter for `'h e l l o'`.
- Collect all unique characters from the text.

```python
word_freqs = defaultdict(int)
all_chars = set()

for text in texts:
    tokens = self.preprocess_text(text)
    for token in tokens:
        chars = list(token)
        word_freqs[' '.join(chars)] += 1
        all_chars.update(chars)
```

**2. Adding Characters to Vocabulary:**
- Each unique character (e.g., `'h'`, `'e'`) is added to `token2id` and `vocab` if not already present. This ensures even individual characters have IDs.

```python
for char in sorted(all_chars):
    if char not in self.token2id:
        idx = len(self.token2id)
        self.token2id[char] = idx
        self.id2token[idx] = char
        self.vocab.add(char)
```

**3. Main Merge Loop:**
- Each iteration counts the frequency of all possible character pairs in the current word representations. For example, the word `'h e l l o'` has pairs `('h', 'e')`, `('e', 'l')`, etc.
- Select the most frequent pair (e.g., `('l', 'l')` for `hello`) and create a new token `'ll'`.
- Update `merges`, `vocab`, `token2id`, and `id2token`.
- Recalculate word frequencies by replacing the selected pair with the new token. For example, `'h e l l o'` becomes `'h e ll o'` after merging `('l', 'l')`.

```python
for i in tqdm(range(num_merges)):
    pair_freqs = defaultdict(int)
    # Count pair frequencies
    for word, freq in word_freqs.items():
        symbols = word.split()
        pairs = self.get_pairs(symbols)
        for pair in pairs:
            pair_freqs[pair] += freq

    if not pair_freqs:
        break  # No pairs left → stop

    best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
    new_token = ''.join(best_pair)
    self.merges[best_pair] = new_token
    self.vocab.add(new_token)

    # Update vocabulary
    if new_token not in self.token2id:
        idx = len(self.token2id)
        self.token2id[new_token] = idx
        self.id2token[idx] = new_token

    # Recalculate frequencies with new token
    new_word_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        symbols = word.split()
        i = 0
        new_symbols = []
        while i < len(symbols):
            if i < len(symbols)-1 and (symbols[i], symbols[i+1]) == best_pair:
                new_symbols.append(new_token)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_word = ' '.join(new_symbols)
        new_word_freqs[new_word] += freq

    word_freqs = new_word_freqs
```

If `num_merges` is too high and pairs are exhausted early, training stops. Progress is printed every 1000 iterations to track vocabulary growth.

---

### Text Tokenization

The `tokenize` method converts text to token IDs:
1. Text is split into tokens via `preprocess_text`.
2. The `<BOS>` special token is prepended.
3. For each token (e.g., `"hello"`):
   - Characters are split into `['h', 'e', 'l', 'l', 'o']`.
   - Merges from `merges` are applied iteratively. For example, if `('l', 'l')` is in `merges`, the character list becomes `['h', 'e', 'll', 'o']`, then remaining pairs are checked.
   - Unknown characters (e.g., `'#'`) are replaced with `<UNK>`.
4. The `<EOS>` token is appended.

```python
def tokenize(self, text: str) -> List[int]:
    tokens = self.preprocess_text(text)
    result = [self.token2id['<BOS>']]

    for token in tokens:
        symbols = list(token)

        while len(symbols) > 1:
            pairs = self.get_pairs(symbols)
            # Find first available merge pair
            pair_to_merge = None
            for pair in pairs:
                if pair in self.merges:
                    pair_to_merge = pair
                    break
            if not pair_to_merge:
                break

            # Replace pair with new token
            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols)-1 and (symbols[i], symbols[i+1]) == pair_to_merge:
                    new_symbols.append(self.merges[pair_to_merge])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        # Add final symbols to result
        for symbol in symbols:
            if symbol in self.token2id:
                result.append(self.token2id[symbol])
            else:
                result.append(self.token2id['<UNK>'])

    result.append(self.token2id['<EOS>'])
    return result
```

---

During tokenization, merges are applied left-to-right, and the **first** available pair from `merges` is chosen. This can yield different results depending on the merge order. For example, if `merges` contains `('h', 'e')` and `('e', 'l')`, the first encountered pair is merged.

```python
@dataclass
class BPETokenizer(BaseTokenizer):
    def __post_init__(self):
        super().__post_init__()
        self.merges = {}
        self.vocab = set(self.special_tokens)

    def get_pairs(self, word: List[str]) -> List[Tuple[str, str]]:
        return [(word[i], word[i+1]) for i in range(len(word)-1)]

    def train(self, texts: List[str], num_merges: int):
        word_freqs = defaultdict(int)
        all_chars = set()

        for text in texts:
            tokens = self.preprocess_text(text)
            for token in tokens:
                chars = list(token)
                word_freqs[' '.join(chars)] += 1
                all_chars.update(chars)

        for char in sorted(all_chars):
            if char not in self.token2id:
                idx = len(self.token2id)
                self.token2id[char] = idx
                self.id2token[idx] = char
                self.vocab.add(char)

        word_freqs = defaultdict(int)
        for text in texts:
            tokens = self.preprocess_text(text)
            for token in tokens:
                chars = list(token)
                word = ' '.join(chars)
                word_freqs[word] += 1

        print(f"Training BPE tokenizer for {self.language}...")
        for i in tqdm(range(num_merges)):
            pair_freqs = defaultdict(int)

            for word, freq in word_freqs.items():
                symbols = word.split()
                pairs = self.get_pairs(symbols)
                for pair in pairs:
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            new_token = ''.join(best_pair)

            self.merges[best_pair] = new_token
            self.vocab.add(new_token)

            if new_token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[new_token] = idx
                self.id2token[idx] = new_token

            new_word_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                symbols = word.split()
                i = 0
                new_symbols = []
                while i < len(symbols):
                    if i < len(symbols)-1 and (symbols[i], symbols[i+1]) == best_pair:
                        new_symbols.append(new_token)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                new_word = ' '.join(new_symbols)
                new_word_freqs[new_word] += freq

            word_freqs = new_word_freqs

            if (i + 1) % 1000 == 0:
                print(f"Merges completed: {i+1}/{num_merges}")
                print(f"Current vocabulary size: {len(self.token2id)}")

    def tokenize(self, text: str) -> List[int]:
        tokens = self.preprocess_text(text)
        result = [self.token2id['<BOS>']]

        for token in tokens:
            symbols = list(token)

            while len(symbols) > 1:
                pairs = self.get_pairs(symbols)
                pair_to_merge = None
                for pair in pairs:
                    if pair in self.merges:
                        pair_to_merge = pair
                        break
                if not pair_to_merge:
                    break

                i = 0
                new_symbols = []
                while i < len(symbols):
                    if i < len(symbols)-1 and (symbols[i], symbols[i+1]) == pair_to_merge:
                        new_symbols.append(self.merges[pair_to_merge])
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols

            for symbol in symbols:
                if symbol in self.token2id:
                    result.append(self.token2id[symbol])
                else:
                    result.append(self.token2id['<UNK>'])

        result.append(self.token2id['<EOS>'])
        return result
```

When applying the tokenizer to convert text to tokens, the algorithm first splits text into base characters, then iteratively merges character pairs using the built merge dictionary. Each word in the text is represented as a sequence of subwords (or tokens) created during training.

The number of merges (`num_merges` parameter) determines how many times the algorithm will merge characters into new tokens. More merges create larger, more informative tokens. However, excessive merges can lead to loss of fine-grained details.

This algorithm performs well with large text corpora and helps models handle rare or unseen words by replacing them with subwords from more frequent character combinations. Additionally, BPE works with any language, even those with unusual or complex alphabets, as it starts from base characters.

```python
en_bpe = BPETokenizer(language='en', vocab_size=32000)
ru_bpe = BPETokenizer(language='ru', vocab_size=32000)

n_samples = 80000
train_samples = dataset['train'].select(range(n_samples))
en_texts = [s['translation']['en'] for s in train_samples]
ru_texts = [s['translation']['ru'] for s in train_samples]

en_bpe.train(en_texts, num_merges=3000)
ru_bpe.train(ru_texts, num_merges=3000)

print(f"English vocabulary size: {len(en_bpe.token2id)}")
print(f"Russian vocabulary size: {len(ru_bpe.token2id)}")
```

```python
def test_tokenization(text: str, tokenizer: BPETokenizer):
    print(f"\nOriginal text: {text}")

    token_ids = tokenizer.tokenize(text)
    print(f"Token IDs: {token_ids}")

    tokens = [tokenizer.id2token[id] for id in token_ids]
    print(f"Tokens: {tokens}")

    return token_ids
```

```python
en_sample = dataset['train'][0]['translation']['en']
ru_sample = dataset['train'][0]['translation']['ru']

print("English tokenization:")
en_tokens = test_tokenization(en_sample, en_bpe)

print("\nRussian tokenization:")
ru_tokens = test_tokenization(ru_sample, ru_bpe)
```

Overall, BPE effectively addresses rare and complex words, improving tokenization quality and NLP model performance.

However, even after training, artifacts remain. For example, "useless" splits into ["us", "el", "ess"], and "бесполезно" into ["бес", "пол", "ез", "но"]. This stems from the limited number of merges and the lack of explicit morpheme boundary consideration in our educational implementation.

In production tokenizers (e.g., Hugging Face's), such issues are mitigated by pretraining on massive corpora and tens of thousands of merges.

## Batch Preparation

The `prepare_batch` function converts tokenized sequences into tensors suitable for training. Each sentence is padded to a fixed length (`max_length=64`) with the `<PAD>` token, and attention masks tell the model to ignore these "empty" positions.

For example, a sentence with 24 tokens becomes a vector of length 64, where the last 40 elements are zeros (ID `<PAD>`). Masking is critical for transformers, as the attention mechanism would otherwise account for meaningless padding tokens, distorting weights.

```python
def prepare_batch(batch: List[Dict],
                 src_tokenizer: BPETokenizer,
                 tgt_tokenizer: BPETokenizer,
                 max_length: int):

    src_texts = [item['translation']['en'] for item in batch]
    tgt_texts = [item['translation']['ru'] for item in batch]

    src_tokens = [src_tokenizer.tokenize(text) for text in src_texts]
    tgt_tokens = [tgt_tokenizer.tokenize(text) for text in tgt_texts]

    src_padded = []
    tgt_padded = []
    src_masks = []
    tgt_masks = []

    for src, tgt in zip(src_tokens, tgt_tokens):
        if len(src) > max_length:
            src_pad = src[:max_length]
            src_mask = [1] * max_length
        else:
            src_pad = src + [src_tokenizer.token2id['<PAD>']] * (max_length - len(src))
            src_mask = [1] * len(src) + [0] * (max_length - len(src))

        if len(tgt) > max_length:
            tgt_pad = tgt[:max_length]
            tgt_mask = [1] * max_length
        else:
            tgt_pad = tgt + [tgt_tokenizer.token2id['<PAD>']] * (max_length - len(tgt))
            tgt_mask = [1] * len(tgt) + [0] * (max_length - len(tgt))

        src_padded.append(src_pad)
        tgt_padded.append(tgt_pad)
        src_masks.append(src_mask)
        tgt_masks.append(tgt_mask)

    return {
        'src_tokens': np.array(src_padded),
        'tgt_tokens': np.array(tgt_padded),
        'src_mask': np.array(src_masks),
        'tgt_mask': np.array(tgt_masks)
    }
```

```python
test_samples = dataset['train'].select(range(5))
prepared_data = prepare_batch(test_samples, en_bpe, ru_bpe, max_length=64)

print("Prepared batch shapes:")
for key, value in prepared_data.items():
    print(f"{key}: {value.shape}")

print("\nExample source tokens:")
print(prepared_data['src_tokens'][0])
print("\nCorresponding mask:")
print(prepared_data['src_mask'][0])
```

```python
def verify_bpe_tokenization(tokenizer: BPETokenizer, text: str):
    print(f"Original text: {text}")

    base_tokens = tokenizer.preprocess_text(text)
    print(f"\nBase tokenization: {base_tokens}")

    print(f"\nNumber of merges learned: {len(tokenizer.merges)}")
    print("Sample merges (first 5):")
    for pair, merged in list(tokenizer.merges.items())[:5]:
        print(f"{pair} -> {merged}")

    print(f"\nVocabulary size: {len(tokenizer.token2id)}")
    print("Sample vocabulary items (first 10):")
    for token, idx in list(tokenizer.token2id.items())[:10]:
        print(f"{token}: {idx}")

    tokens = tokenizer.tokenize(text)
    decoded = [tokenizer.id2token[id] for id in tokens]

    print(f"\nFinal tokenization:")
    print(f"Token IDs: {tokens}")
    print(f"Decoded tokens: {decoded}")

print("Testing English tokenizer:")
verify_bpe_tokenization(en_bpe, dataset['train'][0]['translation']['en'])
```

## Hugging Face Tokenizers

All this seems quite complex. Our current tokenizer works imperfectly and is slow. Fortunately, programmers avoid reinventing the wheel. In practice, it's much easier to use a ready-made tokenizer via `AutoTokenizer` from the `transformers` library.

The `opus-mt-en-ru` model already has a pretrained BPE vocabulary optimized for the language pair. The tokenizer automatically adds special tokens, handles case, and rare symbols. When processing the dataset, the `map` function applies tokenization in parallel to all examples, speeding up work via batching.

```python
from transformers import AutoTokenizer

def prepare_data_with_hf(
    dataset,
    model_name: str = "Helsinki-NLP/opus-mt-en-ru",
    max_length: int = 128,
    batch_size: int = 32
):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        source_texts = [item['en'] for item in examples['translation']]
        target_texts = [item['ru'] for item in examples['translation']]

        source_encoding = tokenizer(
            source_texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='np'
        )

        target_encoding = tokenizer(
            target_texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='np'
        )

        return {
            'input_ids': source_encoding['input_ids'],
            'attention_mask': source_encoding['attention_mask'],
            'labels': target_encoding['input_ids'],
            'decoder_attention_mask': target_encoding['attention_mask']
        }

    processed_dataset = dataset['train'].map(
        preprocess_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset['train'].column_names
    )

    return processed_dataset, tokenizer
```

```python
processed_data, hf_tokenizer = prepare_data_with_hf(dataset)
```

## Comparing Tokenizers

```python
def print_custom_bpe_data_shape(prepared_data):
    print("\n" + "="*50)
    print("Custom BPE Tokenizer Data Structure:")
    print("Shape of prepared batches:")
    for key, array in prepared_data.items():
        print(f"{key}: {array.shape} (dtype: {array.dtype})")

    print("\nSample data from first batch:")
    print("Source tokens (first example):")
    print(prepared_data['src_tokens'][0])
    print("\nTarget tokens (first example):")
    print(prepared_data['tgt_tokens'][0])
    print("\nSource mask (first example):")
    print(prepared_data['src_mask'][0])
    print("="*50 + "\n")

def print_hf_data_details(processed_dataset, tokenizer):
    print("\n" + "="*50)
    print("Hugging Face Tokenizer Data Structure:")
    print(f"Dataset features: {processed_dataset.features}")
    print(f"Number of examples: {len(processed_dataset)}")

    first_example = processed_dataset[0]
    print("\nFirst example details:")
    print("Input IDs shape:", len(first_example['input_ids']))
    print("Decoded input:", tokenizer.decode(first_example['input_ids'], skip_special_tokens=True))
    print("Labels shape:", len(first_example['labels']))
    print("Decoded labels:", tokenizer.decode(first_example['labels'], skip_special_tokens=True))
    print("Attention mask sample:", first_example['attention_mask'][:10])
    print("="*50 + "\n")
```

```python
test_samples = dataset['train'].select(range(5))
prepared_data = prepare_batch(test_samples, en_bpe, ru_bpe, max_length=64)
print_custom_bpe_data_shape(prepared_data)


processed_data, hf_tokenizer = prepare_data_with_hf(dataset)
print_hf_data_details(processed_data, hf_tokenizer)
```