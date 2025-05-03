---
layout: post
title: Building a Transformer (Cross-Attention and MHA Explained)
date: 2025-04-16 15:00:00
description: Learn and implement the most iconic architecture in modern deep learning.
tags: machine-learning, transformers, multihead-attention, positional-encoding, nlp
categories: featured-posts
featured: false
---

[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/qmarva/implementing-transformer-en) [![Colab](https://img.shields.io/badge/Colab-F9AB00?style=plastic&logo=google-colab&logoColor=white)](https://colab.research.google.com/drive/1m34XYFZZTt-jbHo2OXlUfxR33zvFbXJZ?usp=sharing)

## Building a Transformer

The Transformer architecture marked a revolutionary step in sequence processing. Unlike traditional models such as RNNs and LSTMs, which handle data sequentially, the Transformer uses an attention mechanism that enables parallel processing of the entire sequence. This significantly accelerates training and improves performance.

The key innovation of the Transformer is the use of **self-attention**, which allows the model to effectively take into account the context of each word or token, regardless of its position in the sequence. This architecture has become the foundation of many modern models, including BERT, GPT, and others, and has greatly improved the quality of solutions in the field of natural language processing.

We'll need to install several libraries (and import even more), and the number of dependencies will only grow as we move forward. You can ignore these setup cells and just run them — the main focus should be on the core code.

```python
%%capture
!pip install -q torchdata==0.3.0 torchtext==0.12 spacy==3.2 altair==5.5.0 GPUtil==1.4.0
!python -m spacy download de_core_news_sm
!python -m spacy download en_core_web_sm
```

```python
import os
import time
import math
import copy
import spacy
import GPUtil
import pandas as pd
from typing import *
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

import altair as alt
from altair import Chart

alt.data_transformers.disable_max_rows()
```

## Positional Encoding

Transformer models work with numbers. To process text, it must be converted into a numerical format that the model can understand and work with. The first step is to convert text into tokens — for more details, see the notebook on tokenization. Tokens represented as vectors are called **embeddings**.

**Embeddings** are numerical vectors that capture the semantic meaning of words or subwords.

Before the Transformer architecture, sequence-processing models like RNNs and LSTMs handled data sequentially, inherently preserving the order of elements. However, their computational inefficiency due to step-by-step processing and poor parallelization led researchers to seek alternatives.

The Transformer overcame these limitations with a fully parallel approach. However, without positional information, the model wouldn't be able to distinguish between sentences with the same words in different orders. In Russian, word relationships are often expressed via case endings, but in English and many other languages, word order is crucial.

**Positional Encoding** is the mechanism in the Transformer architecture that enables the model to account for the order of words in a sequence.

Positional Encoding adds special signals (positional encodings) to the token embeddings based on their position in the sequence. These encodings have the same dimensionality as the embeddings so that they can be summed together. This additional information allows the model to distinguish, for instance, between the word "cat" at position 1 and "cat" at position 5, even if their semantic embeddings are identical. The encoding uses a formula that combines sine and cosine functions at different frequencies:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

where $$pos$$ is the position, $$d_{\text{model}}$$ is the embedding dimensionality, and $$i$$ is the index of the vector dimension.

The core idea is that these sinusoidal functions allow the model to pay attention to **relative positions**.

### Why does this strange formula encode relative positions?

#### First, the model can generalize to sequences longer than those seen during training.

Imagine that each position in a sequence is a point on a number line. If we generate signals for position $$pos$$ using sine and cosine, then the signals for position $$pos + k$$ can be expressed as a combination of the original values. For example, using the angle addition formula:

$$\sin(pos + k) = \sin(pos)\cos(k) + \cos(pos)\sin(k)$$

A shift of $$k$$ positions can be expressed as a weighted sum of the original sine and cosine values. This allows the model to infer that a word "three positions later" is related to the original word, even if it never saw such a long sequence during training.

#### Second, the distance between any two time steps is consistent across the sequence.

The logarithmic decay of frequencies in the term $$10000^{2i/d_{\text{model}}}$$ ensures that different dimensions of the positional vector capture different levels of positional detail. For small $$i$$ (early vector components), the denominator becomes large, causing the sine and cosine arguments to grow slowly with $$pos$$. This creates low-frequency oscillations that help distinguish between distant positions — for example, the beginning of the text (positions 1–100) versus the middle (positions 101–200). For larger $$i$$, the denominator shrinks, the argument grows faster, and high-frequency oscillations emerge, encoding fine-grained differences between neighboring positions (e.g., 101 and 102).

#### Third, this formula yields unique encodings for each position.

Alternating sine and cosine for even and odd indices solves the uniqueness issue. If we used only sine, different positions might accidentally match due to the periodicity of the function (e.g., $$\sin(pos)$$ and $$\sin(pos + 2\pi)$$). Adding cosine for neighboring vector components eliminates this symmetry: the combination of $$\sin(f(pos))$$ and $$\cos(f(pos))$$ across different frequencies $$f$$ ensures that each position $$pos$$ has a unique vector. The orthogonality of sine and cosine (their dot product is close to zero) minimizes overlap with word embeddings, allowing the model to separately process semantics and position.

---

The sum $$\text{Embedding} + PE$$ is possible because word embeddings and positional encodings have the same dimensionality $$d_{\text{model}}$$. This addition requires no trainable parameters: the model receives a combined signal where the word's semantics are modulated by its position. Gradients flow through this operation without distortion, as the derivative of a sum is the sum of the derivatives. As a result, during training, the model automatically learns to adjust both the semantic embeddings and the use of positional information (via attention), without conflicting signals.

While it's also possible to use **learned positional embeddings**, the sinusoidal version was chosen in the original paper because it enables the model to extrapolate to sequence lengths not seen during training. Experiments have shown that both versions yield nearly identical results.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)
```

## Attention, Self-Attention, Multi-Head Attention

### Attention

**Attention** is a mechanism that allows a model to weigh the importance of different elements in the input sequence.

It can be described as a function that takes a **query** and a set of **key-value** pairs, and produces an output — a weighted sum of the values. The weight assigned to each value is computed based on a compatibility function between the query and the corresponding key.

Imagine you're at a table with 50 experts. At the start, none of them knows anything about themselves or each other, but their goal during the meeting is to figure out:

* $$V$$: what they themselves know (their **Value** — knowledge/opinion),
* $$K$$: the best way to describe what they're good at (their **Key**),
* $$Q$$: the best way to express what information they're looking for (their **Query**).

If we used only $$Q$$ and $$K$$, the model wouldn't be able to transform the discovered dependencies into new features. The matrix $$V$$ adds flexibility, allowing the model to reweight values according to context.

Let's say you're one of those experts. You have a question (query), such as:

> "I need an opinion on Japanese cars."

You look around. Each expert has published a short description (key), for example:

* "I'm a mechanic specializing in Japanese cars"
* "I'm a chef who knows Italian cuisine"
* "I'm a driver who owned a Subaru"

You compare your query against the keys of the others. If someone's key matches well, you pay more attention to their value (opinion). You'll likely give the most weight to the mechanic and less to the driver.

As training progresses, you refine your query. Maybe next time, you realize you're not interested in Japanese cars, but in **Italian sewing machines**. And it turns out the chef, initially thinking they specialize in Italian food, actually knows sewing machines well.

So, you update the attention weights accordingly and learn to listen to the right expert.

---

### **Self-Attention**

**Self-attention** is a type of attention mechanism used in Transformers where the queries, keys, and values come from the same sequence. The original Transformer uses **Scaled Dot-Product Attention**, which works as follows:

1. **Create query, key, and value vectors**. For each input vector $$x$$ (e.g., a word embedding), three vectors are computed:

$$Q = xW_q,\quad K = xW_k,\quad V = xW_v$$

Here, $$W_q$$, $$W_k$$, and $$W_v$$ are trainable weight matrices. The dimensions of $$Q$$ and $$K$$ must match: $$d_k$$.

2. **Compute scores.** For each query vector $$Q_i$$ (corresponding to position $$i$$), scores are computed with all keys $$K_j$$ using the dot product:

$$\text{score}(i, j) = Q_i \cdot K_j^T$$

3. **Scale the scores.** To prevent large dot product values with high dimensions, the scores are divided by $$\sqrt{d_k}$$:

$$\text{scaled\_score}(i, j) = \frac{Q_i \cdot K_j^T}{\sqrt{d_k}}$$

4. **Apply Softmax.** Each row of scores is passed through Softmax for normalization:

$$\alpha_{ij} = \text{softmax}\left( \frac{Q_i \cdot K_j^T}{\sqrt{d_k}} \right)$$

The resulting $$\alpha_{ij}$$ are the attention weights.

5. **Compute the weighted sum of values.** Each value vector $$V_j$$ is multiplied by the attention weight $$\alpha_{ij}$$ and aggregated:

$$\text{Attention}(Q_i, K, V) = \sum_j \alpha_{ij} V_j$$

6. **Form the output vector.** The result is a vector containing contextual information relevant to position $$i$$. For the entire sequence:

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V$$

In practice, all computations are done in parallel using matrix operations, making the mechanism efficient and scalable.

---

### **Multi-Head Attention**

**Multi-Head Attention** is an extension of self-attention. While single-head attention focuses on one type of dependency (e.g., syntax or semantics), multi-head attention enables the model to capture multiple aspects of context simultaneously: grammatical relationships, anaphora, semantic parallels, etc.

Let the input be an embedding matrix $$X \in \mathbb{R}^{n \times d_{\text{model}}}$$, where $$n$$ is the sequence length and $$d_{\text{model}}$$ is the embedding dimension.

The idea remains the same: for each attention head $$h \in {1, \dots, H}$$, the input $$X$$ is projected into queries, keys, and values via trainable matrices:

$$Q_h = X W_h^Q,\quad K_h = X W_h^K,\quad V_h = X W_h^V$$

Typically, $$d_k = d_v = \frac{d_{\text{model}}}{H}$$ so that concatenating all heads results in the original dimension $$d_{\text{model}}$$.

Each head performs standard attention:

$$\text{Attention}_h(Q_h, K_h, V_h) = \text{softmax}\left( \frac{Q_h K_h^T}{\sqrt{d_k}} \right) V_h$$

The outputs from all $$H$$ heads are concatenated along the last dimension:

$$\text{Concat}( \text{head}_1, \dots, \text{head}_H ) \in \mathbb{R}^{n \times (H \cdot d_v)}$$

This combined output is projected back into $$d_{\text{model}}$$ using a final linear layer:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W^O$$

where $$W^O \in \mathbb{R}^{(H \cdot d_v) \times d_{\text{model}}}$$ is a trainable weight matrix of the final output projection layer.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q: [batch_size, num_heads, seq_len, head_dim]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(attn_output)
        return output
```

## FeedForward

Imagine that after passing through the Multi-Head Attention mechanism, the information for each word or token has become richer and more contextualized. Attention has blended information from different tokens to better understand each one in the context of the sentence. But now, this enriched information needs to be **processed and refined individually** for each token.

That's the role of the **FeedForward Network (FFN)**, which comes after the attention layer in each encoder and decoder block.

An FFN consists of two linear transformations. Between these two linear layers, there's a non-linear activation function — usually **ReLU**. Simply put, it's a small two-layer neural network.

One of the key features of the FFN in the Transformer is that it is **applied position-wise**. This means the *same* feedforward network is applied **independently** to the representation of **each token** in the sequence.

The dimension of the inner layer in the FFN is typically **larger** than the model dimension ($$d_{\text{model}}$$). In the original *"Attention Is All You Need"* paper, this inner dimension ($$d_{\text{ff}}$$) was **four times larger** than $$d_{\text{model}}$$ — that is, 2048 vs. 512 for the base model. However, other ratios may be used, such as doubling the size.


```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x
```

## EncoderLayer

Most competitive sequence transformation models follow an **encoder-decoder** structure.

The **encoder** receives the input and builds its representation (i.e., its features).

The encoder is composed of a stack of **identical layers**. The original paper uses a stack of **6 such layers**, though the number can vary. Each encoder layer consists of **two sub-layers**:

1. A **Multi-Head Self-Attention** mechanism
2. A **Feed-Forward Network (FFN)** — which we discussed earlier

Each of these sub-layers is wrapped in a **residual connection**, followed by a **layer normalization** step.

Residual connections help ensure smooth gradient flow when training very deep models and preserve information from the original input sequence.

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        # Self attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x
```

## DecoderLayer

The **decoder** uses the encoder's embeddings along with other inputs to generate the **target sequence**.

Like the encoder, the decoder is composed of a **stack of identical layers**, typically matching the encoder in depth.

In addition to the two sub-layers found in the encoder (Multi-Head Self-Attention and Feed-Forward Network), each decoder layer includes a **third sub-layer**: **Encoder-Decoder Attention**. This allows the decoder to focus on relevant parts of the input sequence — that is, the encoder's output.

The self-attention sub-layer in the decoder is **modified** to prevent attending to **future positions**. This is implemented by **masking** — setting the scores corresponding to illegal connections in the Softmax input to $$-\infty$$. This ensures that predictions for position $$i$$ depend only on known outputs at positions less than $$i$$.

As in the encoder, **residual connections** and **layer normalization** are applied around each sub-layer.

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Self attention (маскированное)
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross attention (с выходом энкодера)
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x
```

## Transformer

Once all the components of the Transformer — the encoder, decoder, attention mechanisms, and positional encodings — are implemented, the final step is to combine them into a single model that can be trained on sequence pairs (e.g., source text and its translation).

The **encoder** and **decoder** are each constructed as a **stack of `num_layers` layers**. Each `EncoderLayer` in the encoder sequentially refines the input representations: self-attention captures global dependencies, the feed-forward network introduces non-linearity, and residual connections with layer normalization ensure stability.

Similarly, each `DecoderLayer` applies masked self-attention, cross-attention to the encoder output, and a feed-forward network. Repeating these layers multiple times allows the model to iteratively refine representations — as if it is "re-reading" the data at different levels of abstraction.

The **final output layer** `fc_out` projects from the model dimension $$d_{\text{model}}$$ to the size of the target language vocabulary. This projection interprets the decoder's output vectors as **logits** — unnormalized scores for each token in the vocabulary:

$$\text{output} = W_{\text{out}} \cdot \text{dec\_output} + b_{\text{out}}$$

A Softmax (not explicitly shown in code, but implied in the loss function) is applied to these logits to produce a probability distribution over the vocabulary, from which the next word is selected.

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=0.1)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.positional_encoding(self.encoder_embedding(src))
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        tgt_emb = self.positional_encoding(self.decoder_embedding(tgt))
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        output = self.fc_out(dec_output)
        return output
```

## Testing Transformer (just run it)

```python
def test_transformer():
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    src_vocab_size = 100
    tgt_vocab_size = 100
    num_layers = 2

    # Generate synthetic data
    src = torch.randint(0, src_vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, seq_len))
    
    # Generate masks (example)
    src_mask = torch.ones(batch_size, 1, 1, seq_len)  # No masking
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).expand(batch_size, 1, seq_len, seq_len)  # Causal mask

    # Initialize the model
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    )

    print("=" * 50)
    print("1. Positional Encoding Test")
    pe = PositionalEncoding(d_model, dropout=0.1)
    x = torch.randn(1, seq_len, d_model)
    print(f"Before PE: mean={x.mean().item():.4f}, std={x.std().item():.4f}")
    x_pe = pe(x)
    print(f"After PE: mean={x_pe.mean().item():.4f}, std={x_pe.std().item():.4f}")
    print(f"PE Shape: {x_pe.shape} (should be [1, {seq_len}, {d_model}])")

    print("\n2. Multi-Head Attention Test")
    mha = MultiHeadAttention(d_model, num_heads)
    q = k = v = torch.randn(batch_size, seq_len, d_model)
    attn_output = mha(q, k, v)
    print(f"Attention output shape: {attn_output.shape} (should match {q.shape})")
    print(f"Max value: {attn_output.max().item():.4f}")
    print(f"Min value: {attn_output.min().item():.4f}")

    print("\n3. Encoder Layer Test")
    encoder_layer = EncoderLayer(d_model, num_heads)
    enc_input = torch.randn(batch_size, seq_len, d_model)
    enc_output = encoder_layer(enc_input)
    print(f"Encoder output shape: {enc_output.shape} (should match {enc_input.shape})")
    print(f"Data changed: {torch.allclose(enc_input, enc_output, atol=1e-4)} (should be False)")

    print("\n4. Decoder Layer Test")
    decoder_layer = DecoderLayer(d_model, num_heads)
    dec_input = torch.randn(batch_size, seq_len, d_model)
    dec_output = decoder_layer(dec_input, enc_output, src_mask, tgt_mask)
    print(f"Decoder output shape: {dec_output.shape} (should match {dec_input.shape})")
    print(f"Output norm: {dec_output.norm().item():.4f}")

    print("\n5. Full Transformer Test")
    print("Input data:")
    print(f"src: {src.shape} (max={src.max().item()}, min={src.min().item()})")
    print(f"tgt: {tgt.shape} (max={tgt.max().item()}, min={tgt.min().item()})")

    output = transformer(src, tgt, src_mask, tgt_mask)
    print("\nOutput shape check:")
    print(f"Expected shape: ({batch_size}, {seq_len}, {tgt_vocab_size})")
    print(f"Actual shape:   {output.shape}")

    print("\nGradient check:")
    dummy_loss = output.sum()
    dummy_loss.backward()
    has_gradients = any(p.grad is not None for p in transformer.parameters())
    print(f"Gradients computed: {has_gradients} (should be True)")

    print("\n6. Model Parameters Check:")
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Encoder embedding params: {sum(p.numel() for p in transformer.encoder_embedding.parameters())}")
    print(f"Decoder embedding params: {sum(p.numel() for p in transformer.decoder_embedding.parameters())}")

    print("\nTest completed!")
```