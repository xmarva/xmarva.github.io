---
layout: post
title: Building a Transformer
date: 2025-04-16 15:00:00
description: Learn and implement the most iconic architecture in modern deep learning.
tags: machine-learning, transformers, multihead-attention, positional-encoding, nlp
categories: featured-posts
featured: false
---

## Building a Transformer

The Transformer architecture was a groundbreaking development in the field of sequence processing. Unlike traditional models like RNNs and LSTMs, which process data sequentially, the Transformer uses an attention mechanism that allows it to process the entire sequence in parallel. This dramatically speeds up training and improves performance.

The key innovation of the Transformer is the use of self-attention, which enables the model to effectively take into account the context of each word or token, regardless of its position in the sequence. This architecture has become the foundation for many modern models, including BERT, GPT, and others, significantly improving the performance of natural language processing tasks.

We'll need to install a few libraries (and import even more), and as we go, the list will only grow. You can safely ignore these setup cells and just run them. Focus on the main code instead.

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

Before the Transformer architecture came along, sequence models like RNNs and LSTMs processed data step by step, inherently taking element order into account. But their inefficiency (due to sequential computation and difficulty with parallelization) drove the search for alternatives.

The Transformer eliminated these limitations by introducing a fully parallel approach. But that parallelism introduced a new problem: if all tokens are processed simultaneously, how can the model know their order?

To address this, researchers proposed **Positional Encoding** — a mechanism that encodes positional information into each element.

Positional Encoding adds special signals to the token embeddings that depend on their position in the sequence. This helps the model distinguish between, say, the word "cat" at position 1 and "cat" at position 5 — even if their semantic embeddings are identical. The encoding formula uses a mix of sine and cosine functions with different frequencies:

$$
PE_{(pos,\, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right), \quad
PE_{(pos,\, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)
$$


where $$pos$$ is the position in the sequence, $$d_{\text{model}}$$ is the embedding dimension, and $$i$$ is the index of the vector component.

The core idea is that these sinusoidal functions allow the model to pay attention to **relative positions**.

### Why does this formula encode relative positions?

Imagine every position in the sequence as a point on a number line. If we generate sine and cosine signals for position $$pos$$, then for position $$pos + k$$ those signals can be expressed using combinations of the originals. For example, using the angle addition formula:

$$  
\sin(pos + k) = \sin(pos)\cos(k) + \cos(pos)\sin(k),  
$$  

This means a shift by $$k$$ positions becomes a weighted sum of the original sine and cosine values. That lets the model naturally pick up on things like "a word three positions away" being related to the current one — even if it’s never seen sequences that long during training.

The logarithmic frequency decay in the denominator, $$10000^{2i/d_{\text{model}}}$$, ensures that different components of the position vector focus on different scales. For small $$i$$ (i.e., early dimensions of the vector), the denominator is large, so the sine and cosine functions grow slowly with $$pos$$. These low-frequency oscillations help distinguish broad regions of the sequence — like the beginning (positions 1–100) from the middle (positions 101–200). For large $$i$$, the denominator shrinks, so the functions grow faster and produce high-frequency oscillations that capture fine-grained position differences — like 101 vs. 102.

Alternating between sine and cosine for even and odd indices ensures unique positional encodings. If we used only sine, some positions could accidentally overlap because of its periodic nature (e.g., $$\sin(pos)$$ and $$\sin(pos + 2\pi)$$). Including cosine for neighboring vector components breaks that symmetry: the combination of $$\sin(f(pos))$$ and $$\cos(f(pos))$$ across various frequencies $$f$$ guarantees that every $$pos$$ has a unique vector. Since sine and cosine are nearly orthogonal (their dot product is close to zero), their signals don’t interfere with the word embeddings, letting the model process semantics and position independently.

The sum $$\text{Embedding} + PE$$ works because both word embeddings and positional encodings have the same dimensionality, $$d_{\text{model}}$$. This addition requires no learnable parameters — the model receives a unified signal where the meaning of a word is adjusted based on its position. Gradients flow cleanly through this operation since the derivative of a sum is just the sum of derivatives. As a result, during training, the model naturally learns to refine both the semantic embeddings and the use of positional signals (via the attention mechanism), without signal conflict.

Researchers explored other options too, like learnable positional embeddings. But the sinusoidal approach proved more effective at generalizing to sequences longer than those seen during training. So, Positional Encoding became a well-balanced solution — expressive, efficient, and free of extra learnable parameters — making it a perfect fit for the Transformer’s parallel architecture.

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

## MultiHeadAttention

**MultiHeadAttention** is the core of the Transformer architecture. Attention in Transformers emerged as a response to the limitations of earlier attention mechanisms used in seq2seq models. Initially, self-attention allowed each element in a sequence to interact with others by computing weighted sums of their features. But there was a problem: a single-head attention mechanism could only focus on **one type of dependency** — like syntactic relationships or semantic similarity. For complex tasks like translation, we need to capture **multiple types of interactions** at once: subject-verb agreement, anaphora, contextual synonyms, and more.

**The solution**: instead of one attention mechanism, use several parallel “heads,” each learning to capture its own type of dependency. Formally, for input vectors (embeddings) $$X \in \mathbb{R}^{n \times d_{\text{model}}}$$ — where $$n$$ is the sequence length and $$d_{\text{model}}$$ is the embedding size — each head $$h$$ projects $$X$$ into three separate spaces: queries ($$Q_h$$), keys ($$K_h$$), and values ($$V_h$$), using learned weight matrices:

$$  
Q_h = X W_h^Q, \quad K_h = X W_h^K, \quad V_h = X W_h^V,  
$$  

where $$W_h^Q, W_h^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$$, $$W_h^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$$, and $$d_k$$, $$d_v$$ are the dimensionalities of the key/query and value subspaces, respectively. This triplet ($$Q, K, V$$) mirrors concepts from information retrieval:

- **Queries** ($$Q$$) — what we're looking for,  
- **Keys** ($$K$$) — where we’re looking,  
- **Values** ($$V$$) — what we retrieve.

If we only had $$Q$$ and $$K$$, the model could measure similarity, but not transform or reweight information based on context. The $$V$$ matrix introduces that flexibility — it allows the model to adapt the retrieved features.

For each head, we compute **scaled dot-product attention**:

$$  
\text{Attention}(Q_h, K_h, V_h) = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}}\right) V_h.  
$$  

**Why softmax?** Softmax turns unbounded similarity scores (logits) into a probability distribution where the attention weights sum to 1. This keeps outputs within a stable range and ensures the model focuses on the most relevant tokens.

**Why scale by $$\sqrt{d_k}$$?** Without scaling, when $$d_k$$ is large, the dot product $$Q_h K_h^T$$ can have high variance. This leads to extremely sharp softmax distributions, causing gradients to vanish and slowing training. Scaling by $$\sqrt{d_k}$$ keeps gradients in a healthy range.

**Combining heads**: the outputs of all heads are concatenated and projected back to $$d_{\text{model}}$$:

$$  
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O,  
$$  

where $$W^O \in \mathbb{R}^{H d_v \times d_{\text{model}}}$$ is a learned projection matrix. Typically, we set $$d_k = d_v = d_{\text{model}} / H$$ to keep the total computation manageable. For example, with $$d_{\text{model}} = 512$$ and $$H = 8$$, we get $$d_k = d_v = 64$$.

**Why this choice of dimensions?**  
- If $$d_k$$ and $$d_v$$ stayed constant as $$H$$ increased, the computation cost would grow quadratically: $$O(H n^2 d_k)$$.  
- By reducing them to $$d_{\text{model}} / H$$, we keep the overall complexity at $$O(n^2 d_{\text{model}})$$ — the same as single-head attention.  
- The output projection $$W^O$$ restores the dimensionality to $$d_{\text{model}}$$, keeping it compatible with the rest of the Transformer stack.

**Why this structure works:**  
1. **Subspace separation**: Each head operates in its own $$d_k$$-dimensional subspace, letting the model learn **independent types of interactions**. One head might track noun-adjective agreement, another might attend to pronoun references. The projection matrices $$W_h^Q, W_h^K, W_h^V$$ effectively decompose the original embeddings into interpretable components.
2. **Parallelism**: Independent heads allow for efficient parallel computation on GPUs.  
3. **Interpretability**: After training, analyzing attention weights per head reveals the kinds of patterns each one has learned.

**Example computation for a single head:**  
Let $$X$$ be an embedding matrix of shape $$n \times d_{\text{model}}$$. For head $$h$$:

- $$Q_h = X W_h^Q$$ → shape $$n \times d_k$$
- $$K_h = X W_h^K$$ → shape $$n \times d_k$$  
- $$V_h = X W_h^V$$ → shape $$n \times d_v$$  

Then the attention matrix $$A_h = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}}\right)$$ (shape $$n \times n$$) is multiplied by $$V_h$$, giving an output of shape $$n \times d_v$$. Concatenating the outputs from all heads gives a $$n \times (H d_v)$$ matrix, which is projected back to $$n \times d_{\text{model}}$$ via $$W^O$$.

Preserving the **dimensionality** is critical: MultiHeadAttention outputs have the same shape $$d_{\text{model}}$$ as the input, allowing seamless integration with other Transformer components (like normalization and feed-forward layers) without extra transformation. This consistency also helps stabilize gradients in deep models.

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

We need the position-wise Feed Forward layer in the Transformer to provide **non-linear feature transformation** after the attention step. While MultiHeadAttention effectively captures global dependencies between tokens, that alone isn’t enough for complex tasks like translation — the model also needs to combine the extracted patterns and transform them into new semantic representations.

Each token in the sequence is processed **independently** through two linear layers. The first layer expands the dimensionality from $$d_{\text{model}}$$ (e.g., 512) to $$d_{\text{ff}}$$ (typically 2048), followed by a ReLU activation function:

$$  
\text{hidden} = \text{ReLU}(x W_1 + b_1),  
$$

where $$W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$$.

Expanding the dimensionality by a factor of 4 ($$d_{\text{ff}} = 4d_{\text{model}}$$) gives the model enough capacity to learn non-obvious combinations of features. The second linear layer projects the representation back down to the original dimensionality:

$$  
\text{output} = \text{hidden} W_2 + b_2,  
$$

where $$W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$$. Dropout (typically with a rate like 0.1) is applied between the layers for regularization.

**Why this design?**

- **Non-linearity**: ReLU breaks linearity, enabling the model to approximate complex functions. Without it, the two linear layers would collapse into a single matrix multiplication.
- **Expansion and compression**: Increasing the dimensionality creates a kind of "bottleneck" that forces the model to filter out noise and extract more abstract features. This is similar to how an autoencoder works — but without information loss, since the output returns to the original size.
- **Position-wise independence**: Processing each token separately helps compensate for any local information that might be diluted by the global attention mechanism. For example, in the phrase "blue ball", attention may link the adjective to the noun, but the FFN refines their joint representation into a vector encoding both color and shape.

The input and output of the FeedForward layer both have the same dimensionality, $$d_{\text{model}}$$, which allows for stacking multiple Encoder/Decoder blocks. Dropout and residual connections (implemented outside this layer) help stabilize training in deep networks.


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

Before the introduction of the **EncoderLayer** in the Transformer, researchers faced a dilemma: how to combine global contextual understanding with local feature transformation, while maintaining stable training in deep networks. Earlier approaches like RNNs suffered from vanishing gradients, and convolutional networks required many layers to capture long-range dependencies. Self-attention solved the problem of modeling global context — but on its own, it couldn’t provide deep, hierarchical feature transformation. This raised the question: how can we structure sequential transformations so the model first identifies relationships between tokens, then "rethinks" them, all while staying robust as the network grows deeper?

The **EncoderLayer** was the answer — a module that combines two essential stages. First, the input embeddings $$x$$, already enriched with positional information (via Positional Encoding), are passed through **MultiHeadAttention**. Here, each token “asks” the rest of the sequence:

$$  
\text{attn\_output} = \text{MultiHeadAttention}(x, x, x, mask),  
$$  

where `mask` is used to ignore future tokens (in the decoder) or padding tokens. This lets the model, for instance, link the pronoun “he” to the correct noun, even if they’re separated by dozens of words. But attention is a **linear** operation in feature space. To introduce **non-linearity and depth**, we follow up with a **Feed Forward Network (FFN)** — two linear layers with an intermediate dimensionality expansion:

$$  
\text{ffn\_output} = \text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2.  
$$  

The FFN acts like the model’s “thought process”: it transforms the global dependencies identified by attention into new semantic representations. For example, if attention links “apple” and “green,” the FFN can encode that as a combined vector representing both fruit and color.

But simply chaining these operations wasn’t enough. Deep networks often “forgot” the original inputs — gradients vanished, and features got distorted. This is where **residual connections** and **layer normalization** came in. After each sub-step (attention or FFN), the layer adds the original input $$x$$ to the output and applies normalization:

$$  
x = \text{LayerNorm}(x + \text{Dropout}(\text{sublayer}(x))).  
$$  

Residuals act like bridges, allowing gradients and raw input information to flow freely through even dozens of layers. LayerNorm stabilizes activation distributions by computing the mean and variance across the $$d_{\text{model}}$$ dimensions — preventing exploding or vanishing values.

**Why this order?** If the FFN came before attention, the ReLU non-linearity could "break" the positional information that's critical for self-attention. And the use of **post-layer normalization** (after the residual connection) instead of pre-normalization (before the sub-step) wasn’t arbitrary: in the original Transformer, this design helped gradients flow through both the transformed path and the original input path, balancing parameter updates.

**Example**: An embedding for the word “bank” may, after attention, be linked to “river” (bank as a shore) or “money” (bank as a financial institution). The FFN then transforms these associations into a context-specific representation. Residuals and normalization ensure that the signal remains stable. As this process repeats across multiple EncoderLayers, the model refines the meaning iteratively — like rereading a sentence and noticing new details each time.

Historically, the EncoderLayer became the blueprint for scalability. It could be stacked N times (e.g., 6 or 12 layers), enabling deep models without collapsing gradients. The combination of self-attention and FFN turned out to be so effective that even today’s large language models — like GPT-4 — retain this core structure, merely enhancing it with new mechanisms.

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

If the EncoderLayer in a Transformer learns to **understand** the input text by compressing its context into dense vectors, then the **DecoderLayer** was designed to **generate output** — word by word — while taking into account both previous predictions and the encoder’s information. Early approaches like seq2seq with attention already connected encoder and decoder, but their recurrent nature limited parallelism and made it harder to model complex dependencies. In the Transformer, the decoder had to be **autoregressive**, yet still **parallelizable** — and this is where **masked self-attention** combined with **cross-attention** became crucial.

A **DecoderLayer** starts with a partially generated output sequence (e.g., the translation generated up to the current word). To ensure that the model doesn’t “peek” at future tokens, it uses **masked self-attention**:

$$  
\text{attn\_output} = \text{MultiHeadAttention}(x, x, x, tgt\_mask),  
$$  

where $$tgt\_mask$$ is an upper-triangular matrix with $$-\infty$$ in positions corresponding to future tokens. When passed through softmax, these become zeros — effectively blocking attention to future positions. For instance, while generating the third word, the mask hides all tokens beyond the third, forcing the model to rely only on previously generated context.

But self-attention isn’t enough — the decoder also needs to **relate the output to the input**. This is where **cross-attention** comes in: queries ($$Q$$) come from the decoder, while keys ($$K$$) and values ($$V$$) are taken from the encoder output:

$$  
\text{cross\_attn\_output} = \text{MultiHeadAttention}(x, enc\_output, enc\_output, src\_mask).  
$$  

Here, $$src\_mask$$ hides padding tokens from the source sequence. This step acts like an “interrogation” of the encoder: the decoder asks which parts of the input are relevant at this step in the output. For example, when translating the word “apple,” the decoder can use cross-attention to link it to either “яблоко” or “компания,” depending on context.

After cross-attention, as in the encoder, comes a **Feed Forward Network** to inject non-linearity:

$$  
\text{ffn\_output} = \text{FFN}(x).  
$$  

Each sublayer is wrapped with **residual connections** and **layer normalization**:

$$  
x = \text{LayerNorm}(x + \text{Dropout}(\text{sublayer}(x))),  
$$  

ensuring gradient stability even in very deep networks.

**Why three stages?**  
1. **Masked self-attention** isolates the already-generated portion of the sequence, simulating RNN-like autoregression.  
2. **Cross-attention** synchronizes encoder and decoder, letting the decoder “look into” the input — akin to alignment in statistical machine translation.  
3. **FFN** transforms the combined context into a decision — a final refinement before predicting the next token.

**Example**: When translating “I hit the bank” into Russian:  
1. Masked self-attention links “I hit” to “the,” while blocking future tokens.  
2. Cross-attention identifies whether “bank” aligns to “берег” (if the context is a river) or “банк” (if financial).  
3. The FFN processes this and outputs either “по берегу” or “в банк,” preserving grammar and meaning.

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

Once all the components of the Transformer — encoder, decoder, attention mechanisms, and positional encodings — were developed, the final task was to **assemble them into a complete model** capable of learning from sequence pairs (e.g., source text and translation). Early approaches like Seq2Seq already used an encoder-decoder separation, but their recurrent nature limited both parallelism and depth. The Transformer architecture, as implemented in code, emerged as a balance between expressiveness and computational efficiency.

**Model assembly** starts with turning tokens into vectors. The embeddings (`encoder_embedding` and `decoder_embedding`) map words into a $$d_{\text{model}}$$-dimensional space, while `positional_encoding` injects positional information:

$$  
X_{\text{enc}} = \text{Embedding}(src) + \text{PositionalEncoding}(src),  
$$  
$$  
X_{\text{dec}} = \text{Embedding}(tgt) + \text{PositionalEncoding}(tgt).  
$$

Without positional encoding, the model wouldn’t be able to distinguish between permutations of words, since self-attention alone is order-invariant.

Next, the encoder and decoder are built as **stacks of layers** (`num_layers`). Each layer in the encoder (`EncoderLayer`) progressively refines the input representations: self-attention extracts global dependencies, the FFN adds non-linearity, and residual connections with layer normalization ensure stability. Similarly, the decoder (`DecoderLayer`) applies masked self-attention, cross-attention to the encoder output, and the FFN — in that order. Repeating these layers allows the model to iteratively refine its understanding, as if "rereading" the data at different levels of abstraction.

The **final layer** (`fc_out`) projects from $$d_{\text{model}}$$ to the size of the target vocabulary. This projection interprets decoder vectors as logits — scores for each token in the vocabulary:

$$  
\text{output} = W_{\text{out}} \cdot \text{dec\_output} + b_{\text{out}}.  
$$

A softmax (not explicitly shown in code but implied in the loss function) converts these logits into a probability distribution, from which the next word is sampled or selected.

**Why this particular structure?**  
- **Depth (`num_layers`)**: Each layer captures different facets of the data. Early encoder layers may pick up syntax, while later ones capture semantics. In the decoder, lower layers focus on alignment with the encoder, while upper ones refine output grammar and fluency.  
- **Separate embeddings**: Using different embedding matrices for source and target languages allows the model to work effectively in multilingual settings.  
- **Dimensional consistency**: All components maintain the same dimensionality $$d_{\text{model}}$$, which simplifies training — gradients flow freely through residual paths, and parameters update coherently.
- 

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

## Testing

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