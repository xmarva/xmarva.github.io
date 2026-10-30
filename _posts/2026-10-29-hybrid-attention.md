---
layout: post
title: "Hybrid attention: why new LLMs keep full attention in only a few layers"
date: 2026-10-29 12:00:00
description: "Linear attention, the delta rule and Gated DeltaNet in PyTorch, a recall test, GPU timings, and why Qwen3-Next and Kimi Linear keep few attention layers"
tags: [machine-learning, transformers, attention, linear-attention, llm]
categories: machine-learning
featured: false
thumbnail: /assets/posts/hybrid-attention/cover.png
og_image: https://xmarva.github.io/assets/posts/hybrid-attention/cover.png
stylesheets: [/assets/css/tfs-course.css]
toc:
  beginning: true
---
{% raw %}

Qwen3-32B, a dense model from April 2025, keeps 32 GiB of keys and values for one conversation of 128 thousand tokens, about half of what its weights take. Qwen3-Next, released by the same team a few months later, keeps 3 GiB at the same length. It doesn't compress its cache or skip tokens.

Three of every four of its layers simply have no cache: they read the text the way a recurrent network does, into a state of fixed size, and only every fourth layer is ordinary attention.

Qwen3-Next is not the only one. Jamba, the Nemotron-H models, Kimi Linear and, for a while, MiniMax made the same choice: full attention in a few layers, something cheaper in the rest.

This post builds the cheaper part from scratch, from linear attention to the gated delta rule, tests what it can and cannot remember, and then looks at how real models combine the two and what that buys on a GPU.

## What attention remembers

Every attention layer of a Transformer answers the same question for every token: which of the earlier tokens matter here, and what do they say. Softmax attention answers it by brute force. Each token leaves behind a key, which says what it is about, and a value, which says what it contributes.

A new token arrives with a query, compares it with every key, turns the scores into weights with a softmax, and takes the weighted sum of the values:

$$o_t = \sum_{j \le t} \frac{\exp(q_t \cdot k_j / \sqrt{d})}{\sum_{i \le t} \exp(q_t \cdot k_i / \sqrt{d})}\, v_j .$$

Nothing is ever thrown away. To write token $$t$$, the layer needs the keys and values of all $$t-1$$ tokens before it, which is why serving engines keep them in a **KV cache** instead of recomputing them.

Memory therefore grows linearly with the text, and since each new token reads all of it, reading a whole text of $$n$$ tokens takes time proportional to $$n^2$$.

Think of a library that keeps every book it has ever received and checks the whole catalogue for every question. The answers are exact, but the building has to grow with the collection.

## Attention without the softmax

The softmax is what forces the rereading: the exponential ties every query to every key. Katharopoulos and colleagues replaced it, putting a dot product of features $$\phi(q) \cdot \phi(k)$$ in place of $$\exp(q \cdot k)$$, with a positive map such as $$\phi(x) = \mathrm{elu}(x) + 1$$, and noticed that the rest follows from associativity<sup class="cite">[<a href="#ref-1" id="cite-1" data-ref="1" aria-label="Reference 1: Katharopoulos, A., Vyas, A., Pappas, N., &amp; Fleuret, F. (2020). Transformers are RNNs: fast autoregressive Transformers with linear attention. ICML 2020.">1</a>]</sup>:

$$o_t = \frac{\sum_{j \le t} \big(\phi(q_t)^\top \phi(k_j)\big) v_j}{\sum_{j \le t} \phi(q_t)^\top \phi(k_j)} = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}, \qquad S_t = \sum_{j \le t} \phi(k_j)\, v_j^\top, \quad z_t = \sum_{j \le t} \phi(k_j).$$

The query no longer sits inside the sum. Everything the layer needs from the past now sits in $$S_t$$, a $$d \times d$$ matrix, and $$z_t$$, a vector, and both are updated by adding one term per token:

$$S_t = S_{t-1} + \phi(k_t)\, v_t^\top .$$

This is a recurrent network. Its state has the same size after ten tokens and after ten million, writing a token costs the same at any position, and there is no cache. Done for all positions at once, the same computation is the attention formula with the brackets moved:

<img class="tfs-figure tfs-light" src="/assets/posts/hybrid-attention/associativity-light.png" width="760" alt="The same product computed in two orders, for 32,768 tokens and 128-dimensional heads. Multiplying the queries by the keys first gives a 32,768 by 32,768 matrix of scores, 1,073,741,824 numbers, before the values are applied. Multiplying the keys by the values first gives a 128 by 128 matrix, 16,384 numbers, that every query then reads. Without the softmax between them, the order is free" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/hybrid-attention/associativity-dark.png" width="760" alt="The same product computed in two orders, for 32,768 tokens and 128-dimensional heads. Multiplying the queries by the keys first gives a 32,768 by 32,768 matrix of scores, 1,073,741,824 numbers, before the values are applied. Multiplying the keys by the values first gives a 128 by 128 matrix, 16,384 numbers, that every query then reads. Without the softmax between them, the order is free" loading="lazy">

Both forms compute the same numbers, only in a different order. In code, the recurrent form loops over positions and the parallel one builds the masked score matrix:

```python
def linear_recurrent(q, k, v):
    q, k = feature(q), feature(k)
    state = q.new_zeros(*q.shape[:2], q.shape[-1], v.shape[-1])  # S = sum of k v^T
    norm = q.new_zeros(*q.shape[:2], q.shape[-1])                # z = sum of k
    out = []
    for t in range(q.shape[2]):
        state = state + k[:, :, t, :, None] * v[:, :, t, None, :]
        norm = norm + k[:, :, t]
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], state)
                   / torch.einsum("bhk,bhk->bh", q[:, :, t], norm)[..., None])
    return torch.stack(out, dim=2)

def linear_attention(q, k, v):
    q, k = feature(q), feature(k)
    causal = torch.ones(q.shape[2], q.shape[2], dtype=torch.bool).tril()
    scores = (q @ k.transpose(-1, -2)).masked_fill(~causal, 0)
    return (scores @ v) / scores.sum(-1, keepdim=True)
```

On random inputs in double precision the two functions agree to about $$10^{-15}$$, which is rounding error. The recurrent form is how the layer writes text, one token at a time; the parallel form is how it trains, all positions at once.

The parallel form also shows a catch: it still builds an $$n \times n$$ matrix of scores, because with a causal mask every position needs its own prefix of the sum and the brackets can't simply move. Training a linear layer efficiently needs a third form, which comes later.

## A memory of fixed size

The price of the fixed state can be read straight from the formula. The state is one matrix. Every pair $$(k_j, v_j)$$ is added into the same $$d^2$$ numbers, and to recall the value stored under a key, the layer multiplies the state by it:

$$S^\top k_i = \sum_j (k_j \cdot k_i)\, v_j = (k_i \cdot k_i)\, v_i + \sum_{j \ne i} (k_j \cdot k_i)\, v_j .$$

The first term is the stored value. The second is interference: every other pair, weighted by how similar its key is to $$k_i$$. If the keys are orthogonal, the interference is exactly zero, but a $$d$$-dimensional space has room for only $$d$$ orthogonal directions.

Beyond that, each new pair blurs the ones already stored. The library has turned into one notebook page on which every note is written over the others; it stays readable while there are few notes.

Softmax attention has no such limit because it never mixes the notes: it keeps them all and lets the exponential pick out the right one. It pays for that with the growing cache. Everything below is about this trade between a fixed state and exact recall.

Arora and colleagues measured it across efficient architectures and called it "a key tradeoff between a model's state size and recall ability"<sup class="cite">[<a href="#ref-2" id="cite-2" data-ref="2" aria-label="Reference 2: Arora, S., et al. (2024). Simple linear attention language models balance the recall-throughput tradeoff. ICML 2024.">2</a>]</sup>: the smaller the state a model keeps while reading, the worse it recalls.

## Writing with the delta rule

Plain addition has a second, quieter problem. If the same key comes back with a new value, say a variable that has been reassigned, the state holds the sum of the old and the new value, and reading it returns a mixture of both. Nothing in the notebook can be crossed out.

The remedy dates from 1960, when Widrow and Hoff trained their linear neurons with what became known as the delta rule: compare the output with the target and move the weights in proportion to the error<sup class="cite">[<a href="#ref-3" id="cite-3" data-ref="3" aria-label="Reference 3: Widrow, B., &amp; Hoff, M. E. (1960). Adaptive switching circuits. IRE WESCON Convention Record, 4, 96–104.">3</a>]</sup>.

Schlag, Irie and Schmidhuber noticed that the state of linear attention is a weight matrix, written at every step, and proposed writing it the same way<sup class="cite">[<a href="#ref-4" id="cite-4" data-ref="4" aria-label="Reference 4: Schlag, I., Irie, K., &amp; Schmidhuber, J. (2021). Linear Transformers are secretly fast weight programmers. ICML 2021.">4</a>]</sup>. Before storing $$v_t$$ under $$k_t$$, read what is already there, and write only the difference:

$$S_t = S_{t-1} + \beta_t\, k_t \big(v_t - S_{t-1}^\top k_t\big)^\top = (I - \beta_t k_t k_t^\top)\, S_{t-1} + \beta_t\, k_t v_t^\top ,$$

with $$k_t$$ of unit length and $$\beta_t \in (0, 1)$$, a writing strength the layer computes for each token. This is one step of gradient descent on the memory's error for this key, $$\tfrac12 \lVert S^\top k_t - v_t \rVert^2$$, with learning rate $$\beta_t$$: the gradient with respect to $$S$$ is $$k_t (S^\top k_t - v_t)^\top$$.

With $$\beta_t = 1$$, the old value under $$k_t$$ is erased and the new one is stored exactly. A layer that writes this way is called **DeltaNet**. In the space of values, one write looks like this:

<img class="tfs-figure tfs-light" src="/assets/posts/hybrid-attention/delta-step-light.png" width="760" alt="One write of the delta rule, drawn in the space of values. The memory returns an old value for the key; the new value is elsewhere; the error is the arrow between them. A write of strength 0.5 moves the memory&#x27;s answer halfway along the error, a write of strength 1 lands on the new value exactly. Plain addition would instead add the new value to the old one" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/hybrid-attention/delta-step-dark.png" width="760" alt="One write of the delta rule, drawn in the space of values. The memory returns an old value for the key; the new value is elsewhere; the error is the arrow between them. A write of strength 0.5 moves the memory&#x27;s answer halfway along the error, a write of strength 1 lands on the new value exactly. Plain addition would instead add the new value to the old one" loading="lazy">

Side by side, the three memories differ in what they keep and in how a new pair changes it:

<img class="tfs-figure tfs-light" src="/assets/posts/hybrid-attention/memories-light.png" width="760" alt="Three ways for a layer to remember the tokens it has read. Softmax attention keeps the key and the value of every token, so its memory grows with the text, and a query is compared with every key. Linear attention keeps one matrix and adds the outer product of each key and value to it, so its memory has a fixed size. The delta rule keeps the same matrix but first reads what it holds under the new key and writes only the difference" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/hybrid-attention/memories-dark.png" width="760" alt="Three ways for a layer to remember the tokens it has read. Softmax attention keeps the key and the value of every token, so its memory grows with the text, and a query is compared with every key. Linear attention keeps one matrix and adds the outer product of each key and value to it, so its memory has a fixed size. The delta rule keeps the same matrix but first reads what it holds under the new key and writes only the difference" loading="lazy">

How much of a difference does editing make? A small experiment shows both sides of it: a $$64 \times 64$$ memory, random unit keys and values, written once by adding and once by the delta rule, and read back by finding the nearest stored value. Softmax attention, which keeps every pair, is the reference.

<img class="tfs-figure tfs-light" src="/assets/posts/hybrid-attention/memory-light.png" width="760" alt="Left: share of 64-dimensional key-value pairs recalled from a 64 by 64 memory. Keeping every pair, as softmax attention does, recalls all of them at any number. Adding outer products recalls 98% of 128 pairs and 34% of 512; editing with the delta rule recalls 53% and 13%, keeping the newest. Right: after 32 pairs, some keys get a new value; the delta rule returns the new value 100% of the time, adding and softmax attention about half the time" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/hybrid-attention/memory-dark.png" width="760" alt="Left: share of 64-dimensional key-value pairs recalled from a 64 by 64 memory. Keeping every pair, as softmax attention does, recalls all of them at any number. Adding outer products recalls 98% of 128 pairs and 34% of 512; editing with the delta rule recalls 53% and 13%, keeping the newest. Right: after 32 pairs, some keys get a new value; the delta rule returns the new value 100% of the time, adding and softmax attention about half the time" loading="lazy">

The left panel holds a surprise: from 64 pairs on, plain addition recalls more than the delta rule. With $$\beta_t = 1$$, each write makes the memory return exactly $$v_t$$ for $$k_t$$, and since random keys are not orthogonal, it also shifts what the memory returns for every older key that overlaps with $$k_t$$.

Once the memory is full, the delta rule holds the recent pairs well and the old ones poorly. Addition keeps a blur of everything, in which the nearest value is still often the right one.

The right panel shows what the delta rule is for. After 32 pairs, some of the keys get a new value. The delta rule returns the new value nearly every time.

Addition returns the sum of the old and the new value, which is nearer to the new one only about half the time, and softmax attention as written here does no better: it sees two identical keys and averages their values.

A trained Transformer can tell the later pair from the earlier one by their positions; this toy has no positions.

So the delta rule doesn't enlarge the memory. A $$d \times d$$ matrix stays a $$d \times d$$ matrix, and it holds at most $$d$$ pairs exactly, whichever way they are written. What the delta rule adds is the ability to correct what is stored, which plain addition doesn't have at all.

## Forgetting on purpose

Editing handles values that change. It does nothing about the opposite problem: a long text moves on, and much of what the state holds stops mattering. A memory of fixed size needs a way to let go of it.

That is what gates are for. RetNet multiplied the state by a fixed decay at every step<sup class="cite">[<a href="#ref-5" id="cite-5" data-ref="5" aria-label="Reference 5: Sun, Y., et al. (2023). Retentive network: a successor to Transformer for large language models. arXiv.">5</a>]</sup>, Gated Linear Attention made the decay depend on the token<sup class="cite">[<a href="#ref-6" id="cite-6" data-ref="6" aria-label="Reference 6: Yang, S., Wang, B., Shen, Y., Panda, R., &amp; Kim, Y. (2024). Gated linear attention Transformers with hardware-efficient training. ICML 2024.">6</a>]</sup>, and Mamba-2 showed that its own selective state space model is linear attention with a data-dependent scalar gate<sup class="cite">[<a href="#ref-7" id="cite-7" data-ref="7" aria-label="Reference 7: Dao, T., &amp; Gu, A. (2024). Transformers are SSMs: generalized models and efficient algorithms through structured state space duality. ICML 2024.">7</a>]</sup>. **Gated DeltaNet** puts the gate and the delta rule together<sup class="cite">[<a href="#ref-8" id="cite-8" data-ref="8" aria-label="Reference 8: Yang, S., Kautz, J., &amp; Hatamizadeh, A. (2025). Gated delta networks: improving Mamba2 with delta rule. ICLR 2025.">8</a>]</sup>:

$$S_t = \alpha_t \,(I - \beta_t k_t k_t^\top)\, S_{t-1} + \beta_t\, k_t v_t^\top, \qquad \alpha_t \in (0, 1).$$

Both $$\alpha_t$$ and $$\beta_t$$ are computed from the current token, so the layer decides as it reads when to forget and how firmly to write.

The two parts split the work: in the authors' words, the gate "can promptly clear memory by setting $$\alpha_t \to 0$$", and with $$\alpha_t \to 1$$ the layer is "effectively switching to the pure delta rule" and edits single entries without touching the rest.

This is the layer Qwen3-Next is built on, and Kimi Delta Attention is the same update with a separate gate for every channel of the state instead of one per head<sup class="cite">[<a href="#ref-9" id="cite-9" data-ref="9" aria-label="Reference 9: Kimi Team (2025). Kimi Linear: an expressive, efficient attention architecture. arXiv.">9</a>]</sup>.

## Training a recurrence in parallel

A recurrence over positions is awkward to train on a GPU, which is fast at large matrix products and slow at loops of ten thousand small updates. The solution is to cut the sequence into chunks.

Inside a chunk of 64 tokens, the writes depend on each other only through a small lower-triangular system that can be solved with matrix products; between chunks, only the $$d \times d$$ state is passed along<sup class="cite">[<a href="#ref-10" id="cite-10" data-ref="10" aria-label="Reference 10: Yang, S., Wang, B., Zhang, Y., Shen, Y., &amp; Kim, Y. (2024). Parallelizing linear Transformers with the delta rule over sequence length. NeurIPS 2024.">10</a>]</sup>.

<img class="tfs-figure tfs-light" src="/assets/posts/hybrid-attention/chunks-light.png" width="760" alt="How a recurrent layer is trained in parallel. The sequence is cut into chunks of 64 tokens. Inside a chunk, all positions are computed at once with matrix products; between chunks, only the d by d state is passed on, one step per chunk instead of one per token" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/hybrid-attention/chunks-dark.png" width="760" alt="How a recurrent layer is trained in parallel. The sequence is cut into chunks of 64 tokens. Inside a chunk, all positions are computed at once with matrix products; between chunks, only the d by d state is passed on, one step per chunk instead of one per token" loading="lazy">

The chunked form is exact. My plain-PyTorch version of it agrees with the token-by-token loop to $$1.4 \times 10^{-15}$$ in double precision, with the gate or without it, and it is what the experiments below train with. Production models use the Triton kernels of [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), which do the same thing much faster.

## A test of memory: associative recall

The language modelling loss says little about memory, because much of predicting the next word needs little of it. A more direct test asks a model to remember what it was told. In multi-query associative recall, a sequence lists key-value pairs and then asks for the values of the keys in a shuffled order<sup class="cite">[<a href="#ref-11" id="cite-11" data-ref="11" aria-label="Reference 11: Arora, S., et al. (2023). Zoology: measuring and improving recall in efficient language models. ICLR 2024.">11</a>]</sup>:

```
k1 v1  k2 v2  ...  kN vN   |   q1 a1  q2 a2  ...  qN aN
```

Each $$q_i$$ is one of the keys, and the model must put the right value after it. It is the skill behind answering a question about a long document, using a name defined forty pages earlier, or calling a function with the arguments described at the top of a prompt.

I trained six small models on it, each with four layers of width 64 and two heads, the same in everything except how their layers mix tokens: softmax attention in all four, linear attention, DeltaNet, Gated DeltaNet, and two hybrids with three Gated DeltaNet layers and one softmax attention layer.

In one hybrid the attention layer is second; in the other it is on top, the place it has in each block of four in Qwen3-Next.

Every layer, attention included, first passes its queries, keys and values through a short convolution over the last four positions, as Mamba and DeltaNet do, so that a value can carry its key with it.

Each model saw 64 fresh random sequences at each of 4,000 steps and was scored on held-out ones. The models with Gated DeltaNet layers trained on an Apple M2, the others on a T4 ([code](https://github.com/xmarva/transformers-from-scratch/tree/main/articles/hybrid_attention)).

<img class="tfs-figure tfs-light" src="/assets/posts/hybrid-attention/mqar-light.png" width="760" alt="Share of values recalled with 4, 8, 16, 32 and 64 key-value pairs, by six models of four layers and width 64: Transformer 100%, 100%, 100%, 100%, 100%; Hybrid (attention 2nd) 100%, 100%, 100%, 100%, 100%; Hybrid (attention on top) 100%, 100%, 26%, 12%, 1%; Gated DeltaNet 100%, 98%, 19%, 2%, 0%; DeltaNet 100%, 100%, 71%, 36%, 13%; Linear attention 97%, 82%, 76%, 26%, 0%" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/hybrid-attention/mqar-dark.png" width="760" alt="Share of values recalled with 4, 8, 16, 32 and 64 key-value pairs, by six models of four layers and width 64: Transformer 100%, 100%, 100%, 100%, 100%; Hybrid (attention 2nd) 100%, 100%, 100%, 100%, 100%; Hybrid (attention on top) 100%, 100%, 26%, 12%, 1%; Gated DeltaNet 100%, 98%, 19%, 2%, 0%; DeltaNet 100%, 100%, 71%, 36%, 13%; Linear attention 97%, 82%, 76%, 26%, 0%" loading="lazy">

The Transformer recalls every value at every size on the chart. The three recurrent models hold up while there are few pairs and fall apart between 8 and 32: at 32 pairs DeltaNet recalls 36% of the values, linear attention 26% and Gated DeltaNet 2%, and at 64 the best of them, DeltaNet, recalls 13%.

Each head here works with keys of 32 numbers, so by the bound from earlier its state holds at most 32 pairs exactly, and at 16 pairs all three are already well short of full recall.

At 128 pairs even the Transformer didn't learn the task in 4,000 steps, which is why the chart stops at 64.

Gated DeltaNet, the newest of the three, comes last from 16 pairs on. That is less strange than it looks, because this test gives the gate nothing to do. Every pair has to be kept until its question comes, so forgetting can only lose information.

The authors of Gated DeltaNet, who trained models of 1.3 billion parameters on 100 billion tokens of real text, found that it "consistently surpasses existing models like Mamba2 and DeltaNet", in-context retrieval included<sup class="cite">[<a href="#ref-8" id="cite-12" data-ref="8" aria-label="Reference 8: Yang, S., Kautz, J., &amp; Hatamizadeh, A. (2025). Gated delta networks: improving Mamba2 with delta rule. ICLR 2025.">8</a>]</sup>. Four layers and 4,000 steps of a synthetic task are a different regime.

The gate is also the easiest part to get wrong. In my first version it started at about one half, and weight decay kept pulling it back there, so the state lost half of what it held at every token, and Gated DeltaNet and the hybrid stayed at chance level.

Starting the gate as the Gated DeltaNet and Mamba-2 code does, where a typical head keeps about 94% of its state from one token to the next, and leaving it out of weight decay, fixed it.

The two hybrids differ only in the order of their layers. With attention second, the hybrid recalls every value at every size, like the Transformer, though three of its four layers are Gated DeltaNet, which on its own recalls the least.

With attention on top, the same layers recall 26% of the values at 16 pairs and 12% at 32, at the better of two learning rates: a little more than Gated DeltaNet alone and less than DeltaNet.

I can't say for certain why. The attention layer on top has what it needs: the convolution hands each value its key, and the other hybrid shows that one attention layer among three Gated DeltaNet layers is enough. This one just doesn't learn the lookup in 4,000 steps.

A model of four layers is far from Qwen3-Next, where twelve attention layers are spread through 48 and most of them have many recurrent layers on both sides, so this says nothing about whether its layout is a good one.

It does show that the ratio alone doesn't describe a hybrid. With the same layers in a different order, one model recalled like a Transformer and the other like a recurrent network.

## Why one layer of attention is enough

Attention solves associative recall in one layer: the query for $$k_i$$ meets the stored key $$k_i$$, the softmax puts nearly all the weight on it, and out comes $$v_i$$, however many pairs came before.

Arora and colleagues proved that attention needs a model dimension independent of the sequence length for this, while the gated convolutions they studied need one that grows with it.

Measuring the gap in quality between the two kinds of models on real text, they found that "82% of the gap is explained by each model's ability to recall information that is previously mentioned in-context"<sup class="cite">[<a href="#ref-11" id="cite-13" data-ref="11" aria-label="Reference 11: Arora, S., et al. (2023). Zoology: measuring and improving recall in efficient language models. ICLR 2024.">11</a>]</sup>.

Copying gives the same picture. Jelassi and colleagues proved that two layers of attention can copy strings of exponential length, while models with a fixed state are "fundamentally limited by their fixed-size latent state"<sup class="cite">[<a href="#ref-12" id="cite-14" data-ref="12" aria-label="Reference 12: Jelassi, S., Brandfonbrener, D., Kakade, S. M., &amp; Malach, E. (2024). Repeat after me: Transformers are better than state space models at copying. ICML 2024.">12</a>]</sup>.

None of this requires attention everywhere. Much of predicting the next token is local, or a matter of keeping a running summary of what the text is about, and a gated recurrent layer does that well and cheaply.

Exact lookups into the distant past are rarer, and a layer that can make them serves the whole model: the recurrent layers around it carry the local context, the attention layer finds the match. In the library picture, most floors work from a notebook page, and every few floors someone is allowed into the archive.

The authors of Mamba-2 tried such hybrids and described the division of labour the same way: "attention layers act as a retrieval mechanism to quickly refer to previous tokens in the sequence instead of forcing the model to compress all the context to its memory". They found that "around 10% of the total number of layers being attention performs best"<sup class="cite">[<a href="#ref-7" id="cite-15" data-ref="7" aria-label="Reference 7: Dao, T., &amp; Gu, A. (2024). Transformers are SSMs: generalized models and efficient algorithms through structured state space duality. ICML 2024.">7</a>]</sup>.

A study at NVIDIA trained an 8-billion-parameter Transformer, pure Mamba and Mamba-2 models and a hybrid with 4 attention layers out of 56 on the same data.

The pure Mamba models began to fail at looking up a number in a phone book beyond about 500 tokens. The hybrid exceeded the Transformer on all 12 standard tasks the authors evaluated, by 2.65 points on average<sup class="cite">[<a href="#ref-13" id="cite-16" data-ref="13" aria-label="Reference 13: Waleffe, R., et al. (2024). An empirical study of Mamba-based language models. arXiv.">13</a>]</sup>.

## Hybrids in practice

The first large hybrid was Jamba from AI21, in March 2024: Mamba layers with one attention layer in every eight<sup class="cite">[<a href="#ref-14" id="cite-17" data-ref="14" aria-label="Reference 14: Lieber, O., et al. (2024). Jamba: a hybrid Transformer-Mamba language model. arXiv.">14</a>]</sup>. MiniMax, NVIDIA, Qwen and Moonshot followed, each with its own recurrent layer and its own ratio.

MiniMax-Text-01 put one softmax layer after every seven layers of lightning attention, a fast form of linear attention, after finding that lightning attention alone "demonstrates limited retrieval capabilities"<sup class="cite">[<a href="#ref-15" id="cite-18" data-ref="15" aria-label="Reference 15: MiniMax (2025). MiniMax-01: scaling foundation models with lightning attention. arXiv.">15</a>]</sup>. NVIDIA's Nemotron-H models keep "roughly 8% of the total number of layers" as attention, and the rest are Mamba-2 and MLP layers<sup class="cite">[<a href="#ref-16" id="cite-19" data-ref="16" aria-label="Reference 16: NVIDIA (2025). Nemotron-H: a family of accurate and efficient hybrid Mamba-Transformer models. arXiv.">16</a>]</sup>.

Qwen3-Next uses Gated DeltaNet in three layers out of four. According to the Qwen team, at this ratio "the model consistently outperforms any monolithic architecture"<sup class="cite">[<a href="#ref-17" id="cite-20" data-ref="17" aria-label="Reference 17: Qwen Team (2025). Qwen3-Next: towards ultimate training &amp; inference efficiency. Alibaba Cloud blog.">17</a>]</sup>.

Kimi Linear uses Kimi Delta Attention, the variant with a gate for every channel, also at 3:1. It leaves word order entirely to the recurrent layers: its attention layers use no positional encoding at all<sup class="cite">[<a href="#ref-9" id="cite-21" data-ref="9" aria-label="Reference 9: Kimi Team (2025). Kimi Linear: an expressive, efficient attention architecture. arXiv.">9</a>]</sup>.

Here is where each model keeps its full attention, read from the config files the models are published with:

<img class="tfs-figure tfs-light" src="/assets/posts/hybrid-attention/patterns-light.png" width="760" alt="The layers of real models, one mark per token-mixing layer, read from their config files. Jamba: 4 of 32 layers are full attention. MiniMax-Text-01: 10 of 80 layers are full attention. Nemotron-H-8B: 4 of 28 layers are full attention. Qwen3-Next-80B-A3B: 12 of 48 layers are full attention. Kimi Linear 48B-A3B: 7 of 27 layers are full attention. Qwen3.5-397B-A17B: 15 of 60 layers are full attention. Kimi K3: 24 of 93 layers are full attention. gpt-oss-20b: 12 of 24 layers are full attention, 12 use a sliding window. Gemma 4 31B: 10 of 60 layers are full attention, 50 use a sliding window. MiniMax-M2: 62 of 62 layers are full attention. MiniMax-M2, the successor of MiniMax-Text-01, uses full attention in all of its layers" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/hybrid-attention/patterns-dark.png" width="760" alt="The layers of real models, one mark per token-mixing layer, read from their config files. Jamba: 4 of 32 layers are full attention. MiniMax-Text-01: 10 of 80 layers are full attention. Nemotron-H-8B: 4 of 28 layers are full attention. Qwen3-Next-80B-A3B: 12 of 48 layers are full attention. Kimi Linear 48B-A3B: 7 of 27 layers are full attention. Qwen3.5-397B-A17B: 15 of 60 layers are full attention. Kimi K3: 24 of 93 layers are full attention. gpt-oss-20b: 12 of 24 layers are full attention, 12 use a sliding window. Gemma 4 31B: 10 of 60 layers are full attention, 50 use a sliding window. MiniMax-M2: 62 of 62 layers are full attention. MiniMax-M2, the successor of MiniMax-Text-01, uses full attention in all of its layers" loading="lazy">

Drawn to scale, the designs differ more than their headline ratios. Jamba and MiniMax-Text-01 keep one attention layer in eight, Nemotron-H-8B 4 of its 28 token-mixing layers, the Qwen and Kimi models one in four.

Kimi Linear's released model has 27 layers, so its 3:1 turns into 20 recurrent layers and 7 attention layers, the last one attention.

Kimi K3, 93 layers deep, adds an extra attention layer at the top, "ensuring that the final layer always performs global attention"<sup class="cite">[<a href="#ref-18" id="cite-22" data-ref="18" aria-label="Reference 18: Kimi Team (2026). Kimi K3: open frontier intelligence. arXiv.">18</a>]</sup>. All of them spread the attention layers evenly through the depth.

The second group shrinks the cache in another way. gpt-oss and Gemma use softmax attention in every layer, but most of their layers see only a sliding window of recent tokens: 128 in gpt-oss, where windowed and full layers alternate<sup class="cite">[<a href="#ref-19" id="cite-23" data-ref="19" aria-label="Reference 19: OpenAI (2025). gpt-oss-120b &amp; gpt-oss-20b model card. arXiv.">19</a>]</sup>, and 1,024 in Gemma 3, with five windowed layers for every full one<sup class="cite">[<a href="#ref-20" id="cite-24" data-ref="20" aria-label="Reference 20: Gemma Team (2025). Gemma 3 technical report. arXiv.">20</a>]</sup>.

Gemma 4 kept the ratio. A window is the opposite trade to a recurrent state: it keeps recent tokens exactly and forgets older ones completely, while a state keeps a blurred summary of everything.

The last row is MiniMax-M2, the successor of MiniMax-Text-01, with full attention in all 62 layers. The team explained why in a blog post.

Small hybrids had matched full attention on the leaderboards, but "the price paid became obvious at a larger scale: the model had clear deficits in complex, multi-hop reasoning tasks"<sup class="cite">[<a href="#ref-21" id="cite-25" data-ref="21" aria-label="Reference 21: MiniMax (2025). Why did MiniMax M2 end up as a full attention model? Hugging Face blog.">21</a>]</sup>.

The Qwen and Kimi hybrids keep more attention than MiniMax-Text-01 did, one layer in four rather than one in eight, and use the delta rule, which lightning attention lacks. Still, the episode is a fair warning that parity on small models and leaderboards doesn't settle the question.

What the hybrids save is easiest to see in memory. For each model, the size of its cache follows from the config: the number of attention layers, key-value heads and head dimension.

<img class="tfs-figure tfs-light" src="/assets/posts/hybrid-attention/kv-memory-light.png" width="760" alt="Keys and values one sequence of 128K tokens keeps in bf16, derived from each model&#x27;s config: Qwen3-32B 32 GiB, MiniMax-M2 31 GiB, Qwen3-8B 18 GiB, MiniMax-Text-01 5 GiB, gpt-oss-20b 3 GiB, Qwen3-Next 3 GiB, Jamba 2 GiB, Kimi Linear 0.98 GiB, Nemotron 3 Nano 0.75 GiB. Qwen3-32B, MiniMax-M2 and Qwen3-8B use full attention in every layer, gpt-oss-20b a sliding window in half of them, the others recurrent layers with full attention in a few" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/hybrid-attention/kv-memory-dark.png" width="760" alt="Keys and values one sequence of 128K tokens keeps in bf16, derived from each model&#x27;s config: Qwen3-32B 32 GiB, MiniMax-M2 31 GiB, Qwen3-8B 18 GiB, MiniMax-Text-01 5 GiB, gpt-oss-20b 3 GiB, Qwen3-Next 3 GiB, Jamba 2 GiB, Kimi Linear 0.98 GiB, Nemotron 3 Nano 0.75 GiB. Qwen3-32B, MiniMax-M2 and Qwen3-8B use full attention in every layer, gpt-oss-20b a sliding window in half of them, the others recurrent layers with full attention in a few" loading="lazy">

For one sequence of 128K tokens, Qwen3-Next keeps 3 GiB of keys and values, against 32 GiB for Qwen3-32B. Not all of the factor of ten comes from the hybrid. Each attention layer of Qwen3-Next keeps two key-value heads of 256 dimensions, where Qwen3-32B keeps eight of 128, which halves the cache per layer.

The rest, a factor of 5.3, is 12 attention layers instead of 64. The recurrent layers add a state of 36 MiB that stays the same at any length.

Kimi Linear starts from multi-head latent attention, introduced in DeepSeek-V2<sup class="cite">[<a href="#ref-22" id="cite-26" data-ref="22" aria-label="Reference 22: DeepSeek-AI (2024). DeepSeek-V2: a strong, economical, and efficient mixture-of-experts language model. arXiv.">22</a>]</sup>, which already caches one compressed vector of 576 numbers per token and layer in place of keys and values for every head, and keeps it in 7 layers of 27.

That is 26% of the cache of an all-MLA model of the same depth, in line with the paper's "reducing KV cache usage by up to 75%". MiniMax-M2, at 31 GiB, keeps six times more than its predecessor.

At long context, the cache rather than the weights decides how many conversations fit on a GPU at once, and decoding speed follows from that.

The Kimi Linear authors measured decoding with a million tokens of context and put the result down to memory: the smaller cache "enables larger batches, yielding a 6.3× faster TPOT (1.84ms vs. 11.48ms) than MLA at 1M tokens", TPOT being the time per output token.

## What it costs on a GPU

The recurrent layer pays off only where the cache grows large, and a single layer on one GPU shows where that starts.

I timed one layer with 16 heads of 128 dimensions, about the size of a Gated DeltaNet layer in Qwen3-Next, in two situations: reading a prompt of $$n$$ tokens at once, and writing one new token after $$n$$ tokens.

Softmax attention runs through PyTorch's fused attention kernel in half precision. Gated DeltaNet runs my plain-PyTorch chunked version in single precision, a Python loop over chunks of 64. The comparison is tilted against the recurrent layer on purpose: if it wins anyway, it wins on arithmetic, not on engineering.

<img class="tfs-figure tfs-light" src="/assets/posts/hybrid-attention/prefill-light.png" width="760" alt="Time for one layer with 16 heads of 128 to read a prompt, on a Tesla T4, on logarithmic axes. Softmax attention takes 0.18 ms for 512 tokens and 1,579 ms for 65,536; the chunked Gated DeltaNet in plain PyTorch takes 3.0 ms and 377 ms. Attention is faster up to 8K tokens, the recurrent layer from 16K on" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/hybrid-attention/prefill-dark.png" width="760" alt="Time for one layer with 16 heads of 128 to read a prompt, on a Tesla T4, on logarithmic axes. Softmax attention takes 0.18 ms for 512 tokens and 1,579 ms for 65,536; the chunked Gated DeltaNet in plain PyTorch takes 3.0 ms and 377 ms. Attention is faster up to 8K tokens, the recurrent layer from 16K on" loading="lazy">

<img class="tfs-figure tfs-light" src="/assets/posts/hybrid-attention/decode-light.png" width="760" alt="Writing one token after a context of 512 to 65,536 tokens. Left: softmax attention takes 0.08 ms to 3.7 ms, growing with the context; Gated DeltaNet about 0.2 ms at every length. Right: attention reads 4 to 512 MiB of keys and values, the recurrent layer a state of 1 MiB" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/hybrid-attention/decode-dark.png" width="760" alt="Writing one token after a context of 512 to 65,536 tokens. Left: softmax attention takes 0.08 ms to 3.7 ms, growing with the context; Gated DeltaNet about 0.2 ms at every length. Right: attention reads 4 to 512 MiB of keys and values, the recurrent layer a state of 1 MiB" loading="lazy">

Reading a prompt, softmax attention is faster up to 8K tokens, which is no surprise: its kernel is one fused program, and mine is a loop. The slopes differ, though.

From 512 to 65,536 tokens, a sequence 128 times longer, the attention layer gets 8,600 times slower and the recurrent one 124 times slower. The curves cross between 8K and 16K tokens, and at 65K the recurrent layer is 4.2 times faster.

Writing a token is where the cache costs most. Attention reads the keys and values of every earlier token, 512 MiB at 65K tokens in this one layer, and its time grows in proportion, from 0.08 ms to 3.7 ms. The recurrent layer updates a state of 1 MiB and takes 0.2 ms at any length; at that size the time is mostly the overhead of launching a few small GPU kernels.

From about 2K tokens on, the recurrent step is faster, and at 65K it is 16 times faster.

One layer and one sequence understate the difference in a real server. There, the caches of all the conversations in a batch have to fit in memory at once, so a model whose cache is a quarter or an eighth of the size can serve proportionally more conversations at the same length.

Kimi Linear's 6.3 times, quoted above, comes from exactly that. The attention layers of a hybrid still pay the full price; three quarters of its layers stop paying it.

## When a hybrid is the right choice

A hybrid pays off when contexts are long and there are many of them: agents that carry long transcripts of tool calls, reasoning models that write tens of thousands of tokens, questions over long documents, a server holding hundreds of such conversations at once.

There the cache is most of the memory and reading it is most of the decoding time, and cutting it by a factor of four to ten changes how many users one GPU can serve.

At short contexts the gain is smaller than the headlines suggest. According to Qwen, the decoding throughput of Qwen3-Next is nearly 4 times that of Qwen3-32B at 4K tokens and over 10 times beyond 32K<sup class="cite">[<a href="#ref-17" id="cite-27" data-ref="17" aria-label="Reference 17: Qwen Team (2025). Qwen3-Next: towards ultimate training &amp; inference efficiency. Alibaba Cloud blog.">17</a>]</sup>.

But Qwen3-Next activates 3 billion parameters per token and Qwen3-32B all 32 billion, so the comparison mixes two savings, the mixture of experts and the hybrid, and the blog doesn't separate them. The hybrid's share is the part that grows with the context.

The costs are of three kinds. The first is quality on the tasks attention is best at, precise retrieval and multi-hop reasoning over long inputs: the deficits that sent MiniMax back to full attention.

The second is engineering. Recurrent layers need their own kernels, and serving tricks built around a cache of separate tokens get harder when part of the model keeps a state that has absorbed all of them.

Reusing the cache of a shared prompt needs copies of the state saved at chosen positions, and rolling back a rejected speculative token needs the state from a step earlier.

The third is that the field keeps moving. Qwen3.8-Flash-Next replaced the full attention layers of its hybrid with sparse attention<sup class="cite">[<a href="#ref-23" id="cite-28" data-ref="23" aria-label="Reference 23: Qwen Team (2026). Qwen3.8-Flash-Next model card. Hugging Face.">23</a>]</sup>, and DeepSeek-V4 combines two kinds of compressed attention instead of adding recurrent layers<sup class="cite">[<a href="#ref-24" id="cite-29" data-ref="24" aria-label="Reference 24: DeepSeek-AI (2026). DeepSeek-V4: towards highly efficient million-token context intelligence. arXiv.">24</a>]</sup>. Both directions go after the same cache.

A hybrid's way is to give most layers a memory of fixed size and to keep a few that can still read every token.

## References

1. {:#ref-1} Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). [Transformers are RNNs: fast autoregressive Transformers with linear attention](https://arxiv.org/abs/2006.16236). *ICML 2020*. <a class="ref-back" href="#cite-1" aria-label="Back to the text">&#8617;</a>
2. {:#ref-2} Arora, S., et al. (2024). [Simple linear attention language models balance the recall-throughput tradeoff](https://arxiv.org/abs/2402.18668). *ICML 2024*. <a class="ref-back" href="#cite-2" aria-label="Back to the text">&#8617;</a>
3. {:#ref-3} Widrow, B., & Hoff, M. E. (1960). [Adaptive switching circuits](https://isl.stanford.edu/~widrow/papers/c1960adaptiveswitching.pdf). *IRE WESCON Convention Record*, 4, 96–104. <a class="ref-back" href="#cite-3" aria-label="Back to the text">&#8617;</a>
4. {:#ref-4} Schlag, I., Irie, K., & Schmidhuber, J. (2021). [Linear Transformers are secretly fast weight programmers](https://arxiv.org/abs/2102.11174). *ICML 2021*. <a class="ref-back" href="#cite-4" aria-label="Back to the text">&#8617;</a>
5. {:#ref-5} Sun, Y., et al. (2023). [Retentive network: a successor to Transformer for large language models](https://arxiv.org/abs/2307.08621). arXiv. <a class="ref-back" href="#cite-5" aria-label="Back to the text">&#8617;</a>
6. {:#ref-6} Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024). [Gated linear attention Transformers with hardware-efficient training](https://arxiv.org/abs/2312.06635). *ICML 2024*. <a class="ref-back" href="#cite-6" aria-label="Back to the text">&#8617;</a>
7. {:#ref-7} Dao, T., & Gu, A. (2024). [Transformers are SSMs: generalized models and efficient algorithms through structured state space duality](https://arxiv.org/abs/2405.21060). *ICML 2024*. <a class="ref-back" href="#cite-7" aria-label="Back to the text">&#8617;</a>
8. {:#ref-8} Yang, S., Kautz, J., & Hatamizadeh, A. (2025). [Gated delta networks: improving Mamba2 with delta rule](https://arxiv.org/abs/2412.06464). *ICLR 2025*. <a class="ref-back" href="#cite-8" aria-label="Back to the text">&#8617;</a>
9. {:#ref-9} Kimi Team (2025). [Kimi Linear: an expressive, efficient attention architecture](https://arxiv.org/abs/2510.26692). arXiv. <a class="ref-back" href="#cite-9" aria-label="Back to the text">&#8617;</a>
10. {:#ref-10} Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024). [Parallelizing linear Transformers with the delta rule over sequence length](https://arxiv.org/abs/2406.06484). *NeurIPS 2024*. <a class="ref-back" href="#cite-10" aria-label="Back to the text">&#8617;</a>
11. {:#ref-11} Arora, S., et al. (2023). [Zoology: measuring and improving recall in efficient language models](https://arxiv.org/abs/2312.04927). *ICLR 2024*. <a class="ref-back" href="#cite-11" aria-label="Back to the text">&#8617;</a>
12. {:#ref-12} Jelassi, S., Brandfonbrener, D., Kakade, S. M., & Malach, E. (2024). [Repeat after me: Transformers are better than state space models at copying](https://arxiv.org/abs/2402.01032). *ICML 2024*. <a class="ref-back" href="#cite-14" aria-label="Back to the text">&#8617;</a>
13. {:#ref-13} Waleffe, R., et al. (2024). [An empirical study of Mamba-based language models](https://arxiv.org/abs/2406.07887). arXiv. <a class="ref-back" href="#cite-16" aria-label="Back to the text">&#8617;</a>
14. {:#ref-14} Lieber, O., et al. (2024). [Jamba: a hybrid Transformer-Mamba language model](https://arxiv.org/abs/2403.19887). arXiv. <a class="ref-back" href="#cite-17" aria-label="Back to the text">&#8617;</a>
15. {:#ref-15} MiniMax (2025). [MiniMax-01: scaling foundation models with lightning attention](https://arxiv.org/abs/2501.08313). arXiv. <a class="ref-back" href="#cite-18" aria-label="Back to the text">&#8617;</a>
16. {:#ref-16} NVIDIA (2025). [Nemotron-H: a family of accurate and efficient hybrid Mamba-Transformer models](https://arxiv.org/abs/2504.03624). arXiv. <a class="ref-back" href="#cite-19" aria-label="Back to the text">&#8617;</a>
17. {:#ref-17} Qwen Team (2025). [Qwen3-Next: towards ultimate training & inference efficiency](https://www.alibabacloud.com/blog/602580). Alibaba Cloud blog. <a class="ref-back" href="#cite-20" aria-label="Back to the text">&#8617;</a>
18. {:#ref-18} Kimi Team (2026). [Kimi K3: open frontier intelligence](https://arxiv.org/abs/2607.24653). arXiv. <a class="ref-back" href="#cite-22" aria-label="Back to the text">&#8617;</a>
19. {:#ref-19} OpenAI (2025). [gpt-oss-120b & gpt-oss-20b model card](https://arxiv.org/abs/2508.10925). arXiv. <a class="ref-back" href="#cite-23" aria-label="Back to the text">&#8617;</a>
20. {:#ref-20} Gemma Team (2025). [Gemma 3 technical report](https://arxiv.org/abs/2503.19786). arXiv. <a class="ref-back" href="#cite-24" aria-label="Back to the text">&#8617;</a>
21. {:#ref-21} MiniMax (2025). [Why did MiniMax M2 end up as a full attention model?](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model) Hugging Face blog. <a class="ref-back" href="#cite-25" aria-label="Back to the text">&#8617;</a>
22. {:#ref-22} DeepSeek-AI (2024). [DeepSeek-V2: a strong, economical, and efficient mixture-of-experts language model](https://arxiv.org/abs/2405.04434). arXiv. <a class="ref-back" href="#cite-26" aria-label="Back to the text">&#8617;</a>
23. {:#ref-23} Qwen Team (2026). [Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Hugging Face. <a class="ref-back" href="#cite-28" aria-label="Back to the text">&#8617;</a>
24. {:#ref-24} DeepSeek-AI (2026). [DeepSeek-V4: towards highly efficient million-token context intelligence](https://arxiv.org/abs/2606.19348). arXiv. <a class="ref-back" href="#cite-29" aria-label="Back to the text">&#8617;</a>
{: .tfs-refs}

{% endraw %}
