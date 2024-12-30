---
layout: post
title: Low Rank Adaptation algorithm for LLMs
date: 2024-12-30 15:09:00
description: Let's look at the linear algebra behind the popular algorithm
tags: nlp, llm, lora
categories: 
featured: true
toc:
  sidebar: left
---

Large language models are amazing, but for some practical tasks, they require fine-tuning. However, because of their size and the high cost of training on GPUs, traditional fine-tuning methods have become impractical.

PEFT makes it faster and cheaper to adapt these models. The main idea behind these approaches is to train only a small subset of the model's parameters while keeping the rest frozen.

In this article, I’ll dive into how Low-Rank Adaptation (LoRA) works.

Let’s start with some key linear algebra terms and properties that are essential to understanding the algorithm.

## Matrix

A matrix $$A$$ is a rectangular array of numbers arranged in rows and columns. The element of a matrix at the intersection of the $$i$$-th row and the $$j$$-th column is denoted as $$a_{ij}$$ or $$A_{ij}$$.

In the context of neural networks, a weight matrix is a matrix whose elements represent the weights between neurons in adjacent layers. Each number in the matrix indicates the strength of the connection between two neurons: the larger the number, the stronger the connection.

## Rank of a Matrix

The rank of a matrix $$A$$ (denoted as $$\text{rank}(A)$$ or $$r(A)$$) is the maximum number of linearly independent rows or columns in the matrix. Roughly speaking, it reflects how much information the matrix actually contains. If the rank is low, it means the matrix is compressed, and many rows/columns are linear combinations of others.

The rank of a matrix can range from 0 (for a zero matrix) to the smaller of the number of rows and columns in the matrix. If $$A$$ is a matrix of size $$m \times n$$, then $$0 \le \text{rank}(A) \le \min(m, n)$$.

A matrix $$A$$ of size $$m \times n$$ is called a full-rank matrix if its rank equals the smaller of its dimensions: $$\text{rank}(A) = \min(m, n)$$.

A matrix $$A$$ of size $$m \times n$$ is called a rank-deficient matrix if its rank is less than the smaller of its dimensions: $$\text{rank}(A) < \min(m, n)$$.

## Rank of the Product and Decomposition

The rank of the product of two matrices cannot exceed the rank of either matrix: if $$A$$ is a matrix of size $$m \times n$$ and $$B$$ is a matrix of size $$n \times p$$, then $$\text{rank}(AB) \le \min(\text{rank}(A), \text{rank}(B))$$.

Rank Decomposition is the process of breaking down complex information into simpler components.

Any matrix $$A$$ with rank $$r$$ can be represented as the product of two smaller matrices: $$A = U V^T$$, where $$U$$ is a matrix of size $$m \times r$$ and $$V$$ is a matrix of size $$n \times r$$. This decomposition is also known as **low-rank factorization**.

There are various methods to compute such decompositions, such as Singular Value Decomposition (SVD).

## Low-Rank Adaptation (LoRA) Hypothesis

Previous research has shown that overparameterized large models often have low intrinsic dimensionality. The core idea of LoRA is that weight changes during model adaptation also have low intrinsic rank/dimensionality. Specifically, if $$W_{n \times k}$$ represents the weights of a layer and $$\Delta W_{n \times k}$$ represents the weight changes during adaptation, the authors hypothesize that $$\Delta W_{n \times k}$$ is a low-rank matrix.  

*So, you don’t need to change all the weights of a layer to adapt the model; it’s enough to tweak only a small subset.*

### Why does this make sense?

LLMs are trained to capture general representations of their domain.

These models learn a rich set of features, which allows them to solve various tasks with decent accuracy even without fine-tuning. 

However, when adapting such a model to a specific task or dataset, it’s often sufficient to refine or retrain only a few features. This means that the update matrix ($$\Delta W$$) can have a low rank.  

## Method

LoRA constrains the rank of the update matrix $$\Delta W$$ using rank decomposition. It represents $$\Delta W_{n \times k}$$ as the product of two low-rank matrices, $$B_{n \times r}$$ and $$A_{r \times k}$$, where $$r \ll \min(n, k)$$. This means that the forward pass of the layer, originally $$Wx$$, is modified to $$Wx + BAx$$. 

- The matrix $$A$$ is randomly initialized with Gaussian values.
- The matrix $$B$$ is initialized to 0, so $$BA = 0$$ at the start of training.
- The $$BA$$ update is further scaled by a factor of $$\alpha / r$$.

### Practical Benefits:

1. **Reduced training time and memory usage:** Using this method, only $$r(n + k)$$ parameters need to be tuned during model adaptation. Since $$r \ll \min(n, k)$$, this is significantly fewer than the $$nk$$ parameters that would otherwise need to be updated. This drastically reduces the time and memory required for fine-tuning.
2. **No additional inference overhead:** In production, you can precompute $$W' = W + BA$$ and use the result as usual, ensuring no added latency during inference.
3. **Easier task switching:** Replacing only the LoRA weights instead of all parameters allows for cheaper and faster task switching. You can create multiple fine-tuned models and quickly switch between them.