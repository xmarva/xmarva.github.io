---
layout: page
permalink: /teaching/
title: teaching
description: Welcome to my teaching materials collection. These page and course are currently under development.
nav: true
nav_order: 6
---

<div class="course-container" markdown="1">
  <div class="course-card">
    <h2>Transformer Architectures</h2>
    <p><a href="https://github.com/xmarva/transformer-architectures"><i class="fab fa-github"></i> GitHub Repository</a></p>
    <p class="course-description">Exploration of transformer architectures through hands-on implementation and experimentation.</p>
<div class="course-section">
  <h3>Part 1: Basic Components from Prototyping to Production</h3>
  <table class="table">
    <tbody>
      <tr>
        <td>Lecture 1. Tokenization: BPE, HuggingFace Tokenizers</td>
      </tr>
      <tr>
        <td>Lecture 2. Transformer Architecture: A clean-code implementation (no import transformers allowed).</td>
      </tr>
      <tr>
        <td>Lecture 3. Transformer Training: Train a model on a simple translation task</td>
      </tr>
      <tr>
        <td></td>
      </tr>
      <tr>
        <td></td>
      </tr>
    </tbody>
  </table>
</div>

<div class="course-section">
  <h3>Part 2: Architecture Enhancements and Pretrained Models</h3>
  <table class="table">
    <tbody>
      <tr>
        <td>Lecture 1. BERT: Bidirectional Encoder Representations from Transformers</td>
      </tr>
      <tr>
        <td>Lecture 2. GPT: Generative Pre-trained Transformers</td>
      </tr>
      <tr>
        <td></td>
      </tr>
      <tr>
        <td></td>
      </tr>
      <tr>
        <td></td>
      </tr>
    </tbody>
  </table>
</div>

<div class="course-section">
  <h3>Part 3: Multimodality and Optimization</h3>
  <table class="table">
    <tbody>
      <tr>
        <td>Lecture 1. Vision Transformers and Multimodal Models</td>
      </tr>
      <tr>
        <td>Lecture 2. Optimization and Efficiency: PEFT, LoRA, Quantization</td>
      </tr>
      <tr>
        <td></td>
      </tr>
      <tr>
        <td></td>
      </tr>
      <tr>
        <td></td>
      </tr>
    </tbody>
  </table>
</div>
  </div>
</div>
<!-- Add this CSS to your _sass or assets/css file -->
<style>
.course-container {
  margin-bottom: 3rem;
}

.course-card {
  background-color: var(--global-bg-color);
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  padding: 2rem;
  margin-bottom: 2rem;
  transition: box-shadow 0.3s ease;
  color: var(--global-text-color);
}

.course-card:hover {
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
}

.course-card h2 {
  margin-top: 0;
  border-bottom: 1px solid var(--global-divider-color);
  padding-bottom: 0.75rem;
  color: var(--global-theme-color);
}

.course-description {
  color: var(--global-text-color-light);
  font-size: 1.1rem;
  margin-bottom: 1.5rem;
  line-height: 1.6;
}

.course-section {
  margin-top: 1.5rem;
  background-color: var(--global-code-bg-color);
  border-radius: 6px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}

.course-section h3 {
  margin-top: 0;
  font-size: 1.3rem;
  color: var(--global-theme-color);
}

.table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 1rem;
}

.table td {
  padding: 0.85rem 1rem;
  border-bottom: 1px solid var(--global-divider-color);
  font-size: 1rem;
}

.table tbody tr:last-child td {
  border-bottom: none;
}

.table tbody tr:hover {
  background-color: rgba(0, 0, 0, 0.02);
}
</style>