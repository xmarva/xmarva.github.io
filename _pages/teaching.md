---
layout: page
permalink: /teaching/
title: teaching
description: Courses I teach, as notebooks that run on a free GPU and posts to read here
seo_description: "Transformers: from scratch to production, a free course: build an English to Russian translator with a Transformer written from scratch."
nav: true
nav_order: 6
hide_header: true
---

<style>
  /* Same system as the portfolio: the site's red for the course's one action and its progress, cards that follow
     the theme, the lesson covers (dark, red) as the only pictures. */
  .tc { --tc-border: var(--global-divider-color); --tc-muted: var(--global-text-color-light); }
  .tc-course { margin: 2.6rem 0 3.5rem; }

  .tc-hero {
    display: grid;
    grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr);
    gap: 2rem;
    align-items: center;
    padding: 2.25rem 2.25rem 2.25rem 2.5rem;
    border: 1px solid var(--tc-border);
    border-radius: 16px;
    background: var(--global-card-bg-color);
    overflow: hidden;
  }
  .tc-hero-text { display: flex; flex-direction: column; align-items: flex-start; gap: 1rem; min-width: 0; }

  .tc-new {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    max-width: 100%;
    padding: 0.22rem 0.8rem 0.22rem 0.25rem;
    border: 1px solid var(--tc-border);
    border-radius: 999px;
    font-size: 0.85rem;
    color: var(--global-text-color) !important;
    text-decoration: none !important;
    transition: border-color 0.2s ease;
  }
  .tc-new:hover { border-color: color-mix(in srgb, var(--global-theme-color) 45%, var(--tc-border)); }
  .tc-new-tag {
    flex: none;
    padding: 0.08rem 0.5rem;
    border-radius: 999px;
    background: color-mix(in srgb, var(--global-theme-color) 14%, transparent);
    color: var(--global-theme-color);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .tc-new-text { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  .tc-title { margin: 0; font-size: 2.1rem; font-weight: 500; line-height: 1.18; color: var(--global-text-color); }
  .tc-summary { margin: 0; font-size: 1.05rem; line-height: 1.6; color: var(--tc-muted); }
  .tc-facts { margin: 0; font-size: 0.85rem; color: var(--tc-muted); }

  .tc-progress { display: flex; align-items: center; gap: 0.85rem; }
  .tc-bar { display: flex; gap: 4px; width: 200px; }
  .tc-bar span { flex: 1; height: 6px; border-radius: 3px; background: var(--tc-border); }
  .tc-bar span.on { background: var(--global-theme-color); }
  .tc-progress-label { font-size: 0.85rem; color: var(--tc-muted); white-space: nowrap; }

  .tc-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem 1.25rem; margin-top: 0.35rem; }
  .tc-start {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.5rem 1.15rem;
    border-radius: 999px;
    background: var(--global-button-color);
    color: #fff !important;
    font-size: 0.95rem;
    font-weight: 500;
    text-decoration: none !important;
    transition: filter 0.2s ease;
  }
  .tc-start:hover { filter: brightness(1.1); }
  .tc-github { font-size: 0.95rem; color: var(--global-text-color) !important; text-decoration: none !important; }
  .tc-github:hover { color: var(--global-theme-color) !important; }
  .tc-github i { margin-right: 0.3rem; }

  /* the latest lessons' covers, fanned out, the newest on top */
  .tc-fan { position: relative; display: block; aspect-ratio: 4 / 3; }
  .tc-fan-card {
    position: absolute;
    right: 0;
    top: 50%;
    width: 82%;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.15), 0 14px 32px rgba(0, 0, 0, 0.22);
    transform: translate(calc(var(--i) * -10%), calc(-50% - var(--i) * 10%)) rotate(calc(var(--i) * -4deg));
    z-index: calc(3 - var(--i));
    filter: brightness(calc(1 - var(--i) * 0.12));
    transition: transform 0.35s ease;
  }
  .tc-fan-card figure, .tc-fan-card picture { display: block; margin: 0; }
  .tc-fan-card img { display: block; width: 100%; height: auto; }
  .tc-fan:hover .tc-fan-card {
    transform: translate(calc(var(--i) * -13%), calc(-50% - var(--i) * 12%)) rotate(calc(var(--i) * -6deg));
  }

  .tc-section-head { margin: 3rem 0 1rem; }
  .tc-section-head::before {
    content: "";
    display: block;
    width: 28px;
    height: 3px;
    margin-bottom: 0.75rem;
    background: var(--global-theme-color);
  }
  .tc-section-head h3 { margin: 0; font-size: 1.45rem; font-weight: 500; color: var(--global-text-color); }

  .tc-lessons { list-style: none; margin: 0; padding: 0; border-top: 1px solid var(--tc-border); }
  .tc-lesson {
    display: grid;
    grid-template-columns: 2.75rem minmax(0, 1fr) auto;
    column-gap: 1rem;
    align-items: center;
    padding: 1.05rem 0.25rem;
    border-bottom: 1px solid var(--tc-border);
  }
  .tc-num {
    align-self: start;
    padding-top: 0.15rem;
    font-family: var(--global-code-font-family, monospace);
    font-size: 0.95rem;
    color: var(--global-theme-color);
  }
  .tc-lesson-title { margin: 0; font-size: 1.1rem; font-weight: 500; line-height: 1.4; }
  .tc-lesson-title a { color: var(--global-text-color); text-decoration: none; }
  .tc-lesson-title a:hover { color: var(--global-theme-color); }
  .tc-topics { margin: 0.2rem 0 0; font-size: 0.9rem; line-height: 1.5; color: var(--tc-muted); }
  .tc-links { display: flex; gap: 0.5rem; }
  .tc-link {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.22rem 0.7rem;
    border: 1px solid var(--tc-border);
    border-radius: 999px;
    font-size: 0.8rem;
    color: var(--global-text-color) !important;
    text-decoration: none !important;
    transition: border-color 0.2s ease;
  }
  .tc-link:hover { border-color: var(--dot); }
  .tc-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--dot); }
  .tc-kaggle { --dot: #20beff; }
  .tc-colab { --dot: #f9ab00; }
  .tc-lesson.is-soon .tc-num,
  .tc-lesson.is-soon .tc-lesson-title { color: var(--tc-muted); }
  .tc-soon { font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--tc-muted); }

  @media (max-width: 860px) {
    .tc-hero { grid-template-columns: minmax(0, 1fr); gap: 1.75rem; padding: 1.5rem; }
    .tc-fan { order: -1; aspect-ratio: auto; }
    .tc-fan-card { position: relative; top: auto; width: 100%; transform: none; filter: none; }
    .tc-fan-card:not(:first-child) { display: none; }
    .tc-fan:hover .tc-fan-card { transform: none; }
    .tc-title { font-size: 1.7rem; }
  }
  @media (max-width: 600px) {
    .tc-lesson { grid-template-columns: 2.1rem minmax(0, 1fr); row-gap: 0.65rem; }
    .tc-links { grid-column: 2; }
  }
</style>

<div class="tc">
{% for course in site.data.courses %}
{% if course.show %}
{% assign published = course.lessons | where_exp: "lesson", "lesson.url" %}
{% assign total = course.lessons | size %}
{% assign done = published | size %}
{% assign recent = published | reverse %}
{% assign latest = recent | first %}
{% assign first_lesson = published | first %}
<section class="tc-course" aria-labelledby="tc-{{ course.id }}">
  <div class="tc-hero">
    <div class="tc-hero-text">
      {% if latest %}
      <a class="tc-new" href="{{ latest.url | relative_url }}"><span class="tc-new-tag">New</span><span class="tc-new-text">Lesson {{ done }}: {{ latest.short | default: latest.title }}</span><span aria-hidden="true">→</span></a>
      {% endif %}
      <h2 class="tc-title" id="tc-{{ course.id }}">{{ course.title }}</h2>
      <p class="tc-summary">{{ course.summary }}</p>
      <p class="tc-facts">{{ course.facts | join: " · " }}</p>
      <div class="tc-progress">
        <span class="tc-bar" aria-hidden="true">{% for lesson in course.lessons %}<span{% if lesson.url %} class="on"{% endif %}></span>{% endfor %}</span>
        <span class="tc-progress-label">{{ done }} of {{ total }} lessons out</span>
      </div>
      <div class="tc-actions">
        {% if first_lesson %}<a class="tc-start" href="{{ first_lesson.url | relative_url }}">Start with lesson 1 <span aria-hidden="true">→</span></a>{% endif %}
        <a class="tc-github" href="{{ course.github }}" target="_blank" rel="noopener noreferrer"><i class="fab fa-github" aria-hidden="true"></i>GitHub</a>
      </div>
    </div>
    {% if latest %}
    <a class="tc-fan" href="{{ latest.url | relative_url }}" tabindex="-1" aria-hidden="true">
      {% for lesson in recent limit: 3 %}
      <span class="tc-fan-card" style="--i: {{ forloop.index0 }}">{% include figure.liquid path=lesson.cover alt="" sizes="(min-width: 860px) 420px, 90vw" loading="eager" %}</span>
      {% endfor %}
    </a>
    {% endif %}
  </div>

  <div class="tc-section-head"><h3>Lessons</h3></div>
  <ol class="tc-lessons">
    {% for lesson in course.lessons %}
    <li class="tc-lesson{% unless lesson.url %} is-soon{% endunless %}">
      <span class="tc-num">{{ forloop.index | prepend: "0" | slice: -2, 2 }}</span>
      <div>
        <h4 class="tc-lesson-title">{% if lesson.url %}<a href="{{ lesson.url | relative_url }}">{{ lesson.title }}</a>{% else %}{{ lesson.title }}{% endif %}</h4>
        <p class="tc-topics">{{ lesson.topics }}</p>
      </div>
      <div class="tc-links">
        {% if lesson.url %}
        <a class="tc-link tc-kaggle" href="{{ lesson.kaggle }}" target="_blank" rel="noopener noreferrer" aria-label="Lesson {{ forloop.index }} on Kaggle"><span class="tc-dot" aria-hidden="true"></span>Kaggle</a>
        <a class="tc-link tc-colab" href="{{ lesson.colab }}" target="_blank" rel="noopener noreferrer" aria-label="Lesson {{ forloop.index }} in Colab"><span class="tc-dot" aria-hidden="true"></span>Colab</a>
        {% else %}
        <span class="tc-soon">soon</span>
        {% endif %}
      </div>
    </li>
    {% endfor %}
  </ol>
</section>
{% endif %}
{% endfor %}
</div>

{% comment %}
Archived course, hidden from the page. Its posts stay published in the blog; to show it again, remove this
comment.
{% endcomment %}
{% comment %}

<div class="course-container" markdown="1">
  <div class="course-card">
    <div class="course-header">
      <div class="header-left">
        <h3>Transformer Architectures</h3>
      </div>
    </div>
    <div class="course-content">
      <p class="course-description">End-to-end implementation of transformer architectures with production-grade practices.</p>

      <!-- Part 1: Core Implementation -->
      <div class="course-section">
        <div class="section-header">
          <div class="header-left">
            <h3>Part 1: Core Implementation</h3>
          </div>
          <a href="https://github.com/xmarva/transformer-architectures" class="github-link"><i class="fab fa-github"></i> GitHub</a>
        </div>
        <div class="section-content">
          <table class="table">
            <thead>
              <tr>
                <th>Notebook</th>
                <th>English</th>
                <th>Russian</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><a href="{{ '/blog/2025/tokenization/' | relative_url }}">1. Understanding Tokenization: Byte-Pair Encoding</a></td>
                <td>
                  <a href="https://www.kaggle.com/code/qmarva/1-bpe-tokenization-algorithm-eng?scriptVersionId=231677033" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                  <a href="https://colab.research.google.com/drive/1lmfuMdC8v-lXL_MuyC0uBewdLLCTQzCO?usp=sharing" class="badge-link">
                    <img src="https://img.shields.io/badge/Colab-F9AB00?style=plastic&logo=google-colab&logoColor=white" alt="Colab">
                  </a>
                </td>
                <td>
                  <a href="https://www.kaggle.com/code/qmarva/bpe-tokenization" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                  <a href="https://colab.research.google.com/drive/1nzhgsD2-eyrnPr28wtKZTtf7YMJGiQbe?usp=sharing" class="badge-link">
                    <img src="https://img.shields.io/badge/Colab-F9AB00?style=plastic&logo=google-colab&logoColor=white" alt="Colab">
                  </a>
                </td>
              </tr>
              <tr>
                <td><a href="{{ '/blog/2025/building-a-transformer/' | relative_url }}">2. Implementing Transformer Architecture</a></td>
                <td>
                  <a href="https://www.kaggle.com/code/qmarva/implementing-transformer-en" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                  <a href="https://colab.research.google.com/drive/1m34XYFZZTt-jbHo2OXlUfxR33zvFbXJZ?usp=sharing" class="badge-link">
                    <img src="https://img.shields.io/badge/Colab-F9AB00?style=plastic&logo=google-colab&logoColor=white" alt="Colab">
                  </a>
                </td>
                <td>
                  <a href="https://www.kaggle.com/code/qmarva/building-transformer" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Colab-F9AB00?style=plastic&logo=google-colab&logoColor=white" alt="Colab">
                  </a>
                </td>
              </tr>
              <tr>
                <td>3. Improving Architecture with SoTA Techniques </td>
                <td>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Colab-F9AB00?style=plastic&logo=google-colab&logoColor=white" alt="Colab">
                  </a>
                </td>
                <td>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Colab-F9AB00?style=plastic&logo=google-colab&logoColor=white" alt="Colab">
                  </a>
                </td>
              </tr>
              <tr>
                <td>4. Evaluation Metrics: BLEU, ROUGE, METEOR, WandB</td>
                <td>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Colab-F9AB00?style=plastic&logo=google-colab&logoColor=white" alt="Colab">
                  </a>
                </td>
                <td>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Colab-F9AB00?style=plastic&logo=google-colab&logoColor=white" alt="Colab">
                  </a>
                </td>
              </tr>
              <tr>
                <td>5. Hyperparameter tuning with Wandb Sweeps</td>
                <td>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Colab-F9AB00?style=plastic&logo=google-colab&logoColor=white" alt="Colab">
                  </a>
                </td>
                <td>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Colab-F9AB00?style=plastic&logo=google-colab&logoColor=white" alt="Colab">
                  </a>
                </td>
              </tr>
              <tr>
                <td>6. Complete Transformer: End-to-End Python Pipeline</td>
                <td>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Colab-F9AB00?style=plastic&logo=google-colab&logoColor=white" alt="Colab">
                  </a>
                </td>
                <td>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Colab-F9AB00?style=plastic&logo=google-colab&logoColor=white" alt="Colab">
                  </a>
                </td>
              </tr>
              <tr>
                <td>7. Create API and use Gradio and Streamlit</td>
                <td>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Colab-F9AB00?style=plastic&logo=google-colab&logoColor=white" alt="Colab">
                  </a>
                </td>
                <td>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Colab-F9AB00?style=plastic&logo=google-colab&logoColor=white" alt="Colab">
                  </a>
                </td>
              </tr>
              <tr>
                <td>8. Model Deployment and Monitoring</td>
                <td>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Colab-F9AB00?style=plastic&logo=google-colab&logoColor=white" alt="Colab">
                  </a>
                </td>
                <td>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                  <a href="#" class="badge-link">
                    <img src="https://img.shields.io/badge/Colab-F9AB00?style=plastic&logo=google-colab&logoColor=white" alt="Colab">
                  </a>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Part 2: Transformer-Based Architectures -->
      <div class="course-section">
        <div class="section-header">
          <div class="header-left">
            <h3>Part 2: Transformer-Based Architectures</h3>
          </div>
          <a href="https://github.com/xmarva/transformer-based-architectures" class="github-link"><i class="fab fa-github"></i> GitHub</a>
        </div>
        <div class="section-content">
          <table class="table">
            <thead>
              <tr>
                <th>Notebook</th>
                <th>English</th>
                <th>Russian</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>1. BERT: Bidirectional Encoder Representations</td>
                <td>
                  <a href="https://github.com/xmarva/transformer-based-architectures/blob/main/notebooks/01-bert-explained-and-fine-tuned-for-ner.ipynb" class="badge-link">
                    <img src="https://img.shields.io/badge/GitHub-0F0F0F?style=plastic&logo=github&logoColor=white" alt="GitHub">
                  </a>
                  <a href="https://www.kaggle.com/code/qmarva/bert-explained-and-fine-tuned-for-ner?scriptVersionId=235335954" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                </td>
                <td></td>
              </tr>
              <tr>
                <td>2. GPT: Generative Pre-trained Transformers</td>
                <td>
                  <a href="https://github.com/xmarva/transformer-based-architectures/blob/main/notebooks/02-gpt-explained-and-fine-tuned-for-reasoning.ipynb" class="badge-link">
                    <img src="https://img.shields.io/badge/GitHub-0F0F0F?style=plastic&logo=github&logoColor=white" alt="GitHub">
                  </a>
                  <a href="https://www.kaggle.com/code/qmarva/gpt-explained-and-fine-tuned-for-reasoning?scriptVersionId=235487807" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                </td>
                <td></td>
              </tr>
              <tr>
                <td>3. T5: Text-to-Text Transfer Framework</td>
                <td>
                  <a href="https://github.com/xmarva/transformer-based-architectures/blob/main/notebooks/03-t5-explained-and-fine-tuned-for-summarization.ipynb" class="badge-link">
                    <img src="https://img.shields.io/badge/GitHub-0F0F0F?style=plastic&logo=github&logoColor=white" alt="GitHub">
                  </a>
                  <a href="https://www.kaggle.com/code/qmarva/t5-explained-and-fine-tuned-for-summarization" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                </td>
                <td></td>
              </tr>
              <tr>
                <td>4. Mixture of Experts: Switch Transformers</td>
                <td>
                  <a href="https://github.com/xmarva/transformer-based-architectures/blob/main/notebooks/04-moe-explained-and-fine-tuned.ipynb" class="badge-link">
                    <img src="https://img.shields.io/badge/GitHub-0F0F0F?style=plastic&logo=github&logoColor=white" alt="GitHub">
                  </a>
                  <a href="https://www.kaggle.com/code/qmarva/moe-explained-and-fine-tuned" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                </td>
                <td></td>
              </tr>
              <tr>
                <td>5. Vision Transformers: ViT, DeiT</td>
                <td>
                  <a href="https://github.com/xmarva/transformer-based-architectures/blob/main/notebooks/05-vision-transformer-explained-and-fine-tuned.ipynb" class="badge-link">
                    <img src="https://img.shields.io/badge/GitHub-0F0F0F?style=plastic&logo=github&logoColor=white" alt="GitHub">
                  </a>
                  <a href="https://www.kaggle.com/code/qmarva/vision-transformer-explained-and-fine-tuned" class="badge-link">
                    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=plastic&logo=kaggle&logoColor=white" alt="Kaggle">
                  </a>
                </td>
                <td></td>
              </tr>
              <tr>
                <td>6. Longformers: Efficient Long-Context Attention</td>
                <td>
                  <a href="https://github.com/xmarva/transformer-based-architectures/blob/main/notebooks/06-Longformer.ipynb" class="badge-link">
                    <img src="https://img.shields.io/badge/GitHub-0F0F0F?style=plastic&logo=github&logoColor=white" alt="GitHub">
                  </a>
                </td>
                <td></td>
              </tr>
              <tr>
                <td>7. Multimodal Architectures: CLIP, Llava</td>
                <td>
                  <a href="https://github.com/xmarva/transformer-based-architectures/blob/main/notebooks/07-CLIP.ipynb" class="badge-link">
                    <img src="https://img.shields.io/badge/GitHub-0F0F0F?style=plastic&logo=github&logoColor=white" alt="GitHub">
                  </a>
                </td>
                <td></td>
              </tr>
              <tr>
                <td>8. LLM Architectures: Gemma2, LLaMA, Mistral</td>
                <td>
                  <a href="https://github.com/xmarva/transformer-based-architectures/blob/main/07-CLIP.ipynb" class="badge-link">
                    <img src="https://img.shields.io/badge/GitHub-0F0F0F?style=plastic&logo=github&logoColor=white" alt="GitHub">
                  </a>
                </td>
                <td></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

  </div>
</div>

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

.course-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  border-bottom: 1px solid var(--global-divider-color);
  padding-bottom: 0.75rem;
}

.header-left {
  display: flex;
  align-items: center;
}

.course-header h3 {
  margin: 0;
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

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.section-header h3 {
  margin: 0;
  font-size: 1.3rem;
  color: var(--global-theme-color);
}

.github-link {
  text-decoration: none;
  color: var(--global-theme-color);
  transition: color 0.2s;
  white-space: nowrap;
}

.github-link:hover {
  color: var(--global-hover-color, #0056b3);
}

.table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 1rem;
}

.table td, .table th {
  padding: 0.85rem 1rem;
  border-bottom: 1px solid var(--global-divider-color);
  font-size: 1rem;
  text-align: left;
}

.table thead th {
  text-align: left;
  border-bottom: 2px solid var(--global-divider-color);
}

.table tbody tr:last-child td {
  border-bottom: none;
}

.table tbody tr:hover {
  background-color: rgba(0, 0, 0, 0.02);
}

.badge-link {
  display: inline-block;
  margin: 2px;
  transition: transform 0.2s;
}

.lesson-planned td {
  color: var(--global-text-color-light);
}

.badge-link:hover {
  transform: translateY(-2px);
}
</style>
{% endcomment %}
