---
layout: page
title: portfolio
permalink: /portfolio/
description: Computer vision and ML systems I have built, from the first prototype to production
seo_description: "Projects by Eva Koroleva: virtual try-on with diffusion and 3D bodies, floor plans to 3D, multi-camera tracking, medical NLP and more."
nav: true
nav_order: 2
hide_header: true
display_categories: [work, personal]
---

<style>
  /* One accent (the site's red) for the one action; color otherwise encodes the domain of a project. */
  .pf {
    --pf-cv: #0f8c7e;
    --pf-genai: #6a55d8;
    --pf-3d: #b86e00;
    --pf-nlp: #3567d0;
    --pf-bio: #c4385a;
    --pf-seq: #8a6a12;
    --pf-chip-alpha: 12%;
    --pf-border: var(--global-divider-color);
    --pf-muted: var(--global-text-color-light);
  }
  html[data-theme="dark"] .pf {
    --pf-cv: #4cc9b8;
    --pf-genai: #a596f7;
    --pf-3d: #f0b04a;
    --pf-nlp: #86a8f7;
    --pf-bio: #f4879d;
    --pf-seq: #d9b45a;
    --pf-chip-alpha: 16%;
  }

  .pf-section { margin-bottom: 3.5rem; }
  .pf-section-head { margin-bottom: 1.4rem; }
  .pf-section-head::before {
    content: "";
    display: block;
    width: 28px;
    height: 3px;
    margin-bottom: 0.75rem;
    background: var(--global-theme-color);
  }
  .pf-section-head h2 {
    font-size: 1.45rem;
    font-weight: 500;
    color: var(--global-text-color);
    margin: 0;
  }

  /* domain chips: tinted, one hue per domain */
  .pf-chips { display: flex; flex-wrap: wrap; gap: 0.35rem; }
  .pf-chip {
    --c: var(--pf-muted);
    display: inline-flex;
    align-items: center;
    padding: 0.12rem 0.55rem;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    color: var(--c);
    background: color-mix(in srgb, var(--c) var(--pf-chip-alpha), transparent);
    white-space: nowrap;
  }
  .pf-chip[data-domain="computer-vision"] { --c: var(--pf-cv); }
  .pf-chip[data-domain="genai"] { --c: var(--pf-genai); }
  .pf-chip[data-domain="3d"] { --c: var(--pf-3d); }
  .pf-chip[data-domain="nlp"] { --c: var(--pf-nlp); }
  .pf-chip[data-domain="biomed"] { --c: var(--pf-bio); }
  .pf-chip[data-domain="neurosymbolic-ai"] { --c: var(--pf-genai); }
  .pf-chip[data-domain="sequence-prediction"] { --c: var(--pf-seq); }

  .pf-stack {
    margin: 0;
    font-family: var(--global-code-font-family, monospace);
    font-size: 0.74rem;
    line-height: 1.5;
    color: var(--pf-muted);
  }

  /* selected work: wide cards */
  .pf-work { display: flex; flex-direction: column; gap: 1.5rem; }
  .pf-case {
    display: grid;
    grid-template-columns: minmax(0, 340px) minmax(0, 1fr);
    border: 1px solid var(--pf-border);
    border-radius: 14px;
    overflow: hidden;
    background: var(--global-card-bg-color);
    transition: border-color 0.2s ease, transform 0.2s ease;
  }
  .pf-case:hover { border-color: color-mix(in srgb, var(--global-theme-color) 45%, var(--pf-border)); }
  /* the covers are drawn on #161618: the column fills the card's height in that color, the cover centred in it */
  .pf-case-media { display: flex; align-items: center; overflow: hidden; background: #161618; }
  .pf-case-media figure, .pf-case-media picture { display: block; width: 100%; margin: 0; }
  .pf-case-media img { width: 100%; height: auto; display: block; }
  .pf-case-body { padding: 1.25rem 1.5rem; display: flex; flex-direction: column; gap: 0.55rem; min-width: 0; }
  .pf-head { display: flex; flex-wrap: wrap; align-items: flex-start; justify-content: space-between; gap: 0.4rem 1rem; }
  .pf-head .pf-chips { justify-content: flex-end; margin-left: auto; padding-top: 0.3rem; }
  .pf-title { font-size: 1.3rem; font-weight: 500; margin: 0; line-height: 1.3; }
  .pf-title a { color: var(--global-text-color); text-decoration: none; }
  .pf-title a:hover { color: var(--global-theme-color); }
  .pf-desc { margin: 0; color: var(--pf-muted); line-height: 1.6; }
  .pf-foot { margin-top: auto; display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 1.25rem; padding-top: 0.35rem; }
  .pf-more { font-size: 0.9rem; font-weight: 500; color: var(--global-text-color) !important; text-decoration: none !important; }
  .pf-more:hover { color: var(--global-theme-color) !important; }
  .pf-ext { font-size: 0.85rem; color: var(--pf-muted) !important; text-decoration: none !important; }
  .pf-ext:hover { color: var(--global-text-color) !important; }
  .pf-ext i { margin-right: 0.3rem; }

  /* experiments: a compact grid */
  .pf-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1.1rem; }
  .pf-mini {
    display: flex;
    flex-direction: column;
    border: 1px solid var(--pf-border);
    border-radius: 12px;
    overflow: hidden;
    background: var(--global-card-bg-color);
    transition: border-color 0.2s ease;
  }
  .pf-mini:hover { border-color: color-mix(in srgb, var(--global-theme-color) 45%, var(--pf-border)); }
  .pf-mini-media { display: block; aspect-ratio: 16 / 9; overflow: hidden; background: #f3f3f3; border-bottom: 1px solid var(--pf-border); }
  .pf-mini-media figure, .pf-mini-media picture { display: block; height: 100%; margin: 0; }
  .pf-mini-media img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .pf-mini-body { padding: 0.85rem 1rem 1rem; display: flex; flex-direction: column; gap: 0.45rem; flex: 1; }
  .pf-mini h3 { font-size: 1rem; font-weight: 500; margin: 0; line-height: 1.35; }
  .pf-mini h3 a { color: var(--global-text-color); text-decoration: none; }
  .pf-mini h3 a:hover { color: var(--global-theme-color); }
  .pf-mini p {
    margin: 0; font-size: 0.88rem; line-height: 1.5; color: var(--pf-muted);
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
  }
  .pf-mini .pf-stack { font-size: 0.74rem; display: block; -webkit-line-clamp: none; }
  .pf-mini .pf-foot { padding-top: 0.2rem; }
  .pf-mini .pf-foot .pf-chips { margin-left: auto; }

  @media (max-width: 991px) {
    .pf-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .pf-head .pf-chips { margin-left: 0; justify-content: flex-start; padding-top: 0; }  /* under the title here */
  }
  @media (max-width: 760px) {
    .pf-case { grid-template-columns: minmax(0, 1fr); }
    .pf-grid { grid-template-columns: minmax(0, 1fr); }
  }
</style>

<div class="pf">

  <section class="pf-section" aria-labelledby="pf-work">
    <div class="pf-section-head">
      <h2 id="pf-work">Case studies</h2>
    </div>
    <div class="pf-work">
      {% assign work_projects = site.projects | where: "category", "work" | sort: "importance" %}
      {% for project in work_projects %}
      <article class="pf-case">
        <a class="pf-case-media" href="{{ project.url | relative_url }}" tabindex="-1" aria-hidden="true">
          {% if project.img %}{% include figure.liquid path=project.img alt="" sizes="(min-width: 760px) 340px, 100vw" %}{% endif %}
        </a>
        <div class="pf-case-body">
          <div class="pf-head">
            <h3 class="pf-title"><a href="{{ project.url | relative_url }}">{{ project.title }}</a></h3>
            <span class="pf-chips">{% for domain in project.domains %}<span class="pf-chip" data-domain="{{ domain | slugify }}">{{ domain }}</span>{% endfor %}</span>
          </div>
          <p class="pf-desc">{{ project.description }}</p>
          {% if project.tech %}<p class="pf-stack">{{ project.tech | join: " · " }}</p>{% endif %}
          <div class="pf-foot">
            <a class="pf-more" href="{{ project.url | relative_url }}">Read the case study →</a>
            {% if project.website %}<a class="pf-ext" href="{{ project.website }}" target="_blank" rel="noopener noreferrer"><i class="fas fa-globe" aria-hidden="true"></i>{{ project.website | remove: "https://" | remove: "http://" | remove: "www." }}</a>{% endif %}
            {% if project.github %}<a class="pf-ext" href="{{ project.github }}" target="_blank" rel="noopener noreferrer"><i class="fab fa-github" aria-hidden="true"></i>code</a>{% endif %}
          </div>
        </div>
      </article>
      {% endfor %}
    </div>
  </section>

  <section class="pf-section" aria-labelledby="pf-lab">
    <div class="pf-section-head">
      <h2 id="pf-lab">Experiments</h2>
    </div>
    <div class="pf-grid">
      {% assign personal_projects = site.projects | where: "category", "personal" | sort: "importance" %}
      {% for project in personal_projects %}
      {% if project.github %}{% assign link = project.github %}{% elsif project.kaggle %}{% assign link = project.kaggle %}{% else %}{% assign link = project.url | relative_url %}{% endif %}
      <article class="pf-mini">
        <a class="pf-mini-media" href="{{ link }}" target="_blank" rel="noopener noreferrer" tabindex="-1" aria-hidden="true">
          {% if project.img %}{% include figure.liquid path=project.img alt="" sizes="(min-width: 992px) 300px, (min-width: 760px) 50vw, 100vw" %}{% endif %}
        </a>
        <div class="pf-mini-body">
          <h3><a href="{{ link }}" target="_blank" rel="noopener noreferrer">{{ project.title }}</a></h3>
          <p>{{ project.description }}</p>
          {% if project.tech %}<p class="pf-stack">{{ project.tech | join: " · " }}</p>{% endif %}
          <div class="pf-foot">
            {% if project.github %}<a class="pf-ext" href="{{ project.github }}" target="_blank" rel="noopener noreferrer"><i class="fab fa-github" aria-hidden="true"></i>GitHub</a>{% endif %}
            {% if project.kaggle %}<a class="pf-ext" href="{{ project.kaggle }}" target="_blank" rel="noopener noreferrer"><i class="fab fa-kaggle" aria-hidden="true"></i>Kaggle</a>{% endif %}
            <span class="pf-chips">{% for domain in project.domains %}<span class="pf-chip" data-domain="{{ domain | slugify }}">{{ domain }}</span>{% endfor %}</span>
          </div>
        </div>
      </article>
      {% endfor %}
    </div>
  </section>
</div>
