---
layout: default
permalink: /
title: blog
seo_description: "Eva Koroleva's blog: long reads on machine learning, Transformers and LLMs, AI security and cognitive science, with experiments, code and figures drawn from data."
---

{% comment %}
  The home page is the blog, laid out like a magazine: the newest post large, the course as one card for its whole
  series, the other posts as a grid of covers. Chips filter by category in the page (ES2017 script at the end:
  the production build minifies it with Uglifier). The full list by date stays at /blog/.
{% endcomment %}

{% assign series_tag = 'transformers-from-scratch' %}
{% assign course = site.data.courses | where: 'id', 'tfs' | first %}
{% assign posts = '' | split: '' %}
{% for post in site.posts %}
  {% unless post.tags contains series_tag %}{% assign posts = posts | push: post %}{% endunless %}
{% endfor %}
{% assign lead = posts | first %}
{% assign rest = posts | slice: 1, 12 %}

{% assign chip_names = 'machine-learning,cognitive-science' | split: ',' %}
{% assign chip_labels = 'Machine learning,Cognitive science' | split: ',' %}

<style>
  .bh { --bh-border: var(--global-divider-color); --bh-muted: var(--global-text-color-light); margin: 2.4rem 0 1rem; }

  .bh-head { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 1rem 2rem; }
  .bh-head h1 { margin: 0; font-size: 2.3rem; font-weight: 500; line-height: 1.15; color: var(--global-text-color); }
  .bh-head p { margin: 0.6rem 0 0; max-width: 40rem; font-size: 1.05rem; line-height: 1.6; color: var(--bh-muted); }

  .bh-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0; padding: 0; list-style: none; }
  .bh-chip {
    padding: 0.3rem 0.9rem;
    border: 1px solid var(--bh-border);
    border-radius: 999px;
    background: transparent;
    color: var(--global-text-color);
    font-size: 0.88rem;
    line-height: 1.4;
    cursor: pointer;
    transition: border-color 0.2s ease, color 0.2s ease, background 0.2s ease;
  }
  .bh-chip:hover { border-color: color-mix(in srgb, var(--global-theme-color) 45%, var(--bh-border)); }
  .bh-chip[aria-pressed="true"] {
    border-color: var(--global-theme-color);
    background: color-mix(in srgb, var(--global-theme-color) 12%, transparent);
    color: var(--global-theme-color);
  }

  /* covers keep their 1200x630 shape; a post without one gets a typographic cover in the same dark-and-red style */
  .bh-cover {
    position: relative;
    display: block;
    aspect-ratio: 1200 / 630;
    border: 1px solid var(--bh-border);
    border-radius: 12px;
    overflow: hidden;
    background: #161618;
  }
  .bh-cover img { display: block; width: 100%; height: 100%; object-fit: cover; transition: transform 0.4s ease; }
  .bh-card:hover .bh-cover img, .bh-lead:hover .bh-cover img { transform: scale(1.02); }
  .bh-cover-text {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 8% 7%;
    color: #f2f3f5;
  }
  .bh-cover-text small {
    font-family: var(--global-code-font-family, monospace);
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #9da3ae;
  }
  .bh-cover-text small::before {
    content: "";
    display: inline-block;
    width: 0.5em;
    height: 0.5em;
    margin-right: 0.5em;
    border-radius: 50%;
    background: #ff3636;
  }
  .bh-cover-text strong { font-size: 1.35rem; font-weight: 600; line-height: 1.2; }

  .bh-meta { margin: 0; font-size: 0.82rem; color: var(--bh-muted); }
  .bh-meta a { position: relative; z-index: 2; color: inherit; }
  .bh-meta a:hover { color: var(--global-theme-color); }
  .bh-title { margin: 0; font-weight: 500; line-height: 1.3; }
  .bh-title a { color: var(--global-text-color); text-decoration: none; }
  .bh-title a:hover { color: var(--global-theme-color); }
  .bh-desc { margin: 0; color: var(--bh-muted); line-height: 1.55; }

  .bh-lead {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr);
    gap: 2rem;
    align-items: center;
    margin: 2rem 0 0;
  }
  .bh-lead-text { display: flex; flex-direction: column; gap: 0.75rem; }
  .bh-lead .bh-title { font-size: 1.85rem; }
  .bh-lead .bh-desc { font-size: 1.02rem; }
  .bh-label {
    align-self: flex-start;
    padding: 0.08rem 0.55rem;
    border-radius: 999px;
    background: color-mix(in srgb, var(--global-theme-color) 14%, transparent);
    color: var(--global-theme-color);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  .bh-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 2.4rem 1.75rem; margin: 3rem 0 0; }
  .bh-card { position: relative; display: flex; flex-direction: column; gap: 0.6rem; min-width: 0; }
  .bh-card .bh-cover { margin-bottom: 0.25rem; }
  .bh-card .bh-title { font-size: 1.08rem; }
  .bh-card .bh-desc {
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    font-size: 0.9rem;
  }

  /* the course: one card for the whole series, its newest covers fanned out as on /teaching/ */
  .bh-series {
    position: relative;
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr);
    gap: 2rem;
    align-items: center;
    padding: 1.9rem 2.1rem;
    border: 1px solid var(--bh-border);
    border-radius: 16px;
    background: var(--global-card-bg-color);
    overflow: hidden;
  }
  .bh-series-text { display: flex; flex-direction: column; align-items: flex-start; gap: 0.75rem; min-width: 0; }
  .bh-series .bh-title { font-size: 1.5rem; }
  .bh-progress { display: flex; align-items: center; gap: 0.8rem; font-size: 0.85rem; color: var(--bh-muted); }
  .bh-bar { display: flex; gap: 4px; width: 180px; }
  .bh-bar span { flex: 1; height: 6px; border-radius: 3px; background: var(--bh-border); }
  .bh-bar span.on { background: var(--global-theme-color); }
  .bh-latest { position: relative; z-index: 2; font-size: 0.95rem; color: var(--global-text-color); }
  .bh-latest:hover { color: var(--global-theme-color); }
  .bh-fan { position: relative; display: block; aspect-ratio: 16 / 10; }
  .bh-fan img {
    position: absolute;
    right: 0;
    top: 50%;
    width: 80%;
    height: auto;
    border-radius: 10px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.15), 0 14px 32px rgba(0, 0, 0, 0.22);
    transform: translate(calc(var(--i) * -11%), calc(-50% - var(--i) * 9%)) rotate(calc(var(--i) * -4deg));
    z-index: calc(3 - var(--i));
    filter: brightness(calc(1 - var(--i) * 0.12));
    transition: transform 0.35s ease;
  }
  .bh-series:hover .bh-fan img {
    transform: translate(calc(var(--i) * -14%), calc(-50% - var(--i) * 11%)) rotate(calc(var(--i) * -6deg));
  }

  .bh-more { display: flex; justify-content: flex-end; margin: 3rem 0 0; padding-top: 1.25rem; border-top: 1px solid var(--bh-border); }
  .bh-more a { font-size: 0.95rem; color: var(--global-text-color); text-decoration: none; }
  .bh-more a:hover { color: var(--global-theme-color); }

  .bh [hidden] { display: none !important; }

  @media (max-width: 991px) {
    .bh-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }
  @media (max-width: 767px) {
    .bh { margin-top: 1.5rem; }
    .bh-head h1 { font-size: 1.85rem; }
    .bh-lead, .bh-series { grid-template-columns: minmax(0, 1fr); gap: 1.25rem; }
    .bh-lead .bh-title { font-size: 1.45rem; }
    .bh-series { padding: 1.4rem 1.25rem; }
    .bh-fan { order: -1; width: 88%; margin: 0.5rem auto 0; }
  }
  @media (max-width: 575px) {
    .bh-grid { grid-template-columns: minmax(0, 1fr); gap: 2rem; }
  }
</style>

<div class="bh">
  <header class="bh-head">
    <div>
      <h1>Writing on machine learning and the mind</h1>
      <p>Long reads with experiments, code and figures drawn from data: Transformers and LLMs, AI security, cognitive science.</p>
    </div>
    <ul class="bh-chips" aria-label="Filter posts by topic">
      <li><button type="button" class="bh-chip" data-filter="all" aria-pressed="true">All</button></li>
      {% for name in chip_names %}
        <li><button type="button" class="bh-chip" data-filter="{{ name }}" aria-pressed="false">{{ chip_labels[forloop.index0] }}</button></li>
      {% endfor %}
    </ul>
  </header>

  {% assign category = lead.categories | first %}
  {% assign read_time = lead.content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
  <article class="bh-lead" data-category="{{ category }}">
    <a class="bh-cover" href="{{ lead.url | relative_url }}" tabindex="-1" aria-hidden="true">
      {% if lead.thumbnail %}
        <img src="{{ lead.thumbnail | relative_url }}" alt="" width="1200" height="630" decoding="async">
      {% else %}
        <span class="bh-cover-text"><small>{{ category | replace: '-', ' ' }}</small><strong>{{ lead.title }}</strong></span>
      {% endif %}
    </a>
    <div class="bh-lead-text">
      <span class="bh-label">Latest</span>
      <p class="bh-meta">
        <time datetime="{{ lead.date | date_to_xmlschema }}">{{ lead.date | date: '%b %-d, %Y' }}</time> · {{ read_time }} min read
        {% if category %}· <a href="{{ category | slugify | prepend: '/blog/category/' | append: '/' | relative_url }}">{{ category | replace: '-', ' ' }}</a>{% endif %}
      </p>
      <h2 class="bh-title"><a href="{{ lead.url | relative_url }}">{{ lead.title }}</a></h2>
      {% if lead.description %}<p class="bh-desc">{{ lead.description }}</p>{% endif %}
    </div>
  </article>

  <div class="bh-grid">
    {% for post in rest %}
      {% if forloop.index0 == 3 and course %}
        {% assign published = course.lessons | where_exp: 'l', 'l.url' %}
        {% assign newest = published | last %}
        <article class="bh-series" data-category="machine-learning">
          <div class="bh-series-text">
            <span class="bh-label">Course · series</span>
            <h2 class="bh-title"><a href="{{ '/teaching/' | relative_url }}">{{ course.title }}</a></h2>
            <p class="bh-desc">{{ course.summary }}</p>
            <div class="bh-progress">
              <span class="bh-bar" aria-hidden="true">{% for l in course.lessons %}<span{% if l.url %} class="on"{% endif %}></span>{% endfor %}</span>
              {{ published.size }} of {{ course.lessons.size }} lessons
            </div>
            <a class="bh-latest" href="{{ newest.url | relative_url }}">Newest: {{ newest.title }}&nbsp;→</a>
          </div>
          <a class="bh-fan" href="{{ '/teaching/' | relative_url }}" tabindex="-1" aria-hidden="true">
            {% assign fan = published | reverse %}
            {% for l in fan limit: 3 %}
              <img src="{{ l.cover | relative_url }}" alt="" width="1200" height="630" loading="lazy" decoding="async" style="--i: {{ forloop.index0 }}">
            {% endfor %}
          </a>
        </article>
      {% endif %}
      {% assign category = post.categories | first %}
      {% assign read_time = post.content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
      <article class="bh-card" data-category="{{ category }}">
        <a class="bh-cover" href="{{ post.url | relative_url }}" tabindex="-1" aria-hidden="true">
          {% if post.thumbnail %}
            <img src="{{ post.thumbnail | relative_url }}" alt="" width="1200" height="630" loading="lazy" decoding="async">
          {% else %}
            <span class="bh-cover-text"><small>{{ category | replace: '-', ' ' }}</small><strong>{{ post.title }}</strong></span>
          {% endif %}
        </a>
        <p class="bh-meta">
          <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: '%b %-d, %Y' }}</time> · {{ read_time }} min read
        </p>
        <h3 class="bh-title"><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
        {% if post.description %}<p class="bh-desc">{{ post.description }}</p>{% endif %}
      </article>
    {% endfor %}
  </div>

  <nav class="bh-more"><a href="{{ '/blog/' | relative_url }}">All posts by date →</a></nav>
</div>

<script>
  (function () {
    var chips = document.querySelectorAll(".bh-chip");
    var items = document.querySelectorAll(".bh [data-category]");
    function show(filter) {
      for (var i = 0; i < chips.length; i++) {
        chips[i].setAttribute("aria-pressed", chips[i].getAttribute("data-filter") === filter ? "true" : "false");
      }
      for (var j = 0; j < items.length; j++) {
        items[j].hidden = filter !== "all" && items[j].getAttribute("data-category") !== filter;
      }
    }
    for (var k = 0; k < chips.length; k++) {
      chips[k].addEventListener("click", function (event) {
        show(event.currentTarget.getAttribute("data-filter"));
      });
    }
  })();
</script>
