---
layout: default
permalink: /services/
title: services
seo_title: Machine learning consulting and development
seo_description: "Machine learning consulting and development by Eva Koroleva: computer vision, generative AI and language models, from a feasibility check to a prototype on your data and a system in production."
description: Machine learning consulting and development
nav: true
nav_order: 3
---

<style>
  /* The teaching page's language: a hero card with covers fanned out, numbered rows, the site's red for the one
     action and the numbers; the portfolio's domain colours for the chips. */
  .sv {
    --sv-border: var(--global-divider-color);
    --sv-muted: var(--global-text-color-light);
    --sv-cv: #0f8c7e;
    --sv-genai: #6a55d8;
    --sv-3d: #b86e00;
    --sv-nlp: #3567d0;
    --sv-bio: #c4385a;
    --sv-chip-alpha: 12%;
  }
  html[data-theme="dark"] .sv {
    --sv-cv: #4cc9b8;
    --sv-genai: #a596f7;
    --sv-3d: #f0b04a;
    --sv-nlp: #86a8f7;
    --sv-bio: #f4879d;
    --sv-chip-alpha: 16%;
  }
  .sv-page { margin: 2.6rem 0 3.5rem; }

  /* hero */
  .sv-hero {
    display: grid;
    grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr);
    gap: 2rem;
    align-items: center;
    padding: 2.25rem 2.25rem 2.25rem 2.5rem;
    border: 1px solid var(--sv-border);
    border-radius: 16px;
    background: var(--global-card-bg-color);
    overflow: hidden;
  }
  .sv-hero-text { display: flex; flex-direction: column; align-items: flex-start; gap: 1rem; min-width: 0; }
  .sv-hero h1 { margin: 0; font-size: 2.1rem; font-weight: 500; line-height: 1.18; color: var(--global-text-color); }
  .sv-summary { margin: 0; font-size: 1.05rem; line-height: 1.6; color: var(--sv-muted); }
  .sv-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem 1.25rem; margin-top: 0.35rem; }
  .sv-primary {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.5rem 1.15rem;
    border-radius: 999px;
    background: var(--global-theme-color);
    color: var(--global-bg-color) !important;
    font-size: 0.95rem;
    font-weight: 500;
    text-decoration: none !important;
    transition: filter 0.2s ease;
  }
  .sv-primary:hover { filter: brightness(1.1); }
  .sv-quiet { font-size: 0.95rem; color: var(--global-text-color) !important; text-decoration: none !important; }
  .sv-quiet:hover { color: var(--global-theme-color) !important; }
  .sv-quiet i { margin-right: 0.35rem; }

  .sv-chips { display: flex; flex-wrap: wrap; gap: 0.35rem; }
  .sv-chip {
    --c: var(--sv-muted);
    display: inline-flex;
    align-items: center;
    padding: 0.12rem 0.55rem;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    color: var(--c);
    background: color-mix(in srgb, var(--c) var(--sv-chip-alpha), transparent);
    white-space: nowrap;
  }
  .sv-chip[data-domain="computer-vision"] { --c: var(--sv-cv); }
  .sv-chip[data-domain="genai"] { --c: var(--sv-genai); }
  .sv-chip[data-domain="3d"] { --c: var(--sv-3d); }
  .sv-chip[data-domain="nlp"] { --c: var(--sv-nlp); }
  .sv-chip[data-domain="biomed"] { --c: var(--sv-bio); }
  .sv-chip[data-domain="neurosymbolic-ai"] { --c: var(--sv-genai); }

  /* the case studies' covers, fanned out */
  .sv-fan { position: relative; display: block; aspect-ratio: 4 / 3; }
  .sv-fan-card {
    position: absolute;
    right: 0;
    top: 50%;
    width: 78%;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.15), 0 14px 32px rgba(0, 0, 0, 0.22);
    transform: translate(calc(var(--i) * -12%), calc(-50% - var(--i) * 9%)) rotate(calc(var(--i) * -4deg));
    z-index: calc(3 - var(--i));
    filter: brightness(calc(1 - var(--i) * 0.12));
    transition: transform 0.35s ease;
  }
  .sv-fan-card figure, .sv-fan-card picture { display: block; margin: 0; }
  .sv-fan-card img { display: block; width: 100%; height: auto; }
  .sv-fan:hover .sv-fan-card {
    transform: translate(calc(var(--i) * -15%), calc(-50% - var(--i) * 11%)) rotate(calc(var(--i) * -6deg));
  }

  /* sections */
  .sv-head { margin: 3rem 0 1rem; }
  .sv-head::before {
    content: "";
    display: block;
    width: 28px;
    height: 3px;
    margin-bottom: 0.75rem;
    background: var(--global-theme-color);
  }
  .sv-head h2 { margin: 0; font-size: 1.45rem; font-weight: 500; color: var(--global-text-color); }

  /* numbered rows, as the lessons on /teaching/ */
  .sv-rows { list-style: none; margin: 0; padding: 0; border-top: 1px solid var(--sv-border); }
  .sv-row {
    display: grid;
    grid-template-columns: 2.75rem minmax(0, 1fr) auto;
    column-gap: 1rem;
    align-items: center;
    padding: 1.15rem 0.25rem;
    border-bottom: 1px solid var(--sv-border);
    scroll-margin-top: 5rem;
  }
  .sv-num {
    align-self: start;
    padding-top: 0.15rem;
    font-family: var(--global-code-font-family, monospace);
    font-size: 0.95rem;
    color: var(--global-theme-color);
  }
  .sv-row-title { margin: 0; font-size: 1.1rem; font-weight: 500; line-height: 1.4; color: var(--global-text-color); }
  .sv-row-text { margin: 0.2rem 0 0; font-size: 0.98rem; line-height: 1.55; color: var(--global-text-color); }
  .sv-row-meta { margin: 0.3rem 0 0; font-size: 0.9rem; line-height: 1.5; color: var(--sv-muted); }
  .sv-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.22rem 0.8rem;
    border: 1px solid var(--sv-border);
    border-radius: 999px;
    font-size: 0.8rem;
    white-space: nowrap;
    color: var(--global-text-color) !important;
    text-decoration: none !important;
    transition: border-color 0.2s ease, color 0.2s ease;
  }
  .sv-pill:hover { border-color: var(--global-theme-color); color: var(--global-theme-color) !important; }

  /* case studies: a row each, the cover as a thumbnail */
  .sv-case {
    display: grid;
    grid-template-columns: 9rem minmax(0, 1fr) auto;
    column-gap: 1.25rem;
    align-items: center;
    padding: 0.9rem 0.25rem;
    border-bottom: 1px solid var(--sv-border);
    color: var(--global-text-color) !important;
    text-decoration: none !important;
  }
  .sv-thumb { display: block; aspect-ratio: 16 / 10; border-radius: 8px; overflow: hidden; background: #161618; }
  .sv-thumb figure, .sv-thumb picture { display: block; height: 100%; margin: 0; }
  .sv-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .sv-case .sv-row-title { display: block; }
  .sv-case:hover .sv-row-title { color: var(--global-theme-color); }
  .sv-case .sv-row-meta {
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
  }
  .sv-case .sv-chips { justify-content: flex-end; max-width: 15rem; }

  /* how it goes */
  .sv-steps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1.5rem; margin: 0; padding: 0; list-style: none; counter-reset: step; }
  .sv-step { counter-increment: step; padding-top: 1rem; border-top: 1px solid var(--sv-border); }
  .sv-step::before {
    content: counter(step, decimal-leading-zero);
    display: block;
    margin-bottom: 0.5rem;
    font-family: var(--global-code-font-family, monospace);
    font-size: 0.95rem;
    color: var(--global-theme-color);
  }
  .sv-step h3 { margin: 0 0 0.35rem; font-size: 1.08rem; font-weight: 500; color: var(--global-text-color); }
  .sv-step p { margin: 0; color: var(--sv-muted); line-height: 1.6; }

  /* contact */
  .sv-contact {
    display: grid;
    grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr);
    gap: 2rem;
    align-items: center;
    margin-top: 3.5rem;
    padding: 2rem 2.5rem;
    border: 1px solid var(--sv-border);
    border-radius: 16px;
    background: var(--global-card-bg-color);
    scroll-margin-top: 5rem;
  }
  .sv-contact h2 { margin: 0 0 0.5rem; font-size: 1.45rem; font-weight: 500; color: var(--global-text-color); }
  .sv-contact p { margin: 0; color: var(--sv-muted); line-height: 1.6; }
  .sv-channels { display: flex; flex-direction: column; gap: 0.7rem; align-items: flex-start; }
  .sv-address { display: inline-flex; align-items: center; gap: 0.6rem; font-family: var(--global-code-font-family, monospace); font-size: 0.88rem; color: var(--sv-muted); }
  .sv-copy {
    padding: 0.1rem 0.55rem;
    border: 1px solid var(--sv-border);
    border-radius: 999px;
    background: transparent;
    color: var(--sv-muted);
    font-family: inherit;
    font-size: 0.78rem;
    cursor: pointer;
  }
  .sv-copy:hover { color: var(--global-text-color); border-color: var(--sv-muted); }
  .sv-others { display: flex; flex-wrap: wrap; gap: 0.5rem 1.4rem; }

  @media (max-width: 860px) {
    .sv-hero { grid-template-columns: minmax(0, 1fr); gap: 1.75rem; padding: 1.5rem; }
    .sv-fan { order: -1; aspect-ratio: auto; }
    .sv-fan-card { position: relative; top: auto; width: 100%; transform: none; filter: none; }
    .sv-fan-card:not(:first-child) { display: none; }
    .sv-fan:hover .sv-fan-card { transform: none; }
    .sv-hero h1 { font-size: 1.7rem; }
    .sv-contact { grid-template-columns: minmax(0, 1fr); gap: 1.4rem; padding: 1.5rem; }
  }
  @media (max-width: 680px) {
    .sv-row { grid-template-columns: 2.1rem minmax(0, 1fr); row-gap: 0.75rem; }
    .sv-row .sv-pill { grid-column: 2; justify-self: start; }
    .sv-case { grid-template-columns: 6.5rem minmax(0, 1fr); row-gap: 0.6rem; }
    .sv-case .sv-chips { grid-column: 2; justify-content: flex-start; }
    .sv-steps { grid-template-columns: minmax(0, 1fr); gap: 1rem; }
  }
</style>

{% assign email_parts = site.email | split: "@" %}
{% capture mailto %}mailto:{{ site.email | encode_email }}?body=Hi%20Eva%2C%0A%0AWhat%20we%20are%20building%3A%0AWhat%20data%20we%20have%3A%0AWhen%20we%20need%20it%3A%0A{% endcapture %}
{% assign work_projects = site.projects | where: "category", "work" | sort: "importance" %}

<div class="post sv sv-page">
  <section class="sv-hero" aria-labelledby="sv-title">
    <div class="sv-hero-text">
      <span class="sv-chips"><span class="sv-chip" data-domain="computer-vision">Computer Vision</span><span class="sv-chip" data-domain="genai">GenAI</span><span class="sv-chip" data-domain="3d">3D</span><span class="sv-chip" data-domain="nlp">NLP</span></span>
      <h1 id="sv-title">Machine learning consulting and development</h1>
      <p class="sv-summary">I help teams find out what machine learning can do for their product, and then build it, from a prototype on their data to a system in production.</p>
      <div class="sv-actions">
        <a class="sv-primary" href="#contact">Write to me <span aria-hidden="true">→</span></a>
        <a class="sv-quiet" href="{{ '/portfolio/' | relative_url }}"><i class="fas fa-briefcase" aria-hidden="true"></i>Portfolio</a>
      </div>
    </div>
    <a class="sv-fan" href="{{ '/portfolio/' | relative_url }}" tabindex="-1" aria-hidden="true">
      {% for project in work_projects limit: 3 %}
      <span class="sv-fan-card" style="--i: {{ forloop.index0 }}">{% include figure.liquid path=project.img alt="" sizes="(min-width: 860px) 360px, 90vw" loading="eager" %}</span>
      {% endfor %}
    </a>
  </section>

  <div class="sv-head"><h2 id="sv-what">What I do</h2></div>
  <ol class="sv-rows" aria-labelledby="sv-what">
    <li class="sv-row" id="consulting">
      <span class="sv-num">01</span>
      <div>
        <h3 class="sv-row-title">Consulting</h3>
        <p class="sv-row-text">For deciding what to build, or finding out why something doesn't work.</p>
        <p class="sv-row-meta">Feasibility checks: whether ML can solve the problem, with what data and at what cost · reviews of models and ML systems already running · roadmaps: what to build first, buy or skip</p>
      </div>
      <a class="sv-pill" href="{{ mailto | replace: '?body=', '?subject=Consulting&body=' }}">Ask about it</a>
    </li>
    <li class="sv-row" id="development">
      <span class="sv-num">02</span>
      <div>
        <h3 class="sv-row-title">Development</h3>
        <p class="sv-row-text">A prototype or a production system, built with a small team of engineers I have worked with.</p>
        <p class="sv-row-meta">Prototypes that test the idea on your data · production services with an API, tests and monitoring · models made small and fast for phones and edge devices · slow or inaccurate models, fixed</p>
      </div>
      <a class="sv-pill" href="{{ mailto | replace: '?body=', '?subject=Development&body=' }}">Ask about it</a>
    </li>
  </ol>

  <div class="sv-head"><h2 id="sv-cases">Case studies</h2></div>
  <div class="sv-rows" role="list" aria-labelledby="sv-cases">
    {% for project in work_projects %}
    <a class="sv-case" role="listitem" href="{{ project.url | relative_url }}">
      <span class="sv-thumb">{% if project.img %}{% include figure.liquid path=project.img alt="" sizes="144px" %}{% endif %}</span>
      <span>
        <span class="sv-row-title">{{ project.title }}</span>
        <span class="sv-row-meta">{{ project.description }}</span>
      </span>
      <span class="sv-chips">{% for domain in project.domains %}<span class="sv-chip" data-domain="{{ domain | slugify }}">{{ domain }}</span>{% endfor %}</span>
    </a>
    {% endfor %}
  </div>

  <div class="sv-head"><h2 id="sv-how">How a project goes</h2></div>
  <ol class="sv-steps" aria-labelledby="sv-how">
    <li class="sv-step">
      <h3>A call</h3>
      <p>You describe the problem. I ask about the data, the constraints and what a good result looks like.</p>
    </li>
    <li class="sv-step">
      <h3>A proposal</h3>
      <p>A written plan: the approach, the risks, the timeline and the cost.</p>
    </li>
    <li class="sv-step">
      <h3>The work</h3>
      <p>In short iterations, each ending with something you can run and judge.</p>
    </li>
  </ol>

  <section class="sv-contact" id="contact" aria-labelledby="sv-contact">
    <div>
      <h2 id="sv-contact">Contact</h2>
      <p>Tell me what you are building, what data you have and when you need it. I work in English and Russian.</p>
    </div>
    <div class="sv-channels">
      <a class="sv-primary" href="{{ mailto | replace: '?body=', '?subject=Project%20inquiry&body=' }}"><i class="fas fa-envelope" aria-hidden="true"></i> Email me</a>
      <span class="sv-address" hidden><span class="sv-address-text"></span><button class="sv-copy" type="button" data-u="{{ email_parts[0] }}" data-d="{{ email_parts[1] }}">Copy</button></span>
      <div class="sv-others">
        <a class="sv-quiet" href="https://www.linkedin.com/in/{{ site.linkedin_username }}" target="_blank" rel="noopener noreferrer"><i class="fab fa-linkedin" aria-hidden="true"></i>LinkedIn</a>
        <a class="sv-quiet" href="https://t.me/qmarva" target="_blank" rel="noopener noreferrer"><i class="fab fa-telegram" aria-hidden="true"></i>Telegram</a>
      </div>
    </div>
  </section>
</div>

<script>
  // The address is put together here rather than written into the page, which keeps it from the simplest scrapers.
  document.querySelectorAll('.sv-copy').forEach((button) => {
    const address = `${button.dataset.u}@${button.dataset.d}`;
    const box = button.closest('.sv-address');
    box.querySelector('.sv-address-text').textContent = address;
    box.hidden = false;
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(address);
        button.textContent = 'Copied';
      } catch {
        button.textContent = 'Select and copy';
      }
      setTimeout(() => (button.textContent = 'Copy'), 2000);
    });
  });
</script>

<script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Service",
        "name": "Machine learning consulting",
        "serviceType": "Machine learning consulting",
        "description": "Feasibility checks, reviews of ML systems and roadmaps for computer vision, generative AI and language models.",
        "url": {{ '/services/#consulting' | absolute_url | jsonify }},
        "areaServed": "Worldwide",
        "provider": {"@type": "Person", "name": "Eva Koroleva", "url": {{ '/' | absolute_url | jsonify }}}
      },
      {
        "@type": "Service",
        "name": "Machine learning development",
        "serviceType": "Machine learning development",
        "description": "Prototypes and production systems for computer vision, generative AI and language models, including models for mobile and edge devices.",
        "url": {{ '/services/#development' | absolute_url | jsonify }},
        "areaServed": "Worldwide",
        "provider": {"@type": "Person", "name": "Eva Koroleva", "url": {{ '/' | absolute_url | jsonify }}}
      }
    ]
  }
</script>
