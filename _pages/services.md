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
  /* The portfolio's system: the site's red for the one action, a hue per domain, cards that follow the theme. */
  .sv {
    --sv-border: var(--global-divider-color);
    --sv-muted: var(--global-text-color-light);
    --sv-cv: #0f8c7e;
    --sv-genai: #6a55d8;
    --sv-3d: #b86e00;
    --sv-nlp: #3567d0;
  }
  html[data-theme="dark"] .sv {
    --sv-cv: #4cc9b8;
    --sv-genai: #a596f7;
    --sv-3d: #f0b04a;
    --sv-nlp: #86a8f7;
  }

  .sv-hero { margin: 3.2rem 0 3.5rem; max-width: 46rem; }
  .sv-eyebrow {
    margin: 0 0 0.9rem;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--sv-muted);
  }
  .sv-hero h1 { margin: 0 0 1rem; font-size: 2.4rem; font-weight: 500; line-height: 1.15; color: var(--global-text-color); }
  .sv-lead { margin: 0 0 1.6rem; font-size: 1.12rem; line-height: 1.65; color: var(--sv-muted); }
  .sv-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem 1.4rem; }
  .sv-primary {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.5rem 1.15rem;
    border: 0;
    border-radius: 999px;
    background: var(--global-theme-color);
    color: var(--global-bg-color) !important;
    font-size: 0.95rem;
    font-weight: 500;
    text-decoration: none !important;
    cursor: pointer;
    transition: filter 0.2s ease;
  }
  .sv-primary:hover { filter: brightness(1.1); }
  .sv-quiet { font-size: 0.95rem; color: var(--global-text-color) !important; text-decoration: none !important; }
  .sv-quiet:hover { color: var(--global-theme-color) !important; }
  .sv-quiet i { margin-right: 0.35rem; }

  .sv-section { margin-bottom: 3.5rem; }
  .sv-head { margin-bottom: 1.4rem; }
  .sv-head::before {
    content: "";
    display: block;
    width: 28px;
    height: 3px;
    margin-bottom: 0.75rem;
    background: var(--global-theme-color);
  }
  .sv-head h2 { margin: 0; font-size: 1.45rem; font-weight: 500; color: var(--global-text-color); }

  .sv-offers { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.25rem; }
  .sv-offer {
    display: flex;
    flex-direction: column;
    padding: 1.6rem 1.6rem 1.4rem;
    border: 1px solid var(--sv-border);
    border-radius: 14px;
    background: var(--global-card-bg-color);
    scroll-margin-top: 5rem;
  }
  .sv-offer h3 { margin: 0 0 0.4rem; font-size: 1.3rem; font-weight: 500; color: var(--global-text-color); }
  .sv-offer > p { margin: 0 0 1.1rem; color: var(--sv-muted); line-height: 1.6; }
  .sv-offer ul { margin: 0 0 1.4rem; padding: 0; list-style: none; }
  .sv-offer li { position: relative; padding: 0.42rem 0 0.42rem 1.3rem; line-height: 1.5; color: var(--global-text-color); }
  .sv-offer li::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0.95rem;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    border: 1.5px solid var(--global-theme-color);
  }
  .sv-offer .sv-price {
    margin: auto 0 0;
    padding-top: 1rem;
    border-top: 1px solid var(--sv-border);
    font-size: 1.1rem;
    font-weight: 500;
    color: var(--global-text-color);
  }
  .sv-price span { margin-left: 0.2rem; font-size: 0.9rem; font-weight: 400; color: var(--sv-muted); }

  .sv-areas { margin: 0; padding: 0; list-style: none; border-top: 1px solid var(--sv-border); }
  .sv-area {
    display: grid;
    grid-template-columns: 12rem minmax(0, 1fr);
    gap: 1rem;
    align-items: baseline;
    padding: 0.85rem 0.25rem;
    border-bottom: 1px solid var(--sv-border);
  }
  .sv-area-name { display: inline-flex; align-items: center; gap: 0.55rem; font-weight: 500; color: var(--global-text-color); }
  .sv-area-name::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--c, var(--sv-muted)); }
  .sv-area-what { color: var(--sv-muted); line-height: 1.5; }

  .sv-work { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem; }
  .sv-case {
    display: flex;
    flex-direction: column;
    border: 1px solid var(--sv-border);
    border-radius: 12px;
    overflow: hidden;
    background: var(--global-card-bg-color);
    color: var(--global-text-color) !important;
    text-decoration: none !important;
    transition: border-color 0.2s ease;
  }
  .sv-case:hover { border-color: color-mix(in srgb, var(--global-theme-color) 45%, var(--sv-border)); }
  .sv-case-media { aspect-ratio: 16 / 10; overflow: hidden; background: #161618; }
  .sv-case-media figure, .sv-case-media picture { display: block; height: 100%; margin: 0; }
  .sv-case-media img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .sv-case-body { padding: 0.8rem 0.9rem 0.95rem; }
  .sv-case-title { margin: 0; font-size: 0.98rem; font-weight: 500; line-height: 1.35; }
  .sv-case:hover .sv-case-title { color: var(--global-theme-color); }

  .sv-steps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1.5rem; margin: 0; padding: 0; list-style: none; counter-reset: step; }
  .sv-step { counter-increment: step; }
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

  .sv-contact {
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
    gap: 2rem;
    align-items: center;
    padding: 2rem 2.2rem;
    border: 1px solid var(--sv-border);
    border-radius: 16px;
    background: var(--global-card-bg-color);
    scroll-margin-top: 5rem;
  }
  .sv-contact h2 { margin: 0 0 0.5rem; font-size: 1.45rem; font-weight: 500; color: var(--global-text-color); }
  .sv-contact p { margin: 0; color: var(--sv-muted); line-height: 1.6; }
  .sv-channels { display: flex; flex-direction: column; gap: 0.65rem; align-items: flex-start; }
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
  .sv-others { display: flex; flex-wrap: wrap; gap: 0.5rem 1.4rem; margin-top: 0.2rem; }

  @media (max-width: 860px) {
    .sv-work { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .sv-contact { grid-template-columns: minmax(0, 1fr); gap: 1.4rem; padding: 1.5rem; }
  }
  @media (max-width: 680px) {
    .sv-hero { margin-top: 2rem; }
    .sv-hero h1 { font-size: 1.9rem; }
    .sv-offers, .sv-steps { grid-template-columns: minmax(0, 1fr); }
    .sv-area { grid-template-columns: minmax(0, 1fr); gap: 0.2rem; }
  }
</style>

{% assign email_parts = site.email | split: "@" %}
{% capture mailto %}mailto:{{ site.email | encode_email }}?subject=Project%20inquiry&body=Hi%20Eva%2C%0A%0AWhat%20we%20are%20building%3A%0AWhat%20data%20we%20have%3A%0AWhen%20we%20need%20it%3A%0A{% endcapture %}

<div class="post sv">
  <section class="sv-hero">
    <p class="sv-eyebrow">Consulting · Development</p>
    <h1>Machine learning consulting and development</h1>
    <p class="sv-lead">I help teams find out what machine learning can do for their product, and then build it: computer vision, generative models and language models, from a prototype on your data to a system in production.</p>
    <div class="sv-actions">
      <a class="sv-primary" href="#contact">Write to me <span aria-hidden="true">→</span></a>
      <a class="sv-quiet" href="{{ '/portfolio/' | relative_url }}">See the case studies</a>
    </div>
  </section>

  <section class="sv-section" aria-labelledby="sv-offers">
    <div class="sv-head"><h2 id="sv-offers">Two ways to work together</h2></div>
    <div class="sv-offers">
      <article class="sv-offer" id="consulting">
        <h3>Consulting</h3>
        <p>When you need to decide what to build, or find out why something doesn't work.</p>
        <ul>
          <li>Feasibility: whether machine learning can solve the problem, with what data and at what cost</li>
          <li>A review of a model or an ML system that is already running</li>
          <li>A roadmap: what to build first, what to buy and what to skip</li>
        </ul>
        <p class="sv-price">$100 <span>per hour</span></p>
      </article>
      <article class="sv-offer" id="development">
        <h3>Development</h3>
        <p>A prototype or a production system, built with a small team of engineers I have worked with.</p>
        <ul>
          <li>A prototype that tests the idea on your data</li>
          <li>A production service with an API, tests and monitoring</li>
          <li>Models made small and fast for phones and edge devices</li>
          <li>A model that is too slow or not accurate enough, fixed</li>
        </ul>
        <p class="sv-price">Priced per project <span>after a first call</span></p>
      </article>
    </div>
  </section>

  <section class="sv-section" aria-labelledby="sv-areas">
    <div class="sv-head"><h2 id="sv-areas">What I work with</h2></div>
    <ul class="sv-areas">
      <li class="sv-area" style="--c: var(--sv-cv)"><span class="sv-area-name">Computer vision</span><span class="sv-area-what">Detection, segmentation, tracking, multi-camera systems</span></li>
      <li class="sv-area" style="--c: var(--sv-genai)"><span class="sv-area-name">Generative AI</span><span class="sv-area-what">Diffusion models, virtual try-on, fine-tuning language models, retrieval-augmented generation</span></li>
      <li class="sv-area" style="--c: var(--sv-3d)"><span class="sv-area-name">3D</span><span class="sv-area-what">Reconstruction, human bodies, floor plans turned into 3D scenes</span></li>
      <li class="sv-area" style="--c: var(--sv-nlp)"><span class="sv-area-name">Language</span><span class="sv-area-what">Entity extraction, medical texts, knowledge graphs, translation</span></li>
      <li class="sv-area"><span class="sv-area-name">Production</span><span class="sv-area-what">Serving, quantization, inference on edge devices, monitoring</span></li>
    </ul>
  </section>

  <section class="sv-section" aria-labelledby="sv-work">
    <div class="sv-head"><h2 id="sv-work">Case studies</h2></div>
    <div class="sv-work">
      {% assign work_projects = site.projects | where: "category", "work" | sort: "importance" %}
      {% for project in work_projects %}
      <a class="sv-case" href="{{ project.url | relative_url }}">
        <span class="sv-case-media">{% if project.img %}{% include figure.liquid path=project.img alt="" sizes="(min-width: 860px) 220px, 50vw" %}{% endif %}</span>
        <span class="sv-case-body"><span class="sv-case-title">{{ project.title }}</span></span>
      </a>
      {% endfor %}
    </div>
  </section>

  <section class="sv-section" aria-labelledby="sv-how">
    <div class="sv-head"><h2 id="sv-how">How it goes</h2></div>
    <ol class="sv-steps">
      <li class="sv-step">
        <h3>A call</h3>
        <p>You describe the problem. I ask about the data, the constraints and what a good result looks like.</p>
      </li>
      <li class="sv-step">
        <h3>A proposal</h3>
        <p>A written plan: the approach, the risks, the timeline and the price.</p>
      </li>
      <li class="sv-step">
        <h3>The work</h3>
        <p>In short iterations, each ending with something you can run and judge.</p>
      </li>
    </ol>
  </section>

  <section class="sv-section sv-contact" id="contact" aria-labelledby="sv-contact">
    <div>
      <h2 id="sv-contact">Contact</h2>
      <p>Tell me what you are building, what data you have and when you need it. I work in English and Russian.</p>
    </div>
    <div class="sv-channels">
      <a class="sv-primary" href="{{ mailto }}"><i class="fas fa-envelope" aria-hidden="true"></i> Email me</a>
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
        "provider": {"@type": "Person", "name": "Eva Koroleva", "url": {{ '/' | absolute_url | jsonify }}},
        "offers": {
          "@type": "Offer",
          "priceSpecification": {"@type": "UnitPriceSpecification", "price": 100, "priceCurrency": "USD", "unitText": "hour"}
        }
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
