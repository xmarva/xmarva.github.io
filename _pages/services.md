---
layout: default
permalink: /services/
title: services
seo_title: Machine learning consulting and development
seo_description: "Machine learning consulting and development by Eva Koroleva: AI for finance, biomedicine and cybersecurity, and production ML systems, built with computer vision, generative AI and LLMs."
description: Machine learning consulting and development
# the hero's pictures, the top one first; drawn as HTML in _art/services/
pictures:
  - assets/img/services/finance.png
  - assets/img/services/biomed.png
  - assets/img/services/security.png
nav: true
nav_order: 3
---

<style>
  /* The teaching page's language: a hero card with pictures fanned out, the site's red for the one action, the
     portfolio's domain colours for the chips and the areas. */
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
    grid-template-columns: minmax(0, 1fr) minmax(0, 1.1fr);
    gap: 1.5rem;
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

  /* the areas' pictures: finance in front, the brain lower left, security above */
  .sv-fan { position: relative; display: block; aspect-ratio: 5 / 4; }
  .sv-fan-card {
    position: absolute;
    top: 50%;
    width: 74%;
    border-radius: 10px;
    overflow: hidden;
    /* a hairline keeps the dark windows apart from each other and from a dark card */
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.1), 0 1px 2px rgba(0, 0, 0, 0.15), 0 16px 36px rgba(0, 0, 0, 0.32);
    transform: translate(var(--x, 0%), calc(-50% + var(--y, 0%))) rotate(var(--r, 0deg));
    transition: transform 0.35s ease;
  }
  .sv-fan-card:nth-child(1) { right: 0; --y: 4%; z-index: 3; }
  .sv-fan-card:nth-child(2) { left: 0; --y: 22%; --r: -4deg; z-index: 2; filter: brightness(0.92); }
  .sv-fan-card:nth-child(3) { left: 9%; width: 66%; --y: -34%; --r: 3deg; z-index: 1; filter: brightness(0.82); }
  .sv-fan:hover .sv-fan-card:nth-child(2) { --x: -3%; --r: -6deg; }
  .sv-fan:hover .sv-fan-card:nth-child(3) { --y: -40%; --r: 4deg; }
  .sv-fan-card figure, .sv-fan-card picture { display: block; margin: 0; }
  .sv-fan-card img { display: block; width: 100%; height: auto; }

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

  /* what I do: four areas, an icon each */
  .sv-areas { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.1rem; }
  .sv-area {
    --c: var(--sv-muted);
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    gap: 0 1rem;
    padding: 1.35rem 1.4rem;
    border: 1px solid var(--sv-border);
    border-radius: 14px;
    background: var(--global-card-bg-color);
    transition: border-color 0.2s ease;
  }
  .sv-area:hover { border-color: color-mix(in srgb, var(--c) 50%, var(--sv-border)); }
  .sv-icon {
    display: grid;
    place-items: center;
    width: 2.75rem;
    height: 2.75rem;
    border-radius: 12px;
    font-size: 1.15rem;
    color: var(--c);
    background: color-mix(in srgb, var(--c) var(--sv-chip-alpha), transparent);
    box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--c) 22%, transparent);
  }
  .sv-area h3 { margin: 0.15rem 0 0.35rem; font-size: 1.12rem; font-weight: 500; color: var(--global-text-color); }
  .sv-area p { margin: 0; font-size: 0.95rem; line-height: 1.55; color: var(--global-text-color); }
  .sv-area .sv-tags { margin-top: 0.55rem; font-size: 0.85rem; color: var(--sv-muted); }
  .sv-finance { --c: var(--sv-cv); }
  .sv-bio { --c: var(--sv-bio); }
  .sv-security { --c: var(--sv-nlp); }
  .sv-production { --c: var(--sv-3d); }

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
    .sv-fan-card:nth-child(1) { position: relative; top: auto; width: 100%; transform: none; }
    .sv-fan-card:not(:first-child) { display: none; }
    .sv-hero h1 { font-size: 1.7rem; }
    .sv-contact { grid-template-columns: minmax(0, 1fr); gap: 1.4rem; padding: 1.5rem; }
  }
  @media (max-width: 680px) {
    .sv-areas { grid-template-columns: minmax(0, 1fr); }
    .sv-area { padding: 1.1rem 1.15rem; }
    .sv-steps { grid-template-columns: minmax(0, 1fr); gap: 1rem; }
  }
</style>

{% assign email_parts = site.email | split: "@" %}
{% capture mailto %}mailto:{{ site.email | encode_email }}?body=Hi%20Eva%2C%0A%0AWhat%20we%20are%20building%3A%0AWhat%20data%20we%20have%3A%0AWhen%20we%20need%20it%3A%0A{% endcapture %}

<div class="post sv sv-page">
  <section class="sv-hero" aria-labelledby="sv-title">
    <div class="sv-hero-text">
      <span class="sv-chips"><span class="sv-chip" data-domain="computer-vision">Computer Vision</span><span class="sv-chip" data-domain="genai">GenAI</span><span class="sv-chip" data-domain="nlp">LLM</span></span>
      <h1 id="sv-title">Machine learning consulting and development</h1>
      <p class="sv-summary">I help teams find out what machine learning can do for their product, and then build it, from a prototype on their data to a system in production.</p>
      <div class="sv-actions">
        <a class="sv-primary" href="#contact">Write to me <span aria-hidden="true">→</span></a>
        <a class="sv-quiet" href="{{ '/portfolio/' | relative_url }}"><i class="fas fa-briefcase" aria-hidden="true"></i>Portfolio</a>
      </div>
    </div>
    <div class="sv-fan" aria-hidden="true">
      {% for picture in page.pictures %}
      <span class="sv-fan-card" style="--i: {{ forloop.index0 }}">{% include figure.liquid path=picture alt="" sizes="(min-width: 860px) 360px, 90vw" loading="eager" %}</span>
      {% endfor %}
    </div>
  </section>

  <div class="sv-head"><h2 id="sv-what">What I do</h2></div>
  <div class="sv-areas" role="list" aria-labelledby="sv-what">
    <article class="sv-area sv-finance" role="listitem" id="finance">
      <span class="sv-icon" aria-hidden="true"><i class="fa-solid fa-chart-line"></i></span>
      <div>
        <h3>AI for finance</h3>
        <p>Assistants for traders and analysts that read markets, portfolios and reports, in crypto and traditional markets.</p>
        <p class="sv-tags">Market assistants · forecasting · risk · on-chain data</p>
      </div>
    </article>
    <article class="sv-area sv-bio" role="listitem" id="biomedicine">
      <span class="sv-icon" aria-hidden="true"><i class="fa-solid fa-brain"></i></span>
      <div>
        <h3>AI for biomedicine</h3>
        <p>Models for clinical text, medical images and signals, and tools for psychiatry that follow a patient between visits.</p>
        <p class="sv-tags">Clinical NLP · medical imaging · psychiatry · knowledge graphs</p>
      </div>
    </article>
    <article class="sv-area sv-security" role="listitem" id="cybersecurity">
      <span class="sv-icon" aria-hidden="true"><i class="fa-solid fa-shield-halved"></i></span>
      <div>
        <h3>AI for cybersecurity</h3>
        <p>Anomaly detection in logs and traffic, phishing and malware classification, assistants that help analysts triage incidents.</p>
        <p class="sv-tags">Log anomalies · phishing · threat intelligence · LLM agents</p>
      </div>
    </article>
    <article class="sv-area sv-production" role="listitem" id="production">
      <span class="sv-icon" aria-hidden="true"><i class="fa-solid fa-server"></i></span>
      <div>
        <h3>Production systems</h3>
        <p>Models that keep working after the demo: serving, monitoring and pipelines, and models made small and fast for edge devices.</p>
        <p class="sv-tags">Serving · MLOps · monitoring · edge inference</p>
      </div>
    </article>
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
        "name": "AI for finance",
        "serviceType": "AI for finance",
        "description": "Market and portfolio assistants, forecasting, risk and anomaly detection for crypto and traditional markets.",
        "url": {{ '/services/#finance' | absolute_url | jsonify }},
        "areaServed": "Worldwide",
        "provider": {"@type": "Person", "name": "Eva Koroleva", "url": {{ '/' | absolute_url | jsonify }}}
      },
      {
        "@type": "Service",
        "name": "AI for biomedicine",
        "serviceType": "AI for biomedicine",
        "description": "Models for clinical text, medical images and signals, and tools for psychiatry.",
        "url": {{ '/services/#biomedicine' | absolute_url | jsonify }},
        "areaServed": "Worldwide",
        "provider": {"@type": "Person", "name": "Eva Koroleva", "url": {{ '/' | absolute_url | jsonify }}}
      },
      {
        "@type": "Service",
        "name": "AI for cybersecurity",
        "serviceType": "AI for cybersecurity",
        "description": "Anomaly detection in logs and network traffic, phishing and malware classification, assistants for security analysts.",
        "url": {{ '/services/#cybersecurity' | absolute_url | jsonify }},
        "areaServed": "Worldwide",
        "provider": {"@type": "Person", "name": "Eva Koroleva", "url": {{ '/' | absolute_url | jsonify }}}
      },
      {
        "@type": "Service",
        "name": "Production ML systems",
        "serviceType": "Production ML systems",
        "description": "Serving, monitoring, pipelines and edge inference for machine learning models.",
        "url": {{ '/services/#production' | absolute_url | jsonify }},
        "areaServed": "Worldwide",
        "provider": {"@type": "Person", "name": "Eva Koroleva", "url": {{ '/' | absolute_url | jsonify }}}
      }
    ]
  }
</script>
