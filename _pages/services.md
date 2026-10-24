---
layout: default
permalink: /services/
title: services
seo_title: Machine learning consulting and development
seo_description: "Machine learning consulting and development by Eva Koroleva: computer vision, AI for finance, biomedicine and cybersecurity, and production ML systems, with generative AI and LLMs."
description: Machine learning consulting and development
# the hero's pictures: the front one, then the left and the right; sources in _art/services/
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

  /* hero: the title across the card, a line and the actions under it, the pictures rising from the bottom edge */
  .sv-hero {
    border: 1px solid var(--sv-border);
    border-radius: 16px;
    background: var(--global-card-bg-color);
    overflow: hidden;
  }
  .sv-hero-text { padding: 2.4rem 2.4rem 0; }
  .sv-hero h1 { margin: 0.9rem 0 0.9rem; font-size: 2.4rem; font-weight: 500; line-height: 1.15; color: var(--global-text-color); text-wrap: balance; }
  .sv-hero-row { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 1rem 2rem; }
  .sv-summary { margin: 0; max-width: 40rem; font-size: 1.08rem; line-height: 1.6; color: var(--sv-muted); }
  .sv-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem 1.25rem; }
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

  /* the stage: the exchange in front, the brain to the left, CyberChef to the right; the card's edge crops them */
  .sv-stage { position: relative; height: 22rem; margin-top: 2.2rem; }
  .sv-shot {
    position: absolute;
    border-radius: 10px;
    overflow: hidden;
    /* a hairline keeps the dark windows apart from each other and from a dark card */
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.1), 0 1px 2px rgba(0, 0, 0, 0.15), 0 18px 40px rgba(0, 0, 0, 0.35);
    transition: transform 0.35s ease;
  }
  .sv-shot figure, .sv-shot picture { display: block; margin: 0; }
  .sv-shot img { display: block; width: 100%; height: auto; }
  .sv-shot:nth-child(1) { left: 22%; width: 56%; top: 0; z-index: 3; }
  .sv-shot:nth-child(2) { left: 2.5%; width: 46%; top: 3.6rem; z-index: 2; transform: rotate(-3deg); filter: brightness(0.9); }
  .sv-shot:nth-child(3) { right: 2.5%; width: 46%; top: 3.6rem; z-index: 1; transform: rotate(3deg); filter: brightness(0.9); }
  .sv-stage:hover .sv-shot:nth-child(1) { transform: translateY(-6px); }
  .sv-stage:hover .sv-shot:nth-child(2) { transform: rotate(-4deg) translateX(-8px); }
  .sv-stage:hover .sv-shot:nth-child(3) { transform: rotate(4deg) translateX(8px); }

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

  /* what I do: a row per area, an icon each, as the lessons on /teaching/ */
  .sv-areas { list-style: none; margin: 0; padding: 0; border-top: 1px solid var(--sv-border); }
  .sv-area {
    --c: var(--sv-muted);
    display: grid;
    grid-template-columns: 2.75rem minmax(0, 1fr);
    column-gap: 1.25rem;
    padding: 1.2rem 0.25rem;
    border-bottom: 1px solid var(--sv-border);
    scroll-margin-top: 5rem;
  }
  .sv-icon {
    display: grid;
    place-items: center;
    width: 2.75rem;
    height: 2.75rem;
    border-radius: 12px;
    font-size: 1.1rem;
    color: var(--c);
    background: color-mix(in srgb, var(--c) var(--sv-chip-alpha), transparent);
    box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--c) 22%, transparent);
  }
  .sv-area h3 { margin: 0.1rem 0 0.3rem; font-size: 1.12rem; font-weight: 500; color: var(--global-text-color); }
  .sv-area p { margin: 0; font-size: 0.98rem; line-height: 1.55; color: var(--global-text-color); }
  .sv-cv-area { --c: var(--sv-cv); }
  .sv-finance { --c: var(--sv-3d); }
  .sv-bio { --c: var(--sv-bio); }
  .sv-security { --c: var(--sv-nlp); }
  .sv-production { --c: var(--sv-genai); }

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
    .sv-hero-text { padding: 1.5rem 1.5rem 0; }
    .sv-hero h1 { font-size: 1.8rem; }
    .sv-stage { height: auto; margin-top: 1.5rem; padding: 0 1.5rem; }
    .sv-shot:nth-child(1) { position: relative; left: auto; width: 100%; margin-bottom: -1.2rem; }
    .sv-shot:not(:first-child) { display: none; }
    .sv-contact { grid-template-columns: minmax(0, 1fr); gap: 1.4rem; padding: 1.5rem; }
  }
  @media (max-width: 680px) {
    .sv-area { grid-template-columns: 2.4rem minmax(0, 1fr); column-gap: 1rem; }
    .sv-icon { width: 2.4rem; height: 2.4rem; font-size: 1rem; }
  }
</style>

{% assign email_parts = site.email | split: "@" %}
{% capture mailto %}mailto:{{ site.email | encode_email }}?body=Hi%20Eva%2C%0A%0AWhat%20we%20are%20building%3A%0AWhat%20data%20we%20have%3A%0AWhen%20we%20need%20it%3A%0A{% endcapture %}

<div class="post sv sv-page">
  <section class="sv-hero" aria-labelledby="sv-title">
    <div class="sv-hero-text">
      <span class="sv-chips"><span class="sv-chip" data-domain="computer-vision">Computer Vision</span><span class="sv-chip" data-domain="genai">GenAI</span><span class="sv-chip" data-domain="nlp">LLM</span></span>
      <h1 id="sv-title">Machine learning consulting and development</h1>
      <div class="sv-hero-row">
        <p class="sv-summary">I'm an ML engineer. I design and build models for finance, biomedicine and cybersecurity, and run them in production.</p>
        <div class="sv-actions">
          <a class="sv-primary" href="#contact">Write to me <span aria-hidden="true">→</span></a>
          <a class="sv-quiet" href="{{ '/portfolio/' | relative_url }}"><i class="fas fa-briefcase" aria-hidden="true"></i>Portfolio</a>
        </div>
      </div>
    </div>
    <div class="sv-stage" aria-hidden="true">
      {% for picture in page.pictures %}
      <span class="sv-shot">{% include figure.liquid path=picture alt="" sizes="(min-width: 860px) 560px, 90vw" loading="eager" %}</span>
      {% endfor %}
    </div>
  </section>

  <div class="sv-head"><h2 id="sv-what">What I do</h2></div>
  <ul class="sv-areas" aria-labelledby="sv-what">
    <li class="sv-area sv-cv-area" id="computer-vision">
      <span class="sv-icon" aria-hidden="true"><i class="fa-solid fa-eye"></i></span>
      <div>
        <h3>Computer vision</h3>
        <p>Detection, segmentation and tracking in images and video. GenAI and 3D model adaptation.</p>
      </div>
    </li>
    <li class="sv-area sv-finance" id="finance">
      <span class="sv-icon" aria-hidden="true"><i class="fa-solid fa-chart-line"></i></span>
      <div>
        <h3>AI for finance</h3>
        <p>Assistants and custom software for traders and analysts that read markets, portfolios and reports.</p>
      </div>
    </li>
    <li class="sv-area sv-bio" id="biomedicine">
      <span class="sv-icon" aria-hidden="true"><i class="fa-solid fa-brain"></i></span>
      <div>
        <h3>AI for biomedicine</h3>
        <p>Models for clinical text, medical images and signals, and tools for neuroscience.</p>
      </div>
    </li>
    <li class="sv-area sv-security" id="cybersecurity">
      <span class="sv-icon" aria-hidden="true"><i class="fa-solid fa-shield-halved"></i></span>
      <div>
        <h3>AI for cybersecurity</h3>
        <p>Anomaly detection in logs and traffic, phishing and malware classification.</p>
      </div>
    </li>
    <li class="sv-area sv-production" id="production">
      <span class="sv-icon" aria-hidden="true"><i class="fa-solid fa-server"></i></span>
      <div>
        <h3>Production systems</h3>
        <p>Serving, monitoring and pipelines, and models made small and fast for edge devices.</p>
      </div>
    </li>
  </ul>

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
        "name": "Computer vision",
        "serviceType": "Computer vision",
        "description": "Detection, segmentation and tracking in images and video. GenAI and 3D model adaptation.",
        "url": {{ '/services/#computer-vision' | absolute_url | jsonify }},
        "areaServed": "Worldwide",
        "provider": {"@type": "Person", "name": "Eva Koroleva", "url": {{ '/' | absolute_url | jsonify }}}
      },
      {
        "@type": "Service",
        "name": "AI for finance",
        "serviceType": "AI for finance",
        "description": "Assistants and custom software for traders and analysts that read markets, portfolios and reports.",
        "url": {{ '/services/#finance' | absolute_url | jsonify }},
        "areaServed": "Worldwide",
        "provider": {"@type": "Person", "name": "Eva Koroleva", "url": {{ '/' | absolute_url | jsonify }}}
      },
      {
        "@type": "Service",
        "name": "AI for biomedicine",
        "serviceType": "AI for biomedicine",
        "description": "Models for clinical text, medical images and signals, and tools for neuroscience.",
        "url": {{ '/services/#biomedicine' | absolute_url | jsonify }},
        "areaServed": "Worldwide",
        "provider": {"@type": "Person", "name": "Eva Koroleva", "url": {{ '/' | absolute_url | jsonify }}}
      },
      {
        "@type": "Service",
        "name": "AI for cybersecurity",
        "serviceType": "AI for cybersecurity",
        "description": "Anomaly detection in logs and traffic, phishing and malware classification.",
        "url": {{ '/services/#cybersecurity' | absolute_url | jsonify }},
        "areaServed": "Worldwide",
        "provider": {"@type": "Person", "name": "Eva Koroleva", "url": {{ '/' | absolute_url | jsonify }}}
      },
      {
        "@type": "Service",
        "name": "Production ML systems",
        "serviceType": "Production ML systems",
        "description": "Serving, monitoring and pipelines, and models made small and fast for edge devices.",
        "url": {{ '/services/#production' | absolute_url | jsonify }},
        "areaServed": "Worldwide",
        "provider": {"@type": "Person", "name": "Eva Koroleva", "url": {{ '/' | absolute_url | jsonify }}}
      }
    ]
  }
</script>
