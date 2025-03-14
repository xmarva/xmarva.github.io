---
layout: about
title: about
permalink: /
subtitle:
profile:
  align: right
  image: eva_koroleva.jpg
  image_circular: true # crops the image to make it circular
  more_info: >
    <p>📍 Buenos Aires, Argentina </p>
news: true # includes a list of news items
selected_papers: false # includes a list of papers marked as "selected={true}"
social: true # includes social icons at the bottom of the page
---

/* Projects Section Styling */
.projects-section {
  margin: 1.5rem 0 2.5rem 0;
}

.project-card {
  background-color: var(--global-bg-color);
  border-radius: 10px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  border: 1px solid rgba(var(--global-theme-color-rgb), 0.2);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.project-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08);
}

.project-card h3 {
  margin-top: 0;
  margin-bottom: 0.75rem;
  font-family: 'Playfair Display', var(--global-serif-font-family), serif;
  font-weight: 600;
}

.project-card p {
  margin-bottom: 0.75rem;
  color: var(--global-text-color-light);
}

.project-card p:last-child {
  margin-bottom: 0;
}

.project-card a {
  color: var(--global-theme-color);
  text-decoration: none;
  font-weight: 500;
  transition: color 0.2s ease;
}

.project-card a:hover {
  color: var(--global-hover-color);
  text-decoration: underline;
}

/* Latest Posts Section Styling */
.latest-posts {
  margin: 1.5rem 0 2.5rem 0;
}

.post-item {
  margin-bottom: 1.25rem;
  padding-bottom: 1.25rem;
  border-bottom: 1px solid var(--global-divider-color);
}

.post-item:last-child {
  border-bottom: none;
}

.post-item h3 {
  margin: 0;
  font-size: 1.25rem;
  font-family: 'Playfair Display', var(--global-serif-font-family), serif;
}

.post-item h3 a {
  color: var(--global-text-color);
  text-decoration: none;
  transition: color 0.2s ease;
}

.post-item h3 a:hover {
  color: var(--global-theme-color);
}

.post-date {
  font-size: 0.9rem;
  color: var(--global-text-color-light);
  margin-top: 0.25rem;
  margin-bottom: 0;
}

.latest-posts > p {
  margin-top: 1.5rem;
}

.latest-posts > p a {
  color: var(--global-theme-color);
  font-weight: 500;
  text-decoration: none;
  transition: color 0.2s ease;
}

.latest-posts > p a:hover {
  color: var(--global-hover-color);
  text-decoration: underline;
}

/* Section Headings */
h2 {
  font-family: 'Playfair Display', var(--global-serif-font-family), serif;
  font-size: 1.75rem;
  font-weight: 600;
  margin: 2.5rem 0 1rem 0;
  color: var(--global-theme-color);
}

Hey there, welcome to my website!

I'm an 👩‍💻 **ML Engineer & Researcher** working with: 

&nbsp;&nbsp;&nbsp;&nbsp; 💬 Natural Language Processing  
&nbsp;&nbsp;&nbsp;&nbsp; 👀 Computer Vision  
&nbsp;&nbsp;&nbsp;&nbsp; 🚀 Generative Models

I'm also interested in **Cognitive Science** and **Philosophy of Mind**.

What I love is creating ✨**beautiful**✨ solutions — clean code, creative ideas, efficient models, and solid results.

It would be very bold to claim I always succeed. Sometimes it's a bit messy (who hasn't been there?), but I'm always chasing that ideal.

So, can develop and train models to generalize your data and generate new ones to solve practical tasks. But who in AI doesn't secretly hope to solve the problem of consciousness and play a part in creating strong AI?

Outside of work I enjoy nature, hiking, staying active, cooking, learning languages, and exploring mindfulness through therapy. Good work-life balance and happiness is key to productive thinking.

Also, salmon nigiri might just be the peak of human existence 🍣.

## Current Projects

<div class="projects-section">
  <div class="project-card">
    <h3>LLM-Powered Agent Framework</h3>
    <p>Building a modular framework for autonomous LLM agents with custom tools and memory systems.</p>
    <p><strong>Tech:</strong> Python, LangChain, HuggingFace Transformers, Redis</p>
    <p><a href="/projects/llm-agents">View Project →</a></p>
  </div>
  
  <div class="project-card">
    <h3>Multimodal Foundation Model Fine-tuning</h3>
    <p>Fine-tuning foundation models for specialized domain applications with efficient low-rank adaptation techniques.</p>
    <p><strong>Tech:</strong> PyTorch, PEFT, LoRA, Vision-Language Models</p>
    <p><a href="/projects/multimodal-finetuning">View Project →</a></p>
  </div>
</div>

## News

<div class="news">
  {% if site.news != blank -%}
  {%- include news.html %}
  {%- endif %}
</div>

## Latest Posts

<div class="latest-posts">
  {% assign latest_posts = site.posts | slice: 0, 3 %}
  {% for post in latest_posts %}
  <div class="post-item">
    <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
    <p class="post-date">{{ post.date | date: "%B %d, %Y" }}</p>
  </div>
  {% endfor %}
  <p><a href="{{ '/blog/' | relative_url }}">View All Posts →</a></p>
</div>