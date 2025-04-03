---
layout: page
title: portfolio
permalink: /portfolio/
description: Welcome to my project collection
nav: true
nav_order: 2
display_categories: [work, personal]
---

<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Portfolio</title>
  <style>
    /* Base styling for projects page */
    .projects-wrapper {
      font-family: var(--global-font-family), sans-serif;
      color: var(--global-text-color);
      max-width: 100%;
      margin: 0 auto;
    }

    /* Section styling */
    .project-section {
      margin-bottom: 4rem;
    }

    .section-title {
      font-family: var(--global-serif-font-family), serif;
      font-size: 2.4rem;
      font-weight: normal;
      color: var(--global-theme-color);
      margin-bottom: 2rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid var(--global-divider-color);
      text-decoration: none;
      pointer-events: none;
    }

    /* Project card styling */
    .project-list {
      display: flex;
      flex-direction: column;
      gap: 2.5rem;
    }

    .project-card {
      background-color: var(--global-bg-color);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
      transition: transform 0.2s ease, box-shadow 0.3s ease;
      height: 220px;
      width: 100%;
    }

    .project-card:hover {
      box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08);
      transform: translateY(-3px);
      background-color: var(--card-hover, rgba(0, 0, 0, 0.03));
    }

    html[data-theme="dark"] .project-card:hover {
      background-color: var(--card-hover, rgba(255, 255, 255, 0.05));
    }

    .project-content {
      display: flex;
      flex-direction: row;
      height: 100%;
    }

    /* Project image */
    .project-image {
      flex: 0 0 300px;
      width: 300px;
      height: 220px;
      overflow: hidden;
      position: relative;
    }

    .project-image img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      object-position: center;
      display: block;
    }

    .no-image {
      background-color: var(--global-code-bg-color);
      height: 100%;
    }

    /* Project details */
    .project-details {
      flex: 1;
      padding: 1.2rem 1.5rem; /* Уменьшен внутренний отступ */
      display: flex;
      flex-direction: column;
      position: relative;
    }

    /* Domain tags in top right corner */
    .project-domains {
      position: absolute;
      top: 0.7rem; /* Уменьшен отступ сверху */
      right: 1.5rem;
      display: flex;
      flex-direction: row;
      align-items: center;
      gap: 0.4rem;
      max-width: 70%;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .domain-tag {
      display: inline-block;
      padding: 0.25rem 0.7rem;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 500;
      background-color: transparent;
      color: var(--global-theme-color);
      border: 1px solid var(--global-theme-color);
      white-space: nowrap;
    }

    .project-title {
      font-family: var(--global-serif-font-family), serif;
      font-size: 1.7rem;
      font-weight: normal;
      margin: 0 0 0.6rem 0; /* Уменьшен верхний отступ */
      letter-spacing: -0.02em;
      padding-right: 6rem; /* Make space for domain tags */
      padding-top: 0; /* Убран верхний отступ */
    }

    .project-title a {
      color: var(--global-text-color);
      text-decoration: none;
      transition: color 0.2s ease;
    }

    .project-title a:hover {
      color: var(--global-theme-color);
    }

    .project-role {
      display: none;
    }

    .project-description {
      font-size: 1rem;
      line-height: 1.6;
      margin-bottom: 0.5rem; /* Уменьшен нижний отступ */
      color: var(--global-text-color-light);
      overflow: hidden;
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
    }

    /* Кнопки Read More и Take a Peek удалены */

    /* Project metadata (technologies, links) */
    .project-meta {
      margin-top: auto;
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
    }

    .project-tech {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 0.5rem; /* Добавлен отступ сверху */
    }

    .tech-tag {
      display: inline-block;
      padding: 0.25rem 0.7rem;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: normal;
      background-color: transparent;
      color: var(--global-theme-color);
      border: 1px solid var(--global-theme-color);
      transition: transform 0.2s ease;
    }

    .tech-tag:hover {
      transform: translateY(-2px);
    }

    /* Project links */
    .project-links {
      position: absolute;
      bottom: 1.5rem;
      right: 1.5rem;
      display: flex;
      gap: 1rem;
    }

    .project-link {
      display: inline-flex;
      align-items: center;
      color: var(--global-theme-color); /* Изменен цвет значков на цвет темы */
      text-decoration: none;
      font-weight: normal;
      font-size: 0.9rem;
      transition: transform 0.2s ease, opacity 0.2s ease;
    }

    .project-link:hover {
      transform: translateY(-2px);
      opacity: 0.8;
    }

    .project-link i {
      font-size: 1.3rem; /* Увеличен размер иконок */
    }

    /* Убраны подписи к иконкам (текст GitHub, Kaggle, Website) */
    .project-link span {
      display: none;
    }

    /* Resume button styling */
    .resume-btn-container {
      display: flex;
      justify-content: flex-end;
      margin-top: -3.5rem;
      margin-bottom: 2rem;
      position: relative;
      z-index: 10;
    }

    .resume-btn {
      display: inline-flex;
      align-items: center;
      padding: 0.4rem 1rem;
      background-color: var(--global-theme-color);
      color: var(--global-bg-color);
      border-radius: 20px;
      font-size: 0.9rem;
      font-weight: normal;
      text-decoration: none;
      position: relative;
      overflow: hidden;
      z-index: 1;
      transition: transform 0.3s ease;
    }

    .resume-btn:before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-color: rgba(0, 0, 0, 0.2);
      transform: scaleX(0);
      transform-origin: right;
      transition: transform 0.3s ease;
      z-index: -1;
    }

    .resume-btn:hover {
      transform: translateY(-3px);
    }

    .resume-btn:hover:before {
      transform: scaleX(1);
      transform-origin: left;
    }

    .resume-btn i {
      margin-right: 0.5rem;
    }

    /* Media queries for responsiveness */
    @media (max-width: 900px) {
      .project-card {
        height: auto;
        min-height: 440px;
      }
      
      .project-content {
        flex-direction: column;
      }
      
      .project-image {
        flex: none;
        width: 100%;
        height: 200px;
      }
      
      .project-details {
        padding: 1.5rem;
        min-height: 240px;
      }
      
      .project-domains {
        position: relative;
        top: 0;
        right: 0;
        flex-direction: row;
        justify-content: flex-start;
        max-width: 100%;
        margin-bottom: 0.8rem;
      }
      
      .project-title {
        padding-right: 0;
        margin-top: 0.5rem;
      }
      
      .project-links {
        position: relative;
        bottom: 0;
        right: 0;
        margin-top: 1rem;
      }
      
      .resume-btn-container {
        margin-top: 1rem;
        justify-content: flex-start;
      }
    }
  </style>
</head>
<body>
  <!-- Resume button container -->
  <div class="resume-btn-container">
    <a href="https://xmarva.github.io/assets/pdf/eva_koroleva_cv.pdf" class="resume-btn" target="_blank">
      <i class="fas fa-file-download"></i> Download Resume
    </a>
  </div>

  <div class="projects-wrapper">
    <!-- Work projects section -->
    <section id="work" class="project-section">
      <h2 class="section-title">what I worked on</h2>
      <div class="project-list">
        {% assign work_projects = site.projects | where: "category", "work" | sort: "importance" %}
        {% for project in work_projects %}
        <div class="project-card">
          <div class="project-content">
            <div class="project-image">
              {% if project.img %}
              <a href="{{ project.url | relative_url }}">
                <img src="{{ project.img | relative_url }}" alt="{{ project.title }}" />
              </a>
              {% else %}
              <a href="{{ project.url | relative_url }}">
                <div class="no-image"></div>
              </a>
              {% endif %}
            </div>
            <div class="project-details">
              {% if project.domains %}
              <div class="project-domains">
                {% for domain in project.domains %}
                <span class="domain-tag">{{ domain }}</span>
                {% endfor %}
              </div>
              {% endif %}
              
              <h3 class="project-title">
                <a href="{{ project.url | relative_url }}">{{ project.title }}</a>
              </h3>
              
              <div class="project-description">{{ project.description }}</div>
              
              <!-- Кнопка Read More удалена -->
              
              <div class="project-meta">
                {% if project.tech %}
                <div class="project-tech">
                  {% for tech in project.tech %}
                  <span class="tech-tag">{{ tech }}</span>
                  {% endfor %}
                </div>
                {% endif %}
              </div>
              
              <div class="project-links">
                {% if project.github %}
                <a href="{{ project.github }}" class="project-link" target="_blank" rel="noopener noreferrer">
                  <i class="fab fa-github"></i>
                </a>
                {% endif %}
                
                {% if project.kaggle %}
                <a href="{{ project.kaggle }}" class="project-link" target="_blank" rel="noopener noreferrer">
                  <i class="fab fa-kaggle"></i>
                </a>
                {% endif %}
                
                {% if project.website %}
                <a href="{{ project.website }}" class="project-link" target="_blank" rel="noopener noreferrer">
                  <i class="fas fa-globe"></i>
                </a>
                {% endif %}
              </div>
            </div>
          </div>
        </div>
        {% endfor %}
      </div>
    </section>

    <!-- Personal projects section -->
    <section id="personal" class="project-section">
      <h2 class="section-title">fun / experimental</h2>
      <div class="project-list">
        {% assign personal_projects = site.projects | where: "category", "personal" | sort: "importance" %}
        {% for project in personal_projects %}
        <div class="project-card">
          <div class="project-content">
            <div class="project-image">
              {% if project.img %}
              <a href="{% if project.github %}{{ project.github }}{% elsif project.kaggle %}{{ project.kaggle }}{% else %}{{ project.url | relative_url }}{% endif %}" {% if project.github or project.kaggle %}target="_blank" rel="noopener noreferrer"{% endif %}>
                <img src="{{ project.img | relative_url }}" alt="{{ project.title }}" />
              </a>
              {% else %}
              <a href="{% if project.github %}{{ project.github }}{% elsif project.kaggle %}{{ project.kaggle }}{% else %}{{ project.url | relative_url }}{% endif %}" {% if project.github or project.kaggle %}target="_blank" rel="noopener noreferrer"{% endif %}>
                <div class="no-image"></div>
              </a>
              {% endif %}
            </div>
            <div class="project-details">
              {% if project.domains %}
              <div class="project-domains">
                {% for domain in project.domains %}
                <span class="domain-tag">{{ domain }}</span>
                {% endfor %}
              </div>
              {% endif %}
              
              <h3 class="project-title">
                <a href="{% if project.github %}{{ project.github }}{% elsif project.kaggle %}{{ project.kaggle }}{% else %}{{ project.url | relative_url }}{% endif %}" {% if project.github or project.kaggle %}target="_blank" rel="noopener noreferrer"{% endif %}>{{ project.title }}</a>
              </h3>
              
              <div class="project-description">{{ project.description }}</div>
              
              <!-- Кнопка Take a Peek удалена -->
              
              <div class="project-meta">
                {% if project.tech %}
                <div class="project-tech">
                  {% for tech in project.tech %}
                  <span class="tech-tag">{{ tech }}</span>
                  {% endfor %}
                </div>
                {% endif %}
              </div>
              
              <div class="project-links">
                {% if project.github %}
                <a href="{{ project.github }}" class="project-link" target="_blank" rel="noopener noreferrer">
                  <i class="fab fa-github"></i>
                </a>
                {% endif %}
                
                {% if project.kaggle %}
                <a href="{{ project.kaggle }}" class="project-link" target="_blank" rel="noopener noreferrer">
                  <i class="fab fa-kaggle"></i>
                </a>
                {% endif %}
                
                {% if project.website %}
                <a href="{{ project.website }}" class="project-link" target="_blank" rel="noopener noreferrer">
                  <i class="fas fa-globe"></i>
                </a>
                {% endif %}
              </div>
            </div>
          </div>
        </div>
        {% endfor %}
      </div>
    </section>
  </div>
</body>
</html>