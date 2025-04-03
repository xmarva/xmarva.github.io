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
    /* Удалены импорты кастомных шрифтов */
    
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
      /* Удаляем подчеркивание */
      text-decoration: none;
      /* Нельзя кликнуть на название категории */
      pointer-events: none;
    }

    /* Project card styling - изменены размеры */
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
      height: 220px;  /* Изменено на более компактный размер */
      width: 100%;    /* Ширина будет подстраиваться под контент */
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

    /* Project image - изменена ориентация на горизонтальную */
    .project-image {
      flex: 0 0 300px;  /* Горизонтальная ориентация */
      width: 300px;
      height: 220px;    /* Совпадает с высотой карточки */
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
      padding: 1.5rem 2rem;
      display: flex;
      flex-direction: column;
      position: relative;
    }

    /* Domain tags in top right corner */
    .project-domains {
      position: absolute;
      top: 1rem;
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
      margin: 0 0 0.4rem 0; /* Reduced top margin */
      letter-spacing: -0.02em;
      padding-right: 6rem; /* Make space for domain tags */
    }

    .project-title a {
      color: var(--global-text-color);
      text-decoration: none;
      transition: color 0.2s ease;
    }

    .project-title a:hover {
      color: var(--global-theme-color);
    }

    /* Remove role display as requested */
    .project-role {
      display: none;
    }

    .project-description {
      font-size: 1rem;
      line-height: 1.6;
      margin-bottom: 1rem;
      color: var(--global-text-color-light);
      overflow: hidden;
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
    }

    /* Read More button - сохраняем эффект при наведении, но шрифт не меняется */
    .read-more-btn {
      display: inline-block;
      padding: 0.4rem 1rem;
      background-color: var(--global-theme-color);
      color: var(--global-bg-color);
      border-radius: 20px;
      font-size: 0.9rem;
      font-weight: normal;
      text-decoration: none;
      margin-bottom: 1rem;
      align-self: flex-start;
      position: relative;
      overflow: hidden;
      z-index: 1;
      transition: transform 0.3s ease;
    }

    .read-more-btn:before {
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

    .read-more-btn:hover {
      transform: translateY(-3px);
    }

    .read-more-btn:hover:before {
      transform: scaleX(1);
      transform-origin: left;
    }

    /* Take a peek button */
    .take-peek-btn {
      display: inline-block;
      padding: 0.4rem 1rem;
      background-color: var(--global-theme-color);
      color: var(--global-bg-color);
      border-radius: 20px;
      font-size: 0.9rem;
      font-weight: normal;
      text-decoration: none;
      margin-bottom: 1rem;
      align-self: flex-start;
      position: relative;
      overflow: hidden;
      z-index: 1;
      transition: transform 0.3s ease;
    }

    .take-peek-btn:before {
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

    .take-peek-btn:hover {
      transform: translateY(-3px);
    }

    .take-peek-btn:hover:before {
      transform: scaleX(1);
      transform-origin: left;
    }

    /* Project metadata (technologies, links) */
    .project-meta {
      margin-top: auto;
      display: flex;
      flex-direction: column;
      gap: 0.8rem;
    }

    .project-tech {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
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
      gap: 0.4rem;
      color: var(--global-text-color);
      text-decoration: none;
      font-weight: normal;
      font-size: 0.9rem;
      transition: color 0.2s ease;
    }

    .project-link:hover {
      color: var(--global-theme-color);
    }

    .project-link i {
      font-size: 1.1rem;
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
    <!-- Work projects section with updated heading -->
    <section id="work" class="project-section">
      <!-- Убрана возможность клика на названии категории -->
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
              
              <a href="{{ project.url | relative_url }}" class="read-more-btn">Read More</a>
              
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
                  <i class="fab fa-github"></i> GitHub
                </a>
                {% endif %}
                
                {% if project.kaggle %}
                <a href="{{ project.kaggle }}" class="project-link" target="_blank" rel="noopener noreferrer">
                  <i class="fab fa-kaggle"></i> Kaggle
                </a>
                {% endif %}
                
                {% if project.website %}
                <a href="{{ project.website }}" class="project-link" target="_blank" rel="noopener noreferrer">
                  <i class="fas fa-globe"></i> Website
                </a>
                {% endif %}
              </div>
            </div>
          </div>
        </div>
        {% endfor %}
      </div>
    </section>

    <!-- Personal projects section with updated heading and renamed -->
    <section id="personal" class="project-section">
      <!-- Убрана возможность клика на названии категории -->
      <h2 class="section-title">fun / experimental</h2>
      <div class="project-list">
        {% assign personal_projects = site.projects | where: "category", "personal" | sort: "importance" %}
        {% for project in personal_projects %}
        <div class="project-card">
          <div class="project-content">
            <div class="project-image">
              {% if project.img %}
              <!-- Link image to GitHub/Kaggle for personal projects -->
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
                <!-- Link title to GitHub/Kaggle for personal projects -->
                <a href="{% if project.github %}{{ project.github }}{% elsif project.kaggle %}{{ project.kaggle }}{% else %}{{ project.url | relative_url }}{% endif %}" {% if project.github or project.kaggle %}target="_blank" rel="noopener noreferrer"{% endif %}>{{ project.title }}</a>
              </h3>
              
              <div class="project-description">{{ project.description }}</div>
              
              {% if project.github or project.kaggle or project.website %}
              <a href="{% if project.github %}{{ project.github }}{% elsif project.kaggle %}{{ project.kaggle }}{% else %}{{ project.website }}{% endif %}" class="take-peek-btn" target="_blank" rel="noopener noreferrer">Take a Peek</a>
              {% endif %}
              
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
                  <i class="fab fa-github"></i> GitHub
                </a>
                {% endif %}
                
                {% if project.kaggle %}
                <a href="{{ project.kaggle }}" class="project-link" target="_blank" rel="noopener noreferrer">
                  <i class="fab fa-kaggle"></i> Kaggle
                </a>
                {% endif %}
                
                {% if project.website %}
                <a href="{{ project.website }}" class="project-link" target="_blank" rel="noopener noreferrer">
                  <i class="fas fa-globe"></i> Website
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