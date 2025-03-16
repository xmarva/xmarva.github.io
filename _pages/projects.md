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
    /* Import elegant fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Source+Sans+Pro:wght@300;400;600&display=swap');

    /* Base styling for projects page */
    .projects-wrapper {
      font-family: 'Source Sans Pro', var(--global-font-family), sans-serif;
      color: var(--global-text-color);
      max-width: 100%;
      margin: 0 auto;
    }

    /* Section styling */
    .project-section {
      margin-bottom: 4rem;
    }

    .section-title {
      font-family: 'Playfair Display', var(--global-serif-font-family), serif;
      font-size: 2.4rem;
      font-weight: 600;
      color: var(--global-theme-color);
      margin-bottom: 2rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid var(--global-divider-color);
    }

    a .section-title {
      text-decoration: none;
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
      transition: box-shadow 0.3s ease;
      height: 320px;  /* Fixed height for uniformity */
    }

    .project-card:hover {
      box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08);
    }

    .project-content {
      display: flex;
      flex-direction: row;
      height: 100%;
    }

    /* Project image */
    .project-image {
      flex: 0 0 280px;
      width: 280px;
      height: 320px;
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
      font-family: 'Playfair Display', var(--global-serif-font-family), serif;
      font-size: 1.7rem;
      font-weight: 600;
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

    /* Role styling - moved under title */
    .project-role {
      margin-bottom: 0.8rem;
    }

    .role-label {
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

    /* Read More button */
    .read-more-btn {
      display: inline-block;
      padding: 0.4rem 1rem;
      background-color: var(--global-theme-color);
      color: var(--global-bg-color);
      border-radius: 20px;
      font-size: 0.9rem;
      font-weight: 500;
      text-decoration: none;
      transition: background-color 0.2s ease, transform 0.2s ease;
      margin-bottom: 1rem;
      align-self: flex-start;
    }

    .read-more-btn:hover {
      transform: scale(1.05);
      /* Modified hover effect to not change text color */
      box-shadow: 0 3px 8px rgba(0, 0, 0, 0.15);
    }

    /* Take a peek button for personal projects */
    .take-peek-btn {
      display: inline-block;
      padding: 0.4rem 1rem;
      background-color: var(--global-theme-color);
      color: var(--global-bg-color);
      border-radius: 20px;
      font-size: 0.9rem;
      font-weight: 500;
      text-decoration: none;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
      margin-bottom: 1rem;
      align-self: flex-start;
    }

    .take-peek-btn:hover {
      transform: scale(1.05);
      box-shadow: 0 3px 8px rgba(0, 0, 0, 0.15);
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
      font-weight: 500;
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
      font-weight: 500;
      font-size: 0.9rem;
      transition: color 0.2s ease;
    }

    .project-link:hover {
      color: var(--global-theme-color);
    }

    .project-link i {
      font-size: 1.1rem;
    }

    /* Media queries for responsiveness */
    @media (max-width: 900px) {
      .project-card {
        height: auto;
        min-height: 500px;
      }
      
      .project-content {
        flex-direction: column;
      }
      
      .project-image {
        flex: none;
        width: 100%;
        height: 220px;
      }
      
      .project-details {
        padding: 1.5rem;
        min-height: 280px;
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
    }
  </style>
</head>
<body>
  <div class="projects-wrapper">
    <!-- Work projects section with updated heading -->
    <section id="work" class="project-section">
      <a id="work" href=".#work">
        <h2 class="section-title">what I worked on</h2>
      </a>
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
              
              {% if project.role %}
              <div class="project-role">
                <span class="role-label">{{ project.role }}</span>
              </div>
              {% endif %}
              
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

    <!-- Personal projects section with updated heading -->
    <section id="personal" class="project-section">
      <a id="personal" href=".#personal">
        <h2 class="section-title">personal / fun / experimental</h2>
      </a>
      <div class="project-list">
        {% assign personal_projects = site.projects | where: "category", "personal" | sort: "importance" %}
        {% for project in personal_projects %}
        <div class="project-card">
          <div class="project-content">
            <div class="project-image">
              {% if project.img %}
              <a href="{% if project.github %}{{ project.github }}{% elsif project.kaggle %}{{ project.kaggle }}{% else %}{{ project.url | relative_url }}{% endif %}" target="_blank" rel="noopener noreferrer">
                <img src="{{ project.img | relative_url }}" alt="{{ project.title }}" />
              </a>
              {% else %}
              <a href="{% if project.github %}{{ project.github }}{% elsif project.kaggle %}{{ project.kaggle }}{% else %}{{ project.url | relative_url }}{% endif %}" target="_blank" rel="noopener noreferrer">
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
                <a href="{% if project.github %}{{ project.github }}{% elsif project.kaggle %}{{ project.kaggle }}{% else %}{{ project.url | relative_url }}{% endif %}" target="_blank" rel="noopener noreferrer">{{ project.title }}</a>
              </h3>
              
              <div class="project-description">{{ project.description }}</div>
              
              {% if project.github or project.kaggle %}
              <a href="{% if project.github %}{{ project.github }}{% else %}{{ project.kaggle }}{% endif %}" class="take-peek-btn" target="_blank" rel="noopener noreferrer">Take a peek