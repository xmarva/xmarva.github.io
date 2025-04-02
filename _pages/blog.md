---
layout: default
permalink: /blog/
title: blog
nav: true
nav_order: 1
pagination:
  enabled: true
  collection: posts
  permalink: /page/:num/
  per_page: 5
  sort_field: date
  sort_reverse: true
  trail:
    before: 1 # The number of links before the current page
    after: 3 # The number of links after the current page
---

<div class="post">

{% if site.display_tags and site.display_tags.size > 0 or site.display_categories and site.display_categories.size > 0 %}
  <div class="tag-category-list">
    <ul class="p-0 m-0">
      {% for tag in site.display_tags %}
        <li>
          <i class="fa-solid fa-hashtag fa-sm"></i> <a href="{{ tag | slugify | prepend: '/blog/tag/' | relative_url }}">{{ tag }}</a>
        </li>
        {% unless forloop.last %}
          <p>&bull;</p>
        {% endunless %}
      {% endfor %}
      {% if site.display_categories.size > 0 and site.display_tags.size > 0 %}
        <p>&bull;</p>
      {% endif %}
      {% for category in site.display_categories %}
        <li>
          <i class="fa-solid fa-tag fa-sm"></i> <a href="{{ category | slugify | prepend: '/blog/category/' | relative_url }}">{{ category }}</a>
        </li>
        {% unless forloop.last %}
          <p>&bull;</p>
        {% endunless %}
      {% endfor %}
    </ul>
  </div>
  {% endif %}

{% assign featured_posts = site.posts | where: "featured", "true" | limit: 1 %}
{% if featured_posts.size > 0 %}
<div class="featured-post">
  {% for post in featured_posts %}
  <a href="{{ post.url | relative_url }}">
    <div class="featured-card">
      <div class="card-content">
        <div class="pin-icon">
          <i class="fa-solid fa-thumbtack fa-xs"></i>
        </div>
        <h3 class="card-title">{{ post.title }}</h3>
        <p class="card-description">{{ post.description }}</p>

        {% if post.external_source == blank %}
          {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
        {% else %}
          {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
        {% endif %}
        {% assign year = post.date | date: "%Y" %}

        <p class="post-meta">
          {{ read_time }} min read &nbsp; &middot; &nbsp;
          <a href="{{ year | prepend: '/blog/' | relative_url }}">
            <i class="fa-solid fa-calendar fa-sm"></i> {{ year }}
          </a>
        </p>
      </div>
    </div>
  </a>
  {% endfor %}
</div>
<hr>
{% endif %}

<div class="post-list">
  {% if page.pagination.enabled %}
    {% assign postlist = paginator.posts %}
  {% else %}
    {% assign postlist = site.posts %}
  {% endif %}

  {% for post in postlist %}
    {% if post.external_source == blank %}
      {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
    {% else %}
      {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
    {% endif %}
    {% assign year = post.date | date: "%Y" %}
    {% assign tags = post.tags | join: "" %}
    {% assign categories = post.categories | join: "" %}

    <div class="post-card">
      <a class="post-link" href="{{ post.url | relative_url }}">
        {% if post.thumbnail %}
        <div class="post-thumbnail">
          <img src="{{ post.thumbnail | relative_url }}" alt="Post thumbnail">
        </div>
        {% endif %}
        <div class="post-card-content">
          <h3 class="post-card-title">{{ post.title }}</h3>
          <p class="post-card-description">{{ post.description }}</p>
          <div class="post-card-meta">
            <span>{{ read_time }} min read</span>
            <span>&nbsp; &middot; &nbsp;</span>
            <span>{{ post.date | date: '%B %d, %Y' }}</span>
            {% if post.external_source %}
            <span>&nbsp; &middot; &nbsp;</span>
            <span>{{ post.external_source }}</span>
            {% endif %}
          </div>
          <div class="post-card-tags">
            <a href="{{ year | prepend: '/blog/' | relative_url }}">
              <i class="fa-solid fa-calendar fa-sm"></i> {{ year }}
            </a>

            {% if tags != "" %}
            <span>&nbsp; &middot; &nbsp;</span>
              {% for tag in post.tags %}
              <a href="{{ tag | slugify | prepend: '/blog/tag/' | relative_url }}">
                <i class="fa-solid fa-hashtag fa-sm"></i> {{ tag }}</a>
                {% unless forloop.last %}
                  &nbsp;
                {% endunless %}
                {% endfor %}
            {% endif %}

            {% if categories != "" %}
            <span>&nbsp; &middot; &nbsp;</span>
              {% for category in post.categories %}
              <a href="{{ category | slugify | prepend: '/blog/category/' | relative_url }}">
                <i class="fa-solid fa-tag fa-sm"></i> {{ category }}</a>
                {% unless forloop.last %}
                  &nbsp;
                {% endunless %}
                {% endfor %}
            {% endif %}
          </div>
        </div>
      </a>
    </div>
  {% endfor %}
</div>

{% if page.pagination.enabled %}
{% include pagination.liquid %}
{% endif %}

</div>

<style>
/* Custom card colors utilizing theme variables */
/* Custom card colors utilizing theme variables */
:root {
  --card-bg: var(--global-bg-color);
  --card-border: var(--global-divider-color);
  --card-text: var(--global-text-color);
  --card-link: var(--global-theme-color);
  --card-hover: rgba(0, 0, 0, 0.03);
}

html[data-theme="dark"] {
  --card-hover: rgba(255, 255, 255, 0.05);
}

.header-bar {
  display: none;
}

/* Полностью скрываем секцию с тегами */
.tag-category-list {
  display: none;
}

/* Featured post styling */  
.featured-post {
  margin: 1.5rem 0 2rem 0;
  width: 100%;
}

.featured-card {
  background-color: var(--card-bg);
  border-radius: 12px;
  border: 1px solid var(--card-border);
  padding: 1.5rem;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.featured-post a {
  color: var(--card-text);
  text-decoration: none;
  display: block;
}

.featured-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
  background-color: var(--card-hover);
}

.pin-icon {
  float: right;
  color: var(--global-theme-color);
}

.card-title {
  margin-top: 0;
  font-size: 1.6rem;
  color: var(--global-theme-color);
}

.card-description {
  margin-top: 1rem;
  margin-bottom: 1.5rem;
  font-size: 1rem;
  line-height: 1.6;
  color: var(--card-text);
}

/* Метаданные для featured post на одной строке */
.post-meta {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.post-meta a {
  color: var(--global-theme-color);
  text-decoration: none;
}

/* Убираем подчеркивание при наведении на метаданные */
.post-meta a:hover {
  text-decoration: none;
}

hr {
  background-color: var(--card-border);
  height: 1px;
  border: none;
  margin: 2rem 0;
}

/* Post list styling - карточки одинакового размера */
.post-list {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  margin-top: 1.5rem;
}

.post-card {
  background-color: var(--card-bg);
  border-radius: 12px;
  overflow: hidden;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  border: 1px solid var(--card-border);
  width: 100%;
  height: 180px; /* Фиксированная высота для одинакового размера карточек */
}

.post-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
  background-color: var(--card-hover);
}

.post-link {
  display: flex;
  flex-direction: row;
  color: var(--card-text);
  text-decoration: none;
  height: 100%;
}

.post-thumbnail {
  flex: 0 0 280px;
  max-width: 280px;
  height: 180px;
  overflow: hidden;
}

.post-thumbnail img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.post-card-content {
  padding: 1.2rem;
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: space-between; /* Распределяет пространство равномерно */
}

.post-card-title {
  margin-top: 0;
  margin-bottom: 0.75rem;
  font-size: 1.25rem;
  color: var(--global-theme-color);
  /* Убираем возможные проблемы с отображением */
  display: block;
  visibility: visible;
}

.post-card-description {
  margin-bottom: 0.75rem;
  font-size: 0.9rem;
  line-height: 1.5;
  color: var(--card-text);
  display: -webkit-box;
  -webkit-line-clamp: 2; /* Уменьшил количество строк для более компактного отображения */
  -webkit-box-orient: vertical;
  overflow: hidden;
  /* Убираем возможные проблемы с отображением */
  visibility: visible;
}

/* Скрываем метаданные о времени чтения */
.post-card-meta {
  display: none;
}

.post-card-tags {
  font-size: 0.8rem;
  color: var(--global-text-color-light);
  margin-top: auto; /* Прижимает теги к низу карточки */
  /* Убираем возможные проблемы с отображением */
  display: block;
  visibility: visible;
}

.post-card-tags a {
  color: var(--global-theme-color);
  text-decoration: none;
}

/* Убираем подчеркивание при наведении */
.post-card-tags a:hover {
  text-decoration: none;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .post-link {
    flex-direction: column;
  }
  
  .post-thumbnail {
    max-width: 100%;
    height: 200px;
  }
  
  .card-title {
    font-size: 1.4rem;
  }
  
  .post-card {
    height: auto; /* Для мобильных устройств высота адаптивная */
  }
  
  .post-card-content {
    padding: 1rem;
  }
}