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
        <div class="pin-icon">
          <i class="fa-solid fa-thumbtack fa-xs"></i>
        </div>
        <h3 class="post-card-title">{{ post.title }}</h3>
        <p class="post-card-description">{{ post.description }}</p>
        <div class="post-card-tags">
          <a href="{{ year | prepend: '/blog/' | relative_url }}">
            <i class="fa-solid fa-calendar fa-sm"></i> {{ year }}
          </a>
          <span>&nbsp; &middot; &nbsp;</span>
          {{ read_time }} min read

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
<hr>
{% endif %}

<div class="post-list">
  {% if page.pagination.enabled %}
    {% assign postlist = paginator.posts %}
  {% else %}
    {% assign postlist = site.posts %}
  {% endif %}

  {% for post in postlist %}
    {% if post.featured != true %}
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
            <div class="post-card-tags">
              <a href="{{ year | prepend: '/blog/' | relative_url }}">
                <i class="fa-solid fa-calendar fa-sm"></i> {{ year }}
              </a>
              <span>&nbsp; &middot; &nbsp;</span>
              {{ read_time }} min read

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
    {% endif %}
  {% endfor %}
</div>

{% if page.pagination.enabled %}
{% include pagination.liquid %}
{% endif %}

</div>

<style>
/* Custom card colors utilizing theme variables */
:root {
  --card-bg: var(--global-bg-color);
  --card-text: var(--global-text-color);
  --card-link: var(--global-theme-color);
  --card-hover: rgba(0, 0, 0, 0.03);
  --card-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  --card-shadow-hover: 0 5px 15px rgba(0, 0, 0, 0.1);
}

html[data-theme="dark"] {
  --card-hover: rgba(255, 255, 255, 0.05);
  --card-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
  --card-shadow-hover: 0 5px 15px rgba(0, 0, 0, 0.3);
}

.header-bar {
  display: none;
}

.tag-category-list {
  display: block;
  margin: 0 0 2rem 0;
}

.tag-category-list ul {
  display: flex;
  flex-wrap: wrap;
  list-style: none;
  align-items: center;
  gap: 0.5rem;
}

.tag-category-list li {
  display: inline-flex;
  align-items: center;
}

.tag-category-list a {
  color: var(--global-theme-color);
  text-decoration: none;
  font-size: 0.9rem;
}

.tag-category-list a:hover {
  text-decoration: underline;
}

.featured-post {
  margin: 1.5rem 0 2rem 0;
  width: 100%;
}

.pin-icon {
  float: right;
  color: var(--global-theme-color);
}

hr {
  background-color: var(--global-divider-color);
  height: 1px;
  border: none;
  margin: 2rem 0;
}

.post-list {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  margin-top: 1.5rem;
}

.post-card {
  display: flex;
  flex-direction: column;
  height: auto;
  min-height: 150px;
  border-radius: 12px;
  border: none;
  padding: 1.5rem;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  background-color: var(--card-bg);
  box-shadow: var(--card-shadow);
}

.post-card:hover {
  transform: translateY(-3px);
  box-shadow: var(--card-shadow-hover);
  background-color: var(--card-hover);
}

.post-link {
  display: block;
  color: var(--card-text);
  text-decoration: none;
}

.post-link:hover {
  text-decoration: none;
}

.post-link:hover .post-card-title,
.post-link:hover .post-card-description {
  text-decoration: none !important;
}

.post-thumbnail {
  flex: 0 0 auto;
  margin-bottom: 1rem;
}

.post-thumbnail img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 8px;
}

.post-card-content {
  display: flex;
  flex-direction: column;
  min-height: 120px;
}

.post-card-title {
  font-size: 1.25rem;
  margin: 0 0 0.75rem 0;
  color: var(--global-theme-color);
  line-height: 1.3;
  flex-grow: 0;
}

.post-card-description {
  font-size: 0.95rem;
  line-height: 1.6;
  margin: 0 0 1rem 0; 
  color: var(--card-text);
  flex-grow: 1;
}

.post-card-tags {
  font-size: 0.8rem;
  color: var(--global-text-color-light);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.post-card-tags a {
  color: var(--global-theme-color);
  text-decoration: none;
}

.post-card-tags a:hover {
  text-decoration: none;
}

@media (min-width: 768px) {
  .post-link {
    display: block;
  }
  
  .post-thumbnail {
    height: 150px; 
  }
}
</style>