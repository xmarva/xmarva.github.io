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

{% comment %} only link tags/categories that have posts, otherwise their archive page doesn't exist {% endcomment %}
{% assign shown_tags = '' | split: '' %}
{% for tag in site.display_tags %}
  {% if site.tags[tag] %}{% assign shown_tags = shown_tags | push: tag %}{% endif %}
{% endfor %}
{% assign shown_categories = '' | split: '' %}
{% for category in site.display_categories %}
  {% if site.categories[category] %}{% assign shown_categories = shown_categories | push: category %}{% endif %}
{% endfor %}

{% if shown_tags.size > 0 or shown_categories.size > 0 %}
  <div class="tag-category-list">
    <ul class="p-0 m-0">
      {% for tag in shown_tags %}
        <li>
          <i class="fa-solid fa-hashtag fa-sm"></i> <a href="{{ tag | slugify | prepend: '/blog/tag/' | append: '/' | relative_url }}">{{ tag }}</a>
        </li>
        {% unless forloop.last %}
          <p>&bull;</p>
        {% endunless %}
      {% endfor %}
      {% if shown_categories.size > 0 and shown_tags.size > 0 %}
        <p>&bull;</p>
      {% endif %}
      {% for category in shown_categories %}
        <li>
          <i class="fa-solid fa-tag fa-sm"></i> <a href="{{ category | slugify | prepend: '/blog/category/' | append: '/' | relative_url }}">{{ category }}</a>
        </li>
        {% unless forloop.last %}
          <p>&bull;</p>
        {% endunless %}
      {% endfor %}
    </ul>
  </div>
{% endif %}

{% assign featured_posts = site.posts | where: "featured", "true" %}
{% if featured_posts.size > 0 %}
<div class="featured-post">
  {% for post in featured_posts %}
    {% include post_card.liquid post=post featured=true %}
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
      {% include post_card.liquid post=post %}
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

.tag-category-list {
  display: block;
  margin: 0 0 2rem 0;
  margin-top: -3.5rem;
}

.tag-category-list ul {
  display: flex;
  flex-wrap: wrap;
  list-style: none;
  align-items: center;
  gap: 0.5rem;
}

.tag-category-list ul p {
  margin: 0;
  display: flex;
  align-items: center;
  height: 100%;
  line-height: 1;
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
  position: absolute;
  top: 1.5rem;
  right: 1.5rem;
  color: var(--global-theme-color);
}

.featured-post .post-card-title {
  padding-right: 1.5rem; /* room for the pin icon */
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
  position: relative; /* containing block for the title's .stretched-link */
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

.post-link,
.post-link:hover {
  color: inherit;
  text-decoration: none;
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
}

.post-card-tags {
  font-size: 0.8rem;
  color: var(--global-text-color-light);
}

.post-card-tags a {
  /* keep the tag links clickable above the card-wide title link */
  position: relative;
  z-index: 2;
  white-space: nowrap;
  color: var(--global-theme-color);
  text-decoration: none;
}

.post-card-tags a:hover {
  text-decoration: none;
}

@media (min-width: 768px) {
  .post-thumbnail {
    height: 150px; 
  }
}
</style>