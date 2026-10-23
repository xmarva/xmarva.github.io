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
  per_page: 12
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

<div class="post-rows">
  {% if paginator.page == 1 or page.pagination.enabled != true %}
    {% for post in featured_posts %}
      {% include post_card.liquid post=post featured=true %}
    {% endfor %}
  {% endif %}

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
