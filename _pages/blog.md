<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Blog</title>
  <style>
    /* Import elegant fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Source+Sans+Pro:wght@300;400;600&display=swap');

    /* Base styling */
    .blog-wrapper {
      font-family: 'Source Sans Pro', var(--global-font-family), sans-serif;
      color: var(--global-text-color);
      max-width: 100%;
      margin: 0 auto;
    }

    /* Tags styling - matching project page */
    .tag-category-list {
      display: flex;
      justify-content: center;
      margin: 2rem 0;
    }

    .tag-category-list ul {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      justify-content: center;
      list-style: none;
    }

    .tag-category-list li {
      display: inline-block;
    }

    .tag-category-list a {
      display: inline-block;
      padding: 0.25rem 0.7rem;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 500;
      background-color: transparent;
      color: var(--global-theme-color);
      border: 1px solid var(--global-theme-color);
      text-decoration: none;
      transition: transform 0.2s ease;
    }

    .tag-category-list a:hover {
      transform: translateY(-2px);
    }

    .tag-category-list p {
      margin: 0 0.3rem;
      display: flex;
      align-items: center;
      color: var(--global-text-color-light);
    }

    /* Featured posts styling */
    .featured-posts {
      margin-bottom: 3rem;
    }

    .featured-post-card {
      background-color: var(--global-bg-color);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
      transition: box-shadow 0.3s ease;
      margin-bottom: 1.5rem;
      border: 1px solid rgba(var(--global-theme-color-rgb), 0.2);
      position: relative;
      background-color: rgba(var(--global-theme-color-rgb), 0.05);
    }

    .featured-post-card:hover {
      box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08);
    }

    .featured-post-content {
      padding: 1.5rem 2rem;
      position: relative;
    }

    .featured-pin {
      position: absolute;
      top: 1.5rem;
      right: 1.5rem;
      color: var(--global-theme-color);
      font-size: 1.2rem;
    }

    .featured-post-title {
      font-family: 'Playfair Display', var(--global-serif-font-family), serif;
      font-size: 1.7rem;
      font-weight: 600;
      margin: 0 0 0.6rem 0;
      letter-spacing: -0.02em;
      padding-right: 2rem;
    }

    .featured-post-title a {
      color: var(--global-text-color);
      text-decoration: none;
      transition: color 0.2s ease;
    }

    .featured-post-title a:hover {
      color: var(--global-theme-color);
    }

    .featured-post-description {
      font-size: 1rem;
      line-height: 1.6;
      margin-bottom: 1rem;
      color: var(--global-text-color-light);
    }

    /* Regular posts styling */
    .post-list {
      list-style: none;
      padding: 0;
      margin: 0;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }

    .post-card {
      background-color: var(--global-bg-color);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
      transition: box-shadow 0.3s ease;
      border: 1px solid var(--global-divider-color);
    }

    .post-card:hover {
      box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08);
    }

    .post-content {
      padding: 1.5rem 2rem;
      position: relative;
      display: flex;
      flex-direction: row;
    }

    .post-text {
      flex: 1;
      padding-right: 1rem;
    }

    .post-thumbnail {
      flex: 0 0 220px;
      height: 160px;
      overflow: hidden;
      border-radius: 8px;
    }

    .post-thumbnail img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      object-position: center;
    }

    .post-title {
      font-family: 'Playfair Display', var(--global-serif-font-family), serif;
      font-size: 1.5rem;
      font-weight: 600;
      margin: 0 0 0.6rem 0;
      letter-spacing: -0.02em;
    }

    .post-title a {
      color: var(--global-text-color);
      text-decoration: none;
      transition: color 0.2s ease;
    }

    .post-title a:hover {
      color: var(--global-theme-color);
    }

    .post-description {
      font-size: 1rem;
      line-height: 1.6;
      margin-bottom: 1rem;
      color: var(--global-text-color-light);
    }

    .post-meta {
      display: flex;
      align-items: center;
      font-size: 0.9rem;
      color: var(--global-text-color-light);
      margin-bottom: 0.5rem;
      flex-wrap: wrap;
    }

    .post-meta a {
      color: var(--global-text-color-light);
      text-decoration: none;
      transition: color 0.2s ease;
    }

    .post-meta a:hover {
      color: var(--global-theme-color);
    }

    .post-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin: 0;
      align-items: center;
    }

    .post-tags a {
      display: inline-flex;
      align-items: center;
      padding: 0.25rem 0.7rem;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 500;
      background-color: transparent;
      color: var(--global-theme-color);
      border: 1px solid var(--global-theme-color);
      text-decoration: none;
      transition: transform 0.2s ease;
    }

    .post-tags a:hover {
      transform: translateY(-2px);
    }

    .post-tags i {
      margin-right: 0.25rem;
    }

    /* Pagination */
    .pagination {
      margin-top: 3rem;
      display: flex;
      justify-content: center;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
      .post-content {
        flex-direction: column;
      }

      .post-text {
        padding-right: 0;
        margin-bottom: 1rem;
      }

      .post-thumbnail {
        flex: none;
        width: 100%;
        height: 180px;
        order: -1;
        margin-bottom: 1rem;
      }

      .tag-category-list ul {
        justify-content: center;
      }
    }
  </style>
</head>
<body>
  <div class="blog-wrapper">
    <!-- Tags and Categories section -->
    {% if site.display_tags and site.display_tags.size > 0 or site.display_categories and site.display_categories.size > 0 %}
    <div class="tag-category-list">
      <ul class="p-0 m-0">
        {% for tag in site.display_tags %}
          <li>
            <a href="{{ tag | slugify | prepend: '/blog/tag/' | relative_url }}">
              <i class="fa-solid fa-hashtag fa-sm"></i> {{ tag }}
            </a>
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
            <a href="{{ category | slugify | prepend: '/blog/category/' | relative_url }}">
              <i class="fa-solid fa-tag fa-sm"></i> {{ category }}
            </a>
          </li>
          {% unless forloop.last %}
            <p>&bull;</p>
          {% endunless %}
        {% endfor %}
      </ul>
    </div>
    {% endif %}

    <!-- Featured Posts section - one per row -->
    {% assign featured_posts = site.posts | where: "featured", "true" %}
    {% if featured_posts.size > 0 %}
    <div class="featured-posts">
      {% for post in featured_posts %}
      <div class="featured-post-card">
        <div class="featured-post-content">
          <div class="featured-pin">
            <i class="fa-solid fa-thumbtack"></i>
          </div>
          <h3 class="featured-post-title">
            <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
          </h3>
          <p class="featured-post-description">{{ post.description }}</p>
          
          {% if post.external_source == blank %}
            {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
          {% else %}
            {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
          {% endif %}
          {% assign year = post.date | date: "%Y" %}
          
          <div class="post-meta">
            {{ read_time }} min read &nbsp; &middot; &nbsp;
            {{ post.date | date: '%B %d, %Y' }}
            {% if post.external_source %}
            &nbsp; &middot; &nbsp; {{ post.external_source }}
            {% endif %}
          </div>
          
          <div class="post-tags">
            <a href="{{ year | prepend: '/blog/' | prepend: site.baseurl}}">
              <i class="fa-solid fa-calendar fa-sm"></i> {{ year }}
            </a>
            
            {% assign tags = post.tags | join: "" %}
            {% if tags != "" %}
            &nbsp; &middot; &nbsp;
              {% for tag in post.tags %}
              <a href="{{ tag | slugify | prepend: '/blog/tag/' | prepend: site.baseurl}}">
                <i class="fa-solid fa-hashtag fa-sm"></i> {{ tag }}</a>
              {% endfor %}
            {% endif %}
            
            {% assign categories = post.categories | join: "" %}
            {% if categories != "" %}
            &nbsp; &middot; &nbsp;
              {% for category in post.categories %}
              <a href="{{ category | slugify | prepend: '/blog/category/' | prepend: site.baseurl}}">
                <i class="fa-solid fa-tag fa-sm"></i> {{ category }}</a>
              {% endfor %}
            {% endif %}
          </div>
        </div>
      </div>
      {% endfor %}
    </div>
    <hr>
    {% endif %}

    <!-- Regular Posts section - one per row -->
    <ul class="post-list">
      {% if page.pagination.enabled %}
        {% assign postlist = paginator.posts %}
      {% else %}
        {% assign postlist = site.posts %}
      {% endif %}

      {% for post in postlist %}
        {% unless post.featured %}
        
        {% if post.external_source == blank %}
          {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
        {% else %}
          {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
        {% endif %}
        {% assign year = post.date | date: "%Y" %}
        {% assign tags = post.tags | join: "" %}
        {% assign categories = post.categories | join: "" %}
        
        <li class="post-card">
          <div class="post-content">
            <div class="post-text">
              <h3 class="post-title">
                {% if post.redirect == blank %}
                  <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
                {% elsif post.redirect contains '://' %}
                  <a href="{{ post.redirect }}" target="_blank">{{ post.title }}</a>
                  <svg width="1rem" height="1rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
                    <path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path>
                  </svg>
                {% else %}
                  <a href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
                {% endif %}
              </h3>
              <p class="post-description">{{ post.description }}</p>
              <div class="post-meta">
                {{ read_time }} min read &nbsp; &middot; &nbsp;
                {{ post.date | date: '%B %d, %Y' }}
                {% if post.external_source %}
                &nbsp; &middot; &nbsp; {{ post.external_source }}
                {% endif %}
              </div>
              <div class="post-tags">
                <a href="{{ year | prepend: '/blog/' | prepend: site.baseurl}}">
                  <i class="fa-solid fa-calendar fa-sm"></i> {{ year }}
                </a>
                
                {% if tags != "" %}
                &nbsp; &middot; &nbsp;
                  {% for tag in post.tags %}
                  <a href="{{ tag | slugify | prepend: '/blog/tag/' | prepend: site.baseurl}}">
                    <i class="fa-solid fa-hashtag fa-sm"></i> {{ tag }}</a>
                  {% endfor %}
                {% endif %}
                
                {% if categories != "" %}
                &nbsp; &