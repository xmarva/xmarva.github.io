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

    /* Section headings - changed to black */
    .section-heading {
      font-family: 'Playfair Display', var(--global-serif-font-family), serif;
      font-size: 2rem;
      font-weight: 600;
      margin: 2rem 0 1.5rem 0;
      text-align: left;
      color: #000000; /* Changed to black */
    }

    /* Common card styling for both featured and regular posts */
    .featured-post-card,
    .post-card {
      background-color: var(--global-bg-color);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
      transition: box-shadow 0.3s ease;
      margin-bottom: 1.5rem;
      border: 1px solid var(--global-divider-color);
      height: 220px; /* Fixed height for all cards */
    }

    .featured-post-card:hover,
    .post-card:hover {
      box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08);
    }

    /* Featured posts styling - keeping structure but aligning visuals with regular posts */
    .featured-posts {
      margin-bottom: 3rem;
    }

    .featured-post-content {
      padding: 1.5rem 2rem;
      position: relative;
      height: 100%;
      box-sizing: border-box;
    }

    .featured-pin {
      position: absolute;
      top: 1.5rem;
      right: 1.5rem;
      color: #000000; /* Changed to black */
      font-size: 1.2rem;
    }

    .featured-post-title {
      font-family: 'Playfair Display', var(--global-serif-font-family), serif;
      font-size: 1.5rem;
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
      color: #000000; /* Changed to black */
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

    .post-content {
      padding: 1.5rem 2rem;
      position: relative;
      display: flex;
      flex-direction: row;
      height: 100%;
      box-sizing: border-box;
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
      color: #000000; /* Changed to black */
    }

    .post-description {
      font-size: 1rem;
      line-height: 1.6;
      margin-bottom: 1rem;
      color: var(--global-text-color-light);
    }

    /* Unified meta styling for both post types */
    .post-meta,
    .featured-post-content .post-meta {
      display: flex;
      align-items: center;
      font-size: 0.9rem;
      color: var(--global-text-color-light);
      margin-bottom: 1rem;
      flex-wrap: wrap;
    }

    .post-meta a,
    .featured-post-content .post-meta a {
      color: var(--global-text-color-light);
      text-decoration: none;
      transition: color 0.2s ease;
    }

    .post-meta a:hover,
    .featured-post-content .post-meta a:hover {
      color: #000000; /* Changed to black */
    }

    /* Unified tag styling for both post types - changed to black */
    .post-tags,
    .featured-post-content .post-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin: 1rem 0 0 0;
      align-items: center;
    }

    .post-tags a,
    .featured-post-content .post-tags a {
      display: inline-flex;
      align-items: center;
      padding: 0.25rem 0.7rem;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 500;
      background-color: transparent;
      color: #000000; /* Changed to black */
      border: 1px solid #000000; /* Changed to black */
      text-decoration: none;
      transition: transform 0.2s ease;
    }

    .post-tags a:hover,
    .featured-post-content .post-tags a:hover {
      transform: translateY(-2px);
    }

    .post-tags i,
    .featured-post-content .post-tags i {
      margin-right: 0.25rem;
      color: #000000; /* Changed to black */
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
      .featured-post-card,
      .post-card {
        height: auto;
        min-height: 220px;
      }
      
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
    }
  </style>
</head>
<body>
  <div class="blog-wrapper">
    <!-- Featured Posts section with heading -->
    {% assign featured_posts = site.posts | where: "featured", "true" %}
    {% if featured_posts.size > 0 %}
    <h2 class="section-heading">Pinned Posts</h2>
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
              {% assign clean_tag = tag | strip %}
              <a href="{{ clean_tag | slugify | prepend: '/blog/tag/' | prepend: site.baseurl}}">
                <i class="fa-solid fa-hashtag fa-sm"></i> {{ clean_tag }}</a>
              {% endfor %}
            {% endif %}
            
            {% assign categories = post.categories | join: "" %}
            {% if categories != "" %}
            &nbsp; &middot; &nbsp;
              {% for category in post.categories %}
              {% assign clean_category = category | strip %}
              <a href="{{ clean_category | slugify | prepend: '/blog/category/' | prepend: site.baseurl}}">
                <i class="fa-solid fa-tag fa-sm"></i> {{ clean_category }}</a>
              {% endfor %}
            {% endif %}
          </div>
        </div>
      </div>
      {% endfor %}
    </div>
    <hr>
    {% endif %}

    <!-- Regular Posts section with heading -->
    <h2 class="section-heading">Latest Blogposts</h2>
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
                  {% assign clean_tag = tag | strip %}
                  <a href="{{ clean_tag | slugify | prepend: '/blog/tag/' | prepend: site.baseurl}}">
                    <i class="fa-solid fa-hashtag fa-sm"></i> {{ clean_tag }}</a>
                  {% endfor %}
                {% endif %}
                
                {% if categories != "" %}
                &nbsp; &middot; &nbsp;
                  {% for category in post.categories %}
                  {% assign clean_category = category | strip %}
                  <a href="{{ clean_category | slugify | prepend: '/blog/category/' | prepend: site.baseurl}}">
                    <i class="fa-solid fa-tag fa-sm"></i> {{ clean_category }}</a>
                  {% endfor %}
                {% endif %}
              </div>
            </div>
            {% if post.thumbnail %}
            <div class="post-thumbnail">
              <img src="{{ post.thumbnail | relative_url }}" alt="{{ post.title | escape }}" />
            </div>
            {% endif %}
          </div>
        </li>
        {% endunless %}
      {% endfor %}
    </ul>
  </div>

  <!-- Add FontAwesome for icons -->
  <script src="https://kit.fontawesome.com/your-code-here.js" crossorigin="anonymous"></script>
</body>
</html>