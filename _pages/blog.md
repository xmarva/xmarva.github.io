---
layout: default
permalink: /blog/
title: blog
seo_description: "Posts by Eva Koroleva on machine learning, LLMs, Transformers, system design and cognitive science, with code, experiments and figures."
nav: true
# its first page is the home page: search engines are pointed there (head.liquid)
canonical_first_page: /
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

{% comment %} The blog, page by page: the same list as the home page, whose first page it repeats. {% endcomment %}
{% if paginator.page == 1 %}{% assign first = true %}{% else %}{% assign first = false %}{% endif %}
{% include blog_index.liquid posts=paginator.posts pinned=first newer=paginator.previous_page_path older=paginator.next_page_path %}
