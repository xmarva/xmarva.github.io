---
layout: default
permalink: /
title: blog
seo_description: "Eva Koroleva's blog: long reads on machine learning, Transformers and LLMs, AI security and cognitive science, with experiments, code and figures drawn from data."
---

{% comment %}
  The home page is the first page of the blog, the same as /blog/ (which points search engines here with its
  canonical link): the same 12 posts as the paginator's first page, then a link to the second.
{% endcomment %}
{% assign per_page = site.pages | where: 'permalink', '/blog/' | first %}
{% assign per_page = per_page.pagination.per_page | default: 12 %}
{% assign posts = site.posts | slice: 0, per_page %}
{% if site.posts.size > per_page %}{% assign older = '/blog/page/2/' %}{% endif %}
{% include blog_index.liquid posts=posts pinned=true older=older %}
