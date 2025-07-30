---
layout: default
permalink: /services/
title: services
nav: true
nav_order: 3
dropdown: true
children:
  - title: AI/ML Consulting
    permalink: /services/ai-consulting/
  - title: Software Design
    permalink: /services/software-design/
  - title: Career Mentoring
    permalink: /services/career-mentoring/
---

<div class="post">
  <div class="services-overview">
    <h1 class="visually-hidden">Services</h1>
    <p class="service-description">Strategic AI consulting and custom development to help organizations and professionals thrive in the age of artificial intelligence.</p>
    
    <div class="services-grid">
      <div class="service-preview-card">
        <h3><i class="fa-solid fa-brain"></i> AI/ML Consulting</h3>
        <p>Strategic support for companies looking to integrate AI effectively.</p>
        <a href="/services/ai-consulting/" class="preview-link">Learn more →</a>
      </div>
      
      <div class="service-preview-card">
        <h3><i class="fa-solid fa-rocket"></i> AI/ML Software Design</h3>
        <p>Custom AI projects from POCs to production-ready systems.</p>
        <a href="/services/software-design/" class="preview-link">Learn more →</a>
      </div>
      
      <div class="service-preview-card">
        <h3><i class="fa-solid fa-user-graduate"></i> Career Mentoring</h3>
        <p>Strategic guidance for technical professionals in AI.</p>
        <a href="/services/career-mentoring/" class="preview-link">Learn more →</a>
      </div>
    </div>
  </div>
</div>

<style>
.visually-hidden {
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  padding: 0 !important;
  margin: -1px !important;
  overflow: hidden !important;
  clip: rect(0, 0, 0, 0) !important;
  white-space: nowrap !important;
  border: 0 !important;
}

.services-overview {
  margin: 2rem 0;
}

.service-description {
  color: var(--global-text-color-light);
  font-size: 1.1rem;
  margin-bottom: 2rem;
  line-height: 1.6;
  text-align: center;
}

.service-preview-card {
  background-color: var(--global-bg-color);
  border-radius: 8px;
  box-shadow: var(--global-card-shadow, 0 4px 12px rgba(0, 0, 0, 0.08));
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  transition: box-shadow 0.3s ease;
  border: 1px solid var(--global-divider-color);
}

.service-preview-card:hover {
  box-shadow: var(--global-card-shadow-hover, 0 6px 16px rgba(0, 0, 0, 0.12));
}

.service-preview-card h3 {
  color: var(--global-theme-color);
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
}

.service-preview-card h3 i {
  margin-right: 0.75rem;
}

.service-preview-card p {
  color: var(--global-text-color-light);
  margin-bottom: 1rem;
}

.preview-link {
  color: var(--global-theme-color);
  text-decoration: none;
  font-weight: 500;
}

.preview-link:hover {
  text-decoration: underline;
}

@media (min-width: 768px) {
  .services-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
  }
}
</style>