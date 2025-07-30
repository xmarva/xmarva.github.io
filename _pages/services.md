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
        <div class="service-preview-header">
          <h3><i class="fa-solid fa-brain"></i> AI/ML Consulting</h3>
        </div>
        <p class="service-preview-description">Strategic support for companies looking to integrate AI effectively. From feasibility assessments to designing robust ML roadmaps, I help you align AI initiatives with your business goals and market realities.</p>
        

        
        <div class="service-preview-footer">
          <div class="pricing-preview">
            <strong>Consulting rate: $100/hour</strong>
          </div>
          <a href="/services/ai-consulting/" class="preview-link">Learn more →</a>
        </div>
      </div>
      
      <div class="service-preview-card">
        <div class="service-preview-header">
          <h3><i class="fa-solid fa-rocket"></i> AI/ML Software Design</h3>
        </div>
        <p class="service-preview-description">Working with a network of trusted developers and ML experts, I lead and deliver custom AI projects — from early POCs to production-ready systems. Human-centered design meets cutting-edge technology.</p>
        

        
        <div class="service-preview-footer">
          <div class="pricing-preview">
            <strong>Custom pricing based on project scope</strong>
          </div>
          <a href="/services/software-design/" class="preview-link">Learn more →</a>
        </div>
      </div>
      
      <div class="service-preview-card">
        <div class="service-preview-header">
          <h3><i class="fa-solid fa-user-graduate"></i> Career Mentoring</h3>
        </div>
        <p class="service-preview-description">Helping technical professionals break into AI or level up within the field through strategic, no-fluff guidance. Skip the endless theory rabbit holes and focus on what actually matters for career advancement.</p>
        

        
        <div class="service-preview-footer">
          <div class="pricing-preview">
            <strong>Mentoring rate: $100/hour</strong>
          </div>
          <a href="/services/career-mentoring/" class="preview-link">Learn more →</a>
        </div>
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

.services-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 2rem;
  margin-top: 2rem;
}

.service-preview-card {
  background-color: var(--global-bg-color);
  border-radius: 8px;
  box-shadow: var(--global-card-shadow, 0 4px 12px rgba(0, 0, 0, 0.08));
  padding: 2rem;
  transition: box-shadow 0.3s ease;
  border: 1px solid var(--global-divider-color);
}

.service-preview-card:hover {
  box-shadow: var(--global-card-shadow-hover, 0 6px 16px rgba(0, 0, 0, 0.12));
}

.service-preview-header h3 {
  color: var(--global-theme-color);
  margin: 0 0 1rem 0;
  font-size: 1.4rem;
  display: flex;
  align-items: center;
}

.service-preview-header h3 i {
  margin-right: 0.75rem;
}

.service-preview-description {
  color: var(--global-text-color);
  margin-bottom: 2rem;
  font-size: 1rem;
  line-height: 1.6;
}

.service-preview-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-top: 1rem;
  border-top: 1px solid var(--global-divider-color);
}

.pricing-preview {
  color: var(--global-theme-color);
  font-size: 0.9rem;
}

.preview-link {
  color: var(--global-theme-color);
  text-decoration: none;
  font-weight: 500;
  padding: 0.5rem 1rem;
  border: 1px solid var(--global-theme-color);
  border-radius: 20px;
  transition: all 0.3s ease;
}

.preview-link:hover {
  background: var(--global-theme-color);
  color: var(--global-bg-color);
  text-decoration: none;
}

@media (max-width: 768px) {
  .service-preview-footer {
    flex-direction: column;
    gap: 1rem;
    align-items: stretch;
  }
  
  .preview-link {
    text-align: center;
  }
}
</style>