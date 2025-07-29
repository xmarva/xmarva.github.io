---
layout: default
permalink: /services/
title: services
nav: true
nav_order: 3
---

<div class="post">

<div class="services-intro">
  <h1>services</h1>
  <p class="lead-text">Strategic AI consulting and custom development to help organizations and professionals thrive in the age of AI</p>
</div>

<div class="services-grid">

  <!-- AI/ML Consulting -->
  <div class="service-card">
    <h2><i class="fa-solid fa-brain"></i> AI/ML Consulting</h2>
    <div class="service-content">
      <p class="service-description">
        Strategic support for companies looking to integrate AI effectively. From feasibility assessments to designing robust ML roadmaps, I help you align AI initiatives with your business goals and market realities.
      </p>
      
      <div class="service-details">
        <h3>What I Do:</h3>
        <ul>
          <li><strong>Computer Vision Solutions</strong> — Custom visual AI systems for industry-specific challenges</li>
          <li><strong>Large Language Models</strong> — Implementation and fine-tuning for your unique use cases</li>
          <li><strong>Custom Model Design</strong> — Bespoke AI architectures tailored to your data and constraints</li>
          <li><strong>Technical Audits</strong> — Comprehensive evaluation of existing AI initiatives and infrastructure</li>
          <li><strong>Feasibility Assessment</strong> — Realistic ROI analysis and implementation roadmaps</li>
          <li><strong>Roadmap Development</strong> — Strategic planning and optimization for long-term AI success</li>
        </ul>
      </div>

      <div class="pricing-info">
        <p><strong>Free initial consultation call • $100/hour thereafter</strong></p>
      </div>

      <div class="cta-section">
        <a href="mailto:elizaveta.victoria.koroleva@gmail.com?subject=AI/ML Consulting Inquiry" class="cta-button">Get Started</a>
        <a href="https://www.linkedin.com/in/marva-s" class="cta-button secondary" target="_blank">
          <i class="fa-brands fa-linkedin"></i> Connect on LinkedIn
        </a>
      </div>
    </div>
  </div>

  <!-- ML Software Development -->
  <div class="service-card">
    <h2><i class="fa-solid fa-rocket"></i> ML Software Development</h2>
    <div class="service-content">
      <p class="service-description">
        Working with a network of trusted developers and ML experts, I lead and deliver custom AI projects — from early POCs to production-ready systems. Human-centered design meets cutting-edge technology.
      </p>
      
      <div class="service-details">
        <h3>Full-Stack AI Delivery:</h3>
        
        <div class="sub-service">
          <h4><i class="fa-solid fa-lightbulb"></i> Proof of Concepts</h4>
          <p>Rapid prototyping with human factors consideration from day one. Validate your AI ideas quickly and cost-effectively.</p>
        </div>
        
        <div class="sub-service">
          <h4><i class="fa-solid fa-cogs"></i> Production Systems</h4>
          <p>Scalable ML solutions designed for real-world cognitive constraints. Built to last, scale, and integrate seamlessly.</p>
        </div>
        
        <div class="sub-service">
          <h4><i class="fa-solid fa-wrench"></i> Project Optimization</h4>
          <p>Audit and enhancement of existing ML projects. Performance optimization, code refactoring, and system improvements.</p>
        </div>
        
        <div class="sub-service">
          <h4><i class="fa-solid fa-project-diagram"></i> Project Leadership</h4>
          <p>Technical project management informed by organizational psychology. Keep your AI initiatives on track and stakeholders aligned.</p>
        </div>
      </div>

      <div class="pricing-info">
        <p><strong>Custom pricing based on project scope and requirements</strong></p>
      </div>

      <div class="cta-section">
        <a href="mailto:elizaveta.victoria.koroleva@gmail.com?subject=Project Execution Inquiry" class="cta-button">Discuss Your Project</a>
        <a href="https://www.linkedin.com/in/marva-s" class="cta-button secondary" target="_blank">
          <i class="fa-brands fa-linkedin"></i> Connect on LinkedIn
        </a>
      </div>
    </div>
  </div>

  <!-- Career Mentoring -->
  <div class="service-card">
    <h2><i class="fa-solid fa-user-graduate"></i> Career Mentoring</h2>
    <div class="service-content">
      <p class="service-description">
        Helping technical professionals break into AI or level up within the field through strategic, no-fluff guidance. Skip the endless theory rabbit holes and focus on what actually matters for career advancement.
      </p>
      <div class="service-details">
        <h3>My Approach:</h3>
        <ul>
          <li><strong>Strategic Learning Plans</strong> — Skip the math deep-dives, focus on practical skills that employers value</li>
          <li><strong>Professional Positioning</strong> — Craft compelling narratives that showcase your unique AI potential</li>
          <li><strong>Resume & Portfolio Optimization</strong> — Make your experience stand out in competitive AI job markets</li>
          <li><strong>Interview Preparation</strong> — Navigate technical assessments and behavioral rounds with confidence</li>
          <li><strong>Industry Navigation</strong> — Understand the AI landscape, compensation benchmarks, and career trajectories</li>
          <li><strong>Network Building</strong> — Connect with the right people and communities in AI</li>
        </ul>
        
        <p><strong>Format:</strong> Bi-weekly 1-hour video sessions with ongoing support via messaging between calls.</p>
      </div>

      <div class="pricing-info">
        <p><strong>Free initial consultation call • $100/hour for ongoing mentoring</strong></p>
      </div>

      <div class="cta-section">
        <a href="mailto:elizaveta.victoria.koroleva@gmail.com?subject=Career Mentoring Inquiry" class="cta-button">Start Your Journey</a>
        <a href="https://www.linkedin.com/in/marva-s" class="cta-button secondary" target="_blank">
          <i class="fa-brands fa-linkedin"></i> Connect on LinkedIn
        </a>
      </div>
    </div>
  </div>

</div>

</div>

<style>
/* Inherit theme variables from your existing style */
:root {
  --services-hover: rgba(0, 0, 0, 0.02);
  --services-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  --services-shadow-hover: 0 6px 16px rgba(0, 0, 0, 0.12);
}

html[data-theme="dark"] {
  --services-hover: rgba(255, 255, 255, 0.05);
  --services-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  --services-shadow-hover: 0 6px 16px rgba(0, 0, 0, 0.3);
}

.services-intro {
  margin: 2rem 0 3rem 0;
}

.services-intro h1 {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  color: var(--global-theme-color);
}

.lead-text {
  font-size: 1.1rem;
  color: var(--global-text-color-light);
  line-height: 1.6;
  margin-bottom: 2rem;
}

.services-grid {
  display: grid;
  gap: 2rem;
  margin: 3rem 0;
}

.service-card {
  background: var(--global-bg-color);
  border-radius: 8px;
  padding: 2rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  border: 1px solid var(--global-divider-color);
  margin-bottom: 2rem;
  color: var(--global-text-color);
}

.service-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
}

.service-card h2 {
  font-size: 1.8rem;
  margin: 0 0 1rem 0;
  color: var(--global-theme-color);
  display: flex;
  align-items: center;
}

.service-card h2 i {
  font-size: 1.5rem;
  margin-right: 0.75rem;
}

.service-description {
  font-size: 1.1rem;
  line-height: 1.6;
  margin: 0 0 1.5rem 0;
  color: var(--global-text-color-light);
}

.service-details h3 {
  font-size: 1.2rem;
  margin: 1.5rem 0 1rem 0;
  color: var(--global-theme-color);
}

.service-details ul {
  list-style: none;
  padding: 0;
}

.service-details li {
  margin-bottom: 0.75rem;
  padding-left: 1.5rem;
  position: relative;
}

.service-details li::before {
  content: "▸";
  position: absolute;
  left: 0;
  color: var(--global-theme-color);
  font-weight: bold;
}

.sub-service {
  margin: 1.5rem 0;
  padding: 1rem;
  border-left: 3px solid var(--global-theme-color);
  background: rgba(0, 0, 0, 0.02);
  border-radius: 0 6px 6px 0;
}

html[data-theme="dark"] .sub-service {
  background: rgba(255, 255, 255, 0.05);
}

.sub-service h4 {
  margin: 0 0 0.5rem 0;
  color: var(--global-theme-color);
  font-size: 1.1rem;
}

.sub-service h4 i {
  margin-right: 0.5rem;
}

.sub-service p {
  margin: 0;
  font-size: 0.95rem;
  line-height: 1.5;
  color: var(--global-text-color);
}

.pricing-info {
  background: rgba(0, 0, 0, 0.02);
  padding: 1rem;
  border-radius: 6px;
  margin: 1.5rem 0;
  border-left: 4px solid var(--global-theme-color);
}

html[data-theme="dark"] .pricing-info {
  background: rgba(255, 255, 255, 0.05);
}

.pricing-info p {
  margin: 0;
  font-weight: 500;
  color: var(--global-text-color);
}

.cta-section {
  text-align: center;
  margin-top: 2rem;
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  justify-content: center;
}

.cta-button {
  display: inline-block;
  background: var(--global-theme-color);
  color: var(--global-bg-color);
  padding: 0.75rem 2rem;
  border-radius: 25px;
  text-decoration: none;
  font-weight: bold;
  transition: all 0.3s ease;
  border: 2px solid var(--global-theme-color);
  flex: 0 0 auto;
}

.cta-button:hover {
  background: transparent;
  color: var(--global-theme-color);
  text-decoration: none;
  transform: translateY(-2px);
}

.cta-button.primary {
  background: var(--global-theme-color);
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.cta-button.secondary {
  background: transparent;
  color: var(--global-theme-color);
  border: 2px solid var(--global-theme-color);
}

.cta-button.secondary:hover {
  background: var(--global-theme-color);
  color: var(--global-bg-color);
}

.cta-button i {
  margin-right: 0.5rem;
}

/* Remove unused styles */

/* Responsive Design */
@media (min-width: 768px) {
  .services-grid {
    grid-template-columns: 1fr;
  }
  
  .contact-buttons {
    flex-direction: row;
  }
  
  .cta-section {
    flex-direction: row;
  }
}

@media (min-width: 1024px) {
  .services-intro h1 {
    font-size: 3rem;
  }
  
  .lead-text {
    font-size: 1.3rem;
  }
}
</style>