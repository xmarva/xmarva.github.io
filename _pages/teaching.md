---
layout: page
permalink: /teaching/
title: teaching
description: Materials for courses you taught. Replace this text with your description.
nav: true
nav_order: 6
---

<div class="course-container" markdown="1">
  <div class="course-card">
    <h2>Transformer Architectures</h2>
    <p><a href="https://github.com/yourusername/transformer-course"><i class="fab fa-github"></i> GitHub Repository</a></p>
    <p class="course-description">{{ course_description }}</p>
<div class="course-section">
  <h3>Part 1: {{ part_1_title }}</h3>
  <table class="table">
    <thead>
      <tr>
        <th>Lecture</th>
        <th>Materials</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>{{ lecture_1_1 }}</td>
        <td class="materials-column">
          <a href="{{ lecture_1_1_materials }}"><i class="fas fa-book"></i> Notes</a>
          <a href="{{ lecture_1_1_notebook }}"><i class="fas fa-code"></i> Notebook</a>
        </td>
      </tr>
      <tr>
        <td>{{ lecture_1_2 }}</td>
        <td class="materials-column">
          <a href="{{ lecture_1_2_materials }}"><i class="fas fa-book"></i> Notes</a>
          <a href="{{ lecture_1_2_notebook }}"><i class="fas fa-code"></i> Notebook</a>
        </td>
      </tr>
      <tr>
        <td>{{ lecture_1_3 }}</td>
        <td class="materials-column">
          <a href="{{ lecture_1_3_materials }}"><i class="fas fa-book"></i> Notes</a>
          <a href="{{ lecture_1_3_notebook }}"><i class="fas fa-code"></i> Notebook</a>
        </td>
      </tr>
    </tbody>
  </table>
</div>

<div class="course-section">
  <h3>Part 2: {{ part_2_title }}</h3>
  <table class="table">
    <thead>
      <tr>
        <th>Lecture</th>
        <th>Materials</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>{{ lecture_2_1 }}</td>
        <td class="materials-column">
          <a href="{{ lecture_2_1_materials }}"><i class="fas fa-book"></i> Notes</a>
          <a href="{{ lecture_2_1_notebook }}"><i class="fas fa-code"></i> Notebook</a>
        </td>
      </tr>
      <tr>
        <td>{{ lecture_2_2 }}</td>
        <td class="materials-column">
          <a href="{{ lecture_2_2_materials }}"><i class="fas fa-book"></i> Notes</a>
          <a href="{{ lecture_2_2_notebook }}"><i class="fas fa-code"></i> Notebook</a>
        </td>
      </tr>
    </tbody>
  </table>
</div>

<div class="course-section">
  <h3>Part 3: {{ part_3_title }}</h3>
  <table class="table">
    <thead>
      <tr>
        <th>Lecture</th>
        <th>Materials</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>{{ lecture_3_1 }}</td>
        <td class="materials-column">
          <a href="{{ lecture_3_1_materials }}"><i class="fas fa-book"></i> Notes</a>
          <a href="{{ lecture_3_1_notebook }}"><i class="fas fa-code"></i> Notebook</a>
        </td>
      </tr>
      <tr>
        <td>{{ lecture_3_2 }}</td>
        <td class="materials-column">
          <a href="{{ lecture_3_2_materials }}"><i class="fas fa-book"></i> Notes</a>
          <a href="{{ lecture_3_2_notebook }}"><i class="fas fa-code"></i> Notebook</a>
        </td>
      </tr>
    </tbody>
  </table>
</div>
  </div>
</div>
<!-- Add this CSS to your _sass or assets/css file -->
<style>
.course-container {
  margin-bottom: 3rem;
}

.course-card {
  background-color: var(--global-bg-color);
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  padding: 2rem;
  margin-bottom: 2rem;
  transition: box-shadow 0.3s ease;
  color: var(--global-text-color);
}

.course-card:hover {
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
}

.course-card h2 {
  margin-top: 0;
  border-bottom: 1px solid var(--global-divider-color);
  padding-bottom: 0.75rem;
  color: var(--global-theme-color);
}

.course-description {
  color: var(--global-text-color-light);
  font-size: 1.1rem;
  margin-bottom: 1.5rem;
}

.course-section {
  margin-top: 1.5rem;
  background-color: var(--global-code-bg-color);
  border-radius: 6px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}

.course-section h3 {
  margin-top: 0;
  font-size: 1.3rem;
  color: var(--global-theme-color);
}

.table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 1rem;
}

.table th, .table td {
  padding: 0.75rem;
  border-bottom: 1px solid var(--global-divider-color);
}

.table th {
  text-align: left;
  font-weight: 600;
  color: var(--global-text-color);
}

.table tbody tr:last-child td {
  border-bottom: none;
}

.table tbody tr:hover {
  background-color: var(--global-hover-color);
}

/* Column width adjustments */
.table th:first-child,
.table td:first-child {
  width: 70%;
}

.table th:last-child,
.table td:last-child {
  width: 30%;
}

.materials-column {
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
}

.materials-column a {
  white-space: nowrap;
  color: var(--global-theme-color);
  text-decoration: none;
}

.materials-column a:hover {
  text-decoration: underline;
}
</style>