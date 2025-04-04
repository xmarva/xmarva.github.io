---
layout: page
title: AI Fashion App
description: We explored two fundamentally different approaches to digital clothing try-on. Our mission was to investigate both image manipulation and 3D mesh-based solutions, each serving distinct use cases in the virtual fashion industry.
img: assets/img/projects/4-ai-fashion-app/main_img.png
importance: 1
category: work
related_publications: false
toc:
  beginning: true

role: Lead ML Engineer

domains: 
  - Computer Vision
  - GenAI
  - 3D

tech:
  - Stable Diffusion
  - Human Pose Estimation
  - 3D Shape Reconstruction

website: https://texelmoda.com
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/4-ai-fashion-app/main_img.png" title="AI Fashion App" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    AI Fashion App
</div>

We explored two fundamentally different approaches to digital clothing try-on. Our mission was to investigate both image manipulation and 3D mesh-based solutions, each serving distinct use cases in the virtual fashion industry.

<br>

## Image-Based Try-On Approach

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/4-ai-fashion-app/virtual_try_on_app.gif" title="Virtual Try On App" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Virtual Try On App
</div>

Our first approach focused on achieving visually perfect try-on results through sophisticated image manipulation techniques. We implemented **[BipartGraph](https://arxiv.org/abs/2106.16076)**, a bipartite graph network that establishes dense correspondences between clothing and body regions.

The core BipartGraph formulation can be expressed as:

$$
\mathcal{G} = \{(\mathcal{V}_c \cup \mathcal{V}_b, \mathcal{E}), \mathcal{A}\}
$$

where $\mathcal{V}_c$ represents clothing nodes, $\mathcal{V}_b$ represents body nodes, and $\mathcal{A}$ is the adjacency tensor.

### Human Parsing and Segmentation

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/4-ai-fashion-app/densepose.png" title="densepose" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    DensePose Inference
</div>

For precise human parsing, we utilized **[SCHP (Self-Correction Human Parsing)](https://arxiv.org/abs/1910.09777)** with a novel hierarchical correction mechanism:

$$
P_{t+1} = \text{SoftMax}(f(P_t, F) + P_t)
$$

where $P_t$ represents parsing results at stage $t$, and $F$ are deep features extracted through **[HRNet](https://arxiv.org/abs/1908.07919)**.

The hierarchical feature fusion is defined as:

$$
Y^h = \sum_{l=1}^L w_l * X^l * M^l
$$

where $M^l$ represents attention masks at different resolutions.

### Diffusion-Based Image Generation

We implemented **[Stable Diffusion](https://arxiv.org/abs/2112.10752)** with custom conditioning for clothing synthesis. The diffusion process follows:

$$
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)
$$

The reverse process is guided by:

$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

### Style Transfer and Refinement

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/4-ai-fashion-app/virtual_try_on.png" title="virtual try on" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Stable Diffusion Based Virtual Try On
</div>

For style preservation, we employed **[AdaIN (Adaptive Instance Normalization)](https://arxiv.org/abs/1703.06868)**:

$$
\text{AdaIN}(x, y) = \sigma(y)(\frac{x - \mu(x)}{\sigma(x)}) + \mu(y)
$$

This was enhanced with **[SPADE (Spatially-Adaptive Normalization)](https://arxiv.org/abs/1903.07291)** for spatial awareness:

$$
\gamma_{c,y,x}(s) \frac{h_{c,y,x} - \mu_c}{\sigma_c} + \beta_{c,y,x}(s)
$$

<br>

## 3D Mesh-Based Approach

Our second approach focused on physical accuracy through detailed 3D modeling and simulation. We implemented **[SMPL-X](https://smpl-x.is.tue.mpg.de/)**, an expressive parametric body model.

### Body Shape Estimation

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/4-ai-fashion-app/smpl_unity.png" title="SMPL in Unity" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    SMPL in Unity
</div>

The SMPL-X model defines body shape through a differentiable function:

$$
M(\beta, \theta, \psi) = W(T_P(\beta, \theta, \psi), J(\beta), \theta, \mathcal{W})
$$

where:
- $\beta$ represents shape parameters
- $\theta$ represents pose parameters
- $\psi$ represents expression parameters
- $W$ is the skinning function

We enhanced this with **[ICON (Implicit Clothed Humans Obtained from Normals)](https://arxiv.org/abs/2112.09127)** for better surface detail:

$$
f_\theta: (x, z, c) \mapsto (s, n)
$$

where $s$ is the occupancy value and $n$ is the predicted surface normal.

### Garment Modeling

For garment simulation, we implemented **[POP (Physics-based Optimization of Patterns)](https://arxiv.org/abs/2203.15720)**. The physical simulation follows:

$$
M\ddot{x} + C\dot{x} + K(x-x_0) = f_{ext}
$$

where:
- $M$ is the mass matrix
- $C$ is the damping matrix
- $K$ is the stiffness matrix
- $x_0$ is the rest pose
- $f_{ext}$ represents external forces

### Contact Handling

We utilized **[SDF-based Contact Handling](https://arxiv.org/abs/2012.07962)** for realistic cloth-body interaction:

$$
E_{contact} = \sum_{p \in \mathcal{P}} w_p \max(0, -\phi(p))^2
$$

where $\phi(p)$ is the signed distance function at point $p$.

<br>