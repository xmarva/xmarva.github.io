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
  - Computer Vision---
layout: page
title: AI Fashion App
description: We explored two fundamentally different approaches to digital clothing try-on. Our mission was to investigate both image manipulation and 3D mesh-based solutions, each serving distinct use cases in the virtual fashion industry.
img: assets/img/projects/4-ai-fashion-app/main_img.png
importance: 1
category: work
related_ublications: false
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

## Image-Based Virtual Try-On Pipeline

The 2D clothing synthesis system revolved around a custom control network architecture developed in-house, drawing inspiration from ControlNet but optimized for fashion-specific constraints. 

Unlike traditional approaches, our implementation used a dual-stream encoder that processed both body pose estimates from OpenPose and semantic segmentation masks from SCHP simultaneously. This allowed precise spatial control over garment placement while maintaining texture consistency.

<div class="row">
    <div class="col-sm mt-3 mt-md-0" style="max-height: 400px; overflow: hidden;">
        {% include figure.liquid loading="eager" 
        path="assets/img/projects/4-ai-fashion-app/virtual_try_on_app.gif" 
        title="Virtual Try On App" 
        class="img-fluid rounded z-depth-1"
        style="object-fit: contain; width: 100%; height: 100%;" %}
    </div>
</div>
<div class="caption">
    Real-Time Try-On App
</div>

Key innovations in our control network included attention-guided warping modules that adapted garment textures to individual body proportions, and a differentiable renderer that simulated fabric-light interactions. 

The warping process used thin-plate spline transformations guided by 68 body landmark points extracted through a modified HRNet architecture. 

To handle complex draping effects, we introduced a physics-inspired regularization loss that penalized unrealistic cloth deformation patterns.

## Diffusion-Based Synthesis with Spatial Constraints

Building on Stable Diffusion we developed a multi-conditioned generation pipeline that combined textual prompts with geometric constraints. 

The system accepted three input types: product images from flat lays, textual descriptions of materials, and body measurement profiles. 

A novel cross-attention mechanism dynamically weighted these inputs based on garment category - for example, prioritizing texture details for silk blouses versus geometric precision for structured blazers.

The diffusion process was guided by a composite loss function combining VGG-based perceptual loss, face identity preservation using ArcFace embeddings, and a fabric consistency metric derived from CLIP space comparisons.

To accelerate inference, we implemented progressive latent space refinement that reduced generation time from 12 seconds to 3.8 seconds per image on A100 GPUs.

## 3D Body Modeling

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/4-ai-fashion-app/smpl_unity.png" title="SMPL in Unity" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    SMPL-X Body Model with Measurement Overlays
</div>

For the 3D pipeline, we extended the SMPL-X parametric model with measurement-aware shape estimation. 

A custom CNN architecture predicted 92 anthropometric parameters from single RGB images, achieving ±1.5cm accuracy on key measurements like bust, waist, and hip circumferences. 

The model was trained on synthetic data generated using Unreal Engine's MetaHuman framework, augmented with real-world scanning data from our partner tailors.

## Physics-Based Garment Simulation

The 3D garment pipeline combined projective dynamics for fast convergence with XPBD constraints for accurate cloth behavior. 

We developed a material authoring system that converted real fabric swatches into simulation parameters through spectral analysis of drape patterns and CNN-based stiffness prediction. 

Collision handling used hierarchical signed distance fields that adaptively refined resolution near contact points, reducing computation time by 40% compared to uniform grids.

Garment patterns were parameterized using UV maps that encoded seam allowances and stitching constraints. During simulation, these maps guided the virtual sewing process, ensuring realistic garment construction. 

The system supported multi-layer outfits through depth-aware collision detection that properly handled interactions between base layers and outerwear.

## Production Deployment Architecture

The final system used a microservices architecture with ONNX Runtime for model inference and Redis for caching frequent measurements. Web-based 2D try-ons leveraged WebGL with progressive texture streaming, while the 3D pipeline integrated with Unity's HDRP for VR experiences. 

A shared GraphQL API managed user sessions and synchronized state between different try-on modalities, maintaining under 500ms latency for 95% of requests.