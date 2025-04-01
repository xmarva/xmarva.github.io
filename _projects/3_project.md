---
layout: page
title: 3D Florplan Reconstruction 
description: Our mission was to transform raw architectural blueprints into realistic and accurate 3D representations that could be explored in any browser—or even with VR headsets.
img: assets/img/projects/3-vr-floorplan-reconstruction/main_img.png
importance: 2
category: work
related_publications: false

role: Computer Vision Engineer

domains: 
  - Computer Vision

tech:
  - SAM
  - Mask-RCNN
  - ResNet
  - UnrealEngine

website: https://getfloorplan.com
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/3-vr-floorplan-reconstruction/main_img.png" title="3D Florplan Reconstruction" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    VR Florplan reconstruction
</div>


The goal of [this project](https://getfloorplan.com) was to develop a solution for automatically creating 3D floor plans and virtual tours. This kind of software is incredibly useful for architectural firms and real estate agents, as it provides a visually engaging way to showcase properties to potential buyers.

Our mission was to transform raw architectural blueprints into realistic and accurate 3D representations that could be explored in any browser—or even with VR headsets.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid path="https://www.youtube.com/embed/ZECQHFttNLQ?si=X_XsCcJDADp0hXU5" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

To accurately recognize elements in floor plan drawings, we built a computer vision model system. For reconstructing those elements, our game development team used Unreal Engine. Additionally, we developed an algorithm for semi-automatically placing furniture, leveraging design presets and accounting for the identified parameters of the living space.

## Dataset

We focused heavily on creating and annotating a high-quality, diverse dataset. Initially, we relied on manual efforts and tools like LabelBox. Later, we integrated segmentation models. While their performance wasn’t yet good enough for final outputs, they significantly sped up the manual annotation process.

Key components in the floor plans included:

- Walls
- Windows and Doors
- Scale and Dimension Indicators

In addition to these, the annotations also captured masks for various secondary objects, furniture placement markers, and more.

## Floorplan Recognition

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/3-vr-floorplan-reconstruction/3d-example.png" title="3D FloorPlan Resonctruction" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    3D FloorPlan Resonctruction
</div>

### Segmentation Models

We developed a custom deep learning architecture based on the **U-Net** model with a **ResNet** backbone, specifically tailored for architectural element detection.

The core of our segmentation approach was the **Dice Loss Function**, which allowed for precise pixel-level classification:

$$\text{Dice Loss} = 1 - \frac{2 \sum_{i} p_i g_i}{\sum_{i} p_i^2 + \sum_{i} g_i^2}$$

This formula enabled our model to achieve exceptional accuracy in identifying complex architectural elements, even in challenging or partially obscured drawings.

### Object Detection Refinement

For enhanced detection precision, we implemented **EfficientDet** with a custom architectural configuration. The model utilized a sophisticated Focal Loss approach to handle class imbalance:

$$\text{FL}(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)$$

This advanced technique allowed our system to prioritize and accurately detect small-scale architectural elements that traditional object detection methods might miss.

<br>

## Spatial Coordinate Transformation

One of the most challenging aspects of our project was converting 2D drawings into accurate 3D representations. We developed a sophisticated coordinate correction algorithm that goes beyond simple transformation:

$$\mathbf{X}_{\text{corrected}} = \mathbf{X}_{\text{detected}} + \epsilon(\text{spatial constraints})$$

This innovative approach incorporates:

- Geometric constraint validation
- Spatial relationship analysis
- Error correction using advanced filtering techniques


## Automatic Furnishing

We developed a groundbreaking algorithm for automatic interior design that goes beyond simple object placement. Our system understands architectural context and applies intelligent furnishing rules.

The core of our placement strategy is a comprehensive scoring function:

$$\text{Placement Score} = f(\text{Spatial Proximity}, \text{Style Compatibility}, \text{Functional Constraints})$$


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid path="https://www.youtube.com/embed/2i-dVo22lfg?si=vCV6JhYxwyXiaLDZ" class="img-fluid rounded z-depth-1" %}
    </div>
</div>