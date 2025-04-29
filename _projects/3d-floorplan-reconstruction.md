---
layout: page
title: 3D Floorplan Reconstruction 
description: Our mission was to transform raw architectural blueprints into realistic and accurate 3D representations that could be explored in any browser—or even with VR headsets.
img: assets/img/projects/3-vr-floorplan-reconstruction/main_img.png
importance: 2
category: work
related_publications: false
toc:
  beginning: true

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
    VR Floorplan Reconstruction
</div>

The goal of [this project](https://getfloorplan.com) was to develop a solution for automatically creating 3D floor plans and virtual tours. Our mission was to transform raw architectural blueprints into realistic and accurate 3D representations that could be explored in any browser or with VR headsets, providing a valuable tool for real estate and architectural visualization.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid path="https://www.youtube.com/embed/ZECQHFttNLQ?si=X_XsCcJDADp0hXU5" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## Dataset

We curated a dataset of over 50,000 annotated floorplans, focusing on key architectural elements like walls, windows, doors, and dimension markers. 

To accelerate annotation, we leveraged [Segment Anything Model (SAM)](https://arxiv.org/abs/2304.02643) for preliminary segmentation masks, which reduced manual labeling time by 40%. 

Final annotations included pixel-level masks for structural elements and furniture placement zones, validated through a hybrid pipeline combining [Labelbox](https://labelbox.com) tools and custom quality checks.

<br>

## Floorplan Recognition

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/3-vr-floorplan-reconstruction/3d-example.png" title="3D FloorPlan Reconstruction" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    3D FloorPlan Reconstruction
</div>

Our core segmentation stack used a [U-Net](https://arxiv.org/abs/1505.04597) architecture with a [ResNet-50](https://arxiv.org/abs/1512.03385) backbone, optimized for architectural drawings through domain-specific augmentations like line thickness variations and simulated scan artifacts. 

The model achieved 94.2% mIoU on our test set by leveraging multi-scale feature fusion and a hybrid loss combining Dice and boundary-aware losses. For challenging elements like curved walls, we integrated [Mask R-CNN](https://arxiv.org/abs/1703.06870) as a refinement stage to capture fine geometric details.

## Object Detection

Element localization used [EfficientDet-D2](https://arxiv.org/abs/1911.09070) trained with focal loss to handle class imbalance between large structural components and small decorative elements. 

We introduced rotational bounding boxes to better represent angled architectural features, improving window/door detection accuracy by 18% compared to standard axis-aligned approaches.

<br>

## Spatial Coordinate Transformation

Converting 2D layouts to 3D required solving perspective distortions and inconsistent scaling. 

We developed a hybrid approach using [homography estimation](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html) for planar alignment and [RANSAC](https://www.cs.ubc.ca/~lowe/papers/ijcv05.pdf)-based outlier rejection to handle drawing artifacts. 

Scale inference combined explicit dimension markers with learned priors about standard room sizes, achieving ±3% dimensional accuracy compared to ground-truth LIDAR scans.

<br>

## Automatic Furnishing

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid path="https://www.youtube.com/embed/2i-dVo22lfg?si=vCV6JhYxwyXiaLDZ" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The furnishing system combined rule-based layout principles with learned style preferences. Using [Graph Neural Networks](https://arxiv.org/abs/1812.08434), we modeled spatial relationships between furniture items and room dimensions.

The system generated 3D renders in [Unreal Engine 5](https://www.unrealengine.com) using procedural material generation for realistic textures.

<br>

## System Integration

The final pipeline processed floorplans in under 90 seconds per 100m², leveraging [ONNX Runtime](https://onnxruntime.ai/) for model inference and [Open3D](http://www.open3d.org) for 3D reconstruction. 

VR exports used [WebXR](https://immersive-web.github.io/webxr-samples/) for browser compatibility, while high-fidelity renders utilized Unreal Engine's [Nanite](https://docs.unrealengine.com/5.0/en-US/nanite-virtualized-geometry-in-unreal-engine/) geometry system. 

The solution reduced 3D modeling costs by 70% for real estate clients compared to manual workflows.