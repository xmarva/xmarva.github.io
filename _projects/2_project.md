---
layout: page
title: Cashierless Store
description: Creating a seamless and secure environment where customers could browse, select products, and leave without traditional checkout processes.
img: assets/img/projects/2_cachierless_store/main_img.png
importance: 3
category: work
related_publications: false
toc:
  beginning: true

role: ML Engineer

domains: 
  - Computer Vision

tech:
  - OpenCV
  - ResNet
  - MultiCamera Object Tracking

website: https://digital.tatarstan.ru
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/2_cachierless_store/main_img.png" title="cashierless store" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Cashierless Store
</div>

This project aimed to develop a fully automated, cashier-free retail experience for the [Ministry of Digital Development of the Republic of Tatarstan, Russia](https://digital.tatarstan.ru). 

Our mission was to create a seamless and secure environment where customers could browse, select products, and leave without traditional checkout processes. 

<br>

## Customer Identification

For secure entry, customers registered via a kiosk system combining biometric authentication and RFID-enabled access cards. 

The facial recognition pipeline leveraged [ArcFace](https://arxiv.org/abs/1801.07698), a state-of-the-art model that uses additive angular margin loss to enhance feature discriminability. 

We integrated live anti-spoofing checks using vision transformer architectures to detect presentation attacks. For edge deployment, we quantized the model using [TensorRT](https://developer.nvidia.com/tensorrt) and optimized inference latency to under 80ms per frame on NVIDIA Jetson devices.

<br>

## Camera System and Calibration

The store deployed a hybrid camera system combining stereo depth sensors (Intel RealSense D455) and 4K PTZ cameras for wide-area coverage. 

Calibration used a combination of traditional checkerboard patterns for intrinsic parameters and deep learning-based methods for dynamic distortion correction. Overhead fisheye cameras were processed using distortion-invariant object detection models to mitigate occlusion challenges in crowded areas.

<br>

## Virtual Mapping and Tracking

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/2_cachierless_store/3.jpg" title="Virtual Mapping and Tracking" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Virtual Mapping and Tracking
</div>

Customer tracking combined [DeepSORT](https://arxiv.org/abs/1703.07402) for robust association of detections across frames and Kalman filters for motion prediction. 

To handle cross-camera re-identification, we trained a transformer-based model ([TransReID](https://arxiv.org/abs/2106.03896)) on synthetic data. Real-time tracking latency was maintained at <200ms using [ONNX Runtime](https://onnxruntime.ai/) with CUDA acceleration.

<br>

## Object Detection and Recognition

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/2_cachierless_store/4.png" title="Object Detection" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Object Detection
</div>

Product recognition used a cascaded approach: [YOLOv8](https://arxiv.org/abs/2304.00501) for real-time item localization followed by [DETR-ResNet50](https://arxiv.org/abs/2005.12872) for fine-grained classification. 

To distinguish visually similar products, we incorporated metric learning with triplet loss, trained on a dataset of 5000 SKUs. For edge cases like occluded items, we fused 2D visual data with RFID spatial signals from smart shelves.

<br>

## Action Recognition

To detect product interactions, we deployed a temporal model combining [SlowFast Networks](https://arxiv.org/abs/1812.03982) for multi-speed feature extraction and MediaPipe Hands for precise hand localization. 

The system achieved 89.4% F1-score on the [EPIC-KITCHENS](https://epic-kitchens.github.io/2023) benchmark, validated against real-world shopping scenarios.

<br>

## System Integration

The architecture used microservices for modular scalability, with gRPC ensuring low-latency communication between components. 

Camera streams were processed through an optimized video analytics pipeline using [NVIDIA DeepStream](https://developer.nvidia.com/deepstream-sdk). The final system supported 150 concurrent users with end-to-end latency under 1.2 seconds.