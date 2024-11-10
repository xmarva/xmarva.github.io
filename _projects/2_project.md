---
layout: page
title: Cashierless Store
description: 
img: assets/img/projects/2_cachierless_store/main_img.png
importance: 3
category: work
related_publications: false
toc:
  sidebar: left
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/projects/2_cachierless_store/main_img.png" title="cashierless store" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Cashierless Store
</div>

This project aimed to develop a **fully automated, cashier-free retail experience** for the **[Ministry of Digital Development of the Republic of Tatarstan, Russia](https://digital.tatarstan.ru)**. 

Our mission was to create a seamless and secure environment where customers could browse, select products, and leave without traditional checkout processes, leveraging state-of-the-art computer vision and machine learning techniques.

<br>

## Customer Identification

For secure entry, customers registered with a store assistant, providing identification data, a photograph, and payment information. 

A **personalized access card** was then issued, and **multi-factor identification** was implemented using **facial recognition technology** for an added layer of security.

Our facial recognition model was based on **[FaceNet](https://arxiv.org/pdf/1503.03832)**, a deep convolutional neural network that uses **triplet loss** to map faces into a high-dimensional Euclidean space. 

The triplet loss function, designed to minimize the distance between the embedding of a face and its positive sample while maximizing the distance to a negative sample, is as follows:

$$
\mathcal{L}(A, P, N) = \max(0, ||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + \alpha)
$$

where $$ A $$ is the anchor image, $$ P $$ is a positive image, $$ N $$ is a negative image, $$ f $$ represents the embedding function, and $$ \alpha $$ is a margin. 

By training the network with this loss, we achieved a robust facial recognition system that could handle minor variations in lighting, angles, and expressions.

Implementing FaceNet in real-time presented unique challenges, especially around latency and accuracy in varied lighting. We experimented with model quantization and optimized embedding computation to maintain both speed and accuracy. 

<br>

## Camera System and Calibration

To ensure comprehensive visual coverage, the store was outfitted with strategically placed cameras offering **overlapping fields of view**. 

Calibration was crucial to synchronize data from these multiple perspectives, so we used a **[checkerboard calibration technique](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)** with a **homography matrix** to map points from 2D camera frames to the 3D store layout. 

The homography matrix $$ H $$ for a transformation between two planes can be defined as:

$$
\mathbf{x'} = H \cdot \mathbf{x}
$$

where $$ \mathbf{x} $$ represents points in the camera view and $$ \mathbf{x'} $$ represents mapped points in the store’s 3D coordinate system.

We faced challenges around maintaining calibration stability due to minor shifts in camera positions and changes in store lighting. Regular recalibration schedules were set, and camera lens distortion correction was applied to reduce discrepancies between feeds. 

<br>

## Virtual Mapping and Tracking

For customer tracking, we used **multi-camera tracking algorithms** based on **deep sort** and **[Kalman filters](https://en.wikipedia.org/wiki/Kalman_filter#:~:text=The%20Kalman%20filter%20produces%20an,uncertainty%20are%20%22trusted%22%20more.)**. 

Multi-camera tracking in a crowded environment is complex, as it requires handling occlusions, re-identifications, and maintaining continuity across camera views.

The Kalman filter predicts the position and velocity of each customer as they move, using the following state transition:

$$
\mathbf{x}_{k} = F \cdot \mathbf{x}_{k-1} + G \cdot \mathbf{u}_{k-1} + \mathbf{w}_{k-1}
$$

where $$ F $$ is the state transition matrix, $$\mathbf{u}_{k-1} $$ is the control vector, and $$ \mathbf{w}_{k-1} $$ is process noise. 

With **deep sort**, re-identification embeddings were generated to handle instances when customers were briefly occluded or left the view of one camera, then reappeared in another. 

We also implemented **Non-Maximum Suppression (NMS)** to reduce false positives from overlapping detections.

<br>

## Object Detection and Recognition

Our object recognition system was powered by **[EfficientDet](https://arxiv.org/pdf/1911.09070)**, a state-of-the-art object detection model. EfficientDet utilizes a **bi-directional feature pyramid network (BiFPN)** and compound scaling, allowing for optimal accuracy and computational efficiency. 

Unlike traditional object detection models, EfficientDet can scale efficiently across different input sizes and feature resolutions, which is crucial for real-time performance in dynamic retail settings.

The objective function used in EfficientDet is based on **weighted focal loss**, which addresses class imbalance issues by down-weighting easy examples:

$$
\text{FL}(p_t) = - \alpha_t (1 - p_t)^\gamma \log(p_t)
$$

where $$ p_t $$ is the predicted probability of the true class, $$ \alpha_t $$ is a weighting factor, and $$ \gamma $$ adjusts the rate at which easy examples are down-weighted. This formulation allows EfficientDet to prioritize hard-to-detect items and improves overall detection quality.

<br>

## Action Recognition

To track customer interactions with products, we implemented an action recognition model based on **I3D (Inflated 3D ConvNet)** architecture, known for its effectiveness in capturing spatiotemporal features by inflating 2D filters into 3D. 

The I3D model applies 3D convolutions across consecutive frames, which allows it to capture motion, direction, and object interactions effectively.

The I3D model learns temporal patterns by extending 2D convolutional filters along the temporal dimension:

$$
f(x) = \sigma \left( W_{3D} * x + b \right)
$$

where $$ W_{3D} $$ is a 3D convolutional kernel, $$ x $$ is the input video clip, and $$ \sigma $$ is the activation function. 

This design enables the model to capture motion patterns across frames, effectively identifying actions like “picking up,” “examining,” or “returning” items.

<br>
