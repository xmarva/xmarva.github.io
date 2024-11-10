---
layout: page
title: Cashierless store
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
    Cachierless Store
</div>

This project aimed to build a fully automated, cashier-free retail experience for the Ministry of Digital Development of the Republic of Tatarstan, Russia.

Our goal was to create an intuitive shopping environment where customers could enter, pick up products, and leave without going through a traditional checkout.

<br>

## Customer Identification

To gain access, customers registered with a store assistant, providing their passport, personal photo, and payment information. Once registered, each customer was issued a personalized access card.

Upon arriving at the store, customers would pass through turnstiles using this access card. The system also verified customer identity through facial recognition, creating a seamless and secure entry process.

Each customer received a unique identifier upon entry, enabling tracking throughout their entire visit.

<br>

## Camera System and Calibration

The store was equipped with multiple inexpensive cameras, similar to those used in standard video surveillance.

Positioned throughout the store, these cameras had overlapping fields of view to cover the entire area without gaps.

To ensure consistent data from these multiple perspectives, we calibrated the cameras using a checkerboard pattern, allowing for precise alignment and accurate tracking across feeds.

<br>

## Virtual Mapping and Tracking

To monitor each customer’s movements, we projected their route as a virtual dot on a digital map of the store.

As they moved through the aisles, the dot moved in real-time, providing a continuous, mapped record of their location and activity.

This enabled not only accurate tracking but also allowed us to analyze traffic patterns and identify popular sections within the store.

<br>

## Object Detection and Recognition

For recognizing products and tracking customer interactions, we built an object detection model using a custom dataset of product images.

Every item in the store's inventory was represented by a set of images used to train the model. This allowed for high recognition accuracy even with varying product orientations and lighting conditions.

Our object detection system relied on architectures such as YOLO, which are known for real-time, high-precision object recognition and classification.

<br>

# Action Recognition

We also integrated action recognition models focused on customer interactions with products.

This system could distinguish between actions such as picking up, examining, or placing items back on shelves.

Recognizing these actions was very important for automating the checkout process, as it allowed us to accurately record products being "purchased" or returned without direct checkout scanning.

<br>
