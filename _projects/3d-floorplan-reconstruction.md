---
layout: page
title: 3D Floorplan Reconstruction
description: "Turning an architectural floor plan, scanned, photographed or exported, into a furnished 3D apartment you can walk through in a game engine or a VR headset"
img: assets/img/projects/3-vr-floorplan-reconstruction/figures/card.png
cover: /assets/img/projects/3-vr-floorplan-reconstruction/figures/cover.png
og_image: https://xmarva.github.io/assets/img/projects/3-vr-floorplan-reconstruction/figures/cover.png
stylesheets: [/assets/css/tfs-course.css]
importance: 2
category: work
related_publications: false

role: Computer Vision Engineer

domains:
  - Computer Vision
  - 3D

tech:
  - Mask R-CNN
  - U-Net
  - OpenCV
  - Unreal Engine 4

website: https://getfloorplan.com
---

{% raw %}

In the Louvre there is a diorite statue of Gudea, a ruler of Lagash, made around 2120 BC. He sits with a tablet on his knees, and on the tablet is the plan of a building, drawn from above, with a stylus and a graduated ruler beside it ([Louvre, AO 2](https://collections.louvre.fr/en/ark:/53355/cl010119540)).

The convention has hardly changed in four thousand years. A plan is a building seen from directly above, with the walls cut about a metre above the floor and drawn as thick lines, and a scale that tells you how long a line is.

Architects read it at a glance. Most people buying a flat do not: they see lines and arcs and try to imagine a room.

From November 2018 to December 2019 I worked on the machine learning side of a product that did the imagining for them. A real-estate company sends a plan as an image, and gets back the apartment in 3D, furnished, ready to walk through in a game engine or in a VR headset.

This page is about how a plan is read by a machine, what made it hard, and how an empty model becomes a room someone might live in. The figures are computed on one real plan from the project, a small studio with a balcony.

## The plan as a picture

<img class="tfs-figure tfs-light" src="/assets/img/projects/3-vr-floorplan-reconstruction/figures/pipeline-light.png" width="760" alt="The pipeline in two rows. A plan image, scanned, photographed or exported as a PDF, goes through segmentation of walls, doors and rooms, then becomes vectors, rectangles and polygons; the scale comes from doors and dimension text; the result is a 3D model of walls, floors and openings. Then rooms get their types from fixtures and sizes, furniture is placed by a search with rules, materials are chosen per room style, and the scene goes to Unreal Engine 4 for a walk-through in VR or in a browser" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/3-vr-floorplan-reconstruction/figures/pipeline-dark.png" width="760" alt="The pipeline in two rows. A plan image, scanned, photographed or exported as a PDF, goes through segmentation of walls, doors and rooms, then becomes vectors, rectangles and polygons; the scale comes from doors and dimension text; the result is a 3D model of walls, floors and openings. Then rooms get their types from fixtures and sizes, furniture is placed by a search with rules, materials are chosen per room style, and the scene goes to Unreal Engine 4 for a walk-through in VR or in a browser" loading="lazy">

A plan arrives as pixels. Some are exported from architectural software and are perfectly clean; some are scans of paper, photos of a printout on a kitchen table, or screenshots of a developer's brochure with a logo on top.

The first job is to find the three things the 3D model is made of: walls, openings and rooms. On a clean drawing this can be done with a handful of image operations, and doing it that way shows what each step needs to know.

<img class="tfs-figure tfs-light" src="/assets/img/projects/3-vr-floorplan-reconstruction/figures/reading-light.png" width="760" alt="Four panels of the same studio plan: the drawing; the walls left after a morphological opening, with thin lines removed; the rooms found as connected floor, coloured and labelled (living room and hall 22.8 m²; bathroom 3.5 m²; balcony 2.4 m²; wardrobe 1.6 m²; closet 0.9 m²); and the doors, found as thin drawings that touch two rooms, with their widths (148 px, 142 px, 58 px, 56 px, 73 px). Scale 1.23 cm per pixel" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/3-vr-floorplan-reconstruction/figures/reading-dark.png" width="760" alt="Four panels of the same studio plan: the drawing; the walls left after a morphological opening, with thin lines removed; the rooms found as connected floor, coloured and labelled (living room and hall 22.8 m²; bathroom 3.5 m²; balcony 2.4 m²; wardrobe 1.6 m²; closet 0.9 m²); and the doors, found as thin drawings that touch two rooms, with their widths (148 px, 142 px, 58 px, 56 px, 73 px). Scale 1.23 cm per pixel" loading="lazy">

Walls are drawn as the thickest strokes. If you erode the image with a 5 by 5 square and grow it back, everything thinner than the square disappears, and what is left are the walls: the morphological opening of the dark pixels. Small dark pieces that survive, like the burners of the stove, are dropped by size.

Rooms are the pieces of floor between the lines. Here every drawn line counts as a barrier, including door leaves and the arcs of their swing, so a door closes its room. The floor then falls apart into five connected regions: the living room with its kitchen corner and hall, the bathroom, a closet, a built-in wardrobe and the balcony.

Doors are the thin drawings that touch two rooms at once, or a room and the outside. Their sizes in pixels, 56 and 58 for the bathroom's two doors, 73 for the entrance and 148 for the double balcony door, are what the scale will come from.

## Why rules were not enough

Rules like these are written for one style of drawing. A different architect's office draws walls hatched instead of solid, doors as gaps without arcs, furniture with lines as thick as walls, and the rules quietly fail.

A photographed plan is worse. The figure runs exactly the same threshold and opening on the same plan after a phone photo of a printout: a little perspective, a shadow from the phone, blur, sensor noise and some handwriting.

<img class="tfs-figure tfs-light" src="/assets/img/projects/3-vr-floorplan-reconstruction/figures/scan-light.png" width="760" alt="Three panels: the plan as if photographed from a printout, tilted, with a shadow in one corner, noise and handwritten scribbles; the walls the same rules find on it, with the shadowed corner turned solid; and a comparison with the true walls, where false walls are coloured and almost nothing is missed" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/3-vr-floorplan-reconstruction/figures/scan-dark.png" width="760" alt="Three panels: the plan as if photographed from a printout, tilted, with a shadow in one corner, noise and handwritten scribbles; the walls the same rules find on it, with the shadowed corner turned solid; and a comparison with the true walls, where false walls are coloured and almost nothing is missed" loading="lazy">

In the shadow the floor itself turns dark enough to count as wall, and the door leaves there thicken into walls too. Almost no true wall is missed, but nearly half of what the rules call walls is something else, and the overlap with the true walls drops to 55%. Each fault could be fixed by another rule, and every rule would break some other plan.

That is why the segmentation itself was done by neural networks trained on annotated plans. The published work of the time showed both the promise and the cost. Raster-to-Vector, at ICCV 2017, predicted wall corners and door ends with a network and joined them by integer programming, at "around 90% precision and recall" ([Liu et al., 2017](http://art-programmer.github.io/floorplan-transformation.html); [code](https://github.com/art-programmer/FloorplanTransformation)).

It was trained on 870 plans annotated by hand, out of a collection of five million. Its authors also noted a limit: "Relying on Manhattan assumption, our method is unable to detect walls which are neither horizontal nor vertical".

Researchers at Arizona State University and Rakuten segmented walls with a fully convolutional network and detected objects with Faster R-CNN ([Dodge et al., 2017](https://web.archive.org/web/20201111170058/http://www.mva-org.jp/Proceedings/2017USB/papers/10-01.pdf)). Mask R-CNN gave a mask for every detected object ([He et al., 2017](https://arxiv.org/abs/1703.06870)), and the most used open implementation of it was published by Matterport, a company that makes 3D tours of real estate ([code](https://github.com/matterport/Mask_RCNN)).

The data was the bottleneck. In April 2019, in the middle of our project, CubiCasa released 5,000 Finnish plans annotated into more than 80 categories ([Kalervo et al., 2019](https://arxiv.org/abs/1904.01920); [code](https://github.com/CubiCasa/CubiCasa5k)). They report that one plan took "from 5 to 120 minutes" to annotate. Their licence is non-commercial, so for a product the annotation still had to be our own.

## The scale

A plan in pixels has no size. Everything in the 3D model, the height of the walls, the width of a bed, whether a sofa fits, depends on knowing how many centimetres a pixel is.

Some plans print dimension lines with numbers, and reading them means OCR on small rotated digits. Tesseract switched to an LSTM recognizer in its 4.0 release in October 2018 ([Tesseract 4.0.0](https://github.com/tesseract-ocr/tesseract/releases/tag/4.0.0)). Japanese plans write each room's area in tatami units, and Dodge and colleagues read those labels with OCR to get pixels per tatami; their OCR found "74.1% of the room size annotations".

When a plan prints nothing, the doors are the most reliable ruler. Door sizes are standardized: the Russian standard in force then, GOST 475-2016, gives openings of 710, 810, 910 and 1010 mm, and the older GOST 6629-88 drew door leaves of 600, 700, 800 and 900 mm ([GOST 475-2016](https://meganorm.ru/Data2/1/4293749/4293749722.htm)).

On this plan, taking interior doors as 70 cm and the entrance as 90 cm gives three estimates that agree within 4%, and a scale of 1.23 cm per pixel. The flat comes out at 31.2 m², with a 22.8 m² main room.

<img class="tfs-figure tfs-light" src="/assets/img/projects/3-vr-floorplan-reconstruction/figures/scale-light.png" width="760" alt="A line chart of the flat&#x27;s floor area against the width assumed for an interior door, from 55 to 95 cm; the area grows with the square of the assumed width, marked at 60, 70, 80 and 90 cm" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/3-vr-floorplan-reconstruction/figures/scale-dark.png" width="760" alt="A line chart of the flat&#x27;s floor area against the width assumed for an interior door, from 55 to 95 cm; the area grows with the square of the assumed width, marked at 60, 70, 80 and 90 cm" loading="lazy">

The assumption carries a lot of weight. Area grows with the square of the scale, so if the bathroom door is really 60 cm, the same flat shrinks to about 23 m², and at 80 cm it grows to about 41. A model that disagrees with the developer's brochure by ten square metres is useless, which is why each estimate had to be checked against the others.

## From a drawing to a model

Once the walls are known in metres, the model itself is simple geometry. The wall mask is cut into rectangles, each rectangle is raised into a box, and the floors are the room regions.

<img class="tfs-figure tfs-light" src="/assets/img/projects/3-vr-floorplan-reconstruction/figures/model-light.png" width="760" alt="Left: the plan with its walls covered by 48 rectangles. Right: those rectangles extruded to full height in an axonometric view, with the plan&#x27;s own drawing of doors and fixtures on the floors and each room tinted" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/3-vr-floorplan-reconstruction/figures/model-dark.png" width="760" alt="Left: the plan with its walls covered by 48 rectangles. Right: those rectangles extruded to full height in an axonometric view, with the plan&#x27;s own drawing of doors and fixtures on the floors and each room tinted" loading="lazy">

Wall height is the one number the plan does not contain. Dodge and colleagues simply wrote: "In absence of this information, we arbitrarily choose a height." For Russian flats the building code gives a floor: living rooms and kitchens at least 2.5 m, or 2.7 m in the colder regions ([SP 54.13330.2016](https://web.archive.org/web/20191226232843/http://docs.cntd.ru/document/456054198)).

The doors and windows become openings in the boxes, and the plan's own drawing of the bathtub and the stove tells the engine where to place real fixtures.

Curved and diagonal walls were the persistent trouble. A rectangle cover handles them badly, and the methods built on corner junctions assumed a Manhattan world. Zeng and colleagues showed a network that handled curved walls and walls of uneven thickness at ICCV in the autumn of 2019, shortly before the project ended ([Zeng et al., 2019](https://arxiv.org/abs/1908.11025); [code](https://github.com/zlzeng/DeepFloorplan)).

## Furniture

An empty apartment in 3D looks like a warehouse. The product placed furniture automatically, and the rules for it are older than deep learning.

Merrell and colleagues turned interior design guidelines into terms of a cost function: clearance in front of furniture, circulation paths through the room, pairs of objects that belong together, balance, alignment ([Merrell et al., 2011](https://paulmerrell.org//furnitureLayout2.pdf)). "Make it Home", from the same year, optimized a similar cost by "simulated annealing using a Metropolis-Hastings state search step" ([Yu et al., 2011](https://ww3.math.ucla.edu/camreport/cam11-49.pdf)).

The figure does a small version of this in the studio's main room. A bed, a desk, a table and an armchair are placed one by one at random positions where they fit; the bed's head and the desk's back must touch a wall, and each piece keeps a clearance zone in front of it free of the others.

A layout is rejected if a person can no longer walk from the entrance to every door along a path at least 70 cm wide. Among the layouts that pass, the best keeps the bed far from the entrance, the table near the kitchen and the desk near the balcony light.

<img class="tfs-figure tfs-light" src="/assets/img/projects/3-vr-floorplan-reconstruction/figures/furnish-light.png" width="760" alt="Left: the plan with a bed, a desk, a table and an armchair placed in the living room, dashed zones in front of each, and the floor where a person can still walk shaded, joining every door. Right: the same layout in 3D. It is the best of 290 layouts that passed the rules" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/3-vr-floorplan-reconstruction/figures/furnish-dark.png" width="760" alt="Left: the plan with a bed, a desk, a table and an armchair placed in the living room, dashed zones in front of each, and the floor where a person can still walk shaded, joining every door. Right: the same layout in 3D. It is the best of 290 layouts that passed the rules" loading="lazy">

Out of 3,000 random attempts, 290 passed. Every one of the others failed in the same way: the furniture fitted, but the path was blocked. A bed that fits the room perfectly can cut the hall in two.

Deep learning arrived in this task while we worked. Wang and colleagues learned where objects go from top-down images of rooms at SIGGRAPH 2018, taking "an average overall room synthesis time of 4 minutes" ([Wang et al., 2018](https://kwang-ether.github.io/pdf/deepsynth.pdf)), and Ritchie and colleagues cut that to "under 2 seconds" at CVPR 2019 ([Ritchie et al., 2019](https://arxiv.org/abs/1811.12463)).

Both learned from SUNCG, a dataset of 45,622 indoor scenes designed by people. In June 2019 Planner 5D sued over where the data came from, and the dataset disappeared from the web; a lab's code page from that month reads: "Due to an ongoing legal dispute, SUNCG is currently unavailable to download" ([fast-synth](https://github.com/brownvc/fast-synth)).

## Walking through it

<img class="img-fluid rounded d-block mx-auto my-4" src="/assets/img/projects/3-vr-floorplan-reconstruction/ue4.png" width="1678" height="1042" alt="A rendered kitchen in a small apartment: red cabinets, a stove, a sink, a steel fridge, a light wooden floor and a corridor to the front door. A ring of coloured circles in the middle of the view, labelled Закрыть, offers colours for the cabinets, and blue pins mark objects that can be changed" loading="lazy">

<p class="tfs-caption">An apartment in Unreal Engine 4, from the project: the ring in the middle changes the colour of the kitchen.</p>

The finished model went to a game engine, where it could be lit, textured and walked through. Epic had opened Unreal Studio in March 2018 with a toolkit that "simplifies importing CAD and 3ds Max data" into Unreal Engine 4 ([Epic Games, 2018](https://web.archive.org/web/20190925131023/https://www.unrealengine.com/en-US/blog/announcing-unreal-studio-now-in-open-beta)).

VR headsets also changed during the project. The Oculus Go, the company's first standalone headset, went on sale for &#36;199 in May 2018 ([Facebook, 2018](https://about.fb.com/news/2018/05/oculus-go/)), and the Oculus Quest, "our first all-in-one VR gaming system" as the company put it, followed in May 2019 for &#36;399 ([Oculus, 2019](https://web.archive.org/web/20191208062944/https://www.oculus.com/blog/game-on-oculus-quest-and-rift-s-pre-orders-are-live-shipping-begins-may-21/)).

With a headset a buyer could stand in a kitchen that did not exist yet, look around, and change the colour of the cabinets.

## References

Reading plans:

- Liu, Wu, Kohli and Furukawa. [Raster-to-Vector: Revisiting Floorplan Transformation](http://art-programmer.github.io/floorplan-transformation.html). ICCV 2017. [Code](https://github.com/art-programmer/FloorplanTransformation).
- Dodge, Xu and Stenger. [Parsing Floor Plan Images](https://web.archive.org/web/20201111170058/http://www.mva-org.jp/Proceedings/2017USB/papers/10-01.pdf). MVA 2017.
- Kalervo et al. [CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis](https://arxiv.org/abs/1904.01920). SCIA 2019. [Code](https://github.com/CubiCasa/CubiCasa5k).
- Zeng, Li, Yu and Fu. [Deep Floor Plan Recognition Using a Multi-Task Network with Room-Boundary-Guided Attention](https://arxiv.org/abs/1908.11025). ICCV 2019. [Code](https://github.com/zlzeng/DeepFloorplan).
- He, Gkioxari, Dollár and Girshick. [Mask R-CNN](https://arxiv.org/abs/1703.06870). ICCV 2017. [Matterport's implementation](https://github.com/matterport/Mask_RCNN).

Furniture:

- Merrell, Schkufza, Li, Agrawala and Koltun. [Interactive Furniture Layout Using Interior Design Guidelines](https://paulmerrell.org//furnitureLayout2.pdf). SIGGRAPH 2011.
- Yu et al. [Make it Home: Automatic Optimization of Furniture Arrangement](https://ww3.math.ucla.edu/camreport/cam11-49.pdf). SIGGRAPH 2011.
- Wang, Savva, Chang and Ritchie. [Deep Convolutional Priors for Indoor Scene Synthesis](https://kwang-ether.github.io/pdf/deepsynth.pdf). SIGGRAPH 2018.
- Ritchie, Wang and Lin. [Fast and Flexible Indoor Scene Synthesis via Deep Convolutional Generative Models](https://arxiv.org/abs/1811.12463). CVPR 2019.

Standards:

- [GOST 475-2016](https://meganorm.ru/Data2/1/4293749/4293749722.htm), wooden doors.
- [SP 54.13330.2016](https://web.archive.org/web/20191226232843/http://docs.cntd.ru/document/456054198), apartment buildings.

{% endraw %}
