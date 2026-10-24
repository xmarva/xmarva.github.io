---
layout: page
title: AI Fashion App
description: "Virtual try-on for digital and real clothes, built at a startup: pose estimation, garment warping, inpainting with GANs and Stable Diffusion, 3D bodies, cloth simulation and the system around them"
img: assets/img/projects/4-ai-fashion-app/figures/card.png
cover: /assets/img/projects/4-ai-fashion-app/figures/cover.png
og_image: https://xmarva.github.io/assets/img/projects/4-ai-fashion-app/figures/cover.png
stylesheets: [/assets/css/tfs-course.css]
importance: 1
category: work
related_publications: false
period: 2021–2023

role: Lead ML Engineer

domains:
  - Computer Vision
  - GenAI
  - 3D

tech:
  - Stable Diffusion
  - Human Pose Estimation
  - 3D Shape Reconstruction
---

{% raw %}

In May 2019 the Amsterdam studio The Fabricant sold a dress called Iridescence for &#36;9,500 at an auction in New York<sup class="cite">[<a href="#ref-1" id="cite-1" data-ref="1" aria-label="Reference 1: Engadget, 2019. engadget.com.">1</a>]</sup>. The dress existed only as a 3D file.

The buyer, Richard Ma, bought it for his wife, Mary Ren, and she received a photograph of herself wearing it: the designers "digitally superimposed their creation onto an image of Ma's wife"<sup class="cite">[<a href="#ref-2" id="cite-2" data-ref="2" aria-label="Reference 2: Smithsonian, 2019. smithsonianmag.com.">2</a>]</sup>.

That photo was made by hand. From January 2021 to June 2023 I led machine learning research and development for a digital fashion product at a startup that is still in stealth. Our task was to make such photos automatically: a person uploads a photo, picks a garment and gets the photo back with the garment on.

This page describes the approaches we tried, the problems we ran into and the system around the models. The NDA covers the product, the data and the numbers, so the mechanics are shown on figures I made again from open models and data. The photos and the video are from the project.

## The product

The product had three uses, shown on a slide from the project:

<img class="img-fluid rounded d-block mx-auto my-4" src="/assets/img/projects/4-ai-fashion-app/slide.jpg" width="1800" height="1011" alt="A slide titled 'Digital clothing can be both futuristic and indistinguishable from the tangible one'. In the middle a model in a white two-piece, labelled Tangible. An arrow labelled Digital Outfits leads left to the same model in an iridescent violet jacket and cut-out flared trousers, labelled Digital. An arrow labelled Virtual Fitting leads right to her in a red leather jacket and black shorts, labelled Digital/Tangible, and an arrow labelled Hyper Personalized Ads leads back" loading="lazy">

<p class="tfs-caption">A slide from the project. Middle: the photo as taken. Left: an outfit that exists only as a 3D model. Right: real clothes, fitted virtually.</p>

The first use was digital outfits: clothes that exist only as 3D models and are worn only in photos and videos. DressX, launched in August 2020, sold exactly that, "virtual looks that can be exclusively worn on photos and videos" ([DressX](https://dressx.com/pages/about)).

The second was virtual fitting, where real clothes from a shop are shown on the buyer before the order. The third turned the second around: a shop's ad shows its garment on the customer's own photo instead of on a model.

In a shop's app the first case looked like this: the user opens a digital outfit, taps Try now, picks a photo and gets it back with the outfit on.

<video class="d-block mx-auto my-4 rounded" width="300" height="533" style="max-width: 100%; height: auto" autoplay muted loop playsinline preload="metadata" poster="/assets/img/projects/4-ai-fashion-app/try_on_app_poster.jpg" aria-label="A phone screen recording: browsing a shop's digital fashion section, opening an iridescent violet outfit, tapping Try now, choosing a photo, a progress indicator, and the same person wearing the outfit, with Add to cart below">
  <source src="/assets/img/projects/4-ai-fashion-app/try_on_app.webm" type="video/webm">
  <source src="/assets/img/projects/4-ai-fashion-app/try_on_app.mp4" type="video/mp4">
</video>

The two main cases set different requirements. For a digital outfit the generated photo is the product, so what matters is image quality and the designer's intent.

For virtual fitting the photo has to show how a real garment will fit, including when it is too tight or too long. Shops care about this because of returns: in 2022 size and fit were the most common reason for them, 45% of all returns<sup class="cite">[<a href="#ref-3" id="cite-3" data-ref="3" aria-label="Reference 3: Narvar, 2022. corp.narvar.com.">3</a>]</sup>.

## Two approaches

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/routes-light.png" width="760" alt="Two ways to put a garment on a photo. The image route reads the body from a person&#x27;s photo, warps the product photo onto it and repaints the result with a generator, at first a GAN and later a diffusion model: it works from any product photo but guesses what it cannot see and knows nothing of sizes. The 3D route fits a body model, SMPL-X, to the photo or to measurements, drapes a 3D garment made from its sewing pattern on that body with a cloth simulation and renders it into the photo: it knows sizes and fit but needs a 3D file for every garment" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/routes-dark.png" width="760" alt="Two ways to put a garment on a photo. The image route reads the body from a person&#x27;s photo, warps the product photo onto it and repaints the result with a generator, at first a GAN and later a diffusion model: it works from any product photo but guesses what it cannot see and knows nothing of sizes. The 3D route fits a body model, SMPL-X, to the photo or to measurements, drapes a 3D garment made from its sewing pattern on that body with a cloth simulation and renders it into the photo: it knows sizes and fit but needs a 3D file for every garment" loading="lazy">

We worked on two approaches in parallel.

**Image-based try-on** treats the task as editing a photo. The system finds the body in the photo, bends the product photo onto it and lets a generative model repaint the region so that the result looks like one photograph.

It needs only the person's photo and a product photo. It knows nothing about sizes: a size S dress will look fine on any body.

**3D try-on** treats the task as dressing a body. The system reconstructs the person's body in 3D, puts a 3D model of the garment on it with a cloth simulation and renders the result into the photo.

It knows sizes and how the fabric behaves, but it needs a 3D model of every garment, and the render has to be made to look like a photo.

The 3D approach had a second purpose besides fit: digital clothing. Digital outfits are designed as 3D models and often could not exist physically: iridescent and transparent materials, shapes that nothing holds up, surfaces that glow.

A generator trained on photos of real clothes has seen little of this. A 3D garment can be draped and rendered with exactly the materials its designer chose.

Both approaches start from the same step: reading the person in the photo.

## Reading the person in a photo

### Keypoints, parsing and DensePose

Image-based try-on in 2021 ran three separate networks on each photo. The first finds the skeleton: OpenPose gives 25 body points in its BODY_25 model, feet included<sup class="cite">[<a href="#ref-4" id="cite-4" data-ref="4" aria-label="Reference 4: Cao et al. OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields. TPAMI 2019.">4</a>]</sup> ([code](https://github.com/CMU-Perceptual-Computing-Lab/openpose)), and BlazePose, made to run on phones, gives 33<sup class="cite">[<a href="#ref-5" id="cite-5" data-ref="5" aria-label="Reference 5: Bazarevsky et al. BlazePose: On-device Real-time Body Pose tracking. CVPR Workshops 2020.">5</a>]</sup>.

The second, a human parser, labels every pixel as hair, face, upper clothes, dress, left arm and so on. SCHP, trained on the LIP dataset with its 20 labels, was a common choice<sup class="cite">[<a href="#ref-6" id="cite-6" data-ref="6" aria-label="Reference 6: Li et al. Self-Correction for Human Parsing. TPAMI 2020.">6</a>]</sup> ([code](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)).

The third, DensePose, goes further. It assigns each pixel of a person to one of 24 patches of the body's surface and gives its position on that patch as two coordinates, U and V<sup class="cite">[<a href="#ref-7" id="cite-7" data-ref="7" aria-label="Reference 7: Güler, Neverova and Kokkinos. DensePose: Dense Human Pose Estimation In The Wild. CVPR 2018.">7</a>]</sup> ([code](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose)).

<img class="img-fluid rounded d-block mx-auto my-4" src="/assets/img/projects/4-ai-fashion-app/densepose.png" width="800" height="512" alt="Left: a young woman in a pink tracksuit and white sneakers crouching by a window. Right: the same silhouette on black, split into coloured patches: the head yellow, the torso and arms in blues and greens, the legs and feet in teal and blue" loading="lazy">

<p class="tfs-caption">A photo from the project and its DensePose body-part map. Each colour is one patch of the body's surface.</p>

The three networks answer different questions. Keypoints give the pose but not the outline of the body. Parsing gives the outline and the old clothes but not which part of the body a pixel shows.

DensePose gives each pixel a position on the body's surface, which is what clothes are attached to: a sleeve follows the surface of the arm. In a pose like this crouch, the keypoints alone do not say which surfaces face the camera.

Parsing also finds the old clothes, which have to be removed from the input. If the person wears jeans and the new garment is a dress, the outline of the jeans must not reach the generator, or the dress comes out in the shape of the jeans.

Try-on papers build a "clothing-agnostic" representation of the person, which keeps the face, hair and pose and masks the clothes<sup class="cite">[<a href="#ref-8" id="cite-8" data-ref="8" aria-label="Reference 8: Han et al. VITON: An Image-based Virtual Try-on Network. CVPR 2018.">8</a>]</sup>. VITON-HD masks the arms too, because they "hint at the sleeve length"<sup class="cite">[<a href="#ref-9" id="cite-9" data-ref="9" aria-label="Reference 9: Choi et al. VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization. CVPR 2021. Code.">9</a>]</sup>.

### Pose and shape in 3D

For 3D try-on the 2D keypoints are only the input. The body has to be reconstructed as a 3D model in the same pose and with the person's proportions.

The standard body model for this was SMPL: a mesh of 6,890 vertices controlled by a few numbers. Shape is usually 10 coefficients $$\beta$$. Pose is 72 numbers $$\theta$$: a rotation of three numbers for each of 23 joints and one for the body as a whole<sup class="cite">[<a href="#ref-10" id="cite-10" data-ref="10" aria-label="Reference 10: Loper et al. SMPL: A Skinned Multi-Person Linear Model. SIGGRAPH Asia 2015.">10</a>]</sup>:

$$
M(\beta, \theta) = W\big(\bar{T} + B_S(\beta) + B_P(\theta),\; J(\beta),\; \theta,\; \mathcal{W}\big).
$$

The formula reads from the inside out. $$\bar{T}$$ is an average body standing with its arms out. $$B_S(\beta)$$ moves its vertices according to the shape, making the body taller, wider or heavier, and $$B_P(\theta)$$ adds the corrections a pose needs, such as the bulge of a bent elbow.

$$J(\beta)$$ places the joints inside this particular body. The skinning function $$W$$ then rotates each vertex with the bones near it, in proportions given by the weights $$\mathcal{W}$$.

SMPL-X (2019) adds hands and a face: 10,475 vertices and 54 joints<sup class="cite">[<a href="#ref-11" id="cite-11" data-ref="11" aria-label="Reference 11: Pavlakos et al. Expressive Body Capture: 3D Hands, Face, and Body from a Single Image. CVPR 2019.">11</a>]</sup> ([code](https://github.com/vchoutas/smplx)).

There were two ways to get these numbers from a photo. SMPLify fits the model: it detects 2D joints and searches for the body whose projected joints land on them, "minimizing an objective function that penalizes the error between the projected 3D model joints and detected 2D joints"<sup class="cite">[<a href="#ref-12" id="cite-12" data-ref="12" aria-label="Reference 12: Bogo et al. Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image. ECCV 2016.">12</a>]</sup>. It is accurate but slow, and where it ends up depends on where it starts.

HMR and ExPose regress instead. A network looks at the pixels and outputs the parameters in one pass, for SMPL<sup class="cite">[<a href="#ref-13" id="cite-13" data-ref="13" aria-label="Reference 13: Kanazawa et al. End-to-end Recovery of Human Shape and Pose. CVPR 2018.">13</a>]</sup> and for SMPL-X with hands and face<sup class="cite">[<a href="#ref-14" id="cite-14" data-ref="14" aria-label="Reference 14: Choutas et al. Monocular Expressive Body Regression through Body-Driven Attention. ECCV 2020.">14</a>]</sup>. SPIN made the two work together: the network's guess is where the fitting starts, and the fitted result becomes training data for the network<sup class="cite">[<a href="#ref-15" id="cite-15" data-ref="15" aria-label="Reference 15: Kolotouros et al. Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop. ICCV 2019.">15</a>]</sup>.

The hard part is depth. A photo shows each bone as a 2D segment. Even when its 3D length is known, the bone can point towards the camera or away from it and project onto the same two points<sup class="cite">[<a href="#ref-16" id="cite-16" data-ref="16" aria-label="Reference 16: Taylor. Reconstruction of Articulated Objects from Point Correspondences in a Single Uncalibrated Image. Computer Vision and Image Understanding, 2000.">16</a>]</sup>.

With the eight long bones of the arms and legs, that gives $$2^8 = 256$$ poses for one 2D skeleton. Models choose among them using what they have learned about real poses: which way knees bend, how people stand and sit.

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/pose-light.png" width="760" alt="Three panels. The standing person of the try-on example with BlazePose&#x27;s 33 keypoints joined into a skeleton; a woman in a pink tracksuit crouching, with her skeleton and the right elbow and the left wrist marked as likely hidden; and the crouch&#x27;s 3D keypoints seen from the side, with dashed alternatives in which the left forearm, the left thigh or the right forearm is turned the other way along the camera&#x27;s line of sight: each of them projects onto the same photo" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/pose-dark.png" width="760" alt="Three panels. The standing person of the try-on example with BlazePose&#x27;s 33 keypoints joined into a skeleton; a woman in a pink tracksuit crouching, with her skeleton and the right elbow and the left wrist marked as likely hidden; and the crouch&#x27;s 3D keypoints seen from the side, with dashed alternatives in which the left forearm, the left thigh or the right forearm is turned the other way along the camera&#x27;s line of sight: each of them projects onto the same photo" loading="lazy">

BlazePose also estimates the 3D positions of its points. On the standing photo depth barely matters, because the limbs lie almost in the plane of the photo.

In the crouch the thighs and forearms point towards the camera. The third panel shows the model's 3D estimate together with three alternatives that project onto the same keypoints.

### Photos that do not work

Many failures in practice came from the photos rather than from the models. The datasets of the time were catalogue photos: VITON-HD collected "frontal-view woman and top clothing image pairs", and its authors wrote that existing methods, their own included, "do not provide robust performance for in-the-wild images". Users send other photos.

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/crops-light.png" width="760" alt="Four crops of the standing example with its keypoints, those outside the frame drawn hollow: full length, 33 of 33 keypoints in frame; cut at the thighs, 25 of 33 keypoints in frame; waist up, 15 of 33 keypoints in frame; a selfie, 13 of 33 keypoints in frame" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/crops-dark.png" width="760" alt="Four crops of the standing example with its keypoints, those outside the frame drawn hollow: full length, 33 of 33 keypoints in frame; cut at the thighs, 25 of 33 keypoints in frame; waist up, 15 of 33 keypoints in frame; a selfie, 13 of 33 keypoints in frame" loading="lazy">

The most common problem was the frame. A dress or trousers need the legs, and when the knees are cut off, the model has to invent both the legs and the place where the hem ends. In the figure a crop at the thighs leaves 25 of the 33 keypoints, and a crop at the waist leaves 15.

Selfies were worse. A photo at arm's length shows the head and shoulders, 13 keypoints out of 33, through a wide-angle lens that enlarges whatever is closest to it. In a mirror selfie the phone and the arm holding it cover the torso, exactly where the new garment has to go.

Then there is occlusion: hands in pockets, crossed arms, bags, long hair over the shoulders, a friend in the frame. The garment has to go behind all of it, and the parsing and pose models make more mistakes around it.

HR-VITON's authors write that results are "significantly impaired when the body parts occlude the garment"<sup class="cite">[<a href="#ref-17" id="cite-17" data-ref="17" aria-label="Reference 17: Lee et al. High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions. ECCV 2022. Code.">17</a>]</sup>. In the crouch above, the pose model itself marks the right elbow and the left wrist as likely hidden.

Sitting, crouching, and side and back views were rare in the training data. A loose sweater or a coat hides the body shape that 3D try-on needs. All of this is why the photo was checked on the phone before upload, as the section on system design describes.

## Image-based try-on

### Garment warping

The product photo shows the garment flat, from the front, without a person. Before generation it has to be moved and bent to where the body is.

The classic tool for this is the thin-plate spline. Matching points are marked on the product and on the person: shoulders, waist, hem. The spline is the smoothest map of the plane that carries each point exactly onto its partner:

$$
f(\mathbf{x}) = \mathbf{a} + A\mathbf{x} + \sum_{i=1}^{n} \mathbf{w}_i \, U\big(\lVert \mathbf{x} - \mathbf{p}_i \rVert\big), \qquad U(r) = r^2 \log r .
$$

The first two terms are an affine map: shift, rotation, scale and shear. The sum adds a smooth bump around each control point $$\mathbf{p}_i$$.

The weights $$\mathbf{w}_i$$ come from a linear system. It sends each $$\mathbf{p}_i$$ to its partner and keeps the bending energy, the integral of the squared second derivatives of $$f$$, as small as possible. The name comes from a thin metal plate pinned at the points, which bends into the same shape<sup class="cite">[<a href="#ref-18" id="cite-18" data-ref="18" aria-label="Reference 18: Bookstein. Principal Warps: Thin-Plate Splines and the Decomposition of Deformations. TPAMI 1989.">18</a>]</sup>.

CP-VTON made the spline learnable. A network looks at the person and the product and predicts the offsets of a 5×5 grid of control points, 2×5×5 = 50 numbers in total<sup class="cite">[<a href="#ref-19" id="cite-19" data-ref="19" aria-label="Reference 19: Wang et al. Toward Characteristic-Preserving Image-based Virtual Try-On Network. ECCV 2018. Code.">19</a>]</sup>. The figure does the same by hand, with 14 points on the project's example.

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/warp-light.png" width="760" alt="Four panels. The product photo of a floral dress with 14 points marked on its sleeves, neckline, waist and hem; the person in a white dress with the same points marked on her body; the floral dress bent onto her by a thin-plate spline, with the old white sleeves and skirt still showing around it and the fabric as flat as on the product photo; and the project&#x27;s trained model, which replaces the old dress, shapes the puffed sleeves to the arms and matches the light of the scene" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/warp-dark.png" width="760" alt="Four panels. The product photo of a floral dress with 14 points marked on its sleeves, neckline, waist and hem; the person in a white dress with the same points marked on her body; the floral dress bent onto her by a thin-plate spline, with the old white sleeves and skirt still showing around it and the fabric as flat as on the product photo; and the project&#x27;s trained model, which replaces the old dress, shapes the puffed sleeves to the arms and matches the light of the scene" loading="lazy">

The third panel shows what a warp alone cannot do. The fabric is bent in the plane of the picture, not in space, so there is no shading where it turns away from the camera. The old white dress still shows around the new one.

The puffed sleeves are squeezed flat instead of gathered. Anything in front of the garment, a hand or a strand of hair, would simply be painted over.

The papers of the time name the same limits. TPS has "limited degrees of freedom" for "highly non-rigid deformation"<sup class="cite">[<a href="#ref-20" id="cite-20" data-ref="20" aria-label="Reference 20: Ge et al. Parser-Free Virtual Try-on via Distilling Appearance Flows. CVPR 2021. Code.">20</a>]</sup>, and where the body hides part of the garment, warping causes "pixel-squeezing artifacts"<sup class="cite">[<a href="#ref-17" id="cite-21" data-ref="17" aria-label="Reference 17: Lee et al. High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions. ECCV 2022. Code.">17</a>]</sup>.

The fourth panel shows what a trained model makes of the same inputs. It uses the warped garment as a hint and repaints the region: the old dress is removed, the sleeves follow the arms, and the light matches the scene.

### Training data

In 2021 the repainting model was a GAN. VITON-HD, the first to work at high resolution, generated 1024×768 images and was trained on 11,647 pairs of a product photo and a person wearing that product<sup class="cite">[<a href="#ref-9" id="cite-22" data-ref="9" aria-label="Reference 9: Choi et al. VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization. CVPR 2021. Code.">9</a>]</sup>.

Such datasets have no photos of the same person in the same pose in two different garments. So models are trained on reconstruction: the garment in a photo is masked, and the model gets the product photo of the same garment and has to paint it back. In use, the product photo shows a different garment.

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/pairs-light.png" width="760" alt="How an image try-on model is trained and how it is used. In training, a photo of a person in shirt A has the shirt hidden, keeping the face, hair and pose; the generator gets the shop photo of the same shirt A, paints it back, and the result is compared with the original photo. In use, the same masked photo gets the shop photo of a different shirt B, and there is no photo to compare with. If the mask lets the outline of shirt A through, shirt B comes out in the shape of A" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/pairs-dark.png" width="760" alt="How an image try-on model is trained and how it is used. In training, a photo of a person in shirt A has the shirt hidden, keeping the face, hair and pose; the generator gets the shop photo of the same shirt A, paints it back, and the result is compared with the original photo. In use, the same masked photo gets the shop photo of a different shirt B, and there is no photo to compare with. If the mask lets the outline of shirt A through, shirt B comes out in the shape of A" loading="lazy">

This difference between training and use explains a common failure. If the mask follows the outline of the old garment, the model learns to rely on that outline, and in use the new garment comes out in the shape of the old one. This is what the clothing-agnostic masks prevent.

### Inpainting with diffusion

Stable Diffusion went public on 22 August 2022 with open weights, trained "on 512x512 images from a subset of the LAION-5B database"<sup class="cite">[<a href="#ref-21" id="cite-23" data-ref="21" aria-label="Reference 21: Stability AI, 2022. stability.ai.">21</a>]</sup> ([code](https://github.com/CompVis/stable-diffusion)).

An inpainting version followed in October. It takes a photo, a mask and a text prompt, and its UNet "has 5 additional input channels (4 for the encoded masked-image and 1 for the mask itself)" ([model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)).

A model trained on that much data already knows how fabrics, skin and light look, which a GAN trained on twelve thousand pairs did not. Turning it into try-on still took several decisions.

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/masks-light.png" width="760" alt="Four panels. The person with her white dress masked tightly, so the mask keeps the dress&#x27;s outline; a larger clothing-agnostic mask that also covers both arms; the photo with that region grey and the skeleton on top, which is what the generator keeps; and a close-up of the generated floral dress under a grid of 8 by 8 pixel cells, in which a flower of the print spans one or two cells" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/masks-dark.png" width="760" alt="Four panels. The person with her white dress masked tightly, so the mask keeps the dress&#x27;s outline; a larger clothing-agnostic mask that also covers both arms; the photo with that region grey and the skeleton on top, which is what the generator keeps; and a close-up of the generated floral dress under a grid of 8 by 8 pixel cells, in which a flower of the print spans one or two cells" loading="lazy">

The first decision is the mask. It has to cover where the new garment will be, not where the old one was. A dress replacing shorts needs the legs masked, and a T-shirt replacing a long-sleeved top needs the arms masked, because the model has to paint skin there.

A tight mask would also hand the old outline to the model, so the mask is a generous clothing-agnostic region, as in the second panel.

What stays outside the mask, the face, hair, hands and background, should stay untouched. The autoencoder slightly changes even the pixels it was not asked to change, so after generation the original pixels are pasted back outside the mask.

The garment itself is the hardest thing to get in. A prompt like "a floral dress with puffed sleeves" gives a floral dress, not the one in the shop, so the garment has to enter as an image.

It can be warped onto the body as a starting point, passed through an image encoder as in Paint by Example<sup class="cite">[<a href="#ref-22" id="cite-24" data-ref="22" aria-label="Reference 22: Yang et al. Paint by Example: Exemplar-based Image Editing with Diffusion Models. CVPR 2023.">22</a>]</sup> ([code](https://github.com/Fantasy-Studio/Paint-by-Example)), or taught to the model by fine-tuning on the product's photos, as in DreamBooth<sup class="cite">[<a href="#ref-23" id="cite-25" data-ref="23" aria-label="Reference 23: Ruiz et al. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. CVPR 2023.">23</a>]</sup> and Textual Inversion<sup class="cite">[<a href="#ref-24" id="cite-26" data-ref="24" aria-label="Reference 24: Gal et al. An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion. ICLR 2023.">24</a>]</sup>.

With a large mask the model may also move the arms. ControlNet, released in February 2023, holds the generator to a skeleton, edges or a segmentation; its OpenPose model "was trained on 200k pose-image, caption pairs"<sup class="cite">[<a href="#ref-25" id="cite-27" data-ref="25" aria-label="Reference 25: Zhang and Agrawala. Adding Conditional Control to Text-to-Image Diffusion Models. ICCV 2023.">25</a>]</sup> ([code](https://github.com/lllyasviel/ControlNet); [model](https://huggingface.co/lllyasviel/sd-controlnet-openpose)).

Resolution limits the detail. The autoencoder of Stable Diffusion 1.x turns each 8×8 block of pixels into one latent, so on a full-length photo 512 pixels tall a flower of the print is one or two latents wide, as the last panel shows.

Fine prints, logos and text come out blurred; LaDI-VTON's authors write that "high-frequency details may not be accurately represented"<sup class="cite">[<a href="#ref-26" id="cite-28" data-ref="26" aria-label="Reference 26: Morelli et al. LaDI-VTON: Latent Diffusion Textual-Inversion Enhanced Virtual Try-On. ACM Multimedia 2023. Code.">26</a>]</sup>. The usual fix is to generate on a crop around the garment and paste the result back into the photo.

Finally, every random seed gives a different result, while the same photo and garment should give the same picture every time. Either the seed is fixed, or several candidates are generated and the best one is kept, at several times the cost.

Over the project I fine-tuned both kinds of generator, GANs first and then Stable Diffusion. With GANs most of the work went into getting a result that looks like a photo at all.

With diffusion the result looked like a photo from the start, and the work moved to keeping the garment, the face and the body the same as in the inputs.

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/timeline-light.png" width="760" alt="A timeline from late 2017 to mid-2023 in four lanes, with the project, January 2021 to June 2023, shaded. Try-on models: VITON (Nov 2017), CP-VTON (Jul 2018), ACGPN (Mar 2020), VITON-HD (Mar 2021), HR-VITON (Jun 2022), TryOnDiffusion (Jun 2023). Image generators: StyleGAN2 (Dec 2019), DDPM (Jun 2020), DALL·E (Jan 2021), Latent Diffusion (Dec 2021), DALL·E 2 (Apr 2022), Stable Diffusion (Aug 2022), ControlNet (Feb 2023). Bodies in 3D: DensePose (Feb 2018), SMPL-X (Apr 2019), PIFuHD (Apr 2020), SMPL-X in Unity (Jun 2021), ICON (Dec 2021), SHAPY (Jun 2022), ECON (Dec 2022). Digital fashion: a digital dress, 9,500 dollars (May 2019), DressX (Aug 2020), Gucci bag on Roblox (May 2021), Facebook → Meta (Oct 2021), Metaverse Fashion Week (Mar 2022), try-on in Google Search (Jun 2023)." loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/timeline-dark.png" width="760" alt="A timeline from late 2017 to mid-2023 in four lanes, with the project, January 2021 to June 2023, shaded. Try-on models: VITON (Nov 2017), CP-VTON (Jul 2018), ACGPN (Mar 2020), VITON-HD (Mar 2021), HR-VITON (Jun 2022), TryOnDiffusion (Jun 2023). Image generators: StyleGAN2 (Dec 2019), DDPM (Jun 2020), DALL·E (Jan 2021), Latent Diffusion (Dec 2021), DALL·E 2 (Apr 2022), Stable Diffusion (Aug 2022), ControlNet (Feb 2023). Bodies in 3D: DensePose (Feb 2018), SMPL-X (Apr 2019), PIFuHD (Apr 2020), SMPL-X in Unity (Jun 2021), ICON (Dec 2021), SHAPY (Jun 2022), ECON (Dec 2022). Digital fashion: a digital dress, 9,500 dollars (May 2019), DressX (Aug 2020), Gucci bag on Roblox (May 2021), Facebook → Meta (Oct 2021), Metaverse Fashion Week (Mar 2022), try-on in Google Search (Jun 2023)." loading="lazy">

The field changed during the project. OpenAI showed DALL·E on 5 January 2021, the month I started<sup class="cite">[<a href="#ref-27" id="cite-29" data-ref="27" aria-label="Reference 27: OpenAI, 2021. openai.com.">27</a>]</sup>.

On 14 June 2023, the month I left, Google added virtual try-on to Search<sup class="cite">[<a href="#ref-28" id="cite-30" data-ref="28" aria-label="Reference 28: Google, 2023. blog.google.">28</a>]</sup>. It runs on a diffusion model ([how it works](https://blog.google/products/shopping/virtual-try-on-google-generative-ai/)), and its paper, TryOnDiffusion, argued that "the sequence of finding displacements, warping, and blending often creates artifacts"<sup class="cite">[<a href="#ref-29" id="cite-31" data-ref="29" aria-label="Reference 29: Zhu et al. TryOnDiffusion: A Tale of Two UNets. CVPR 2023.">29</a>]</sup>.

That sequence is the pipeline described above. TryOnDiffusion warps the garment "implicitly via a cross attention mechanism" and was trained on 4 million pairs of the same person in the same garment in different poses.

## 3D try-on

### The body in a game engine

<img class="img-fluid rounded d-block mx-auto my-4" src="/assets/img/projects/4-ai-fashion-app/smpl_unity.png" width="760" height="553" alt="The Unity editor with two grey SMPL bodies, a man and a woman, standing in a T-pose on a plane. Below them an animation state machine with the states Idle, WalkForward, WalkBackward, Jump, Crouch and crouch2idle" loading="lazy">

<p class="tfs-caption">SMPL's female and male bodies in Unity, from the project, driven by an ordinary game animation controller.</p>

A fitted body is useful for a product only if it can be rendered and animated. SMPL comes with code for Unity, and an SMPL-X integration for Unity appeared in June 2021 ([SMPL-X](https://smpl-x.is.tue.mpg.de/)). In a game engine the body can walk, turn and be lit, and the scene can run on a phone.

SMPL's licence covers "non-commercial scientific research" and excludes "incorporation in a commercial product"; companies are pointed to commercial licences sold through Meshcapade ([SMPL licence](https://smpl.is.tue.mpg.de/modellicense.html)). The bodies in my figures below are from MakeHuman instead, whose assets are in the public domain under CC0 ([MakeHuman](https://github.com/makehumancommunity/makehuman)).

### Measurements

Fit needs the numbers a size chart uses: bust, waist and hips. A mesh does not come with them, so they have to be measured.

The mesh is cut with a horizontal plane. Each triangle the plane crosses gives a short segment, and the segments join into closed loops: one around the torso and one around each arm. The torso loop is the one to measure.

A tape measure does not follow the hollow between the breasts or the small of the back; it stretches across them. So the girth is the perimeter of the convex hull of the cross-section, the shortest closed curve around it.

Where to cut depends on the measurement. The waist is the narrowest level of the torso, the hips the widest level below it, and the bust the level where the chest comes furthest forward. Each is a small search over heights, here in steps of half a centimetre.

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/bodies-light.png" width="760" alt="Three versions of the same open 3D body, thin, average and full, with tape-measure loops drawn around the bust, waist and hips: thin: bust 81 cm, waist 59 cm, hips 87 cm; average: bust 84 cm, waist 65 cm, hips 95 cm; full: bust 90 cm, waist 71 cm, hips 103 cm" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/bodies-dark.png" width="760" alt="Three versions of the same open 3D body, thin, average and full, with tape-measure loops drawn around the bust, waist and hips: thin: bust 81 cm, waist 59 cm, hips 87 cm; average: bust 84 cm, waist 65 cm, hips 95 cm; full: bust 90 cm, waist 71 cm, hips 103 cm" loading="lazy">

From the thin body to the full one the girths grow by about 9 cm at the bust, 11 cm at the waist and 16 cm at the hips. The next figure shows where on the body this growth happens.

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/shape-light.png" width="760" alt="The average body seen from the front and the side, coloured by how far each point moves between the thin and the full build, up to about 3.5 cm: most around the belly, hips, buttocks and thighs, a little on the arms, almost nothing on the head, hands and feet" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/shape-dark.png" width="760" alt="The average body seen from the front and the side, coloured by how far each point moves between the thin and the full build, up to about 3.5 cm: most around the belly, hips, buttocks and thighs, a little on the arms, almost nothing on the head, hands and feet" loading="lazy">

The belly, hips and thighs move by up to about 3.5 cm; the head, hands and feet hardly move.

These are the parts of the body that clothes cover, which makes body shape hard to estimate from a photo. A person in a loose sweater shows their face, hands and shoes clearly and their waist not at all.

SHAPY (2022), a method built for this task, estimated body shape from a single photo with average errors of 6.5 cm at the chest, 6.9 cm at the waist and 5.7 cm at the hips<sup class="cite">[<a href="#ref-30" id="cite-32" data-ref="30" aria-label="Reference 30: Choutas et al. Accurate 3D Body Shape Regression using Metric and Semantic Attributes. CVPR 2022.">30</a>]</sup> ([code](https://github.com/muelea/shapy)).

That is more than half of the whole difference between my thin and full bodies, so measurements entered by the user are a useful input alongside the photo.

### Cloth and digital garments

A 3D garment is a mesh cut from flat pattern pieces and sewn together in software, in programs such as [CLO 3D](https://www.clo3d.com/en/) or [Marvelous Designer](https://www.marvelousdesigner.com/). To put it on a body it is simulated: gravity pulls, the fabric resists stretching and, a little, bending, and it must not pass through the skin.

Position-based dynamics turns each of these rules into a constraint on the positions of the mesh's vertices. Two neighbours $$i$$ and $$j$$ that were $$L$$ apart on the pattern should stay $$L$$ apart:

$$
C(\mathbf{x}_i, \mathbf{x}_j) = \lVert \mathbf{x}_i - \mathbf{x}_j \rVert - L = 0 .
$$

Every time step moves the vertices freely under gravity, then corrects their positions, constraint by constraint, until the rules hold again<sup class="cite">[<a href="#ref-31" id="cite-33" data-ref="31" aria-label="Reference 31: Müller et al. Position Based Dynamics. Journal of Visual Communication and Image Representation, 2007.">31</a>]</sup>.

In plain PBD the cloth gets stiffer the more correction iterations are run. XPBD fixes this by giving each constraint a compliance $$\alpha$$, the inverse of stiffness, and computing each correction as

$$
\Delta\lambda = \frac{-C - \tilde\alpha \lambda}{\nabla C \, M^{-1} \nabla C^{\top} + \tilde\alpha}, \qquad \tilde\alpha = \frac{\alpha}{\Delta t^2} .
$$

Here $$\lambda$$ is how hard the constraint has pushed so far, and $$M$$ holds the vertices' masses. With the compliance in the denominator the cloth behaves the same "regardless of iteration count"<sup class="cite">[<a href="#ref-32" id="cite-34" data-ref="32" aria-label="Reference 32: Macklin, Müller and Chentanez. XPBD: Position-Based Simulation of Compliant Constrained Dynamics. Motion in Games 2016.">32</a>]</sup>.

Silk and wool both resist stretching; they differ in bending. In the simulation below, silk's bending constraints are compliant and wool's are rigid.

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/drape-light.png" width="760" alt="The same A-line skirt simulated on the thin and the full body in two fabrics. Silk, which bends freely, falls in narrow folds close to the legs; wool, which resists bending, keeps a wide A-line flare. On the full body the hips push the skirt outward" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/drape-dark.png" width="760" alt="The same A-line skirt simulated on the thin and the full body in two fabrics. Silk, which bends freely, falls in narrow folds close to the legs; wool, which resists bending, keeps a wide A-line flare. On the full body the hips push the skirt outward" loading="lazy">

Silk falls in narrow folds along the legs, and wool keeps its A-line shape. On the full body the hips push the skirt outward.

For digital clothing the simulation does not have to stop at real fabrics. Compliances, masses and materials are parameters, so a designer's garment can hold a shape no fabric would, or be rendered as metal, glass or light, and it still follows the body's pose and size. The image-based approach has no such parameters.

The practical problems of this approach lay elsewhere. Every garment needs a 3D model, which is natural for a digital outfit but not for a shop that adds thousands of products a season and has only photos of them. Every real fabric needs its stretch and bending measured or guessed.

Simulation is also too slow for an interactive request, so drapes are computed in advance or predicted by a network; TailorNet (2020) predicts them "more than 1000 times faster" than simulating<sup class="cite">[<a href="#ref-33" id="cite-35" data-ref="33" aria-label="Reference 33: Patel, Liao and Pons-Moll. TailorNet: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style. CVPR 2020.">33</a>]</sup> ([code](https://github.com/chaitanya100100/TailorNet)). And a body from a game engine next to a real face looks like a game character.

## Combining the two

The two approaches fail in different places. Image generation gives realistic light, skin and fabric texture, but it guesses the geometry and ignores size. The 3D pipeline gets the geometry and size right, but its output looks rendered.

We experimented with combining them: the 3D part decides where the garment goes and how it falls, and the image model paints it into the photo.

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/hybrid-light.png" width="760" alt="An experimental pipeline that combines the two routes. In the 3D part, a body model, SMPL-X, is fitted to the person&#x27;s photo, the 3D garment is draped on it by a cloth simulation, and the result is rendered as a garment mask, a depth map and a normal map. In the 2D part, the clothes in the photo are masked, keeping the face and hands, and a diffusion model inpaints the region conditioned on those maps, giving the new photo" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/hybrid-dark.png" width="760" alt="An experimental pipeline that combines the two routes. In the 3D part, a body model, SMPL-X, is fitted to the person&#x27;s photo, the 3D garment is draped on it by a cloth simulation, and the result is rendered as a garment mask, a depth map and a normal map. In the 2D part, the clothes in the photo are masked, keeping the face and hands, and a diffusion model inpaints the region conditioned on those maps, giving the new photo" loading="lazy">

The pipeline starts with the SMPL-X fit from the section on pose and shape. The 3D garment is draped on that body by the cloth simulation, and the result is rendered not as a finished picture but as maps: the garment's mask, its depth and its surface normals.

A diffusion model then inpaints the masked region of the photo, conditioned on those maps. ControlNet's first release already included models for depth maps and normal maps ([ControlNet](https://github.com/lllyasviel/ControlNet)).

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/maps-light.png" width="760" alt="The silk skirt on the full body drawn four ways: shaded, as a white garment mask on black, as a normal map in which colour encodes the direction of the surface, and as a depth map in which nearer is lighter" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/maps-dark.png" width="760" alt="The silk skirt on the full body drawn four ways: shaded, as a white garment mask on black, as a normal map in which colour encodes the direction of the surface, and as a depth map in which nearer is lighter" loading="lazy">

Another variant starts from the render itself. SDEdit adds noise to a rough image and then removes it with a diffusion model, balancing "faithfulness to the user input" against "realism of the synthesized image"<sup class="cite">[<a href="#ref-34" id="cite-36" data-ref="34" aria-label="Reference 34: Meng et al. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations. ICLR 2022.">34</a>]</sup>. A rendered garment comes out closer to a photo and keeps its shape.

For digital clothing this combination fits well. The shape and the unusual material come from the designer's 3D model. The integration into the photo comes from the image model: the light, the shadows on the skin, the edges against the background.

The difficulty is consistency. The generator can still move a fold or change a colour that the render had fixed, and two pipelines have to be kept in step.

## System design

The models were one part of the product. Around them were mobile apps, a server and a pool of GPU workers, and the clearest way to describe them is to follow one photo through the system.

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/system-light.png" width="760" alt="The system around the models, in three lanes. On the phone: take or pick a photo, check it with a small pose model, upload it resized and compressed, and later show the result by push notification or polling. On the server: an API with accounts and limits, a job queue with one job per pair of photo and garment, and results in storage behind a CDN. On GPU workers: body analysis once per photo, garment preparation once per product, and generation once per pair, which uses both and writes to the results" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/system-dark.png" width="760" alt="The system around the models, in three lanes. On the phone: take or pick a photo, check it with a small pose model, upload it resized and compressed, and later show the result by push notification or polling. On the server: an API with accounts and limits, a job queue with one job per pair of photo and garment, and results in storage behind a CDN. On GPU workers: body analysis once per photo, garment preparation once per product, and generation once per pair, which uses both and writes to the results" loading="lazy">

It starts on the phone. The app asks for a full-length photo, facing the camera, arms slightly away from the body, and a small pose model checks the result before anything leaves the device: is there one person, and is the whole body in the frame.

BlazePose was built for this kind of job and "runs at over 30 frames per second on a Pixel 2 phone"<sup class="cite">[<a href="#ref-5" id="cite-37" data-ref="5" aria-label="Reference 5: Bazarevsky et al. BlazePose: On-device Real-time Body Pose tracking. CVPR Workshops 2020.">5</a>]</sup>. A selfie is turned back on the phone, with a hint about what to fix, before it costs any GPU time.

A photo that passes is resized, compressed and uploaded. From there the work is split by what it depends on. Whatever depends only on the photo, the keypoints, parsing, DensePose and the 3D body fit, runs once, right after upload, often before the user has picked a garment.

Whatever depends only on the garment, the segmented product photo, the 3D model and its drapes, was computed when the product was added to the catalogue. Only generation needs both, so trying one more garment on a known photo costs one generation and nothing else.

That generation is a job in a queue, one per pair of photo and garment. A burst of requests makes the queue longer instead of making requests fail, and the app does not hold a connection open while it waits: the result arrives by a push notification or by polling, and the user keeps browsing.

Finished images go to storage behind a CDN, so asking for the same pair again costs nothing.

Most of the cost was in the diffusion model, and the ways to cut it changed quickly over 2022 and 2023. DDIM had already made sampling "10× to 50× faster" than the original diffusion models<sup class="cite">[<a href="#ref-35" id="cite-38" data-ref="35" aria-label="Reference 35: Song, Meng and Ermon. Denoising Diffusion Implicit Models. ICLR 2021.">35</a>]</sup>, and DPM-Solver++ "can generate high-quality samples within only 15 to 20 steps"<sup class="cite">[<a href="#ref-36" id="cite-39" data-ref="36" aria-label="Reference 36: Lu et al. DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models. 2022.">36</a>]</sup>.

Half precision halves the memory. FlashAttention computes exact attention with fewer reads and writes to GPU memory<sup class="cite">[<a href="#ref-37" id="cite-40" data-ref="37" aria-label="Reference 37: Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022.">37</a>]</sup>, and runtimes such as [ONNX Runtime](https://onnxruntime.ai/) and [TensorRT](https://developer.nvidia.com/tensorrt) compile the networks for inference.

A crop around the garment at the model's native resolution is cheaper to generate than the whole photo, and several jobs batched on one GPU use it better than one at a time.

## References

1. {:#ref-1} Engadget, 2019. [engadget.com](https://www.engadget.com/2019-05-27-fabricant-blockchain-digital-dress.html). <a class="ref-back" href="#cite-1" aria-label="Back to the text">&#8617;</a>
2. {:#ref-2} Smithsonian, 2019. [smithsonianmag.com](https://www.smithsonianmag.com/smart-news/designers-are-selling-expensive-clothes-dont-existand-people-are-actually-buying-them-180973568/). <a class="ref-back" href="#cite-2" aria-label="Back to the text">&#8617;</a>
3. {:#ref-3} Narvar, 2022. [corp.narvar.com](https://corp.narvar.com/resources/2022-the-state-of-returns). <a class="ref-back" href="#cite-3" aria-label="Back to the text">&#8617;</a>
4. {:#ref-4} Cao et al. [OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1812.08008). TPAMI 2019. <a class="ref-back" href="#cite-4" aria-label="Back to the text">&#8617;</a>
5. {:#ref-5} Bazarevsky et al. [BlazePose: On-device Real-time Body Pose tracking](https://arxiv.org/abs/2006.10204). CVPR Workshops 2020. <a class="ref-back" href="#cite-5" aria-label="Back to the text">&#8617;</a>
6. {:#ref-6} Li et al. [Self-Correction for Human Parsing](https://arxiv.org/abs/1910.09777). TPAMI 2020. <a class="ref-back" href="#cite-6" aria-label="Back to the text">&#8617;</a>
7. {:#ref-7} Güler, Neverova and Kokkinos. [DensePose: Dense Human Pose Estimation In The Wild](https://arxiv.org/abs/1802.00434). CVPR 2018. <a class="ref-back" href="#cite-7" aria-label="Back to the text">&#8617;</a>
8. {:#ref-8} Han et al. [VITON: An Image-based Virtual Try-on Network](https://arxiv.org/abs/1711.08447). CVPR 2018. <a class="ref-back" href="#cite-8" aria-label="Back to the text">&#8617;</a>
9. {:#ref-9} Choi et al. [VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization](https://arxiv.org/abs/2103.16874). CVPR 2021. [Code](https://github.com/shadow2496/VITON-HD). <a class="ref-back" href="#cite-9" aria-label="Back to the text">&#8617;</a>
10. {:#ref-10} Loper et al. [SMPL: A Skinned Multi-Person Linear Model](https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf). SIGGRAPH Asia 2015. <a class="ref-back" href="#cite-10" aria-label="Back to the text">&#8617;</a>
11. {:#ref-11} Pavlakos et al. [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://arxiv.org/abs/1904.05866). CVPR 2019. <a class="ref-back" href="#cite-11" aria-label="Back to the text">&#8617;</a>
12. {:#ref-12} Bogo et al. [Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image](https://arxiv.org/abs/1607.08128). ECCV 2016. <a class="ref-back" href="#cite-12" aria-label="Back to the text">&#8617;</a>
13. {:#ref-13} Kanazawa et al. [End-to-end Recovery of Human Shape and Pose](https://arxiv.org/abs/1712.06584). CVPR 2018. <a class="ref-back" href="#cite-13" aria-label="Back to the text">&#8617;</a>
14. {:#ref-14} Choutas et al. [Monocular Expressive Body Regression through Body-Driven Attention](https://arxiv.org/abs/2008.09062). ECCV 2020. <a class="ref-back" href="#cite-14" aria-label="Back to the text">&#8617;</a>
15. {:#ref-15} Kolotouros et al. [Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop](https://arxiv.org/abs/1909.12828). ICCV 2019. <a class="ref-back" href="#cite-15" aria-label="Back to the text">&#8617;</a>
16. {:#ref-16} Taylor. [Reconstruction of Articulated Objects from Point Correspondences in a Single Uncalibrated Image](https://doi.org/10.1006/cviu.2000.0878). Computer Vision and Image Understanding, 2000. <a class="ref-back" href="#cite-16" aria-label="Back to the text">&#8617;</a>
17. {:#ref-17} Lee et al. [High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions](https://arxiv.org/abs/2206.14180). ECCV 2022. [Code](https://github.com/sangyun884/HR-VITON). <a class="ref-back" href="#cite-17" aria-label="Back to the text">&#8617;</a>
18. {:#ref-18} Bookstein. [Principal Warps: Thin-Plate Splines and the Decomposition of Deformations](https://doi.org/10.1109/34.24792). TPAMI 1989. <a class="ref-back" href="#cite-18" aria-label="Back to the text">&#8617;</a>
19. {:#ref-19} Wang et al. [Toward Characteristic-Preserving Image-based Virtual Try-On Network](https://arxiv.org/abs/1807.07688). ECCV 2018. [Code](https://github.com/sergeywong/cp-vton). <a class="ref-back" href="#cite-19" aria-label="Back to the text">&#8617;</a>
20. {:#ref-20} Ge et al. [Parser-Free Virtual Try-on via Distilling Appearance Flows](https://arxiv.org/abs/2103.04559). CVPR 2021. [Code](https://github.com/geyuying/PF-AFN). <a class="ref-back" href="#cite-20" aria-label="Back to the text">&#8617;</a>
21. {:#ref-21} Stability AI, 2022. [stability.ai](https://stability.ai/news/stable-diffusion-public-release). <a class="ref-back" href="#cite-23" aria-label="Back to the text">&#8617;</a>
22. {:#ref-22} Yang et al. [Paint by Example: Exemplar-based Image Editing with Diffusion Models](https://arxiv.org/abs/2211.13227). CVPR 2023. <a class="ref-back" href="#cite-24" aria-label="Back to the text">&#8617;</a>
23. {:#ref-23} Ruiz et al. [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242). CVPR 2023. <a class="ref-back" href="#cite-25" aria-label="Back to the text">&#8617;</a>
24. {:#ref-24} Gal et al. [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618). ICLR 2023. <a class="ref-back" href="#cite-26" aria-label="Back to the text">&#8617;</a>
25. {:#ref-25} Zhang and Agrawala. [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543). ICCV 2023. <a class="ref-back" href="#cite-27" aria-label="Back to the text">&#8617;</a>
26. {:#ref-26} Morelli et al. [LaDI-VTON: Latent Diffusion Textual-Inversion Enhanced Virtual Try-On](https://arxiv.org/abs/2305.13501). ACM Multimedia 2023. [Code](https://github.com/miccunifi/ladi-vton). <a class="ref-back" href="#cite-28" aria-label="Back to the text">&#8617;</a>
27. {:#ref-27} OpenAI, 2021. [openai.com](https://openai.com/index/dall-e/). <a class="ref-back" href="#cite-29" aria-label="Back to the text">&#8617;</a>
28. {:#ref-28} Google, 2023. [blog.google](https://blog.google/products/shopping/ai-virtual-try-on-google-shopping/). <a class="ref-back" href="#cite-30" aria-label="Back to the text">&#8617;</a>
29. {:#ref-29} Zhu et al. [TryOnDiffusion: A Tale of Two UNets](https://arxiv.org/abs/2306.08276). CVPR 2023. <a class="ref-back" href="#cite-31" aria-label="Back to the text">&#8617;</a>
30. {:#ref-30} Choutas et al. [Accurate 3D Body Shape Regression using Metric and Semantic Attributes](https://arxiv.org/abs/2206.07036). CVPR 2022. <a class="ref-back" href="#cite-32" aria-label="Back to the text">&#8617;</a>
31. {:#ref-31} Müller et al. [Position Based Dynamics](https://doi.org/10.1016/j.jvcir.2007.01.005). Journal of Visual Communication and Image Representation, 2007. <a class="ref-back" href="#cite-33" aria-label="Back to the text">&#8617;</a>
32. {:#ref-32} Macklin, Müller and Chentanez. [XPBD: Position-Based Simulation of Compliant Constrained Dynamics](https://matthias-research.github.io/pages/publications/XPBD.pdf). Motion in Games 2016. <a class="ref-back" href="#cite-34" aria-label="Back to the text">&#8617;</a>
33. {:#ref-33} Patel, Liao and Pons-Moll. [TailorNet: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style](https://arxiv.org/abs/2003.04583). CVPR 2020. <a class="ref-back" href="#cite-35" aria-label="Back to the text">&#8617;</a>
34. {:#ref-34} Meng et al. [SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073). ICLR 2022. <a class="ref-back" href="#cite-36" aria-label="Back to the text">&#8617;</a>
35. {:#ref-35} Song, Meng and Ermon. [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502). ICLR 2021. <a class="ref-back" href="#cite-38" aria-label="Back to the text">&#8617;</a>
36. {:#ref-36} Lu et al. [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095). 2022. <a class="ref-back" href="#cite-39" aria-label="Back to the text">&#8617;</a>
37. {:#ref-37} Dao et al. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). NeurIPS 2022. <a class="ref-back" href="#cite-40" aria-label="Back to the text">&#8617;</a>
38. {:#ref-38} Rombach et al. [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752). CVPR 2022.
{: .tfs-refs}

{% endraw %}
