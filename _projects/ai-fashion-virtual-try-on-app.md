---
layout: page
title: AI Fashion App
description: "Putting clothes that were never sewn on photos of real people: two and a half years of virtual try-on at a startup, with image generators and with 3D bodies, 2021 to 2023"
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

In May 2019 a dress that had never been sewn sold for &#36;9,500. It was called Iridescence, the Amsterdam studio The Fabricant made it, and it existed only as a 3D file. It went at an auction during a blockchain conference in New York ([Engadget, 2019](https://www.engadget.com/2019-05-27-fabricant-blockchain-digital-dress.html)).

The buyer, Richard Ma, bought it for his wife, Mary Ren. What she received was a photograph: the designers "digitally superimposed their creation onto an image of Ma's wife" ([Smithsonian, 2019](https://www.smithsonianmag.com/smart-news/designers-are-selling-expensive-clothes-dont-existand-people-are-actually-buying-them-180973568/)).

So the dress is the photo. Nobody will ever touch it, and one picture carries the whole price. That picture was made by people, by hand, for one customer.

From January 2021 to June 2023 I led machine learning research and development for a digital fashion product at a startup that is still in stealth. The question was how to make that photograph by machine: any person, any garment, no 3D artist in the loop.

This page is about how we went at it, what turned out to be hard, and how the field moved while we worked. The NDA covers the product and its numbers, so the mechanics are explained with figures I drew again on open data and open models. The photos and the video are from the project.

## One photo, three products

<img class="img-fluid rounded d-block mx-auto my-4" src="/assets/img/projects/4-ai-fashion-app/slide.jpg" width="1800" height="1011" alt="A slide titled 'Digital clothing can be both futuristic and indistinguishable from the tangible one'. In the middle a model in a white two-piece, labelled Tangible. An arrow labelled Digital Outfits leads left to the same model in an iridescent violet jacket and cut-out flared trousers, labelled Digital. An arrow labelled Virtual Fitting leads right to her in a red leather jacket and black shorts, labelled Digital/Tangible, and an arrow labelled Hyper Personalized Ads leads back" loading="lazy">

<p class="tfs-caption">A slide from the project. Middle: the photo as taken. Left: an outfit that exists only as a file. Right: real clothes, fitted virtually.</p>

The slide shows three things the same technology could sell:

- **Digital outfits.** Clothes that exist only as files and are worn only in pictures. DressX, launched in August 2020, sells "virtual looks that can be exclusively worn on photos and videos" ([DressX](https://dressx.com/pages/about)).
- **Virtual fitting.** Real clothes from a shop, put on the buyer before the order.
- **Personalized ads.** The same thing pointed the other way: the shop shows its jacket on you instead of on a model.

They look like one feature. They make two different promises.

For a digital outfit, the picture is what you paid for. It has to be beautiful, and nobody checks whether the fabric would really sit like that.

For virtual fitting, the picture is a forecast about a parcel. A jacket that will be tight in the shoulders has to look tight. This is where the money is: in 2022 size and fit were the most common reason people sent purchases back, 45% of all returns ([Narvar, 2022](https://corp.narvar.com/resources/2022-the-state-of-returns)).

A beautiful picture and an honest one are not the same picture. That tension ran through the whole project, and it split the work in two.

## The demo

<video class="d-block mx-auto my-4 rounded" width="300" height="533" style="max-width: 100%; height: auto" autoplay muted loop playsinline preload="metadata" poster="/assets/img/projects/4-ai-fashion-app/try_on_app_poster.jpg" aria-label="A phone screen recording: browsing a shop's digital fashion section, opening an iridescent violet outfit, tapping Try now, choosing a photo, a progress indicator, and the same person wearing the outfit, with Add to cart below">
  <source src="/assets/img/projects/4-ai-fashion-app/try_on_app.webm" type="video/webm">
  <source src="/assets/img/projects/4-ai-fashion-app/try_on_app.mp4" type="video/mp4">
</video>

<p class="tfs-caption">The flow inside a shop's app: open a digital outfit, tap Try now, pick a photo of yourself, see yourself in it.</p>

The interface was meant to be dull: a product page with one more button.

Everything difficult happens while the progress indicator spins. The rest of this page is about those seconds.

## Two routes

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/routes-light.png" width="760" alt="Two ways to put a garment on a photo. The image route reads the body from a person&#x27;s photo, warps the product photo onto it and repaints the result with a generator, at first a GAN and later a diffusion model: it works from any product photo but guesses what it cannot see and knows nothing of sizes. The 3D route fits a body model, SMPL-X, to the photo or to measurements, drapes a 3D garment made from its sewing pattern on that body with a cloth simulation and renders it into the photo: it knows sizes and fit but needs a 3D file for every garment" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/routes-dark.png" width="760" alt="Two ways to put a garment on a photo. The image route reads the body from a person&#x27;s photo, warps the product photo onto it and repaints the result with a generator, at first a GAN and later a diffusion model: it works from any product photo but guesses what it cannot see and knows nothing of sizes. The 3D route fits a body model, SMPL-X, to the photo or to measurements, drapes a 3D garment made from its sewing pattern on that body with a cloth simulation and renders it into the photo: it knows sizes and fit but needs a 3D file for every garment" loading="lazy">

There were two ways to make the picture, and we worked on both.

**The image route** treats try-on as photo editing. Find the body in the photo, bend the product photo onto it, and let a generative model repaint the region so that the result reads as one photograph. It needs nothing but two pictures. It also knows nothing about size: it will put a size S dress on any body and make it look fine.

**The 3D route** treats try-on as dressing a body. Estimate the person's body in 3D, take the garment as a 3D model sewn from its pattern, let cloth physics drape it, and render the result. It knows size, pose and fabric. It needs a 3D model of every garment, and its renders easily look like a video game.

The routes line up with the two promises. A digital outfit is a 3D model from birth, while a shop usually has nothing but photos of its products. So the image route was the one that could reach every shop, and the 3D route was the one that could tell the truth about fit.

## The image route

### Reading the body

Before a garment can go onto a photo, the system has to know which pixels are the body and which are the clothes. In 2021 that meant running three separate networks on every photo:

- **Pose.** OpenPose finds the skeleton: 25 points from the nose to the toes, feet included ([Cao et al., 2019](https://arxiv.org/abs/1812.08008); [code](https://github.com/CMU-Perceptual-Computing-Lab/openpose)).
- **Parsing.** A human parser labels every pixel: hair, face, upper clothes, dress, left arm and so on. SCHP, trained on the LIP dataset with its 20 labels, was a common choice ([Li et al., 2019](https://arxiv.org/abs/1910.09777); [code](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)).
- **Surface.** DensePose goes furthest. For each pixel of a person it says which of 24 patches of the body's surface the pixel shows, and where on that patch, as two coordinates U and V ([Güler et al., 2018](https://arxiv.org/abs/1802.00434); [code](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose)).

<img class="img-fluid rounded d-block mx-auto my-4" src="/assets/img/projects/4-ai-fashion-app/densepose.png" width="800" height="512" alt="Left: a young woman in a pink tracksuit and white sneakers crouching by a window. Right: the same silhouette on black, split into coloured patches: the head yellow, the torso and arms in blues and greens, the legs and feet in teal and blue" loading="lazy">

<p class="tfs-caption">A crouching pose and its DensePose body-part map, from the project. Each colour is one patch of the body's surface.</p>

The crouch is the reason to bother. A skeleton says where the joints are, but not which side of a limb faces the camera. Parsing says what is visible, but not which part of the body it belongs to.

DensePose gives every pixel an address on the skin, and a garment is attached to skin. A sleeve follows the surface of the arm, not the line from the elbow to the wrist.

Parsing has a trap of its own: it describes the old clothes. If the person wears jeans and we want a dress, the jeans have to be forgotten, or the dress comes out shaped like jeans.

Try-on papers handle this with a "clothing-agnostic" picture of the person, which keeps the face, hair and pose and masks out the clothes ([Han et al., 2018](https://arxiv.org/abs/1711.08447)). VITON-HD masks the arms as well, because they "hint at the sleeve length" ([Choi et al., 2021](https://arxiv.org/abs/2103.16874)).

### Bending the garment

The product photo shows the garment flat, from the front, on nobody. Before anything is generated, it has to be moved and bent to where the body is.

The classic tool is the thin-plate spline. Mark matching points on the product and on the person: shoulder to shoulder, waist to waist, hem to hem. The spline is the smoothest map of the plane that carries each point exactly onto its partner:

$$
f(\mathbf{x}) = \mathbf{a} + A\mathbf{x} + \sum_{i=1}^{n} \mathbf{w}_i \, U\big(\lVert \mathbf{x} - \mathbf{p}_i \rVert\big), \qquad U(r) = r^2 \log r .
$$

The first two terms are an affine map: shift, rotation, scale and shear. The sum adds a smooth bump around each control point $$\mathbf{p}_i$$.

The weights $$\mathbf{w}_i$$ come from a linear system. It makes each $$f(\mathbf{p}_i)$$ land on its partner and keeps the bending energy, the integral of the squared second derivatives of $$f$$, as small as possible. Picture a thin steel sheet pinned at the points: it settles into exactly this shape, hence the name ([Bookstein, 1989](https://doi.org/10.1109/34.24792)).

CP-VTON made the spline learnable. A network looks at the person and the product and predicts the offsets of a 5×5 grid of control points, an output of 2×5×5 = 50 numbers ([Wang et al., 2018](https://arxiv.org/abs/1807.07688); [code](https://github.com/sergeywong/cp-vton)). A whole garment is placed by fifty numbers.

The figure does the same by hand, with 14 points, on the project's own example.

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/warp-light.png" width="760" alt="Four panels. The product photo of a floral dress with 14 points marked on its sleeves, neckline, waist and hem; the person in a white dress with the same points marked on her body; the floral dress bent onto her by a thin-plate spline, with the old white sleeves and skirt still showing around it and the fabric as flat as on the product photo; and the project&#x27;s trained model, which replaces the old dress, shapes the puffed sleeves to the arms and matches the light of the scene" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/warp-dark.png" width="760" alt="Four panels. The product photo of a floral dress with 14 points marked on its sleeves, neckline, waist and hem; the person in a white dress with the same points marked on her body; the floral dress bent onto her by a thin-plate spline, with the old white sleeves and skirt still showing around it and the fabric as flat as on the product photo; and the project&#x27;s trained model, which replaces the old dress, shapes the puffed sleeves to the arms and matches the light of the scene" loading="lazy">

The third panel shows what a warp cannot do:

- **The light stays flat.** The fabric bends in the picture but not in space, so there is no shading where it turns away from the camera.
- **The old garment stays.** The white sleeves and skirt show around the new dress.
- **Volume gets squeezed.** The puffed sleeves are compressed like a picture, not gathered like cloth.
- **Nothing can be in front.** A hand on the waist or a strand of hair would be painted over.

The papers of the time name the same limits. TPS has "limited degrees of freedom" and cannot model "highly non-rigid deformation" ([Ge et al., 2021](https://arxiv.org/abs/2103.04559)). Where the body hides part of the garment, warping causes "pixel-squeezing artifacts" ([Lee et al., 2022](https://arxiv.org/abs/2206.14180)).

The fourth panel is what a trained model makes of the same inputs. It takes the warped garment as a hint and repaints the whole region: the old dress is gone, the sleeves sit on the arms, and the light matches the scene.

### Repainting

In 2021 the repainting model was a GAN. VITON-HD, the first to work at high resolution, generated images at 1024×768 and was trained on 11,647 pairs of a product photo and a person wearing that product ([Choi et al., 2021](https://arxiv.org/abs/2103.16874); [code](https://github.com/shadow2496/VITON-HD)).

Those pairs hide a trick. Nobody has photos of the same person, in the same pose, in two different shirts.

So the model learns by reconstruction. Take a photo of someone in a shirt, hide the shirt, give the model the shop photo of that same shirt, and ask it to paint the shirt back. At test time the shirt is a different one.

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/pairs-light.png" width="760" alt="How an image try-on model is trained and how it is used. In training, a photo of a person in shirt A has the shirt hidden, keeping the face, hair and pose; the generator gets the shop photo of the same shirt A, paints it back, and the result is compared with the original photo. In use, the same masked photo gets the shop photo of a different shirt B, and there is no photo to compare with. If the mask lets the outline of shirt A through, shirt B comes out in the shape of A" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/pairs-dark.png" width="760" alt="How an image try-on model is trained and how it is used. In training, a photo of a person in shirt A has the shirt hidden, keeping the face, hair and pose; the generator gets the shop photo of the same shirt A, paints it back, and the result is compared with the original photo. In use, the same masked photo gets the shop photo of a different shirt B, and there is no photo to compare with. If the mask lets the outline of shirt A through, shirt B comes out in the shape of A" loading="lazy">

If the mask leaks the old shirt's outline, the model learns to copy it, and a new garment comes out in the shape of the old one. This is why the clothing-agnostic picture of the person matters so much.

The datasets had another limit: they were catalogue photos. VITON-HD collected "frontal-view woman and top clothing image pairs", and its authors wrote that existing methods, their own included, "do not provide robust performance for in-the-wild images". A photo a buyer uploads is taken in a hallway, in a mirror, at arm's length.

Then diffusion arrived. Stable Diffusion went public on 22 August 2022 with open weights, trained "on 512x512 images from a subset of the LAION-5B database" ([Stability AI, 2022](https://stability.ai/news/stable-diffusion-public-release); [code](https://github.com/CompVis/stable-diffusion)). An inpainting version followed in October: give it a photo, a mask and a prompt, and it repaints only the masked region ([model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)).

For try-on this moved the hard part. A GAN trained on twelve thousand pairs knows little beyond its dataset. A diffusion model trained on a large share of the internet's images already knows how wool sits on a shoulder, how silk catches light and what a hand in a pocket looks like.

It knows nothing about this particular dress, and words cannot describe a print exactly. Ask for "a floral dress with puffed sleeves" and you get a floral dress, not the one in the shop. In try-on the print is the product: a melted logo makes it a different product.

So the work of 2022 and 2023 was about making the generator look at the garment instead of imagining one:

- **Teach it one object.** DreamBooth fine-tunes the model on a few photos of a subject, so that it "learns to bind a unique identifier with that specific subject" ([Ruiz et al., 2022](https://arxiv.org/abs/2208.12242)). Textual Inversion learns a new word for the subject instead and leaves the model alone ([Gal et al., 2022](https://arxiv.org/abs/2208.01618)).
- **Show it an example.** Paint by Example replaces the text prompt with a picture of the object to paint in ([Yang et al., 2022](https://arxiv.org/abs/2211.13227); [code](https://github.com/Fantasy-Studio/Paint-by-Example)).
- **Pin it to the body.** ControlNet attaches a trainable copy of the generator's encoder that reads a skeleton, an edge map or a segmentation and holds the image to it ([Zhang and Agrawala, 2023](https://arxiv.org/abs/2302.05543); [code](https://github.com/lllyasviel/ControlNet)).

I fine-tuned both kinds of generator over the project, GANs first and then Stable Diffusion. The difference was less in quality than in where the effort went.

With a GAN, the effort went into getting a believable photo at all. With diffusion the photo was believable from the first try, and the effort moved to keeping it faithful: the same print, the same face, the same person.

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/timeline-light.png" width="760" alt="A timeline from late 2017 to mid-2023 in four lanes, with the project, January 2021 to June 2023, shaded. Try-on models: VITON (Nov 2017), CP-VTON (Jul 2018), ACGPN (Mar 2020), VITON-HD (Mar 2021), HR-VITON (Jun 2022), TryOnDiffusion (Jun 2023). Image generators: StyleGAN2 (Dec 2019), DDPM (Jun 2020), DALL·E (Jan 2021), Latent Diffusion (Dec 2021), DALL·E 2 (Apr 2022), Stable Diffusion (Aug 2022), ControlNet (Feb 2023). Bodies in 3D: DensePose (Feb 2018), SMPL-X (Apr 2019), PIFuHD (Apr 2020), SMPL-X in Unity (Jun 2021), ICON (Dec 2021), SHAPY (Jun 2022), ECON (Dec 2022). Digital fashion: a digital dress, 9,500 dollars (May 2019), DressX (Aug 2020), Gucci bag on Roblox (May 2021), Facebook → Meta (Oct 2021), Metaverse Fashion Week (Mar 2022), try-on in Google Search (Jun 2023)." loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/timeline-dark.png" width="760" alt="A timeline from late 2017 to mid-2023 in four lanes, with the project, January 2021 to June 2023, shaded. Try-on models: VITON (Nov 2017), CP-VTON (Jul 2018), ACGPN (Mar 2020), VITON-HD (Mar 2021), HR-VITON (Jun 2022), TryOnDiffusion (Jun 2023). Image generators: StyleGAN2 (Dec 2019), DDPM (Jun 2020), DALL·E (Jan 2021), Latent Diffusion (Dec 2021), DALL·E 2 (Apr 2022), Stable Diffusion (Aug 2022), ControlNet (Feb 2023). Bodies in 3D: DensePose (Feb 2018), SMPL-X (Apr 2019), PIFuHD (Apr 2020), SMPL-X in Unity (Jun 2021), ICON (Dec 2021), SHAPY (Jun 2022), ECON (Dec 2022). Digital fashion: a digital dress, 9,500 dollars (May 2019), DressX (Aug 2020), Gucci bag on Roblox (May 2021), Facebook → Meta (Oct 2021), Metaverse Fashion Week (Mar 2022), try-on in Google Search (Jun 2023)." loading="lazy">

The project's dates make a neat pair of bookends. OpenAI showed DALL·E on 5 January 2021, the month I started ([OpenAI, 2021](https://openai.com/index/dall-e/)).

On 14 June 2023, the month I left, Google added virtual try-on to Search ([Google, 2023](https://blog.google/products/shopping/ai-virtual-try-on-google-shopping/)). It runs on a diffusion model ([how it works](https://blog.google/products/shopping/virtual-try-on-google-generative-ai/)), and the paper behind it, TryOnDiffusion, argued that "the sequence of finding displacements, warping, and blending often creates artifacts" ([Zhu et al., 2023](https://arxiv.org/abs/2306.08276)).

That sequence was the image route as everyone had built it since 2018. TryOnDiffusion warps the garment "implicitly via a cross attention mechanism" and was trained on 4 million pairs of the same person in the same garment in different poses.

In two and a half years the field replaced the generator, and then the pipeline around it.

## The 3D route

### A body in a few numbers

SMPL describes a human body as a mesh of 6,890 vertices driven by a few numbers. Shape is usually 10 coefficients $$\beta$$. Pose is 72 numbers $$\theta$$: a rotation for each of 23 joints and one for the body as a whole, three numbers each ([Loper et al., 2015](https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf)). In outline:

$$
M(\beta, \theta) = W\big(\bar{T} + B_S(\beta) + B_P(\theta),\; J(\beta),\; \theta,\; \mathcal{W}\big).
$$

Read it from the inside:

- $$\bar{T}$$ is an average body standing with its arms out.
- $$B_S(\beta)$$ moves every vertex according to the shape: taller, wider, heavier.
- $$B_P(\theta)$$ adds the corrections a pose needs, such as the bulge of a bent elbow.
- $$J(\beta)$$ places the joints inside this particular body.
- $$W$$, the skinning function, turns each vertex with the bones near it, in proportions given by the weights $$\mathcal{W}$$.

SMPL-X (2019) adds hands and a face: 10,475 vertices and 54 joints ([Pavlakos et al., 2019](https://arxiv.org/abs/1904.05866); [code](https://github.com/vchoutas/smplx)).

<img class="img-fluid rounded d-block mx-auto my-4" src="/assets/img/projects/4-ai-fashion-app/smpl_unity.png" width="760" height="553" alt="The Unity editor with two grey SMPL bodies, a man and a woman, standing in a T-pose on a plane. Below them an animation state machine with the states Idle, WalkForward, WalkBackward, Jump, Crouch and crouch2idle" loading="lazy">

<p class="tfs-caption">SMPL's female and male bodies in Unity, from the project, driven by an ordinary game animation controller.</p>

For a product, the format mattered as much as the model. SMPL comes with code for Unity, and the SMPL-X integration for Unity appeared in June 2021 ([SMPL-X](https://smpl-x.is.tue.mpg.de/)). In a game engine the body can walk, turn, be lit and run on a phone.

There is a catch for a startup. SMPL's licence covers "non-commercial scientific research", excludes "incorporation in a commercial product", and points companies to commercial licences sold through Meshcapade ([SMPL licence](https://smpl.is.tue.mpg.de/modellicense.html)).

The bodies in my figures below are MakeHuman's instead. Its assets are in the public domain under CC0 and can be shown and changed freely ([MakeHuman](https://github.com/makehumancommunity/makehuman)).

### What a tape measure computes

Fitting needs the numbers a size chart uses: bust, waist and hips. A mesh doesn't come with them, so they have to be measured.

Cut the mesh with a horizontal plane. Every triangle the plane crosses gives a short segment, and the segments join into closed loops: one around the torso, one around each arm. Keep the torso's loop.

Then don't take its length. A tape measure does not dip into the hollow between the breasts or follow the small of the back; it stretches across them. What it reads is the perimeter of the convex hull of the cross-section, the shortest closed curve that goes around it.

The height of the cut is the subtle part. Each measurement has its own rule:

- **Waist:** the narrowest level of the torso.
- **Hips:** the widest level below it.
- **Bust:** the level where the chest comes furthest forward.

Each rule is a small search over heights, here in steps of half a centimetre.

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/bodies-light.png" width="760" alt="Three versions of the same open 3D body, thin, average and full, with tape-measure loops drawn around the bust, waist and hips: thin: bust 81 cm, waist 59 cm, hips 87 cm; average: bust 84 cm, waist 65 cm, hips 95 cm; full: bust 90 cm, waist 71 cm, hips 103 cm" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/bodies-dark.png" width="760" alt="Three versions of the same open 3D body, thin, average and full, with tape-measure loops drawn around the bust, waist and hips: thin: bust 81 cm, waist 59 cm, hips 87 cm; average: bust 84 cm, waist 65 cm, hips 95 cm; full: bust 90 cm, waist 71 cm, hips 103 cm" loading="lazy">

### Where size lives

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/shape-light.png" width="760" alt="The average body seen from the front and the side, coloured by how far each point moves between the thin and the full build, up to about 3.5 cm: most around the belly, hips, buttocks and thighs, a little on the arms, almost nothing on the head, hands and feet" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/shape-dark.png" width="760" alt="The average body seen from the front and the side, coloured by how far each point moves between the thin and the full build, up to about 3.5 cm: most around the belly, hips, buttocks and thighs, a little on the arms, almost nothing on the head, hands and feet" loading="lazy">

The figure colours each point of the body by how far it moves between the thin and the full build. The belly, hips and thighs move by up to about 3.5 cm. The head, hands and feet hardly move at all.

This is what makes size from a photo hard. The parts of the body that carry information about size are the parts clothes cover. A person in a loose sweater shows their face, hands and shoes perfectly, and their waist not at all.

SHAPY, a 2022 method built for exactly this task, estimated body shape from a single photo with average errors of 6.5 cm at the chest, 6.9 cm at the waist and 5.7 cm at the hips ([Choutas et al., 2022](https://arxiv.org/abs/2206.07036); [code](https://github.com/muelea/shapy)).

Compare that with the measured bodies above. The whole distance between my thin and full bodies is 9 cm at the bust and 12 cm at the waist.

### Cloth

A 3D garment is a mesh cut from flat pattern pieces and sewn together in software, the way a tailor would do it, in programs such as [CLO 3D](https://www.clo3d.com/en/) or [Marvelous Designer](https://www.marvelousdesigner.com/). To put it on a body it is simulated: gravity pulls, the fabric resists stretching, resists bending a little, and must not pass through the skin.

Position-based dynamics turns each of these rules into a constraint on the positions of the mesh's vertices. Two neighbours $$i$$ and $$j$$ that were $$L$$ apart on the pattern should stay $$L$$ apart:

$$
C(\mathbf{x}_i, \mathbf{x}_j) = \lVert \mathbf{x}_i - \mathbf{x}_j \rVert - L = 0 .
$$

Every time step lets the vertices fly freely under gravity, then nudges them, constraint by constraint, until the rules hold again ([Müller et al., 2007](https://doi.org/10.1016/j.jvcir.2007.01.005)).

Plain PBD has a flaw: the more nudging iterations you run, the stiffer the cloth becomes. XPBD fixes that by giving each constraint a compliance $$\alpha$$, the inverse of stiffness, and computing each nudge as

$$
\Delta\lambda = \frac{-C - \tilde\alpha \lambda}{\nabla C \, M^{-1} \nabla C^{\top} + \tilde\alpha}, \qquad \tilde\alpha = \frac{\alpha}{\Delta t^2} .
$$

Here $$\lambda$$ is how hard the constraint has pushed so far, and $$M$$ holds the vertices' masses. With compliance in the denominator the cloth behaves the same "regardless of iteration count" ([Macklin et al., 2016](https://matthias-research.github.io/pages/publications/XPBD.pdf)).

A fabric then becomes a few numbers. Silk and wool both resist stretching; where they differ is bending. In the simulation below, silk's bending constraints are compliant and wool's are rigid.

<img class="tfs-figure tfs-light" src="/assets/img/projects/4-ai-fashion-app/figures/drape-light.png" width="760" alt="The same A-line skirt simulated on the thin and the full body in two fabrics. Silk, which bends freely, falls in narrow folds close to the legs; wool, which resists bending, keeps a wide A-line flare. On the full body the hips push the skirt outward" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/4-ai-fashion-app/figures/drape-dark.png" width="760" alt="The same A-line skirt simulated on the thin and the full body in two fabrics. Silk, which bends freely, falls in narrow folds close to the legs; wool, which resists bending, keeps a wide A-line flare. On the full body the hips push the skirt outward" loading="lazy">

Silk falls in narrow folds along the legs; wool holds its A-line. On the full body the hips push the skirt outward.

This is the information the image route lacks. A generator has no numbers for the fabric: it draws the drape it has seen most often in pictures like this one.

The difficulties of this route were not in the physics:

- **Every garment needs a 3D model.** That is natural for a digital outfit, which starts as one. A shop that adds thousands of products a season has photos of them.
- **Every fabric needs measuring.** A real fabric's stretch and bending have to be measured or guessed.
- **Phones are slow.** A simulation does not fit into a progress bar on a phone. The drape has to be computed in advance or learned: TailorNet (2020) predicts it with a network "more than 1000 times faster" than simulating ([Patel et al., 2020](https://arxiv.org/abs/2003.04583); [code](https://github.com/chaitanya100100/TailorNet)).
- **Renders look rendered.** A body from a game engine next to a real face looks like a game character.

## What I took from it

**The routes met.** By 2023 the image route was steered by what the 3D route computes: skeletons, body-part maps, depth. ControlNet made a pose an ordinary input to an image generator. The 3D body did not become the picture; it became the scaffolding the picture is built on.

**Beauty arrived before honesty.** Diffusion made the picture convincing within a year. Fit stayed a measurement problem.

Google's 2023 launch answered it without estimating anyone's body: it shows the garment on real models "ranging in sizes XXS-4XL", so that a shopper can find a body like their own.

**In a fast field, a plan is a bet on direction.** In January 2021 the best try-on was a GAN at 1024×768 that needed a parser and a warp. In June 2023 it was a diffusion model with no explicit warp, trained on 4 million pairs.

A plan made at the start was out of date by the middle. What kept its value was the work that did not depend on the generator: reading the body and knowing the garment.

In 2019 it took designers to put one digital dress on one photo. In 2023 a search engine dressed models of every size for any shopper who asked. Whether a garment will fit you is still a question for a tape measure.

## References

Virtual try-on:

- Han et al. [VITON: An Image-based Virtual Try-on Network](https://arxiv.org/abs/1711.08447). CVPR 2018.
- Wang et al. [Toward Characteristic-Preserving Image-based Virtual Try-On Network](https://arxiv.org/abs/1807.07688). ECCV 2018. [Code](https://github.com/sergeywong/cp-vton).
- Yang et al. [Towards Photo-Realistic Virtual Try-On by Adaptively Generating↔Preserving Image Content](https://arxiv.org/abs/2003.05863). CVPR 2020. [Code](https://github.com/switchablenorms/DeepFashion_Try_On).
- Ge et al. [Parser-Free Virtual Try-on via Distilling Appearance Flows](https://arxiv.org/abs/2103.04559). CVPR 2021. [Code](https://github.com/geyuying/PF-AFN).
- Choi et al. [VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization](https://arxiv.org/abs/2103.16874). CVPR 2021. [Code](https://github.com/shadow2496/VITON-HD).
- Lee et al. [High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions](https://arxiv.org/abs/2206.14180). ECCV 2022. [Code](https://github.com/sangyun884/HR-VITON).
- Zhu et al. [TryOnDiffusion: A Tale of Two UNets](https://arxiv.org/abs/2306.08276). CVPR 2023.

Reading the body:

- Cao et al. [OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1812.08008). TPAMI 2019.
- Güler, Neverova and Kokkinos. [DensePose: Dense Human Pose Estimation In The Wild](https://arxiv.org/abs/1802.00434). CVPR 2018.
- Li et al. [Self-Correction for Human Parsing](https://arxiv.org/abs/1910.09777). TPAMI 2020.
- Bookstein. [Principal Warps: Thin-Plate Splines and the Decomposition of Deformations](https://doi.org/10.1109/34.24792). TPAMI 1989.

Image generators:

- Rombach et al. [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752). CVPR 2022.
- Ruiz et al. [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242). CVPR 2023.
- Gal et al. [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618). ICLR 2023.
- Yang et al. [Paint by Example: Exemplar-based Image Editing with Diffusion Models](https://arxiv.org/abs/2211.13227). CVPR 2023.
- Zhang and Agrawala. [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543). ICCV 2023.

Bodies and cloth:

- Loper et al. [SMPL: A Skinned Multi-Person Linear Model](https://files.is.tue.mpg.de/black/papers/SMPL2015.pdf). SIGGRAPH Asia 2015.
- Pavlakos et al. [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://arxiv.org/abs/1904.05866). CVPR 2019.
- Choutas et al. [Accurate 3D Body Shape Regression using Metric and Semantic Attributes](https://arxiv.org/abs/2206.07036). CVPR 2022.
- Müller et al. [Position Based Dynamics](https://doi.org/10.1016/j.jvcir.2007.01.005). Journal of Visual Communication and Image Representation, 2007.
- Macklin, Müller and Chentanez. [XPBD: Position-Based Simulation of Compliant Constrained Dynamics](https://matthias-research.github.io/pages/publications/XPBD.pdf). Motion in Games 2016.
- Patel, Liao and Pons-Moll. [TailorNet: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style](https://arxiv.org/abs/2003.04583). CVPR 2020.

{% endraw %}
