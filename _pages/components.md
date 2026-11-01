---
layout: page
title: Post components
permalink: /components/
description: The blocks that give a long post its shape, with their markup and when to use them.
nav: false
sitemap: false
robots: noindex, follow
---

<!-- the posts' container, so that the blocks look exactly as they do in a post -->
<div id="markdown-content" markdown="1">

A long post is read in two passes: a quick one through the summary, the headings, the charts and the marked lines, and a slow one through everything else. These blocks serve the quick pass without getting in the way of the slow one. Each is plain HTML in a post's markdown; `markdown="1"` lets the markdown inside it render. Their styles are in `_sass/_components.scss`.

The rule for all of them is scarcity. A summary at the top, one point to remember or one takeaway per section at most, a marked phrase every few screens. A page where everything is highlighted has nothing highlighted.

## Summary at the top

The first thing in a long post, before the story starts: three or four sentences that a reader could stop after, and up to three numbers worth remembering.

<aside class="tldr" markdown="1">
<p class="tldr-label">In short</p>

Polymarket's prices are good probabilities where people actually trade: a week before the close they are off by under two points on average. The famous accuracy figures say less than they seem, because most questions resolve No, and grouping markets by volume manufactures a bias that isn't there.

<div class="stats">
<div class="stat"><span class="stat-value">1.8</span><span class="stat-label">points between price and outcome, on average, a week before the close</span></div>
<div class="stat"><span class="stat-value">77%</span><span class="stat-label">right for a forecaster who always says No</span></div>
<div class="stat"><span class="stat-value">−39%</span><span class="stat-label">the average return on contracts bought at 2–5 cents</span></div>
</div>
</aside>

```html
<aside class="tldr" markdown="1">
<p class="tldr-label">In short</p>

Three or four sentences: the answer, then what is surprising about it.

<div class="stats">
<div class="stat"><span class="stat-value">1.8</span><span class="stat-label">what the number measures</span></div>
...
</div>
</aside>
```

## Key numbers

The same grid works anywhere a section turns on two or three numbers. Each value needs its unit and a line that says what it measures.

<div class="stats">
<div class="stat"><span class="stat-value">48%</span><span class="stat-label">of the base rate's error removed a month before the close</span></div>
<div class="stat"><span class="stat-value">70%</span><span class="stat-label">a week before</span></div>
<div class="stat"><span class="stat-value">82%</span><span class="stat-label">a day before</span></div>
</div>

```html
<div class="stats">
<div class="stat"><span class="stat-value">48%</span><span class="stat-label">what it measures</span></div>
</div>
```

## Points to remember, traps and notes

A callout lifts one idea out of the flow. `callout-key` is the point of a section, `callout-warn` a mistake that is easy to make, and a plain callout an aside with a label of its own, such as "For traders" or "For engineers".

<div class="callout callout-key" markdown="1">
<p class="callout-label">Key point</p>

When a liquid market says 70%, read it as about 70%. A single outcome, either way, says almost nothing about whether that was right.
</div>

<div class="callout callout-warn" markdown="1">
<p class="callout-label">Trap</p>

Never sort markets by their total volume. It counts the trading after the forecast, and surprises draw trading.
</div>

<div class="callout" markdown="1">
<p class="callout-label">For traders</p>

The longshot discount is about the size of the spread. That is why it survives.
</div>

```html
<div class="callout callout-key" markdown="1">
<p class="callout-label">Key point</p>

One or two sentences.
</div>
```

## Method, folded

Details that a careful reader needs and a quick one does not: how the sample was drawn, the full formula, the assumptions. Closed by default; the summary line says what is inside.

<details class="method" markdown="1">
<summary>How the markets were drawn and weighted</summary>

Two independent random draws, one by topic and one by volume. Each market counts in inverse proportion to its chance of being in either, so pooled figures describe all markets:

$$
\hat\mu = \frac{\sum_i y_i / \pi_i}{\sum_i 1 / \pi_i}.
$$

Uncertainty comes from a bootstrap over events.
</details>

```html
<details class="method" markdown="1">
<summary>What is inside, in a few words</summary>

The method, with formulas and code if needed.
</details>
```

## Takeaway

The one sentence a reader should keep from a chart or a section, right after it. It is a conclusion, not a slogan: it states a finding with its number.

<p class="takeaway" markdown="1">Markets with real trading a week before the close are off by **1.8 points** on average.</p>

```html
<p class="takeaway" markdown="1">The finding, with its number.</p>
```

## A chain of causes

For an argument that runs through a few steps, each step a short phrase; the last one, the consequence, is in red.

<p class="flow"><span>a longshot starts winning</span><span>everyone trades</span><span>the market grows</span><span>"big markets underprice longshots"</span></p>

```html
<p class="flow"><span>cause</span><span>step</span><span>consequence</span></p>
```

## A quotation set apart

For a sentence that deserves to be read twice. Not for every quote; ordinary quotations stay in the text.

<blockquote class="pull" markdown="1">
Insiders having an edge to the market is a good thing.

<cite>Shayne Coplan, Polymarket, to CBS</cite>
</blockquote>

```html
<blockquote class="pull" markdown="1">
The quotation.

<cite>Who said it, where</cite>
</blockquote>
```

## Marking a phrase

Inside a sentence, `<mark>` puts a highlighter stroke under <mark>the phrase that carries the paragraph</mark>, and a hand-drawn line marks <span class="underline">a term defined here</span> or a contrast. Bold is for the one sentence a skimming reader should not miss; italics are for a *new term* the first time it appears and for stress. Names of layers, functions, files and APIs go in code: `RMSNorm`, `prices-history`, `torch.compile`.

```html
<mark>the key phrase</mark>   <span class="underline">a term or a contrast</span>
```

## A pointer

A link to another section or post that the reader is invited to follow: <a class="arrow-link" href="#a-chain-of-causes">the chain of causes above</a>.

```html
<a class="arrow-link" href="#section">where it leads</a>
```

## Headings

`##` headings say what the section finds or answers, not only its topic: "Calibration: a week out, prices are off by 1.8 points" rather than "Calibration". `###` headings split a section longer than about five screens into steps, and nothing shorter. Both stay plain: no puns, no questions that the section does not answer.

</div>
