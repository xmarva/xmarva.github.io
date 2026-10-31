---
layout: post
title: "Can AI be creative?"
date: 2025-02-13 15:00:00
description: "What happens in a brain that has a new idea, where recombination stops, and what the measurements of machine creativity show, from AlphaGo's move 37 to GPT-4 on the Alternate Uses Task"
tags: [ai, neuroscience, creativity]
categories: [cognitive-science, machine-learning]
featured: true
thumbnail: /assets/posts/creative-ai/cover.png
og_image: https://xmarva.github.io/assets/posts/creative-ai/cover.png
stylesheets: [/assets/css/tfs-course.css]
toc:
  beginning: true
---
{% raw %}

On 10 March 2016, in the second game of its match against Lee Sedol in Seoul, AlphaGo played its 37th move on the fifth line, a shoulder hit at P10 of the kind professionals learn not to play.

One of the English commentators "chuckled and said: 'I thought it was a mistake.'" Fan Hui, the European champion, who had lost to AlphaGo a few months earlier, was watching: "It's not a human move. I've never seen a human play this move. … So beautiful"<sup class="cite">[<a href="#ref-1" id="cite-1" data-ref="1" aria-label="Reference 1: Metz, C. (2016). The sadness and beauty of watching Google&#x27;s AI play Go. Wired.">1</a>]</sup>.

Lee Sedol stood up and left the room, and spent nearly fifteen minutes on his reply. AlphaGo itself had put the chance that a human would play the move at one in ten thousand<sup class="cite">[<a href="#ref-2" id="cite-2" data-ref="2" aria-label="Reference 2: Hassabis, D. (2016). What we learned in Seoul with AlphaGo. Google.">2</a>]</sup>.

<img class="tfs-figure tfs-light" src="/assets/posts/creative-ai/move37-light.png" width="760" alt="The Go board after move 37 of the second game between AlphaGo and Lee Sedol: AlphaGo&#x27;s black stone at P10, a shoulder hit on the fifth line against White&#x27;s stone at Q11, which AlphaGo estimated a human would play with a probability of 1 in 10,000" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/creative-ai/move37-dark.png" width="760" alt="The Go board after move 37 of the second game between AlphaGo and Lee Sedol: AlphaGo&#x27;s black stone at P10, a shoulder hit on the fifth line against White&#x27;s stone at Q11, which AlphaGo estimated a human would play with a probability of 1 in 10,000" loading="lazy">

Lee said afterwards: "I thought AlphaGo was based on probability calculation and that it was merely a machine. But when I saw this move, I changed my mind. Surely, AlphaGo is creative"<sup class="cite">[<a href="#ref-3" id="cite-3" data-ref="3" aria-label="Reference 3: DeepMind. AlphaGo. Google DeepMind research page.">3</a>]</sup>.

The view he changed his mind about is older than computers. In 1843 Ada Lovelace wrote of Babbage's Analytical Engine: "The Analytical Engine has no pretensions whatever to originate anything. It can do whatever we know how to order it to perform"<sup class="cite">[<a href="#ref-4" id="cite-4" data-ref="4" aria-label="Reference 4: Lovelace, A. A. (1843). Notes on the Sketch of the Analytical Engine by L. F. Menabrea. Scientific Memoirs, 3.">4</a>]</sup>.

In 1950 Alan Turing answered with a question that turns the objection on the people making it: "Who can be certain that 'original work' that he has done was not simply the growth of the seed planted in him by teaching…"<sup class="cite">[<a href="#ref-5" id="cite-5" data-ref="5" aria-label="Reference 5: Turing, A. M. (1950). Computing machinery and intelligence. Mind, 59(236), 433–460.">5</a>]</sup>.

That is the question here. Not whether a machine can create something from nothing, but whether anyone can: what a brain does when a person has a new idea, how much of it is recombining what was already there, where recombination runs out, and what the measurements of machine creativity actually show.

## What counts as creative

Psychologists use a definition with two parts: an idea or a product is creative if it is both original and effective, new and fit for some purpose<sup class="cite">[<a href="#ref-6" id="cite-6" data-ref="6" aria-label="Reference 6: Runco, M. A., &amp; Jaeger, G. J. (2012). The standard definition of creativity. Creativity Research Journal, 24(1), 92–96.">6</a>]</sup>. A random string is new and useless; a correct sum is useful and not new.

The philosopher Margaret Boden, who has written about creativity and AI since the 1980s, adds surprise: "Creativity is the ability to come up with ideas or artefacts that are new, surprising, and valuable"<sup class="cite">[<a href="#ref-7" id="cite-7" data-ref="7" aria-label="Reference 7: Boden, M. A. (2007). Creativity in a nutshell. Think, 5(15), 83–96; from The Creative Mind (2nd ed., 2004).">7</a>]</sup>.

Boden makes two distinctions that clear up much of the argument about machines. The first is between ideas that are new to the person who has them, **P-creativity**, and ideas that are new to the world, "for the first time in human history", **H-creativity**.

A child who works out for herself that there is no largest prime number is P-creative; Euclid, who proved it first, was H-creative. The second distinction is between three ways of being creative at all:

<img class="tfs-figure tfs-light" src="/assets/posts/creative-ai/boden-light.png" width="760" alt="Margaret Boden&#x27;s three kinds of creativity drawn on a space of ideas. Combinational: two familiar ideas from distant parts of the space are joined. Exploratory: a path inside the space&#x27;s rules reaches a place nobody visited. Transformational: one rule of the space changes, and ideas that were impossible become possible" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/creative-ai/boden-dark.png" width="760" alt="Margaret Boden&#x27;s three kinds of creativity drawn on a space of ideas. Combinational: two familiar ideas from distant parts of the space are joined. Exploratory: a path inside the space&#x27;s rules reaches a place nobody visited. Transformational: one rule of the space changes, and ideas that were impossible become possible" loading="lazy">

- **Combinational** creativity is "making unfamiliar combinations of familiar ideas": a metaphor, a pun, a collage.
- **Exploratory** creativity works inside a conceptual space, the set of possibilities that a style or a set of rules allows, and reaches places in it nobody had been.
- **Transformational** creativity changes the space itself, so that "thoughts are now possible which previously… were literally inconceivable."

Move 37 was exploratory. The rules of Go did not change; the move was always legal. What changed was that a player had looked, and evaluated, where generations of professionals had learned not to look.

## Memory is the raw material

If people created from nothing, imagination would not depend on memory. It depends on it heavily.

Daniel Schacter and Donna Rose Addis argued in 2007 that memory is "fundamentally constructive, rather than reproductive", and that its constructive nature is what makes imagination possible: imagining the future "requires a system that can draw on the past in a manner that flexibly extracts and recombines elements of previous experiences"<sup class="cite">[<a href="#ref-8" id="cite-8" data-ref="8" aria-label="Reference 8: Schacter, D. L., &amp; Addis, D. R. (2007). The cognitive neuroscience of constructive memory. Philosophical Transactions of the Royal Society B, 362, 773–786.">8</a>]</sup>.

The same year Demis Hassabis, then a neuroscientist and later a founder of DeepMind, tested the idea on five patients whose hippocampi, the brain's structure for forming new memories, were damaged on both sides<sup class="cite">[<a href="#ref-9" id="cite-9" data-ref="9" aria-label="Reference 9: Hassabis, D., Kumaran, D., Vann, S. D., &amp; Maguire, E. A. (2007). Patients with hippocampal amnesia cannot imagine new experiences. PNAS, 104(5), 1726–1731.">9</a>]</sup>. They were asked to imagine new scenes from short cues and describe them, and told not to recall real memories.

One patient, asked to imagine lying on a beach, said: "As for seeing I can't really, apart from just sky. I can hear the sound of seagulls and of the sea… the only thing I can see is blue…"

<img class="tfs-figure tfs-light" src="/assets/posts/creative-ai/hassabis-light.png" width="760" alt="Imagined scenes of five patients with hippocampal damage against ten controls: richness 27.54 against 45.06, spatial coherence 0.10 against 3.68, the person&#x27;s own quality rating 3.88 against 7.13" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/creative-ai/hassabis-dark.png" width="760" alt="Imagined scenes of five patients with hippocampal damage against ten controls: richness 27.54 against 45.06, spatial coherence 0.10 against 3.68, the person&#x27;s own quality rating 3.88 against 7.13" loading="lazy">

The patients' imagined scenes were poorer and, above all, fell apart in space: "fragmented images in the absence of a holistic representation of the environmental setting." The result has a published dispute behind it.

One of the five patients was unimpaired, and Larry Squire and colleagues found hippocampal patients who could imagine the future normally with a different method<sup class="cite">[<a href="#ref-10" id="cite-10" data-ref="10" aria-label="Reference 10: Squire, L. R., et al. (2010). Role of the hippocampus in remembering the past and imagining the future. PNAS, 107(44), 19044–19048.">10</a>]</sup>; later studies with damage restricted to the hippocampus found the impairment again<sup class="cite">[<a href="#ref-11" id="cite-11" data-ref="11" aria-label="Reference 11: Race, E., Keane, M. M., &amp; Verfaellie, M. (2011). Medial temporal lobe damage causes deficits in episodic memory and episodic future thinking not attributable to deficits in narrative construction. Journal of Neuroscience, 31(28), 10262–10269.">11</a>]</sup>.

The weight of the evidence is that the system which stores experiences is also the one that assembles new scenes out of them.

The same system shows up when healthy people create. Melissa Ellamil and colleagues had people design book covers in a scanner, switching between generating ideas and evaluating them. "Creative generation was associated with preferential recruitment of medial temporal lobe regions", the memory system, while evaluation recruited executive and default network regions<sup class="cite">[<a href="#ref-12" id="cite-12" data-ref="12" aria-label="Reference 12: Ellamil, M., Dobson, C., Beeman, M., &amp; Christoff, K. (2012). Evaluative and generative modes of thought during the creative process. NeuroImage, 59(2), 1783–1794.">12</a>]</sup>.

## Where recombination runs out

A good test of the recombination view is colour. Can you imagine a colour you have never seen?

John Locke told of a blind man who said he now understood scarlet: it "was like the sound of a trumpet." Locke told the story to show that the man had *not* understood the word<sup class="cite">[<a href="#ref-13" id="cite-13" data-ref="13" aria-label="Reference 13: Locke, J. (1690). An Essay Concerning Human Understanding, III.iv.11.">13</a>]</sup>.

Modern experiments find something more interesting. Congenitally blind adults understand colour much as sighted people do, and they learned it from language: "blind and sighted adults share causal understanding of color, despite not always agreeing about arbitrary color facts".

They are less likely to say that a banana is yellow, but they know what colours are for and how they relate. "Living among people who talk about color is sufficient for color understanding"<sup class="cite">[<a href="#ref-14" id="cite-14" data-ref="14" aria-label="Reference 14: Kim, J. S., Aheimer, B., Montané Manrara, V., &amp; Bedny, M. (2021). Shared understanding of color among sighted and blind adults. PNAS, 118(33), e2020192118.">14</a>]</sup>.

Knowing about a colour is not seeing it, though, and in April 2025 five people saw a colour nobody had seen before.

A team at Berkeley built a system that aims laser microdoses at individually mapped cones in the retina and tried to stimulate only the M cones, the ones most sensitive to green. Natural light never does that: every wavelength that excites M cones also excites L cones.

The subjects described the result, which the authors named olo, as "blue-green of unprecedented saturation"<sup class="cite">[<a href="#ref-15" id="cite-15" data-ref="15" aria-label="Reference 15: Fong, J., et al. (2025). Novel color via stimulation of individual photoreceptors at population scale. Science Advances, 11(16), eadu1052.">15</a>]</sup>. To match it with ordinary light, they had to desaturate olo by adding white, "unequivocal proof that olo lies beyond the gamut."

<img class="tfs-figure tfs-light" src="/assets/posts/creative-ai/olo-light.png" width="760" alt="The MacLeod-Boynton chromaticity diagram. Every colour a light can produce lies inside the spectral locus, which runs from l of about 0.6 at short wavelengths to nearly 1 at long ones; a signal from the M cones alone sits at l = 0, far outside it" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/creative-ai/olo-dark.png" width="760" alt="The MacLeod-Boynton chromaticity diagram. Every colour a light can produce lies inside the spectral locus, which runs from l of about 0.6 at short wavelengths to nearly 1 at long ones; a signal from the M cones alone sits at l = 0, far outside it" loading="lazy">

Olo is not a new hue but a saturation no light can produce; the authors call it "a partial expansion of colorspace." Nobody could have imagined it in advance, and nobody did. It arrived through a new input.

There is even a hint that the input then extends imagination: after seeing "forbidden" colours such as reddish-green under special viewing conditions, "two subjects noted independently that reddish-green and yellowish-blue colors could now be imagined"<sup class="cite">[<a href="#ref-16" id="cite-16" data-ref="16" aria-label="Reference 16: Billock, V. A., Gleason, G. A., &amp; Tsou, B. H. (2001). Perception of forbidden colors in retinally stabilized equiluminant images. JOSA A, 18(10), 2398–2403.">16</a>]</sup>, although whether such colours are really seen is disputed<sup class="cite">[<a href="#ref-17" id="cite-17" data-ref="17" aria-label="Reference 17: Hsieh, P.-J., &amp; Tse, P. U. (2006). Illusory color mixing upon perceptual fading and filling-in does not result in &#x27;forbidden colors&#x27;. Vision Research, 46(14), 2251–2258.">17</a>]</sup>.

Image generators show the same boundary in their own way. They combine concepts they have learned, with what the DALL·E authors called "a rudimentary ability to compose unusual concepts"<sup class="cite">[<a href="#ref-18" id="cite-18" data-ref="18" aria-label="Reference 18: Ramesh, A., et al. (2021). Zero-shot text-to-image generation. ICML.">18</a>]</sup>, and they render a concept well only if they saw it often: across models including Stable Diffusion, linear improvements on a concept took "exponentially more data" about it<sup class="cite">[<a href="#ref-19" id="cite-19" data-ref="19" aria-label="Reference 19: Udandarao, V., et al. (2024). No &quot;zero-shot&quot; without exponential data: pretraining concept frequency determines multimodal model performance. arXiv:2404.04125.">19</a>]</sup>.

A genuinely new object can be taught to them from "only 3-5 images"<sup class="cite">[<a href="#ref-20" id="cite-20" data-ref="20" aria-label="Reference 20: Gal, R., et al. (2022). An image is worth one word: personalizing text-to-image generation using textual inversion. arXiv:2208.01618.">20</a>]</sup>: a new input, again.

## Generate, then select

In 1960 the psychologist Donald Campbell proposed that creative thought works like evolution: **blind variation and selective retention**, many candidates produced without foresight and a few kept<sup class="cite">[<a href="#ref-21" id="cite-21" data-ref="21" aria-label="Reference 21: Campbell, D. T. (1960). Blind variation and selective retention in creative thought as in other knowledge processes. Psychological Review, 67(6), 380–400.">21</a>]</sup>.

Dean Keith Simonton has defended the theory for decades<sup class="cite">[<a href="#ref-22" id="cite-22" data-ref="22" aria-label="Reference 22: Simonton, D. K. (2011). Creativity and discovery as blind variation: Campbell&#x27;s (1960) BVSR model after the half-century mark. Review of General Psychology, 15(2), 158–174.">22</a>]</sup>, others argue it does not fit how ideas actually develop<sup class="cite">[<a href="#ref-23" id="cite-23" data-ref="23" aria-label="Reference 23: Gabora, L. (2011). An analysis of the blind variation and selective retention theory of creativity. Creativity Research Journal, 23(2), 155–165.">23</a>]</sup>, and after more than sixty years "without any resolution of the controversy"<sup class="cite">[<a href="#ref-24" id="cite-24" data-ref="24" aria-label="Reference 24: Simonton, D. K. (2013). Creative thought as blind variation and selective retention: why creativity is inversely related to sightedness. Journal of Theoretical and Philosophical Psychology, 33(4), 253–266.">24</a>]</sup> the core distinction has stuck: producing candidates and choosing among them are different jobs.

The brain seems to split them between networks that usually work against each other. The **default mode network** is active when attention turns inward, to memories, plans and daydreams. The **executive control network** holds a goal and checks progress toward it.

The **salience network** decides which of the two the moment calls for. In creative thought, Roger Beaty and colleagues write, "the default and executive control networks, which can show an antagonistic relation, tend to cooperate"<sup class="cite">[<a href="#ref-25" id="cite-25" data-ref="25" aria-label="Reference 25: Beaty, R. E., Benedek, M., Silvia, P. J., &amp; Schacter, D. L. (2016). Creative cognition and brain network dynamics. Trends in Cognitive Sciences, 20(2), 87–95.">25</a>]</sup>.

<img class="tfs-figure tfs-light" src="/assets/posts/creative-ai/networks-light.png" width="760" alt="The left hemisphere seen from the side and from the middle, with three of Yeo&#x27;s seven networks coloured: the default mode network on the medial prefrontal and posterior cingulate cortex, the angular gyrus and the lateral temporal lobe; the executive control network on the lateral prefrontal and parietal cortex; the salience network on the insula and the anterior cingulate" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/creative-ai/networks-dark.png" width="760" alt="The left hemisphere seen from the side and from the middle, with three of Yeo&#x27;s seven networks coloured: the default mode network on the medial prefrontal and posterior cingulate cortex, the angular gyrus and the lateral temporal lobe; the executive control network on the lateral prefrontal and parietal cortex; the salience network on the insula and the anterior cingulate" loading="lazy">

In 2018 the same group scanned 163 people while they thought of unusual uses for objects and found a pattern of connections among these three networks that predicted the originality of the ideas of people the model had not seen, with a correlation of 0.30<sup class="cite">[<a href="#ref-26" id="cite-26" data-ref="26" aria-label="Reference 26: Beaty, R. E., et al. (2018). Robust prediction of individual creative ability from brain functional connectivity. PNAS, 115(5), 1087–1092.">26</a>]</sup>.

It held in samples from Austria and China, at 0.35, 0.28 and 0.13. These are modest numbers: brain connectivity tells you something about a person's creativity, not much.

The best-known finding about letting go is also less simple than its retellings. Charles Limb and Allen Braun scanned six jazz pianists improvising and playing memorized music.

Improvisation showed "extensive deactivation of dorsolateral prefrontal and lateral orbital regions with focal activation of the medial prefrontal (frontal polar) cortex"<sup class="cite">[<a href="#ref-27" id="cite-27" data-ref="27" aria-label="Reference 27: Limb, C. J., &amp; Braun, A. R. (2008). Neural substrates of spontaneous musical performance: an fMRI study of jazz improvisation. PLoS ONE, 3(2), e1679.">27</a>]</sup>: parts of the brain that monitor behaviour went quiet, a part linked to self-expression switched on.

Later studies found that the involvement of the dorsolateral prefrontal cortex depends on the task and on expertise<sup class="cite">[<a href="#ref-28" id="cite-28" data-ref="28" aria-label="Reference 28: Pinho, A. L., Ullén, F., Castelo-Branco, M., Fransson, P., &amp; de Manzano, Ö. (2016). Addressing a paradox: dual strategies for creative performance in introspective and extrospective networks. Cerebral Cortex, 26(7), 3052–3063.">28</a>]</sup>. Creativity is not switching control off; it is choosing when control applies.

Machines have the same two jobs. AlphaGo's networks were trained first on games between strong human players, then by playing against themselves; a search over continuations decided which of the proposed moves was worth playing<sup class="cite">[<a href="#ref-29" id="cite-29" data-ref="29" aria-label="Reference 29: Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. Nature, 529, 484–489.">29</a>]</sup>.

Move 37 was a move that, by AlphaGo's own estimate, a human would almost never play, and that its search judged good.

A language model has the variation half built in, in its sampling, and the selection half only if somebody adds it. Here is a small language model asked eight times, at three temperatures, how else a brick can be used:

<img class="tfs-figure tfs-light" src="/assets/posts/creative-ai/variation-light.png" width="760" alt="Eight continuations of &quot;Besides building walls, a brick can be used as a&quot; at each of three temperatures. at 0.3: 3 different first words, mean log-probability -1.0 per token; at 1.0: 8 different first words, mean log-probability -2.2 per token; at 1.8: 8 different first words, mean log-probability -8.7 per token. Low temperature repeats one idea, high temperature produces new and mostly broken text" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/creative-ai/variation-dark.png" width="760" alt="Eight continuations of &quot;Besides building walls, a brick can be used as a&quot; at each of three temperatures. at 0.3: 3 different first words, mean log-probability -1.0 per token; at 1.0: 8 different first words, mean log-probability -2.2 per token; at 1.8: 8 different first words, mean log-probability -8.7 per token. Low temperature repeats one idea, high temperature produces new and mostly broken text" loading="lazy">

At a low temperature the model repeats one idea. At a medium one it produces eight different, sensible ones. At a high one it produces eight new ones, most of them broken, and the model itself rates them as very unlikely text.

Variation on its own is noise. Everything that makes it useful, a person choosing, a reward model trained on human preferences, a search that tries continuations, is selection.

## Measuring machine creativity

Psychologists measure divergent thinking with tasks like the Alternate Uses Task: list unusual uses for a brick, a rope, a box. In 2023 Mika Koivisto and Simone Grassini gave it to 256 people and to three chatbots, ChatGPT-3.5, ChatGPT-4 and Copy.Ai, eleven sessions each, with four objects: rope, box, pencil and candle.

Six judges who did not know which answers came from machines rated every idea<sup class="cite">[<a href="#ref-30" id="cite-30" data-ref="30" aria-label="Reference 30: Koivisto, M., &amp; Grassini, S. (2023). Best humans still outperform artificial intelligence in a creative divergent thinking task. Scientific Reports, 13, 13601. Data: OSF.">30</a>]</sup>. Their data are open, so here is every person and every session:

<img class="tfs-figure tfs-light" src="/assets/posts/creative-ai/aut-light.png" width="760" alt="Ratings of ideas in the Alternate Uses Task, per person or chatbot session. Mean rating: humans 2.47, Copy.Ai 2.67, ChatGPT-3.5 2.70, ChatGPT-4 3.35; the humans spread from the lowest scores to near the top" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/creative-ai/aut-dark.png" width="760" alt="Ratings of ideas in the Alternate Uses Task, per person or chatbot session. Mean rating: humans 2.47, Copy.Ai 2.67, ChatGPT-3.5 2.70, ChatGPT-4 3.35; the humans spread from the lowest scores to near the top" loading="lazy">

"On average, the AI chatbots outperformed human participants", the authors write, and "the best human ideas still matched or exceed those of the chatbots". Both are visible above.

The mean of 256 people sits below every chatbot, and only 7 people did better on average than ChatGPT-4. But the people spread from the bottom of the scale to near the top, and the best person's best idea beat 31 of the 33 chatbot sessions.

The authors add a caution about where the chatbots' ideas come from: "It remains possible that they simply retrieve ideas that exist in their database."

A larger study found GPT-4 "robustly more creative along each divergent thinking measurement" than 151 people, while noting that it scored originality automatically and did not measure usefulness at all, and that "creative potential is not a guarantee for creative achievement"<sup class="cite">[<a href="#ref-31" id="cite-31" data-ref="31" aria-label="Reference 31: Hubert, K. F., Awa, K. N., &amp; Zabelina, D. L. (2024). The current state of artificial intelligence generative language models is more creative than humans on divergent thinking tasks. Scientific Reports, 14, 3440.">31</a>]</sup>.

What happens when people create *with* the machine? Anil Doshi and Oliver Hauser asked 293 people to write short stories, some with no help and some with story ideas from GPT-4, and had 600 readers rate them<sup class="cite">[<a href="#ref-32" id="cite-32" data-ref="32" aria-label="Reference 32: Doshi, A. R., &amp; Hauser, O. P. (2024). Generative AI enhances individual creativity but reduces the collective diversity of novel content. Science Advances, 10, eadn5290.">32</a>]</sup>:

<img class="tfs-figure tfs-light" src="/assets/posts/creative-ai/doshi-light.png" width="760" alt="With GPT-4 story ideas, novelty rose by 5.4% (one idea) and 8.1% (up to five), usefulness by 3.7% and 9.0%, and the similarity of the stories to each other by 10.7% and 8.9% of its range without AI" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/creative-ai/doshi-dark.png" width="760" alt="With GPT-4 story ideas, novelty rose by 5.4% (one idea) and 8.1% (up to five), usefulness by 3.7% and 9.0%, and the similarity of the stories to each other by 10.7% and 8.9% of its range without AI" loading="lazy">

The stories written with AI ideas were judged more novel and more useful, especially those by the writers who were least creative on their own. They were also more similar to each other. "These results point to an increase in individual creativity at the risk of losing collective novelty."

At the scale of a whole culture, the same pull toward the middle has a sharper version: models trained on the output of earlier models lose the rare cases first, "tails of the original content distribution disappear"<sup class="cite">[<a href="#ref-33" id="cite-33" data-ref="33" aria-label="Reference 33: Shumailov, I., et al. (2024). AI models collapse when trained on recursively generated data. Nature, 631, 755–759.">33</a>]</sup>, unless real data keeps being added<sup class="cite">[<a href="#ref-34" id="cite-34" data-ref="34" aria-label="Reference 34: Gerstgrasser, M., et al. (2024). Is model collapse inevitable? Breaking the curse of recursion by accumulating real and synthetic data. arXiv:2404.01413.">34</a>]</sup>.

## Bodies and words

A common objection is that machines lack the bodily experience that grounds human ideas. Some of the evidence quoted for it is weaker than it sounds.

In a well-known study, reading the verbs "lick", "pick" and "kick" activated areas along the motor strip next to those that move the tongue, fingers and feet<sup class="cite">[<a href="#ref-35" id="cite-35" data-ref="35" aria-label="Reference 35: Hauk, O., Johnsrude, I., &amp; Pulvermüller, F. (2004). Somatotopic representation of action words in human motor and premotor cortex. Neuron, 41(2), 301–307.">35</a>]</sup>.

That was fourteen people in a scanner; a later study found "no strong evidence for a somatotopic organisation of action word meaning representations"<sup class="cite">[<a href="#ref-36" id="cite-36" data-ref="36" aria-label="Reference 36: Postle, N., McMahon, K. L., Ashton, R., Meredith, M., &amp; de Zubicaray, G. I. (2008). Action word meaning representations in cytoarchitectonically defined primary and premotor cortices. NeuroImage, 43(3), 634–644.">36</a>]</sup>, and the activation can be explained without making it part of the meaning<sup class="cite">[<a href="#ref-37" id="cite-37" data-ref="37" aria-label="Reference 37: Mahon, B. Z., &amp; Caramazza, A. (2008). A critical look at the embodied cognition hypothesis. Journal of Physiology–Paris, 102, 59–70.">37</a>]</sup>.

And words carry more of the world than the objection assumes. The blind adults above learned the structure of colour from talk.

GPT-4's judgments of similarity between colours recover "the color wheel", agreeing with human judgments at r = 0.79 when it is given only the colour words<sup class="cite">[<a href="#ref-38" id="cite-38" data-ref="38" aria-label="Reference 38: Marjieh, R., et al. (2024). Large language models predict human sensory judgments across six modalities. Scientific Reports, 14, 21445.">38</a>]</sup>; in [another post](/blog/2025/minds-as-computers/) I measured how much of the same structure sits inside a small open model.

What words cannot carry is olo. The difference between people and models here is not that one recombines and the other does not. It is in the inputs: a person's stream of experience keeps bringing things no text describes.

## Who does the selecting

If creativity is variation plus selection, the interesting question about AI art is who selects. In 2022 Jason Allen won the digital arts category of the Colorado State Fair with an image made in Midjourney, after about 80 hours and more than 900 renderings<sup class="cite">[<a href="#ref-39" id="cite-39" data-ref="39" aria-label="Reference 39: Smithsonian, 2022. smithsonianmag.com.">39</a>]</sup>.

The US Copyright Office refused to register it as submitted because it "contains more than a de minimis amount of content generated by artificial intelligence"<sup class="cite">[<a href="#ref-40" id="cite-40" data-ref="40" aria-label="Reference 40: U.S. Copyright Office Review Board (2023). Théâtre D&#x27;opéra Spatial.">40</a>]</sup>, although Allen had revised his prompts at least 624 times.

In 2025 a federal appeals court confirmed that copyright requires "all eligible work to be authored in the first instance by a human being", while noting that this "does not prohibit copyrighting work that was made by or with the assistance of artificial intelligence"<sup class="cite">[<a href="#ref-41" id="cite-41" data-ref="41" aria-label="Reference 41: Thaler v. Perlmutter, No. 23-5233 (D.C. Cir. 2025); certiorari denied, No. 25-449 (2026).">41</a>]</sup>.

The Supreme Court declined to hear the case in March 2026. The law, in effect, puts the author where the selection is.

## After move 37

Three days after move 37, in the fourth game, Lee Sedol played a move of his own, a wedge at move 78 that AlphaGo had also put at one in ten thousand<sup class="cite">[<a href="#ref-2" id="cite-42" data-ref="2" aria-label="Reference 2: Hassabis, D. (2016). What we learned in Seoul with AlphaGo. Google.">2</a>]</sup>.

Commentators called it the divine move. Analysis later showed it does not actually work; AlphaGo's reply was the real mistake, and Lee won the game, the only one a human took from AlphaGo in that match.

Then something happened to the players. Minkyu Shin and colleagues analysed 5.8 million moves by professional Go players from 1950 to 2021. For 66 years the novelty of human play stayed flat.

After superhuman Go programs appeared, "novel decisions (i.e., previously unobserved moves) occurred more frequently and became associated with higher decision quality"<sup class="cite">[<a href="#ref-42" id="cite-43" data-ref="42" aria-label="Reference 42: Shin, M., Kim, J., van Opheusden, B., &amp; Griffiths, T. L. (2023). Superhuman artificial intelligence can improve human decision-making by increasing novelty. PNAS, 120(12), e2214840120.">42</a>]</sup>. The players were not copying the machines' moves, the authors argue, or not only. They were searching a space that had turned out to be larger than they had thought.

## References

1. {:#ref-1} Metz, C. (2016). [The sadness and beauty of watching Google's AI play Go](https://www.wired.com/2016/03/sadness-beauty-watching-googles-ai-play-go/). *Wired*. <a class="ref-back" href="#cite-1" aria-label="Back to the text">&#8617;</a>
2. {:#ref-2} Hassabis, D. (2016). [What we learned in Seoul with AlphaGo](https://blog.google/technology/ai/what-we-learned-in-seoul-with-alphago/). Google. <a class="ref-back" href="#cite-2" aria-label="Back to the text">&#8617;</a>
3. {:#ref-3} DeepMind. [AlphaGo](https://deepmind.google/research/breakthroughs/alphago/). Google DeepMind research page. <a class="ref-back" href="#cite-3" aria-label="Back to the text">&#8617;</a>
4. {:#ref-4} Lovelace, A. A. (1843). [Notes on the Sketch of the Analytical Engine by L. F. Menabrea](https://www.fourmilab.ch/babbage/sketch.html). *Scientific Memoirs*, 3. <a class="ref-back" href="#cite-4" aria-label="Back to the text">&#8617;</a>
5. {:#ref-5} Turing, A. M. (1950). [Computing machinery and intelligence](https://doi.org/10.1093/mind/LIX.236.433). *Mind*, 59(236), 433–460. <a class="ref-back" href="#cite-5" aria-label="Back to the text">&#8617;</a>
6. {:#ref-6} Runco, M. A., & Jaeger, G. J. (2012). [The standard definition of creativity](https://doi.org/10.1080/10400419.2012.650092). *Creativity Research Journal*, 24(1), 92–96. <a class="ref-back" href="#cite-6" aria-label="Back to the text">&#8617;</a>
7. {:#ref-7} Boden, M. A. (2007). [Creativity in a nutshell](https://doi.org/10.1017/S147717560000230X). *Think*, 5(15), 83–96; from *The Creative Mind* (2nd ed., 2004). <a class="ref-back" href="#cite-7" aria-label="Back to the text">&#8617;</a>
8. {:#ref-8} Schacter, D. L., & Addis, D. R. (2007). [The cognitive neuroscience of constructive memory](https://doi.org/10.1098/rstb.2007.2087). *Philosophical Transactions of the Royal Society B*, 362, 773–786. <a class="ref-back" href="#cite-8" aria-label="Back to the text">&#8617;</a>
9. {:#ref-9} Hassabis, D., Kumaran, D., Vann, S. D., & Maguire, E. A. (2007). [Patients with hippocampal amnesia cannot imagine new experiences](https://doi.org/10.1073/pnas.0610561104). *PNAS*, 104(5), 1726–1731. <a class="ref-back" href="#cite-9" aria-label="Back to the text">&#8617;</a>
10. {:#ref-10} Squire, L. R., et al. (2010). [Role of the hippocampus in remembering the past and imagining the future](https://doi.org/10.1073/pnas.1014391107). *PNAS*, 107(44), 19044–19048. <a class="ref-back" href="#cite-10" aria-label="Back to the text">&#8617;</a>
11. {:#ref-11} Race, E., Keane, M. M., & Verfaellie, M. (2011). [Medial temporal lobe damage causes deficits in episodic memory and episodic future thinking not attributable to deficits in narrative construction](https://doi.org/10.1523/JNEUROSCI.1145-11.2011). *Journal of Neuroscience*, 31(28), 10262–10269. <a class="ref-back" href="#cite-11" aria-label="Back to the text">&#8617;</a>
12. {:#ref-12} Ellamil, M., Dobson, C., Beeman, M., & Christoff, K. (2012). [Evaluative and generative modes of thought during the creative process](https://doi.org/10.1016/j.neuroimage.2011.08.008). *NeuroImage*, 59(2), 1783–1794. <a class="ref-back" href="#cite-12" aria-label="Back to the text">&#8617;</a>
13. {:#ref-13} Locke, J. (1690). [*An Essay Concerning Human Understanding*](https://www.gutenberg.org/ebooks/10616), III.iv.11. <a class="ref-back" href="#cite-13" aria-label="Back to the text">&#8617;</a>
14. {:#ref-14} Kim, J. S., Aheimer, B., Montané Manrara, V., & Bedny, M. (2021). [Shared understanding of color among sighted and blind adults](https://doi.org/10.1073/pnas.2020192118). *PNAS*, 118(33), e2020192118. <a class="ref-back" href="#cite-14" aria-label="Back to the text">&#8617;</a>
15. {:#ref-15} Fong, J., et al. (2025). [Novel color via stimulation of individual photoreceptors at population scale](https://doi.org/10.1126/sciadv.adu1052). *Science Advances*, 11(16), eadu1052. <a class="ref-back" href="#cite-15" aria-label="Back to the text">&#8617;</a>
16. {:#ref-16} Billock, V. A., Gleason, G. A., & Tsou, B. H. (2001). [Perception of forbidden colors in retinally stabilized equiluminant images](https://doi.org/10.1364/josaa.18.002398). *JOSA A*, 18(10), 2398–2403. <a class="ref-back" href="#cite-16" aria-label="Back to the text">&#8617;</a>
17. {:#ref-17} Hsieh, P.-J., & Tse, P. U. (2006). [Illusory color mixing upon perceptual fading and filling-in does not result in 'forbidden colors'](https://doi.org/10.1016/j.visres.2005.11.030). *Vision Research*, 46(14), 2251–2258. <a class="ref-back" href="#cite-17" aria-label="Back to the text">&#8617;</a>
18. {:#ref-18} Ramesh, A., et al. (2021). [Zero-shot text-to-image generation](https://arxiv.org/abs/2102.12092). *ICML*. <a class="ref-back" href="#cite-18" aria-label="Back to the text">&#8617;</a>
19. {:#ref-19} Udandarao, V., et al. (2024). [No "zero-shot" without exponential data: pretraining concept frequency determines multimodal model performance](https://arxiv.org/abs/2404.04125). arXiv:2404.04125. <a class="ref-back" href="#cite-19" aria-label="Back to the text">&#8617;</a>
20. {:#ref-20} Gal, R., et al. (2022). [An image is worth one word: personalizing text-to-image generation using textual inversion](https://arxiv.org/abs/2208.01618). arXiv:2208.01618. <a class="ref-back" href="#cite-20" aria-label="Back to the text">&#8617;</a>
21. {:#ref-21} Campbell, D. T. (1960). [Blind variation and selective retention in creative thought as in other knowledge processes](https://doi.org/10.1037/h0040373). *Psychological Review*, 67(6), 380–400. <a class="ref-back" href="#cite-21" aria-label="Back to the text">&#8617;</a>
22. {:#ref-22} Simonton, D. K. (2011). [Creativity and discovery as blind variation: Campbell's (1960) BVSR model after the half-century mark](https://doi.org/10.1037/a0022912). *Review of General Psychology*, 15(2), 158–174. <a class="ref-back" href="#cite-22" aria-label="Back to the text">&#8617;</a>
23. {:#ref-23} Gabora, L. (2011). [An analysis of the blind variation and selective retention theory of creativity](https://doi.org/10.1080/10400419.2011.571187). *Creativity Research Journal*, 23(2), 155–165. <a class="ref-back" href="#cite-23" aria-label="Back to the text">&#8617;</a>
24. {:#ref-24} Simonton, D. K. (2013). [Creative thought as blind variation and selective retention: why creativity is inversely related to sightedness](https://doi.org/10.1037/a0030705). *Journal of Theoretical and Philosophical Psychology*, 33(4), 253–266. <a class="ref-back" href="#cite-24" aria-label="Back to the text">&#8617;</a>
25. {:#ref-25} Beaty, R. E., Benedek, M., Silvia, P. J., & Schacter, D. L. (2016). [Creative cognition and brain network dynamics](https://doi.org/10.1016/j.tics.2015.10.004). *Trends in Cognitive Sciences*, 20(2), 87–95. <a class="ref-back" href="#cite-25" aria-label="Back to the text">&#8617;</a>
26. {:#ref-26} Beaty, R. E., et al. (2018). [Robust prediction of individual creative ability from brain functional connectivity](https://doi.org/10.1073/pnas.1713532115). *PNAS*, 115(5), 1087–1092. <a class="ref-back" href="#cite-26" aria-label="Back to the text">&#8617;</a>
27. {:#ref-27} Limb, C. J., & Braun, A. R. (2008). [Neural substrates of spontaneous musical performance: an fMRI study of jazz improvisation](https://doi.org/10.1371/journal.pone.0001679). *PLoS ONE*, 3(2), e1679. <a class="ref-back" href="#cite-27" aria-label="Back to the text">&#8617;</a>
28. {:#ref-28} Pinho, A. L., Ullén, F., Castelo-Branco, M., Fransson, P., & de Manzano, Ö. (2016). [Addressing a paradox: dual strategies for creative performance in introspective and extrospective networks](https://doi.org/10.1093/cercor/bhv130). *Cerebral Cortex*, 26(7), 3052–3063. <a class="ref-back" href="#cite-28" aria-label="Back to the text">&#8617;</a>
29. {:#ref-29} Silver, D., et al. (2016). [Mastering the game of Go with deep neural networks and tree search](https://doi.org/10.1038/nature16961). *Nature*, 529, 484–489. <a class="ref-back" href="#cite-29" aria-label="Back to the text">&#8617;</a>
30. {:#ref-30} Koivisto, M., & Grassini, S. (2023). [Best humans still outperform artificial intelligence in a creative divergent thinking task](https://doi.org/10.1038/s41598-023-40858-3). *Scientific Reports*, 13, 13601. Data: [OSF](https://osf.io/qdz3n/). <a class="ref-back" href="#cite-30" aria-label="Back to the text">&#8617;</a>
31. {:#ref-31} Hubert, K. F., Awa, K. N., & Zabelina, D. L. (2024). [The current state of artificial intelligence generative language models is more creative than humans on divergent thinking tasks](https://doi.org/10.1038/s41598-024-53303-w). *Scientific Reports*, 14, 3440. <a class="ref-back" href="#cite-31" aria-label="Back to the text">&#8617;</a>
32. {:#ref-32} Doshi, A. R., & Hauser, O. P. (2024). [Generative AI enhances individual creativity but reduces the collective diversity of novel content](https://doi.org/10.1126/sciadv.adn5290). *Science Advances*, 10, eadn5290. <a class="ref-back" href="#cite-32" aria-label="Back to the text">&#8617;</a>
33. {:#ref-33} Shumailov, I., et al. (2024). [AI models collapse when trained on recursively generated data](https://doi.org/10.1038/s41586-024-07566-y). *Nature*, 631, 755–759. <a class="ref-back" href="#cite-33" aria-label="Back to the text">&#8617;</a>
34. {:#ref-34} Gerstgrasser, M., et al. (2024). [Is model collapse inevitable? Breaking the curse of recursion by accumulating real and synthetic data](https://arxiv.org/abs/2404.01413). arXiv:2404.01413. <a class="ref-back" href="#cite-34" aria-label="Back to the text">&#8617;</a>
35. {:#ref-35} Hauk, O., Johnsrude, I., & Pulvermüller, F. (2004). [Somatotopic representation of action words in human motor and premotor cortex](https://doi.org/10.1016/s0896-6273%2803%2900838-9). *Neuron*, 41(2), 301–307. <a class="ref-back" href="#cite-35" aria-label="Back to the text">&#8617;</a>
36. {:#ref-36} Postle, N., McMahon, K. L., Ashton, R., Meredith, M., & de Zubicaray, G. I. (2008). [Action word meaning representations in cytoarchitectonically defined primary and premotor cortices](https://doi.org/10.1016/j.neuroimage.2008.08.006). *NeuroImage*, 43(3), 634–644. <a class="ref-back" href="#cite-36" aria-label="Back to the text">&#8617;</a>
37. {:#ref-37} Mahon, B. Z., & Caramazza, A. (2008). [A critical look at the embodied cognition hypothesis](https://doi.org/10.1016/j.jphysparis.2008.03.004). *Journal of Physiology–Paris*, 102, 59–70. <a class="ref-back" href="#cite-37" aria-label="Back to the text">&#8617;</a>
38. {:#ref-38} Marjieh, R., et al. (2024). [Large language models predict human sensory judgments across six modalities](https://doi.org/10.1038/s41598-024-72071-1). *Scientific Reports*, 14, 21445. <a class="ref-back" href="#cite-38" aria-label="Back to the text">&#8617;</a>
39. {:#ref-39} Smithsonian, 2022. [smithsonianmag.com](https://www.smithsonianmag.com/smart-news/artificial-intelligence-art-wins-colorado-state-fair-180980703/). <a class="ref-back" href="#cite-39" aria-label="Back to the text">&#8617;</a>
40. {:#ref-40} U.S. Copyright Office Review Board (2023). [Théâtre D'opéra Spatial](https://www.copyright.gov/rulings-filings/review-board/docs/Theatre-Dopera-Spatial.pdf). <a class="ref-back" href="#cite-40" aria-label="Back to the text">&#8617;</a>
41. {:#ref-41} Thaler v. Perlmutter, No. 23-5233 ([D.C. Cir. 2025](https://media.cadc.uscourts.gov/opinions/docs/2025/03/23-5233.pdf)); certiorari denied, No. 25-449 ([2026](https://www.supremecourt.gov/docket/docketfiles/html/public/25-449.html)). <a class="ref-back" href="#cite-41" aria-label="Back to the text">&#8617;</a>
42. {:#ref-42} Shin, M., Kim, J., van Opheusden, B., & Griffiths, T. L. (2023). [Superhuman artificial intelligence can improve human decision-making by increasing novelty](https://doi.org/10.1073/pnas.2214840120). *PNAS*, 120(12), e2214840120. <a class="ref-back" href="#cite-43" aria-label="Back to the text">&#8617;</a>
{: .tfs-refs}

The figures, the temperature experiment and the data behind the charts: [articles/creative_ai](https://github.com/xmarva/transformers-from-scratch/tree/main/articles/creative_ai).

{% endraw %}
