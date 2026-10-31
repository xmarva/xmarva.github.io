---
layout: post
title: "Is the mind a computer? The computational theory of mind"
date: 2025-05-01 15:00:00
description: "Symbols, rules and meaning: the computational theory of mind from McCulloch and Pitts to the Chinese Room, and what language models add to an old argument"
tags: [cognitive-science, philosophy-of-mind, symbolic-systems, language-of-thought]
categories: [cognitive-science, philosophy]
featured: false
thumbnail: /assets/posts/minds-as-computers/cover.png
og_image: https://xmarva.github.io/assets/posts/minds-as-computers/cover.png
stylesheets: [/assets/css/tfs-course.css]
toc:
  beginning: true
---
{% raw %}

In 1950 Alan Turing made a prediction with numbers in it. In about fifty years, he wrote, computers would "play the imitation game so well that an average interrogator will not have more than 70 per cent chance of making the right identification after five minutes of questioning"<sup class="cite">[<a href="#ref-1" id="cite-1" data-ref="1" aria-label="Reference 1: Turing, A. M. (1950). Computing machinery and intelligence. Mind, 59(236), 433–460.">1</a>]</sup>.

He was a quarter of a century late, and then the machines overshot. In 2025 Cameron Jones and Benjamin Bergen ran Turing's game as he described it: an interrogator chats for five minutes with two witnesses at once, one human and one machine, and has to say which is which.

Prompted to play "a young person who is introverted, knowledgeable about internet culture, and uses slang", GPT-4.5 was picked as the human 73% of the time; the interrogators found the real human in only 27% of games against it<sup class="cite">[<a href="#ref-2" id="cite-2" data-ref="2" aria-label="Reference 2: Jones, C. R., &amp; Bergen, B. K. (2025). Large language models pass the Turing test. arXiv:2503.23674.">2</a>]</sup>.

The study also included, as a control, ELIZA, Joseph Weizenbaum's program from 1966 that turns your sentences into questions. It was picked 23% of the time. In 1966 Weizenbaum had already noticed that "some subjects have been very hard to convince that ELIZA (with its present script) is not human"<sup class="cite">[<a href="#ref-3" id="cite-3" data-ref="3" aria-label="Reference 3: Weizenbaum, J. (1966). ELIZA: a computer program for the study of natural language communication between man and machine. Communications of the ACM, 9(1), 36–45.">3</a>]</sup>.

So a machine passes the test. Does it think? The answer depends on what you take thinking to be, and the most influential answer of the last eighty years is that thinking is computation: the mind is to the brain what a program is to the machine that runs it.

This is the **computational theory of mind**. It gave cognitive science its first research program and artificial intelligence its first name, and it has been attacked from the start on one question that language models have now made practical: can a system that manipulates symbols by their shapes ever mean anything by them?

## Logic in the nervous system

The idea is older than computers. In 1943 the neurophysiologist Warren McCulloch and the logician Walter Pitts published "A logical calculus of the ideas immanent in nervous activity"<sup class="cite">[<a href="#ref-4" id="cite-4" data-ref="4" aria-label="Reference 4: McCulloch, W. S., &amp; Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. Bulletin of Mathematical Biophysics, 5, 115–133.">4</a>]</sup>.

They simplified the neuron to a unit that adds its weighted inputs and fires if the sum reaches a threshold, and showed that networks of such units can compute any proposition of logic.

<img class="tfs-figure tfs-light" src="/assets/posts/minds-as-computers/mcculloch-pitts-light.png" width="760" alt="Three McCulloch-Pitts units. Each adds its weighted inputs and fires if the sum reaches its threshold. Weights 1 and 1 with threshold 2 compute AND, with threshold 1 OR; weight -1 with threshold 0 computes NOT. The truth tables are computed by running the units" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/minds-as-computers/mcculloch-pitts-dark.png" width="760" alt="Three McCulloch-Pitts units. Each adds its weighted inputs and fires if the sum reaches its threshold. Weights 1 and 1 with threshold 2 compute AND, with threshold 1 OR; weight -1 with threshold 0 computes NOT. The truth tables are computed by running the units" loading="lazy">

Three units suffice for AND, OR and NOT, and from those any Boolean function can be built. The neuron in this model is wrong in almost every biological detail, and the authors knew it.

The point was elsewhere: if neurons compute logical functions, then a brain is a kind of logic machine, and a thought might be a computation running on it.

Seven years earlier Turing had described a universal machine that can carry out any computation that can be written down as rules<sup class="cite">[<a href="#ref-5" id="cite-5" data-ref="5" aria-label="Reference 5: Turing, A. M. (1936). On computable numbers, with an application to the Entscheidungsproblem. Proceedings of the London Mathematical Society, s2-42(1), 230–265.">5</a>]</sup>. Put the two papers together and you get the founding bet of cognitive science: the brain implements a computer, and the mind is what it computes.

## Symbols and search

The bet was stated most sharply by Allen Newell and Herbert Simon, whose Logic Theorist proved theorems from *Principia Mathematica* in 1956. Twenty years later, in their Turing Award lecture, they gave it the form of an empirical hypothesis<sup class="cite">[<a href="#ref-6" id="cite-6" data-ref="6" aria-label="Reference 6: Newell, A., &amp; Simon, H. A. (1976). Computer science as empirical inquiry: symbols and search. Communications of the ACM, 19(3), 113–126.">6</a>]</sup>:

> A physical symbol system has the necessary and sufficient means for general intelligent action.

A **physical symbol system** is a machine that holds physical patterns, symbols, combines them into expressions, and runs processes that create, copy, modify and destroy those expressions. "Necessary" means that anything intelligent is such a system, people included.

"Sufficient" means that a large enough system of this kind, suitably organized, would be intelligent. Both halves are strong claims, and Newell and Simon offered them the way a physicist offers a law: to be tested against what intelligent systems actually do.

What does symbol processing look like in practice? Take a puzzle from *Propositiones ad acuendos juvenes*, "problems to sharpen the young", a collection almost certainly by Alcuin of York from around 800 ([MacTutor](https://mathshistory.st-andrews.ac.uk/HistTopics/Alcuin_book/)).

A man must take a wolf, a goat and a bundle of cabbages across a river in a boat that carries only two: him and one passenger. Left alone, the wolf eats the goat and the goat eats the cabbage.

For a symbol system the puzzle is a set of expressions, one per arrangement on the banks, and a rule that turns one arrangement into another. Solving it is searching:

<img class="tfs-figure tfs-light" src="/assets/posts/minds-as-computers/river-light.png" width="760" alt="The 10 safe arrangements of farmer, wolf, goat and cabbage out of 16, joined by the crossings between them. Breadth-first search finds the far bank in 7 crossings, by two routes that differ only in whether the wolf or the cabbage goes over first" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/minds-as-computers/river-dark.png" width="760" alt="The 10 safe arrangements of farmer, wolf, goat and cabbage out of 16, joined by the crossings between them. Breadth-first search finds the far bank in 7 crossings, by two routes that differ only in whether the wolf or the cabbage goes over first" loading="lazy">

The search finds what people find by trial and error: there are only ten safe arrangements, and two shortest solutions that differ in whether the wolf or the cabbage goes over first.

Newell and Simon's claim is that this is not an analogy for what you did when you solved the puzzle as a child. It is a description of it, written at a different level of detail.

## A language of thought

The philosopher Jerry Fodor turned the hypothesis into a theory of the mind as psychology knows it<sup class="cite">[<a href="#ref-7" id="cite-7" data-ref="7" aria-label="Reference 7: Fodor, J. A. (1975). The Language of Thought. Crowell; Harvard University Press.">7</a>]</sup>. We explain behaviour by beliefs and desires: she took the umbrella because she believed it would rain and wanted to stay dry.

Fodor argued that this kind of explanation works because it is literally true. Beliefs and desires are real states of the brain, and they have the structure of sentences in an inner language, a **language of thought**, sometimes called Mentalese.

The hard part is how the content of a belief, what it is about, can cause anything. Fodor's answer came from logic and computers.

A proof checker does not understand the formulas it checks; it looks at their shapes. Yet logicians have designed the rules so that whenever the premises are true, the conclusion the rules produce is true too.

Syntax, done right, keeps track of semantics. If the mind is such a system, then thoughts cause other thoughts by their shapes, and the causation respects what they mean.

Here is the idea in the smallest form, in an example from José Luis Bermúdez's textbook *Cognitive Science*<sup class="cite">[<a href="#ref-8" id="cite-8" data-ref="8" aria-label="Reference 8: Bermúdez, J. L. (2014). Cognitive Science: An Introduction to the Science of the Mind (2nd ed.). Cambridge University Press.">8</a>]</sup>. Let $$Ga$$ say that Georgina is tall and $$Fa$$ that she has red hair.

One rule joins two sentences with $$\wedge$$; another replaces a name by a variable bound by "there exists". Neither rule looks at what $$G$$, $$F$$ or $$a$$ mean:

<img class="tfs-figure tfs-light" src="/assets/posts/minds-as-computers/derivation-light.png" width="760" alt="A derivation in a language of thought: from Ga and Fa the rule of conjunction gives (Ga ∧ Fa), and existential introduction gives ∃x (Gx ∧ Fx). The rules look only at the shapes of the formulas. Of all 16 ways to assign G and F to two people, the premises are true in 4, and the conclusion is true in each of them. A rule that is not sound, from ∃x (Gx ∧ Fx) to Gb, fails in 2 worlds" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/minds-as-computers/derivation-dark.png" width="760" alt="A derivation in a language of thought: from Ga and Fa the rule of conjunction gives (Ga ∧ Fa), and existential introduction gives ∃x (Gx ∧ Fx). The rules look only at the shapes of the formulas. Of all 16 ways to assign G and F to two people, the premises are true in 4, and the conclusion is true in each of them. A rule that is not sound, from ∃x (Gx ∧ Fx) to Gb, fails in 2 worlds" loading="lazy">

The check on the right is the whole point. Over every way the world could be, as far as these sentences can tell, the rules never lead from truths to a falsehood.

A rule that is not sound, such as concluding $$Gb$$ from "someone is tall and red-haired", fails in some world, and a well-built system simply does not have it.

Fodor and Zenon Pylyshyn later turned this into an argument against the alternative that was then on the rise, neural networks<sup class="cite">[<a href="#ref-9" id="cite-9" data-ref="9" aria-label="Reference 9: Fodor, J. A., &amp; Pylyshyn, Z. W. (1988). Connectionism and cognitive architecture: a critical analysis. Cognition, 28(1–2), 3–71.">9</a>]</sup>.

Thought is **systematic**: "you don't find people who can understand the sentence 'John loves the girl' but not the sentence 'the girl loves John'", and the same holds for thinking those thoughts. It is **productive**: from finite means we can think indefinitely many new thoughts. A language with combinatorial syntax explains both for free.

A network that learns associations, they argued, either implements such a language, and is then just its hardware ("models of the implementation of a Classical architecture"), or fails to be systematic. Their verdict was blunt: "the mind cannot be, in its general structure, a Connectionist network."

## The same question at three levels

Part of the confusion in this debate is that "the mind is a computer" can mean three different things. The vision scientist David Marr separated them<sup class="cite">[<a href="#ref-10" id="cite-10" data-ref="10" aria-label="Reference 10: Marr, D. (1982). Vision. W. H. Freeman.">10</a>]</sup>.

At the level of **computational theory** you say what is computed and why: a cash register computes addition, because combining prices must not depend on their order or grouping.

At the level of **representation and algorithm** you say how: decimal digits and carrying, or binary and shifts. At the level of **implementation** you say what physically does it: gears, transistors, neurons.

<img class="tfs-figure tfs-light" src="/assets/posts/minds-as-computers/marr-light.png" width="760" alt="Marr&#x27;s three levels of explanation for two systems. Computational theory: a cash register adds because the total must not depend on order or grouping; the river puzzle needs a sequence of crossings that never leaves a dangerous pair alone. Representation and algorithm: decimal digits with carries; arrangements of the banks searched breadth-first. Implementation: gears or transistors; Python on a laptop or a child&#x27;s neurons" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/minds-as-computers/marr-dark.png" width="760" alt="Marr&#x27;s three levels of explanation for two systems. Computational theory: a cash register adds because the total must not depend on order or grouping; the river puzzle needs a sequence of crossings that never leaves a dangerous pair alone. Representation and algorithm: decimal digits with carries; arrangements of the banks searched breadth-first. Implementation: gears or transistors; Python on a laptop or a child&#x27;s neurons" loading="lazy">

The computational theory of mind is a claim about the first two levels. It is perfectly compatible with neurons being nothing like transistors. It is also why "but the brain is not a von Neumann computer" is not an objection: nobody serious claimed that it was.

## The Chinese Room

The best-known objection grants the theory everything at the level of behaviour and denies it at the level of mind. In 1980 John Searle asked you to imagine yourself locked in a room with boxes of Chinese characters and a rule book in English<sup class="cite">[<a href="#ref-11" id="cite-11" data-ref="11" aria-label="Reference 11: Searle, J. R. (1980). Minds, brains, and programs. Behavioral and Brain Sciences, 3(3), 417–457.">11</a>]</sup>. Questions in Chinese come in through a slot.

You look up their shapes in the book, which tells you which shapes to send back, and you send them. The answers are good enough to fool native speakers. You do not understand a word of Chinese.

<img class="tfs-figure tfs-light" src="/assets/posts/minds-as-computers/chinese-room-light.png" width="760" alt="Searle&#x27;s Chinese Room. A question in Chinese, 你会说中文吗？ (&#x27;Do you speak Chinese?&#x27;), comes in through a slot. Inside, a man who knows no Chinese finds its shape in a rule book written in English and sends back the shapes the book prescribes, 会，说得很流利。 (&#x27;Yes, fluently.&#x27;). Outside, the room seems to understand Chinese" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/minds-as-computers/chinese-room-dark.png" width="760" alt="Searle&#x27;s Chinese Room. A question in Chinese, 你会说中文吗？ (&#x27;Do you speak Chinese?&#x27;), comes in through a slot. Inside, a man who knows no Chinese finds its shape in a rule book written in English and sends back the shapes the book prescribes, 会，说得很流利。 (&#x27;Yes, fluently.&#x27;). Outside, the room seems to understand Chinese" loading="lazy">

The room runs a program for understanding Chinese, Searle says, and nothing in it understands Chinese. So running a program is not sufficient for understanding.

He called the claim he was attacking **strong AI**: that an appropriately programmed computer literally has a mind. **Weak AI**, the use of computers to model and test theories of the mind, he had no quarrel with.

Searle listed the replies he had already heard and answered each one in the same paper:

- **The systems reply**: you do not understand Chinese, but the whole system, you plus the book plus the paper, does. Searle: then memorize the book and do the lookups in your head. Now you are the whole system, and you still understand nothing.
- **The robot reply**: put the program in a robot with cameras and arms, so its symbols connect to things. Searle: the man in the room now also receives symbols from cameras and sends symbols to motors. Still shapes.
- **The brain simulator reply**: make the program simulate the firing of a Chinese speaker's neurons. Searle: simulate it with water pipes and valves, operated by the man. The pipes do not understand Chinese either.

The argument is still argued about. One line of reply, from Paul and Patricia Churchland, is that intuitions about rooms are a poor guide to large systems: a person waving a magnet in a dark room produces electromagnetic waves too slow to see, and would conclude, wrongly, that light is not electromagnetism<sup class="cite">[<a href="#ref-12" id="cite-12" data-ref="12" aria-label="Reference 12: Churchland, P. M., &amp; Churchland, P. S. (1990). Could a machine think? Scientific American, 262(1), 32–37.">12</a>]</sup>.

Searle had also anticipated neural networks: replace the room with a "Chinese gym", a hall full of English speakers each doing what a unit does, and still nobody understands Chinese<sup class="cite">[<a href="#ref-13" id="cite-13" data-ref="13" aria-label="Reference 13: Searle, J. R. (1990). Is the brain&#x27;s mind a computer program? Scientific American, 262(1), 26–31.">13</a>]</sup>.

What he established beyond doubt is narrower than his conclusion, and still important: behaviour alone does not settle the question. In 1980 he had already granted that "there could be two 'systems,' both of which pass the Turing test, but only one of which understands".

## Where do symbols get their meaning?

Stevan Harnad sharpened Searle's point into a problem that any theory has to solve, not just computational ones<sup class="cite">[<a href="#ref-14" id="cite-14" data-ref="14" aria-label="Reference 14: Harnad, S. (1990). The symbol grounding problem. Physica D, 42(1–3), 335–346.">14</a>]</sup>. Imagine trying to learn Chinese from a Chinese-Chinese dictionary.

Every word is defined by other words, which are defined by other words, and you go round forever without touching anything. This is the **symbol grounding problem**: how do the symbols of a system come to be about things in the world, rather than only about each other?

Harnad's own answer was that some symbols must be grounded in perception, in the categories that sensory systems learn to pick out.

Language models are the dictionary-go-round built at scale. They learn from text alone, and on Harnad's view they should have no grounded meanings at all.

Emily Bender and Alexander Koller made the argument with an octopus that taps a telegraph cable between two islands and learns to imitate one of the speakers perfectly, until someone asks it for help against a bear<sup class="cite">[<a href="#ref-15" id="cite-15" data-ref="15" aria-label="Reference 15: Bender, E. M., &amp; Koller, A. (2020). Climbing towards NLU: on meaning, form, and understanding in the age of data. ACL.">15</a>]</sup>. Form, they argue, is not enough to learn meaning.

It turns out that form carries more than this suggests. Here is a small test you can run. Take fifty colour names, and for each the colour people mean by it, from an online survey in which over five million colours were named in 222,500 sessions<sup class="cite">[<a href="#ref-16" id="cite-16" data-ref="16" aria-label="Reference 16: Munroe, R. (2010). Color survey results; the names and colours: rgb.txt, public domain.">16</a>]</sup>.

Place the colours in CIELAB, a space built so that equal distances look equally different. Then take Qwen2.5-1.5B, a model that has never seen a colour, give it each name in a sentence, and compare the distances between its internal states:

<img class="tfs-figure tfs-light" src="/assets/posts/minds-as-computers/color-map-light.png" width="760" alt="Fifty colour names placed by their colours in CIELAB and by the hidden states of Qwen2.5-1.5B-Instruct at layer 12, rotated and scaled onto CIELAB. Blues, purples and reds keep their neighbourhoods; the agreement of the two sets of distances is Spearman&#x27;s rho = 0.38" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/minds-as-computers/color-map-dark.png" width="760" alt="Fifty colour names placed by their colours in CIELAB and by the hidden states of Qwen2.5-1.5B-Instruct at layer 12, rotated and scaled onto CIELAB. Blues, purples and reds keep their neighbourhoods; the agreement of the two sets of distances is Spearman&#x27;s rho = 0.38" loading="lazy">

The geometry is far from perfect, but it is there. Blues sit with blues, reds with reds and pinks, teal between green and blue.

The rank correlation between the two sets of distances is 0.38, and none of 2,000 shuffles of the words comes close. It is not an accident of one layer either:

<img class="tfs-figure tfs-light" src="/assets/posts/minds-as-computers/color-layers-light.png" width="760" alt="Spearman&#x27;s rho between the model&#x27;s distances and CIELAB distances, by layer: about 0.1 at the embedding, between 0.3 and 0.4 from layer 6 on, against 0.07 for shuffled words" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/minds-as-computers/color-layers-dark.png" width="760" alt="Spearman&#x27;s rho between the model&#x27;s distances and CIELAB distances, by layer: about 0.1 at the embedding, between 0.3 and 0.4 from layer 6 on, against 0.07 for shuffled words" loading="lazy">

The embedding layer knows little; from the sixth layer on, the model's states keep a stable fraction of the colour space.

[Abdou et al. (2021)](https://aclanthology.org/2021.conll-1.9/) found a modest but significant alignment of the same kind in several language models, and GPT-4's judgments of how similar two colours are, given only their names, agree with people's at r = 0.79<sup class="cite">[<a href="#ref-17" id="cite-17" data-ref="17" aria-label="Reference 17: Marjieh, R., et al. (2024). Large language models predict human sensory judgments across six modalities. Scientific Reports, 14, 21445.">17</a>]</sup>.

Text is not as cut off from the world as the dictionary image suggests, because people wrote it about the world: about skies that were deep blue, and teal that sits between green and blue. Congenitally blind people show the same from the other side.

Asked how similar colour words are, they produce a map that resembles the colour space of sighted people<sup class="cite">[<a href="#ref-18" id="cite-18" data-ref="18" aria-label="Reference 18: Saysani, A., Corballis, M. C., &amp; Corballis, P. M. (2018). Colour envisioned: concepts of colour in the blind and sighted. Visual Cognition, 26(5), 382–392.">18</a>]</sup>, and they share sighted people's understanding of what colours are for, learned from "living among people who talk about color"<sup class="cite">[<a href="#ref-19" id="cite-19" data-ref="19" aria-label="Reference 19: Kim, J. S., Aheimer, B., Montané Manrara, V., &amp; Bedny, M. (2021). Shared understanding of color among sighted and blind adults. PNAS, 118(33), e2020192118.">19</a>]</sup>.

None of this answers Searle. A geometry of colour words is not the experience of red, and a system can have the first without the second. What it does change is the shape of the argument.

The question is no longer whether meaning can come from symbols alone, yes or no, but how much of it does, and which part can only come from somewhere else.

## Networks, systematicity and language models

Fodor and Pylyshyn's challenge to neural networks became testable. In 2018 Brenden Lake and Marco Baroni built SCAN, a set of commands such as "jump twice and walk left", and trained sequence-to-sequence networks on it<sup class="cite">[<a href="#ref-20" id="cite-20" data-ref="20" aria-label="Reference 20: Lake, B., &amp; Baroni, M. (2018). Generalization without systematicity. ICML.">20</a>]</sup>.

On a random split of the data the best network got 99.8% of new commands right. When "jump" appeared in training only on its own and had to be combined with the rest of the grammar at test time, the best network got 1.2%.

People do this effortlessly. Five years later the same authors trained a standard transformer on a stream of small composition tasks, each with its own made-up grammar, and on a new task it matched the algebraic answer as often as people did, or more<sup class="cite">[<a href="#ref-21" id="cite-21" data-ref="21" aria-label="Reference 21: Lake, B. M., &amp; Baroni, M. (2023). Human-like systematic generalization through a meta-learning neural network. Nature, 623, 115–121.">21</a>]</sup>.

They wrote that they had addressed "Fodor and Pylyshyn's challenge", and in the same paper listed what was left: much longer outputs and new sentence structures still defeated the model. Systematicity was not built in; it was learned from practice at being systematic.

Something similar shows up inside sequence models. A transformer trained only on lists of moves in the board game Othello, never shown a board, develops an internal representation of the board that can be read out and edited: change it, and the model's next moves change as if the board were different<sup class="cite">[<a href="#ref-22" id="cite-22" data-ref="22" aria-label="Reference 22: Li, K., et al. (2023). Emergent world representations: exploring a sequence model trained on a synthetic task. ICLR.">22</a>]</sup>.

Read as "my stones and the opponent's" rather than "black and white", the board can be recovered by a simple linear probe with 99.6% accuracy<sup class="cite">[<a href="#ref-23" id="cite-23" data-ref="23" aria-label="Reference 23: Nanda, N., Lee, A., &amp; Wattenberg, M. (2023). Emergent linear representations in world models of self-supervised sequence models. BlackboxNLP.">23</a>]</sup>. Othello is a toy world, fully determined by the moves, so this shows structure rather than reference to anything outside the text.

But it does show that such a network is not only associating surface forms. It builds structured states and computes with them, which is closer to what Fodor asked for than either side expected.

## What the metaphor leaves out

The computational theory never lacked critics who thought it was looking in the wrong place. Hubert Dreyfus argued from the 1970s that human expertise rests on embodied know-how, not explicit rules<sup class="cite">[<a href="#ref-24" id="cite-24" data-ref="24" aria-label="Reference 24: Dreyfus, H. L. (1972). What Computers Can&#x27;t Do. Harper &amp; Row.">24</a>]</sup>.

The enactive tradition of Francisco Varela, Evan Thompson and Eleanor Rosch put the body and its activity at the centre<sup class="cite">[<a href="#ref-25" id="cite-25" data-ref="25" aria-label="Reference 25: Varela, F. J., Thompson, E., &amp; Rosch, E. (1991). The Embodied Mind. MIT Press.">25</a>]</sup>. More recent **predictive processing** theories describe the brain as a hierarchy that predicts its own sensory input and learns from the errors<sup class="cite">[<a href="#ref-26" id="cite-26" data-ref="26" aria-label="Reference 26: Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. Behavioral and Brain Sciences, 36(3), 181–204.">26</a>, <a href="#ref-27" id="cite-27" data-ref="27" aria-label="Reference 27: Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11(2), 127–138.">27</a>]</sup>.

Notice, though, that most of these are still computational at Marr's first two levels. They disagree about what is computed and in what format, not about whether the mind computes.

The numbers also remind you how different the hardware is. A human brain has about 86 billion neurons<sup class="cite">[<a href="#ref-28" id="cite-28" data-ref="28" aria-label="Reference 28: Azevedo, F. A. C., et al. (2009). Equal numbers of neuronal and nonneuronal cells make the human brain an isometrically scaled-up primate brain. Journal of Comparative Neurology, 513(5), 532–541.">28</a>]</sup>, weighs about 2% of the body and uses about 20% of its oxygen<sup class="cite">[<a href="#ref-29" id="cite-29" data-ref="29" aria-label="Reference 29: Raichle, M. E., &amp; Gusnard, D. A. (2002). Appraising the brain&#x27;s energy budget. PNAS, 99(16), 10237–10239.">29</a>]</sup>.

That comes to roughly 20 watts, "a paltry sum given that the computer on which this article is being typed requires 80 W", as the physicist Vijay Balasubramanian put it<sup class="cite">[<a href="#ref-30" id="cite-30" data-ref="30" aria-label="Reference 30: Balasubramanian, V. (2021). Brain power. PNAS, 118(32), e2107022118.">30</a>]</sup>.

Which brings back the test. The interrogators who picked GPT-4.5 over a person were not wrong about anything they could check: the replies were what a person would write.

Turing proposed the game because he found "Can machines think?" "too meaningless to deserve discussion", and behaviour was all anyone could go on. Seventy-five years later that is no longer true.

We can open the model, read off the board it keeps for Othello or the geometry it keeps for colour, and ask how much of a mind is in there, piece by piece. Searle's room had no windows. This one does.

## References

1. {:#ref-1} Turing, A. M. (1950). [Computing machinery and intelligence](https://doi.org/10.1093/mind/LIX.236.433). *Mind*, 59(236), 433–460. <a class="ref-back" href="#cite-1" aria-label="Back to the text">&#8617;</a>
2. {:#ref-2} Jones, C. R., & Bergen, B. K. (2025). [Large language models pass the Turing test](https://arxiv.org/abs/2503.23674). arXiv:2503.23674. <a class="ref-back" href="#cite-2" aria-label="Back to the text">&#8617;</a>
3. {:#ref-3} Weizenbaum, J. (1966). [ELIZA: a computer program for the study of natural language communication between man and machine](https://doi.org/10.1145/365153.365168). *Communications of the ACM*, 9(1), 36–45. <a class="ref-back" href="#cite-3" aria-label="Back to the text">&#8617;</a>
4. {:#ref-4} McCulloch, W. S., & Pitts, W. (1943). [A logical calculus of the ideas immanent in nervous activity](https://doi.org/10.1007/BF02478259). *Bulletin of Mathematical Biophysics*, 5, 115–133. <a class="ref-back" href="#cite-4" aria-label="Back to the text">&#8617;</a>
5. {:#ref-5} Turing, A. M. (1936). [On computable numbers, with an application to the Entscheidungsproblem](https://doi.org/10.1112/plms/s2-42.1.230). *Proceedings of the London Mathematical Society*, s2-42(1), 230–265. <a class="ref-back" href="#cite-5" aria-label="Back to the text">&#8617;</a>
6. {:#ref-6} Newell, A., & Simon, H. A. (1976). [Computer science as empirical inquiry: symbols and search](https://doi.org/10.1145/360018.360022). *Communications of the ACM*, 19(3), 113–126. <a class="ref-back" href="#cite-6" aria-label="Back to the text">&#8617;</a>
7. {:#ref-7} Fodor, J. A. (1975). [*The Language of Thought*](https://archive.org/details/languageofthough0000fodo). Crowell; Harvard University Press. <a class="ref-back" href="#cite-7" aria-label="Back to the text">&#8617;</a>
8. {:#ref-8} Bermúdez, J. L. (2014). [*Cognitive Science: An Introduction to the Science of the Mind*](https://archive.org/details/cognitivescience0000berm) (2nd ed.). Cambridge University Press. <a class="ref-back" href="#cite-8" aria-label="Back to the text">&#8617;</a>
9. {:#ref-9} Fodor, J. A., & Pylyshyn, Z. W. (1988). [Connectionism and cognitive architecture: a critical analysis](https://doi.org/10.1016/0010-0277%2888%2990031-5). *Cognition*, 28(1–2), 3–71. <a class="ref-back" href="#cite-9" aria-label="Back to the text">&#8617;</a>
10. {:#ref-10} Marr, D. (1982). [*Vision*](https://archive.org/details/vision00davi). W. H. Freeman. <a class="ref-back" href="#cite-10" aria-label="Back to the text">&#8617;</a>
11. {:#ref-11} Searle, J. R. (1980). [Minds, brains, and programs](https://doi.org/10.1017/S0140525X00005756). *Behavioral and Brain Sciences*, 3(3), 417–457. <a class="ref-back" href="#cite-11" aria-label="Back to the text">&#8617;</a>
12. {:#ref-12} Churchland, P. M., & Churchland, P. S. (1990). [Could a machine think?](https://doi.org/10.1038/scientificamerican0190-32) *Scientific American*, 262(1), 32–37. <a class="ref-back" href="#cite-12" aria-label="Back to the text">&#8617;</a>
13. {:#ref-13} Searle, J. R. (1990). [Is the brain's mind a computer program?](https://doi.org/10.1038/scientificamerican0190-26) *Scientific American*, 262(1), 26–31. <a class="ref-back" href="#cite-13" aria-label="Back to the text">&#8617;</a>
14. {:#ref-14} Harnad, S. (1990). [The symbol grounding problem](https://doi.org/10.1016/0167-2789%2890%2990087-6). *Physica D*, 42(1–3), 335–346. <a class="ref-back" href="#cite-14" aria-label="Back to the text">&#8617;</a>
15. {:#ref-15} Bender, E. M., & Koller, A. (2020). [Climbing towards NLU: on meaning, form, and understanding in the age of data](https://aclanthology.org/2020.acl-main.463/). *ACL*. <a class="ref-back" href="#cite-15" aria-label="Back to the text">&#8617;</a>
16. {:#ref-16} Munroe, R. (2010). [Color survey results](https://blog.xkcd.com/2010/05/03/color-survey-results/); the names and colours: [rgb.txt](https://xkcd.com/color/rgb.txt), public domain. <a class="ref-back" href="#cite-16" aria-label="Back to the text">&#8617;</a>
17. {:#ref-17} Marjieh, R., et al. (2024). [Large language models predict human sensory judgments across six modalities](https://doi.org/10.1038/s41598-024-72071-1). *Scientific Reports*, 14, 21445. <a class="ref-back" href="#cite-17" aria-label="Back to the text">&#8617;</a>
18. {:#ref-18} Saysani, A., Corballis, M. C., & Corballis, P. M. (2018). [Colour envisioned: concepts of colour in the blind and sighted](https://doi.org/10.1080/13506285.2018.1465148). *Visual Cognition*, 26(5), 382–392. <a class="ref-back" href="#cite-18" aria-label="Back to the text">&#8617;</a>
19. {:#ref-19} Kim, J. S., Aheimer, B., Montané Manrara, V., & Bedny, M. (2021). [Shared understanding of color among sighted and blind adults](https://doi.org/10.1073/pnas.2020192118). *PNAS*, 118(33), e2020192118. <a class="ref-back" href="#cite-19" aria-label="Back to the text">&#8617;</a>
20. {:#ref-20} Lake, B., & Baroni, M. (2018). [Generalization without systematicity](https://proceedings.mlr.press/v80/lake18a.html). *ICML*. <a class="ref-back" href="#cite-20" aria-label="Back to the text">&#8617;</a>
21. {:#ref-21} Lake, B. M., & Baroni, M. (2023). [Human-like systematic generalization through a meta-learning neural network](https://doi.org/10.1038/s41586-023-06668-3). *Nature*, 623, 115–121. <a class="ref-back" href="#cite-21" aria-label="Back to the text">&#8617;</a>
22. {:#ref-22} Li, K., et al. (2023). [Emergent world representations: exploring a sequence model trained on a synthetic task](https://arxiv.org/abs/2210.13382). *ICLR*. <a class="ref-back" href="#cite-22" aria-label="Back to the text">&#8617;</a>
23. {:#ref-23} Nanda, N., Lee, A., & Wattenberg, M. (2023). [Emergent linear representations in world models of self-supervised sequence models](https://aclanthology.org/2023.blackboxnlp-1.2/). *BlackboxNLP*. <a class="ref-back" href="#cite-23" aria-label="Back to the text">&#8617;</a>
24. {:#ref-24} Dreyfus, H. L. (1972). [*What Computers Can't Do*](https://archive.org/details/whatcomputerscan00dreyrich). Harper & Row. <a class="ref-back" href="#cite-24" aria-label="Back to the text">&#8617;</a>
25. {:#ref-25} Varela, F. J., Thompson, E., & Rosch, E. (1991). [*The Embodied Mind*](https://archive.org/details/embodiedmindcogn0000vare). MIT Press. <a class="ref-back" href="#cite-25" aria-label="Back to the text">&#8617;</a>
26. {:#ref-26} Clark, A. (2013). [Whatever next? Predictive brains, situated agents, and the future of cognitive science](https://doi.org/10.1017/S0140525X12000477). *Behavioral and Brain Sciences*, 36(3), 181–204. <a class="ref-back" href="#cite-26" aria-label="Back to the text">&#8617;</a>
27. {:#ref-27} Friston, K. (2010). [The free-energy principle: a unified brain theory?](https://doi.org/10.1038/nrn2787) *Nature Reviews Neuroscience*, 11(2), 127–138. <a class="ref-back" href="#cite-27" aria-label="Back to the text">&#8617;</a>
28. {:#ref-28} Azevedo, F. A. C., et al. (2009). [Equal numbers of neuronal and nonneuronal cells make the human brain an isometrically scaled-up primate brain](https://doi.org/10.1002/cne.21974). *Journal of Comparative Neurology*, 513(5), 532–541. <a class="ref-back" href="#cite-28" aria-label="Back to the text">&#8617;</a>
29. {:#ref-29} Raichle, M. E., & Gusnard, D. A. (2002). [Appraising the brain's energy budget](https://doi.org/10.1073/pnas.172399499). *PNAS*, 99(16), 10237–10239. <a class="ref-back" href="#cite-29" aria-label="Back to the text">&#8617;</a>
30. {:#ref-30} Balasubramanian, V. (2021). [Brain power](https://doi.org/10.1073/pnas.2107022118). *PNAS*, 118(32), e2107022118. <a class="ref-back" href="#cite-30" aria-label="Back to the text">&#8617;</a>
31. {:#ref-31} Abdou, M., et al. (2021). [Can language models encode perceptual structure without grounding? A case study in color](https://aclanthology.org/2021.conll-1.9/). *CoNLL*.
32. {:#ref-32} Stanford Encyclopedia of Philosophy: [The Computational Theory of Mind](https://plato.stanford.edu/entries/computational-mind/), [The Chinese Room Argument](https://plato.stanford.edu/entries/chinese-room/), [The Language of Thought Hypothesis](https://plato.stanford.edu/entries/language-thought/), [Connectionism](https://plato.stanford.edu/entries/connectionism/).
{: .tfs-refs}

The measurement of colour in Qwen2.5-1.5B and the figures: [articles/minds_as_computers](https://github.com/xmarva/transformers-from-scratch/tree/main/articles/minds_as_computers).

{% endraw %}
