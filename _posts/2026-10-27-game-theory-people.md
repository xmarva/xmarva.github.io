---
layout: post
title: How to treat people, according to game theory
date: 2026-10-27 18:00:00
description: "Trust, forgiveness, revenge and reputation, worked out with the repeated prisoner's dilemma: Axelrod's tournaments, the arithmetic of forgiveness, extortion strategies, indirect reciprocity and the stag hunt, with simulations"
tags: [game-theory, cooperation, prisoners-dilemma, decision-making]
categories: cognitive-science
featured: false
thumbnail: /assets/posts/game-theory-people/cover.png
og_image: https://xmarva.github.io/assets/posts/game-theory-people/cover.png
stylesheets: [/assets/css/tfs-course.css]
toc:
  beginning: true
---
{% raw %}

On quiet stretches of the Western Front in the First World War, British and German soldiers sat in trenches within rifle range of each other and, for weeks at a time, tried not to kill each other.

A British staff officer arriving in one sector was "astonished to observe German soldiers walking about within rifle range behind their own line. Our men appeared to take no notice.

I privately made up my mind to do away with that sort of thing when we took over; such things should not be allowed. These people evidently did not know there was a war on. Both sides apparently believed in the policy of 'live and let live'"<sup class="cite">[<a href="#ref-1" id="cite-1" data-ref="1" aria-label="Reference 1: Axelrod, R. (1984). The Evolution of Cooperation. Basic Books. Chapter 4, The live-and-let-live system in trench warfare in World War I, pp. 73–87.">1</a>]</sup>.

Nobody negotiated this, and fraternising with the enemy was forbidden. The truce lived in the rhythm of the shooting.

Artillery fired at the same targets at the same hours, and a German soldier described the British "evening gun": "At seven it came – so regularly that you could set your watch by it.... There were even some inquisitive fellows who crawled out... a little before seven, in order to see it burst."

When a salvo did land by mistake, a German climbed onto his parapet and shouted across: "We are very sorry about that; we hope no one was hurt. It is not our fault, it is that damned Prussian artillery."

Tony Ashworth reconstructed this "live and let live system" from soldiers' memoirs and regimental histories<sup class="cite">[<a href="#ref-2" id="cite-2" data-ref="2" aria-label="Reference 2: Ashworth, T. (1980). Trench Warfare 1914–1918: The Live and Let Live System. Macmillan.">2</a>]</sup>, and the political scientist Robert Axelrod made it the central case of *The Evolution of Cooperation*.

The soldiers were enemies with every reason to shoot. What held them back was that the same battalions faced each other for weeks, and each knew that a shell sent over today would come back tomorrow.

The system did not end because anyone taught them to hate each other. It ended when headquarters started ordering raids, attacks whose results it could check: "There was no effective way to pretend that a raid had been undertaken when it had not."

Once the generals could see what each unit did, the game the soldiers had been playing with each other was replaced by one they played with their own command.

Without a word of theory, the soldiers had found what game theory would later prove: between people who expect to meet again, cooperation can be the selfish choice. This post follows that idea as far as the mathematics goes. It is not a list of tips about people.

It is a set of small, exact models, each of which answers one question that every relationship eventually asks: whether to trust first, how to answer a betrayal, how much to forgive, what a reputation is worth, and when to walk away.

The answers are not always the ones intuition gives, and where they agree with intuition, the models say *why*, which is more useful, because a reason tells you when the advice stops working.

## One round of the prisoner's dilemma

Two people can each do something for the other at a cost to themselves: cover a shift, lend money, keep a secret, share credit. If both do, both are better off.

If one does and the other doesn't, the one who didn't is best off of all, and the one who did is worst off. If neither does, they lose what they could have had together.

Game theory calls this structure the **prisoner's dilemma**, after a story Albert Tucker told to illustrate it in 1950; the game itself was invented a few months earlier at the RAND Corporation by Merrill Flood and Melvin Dresher<sup class="cite">[<a href="#ref-3" id="cite-3" data-ref="3" aria-label="Reference 3: Flood, M. M. (1952). Some Experimental Games. RAND Research Memorandum RM-789-1.">3</a>]</sup>.

<img class="tfs-figure tfs-light" src="/assets/posts/game-theory-people/dilemma-light.png" width="760" alt="The prisoner&#x27;s dilemma with Axelrod&#x27;s payoffs. Both cooperate: 3 points each. You cooperate and they defect: 0 for you, 5 for them. You defect and they cooperate: 5 for you, 0 for them. Both defect: 1 each. Whatever the other does, defecting gives you more, and two players who reason this way get 1 each" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/game-theory-people/dilemma-dark.png" width="760" alt="The prisoner&#x27;s dilemma with Axelrod&#x27;s payoffs. Both cooperate: 3 points each. You cooperate and they defect: 0 for you, 5 for them. You defect and they cooperate: 5 for you, 0 for them. Both defect: 1 each. Whatever the other does, defecting gives you more, and two players who reason this way get 1 each" loading="lazy">

The numbers are the ones Robert Axelrod used, and only their order matters: the **temptation** to exploit, $$T = 5$$, beats the **reward** for mutual cooperation, $$R = 3$$, which beats the **punishment** for mutual defection, $$P = 1$$, which beats the **sucker's payoff**, $$S = 0$$. One more condition, $$2R > T + S$$, says that taking turns at exploiting each other is worse than cooperating.

The logic of a single round is merciless. Whatever the other player does, you get more by defecting: $$5 > 3$$ if they cooperate, $$1 > 0$$ if they defect.

Defection is a **dominant strategy**, and two players who reason correctly end up with 1 point each while 3 each was on the table. There is no mistake in the reasoning to point to, which is what makes this a dilemma and not a puzzle.

## How long the relationship lasts

The first time anyone played the game, in January 1950 at RAND, it was not played once. Flood and Dresher had two colleagues play a version of it a hundred times in a row.

The outcome the theory predicts came up 14 times; mutual cooperation came up 60 times. One of the players kept notes. "Hope he's bright," he wrote after the first round, and after the seventy-first: "This is like toilet training a child—you have to be very patient"<sup class="cite">[<a href="#ref-3" id="cite-4" data-ref="3" aria-label="Reference 3: Flood, M. M. (1952). Some Experimental Games. RAND Research Memorandum RM-789-1.">3</a>]</sup>.

John Nash, shown the results, objected that "the experiment really amounts to having the players play one large multimove game." He was right, and that objection is where the interesting part of the theory begins.

People rarely meet once. Suppose that after each round there is another one with probability $$w$$.

The relationship then lasts $$1/(1 - w)$$ rounds on average: two rounds when $$w = 0.5$$, ten when $$w = 0.9$$. Axelrod later called this the **shadow of the future**<sup class="cite">[<a href="#ref-1" id="cite-5" data-ref="1" aria-label="Reference 1: Axelrod, R. (1984). The Evolution of Cooperation. Basic Books. Chapter 4, The live-and-let-live system in trench warfare in World War I, pp. 73–87.">1</a>]</sup>.

Now the other player's strategy can depend on what you did. Take the simplest such strategy, **tit for tat**: cooperate in the first round, then do whatever the other player did in the previous one.

Against it, a defection buys you $$T$$ today and a punishment tomorrow. Three ways of playing against tit for tat, and what each earns per round, averaged over the relationship:

<img class="tfs-figure tfs-light" src="/assets/posts/game-theory-people/shadow-light.png" width="760" alt="Points per round against tit for tat as a function of w, the probability of another round. Cooperating earns 3. Always defecting earns 5 - 4w and wins below w = 1/2; defecting and making up in turns earns 5/(1 + w) and wins below w = 2/3. Above 2/3 cooperating is the best reply" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/game-theory-people/shadow-dark.png" width="760" alt="Points per round against tit for tat as a function of w, the probability of another round. Cooperating earns 3. Always defecting earns 5 - 4w and wins below w = 1/2; defecting and making up in turns earns 5/(1 + w) and wins below w = 2/3. Above 2/3 cooperating is the best reply" loading="lazy">

Always defecting earns $$T$$ once and $$P$$ ever after; per round that is $$(1 - w)T + wP = 5 - 4w$$, which falls below the reward $$R = 3$$ when $$w > (T - R)/(T - P) = 1/2$$.

A subtler exploiter defects and then makes up with one cooperation, in turns, collecting $$T$$ and $$S$$ alternately; that earns $$(T + wS)/(1 + w)$$ per round and falls below $$R$$ when $$w > (T - R)/(R - S) = 2/3$$. Above both thresholds, against tit for tat, nothing beats cooperating.

Axelrod and the biologist William Hamilton showed that these two conditions are exactly what makes tit for tat **evolutionarily stable**: when both hold, no strategy can invade a population that plays it<sup class="cite">[<a href="#ref-4" id="cite-6" data-ref="4" aria-label="Reference 4: Axelrod, R., &amp; Hamilton, W. D. (1981). The evolution of cooperation. Science, 211(4489), 1390–1396.">4</a>]</sup>.

The first practical conclusion is about the structure of relationships, not about character. Cooperation is rational when the future is long enough and the present is not too tempting, and both can be changed.

People who expect to meet again, who live in the same building, work in the same field, belong to the same small community, face a long shadow of the future.

A relationship that both sides know is ending is a game with $$w$$ near zero, and the last rounds of such games are where the defections cluster. This is why a promise to a stranger at an airport is worth less than the same promise to a neighbour, whatever the character of the person who made it.

## Axelrod's tournaments

Knowing that cooperation *can* be stable doesn't say how to play. Axelrod asked game theorists, economists, psychologists and mathematicians to send him programs that play the repeated prisoner's dilemma, and ran them against each other in a round robin of 200-move games<sup class="cite">[<a href="#ref-5" id="cite-7" data-ref="5" aria-label="Reference 5: Axelrod, R. (1980a). Effective choice in the prisoner&#x27;s dilemma. Journal of Conflict Resolution, 24(1), 3–25.">5</a>]</sup>.

Fourteen programs took part, and the highest score went to "the simplest of all strategies submitted", tit for tat, sent by the psychologist Anatol Rapoport<sup class="cite">[<a href="#ref-4" id="cite-8" data-ref="4" aria-label="Reference 4: Axelrod, R., &amp; Hamilton, W. D. (1981). The evolution of cooperation. Science, 211(4489), 1390–1396.">4</a>]</sup>.

Axelrod published the results and invited a second round. This time there were 62 entries from six countries, from people who knew why tit for tat had won and tried to beat it, and tit for tat won again<sup class="cite">[<a href="#ref-6" id="cite-9" data-ref="6" aria-label="Reference 6: Axelrod, R. (1980b). More effective choice in the prisoner&#x27;s dilemma. Journal of Conflict Resolution, 24(3), 379–403.">6</a>]</sup>.

The victory was less certain than the story usually told. In 2025 Vincent Knight and colleagues reran the second tournament from Axelrod's original code with 25,000 different random seeds, and tit for tat won 63% of the runs<sup class="cite">[<a href="#ref-7" id="cite-10" data-ref="7" aria-label="Reference 7: Knight, V., Campbell, O., Harper, M., Gaffney, T. J., &amp; Glynatsi, N. E. (2025). Reviving, reproducing, and revisiting Axelrod&#x27;s second tournament. arXiv:2510.15438.">7</a>]</sup>.

A tournament is one draw of a random process, which is worth remembering before treating any single ranking, including the one below, as a law of nature.

Here is a small tournament in the same spirit, with twelve classic strategies, each playing every other and itself, the continuation probability of Axelrod's second tournament (289 rounds on average) and no mistakes:

<img class="tfs-figure tfs-light" src="/assets/posts/game-theory-people/tournament-light.png" width="760" alt="Matches won against mean points per round for twelve strategies. Generous ZD scores highest, 2.50, winning 0 matches; the nice strategies take the top places without winning any. Always defect wins 9 of 11 matches and scores 2.05; Joss is last with 1.87" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/game-theory-people/tournament-dark.png" width="760" alt="Matches won against mean points per round for twelve strategies. Generous ZD scores highest, 2.50, winning 0 matches; the nice strategies take the top places without winning any. Always defect wins 9 of 11 matches and scores 2.05; Joss is last with 1.87" loading="lazy">

The picture has an odd shape. The six strategies that are never the first to defect take the top six places, and between them they win almost no matches: none at all, except grim trigger, which won two.

Always defect wins nine of its eleven matches and finishes ninth. An extortioner, which we will meet later, wins nine and finishes eleventh.

The paradox disappears once you see what a match is. Tit for tat can never get more than its partner in a single game, because it only defects in answer to a defection; at best it ties.

It wins the tournament by doing reasonably well against everyone, while the strategies that "win" their matches do it by dragging both sides down to the punishment payoff. Axelrod turned this into his first piece of advice: don't be envious.

Your score is compared with other people in your position, not with the person across the table, and a relationship you "win" at 1 point a round is worse than one you share at 3.

Axelrod summed up why tit for tat did well in four words, "nice, retaliatory, forgiving, and clear"<sup class="cite">[<a href="#ref-1" id="cite-11" data-ref="1" aria-label="Reference 1: Axelrod, R. (1984). The Evolution of Cooperation. Basic Books. Chapter 4, The live-and-let-live system in trench warfare in World War I, pp. 73–87.">1</a>]</sup>. It is **nice**: never the first to defect. It is **retaliatory**: a defection is answered at once, so exploiting it doesn't pay.

It is **forgiving**: once the other side returns to cooperation, so does tit for tat, without holding a grudge. And it is **clear**: after a few rounds anyone can tell what it does, so the other side learns that cooperation is the only profitable reply.

## Mistakes

Tit for tat has a weakness that a tournament without mistakes cannot show. Real moves are noisy. A message is misread, a favour is forgotten, a busy week looks like a cold shoulder.

Suppose one move in a while comes out as the opposite of what was meant. Here is what one mistake does between two players of the same strategy:

<img class="tfs-figure tfs-light" src="/assets/posts/game-theory-people/echo-light.png" width="760" alt="Sixteen rounds between two players of the same strategy, with one mistake in round 4: player one means to cooperate and defects. Two tit for tat players then punish each other in turns for the rest of the game. Two generous tit for tat players stop the echo when one of them forgives, here after 2 rounds. Two win-stay, lose-shift players defect together once and are back to cooperating in round 6" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/game-theory-people/echo-dark.png" width="760" alt="Sixteen rounds between two players of the same strategy, with one mistake in round 4: player one means to cooperate and defects. Two tit for tat players then punish each other in turns for the rest of the game. Two generous tit for tat players stop the echo when one of them forgives, here after 2 rounds. Two win-stay, lose-shift players defect together once and are back to cooperating in round 6" loading="lazy">

Two tit for tat players turn one accident into a feud. Each faithfully answers the other's last defection, and the defection bounces between them forever, with each side certain it is only responding. Many long-running quarrels between reasonable people have this structure: there is no villain in the story, just two reciprocators with a short memory.

The other two strategies recover. **Generous tit for tat** answers a defection with a defection only most of the time and forgives with some probability $$q$$, so the echo dies out after a few rounds<sup class="cite">[<a href="#ref-8" id="cite-12" data-ref="8" aria-label="Reference 8: Nowak, M., &amp; Sigmund, K. (1992). Tit for tat in heterogeneous populations. Nature, 355, 250–253.">8</a>]</sup>. **Win-stay, lose-shift** repeats its last move if it went well (it got $$R$$ or $$T$$) and switches if it went badly ($$P$$ or $$S$$); after a mistake both players defect once, both did badly, and both switch back to cooperating<sup class="cite">[<a href="#ref-9" id="cite-13" data-ref="9" aria-label="Reference 9: Nowak, M., &amp; Sigmund, K. (1993). A strategy of win-stay, lose-shift that outperforms tit-for-tat in the Prisoner&#x27;s Dilemma game. Nature, 364, 56–58.">9</a>]</sup>.

How much this matters is easy to compute. The payoffs below are exact: two memory-one strategies playing each other form a Markov chain over the four outcomes of a round, and its discounted stationary distribution gives the expected payoff without playing a single game.

<img class="tfs-figure tfs-light" src="/assets/posts/game-theory-people/noise-light.png" width="760" alt="Points per round for two players of the same strategy as mistakes grow from 0 to 10 percent. At 1 percent: generous tit for tat 2.97, win-stay lose-shift 2.95, tit for tat 2.53, grim trigger 1.69" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/game-theory-people/noise-dark.png" width="760" alt="Points per round for two players of the same strategy as mistakes grow from 0 to 10 percent. At 1 percent: generous tit for tat 2.97, win-stay lose-shift 2.95, tit for tat 2.53, grim trigger 1.69" loading="lazy">

One mistake in a hundred moves costs two tit for tat players about a sixth of everything they earn together (2.53 points a round instead of 3); grim trigger, which never forgives at all, loses almost half.

Generous tit for tat and win-stay, lose-shift barely notice. The difference between a strategy that punishes forever and one that forgives is small in a world without mistakes and enormous in the world we live in.

## How much to forgive

Forgiveness has a price, and the price has an exact size. If you forgive too often, a player who exploits forgiveness earns more than people who cooperate. Picture a population that plays generous tit for tat with forgiveness probability $$q$$, and an invader who tries to live off it:

<img class="tfs-figure tfs-light" src="/assets/posts/game-theory-people/forgiveness-light.png" width="760" alt="Left: residents who forgive a defection with probability q earn 3; an invader who defects every other round earns more once q passes 1/3, one who always defects once q passes 1/2. Right: with 5 percent mistakes the residents earn more the more they forgive" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/game-theory-people/forgiveness-dark.png" width="760" alt="Left: residents who forgive a defection with probability q earn 3; an invader who defects every other round earns more once q passes 1/3, one who always defects once q passes 1/2. Right: with 5 percent mistakes the residents earn more the more they forgive" loading="lazy">

An invader who always defects collects $$T$$ whenever it is forgiven and $$P$$ otherwise, $$qT + (1 - q)P$$ per round, and beats the residents' $$R$$ when $$q > (R - P)/(T - P) = 1/2$$.

A cleverer invader defects every other round: it collects $$T$$ on its defections and is forgiven on half of the remaining rounds, and beats the residents when $$q > 1 - (T - R)/(R - S) = 1/3$$. The most forgiving strategy that neither invader can exploit forgives with probability

$$q^* = \min\left(1 - \frac{T - R}{R - S},\; \frac{R - P}{T - P}\right),$$

which with Axelrod's numbers is $$1/3$$<sup class="cite">[<a href="#ref-10" id="cite-14" data-ref="10" aria-label="Reference 10: Molander, P. (1985). The optimal level of generosity in a selfish, uncertain environment. Journal of Conflict Resolution, 29(4), 611–618.">10</a>, <a href="#ref-8" id="cite-15" data-ref="8" aria-label="Reference 8: Nowak, M., &amp; Sigmund, K. (1992). Tit for tat in heterogeneous populations. Nature, 355, 250–253.">8</a>]</sup>. Below $$q^*$$, more forgiveness is pure gain, as the right panel shows: it shortens every echo of every mistake. Above it, forgiveness becomes a subsidy for exploiters.

"Forgive one betrayal in three" is therefore the solution of an optimisation problem, and it moves with the payoffs. When the temptation is large compared with the reward, $$T - R$$ grows, $$q^*$$ shrinks, and the right amount of forgiveness goes down: people with more to gain from betraying you deserve less benefit of the doubt.

When cooperation is very valuable compared with exploitation, you can afford to be more generous, and should be, because every forgiven accident pays.

## Populations that copy what works

A tournament fixes the players. In life, strategies spread: people imitate those who seem to do well, and drop what doesn't work.

A hundred players, each copying a random other player with a probability that grows with how much better that player did, and now and then trying a strategy at random, settle the question of which strategies last. With relationships of twenty rounds on average and one move in fifty going wrong:

<img class="tfs-figure tfs-light" src="/assets/posts/game-theory-people/cycles-light.png" width="760" alt="Shares of strategies in a population of a hundred over time. Without generous tit for tat the population cycles: defectors are replaced by tit for tat, which drifts into unconditional cooperation, which defectors take over again; over twenty runs defectors hold the majority 49% of the time. With generous tit for tat it mostly holds the population, and defectors hold the majority 18% of the time" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/game-theory-people/cycles-dark.png" width="760" alt="Shares of strategies in a population of a hundred over time. Without generous tit for tat the population cycles: defectors are replaced by tit for tat, which drifts into unconditional cooperation, which defectors take over again; over twenty runs defectors hold the majority 49% of the time. With generous tit for tat it mostly holds the population, and defectors hold the majority 18% of the time" loading="lazy">

Without a forgiving strategy the population never settles. Defectors are replaced by tit for tat, because tit for tat players do well among themselves and refuse to be exploited.

In a population of tit for tat, unconditional cooperators do just as well, even slightly better, since mistakes don't start feuds among them, so they drift in unnoticed. Once there are enough of them, defectors return and feast.

Lorens Imhof, Drew Fudenberg and Martin Nowak described this cycle in finite populations, where a small cost of being conditional plays the role that mistakes play here<sup class="cite">[<a href="#ref-11" id="cite-16" data-ref="11" aria-label="Reference 11: Imhof, L. A., Fudenberg, D., &amp; Nowak, M. A. (2005). Evolutionary cycles of cooperation and defection. PNAS, 102(31), 10797–10800.">11</a>]</sup>. In the simulation it runs over and over. Over twenty runs, defectors held the majority 49% of the time.

Add generous tit for tat and the picture changes. It takes over within the first twenty thousand steps and holds the population for most of the run; over twenty runs defectors held the majority 18% of the time, and the average payoff rose from 1.94 to 2.58 points per round.

The cycle carries the least comfortable lesson of the whole subject. Unconditional kindness is not stable: in a world of reciprocators it is harmless and pleasant, and it is exactly what makes the next wave of exploitation possible.

A group protects its cooperators only as long as enough of its members are willing to punish, and the willingness to punish erodes precisely when it is not needed.

## Extortion

In 2012 the physicist William Press and the mathematician Freeman Dyson found something everyone had missed in fifty years of the iterated prisoner's dilemma: strategies that unilaterally fix a linear relation between their own payoff and the opponent's, whatever the opponent does<sup class="cite">[<a href="#ref-12" id="cite-17" data-ref="12" aria-label="Reference 12: Press, W. H., &amp; Dyson, F. J. (2012). Iterated Prisoner&#x27;s Dilemma contains strategies that dominate any evolutionary opponent. PNAS, 109(26), 10409–10413.">12</a>]</sup>. They called them **zero-determinant** strategies.

One family, **extortioners**, guarantees that its surplus over the punishment payoff is $$\chi$$ times the opponent's: $$s_\text{me} - P = \chi\,(s_\text{them} - P)$$. Against an extortioner with $$\chi = 3$$, every extra point you earn by cooperating gives it three.

It sounds like the perfect strategy for a manipulator, and in a single relationship it is: an extortioner never earns less than its opponent. It also rarely earns much.

<img class="tfs-figure tfs-light" src="/assets/posts/game-theory-people/extortion-light.png" width="760" alt="Points per round of an extortioner and of a generous zero-determinant strategy against six opponents, next to their opponents&#x27; points. The extortioner never earns less than its opponent but earns little against anyone who retaliates; the generous strategy never earns more than its opponent and earns 3 against every cooperator" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/game-theory-people/extortion-dark.png" width="760" alt="Points per round of an extortioner and of a generous zero-determinant strategy against six opponents, next to their opponents&#x27; points. The extortioner never earns less than its opponent but earns little against anyone who retaliates; the generous strategy never earns more than its opponent and earns 3 against every cooperator" loading="lazy">

Against a pushover that always cooperates, the extortioner collects 3.73 points a round. Against anyone who answers defection with defection, tit for tat, always defect, it falls to the punishment payoff, because the only way to deny an extortioner its share is to deny it everything.

The mirror image, a **generous** zero-determinant strategy, guarantees the opposite inequality, $$s_\text{me} - R = \chi\,(s_\text{them} - R)$$: it never earns more than its opponent, and it earns the full reward with every cooperator. In the tournament above, the generous one came first and the extortioner eleventh.

Alexander Stewart and Joshua Plotkin put the evolutionary verdict in one sentence: "Extortion strategies perform very well in head-to-head competitions, but they fare poorly in large, evolving populations"<sup class="cite">[<a href="#ref-13" id="cite-18" data-ref="13" aria-label="Reference 13: Stewart, A. J., &amp; Plotkin, J. B. (2013). From extortion to generosity, evolution in the Iterated Prisoner&#x27;s Dilemma. PNAS, 110(38), 15348–15353.">13</a>]</sup>. People reach the same verdict without the mathematics.

In an experiment by Christian Hilbe, Torsten Röhl and Manfred Milinski, players facing an extortion strategy punished it by refusing to cooperate fully, and "although extortioners succeeded against each of their human opponents, extortion resulted in lower payoffs than generosity"<sup class="cite">[<a href="#ref-14" id="cite-19" data-ref="14" aria-label="Reference 14: Hilbe, C., Röhl, T., &amp; Milinski, M. (2014). Extortion subdues human players but is finally punished in the prisoner&#x27;s dilemma. Nature Communications, 5, 3976.">14</a>]</sup>.

The general point is about two different scoreboards. Extortion maximises your share of each relationship; cooperation maximises the total you take home from all of them. Someone who keeps winning arguments, deals and comparisons while the people around them give them the minimum is maximising the first at the expense of the second.

## Reputation

So far every relationship has been a pair that meets again. But most help we give goes to people who will never be in a position to repay it, and much of what we receive comes from strangers.

The biologist Richard Alexander called the mechanism that makes this rational **indirect reciprocity**: "the return is expected from someone other than the recipient of the beneficence"<sup class="cite">[<a href="#ref-15" id="cite-20" data-ref="15" aria-label="Reference 15: Alexander, R. D. (1987). The Biology of Moral Systems. Aldine de Gruyter. He introduced the term in Darwinism and Human Affairs (1979).">15</a>]</sup>.

I help you, and someone else helps me, because they saw or heard what I did. David Haig put the difference in one line: "For direct reciprocity you need a face, for indirect reciprocity you need a name"<sup class="cite">[<a href="#ref-16" id="cite-21" data-ref="16" aria-label="Reference 16: Milinski, M. (2016). Reputation, a universal currency for human social interactions. Philosophical Transactions of the Royal Society B, 371, 20150100.">16</a>]</sup>.

Martin Nowak and Karl Sigmund showed in 1998 that this can work with nothing but a public score: help those with a good image, refuse those with a bad one, and your own image goes up when you help and down when you refuse<sup class="cite">[<a href="#ref-17" id="cite-22" data-ref="17" aria-label="Reference 17: Nowak, M. A., &amp; Sigmund, K. (1998). Evolution of indirect reciprocity by image scoring. Nature, 393, 573–577.">17</a>]</sup>.

This rule, **image scoring**, has a flaw that matters a great deal in practice: a player who refuses to help a known defector loses reputation for doing so, so the people who enforce the norm are punished for enforcing it.

Olof Leimar and Peter Hammerstein showed that a rule called **standing**, which takes into account whom you refused, is far more robust<sup class="cite">[<a href="#ref-18" id="cite-23" data-ref="18" aria-label="Reference 18: Leimar, O., &amp; Hammerstein, P. (2001). Evolution of cooperation through indirect reciprocity. Proceedings of the Royal Society B, 268, 745–753.">18</a>]</sup>:

<img class="tfs-figure tfs-light" src="/assets/posts/game-theory-people/reputation-light.png" width="760" alt="How an observer updates a donor&#x27;s reputation. Image scoring looks only at the act: helping makes the donor good, refusing makes the donor bad. Standing also looks at whom the donor refused: refusing someone with a bad reputation keeps the donor good. The two rules differ only in that one case" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/game-theory-people/reputation-dark.png" width="760" alt="How an observer updates a donor&#x27;s reputation. Image scoring looks only at the act: helping makes the donor good, refusing makes the donor bad. Standing also looks at whom the donor refused: refusing someone with a bad reputation keeps the donor good. The two rules differ only in that one case" loading="lazy">

Hisashi Ohtsuki and Yoh Iwasa then checked all 4096 possible rules of this kind and found that only eight, the "leading eight", keep cooperation stable<sup class="cite">[<a href="#ref-19" id="cite-24" data-ref="19" aria-label="Reference 19: Ohtsuki, H., &amp; Iwasa, Y. (2004). How should we define goodness? Reputation dynamics in indirect reciprocity. Journal of Theoretical Biology, 231(1), 107–120.">19</a>]</sup>. All eight share two features: helping a good person is good and refusing one is bad; and refusing a bad person is good, "because it works as sanction".

Two practical conclusions follow. The first is that how you treat people who can do nothing for you is not a private matter: it is the main evidence others have about you, and reputation-based cooperation runs on exactly this evidence.

Claus Wedekind and Manfred Milinski found that in an experiment where people could give money to others who could never give back, they gave more to those who had been generous before<sup class="cite">[<a href="#ref-20" id="cite-25" data-ref="20" aria-label="Reference 20: Wedekind, C., &amp; Milinski, M. (2000). Cooperation through image scoring in humans. Science, 288, 850–852.">20</a>]</sup>.

The second is a rule for judging others: judge an act together with its target. Someone who refuses to help a person who has been cheating everyone is not unkind, and a community that can't tell the difference will end up punishing its own defenders.

## Leaving

The prisoner's dilemma assumes you are stuck with your partner. Usually you aren't. The option to leave changes the game more than any refinement of tit for tat: a defector who is abandoned after its first defection has to keep finding new partners, and each new partner costs it time and gives it one round of profit at most.

Athena Aktipis showed in simulations that a rule this simple, "move after partner defects", resists invasion by defectors and can even take over a population of them, without any retaliation at all<sup class="cite">[<a href="#ref-21" id="cite-26" data-ref="21" aria-label="Reference 21: Aktipis, C. A. (2004). Know when to walk away: contingent movement and the evolution of cooperation. Journal of Theoretical Biology, 231(2), 249–260.">21</a>]</sup>.

This matters because retaliation is costly in a way the payoff matrix hides. To punish, you must stay in the relationship and defect yourself, which also lowers your own payoff and invites an echo.

Leaving achieves the same deterrence at a lower price, as long as finding a new partner is not much harder than keeping the old one. Where exit is cheap, leaving answers repeated defection better than any counterattack.

## Trust as a coordination problem

Not every social situation is a prisoner's dilemma. In many, nobody gains by betraying anyone; what is risky is relying on someone. Jean-Jacques Rousseau told the canonical story: "Was a deer to be taken?

Every one saw that to succeed he must faithfully stand to his post; but suppose a hare to have slipped by within reach of any one of them, it is not to be doubted but he pursued it without scruple, and when he had seized his prey never reproached himself with having made his companions miss theirs"<sup class="cite">[<a href="#ref-22" id="cite-27" data-ref="22" aria-label="Reference 22: Rousseau, J.-J. (1755). A Discourse upon the Origin and the Foundation of the Inequality among Mankind, part II.">22</a>]</sup>.

<img class="tfs-figure tfs-light" src="/assets/posts/game-theory-people/stag-light.png" width="760" alt="Expected catch of hunting the stag (4 times the share p of others who hunt it) and of hunting hare (3). Hunting the stag pays when more than three quarters of the others hunt it; below that the share of stag hunters falls, above it it rises" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/game-theory-people/stag-dark.png" width="760" alt="Expected catch of hunting the stag (4 times the share p of others who hunt it) and of hunting hare (3). Hunting the stag pays when more than three quarters of the others hunt it; below that the share of stag hunters falls, above it it rises" loading="lazy">

With a stag worth 4 to each hunter if all stay, a hare worth 3, and nothing for staying alone, hunting the stag pays only if more than three quarters of the others will also hunt it.

Unlike the prisoner's dilemma, mutual cooperation here is an equilibrium: nobody wants to leave a successful stag hunt. But so is mutual distrust, and the distrustful equilibrium is the safer of the two, what John Harsanyi and Reinhard Selten called **risk-dominant**<sup class="cite">[<a href="#ref-23" id="cite-28" data-ref="23" aria-label="Reference 23: Harsanyi, J. C., &amp; Selten, R. (1988). A General Theory of Equilibrium Selection in Games. MIT Press.">23</a>]</sup>.

The philosopher Brian Skyrms argued that this game deserves at least as much attention as the prisoner's dilemma: "The Stag Hunt is a story that became a game.

The game is a prototype of the social contract"<sup class="cite">[<a href="#ref-24" id="cite-29" data-ref="24" aria-label="Reference 24: Skyrms, B. (2001). The stag hunt. Proceedings and Addresses of the American Philosophical Association, 75(2), 31–41. The book: The Stag Hunt and the Evolution of Social Structure (2004), Cambridge University Press.">24</a>]</sup>. He also noted the link to everything above: "If the probability of repetition is high enough, the stag hunting equilibrium becomes risk dominant."

The arrows in the chart are the reason that trust in groups is so hard to build and so easy to lose.

Above the threshold, every season of successful cooperation makes cooperation more attractive; below it, every season of caution makes caution more attractive. The same people, with the same values, end up trusting or distrusting each other depending on where they started.

What moves a group across the threshold is evidence that others will hunt the stag: small public commitments, costly signals that only a committed hunter would send, or simply a history.

Thomas Schelling built a theory of bargaining on the observation that "the power to constrain an adversary may depend on the power to bind oneself"<sup class="cite">[<a href="#ref-25" id="cite-30" data-ref="25" aria-label="Reference 25: Schelling, T. C. (1956). An essay on bargaining. American Economic Review, 46(3), 281–306; reprinted in The Strategy of Conflict (1960), ch. 2.">25</a>]</sup>: a promise you cannot take back is worth more than one you can.

## What the models leave out

Every model here gives the players fixed payoffs and asks only how to play. People also choose what to want. Someone who cooperates because they enjoy being trustworthy plays a different game from someone who cooperates because they computed that $$w > 2/3$$, and the first person is much harder to tempt.

In the ultimatum game, one player proposes how to split a sum and the other can accept or let both get nothing. The narrowly rational offer is the smallest possible one, and it is rarely made.

In the first experiment, with economics students in Cologne, the most common proposal was an even split, and the authors concluded that "subjects do not hesitate to punish if their opponent asks for 'too much'"<sup class="cite">[<a href="#ref-26" id="cite-31" data-ref="26" aria-label="Reference 26: Güth, W., Schmittberger, R., &amp; Schwarze, B. (1982). An experimental analysis of ultimatum bargaining. Journal of Economic Behavior &amp; Organization, 3(4), 367–388.">26</a>]</sup>.

Forty years of replications put typical offers at 40 to 50 percent, with acceptance falling fast for offers below 20 percent<sup class="cite">[<a href="#ref-27" id="cite-32" data-ref="27" aria-label="Reference 27: Güth, W., &amp; Kocher, M. G. (2014). More than thirty years of ultimatum bargaining experiments. Journal of Economic Behavior &amp; Organization, 108, 396–409.">27</a>]</sup>.

Joseph Henrich and colleagues ran the game in fifteen small-scale societies. Mean offers, "typically close to 44 percent" in industrial societies, ranged there from 26 to 58 percent, and "the higher the degree of market integration and the higher the payoffs to cooperation, the greater the level of cooperation in experimental games".

Among the Au and Gnau of Papua New Guinea, people "rejected both unfair and hyper-fair (greater than 50 percent) offers with nearly equal frequency"<sup class="cite">[<a href="#ref-28" id="cite-33" data-ref="28" aria-label="Reference 28: Henrich, J., Boyd, R., Bowles, S., Camerer, C., Fehr, E., Gintis, H., &amp; McElreath, R. (2001). In search of homo economicus: behavioral experiments in 15 small-scale societies. American Economic Review, 91(2), 73–78.">28</a>]</sup>: where a gift creates an obligation, too generous an offer is a move in a longer game.

Even the trench truce is contested as an example: Andrew Gelman argued that not shooting simply lowered each side's own risk, so the soldiers may not have faced a prisoner's dilemma at all<sup class="cite">[<a href="#ref-29" id="cite-34" data-ref="29" aria-label="Reference 29: Gelman, A. (2008). Methodology as ideology: some comments on Robert Axelrod&#x27;s The Evolution of Cooperation. QA Rivista dell&#x27;Associazione Rossi-Doria, 2/2008, 167–176.">29</a>]</sup>.

A plausible reading of these results is that human intuitions about fairness were tuned for repeated games with reputations, which is the world the models above describe.

The sense that a low offer must be refused even when no one is watching is the reflex of a player who has always been watched. The models do not replace those intuitions. They explain what the intuitions are for, and when, in an unusual situation, they might be miscalibrated.

## References

1. {:#ref-1} Axelrod, R. (1984). *The Evolution of Cooperation*. Basic Books. Chapter 4, [The live-and-let-live system in trench warfare in World War I](https://gwern.net/doc/economics/1984-axelrod-theevolutionofcooperation-ch4-theliveandletlivesysteminwwi.html), pp. 73–87. <a class="ref-back" href="#cite-1" aria-label="Back to the text">&#8617;</a>
2. {:#ref-2} Ashworth, T. (1980). [*Trench Warfare 1914–1918: The Live and Let Live System*](https://doi.org/10.1007/978-1-349-04356-9). Macmillan. <a class="ref-back" href="#cite-2" aria-label="Back to the text">&#8617;</a>
3. {:#ref-3} Flood, M. M. (1952). [*Some Experimental Games*](https://www.rand.org/pubs/research_memoranda/RM789-1.html). RAND Research Memorandum RM-789-1. <a class="ref-back" href="#cite-3" aria-label="Back to the text">&#8617;</a>
4. {:#ref-4} Axelrod, R., & Hamilton, W. D. (1981). [The evolution of cooperation](https://doi.org/10.1126/science.7466396). *Science*, 211(4489), 1390–1396. <a class="ref-back" href="#cite-6" aria-label="Back to the text">&#8617;</a>
5. {:#ref-5} Axelrod, R. (1980a). [Effective choice in the prisoner's dilemma](https://doi.org/10.1177/002200278002400101). *Journal of Conflict Resolution*, 24(1), 3–25. <a class="ref-back" href="#cite-7" aria-label="Back to the text">&#8617;</a>
6. {:#ref-6} Axelrod, R. (1980b). [More effective choice in the prisoner's dilemma](https://doi.org/10.1177/002200278002400301). *Journal of Conflict Resolution*, 24(3), 379–403. <a class="ref-back" href="#cite-9" aria-label="Back to the text">&#8617;</a>
7. {:#ref-7} Knight, V., Campbell, O., Harper, M., Gaffney, T. J., & Glynatsi, N. E. (2025). [Reviving, reproducing, and revisiting Axelrod's second tournament](https://arxiv.org/abs/2510.15438). arXiv:2510.15438. <a class="ref-back" href="#cite-10" aria-label="Back to the text">&#8617;</a>
8. {:#ref-8} Nowak, M., & Sigmund, K. (1992). [Tit for tat in heterogeneous populations](https://doi.org/10.1038/355250a0). *Nature*, 355, 250–253. <a class="ref-back" href="#cite-12" aria-label="Back to the text">&#8617;</a>
9. {:#ref-9} Nowak, M., & Sigmund, K. (1993). [A strategy of win-stay, lose-shift that outperforms tit-for-tat in the Prisoner's Dilemma game](https://doi.org/10.1038/364056a0). *Nature*, 364, 56–58. <a class="ref-back" href="#cite-13" aria-label="Back to the text">&#8617;</a>
10. {:#ref-10} Molander, P. (1985). [The optimal level of generosity in a selfish, uncertain environment](https://doi.org/10.1177/0022002785029004004). *Journal of Conflict Resolution*, 29(4), 611–618. <a class="ref-back" href="#cite-14" aria-label="Back to the text">&#8617;</a>
11. {:#ref-11} Imhof, L. A., Fudenberg, D., & Nowak, M. A. (2005). [Evolutionary cycles of cooperation and defection](https://doi.org/10.1073/pnas.0502589102). *PNAS*, 102(31), 10797–10800. <a class="ref-back" href="#cite-16" aria-label="Back to the text">&#8617;</a>
12. {:#ref-12} Press, W. H., & Dyson, F. J. (2012). [Iterated Prisoner's Dilemma contains strategies that dominate any evolutionary opponent](https://doi.org/10.1073/pnas.1206569109). *PNAS*, 109(26), 10409–10413. <a class="ref-back" href="#cite-17" aria-label="Back to the text">&#8617;</a>
13. {:#ref-13} Stewart, A. J., & Plotkin, J. B. (2013). [From extortion to generosity, evolution in the Iterated Prisoner's Dilemma](https://doi.org/10.1073/pnas.1306246110). *PNAS*, 110(38), 15348–15353. <a class="ref-back" href="#cite-18" aria-label="Back to the text">&#8617;</a>
14. {:#ref-14} Hilbe, C., Röhl, T., & Milinski, M. (2014). [Extortion subdues human players but is finally punished in the prisoner's dilemma](https://doi.org/10.1038/ncomms4976). *Nature Communications*, 5, 3976. <a class="ref-back" href="#cite-19" aria-label="Back to the text">&#8617;</a>
15. {:#ref-15} Alexander, R. D. (1987). [*The Biology of Moral Systems*](https://openlibrary.org/isbn/0202011739). Aldine de Gruyter. He introduced the term in *Darwinism and Human Affairs* (1979). <a class="ref-back" href="#cite-20" aria-label="Back to the text">&#8617;</a>
16. {:#ref-16} Milinski, M. (2016). [Reputation, a universal currency for human social interactions](https://doi.org/10.1098/rstb.2015.0100). *Philosophical Transactions of the Royal Society B*, 371, 20150100. <a class="ref-back" href="#cite-21" aria-label="Back to the text">&#8617;</a>
17. {:#ref-17} Nowak, M. A., & Sigmund, K. (1998). [Evolution of indirect reciprocity by image scoring](https://doi.org/10.1038/31225). *Nature*, 393, 573–577. <a class="ref-back" href="#cite-22" aria-label="Back to the text">&#8617;</a>
18. {:#ref-18} Leimar, O., & Hammerstein, P. (2001). [Evolution of cooperation through indirect reciprocity](https://doi.org/10.1098/rspb.2000.1573). *Proceedings of the Royal Society B*, 268, 745–753. <a class="ref-back" href="#cite-23" aria-label="Back to the text">&#8617;</a>
19. {:#ref-19} Ohtsuki, H., & Iwasa, Y. (2004). [How should we define goodness? Reputation dynamics in indirect reciprocity](https://doi.org/10.1016/j.jtbi.2004.06.005). *Journal of Theoretical Biology*, 231(1), 107–120. <a class="ref-back" href="#cite-24" aria-label="Back to the text">&#8617;</a>
20. {:#ref-20} Wedekind, C., & Milinski, M. (2000). [Cooperation through image scoring in humans](https://doi.org/10.1126/science.288.5467.850). *Science*, 288, 850–852. <a class="ref-back" href="#cite-25" aria-label="Back to the text">&#8617;</a>
21. {:#ref-21} Aktipis, C. A. (2004). [Know when to walk away: contingent movement and the evolution of cooperation](https://doi.org/10.1016/j.jtbi.2004.06.020). *Journal of Theoretical Biology*, 231(2), 249–260. <a class="ref-back" href="#cite-26" aria-label="Back to the text">&#8617;</a>
22. {:#ref-22} Rousseau, J.-J. (1755). [*A Discourse upon the Origin and the Foundation of the Inequality among Mankind*](https://www.gutenberg.org/ebooks/11136), part II. <a class="ref-back" href="#cite-27" aria-label="Back to the text">&#8617;</a>
23. {:#ref-23} Harsanyi, J. C., & Selten, R. (1988). [*A General Theory of Equilibrium Selection in Games*](https://openlibrary.org/isbn/0262081733). MIT Press. <a class="ref-back" href="#cite-28" aria-label="Back to the text">&#8617;</a>
24. {:#ref-24} Skyrms, B. (2001). [The stag hunt](https://doi.org/10.2307/3218711). *Proceedings and Addresses of the American Philosophical Association*, 75(2), 31–41. The book: [*The Stag Hunt and the Evolution of Social Structure*](https://doi.org/10.1017/CBO9781139165228) (2004), Cambridge University Press. <a class="ref-back" href="#cite-29" aria-label="Back to the text">&#8617;</a>
25. {:#ref-25} Schelling, T. C. (1956). [An essay on bargaining](https://www.jstor.org/stable/1805498). *American Economic Review*, 46(3), 281–306; reprinted in *The Strategy of Conflict* (1960), ch. 2. <a class="ref-back" href="#cite-30" aria-label="Back to the text">&#8617;</a>
26. {:#ref-26} Güth, W., Schmittberger, R., & Schwarze, B. (1982). [An experimental analysis of ultimatum bargaining](https://doi.org/10.1016/0167-2681%2882%2990011-7). *Journal of Economic Behavior & Organization*, 3(4), 367–388. <a class="ref-back" href="#cite-31" aria-label="Back to the text">&#8617;</a>
27. {:#ref-27} Güth, W., & Kocher, M. G. (2014). [More than thirty years of ultimatum bargaining experiments](https://doi.org/10.1016/j.jebo.2014.06.006). *Journal of Economic Behavior & Organization*, 108, 396–409. <a class="ref-back" href="#cite-32" aria-label="Back to the text">&#8617;</a>
28. {:#ref-28} Henrich, J., Boyd, R., Bowles, S., Camerer, C., Fehr, E., Gintis, H., & McElreath, R. (2001). [In search of homo economicus: behavioral experiments in 15 small-scale societies](https://doi.org/10.1257/aer.91.2.73). *American Economic Review*, 91(2), 73–78. <a class="ref-back" href="#cite-33" aria-label="Back to the text">&#8617;</a>
29. {:#ref-29} Gelman, A. (2008). [Methodology as ideology: some comments on Robert Axelrod's *The Evolution of Cooperation*](https://francoangeli.it/riviste/articolo/33851). *QA Rivista dell'Associazione Rossi-Doria*, 2/2008, 167–176. <a class="ref-back" href="#cite-34" aria-label="Back to the text">&#8617;</a>
{: .tfs-refs}

The models, the simulations and the figures: [articles/game_theory](https://github.com/xmarva/transformers-from-scratch/tree/main/articles/game_theory).

{% endraw %}
