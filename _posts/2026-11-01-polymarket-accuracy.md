---
layout: post
title: "How accurate is Polymarket? Calibration, longshot bias and a volume trap"
date: 2026-11-01 00:00:00
description: "Polymarket's prices checked against what happened across almost three million resolved markets: how well calibrated they are, what accuracy figures hide, where longshots and favourites are mispriced, insiders, oracles and the cost of trading"
tags: [prediction-markets, polymarket, calibration, forecasting, trading]
categories: [finance, crypto]
featured: false
thumbnail: /assets/posts/polymarket-accuracy/cover.png
og_image: https://xmarva.github.io/assets/posts/polymarket-accuracy/cover.png
stylesheets: [/assets/css/tfs-course.css]
toc:
  beginning: true
---
{% raw %}

<aside class="tldr" markdown="1">
<p class="tldr-label">In short</p>

Polymarket's prices are good probabilities where people actually trade. A week before the close, markets with real trading are off by under two percentage points on average, and a day before the close they remove 82% of the error of a forecaster who knows only the base rate.

The famous accuracy figures say less than they seem, because most questions resolve No. What mispricing is left sits at the tails, cheap longshots a little too dear and near-certain favourites a little too cheap, and it is about the size of the spread.

<div class="stats">
<div class="stat"><span class="stat-value">1.8</span><span class="stat-label">points between price and outcome, on average, a week before the close</span></div>
<div class="stat"><span class="stat-value">77%</span><span class="stat-label">right for a forecaster who answers No to every question</span></div>
<div class="stat"><span class="stat-value">−39%</span><span class="stat-label">average return on contracts bought at 2 to 5 cents</span></div>
</div>
</aside>

On the last evening of October I opened Polymarket to look at one number. The US midterm elections were three days away, and the market on which party would control the Senate had taken &#36;13.5 million of bets.

A share that pays a dollar if Democrats win the Senate cost 71.5 cents. The House was hardly a contest by the market's account: 94.5 cents for Democrats.

A week earlier the same kind of number had been wrong in public. Going into Brazil's presidential runoff on 25 October, Polymarket and Kalshi gave Flávio Bolsonaro about an 80% chance of beating Lula, and Lula won<sup class="cite">[<a href="#ref-1" id="cite-1" data-ref="1" aria-label="Reference 1: CNBC (2026). Lula wins Brazil election despite prediction markets favoring Bolsonaro. 26 October 2026.">1</a>]</sup>. Semafor's headline that week was that prediction markets had "blown three big elections"<sup class="cite">[<a href="#ref-2" id="cite-2" data-ref="2" aria-label="Reference 2: Semafor (2026). Prediction markets flop on recent elections. 30 October 2026.">2</a>]</sup>.

By the time you read this you probably know how the Senate went, and I don't. Suppose Republicans keep it. Was 71.5 wrong?

Suppose they lose it. Was it right? Neither follows, and Brazil does not prove the 80% wrong either.

A forecast of 71.5% claims something only about *many situations like this one*: that roughly seven in ten of them end the way the market leaned, and three in ten don't.

So the question in the title cannot be settled by any one market, however famous. It can be settled by many. I downloaded the record of every market Polymarket resolved from 2024 on, almost three million of them, took the price histories of about 150,000 and checked what the prices said against what then happened.

This post is about what came out: how good the prices are as probabilities, where they are systematically off, a trap in the most natural way to slice the data, what the "accuracy" figures in the press leave out, and what a trader can and cannot do with any of it.

## What a Polymarket price is, and why it should be a probability

Polymarket sells shares in outcomes. A share of "Yes" pays one dollar if the event happens and nothing if it does not; a share of "No" pays a dollar in the opposite case.

Since April 2026 the dollar is pUSD, Polymarket's own token backed by the USDC stablecoin; before that it was USDC itself<sup class="cite">[<a href="#ref-3" id="cite-3" data-ref="3" aria-label="Reference 3: Polymarket (2026). pUSD. Documentation.">3</a>]</sup>. One Yes and one No together always pay exactly a dollar, so their prices add up to about a dollar too.

Every market is binary. A question with many answers, such as who wins a prize, is a set of Yes/No markets, one per candidate.

Shares trade in an order book, as on a stock exchange: buyers post bids, sellers post asks, and a trade happens when they meet.

The price the site shows, the number the press calls "the odds", is the midpoint between the best bid and the best ask, or the last trade if the gap between them is wider than ten cents<sup class="cite">[<a href="#ref-4" id="cite-4" data-ref="4" aria-label="Reference 4: Polymarket (2026). Prices and the order book. Documentation.">4</a>]</sup>.

<img class="tfs-figure tfs-light" src="/assets/posts/polymarket-accuracy/anatomy-light.png" width="760" alt="A Polymarket share on the US Senate on 31 October 2026: Yes, Democrats control the Senate, at 71.5¢, and No at 28.5¢; together they make a dollar. The best bid is 71¢ and the best ask 72¢. A Yes share bought at 72¢ pays $$1 if Democrats win, a gain of 28¢, and nothing if they lose, a loss of 72¢. It pays off on average only if the chance of a Democratic Senate is above 72%" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/polymarket-accuracy/anatomy-dark.png" width="760" alt="A Polymarket share on the US Senate on 31 October 2026: Yes, Democrats control the Senate, at 71.5¢, and No at 28.5¢; together they make a dollar. The best bid is 71¢ and the best ask 72¢. A Yes share bought at 72¢ pays $$1 if Democrats win, a gain of 28¢, and nothing if they lose, a loss of 72¢. It pays off on average only if the chance of a Democratic Senate is above 72%" loading="lazy">

Why should that price be a probability? Because of a simple argument about money. If you believe Democrats win the Senate with probability $$q$$, a Yes share bought at price $$p$$ is worth $$q \cdot 1 + (1 - q) \cdot 0 = q$$ to you on average.

You buy when $$q > p$$ and sell when $$q < p$$. The price stops moving where nobody with money left is willing to push it either way.

That argument has holes, and economists have spent twenty years mapping them. The price is a compromise between traders, not anyone's belief.

With logarithmic utility it equals the traders' average belief, and for other reasonable preferences it stays close to it between about 20 and 80 cents<sup class="cite">[<a href="#ref-5" id="cite-5" data-ref="5" aria-label="Reference 5: Wolfers, J., &amp; Zitzewitz, E. (2006). Interpreting prediction market prices as probabilities. NBER Working Paper 12200.">5</a>]</sup>; for risk-neutral traders with different beliefs, a price of 75 cents is compatible with an average belief anywhere from 56% to 94%<sup class="cite">[<a href="#ref-6" id="cite-6" data-ref="6" aria-label="Reference 6: Manski, C. F. (2006). Interpreting the predictions of prediction markets. Economics Letters, 91(3), 425–429.">6</a>]</sup>.

Money tied up in a bet until December could have earned interest. Since March 2026 most categories charge the taker a fee<sup class="cite">[<a href="#ref-7" id="cite-7" data-ref="7" aria-label="Reference 7: Polymarket (2026). Trading fees. Documentation.">7</a>]</sup>. And the dollar arrives only if the event is resolved the way you expect, by a process with its own failures.

Whether the price works as a probability despite all that is an empirical question, and the rest of this post answers it with data.

## How to check a forecaster

### Calibration

A forecaster who said 70% is not wrong when the event fails to happen, but a forecaster who says 70% a thousand times should see about seven hundred of those events happen. That property is called *calibration*.

Weather services have been checked this way for seventy years: on the days a calibrated service gives a 30% chance of rain, it rains on about 30% of them.

The picture that shows calibration is the *reliability diagram*. Take all the forecasts, group them by value (all forecasts between 60% and 70%, say), and for each group plot the average forecast against the share of events that happened. A calibrated forecaster's points fall on the diagonal.

If you have ever trained a classifier and checked whether its predicted probabilities can be trusted, you have drawn this picture already; modern neural networks are famously overconfident on it<sup class="cite">[<a href="#ref-8" id="cite-8" data-ref="8" aria-label="Reference 8: Guo, C., Pleiss, G., Sun, Y., &amp; Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML 2017.">8</a>]</sup>, and how to bin the forecasts is a small science of its own<sup class="cite">[<a href="#ref-9" id="cite-9" data-ref="9" aria-label="Reference 9: Dimitriadis, T., Gneiting, T., &amp; Jordan, A. I. (2021). Stable reliability diagrams for probabilistic classifiers. PNAS, 118(8), e2016191118.">9</a>]</sup>.

Calibration is not enough. A forecaster who always announces the *base rate*, the share of all events that happen, is perfectly calibrated and perfectly useless. About 34% of Polymarket's markets resolve Yes, so saying "34%" about every market is calibrated: of all the markets you said it about, 34% happened.

What a good forecaster adds is *resolution*, forecasts that move away from the base rate in the right direction, high for the events that happen and low for those that don't.

### The Brier score

The *Brier score* measures both at once. For $$N$$ forecasts $$p_i$$ of events with outcomes $$y_i$$ (1 if it happened, 0 if not), it is the mean squared error

$$
\mathrm{BS} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2 .
$$

A forecast of 70% scores $$0.3^2 = 0.09$$ if the event happens and $$0.7^2 = 0.49$$ if it does not. A forecaster who always says 50% scores $$0.25$$, and a clairvoyant scores zero. Glenn Brier's original score summed the error over both outcomes and ran from 0 to 2, so it is twice this one<sup class="cite">[<a href="#ref-10" id="cite-10" data-ref="10" aria-label="Reference 10: Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1), 1–3.">10</a>]</sup>.

The score splits into a part for calibration, called reliability, a part for resolution, and a part that belongs to the questions rather than the forecaster. The *skill score* says how much of the worst honest score, that of the base-rate forecaster, a forecaster removes.

<details class="method" markdown="1">
<summary>The Brier score in three parts: Murphy's decomposition</summary>

Allan Murphy showed that the score splits into three parts<sup class="cite">[<a href="#ref-11" id="cite-11" data-ref="11" aria-label="Reference 11: Murphy, A. H. (1973). A new vector partition of the probability score. Journal of Applied Meteorology, 12(4), 595–600.">11</a>]</sup>. Group the forecasts into bins $$k$$, each holding a share $$n_k / N$$ of them, with average forecast $$\bar p_k$$ and observed frequency $$\bar y_k$$, and let $$\bar y$$ be the overall base rate:

$$
\mathrm{BS} \approx \underbrace{\sum_k \frac{n_k}{N} (\bar p_k - \bar y_k)^2}_{\text{reliability}} \;-\; \underbrace{\sum_k \frac{n_k}{N} (\bar y_k - \bar y)^2}_{\text{resolution}} \;+\; \underbrace{\bar y (1 - \bar y)}_{\text{uncertainty}} .
$$

Reliability is the calibration error, the squared distance of the points from the diagonal, and should be small. Resolution is how far the groups' outcomes spread away from the base rate, and should be large. Uncertainty belongs to the questions, not the forecaster: it is the score of the base-rate forecaster. The skill score $$1 - \mathrm{BS} / \bar y(1 - \bar y)$$ says what share of that score a forecaster removes.
</details>

### Three forecasters on the same markets

The three charts below put this on real data. All three forecasters see the same 18,157 markets a week before they closed. The first is Polymarket's prices.

The second is a bolder copy that doubles the prices' log-odds, so that 70% becomes 84% and 30% becomes 16%. The third is a timid copy that halves them, so that 70% becomes 60%. The grey square is the base-rate forecaster.

<img class="tfs-figure tfs-light" src="/assets/posts/polymarket-accuracy/forecasters-light.png" width="760" alt="Three reliability diagrams for the same markets a week before they closed. Polymarket&#x27;s own prices lie close to the diagonal, Brier score 0.079. A bolder copy, with the log-odds doubled, bends away from it, Brier 0.084; a timid copy, with them halved, bends the other way, Brier 0.089. A forecaster who always says the base rate, 25%, is a single point on the diagonal, Brier 0.186" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/polymarket-accuracy/forecasters-dark.png" width="760" alt="Three reliability diagrams for the same markets a week before they closed. Polymarket&#x27;s own prices lie close to the diagonal, Brier score 0.079. A bolder copy, with the log-odds doubled, bends away from it, Brier 0.084; a timid copy, with them halved, bends the other way, Brier 0.089. A forecaster who always says the base rate, 25%, is a single point on the diagonal, Brier 0.186" loading="lazy">

The market's own prices score 0.079. Both copies score worse, 0.084 for the bold one and 0.089 for the timid one, and they fail in opposite ways.

The bold copy's points lie above the diagonal on the left and below it on the right: when it says 5%, the event happens more often than that, because it took a modest forecast and inflated its confidence.

The timid copy bends the other way. Neither copy knows anything the market didn't, and that is the point of the exercise. Distorting a calibrated forecaster in either direction only makes its score worse.

The base-rate forecaster says 25% about everything, the share of these markets that resolved Yes, and scores 0.186, more than twice the market's error.

## Why one election proves nothing

The case people quote as proof is the 2024 US presidential election. On the eve of the vote Polymarket priced a Trump win at 59 cents, while 538's final model gave Kamala Harris 50 chances in 100<sup class="cite">[<a href="#ref-12" id="cite-12" data-ref="12" aria-label="Reference 12: 538 (2024). 538&#x27;s final forecasts for the 2024 presidential election. Republished by 6abc, 5 November 2024.">12</a>]</sup>. Trump won all seven swing states.

<img class="tfs-figure tfs-light" src="/assets/posts/polymarket-accuracy/election-light.png" width="760" alt="Polymarket&#x27;s price for Donald Trump winning the 2024 US presidential election over 2024: near 50 cents in spring, up after the first debate, down to about even after Biden withdrew and Harris entered, rising again in October; 1.53 billion dollars traded" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/polymarket-accuracy/election-dark.png" width="760" alt="Polymarket&#x27;s price for Donald Trump winning the 2024 US presidential election over 2024: near 50 cents in spring, up after the first debate, down to about even after Biden withdrew and Harris entered, rising again in October; 1.53 billion dollars traded" loading="lazy">

The market's best-known trader that year, a Frenchman the press called Théo, had bet more than &#36;30 million on Trump across several accounts.

He told the Wall Street Journal he had "absolutely no political agenda"<sup class="cite">[<a href="#ref-13" id="cite-13" data-ref="13" aria-label="Reference 13: The Wall Street Journal (2024). The Trump betting whale speaks out: &quot;I have absolutely no political agenda&quot;. Republished by To Vima, 1 November 2024.">13</a>]</sup>, and later turned out to have commissioned his own polls, asking people not whom they would vote for but whom their neighbours would<sup class="cite">[<a href="#ref-14" id="cite-14" data-ref="14" aria-label="Reference 14: CBS News (2025). French whale made over &amp;#36;80 million on Polymarket betting on Trump&#x27;s election win. 60 Minutes, 30 November 2025.">14</a>]</sup>. He made tens of millions of dollars, and he was right.

Think about what that shows. A forecaster who put 59% on the winner scores $$0.41^2 \approx 0.17$$ for that election; one who put 50% scores $$0.25$$.

A single draw from a coin that lands heads 59% of the time says almost nothing about the coin, and the same arithmetic run on Brazil a week ago says almost nothing in the other direction. One outcome cannot separate a good forecaster from a lucky one.

Thousands can.

## Three million markets, and which of them are forecasts

Polymarket publishes everything needed to check it. Its Gamma API lists every event and market with its question, its tags, the dollars traded and the final outcome; its order-book API returns the price history of every outcome token.

I took every event that closed from 2024 to the end of October 2026 with at least &#36;1,000 traded: 2,959,534 binary markets that resolved cleanly to Yes or No. (The API records only a few thousand markets before 2024, and I leave them out of the counts.)

About 0.4% of markets with more than &#36;10,000 traded resolved 50-50, the platform's answer to a question that turned out ambiguous, and I set those aside.

### What Polymarket has become

<img class="tfs-figure tfs-light" src="/assets/posts/polymarket-accuracy/population-light.png" width="760" alt="Stacked bars of resolved Polymarket markets by the year they closed and topic: 2024: 12,964, 2025: 193,734, 2026: 2,752,836. 34% of all of them resolved Yes" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/polymarket-accuracy/population-dark.png" width="760" alt="Stacked bars of resolved Polymarket markets by the year they closed and topic: 2024: 12,964, 2025: 193,734, 2026: 2,752,836. 34% of all of them resolved Yes" loading="lazy">

Polymarket in 2026 is mostly a sports book. One football match can carry a dozen markets (the winner, the handicap, the total goals), and sports make up 73% of all the markets that closed this year.

Next come the short crypto markets that ask whether Bitcoin will be higher at the end of the next fifteen minutes, hour or day, 11%, and science and weather questions, 6%. Politics, the subject of nearly every article about prediction markets, is 2% of the count.

Downloading the price history of three million markets is not practical, so I sampled about 150,000 of them, and how to sample turned out to matter more than anything else in this study.

Big markets were drawn more often than small ones, and every pooled figure is weighted back to the whole population; <a class="arrow-link" href="#the-volume-trap-why-big-markets-seem-to-underprice-longshots">a later section</a> shows what goes wrong without that.

<details class="method" markdown="1">
<summary>How the markets were drawn and weighted</summary>

I made two independent random draws. The first takes a share of the markets in each topic, whatever their size: 1.7% of sports, 11% of politics, 16% of economics, 20% of culture, 5.5% of crypto, 3.5% of science and weather.

The second takes every market with more than &#36;100,000 traded (every sports market with more than a million) and a share of the smaller ones above &#36;10,000. Together they hold 153,602 markets in 80,320 events.

Every market had a known chance of being drawn, and when I pool markets each counts in inverse proportion to that chance, so the pooled figures describe all resolved markets rather than my sample; statisticians call this the *Horvitz–Thompson estimator*<sup class="cite">[<a href="#ref-15" id="cite-15" data-ref="15" aria-label="Reference 15: Horvitz, D. G., &amp; Thompson, D. J. (1952). A generalization of sampling without replacement from a finite universe. Journal of the American Statistical Association, 47(260), 663–685.">15</a>]</sup>.

Drawing big markets more often is harmless as long as the weights undo it. Leaving markets out, or grouping them by size, is not. Uncertainty comes from a bootstrap over events rather than markets, because the twenty candidates of one election are not twenty independent questions.

Market lists come from the Gamma API (`events/keyset`), prices from the order-book API (`prices-history`, an hourly series for the last two weeks of a market and a 12-hour one for its whole life), and trades from the data API (`trades`).
</details>

### What counts as a forecast

The forecast I score is the price of a market's first outcome (Yes, in a Yes/No market) a fixed time before the market closed: a week, mostly, and a day, a month or three months where I compare horizons.

The close comes after the outcome is known, typically hours after a match ends or a day or two after an election, so a week before the close is before the event itself.

A market counts as a forecast only if at least &#36;1,000 had been traded in it in the week before. This matters more than it sounds. Most markets on Polymarket are created, barely traded and resolved, and the price history of such a market is the middle of an almost empty order book.

One football market in my sample, on whether a Bundesliga match would have five goals or more, stood at 48 cents a week before the game; most matches end with fewer, this one did too, and over its whole life the market traded &#36;134. That price is not a forecast of a high-scoring game.

A bid at one cent and an ask at 95 cents would give it. The test uses only what was known at the time, the trades of that week, which I took from Polymarket's public trades API.

## Calibration: a week out, prices are off by 1.8 points

Here is the main result.

<img class="tfs-figure tfs-light" src="/assets/posts/polymarket-accuracy/calibration-light.png" width="760" alt="Reliability diagram: markets grouped by their price a week before closing, against the share that resolved Yes: priced 1%, happened 0%; priced 3%, happened 2%; priced 7%, happened 9%; priced 14%, happened 15%; priced 25%, happened 28%; priced 35%, happened 36%; priced 45%, happened 46%; priced 55%, happened 60%; priced 65%, happened 70%; priced 75%, happened 82%; priced 85%, happened 89%; priced 93%, happened 92%; priced 97%, happened 99%; priced 99%, happened 100%. Next to it, a histogram of those prices" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/polymarket-accuracy/calibration-dark.png" width="760" alt="Reliability diagram: markets grouped by their price a week before closing, against the share that resolved Yes: priced 1%, happened 0%; priced 3%, happened 2%; priced 7%, happened 9%; priced 14%, happened 15%; priced 25%, happened 28%; priced 35%, happened 36%; priced 45%, happened 46%; priced 55%, happened 60%; priced 65%, happened 70%; priced 75%, happened 82%; priced 85%, happened 89%; priced 93%, happened 92%; priced 97%, happened 99%; priced 99%, happened 100%. Next to it, a histogram of those prices" loading="lazy">

The points sit close to the diagonal. Of the 18,157 markets with real trading a week before the close, those priced around 25 cents resolved Yes 28% of the time and those around 75 cents 82%.

The average gap between price and frequency is 1.8 percentage points, and the reliability term of the Brier score is 0.001, a hundredth of the resolution term, 0.108. A weather service would be pleased with that.

<p class="takeaway" markdown="1">When a liquid market says 70%, read it as about 70%, and remember that 70 means it fails three times in ten.</p>

The small deviations have a shape. Between 55 and 85 cents events happened a little more often than priced (60% of the markets at 55 cents, 82% of those at 75), while contracts at 2 to 5 cents happened less often than priced, 2.3% of the time against a price of 3.2. The prices are slightly compressed toward the middle: favourites a little too cheap, longshots a little too dear.

That is the pattern economic theory predicts for traders who like long shots or dislike tying up money, and it is what a Princeton thesis on 188,509 Polymarket markets found too<sup class="cite">[<a href="#ref-16" id="cite-16" data-ref="16" aria-label="Reference 16: Walker, A. (2026). The price is (almost) right: a calibration analysis of Polymarket pricing data. Senior thesis, Princeton University.">16</a>]</sup>.

The right panel shows something just as important: where the forecasts are. Almost half of these markets are priced below 5 cents a week before they close.

Most questions on Polymarket are of the form "will this specific thing happen", most of them don't happen, and the market knows. Those confident, correct and uninteresting forecasts are a large part of every headline accuracy figure.

## Accuracy against a forecaster who always says no

Polymarket advertises its record on a page titled "How accurate is Polymarket?": 98.7% accurate four hours before resolution, 90.3% a month before<sup class="cite">[<a href="#ref-17" id="cite-17" data-ref="17" aria-label="Reference 17: Polymarket (2026). How accurate is Polymarket? Accessed 31 October 2026.">17</a>]</sup>. Accuracy there means the share of markets where the leading outcome won.

It sounds like a hard bar to clear. The same page notes that 78.3% of its markets resolve No, and explains why: "There are usually more ways for something not to happen than to happen in one exact way".

A forecaster who answers "No" to every question without reading it is therefore 78% accurate on Polymarket's own sample. The left half of the next chart shows what that does to the comparison on mine.

<img class="tfs-figure tfs-light" src="/assets/posts/polymarket-accuracy/accuracy-light.png" width="760" alt="The same 5,420 markets a month, a week and a day before they closed. Left: share where the side priced above 50 cents won, against the share a forecaster who always says No gets right: 1 month before 87% against 77%; 1 week before 93% against 77%; 1 day before 96% against 77%. Right: Brier skill over the base rate: 1 month 48% for all, 40% for those priced 2 to 98 cents; 1 week 70% for all, 54% for those priced 2 to 98 cents; 1 day 82% for all, 57% for those priced 2 to 98 cents" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/polymarket-accuracy/accuracy-dark.png" width="760" alt="The same 5,420 markets a month, a week and a day before they closed. Left: share where the side priced above 50 cents won, against the share a forecaster who always says No gets right: 1 month before 87% against 77%; 1 week before 93% against 77%; 1 day before 96% against 77%. Right: Brier skill over the base rate: 1 month 48% for all, 40% for those priced 2 to 98 cents; 1 week 70% for all, 54% for those priced 2 to 98 cents; 1 day 82% for all, 57% for those priced 2 to 98 cents" loading="lazy">

The chart uses the 5,420 markets that had real trading a month, a week and a day before they closed, so the same questions are scored at every horizon.

The favourite won 87% of them a month before the close, 93% a week before and 96% a day before. A forecaster who says No to everything gets 77% right on these markets at every horizon, since only 23% of them resolved Yes.

The honest comparison is skill: the market removes 48% of the base-rate forecaster's Brier score a month out, 70% a week out and 82% a day out. On the questions still in doubt at the time, priced between 2 and 98 cents, it removes 40%, 54% and 57%.

<div class="callout callout-key" markdown="1">
<p class="callout-label">Key point</p>

Accuracy flatters any forecaster whose questions mostly resolve one way. Ask for skill: the share of the base-rate forecaster's error that is gone. On Polymarket it is 70% a week before the close, and 54% on the questions still in doubt.
</div>

Those are the numbers to keep in mind the next time anyone, a platform or a pundit, advertises 90% accuracy.

## How the forecast sharpens as the close approaches

Information arrives over time. The Brier score of markets that lived at least two months traces how quickly.

<img class="tfs-figure tfs-light" src="/assets/posts/polymarket-accuracy/convergence-light.png" width="760" alt="Brier score of 3,492 markets, each open for at least 60 days, by days before the close: 0.084 at 60 days, 0.056 at 30, 0.033 at 7 and 0.021 at half a day, against 0.150 for always saying the base rate. The share priced below 1 cent or above 99 rises from 26% to 75%" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/polymarket-accuracy/convergence-dark.png" width="760" alt="Brier score of 3,492 markets, each open for at least 60 days, by days before the close: 0.084 at 60 days, 0.056 at 30, 0.033 at 7 and 0.021 at half a day, against 0.150 for always saying the base rate. The share priced below 1 cent or above 99 rises from 26% to 75%" loading="lazy">

Sixty days before the close, the 3,492 markets that lived that long and were traded a month out already score 0.084, well below the base rate's 0.150.

The score falls steadily, to 0.056 a month out and 0.033 a week out, and drops again in the last days, when scheduled events (a vote, a match, a data release) take place.

The lower panel counts the markets already priced at the extremes, below a cent or above 99 cents: a quarter of them at sixty days, more than half a week out, three quarters at the end. A market at 99.5 cents is not a forecast any more; it is a receipt waiting to be cashed.

## The volume trap: why big markets seem to underprice longshots

The obvious next question is whether big markets forecast better than small ones, and the obvious way to answer it is to group markets by the dollars traded. Polymarket's accuracy page does exactly this. It is a trap, and I walked into it first.

My first sample took every market with more than &#36;100,000 traded and a share of those between &#36;10,000 and &#36;100,000, the way one naturally filters out dead markets.

In that sample, contracts priced between 10 and 20 cents a week before the close returned 25% on average, and the bigger the market, the more the longshots won.

It looked like the reverse of everything the racetrack literature says, and a trading strategy in one line. Here are the longshots of the sample drawn by topic, grouped two ways.

<img class="tfs-figure tfs-light" src="/assets/posts/polymarket-accuracy/trap-light.png" width="760" alt="Contracts priced 2 to 20 cents a week before the close: how often they won divided by their price, grouped by the market&#x27;s total volume and by the dollars traded in it in the week before. By total volume: under $$1K 0.49; $$1K–10K 0.60; $$10K–100K 1.13; $$100K–1M 1.34; $$1M–10M 2.14. By the week before: under $$100 0.72; $$100–1K 0.87; $$1K–10K 0.96; $$10K–100K 0.81" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/polymarket-accuracy/trap-dark.png" width="760" alt="Contracts priced 2 to 20 cents a week before the close: how often they won divided by their price, grouped by the market&#x27;s total volume and by the dollars traded in it in the week before. By total volume: under $$1K 0.49; $$1K–10K 0.60; $$10K–100K 1.13; $$100K–1M 1.34; $$1M–10M 2.14. By the week before: under $$100 0.72; $$100–1K 0.87; $$1K–10K 0.96; $$10K–100K 0.81" loading="lazy">

On the left, grouped by the dollars a market traded over its whole life, contracts priced 2 to 20 cents won about half as often as priced in markets under &#36;1,000 (a ratio of 0.49) and twice as often in markets of &#36;1–10 million (2.14).

On the right, the same contracts grouped by the dollars traded in the week before the forecast show *nothing of the kind*: 0.72 in the thinnest markets, 0.87 and 0.96 as trading grows. Favourites mirror it, from 1.05 to 0.88 on the left and flat at about 1.0 on the right.

The left panel is an artefact, and the reason is in what volume counts. A market's total volume includes all the trading after the forecast, and nothing makes people trade like a surprise.

When a 10-cent candidate starts winning, the market fills with buyers, sellers, hedgers and spectators, and its volume jumps; when the favourite wins as expected, the market stays quiet and small.

<p class="flow"><span>a longshot starts winning</span><span>everyone trades it</span><span>its volume grows</span><span>"big markets underprice longshots"</span></p>

So sorting markets by total volume quietly sorts them by whether the unlikely happened. In statistics this is *selection on a consequence of the outcome*, the same family of error as look-ahead bias in a backtest, which I wrote about in the post on [AI crypto trading bots](/blog/2026/ai-crypto-trading/). Polymarket's own accuracy page breaks its Brier scores down by volume in exactly this way.

<div class="callout callout-warn" markdown="1">
<p class="callout-label">Trap</p>

Never filter, group or weight markets by their total volume, or by anything else that is counted after the forecast. Use what was known when the price was set: the dollars traded in the week before, the order book at the time, the topic.
</div>

The right panel is the honest version, because *the week before a forecast is over when the forecast is made*. It says something useful.

In thin markets longshots are overpriced, and the more a market trades, the closer to fair they become. That is why the rest of this post, like the calibration above, uses the test of &#36;1,000 traded in the week before and never groups markets by their total volume.

## Longshots lose, favourites earn a little

Calibration is an average, and averages hide the tails. The tails are where traders make or lose money, and economists have known for seventy years where to look: at the racetrack, bettors overpay for longshots and underpay for favourites<sup class="cite">[<a href="#ref-18" id="cite-18" data-ref="18" aria-label="Reference 18: Griffith, R. M. (1949). Odds adjustments by American horse-race bettors. American Journal of Psychology, 62(2), 290–294.">18</a>]</sup>.

In six million American horse races, betting on horses at 100 to 1 or longer lost about 61% of the stake, while betting on the favourite in every race lost only 5.5%<sup class="cite">[<a href="#ref-19" id="cite-19" data-ref="19" aria-label="Reference 19: Snowberg, E., &amp; Wolfers, J. (2010). Explaining the favorite–long shot bias: is it risk-love or misperceptions? Journal of Political Economy, 118(4), 723–746.">19</a>]</sup>. On Kalshi, contracts bought below 10 cents lost over 60% of the money put into them<sup class="cite">[<a href="#ref-20" id="cite-20" data-ref="20" aria-label="Reference 20: Bürgi, C., Deng, W., &amp; Whelan, K. (2026). Makers and takers: the economics of the Kalshi prediction market. GWU Center for Economic Research Working Paper 2026-001.">20</a>]</sup>.

To look for it on Polymarket, I treated both sides of every market as a contract: a Yes at the price $$p$$ and a No at $$1 - p$$. Buying a contract at price $$p$$ that pays $$y \in \{0, 1\}$$ returns $$y / p - 1$$ on the stake. If prices were exact probabilities, the average return in every price band would be zero.

<img class="tfs-figure tfs-light" src="/assets/posts/polymarket-accuracy/returns-light.png" width="760" alt="Average return of buying a contract a week before the close, by price band: 0.5–2¢ -22.5%; 2–5¢ -38.8%; 5–10¢ +19.4%; 10–20¢ -2.5%; 20–30¢ -4.1%; 30–40¢ -3.8%; 40–50¢ -3.9%; 50–60¢ +3.3%; 60–70¢ +2.9%; 70–80¢ +0.3%; 80–90¢ +0.0%; 90–95¢ -1.4%; 95–98¢ +1.1%; 98–99.5¢ +0.3%" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/polymarket-accuracy/returns-dark.png" width="760" alt="Average return of buying a contract a week before the close, by price band: 0.5–2¢ -22.5%; 2–5¢ -38.8%; 5–10¢ +19.4%; 10–20¢ -2.5%; 20–30¢ -4.1%; 30–40¢ -3.8%; 40–50¢ -3.9%; 50–60¢ +3.3%; 60–70¢ +2.9%; 70–80¢ +0.3%; 80–90¢ +0.0%; 90–95¢ -1.4%; 95–98¢ +1.1%; 98–99.5¢ +0.3%" loading="lazy">

The bias is there, and it lives at the extremes. Contracts priced between 2 and 5 cents a week before the close won 2.1% of the time while costing 3.2 cents on average, so buying them lost 39% of the stake, with a 95% interval from −53% to −22%.

The cheapest band, under 2 cents, lost 22%, with an interval wide enough to include zero.

At the other end, contracts at 95 to 98 cents won 97.8% of the time and returned 1.15% in the week, and those at 98 to 99.5 cents returned 0.35%; both intervals stay above zero. Between 5 and 95 cents the averages cannot be told apart from zero.

<p class="takeaway" markdown="1">Longshots under 5 cents lose a quarter to two fifths of the stake; favourites above 95 cents earn a fraction of a percent a week, before costs.</p>

<details class="method" markdown="1">
<summary>How the returns are counted, and why the weighting matters</summary>

How contracts are weighted matters for this kind of result. A study of 588 million Polymarket trades found that buys under 10 cents lost 19.3 cents per dollar when weighted by the money traded, 6.3 cents when each contract counted once, and gained 4.1 cents when related contracts were first grouped by event<sup class="cite">[<a href="#ref-21" id="cite-21" data-ref="21" aria-label="Reference 21: Cardozo, M., &amp; Rivero-Wildemauwe, J. I. (2026). The favorite-longshot bias in prediction markets: evidence from Polymarket. arXiv:2609.12878.">21</a>]</sup>.

My numbers count each market once, weighted to the population of markets, with uncertainty from resampling whole events.

A return here is that of one contract bought at the mid price a week before the close and held to the end, averaged over contracts in a price band with the population weights; the interval resamples whole events 400 times.
</details>

A favourite at 97 cents that earns a percent in a week sounds like free money, and it is not. The bars are mid prices, and nobody buys at the mid.

The downside is a total loss on the rare occasion the favourite fails, and the 50-50 resolutions I excluded would cost a favourite buyer half the stake; at about 0.4% of markets they take back up to 0.2 points of that return.

## What the spread and the fees take

The price shown is the middle of the order book, but a buyer pays the ask. On the last night of October I recorded the best bid and ask of all 124,068 open markets, 57,468 of them with orders on both sides, and measured what buying at the ask costs above the middle, as a share of the price paid.

<img class="tfs-figure tfs-light" src="/assets/posts/polymarket-accuracy/spread-light.png" width="760" alt="Median cost of buying at the ask instead of the middle of the book, as a share of the price, by contract price: 0.5–2¢ 66.7%; 2–5¢ 66.1%; 5–10¢ 29.4%; 10–20¢ 13.0%; 20–30¢ 6.7%; 30–40¢ 4.8%; 40–50¢ 2.5%; 50–60¢ 2.0%; 60–70¢ 2.4%; 70–80¢ 2.1%; 80–90¢ 2.3%; 90–95¢ 2.2%; 95–98¢ 2.1%; 98–99.5¢ 0.6%" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/polymarket-accuracy/spread-dark.png" width="760" alt="Median cost of buying at the ask instead of the middle of the book, as a share of the price, by contract price: 0.5–2¢ 66.7%; 2–5¢ 66.1%; 5–10¢ 29.4%; 10–20¢ 13.0%; 20–30¢ 6.7%; 30–40¢ 4.8%; 40–50¢ 2.5%; 50–60¢ 2.0%; 60–70¢ 2.4%; 70–80¢ 2.1%; 80–90¢ 2.3%; 90–95¢ 2.2%; 95–98¢ 2.1%; 98–99.5¢ 0.6%" loading="lazy">

For contracts between 40 and 95 cents the median cost is about 2% of the price. Below 20 cents it explodes: 13% for contracts at 10 to 20 cents, 29% at 5 to 10, and about two thirds of the price under 5 cents.

A two-cent contract with a bid at one cent and an ask at three costs a full cent above its middle, half of what it is worth on paper. At the other end, near-certain contracts above 98 cents are cheap to buy, 0.6% of the price.

On top of the spread comes the fee. Since 30 March 2026 a taker on most categories pays $$r \cdot p(1 - p)$$ per share, where $$p$$ is the price and the rate $$r$$ is 0.04 for politics, 0.05 for sports and most other categories and 0.07 for crypto; makers pay nothing, and geopolitical markets are free<sup class="cite">[<a href="#ref-7" id="cite-22" data-ref="7" aria-label="Reference 7: Polymarket (2026). Trading fees. Documentation.">7</a>]</sup>.

As a share of the money paid that is $$r(1 - p)$$: about 5% of the stake for a 5-cent sports contract, a quarter of a percent for a 95-cent one.

Many of the busiest markets ran with the fee switched off on the day I looked, so the rate has to be checked market by market.

Put together, a longshot buyer who crosses the spread gives up between a tenth and two thirds of the stake before the event even happens.

A favourite buyer gives up a couple of percent, more than the whole edge the previous chart found: after half the spread the 95–98 cent band goes from +1.15% to −0.9% a week, and the 98–99.5 cent band from +0.35% to −0.3%. The diamonds on that chart show it.

<div class="callout" markdown="1">
<p class="callout-label">For traders</p>

The longshot discount and the favourite premium are real, but they are about the size of what it costs to trade them, which is why they survive. Selling longshots and buying near-certainties works only for those who post limit orders and let others cross the spread.
</div>

## Topics: politics is calibrated best

<img class="tfs-figure tfs-light" src="/assets/posts/polymarket-accuracy/topics-light.png" width="760" alt="Brier skill and calibration error a week before the close, by topic: Politics and world skill 58%, error 1.9 points; Sports skill 53%, error 4.4 points; Crypto prices and events skill 54%, error 3.9 points; Economy and markets skill 60%, error 3.3 points; Culture and media skill 59%, error 2.3 points; Science, tech, weather skill 70%, error 4.1 points" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/polymarket-accuracy/topics-dark.png" width="760" alt="Brier skill and calibration error a week before the close, by topic: Politics and world skill 58%, error 1.9 points; Sports skill 53%, error 4.4 points; Crypto prices and events skill 54%, error 3.9 points; Economy and markets skill 60%, error 3.3 points; Culture and media skill 59%, error 2.3 points; Science, tech, weather skill 70%, error 4.1 points" loading="lazy">

A week before the close, markets with real trading are skilled in every topic, from about 53% of the base rate's error removed in sports and crypto to 70% in science and weather.

Politics, with 8,929 markets the largest group, is also the best calibrated, with an average gap of 1.9 points; sports and science have the largest gaps, 4.4 and 4.1, with crypto close behind at 3.9. Sports here means the markets traded a week before the close, mostly season futures (who wins the league, who wins the award) rather than single games, which are listed only days before kick-off.

### Crypto markets resolve together

Crypto deserves a chart of its own, because its errors have a cause that has nothing to do with the traders' judgement.

<img class="tfs-figure tfs-light" src="/assets/posts/polymarket-accuracy/crypto-light.png" width="760" alt="Two panels by month. Top: among crypto price markets priced 30 to 70 cents a week before the close, the share where the side betting on rising prices won, against that side&#x27;s average price. Bottom: Bitcoin&#x27;s return that month. The two move together: 2024-12: 29% of 17, Bitcoin -3%; 2025-01: 58% of 12, Bitcoin +9%; 2025-02: 39% of 18, Bitcoin -18%; 2025-03: 31% of 16, Bitcoin -2%; 2025-04: 67% of 12, Bitcoin +14%; 2025-05: 73% of 15, Bitcoin +11%; 2025-06: 40% of 10, Bitcoin +2%; 2025-07: 75% of 16, Bitcoin +8%; 2025-09: 60% of 10, Bitcoin +5%; 2025-10: 20% of 10, Bitcoin -4%; 2025-11: 29% of 14, Bitcoin -18%; 2026-01: 41% of 17, Bitcoin -10%; 2026-02: 30% of 10, Bitcoin -15%; 2026-04: 67% of 15, Bitcoin +12%; 2026-09: 50% of 10, Bitcoin +6%" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/polymarket-accuracy/crypto-dark.png" width="760" alt="Two panels by month. Top: among crypto price markets priced 30 to 70 cents a week before the close, the share where the side betting on rising prices won, against that side&#x27;s average price. Bottom: Bitcoin&#x27;s return that month. The two move together: 2024-12: 29% of 17, Bitcoin -3%; 2025-01: 58% of 12, Bitcoin +9%; 2025-02: 39% of 18, Bitcoin -18%; 2025-03: 31% of 16, Bitcoin -2%; 2025-04: 67% of 12, Bitcoin +14%; 2025-05: 73% of 15, Bitcoin +11%; 2025-06: 40% of 10, Bitcoin +2%; 2025-07: 75% of 16, Bitcoin +8%; 2025-09: 60% of 10, Bitcoin +5%; 2025-10: 20% of 10, Bitcoin -4%; 2025-11: 29% of 14, Bitcoin -18%; 2026-01: 41% of 17, Bitcoin -10%; 2026-02: 30% of 10, Bitcoin -15%; 2026-04: 67% of 15, Bitcoin +12%; 2026-09: 50% of 10, Bitcoin +6%" loading="lazy">

Most crypto markets ask about one number, the price of Bitcoin or Ether on a date. Take the close calls, markets priced between 30 and 70 cents a week out, and orient each so that Yes means the price went up.

In the months when Bitcoin rose, the bet on rising prices won 63% of the time; in the months when it fell, 32%. Its price a week before was about 50 cents in both. Across fifteen months, the share of winning bullish bets and Bitcoin's monthly return have a correlation of 0.78.

The market was not wrong in any useful sense: a week before the close nobody knew which way Bitcoin would go, and 50 cents was a fair price for a coin toss. But a dozen such markets in one month are not a dozen tosses. They are one toss recorded a dozen times.

A calibration chart of crypto markets mostly shows whether the period studied was a good one for Bitcoin. Calibration needs independent questions, which is why I resample whole events rather than markets, and why even that understates the uncertainty for crypto.

## Long-dated favourites and the price of waiting

A market that resolves in six months has to pay its traders for waiting. A No share at 95 cents on a question that will almost certainly resolve No pays 5 cents on 95 after half a year, about 10.8% a year, if it pays at all; three-month US Treasury bills paid 4.07% on average in 2025 and 3.71% so far in 2026<sup class="cite">[<a href="#ref-22" id="cite-23" data-ref="22" aria-label="Reference 22: Federal Reserve Bank of St. Louis (2026). 3-month Treasury bill secondary market rate (DTB3). FRED.">22</a>]</sup>.

Traders who could earn interest elsewhere will not bid near-certain long-dated contracts all the way to a dollar, and the people with the most money to commit, favourite buyers, drop out first.

Page and Clemen found the result on the old Intrade exchange: markets far from expiry were poorly calibrated, with prices "systematically biased toward 0.50"<sup class="cite">[<a href="#ref-23" id="cite-24" data-ref="23" aria-label="Reference 23: Page, L., &amp; Clemen, R. T. (2013). Do prediction markets produce well-calibrated probability forecasts? The Economic Journal, 123(568), 491–513.">23</a>]</sup>.

Polymarket noticed. Since September 2025 it has paid interest on positions in a short list of long-dated markets "to keep long-term pricing accurate", 4% a year at launch and 3.25% since June 2026<sup class="cite">[<a href="#ref-24" id="cite-25" data-ref="24" aria-label="Reference 24: Polymarket (2026). Holding rewards. Help center, 1 June 2026.">24</a>]</sup>.

The list holds about eighteen events, the 2028 presidential race and the midterms among them. Every other market pays nothing for waiting.

<img class="tfs-figure tfs-light" src="/assets/posts/polymarket-accuracy/long-light.png" width="760" alt="Contracts priced above 90 cents one, three and six months before the close: their price, how often they won and what holding them to the end returned per year: 1 month 90–95¢: price 92.6%, won 94.3%, +25.0% a year; 1 month 95–98¢: price 96.6%, won 98.4%, +23.9% a year; 1 month 98–99.5¢: price 98.8%, won 99.3%, +6.4% a year; 3 months 90–95¢: price 92.6%, won 94.7%, +9.3% a year; 3 months 95–98¢: price 96.7%, won 98.1%, +6.2% a year; 3 months 98–99.5¢: price 98.9%, won 97.7%, -4.9% a year; 6 months 90–95¢: price 92.7%, won 97.0%, +9.3% a year; 6 months 95–98¢: price 96.6%, won 93.3%, -6.8% a year; 6 months 98–99.5¢: price 98.9%, won 98.7%, -0.4% a year" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/polymarket-accuracy/long-dark.png" width="760" alt="Contracts priced above 90 cents one, three and six months before the close: their price, how often they won and what holding them to the end returned per year: 1 month 90–95¢: price 92.6%, won 94.3%, +25.0% a year; 1 month 95–98¢: price 96.6%, won 98.4%, +23.9% a year; 1 month 98–99.5¢: price 98.8%, won 99.3%, +6.4% a year; 3 months 90–95¢: price 92.6%, won 94.7%, +9.3% a year; 3 months 95–98¢: price 96.7%, won 98.1%, +6.2% a year; 3 months 98–99.5¢: price 98.9%, won 97.7%, -4.9% a year; 6 months 90–95¢: price 92.7%, won 97.0%, +9.3% a year; 6 months 95–98¢: price 96.6%, won 93.3%, -6.8% a year; 6 months 98–99.5¢: price 98.9%, won 98.7%, -0.4% a year" loading="lazy">

Polymarket shows the same thing. Contracts priced between 90 and 95 cents a month before the close won 94.3% of the time against a price of 92.6, and those at 95 to 98 cents won 98.4% against 96.6.

Held to the end, both earned about 1.8% in a month, which compounds to roughly 25% a year, six times what a Treasury bill paid.

Three months out, the same two bands earned 6 to 9% a year and the most expensive band lost money; six months out the samples are small and the results noisy.

<p class="takeaway" markdown="1">A month before the close, favourites at 90 to 98 cents paid about 25% a year at mid prices: the price of waiting, not of ignorance.</p>

The pattern fits Page and Clemen: near-certain long-dated contracts are a little too cheap, and the discount is about what impatient money charges. It is not free money either.

These are mid prices, a single failure costs the whole stake, and a trader who buys favourites a month out is lending to the market at a rate that has to cover the risk that the near-certain doesn't happen.

## A two-number correction that doesn't help

An engineer looking at the calibration chart will ask the obvious question: if the market is slightly off in a systematic way, can a model correct it? This is the same move as calibrating a classifier. Fit two numbers, $$a$$ and $$b$$, so that a corrected probability

$$
\hat q = \sigma\big(a \cdot \operatorname{logit}(p) + b\big), \qquad \operatorname{logit}(p) = \ln \frac{p}{1 - p}, \quad \sigma(z) = \frac{1}{1 + e^{-z}},
$$

fits the outcomes better. With $$b = 0$$, a slope $$a > 1$$ means the prices should have been bolder, longshots cheaper and favourites dearer, and $$a < 1$$ means they should have been more timid. This is *Platt scaling*, a standard way to recalibrate a classifier<sup class="cite">[<a href="#ref-25" id="cite-26" data-ref="25" aria-label="Reference 25: Niculescu-Mizil, A., &amp; Caruana, R. (2005). Predicting good probabilities with supervised learning. ICML 2005.">25</a>]</sup>.

A Princeton thesis that fitted it to 188,509 Polymarket markets found a slope of 1.112<sup class="cite">[<a href="#ref-16" id="cite-27" data-ref="16" aria-label="Reference 16: Walker, A. (2026). The price is (almost) right: a calibration analysis of Polymarket pricing data. Senior thesis, Princeton University.">16</a>]</sup>, and a study of Kalshi and Polymarket found political markets "compressed toward 50%" in the same sense<sup class="cite">[<a href="#ref-26" id="cite-28" data-ref="26" aria-label="Reference 26: Le, N. A. (2026). Decomposing crowd wisdom: domain-specific calibration dynamics in prediction markets. arXiv:2602.19520.">26</a>]</sup>.

The interesting question is whether such a correction helps on markets it has not seen. I fitted $$a$$ and $$b$$ only on markets that closed before July 2025 and scored the correction on markets that closed after.

<img class="tfs-figure tfs-light" src="/assets/posts/polymarket-accuracy/recalibration-light.png" width="760" alt="The fitted correction from Polymarket&#x27;s price to a probability, against the diagonal, for one week and one month before the close: 1 week: slope 1.12, Brier on later markets 0.0796 before and 0.0797 after; 1 month: slope 1.18, Brier on later markets 0.0758 before and 0.0761 after" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/polymarket-accuracy/recalibration-dark.png" width="760" alt="The fitted correction from Polymarket&#x27;s price to a probability, against the diagonal, for one week and one month before the close: 1 week: slope 1.12, Brier on later markets 0.0796 before and 0.0797 after; 1 month: slope 1.18, Brier on later markets 0.0758 before and 0.0761 after" loading="lazy">

Fitted on the markets that closed before July 2025, the correction has a slope of 1.12 a week before the close and 1.18 a month before: the prices should have been a little bolder, the same compression the calibration chart showed, and close to Walker's 1.112.

Applied to the 13,618 markets that closed later, it changes the Brier score a week out by −0.0001, with a 95% interval from −0.0009 to +0.0008, and a month out by −0.0003. In other words, *nothing*.

The bias is real on average, but too small and too unstable over time for a formula to turn it into a better forecast, let alone a profit after the spread.

<div class="callout" markdown="1">
<p class="callout-label">For engineers</p>

The same discipline applies to any recalibration of a model: fit it on the past, score it on the future, and report the difference with its interval. Here a correction that every in-sample diagnostic recommends turns out to be worth nothing.
</div>

## Insiders: when the market knows too early

The most impressive forecasts on Polymarket are sometimes not forecasts at all.

### The Nobel Peace Prize

<img class="tfs-figure tfs-light" src="/assets/posts/polymarket-accuracy/nobel-light.png" width="760" alt="Prices of the leading candidates for the 2025 Nobel Peace Prize on Polymarket from midday on 8 October to the announcement on 10 October. María Corina Machado&#x27;s price is near the bottom until late on 9 October, then rises within hours to about two thirds of a dollar before the announcement at 09:00 UTC" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/posts/polymarket-accuracy/nobel-dark.png" width="760" alt="Prices of the leading candidates for the 2025 Nobel Peace Prize on Polymarket from midday on 8 October to the announcement on 10 October. María Corina Machado&#x27;s price is near the bottom until late on 9 October, then rises within hours to about two thirds of a dollar before the announcement at 09:00 UTC" loading="lazy">

The Norwegian Nobel Committee announced the 2025 Peace Prize at 09:00 UTC on 10 October. For months María Corina Machado had traded at a cent or two. Shortly after midnight in Oslo her price started to climb, passed 43 cents before one in the morning and reached 73 by two<sup class="cite">[<a href="#ref-27" id="cite-29" data-ref="27" aria-label="Reference 27: Decrypt (2025). Nobel Peace Prize organizers investigating Polymarket insider trades. 10 October 2025.">27</a>]</sup>.

In the price history I downloaded she was still at 67 cents five minutes before the announcement. An account created days earlier bought early and made tens of thousands of dollars.

The director of the Norwegian Nobel Institute said, "We take this very seriously"<sup class="cite">[<a href="#ref-28" id="cite-30" data-ref="28" aria-label="Reference 28: Bloomberg (2025). Unusual Polymarket bets on Peace Prize winner spur probe. Via Bloomberg Law, 10 October 2025.">28</a>]</sup>. In January the institute concluded that the winner's name had most likely leaked through "the digital domain" and that it could not say who was behind it<sup class="cite">[<a href="#ref-29" id="cite-31" data-ref="29" aria-label="Reference 29: Reuters (2026). Nobel committee says Peace Prize winner likely revealed early by digital spying. Via CP24, 30 January 2026.">29</a>]</sup>.

### Maduro, the Iran strikes and Google

That was not the only case. On the night of 2 January 2026, hours before Donald Trump announced the capture of Nicolás Maduro, a weeks-old account bet about &#36;33,000 that Maduro would be out by the end of the month and made about &#36;400,000.

In April the Justice Department charged a Special Forces master sergeant who had helped plan the operation, and the CFTC called it the first insider-trading case it had ever brought over event contracts<sup class="cite">[<a href="#ref-30" id="cite-32" data-ref="30" aria-label="Reference 30: U.S. Department of Justice (2026). U.S. soldier charged with using classified information to profit from prediction market bets. 23 April 2026.">30</a>, <a href="#ref-31" id="cite-33" data-ref="31" aria-label="Reference 31: Commodity Futures Trading Commission (2026). Press release 9217-26. 23 April 2026.">31</a>]</sup>. He has pleaded not guilty.

Hours before the US strikes on Iran on 28 February, six freshly funded wallets bought Yes on the strike market and made about a million dollars<sup class="cite">[<a href="#ref-32" id="cite-34" data-ref="32" aria-label="Reference 32: The Block (2026). Fresh accounts netted &amp;#36;1 million on Polymarket hours before US airstrikes on Iran. 28 February 2026.">32</a>]</sup>.

In May a Google engineer was charged over about &#36;1.2 million made on markets about Google's "Year in Search" lists, using, according to Google, marketing material he could see at work<sup class="cite">[<a href="#ref-33" id="cite-35" data-ref="33" aria-label="Reference 33: CBS News (2026). Google employee charged with insider trading on Polymarket. 28 May 2026.">33</a>]</sup>.

Polymarket's chief executive once put it plainly to CBS<sup class="cite">[<a href="#ref-34" id="cite-36" data-ref="34" aria-label="Reference 34: CBS News (2026). Polymarket trader made &amp;#36;400,000 on Maduro&#x27;s capture. 6 January 2026.">34</a>]</sup>:

<blockquote class="pull" markdown="1">
Insiders having an edge to the market is a good thing.

<cite>Shayne Coplan, chief executive of Polymarket</cite>
</blockquote>

For accuracy, in a narrow sense, he is right: the price moved to the truth before the public knew it. It is not prediction, though, and it is no use to an ordinary trader, who is the one on the other side of each of those trades.

The company now says it referred the Maduro trader to the Justice Department itself. In Congress, one bill would bar officials and their staff from trading on government decisions<sup class="cite">[<a href="#ref-35" id="cite-37" data-ref="35" aria-label="Reference 35: Torres, R. (2026). Rep. Ritchie Torres introduces legislation to crack down on insider trading on prediction markets. Press release.">35</a>]</sup>, and another would ban markets on elections, sports and military action altogether<sup class="cite">[<a href="#ref-36" id="cite-38" data-ref="36" aria-label="Reference 36: Raskin, J. (2026). Raskin, Merkley legislation would ban prediction market gambling on elections, sports, war and government activity. Press release, 26 March 2026.">36</a>]</sup>.

## Resolution: when the truth is decided by a vote

A share pays a dollar if the event happens, but someone has to say whether it happened. On Polymarket that someone is usually UMA's *optimistic oracle*. A proposer posts an outcome with a bond, typically &#36;750, and from there it is a chain of deadlines<sup class="cite">[<a href="#ref-37" id="cite-39" data-ref="37" aria-label="Reference 37: Polymarket (2026). Resolution. Documentation.">37</a>]</sup>:

<p class="flow"><span>a proposal</span><span>two hours to dispute it</span><span>a second dispute</span><span>a vote of UMA holders, two days</span></p>

Most markets resolve in two hours without a dispute. The ones that don't are where prediction becomes a governance problem.

In March 2025 a &#36;7 million market on whether Ukraine would agree to a minerals deal with the US before April resolved Yes, although Ukraine had not agreed; a single large token holder was accused of swinging the vote, and Polymarket called the situation "unprecedented" but paid no refunds, since "this wasn't a market failure"<sup class="cite">[<a href="#ref-38" id="cite-40" data-ref="38" aria-label="Reference 38: The Block (2025). Polymarket says governance attack by UMA whale to hijack a bet&#x27;s resolution is &quot;unprecedented&quot;. 26 March 2025.">38</a>]</sup>.

That summer a market with hundreds of millions of dollars traded asked whether Volodymyr Zelensky would be photographed in a suit; he appeared at a NATO summit in a black jacket and matching trousers that many outlets called a suit, and UMA's voters decided it was not<sup class="cite">[<a href="#ref-39" id="cite-41" data-ref="39" aria-label="Reference 39: DL News (2025). Polymarket bettors clash over Zelenskyy suit dispute. 30 June 2025.">39</a>]</sup>.

In May 2026 Bloomberg found that nine wallets held about half of the UMA votes cast in Polymarket disputes over three years<sup class="cite">[<a href="#ref-40" id="cite-42" data-ref="40" aria-label="Reference 40: Bloomberg (2026). Las disputas en Polymarket exponen el lado oscuro de los mercados de predicción. Via Bloomberg Línea, 29 May 2026.">40</a>]</sup>.

The platform has been narrowing the problem: since September 2025 only whitelisted proposers can propose outcomes, which cut disputes by two thirds in the first month<sup class="cite">[<a href="#ref-41" id="cite-43" data-ref="41" aria-label="Reference 41: UMA (2025). Managed proposers: an update. Blog, 8 November 2025.">41</a>]</sup>, crypto price markets now settle from Chainlink prices without a vote, and a new contract system that can combine several oracles is due to switch on for new markets this week<sup class="cite">[<a href="#ref-42" id="cite-44" data-ref="42" aria-label="Reference 42: The Block (2026). Polymarket rolls out V2 upgrade. 6 October 2026.">42</a>]</sup>.

For a trader, resolution risk is a cost like the spread, small on most markets and large on exactly the ones with ambiguous wording.

## Wash trading: volume that isn't

Volume is the number every article about Polymarket leads with, and it is the least reliable number on the platform.

Researchers at Columbia classified nearly 25% of Polymarket's historical volume as consistent with *wash trading*, traders buying from and selling to themselves without changing their position; it peaked at nearly 60% of weekly volume in December 2024, and in sports markets it reached 45% of all-time volume<sup class="cite">[<a href="#ref-43" id="cite-45" data-ref="43" aria-label="Reference 43: Sirolly, A., Ma, H., Kanoria, Y., &amp; Sethi, R. (2025). Network-based detection of wash trading. SSRN Working Paper 5714122.">43</a>]</sup>.

Separately, Paradigm showed in December 2025 that most dashboards counted every trade twice, once for each side<sup class="cite">[<a href="#ref-44" id="cite-46" data-ref="44" aria-label="Reference 44: Paradigm (2025). Polymarket volume is being double-counted. 8 December 2025.">44</a>]</sup>.

Neither changes the prices this post studies: a wash trade moves money in a circle, not the price. But together with the trap above, they are good reasons not to read a market's volume as a measure of how much it knows.

## What a Polymarket price is worth

As a probability, a Polymarket price on a market people actually trade is about as good as public forecasts get.

A week before the close the prices were calibrated to within 1.8 percentage points on average, and on the markets open for at least a month they removed 70% of the base-rate forecaster's error a week out and 82% a day out.

When a liquid market says 70%, the honest reading is "about 70%", give or take a few points.

That is a statement about many markets, not about one. The Senate at 71.5 cents is a single draw, and so was Brazil at 80. A calibrated price is still wrong three times in ten when it says 70, which is what 70 means.

The weak spots are where theory and the data agree they should be. Thin markets are not forecasts at all, and in them longshots are overpriced. Prices months before the close drift toward 50 cents, because waiting costs money and Polymarket pays for the wait on only a handful of markets.

Crypto price markets share one driver and resolve together, so a bad month for Bitcoin looks like miscalibration. Insiders sometimes move prices before the news, which makes the forecast *right* and the trade *unfair*. And the dollar arrives only if the oracle agrees that the event happened.

For a trader the useful findings are modest and specific. The mispricing that survives is at the tails: contracts under 5 cents lose a quarter to two fifths of the stake on average, and near-certain favourites earn a fraction of a percent a week, more when they are bought a month or more before the close.

It is about the size of the spread and the fee, which is why it survives. Nothing in the data suggests a general bias that a formula could exploit, and a two-number correction fitted on the past made the future no better.

The people who make money on Polymarket, according to a study of 588 million trades, are mostly the ones who post limit orders and let others cross the spread; the top 1% of profitable accounts took 76.5% of the profits<sup class="cite">[<a href="#ref-45" id="cite-47" data-ref="45" aria-label="Reference 45: Akey, P., Grégoire, V., Harvie, N., &amp; Martineau, C. (2026). Who wins and who loses in prediction markets? Evidence from Polymarket. CEPR Discussion Paper 21615.">45</a>]</sup>.

For an engineer the lessons are the ones from evaluating any model, sharpened by a market's habits. Accuracy flatters when the classes are unbalanced, and most questions resolve No. Comparisons across horizons are confounded by which questions exist at each horizon, so they have to be made on the same questions.

And a filter on anything that depends on the outcome, a market's total volume included, manufactures effects that aren't there. The trap that caught my first sample is the most transferable thing in this post.

## References

1. {:#ref-1} CNBC (2026). [Lula wins Brazil election despite prediction markets favoring Bolsonaro](https://www.cnbc.com/2026/10/26/lula-brazil-election-prediction-markets.html). 26 October 2026. <a class="ref-back" href="#cite-1" aria-label="Back to the text">&#8617;</a>
2. {:#ref-2} Semafor (2026). [Prediction markets flop on recent elections](https://www.semafor.com/article/10/30/2026/prediction-markets-flop-on-recent-elections). 30 October 2026. <a class="ref-back" href="#cite-2" aria-label="Back to the text">&#8617;</a>
3. {:#ref-3} Polymarket (2026). [pUSD](https://docs.polymarket.com/concepts/pusd). Documentation. <a class="ref-back" href="#cite-3" aria-label="Back to the text">&#8617;</a>
4. {:#ref-4} Polymarket (2026). [Prices and the order book](https://docs.polymarket.com/concepts/prices-orderbook). Documentation. <a class="ref-back" href="#cite-4" aria-label="Back to the text">&#8617;</a>
5. {:#ref-5} Wolfers, J., & Zitzewitz, E. (2006). [Interpreting prediction market prices as probabilities](https://www.nber.org/papers/w12200). NBER Working Paper 12200. <a class="ref-back" href="#cite-5" aria-label="Back to the text">&#8617;</a>
6. {:#ref-6} Manski, C. F. (2006). [Interpreting the predictions of prediction markets](https://doi.org/10.1016/j.econlet.2006.01.004). *Economics Letters*, 91(3), 425–429. <a class="ref-back" href="#cite-6" aria-label="Back to the text">&#8617;</a>
7. {:#ref-7} Polymarket (2026). [Trading fees](https://docs.polymarket.com/trading/fees). Documentation. <a class="ref-back" href="#cite-7" aria-label="Back to the text">&#8617;</a>
8. {:#ref-8} Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). [On calibration of modern neural networks](https://proceedings.mlr.press/v70/guo17a.html). *ICML 2017*. <a class="ref-back" href="#cite-8" aria-label="Back to the text">&#8617;</a>
9. {:#ref-9} Dimitriadis, T., Gneiting, T., & Jordan, A. I. (2021). [Stable reliability diagrams for probabilistic classifiers](https://doi.org/10.1073/pnas.2016191118). *PNAS*, 118(8), e2016191118. <a class="ref-back" href="#cite-9" aria-label="Back to the text">&#8617;</a>
10. {:#ref-10} Brier, G. W. (1950). [Verification of forecasts expressed in terms of probability](https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml). *Monthly Weather Review*, 78(1), 1–3. <a class="ref-back" href="#cite-10" aria-label="Back to the text">&#8617;</a>
11. {:#ref-11} Murphy, A. H. (1973). [A new vector partition of the probability score](https://doi.org/10.1175/1520-0450%281973%29012%3C0595%3AANVPOT%3E2.0.CO%3B2). *Journal of Applied Meteorology*, 12(4), 595–600. <a class="ref-back" href="#cite-11" aria-label="Back to the text">&#8617;</a>
12. {:#ref-12} 538 (2024). [538's final forecasts for the 2024 presidential election](https://6abc.com/post/538s-final-forecasts-2024-presidential-election/15512334/). Republished by 6abc, 5 November 2024. <a class="ref-back" href="#cite-12" aria-label="Back to the text">&#8617;</a>
13. {:#ref-13} The Wall Street Journal (2024). [The Trump betting whale speaks out: "I have absolutely no political agenda"](https://tovima.com/wsj/the-trump-betting-whale-speaks-out-i-have-absolutely-no-political-agenda). Republished by To Vima, 1 November 2024. <a class="ref-back" href="#cite-13" aria-label="Back to the text">&#8617;</a>
14. {:#ref-14} CBS News (2025). [French whale made over &#36;80 million on Polymarket betting on Trump's election win](https://www.cbsnews.com/news/french-whale-made-over-80-million-on-polymarket-betting-on-trump-election-win-60-minutes/). *60 Minutes*, 30 November 2025. <a class="ref-back" href="#cite-14" aria-label="Back to the text">&#8617;</a>
15. {:#ref-15} Horvitz, D. G., & Thompson, D. J. (1952). [A generalization of sampling without replacement from a finite universe](https://doi.org/10.1080/01621459.1952.10483446). *Journal of the American Statistical Association*, 47(260), 663–685. <a class="ref-back" href="#cite-15" aria-label="Back to the text">&#8617;</a>
16. {:#ref-16} Walker, A. (2026). [The price is (almost) right: a calibration analysis of Polymarket pricing data](https://theses-dissertations.princeton.edu/handle/88435/dsp01mw22v8973). Senior thesis, Princeton University. <a class="ref-back" href="#cite-16" aria-label="Back to the text">&#8617;</a>
17. {:#ref-17} Polymarket (2026). [How accurate is Polymarket?](https://polymarket.com/accuracy) Accessed 31 October 2026. <a class="ref-back" href="#cite-17" aria-label="Back to the text">&#8617;</a>
18. {:#ref-18} Griffith, R. M. (1949). [Odds adjustments by American horse-race bettors](https://doi.org/10.2307/1418469). *American Journal of Psychology*, 62(2), 290–294. <a class="ref-back" href="#cite-18" aria-label="Back to the text">&#8617;</a>
19. {:#ref-19} Snowberg, E., & Wolfers, J. (2010). [Explaining the favorite–long shot bias: is it risk-love or misperceptions?](https://doi.org/10.1086/655844) *Journal of Political Economy*, 118(4), 723–746. <a class="ref-back" href="#cite-19" aria-label="Back to the text">&#8617;</a>
20. {:#ref-20} Bürgi, C., Deng, W., & Whelan, K. (2026). [Makers and takers: the economics of the Kalshi prediction market](https://www2.gwu.edu/~forcpgm/2026-001.pdf). GWU Center for Economic Research Working Paper 2026-001. <a class="ref-back" href="#cite-20" aria-label="Back to the text">&#8617;</a>
21. {:#ref-21} Cardozo, M., & Rivero-Wildemauwe, J. I. (2026). [The favorite-longshot bias in prediction markets: evidence from Polymarket](https://arxiv.org/abs/2609.12878). arXiv:2609.12878. <a class="ref-back" href="#cite-21" aria-label="Back to the text">&#8617;</a>
22. {:#ref-22} Federal Reserve Bank of St. Louis (2026). [3-month Treasury bill secondary market rate (DTB3)](https://fred.stlouisfed.org/series/DTB3). FRED. <a class="ref-back" href="#cite-23" aria-label="Back to the text">&#8617;</a>
23. {:#ref-23} Page, L., & Clemen, R. T. (2013). [Do prediction markets produce well-calibrated probability forecasts?](https://doi.org/10.1111/j.1468-0297.2012.02561.x) *The Economic Journal*, 123(568), 491–513. <a class="ref-back" href="#cite-24" aria-label="Back to the text">&#8617;</a>
24. {:#ref-24} Polymarket (2026). [Holding rewards](https://help.polymarket.com/en/articles/13364459-holding-rewards). Help center, 1 June 2026. <a class="ref-back" href="#cite-25" aria-label="Back to the text">&#8617;</a>
25. {:#ref-25} Niculescu-Mizil, A., & Caruana, R. (2005). [Predicting good probabilities with supervised learning](https://doi.org/10.1145/1102351.1102430). *ICML 2005*. <a class="ref-back" href="#cite-26" aria-label="Back to the text">&#8617;</a>
26. {:#ref-26} Le, N. A. (2026). [Decomposing crowd wisdom: domain-specific calibration dynamics in prediction markets](https://arxiv.org/abs/2602.19520). arXiv:2602.19520. <a class="ref-back" href="#cite-28" aria-label="Back to the text">&#8617;</a>
27. {:#ref-27} Decrypt (2025). [Nobel Peace Prize organizers investigating Polymarket insider trades](https://decrypt.co/343840/nobel-peace-prize-organizers-intestigating-polymarket-insider-trades). 10 October 2025. <a class="ref-back" href="#cite-29" aria-label="Back to the text">&#8617;</a>
28. {:#ref-28} Bloomberg (2025). [Unusual Polymarket bets on Peace Prize winner spur probe](https://news.bloomberglaw.com/capital-markets/unusual-polymarket-bets-on-peace-prize-winner-spur-probe-1). Via Bloomberg Law, 10 October 2025. <a class="ref-back" href="#cite-30" aria-label="Back to the text">&#8617;</a>
29. {:#ref-29} Reuters (2026). [Nobel committee says Peace Prize winner likely revealed early by digital spying](https://www.cp24.com/news/world/2026/01/30/nobel-committee-says-peace-prize-winner-likely-revealed-early-by-digital-spying/). Via CP24, 30 January 2026. <a class="ref-back" href="#cite-31" aria-label="Back to the text">&#8617;</a>
30. {:#ref-30} U.S. Department of Justice (2026). [U.S. soldier charged with using classified information to profit from prediction market bets](https://www.justice.gov/opa/pr/us-soldier-charged-using-classified-information-profit-prediction-market-bets). 23 April 2026. <a class="ref-back" href="#cite-32" aria-label="Back to the text">&#8617;</a>
31. {:#ref-31} Commodity Futures Trading Commission (2026). [Press release 9217-26](https://www.cftc.gov/PressRoom/PressReleases/9217-26). 23 April 2026. <a class="ref-back" href="#cite-33" aria-label="Back to the text">&#8617;</a>
32. {:#ref-32} The Block (2026). [Fresh accounts netted &#36;1 million on Polymarket hours before US airstrikes on Iran](https://www.theblock.co/post/391650/fresh-accounts-netted-1-million-on-polymarket-hours-before-us-airstrikes-on-iran-bubblemaps). 28 February 2026. <a class="ref-back" href="#cite-34" aria-label="Back to the text">&#8617;</a>
33. {:#ref-33} CBS News (2026). [Google employee charged with insider trading on Polymarket](https://www.cbsnews.com/news/polymarket-google-employee-insider-trading-charges-spagnuolo-d4vd/). 28 May 2026. <a class="ref-back" href="#cite-35" aria-label="Back to the text">&#8617;</a>
34. {:#ref-34} CBS News (2026). [Polymarket trader made &#36;400,000 on Maduro's capture](https://www.cbsnews.com/news/polymarket-maduro-capture-bet-400000). 6 January 2026. <a class="ref-back" href="#cite-36" aria-label="Back to the text">&#8617;</a>
35. {:#ref-35} Torres, R. (2026). [Rep. Ritchie Torres introduces legislation to crack down on insider trading on prediction markets](https://ritchietorres.house.gov/posts/in-response-to-suspicious-polymarket-trade-preceding-maduro-operation-rep-ritchie-torres-introduces-legislation-to-crack-down-on-insider-trading-on-prediction-markets). Press release. <a class="ref-back" href="#cite-37" aria-label="Back to the text">&#8617;</a>
36. {:#ref-36} Raskin, J. (2026). [Raskin, Merkley legislation would ban prediction market gambling on elections, sports, war and government activity](https://raskin.house.gov/2026/3/raskin-merkley-legislation-would-ban-prediction-market-gambling-on-elections-sports-war-and-government-activity). Press release, 26 March 2026. <a class="ref-back" href="#cite-38" aria-label="Back to the text">&#8617;</a>
37. {:#ref-37} Polymarket (2026). [Resolution](https://docs.polymarket.com/concepts/resolution). Documentation. <a class="ref-back" href="#cite-39" aria-label="Back to the text">&#8617;</a>
38. {:#ref-38} The Block (2025). [Polymarket says governance attack by UMA whale to hijack a bet's resolution is "unprecedented"](https://www.theblock.co/post/348171/polymarket-says-governance-attack-by-uma-whale-to-hijack-a-bets-resolution-is-unprecedented). 26 March 2025. <a class="ref-back" href="#cite-40" aria-label="Back to the text">&#8617;</a>
39. {:#ref-39} DL News (2025). [Polymarket bettors clash over Zelenskyy suit dispute](https://www.dlnews.com/articles/markets/polymarket-bettors-clash-over-zelenskyy-suit-dispute/). 30 June 2025. <a class="ref-back" href="#cite-41" aria-label="Back to the text">&#8617;</a>
40. {:#ref-40} Bloomberg (2026). [Las disputas en Polymarket exponen el lado oscuro de los mercados de predicción](https://www.bloomberglinea.com/cripto/las-disputas-en-polymarket-exponen-el-lado-oscuro-de-los-mercados-de-prediccion/). Via Bloomberg Línea, 29 May 2026. <a class="ref-back" href="#cite-42" aria-label="Back to the text">&#8617;</a>
41. {:#ref-41} UMA (2025). [Managed proposers: an update](https://blog.uma.xyz/articles/managed-proposers-update). Blog, 8 November 2025. <a class="ref-back" href="#cite-43" aria-label="Back to the text">&#8617;</a>
42. {:#ref-42} The Block (2026). [Polymarket rolls out V2 upgrade](https://www.theblock.co/news/business/2026-10-06-polymarket-rolls-out-v2-upgrade-with-single-erc-1155-positions-contract-pusd-collateral-417804). 6 October 2026. <a class="ref-back" href="#cite-44" aria-label="Back to the text">&#8617;</a>
43. {:#ref-43} Sirolly, A., Ma, H., Kanoria, Y., & Sethi, R. (2025). [Network-based detection of wash trading](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5714122). SSRN Working Paper 5714122. <a class="ref-back" href="#cite-45" aria-label="Back to the text">&#8617;</a>
44. {:#ref-44} Paradigm (2025). [Polymarket volume is being double-counted](https://www.paradigm.xyz/2025/12/polymarket-volume-is-being-double-counted). 8 December 2025. <a class="ref-back" href="#cite-46" aria-label="Back to the text">&#8617;</a>
45. {:#ref-45} Akey, P., Grégoire, V., Harvie, N., & Martineau, C. (2026). [Who wins and who loses in prediction markets? Evidence from Polymarket](https://ideas.repec.org/p/cpr/ceprdp/21615.html). CEPR Discussion Paper 21615. <a class="ref-back" href="#cite-47" aria-label="Back to the text">&#8617;</a>
{: .tfs-refs}

{% endraw %}
