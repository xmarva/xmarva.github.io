The pictures on /services/: real data in real tools, in the site's colours. Jekyll skips this folder.

**finance.png**: BTC/USDT on Binance, a snapshot of its public API (`/api/v3/klines`, 4h candles; `/depth`;
`/ticker/24hr`) taken on 24 October 2026 and saved in `data.js`, drawn by `exchange.html` with TradingView
Lightweight Charts 4.2 (Apache-2.0; its logo stays on the chart as the library asks). Render `exchange.html` at
640 × 440 with a device scale of 2.

**biomed.png**: resting-state fMRI of ten ADHD-200 participants (nilearn's copy of the ADHD-200 sample), the
mean correlation between the 39 regions of the MSDL atlas over the seven with ADHD, drawn with nilearn.
`brain_fetch.py <data dir>` downloads and computes, `brain_plot.py <data dir>/connectome.npz biomed.png <fonts dir>`
draws (the fonts dir holds IBMPlexMono-Regular.ttf and -Medium.ttf).

**security.png**: CyberChef (GCHQ, Apache-2.0) in its dark theme with the site's greys (`cyberchef-style.js`),
decoding a PowerShell launcher: a regular expression, From Base64, UTF-16LE, Defang URL. The input is in
`cyberchef.url`; the command is a defanged sample (hxxp, a documentation address from 203.0.113.0/24), nothing
that runs. The recipe is set through CyberChef's `app.setRecipeConfig` after the page loads.
