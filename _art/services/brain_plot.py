# The ADHD-200 connectome as a neuroimaging tool shows it, in the site's dark palette
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ["IBMPlexMono-Regular.ttf", "IBMPlexMono-Medium.ttf"]:
    font_manager.fontManager.addfont(f"{sys.argv[3]}/{f}")
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from nilearn import plotting

npz, out = sys.argv[1], sys.argv[2]
d = np.load(npz, allow_pickle=True)
mean, coords, labels, networks, series = d["mean"], d["coords"], list(d["labels"]), list(d["networks"]), d["series"]
BG, INK, MUTED, FAINT, LINE = "#131316", "#e6e6eb", "#8b8b96", "#5d5d68", "#25252b"
plt.rcParams.update({"font.family": "IBM Plex Mono", "text.color": MUTED, "axes.labelcolor": MUTED,
                     "xtick.color": FAINT, "ytick.color": FAINT})
palette = ["#4cc9b8", "#a596f7", "#f0b04a", "#86a8f7", "#f4879d", "#ff4545", "#9ad06a", "#d9b45a", "#c7c7d1", "#6fb7e9", "#e58fd0", "#b5a28a", "#7a7a86", "#58d68d", "#f2994a", "#bb86fc", "#7ec8e3"]
names = list(dict.fromkeys(networks))
node_colors = [palette[names.index(n) % len(palette)] for n in networks]

fig = plt.figure(figsize=(12.8, 8.8), dpi=100, facecolor=BG)
fig.text(0.018, 0.962, "ADHD-200 · resting-state fMRI · 7 participants with ADHD · MSDL atlas, 39 regions", fontsize=11, color=MUTED)
fig.text(0.982, 0.962, "nilearn", fontsize=11, color=FAINT, ha="right")
fig.add_artist(plt.Line2D([0, 1], [0.94, 0.94], color=LINE, lw=1))

edges = LinearSegmentedColormap.from_list("edges", ["#5a3a46", "#f4879d", "#ff4545"])
matrix = mean.copy()
np.fill_diagonal(matrix, 0)
display = plotting.plot_connectome(matrix, coords, node_color=node_colors, node_size=34, edge_threshold="96%",
                                   edge_cmap=edges, edge_vmin=0.3, edge_vmax=0.8, display_mode="lzry", black_bg=True,
                                   figure=fig, axes=(0.0, 0.43, 1.0, 0.5), colorbar=False,
                                   edge_kwargs={"linewidth": 2.0, "alpha": 0.9})

# the matrix, regions grouped by network
order = np.argsort([names.index(n) for n in networks], kind="stable")
ax = fig.add_axes((0.045, 0.08, 0.33, 0.34))
ax.imshow(mean[np.ix_(order, order)], cmap=LinearSegmentedColormap.from_list("div", ["#86a8f7", BG, "#f4879d"]), vmin=-0.8, vmax=0.8)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_color(LINE)
ax.set_title("correlation between regions", fontsize=10, color=MUTED, loc="left", pad=8)
bounds = np.cumsum([networks.count(n) for n in names])[:-1] - 0.5
for b in bounds:
    ax.axhline(b, color=BG, lw=0.8); ax.axvline(b, color=BG, lw=0.8)

# a real signal: four regions of the first subject
ax2 = fig.add_axes((0.43, 0.08, 0.545, 0.34), facecolor=BG)
picks = [labels.index(n) for n in ["L DMN", "R DMN", "L Ins", "Med DMN"] if n in labels][:4] or [0, 1, 2, 3]
t = np.arange(series.shape[0]) * 2.5
for k, r in enumerate(picks):
    ax2.plot(t, series[:, r] + k * 3.2, color=node_colors[r], lw=1.1)
    ax2.text(t[-1] + 6, k * 3.2, labels[r], color=node_colors[r], fontsize=9, va="center")
ax2.set_title("BOLD signal, one subject", fontsize=10, color=MUTED, loc="left", pad=8)
ax2.set_yticks([]); ax2.set_xlabel("seconds", fontsize=9)
ax2.set_xlim(0, t[-1] + 70)
for side in ["top", "right", "left"]:
    ax2.spines[side].set_visible(False)
ax2.spines["bottom"].set_color(LINE)

from matplotlib.patches import Patch, PathPatch
from matplotlib.lines import Line2D
for a in fig.axes:  # nilearn paints black and outlines in white; the site's background and a softer grey instead
    if np.allclose(a.get_facecolor()[:3], 0):
        a.set_facecolor(BG)
    for artist in a.get_children():
        if isinstance(artist, (Patch, PathPatch)) and np.allclose(artist.get_facecolor()[:3], 0) and artist.get_facecolor()[3] > 0:
            artist.set_facecolor(BG)
        if isinstance(artist, (Patch, PathPatch)) and np.mean(artist.get_edgecolor()[:3]) > 0.6:
            artist.set_edgecolor("#77777f")
        if isinstance(artist, Line2D) and np.mean(matplotlib.colors.to_rgb(artist.get_color())) > 0.6 and artist.axes not in (ax, ax2):
            artist.set_color("#77777f")
    for text in a.texts:
        if text.get_text() in ("L", "R"):
            text.set_color(FAINT)
            text.set_bbox({"facecolor": BG, "edgecolor": "none"})
for a in fig.axes:  # nilearn's black backdrop is an image of zeros in its frame
    if a in (ax, ax2):
        continue
    a.set_facecolor(BG)
    for img in a.images:
        values = np.asarray(img.get_array(), dtype=float)
        if values.size and np.nanmax(np.abs(values)) < 1e-6:
            img.set_visible(False)
fig.savefig(out, facecolor=BG, dpi=100)
print("saved", out, "networks", len(names))
