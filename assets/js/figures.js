// Every picture that stands on its own in a post becomes a numbered figure with a caption: the chart's title for a
// plot made by the notebook, the first sentence of the description for a drawn figure. The light and dark versions of
// one figure share a number. Covers, icons and pictures inside text are left alone.
(function () {
  const content = document.querySelector(".post-page .post-content");
  if (!content) return;
  const numbers = {};
  let count = 0;

  function caption(img) {
    const alt = (img.getAttribute("alt") || "").trim();
    if (!alt) return "";
    if (img.classList.contains("tfs-output")) return alt;
    const match = alt.match(/^[\s\S]*?[.!?](?=\s|$)/);
    const first = match ? match[0] : alt;
    return first.length > 180 ? first.slice(0, 177).replace(/\s\S*$/, "") + "…" : first;
  }

  const seen = [];
  content.querySelectorAll("p > img, p > picture").forEach(function (node) {
    const holder = node.parentElement;
    if (seen.indexOf(holder) !== -1) return;
    seen.push(holder);
    const pictures = Array.prototype.every.call(holder.children, function (child) {
      return child.tagName === "IMG" || child.tagName === "PICTURE";
    });
    if (!pictures || holder.textContent.trim() !== "") return; // a picture inside a sentence
    const img = node.tagName === "IMG" ? node : node.querySelector("img");
    if (!img || img.closest(".post-cover")) return;
    const key = (img.getAttribute("src") || "").replace(/-(light|dark)(\.\w+)(\?.*)?$/, "$2");
    if (!numbers[key]) {
      count += 1;
      numbers[key] = count;
    }
    const figure = document.createElement("figure");
    figure.className = "post-figure";
    if (holder.children.length === 1 && img.classList.contains("tfs-dark")) figure.classList.add("tfs-dark");
    if (holder.children.length === 1 && img.classList.contains("tfs-light")) figure.classList.add("tfs-light");
    holder.parentNode.insertBefore(figure, holder);
    figure.appendChild(holder);
    const text = caption(img);
    const label = document.createElement("figcaption");
    const number = document.createElement("b");
    number.textContent = "Fig. " + numbers[key];
    label.appendChild(number);
    if (text) label.appendChild(document.createTextNode(text));
    figure.appendChild(label);
  });
})();
