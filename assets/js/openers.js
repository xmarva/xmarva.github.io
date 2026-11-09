// The post's summary and its contents open on request: both become small folded rows side by side under the header
// (tl;dr and Contents), and the one opened unfolds below them. Without JavaScript the summary stays open in the text
// and the contents keep their own fold.
(function () {
  const content = document.querySelector(".post-page .post-content");
  if (!content) return;
  const toc = document.getElementById("table-of-contents");
  const aside = content.querySelector("aside.tldr");
  let fold = null;
  if (aside) {
    fold = document.createElement("details");
    fold.className = "tldr-fold";
    const summary = document.createElement("summary");
    summary.textContent = "tl;dr";
    fold.appendChild(summary);
    const label = aside.querySelector(".tldr-label");
    if (label) label.remove();
    aside.parentNode.insertBefore(fold, aside);
    fold.appendChild(aside);
  }
  if (!toc && !fold) return;
  const row = document.createElement("div");
  row.className = "post-openers";
  const text = document.getElementById("markdown-content") || content; // where the blocks' styles live
  text.insertBefore(row, text.firstChild);
  if (fold) row.appendChild(fold);
  if (toc) {
    toc.removeAttribute("open");
    row.appendChild(toc);
  }
})();
