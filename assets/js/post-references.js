// Fold a post's "References" section into a <details> element, closed by default. Without JavaScript the
// list simply stays open. A link to #references (the table of contents has one) opens it.
(function () {
  function fold() {
    const root = document.getElementById("markdown-content");
    if (!root) return;
    root.querySelectorAll("h2").forEach(function (heading) {
      if (heading.textContent.trim().toLowerCase() !== "references") return;

      const details = document.createElement("details");
      details.className = "post-references";
      if (heading.id) details.id = heading.id + "-section";
      const summary = document.createElement("summary");
      const body = document.createElement("div");
      body.className = "post-references-body";

      let node = heading.nextSibling;
      while (node && !(node.nodeType === Node.ELEMENT_NODE && node.tagName === "H2")) {
        const next = node.nextSibling;
        body.appendChild(node);
        node = next;
      }

      heading.parentNode.insertBefore(details, heading);
      summary.appendChild(heading);
      const count = body.querySelectorAll("li").length;
      if (count > 0) {
        const note = document.createElement("span");
        note.className = "post-references-count";
        note.textContent = count + (count === 1 ? " source" : " sources");
        summary.appendChild(note);
      }
      details.appendChild(summary);
      details.appendChild(body);
    });
  }

  function openTarget() {
    if (!location.hash) return;
    const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    const details = target && target.closest("details.post-references");
    if (details && !details.open) {
      details.open = true;
      target.scrollIntoView();
    }
  }

  function start() {
    fold();
    openTarget();
    window.addEventListener("hashchange", openTarget);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
