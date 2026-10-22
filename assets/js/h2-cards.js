// Wrap every <h2> section of opted-in containers (class "h2-cards") into a card.
// Nodes are moved rather than cloned, so event listeners attached by other
// scripts (code copy buttons, zoomable images, ...) keep working.
(function () {
  function buildCards() {
    document.querySelectorAll(".h2-cards h2").forEach(function (heading) {
      // headings nested inside an already built card stay as they are
      if (heading.closest(".h2-card")) return;

      const card = document.createElement("div");
      card.className = "h2-card";

      const content = document.createElement("div");
      content.className = "h2-card-body";

      let sibling = heading.nextElementSibling;
      while (sibling && sibling.tagName !== "H2") {
        const next = sibling.nextElementSibling;
        content.appendChild(sibling);
        sibling = next;
      }

      heading.parentNode.insertBefore(card, heading);
      card.appendChild(heading);
      card.appendChild(content);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", buildCards);
  } else {
    buildCards();
  }
})();
