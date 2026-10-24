// Numbered citations in posts: hovering or focusing a number shows its reference in a small card; on a touch screen
// the first tap shows the card and the second follows the number to the list under References (post-references.js
// unfolds it). Escape or a tap elsewhere closes the card. Without JavaScript the numbers are plain links to the list.
(function () {
  let card = null;
  let current = null;
  let hideTimer = null;

  function reference(link) {
    return document.getElementById("ref-" + link.dataset.ref);
  }

  function build() {
    card = document.createElement("div");
    card.className = "cite-card";
    card.id = "cite-card";
    card.setAttribute("role", "tooltip");
    card.hidden = true;
    card.addEventListener("mouseenter", function () {
      clearTimeout(hideTimer);
    });
    card.addEventListener("mouseleave", scheduleHide);
    document.body.appendChild(card);
  }

  function place(link) {
    const box = link.getBoundingClientRect();
    const width = card.offsetWidth;
    const margin = 16;
    let left = window.scrollX + box.left + box.width / 2 - width / 2;
    left = Math.max(window.scrollX + margin, Math.min(left, window.scrollX + document.documentElement.clientWidth - width - margin));
    let top = window.scrollY + box.bottom + 8;
    if (box.bottom + 8 + card.offsetHeight > window.innerHeight && box.top - 8 - card.offsetHeight > 0) {
      top = window.scrollY + box.top - 8 - card.offsetHeight; // no room below: open above
    }
    card.style.left = left + "px";
    card.style.top = top + "px";
  }

  function show(link) {
    const ref = reference(link);
    if (!ref) return;
    clearTimeout(hideTimer);
    if (!card) build();
    if (current && current !== link) current.setAttribute("aria-expanded", "false");
    const body = ref.cloneNode(true);
    body.querySelectorAll(".ref-back").forEach(function (back) {
      back.remove();
    });
    card.innerHTML = "";
    const number = document.createElement("span");
    number.className = "cite-card-number";
    number.textContent = "[" + link.dataset.ref + "]";
    card.appendChild(number);
    const text = document.createElement("div");
    text.innerHTML = body.innerHTML;
    card.appendChild(text);
    const jump = document.createElement("a");
    jump.className = "cite-card-jump";
    jump.href = link.getAttribute("href");
    jump.textContent = "In the references ↓";
    card.appendChild(jump);
    card.hidden = false;
    place(link);
    current = link;
    link.setAttribute("aria-expanded", "true");
    link.setAttribute("aria-describedby", "cite-card");
  }

  function hide() {
    if (!card) return;
    card.hidden = true;
    if (current) {
      current.setAttribute("aria-expanded", "false");
      current.removeAttribute("aria-describedby");
    }
    current = null;
  }

  function scheduleHide() {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(hide, 180);
  }

  function citation(target) {
    return target && target.closest ? target.closest("sup.cite a[data-ref]") : null;
  }

  function start() {
    if (!document.querySelector("sup.cite a[data-ref]")) return;
    const hover = window.matchMedia("(hover: hover)").matches;
    if (hover) {
      document.addEventListener("mouseover", function (event) {
        const link = citation(event.target);
        if (link) show(link);
      });
      document.addEventListener("mouseout", function (event) {
        if (citation(event.target)) scheduleHide();
      });
    }
    document.addEventListener("focusin", function (event) {
      const link = citation(event.target);
      if (link) show(link);
      else if (card && !card.contains(event.target)) hide();
    });
    document.addEventListener("click", function (event) {
      const link = citation(event.target);
      if (link) {
        if (current !== link) {
          event.preventDefault(); // first tap on a touch screen: show the card, the second tap follows the link
          show(link);
        } else {
          hide();
        }
        return;
      }
      if (card && !card.contains(event.target)) hide();
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && current) {
        const link = current;
        hide();
        link.focus();
      }
    });
    window.addEventListener("resize", hide);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
