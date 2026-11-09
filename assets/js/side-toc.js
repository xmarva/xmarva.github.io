// On wide screens a post's contents also sit in the margin, next to the text, and follow the reader: the section in
// view is highlighted as the reader scrolls. The list is a copy of the post's own contents at the top, so nothing new is generated; on
// narrow screens the margin is too small and only the contents at the top remain.
(function () {
  const source = document.querySelector("#table-of-contents nav");
  const article = document.querySelector(".post-content");
  if (!source || !article) return;
  const aside = document.createElement("aside");
  aside.className = "side-toc";
  aside.setAttribute("aria-label", "Contents");
  const heading = document.createElement("span");
  heading.className = "side-toc-title";
  heading.textContent = "Contents";
  aside.appendChild(heading);
  const list = source.cloneNode(true);
  list.removeAttribute("aria-label");
  aside.appendChild(list);
  document.body.appendChild(aside);

  const links = {};
  list.querySelectorAll("a[href^='#']").forEach(function (link) {
    links[decodeURIComponent(link.getAttribute("href").slice(1))] = link;
  });
  const headings = Array.prototype.filter.call(article.querySelectorAll("h2[id], h3[id]"), function (h) {
    return links[h.id];
  });

  function fit() {
    const box = article.getBoundingClientRect();
    const room = document.documentElement.clientWidth - box.right - 40;
    const visible = room >= 190 && window.innerHeight >= 560;
    aside.hidden = !visible;
    if (visible) {
      aside.style.left = box.right + 32 + "px";
      aside.style.width = Math.min(260, room - 8) + "px";
    }
  }

  let active = null;
  function mark(id) {
    if (active) active.classList.remove("active");
    active = links[id] || null;
    if (active) {
      active.classList.add("active");
      const top = active.offsetTop - aside.clientHeight / 2;
      aside.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
    }
  }

  let pending = false;
  function track() {
    pending = false;
    let current = headings[0];
    headings.forEach(function (h) {
      if (h.getBoundingClientRect().top < window.innerHeight * 0.3) current = h;
    });
    if (current && links[current.id] !== active) mark(current.id);
  }
  window.addEventListener(
    "scroll",
    function () {
      if (!pending) {
        pending = true;
        window.requestAnimationFrame(track);
      }
    },
    { passive: true }
  );
  window.addEventListener("resize", fit);
  window.addEventListener("load", fit);
  fit();
  track();
})();
