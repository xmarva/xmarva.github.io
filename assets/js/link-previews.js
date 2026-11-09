// Links in a post to other posts of the site show a small card with the post's cover, title and description when
// hovered or focused, so a reader can decide without leaving the page. Only on devices with a pointer that can hover;
// on touch screens a link stays a link. The cards come from /assets/json/previews.json, fetched on the first hover.
(function () {
  if (!window.matchMedia || !window.matchMedia("(hover: hover) and (pointer: fine)").matches) return;
  const content = document.querySelector(".post-content");
  if (!content) return;
  let previews = null;
  let loading = null;
  let card = null;
  let showTimer = null;
  let hideTimer = null;

  function load() {
    if (!loading) {
      loading = fetch("/assets/json/previews.json")
        .then(function (response) {
          return response.json();
        })
        .then(function (data) {
          previews = data;
          return data;
        })
        .catch(function (error) {
          previews = {};
          return previews;
        });
    }
    return loading;
  }

  function target(link) {
    let url;
    try {
      url = new URL(link.href, window.location.href);
    } catch (error) {
      return null;
    }
    if (url.origin !== window.location.origin || !/^\/blog\/\d{4}\//.test(url.pathname)) return null;
    if (url.pathname === window.location.pathname) return null; // a jump within this page
    return url.pathname.replace(/\/?$/, "/");
  }

  function build() {
    card = document.createElement("div");
    card.className = "cite-card link-card";
    card.setAttribute("role", "tooltip");
    card.id = "link-card";
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
    const widest = window.scrollX + document.documentElement.clientWidth - width - margin;
    left = Math.max(window.scrollX + margin, Math.min(left, widest));
    let top = window.scrollY + box.bottom + 8;
    if (box.bottom + 8 + card.offsetHeight > window.innerHeight && box.top - 8 - card.offsetHeight > 0) {
      top = window.scrollY + box.top - 8 - card.offsetHeight;
    }
    card.style.left = left + "px";
    card.style.top = top + "px";
  }

  function show(link, path) {
    const post = previews && previews[path];
    if (!post) return;
    if (!card) build();
    card.innerHTML = "";
    if (post.cover) {
      const image = document.createElement("img");
      image.src = post.cover;
      image.alt = "";
      image.className = "link-card-cover";
      card.appendChild(image);
    }
    const date = document.createElement("span");
    date.className = "cite-card-number";
    date.textContent = post.date;
    card.appendChild(date);
    const title = document.createElement("strong");
    title.className = "link-card-title";
    title.textContent = post.title;
    card.appendChild(title);
    const text = document.createElement("p");
    text.className = "link-card-text";
    text.textContent = post.description;
    card.appendChild(text);
    card.hidden = false;
    place(link);
    link.setAttribute("aria-describedby", "link-card");
  }

  function hide() {
    if (card) card.hidden = true;
  }

  function scheduleHide() {
    clearTimeout(showTimer);
    hideTimer = setTimeout(hide, 180);
  }

  content.querySelectorAll("a[href]").forEach(function (link) {
    const path = target(link);
    if (!path) return;
    function start() {
      clearTimeout(hideTimer);
      clearTimeout(showTimer);
      showTimer = setTimeout(function () {
        load().then(function () {
          show(link, path);
        });
      }, 250);
    }
    link.addEventListener("mouseenter", start);
    link.addEventListener("focus", start);
    link.addEventListener("mouseleave", scheduleHide);
    link.addEventListener("blur", scheduleHide);
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") hide();
  });
})();
