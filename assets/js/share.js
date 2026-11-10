// The share row of a post (_includes/post_share.liquid): the copy button copies the post's address and says so, and
// the system's share sheet is offered where the browser has one and the pointer is a finger, that is on a phone.
// Plain ES5: the production build minifies the site's scripts with Uglifier.
(function () {
  var row = document.querySelector(".post-share");
  if (!row) return;
  var native = row.querySelector(".post-share-native");
  var copy = row.querySelector(".post-share-copy");
  var status = row.querySelector(".post-share-status");
  var url = copy.getAttribute("data-url");
  var heading = document.querySelector(".post-title");
  var timer = null;

  if (navigator.share && window.matchMedia && window.matchMedia("(pointer: coarse)").matches) {
    native.hidden = false;
    native.addEventListener("click", function () {
      var shared = navigator.share({ title: heading ? heading.textContent : document.title, url: url });
      // closing the sheet without sharing rejects the promise, which is not an error
      if (shared && shared.then) shared.then(null, function () {});
    });
  }

  function done(ok) {
    copy.classList.toggle("is-done", ok);
    status.textContent = ok ? "Link copied" : "";
    clearTimeout(timer);
    timer = setTimeout(function () {
      copy.classList.remove("is-done");
      status.textContent = "";
    }, 2000);
  }

  // older browsers and pages outside a secure context have no Clipboard API
  function fallback() {
    var field = document.createElement("textarea");
    field.value = url;
    field.setAttribute("readonly", "");
    field.style.position = "fixed";
    field.style.opacity = "0";
    document.body.appendChild(field);
    field.select();
    var ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (error) {
      ok = false;
    }
    document.body.removeChild(field);
    done(ok);
  }

  copy.addEventListener("click", function () {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(url).then(function () {
        done(true);
      }, fallback);
    } else {
      fallback();
    }
  });
})();
