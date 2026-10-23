// Long code blocks in a post start folded: the first lines show, the rest fades out under a button that opens the
// whole block in place. Short blocks are left alone. Without JavaScript, or when printing, every block is open.
// The copy button still copies the whole block, since folding only clips it.
(function () {
  const FOLD_AFTER = 18; // a block longer than this many lines is folded...
  const SHOW = 10; // ...to this many

  function lines(pre) {
    return pre.innerText.replace(/\n$/, "").split("\n").length;
  }

  function language(block) {
    const holder = block.closest("[class*='language-']");
    const match = holder && holder.className.match(/language-([\w+-]+)/);
    if (!match || ["plaintext", "text"].includes(match[1])) return "";
    const names = { python: "Python", bash: "shell", shell: "shell", sh: "shell", js: "JavaScript" };
    return names[match[1]] || match[1];
  }

  function fold(pre, index) {
    const count = lines(pre);
    if (count <= FOLD_AFTER) return;
    // the outermost box of the block: kramdown's wrapper, or the copy button's wrapper, or the pre itself
    const box = pre.closest(".highlighter-rouge") || pre.closest(".code-display-wrapper") || pre;
    const lang = language(pre);
    const id = pre.id || "code-block-" + index;
    pre.id = id;
    box.classList.add("code-fold", "is-folded");
    box.style.setProperty("--fold-lines", SHOW);

    const button = document.createElement("button");
    button.type = "button";
    button.className = "code-fold-toggle";
    button.setAttribute("aria-controls", id);
    button.setAttribute("aria-expanded", "false");
    const closedLabel = "Show all " + count + " lines" + (lang ? " of " + lang : "");
    button.textContent = closedLabel;
    button.addEventListener("click", function () {
      const opening = box.classList.contains("is-folded");
      box.classList.toggle("is-folded", !opening);
      button.setAttribute("aria-expanded", String(opening));
      button.textContent = opening ? "Show less" : closedLabel;
      if (!opening && box.getBoundingClientRect().top < 0) {
        box.scrollIntoView({ block: "start" });
      }
    });
    box.insertAdjacentElement("afterend", button);
  }

  function start() {
    const root = document.getElementById("markdown-content");
    if (!root) return;
    root.querySelectorAll("pre").forEach(function (pre, index) {
      if (pre.parentElement.closest("pre")) return; // rouge nests a pre for line numbers
      fold(pre, index);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
