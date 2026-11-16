// Spectrograms that play, from the course notebooks (tfs.viz.audio_spectrogram): while the sound plays, a line
// follows it over the picture, and a click on the picture plays from that moment.
(function () {
  function attach(box) {
    var img = box.querySelector("img");
    var audio = box.querySelector("audio");
    var line = box.querySelector(".tfs-playhead");
    if (!img || !audio || !line) {
      return;
    }
    var x0 = parseFloat(box.getAttribute("data-x0"));
    var x1 = parseFloat(box.getAttribute("data-x1"));
    var duration = parseFloat(box.getAttribute("data-duration"));

    function draw() {
      var f = Math.min(audio.currentTime / duration, 1);
      line.style.left = 100 * (x0 + (x1 - x0) * f) + "%";
      line.style.opacity = audio.currentTime > 0 ? 1 : 0;
      if (!audio.paused) {
        window.requestAnimationFrame(draw);
      }
    }

    function playFrom(seconds) {
      if (audio.readyState > 0) {
        audio.currentTime = seconds;
      } else {
        // the sound is not loaded yet (preload="none"): seek as soon as its length is known
        var seek = function () {
          audio.removeEventListener("loadedmetadata", seek);
          audio.currentTime = seconds;
        };
        audio.addEventListener("loadedmetadata", seek);
      }
      var started = audio.play();
      if (started && started["catch"]) {
        started["catch"](function () {}); // a browser that blocks autoplay just leaves it paused
      }
    }

    audio.addEventListener("play", draw);
    audio.addEventListener("seeked", draw);
    audio.addEventListener("ended", draw);
    img.addEventListener("click", function (event) {
      var rect = img.getBoundingClientRect();
      var f = ((event.clientX - rect.left) / rect.width - x0) / (x1 - x0);
      if (f >= 0 && f <= 1) {
        playFrom(f * duration);
      }
    });
  }

  function init() {
    var boxes = document.querySelectorAll(".tfs-audio-spec");
    for (var i = 0; i < boxes.length; i++) {
      attach(boxes[i]);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
