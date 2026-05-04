(function () {
  const saved = localStorage.getItem("theme") || "dark";
  document.documentElement.dataset.theme = saved;
})();

function adSetTheme(next) {
  document.documentElement.dataset.theme = next;
  localStorage.setItem("theme", next);
  if (window.setBgTheme) window.setBgTheme(next === "dark");
}

function adWireThemeButtons() {
  document.querySelectorAll(".theme-toggle").forEach((btn) => {
    btn.setAttribute("type", "button");
    if (btn.dataset.wired) return;
    btn.dataset.wired = "1";
    btn.addEventListener("click", () => {
      const cur = document.documentElement.dataset.theme === "light" ? "light" : "dark";
      adSetTheme(cur === "dark" ? "light" : "dark");
    });
  });
}

function adSyncStepPanels() {
  ["step-config", "step-upload", "step-detect", "step-review"].forEach((id) => {
    const panel = document.getElementById(id);
    if (!panel) return;
    const hidden =
      panel.classList.contains("hide") ||
      panel.hidden ||
      panel.getAttribute("aria-hidden") === "true" ||
      panel.style.display === "none";
    const shell = panel.closest(".step-panel") || panel;
    shell.classList.toggle("ad-step-hidden", hidden);
  });
}

function adWirePills() {
  document.querySelectorAll(".pillgroup").forEach((group) => {
    if (group.dataset.wired) return;
    group.dataset.wired = "1";
    const bg = group.querySelector(".pillbg");
    const pills = Array.from(group.querySelectorAll(".pill"));
    const place = (pill) => {
      if (!pill || !bg) return;
      bg.style.left = pill.offsetLeft + "px";
      bg.style.width = pill.offsetWidth + "px";
    };
    document.fonts.ready.then(() =>
      requestAnimationFrame(() => place(group.querySelector(".pill.active") || pills[0]))
    );
    pills.forEach((pill) => {
      pill.addEventListener("click", () => {
        pills.forEach((p) => p.classList.remove("active"));
        pill.classList.add("active");
        place(pill);
        const hidden = document.querySelector(
          "#" + group.dataset.group + "-val textarea, #" + group.dataset.group + "-val input"
        );
        if (hidden) {
          hidden.value = pill.dataset.value;
          hidden.dispatchEvent(new Event("input", { bubbles: true }));
        }
      });
    });
  });
}

function adWireReviewActions() {
  if (document.documentElement.dataset.reviewActionsWired) return;
  document.documentElement.dataset.reviewActionsWired = "1";
  document.addEventListener("click", (event) => {
    const btn = event.target.closest("[data-review-payload]");
    if (!btn) return;
    event.preventDefault();

    let payload;
    try {
      payload = JSON.parse(btn.dataset.reviewPayload || "{}");
    } catch {
      return;
    }

    if (payload.action === "save" && payload.id) {
      const nameInput = document.getElementById("review-name-" + payload.id);
      const presentInput = document.getElementById("review-present-" + payload.id);
      payload.name = nameInput ? nameInput.value : "";
      payload.present = presentInput ? Boolean(presentInput.checked) : true;
    }

    const hidden = document.querySelector(
      "#review-action-payload textarea, #review-action-payload input"
    );
    const apply = document.querySelector("#review-action-apply button, #review-action-apply");
    if (!hidden || !apply) return;

    hidden.value = JSON.stringify(payload);
    hidden.dispatchEvent(new Event("input", { bubbles: true }));
    hidden.dispatchEvent(new Event("change", { bubbles: true }));
    window.setTimeout(() => apply.click(), 50);
  });
}

/* ── Canvas bokeh orbs background (Issue 14) ──────────────── */
function initCanvas() {
  const canvas = document.getElementById("bg-canvas");
  if (!canvas || canvas.dataset.wired) return;
  canvas.dataset.wired = "1";

  const ctx = canvas.getContext("2d");
  let W, H, orbs;
  let isDark = document.documentElement.dataset.theme !== "light";

  function resize() {
    W = canvas.width = innerWidth;
    H = canvas.height = innerHeight;
  }

  function makeOrbs(dark) {
    return Array.from({ length: 7 }, (_, i) => ({
      x: Math.random() * W,
      y: Math.random() * H,
      r: 180 + Math.random() * 240,
      vx: (Math.random() - 0.5) * 0.22,
      vy: (Math.random() - 0.5) * 0.18,
      hue: (i % 3 === 0 ? 44 : i % 3 === 1 ? 38 : 52) + Math.random() * 8,
      alpha: dark ? 0.045 + Math.random() * 0.05 : 0.065 + Math.random() * 0.07,
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    orbs.forEach((o) => {
      const g = ctx.createRadialGradient(o.x, o.y, 0, o.x, o.y, o.r);
      g.addColorStop(0, `hsla(${o.hue},85%,${isDark ? 55 : 62}%,${o.alpha})`);
      g.addColorStop(0.5, `hsla(${o.hue},70%,${isDark ? 42 : 55}%,${o.alpha * 0.5})`);
      g.addColorStop(1, `hsla(${o.hue},50%,${isDark ? 30 : 50}%,0)`);
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(o.x, o.y, o.r, 0, Math.PI * 2);
      ctx.fill();
      o.x += o.vx;
      o.y += o.vy;
      if (o.x < -o.r || o.x > W + o.r) o.vx *= -1;
      if (o.y < -o.r || o.y > H + o.r) o.vy *= -1;
    });
    requestAnimationFrame(draw);
  }

  window.setBgTheme = (dark) => {
    isDark = dark;
    orbs = makeOrbs(dark);
  };

  resize();
  orbs = makeOrbs(isDark);
  window.addEventListener("resize", () => {
    resize();
    orbs = makeOrbs(isDark);
  });
  draw();
}

/* ── Typewriter hero subtitle (Issue 2) ───────────────────── */
function initTypewriter() {
  const el = document.getElementById("ad-typewriter");
  if (!el || el.dataset.wired) return;
  el.dataset.wired = "1";

  const PHRASES = [
    "Upload property images and detect amenities automatically.",
    "Review and edit AI-detected room features with one click.",
    "Generate listing-ready descriptions in seconds.",
    "Classify rooms, outdoors, storage, and more with ease.",
  ];

  let phraseIdx = 0, charIdx = 0, erasing = false;
  const TYPING = 45, ERASING = 25, PAUSE = 1800;

  function tick() {
    const phrase = PHRASES[phraseIdx];
    if (!erasing) {
      if (charIdx < phrase.length) {
        el.textContent = phrase.slice(0, ++charIdx);
        setTimeout(tick, TYPING);
      } else {
        setTimeout(() => { erasing = true; tick(); }, PAUSE);
      }
    } else {
      if (charIdx > 0) {
        el.textContent = phrase.slice(0, --charIdx);
        setTimeout(tick, ERASING);
      } else {
        erasing = false;
        phraseIdx = (phraseIdx + 1) % PHRASES.length;
        tick();
      }
    }
  }
  tick();
}

function adBoot() {
  adWireThemeButtons();
  adWirePills();
  adWireReviewActions();
  adSyncStepPanels();
  initTypewriter();
  initCanvas();
}

new MutationObserver(adBoot).observe(document.documentElement, { childList: true, subtree: true });
document.addEventListener("DOMContentLoaded", adBoot);
