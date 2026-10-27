(() => {
  const main = document.querySelector("main.report");
  const existing = main?.querySelector("details.toc");
  if (!existing) return;
  const sidebar = document.createElement("aside");
  sidebar.className = "reading-sidebar";
  sidebar.id = "article-contents";
  sidebar.setAttribute("aria-label", "Article contents");
  sidebar.innerHTML = '<button type="button" class="contents-close">Close contents</button><h2>Contents</h2><nav aria-label="Table of contents"></nav>';
  sidebar.querySelector("nav").append(existing.querySelector("ul"));
  existing.remove();
  const content = document.createElement("div");
  content.className = "reading-content";
  while (main.firstChild) content.append(main.firstChild);
  main.append(sidebar, content);
  document.body.classList.add("reading-page");
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "contents-toggle";
  toggle.textContent = "Contents";
  toggle.setAttribute("aria-controls", sidebar.id);
  toggle.setAttribute("aria-expanded", "false");
  const backdrop = document.createElement("button");
  backdrop.type = "button";
  backdrop.className = "contents-backdrop";
  backdrop.setAttribute("aria-label", "Close contents");
  document.body.append(toggle, backdrop);
  const closeButton = sidebar.querySelector(".contents-close");
  const mobile = matchMedia("(max-width: 1000px)");
  const setOpen = (open, restoreFocus = true) => {
    sidebar.classList.toggle("is-open", open);
    backdrop.classList.toggle("is-open", open);
    document.body.classList.toggle("contents-open", open);
    toggle.setAttribute("aria-expanded", String(open));
    if (open) closeButton.focus();
    else if (restoreFocus) toggle.focus();
  };
  toggle.addEventListener("click", () => setOpen(true));
  closeButton.addEventListener("click", () => setOpen(false));
  backdrop.addEventListener("click", () => setOpen(false));
  mobile.addEventListener("change", () => setOpen(false, false));
  document.addEventListener("keydown", (event) => {
    if (!sidebar.classList.contains("is-open")) return;
    if (event.key === "Escape") setOpen(false);
    if (event.key !== "Tab") return;
    const controls = [...sidebar.querySelectorAll("button, a")];
    const first = controls[0];
    const last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });
  const links = [...sidebar.querySelectorAll("a")];
  const headings = links.map((link) => document.getElementById(decodeURIComponent(link.hash.slice(1)))).filter(Boolean);
  links.forEach((link) => link.addEventListener("click", () => {
    if (mobile.matches) {
      setOpen(false, false);
      const heading = document.getElementById(decodeURIComponent(link.hash.slice(1)));
      if (heading) {
        heading.setAttribute("tabindex", "-1");
        heading.focus({ preventScroll: true });
      }
    }
  }));
  const highlight = () => {
    const current = [...headings].reverse().find((heading) => heading.getBoundingClientRect().top <= 140) || headings[0];
    links.forEach((link) => {
      if (current && decodeURIComponent(link.hash.slice(1)) === current.id) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  };
  window.addEventListener("scroll", highlight, { passive: true });
  highlight();
})();
