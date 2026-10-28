(function () {
  "use strict";
  const contents = document.getElementById("contents");
  const compact = matchMedia("(max-width:1000px)");
  contents.open = !compact.matches;
  compact.addEventListener("change", () => {contents.open = !compact.matches;});
  const headings = [...document.querySelectorAll("article h2")];
  const links = [...document.querySelectorAll(".toc a")];
  let pending = false;
  function highlight() {
    const current = headings.filter(h => h.getBoundingClientRect().top < 150).at(-1) || headings[0];
    links.forEach(a => {
      const active = a.hash === "#" + current.id;
      a.classList.toggle("active", active);
      active ? a.setAttribute("aria-current", "location") : a.removeAttribute("aria-current");
    });
    pending = false;
  }
  addEventListener("scroll", () => {if (!pending) {pending = true; requestAnimationFrame(highlight);}}, {passive:true});
  links.forEach(a => a.addEventListener("click", () => {if (innerWidth < 1000) contents.open = false;}));
  highlight();
  document.querySelectorAll(".table-scroll").forEach(region => region.addEventListener("keydown", e => {
    if (e.target !== region || !["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    e.preventDefault();
    region.scrollBy({left:e.key === "ArrowRight" ? 80 : -80, behavior:"instant"});
  }));
  const dialog = document.getElementById("figure-dialog"), image = document.getElementById("zoom-image");
  const caption = document.getElementById("zoom-caption");
  document.querySelectorAll("[data-zoom]").forEach(button => button.addEventListener("click", () => {
    image.src = button.dataset.zoom;
    image.alt = button.querySelector("img").alt;
    caption.innerHTML = button.closest("figure").querySelector("figcaption").innerHTML;
    dialog.showModal();
  }));
  document.getElementById("close-figure").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", e => {if (e.target === dialog) dialog.close();});
  document.querySelectorAll(".copy-formula").forEach(button => button.addEventListener("click", async () => {
    try {await navigator.clipboard.writeText(button.previousElementSibling.textContent);button.textContent="Copied";}
    catch {button.textContent="请选取上方 LaTeX 复制";}
  }));
  if (document.querySelector(".mermaid")) {
    mermaid.initialize({startOnLoad:false, theme:"base", securityLevel:"strict", htmlLabels:false, flowchart:{htmlLabels:false}, themeVariables:{
      primaryColor:"#172334", primaryTextColor:"#e7ebf2", primaryBorderColor:"#6384b5",
      lineColor:"#99baff", secondaryColor:"#111720", tertiaryColor:"#111720",
      clusterBkg:"#101823", clusterBorder:"#374256", fontFamily:"DM Sans, PingFang SC, sans-serif"
    }});
    mermaid.run({querySelector:".mermaid"}).then(() => {window.diagramsReady=true;}).catch(e => console.error(e));
  } else window.diagramsReady=true;
})();
