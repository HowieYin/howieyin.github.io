import { mergeBlock, costs } from './mechanism.mjs';
const $ = id => document.getElementById(id);
const fmt = x => Number.isFinite(x) ? x.toFixed(6) : '−∞';
const empty = () => ({ m: -Infinity, l: 0, u: 0, o: 0 });
let step = 0, timer;
const reduced = matchMedia('(prefers-reduced-motion: reduce)');

function stop() {
  clearInterval(timer); timer = undefined;
  $('soft-play').textContent = 'Play / 播放';
  $('soft-play').setAttribute('aria-pressed', 'false');
}
function inputData() {
  const common = +$('common-shift').value, offset = +$('score-offset').value;
  const scores = [1 + common, 2 + common, 3 + common + offset, 4 + common + offset];
  if ($('mask-mode').value === 'causal') scores[3] = -Infinity;
  if ($('mask-mode').value === 'empty') scores.fill(-Infinity);
  return { scores, values: [10, 20, 30, 40], block: +$('block-width').value };
}
function renderSoftmax() {
  const { scores, values, block } = inputData();
  const count = Math.ceil(scores.length / block);
  step = Math.min(step, count);
  let state = empty();
  const rows = [];
  for (let i = 0; i < step; i++) {
    state = mergeBlock(state, scores.slice(i * block, (i + 1) * block), values.slice(i * block, (i + 1) * block));
    rows.push(`<tr><td>${i + 1}</td><td>${fmt(state.m)}</td><td>${fmt(state.alpha)}</td><td>${fmt(state.beta)}</td><td>${fmt(state.l)}</td><td>${fmt(state.o)}</td></tr>`);
  }
  const full = mergeBlock(empty(), scores, values);
  $('shift-output').textContent = `common shift ${$('common-shift').value}; probability unchanged`;
  $('offset-output').textContent = `last two scores + ${$('score-offset').value}`;
  $('soft-state').innerHTML = `Processed blocks ${step}/${count}<br>m = ${fmt(state.m)} · ℓ = ${fmt(state.l)} · O = ${fmt(state.o)}<br>Dense reference O = ${fmt(full.o)}${step === count ? `<br>Absolute difference = ${Math.abs(state.o - full.o).toExponential(2)}` : '<br>Partial output; it is not the final dense result yet.'}${full.l === 0 ? '<br>All masked: zero-mass guard, output 0.' : ''}`;
  $('merge-rows').innerHTML = rows.join('') || '<tr><td colspan="6">Step to load the first KV block / 单步加载第一块</td></tr>';
  const svg = $('soft-svg');
  const narrow = svg.parentElement.clientWidth < 500;
  svg.setAttribute('viewBox', narrow ? '0 0 300 390' : '0 0 600 215');
  svg.innerHTML = scores.map((s, i) => {
    const active = i < step * block;
    const x = 7 + (narrow ? i % 2 : i) * 143;
    const y = narrow ? Math.floor(i / 2) * 174 : 0;
    const p = Number.isFinite(s) && full.l ? Math.exp(s - full.m) / full.l : 0;
    return `<g transform="translate(${x},${y})"><rect x="0" y="20" width="130" height="151" rx="7" fill="${active ? '#1d3046' : '#111b28'}" stroke="${active ? '#8bb3ef' : '#42516b'}"/><text x="65" y="45" text-anchor="middle">Key ${i + 1} · V=${values[i]}</text><text x="65" y="70" text-anchor="middle">S=${Number.isFinite(s) ? s.toFixed(1) : '−∞'}</text><rect x="18" y="${147 - 62 * p}" width="94" height="${Math.max(1, 62 * p)}" fill="#6bbdb4"/><text x="65" y="164" text-anchor="middle">${(p * 100).toFixed(2)}%</text></g>`;
  }).join('') + (narrow ? '<text x="150" y="366" text-anchor="middle">Teal: final global probability</text><text x="150" y="384" text-anchor="middle">Blue outline: loaded blocks</text>' : '<text x="300" y="198" text-anchor="middle">Teal: final global probability · Blue outline: loaded blocks</text>');
  $('soft-step').disabled = step >= count;
  if (step >= count) stop();
}
for (const id of ['block-width', 'score-offset', 'common-shift', 'mask-mode']) {
  $(id).addEventListener('input', () => { stop(); step = 0; renderSoftmax(); });
}
$('soft-step').addEventListener('click', () => { stop(); step++; renderSoftmax(); });
$('soft-reset').addEventListener('click', () => { stop(); step = 0; renderSoftmax(); });
$('soft-play').addEventListener('click', () => {
  if (timer) return stop();
  if (step >= Math.ceil(4 / +$('block-width').value)) step = 0;
  if (reduced.matches) { step = Math.ceil(4 / +$('block-width').value); renderSoftmax(); return; }
  $('soft-play').textContent = 'Pause / 暂停';
  $('soft-play').setAttribute('aria-pressed', 'true');
  timer = setInterval(() => { step++; renderSoftmax(); }, 1300);
});
reduced.addEventListener('change', stop);
document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
renderSoftmax();

const memory = x => {
  const gb = x / 1024 ** 3;
  return gb >= 1 ? `${gb.toFixed(2)} GiB` : `${(x / 1024 ** 2).toFixed(2)} MiB`;
};
function renderCosts() {
  const n = +$('sequence-length').value, d = +$('head-dimension').value;
  const batch = +$('batch-size').value, heads = +$('head-count').value;
  const bytes = +$('precision').value, sramKB = +$('sram-size').value;
  const c = costs({ n, d, batch, heads, layers: 32, bytes, sramKB });
  const twice = costs({ n: n * 2, d, batch, heads, layers: 32, bytes, sramKB });
  $('sram-output').textContent = `${sramKB} KiB assumed available; M=${c.M.toLocaleString()} words`;
  $('cost-state').innerHTML = `One S or P: ${memory(c.singleQuadraticBytes)}<br>Q/K/V/O: ${memory(c.qkvoBytes)} + row stats ${memory(c.statsBytes)}<br>Persistent KV, 32 layers: ${memory(c.kvBytes)}<br>IO main-term proxy per head: ${c.ioWords.toExponential(3)} scalar accesses<br>N → 2N: IO main term ×${(twice.ioWords / c.ioWords).toFixed(2)}${c.M !== c.capacityWords ? '<br>M capped to Nd to stay within the theorem domain.' : ''}`;
  const entries = [['One S or P', c.singleQuadraticBytes], ['Q/K/V/O', c.qkvoBytes], ['KV · 32 layers', c.kvBytes]];
  const max = Math.max(...entries.map(e => e[1]));
  const narrow = $('cost-svg').parentElement.clientWidth < 500, bar = narrow ? 276 : 570;
  $('cost-svg').setAttribute('viewBox', narrow ? '0 0 300 235' : '0 0 600 235');
  $('cost-svg').innerHTML = entries.map(([label, value], i) => `<text x="12" y="${32 + i * 72}">${label} · ${memory(value)}</text><rect x="12" y="${42 + i * 72}" width="${bar}" height="23" rx="3" fill="#1a2535"/><rect x="12" y="${42 + i * 72}" width="${Math.max(2, bar * value / max)}" height="23" rx="3" fill="${i === 0 ? '#8bb3ef' : '#6bbdb4'}"/>`).join('');
}
for (const id of ['sequence-length', 'head-dimension', 'batch-size', 'head-count', 'precision', 'sram-size']) $(id).addEventListener('input', renderCosts);
renderCosts();
window.addEventListener('resize', () => { renderSoftmax(); renderCosts(); });

const dialog = $('figure-dialog');
document.querySelectorAll('[data-zoom]').forEach(button => button.addEventListener('click', () => {
  $('zoom-image').src = button.dataset.zoom;
  $('zoom-image').alt = button.querySelector('img').alt;
  dialog.showModal();
}));
$('close-dialog').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
if (innerWidth < 1000) $('contents').open = false;
const links = [...document.querySelectorAll('.toc a')];
const headings = [...document.querySelectorAll('.article h2')];
let scrollPending = false;
function updateContents() {
  const current = headings.filter(h => h.getBoundingClientRect().top <= 120).at(-1) || headings[0];
  links.forEach(link => link.classList.toggle('active', link.hash === '#' + current.id));
  scrollPending = false;
}
window.addEventListener('scroll', () => {
  if (!scrollPending) { scrollPending = true; requestAnimationFrame(updateContents); }
}, { passive: true });
updateContents();
