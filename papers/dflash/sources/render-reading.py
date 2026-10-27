from pathlib import Path
import re, html, json, sys
from bs4 import BeautifulSoup
import markdown
from latex2mathml.converter import convert
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

root = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
for p in (root/'sources').iterdir():
    if p.suffix in {'.html','.bib','.py'}:
        p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
paper = BeautifulSoup((root/'sources/paper-v2.html').read_text(), 'html.parser')
equations = {
    '1': ('Original Paper · Equation (1)', r'L=\frac{T_{\mathrm{draft}}+T_{\mathrm{verify}}}{\tau}'),
    '2': ('Original Paper · Equation (2)', r'T_{\mathrm{draft}}=\gamma\cdot t_{\mathrm{step}}'),
    '3': ('Original Paper · Equation (3)', r'T_{\mathrm{draft}}=t_{\mathrm{parallel}}'),
    '4': ('Original Paper · Equation (4)', r'w_k=\exp\left(-\frac{k-1}{\gamma}\right)'),
    'prefix': ('Interpretation · conditional prefix survival', r'\mathbb{E}[\tau]=1+\sum_{k=1}^{\gamma}\prod_{j=1}^{k}a_j'),
    'loss': ('Interpretation · weighted cross entropy', r'\mathcal{L}=-\sum_j\sum_{k=1}^{B-1}w_k\log[\mathrm{softmax}(z_{jk})]_{y_{jk}}'),
    'fuse': ('Original Paper · Appendix A.3 · fusion', r'\mathbf{H}_t=\mathrm{RMSNorm}\left(W_c[\mathbf{H}^{(l_1)};\ldots;\mathbf{H}^{(l_5)}]\right)'),
    'kv': ('Original Paper · Appendix A.3 · per-layer injection', r'\begin{aligned}\mathbf{Q}_i&=W_i^Q\mathbf{H}_d\\\mathbf{K}_i&=[W_i^K\mathbf{H}_t;W_i^K\mathbf{H}_d]_{\mathrm{seq}}\\\mathbf{V}_i&=[W_i^V\mathbf{H}_t;W_i^V\mathbf{H}_d]_{\mathrm{seq}}\end{aligned}'),
    'memory': ('Original Paper · Appendix A.3 · projection weights', r'5\times2048\times2048\times2\approx42\,\mathrm{MB}'),
    'cache': ('Interpretation · persistent draft KV payload', r'M_{\mathrm{draftKV}}=2L_dbSGd_hs_{\mathrm{elem}}'),
}
figs = ['dflash_speedup','dflash_inference_design','draft_latency_bar','dflash_attn','acceptance_length_vs_epoch']
alts = [
    'Qwen3-8B七任务加速柱图：DFlash为2.75至6.08倍，EAGLE-3为1.81至2.23倍',
    '目标模型多层特征融合后进入草稿每层KV，anchor与masks经双向attention并行预测',
    '草稿token预算4、8、16下，一层EAGLE-3与一、三、五层DFlash的延迟对比',
    '训练attention矩阵：每块可见anchor之前的targetfeatures，块内双向，块间隔离',
    '有无位置lossdecay的Math500接受长度随训练epoch变化的曲线'
]
original_figures = paper.select('figure.ltx_figure')

def fig(n):
    n = int(n)
    caption = original_figures[n-1].select_one('figcaption')
    for a in caption.select('a[href^="#"]'):
        a['href'] = 'https://arxiv.org/html/2602.06036v2' + a['href']
    file = f'assets/original/fig-{n:02d}-{figs[n-1]}'
    return f'''<figure class="original" id="original-figure-{n}"><a class="zoom-link" href="{file}.png"><img src="{file}.png" alt="{html.escape(alts[n-1])}" loading="lazy"></a><figcaption><span class="component-label">ORIGINAL PAPER · FIGURE {n} · UNCROPPED</span>{caption.decode_contents()}<br><a href="{file}.pdf">Original vector PDF</a> · Click to zoom · Source: Chen, Liang, Liu / CC BY 4.0 [1]</figcaption></figure>'''

def equation(name):
    label, latex = equations[name]
    math=convert(latex, display="block")
    if name=='kv':
        rows=[]
        for row in [
            r'\mathbf{Q}_i=W_i^Q\mathbf{H}_d',
            r'\mathbf{K}_i=[W_i^K\mathbf{H}_t;W_i^K\mathbf{H}_d]_{\mathrm{seq}}',
            r'\mathbf{V}_i=[W_i^V\mathbf{H}_t;W_i^V\mathbf{H}_d]_{\mathrm{seq}}'
        ]:
            content=BeautifulSoup(convert(row),'html.parser').select_one('math').decode_contents()
            rows.append('<mtr><mtd>'+content+'</mtd></mtr>')
        math='<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mtable columnalign="left" rowspacing="0.6em">'+''.join(rows)+'</mtable></math>'
    return f'<div class="equation" id="equation-{name}"><span class="equation-label">{label}</span>{math}<details class="latex-source"><summary>LaTeX source</summary><code>{html.escape(latex)}</code></details></div>'

def table(n):
    n = int(n)
    soup = BeautifulSoup((root/'sources'/f'table-{n:02d}.html').read_text(), 'html.parser')
    figure = soup.select_one('figure')
    element = figure.select_one('table')
    # ltx column width rules are source-specific; preserve semantic layout,
    # permit horizontal scrolling instead of shrinking text.
    for x in element.select('[style]'):
        x['style'] = re.sub(r'(?:width|max-width|min-width)\s*:[^;]+;?', '', x['style'])
    element['data-original-table'] = str(n)
    for wrapper in figure.select('.ltx_transformed_outer,.ltx_transformed_inner'):
        wrapper.unwrap()
    element.wrap(soup.new_tag('div', attrs={'class':'table-wrap','role':'region','tabindex':'0','aria-label':f'Original Table {n}, horizontal scroll'}))
    figure['class'] = ['original-table']
    return str(figure)

prefix = '''<section class="interactive" id="prefix-explorer" aria-labelledby="prefix-title">
<p class="component-label">AGENT-CREATED INTERACTIVE · MECHANISM, NOT BENCHMARK</p>
<h4 id="prefix-title">第一处拒绝，为什么使后面的“正确”也失效？</h4>
<p>玩具块有一个已知 anchor 和四个 proposals，B=5。改变首个错误位置，逐步观察起草、target 验证、提交与 bonus。后面的相等不能绕过中间错误。</p>
<div class="controls"><label>First mismatch<select id="mismatch"><option value="0">Position 1 · none accepted</option><option value="2" selected>Position 3 · accept first 2</option><option value="4">No mismatch · accept all 4</option></select></label>
<label>Conditional acceptance a<input id="accept-prob" type="range" min="0" max="1" step="0.05" value=".8"><span id="accept-prob-value">0.80</span></label></div>
<ol class="step-track" id="step-track"><li>1 · Context</li><li>2 · Parallel draft</li><li>3 · Target verify</li><li class="current">4 · Commit prefix</li></ol>
<div class="token-line"><span>Draft</span><div id="draft-tokens"><b>speculative</b><b>decoding</b><b>is</b><b>fast</b></div></div>
<div class="token-line"><span>Target</span><div id="target-tokens"><b>speculative</b><b>decoding</b><b>verified</b><b>fast</b></div></div>
<div class="button-row"><button id="prefix-play" type="button">Play</button><button id="prefix-step" type="button">Next step</button><button id="prefix-replay" type="button">Replay</button></div>
<output id="prefix-reading" aria-live="polite">Commit: accepted proposals = 2; τ = 3. Position 4 is discarded after the first mismatch.</output>
<output id="survival-reading">E[τ] = 3.3616 for 4 conditional probabilities of 0.80.</output>
<noscript><p>静态示例：前两位通过，第三位拒绝，第四位即使相等也舍弃。τ=2+1=3。</p></noscript></section>'''

serving = '''<section class="interactive" id="serving-explorer" aria-labelledby="serving-title">
<p class="component-label">ORIGINAL MEASUREMENTS · TABLE 3 · B200 / SGLANG FA4</p>
<h4 id="serving-title">提高并发：吞吐与加速比分别发生了什么？</h4>
<div class="controls"><label>Model / task<select id="serving-task"><option value="q8math">Qwen3-8B · Math500</option><option value="q8code">Qwen3-8B · HumanEval</option><option value="coder">Qwen3-Coder · HumanEval</option></select></label><label>Concurrency<select id="serving-concurrency"><option value="0">1</option><option value="1">4</option><option value="2">8</option><option value="3">16</option><option value="4">32</option></select></label></div>
<div id="serving-bars" class="metric-bars"><div><span>AR baseline · 230 tok/s</span><i style="width:19.57%"></i></div><div><span>DFlash · 1175 tok/s</span><i style="width:100%"></i></div></div>
<output id="serving-reading" aria-live="polite">230 → 1175 tok/s; recalculated speedup 5.109×; +410.9%. Paper rounded: 5.1×.</output>
<p class="fine">只切换原表离散点，不插值。τ 为原表跨并发报告的平均，不虚构每并发 τ；比值从 tok/s 重新计算。</p>
<noscript><p>静态默认：Qwen3-8B Math500，C1，AR230 / DFlash1175 tok/s。全部并发原数据见 Table 3。</p></noscript></section>'''

cost = '''<section class="interactive" id="cost-explorer" aria-labelledby="cost-title">
<p class="component-label">AGENT-CREATED ANALYTICAL MODEL · NOT MEASURED MEMORY OR LATENCY</p>
<h4 id="cost-title">分别计算 projection、持久 KV 和每 token 时间</h4>
<p>固定 D=2048、F=5、BF16；KV head width=128。默认5层/8KV heads是示意配置。调整长度、并发和层数，观察哪项随 S 增长。</p>
<div class="controls"><label>Context length S<select id="memory-length"><option value="2048">2048</option><option value="8192">8192</option><option value="32768">32768</option><option value="131072">131072</option></select></label><label>Batch b<select id="memory-batch"><option value="1">1</option><option value="8">8</option><option value="32">32</option></select></label><label>Draft layers<select id="memory-layers"><option value="3">3</option><option value="5" selected>5</option><option value="8">8</option></select></label><label>KV heads G<select id="memory-groups"><option value="2">2</option><option value="8" selected>8</option><option value="16">16</option></select></label></div>
<output id="memory-reading" aria-live="polite">Projection weights: 40.00 MiB. Input / output activation: 40.00 / 8.00 MiB. Persistent draft KV: 40.00 MiB.</output>
<p class="fine">Projection 权重固定，输入/输出是整个上下文的瞬时示例，decode 只投影新增位置；KV 为持久逻辑 payload。未计 draft Transformer、target KV、logits、allocator、workspace。</p>
<div class="controls"><label>AR target latency (ms/token)<input id="target-ms" type="number" min=".1" max="100" step=".1" value="10"></label><label>Draft latency (ms/cycle)<input id="draft-ms" type="number" min=".1" max="100" step=".1" value="2"></label><label>Verify latency (ms/cycle)<input id="verify-ms" type="number" min=".1" max="100" step=".1" value="12"></label><label>Effective τ<input id="cost-tau" type="range" min="1" max="16" step=".5" value="7"><span id="cost-tau-value">7.0</span></label></div>
<output id="cost-reading" aria-live="polite">Assumed cycle 14.00 ms / τ 7.0 = 2.00 ms/token; speedup 5.00×.</output>
<noscript><p>静态例子：projection40MiB；示意draftKV40MiB；假设10ms target与14ms/cycle、τ7得到5×，非论文测量。</p></noscript></section>'''

text = (root/'index.md').read_text()
text = re.sub(r'\[\[FIGURE:(\d+)\]\]', lambda m:fig(m[1]), text)
text = re.sub(r'\[\[TABLE:(\d+)\]\]', lambda m:table(m[1]), text)
text = re.sub(r'\[\[EQ:(\w+)\]\]', lambda m:equation(m[1]), text)
def diagram(m):
    name = m[1]
    svg = (root/'assets/generated'/f'{name}.svg').read_text()
    svg = re.sub(r'<\?xml.*?\?>', '', svg)
    for old in sorted(re.findall(r'\bid="([^"]+)"',svg), key=len, reverse=True):
        svg=svg.replace(f'id="{old}"',f'id="{name}-{old}"')
        svg=re.sub(r'#'+re.escape(old)+r'(?![\w-])', '#'+name+'-'+old, svg)
    return f'<figure class="generated" id="diagram-{name}"><div class="diagram-scroll" tabindex="0" role="region" aria-label="Explanatory diagram, horizontal scroll">{svg}</div><figcaption>Agent-created Visualization · Simplified diagram, not an original paper figure. <a href="assets/generated/{name}.mmd">Mermaid source</a> · <a href="assets/generated/{name}.svg">SVG</a></figcaption></figure>'
text = re.sub(r'\[\[DIAGRAM:(\w+)\]\]', diagram, text)
for k,v in {'prefix':prefix,'serving':serving,'cost':cost}.items():
    text=text.replace(f'[[INTERACTIVE:{k}]]', v)
assert '[[' not in text
body = BeautifulSoup(markdown.markdown(text, extensions=['tables','fenced_code']), 'html.parser')
toc = []
for i,h in enumerate(body.select('h2,h3,h4'),1):
    if not h.get('id'):h['id']=f'section-{i}'
    if h.name=='h2':
        toc.append(f'<li><a href="#{h["id"]}">{html.escape(h.get_text())}</a></li>')
for t in body.select('table:not([data-original-table])'):
    t.wrap(body.new_tag('div',attrs={'class':'table-wrap','role':'region','tabindex':'0','aria-label':'Explanatory table, horizontal scroll'}))
for pre in body.select('pre'):
    code = pre.get_text()
    block = BeautifulSoup(f'<div class="code-frame"><div class="code-head"><span>PYTHON · EXPLANATORY PSEUDOCODE</span><button class="copy-button" type="button">Copy</button></div>{highlight(code,PythonLexer(),HtmlFormatter())}</div>','html.parser')
    pre.replace_with(block)
# Keep source table ids and caption math while removing arXiv responsive widths.
out = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="dark light"><meta name="theme-color" content="#0c1016"><title>DFlash：并行扩散草稿与目标特征 KV — Paper Readings · Howie Yin</title><meta name="description" content="DFlash v2逐章精读：5张原图、13张完整表、核心公式与附录，prefix验证动画、并发吞吐和KV内存交互，实验审计及2026-10研究更新。"><link rel="canonical" href="https://howieyin.github.io/papers/dflash/"><meta property="og:title" content="DFlash：让目标模型提供上下文，让扩散模型并行起草"><meta property="og:description" content="全图表、逐章方法重构、实验审计与可交互的前缀验证和性能账单。"><meta property="og:type" content="article"><meta property="og:image" content="https://howieyin.github.io/papers/dflash/assets/original/fig-01-dflash_speedup.png"><link rel="icon" href="../../assets/favicon.svg"><link rel="stylesheet" href="../../styles.css?v=3"><link rel="stylesheet" href="assets/reading.css"></head><body><a class="skip-link" href="#main">Skip to content</a><div class="page reading-page"><header class="site-header"><a class="wordmark" href="../../"><span class="brand-mark" aria-hidden="true"></span>howieyin</a><nav aria-label="Main navigation"><a href="../../weekly/">Weekly</a><a href="../../foundations/">Foundations</a><a href="../../lab/">Lab</a><a href="../" aria-current="page">Paper Readings</a><a href="../../#about">About</a></nav></header><div class="page-tools"><span>5 FIGURES · 13 TABLES · ARXIV V2</span><button type="button" id="theme-toggle">Light mode</button></div><div class="reading-layout"><aside class="toc"><details id="contents" open><summary>Contents</summary><ol>{''.join(toc)}</ol></details></aside><main id="main"><article class="reading">{body}</article></main></div><footer class="site-footer"><p>Paper Readings / Howie Yin · Updated 2026-10-27</p><a href="../">Back to archive →</a></footer></div><dialog class="image-modal" id="image-modal" aria-labelledby="zoom-caption"><div class="modal-top"><p id="zoom-caption">Original paper figure</p><button id="zoom-close" type="button">Close</button></div><img id="zoom-image" alt=""><a id="zoom-original">Open full-resolution image →</a></dialog><script src="interactive/model.js"></script><script src="assets/reading.js"></script></body></html>'''
(root/'index.html').write_text('\n'.join(line.rstrip() for line in out.splitlines())+'\n')
# Repair manifest source anchors from an unmodified source tree.
manifest=json.loads((root/'sources/asset-manifest.json').read_text())
manifest=[x for x in manifest if 'figure' in x or 'table' in x]
source_tables=paper.select('figure.ltx_table')
for item in manifest:
    if 'table' in item:item['source']='https://arxiv.org/html/2602.06036v2#'+source_tables[item['table']-1]['id']
manifest += [{'file':f'assets/generated/{n}.svg','source':f'Agent-created from assets/generated/{n}.mmd','role':'simplified explanation'} for n in ['flow','evolution']]
(root/'sources/asset-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('Rendered',len(out),'bytes;',len(toc),'TOC sections; all original evidence included')
