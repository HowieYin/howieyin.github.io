(function () {
  "use strict";
  const css = getComputedStyle(document.documentElement);
  const color = name => css.getPropertyValue(name).trim();
  const accent = color("--accent"), second = color("--accent2"), muted = color("--muted");
  const ink = color("--text"), rule = color("--line");
  const $ = id => document.getElementById(id);
  const chart = id => echarts.init($(id), null, {renderer: "svg"});
  const smoothChart = chart("smooth-chart"), memoryChart = chart("memory-chart"), latencyChart = chart("latency-chart");
  const base = {
    animation: false,
    color: [accent, second],
    textStyle: {color: ink, fontFamily: "DM Sans, PingFang SC, sans-serif"},
    tooltip: {trigger: "axis", appendToBody: true},
    grid: {left: 52, right: 20, top: 62, bottom: 45},
    legend: {top: 5, textStyle: {color: ink}},
    xAxis: {type: "category", axisLabel: {color: muted}, axisLine: {lineStyle: {color: rule}}},
    yAxis: {type: "value", axisLabel: {color: muted}, splitLine: {lineStyle: {color: rule}}, nameTextStyle: {color: muted}}
  };
  const f = x => Number(x.toFixed(4));
  function smoothing() {
    const state = SQMath.smooth(+$("alpha").value, +$("bits").value);
    window.smoothState = state;
    $("alpha-label").textContent = state.alpha.toFixed(2);
    $("bits-label").textContent = `W${state.bits}A${state.bits}`;
    smoothChart.setOption({...base, xAxis: {...base.xAxis, data: ["Ci 1", "Ci 2", "Ci 3", "Ci 4"]},
      yAxis: {...base.yAxis, name: "absmax"},
      series: [{type: "bar", name: "激活 X/s", data: state.a.map(f)}, {type: "bar", name: "权重 sW", data: state.w.map(f)}]});
    $("smooth-readout").textContent = `s=[${state.s.map(f).join(", ")}] · 浮点 max|ΔY|=${state.floatError.toExponential(2)} · ${$("bits-label").textContent} 模拟 max|ΔY|=${state.quantError.toFixed(6)} · Y=[[-16,9,16],[11,0,-11]]`;
    $("smooth-data").innerHTML = state.s.map((s, j) => `<tr><td>${j + 1}</td><td>${f(s)}</td><td>${f(state.a[j])}</td><td>${f(state.w[j])}</td></tr>`).join("");
  }
  function capacity() {
    const ids = {params:"params",layers:"layers",heads:"heads",dim:"dim",batch:"batch",length:"length",wbits:"wbits"};
    const p = {kvbits:+document.querySelector('[name="mem-kvbits"]:checked').value};
    for (const [key, id] of Object.entries(ids)) {
      const e = $("mem-" + id);
      if (e.type === "number" && !e.validity.valid) return;
      p[key] = +e.value;
      if (!Number.isFinite(p[key]) || p[key] <= 0) return;
    }
    const state = SQMath.memory(p);
    window.memoryState = state;
    const gib = b => b / 2 ** 30;
    memoryChart.setOption({...base, legend: {show: false}, xAxis: {...base.xAxis, data: ["权重", "KV cache"]},
      yAxis: {...base.yAxis, name: "GiB"},
      series: [{type: "bar", data: [{value: f(gib(state.weights)), itemStyle: {color: accent}}, {value: f(gib(state.kv)), itemStyle: {color: second}}],
        label: {show: true, position: "top", color: ink, formatter: p => p.value.toFixed(3)}}]});
    $("memory-readout").textContent = `W${p.wbits}=${gib(state.weights).toFixed(3)} GiB · KV${p.kvbits}=${gib(state.kv).toFixed(3)} GiB · 合计=${gib(state.total).toFixed(3)} GiB · 仅逻辑存储`;
  }
  function latencies() {
    const config = +document.querySelector('[name="latency-config"]:checked').value, values = SQMath.latency[config];
    window.latencyState = {config, values, speedup: values[0] / values[4]};
    latencyChart.setOption({...base, legend: {show: false}, grid: {left: 100, right: 52, top: 35, bottom: 45},
      xAxis: {type: "value", name: "ms", axisLabel: {color: muted}, splitLine: {lineStyle: {color: rule}}},
      yAxis: {type: "category", inverse: true, data: ["FP16", "LLM.int8()", "SQ O1", "SQ O2", "SQ O3"], axisLabel: {color: ink}, axisLine: {lineStyle: {color: rule}}},
      series: [{type: "bar", data: values.map((v, j) => ({value: v, itemStyle: {color: j < 2 ? muted : accent}})),
        label: {show: true, position: "right", color: ink, formatter: p => p.value.toFixed(1)}}]});
    $("latency-readout").textContent = `O3 vs FP16：${(values[0] / values[4]).toFixed(3)}× · latency 降低 ${(100 * (1 - values[4] / values[0])).toFixed(2)}% · O1→O3 降低 ${(100 * (1 - values[4] / values[2])).toFixed(2)}%`;
  }
  ["alpha", "bits"].forEach(id => $(id).addEventListener("input", smoothing));
  document.querySelectorAll("#memory-widget input,#memory-widget select").forEach(e => e.addEventListener("input", capacity));
  document.querySelectorAll('[name="latency-config"]').forEach(e => e.addEventListener("change", latencies));
  smoothing(); capacity(); latencies();
  [smoothChart, memoryChart, latencyChart].forEach(c => new ResizeObserver(() => c.resize()).observe(c.getDom()));
  window.addEventListener("resize", () => [smoothChart, memoryChart, latencyChart].forEach(c => c.resize()));
})();
