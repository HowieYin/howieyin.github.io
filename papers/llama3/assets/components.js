(() => {
  'use strict';
  const el = id => document.getElementById(id);
  function bars(target, labels, values) {
    const maximum = Math.max(...values);
    target.replaceChildren(...values.map((v, i) => {
      const row = document.createElement('div'), text = document.createElement('span'), bar = document.createElement('i');
      text.textContent = `${labels[i]} · ${v.toFixed(4)} GiB`;
      bar.style.width = `${v / maximum * 100}%`;
      bar.setAttribute('aria-hidden', 'true');
      row.append(text, bar);
      return row;
    }));
  }
  function kv() {
    const [L, H] = [[32,32],[80,64],[126,128]][+el('kv-model').value];
    const S = +el('kv-seq').value, B = +el('kv-batch').value, shards = +el('kv-shard').value;
    const gqa = 2*L*B*S*8*128*2/2**30, mha = gqa*H/8;
    bars(el('kv-bars'), ['GQA total', 'MHA total', 'GQA / selected partition'], [gqa,mha,gqa/shards]);
    el('kv-result').textContent = `GQA=${gqa.toFixed(4)} GiB; MHA=${mha.toFixed(4)} GiB\nSaving factor=${H/8}×; ideal GQA per partition=${(gqa/shards).toFixed(4)} GiB\nBytes=2 × ${L} × ${B} × ${S} × 8 × 128 × 2`;
  }
  ['kv-model','kv-seq','kv-batch','kv-shard'].forEach(id => el(id).addEventListener('input',kv));
  kv();
  function parallel() {
    const C = +el('cp-size').value, blocks = 2*C, rows = [];
    for (let rank = 0; rank < C; rank++) {
      const row = document.createElement('div');
      row.className = 'cp-row';
      const text = document.createElement('span');
      text.textContent = `Rank ${rank}`;
      row.append(text);
      for (let block = 0; block < blocks; block++) {
        const cell = document.createElement('b');
        cell.textContent = String(block);
        if (block === rank || block === blocks-1-rank) cell.className = 'owned';
        row.append(cell);
      }
      row.style.gridTemplateColumns = `68px repeat(${blocks},minmax(20px,1fr))`;
      rows.push(row);
    }
    el('cp-blocks').replaceChildren(...rows);
    const config = [
      [8192,8,1,16,64,8192,32,43],
      [16384,8,1,16,128,8192,16,41],
      [16384,8,16,16,8,131072,16,38]
    ][+el('cp-config').value];
    const [gpu,tp,cp,pp,dp,s,b,mfu] = config;
    const count = s*b*dp;
    el('cp-result').textContent = `Illustration CP=${C}: rank i owns blocks i and ${blocks}-1-i\nPaired causal work=${blocks} units for every rank\nContiguous pairs range=2 to ${4*C-2} units\nPaper config: TP${tp} × CP${cp} × PP${pp} × DP${dp}=${gpu} GPUs\nSequence × batch/DP × DP=${count.toLocaleString('en-US')} tokens/global batch (~16M)\nPaper MFU=${mfu}% (not availability)`;
  }
  ['cp-size','cp-config'].forEach(id => el(id).addEventListener('input',parallel));
  parallel();
  function metrics() {
    const n = +el('metric-n').value, prevalence = +el('metric-rate').value;
    const p=n*prevalence, negatives=n-p, tp=p*.714, fn=p-tp, fp=negatives*.01, tn=negatives-fp;
    const precision=tp/(tp+fp), recall=tp/p, f1=2*tp/(2*tp+fp+fn), accuracy=(tp+tn)/n;
    el('metric-grid').replaceChildren(...[['TP',tp],['FP',fp],['FN',fn],['TN',tn]].map(([label,value]) => {
      const cell=document.createElement('div');cell.textContent=`${label} = ${value.toFixed(1)}`;return cell;
    }));
    el('metric-result').textContent=`Expected counts: N=${n}, positives=${p}, negatives=${negatives}\nTP=${tp.toFixed(1)}, FP=${fp.toFixed(1)}, TN=${tn.toFixed(1)}, FN=${fn.toFixed(1)}\nAccuracy=${(accuracy*100).toFixed(2)}%; Precision=${(precision*100).toFixed(2)}%\nRecall=${(recall*100).toFixed(2)}%; F1=${(f1*100).toFixed(2)}%; FPR=1.00%\nIllustrative expectation, not paper experiment counts.`;
  }
  ['metric-n','metric-rate'].forEach(id => el(id).addEventListener('input',metrics));
  metrics();
})();
