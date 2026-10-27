## Implementation Notes：先把调用链和缓冲状态迁对

### 一个可核验的 greedy 推理循环

以下是 **解释性伪代码**，省略 tokenizer、EOS 和严格采样细节；不是官方函数逐字复制。输入 prefix/cache 与 LM/Medusa logits，输出合法 token 序列。`posterior` 必须来自各节点实际祖先条件；被拒节点不可进入最终 cache。

```python
while not finished:
    root = argmax(lm_logits_at_last_committed)
    ranked = [topk(x, budget[k]) for k, x in enumerate(head_logits)]
    tree_tokens = instantiate_prefix_closed_tree(root, ranked, topology)
    old_len = cache.logical_length
    outputs = target(
        tree_tokens, cache=cache,
        attention_mask=prefix_plus_ancestor_mask,
        position_ids=old_len + tree_depths,
    )
    paths = retrieve_paths(outputs, retrieve_indices)
    best, extra_accepted = verify_greedy_prefix(paths)
    accepted = retrieve_indices[best, :extra_accepted + 1]
    prefix.extend(tree_tokens[accepted])
    cache.gather_and_compact(old_len + accepted, destination=old_len)
    cache.logical_length = old_len + len(accepted)
    lm_logits_at_last_committed = outputs.lm_logits[accepted[-1]]
    head_logits = outputs.medusa_logits[:, accepted[-1]]
```

verify 中根无条件来自 LM，后续从第一处不匹配截断；根接受数不能重复加两次。复制 cache 需要处理 src/dst 重叠，通常先 gather 临时张量，再写回；更新每层长度必须一致。EOS/stop token 若出现在接受块中，最终输出与 cache 应只保留到停止位置，而不是先提交整块再仅截文本。

### 每轮耗时预算

{{EQ:latency}}

`T_heads` 包括多头残差和词表投影；`T_build` 包括 top-k、索引与 token gather；`T_verify` 是主模型 q 行工作；`T_accept` 包括概率、阈值/前缀计算；`T_commit` 包括 KV gather/copy、长度更新；`T_sync` 包括 `.item()`、CPU 分支、跨设备/分布式同步和 launch gaps。实测应记录接受长度分布与每项耗时，再计算有效 token/s。

把所有损耗归给 CUDA 版本、编译或替换模型会掩盖逻辑问题。先检查是否在 token-by-token CPU 循环反复同步、是否 materialize 所有 [K,B,q,V] logits、是否构造不必要 dense mask、是否复制整个历史 cache、是否在结束 token 上误提交，随后才考虑 kernel 和部署选型。

### TTFT、TPOT 与端侧内存

Medusa 主要减少 decode 轮数；prefill 和首个 token 路径未自然变快。应分别报告 TTFT、decode TPOT、端到端时长和成块提交的用户可见间隔，不能用 decode speedup 等比例折算 TTFT。prefill/decode 分离时，首 token 应保持统一 LM 路径及 logits 数值，避免验证策略与首 token 精度变化混在一起。

对 NPU/HTP：先核验 tree mask、按 depth 的 RoPE、动态 gather、q shape 与 KV logical length 能否在编译图内运行。固定稀疏树可能比动态树容易编译，但放大 q 也可能增加 RPC、回 CPU 的 sampling、dma-buf workspace 或中间输出。测 `dmabuf_dump` 的 HTP/ION 峰值并结合 `VmHWM` 的 CPU 峰值，分别归因权重、KV、candidate logits 与临时拷贝；不能只报模型文件大小。

采用常驻 Native C++ runner、薄 JNI、复用原始缓冲时，关键是每轮都沿同一状态提交路径，按 token/cycle 记录 accept length 和毫秒级计时。量化版头准确率下降应优先检查 hidden/LM/head 量化与转换输出的精度，再检查 calibration 分布；未来边缘预测会放大某些敏感层误差，不能默认归咎于模型尺寸。


## Critique：长处、代价与尚未回答的问题

**Novelty。** 多头并行预测并非 Medusa 首创，2018 blockwise 已有；贡献在适配 LLM 的训练 recipe、树验证、稀疏拓扑和开放实现整合。将它放到正确历史脉络更能看清工程价值。

**Correctness。** tree mask/positions/commit 构成可正确实现的验证流程；typical 是经验放宽规则，不能主张 distribution equality。原文 rejection 的极端例子、Table 5 shape、Figure 8 caption 与部分训练符号需要澄清。保留这些错误信息及纠正说明比改写原图更可靠。

**Experiment quality。** 多模型、多类别、dense/sparse、threshold 和 warmup 有覆盖；缺统一严格/放宽接受拆分、置信区间、端到端强 baseline、动态服务和能耗。主结果相对原文 baseline 有说服力，普遍质量无损与任意场景加速没有被证明。

**Scalability / Memory。** 不加载第二套完整 backbone 是优点；K(d²+dV) heads、q 候选 KV、全词表 logits 和 gather workspace 都会占内存。大词表、多头、长 S、大 B 可能改变瓶颈；分布式输出 head 的 vocab parallel 还会引入通信。

**Generalization / Hardware。** 固定树校准方便、可预计算，但领域迁移可能降低 rank 命中；长 context、高并发和端侧小算力不一定有同样余量。附录 roofline 提供趋势依据，不能代替真实服务测量。

**Reproducibility。** 有官方源码与模型 recipe，复现起点清楚；仍需固定 commit、模型/模板版本、精度、树实际节点数、采样 flag 和 timer 范围。源码中数值分支与论文描述不同处，应如实区分源码观察和运行验证。


## My Takeaways：最值得迁移的是每轮有效工作量

**以下为独立分析 / Interpretation。** Medusa 最有价值的系统观点是：优化单 token 内核之外，还可以增加同一轮权重读取的有效产出。更高 FLOP/s 只有通过接受 token 和真实迭代时间才转换为用户收益；Tables 6–8、Figure 4 和后续同条件 drafter 结果共同说明，accepted length 不是唯一优化目标。

第一个值得做的方向是 **把树预算与当前资源状态联合控制**：按 B、S、当前延迟预算与在线置信度选 q，而不是沿用固定 64。测试时保持输出契约一致，报告每项 per-op 成本；如果 host 同步占主要部分，增加预测头准确率也未必解决速度瓶颈。

第二个方向是 **减少与最终路径无关的大词表投影和临时状态**：复用隐藏状态、仅算需要的 head、延迟投影到可确定的节点，并核验这样做是否能保持下一轮草稿可用。它应与 KV 压实、EOS 处理和异步 prepare 协同设计，而不是孤立加一个更快 GEMM。

第三个方向是 **端侧固定拓扑与精度联动**：在编译图支持的固定 q 下测 HTP/CPU 内存和 TTFT/TPOT，把 head 的预测变化追踪回 hidden/LM/head 转换精度与校准样本。先证明合法路径和 cache 一致，再测接受长度是否补偿 RPC、copy 和工作空间成本。GPU 的空闲算力直觉需要在真实设备上重新成立。

最重要的阅读结论是：应分别写清 **验证的目标模型、输出分布契约和运行时成本**。这三项一致，精度与速度的比较才可解释、可复现，也才能知道改进来自哪里。

<p class="reading-meta">Changelog · 2026-10-27：首次精读；完整原图原表、两幅解释图、三个交互、源码与实验审计、截至当日的技术更新。</p>
