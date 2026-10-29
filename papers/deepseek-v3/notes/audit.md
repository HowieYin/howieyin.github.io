# DeepSeek-V3 v2 — 两轮审校

审核日期：2026-10-29。范围：53 页指定 PDF、作者 HTML/TeX、原始图表、官方固定代码、扩展资料与本页渲染。

## 第一轮：结构与可读性

- 按 Abstract / §1–6 / Appendix A–C 顺序展开，再追加实验审计、相关工作、当前进展、代码边界与个人见解。
- 所有 11 组原图、15 张图像、9 张原表、28 个编号公式保留并相邻解读；两张 Mermaid 与三组交互标明 Agent-created。
- 原图 7 仅将外部分面说明移到纯文本，作者完整 PDF 保留；轴、legend、结构标注不裁除。
- 页内目录、图片键盘放大、横向表格/公式、暗亮主题和离线资产使用现有站点结构。
- 根据全文标题树修正 post-training 原章节编号：§5.3.1 settings、5.3.2 standard、5.3.3 open-ended、5.3.4 reward model；§5.4.1 distill、5.4.2 self-reward、5.4.3 MTP。
- 补充 FP8 mantissa-vs-exponent、online-vs-delayed scale、reward model 构造和 Appendix A 实际角色分类。
- 真实浏览器 QA 与截图记录单独保存，发布前后均检查。

## 第二轮：原始证据与数值

- `audit.py` 对 9 张原表所有 cells 的文本、rowspan、colspan 与 bold 数量逐项一致性检查；编号公式的 MathML alttext 与作者提取结果逐项一致。
- 保留 671B total / 37B activated、61 层、前 3 dense、256+1 experts、Top-8、KV512+RoPE64、正式 MTP D=1；图示 D=2 单独说明。
- MLA 128K/B1/BF16：9,210,691,584 bytes = 8.578125 GiB；参照同形状物化 K/V，压缩比 71.1111，而非推理加速比。
- 固定推理配置 vocab=129280、route_scale=2.5、8 groups 选4；bias 只参与 selection，原 sigmoid 分数参与 mixture。
- 专家矩阵总量复算后修正为 656.463B；参数子项不能替代论文主体/激活总量。
- Table 1 2788K GPUh 与 5.576M 美元仅正式训练、作者价格，不含研究/教师/企业全部成本。
- FP8 E4M3、1×128 activation / 128×128 weights、128 元素 promotion；敏感激活 E5M6，不误写 E5M2。
- Appendix B loss error <.25% 是16B/约230B实验与 EMA=.9，不是671B完整BF16对照或逐位相同。
- Table 4/5 保留 MMLU 等下降项；batch-wise auxiliary 与 aux-free 同验证 loss，限制“无辅助梯度独有收益”的归因。
- Arena-Hard 采用表85.5，正文>86%冲突明确；Appendix PDF重复Figure10/HTML Figure11、五个panel原标签与实际层号同时标明。
- Table 6 数学提升逐项复算为15.9/10.2/27.3pp；SWE/Aider-Edit弱项与Aider-Polyglot例外同时保留。
- Table 9 数学长度增长96.36%，与V3 MTP TPS不混成联合实测；不虚构TP/FP/TN/FN。
- 最新资料截止2026-10-29，含V4.1官方更新；历史/当前benchmark预算分开，不宣称不可比的全局SOTA。
- PyTorch/Nebius 918/651吞吐+41.01%对应时间−29.08%；相对859的FP8增量6.87%，不把系统全部增益归给FP8。
- FlashMLA当前main已删除历史V3/Hopper支持，提供兼容固定提交；DeepEP当前V2.5与NVSHMEM历史实现分开。

## 验证边界

本地运行四项有意义的计算测试：KV字节/复制，路由选择与无偏权重，MTP成本阈值，同组MLA矩阵吸收代数。先确认未实现时失败，再实现并全部通过。没有执行 GPU 训练、NPU 部署、TTFT/TPOT 或吞吐测量。

原始数值与完整 cell 验证结果见 [numerical-check.json](numerical-check.json)；来源见 [sources.json](sources.json)。
