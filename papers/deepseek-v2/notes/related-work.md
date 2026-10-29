# 扩展调研 — 截止 2026-10-29

以下均为访问核验的一手来源。作者性能属于其报告口径，未本地GPU复现。当前信息与2024历史区分。

| 工作 | 日期 | 机制与关系 | 优势与边界 | 一手来源 |
|---|---|---|---|---|
| MQA | 2019 | 所有Q共享1组KV，省带宽；MLA保持多头权重，共享latent | K/V表达约束不同，V2的MQA低分不证明所有MQA弱 | https://arxiv.org/abs/1911.02150 |
| GQA | 2023-05 | Q分组共享KV；MLA576元素相当2.25组容量 | 2.25是容量比喻不是可配置小数group | https://arxiv.org/abs/2305.13245 |
| DeepSeekMoE | 2024-01-11 | fine segmentation+shared isolation，是V2FFN的直接前身 | 更多expert组合也增加路由/通信压力 | https://arxiv.org/abs/2401.06066 |
| DeepSeekMath | 2024-02-05 | GRPO从组奖励估计baseline，V2扩展到双阶段alignment | 无critic但有rollout/RM/ref/old策略与variance问题 | https://arxiv.org/abs/2402.03300 |
| DeepSeek-V2 | 2024-05/06 | latent feature compression+decoupledRoPE+top6deviceboundMoE | 236B21Bactive，kernel/quant/batch共同决定服务效率 | https://arxiv.org/abs/2405.04434 |
| DeepSeek-V3 | 2024-12-27,v2 2025-02-18 | 保留MLA/MoE，aux-loss-free balance、MTP、FP8训练 | 671B/37B，14.8T，不与V2等训练量 | https://arxiv.org/abs/2412.19437 |
| DeepSeek-V3.2 | 2025-12-02 | DSA在MLA上token索引稀疏选取，进一步控制长序列计算 | 同时改posttraining，不能把质量/吞吐全归indexer | https://arxiv.org/abs/2512.02556 |
| DeepSeek-V4 preview | 官宣2026-04-24 | tokenwisecompression+DSA、1Mcontext；Flash284B/13B,Pro1.6T/49B | 不再只是V2latentfeature compression。官宣SOTA是作者claim | https://www.deepseek.com/en/news/v4-preview/ |
| DeepSeek-V4.1-Flash | 官宣2026-09-10 | CED20encoder20decoder,CSA2cross-layer reuse,FP4,SWAboundedreplay,vision | 552Bbackbone+196BEngram；8Bprefill16Bdecode；global890bytes/token，不包含allSWA/transient/allocator | https://www.deepseek.com/en/news/deepseek-v4-1-flash/ |
| FlashMLA current | 2026-09-30 breaking change, inspected2026-10-29 | fusedQnorm/RoPE/attn/OconjugateRoPE/cast；SM100/103 &Ascend950 | 主分支移除Hopper及V3/V3.2/V4.0，不能当V2兼容方案 | https://github.com/deepseek-ai/FlashMLA/tree/395cb3afbb331fef7b1b21e9340861b9436b42d8 |
| FlashMLA older supported tree | pinned historical commit | denseMQAmodeK576,V512；MHA-modeK192,V128;H800densepeak660TFLOPs/3000GBs | 峰值算子数，不是整模型tokens/s；V2未单列supportedmodel需实测 | https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7 |
| vLLM current | inspected2026-10-29 | DeepseekV2MLAAttention+MultiHeadLatentAttentionWrapper,use_mla,runtimecompressedcache | backend硬件支持另核，HFcheckpoint模型声明不代表cache实际布局 | https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_v2.py |

## 当前前沿，按测量问题区分

官方V4.1报告已从HF直接下载并读过概要、架构和Table3。报告链接：
https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf

官方报告Table3有作者实测同表对比：V4.1Flash GPQADiamondPass@1 90.9，GPT5.6Max94.1；TerminalBench2.1 90.6，Opus5Max89.1；TerminalBench3.0 30.0，Opus43.3；DeepSWEv1.1Resolved74.2，Opus74.0。不同benchmark冠军不同，且V2主要MMLU/HumanEval等2024数据不能直接拼榜。Table3给Max reasoning配置、温度1 top-p.95、官方scaffolds，未本地复现，不把小差异写成显著优势。HLEmultimodal/text-only带†，不能混。

V4.1的890byte/token是全局KV在HBM的口径，并非完整运行时cache总数；与V269120byteBF16比较跨精度/压缩token/跨层共享/架构，不能宣称同质量直接77.7×改善。官方摘要还区别persistentSSD/hostcache占用、SWA bounded replay。保留这些口径有助于端侧/CPUdma-buf分析；V2仅文本，无OCR或visionencoder。

## 冲突与复现缺口

V2 README里Mixtral C-Eval58.6，而v5Table2是59.6；本篇以指定v5为准。README部署旧vLLM PR4650不能当2026安装流程。FlashMLA历史与currentABI/cacheformat/hardware约束不同，以固定SHA引用。

没有核验“全行业统一SOTA”、全模型本地吞吐、延迟、量化质量或V2兼容FlashMLA新main的实测。给读者可核验的技术演进与条件，不填造缺失数据。
