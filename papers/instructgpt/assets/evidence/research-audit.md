# InstructGPT v1 — 原文核对与研究底稿

指定版本：[2203.02155v1](https://arxiv.org/pdf/2203.02155v1)，2022-03-04，68 页，OpenAI。后续发表 NeurIPS 2022；v1 叙述与后续成果须区分。完整结构：Abstract、§1 Introduction、§2 Related work、§3 Methods and experimental details（3.1–3.6）、§4 Results（4.1–4.3）、§5 Discussion（5.1–5.5）、Acknowledgements、References、Appendix A–F。

资产清单：50 张编号 Figure、14 张编号 Table、2 个编号公式。编号来自指定 PDF 与 v1 TeX/HTML 的交叉核对。Table 4/5 是无独立 caption 的跨页 prompt longtable；网页标题应明确是编辑补充的识别标签，不能伪装为原图注。Figure 12 是两张原标注界面截图。Figure 14–27 是评测设置文本框；Figure 42–50 是文本样例。图注在图片外，所有原图还保留复制文本入口。

## 已确认的重点

- 一句话：将 GPT-3 的续写能力，经示范、偏好排序和带 KL 约束的在线优化，转换为更符合特定标注者所理解的用户意图的输出。
- 主图横轴是模型大小，纵轴是对 **175B SFT** 的 win rate；不是对 GPT-3 175B 的直接胜率。
- 175B InstructGPT 对 GPT-3 的直接偏好率 **85±3%**，对 few-shot GPT-3 **71±4%**；原图全部误差条是 95% CI。不能把主图约 0.7–0.8 的数读成这两个直接比较。
- 175/1.3=134.615…；“100× fewer”是数量级表达；1.3B 的胜出限于作者 API prompt 与人类偏好，不是整体知识、推理、真实性或所有 benchmark 都胜出。
- InstructGPT 默认指 **PPO-ptx**。
- SFT 精确 train=11,295+1,430=12,725，valid=1,550+103=1,653；RM train=6,623+26,584=33,207，valid=3,488+14,399=17,887；PPO train=31,144，valid=16,185；测试集 Table 9 有 3,196 prompts。
- API 数据只来自 Playground，非生产客户数据。正文按 user ID 切分、每 user 200；附录按 organization ID、约每 org 200。网页应保留两种口径并注明粒度未完全统一，不能宣称严格双重隔离。
- 数据 langid 检测约 96% 英语；作者估计实际可能 ≥99%。约 40 位承包标注者；人口与满意度调查仅 19 自愿回应者。
- 训练标注优先 helpfulness；最终评估优先 truthfulness/harmlessness，存在目标口径变化。训练组一致率 72.6±1.5%，held-out 77.3±1.3%，不是 RM classification accuracy。
- 每 prompt K=4–9 回答，最多 K(K−1)/2=6–36 对。RM 按 prompt 分组、每个 completion 仅算一次，不把 K−1 次相关比较拆成多次 optimizer updates；ties 丢弃。
- 正文 RM 描述从 SFT 改 head；附录 C.2 最终 RM 从经 ARC/BoolQ/CoQA/DROP/MultiNLI/OpenBookQA/QuAC/RACE/Winogrande 任务微调的 GPT-3 6B 初始化。作者说这是历史原因，GPT-3/SFT 初始化类似。必须写清实际设置。
- 奖励仅差值可识别，增加同一 bias 不改变 pairwise loss；RL 前让示范平均 reward=0。这不是把每个输出变成绝对效用或可靠置信度。
- SFT 展示基线 16 epochs，残差 dropout .2；1.3B/6B LR 9.65e-6 batch32，175B LR5.03e-6 batch8；cosine降至10%，无warmup。validation NLL 1 epoch后过拟合，但 RM/human preference 继续改善；最终用 RM validation选模型。
- PPO 初始化用 2 epochs、10% ptx；初始化 SFT LR分别5e-6/1.04e-5/2.45e-6，**不是最终 PPO actor LR**。E.9 PPO小模型扫描2.55e-6–2.55e-5，175B仅2.55e-6/3.74e-6，按human Likert选checkpoint，未明确汇总最终逐variant值。因此不应把16-epoch SFT baseline当作唯一PPO初始参考。
- 全部 policy：GPT-3 decoder架构，2k上下文，模型训练 prompt过滤>1k、response最多1k；Table9 收集数据 prompt最大可到2048，不与训练过滤阶段混为一谈。
- 权重/activation FP16、FP32 master；Adam betas .9/.95。RM固定6B、critic固定6B（从RM初始化），actor1.3B/6B/175B；reference为冻结PPO初始化模型。
- RM1epoch LR9e-6 batch64 **prompts**，up to2304pairs；cosine降到10%。
- PPO256k episodes，batch512、minibatch64、1inner epoch，500outer iterations、4000minibatch updates（按“256k”=256000重算）；temperature1，clip .2，KL beta .02；首10iterations LRwarmup，EMA .992，无discount；critic LR9e-6小模型、5e-6大模型。
- PPO-ptx每RL episode配8个ptx examples，gamma27.8；两路gradient依次算后累积，不是27.8倍tokens或27.8%混合。E.11指出ptx ratio8训练时间约翻倍；32更慢且仅初步结果；minibatch32稍好但最终64为GPU utilization。
- 原式(1) expectation外又放1/binomial，规范实现应按每prompt对的sum/count；必须保留原式并解释expectation归约歧义，避免double averaging。
- 原式(2) log概率比为sequence sum，单条sample可以负，但对当前policy期望=forward KL≥0。每token penalty，terminal RM；KL针对reference，PPO importance ratio针对old rollout policy，两个不同对象。
- §5.1算力4.9 PF/s-days SFT、60 PPO-ptx、3640 GPT-3；4.9/3640=.1346%，60/3640=1.6484%，合计1.7830%。不包含可完整复现的总项目成本/硬件/端侧TTFT。
- Figure7 human eval1729prompts；Table14自动toxicity5000 prompts，按输入toxicity近似均匀采样，不能和标准随机采样直接比均值。
- Table14 175B basic toxicity GPT .231 / PPO-ptx .234（差+.003）；respectful .233/.196（差−.037，−15.8798%）；biased .285/.400（差+.115，+40.3509%）。默认“更安全”的总结不成立。
- Table14自动 TruthfulQA QA prompt true+info GPT .251 / PPO-ptx .689，差43.8pp；Figure6是human评估、不能混为同一测量。1.3B human TQA例外（PPO-ptx略差同规模GPT）；拒答提升true但不保证info。
- Table14 175B DROP zero-shot GPT27.53 PPO13.08 PPO-ptx15.23，ptx恢复2.15 F1，仍低12.30；SQuADv2 zero-shot64.30/43.68/59.85；few-shot69.75/51.95/69.93。alignment tax缓解有任务/shot依赖，并未完全消失。
- entropy: H=−sum p log2p，binary最大1；高entropy不证明群体公平，它可以是单纯不确定。偏见结果未显著改善。
- Figure33 gamma1.3B ablation≥20可恢复若干任务；Figure34 **改用pretrained GPT reference**、gamma0、beta到2依然不能恢复；不是默认SFT reference下只换beta的完美控制比较。
- Figure35 512k episodes/3seeds，后期DROP/SQuAD略回退；Figure36 beta0及2差，.01–.02较好；Figure37 initialization10%ptx较好；Figure38无ptx LR>8.05e-6发散，小模型scan2.55e-6–2.55e-5、175B只2个LR，不能宣称全局最优。
- E.2原文按“inter- and intra-group”顺序写 **72.4±.4%, 69.6±.9%**，不要自行对调；它只测同一承包商总体中held-out group泛化，不是代表性人口验证。
- Appendix F：InstructGPT sampling T1 vs GPT T.7；prompts cherry-picked，outputs不cherry-pick，5选15展示任务范围；不是无偏量化多语言/代码评估。

## 原文异常，保留并解释

- D称multiple choice选“lowest average per-token log probability”，与最大likelihood惯例相反；可能漏negative，但没有证据就不能偷偷改。图模板另写logprob归一。
- Figure26/27摘要模板写“15 additional French/English pairs”，疑似复制WMT内容，不把它当作摘要few-shot真实配置。
- Table14 CNN/DM与TLDR 12个结果逐格相同，可能复制问题，不能当作两组独立相互印证的证据；原表照录。
- Table14 XL指1.3B；QuAC Figure19“7.306 examples”为排版，应读7306，但保留原字样的资产。
- TruthfulQA Table14第三prompt名“QA+instruct”与caption/正文“Instruction+QA”排列差异需要解释，不擅自合并行。

## 扩展研究，一手资料截至2026-10-29

- 2017 Deep reinforcement learning from human preferences：https://arxiv.org/abs/1706.03741；2017 PPO：https://arxiv.org/abs/1707.06347；2020 Learning to summarize from human feedback：https://arxiv.org/abs/2009.01325。已读取官方元信息/摘要。
- 2022-12 Constitutional AI：https://arxiv.org/abs/2212.08073；AI critique/revision + RLAIF，改变监督来源，原则仍人为指定。
- 2023-05 DPO：https://arxiv.org/abs/2305.18290；BT+KL假设下closed-form policy重参数化，离线pairwise loss免显式RM与rollout/critic，不能保证静态数据覆盖新policy。
- 2024-02 DeepSeekMath：https://arxiv.org/abs/2402.03300；GRPO用group relative advantages省critic；数学验证奖励与开放任务偏好仍有差异。
- 2025-06/2026-04 v2 ICLR2026 RewardBench2：https://arxiv.org/abs/2506.01937，完整HTML /private/tmp/instructgpt-work/rewardbench2.html。1865prompts，Factuality475、PreciseIF160、Math183、Safety450、Focus495、Ties102；best-of4+ties，不同于原论文pairwise胜率。
- RB2作者2026v2 Table3公开列表领先Skywork-Reward-V2-Llama-3.1-8B平均84.1，Factuality84.6、IF66.3、Math77.6、Safety96.7、Focus98.4、Ties81.2；AI2 Llama3.1-70B RM-RB2平均76.1；8B版本72.8。有open weights。只称**该官方快照内领先**，不能宣称覆盖2026-10-29所有提交的绝对第一。
- 官方实时入口 https://huggingface.co/spaces/allenai/reward-bench 本次可访问，但静态fetch无榜单内容；结果dataset inventory188files且合计约3GB，未全下载，因此不能假装实时排位全部核验。公开数据链接：https://huggingface.co/datasets/allenai/reward-bench-2-results。
- 当前源码SHA reward-bench main：59f6ff7005a9f9484727ca27f5383420875e5dbf。README确认run_v2.py、bestof4、Ties20+completion rating方式；原benchmark/新benchmark distinct。
- 2026 General Preference Reinforcement Learning：https://arxiv.org/html/2605.18721 （本次fetch v3）；多维/非传递偏好embedding + per-dimension advantage + drift monitor，Llama3-8B-Instruct AlpacaEval2 length-controlled56.51，GRPO+BT41.92，差14.59pp。论文自报、单来源，非复现、非2022指标。
- 原官方repo https://github.com/openai/following-instructions-human-feedback；SHA5c0534c78136809ab0118c5c1b74d159befa2467。仅modelcard、automatic benchmark samples、label instructions，没有训练程序/原偏好data/weights。modelcard January2022明确API实际版本微调procedure稍有不同，论文模型与线上模型不能完全等同。

## 后续产物约束

解释主体按原顺序；工程重建、扩展比较与“我的洞察”置于全文/附录之后。3交互建议：K个ranking共享前向/归约；sequence KL累计（数据仅数学假设、不是实验）；PPO rollout→reference/RM/critic→advantages→update逐步状态。原图/表不可缺；文字图完整copyable转录便于移动阅读。

网页架构已核验：纯静态HTML，站点clone在 `/private/tmp/instructgpt-site`；catalog `papers/readings.json`，run `python3 scripts/build_papers.py`静态导航重建；部署main/root，gh Pages pipeline。本篇category用既有training-scaling。发布需新增article+catalog+生成导航+GPT相关backlink，不改架构。用户已明确授权commit/push/publication。

最终产物 `/Users/bytedance/project/read_papers/paper-readings/instructgpt/`；脚本/临时QA screenshots只在 `/private/tmp/instructgpt-work`；无需GPU复现。双审校和五种宽度375/390/430/1440/1920、dark/light、math、原图/表、interactions、modal/keyboard、internal/external paths、online Actions/Pages/index必验。
