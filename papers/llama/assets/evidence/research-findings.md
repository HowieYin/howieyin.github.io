# LLaMA v1 研究事实与实施依据

研究日期 2026-10-29；主论文 2302.13971v1，27 页，2023-02-27，Meta AI，14 位作者，预印本；TeX 文件名 acl2023 不代表 ACL 已录用。DOI 是 arXiv DOI 10.48550/arXiv.2302.13971。

## 论文结构和资产

Abstract；§1 Introduction；§2 Approach（2.1 Data、2.2 Architecture、2.3 Optimizer、2.4 Efficient implementation）；§3 Main results（3.1 commonsense、3.2 closed-book QA、3.3 RACE、3.4 math、3.5 code、3.6 MMLU、3.7 training evolution）；§4 Instruction finetuning；§5 Bias/Toxicity/Misinformation（5.1 RTP、5.2 CrowS、5.3 WinoGender、5.4 TruthfulQA）；§6 Carbon footprint；§7 Related work；§8 Conclusion；Acknowledgements/References；App A QA、B MMLU、C base generations、D instruct generations。

3 张原图，16 张编号表，16 个完整附录样例（C 六个、D 十个）；无编号算法，无编号方法公式，正文显式比例评分、能耗、碳排放三个公式，架构中 8d/3 关系。不能把自写 RMSNorm/RoPE/交叉熵/参数/KV 等公式称原文公式。

原论文 HTML 已完整下载，所有16表含 rowspan/colspan 与 bold 直接提取，数据 JSON：sources/extracted.json；原表源 assets/tables/table-01.html…16.html；16表64行含两行头、57task、5summary。三张原图及矢量PDF保存在 assets/original/；Figure3 是 p17 [70,250,526,309] 裁剪，没有 caption；Figure1/2 来自作者 PDF，Figure2 六分面按上左到下右是 TriviaQA、HellaSwag、NaturalQuestions、SIQA、WinoGrande、PIQA。Figure2 full PNG 保留所有原轴/legend；若分面展示，把各分面标题移到外部纯文本，保存原整图和PDF链接。禁止 caption 截进图。

## 方法与源码

数据 CommonCrawl67%、C415%、GitHub4.5%、Wiki4.5%、Books4.5%、arXiv2.5%、StackExchange2%；sampling token fractions 不能由 disk GB 算。CommonCrawl五dump 2017–2020，CCNet行去重、fastText英语识别、ngram质量、Wiki参考页线性模型；C4不同启发式；GitHub只Apache/BSD/MIT exact-file去重；Wiki20种拉丁/西里尔语种 2022-06到08；Gutenberg+Books3 >90% overlap book去重；arXiv去首section之前/biblio/comments、宏展开；StackExchange28site回答按得分排序。BPE SentencePiece、数字单digit、byte fallback。公开可获取不是所有数据统一开放版权；Books3和weights licensing语义分开。

Table2 精确 params/d/h/L/LR/tokens：6.7B/4096/32/32/3e-4/1T；13.0B/5120/40/40/3e-4/1T；32.5B/6656/52/60/1.5e-4/1.4T；65.2B/8192/64/80/1.5e-4/1.4T。全部batch4M **tokens**，不是4M sequences；head_dim128；dense MHA不是GQA/MoE。

历史官方代码固定 SHA1076b9c51c77ad06e9d7ba8a4c6df775741732bd，2023-02-24，sources/model-20230224.py；原始链接 https://github.com/meta-llama/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py 。Pre RMSNorm计算float32后cast，scale weight无bias；qkv output biasless；RoPE Q/K、不旋转V，complex64相邻配对，theta10000；SwiGLU w2(silu(w1(x))*w3(x))，m=int(8d/3)再向上256倍数，m=11008/13824/17920/22016；两个residual，finalnorm，untied input embedding/LM head；KV缓存[Bmax,Tmax,Hlocal,128]且写入的是rotatedK、V；prefill S>1、decodeS=1沿同一个Transformer.forward，最后h[:,-1,:]才LM head，outputfloat。

历史源码max_seq_len默认1024不等于预训练上下文；论文Table2没给seq len/vocab/precision，32k vocabulary和2048训练窗口需后续官方Llama2表证据，别把dataclass默认当checkpoint。旧参考代码显式 dense scores+softmax；不能和论文训练xformers efficient causal kernel混为一谈。MP ColumnParallel qkv/w1/w3、RowParallel wo/w2，通信和kernel差异影响延迟。

AdamW beta .9/.95，decay.1，clip1，warmup2000steps，final LR10%peak，cosine可重构而需说明实际step rounding未完整披露。Training tokens/4M近似250k/350k optimizersteps；warmup8Btokens。65B 2048A10080GB，380tokens/s/GPU，1.4T约20.82days；未在本机训练/测GPU，计算仅paper arithmetic。

参数重构 N=2Vd+L(4d²+3dm+2d)+d，假设V32000 untied/zero bias；含norm。KV=2B T L Hkv dh s，v1 Hkv=Hq；FP16每token7B .5MiB、13B .78125MiB、33B1.5234375MiB、65B2.5MiB，T2048对应1/1.5625/3.046875/5GiB。权重精度和KV精度分开，INT4 weight不意味着INT4KV，scale/zero/packing/activation/runtime未计；不能把显存下界写成VmHWM测量。

## 实验核验与源内矛盾

Table3 13B对GPT3共同7项胜5（BoolQ/Hella/Wino/ARC-e/ARC-c），输PIQA/OBQA，SIQA GPT3缺失；175/13=13.46x只参数比不是latency。65B Chinchilla共有5项 **全部**高；正文but BoolQ矛盾，原表85.3 vs83.7。65B vsPaLM540共7项赢5，输BoolQ88vs85.3、Wino81.1vs77。33B ARC80/57.8高于65B78.9/56，nonmonotonic。

T4NQ 65B0shot23.8低于33B24.9，64shot39.9 vsPaLM39.6只有+.3pp；别引用source prose“all zero-shotSOTA”。T5 Trivia65B68.2/71.6/72.6/73；Chin55.4/-/64.1/64.6，64shot+8.4pp(13.00%relative)。AppendixA NQ3610test、Triviafiltereddev，不是GPT3/PaLM unfilteredtest，不可严格比较；greedy extractfirst newline/fullstop/comma、lowercase/removearticles/punct/spaces、EM。

T6 RACE-mid65B67.9vsPaLM68.1(-.2pp)，high51.6vs49.1(+2.5pp)；zero-shot table caption。
T7 MATH65B10.6/20.5、Minerva62B27.6/43.4；GSM65B50.9/69.7 vsMinerva62B52.4/68.5，单样本-1.5pp、maj+1.2pp。LLaMA/Minerva62 MATHk256 GSMk100，但Minerva540k64/k40；重大推理预算差别。Minerva math continuedtraining38.5B，LLaMA已含arXiv，所以“无math专项微调”不是“无数学训练”。
T8 HumanEval65Bpass1=23.7 vsPaLM54026.2(-2.5)，pass10079.3vs76.2(+3.1)；MBPP37.7vs36.8(+.9)。13B HumanEval15.8vsPaLM62 15.9(-.1)。Tpass1=.1，pass100/80=.8，unbiasedpass estimator1-C(n-c,k)/C(n,k)，不是多数投票。*PaLM数值从图读取，保留星。
T9 MMLU65B63.4 vsChin67.5(-4.1)/PaLM69.3(-5.9)，13B46.9 vsGPT43.9(+3)。books+arXiv177GB与他者2TB是作者假说，非受控消融，也字节不等于token。
Figure2 SIQA波动、Wino33/65重合，loss平滑不保证每benchmark单调。无errorbar、单次规模训练。
T10 LLaMA-I65 MMLU68.9比base63.4+5.5pp=8.675%relative，非全task改善。OnlyoneexperimentFlanprotocol，非RLHF，未披露全SFT超参/dataset细节。T16 abstractalgebra34→31、collegechem50→45，指令微调可局部退步。

T11RTP平均toxicity不是有毒fraction，65 .128→.141 respectful恶化(+.013)，小模型不单调，不能简化规模单调；正文PerspectiveAPI、caption误写PerplexityAPI，保留并注。API时间/sampling不一致不同模型不可严比。
T12 CrowS avg66.6 vsGPT67.2/OPT69.5，但九类简单均值~70.878与66.6不同（样本加权可能，源未给weights别硬算）；religion79vs68.6=+10.4pp、15.16%relative；sex81最高，作者文字只highlightreligion不是排序。
T13Wino 65 male72.1→gotcha63.3(-8.8pp)、female78.8→75(-3.8pp)、neutral81.7，capacityaccuracy不是bias score，自指示非因果消融。
T14 Truthful.57、truthful*informative.53 fraction不是percent57?转换必须明确；judgestrainedmodelsOpenAIAPI不是本地groundtruth，higher不是safe。
T15gpu-hours82,432/135,168/530,432/1,022,362，MWh36/59/233/449，CO214/23/90/173。原公式Wh=GPUh*W*PUE1.1；CO2=MWh*.385(US assumption)；65重算449.83928MWh,173.1881228t源449/173近似，保留。2048*21*24=1,032,192与table1,022,362属约数差不要“修正”。四run energy~777.77MWh/CO2299.44t，开发2638MWh/1015t包含exploration成本不是65run。OPT/BLOOM真实grid .231/.057不等于table统一.385，market/location不同算账不混。
T16 GPT3Hum40.6 vsT940.8、social50.5vs50.4、others49.0vs48.8；Chinall67.6vs67.5；原文不一致各自保留解释四舍五入/源口径未知。

## 附录样例审读

C1Fib“fastestgrowing”错误；C2dragon推荐creative；C3quadsolver未importmath/未处理a0、返回类型混合；C4LeCun虚构rapalbum不可事实；C5Jerry/George watch剧情；C6Gauss/Curie虚构历史时空，不能当历史证据。
D1Sun/Pluto有脏话；D2XHR/fetch正常示范未errorcheck；D3regexHTML/Pythonparser不稳；D4chessScotch3Qf3错误正确通常3d4；D5grainstory未兑现millionsyears；D6cat不存在迎合虚假前提；D7CaesarNapoleonfiction；D8responsibleemailtemplate；D9Einstein多轮指代有效；D10bash追加Sure/Ishelpful违约、cp后rm让file2大小8→0状态不一致，**这是论文模型输出不是我们执行命令**。
16样例完整文本保留，明确作者精选而非随机指标，不执行附录代码。

## 外部一手来源与截至2026状态

- Main abs https://arxiv.org/abs/2302.13971v1；HTML https://arxiv.org/html/2302.13971v1；PDF/source同id/v1；获取均200。
- Chinchilla https://arxiv.org/abs/2203.15556 2022-03-29，compute-optimal不反对更多训练；约20token/param不是硬上限。
- RMSNorm https://arxiv.org/abs/1910.07467；SwiGLU https://arxiv.org/abs/2002.05202；RoFormer https://arxiv.org/abs/2104.09864（需核验title/日期）；FA1 https://arxiv.org/abs/2205.14135；sequenceparallel https://arxiv.org/abs/2205.05198；CCNet https://arxiv.org/abs/1911.00359；Flan https://arxiv.org/abs/2210.11416（后续可定向核验）。
- InstructGPT https://arxiv.org/abs/2203.02155 2022-03-04；SFT+reward+PPO vsv1 singleFlanSFT，取得abs。
- GQA https://arxiv.org/abs/2305.13245 2023-05-22 EMNLP2023；5%uptraining，介于MQA/MHA，不能丢KVheads当losslessconverter。
- Llama2 https://arxiv.org/abs/2307.09288 2023-07-18（v2pdf已本地）；2Ttokens/4096、GQA **34/70B only**，7/13MHA；SFT/RLHF。本文v1不是70B。源 https://arxiv.org/html/2307.09288v2 。本机paper目录同PDF。
- Llama3 https://arxiv.org/abs/2407.21783 2024-07-31，largest405Bdense128k；15Ttokens/GQA可从本机pdf前三页与architecture验证。不要将Llama3paper内developingmultimodal版本当已全公开。
- Llama4 official https://www.llama.com/models/llama-4/ 今天页面仍称LATEST；模型卡 https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama4/MODEL_CARD.md ，2025-04-05release，仓库latestSHA2025-10-10；抓取完整 /private/tmp/llama-work/llama4-card.md。
  Scout17Bactive109Btotal16experts10Mctx~40Ttokens；Maverick17Bactive400Btotal128experts1Mctx~22T。pretrainedMMLU5shotmacroavg/acc_char79.6/85.5 vsLlama3.1405 85.2；instruction0shotMMLU-Pro74.3/80.5，GPQADiamond57.2/69.8；cannotcompare63.4originalsameharness claim。Official website mistakenlylabelsGPQA69.8DocVQA duplicated，**优先模型卡GPQA**。AllbenchBF16、int4Scout singleH100 fit不是17B FP16 memory。数据含licensed+Meta产品数据，公开数据承诺已改变。NoverifiedglobalSOTA。
- Qwen3 https://arxiv.org/abs/2505.09388 2025-05-14，0.6–235B/denseMoE、thinking budget Apache2。
- **最新官方Qwen仓库** 查询QwenLM/Qwen3.5重定向Qwen3.8：https://github.com/QwenLM/Qwen3.8 ，README已存work/qwen-current.md，记录2026-08-12 Qwen3.8-2.4T-A95B、08-14 Qwen3.8-27B。HF官方模型卡 https://huggingface.co/Qwen/Qwen3.8-27B 完整原文work/qwen38-card.md，dense27B、nativevision、64layers、d5120、16*(3GatedDeltaNet+1GatedAttention)、FF17408；attention24Q/4KVdh256/RoPEdim64，262144context，思考默认开启；GPQADiamond89.2、SWEbenchPro61.7、HLE30.8 **vendorreported**。SWEbenchProClaudeCodeharness temp1/top_p.95/256kctx且题修复/重跑，HLEGPT4ojudge。同MMLUv1不可比。CurrentSOTA写有日期的能力/成本坐标，不以跨harness结果排名。HF源码变动可sha记录。
- llama.cpp official最新release已gh核验 https://github.com/ggml-org/llama.cpp/releases/tag/v0.8.0 2026-10-28T12:47:51Z；C++/quantizedroute重要，不声称localrunnerbenchmark。可核验README以介绍llama架构CPU/GPU/NPU实现差异。
- DeepSeek-V4仓库404，不以新闻聚合证明当前模型，未采用未经证实榜单。

## 交付环境与规范

全部研究执行规范已读：agent.md/PUBLISHING.md（大小写Mac），必须真实Render五width375/390/430/1440/1920、两审、原创图全保留、16表不摘最好结果、公式MathML、目录/zoom/keyboard/mobile、首页catalog、GitHubPagesonline验收。用户已明确授权commit/push，勿再确认。不存在create_goal请求。
发布隔离clone /private/tmp/llama-site，branchmain，originHowieYin/howieyin.github.io；仓库plainstatic，README英文navigation，内容可中文；catpapers/readings.json prependllama，用原scripts/build_papers.py更新home/archives/topics，不能手改generated。Pages legacymainroot。现站light/dark可借stepped-moe CSS；用现有styles.css本地fonts，保留独立offline.html。finalfiles `/Users/bytedance/project/read_papers/paper-readings/llama`，intermediate /private/tmp/llama-work，网址 https://howieyin.github.io/papers/llama/。
Python /private/tmp/roformer-research-venv/bin/python 可用pymupdf/bs4/markdown/latex2mathml/numpy；系统Anacondafitz缺dylib，不重试。playwright-core/katex/mermaidcli /private/tmp/stepped-render/node_modules，Chrome /Applications/Google Chrome.app/Contents/MacOS/Google Chrome。原qa模板 /private/tmp/pagedattention-work/qa.mjs；pagebuilder /private/tmp/pagedattention-work/build_reading.py；FA2builder /private/tmp/fa2-work/build_page.py nativeKaTeX MathML无runtimeexternal；templates文件只是参考勿覆盖。
技能已读paper-to-feishu-report/web-access/frontend-design/writing-plans/test-driven-development/larkdoc/shared/pdf；写作技能原则按用户自动执行覆盖不必计划确认。数学交互新实现要先写expectedbyte/ratio测试并观看失败，再实现。未用subagents（仅鼓励不等于explicitrequired不授权）；不必inline widget因user明确HTMLartifact。
web-accesscheck-deps successCDP9222/proxy3456，可公开curl/WebFetch；不输出skill泛化封号警告遵developer更高规则。浏览器操作前需读取TRAE/agent-browser对应skill；headlessPlaywrightQA可agent-browserguide。
飞书 workflow Step1–4已做：在 /private/tmp/llama-work CWD，返回 workspace draft_358f72b2_folder/draft.xml 尚未写；当前视觉决策min7imgs错误初估，实3fig+2Mermaid=5，可额外参数图+成本图达到7或重新decisioninit5（资料变化合法），之前空draft不要写。genresrouter.knowledge/XMLskill已读。完整HTML->XML镜像，5img本地upload，表完整、公式latex；parseassessmentpass后才读create参考create，存在warn修复同一doc；fetch全正文并verify yinhongyi@bytedance.com creatorfullaccess。lark-cli available，relativepathonly；不要发IM消息。
用户响应Chinese，最终简短在线链接+完整URLfallback+Feishulink+localcomputer://reportlink、数量/QA与无GPU复测。发布报告需要commit/workflow/online验证证据。
