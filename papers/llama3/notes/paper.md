# Llama 3 v3 — 原文阅读与证据记录

指定输入：`paper/2407.21783v3.pdf`，92 页。标题 The Llama 3 Herd of Models；署名 Llama Team, AI @ Meta，完整 contributors 在 pp.72–74。论文内日期 2024-07-23；arXiv v1 2024-07-31，v3 2024-11-23。DOI 10.48550/arXiv.2407.21783。原文所有主要语言模型结果对应 Llama 3.1，不能混同 April 2024 Llama 3。实际 PDF 正文 §1–10，30 张 Figure、35 张 Table；随后是 contributors 和 References。TeX 包里有 appendices/*.tex，但 paper.tex **没有 input appendix.tex**，故不能声称 PDF 有这些附录实验；如另行讨论，应标为“公开 TeX 中未编入指定 PDF 的材料”。

## Abstract / §1–2

关键杠杆是数据、规模、控制复杂度。405B 是 dense Transformer，15.6T tokens，3.8e25 FLOPs，128K context；不是 MoE。比 Llama 2 70B 约 50× 预训练算力。8B/70B 过度训练以换取推理效率，405B 接近团队数据与预算下的 compute optimum。模型正文以开权重语言模型为主，视觉/视频/语音是 compositional research，论文发表时未广泛发布。

Figure 1 是 language pretrain/posttrain 与 modalities 全链路；Figure 28 是多模态五阶段。视觉与语音保持语言主干 frozen，避免语言退化；不意味着 adapter 无成本。视觉走 cross-attention；语音走 token embedding 拼接。

## §3.1 数据

知识截至 2023 年末；URL 去重保留最近页面，MinHash 全局近重复，30M-doc 桶内出现超过 6 次的行去重。自定义 HTML parser 保留 code/math 结构与 image alt 数学；作者对“以 web 为主”的清洗实验中 Markdown 标记有害，不可外推成所有语料都应去 Markdown。

PII/adult domain filtering、重复 n-gram、dirty words、token KL 异常；fastText 与 Llama2 标注→DistilRoBERTa classifiers 做质量/code/math。176 language LID，语言内去重与定制过滤。最终约 50% general knowledge、25% math/reasoning、17% code、8% multilingual；训练中动态增加多语言、数学、近期数据，减少低质来源。

高质 annealing 评估：半程 8B checkpoint 在 40B tokens 上 linear LR→0，30%新数据 +70%基线。探索将 GSM8K/MATH train 用于 annealing 可使8B validation提升24.0%/6.4%，405B微小；**最终 annealing 不含常用 benchmark train**，不要混成污染证据。

## §3.2 架构与 scaling

Table3：8B=(L32,d4096,f14336,h32,hKV8,LR3e-4)；70B=(80,8192,28672,64,8,1.5e-4)；405B=(126,16384,53248,128,8,8e-5)。dHead=128，全规模 hKV=8，SwiGLU，RoPE theta500000，论文 vocabulary 128000（发布 checkpoint 常为128256，要另外查config）。

document mask 阻止同一packed sequence不同文档互看；短context小影响，长context重要。Tokenizer100K tiktoken+28K non-English；英文字/token3.17→3.94。

Figure2 IsoFLOPs：6e18–1e22算力、40M–16B模型；二次曲线找min，越大预算min附近越flat。原公式 N*(C)=A C^alpha，**N是training tokens而非parameter count**；(alpha,A)=(.53,.29)，作者称外推3.8e25为16.55T tokens、402B；实际选405B。终审核算：直接代入公开系数得到10.46365T，不是16.55T。完整系数/单位归一化未披露，不替作者修正，页面明确区分报告预测和直接计算。Figure3 token-vs-compute，Figure4 FLOPs→downstream normalized NLL→sigmoid accuracy，两阶段避免直接拟合跳变accuracy。单ARC示例跨约四个数量级外推不能保证所有任务。

## §3.3 系统

至多16384 H100 80GB/700W，8GPU服务器NVLink；405B用RoCE，小模型IB，400Gbps；24K RoCE cluster仅最多16K用于这次训练，3,072GPU pod full bisection，跨pod 1:7 oversubscription；每pair16 flows + E-ECMP。Tectonic240PB/7500servers，持续2TB/s、峰值7TB/s，checkpoint1MB–4GB/GPU突发；非平均训练数据读吞吐。

4D顺序 [TP,CP,PP,DP]；FSDP DP最外层，forward后不reshard model以避免backward额外allgather，gradient/moments sharded。Table4：

|GPU|TP|CP|PP|DP|S|batch/DP|tokens/batch|TFLOPs/GPU|MFU|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|8192|8|1|16|64|8192|32|16M|430|43%|
|16384|8|1|16|128|8192|16|16M|400|41%|
|16384|8|16|16|8|131072|16|16M|380|38%|

Figure5所有group各2的16GPU例，不是实际16K配置。CP切2CP块，rank i取i和2CP−1−i均衡causal三角。采用先allgather KV再算local Q，**不是ring attention**，因为GQA使KV较小且支持document mask简单；线性通信vs二次计算不意味着所有S/互联/实现都可忽略通信。

Figure6交错PP，4ranks负责8virtual stages。调节N连续microbatch数（例N5），解除传统DFS/BFS batch约束；移走首尾一层补embedding/output负载；bubble≈(PP−1)/(VM)。async P2P、主动释放input/output、TORCH_NCCL_AVOID_RECORD_STREAMS，8K可无activation checkpoint；不适用于128K全部场景。FP32 microbatch gradients/reduce-scatter与复用vision tensor gradients。

NCCLX fork优化chunk/control msg/copies/GPU cycles/highlatency网络，控制消息优先防HOL。flight recorder ringbuffer、watchdog快照识别finished/pending rank transfer；NVLink stalled kernel无清晰error，通过进程状态与超时定位；slow straggler可拖全job，不能只猜网络。

54day466 interruptions=47planned+419unexpected，manual3次，effective training time>90%（不同于MFU）。Table5列count总和419，但FaultyGPU148对应**35.32%**，原表30.1%；表百分比总和94.9%，保留并解释内部不一致；GPU类数268/419≈63.96%，与正文58.7%亦不同。日间温度DVFS引1–2%吞吐差；同步全群power变化数十MW。

## §3.4 recipe / 原文问题

AdamW LR8e-5，8Ksteps warmup，1.2Msteps cosine→8e-7。initial4Mtokens batch/S4096，原文称预训练**252M tokens**后8M/S8192，2.87T后16M batch；252M与8Kstep warmup(至少32B tokens)次序可疑，原样保留勿擅改252B。六阶段8K→128K，约800B tokens，关卡short恢复+NIH成功；后期才长训避免平方attention。

最终 anneal原文称**40M tokens**、S128K LR→0+Polyak average；与16M全局batch约2.5steps和另一处40B anneal quality experiment对照，应指出但不可替作者改数。warmup与cosine数字原样，缺optimizer betas/fullschedule中间长度列表，不编造。

## §4 后训练（Figure7）

六轮RM→RS→SFT→DPO→model averaging，新prompt与最新model synthetic刷新。RM remove margin，edited>chosen>rejected；多response shuffle/concat一row提升efficiency是pair评分的近似，有潜在上下文泄漏需细看mask。SFT prompt loss mask，CEtarget，LR1e-5、8.5K–9Ksteps。

DPO LR1e-5、beta.1，mask header/termination tokens，+.2 chosen NLL以稳格式与chosen logp。格式共同token冲突为**作者hypothesis**不是证明。正文没给DPO显式公式，重建标准loss要标为agent reconstruction+原DPO来源。

Chat header分role与destination；eom/eot需核对官方tokenprotocol，工具交接是runtime事件，不能把tool result当model prediction。

Table6偏好81.99%general、6.93code、5.19multi、5.89reason/tools（comparison计）；Table7 SFT52.66general、14.89code、3.01multi、8.14exam、21.19reason/tools、.11long（example计）；不能互相比tokens比例。RS K10–30选RMbest，PagedAttention共享prompt pages + admission reserve maxoutput减少swap，作者RS吞吐>2×不是online终端2×。

Quality RMtopquartile **OR** Llamafiltermaxscore，提高recall，非AND；difficulty Instag/Llama；RoBERTa cluster中按quality×difficulty排序，cosine greedy去重。filter导致难例被删是selection bias，修难例而不只是降低难度。

### §4.3 七能力

Code expert主pretrain分支续训1T>85%code、LCFT16K，专家用于annotations/RS，**不是MoE inference专家**。>2.7M synthetic，约1Mexecutiondialog+1.2Mbacktranslation+translation；parser/linter/containerunit tests反馈修复，约20%先错后修。模型可修改test有circular verification风险。8B/70B吃405B生成有益，405B自喂无truth可退化。Figure8Python→PHP、Figure9systemprompt改善style，但图例非因果统计。

Multilingual expert90%multi，SFTsources2.4%human/44.2NLP/18.8RS/34.6translatedreasoning；temp最后.6，先语言匹配后RM；支持8langs不含中文但pretrain176langs不等于官方支持176。Math outcome+stepRM/MCTS/filter/codeexecution/selfcorrection。长SFT .1%按examples合成，不是token .1%；8Kchunk生成QA/层次summary/repo依赖missingfile，避免shortSFT破坏long；shortDPO没见long退化是该setting结论。

Tools Brave/Python/Wolfram，message级pref，**tool数据不RS**（无gain）；single/multistep synthetic bootstrap，30%无效格式filter；function零shot single/nested/parallel/multiturn。Figure10多步tool、11fileupload说明runtime闭环。

Factuality probe pretrain snippets→questions→sample→contexttruthjudge+informativejudge→对持续错误生成refusal，学习know-what-it-knows；pretrain内容也可错，没保证truth。Steerability systemlength/format/tone/persona via allpoststages。

## §5 评测

Table8/16是benchmark inventory；9–14 pretrain，17–22posttrain。重算baseline选自测/原report中更好的可比数，但closedbase无logprob缺值，不聚合405Bcategory。95%CI=1.96sqrt(S(1−S)/N)，S比例/小样本与边界WaldCI问题，有限test variance下界不含prompt版本/seed；AP100%±0不能解释成能力无不确定性。

Figure12categorymean只8/70；Figures13–14 MMLU labels/fewshotbias/answerpermutation/promptformats（405稳）；Figure15adversarialmath/QA仍明显跌，PAWS少跌。

Table15 8gram覆盖threshold per dataset挑max significant gain；HellaSwag85%/405Bgain14.3，BBH95%/41.0，NaturalQuestions52%/.8；HumanEval/MBPP/MMLU无可靠estimate是**unknown不是clean**。gain非clean-counterfactual重训，method选阈值会selection bias。

Table2（post）405B MMLU87.3 vsGPT4o89.1/Sonnet89.9；IFEval88.6 vsSonnet88.0；GSM96.8；MATH73.8 vsGPT4o76.6；GPQA51.1 vsSonnet59.4；Nexus58.7。不能统称超GPT4o。Table18HumanEval89.0→HumanEval+82.3(-6.7pp)，GPT4o90.2/86.0，Sonnet92.0/82.3；tests弱覆盖改变ranking。Table19非Python明显更低尤其C#/Shell。

Table21 caption明确 Qasper F1、SQuALITY ROUGE-L、En.QA F1、QuALITYEM、En.MCaccuracy、multiNIHrecall。正文称prioritize accuracy、few places En.QA claim与实际数冲突：405B30.5<70B36.7，Table2 En.MC GPT4=72.1 vsTable21=72.0；保留两表标差异。长context CIs很宽，NIH≈100不是泛化成功。

Figure16tool human~2000prompts，code/plot wins、fileuploads输；Figure17~7000prompt六single/三multiturn,10/30/60easy/medium/hard,7pointpair, tiesexcluded,95CI, onlybetter/muchbetter considered win。对GPT4onpar，对Sonnetcode/reasoning落后，styleverbosity影响preference；不是blind objective truth证明。

### §5.4 Safety

VRunsafe violations vsFRRborderlinefalse refusal，percap/lang>4000prompts；不能当同一二类confusionmatrix。Figure18size×safetymix Pareto；19multi、20tool/long、21tradeoff。Comp1/2 API systems+oneopenmodel匿名内部，无法独立复现，LG前后应model/system分开。

Table24 verbatim50/1000gram benchmark不涵盖allmemorization，405BEnglish50=1.13/all50=1.03/all1000=3.91%。Figure22promptinjection（正文误指23），405Bsuccess21.7%；interpreterabuse10.4%；Figure23LLMjudgedphishpersuasiveness，不是humanattackoutcomes。cyber62volunteers31expert31novice, fixed two-stage differentchallenge无显著uplift不证明zero risk；biochem2person6hourcontrolinternetvsLLMagent组同样边界。Redteam多轮/语言/工具链组合，覆盖有限。

LlamaGuard3 8B13hazards+codeinterpreter，Table25English fullVR−86%、FRR+102%都是relative不是pp；Table26weapons0%change并不等于rate0。Table27quant int8 EnglishP.947 R.925 F1.936 FPR.040 vsnon.947/.931/.939/.040；multirecall.805→.785/F1.862→.851，tool.884→.865/F1.825→.827，需逐项拆不能只“negligible”。

PromptGuard86M mDeBERTa,Table28IDTPR99.9/99.5，indirectTPR71.4/FPR1.0/AUC.996；AUC好≠该threshold recall好。TP/FP/TN/FN/Accuracy样本数未给无法推绝对counts；可公式定义并给解析控件，勿捏造实验count。CodeShield静态7langs，非完备securityproof。

## §6 Inference

BF16 405B权重约810GB decimal>8×80GB单机，TP8nodeinner+PP2nodeouter16GPU；输入4096/输出256。Figure24two microbatches overlap pref/decode改善throughput，sync increases latency；“no training flush bubbles”不意味串行孤立请求没stageidle，分析持续饱和前提。

FFN约50%inferencecompute time；Attention **不量化**，first/last blocks **不量化**，rowwiseweight&activationdynscale，activation scale upper1200防大outliers导致underflow。需核验code scale convention，别误解1200为FP8max448/直接clipactivations。Figures25 granularity、26100Kresponsesrewarddistribution、27prefill最多+50% vsBF16two-machine &decodefrontier；不外推50%allrequest。reward proxy非pairedlogit/rareerrorquantitative guarantee。sourceFbgemmexperimental/gen_ai与llama-agentic-system公开，需ghcode核查。

## §7 vision

Image dataCLIPthreshold+512DSSCD/FAISS/connectedcomponentdedup+ngramresample probability sqrt(T/f)+OCR concatenation+renderdocs；faceblur/CSAM/NSFW过滤。anneal350Mresample+150Mgrounding/screenshot/QA/synthcaption/structured。videoavg21s/median16/>99%<1min，不过滤aesthetic/res。

ViT-H/14base630M,2.5Bpairs5epochs,224/14=16×16=256；多层4/8/16/24/31+final共6×1280=7680，新增8gatedblocks→40blocks850M；不freezeViT。每4LMselfattn插imagecrossattn(GQA)，405B多约100B参数，实际visiontotal≈505B+encoder，“405B”标语言backbone。4tiles336初训6Bpairs，高resanneal500M。tile336/14=24²576，4×576=2304patch（论文avg2308含extra unspecified不要确定填4tokens）。

Video16frames/pretrainagg16→1 effective; SFT64frames/agg32→2；max64sampling，Perceiver temporal。正文videoadapter0.6B/4.6B写“7B/70B”与family8B不一致，说明原文。Frozen image/LLM只video modules。visionPPstage4self+1cross，ViT replicatedallstage、sequenceparallelvision、microbatch8，FP32visiongradacc。

Image LR正文10×10^-4=1e-3批16384，小scale长schedule泛化差降LR；anneal2e-5wtdecay.01；imageSFT hot-swap instructLM frozen，model soups；visionRM frozenlanguageRM+squaremeanrewardpenalty，DPO referenceEMA每ksteps，不同text fixedDPO；QTselectiveSFT后smallDPO/earlystop。

Table29MMMU64.5<GPT4o69.1/Sonnet68.3，TextVQA84.8>Gemini78.7，Doc92.6<Sonnet95.2，Gemini93.1借externalOCR脚注需显式；Table30Perception60.8与TVQA87.9强，Activity56.3<GPT4o61.9/Gemini57.5，NExT30.3是WUPS非accuracy；TVQAsubtitles，ActivityGPT3.5judge，64frames/~3sec180s clip丢fastmotion。

## §8 speech

15Mhselfsupervised,VAD>.7/PIIfilter；230KhASR34langs，90KhAST33↔English；speechdialog60KhASR+25KhVoiceboxsynthetic。Conformer1B24layersd1536FFN4096kernel7h24，mel80stride4→40ms；adapter100Mconvk3stride2→80ms+Transformer3072+lineartoLMd。~12.5embedding/sec直接入sequence，no cross externalKV。

BEST-RQ16randomfixedcodebooks8192vectors projectingstack4×80→16 dims，mask32framesp2.5%,500Kstepsbatch2048；8B650Kupdatesbatch512LR1e-4，70B600Kbatch7684e-5，LMfrozen。targetLIDonly promptsASRrepeat/ASTtranslate，no sourceLID、zero-shot code-switch examplesFigure30只是qualitative。

Generation只8B16thlayerembeddings；TN LSTMrule tagger55Kexamples，PM50Khstudiovoice sixhead864dual-crossattentionphone+LM streams，predictlogduration/logF0/logpower，trainbatch1024≤500phones1MupdatesLR9e-4warm3K。phonefixedlookahead/tokenvariablechunk，traininfermask一致；TN+PM+acousticTransformer+WaveRNN不是LLM直接生成wave。

Table31MLS70B4.4vsWhisper6.2,Libri3.1vs4.9,FLEURS8.2vsWhisperv3 14.4，CJK/Thai用CER混metric，FLEURSWhisper33langs无Malayalam，not perfectlymatchedaverage。Table32BLEU33.7vsWhisper21.9=+11.8absolute，Covost38.8vs37.9。Table33AT/LT21langs,emptyGeminiassumedsafe可能filter/rate-limit混。Table34TN90.7vs3context73.6=+17.1pp(fullcontext88.0)，Table35PMpreferred60/63.6，无端到端firstaudio/real-time latency数字不能编。

## §9–10 与个人分析方向

Dense成熟baseline的结果支持scale/data/simplicity，可复用orthogonalGQA/RMSNorm/SwiGLU/RoPE；不否定MoE长期。主复现缺口：15T专有data、internalRMsafety、全annotation、frozen multimodal未全释。组织隔离evalteams减少人为overfit但不消除webcontamination。建议：业务pairlogit/rareerror、frozenpipelineruntimeKV、actualTTFT/TPOT+firstaudio、模型/系统/benchmark可比、语言/OCR/工具混淆矩阵、fixedhumantruthtest避免model改tests。

所有当前性能数字均为作者报告，未本地GPU/NPU模型训练或benchmark。交互为公式解析计算而非合成实验性能。
