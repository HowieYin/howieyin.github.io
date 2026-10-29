# Kimi K2 v2 · 两轮内容与版式审校

日期：2026-10-29。主源：32页2507.20534v2.pdf及官方TeX、图件、公开HF配置与reference实现。性能数字无本地GPU复现。

## 第一轮：来源准确性与数据完整性

- 逐章覆盖§1–6、Acknowledgments与附录A–G，按原顺序讲解。
- 核对13张原图/18个面板、6完整原表：T1三配置，T2九项架构，T3四类44项、7模型，T4三项模型信息与23任务，T5全部插件及4项配置策略，T6五类×五策略25行。
- 保留粗体、下划线、星号、缺失值和shots，CSV/TeX供追溯。表内470个小数项逐个在渲染表检验存在；各模型／row顺序按原TeX保留。
- 原11组展示公式与Algorithm1全部操作顺序保留，附录编号(1)–(6)、正文未编号记号分开。另有本文推导与公开配置计算，以标签区分。
- 核对MLA 512+64缓存、64头、qrank1536、qC128/qR64/v128、60MoE层、384选8、shared1、noMTP、不共享embedding、router choice bias与combine权重区分。
- 缩放固定输入的推导不写成未来logit硬界。训练激活FP8 storage与发布权重FP8 quantization分开；公开reference先展开再缓存与优化latent实现分开。
- 纠正Figure11初稿误读：实际为中文内部win/tie/loss比较，不是SFT/RL学习曲线。逐项保留59.6/23.5/16.9，64.6/18.8/16.6，65.4/17.6/17.0。
- 明列ACE、GPQA、PaperBench、ArenaHard、Table2箭头、Base64配置、spectral norm、Smax尺度和recipe计数歧义。原文冲突不静默改表。
- 更新部署指南到已核验README中的docs/deploy_guidance.md。
- 当前模型/API状态与历史论文评测分开；官方同源互引不作独立验证，小模型Muon研究/开源实现不称万亿K2复现。

## 第二轮：可读性、数学与交互的真实渲染

- 深色主站与浅色模式、英文界面、中文正文、当地字体；桌面sticky TOC、手机折叠目录、键盘/点击图片放大与Escape恢复。
- 所有原图从caption-free作者PDF渲染，外部图注与原脚注单列；原图内坐标、图例、标签完整保留。
- 两张原创Mermaid均实际生成SVG与PNG，保留mmd；三组交互有公式/假设和静态文字等价说明。
- Native MathML无需远程KaTeX；74个math对象实际显示；长公式可横滚并有提示；代码保留原例且标注出处。
- 首轮local以及offline的375/390/430/1440/1920px均无整页横向溢出、坏图、错误anchor、console/network错误，完整六表行数一致。
- 交互核验：S400/τ100→γ.25→total100；错误缩放→325；128Ki BF16 latent8.578125GiB，128heads保持latent不变而expanded ratio71.1111；batch2加倍；toy预算share95.238%、超限截断8192。
- 飞书镜像20个图片、74个latex、13个表（其中6原表、7解读表）保留；动态控件转可独立阅读的计算结果及网页链接，非原始占位符。

最终本地/离线/线上浏览器结果、飞书回查与Pages部署证据见publishing-report.md及对应JSON。
