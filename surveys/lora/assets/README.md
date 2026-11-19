# LoRA 实测诊断

这是公开小 checkpoint 上实际运行的教学和工程诊断，不是 LoRA、LoRA-FA、LoRAHub、QLoRA 或 ISO-LoRA 的完整复现。正式结果使用独立随机源产生训练窗口顺序，每个种子的五种方法读取同一批次序列；以 `results-20261119.json`、`results-20261120.json`、`results-20261121.json` 中的 `train_window_indices` 复核。`results.json` 是三种子的算术均值和样本标准差。

## 输入与执行

需要 Python 3.11 或更新版本。依赖见 `requirements.txt`。模型与数据按固定 Hugging Face revision 下载，每个文件校验 SHA-256，下载器不会默认追踪 main。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prepare_inputs.py --data-dir ./inputs
python experiment.py --seed 20261119 --data-dir ./inputs --output-dir ./outputs
python experiment.py --seed 20261120 --data-dir ./inputs --output-dir ./outputs
python experiment.py --seed 20261121 --data-dir ./inputs --output-dir ./outputs
python diagnostics.py --data-dir ./inputs --adapter-dir ./outputs --output-dir ./outputs
```

在 macOS 上自动采用 MPS，有条件时用四个 CPU 线程；其他平台采用 CPU。设备、内核和软件版本会改变数值与时间，不承诺跨设备逐位重现。

## 训练和评测口径

模型为 `EleutherAI/pythia-70m-deduped`，70,426,624 参数，revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`。数据为 `Salesforce/wikitext` 的 `wikitext-2-raw-v1`，冻结 revision 见 `dataset-meta.json`。源行以两个换行连接，再取 tokenizer 编码的前 16,384 个 train tokens 和前 4,096 个 test tokens。测试 16 个独立长度 256 窗口，每窗口评测 255 个 next-token targets，共 4,080 个。窗口间的边界 token 不计入。

训练 96 steps、batch1、length256、AdamW、weight_decay0、梯度裁剪1，模型保持 eval 以关闭 dropout，但梯度仍开启。目标为 24 个 Transformer Linear 的权重，不训练 embedding、lm_head、bias 和 norm。“full”表示全量更新这 24 个权重的 18,874,368 参数，**不是整个 70M 模型全参数微调**。LoRA 的 rank 为4/16/64，alpha=rank，倍率始终1；固定A组rank16。LoRA组196,608 / 786,432 / 3,145,728可训练参数，固定A组442,368。

全量目标权重学习率2e-5，LoRA和固定A组1e-3。每家族只有预设学习率，没有等预算搜索，因此不能用 PPL 排名证明算法优劣或把高rank退化归因于纯表达能力。本次 rank 对比没有固定 alpha，不能用来直接验证 rsLoRA 的固定 alpha 缩放结论。固定A加普通AdamW也不等于LoRA-FA v3的专门optimizer。

## 函数一致性与错误注入

- 初始B=0，确认新adapter不改变实际 logits、第一步A梯度为0、B梯度非0，冻结基座无梯度。
- 实际保存adapter及rank、alpha、模型revision和target元信息，重载后要求探测logits精确恢复。
- 训练后合并，记录最大/平均logits误差、64个位置的argmax差异以及完整4,080 targets测试NLL。验收为有限logits和mean NLL差≤1e-3 nats；这不是bit-exact保证。
- 注入重复合并，恢复后必须恢复原始单次合并logits。rank16首种子还注入错误alpha并检查精确恢复。
- 第一次预设max-logits误差2e-4验收失败。历史权重和原始证据保留在 `../evidence/initial-training-protocol/`；CPU float64合并误差约2.73e-12，而MPS/CPU float32均可有约1e-2最大logits差。不要隐藏这一失败，也不要误称为MPS独有问题。

## 补充诊断的边界

`diagnostics.py`使用实际训练的adapter以及WikiText2真实层输入，比较两个rank4 adapter的因子平均、增量平均和rank≤8串接。这是同任务不同seed的代数检查，不能据此推断多任务能力。

量化顺序比较为 `Q/DQ(W0)+BA` 与 `Q/DQ(W0+BA)`；使用按输出行的对称W4 RTN、qmax7、15个有效等级、浮点矩阵乘法。它不使用NF4、double quantization、量化感知训练或低位内核，不能冒充QLoRA，也不说明低位GPU吞吐。被测adapter原来训练于浮点基座，两个量化路径都是训练后的扰动。

`saved_tensors_hooks`检查单个1536×512 Linear、64个真实输入位置、CPU float32、输入需梯度时，实际保留的非参数storage总字节数。固定A后不再为A梯度保留64×512输入，但仍需rank激活、上游模型反向以及损失输出。这些数不是全模型峰值、显存分配器统计或LoRA-FA论文显存结果。

最终第一层增量的熵有效秩使用奇异值概率 `sigma/sum(sigma)`。ISO-LoRA使用能量概率 `sigma²/sum(sigma²)`并分析每步一阶切空间对象；定义和对象均不同，不能写成ISO复现。Stable rank只由Frobenius范数与谱范数比值确定，对象差异依然存在。

计时只在rank16首种子运行：单进程float32、batch1、64-token prompt，两次warmup，五次采样；prefill+first selection与8个KV-cache decode步骤分别同步计时。排除tokenizer、网络、排队与服务调度；不是用户端到端TTFT，也不支持2026多租户服务排名。

未合并prefill五次采样包含233.1509ms离群，mean60.7171ms、median17.4533ms；合并mean17.4439ms、median18.2622ms。样本和离群均保留，不宣称稳定3.48倍加速。decode合并/未合并mean12.3870/13.6600ms每步也仅是单进程小样本观察；需要独立进程、交替测量和系统状态控制才能判断稳定收益。
