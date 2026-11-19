# 真实模型诊断：范围与复跑

`experiment.py`是实际执行的完整脚本，保持原始数值逻辑与默认路径。结果来自macOS、Python3.12.2、PyTorch2.9.1、MPS；未启用确定性训练。它不是OE／Engram预训练或vLLM性能benchmark。

准备依赖：`torch numpy pyarrow transformers tokenizers sympy`。脚本固定本地输入，不会自动联网下载模型／数据：

| 位置 | 所需文件 |
| --- | --- |
| `/tmp/llm-ptq-survey-work/model/` | Pythia70M-deduped revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c` 的模型及tokenizer |
| `/tmp/llm-ptq-survey-work/train.parquet`、`test.parquet` | Salesforce/wikitext revision `b08601e04326c79dfdd32d625aee71d232d685c3`，配置wikitext-2-raw-v1，保留原行顺序及text列 |
| `/tmp/llm-ptq-survey-work/model-meta.json`、`dataset-meta.json` | 随附冻结元信息，脚本读取sha字段记录身份 |
| `/tmp/engram-survey-work/code/engram_demo_v1.py` | 随附官方冻结demo，SHA256 `9d082070654df217e21bbca9926a4267bdf2cce7777aa6739747c24de30d2044` |

来源：https://huggingface.co/EleutherAI/pythia-70m-deduped ；https://huggingface.co/datasets/Salesforce/wikitext ；https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/engram_demo_v1.py 。

可以修改脚本顶部R/B以使用自己的本地目录；复制数据时用`model-meta.json`、`dataset-meta.json`及`model-files.json`核对身份。CPU也可运行，但训练结果和时间不保证相同。

```bash
python experiment.py 0.001
python experiment.py 0.0001 lowlr
```

分别生成`experiment-results.json`和`experiment-results-lowlr.json`及训练支路权重，后者不是模型checkpoint的公开发布。两种预设LR、两条支路、三seed共12个实际运行；每个运行120steps。高LR退化也保留，不筛掉坏结果。原始日志是随附`experiment.log`和`experiment-lowlr.log`。

输入8192token、128窗口；test4096token、256窗口、4080计分目标；每个窗口新请求历史。真实65,536token hash统计使用同冻结train split，但不等于全部用于训练。input和layer-gated新增参数398208／532864不等，结果不支持普遍方法排名。

状态检查同时包含真实模型的故障PPL／logits恢复与CPUfloat64支路full/chunk等价。错误路径偶然改善PPL不代表正确。量化测试只将code重建为FP32再gather，不是packedkernel访存／速度测试。原结果元信息中的“before embedding-layer normalization”意指首次block归一化之前，准确hook是`gpt_neox.emb_dropout`的输出；Pythia并没有一个独立的embedding LayerNorm。原始结果文件保留不改。
