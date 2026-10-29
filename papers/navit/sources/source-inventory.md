# Source inventory

阅读与核验截止 2026-10-29。

| 类型 | 一手资料 | 用途 |
|---|---|---|
| Primary | https://arxiv.org/pdf/2307.06304v1 | 26 页全文、唯一主版本、全部章节与数值 |
| Primary source package | https://arxiv.org/src/2307.06304v1 | TeX、作者 PDF figures、表格 cells |
| Venue | https://proceedings.neurips.cc/paper_files/paper/2023/file/06ea400b9b7cfce6428ec27a371632eb-Paper-Conference.pdf | 会议身份；不替代 v1 |
| Prior | https://arxiv.org/abs/1906.06423 | FixRes 的 train/test resolution |
| Prior | https://arxiv.org/abs/2210.03347v2 | Pix2Struct variable resolution |
| Prior | https://arxiv.org/abs/2212.08013v2 | FlexiViT patch size |
| Systems | https://arxiv.org/abs/2205.14135 | FlashAttention 的 IO / 显式 map 区分 |
| Follow-up | https://arxiv.org/html/2502.14786v1 | SigLIP 2 NaFlex recipe、resolution interpolation 与 OCR/document retrieval |
| Follow-up | https://arxiv.org/html/2511.21631v2 | Qwen3-VL dynamic resolution / VLM 应用链路 |
| Current local evidence | https://arxiv.org/html/2603.22387v2 | EUPE Table 1/11 多任务与 iPhone CPU 对照 |
| Official implementation | https://github.com/google-research/big_vision/tree/0127fb6b337ee2a27bf4e54dea79cff176527356 | 固定版本 NaFlex encoder / preprocessing / checkpoint README |
| Official current models | https://github.com/facebookresearch/eupe | EUPE 公开代码/模型 |

原 PDF、source tar、相关 appendix TeX、官方 NaFlex 源码和 README 均保留于当前目录。`figures.json` 记录 22 图原始 panel 文件名和生成 SHA256；`tables.json` 记录全 cells。没有找到可验证的独立 NaViT project/slides/talk 时，不制造链接。Scenic 当前目录检索的局限已在正文说明。
