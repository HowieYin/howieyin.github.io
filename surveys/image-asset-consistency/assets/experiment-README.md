# 实际编码器与区域外像素诊断

此实验不运行图像生成器，不复现 PhotoMaker 等模型，不评估身份识别准确率。

输入来自 DreamBench++：https://huggingface.co/datasets/yuangpeng/dreambench_plus 。固定选择 `live_subject/animal/00,01`、`live_subject/human/00,01`、`object/00,01`、`style/00,01`。results.json 保留下载包、每张输入和模型 revision 哈希，及完整预处理配置。

模型为 https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 和 https://huggingface.co/facebook/dino-vits16/tree/abe3b354cb6a9b6f146096b14a4a9d7eecbcb4bd 。

实际环境：macOS ARM64；Python 3.12、torch 2.15.0、transformers 4.57.6、Pillow 13；CPU float32、eval/no_grad、四线程。编码器耗时只统计处理输入和 forward，不是生成时间或严谨性能 benchmark。8张×8种确定性变换共64输入。样本选择不是随机总体样本。

运行前将数据包中的 `images/` 放入工作目录 `data/images/`，保存原包为 `data.zip`；将模型 API metadata 的 revision 保存在 `clip-model-meta.json` 和 `dino-model-meta.json`（字段 `sha`）。安装 torch、transformers、Pillow、numpy，修改脚本顶部 R 为实际工作目录后运行 `python experiment.py`。联网下载仅用于冻结模型和数据；可从已有 cache 离线加载。

脚本输出实验图片、results.json 和 variants-0.png/variants-4.png。每张图强制 LANCZOS resize 为512²，可能改变长宽比；CLIP和DINO各用自身224²预处理。中心64²蒙版为人为已知控制区域，不是主体分割。

检查：精确数组编辑→PNG 的蒙版外误差为零；JPEG90后连带像素变化；亮度／翻转／通道交换时特征相似与像素误差分离。数值只支持上述实际诊断，不能外推任何生成器或身份编码器的准确率。
