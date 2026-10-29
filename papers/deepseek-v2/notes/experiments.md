# 实验审计入口

作者实验：8.1T预训练、中英多项benchmark、128K NIAH、SFT与GRPO、Lite和attention消融。全部37表保留在主稿对应章节；评测prompt26表保留在附录G。

关键公平性：V2/67B同时改变架构、数据与训练量；Lite5.7T vs2T；Table8改layercount；Table9大组420B而非8.1T；服务FP8权重/平均6bitKV，缺乏固定precision/batch/SLO的延迟曲线。

本地仅进行了代数与容量验证、原始表/公式完整性比对、页面与交互测试。没有本地GPU训练/推理复现。结果详见 `numerical-check.json`、`review.md` 和 QA JSON。
