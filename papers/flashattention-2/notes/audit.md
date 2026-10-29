# FlashAttention-2 v1 公式、反向算法与实验独立审计

## 范围与证据

本审计只针对本地 arXiv:2307.08691v1，不引用后续版本、当前实现或网络资料。PDF 首页题面日期为 July 18, 2023，arXiv 侧栏为 17 Jul 2023；共 14 页，下文页码为 PDF 原印刷页码。没有联网，没有调用 GPU，没有执行 git，没有编辑主报告页面或制作图表。

交付目录先通过 `ls` 确认：`paper-readings` 存在，`flashattention-2/notes` 起初不存在，随后仅为本次交付创建。源码包在检查父目录 `/private/tmp`、归档文件列表后，独立解压至 `/private/tmp/fa2-audit-v1/source`。包内是论文 TeX、参考文献及图件，不是 FlashAttention CUDA/Triton 实现。

证据索引：

- **S1**：[本地 v1 PDF](file:///Users/bytedance/project/read_papers/paper/2307.08691v1.pdf)。SHA-256：`4aa8935dfacaf6ae8c68f772ca92f730154a0dd0e1bceeb59c7bc56c512d5868`。
- **S2**：[本地文本抽取](file:///private/tmp/fa2-work/paper.txt)，用来定位，不用来单独裁决排版歧义。
- **S3**：[源码包](file:///private/tmp/fa2-work/source.tar)。SHA-256：`5aea21e5eaa54ebb043336351611454998422e89add584e3d268f0a2f6630bcd`。
- **S4**：[background.tex](file:///private/tmp/fa2-audit-v1/source/src/background.tex)。
- **S5**：[algo.tex](file:///private/tmp/fa2-audit-v1/source/src/algo.tex)。
- **S6**：[experiments.tex](file:///private/tmp/fa2-audit-v1/source/src/experiments.tex)。
- **M1**：[numerical-check.json](file:///Users/bytedance/project/read_papers/paper-readings/flashattention-2/notes/numerical-check.json)，全部为 float64 数学自检，不是 GPU 复现。
- **M2**：[数学自检脚本](file:///private/tmp/fa2-audit-v1/check.py)，三个独立进程分别执行 `softmax`、`backward`、`table` 子问题，再合并结果。

独立子问题为：A，§2.2 的形状与缩放；B，在线 softmax 和 Algorithm 1；C，Algorithm 2 的重计算、梯度与确定性；D，Table 1 数字复算；E，实验可比性。各自先从原文取证，再进行数学或算术交叉检查，最后汇总。PDF 与 TeX 是同一论文的不同表示，不冒充独立发表的两个来源。数学反例、链式法则及有限差分构成另一路检查。

## 主要结论

| 编号 | 原文位置 | 裁决 |
|---|---|---|
| E1 | §2.2，p.3 | `dK = Q dS^T` 的乘法顺序确错，一般形状不兼容；应为 `dS^T Q`。 |
| E2 | §2.2，脚注 2，p.3 | “typically by 1/d” 确实是原文，常规 scaled dot-product attention 应为 `1/sqrt(d)`；`1/d` 本身也可定义另一种温度，不能据此断言作者 GPU 实现用了错误缩放。 |
| E3 | §2.3.1，p.4 | 标准 attention 输出展开漏掉第二项的归一化；在线归一化输出更新同时写反比值、漏掉 max 变化因子。 |
| E4 | §3.1.1，p.5 | 归一化更新重复 E3；“un-scaled” 更新的 `diag(ell1)^(-1) O1` 也错。 |
| E5 | §3.1.1 与 Algorithm 1 line 10，p.6 | 两处都把 `diag(exp(m_old-m_new))` 错写成逆矩阵。Algorithm 1 不是正确勘误基准；原样首次迭代即出现奇异逆矩阵。 |
| E6 | Algorithm 2 line 4，p.7 | `D ∈ R^d` 错，应为 `R^N`。反向核心五个 matmul 的形状和 `dK` 乘法顺序则正确。 |
| E7 | Causal masking 第二点，p.6 | “row indices ... strictly less than column indices” 写反；免 mask 应是整块 key 列不晚于 query 行。 |
| E8 | §4.2，p.11 | “1.3× compared to FlashAttention-2” 为自比较笔误，应是 FlashAttention。 |
| E9 | §4 概述和 §4.2，pp.9–11，相对 Table 1 p.12 | Table 1 的最大 FA2/无 FA 是 `220/72 = 3.0556×`，不是文字声称的最高 `2.8×`。`2.8×` 只匹配 2.7B/8k 那一行。 |

E1、E3–E7 为原文数学或索引错误，置信度高。E8 是内部明确的基线指代错误。E9 是原表与概述不一致，不是认定表中测量造假。E2 是常规缩放的表述勘误，与省略缩放的教学公式本身分开。

## §2.2：形状、链式法则与缩放脚注

### dK 的正确顺序

输入 \(Q,K,V\in\mathbb R^{N\times d}\)，\(S=QK^\top\)，\(dS\in\mathbb R^{N\times N}\)。因此

\[
dQ=dS\,K\in\mathbb R^{N\times d},\qquad
dK=dS^\top Q\in\mathbb R^{N\times d}.
\]

原文 `Q dS^T` 的尺寸为 \((N\times d)(N\times N)\)，除非碰巧 \(N=d\)，矩阵乘法都无法进行；即使 \(N=d\)，也不能凭形状相容改变链式法则的乘法顺序。M1 用 \(N=7,d=3\) 验证原式触发 NumPy 形状错误。Algorithm 2 line 16 正确写成 \((dS_i^{(j)})^\top Q_i\)，是论文内部的直接反证。[S1 p.3、p.7；S4 line 67；S5 line 160]

### 脚注 2

TeX 原文为 `typically by $1/\mathrm{d}$`；PDF 同样印作 `1/d`，不是漏识别根号。常规标准是

\[
S=cQK^\top,\qquad c=1/\sqrt d.
\]

若每个分量独立、零均值、单位方差，点积方差为 \(d\)，乘 \(1/\sqrt d\) 后方差为 1，而乘 \(1/d\) 后方差为 \(1/d\)。这是解释常规缩放的数学依据，不是对实际训练数据分布的假设。

对于任意明确选择的 \(c\)，恢复缩放时两处梯度也必须恢复因子：

\[
dQ=c\,dS\,K,\qquad dK=c\,dS^\top Q.
\]

论文声明为清楚起见省略 scaling、mask、dropout，所以 Algorithm 1/2 不写这些操作并不单独构成错误。不能从该脚注推导其 CUDA kernel 使用了 `1/d`。[S1 p.3 脚注 2；S4 lines 52–54]

## §2.3.1 与 §3.1.1：在线 softmax 勘误

### 正确不变量

对同一 query 行块，先后处理两块 logits \(S_1,S_2\)，定义

\[
m_1=\operatorname{rowmax}(S_1),\quad
m_2=\max(m_1,\operatorname{rowmax}(S_2)),\quad
\alpha=\exp(m_1-m_2).
\]

\[
E_1=\exp(S_1-m_1),\quad E_2=\exp(S_2-m_2),\quad
\ell_1=\operatorname{rowsum}(E_1),\quad
\ell_2=\alpha\ell_1+\operatorname{rowsum}(E_2).
\]

所有行向量均沿列或输出维广播。若 \(O_1=\operatorname{diag}(\ell_1)^{-1}E_1V_1\) 是已经归一化的旧输出，则正确更新为

\[
O_2=\operatorname{diag}(\alpha\ell_1/\ell_2)O_1
    +\operatorname{diag}(\ell_2)^{-1}E_2V_2.
\]

若 \(U_1=E_1V_1\) 是未归一化 numerator，则正确更新为

\[
U_2=\operatorname{diag}(\alpha)U_1+E_2V_2,\qquad
O_2=\operatorname{diag}(\ell_2)^{-1}U_2.
\]

亦即，从已经归一化的 \(O_1\) 开始时应有 \(U_2=\operatorname{diag}(\alpha\ell_1)O_1+E_2V_2\)，不是 `diag(ell1)^(-1) O1`。以上由将旧指数从基准 \(m_1\) 改写为基准 \(m_2\) 直接得到。

### 原文逐项问题

1. **p.4 标准输出展开**：S4 line 120 为 `diag(ell)^(-1) exp(S1-m) V1 + exp(S2-m) V2`。第二项也需要 `diag(ell)^(-1)`；或者括住两个 numerator 后一起左乘归一化矩阵。
2. **p.4 在线输出更新**：S4 line 132 使用 `diag(ell1/ell2)^(-1) O1`，等于乘 `ell2/ell1`，比值写反，且没有 `exp(m1-m2)`。同一行后半段的两个全局归一化项才是正确目标。
3. **p.5 第一项优化解释**：S5 line 33 重复上一错误；S5 line 38 的“un-scaled”旧项为 `diag(ell1)^(-1) O1`，与前面已经归一化的 \(O_1\) 定义不相容。应是 `diag(alpha*ell1) O1`。
4. **p.6 两块未归一化更新**：S5 line 58 把 `diag(alpha)` 写成 `diag(alpha)^(-1)`。同一行后半段全局 numerator 的展开是对的，左侧递推是错的。
5. **p.6 Algorithm 1 line 10**：S5 line 85 仍是 `diag(alpha)^(-1) O_old`；须去掉 `^(-1)`，line 12 的最终 `diag(ell)^(-1)` 保留。初始化 \(m^{(0)}=-\infty\) 时，第一块的 \(\alpha=0\)，原式要求零矩阵求逆。实数意义下未定义，float64 直译为 `0/0 -> NaN`。即使特殊处理了第一块，后续最大值上升仍会用错逆因子。
6. **符号一致性**：p.6 小例子使用的 `tilde P2` 是已经归一化的，但 Algorithm 1 line 9 的 `tilde P` 是未归一化指数。小例子中的该变量没有用于其 numerator 更新，不单独构成数值错误；解释时必须明确区分，不能误称所有 `tilde P` 都是同一种量。

PDF p.6 已本地渲染并视觉检查，两处逆矩阵上标真实存在，不是 OCR 或文本抽取造成。p.7 的“Algorithm 1 returns the correct output”声明对**按印刷式逐字执行**不成立；修正后成立的算法原理与作者实际 GPU 实现是否含笔误，是两个问题，后者没有本地实现可供核实。[S1 pp.4–7；S4 lines 116–132；S5 lines 31–59、77–88、113–119]

### 两个可复核 float64 反例

下面只是数学自检。每个例子 \(B_r=1,B_c=2,d=1\)，每块有两个 key。

| 项目 | 最大值不变：S1=S2=[0,0]，V1=[1,1]，V2=[2,2] | 最大值上升：S1=[0,0]，S2=[2,2]，V1=[1,1]，V2=[0,0] |
|---|---:|---:|
| 稠密正确输出 | 1.5 | 0.11920292202211755 |
| p.4 标准展开漏第二项归一化 | 4.5 | 0.11920292202211755 |
| p.4/p.5 归一化递推原式 | 3.0 | 1.1353352832366128 |
| p.5 un-scaled 原式后再归一化 | 1.125 | 0.22019926949447058 |
| p.6 inverse-rescale 小例子原式 | 1.5 | 6.508259020952766 |
| 修正后的两个递推 | 1.5 | 0.11920292202211755 |

反例有意同时覆盖“max 未变化”与“max 上升”：只测不变 max 会遗漏 inverse-rescale 错误，只令第二块 \(V_2=0\) 会遗漏标准展开漏第二项归一化错误。M1 保留全部输入、中间量和各原式输出。另用 \(Q=[1,1]^\top,K=[0,2]^\top,V=[1,0]^\top,B_r=B_c=1\) 对完整 Algorithm 1 原式进行直译，两行输出都是 NaN，修正版均为 0.11920292202211755。

### 附带发现：causal block 条件写反

原文定义 causal mask 为 key 列索引 \(c>r\) 时设 \(-\infty\)。因此：

- 全块满足 `min(key_column) > max(query_row)` 时，整块跳过。
- 全块满足 `max(key_column) <= min(query_row)` 时，无需逐元素 mask。
- 其余相交块需要元素级 mask；同起点、同大小的方形分块通常只剩对角块。

p.6 第二点却说 query 行“strictly less than”key 列时免 mask，与第一点和其定义相反。应是 key 列位于 query 行之前的块，无需 mask。[S5 lines 98–110]

## §3.1.2：Algorithm 2 完整原文步骤及解释

### 原文结构转录

以下保留原文步骤、初始化与加载结构；`R^d [sic]` 只用于标出原文错误，未悄悄修正。算符用 ASCII 转录，pointwise 行向量广播与论文相同。`S_ij` 对应当前块 `S_i^(j)`。[S1 p.7；S5 lines 130–164]

```text
Require: Q,K,V,O,dO in R^(N*d) in HBM; L in R^N in HBM; Bc,Br.
 1. Divide Q into Tr=ceil(N/Br) blocks Qi of Br*d;
    divide K,V into Tc=ceil(N/Bc) blocks Kj,Vj of Bc*d.
 2. Divide O,dO into Tr blocks Oi,dOi of Br*d;
    divide L into Tr blocks Li of Br.
 3. Initialize dQ=0 in R^(N*d) in HBM and partition into dQi of Br*d.
    Partition dK,dV into Tc blocks dKj,dVj of Bc*d.
 4. Compute D=rowsum(dO * O) in R^d [sic; should be R^N],
    write D to HBM; partition into Tr blocks Di of Br.
 5. for j=1,...,Tc:
 6.   Load Kj,Vj from HBM to SRAM.
 7.   Initialize dKj=0,dVj=0 in R^(Bc*d) on SRAM.
 8.   for i=1,...,Tr:
 9.     Load Qi,Oi,dOi,dQi,Li,Di from HBM to SRAM.
10.     S_i^(j) = Qi Kj^T in R^(Br*Bc).
11.     P_i^(j) = exp(S_ij - Li) in R^(Br*Bc).
12.     dVj += (P_i^(j))^T dOi in R^(Bc*d).
13.     dP_i^(j) = dOi Vj^T in R^(Br*Bc).
14.     dS_i^(j) = P_i^(j) * (dP_i^(j) - Di) in R^(Br*Bc).
15.     Load dQi from HBM to SRAM;
        dQi += dS_i^(j) Kj in R^(Br*d); write back to HBM.
16.     dKj += (dS_i^(j))^T Qi in R^(Bc*d).
17.   end for
18.   Write dKj,dVj to HBM.
19. end for
20. Return dQ,dK,dV.
```

### D、重计算与五个 matmul

\[
D_r=\sum_a dO_{ra}O_{ra}=\sum_c P_{rc}dP_{rc},\qquad D\in\mathbb R^N.
\]

等式来自 \(O_{ra}=\sum_c P_{rc}V_{ca}\) 以及 \(dP_{rc}=\sum_a dO_{ra}V_{ca}\)。它让每个 column tile 用同一个全行归约量计算 \(dS=P\circ(dP-D)\)，而不用在 tile 之间再归约完整 `dP`。原文 line 4 标成 \(R^d\) 与 rowsum 的定义、后续 \(T_r\) 个 \(B_r\) 分块都矛盾。line 9 的 `Oi` 在 `D` 已预计算后不再被后续公式使用；`dQi` 又在 line 15 重载，属于伪代码冗余，不是新的数学错误。

\(L_r=\log\sum_c\exp(S_{rc})\)，因此 tile 上重计算

\[
S_i^{(j)}=Q_iK_j^\top,\qquad
P_i^{(j)}=\exp(S_i^{(j)}-L_i)
\]

恢复的是**完整行 softmax 的对应列块**，不是当前 tile 的局部 softmax；这里不能再按 tile 自己的 rowsum 归一化。`P` 的恢复是逐元素 exp，不是 matmul。无需将全矩阵 `S,P` 写 HBM。

| Algorithm 2 行号 | 矩阵乘法 | 左输入形状 | 右输入形状 | 输出形状 |
|---|---|---|---|---|
| 10 | 重计算 \(S=Q_iK_j^\top\) | \(B_r\times d\) | \(d\times B_c\) | \(B_r\times B_c\) |
| 12 | \(dV_j += P^\top dO_i\) | \(B_c\times B_r\) | \(B_r\times d\) | \(B_c\times d\) |
| 13 | \(dP=dO_iV_j^\top\) | \(B_r\times d\) | \(d\times B_c\) | \(B_r\times B_c\) |
| 15 | \(dQ_i += dS K_j\) | \(B_r\times B_c\) | \(B_c\times d\) | \(B_r\times d\) |
| 16 | \(dK_j += dS^\top Q_i\) | \(B_c\times B_r\) | \(B_r\times d\) | \(B_c\times d\) |

忽略边界和 causal 减半，每个 tile 的五项 matmul 各约 \(2B_rB_cd\) FLOPs，共 \(10B_rB_cd\)；前向两个 matmul 为 \(4B_rB_cd\)。这解释 §4.1 的 backward/forward FLOP 比 2.5。相对于已保存 `P` 的普通反向，额外 matmul 是重计算 `S`；softmax 反向本身为逐元素与归约运算。[S4 lines 150–165；S5 lines 142–160；S6 lines 41–49]

处理 scaled logits 时 line 10 使用 `c*Qi Kj^T`，line 15/16 的增量各乘 `c`。处理 causal mask 时重计算的 `S` 必须恢复相同 mask，屏蔽概率为零；处理 dropout 需相同随机掩码及缩放。这些是恢复论文省略设置时的实现条件，不是声称印刷 Algorithm 2 已包含相应实现。

### 原子 dQ 与确定性

Algorithm 2 以串行 `j` 外循环写 ordinary read/add/write，单独看不存在跨 block 竞争。§3.2 p.8 则明确说明：一个 GPU thread block 负责一个 key/value column block，并用 **atomic adds** 更新共享的 `dQ`。[S5 lines 213–220]

因此，实际并行语义应是对全局零初始化的 `dQi` 累加该 worker 的**局部贡献** `dS_i^(j) Kj`；不能把各 worker 读出的旧 `dQi` 加上贡献后，用普通 store 写回，否则会丢更新；也不能把“旧值加贡献”的完整结果再 atomicAdd，否则重复计入旧值。`dKj,dVj` 则由该 column block worker 在片上累加后独占写回。`D` 和 `L` 先就绪，`dQ` 先清零，属于阶段同步条件。

原子加保证竞争的累加不丢更新，**不保证跨运行 bitwise deterministic**。不同 block 到达原子的次序可能不同，浮点加法不满足结合律。float64 数学反例 `[1e16,-1e16,1]` 按 0,1,2 顺序求和为 1，按 0,2,1 顺序为 0；这不是 GPU 原子测量。M1 的真实分块梯度贡献也在穷举列块累加顺序时出现最多 `4.44e-16` 差异，说明仅调整加法顺序就足以改变末位，不能推断 GPU 上差异的大小或发生频率。

原文没有提供确定性模式、固定跨块归约顺序或其开销的实验。实现确定性可使用固定顺序的独立归约或改变写入所有权，但通常增加工作区、归约或额外访存；这是审计建议，不是 v1 已实现功能。相反，固定前向 row block worker 独占自己的输出，避免了这一种跨 block 输出累加竞争，也仍不自动等于对任意软件、设备和精度的 bitwise 保证。

MQA/GQA 段落另要求在共享 KV 的不同 query heads 间归约 `dK,dV`。上述“每个 column worker 独占 dK/dV”的论证先针对 Algorithm 2 的单 head/MHA 视角；推广到 GQA 时仍需安排 head 间归约，不能从单 head 伪代码推出免同步实现。[S5 lines 168–176]

## Table 1：全部原始数据与派生数值

原始数据位置：PDF p.12；S6 lines 186–197。环境是 **8×A100 80GB SXM**，单位是 **TFLOPs/s/GPU**。下列三个性能列逐项照录，没有把集群总量写成单卡数据。

### 原始数据

| Model | Without FlashAttention | FlashAttention | FlashAttention-2 |
|---|---:|---:|---:|
| GPT3-1.3B 2k context | 142 | 189 | 196 |
| GPT3-1.3B 8k context | 72 | 170 | 220 |
| GPT3-2.7B 2k context | 149 | 189 | 205 |
| GPT3-2.7B 8k context | 80 | 175 | 225 |

### MFU 复算

用论文 §3.1 给出的 A100 FP16/BF16 matmul 峰值 **312 TFLOPs/s** 作分母，按 `100*reported_TFLOPs/312` 复算。以下是从报告值派生的 MFU，并非另外测得的硬件指标。

| Model/context | 无 FA MFU | FA1 MFU | FA2 MFU |
|---|---:|---:|---:|
| 1.3B / 2k | 45.5128% | 60.5769% | 62.8205% |
| 1.3B / 8k | 23.0769% | 54.4872% | 70.5128% |
| 2.7B / 2k | 47.7564% | 60.5769% | 65.7051% |
| 2.7B / 8k | 25.6410% | 56.0897% | 72.1154% |

最高 `225/312 = 72.1153846%`，与原文约 `72%` 一致。该比值不能解释为整个 kernel 的 SM active time、Tensor Core active time 或逐条实际指令计数。

### 同行加速比

| Model/context | FA1 / 无 FA | FA2 / 无 FA | FA2 / FA1 |
|---|---:|---:|---:|
| 1.3B / 2k | 1.3310× | 1.3803× | 1.0370× |
| 1.3B / 8k | 2.3611× | **3.0556×** | **1.2941×** |
| 2.7B / 2k | 1.2685× | 1.3758× | 1.0847× |
| 2.7B / 8k | 2.1875× | 2.8125× | 1.2857× |

同一行若使用相同 FLOP 分子，上述 throughput 比等于运行时间的反比，不能因后述 FLOP 约定的问题否认该算术关系。

原文“up to 1.3× vs FA1”可由 `220/170 = 1.2941176×` 支持。原文“up to 2.8× vs without FA”只符合 `225/80 = 2.8125×`，遗漏表中更大的 `220/72 = 3.0555556×`；不是正常整数四舍五入可以解释的差别。即使假设 220 与 72 均由真实数四舍五入至整数，比值范围仍约为 **3.0276–3.0839×**，不能降为 2.8×。此范围是条件性的舍入区间，不是统计置信区间。

§4.2 的 “1.3× speedup compared to FlashAttention-2” 为自比较，应按 Table 1 和 §4 概述改为 FlashAttention。Table 1 的 72 一格 `TFLOPS/s` 大小写混杂仅是单位排版，不另作性能错误。[S6 lines 15–19、164–168、194–197]

## v1 实验可比性与证据边界

### 已明确报告的条件

§4.1 使用 A100 80GB SXM4；sequence length 为 512、1k、2k、4k、8k、16k；总 tokens 固定 16k；hidden dimension 为 2048；head dimension 为 64 或 128，对应 32 或 16 heads；分别有 causal 和 noncausal 条件。对比 PyTorch、FlashAttention、xformers/CUTLASS、FlashAttention Triton。§4.2 为 8×A100 80GB SXM、1.3B/2.7B、2k/8k。[S6 lines 25–49、164–182]

不在本分支重新提取或重绘 Figures 4–7 的柱状数据；主 agent 负责图表证据和页面。

### 1. FLOP 口径不统一：microbenchmark 与 Table 1 不宜混用

§4.1 写的是每条序列

\[
F_{\rm fwd}=4s^2dH,\quad F_{\rm bwd}=2.5F_{\rm fwd},\quad
F_{\rm fwd+bwd}=3.5F_{\rm fwd},
\]

且 causal 时把上述 forward 分子除以 2。如果 runtime 计量整个 batch，总 FLOPs 还需乘 batch size \(B\)。论文文字没有在公式中写 \(B\)，可以视为省略“per sequence”的说明；缺少本地 benchmark 脚本，不能认定图中的数值实际漏乘了 \(B\)。

Table 1 则按

\[
F_{\rm model}=6sP_{\rm model}+12L_{\rm layers}h s^2
\]

计算，并明确说明为了沿用文献口径，**不对 causal attention 项减半**。这不是未披露的选择，但意味着 kernel 图中的 TFLOPs 与 Table 1 的 MFU 不是同一种工作量定义。前者包含反向 `S` 重计算的第五个 matmul；后者模型 FLOP 公式没有单独加回重计算项。

若只把 Table 1 attention 项改为 causal 减半，而时间和其他条件固定，新旧分子比例为

\[
\frac{6sP+6Lhs^2}{6sP+12Lhs^2}
=\frac{P+Lhs}{P+2Lhs}.
\]

因此新的 MFU 会下降，长上下文通常对此更敏感。v1 没给出 Table 1 的确切层数、hidden size 和参数计数细节，不能仅用“1.3B/2.7B”精确重算每行修正 MFU；也不应擅自代入所谓标准 GPT3 配置。同行速度比的分子仍会抵消，不受这一统一口径变更影响。

### 2. “2.5× backward FLOPs”不是所有 baseline 实际执行指令的精确计数

保存 `P`、不重计算 `S` 的标准反向有四个主要 matmul；FA 的反向有五个。如果为全部 baseline 共用 `5/2` 倍前向的 backward 分子，便是一个统一的 nominal 工作量指标，不是各方法实际执行的 GEMM FLOPs。条件性地，标准四 matmul 反向的显示吞吐相对其真实四 matmul FLOP 计数会高 `5/4 = 1.25` 倍；前向+反向为 `7/6 = 1.1667` 倍。

反之，某些 baseline 若也重计算或融合，以上四 matmul 假设不成立。仅凭 v1 包无法核验具体路径。causal 情况下，dense baseline 也可能仍执行整块 logits 再 mask，而 FA 跳过半数块。故这里的 TFLOPs/s 更适合作为共同工作量/时间的性能指标；不能不区分执行路径就当作实测 Tensor Core 利用率。相同输入条件下，共用分子的 speedup 仍可表示时间比。

### 3. 固定 tokens 同时改变 sequence length 和并行供给

若将 16k 按 16384 tokens 理解，batch sizes 分别为 32、16、8、4、2、1。因此随 sequence length 增长，FA1 的 batch×heads thread blocks 随之减少。测得改善混合了更大矩阵、原算法并行不足和 FA2 增加 sequence 维并行的效果，这是合理目标场景，但不能单独用这些曲线分离各优化的贡献。

特别是“不更改总 tokens 的长上下文”和“同 batch 翻倍 sequence length”的成本增长不同。v1 没有 Table 1 中 16k 的端到端数据，不能用该表直接验证 §5 的“16k 与旧 8k 同价格”泛化表述。

### 4. baseline、环境和计时细节不足

v1 正文/本地源码包未提供足以锁定各路径的 PyTorch/CUDA/Triton/xformers/FlashAttention 版本或 commit，也未提供 microbenchmark 的精度、accumulation 精度、dropout 参数、张量布局、具体 baseline 调用、warmup、同步/计时方式、重复次数和误差条。未给 GPU 时钟、功耗、软件环境和多卡互联/通信配置等完整复现细节。

不能把“PyTorch baseline”自动当成后续版本默认 SDPA，也不能拿当前实现结果替代 v1 实验。在没有实际执行脚本的条件下，无法确定不同路径是否使用完全相同的语义、精度和额外操作。这是可复核性缺口，不是已证明基线不公平。

### 5. Table 1 的端到端设置不足以拆解归因

Table 1 没列确切 architecture、vocabulary/embedding 参数处理、micro/global batch、gradient accumulation、optimizer、activation checkpointing、distributed parallelism、attention dropout、训练步数和计时包含范围。内存更省可能允许更大 microbatch、减少 checkpointing，这些本可作为系统收益，但必须显式报告是否在各列保持一致。

因此表格支持作者所报告场景的训练吞吐差异；不能从仅四行 throughput 精确归因到单一 kernel 优化，或推导同 token budget 的 time-to-quality、最终 loss 一致性和推理/解码性能。v1 没有报告收敛、梯度误差、原子确定性或统计方差实验。

### 6. 缺少逐优化消融

算法减少 non-matmul FLOPs、sequence 维 parallelism、warp partitioning 和 block-size 手工调优共同变化。v1 没有控制其他因素、逐个启停的消融表或 kernel profile 来量化各部分贡献。§3 的机制解释可信，但“哪项独立带来多少倍”不能仅由 Figures 4–7 或 Table 1 推得。[S5 lines 286–299；S6]

## float64 数学自检结果与复核

M1 严格区分原文直译、勘误版本和测量数据。随机输入仅用于数学测试，完整保存于 JSON，不冒充作者实验样本。

- Python：`/private/tmp/paper2610-env/bin/python`；NumPy：`2.5.4`；dtype：float64；RNG seed：`230708691`。
- 四组 `(N,d,Br,Bc,scale,causal)`：`(7,3,3,2,1,false)`、`(7,3,3,2,1/sqrt(3),false)`、`(7,3,3,2,1/sqrt(3),true)`、`(8,4,2,4,1,true)`。包含非整除边界和非方形 tile。
- 修正后的 Algorithm 1 输出 vs 稠密 stable softmax：最大绝对误差 `6.661338147750939e-16`。
- row-wise `L`：最大绝对误差 `4.440892098500626e-16`。
- `exp(S-L)` 重建 `P`：最大绝对误差 `3.3306690738754696e-16`。
- `rowsum(dO*O)` vs `rowsum(P*dP)`：最大绝对误差 `4.440892098500626e-16`。
- 修正 `D` 注记并恢复对应 scale/mask 的分块 Algorithm 2 vs 稠密链式法则：`dQ/dK/dV` 最大绝对误差 `1.3322676295501878e-15`。
- vs 独立的中心有限差分，step `1e-6`，loss 为 `sum(O*dO)`：最大绝对误差 `1.3067959492296666e-9`。
- 穷举 column contributions 的累加次序：最大 `dQ` 差异 `4.440892098500626e-16`。
- 原文 Algorithm 1 的 inverse-rescale：首次迭代输出非有限，统计 `L` 仍有限。测试 Algorithm 2 时使用正确 `O,L`，不是把错误 Algorithm 1 输出带入后声称反向通过。

可独立运行各子问题和合并，命令不联网、不编译 GPU：

```sh
/private/tmp/paper2610-env/bin/python /private/tmp/fa2-audit-v1/check.py softmax
/private/tmp/paper2610-env/bin/python /private/tmp/fa2-audit-v1/check.py backward
/private/tmp/paper2610-env/bin/python /private/tmp/fa2-audit-v1/check.py table
/private/tmp/paper2610-env/bin/python /private/tmp/fa2-audit-v1/check.py merge
```

两轮审阅记录：

1. **可读性与边界**：原式与勘误分开；保留公式符号定义、原始数据单位及派生 MFU 口径；数学自检明确标注，不重叠主 agent 的原图、图表提取和页面工作；“报告不足”和“确证错误”分开；已检查代码围栏、公式括号及全部本地证据链接。
2. **内容准确性**：以 PDF 与 TeX 核对所有定位、逆矩阵上标、Algorithm 2 的 20 个步骤和 Table 1 的 12 个原始值；以矩阵形状、明确不变量、独立有限差分和表格算术再核查。未将 PDF/TeX 视为两篇独立论文，未从纸面错误推出未查看的 GPU 实现错误；所有数字可追溯至本地 v1 或 M1。
