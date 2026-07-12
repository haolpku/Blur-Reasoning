# Proposal:面向真实世界退化的多模态推理鲁棒性(以数学推理为切入)

> **一句话**:MLLM 在"干净渲染图"上训练、却在真实世界拍摄的退化图像上被使用,存在一个被忽视的 train–test 退化 gap。本 proposal 从**数据增强**角度出发,通过对训练图像施加受控的**真实世界退化(模糊、失焦、光照、畸变、噪声、手写、版式等)**,提升模型在真实世界**推理任务**(以 MathScape 多模态数学为切入,可推广到其它真实拍摄推理任务)下的鲁棒性,并配套一个退化鲁棒性评测协议。
>
> **Framing 说明**:核心不是"修图/去模糊",而是"让 MLLM 在真实世界拍摄条件下依然能**推理**"。因此定位为 **real-world reasoning robustness**,而非单纯的 recognition/perception robustness——这也是与现有退化鲁棒性工作(多为识别/VQA)的关键区别。

---

## 1. 动机 (Motivation)

### 1.1 真实场景 vs 训练分布的错配

主流多模态数学训练数据(MathV360K / MAVIS / MultiMath / G-LLaVA 等)大量依赖:

- **程序化渲染**的几何图、函数图、图表(matplotlib / TikZ / GeoGebra 风格)
- **扫描件级别**的清晰题目

而真实使用场景(尤其是 K12 拍照答题、MathScape 这类 benchmark 的数据来源)是**手机随手拍的照片**,天然携带:

| 退化类型 | 现实成因 |
|----------|----------|
| 高斯/运动模糊、失焦 | 手抖、对焦失败、快速拍摄 |
| 光照不均、阴影、反光 | 室内灯光、纸张反光、手影 |
| 透视畸变、旋转、褶皱 | 斜着拍、纸张不平整 |
| 低分辨率、JPEG 压缩噪点 | 缩略图、聊天软件转发压缩 |
| 局部遮挡 | 手指、笔、其他纸张 |

**核心假设**:模型在数学推理上的错误,有相当一部分不是"算不对",而是"看不清"。当前 benchmark 的低分被笼统归因为 reasoning 不足,实际上 **perception 退化** 是一个被低估的独立瓶颈。

> ### ⭐ 这个假设已被 MathScape 论文自己证实(强动机)
>
> MathScape 论文的核心实验数据直接支撑了本假设:
>
> | 模型 | 真实照片 (real photo) | 干净 PDF | 差距 |
> |------|:---:|:---:|:---:|
> | LLaVA-OneVision-72B | **8.31** | **30.56** | **+22.25** |
> | GPT-4o | 42.47 | 43.89 | +1.42 |
> | (人类) | 76.96 | — | — |
>
> **仅仅把输入图像从"手机拍的真实照片"换成"干净渲染 PDF"(题目、答案完全不变),LLaVA-OneVision-72B 就从 8.31 涨到 30.56。** 这 22 分的差距完全来自图像质量/退化,而非数学难度。论文作者的明确结论是:*"strong performance on synthetic or digitally rendered images does NOT guarantee similar effectiveness on real-world tasks — this underscores the necessity of MathScape."*(arXiv:2408.07543)
>
> **含义**:MathScape 巨大的 human gap(76.96 vs GPT-4o 43.89)里,有极大一部分是 **perception 退化**造成的,不是 reasoning 不足。本 proposal 不再需要论证"gap 是否存在"——论文已证实——只需验证**数据增强能否填补它**。这把项目从"验证假设"降级为"验证方法",风险大幅降低。
>
> 开放子问题(值得作为实验重点):具体是哪些退化(透视/光照/手写/遮挡)驱动了这 22 分的 PDF-vs-photo gap?能否只针对这几个因素做合成模拟就恢复大部分损失的准确率?

### 1.2 为什么这是正当研究而非"刷榜"

这条分界线很重要,直接决定 proposal 的学术定位:

- ✅ **补真实能力 gap**:目标是"让模型在真实退化图像上依然能推理",而非贴合某个测试集的表面风格。
- ✅ **泛化可验证**:提升应在 **多个** benchmark(MathScape / MathVista / MathVerse / We-Math)上体现,而非只顶一个数。
- ✅ **退化是外生的**:增强来自通用图像退化算子,不依赖、也不泄漏任何测试集内容 → 天然规避 benchmark contamination。

> 与"用近测试分布的合成题刷分"截然不同:那种做法过拟合基准、不泛化、审稿人视为 gaming;本 proposal 是标准的鲁棒性研究范式。

---

## 2. 研究问题 (Research Questions)

- **RQ1**:当前 SOTA MLLM 在数学推理上,perception 退化贡献了多少错误?(错误归因:看不清 vs 算不对)
- **RQ2**:图像退化数据增强能否显著提升退化图像下的数学推理准确率,且**不损害**干净图像性能?
- **RQ3**:哪类退化最伤性能?增强的"课程/配比"如何设计最优?
- **RQ4**:退化增强训练出的鲁棒性能否**跨 benchmark 泛化**?能否泛化到未见过的退化类型(out-of-distribution degradation)?

---

## 3. 方法 (Method) — 数据增强为核心

### 3.1 退化算子库 (Degradation Operator Bank)

构建一个可组合、参数可控的退化 pipeline,作用于**已有干净训练图**(不改题目、不改答案,label 不变,零标注成本):

```
清晰图 x  ──►  T(x; θ)  ──►  退化图 x̃   (question / answer 保持不变)
```

算子分组(每组含强度参数):

1. **模糊类**:高斯模糊、运动模糊(随机角度/长度)、失焦(disk kernel)
2. **光照类**:亮度/对比度扰动、gamma、局部阴影 mask、高光反射斑
3. **几何类**:随机旋转(±15°)、透视变换(四角扰动)、弹性褶皱扭曲
4. **传感器类**:高斯噪声、泊松噪声、JPEG 压缩(quality 20–70)、降采样再上采样
5. **遮挡类**:随机矩形/手指状遮挡(避开关键区域,或故意小面积覆盖)

> 可直接复用成熟实现:`imagecorruptions`(ImageNet-C 的 15 种 corruption × 5 severity)、`albumentations`、`Augraphy`(专为文档/纸张退化设计,非常契合试卷场景)。

### 3.2 增强策略(几个可对比的设计)

- **策略 A — 随机在线增强**:训练时对每张图以概率 p 施加随机退化组合(最简单 baseline)。
- **策略 B — 课程增强 (curriculum)**:severity 从低到高逐步加大,先学干净后学退化。
- **策略 C — 混合批次**:每 batch 内保留一定比例干净图,防止干净性能退化(缓解 RQ2 的风险)。
- **策略 D — 一致性正则 (consistency regularization)**:同一图的 clean 与 degraded 版本,约束模型输出/中间表征一致(类似 augmentation consistency loss),迫使模型学到退化不变的 perception。

### 3.3 训练配置

- **Base 模型**:选 1–2 个开源 MLLM(如 Qwen2-VL / InternVL / LLaVA-OneVista 系),便于复现。
- **训练方式**:在现有数学 SFT 数据(如 MathV360K)上做增强后微调;可加 LoRA 降成本。
- **对照组**:
  - (a) 无增强 baseline
  - (b) 通用视觉增强(仅裁剪/翻转,不含退化)
  - (c) 本方法(退化增强)

---

## 4. 评测协议 (Evaluation) — 附带一个小 bench 贡献

### 4.1 构建退化鲁棒性评测子集

对现有 benchmark 的测试图,**程序化生成多个退化版本**(clean / blur / lowlight / rotated / noisy / mixed),形成配对评测集:

- 命名建议:**MathScape-C**(-Corruption,致敬 ImageNet-C / MMVP 的思路)
- 每道题 = 1 clean + N 退化版本,答案相同 → 可直接算"退化下掉了多少分"

> 这本身是一个可发表的小贡献:目前多模态数学 bench 几乎都用干净图,**退化鲁棒性维度基本空白**。

### 4.2 指标

- **Accuracy@clean / Accuracy@degraded**:各退化类型分别报告
- **Robustness Gap** = Acc@clean − Acc@degraded(越小越鲁棒)
- **rCE (relative Corruption Error)**:借鉴 ImageNet-C,相对 baseline 的退化误差
- **跨 bench 泛化**:在 MathVista / MathVerse / We-Math 上重复,验证非过拟合单一数据集

### 4.3 错误归因实验(支撑 RQ1,故事性强)

区分错误来源:
- **Perception probe**:让模型只做"读题/识图"(把图中数字、公式、图形转成文本),看退化下识别准确率
- **Reasoning probe**:给模型**干净的文本化题目**(绕过图像),看纯推理准确率
- 二者交叉 → 定量归因"退化下的错误有多少来自看不清 vs 算不对"

---

## 5. 预期贡献 (Contributions)

1. **实证发现**:量化 perception 退化对多模态数学推理的独立影响(RQ1),挑战"低分=推理弱"的默认叙事。
2. **方法**:一套零标注成本的退化数据增强方案 + 一致性正则,显著提升退化鲁棒性且不伤干净性能。
3. **Benchmark**:MathScape-C 等退化鲁棒性评测协议,填补现有 bench 空白。
4. **泛化验证**:跨 benchmark、跨退化类型的鲁棒性泛化证据。

---

## 6. 风险与应对 (Risks & Mitigations)

| 风险 | 应对 |
|------|------|
| 退化增强损害干净图性能 | 策略 C(混合批次)+ 一致性正则;报告 clean 性能作为约束 |
| 合成退化 ≠ 真实退化(domain gap) | 用 Augraphy 等专为真实文档设计的退化;若可能,采集少量真实拍摄图做验证集 |
| 提升被质疑为"数据变多了" | 严格控制:对照组数据量相同,只改增强算子 |
| 被质疑刷榜 | 强调跨 bench 泛化 + 退化算子外生、不接触测试内容 |
| 计算成本 | LoRA 微调 + 选中等规模开源模型;退化在线生成,不额外占存储 |

---

## 7. 相关工作定位 (Related Work,已由深度调研核查)

### 7.1 目标 benchmark
- **MathScape**(arXiv:2408.07543,GitHub: PKU-Baichuan-MLSystemLab/MathScape):1,369 题(清洗后 1,325 条公开发布,图像重编号 1–1325),源自中国 K12 试卷/作业的**真实拍摄照片**,层级结构(每题分解为带类型/知识点/解题过程标签的子问题),多维评测脚本(`judge_all/by_knowledge/by_stage/by_type.py`)。**关键:显式区别于 MathVista/MathVerse 的"数字渲染"图像。**

### 7.2 多模态数学合成数据(证明"领域内增强真的有效"——本 proposal 的正当性基石)
- **Math-LLaVA / MathV360K**(arXiv:2406.17294):40K 图(来自 24 个数据集)+ 合成 320K QA(~89% 合成);微调 LLaVA-1.5 在 MathVista minitest **+19 分**,逼近 GPT-4V。→ 证明大规模合成增强能真实提升能力。
- **MAVIS**(arXiv:2407.08739):自动引擎生成 MAVIS-Caption(558K 图-描述)+ MAVIS-Instruct(834K 带 CoT 的视觉数学题)。→ 合成图+CoT 已是成熟技术。
- **G-LLaVA / Geo170K**(arXiv:2312.11370):>170K LLM 生成的几何图-描述/QA;G-LLaVA-7B 在 MathVista 几何(GPS)上 53.4 > GPT-4V 50.5。→ 定向领域合成数据能让小模型超大模型。
- **MultiMath / MultiMath-300K**(arXiv:2409.00147):300K K12 多模态数学(图注+分步解答);MultiMath-7B 开源 SOTA。→ **与 MathScape 同为 K12,最贴近的数据范式参考。**
- **We-Math**(arXiv:2407.01284):把复合题分解为知识概念子问题,提出 **"Rote Memorization"(死记硬背)指标**——区分"真学会"还是"记住了"。→ 本 proposal 可借它证明增强带来的是真能力而非记忆。

> ⚠️ **共同局限(也是本 proposal 的机会)**:上述工作几乎全部生成**干净/渲染图**——恰恰是 MathScape 证明"不迁移"的分布。**退化/真实拍摄分布的合成增强基本空白。**

### 7.3 Benchmark 污染与检测(界定"增强"与"刷榜"的红线)
- **Rephrased Samples / "Teaching to the Test"**(arXiv:2311.04850):13B 模型在改写(paraphrase/翻译)过的测试数据上训练,可**无真实能力**地刷到 GPT-4 水平;这类改写**能绕过 n-gram/字符串去污染**;且 GPT-3.5/4 生成的合成数据本身被发现含污染。→ **"近测试分布"数据是已被记录的污染向量,不是良性增强。**
- **Min-K% Prob**(arXiv:2310.16789):黑盒检测文本是否在训练集中(未见样本含少量低概率离群 token),无需预训练语料知识或参考模型,已用于下游 benchmark 污染检测。→ **刷榜越来越可被检出。**(注:2024 年后续工作对 MIA 实际效力有争议)

### 7.4 视觉退化鲁棒性(方法工具箱)
- ImageNet-C / `imagecorruptions`(15 种 corruption × 5 severity)、`albumentations`、**`Augraphy`**(专为真实文档/纸张退化设计,最贴试卷场景)。→ 现成退化算子实现。

> ✅ **调研结论**:没有任何来源直接实验过"在 MathScape 式退化/真实拍摄合成数据上训练能否迁移到 MathScape"——**这是一个开放且未被占据的切口**。你的 proposal 正好落在 7.2(证明有效)、7.3(避开红线)、7.4(有工具)三者的交集空白处。

---

## 7.5 查新专项:"合成退化增强" 这个方向的占据地图(已由第二轮深度调研核查)

> 一句话:**"合成退化增强"作为技术是成熟老方法(不能当创新点);但"用它做 MLLM 的 instruction tuning 去修复真实退化下的推理能力"是明确空白(这才是创新点)。**

| 层次 | 代表工作 | 占据程度 | 对本 proposal 的意义 |
|------|----------|:---:|------|
| **通用分类/检测** | ImageNet-C/P (arXiv:1903.12261)、AugMix (1912.02781)、RandAugment (1909.13719)、**DeepAugment / Many Faces of Robustness** (2006.16241) | 🔴 完全占满 | 提供"合成退化增强有效"的**基础背书**;DeepAugment 明确证明收益**能迁移到真实分布偏移**(不只合成 benchmark)。引用它作为方法合理性依据,**不作为创新点**。 |
| **模糊专项(识别鲁棒,非复原)** | Cho et al. IEEE Access 2022(合成运动模糊核训 blur-robust 检测器)、Sayed & Brostow CVPR-W 2021(合成+真实模糊评测)、Brooks & Barron CVPR 2019(模糊合成本身) | 🔴 占满 | 证明"合成模糊→识别鲁棒"在**检测/小模型**上已被做透。你的差异必须在 **MLLM + 推理任务**,不能停在检测。 |
| **OCR/文档退化** | STRAug (2108.06949,36 种退化增强,真实文本测试集涨点)、Augraphy (2208.14558)、DocCreator (Neurocomputing 2017,7 种退化含自适应模糊) | 🔴 占满 | "合成退化文档→鲁棒 OCR"成熟。**Augraphy/DocCreator 可直接拿来当你的退化算子**,但"造工具"本身没新意。 |
| **MLLM/VLM 退化鲁棒性(最相关)** | **OCR-Robust**(arXiv:2506.26041 / 更正 2606.26041,812 样本含数学,合成扰动×3 severity) | 🟡 **仅诊断,方法侧空白** | OCR-Robust 证明了"**VLM 在合成退化下会掉,且干净准确率无法预测鲁棒性**",但**纯评测,没有提出用合成退化训练去修复**。→ **你的训练方法正好补这个洞。** |
| **真实相机退化模拟(缩小 rendered-vs-real gap)** | Real-ESRGAN (ICCVW 2021)、BSRAW (WACV 2024)、Degradation-Independent ISP (CVPR 2024) — 均在**超分/复原**领域 | 🟢 **识别/MLLM 侧基本空白** | 用合成模拟真实拍摄退化的技术存在于**图像复原**领域,但**没人把它用于缩小 MLLM 识别/推理的 rendered-vs-real gap**——正是 MathScape 那 22 分 gap 的方向。 |

### 空白定位(调研 open question 原话对应你的 idea)
> *"Is there any published work that FINE-TUNES an MLLM on synthetically degraded images and shows improved recognition on real degraded inputs — or is this genuinely open whitespace?"*

调研未找到此类工作 → **训练方法侧 = 空白。** 你的创新点应表述为:

> **不是**"提出合成退化增强"(老),**而是**"首次将合成退化增强用于 MLLM 的视觉指令微调,并证明它能填补 MathScape 实测的真实-渲染感知 gap,在多个数学 benchmark 上泛化"。

### ✅ 撞车核实结论(已逐一核查完成 2026-07-10)

| 工作 | 性质 | 训练 or 评测 | 撞车风险 | 对本 proposal 的用途 |
|------|------|:---:|:---:|------|
| **MLLM-IC** (ICCV 2025) | corruption 鲁棒性 benchmark(三级退化 taxonomy) | 仅评测 | 🟢 NONE | 可作评测集 + related work |
| **R-Bench** (arXiv:2410.05474) | 真实世界 corruption 鲁棒 benchmark(含真实拍摄图) | 仅评测 | 🟢 NONE | **天然评测集**;它"测出" gap,你"解决" gap |
| **MMCBench** (arXiv:2401.11943) | 跨模态生成一致性 benchmark | 仅评测 | 🟢 NONE | 弱相关,引用即可 |
| **BenchLMM** (arXiv:2312.02896) | 跨风格视觉能力 benchmark(+training-free prompting) | 仅评测 | 🟢 NONE | 弱相关,引用即可 |
| **Robust-U1** (ICML 2026,arXiv:2606.08063)⚠️ | **训练方法**:MLLM 自恢复退化图 | **训练 SFT+RL** | 🟡 **PARTIAL** | **最近邻 / 强 baseline,必须正面对比** |

**结论:核心 whitespace 未被占据,可立项。** 前四个均为纯评测 benchmark——只证明"MLLM 在退化下掉分",无一用退化数据训练;它们是你的**评测集与 related work**,不是威胁。

**唯一需正面对待的是 Robust-U1(ICML 2026)**,它确实训练 MLLM 处理退化图,但与本 proposal 有三个决定性区别,whitespace 依然开放:

1. **机制不同(最关键)**:Robust-U1 用**显式图像自恢复/复原模块**(SSIM 像素 + CLIP 语义重建奖励,对"corrupted + recovered"双图推理)。本 proposal 是**纯退化数据增强微调**——无复原模块、无重建奖励、无恢复分支。两条不同技术路线。
2. **任务不同**:Robust-U1 做通用 VQA/understanding;本 proposal 做**真实世界推理,特指多模态数学(MathScape)+ rendered-vs-real gap**。它不使用任何数学 benchmark。
3. **命题不同**:它问"模型能否重建丢失像素";本 proposal 问"训练时见过合成退化,能否让**推理**对真实退化鲁棒"。

> **战略含义**:Robust-U1 反而是资产——它给出现成的**强 baseline 和对比故事**:"本方法更简单(纯增强、无复原模块、更省算力),且专攻真实世界数学推理"。审稿人若问"为何不像 Robust-U1 那样复原图像?",用**消融实验**(纯退化增强 vs. 复原模块 baseline)回答。
>
> **待办**:Robust-U1 的 SFT 阶段很可能就在合成退化图上训练——若如此,**数据生成层面有重叠**,你的 novelty 须牢牢落在(a)无复原机制 + (b)数学推理目标 这两点。最终定稿前应精读 arXiv:2606.08063 的 method/experiments 章节。

> ### ⚠️ 一个不能忽视的反面证据(来自调研 caveat)
> 合成退化增强的迁移性**并非普适**:DeepAugment 论文自己承认"no evaluated method consistently improves robustness",且 Taori et al. 2020 曾发现合成干预**迁移到真实退化较差**。→ 因此本 proposal 的 **MVP(第 8 节)必须先验证"退化增强训练真的能恢复 MathScape 的 22 分 gap"**,不能默认成立。这既是风险,也是论文最有价值的实证结论。

---

## 8. 最小可行验证 (MVP,先做这个)

在写大 pipeline 前,用 1–2 天做一个 **信号验证实验**:

1. 取一个开源 MLLM + MathScape(或 MathVista)的一小批题
2. 用 `imagecorruptions` 生成 blur / lowlight / jpeg 三种退化版本
3. 直接测退化前后的 accuracy gap

**如果 gap 明显(比如掉 5–15 个点)→ 说明真实存在这个瓶颈,proposal 立即成立,值得投入。**
如果 gap 很小 → 说明现代 MLLM 已经较鲁棒,需重新审视切口(转向更极端退化或真实拍摄图)。

---

## 9. 可执行实验计划 (Experiment Plan)

> 目标:把第 3–4 节的方法/评测,落成可排期、可跑、有明确决策门的实验序列。全程用 **LoRA + 中等规模开源模型**控制成本;每个阶段设 **Go/No-Go 门**,坏消息尽早暴露。

### 9.0 固定实验设定(全程统一)

| 维度 | 选择 | 理由 |
|------|------|------|
| **Base 模型** | 主:**Qwen2.5-VL-7B**;副(泛化验证):**InternVL2.5-8B** | 开源、强 baseline、社区复现多;双模型证明结论非单模型偶然 |
| **训练数据** | **两者皆可(备选,见下方对照)**:① **Zebra-CoT**(多模态推理 CoT,当前仓库已接入);② **MathV360K**(数学 SFT,已验证 +19 分) | 两条数据线互补,可分别或联合使用 |
| **微调方式** | LoRA(r=64, α=16)+ 冻结 vision encoder 的对照支线 | 降算力;冻结/解冻 vision encoder 本身是一个消融点 |
| **退化实现** | `imagecorruptions`(ImageNet-C 15 类)+ `Augraphy`(文档退化) | 前者标准可比,后者贴试卷真实退化 |
| **评测 harness** | 复用 MathScape 官方 `judge_*.py` + VLMEvalKit | 保证与论文数字可比,不自造评测口径 |
| **随机性** | 固定 3 个 seed,报告 mean±std | 退化增强方差大,单 seed 不可信 |

#### 训练数据两条备选线对照

| | **① Zebra-CoT** | **② MathV360K** |
|---|---|---|
| 内容 | 多模态推理链(CoT)数据,题型更广 | 专注数学,K12,40K 图 + 320K 合成 QA |
| 与 MathScape 对齐度 | 中(推理能力泛化,非纯数学) | 高(同为数学域) |
| 已验证收益 | — | MathVista **+19 分**(Math-LLaVA) |
| 仓库现状 | **已接入**(`setup_zebra_training.sh` + LLaMA-Factory config) | 未接入,需新增数据转换 |
| 定位 | **主线**:验证"退化增强提升真实世界**推理**鲁棒性"(与 real-world reasoning framing 一致) | **对照/补强线**:验证结论在纯数学 SFT 数据上同样成立,并对齐 7.2 已有工作 |
| 建议用法 | 先用 Zebra-CoT 跑通阶段一/二(复用现有 pipeline,启动快) | 阶段三泛化时加入,证明"退化增强有效"不依赖特定数据集 → 强化 RQ4 |

> **策略**:两条线不是二选一,而是**先 Zebra-CoT 快速验证(复用仓库现成 pipeline),再用 MathV360K 做数据集鲁棒性交叉验证**。若两个数据集上退化增强都涨,论证"方法与数据无关"的说服力最强(直接回应"是不是只对某数据集有效"的质疑)。

### 9.1 阶段一:确证 gap 与错误归因(对应 RQ1)—— 1 周

**E1.1 退化敏感度扫描**:对 MathScape / MathVista 测试图,用退化算子库逐类 × 3 severity 生成退化版,测 base 模型 Acc,画"退化类型 × severity → 掉分"热力图。
→ **产出**:哪几类退化最伤(预期:模糊、低光、透视);锁定后续增强重点。

**E1.2 错误归因(perception vs reasoning)**:实现第 4.3 的双 probe。
→ **产出**:退化下的错误中"看不清"占比的定量数字 —— 这是全文最有说服力的 motivating figure。

> **Go/No-Go 门 ①**:若 E1.1 掉分 < 3 分(模型已鲁棒),或 E1.2 显示错误几乎全来自 reasoning(非 perception),则核心假设不成立 → 停,重新 framing。**否则继续。**

### 9.2 阶段二:主方法与消融(对应 RQ2/RQ3)—— 2–3 周

统一在 MathV360K 上微调,只改增强策略,**数据量严格相同**(回应"数据变多了"的质疑):

| 组 | 训练增强 | 目的 |
|----|----------|------|
| **G0** | 无增强(原始 MathV360K SFT) | 下限 baseline |
| **G1** | 通用增强(裁剪/翻转,无退化) | 排除"增强本身"的干扰变量 |
| **G2** | 退化增强·策略A(随机在线) | 主方法最简版 |
| **G3** | 退化增强·策略C(混合批次,保干净图比例) | 防干净性能退化 |
| **G4** | G3 + 策略D(一致性正则) | 完整方法 |
| **G5(关键对比)** | **复原式 baseline**(Robust-U1 思路:加图像重建目标) | 证明"纯增强 ≈ 或 > 复原模块,但更简单" |

**消融子实验**:
- **A1 退化配比**:干净:退化 = 100:0 / 70:30 / 50:50 / 30:70,找最优点(对应 RQ3)。
- **A2 课程 vs 随机**:策略B vs 策略A。
- **A3 vision encoder 冻结 vs 解冻**:退化鲁棒性主要来自哪一部分。
- **A4 增强来源**:`imagecorruptions` only vs +`Augraphy`,验证真实文档退化是否额外有用。

> **Go/No-Go 门 ②**:G4 在退化测试集上显著 > G0/G1(Δ≥ 有意义幅度且 seed 间稳定),**且** G4 干净图性能不低于 G0 超过 ~1 分。达标才进入泛化验证。

### 9.3 阶段三:泛化与 OOD(对应 RQ4)—— 1–2 周

- **G-1 跨 benchmark**:在 MathScape / MathVista / MathVerse / We-Math 上评 G0 vs G4,证明非过拟合单一数据集。
- **G-2 OOD 退化**:训练时**留出**某类退化(如运动模糊),只在测试用 → 验证对未见退化的泛化。
- **G-3 真实退化验证(最强证据)**:MathScape 本身就是真实拍摄图 → **直接用 MathScape 真实照片 vs 其 PDF 版**测 G0 vs G4,看能否缩小那 **22 分 gap**。这是全文的 money experiment。
- **G-4 We-Math 死记硬背指标**:用 We-Math 的 Rote Memorization 指标验证提升来自真能力而非记忆。

### 9.4 阶段四:MathScape-C 评测集 + 收尾 —— 1 周

- 固化第 4.1 的 **MathScape-C**(clean + N 退化版本配对),开源。
- 汇总所有指标(Acc@clean/degraded、Robustness Gap、rCE、跨 bench 表)。
- 补齐与 Robust-U1 / OCR-Robust 的正面对比表述。

### 9.5 主结果表(目标形态,占位)

| Model / Method | MathScape(real) | MathScape(PDF) | MathScape-C avg | MathVista | Clean 均值 | Robustness Gap↓ |
|----------------|:---:|:---:|:---:|:---:|:---:|:---:|
| Base (zero-shot) | … | … | … | … | … | … |
| G0 SFT | … | … | … | … | … | … |
| G1 通用增强 | … | … | … | … | … | … |
| G5 复原式 (Robust-U1 类) | … | … | … | … | … | … |
| **G4 本方法(完整)** | **↑** | … | **↑** | **↑** | ≈G0 | **↓** |

### 9.6 算力与排期

- **总排期**:约 5–7 周(单人 + 少量 GPU)。
- **算力**:LoRA 7–8B,单机 4×A100(或等效)可覆盖;退化在线生成、不占额外存储。
- **最大成本项**:阶段二的 6 组 × 3 seed × 消融 ≈ 20–30 次微调 —— 用 LoRA + 小 epoch 控制。

### 9.7 关键风险与实验层面的对冲

| 风险 | 实验层面对冲 |
|------|------|
| 纯增强打不过复原式(G5>G4) | 那就转向"增强 + 轻量复原"的混合,故事改为"最优配方";仍有贡献 |
| 合成退化不迁移真实(Taori 警告) | G-3 用 MathScape 真实图直接验证,不靠合成测试图自证 |
| 干净性能下降 | A1 配比 + 策略C/D 已内建对冲;干净性能作为硬约束报告 |
| 与 Robust-U1 撞车加剧 | novelty 锁定"无复原 + 数学推理 + rendered-vs-real gap";G5 即为其代理 baseline |

---

*Created: 2026-07-10 · Status: Draft (experiment plan added) · Next step: 跑 9.1 阶段一 E1.1/E1.2,过 Go/No-Go 门 ① 再投入阶段二*
