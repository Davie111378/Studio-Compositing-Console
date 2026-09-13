# 生产实习开题报告

**课 题 题 目**：面向真实感图像合成的图像抠取与光影一致性融合方法研究——多模态交互式 Agent 系统设计与实现

**学 科 方 向**：计算机科学与技术（计算机视觉与智能交互方向）

**汇 报 人**：＿＿＿＿＿　**学号**：＿＿＿＿＿＿＿＿　**学院**：＿＿＿＿＿＿＿＿　**专业**：＿＿＿＿＿＿

**指导教师**：＿＿＿＿＿　**汇报日期**：2026 年 9 月 8 日

> 本报告章节与开题答辩 PPT（共 18 页）一一对应；文中 [n] 为参考文献编号，见文末及《参考文献_GB_T7714》文件。

---

## 一、选题背景与意义

### 1.1 背景

在电商、影视与短视频内容生产中，大量图像需要完成"主体分离—场景替换—氛围重塑"的加工流程，其中图像抠取（matting）与图像合成（composition）是底层核心环节。传统工作流（如 Photoshop）中，发丝级抠图与光影匹配高度依赖资深设计师，单张图处理可达数小时，专业门槛高、难以批量自动化；现有在线工具大多仅提供"一键去背景"，无法完成重打光、阴影生成等光影重塑，更无法通过自然交互完成复杂场景合成。

近年来深度学习推动了该领域的范式转变：扩散模型（DDPM[10]、LDM[11]）使生成图像质量逼近真实照片；交互式分割（SAM[6]、SAM 2[7]）与自然图像抠图（BiRefNet[5]）大幅降低了抠取门槛；指令式编辑（InstructPix2Pix[25]）让"用语言改图"成为可能。但当前研究与工具仍存在两方面短板：其一，抠取、背景生成、重打光、阴影、和谐化各环节以孤立模块为主，合成结果普遍存在"贴纸感"；其二，交互式编辑在多轮连续操作下性能显著退化[27]，且语音这一更自然的交互模态尚未被系统性地引入图像合成流程[29]。

### 1.2 意义

- **应用价值**：为电商虚拟试衣成片、影视后期合成与创意内容生产提供"一句话出图"的自动化工具。本课题与实验室已有基于 Diffusion 的虚拟试衣研究形成上下游闭环：虚拟试衣输出的模特图像可直接进入本系统的抠取—融合流水线，产出可直接交付的成片。
- **学术价值**：其一，探索光照一致性约束下"抠取—重打光—接触阴影—和谐化"的联合建模方法，回应合成领域公认的难点[20][23][24]；其二，探索语音与空间指代相结合的多模态交互式合成范式，并针对多轮编辑退化问题[27]提出基于执行计划的稳定交互机制。
- **工程价值**：沉淀一套可复用的"Agent + 图像工具链"框架，其中的工具封装、数据构造与轻量训练策略均可迁移至实验室后续课题。

## 二、国内外研究现状

### 2.1 图像抠取

图像抠取的目标是估计每个像素的前景不透明度 α（I = α·F + (1−α)·B），发丝、半透明边缘的还原依赖高精度 alpha 估计。现有方法可分为三类：（1）Trimap 引导方法，以 DIM[1] 的编码—解码框架与 Composition-1k 基准为开端，ViTMatte[4] 引入预训练 Vision Transformer 进一步提升精度；（2）无引导方法，以 MatteFormer、BiRefNet[5] 为代表，通过双边参考机制兼顾全局语义与局部细节，BRIA 基于该架构训练的 RMBG-2.0[9] 已达到商用精度；（3）交互/提示引导方法，SAM[6]、SAM 2[7] 支持点、框、文本提示的通用分割，Matte Anything[8] 将其与抠图头结合实现交互式抠取。在 Composition-1k 基准上，最优方法的 SAD 指标已从 DIM 的 59.6 演进至 16.89，人像与商品类目标已达可用水平；但透明/半透明目标（玻璃、薄纱）误差仍高出发丝场景约一个数量级，视频场景的时序稳定性亦依赖专门模型[3]。

### 2.2 图像合成与和谐化

**传统和谐化**方面，DoveNet[13] 构建了首个大规模基准 iHarmony4（约 7.3 万对图像），通过域验证实现前景与背景的外观统一；Harmonizer[14] 将神经网络与白盒滤波器结合，兼顾效果与可解释性。近期工作向更复杂场景扩展：StructFuse[17] 提出可学习的多结构线索（边缘、轮廓、深度）自适应融合；H-CDM[18] 面向噪声与分辨率不一致的退化场景构建 D-iHarmony4 数据集；国内学者提出的 DGFNet[19] 采用动态图融合策略，在 iHarmony4 与 ccHarmony 上取得先进结果。

**生成式合成**方面，Paint-by-Example[15] 以参考图补丁驱动扩散模型完成物体级合成，AnyDoor[16] 实现零样本物体级定制，ControlNet[12] 为扩散模型提供深度、边缘等结构条件控制。

**光影建模**方面，IC-Light[20] 在潜空间施加光照一致性约束，支持文本条件与背景条件的重打光；SwitchLight[21] 基于 Cook-Torrance 反射模型进行物理驱动的人像重打光；DiLightNet[22] 实现细粒度光照控制。阴影生成方面，CoShadow[23] 首次研究多物体联合投影生成；物理推理阴影生成[24] 引入单目几何与光源方向估计，使阴影区域 RMSE 相对下降 23%。总体来看，孤立的外观和谐化已较成熟，"光照 + 阴影 + 几何"联合一致的合成是当前公认的难点。

### 2.3 指令式编辑与交互范式

指令式编辑由 InstructPix2Pix[25] 与数据集 MagicBrush[26] 开创，此后开源模型（Qwen-Image、Step1X-Edit、FLUX 系列）能力快速逼近闭源系统。在评测层面，EdiVal-Agent[27]（ICLR 2026）首次系统评测了连续多轮编辑，结果显示所有模型性能随轮次断崖式下降——开源最强的 Qwen-Image-Edit 指令遵循得分从第一轮的 72.90 跌至第三轮的 22.55；MT-EditFlow[28] 采用强化学习微调仅能小幅缓解。在语音交互方面，DOWIS[29] 构建了语音指令遵循数据集，并证实语音输入一致劣于文本输入、低资源语言差距更大；TalkSketchD[30] 证实将自发语音与视觉指代（草图）联合输入多模态大模型可显著提升生成结果与设计者意图的对齐度。目前，语音驱动的图像合成编辑系统与面向中文的交互评测尚属空白。

## 三、现存问题分析

综合上述现状，本课题聚焦三类现存问题：

1. **透明/半透明目标抠取精度骤降**：玻璃、薄纱、婚纱等半透明区域在现有方法中 alpha 估计易出现灰边、断裂，与发丝等不透明细节存在数量级差距；
2. **合成结果"贴纸感"**：前景光照方向、接触阴影、色彩风格与新背景不一致，现有方法将重打光与阴影生成割裂研究，缺乏光照一致性约束下的联合建模；
3. **多轮与语音交互下的编辑退化**：连续多轮编辑性能断崖式下降[27]，语音指令可靠性不足[29]，且缺乏面向中文的语音驱动合成系统与可复现、可回溯的执行机制。

## 四、研究目标与研究内容

### 4.1 总体目标

构建语音驱动的多模态图像合成 Agent：用户"说一句话"，系统自动完成高精度抠取 → 背景生成 → 光影一致性融合 → 结果自评修正 → 语音反馈的完整闭环，端到端响应不超过 10 秒。

### 4.2 具体目标（可量化）

1. **抠取精度**：在自建含透明目标测试集上，域微调后 SAD / Grad 指标相对基线下降 ≥10%；
2. **合成质量**：主观"贴纸感"评分显著降低，用户研究（n≥10）中优于"直接合成"基线；
3. **交互稳定性**：连续 5 轮以上对话编辑不崩溃，工具调用计划准确率 ≥95%，语音识别—执行—反馈链路稳定可复现。

### 4.3 研究内容

1. **高精度图像抠取**：以 BiRefNet[5] 承担自动抠取、SAM 2[7] 承担交互式分割，结合 alpha 精修与面向透明目标的域微调；
2. **背景生成**：SDXL / FLUX 底座 + ControlNet[12] 结构控制 + 风格 LoRA 轻量训练；
3. **光影一致性融合（重点）**：环境光估计 → 前景重打光 → 接触阴影与反射生成 → 色彩/噪点/景深匹配的联合流水线；
4. **多模态 Agent 框架**：规划器（现成大模型函数调用）→ 执行计划 DAG → VLM 自评与重规划，支持语音与点击联合指代。

## 五、研究方法与技术路线

系统分四层（见 PPT 图 4）：

- **用户交互层**：语音指令、空间指代（点击/框选）、图像与风格参考输入；
- **Agent 智能体层**：流式中文语音识别 → 多模态规划器输出"执行计划 DAG"（可编辑、可回放、可从任意节点重跑）→ 工具调度 → VLM Critic 结果自评（不达标自动重规划）→ 语音合成反馈与解释；
- **图像处理工具层（课题核心）**：抠取、背景生成、光影一致性融合、效果增强四大工具；
- **数据与轻量训练**：领域数据集上的抠取模型域微调、风格/任务 LoRA，采用"短任务 + 断点续训"策略适配外部租用算力，训后权重热更新至推理服务。

## 六、关键技术

### 6.1 光影一致性融合（重点）

方法路线为：背景环境光估计（球谐/方向光）→ 前景重打光（借鉴 IC-Light[20] 的潜空间光照一致性思想）→ 接触阴影与反射生成（借鉴物理推理阴影[24] 的光源方向估计与几何锚定）→ 色彩/噪点/景深匹配。同时引入基于光照/法线估计器的前后景光照一致性评分，作为 Agent Critic 的自动验收指标。实验将与"直接合成""仅和谐化[13]""仅重打光[20]"三类基线对比，采用客观指标与用户研究主观评分双重验证。

### 6.2 高精度图像抠取

采用 BiRefNet[5]（发丝级无引导抠取）与 SAM 2[7]（交互指代）的组合，粗分割掩码经 trimap 构造后由轻量抠图头细化半透明边缘；针对透明目标进行域微调。训练对按合成公式 I = αF + (1−α)B 自动构造；训练策略为单次 ≤24 小时、每 30 分钟保存检查点、支持断点续训与抢占式实例，先在实验室服务器以 1/10 数据冒烟验证再全量训练。评价指标采用 Composition-1k 协议[1]的 SAD、MSE、Grad、Conn。

### 6.3 多模态 Agent 构建

抠取、背景生成、重打光、阴影、效果增强等能力封装为 8 个原子工具并定义统一 JSON Schema；规划器基于成熟大模型的函数调用能力输出执行计划 DAG，不直接生成像素，保证可编辑、可回放、可条件重跑；VLM Critic 对每步结果量化打分并触发重规划。多模态交互上，语音链路（流式 ASR + 端点检测 + TTS）全本地部署，语音描述与点击/框选坐标联合消解指代[30]；通过编辑历史树支持"换回上一步背景、但保留当前光照"的条件回滚，针对性缓解多轮退化[27]。可靠性方面，采用 JSON Schema 强校验与示例约束限制规划器自由度，保留程序化降级方案，并以草稿/精修两级出图控制端到端延迟。

## 七、创新点

1. **光影一致性的联合合成框架**：将"抠取—重打光—接触阴影—和谐化"组织为光照一致性约束下的统一流水线，并引入量化一致性评分，区别于现有将各环节孤立处理的方法[20][23]；
2. **语音 + 空间指代的多模态交互范式**：面向中文场景构建语音驱动的图像合成 Agent，语音与点击/框选联合消解指代，突破纯文本指令编辑的交互局限[25][29]；
3. **基于执行计划 DAG 的稳定多轮机制**：以可编辑、可回放、可条件重跑的执行计划为中心管理编辑状态，缓解多轮编辑性能退化[27]，实现细粒度条件回滚；
4. **训练—推理解耦的轻量训练方案**：短任务化、断点续训与域微调相结合的工程方案，在不稳定外部算力条件下稳定产出自有风格 LoRA 与域适配抠图模型。

## 八、预期成果

1. **可运行系统**：语音交互式图像合成 Agent 网站，支持语音指令、点击指代、多轮修正与结果导出；语音指令 3 秒内给出执行计划、10 秒内输出合成结果；
2. **量化实验结果**：抠取域微调前后 SAD / Grad 对比表；光影一致性融合与多组基线的主观 + 客观对比；用户研究（n≥10）合成自然度优于基线；
3. **软件成果**：图像工具链与 Agent 框架源码、模型权重（含自有风格 LoRA）、部署文档；
4. **文档成果**：技术报告 1 份（含完整实验记录与失败案例分析）、3 分钟演示视频，并可支撑后续学术竞赛与论文投稿。

## 九、进度安排（2026-09-07 — 2026-10-10）

| 阶段 | 时间 | 内容 | 节点 |
|---|---|---|---|
| 1 | 09-07 ~ 09-13 | 文献调研与方案细化；抠取/合成基线流水线搭建 | 09-13 基线跑通 |
| 2 | 09-09 ~ 09-16 | 抠取、背景生成、效果增强工具链联调 | — |
| 3 | 09-14 ~ 09-22 | Agent 工具封装、规划器接入、执行计划可视化 | 09-22 Agent 语音链路贯通 |
| 4 | 09-15 ~ 09-26 | 风格/任务 LoRA 训练（外部算力，断点续训） | — |
| 5 | 09-18 ~ 09-29 | 光影一致性融合模块实现与调优 | — |
| 6 | 09-20 ~ 09-30 | 语音交互与 Web 界面、多轮回滚 | 09-30 系统封版 |
| 7 | 09-25 ~ 10-09 | 实验评测、量化对比、用户研究（10-01~10-07 假期） | — |
| 8 | 10-01 ~ 10-10 | 报告撰写、演示视频、成果整理 | 10-10 交付 |

## 十、可行性分析

- **技术可行**：各模块均有成熟开源模型支撑（BiRefNet[5]、SAM 2[7]、ControlNet[12]、IC-Light[20] 等），均为社区长期维护项目，基线可快速复现；
- **数据可行**：实验室已有标注数据可直接用于域微调；公开基准（Composition-1k[1]、iHarmony4[13]）可验证方法有效性，无需新增大规模人工标注；
- **算力可行**：实验室深度学习服务器（双 Xeon 8352V、4×RTX A6000 48GB、512GB 内存、1TB SSD + 2×18TB HDD）满足推理与小规模训练；外部训练采用"短任务 + 断点续训 + 抢占式实例"策略，成本可控；
- **团队可行**：三人分工明确（Agent 与后端、图像算法与训练、前端与语音交互），接口先行、并行开发，设两周日程对齐与关键节点封版机制；
- **风险与对策**：生成效果不稳 → 保留程序化降级方案（色彩匹配 + 程序化光效）；外部算力不可用 → 实验室服务器夜间兜底训练；进度风险 → 以最小可行系统（MVP）为底线，功能分级取舍。

---

## 参考文献

（完整 30 条见 PPT 第 17–18 页及《参考文献_GB_T7714.md》，此处列正文引用的全部条目）

[1] XU N, PRICE B, COHEN S, et al. Deep image matting[C]//CVPR. Honolulu: IEEE, 2017: 2970-2979.
[3] LIN S, YANG L, SALEEMI I, et al. Robust high-resolution video matting with temporal guidance[C]//WACV. Waikoloa: IEEE, 2022: 238-247.
[4] YAO J, WANG X, YANG S, et al. ViTMatte: boosting image matting with pretrained plain vision transformers[J]. Information Fusion, 2024, 103: 102091.
[5] ZHENG P, GAO D, FAN D P, et al. Bilateral reference for high-resolution dichotomous image segmentation[J]. CAAI Artificial Intelligence Research, 2024, 3(2): 26-36.
[6] KIRILLOV A, MINTUN E, RAVI N, et al. Segment anything[C]//ICCV. Paris: IEEE, 2023: 4015-4026.
[7] RAVI N, GABEUR V, HU Y T, et al. SAM 2: segment anything in images and videos[EB/OL]. arXiv:2408.00714, 2024.
[8] Matte anything: interactive natural image matting with segment anything model[EB/OL]. arXiv:2306.04121, 2023.
[9] BRIA AI. RMBG-2.0 background removal model[EB/OL]. (2024)[2026-09-07]. https://huggingface.co/briaai/RMBG-2.0.
[10] HO J, JAIN A, ABBEEL P. Denoising diffusion probabilistic models[C]//NeurIPS. 2020: 6840-6851.
[11] ROMBACH R, BLATTMANN A, LORENZ D, et al. High-resolution image synthesis with latent diffusion models[C]//CVPR. New Orleans: IEEE, 2022: 10684-10695.
[12] ZHANG L, RAO A, AGRAWALA M. Adding conditional control to text-to-image diffusion models[C]//ICCV. Paris: IEEE, 2023: 3836-3847.
[13] CONG W, ZHANG J, NIU L, et al. DoveNet: deep image harmonization via domain verification[C]//CVPR. Seattle: IEEE, 2020: 8394-8403.
[14] KE Z, SUN C, ZHU L, et al. Harmonizer: learning to perform white-box image and video harmonization[C]//ECCV. Tel Aviv: Springer, 2022: 324-341.
[15] YANG B, GU S, ZHANG B, et al. Paint by example: exemplar-based image editing with diffusion models[C]//CVPR. Vancouver: IEEE, 2023: 18381-18391.
[16] CHEN X, HUANG L, LIU Y, et al. AnyDoor: zero-shot object-level image customization[C]//ECCV. Milan: Springer, 2024.
[17] AHMED W, DIEPEVEEN D, SOHEL F. StructFuse: harmonizing multiple structural cues for diffusion-driven image compositing[J]. Pattern Recognition, 2026, 180: 114609.
[18] LI G, ZHAO B, LI X. Image harmonization in complex degradation scenes[J]. Pattern Recognition, 2026, 171: 112227.
[19] 程显贺, 孟祥瑞, 纪昂霄, 等. 动态图融合的图像和谐化方法[J]. 计算机工程与应用, 2026, 62(5): 293-301.
[20] ZHANG L. IC-Light: imposing consistent light[EB/OL]. (2024)[2026-09-07]. https://github.com/lllyasviel/IC-Light.
[21] KIM H, JANG M, YOON W, et al. SwitchLight: co-design of physics-driven architecture and pre-training framework for human portrait relighting[C]//CVPR. Seattle: IEEE, 2024.
[22] ZENG C, DONG Y, PEERS P, et al. DiLightNet: fine-grained lighting control for diffusion-based image generation[C]//ECCV. Milan: Springer, 2024.
[23] AHMED W, DIEPEVEEN D, SOHEL F. CoShadow: multi-object shadow generation for image compositing via diffusion model[EB/OL]. arXiv:2603.02743, 2026.
[24] HU S, XU J, DAVE A, et al. Embedding physical reasoning into diffusion-based shadow generation[EB/OL]. arXiv:2512.06174, 2026.
[25] BROOKS T, HOLYNSKI A, EFROS A A. InstructPix2Pix: learning to follow image editing instructions[C]//CVPR. Vancouver: IEEE, 2023: 18392-18402.
[26] ZHANG K, MO L, CHEN W, et al. MagicBrush: a manually annotated dataset for instruction-guided image editing[C]//NeurIPS. New Orleans, 2023.
[27] EdiVal-agent: an object-centric framework for automated, fine-grained evaluation of multi-turn editing[C]//ICLR. 2026.
[28] MT-EditFlow: reinforcement learning for multi-turn image editing with flow matching[EB/OL]. arXiv:2606.01985, 2026.
[29] ZÜFLE M, PAPI S, RETKOWSKI F, et al. Do what I say: a spoken prompt dataset for instruction-following[EB/OL]. arXiv:2603.09881, 2026.
[30] SHI W, HERREMANS D, CHOO K T W. When drawing is not enough: exploring spontaneous speech with sketch for intent alignment in multimodal LLMs[EB/OL]. arXiv:2604.11964, 2026.
