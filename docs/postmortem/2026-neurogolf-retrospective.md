# The 2026 NeuroGolf Championship 赛后复盘

更新日期：2026-07-16

## 摘要

团队 `Blacklions.` 在赛后首日公开榜单快照中取得 `7420.93`，排名 `249/3061`。冠军 `Kaggle Agent` 为 `8314.03`，分差 `893.10`。我们的最后一次最高提交是 ref `54736568`，public/private 均为 `7420.93`。

这次比赛证明了三件事：

1. NeuroGolf 的主问题是表示搜索，不是常规训练；免费 MACs 应换取更少的参数和中间内存。
2. 本地正确只是准入条件，隐藏分布、运行时和父包一致性决定候选能否进入冠军包。
3. 400 个独立任务需要任务级状态机、自动回滚和知识沉淀。提交次数不能替代这套系统。

结果证据见 [`final-results-20260716.json`](../evidence/final-results-20260716.json)。榜单仍可能因赛后审核改变，因此本文使用“2026-07-16 快照”，不把它写成永久定榜。

## 比赛目标

每个 task 需要一个输出为固定 one-hot 网格的 ONNX 图。最终阶段计分核心为：

```text
cost = memory_bytes + parameters
score = max(1, 25 - ln(max(1, cost)))
```

MACs 不计入最终 cost。由此，真正高价值的优化不是少做计算，而是：

- 删除或缩小中间张量；
- 将大规模计算融合进不计中间内存的最终 `output`；
- 用低维坐标、相位、代数关系或原生算子替代表格和全画布状态；
- 当图内微调停滞时，重新推导语义并更换表示族。

## 最终结果快照

| 指标 | 我们 | 榜首 |
| --- | ---: | ---: |
| Score | `7420.93` | `8314.03` |
| Rank | `249` | `1` |
| 与榜首差距 | `893.10` | `0` |

快照中第 16 名为 `8000.10`，第 17 名为 `7998.91`。这说明头部竞争最终集中在约 8000 分以上，但奖牌资格仍以 Kaggle/主办方审核结果为准。

Kaggle 榜单只列出 5 个成员账号，而 GitHub 分工表使用了 6 个工作槽。赛后归档应把“仓库协作者”和“Kaggle 正式队员”分别记录，不再默认二者相同。

## 我们实际做了什么

### 1. 六路任务拆分

2026-07-09 的分工快照将 400 个 primary task 分给 A-F，并用 `task233`、`task366` 两个 shared review 槽使每人恰好 67 个槽位。分工以
[`task_assignment_400.csv`](../../assignments/task_assignment_400.csv) 为唯一来源。

这解决了任务覆盖问题，但正式拆分发生在比赛最后六天。仓库证据不能证明整个团队此前没有工作，只能确认共享仓库中的结构化六路 campaign 从 7 月 9 日开始。

### 2. 成员方法

| 工作区 | 主要方法 | 可复核成果 |
| --- | --- | --- |
| A | 高成本和 shape relayout 探索 | 主分支保留实验脚本；部分更完整结果仍在历史远端分支，未形成统一线上账本。 |
| B | 规则重建、低秩分解、QLinear、bool/int32 状态 | 十任务 exact batch 本地 `+1.000883`，ref `54686944` 在线约 `+1.00`。 |
| C | 等价压缩、candidate registry、父包 rebase、公共源审计 | 22-task overlay 本地 `+0.919103`，在线约 `+0.92`；后期承担全局集成。 |
| D | 大规模候选扫描和规则挖掘 | 从 95,451 个原始候选筛选，接受 8 个，本地累计 `+1.692853`。 |
| E | Kaggle A/B、差分 fuzz、异常任务拆分 | 9 个 task 形成带 ref 的接受链；识别 `task110`、`task050` 等隐藏回退。 |
| F | 面向 20 分的代数重写、CRT/动态索引、打包经验 | 记录 task026/274/334/306 等跨过 20 分及运行时失败模式。 |

详细团队证据入口：

- [B 工作记录](../../workplace%20B/readme.md)
- [C 工作区](../../workplace%20C/readme.md)
- [D 工作日志](../../workplace%20D/worklog.md)
- [E 交接记录](../../workplace%20E/TEAM_HANDOFF_20260715.md)
- [F 优化经验](../../workplace%20F/optimization_lessons_20260712.md)

### 3. 最终线上闭环

后期流程逐渐稳定为：

```text
锁定父提交 ref 和 SHA
-> 在准确父包上逐 task 重新计分
-> ONNX checker / shape inference / 全公开样本
-> 生成或差分 fuzz
-> 只覆盖 cost 更低的候选
-> 检查 400 文件、命名、CRC 和 SHA
-> 小批或单任务 Kaggle 提交
-> 等 COMPLETE
-> 按线上差分接受、拆分或回滚
```

最终最高 ref `54736568` 的描述为 `all400 GitHub audit safe11 +1.025027`，说明最后一轮依赖全 400 包审计和 11 个安全候选的集成。

## 做对的事情

### 保存负结果

E 区没有把“本地全过”直接等同于“线上安全”。最典型的两个反例是：

- `task110`：本地 cost `9682 -> 7326`，线上却 `7379.41 -> 7363.58`；
- `task050`：公开样本和 `0/5000` 颜色置换 fuzz 均通过，线上仍 `7385.93 -> 7385.89`。

这些候选被明确标成 rejected，而不是从仓库消失。这个习惯值得保留。

### 精确父包 rebase

候选的历史增益不能直接迁移到新父包。B、C、E 的后期记录均开始保存 parent ref、模型 SHA、ZIP SHA 和 expected delta，使本地预测在规则正确时通常能与线上舍入一致。

### 从通用压缩转向规则级重写

B、C、F 的经验共同表明，完整空间选择图、Cast 链和多层 mask 即使语义正确，也常因中间内存变贵。有效改进更多来自：

- 低秩 `Einsum`；
- 直接输出 contraction；
- `QLinearConv` / `QLinearMatMul`；
- bool、int8、uint8、float16 中间状态；
- 坐标代数、signed margin、CRT 或有限状态表示；
- 共享 initializer、删除 carrier、减少全画布张量。

### 开始形成跨组知识迁移

C 对 B、D 的方法做过独立迁移，E 参考了 2025 code-golf 方案并建立 differential fuzz。说明团队已经越过“每人只看自己的 67 题”的阶段，只是发生得较晚。

## 没做好的事情

### 1. 太晚建立统一系统

共享仓库的正式分工生成于 7 月 9 日，比赛 7 月 15 日结束。六路并行有覆盖，但没有足够时间把成功模式沉淀成自动 scheduler，再让所有任务反复受益。

### 2. 工作区不是统一的 task 状态机

B/C/E 有较完整记录，A/D/F 的入口和账本完整度不同。E 在 7 月 15 日较早的 handoff 仍记录 `7387.15`，而同日全局集成已经继续到 `7407.70`，最终又到 `7420.93`。这不是模型问题，而是状态同步问题。

### 3. 提交很多，但信息利用率不够高

Kaggle 快照记录团队共 `1975` 次提交。提交量已经很大，主要短板不是“验证次数不够”，而是：

- 批次变化不能总是快速归因到单 task；
- hidden failure 没有立即转成全团队可消费的禁用规则；
- 成功架构没有持续进入共享 trick registry；
- 最终父包、任务状态和成员 handoff 之间存在滞后。

### 4. 合规决策链不完整

仓库先记录了公共 archive 的合规待确认，后来又记录了公共模型直接集成，但缺少一份明确的 host 规则解释、许可证证据和团队决策。以后公共源不仅要记录“来自哪里”，还要记录“为什么允许用、允许用到什么程度”。

### 5. 对架构重写投入不足

我们的后期大量工作仍是候选扫描、等价压缩和已有图优化。这些有效，但无法系统性消除数百个 task 的大中间张量。与头部的差距说明，继续在相同表示上压 cost 的边际收益远低于从 ARC 语义重新合成图。

## 金牌选手公开方案

以下内容是 2026-07-16 已公开资料，不包含尚未发布的完整冠军代码。

### 第 1 名：Kaggle Agent，8314.03

[冠军 writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/1st-place-kaggle-agent) 当前披露：

- 使用 dashboard 共享每个 task 的 ONNX、分数和历史；
- agent 的输入信息和 prompt/instructions 是关键资产；
- 统计上，架构重写平均每 task 约 `+0.5`，优化已有图约 `+0.05`；
- 因此 agent 被持续要求探索新架构，避免停在同一表示的局部最优；
- 完整方案仍标记为后续发布。

其[静态 dashboard](https://daxiongshu.github.io/kaggle-agent-neurogolf-2026/)展示 task 级总览和历史，说明团队共享的核心对象不是散落脚本，而是实时 best-of-task 状态。

### 第 6 名：claudex，8118.97

[第 6 名 writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/6th-place-solution-claudex)给出了较完整的方法：

- 两组各做 200 task，一两天后交换任务范围，用不同 prompt 和算子偏好打破局部最优；
- Codex 与 ChatGPT Web 两条流并行；
- prompt 固定包含环境版本、输入输出、当前 best、官方 scorer、ARC-GEN 和目标增益；
- 建立 score-band operator library，按当前分数带推荐图族；
- 常用单 `Einsum`、坐标解析几何、`ConvTranspose`、`QLinearConv`、pool/gather/rank/count、小 latent state；
- 尽量只在最终节点合成完整 `[1,10,30,30]`，优先小 dtype，并把最大张量融合到免费 output；
- 每个候选需通过 checker、shape inference、ONNX Runtime 1.24.4、所有公开样本和 1000 个新 ARC-GEN 样本；
- 对风险 task 使用 singleton submission 隔离线上失败。

其公开仓库为 [PNTAN17/neurogolf](https://github.com/PNTAN17/neurogolf)。

### 第 7 名：Slow and Steady，8067.63

[第 7 名 writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/7th-place-solution-slow-and-steady)目前主要是流程图草稿。作者补充：`NEUROGOLF_PROMPT.md` 保存规则和策略，task ZIP 保存持续更新的知识库，最终积累了 68 条规则。

可确认的重点是“稳定规则书 + task 级动态知识包”，但当前文字不足以可靠复原其图优化细节。

### 第 8 名：kluges clueless，8048.16

[第 8 名 writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/8th-place-writeup-draft)当前只有图示草稿，没有足够文字证据，因此本文不对其实现作进一步推断。

### 第 10 名：Timeout exceeded : (，8028.25

[第 10 名 writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/draft-10th-place-solution)是目前最具体的系统说明之一：

- 每个 task 固定为三件套：ONNX、可复现 builder/attack、task 文档；
- bounded worker pool 为每个 task 保留持久 Codex session；
- 每轮执行 `backup -> experiment -> validate -> promote or restore`；
- 在线 batch 出现差额后，用分组/二分 submission 定位回退 task；
- 只有 online-proven trick 才串行合并进共享知识库；
- 使用 Python code-golf 解作为语义参考，而不是机械转译成 ONNX；
- 核心图技巧包括直接写 output、重算关系而非保存状态、一份 basis 多用途、有限状态/数论代替表、原生算子作为 transport、把 `Einsum` contraction order 当作模型状态。

其五天无人值守案例从本地 `7516.01` 到 `7575.78`，改动 271 个 task，其中 256 个严格降 cost、15 个等 cost，未保留 cost 回退。这说明自动化的价值在于稳定累计大量小中型收益，而不是等待一个万能技巧。

## 我们与金牌方案的对照

| 维度 | 我们 | 金牌方案共同特征 | 结论 |
| --- | --- | --- | --- |
| 任务拆分 | 六人 67 槽 | task 是独立优化单元 | 方向正确，但建立较晚。 |
| 当前 best | 分散在成员目录和 dated handoff | dashboard / central task state | 缺统一事实源。 |
| 可复现性 | 部分 task 有脚本、SHA、ref | 每 task 固定 builder + model + doc | 应强制三件套。 |
| 搜索策略 | 公共候选、等价压缩、部分重写 | 优先架构重写和表示切换 | 我们偏局部优化。 |
| 正确性 | 全例、部分 fuzz、线上 A/B | 全例 + fresh generator + runtime + singleton | E/C 已接近，但未全仓统一。 |
| 失败处理 | E/C 有 rejected 记录 | 自动 rollback + task warning + global tricks | 我们靠人工同步较多。 |
| agent 编排 | 按成员/批次并行 | bounded pool、持久 session、事务化提升 | 缺统一 scheduler。 |
| 知识管理 | README/worklog/报告并存 | 规则书、task memory、trick registry | 资料丰富但检索成本高。 |
| 合规 | 有原则，决策证据不完整 | 来源和验证随 task 单元保存 | 需要统一 source manifest。 |

## 下一次比赛的执行方案

### 第 1 天建立基础设施

1. 固定官方 scorer/runtime 版本并做 golden tests。
2. 建立 task 三件套和中央 registry。
3. 所有候选通过事务化 gate，失败自动回滚。
4. dashboard 同时显示 parent、cost、score、状态、owner、最近失败和线上 ref。
5. 公共源 manifest 在使用前记录许可证、hash 和允许范围。

### 搜索资源分配

- 60%：语义重推导与 from-scratch architecture rewrite；
- 20%：把 online-proven 表示迁移到同类 task；
- 10%：等价压缩、dtype、initializer 和 contraction-order 优化；
- 10%：hidden-risk、runtime、打包和线上归因。

每个 task 至少保留两类并行搜索：改进 incumbent，以及不看 incumbent 的从零重写。agent 不能用“已到理论下限”作为停止理由，除非给出可检查的下界证明。

### 线上验证

- 日常 batch 只包含可归因的小集合；
- 高风险 task 使用 singleton；
- expected delta 与 observed delta 自动对账；
- 有差额立即分组定位，失败 task 加入全局禁用规则；
- 每日冻结一个已确认 champion SHA，不能用模糊的“队友最高包”作父包。

### 赛前最后一周

- 停止大规模目录重构，只做模型和账本集成；
- 提前准备完整包恢复、镜像和离线重建；
- 每日导出 dashboard、registry、accepted builders 和提交 SHA；
- 预留提交额度，避免最后一晚有已验证增益却无法入榜。

## 最终结论

我们的有效能力已经很明确：能做规则级重写、全包 rebase、公共源审计、差分 fuzz 和 Kaggle 隔离验证。最终没有进入头部，主要是这些能力没有在比赛早期统一成一台可持续运行的 task 优化系统。

最值得带走的不是某个 ONNX 算子，而是一个工作原则：每次成功都必须同时更新可执行模型、可复现 builder、task 语义记忆和中央 best 状态；每次失败也必须转化为后续 worker 能直接消费的限制。这样 400 个任务才会形成累积优势，而不是 400 条彼此松散的聊天和脚本。

## 资料来源

- [Kaggle final leaderboard](https://www.kaggle.com/competitions/neurogolf-2026/leaderboard)
- [Competition site and dates](https://sites.google.com/view/neurogolf-2026/home)
- [1st place writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/1st-place-kaggle-agent)
- [6th place writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/6th-place-solution-claudex)
- [7th place writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/7th-place-solution-slow-and-steady)
- [8th place draft](https://www.kaggle.com/competitions/neurogolf-2026/writeups/8th-place-writeup-draft)
- [10th place writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/draft-10th-place-solution)
- [6th place public repository](https://github.com/PNTAN17/neurogolf)
- [1st place static dashboard](https://daxiongshu.github.io/kaggle-agent-neurogolf-2026/)
