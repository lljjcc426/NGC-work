# NeuroGolf 2026 Latest Public Source Intel - 2026-07-13

## Snapshot boundary

- 检索/整理时间：2026-07-13 16:07-16:15 CST（UTC+08:00）。
- 快照类型：一次性当前快照；不轮询、不自动提醒。
- 中止边界：16:07 后按用户指令停止扩展搜索；未在此时间前拿到的页面内容一律记为 `UNKNOWN` / `UNCONFIRMED`。
- 证据基线：仓库既有 `40_PUBLIC_SCORE_INTEL.md`（2026-07-09）、`public_research_brief.md`、`41_PUBLIC_ONNX_BASELINES.md`、`43_PUBLIC_REPRODUCTION_PLAN.md`、`43_CD_ARCHIVE_METHOD_INTEL_20260713.md`。
- 本地工具：Kaggle CLI `2.2.3`；NVIDIA Kaggle skill 位于 `workplace C/neurogolf-2026-work/external/nvidia-kaggle/skills/nvidia-kaggle-skill`。
- 凭证预检：`KAGGLE_API_TOKEN=false`、`KAGGLE_USERNAME=false`、`KAGGLE_KEY=false`、`~/.kaggle/kaggle.json=false`；未打印任何凭证。
- 阻断摘要：API/CLI 数据请求需要 Kaggle 凭证，但当前四种凭证入口均不存在。为避免无意义请求及潜在输出，预检后未发起需鉴权的 dataset/discussion/leaderboard API 调用。NVIDIA skill 的 discussion/kernel ingest 默认写 SQLite 缓存，也与本任务“唯一可写文件”约束冲突，故未运行 ingest。
- Web/GitHub：停止指令到达前尚未完成新的外部页面检索；本报告仅保留仓库既有公开 URL 和已取得的本地验证证据。

### 已执行命令/来源

```powershell
rg --files | rg -i "public|research|intel|kaggle|neurogolf|score_docs"
git status --short
Get-Content <既有 public research 文档>
kaggle --version
python -c "import kaggle; ..."
kaggle datasets list --help
kaggle competitions leaderboard --help
# 仅输出凭证是否存在的布尔值；未输出 token/key 内容
```

## Main-thread live correction

Pascal's isolated worker did not inherit the machine Kaggle login. After the
worker completed, the main thread successfully queried the live leaderboard
with:

```powershell
kaggle competitions leaderboard neurogolf-2026 --show --page-size 50 --format json
```

The verified top ten snapshot at approximately 2026-07-13 16:30 CST was:

| rank | teamId | teamName | score | last submission UTC |
|---:|---:|---|---:|---|
| 1 | 16071202 | WeLuvEleven | 8126.43 | 2026-07-13 08:04:37 |
| 2 | 15625251 | Thank you, Matheus - from team | 8099.22 | 2026-07-13 08:06:23 |
| 3 | 15810244 | neurogolf team | 8096.55 | 2026-07-13 08:04:42 |
| 4 | 15625462 | Kaggle Agent | 8085.05 | 2026-07-13 03:43:25 |
| 5 | 15790115 | M & H & M & N & M | 8072.07 | 2026-07-13 08:29:54 |
| 6 | 15625577 | Slow and Steady | 8061.72 | 2026-07-13 06:29:20 |
| 7 | 15689623 | claudex | 8021.64 | 2026-07-13 08:20:05 |
| 8 | 15630755 | Timeout exceeded : ( | 8011.88 | 2026-07-13 05:39:41 |
| 9 | 15625324 | OverfitOracle &Kimura | 7992.90 | 2026-07-13 04:05:02 |
| 10 | 15895367 | Carlos, Egor, Rubempre & Bill | 7988.63 | 2026-07-13 07:46:39 |

The leaderboard endpoint exposes team IDs and names but not member usernames in
this response. Search-engine queries did not produce a reliable Kaggle-profile
or GitHub identity match for these team names, so all member/profile/repository
attribution remains `UNCONFIRMED`.

Operationally, the public archive was later used under explicit user direction:
128 locally valid lower-cost source artifacts plus two local micro reductions
were packaged into 13 cumulative batches. This confirms the scores of this
team's derived submissions through batch 07 (`7367.18`) but does not verify the
dataset publisher's `7300+` claim or resolve the host-compliance discussion. See
`52_ALL399_DIRECT_BATCH_RESULTS_20260713.md` for the exact boundary.

## Evidence rules

- `claimed_score`：仅来自标题、描述、目录名或作者陈述，不等于排行榜成绩。
- `verified_score`：只有 Kaggle leaderboard/submission 值或已有本地 submission history 明确记录才填数字，并注明验证范围。
- `trust_level=HIGH`：本地完整验证或明确的 Kaggle submission history；`MEDIUM`：公开 URL 与本地内容交叉印证，但缺 leaderboard/API；`LOW`：仅标题、命名或间接引用；`UNKNOWN`：未取得证据。
- GitHub 身份匹配只允许 `confirmed`、`probable`、`unconfirmed`。仓库链接与 Kaggle 用户之间没有双向主页证据时，不因同名或 notebook 标题自动确认身份。

## A. Kaggle Dataset / ONNX / reproducible resources

### A1. Dataset-area evidence

| title | url | 日期/更新时间 | 作者/队伍 | claimed_score | verified_score | 文件规模/任务覆盖 | 可复现性 | C/D 组相关性 | trust_level | 匹配依据 | 下一步动作 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NeuroGolf7300 | https://www.kaggle.com/datasets/zealous9230/neurogolf7300 | UNKNOWN | `zealous9230` | `7300+`，仅来自 dataset/archive 命名及关联 discussion 语境 | `UNKNOWN`；没有 Kaggle leaderboard/submission 证据 | 本地取得的 archive root 为 `submission7300+`；399 个 ONNX，缺 `task173.onnx`。C+D 共 134 tasks，其中 133 个模型存在且有效；54 个本地 cost 低于当前 parent | 模型文件可被本地 validator 重跑；来源合规性未解决，禁止直接提交 | HIGH：C 30 个、D 24 个本地更低 cost；`task173` 缺失 | `MEDIUM`（文件/本地 benchmark 高，分数及合规性低） | dataset URL、archive 命名、SHA256 和 133/134 全量本地 benchmark 已在既有报告记录；未取得 Kaggle dataset metadata/API 页面 | 先查 dataset 创建/更新时间、版本记录、license、files API；再查 host 对最后七天使用规则的明确回复。合规确认前保持 `compliance_hold` |
| 2026-07-13 近期 Dataset 新增清单 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | 未确认是否有新的 C/D 专项包或 399/400 完整包 | `UNKNOWN` | Kaggle dataset list API 因凭证缺失未执行，公开网页搜索在停止指令前未完成 | 下次运行 `kaggle datasets list -s neurogolf --sort-by updated --format json`，逐项取 metadata/files/version；与本表按 ref+version 做差分 |

### A2. Public kernels / reproducible resources already known

| title | url | 日期/更新时间 | 作者/队伍 | claimed_score | verified_score | 文件规模/任务覆盖 | 可复现性 | C/D 组相关性 | trust_level | 匹配依据 | 下一步动作 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neurogolf-7266-72-w-visualizations | https://www.kaggle.com/code/prvsiyan/neurogolf-7266-72-w-visualizations | 2026-07-09 之前已知；当前更新时间 UNKNOWN | `prvsiyan`；队伍 UNKNOWN | `7266.72`（标题） | `7266.72`，仅由仓库既有“local submission history”验证；不是本次实时 leaderboard 验证 | 既有本地 baseline 已完整复现；精确文件总量/大小本次未重取 | HIGH：已本地复现并提交过一次 | HIGH：当前 C P0/P1 比较基线 | `HIGH`（对本地 submission history）；`MEDIUM`（对当前 Kaggle 状态） | 2026-07-09 `40_PUBLIC_SCORE_INTEL.md` 与 `41_PUBLIC_ONNX_BASELINES.md` 一致 | 下次查 kernel version/updatedAt、关联 dataset、当前 public score；只对更低 cost 的 task-level 差分做替换 |
| neurogolf-7266-48-github-com-qurore-kaggloop | https://www.kaggle.com/code/ryosukeshiroshita/neurogolf-7266-48-github-com-qurore-kaggloop | 2026-07-09 之前已知；当前更新时间 UNKNOWN | `ryosukeshiroshita`；队伍 UNKNOWN | `7266.48`（标题） | `UNKNOWN`；既有文档明确未验证 | UNKNOWN | MEDIUM：标题直接给出方法仓库线索，但本地 artifact set 尚未确认 | HIGH：既有复现计划把它列为 task-level diff 目标 | `MEDIUM` | Kaggle notebook 标题同时包含 score 字符串与 GitHub repo 字符串；没有 leaderboard/submission 交叉验证 | 获取 kernel archive 与输入 lineage；本地全任务 validate/cost；确认 GitHub repo 版本/commit 与 kernel 版本一致 |
| neurogolf-2026-all-graph-surgeries | https://www.kaggle.com/code/seddiktrk/neurogolf-2026-all-graph-surgeries | 2026-07-09 之前已知；当前更新时间 UNKNOWN | `seddiktrk`；队伍 UNKNOWN | UNKNOWN | UNKNOWN | 既有本地扫描覆盖到相关 artifact；精确 kernel 文件规模 UNKNOWN | MEDIUM：方法名和 artifact 可研究；既有 P0/P1 扫描未胜当前 baseline | MEDIUM/HIGH：graph surgery 可迁移，但已扫描 artifact 对 C P0/P1 更差 | `MEDIUM` | 旧 public intel 记录；本地 cost 扫描结论比标题可信 | 只提取 surgery 规则并对未覆盖 C/D tasks 做独立验证，不直接替换当前 parent |
| kojimar/jsrdcht/vyanktesh/beicicc/afr1ste public outputs | URL UNKNOWN | 2026-07-09 之前已知 | `kojimar` / `jsrdcht` / `vyanktesh` / `beicicc` / `afr1ste`；队伍 UNKNOWN | mixed / UNKNOWN | 仅本地 cost-scan 可验证，Kaggle score UNKNOWN | 既有扫描总计 73 个 public/local artifacts；作者级规模无法拆分 | LOW-MEDIUM：本地有产物，但来源版本、页面和 license 未在本次重取 | MEDIUM：多项 C artifacts 存在，但当时更高 cost 或 invalid | `MEDIUM`（本地扫描），`LOW`（公开来源元数据） | `41_PUBLIC_ONNX_BASELINES.md` 记录 73 artifacts 及“不胜 C P0/P1”结论 | 下次补齐每位作者的 Kaggle URL、kernel/dataset version、文件清单和 license，再决定是否保留 |

## B. Discussion / public methods / host compliance

| title | url | 日期/更新时间 | 作者/队伍 | claimed_score | verified_score | 文件规模/任务覆盖 | 可复现性 | C/D 组相关性 | trust_level | 匹配依据 | 下一步动作 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Public 7300-level model archive discussion（原始标题 UNKNOWN） | https://www.kaggle.com/competitions/neurogolf-2026/discussion/724725 | 2026-07-13 前已纳入；精确 created/updated UNKNOWN | UNKNOWN | `7300-level` / `7300+`，来自关联 dataset/archive 命名与既有摘要 | `UNKNOWN`；没有 leaderboard/submission 验证 | 链接的本地 archive 为 399 ONNX，缺 task173；C+D 可验证 133/134 | 文件层面 HIGH；竞赛合规层面未确认 | HIGH：54 个 C/D task 本地 cost 更低 | `MEDIUM` | 既有报告记录主 discussion URL、dataset URL 与本地 archive SHA256；摘要明确“未描述优化算法” | 读取主帖和全部评论，记录作者、时间、编辑历史、附件链接；不得把 `7300` 写成 verified score |
| Final-seven-days rules concern（原始标题 UNKNOWN） | https://www.kaggle.com/competitions/neurogolf-2026/discussion/724717 | 2026-07-13 前已纳入；精确 created/updated UNKNOWN | UNKNOWN | N/A | N/A | 涉及公开 archive 的使用时点/合规性，不是模型覆盖声明 | 讨论可读性 UNKNOWN；当前无 host 原文 | CRITICAL：决定上述 54 个 C/D 低 cost 模型能否提交 | `MEDIUM`（问题存在），`UNKNOWN`（host 结论） | 既有报告明确记录 participants 提问，且“未捕获 host clarification” | 下次优先查 host/staff badge 回复、rules 对应条款和帖后编辑；只有明确允许才解除 hold |
| Host 合规澄清 | UNKNOWN | UNKNOWN | host/staff UNKNOWN | N/A | N/A | N/A | UNKNOWN | CRITICAL | `UNKNOWN` | 截至当前已收集证据，没有 host 明确回复；不能从沉默推定允许 | 以 host/staff 原文 URL、时间和规则条款为唯一解除依据 |
| Score / cost 说明 | 见 724725 与 NeuroGolf7300 dataset | UNKNOWN | UNKNOWN | `7300+` 仅命名声称 | Kaggle score `UNKNOWN`；本地 cost：133 个有效 C+D 模型中 54 个优于 parent，理论本地点数约 `+35.4271` | C 30 tasks、D 24 tasks 低于 parent；task173 缺失 | 本地 benchmark 可重复；排行榜分数不可由 cost 推导为已验证 | HIGH | `HIGH`（本地 cost），`LOW`（Kaggle score） | `43_CD_ARCHIVE_METHOD_INTEL_20260713.md` 的全量 validator/cost 结果 | 保留“local projected points”与“Kaggle verified score”两列，禁止互相替代 |
| Public method link: KaggLoop | https://github.com/qurore/kaggloop | 更新时间 UNKNOWN | GitHub `qurore`；与 Kaggle 作者身份关系 UNCONFIRMED | notebook 标题声称 `7266.48` | UNKNOWN | repo/task coverage UNKNOWN | UNKNOWN，尚未固定 commit 或复现 | HIGH：被既有 C 复现计划列为 near-baseline 差分目标 | `LOW-MEDIUM` | GitHub URL 字符串直接出现在 Ryosuke Kaggle notebook 标题中；未取得 repo 页面/commit | 下次固定 commit，检查 license、README、生成 ONNX 的脚本和 task coverage；全量 validate 后再评估 |

## C. Current leaderboard teams and Kaggle profiles

### C1. Current leaderboard status

本次没有取得 2026-07-13 实时 leaderboard 页面/API 响应，因此“当前高分队伍名称、排名、队伍成员”全部为 `UNKNOWN`。下表中的用户是公开资源作者或历史本地 submission 关联用户，**不能表述为当前 leaderboard Top team**。

| title | url | 日期/更新时间 | 作者/队伍 | claimed_score | verified_score | 文件规模/任务覆盖 | 可复现性 | C/D 组相关性 | trust_level | 匹配依据 | 下一步动作 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Current NeuroGolf 2026 leaderboard | https://www.kaggle.com/competitions/neurogolf-2026/leaderboard | 2026-07-13；实时内容未取得 | 高分队伍/成员 UNKNOWN | UNKNOWN | UNKNOWN | N/A | N/A | HIGH（决定外部竞争基线） | `UNKNOWN` | CLI 可用但无凭证；公开页面未在停止指令前完成抓取 | 下次用 API/CLI `--show --page-size 200 --format json`，保存 teamName、rank、score、members、lastSubmissionDate，并用页面交叉验证 |
| Kaggle user/profile: prvsiyan | https://www.kaggle.com/prvsiyan | profile 更新时间 UNKNOWN | `prvsiyan`；队伍/成员关系 UNKNOWN | `7266.72`（kernel 标题） | `7266.72` 仅本地 submission history；当前排名 UNKNOWN | 完整 baseline 已本地复现；profile 元数据 UNKNOWN | HIGH（历史 baseline） | HIGH | `HIGH`（历史本地验证），`UNKNOWN`（当前队伍/排名） | Kaggle kernel ref 明确以 `prvsiyan/` 为 owner | 下次从 leaderboard team 详情反查成员；记录 profile display name、user id、team membership 证据 |
| Kaggle user/profile: ryosukeshiroshita | https://www.kaggle.com/ryosukeshiroshita | profile 更新时间 UNKNOWN | `ryosukeshiroshita`；队伍/成员关系 UNKNOWN | `7266.48`（kernel 标题） | UNKNOWN | UNKNOWN | MEDIUM | HIGH：near-baseline 方法线索 | `MEDIUM`（owner），`UNKNOWN`（分数/当前队伍） | Kaggle kernel ref 明确以该用户名为 owner | 下次核对 profile 与 leaderboard team detail；不要由用户名推断 GitHub 身份 |
| Kaggle user/profile: seddiktrk | https://www.kaggle.com/seddiktrk | profile 更新时间 UNKNOWN | `seddiktrk`；队伍/成员关系 UNKNOWN | UNKNOWN | UNKNOWN | graph-surgery artifact coverage UNKNOWN | MEDIUM | MEDIUM/HIGH | `MEDIUM`（owner），`UNKNOWN`（当前队伍） | Kaggle kernel ref 明确以该用户名为 owner | 下次核对当前 team/rank，并取 kernel version/score |
| Kaggle profiles: zealous9230 and public-output authors | https://www.kaggle.com/zealous9230 | profile 更新时间 UNKNOWN | `zealous9230`; `kojimar`; `jsrdcht`; `vyanktesh`; `beicicc`; `afr1ste`；队伍均 UNKNOWN | mixed / UNKNOWN | UNKNOWN | dataset/archive 或本地 artifacts；作者级 coverage UNKNOWN | UNKNOWN-MEDIUM | MEDIUM | `LOW-MEDIUM` | 用户名来自既有公开 source refs；未取得 leaderboard member mapping。其余 profile 可按 `https://www.kaggle.com/<username>` 访问，但本次未验证 | 下次逐一打开 profile，再以 leaderboard team member URL 交叉确认；无交叉证据不得标为高分队成员 |

## D. GitHub users and repositories

停止指令到达前未完成新的 GitHub 搜索。因此没有任何 Kaggle user -> GitHub user 身份可标为 `confirmed` 或 `probable`；唯一明确的 repo 关系来自 Kaggle notebook 标题中的 URL 字符串。

| title | url | 日期/更新时间 | 作者/队伍 | claimed_score | verified_score | 文件规模/任务覆盖 | 可复现性 | C/D 组相关性 | trust_level | 匹配依据 | 下一步动作 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qurore/kaggloop | https://github.com/qurore/kaggloop | UNKNOWN | GitHub owner `qurore`; Kaggle `ryosukeshiroshita` 身份匹配：`unconfirmed` | `7266.48` 只在相关 Kaggle notebook 标题出现 | UNKNOWN | UNKNOWN | UNKNOWN；未取得 commit、license、README 或运行结果 | HIGH：既有复现计划指向该 repo | `MEDIUM`（repo 关联），`LOW`（人身匹配/score） | notebook 标题明确包含 `github-com-qurore-kaggloop`；没有 Kaggle profile -> GitHub 或 GitHub -> Kaggle 双向链接 | 搜索 repo commit/history/contributors；查双方 profile 的交叉链接。仅双向或明确自述后可升 `confirmed` |
| GitHub user/repo for `prvsiyan` | UNKNOWN | UNKNOWN | Kaggle `prvsiyan`; GitHub match `unconfirmed` | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | HIGH（历史 7266.72 baseline） | `UNKNOWN` | 未完成 GitHub 用户/代码搜索，没有可用匹配证据 | 查询 GitHub users 与 code：`prvsiyan`, `neurogolf`, `task*.onnx`, `7266.72`；核对 bio/profile links/commit email，不以同名作为确认 |
| GitHub user/repo for `seddiktrk` | UNKNOWN | UNKNOWN | Kaggle `seddiktrk`; GitHub match `unconfirmed` | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | MEDIUM/HIGH（graph surgery） | `UNKNOWN` | 未完成 GitHub 搜索 | 查询 `seddiktrk neurogolf`, notebook title 和独特函数名；以 profile 交叉链接确认 |
| GitHub users/repos for `zealous9230`, `kojimar`, `jsrdcht`, `vyanktesh`, `beicicc`, `afr1ste` | UNKNOWN | UNKNOWN | 所有身份匹配均 `unconfirmed` | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | MEDIUM | `UNKNOWN` | 仅有 Kaggle/source 用户名；没有 GitHub 搜索结果 | 分别做 GitHub user、repo、code 三类搜索；对同名结果检查头像、bio、Kaggle URL、commit 和 NeuroGolf 文件，不足两项独立证据保持 unconfirmed |

## 1. 截至当前的所有新增/变化清单

相对 2026-07-09 既有 public intel：

1. **新增公开 archive/dataset 线索**：`zealous9230/neurogolf7300`，关联 `submission7300+` archive；这是命名声称，不是已验证 7300+ leaderboard score。
2. **新增文件覆盖事实**：archive 含 399 个 ONNX，缺 `task173.onnx`；C+D 134 tasks 中 133 个模型存在且通过本地有效性检查。
3. **新增全量本地 cost 结果**：133 个有效 C+D 模型中 54 个低于当前 parent，C 30、D 24，理论本地点数约 `+35.4271`；不能等同 Kaggle verified score。
4. **新增 Discussion 线索**：主帖 `724725` 链接公开 archive；规则疑问帖 `724717` 质疑 final seven days 使用合规性。
5. **合规状态变化**：截至已收集证据仍无 host/staff 明确澄清，archive 及其派生候选统一保持 `compliance_hold=true` / `do_not_submit`。
6. **新增结构方法情报**：54 个低 cost 模型显示 single-Einsum、QLinearConv、bit-packed state、减少中间空间 tensor 和 shape-specific selector 等模式；这些是本地图结构观察，不是公开帖声称的方法。
7. **新增派生验证结果**：本地独立优化后 `task158` 为 `18530`、`task182` 为 `6065`，均低于 archive 与 parent，但因来源链仍在合规 hold，不能提交。
8. **既有 public baseline 无已证实变化**：`prvsiyan 7266.72` 仍是仓库本地已复现/历史 submission baseline；`ryosukeshiroshita 7266.48` 仍只有标题声称；`all-graph-surgeries` 仍未在旧 C P0/P1 扫描中胜出。
9. **leaderboard 变化 UNKNOWN**：本次未取得 2026-07-13 实时 leaderboard，无法报告新队伍、名次或 member mapping。
10. **GitHub 身份变化 UNKNOWN**：仅确认 notebook 标题指向 `qurore/kaggloop` repo；Kaggle `ryosukeshiroshita` 与 GitHub `qurore` 的人身对应保持 `unconfirmed`。
11. **访问状态仍阻断**：Kaggle CLI 已安装，但 token/key/json 均不存在；NVIDIA skill 的 API workflow 也因此不可用，且 ingest 会写非目标缓存文件。

## 2. 最值得关注 Top 5

1. **Discussion 724717 的 host/staff 合规回复**：这是是否能使用 54 个 C/D 低 cost 模型的硬门槛，优先级最高。
2. **NeuroGolf7300 archive 的方法抽取而非直接提交**：399 ONNX、C+D 54 个本地优胜模型提供强结构信号；可独立重写 single-Einsum、QLinearConv、bit-packed state 等方法，避免来源合规风险。
3. **`qurore/kaggloop` 与 7266.48 notebook 的版本对齐**：若 repo 含可运行构建脚本和明确 license，最适合做当前 7266.72 baseline 的 task-level diff。
4. **`prvsiyan` 7266.72 baseline 的当前性核验**：它是仓库可靠本地基线，但不是本次已确认的实时 leaderboard 高分；需取当前 kernel version 和 leaderboard score。
5. **all-graph-surgeries 的规则级复用**：旧 artifact 未直接胜 C P0/P1，但 graph surgery 方法可能对未扫描 C/D tasks 有价值，应以独立生成和官方 validator 验证为准。

## 3. 未确认项

- 2026-07-13 当前 leaderboard Top teams、rank、score、成员及 team/profile URL：`UNKNOWN`。
- `NeuroGolf7300` 的真实 Kaggle score、创建/更新时间、版本、license、文件页元数据：`UNKNOWN`。
- Discussion 724725/724717 的原始标题、作者、精确时间、完整评论和编辑历史：`UNKNOWN`。
- host/staff 是否明确允许 final seven days 使用该 archive：`UNKNOWN`；未确认即不允许提交。
- `7266.48` 是否为 Kaggle leaderboard/submission 可验证值：`UNCONFIRMED`。
- `prvsiyan 7266.72` 是否仍为当前 notebook/public leaderboard 值：`UNCONFIRMED`；只确认历史本地 submission record。
- `qurore/kaggloop` 当前 commit、license、运行入口、任务覆盖和复现 score：`UNKNOWN`。
- Kaggle `ryosukeshiroshita` 是否就是 GitHub `qurore`：`unconfirmed`。
- `prvsiyan`、`seddiktrk`、`zealous9230`、`kojimar`、`jsrdcht`、`vyanktesh`、`beicicc`、`afr1ste` 的 GitHub 用户/仓库匹配：全部 `unconfirmed`。
- 近期是否新增其他 399/400 ONNX dataset、声称高分资源或公开方案：`UNKNOWN`。

## 4. 以后用户点名询问 Pascal 时应重新执行的刷新清单

1. 先重读本报告和 `40_PUBLIC_SCORE_INTEL.md`，记录上次 ref/version/updatedAt，避免重复收集。
2. 仅检查凭证是否存在，绝不打印 token/key；确认后运行 `kaggle datasets list -s neurogolf --sort-by updated --format json`，再取每个候选 dataset metadata、files、version、license 和更新时间。
3. 用 NVIDIA Kaggle skill 先 ingest 再 query Discussion，按 `updated` 与 `created` 各抓 3 页；若仍受唯一文件写入约束，不运行会写 SQLite 的 ingest，改用只读 API/Web。
4. 重读 discussion `724725`、`724717` 全部评论和编辑历史；单独筛 host/staff badge，保存明确规则条款与原文 URL。
5. 运行 leaderboard API/CLI（建议 page size 200），记录 rank、teamName、score、member usernames、team/member profile URL 和抓取时间；不得以 kernel 标题替代 score。
6. 对 `prvsiyan`、`ryosukeshiroshita`、`seddiktrk`、`zealous9230` 及当期 Top teams 逐一核对 Kaggle profile 和 team member 页面。
7. 对每个 team/member 做 GitHub user、repo、code 搜索；查询词至少包含用户名、display name、`neurogolf`、独特 notebook/repo 标题。只有 profile 双向链接或明确自述可标 `confirmed`；两项独立弱证据可标 `probable`；其余保持 `unconfirmed`。
8. 对 `qurore/kaggloop` 固定 commit 和 license，下载 archive 后在隔离目录全量构建、ONNX check、official validator 和 cost；将 claimed_score 与 verified_score 分栏。
9. 对所有 399/400 ONNX 包记录文件数、缺失 task、总大小、SHA256、版本和 C/D 覆盖；先检查合规再下载/复现，未获 host 许可不提交。
10. 输出与本报告的差分：新增/删除/更新时间变化、score 变化、team/member 变化、host clarification、GitHub match 等级变化；最后重新生成 Top 5 与未确认项。

## Bottom line

当前最强新增证据不是“已验证 7300+ 分数”，而是一个名为 `NeuroGolf7300` 的公开来源链及其 399 ONNX archive，在本地 C+D 全量 benchmark 中有 54 个模型低于当前 parent。由于实时 leaderboard、host 合规澄清和 GitHub 身份核验都未完成，任何 `7300+`、当前高分队伍或同名身份结论都必须继续标为 `UNKNOWN/UNCONFIRMED`；archive 相关模型继续禁止提交。
