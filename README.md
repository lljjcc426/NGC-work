# NGC-work

The 2026 NeuroGolf Championship 团队共享仓库，现已进入赛后归档状态。

## 最终结果

截至 Kaggle 公开榜单 `2026-07-16 08:26:45` 快照：

| 项目 | 结果 |
| --- | ---: |
| Team | `Blacklions.` |
| Public / Private score | `7420.93 / 7420.93` |
| Rank | `249 / 3061` |
| 最终最高提交 | ref `54736568` |
| Kaggle 记录的提交数 | `1975` |

榜单可能因赛后资格审核变化；这里保存的是可复核的赛后首日快照，不声称是永久定榜。

## 快速入口

- [团队赛后复盘](docs/postmortem/2026-neurogolf-retrospective.md)
- [赛后证据快照](docs/evidence/final-results-20260716.json)
- [仓库结构与归档规则](docs/repository-guide.md)
- [协作与实验记录规范](CONTRIBUTING.md)
- [400 task 分工表](assignments/task_assignment_400.csv)
- [分工摘要](assignments/task_assignment_summary.md)
- [完整 task 数据镜像](neurogolf_400_tasks/)

## 工作区

六个工作区保留比赛期间的原始路径，避免破坏提交记录和文档链接。

| 工作区 | 比赛职责 | 主要入口 |
| --- | --- | --- |
| A | shape relayout / high-cost tasks | [workplace A](workplace%20A/readme.md) |
| B | same-shape rule rewrites | [workplace B](workplace%20B/readme.md) |
| C | ONNX equivalent compression / integration | [workplace C](workplace%20C/readme.md) |
| D | rule mining / candidate scanning | [workplace D](workplace%20D/readme.md) |
| E | public-source A/B / hidden-set isolation | [workplace E](workplace%20E/readme.md) |
| F | tail validation / packaging lessons | [workplace F](workplace%20F/readme.md) |

任务归属以 `assignments/task_assignment_400.csv` 为唯一事实来源，不在 README 中维护第二份任务清单。

## 目录约定

```text
assignments/          任务分配和优先级快照
docs/                 赛后复盘、证据和仓库规范
neurogolf_400_tasks/  400 个公开 task 及本地查看器
tools/                不依赖私有数据的仓库维护工具
workplace A..F/       各成员比赛期间的源码、实验和记录
.github/workflows/    ONNX 安全检查与仓库结构检查
```

历史工作区是只增不改的比赛档案。新实验采用
`tasks/taskNNN/{README.md,builder.py,result.json}` 三件套；生成的 ONNX、ZIP、外部数据和凭据不进入 Git。

## 本地检查

```powershell
python tools/repository_audit.py
pytest -q "workplace C/neurogolf-2026-work/tests"
```

仓库审计会检查任务分工、必需入口、UTF-8 文档和相对 Markdown 链接。删除文件必须在提交说明或工作日志中列出完整路径和原因。
