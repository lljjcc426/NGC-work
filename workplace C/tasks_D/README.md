# D 组题目镜像到 C 工作区

本目录保存 D 组 67 个 primary 任务的 JSON 镜像，供 C+D 联合建模、viewer、score ledger 和 parent-aware overlay 使用。

- 原始来源：`neurogolf_400_tasks/tasks/taskXXX.json`
- 目标目录：`workplace C/tasks_D/taskXXX.json`
- 复制方式：Git blob SHA 直接复用，文件内容保持逐字节一致
- 原 owner：`D`
- 任务数：`67`
- 清单：`task_manifest_D.csv`

镜像不会改变 `assignments/task_assignment_400.csv` 中的 owner，也不覆盖 `workplace C/tasks/` 中原 C 组任务。后续联合脚本应使用 `source_owner` 或目录名区分 C/D provenance。
