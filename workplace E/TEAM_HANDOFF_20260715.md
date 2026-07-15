# E 组工作交接（更新至 2026-07-15）

这份文档是 E 组当前工作的统一入口。早期逐日实验见
[`worklog.md`](./worklog.md)，每次线上提交的模型、完整 400-task ZIP、哈希和
结构化结果见对应的 `20260715_e_*` 目录。

## 当前结论

- E 分工共 67 个槽位：66 个 primary task，加上 `task233` shared review。
- Kaggle 当前队伍最高分：`7387.15`，submission ref `54711326`。
- 该最高分提交描述为：在 `54709399` (`7386.94`) 上合入
  `task118/task133/task174`；其中 `task118` 是 E 组已验证模型。
- `7387.15` 的精确 400-task 二进制包在最后一次 `git fetch` 时尚未进入
  GitHub，因此不能声称已本地重建该包。
- E 组当前待迁移的已确认增益只有 `task324`；必须在拿到精确 `7387.15`
  包后再替换、提交和看线上结果。
- `task050` 已确认隐藏集回退，禁止复用该候选。`task110` 已确认严重隐藏集
  回退，必须保留基线模型。

## 已接受任务

| Task | Cost | 本地增益 | 官方验证 | Fuzz | Kaggle ref | 在线结果 | 记录 |
|---|---:|---:|---:|---:|---:|---:|---|
| `task035` | `1879 -> 1743` | `+0.075132` | `266/266` | `0/500` | `54707104` | `7384.93 -> 7385.00` | `20260715_e_team738493_online7385_15_v1` |
| `task092` | `6553 -> 5631` | `+0.151636` | `265/265` | `0/500` | `54707273` | `7384.93 -> 7385.08` | 同上 |
| `task216` | `8781 -> 7137` | `+0.207298` | `266/266` | `0/500` | `54707704` | `7385.15 -> 7385.37` | `20260715_e_online7385_37_v1` |
| `task138` | `11656 -> 10096` | `+0.143682` | `266/266` | `0/500` | `54707882` | `7385.37 -> 7385.51` | `20260715_e_online7385_51_v1` |
| `task085` | `2507 -> 2299` | `+0.086613` | `265/265` | `0/500` | `54708032` | `7385.51 -> 7385.60` | `20260715_e_online7385_60_v1` |
| `task174` | `3947 -> 3306` | `+0.177217` | `266/266` | `0/500` | `54708238` | `7385.60 -> 7385.77` | `20260715_e_online7385_77_v1` |
| `task064` | `8356 -> 7507` | `+0.107144` | `267/267` | `0/5000` | `54708724` | `7385.77 -> 7385.88` | `20260715_e_online7385_88_v1` |
| `task118` | `8699 -> 8266` | `+0.051057` | `267/267` | `0/5000` | `54710570` | `7385.88 -> 7385.93` | `20260715_e_task118_isolate7385_93_v1` |
| `task324` | `8635 -> 8556` | `+0.009191` | `266/266` | `0/5000` | `54711156` | `7385.93 -> 7385.94` | `20260715_e_task324_online7385_94_v1` |

注意：队伍最高包已经包含更新后的 `task174`。在精确包上工作时，不要直接用
上表 cost `3306` 的旧 E 模型覆盖队友的新 `task174`；先比较哈希和官方 cost。

## 可复现提交链

| 阶段 | Kaggle ref | Public score | ZIP SHA256 |
|---|---:|---:|---|
| 队友精确基线 | `54686944` | `7384.93` | `8eba05f6379d98c2f99e56760468521d912d23fd0a247556bc1dc2ef2211207` |
| `task035 + task092` | `54707421` | `7385.15` | `09367f8c6bcc438c3e481178a19c1fba5ffa87f0cf301727caff3925cb801d91` |
| 加 `task216` | `54707704` | `7385.37` | `863b7090f8b0a336dea6b5960f271f123f14ccabdcc2d9fdda850673c0e1d1a2` |
| 加 `task138` | `54707882` | `7385.51` | `44136fea71b291af31d9ce6172c5ab2db3d33c6aec7d0b724af187abb1bea79a` |
| 加 `task085` | `54708032` | `7385.60` | `0b113b282c1be0832995f3df716c800200699190a6d7b6cf3add1d35a82575ef` |
| 加 `task174` | `54708238` | `7385.77` | `145d6d3b8d33c1bd4abff397a2c38d324acb973926b7f38c8084bf160780ceb6` |
| 加 `task064` | `54708724` | `7385.88` | `c2196b88f5293b545dd089c802b327408ccb6e37f7841846aec6e00873e1032b` |
| `task118` 单项验证 | `54710570` | `7385.93` | `811be4a604d026a25384b071e10d08d607071b5e118fa725360697e55fc3bbc9` |
| `task324` 单项验证 | `54711156` | `7385.94` | `86190680ef9bbca8686a25868f909782056b217f97a38fa6183408d507eac1ec` |

`20260715_e_online7385_88_v1/submission/submission.zip` 是 GitHub 中最后一份
连续累积、线上确认且完整保存的 E 链路包。后两个目录是单项隔离包，不是当前
队伍最高包。

## 明确拒绝或暂缓

| Task/批次 | 本地情况 | Kaggle/额外验证 | 结论 |
|---|---|---|---|
| `task110` | `9682 -> 7326`，released examples 全过 | ref `54685012`: `7379.41 -> 7363.58` (`-15.83`) | 严格拒绝，保留基线 |
| `task012` | `2251 -> 820`，`265/265` | color permutation `69/500` mismatch | 暂缓，不提交 |
| `task233` aggressive | `30384 -> 26136`，`266/266` | `12/147` comparable fuzz mismatch；另 353 次父模型 runtime reject | 暂缓，不提交 |
| `task138` canonical one-pool | `266/266` | cost `10096 -> 12963` | 拒绝 cost 回退 |
| `task198` threshold gather | `266/266` | cost `7922 -> 7922` | 拒绝，无收益 |
| `task198` selected family | `266/266` | cost `7922 -> 8762` | 拒绝 cost 回退 |
| `task324+050+013` | 本地合计 `+0.019061`，fuzz `0/15000` | ref `54711035`: `7385.93 -> 7385.91` | 拒绝组合包 |
| `task050` | `3000 -> 2981`，`271/271`，fuzz `0/5000` | ref `54711267`: `7385.93 -> 7385.89` | 隐藏集回退，拒绝 |
| `task013` | `1424 -> 1419`，`267/267`，fuzz `0/5000` | ref `54711365`: `7385.93 -> 7385.93` | 暂缓，无可见线上增益 |

三任务隐藏集拆分的完整结构化记录见
[`e_batch3_hidden_audit_20260715.json`](./e_batch3_hidden_audit_20260715.json)。

## 复现入口

### 1. 官方全量计分

本机官方数据实际位于 `F:\kaggle\neurogolf-2026\data`。运行前显式设置：

```powershell
$env:NEUROGOLF_DATA_ROOT='F:\kaggle\neurogolf-2026\data'
$env:NEUROGOLF_UTILS_PATH='F:\kaggle\neurogolf-2026\data\neurogolf_utils\neurogolf_utils.py'
python 'workplace E\e_score_full67_20260715.py' `
  --onnx-dir '<400-model-directory>' `
  --output '<score.csv>' `
  --workers 6
```

`e_score_full67_20260715.py` 会读取
`assignments/task_assignment_400.csv`，对 E 的 67 个槽位跑完整 released examples。

### 2. 颜色置换差分验证

```powershell
python 'workplace E\e_color_permutation_fuzz_20260715.py' `
  --parent-dir '<parent-model-directory>' `
  --candidate 'taskNNN=<candidate.onnx>' `
  --trials 5000 `
  --output '<fuzz.csv>'
```

Fuzz 只能降低风险，不能代替 Kaggle 隐藏集验证；`task050` 已证明 released examples
和 5000 次颜色置换全部通过仍可能线上回退。

### 3. 构建完整包

使用 `e_build_override_package_20260710.py`，从精确 400-task 父 ZIP 加显式 override。
构建后必须检查：

- 恰好 `400` 个条目；
- 文件名是扁平的 `task001.onnx` 到 `task400.onnx`；
- 条目唯一，ZIP CRC 通过；
- 父包 SHA、override 模型 SHA 和输出 ZIP SHA 都进入记录。

### 4. Kaggle 在线闭环

```powershell
kaggle competitions submit -c neurogolf-2026 `
  -f '<path>\submission.zip' `
  -m '<parent ref, changed tasks, local delta, fuzz, sha prefix>'
kaggle competitions submissions -c neurogolf-2026 -v
```

每次最多改 3 个 task。若组合结果异常，立即从同一精确父包做单 task 隔离提交。
线上结果 `COMPLETE` 之前，不把候选写成 accepted。

## 关键脚本和记录

- 全历史：`worklog.md`
- 当前规则和状态：`readme.md`
- 2026-07-15 全 67 task 扫描：`e_round_20260715_summary.json`
- 队伍最高包组件共享盘筛选：`e_team_component_scan_20260715.json` 和
  `e_team_component_quickscore_20260715.csv`
- Research-only 全局拼装清单：
  `artifacts/e_research_champion_20260715/manifest.json`（禁止直接提交）
- 候选差分验证：`e_color_permutation_fuzz_20260715.py`
- 当前全量计分：`e_score_full67_20260715.py`
- `task035`: `e_optimize_task035_compact_scatter_20260713.py`
- `task092`: `e_optimize_task092_deep_20260713.py`
- `task216`: `task216_20260713_optimize.py`
- `task138`: `e_optimize_task138_deep_20260713.py`
- `task085`: `e_optimize_task085_signed_pad_20260711.py`
- `task174`: 对应模型和结果保存在 `20260715_e_online7385_77_v1`
- `task064`: `e_optimize_task064_complement_equal_20260715.py`
- `task118`: `e_optimize_task118_deep_20260713.py`
- `task198` 拒绝实验：`e_optimize_task198_structural_20260715.py`
- `task138` 拒绝实验：`e_optimize_task138_canonical_pool_20260715.py`
- 2025 ARC code-golf 参考整理：`e_analyze_arc_golf_2025_20260713.py` 和
  `e_arc_golf_2025_reference_20260713.csv`

## 数据和合规

- 未把闭源 Kaggle notebook 代码复制到 GitHub。
- 公共 Kaggle 来源的扫描、选择和结果保存在 E 目录；大型外部下载只保存在
  `F:\kaggle\neurogolf-2026\external`，未提交到 GitHub。
- 2026-07-14 快速批次使用的私人 Kaggle dataset/notebook 名称和提交结果记录在
  `e_fast_batch3_submission_20260714.json`；该批次后来因 `task110` 隐藏集回退被拒绝。
- Kaggle 凭据、token、`.env` 和 `kaggle.json` 不得提交。

## 下一步

1. 等队友把 ref `54711326` (`7387.15`) 的精确 `submission.zip` 或等价 400 模型目录推到 GitHub。
2. 校验包 SHA、400-task 清单，并全量重算 E 的 67 个 task。
3. 只替换已线上通过的 `task324`，不要带入 `task050` 或 `task013`。
4. 提交 Kaggle，等待 `COMPLETE`，根据实际结果更新对应 record 目录。
5. 继续优化时优先高 cost 且结构明确的 E task；每个候选依次执行官方全量验证、差分 fuzz、单项或至多三项在线验证。

补充：为尝试重建队伍最高包，已在共享盘扫描
`task118/task133/task174/task249/task315/task321`，共发现 3721 个文件、255 个
唯一模型哈希，并对每个唯一模型执行 3 例快速筛选。该筛选只用于候选排序，不能
代替全量验证；最终没有识别出可证明等同于 ref `54711326` 的完整包。

另有一份 400-task research-only 拼装目录。它包含后来已拒绝的
`task012/task050/task233`，因此 GitHub 只保存 manifest 和风险说明，不保存 400 个
重复 ONNX，也绝不能把该目录打包提交 Kaggle。

本次整理没有删除任何文件。历史上删除过的生成缓存和临时 probe 已在
`worklog.md` 对应日期明确记录。
