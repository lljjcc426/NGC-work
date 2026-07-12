from __future__ import annotations
import csv, sys
from pathlib import Path
HERE=Path(__file__).resolve(); TASK_DIR=HERE.parents[1]; REPO=HERE.parents[4]; COMMON=REPO/'workplace C'/'neurogolf-2026-work'/'scripts'; sys.path.insert(0,str(COMMON))
from c_conv_support_crop import build
from c_score_common import score_onnx
TASK='task201'; BASE=TASK_DIR/'onnx'/'task201_compact_pad_axes.onnx'; OUT=TASK_DIR/'onnx'/'task201_compact_pad_axes_conv_crop.onnx'
if __name__=='__main__':
 new_path=build(BASE,OUT); old=score_onnx(TASK,BASE,True); new=score_onnx(TASK,new_path,True); row={'task':TASK,'method':'compact_pad_plus_conv_crop','old_cost':old.cost,'new_cost':new.cost,'delta_cost':new.cost-old.cost if new.cost is not None else None,'old_points':old.points,'new_points':new.points,'delta_points':new.points-old.points if new.points is not None else None,'examples_passed':new.examples_passed,'examples_checked':new.examples_checked,'local_valid':str(new.ok).lower(),'accepted':str(bool(new.ok and new.cost<old.cost)).lower(),'artifact_path':str(new_path)}; p=TASK_DIR/'reports'/'cost_diff_round2.csv'; f=p.open('w',newline='',encoding='utf-8'); w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row);f.close();print(row)
