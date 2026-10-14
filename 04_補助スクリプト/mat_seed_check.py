# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR（05 フォルダ）：最終モデルを、乱数シード（最適化の初期値）を変えて学習し直したときの、
対数周辺尤度（LML）と 3data の RMSE のばらつきを調べる（補助の確認。ノートの実行が終わった後で行う）。

使い方（リポジトリ直下から）: uv run python 04_補助スクリプト/mat_seed_check.py [cond|shot] [シード数（既定 5）] [最適化のやり直し回数（既定 20。ノートの N_RESTARTS と同じ）]
前提：ノートを、同じ INPUT_MODE で実行済みであること（<材料フォルダ>_学習結果/<入力の取り方>/比較/シフトファクターの推定.csv の係数 θ を読む）。
やること：θ は、ノートの結果に固定する。カーネルのハイパーパラメータは、尤度の格子の初期値を使わず（theta0 = None）、
最適化のやり直しの乱数の初期値だけで学習する（シードを 0, 1, … と変える）。LML の最大と最小の差が小さければ、最適化はほぼ同じ解に着いている。
RMSE のばらつきが、モデルどうしの差より小さければ、モデルの比較は、最適化の運に左右されていない。
結果：<材料フォルダ>_学習結果/<入力の取り方>/比較/乱数シードの確認.csv（1 行が 1 回の学習）と、乱数シードの確認_まとめ.csv（モデルごとの最小・最大）
"""
import os
import sys
import time

os.environ["NB_REPO"] = "."
mode = sys.argv[1] if len(sys.argv) > 1 else "cond"
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
n_restarts = int(sys.argv[3]) if len(sys.argv) > 3 else 20
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

import mat_common as mc

try:
    fm.findfont("IPAexGothic", fallback_to_default=False)
except ValueError:
    raise RuntimeError("日本語フォント IPAexGothic が見つかりません。インストールしてから、もう一度実行してください。")
plt.rcParams["font.family"] = ["IPAexGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
mc.INPUT_MODE = mode
MATS = ["PP", "MIX", "GPPS"]
mc.D = mc.load_all(MATS)
FOLD = mc.MAT_FOLDERS
MODES = {"cond": "条件平均の入力", "shot": "ショットごとの入力"}
for mat in MATS:
    tr = mc.D[(mat, "5data")]
    te = mc.add_groups(tr, mc.D[(mat, "3data")])
    cfg = dict(mc.SHIFT_CFG[mat])
    cmp_dir = os.path.join(mc.DATA_ROOT, FOLD[mat], f"{FOLD[mat]}_学習結果", MODES[mode], "比較")
    sh = pd.read_csv(os.path.join(cmp_dir, "シフトファクターの推定.csv"), encoding="utf-8-sig").set_index("モデル")
    rows = []
    for key in ["B_Tset", "B_Tff", "C"]:
        th, tv = (sh.loc[key, "th_best"], sh.loc[key, "tv_best"]) if key != "C" else (0.0, 0.0)
        for seed in range(n_seeds):
            mc.SEED = seed
            t0 = time.time()
            f = mc.fit_gp(key, tr, cfg, th, tv, restarts=n_restarts, theta0=None, mode=mode)
            mu, sd, sdf, nsd = mc.predict(f, te)
            r = {x["範囲"]: x for x in mc.metrics_rows(mat, key, te, mu, sd, sdf)}
            rows.append({"材料": mc.MAT_NAMES[mat], "入力": MODES[mode], "モデル": key, "シード": seed, "LML": float(f["gp"].log_marginal_likelihood_value_),
                         "RMSE 全体": r["全体"]["RMSE"], "RMSE 低速側": r["低速側"]["RMSE"], "RMSE 高速側": r["高速側"]["RMSE"], "秒": round(time.time() - t0)})
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    mc.to_csv(t, os.path.join(cmp_dir, "乱数シードの確認.csv"))
    s = t.groupby("モデル", sort=False).agg(LML最大=("LML", "max"), LML最小=("LML", "min"), RMSE全体の最小=("RMSE 全体", "min"), RMSE全体の最大=("RMSE 全体", "max"),
                                          RMSE低速側の最小=("RMSE 低速側", "min"), RMSE低速側の最大=("RMSE 低速側", "max")).reset_index()
    s.insert(0, "材料", mc.MAT_NAMES[mat])
    mc.to_csv(s, os.path.join(cmp_dir, "乱数シードの確認_まとめ.csv"))
    print(s.round(4).to_string(), flush=True)
    # 図：横軸 LML、縦軸 3data の RMSE（点が 1 回の学習。同じ LML なのに RMSE が大きく違う点があれば、最適化の運に左右されている）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, col, ttl in [(axes[0], "RMSE 全体", "3data 全体"), (axes[1], "RMSE 低速側", "3data の低速側（3・5 mm/s。外挿）")]:
        for key, mk, c in [("B_Tset", "o", "#1f77b4"), ("B_Tff", "s", "#d62728"), ("C", "^", "#2ca02c")]:
            u = t[t["モデル"] == key]
            ax.scatter(u["LML"], u[col], marker=mk, s=110, color=c, alpha=0.75, edgecolors="k", linewidths=0.6, label=key)
        ax.set_xlabel("対数周辺尤度 LML（右ほど、データによく合う）")
        ax.set_ylabel(f"RMSE（ln η）：{ttl}")
        ax.set_title(ttl)
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend()
    fig.suptitle(f"{mc.MAT_NAMES[mat]}　乱数シード {n_seeds} 通り（最適化のやり直し {n_restarts} 回）で学習し直したときの、LML と 3data の RMSE（{MODES[mode]}）")
    mc.save_fig(fig, cmp_dir, "乱数シードの確認", show=False)
print("完了")
