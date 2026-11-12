# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR（05 フォルダ）：最終モデルを、乱数シード（最適化の初期値）を変えて学習し直したときの、
対数周辺尤度（LML）と 3data の RMSE のばらつきを調べる（補助の確認。シフトファクター検討用のノートの実行が終わった後で行う）。

使い方（リポジトリ直下から）: uv run python 04_補助スクリプト/mat_seed_check.py [cond|shot] [シード数（既定 5）] [最適化のやり直し回数（既定 20。ノートの N_RESTARTS と同じ）] [材料=学習結果フォルダ …]
材料（PP・MIX・GPPS）ごとに、シフトファクター検討用のノートで選んだ学習結果フォルダを、ダイアログで選ぶ（キャンセルした材料は飛ばす）。
引数に「PP=<フォルダ>」のように書いたときは、書いた材料だけを調べ、ダイアログは出さない（相対パスはリポジトリ直下から）。
前提：シフトファクター検討用のノートを、同じ入力の取り方で実行済みであること（<学習結果フォルダ>/<版>/全厚み総合/シフトファクターの検討/ の
シフトファクターの推定.csv の係数 θ と、学習済みモデルに記録された学習データ・検証データ・シフトファクターの設定を読む。設定は、モデルごとに、その学習済みモデルから読む〔2026-11-02 から、型がモデルごとに違うため〕）。
やること：θ は、ノートの結果に固定する。カーネルのハイパーパラメータは、尤度の格子の初期値を使わず（theta0 = None）、
最適化のやり直しの乱数の初期値だけで学習する（シードを 0, 1, … と変える）。LML の最大と最小の差が小さければ、最適化はほぼ同じ解に着いている。
RMSE のばらつきが、モデルどうしの差より小さければ、モデルの比較は、最適化の運に左右されていない。
結果：同じフォルダ（シフトファクターの検討/）の 乱数シードの確認.csv（1 行が 1 回の学習）、乱数シードの確認_まとめ.csv（モデルごとの最小・最大）、乱数シードの確認.png
2026-11-02：検証データにない範囲（低速側・高速側・範囲内）は、記録しない形にした（温度適切版のデータは、検証が高速側〔280・320 mm/s〕だけで、低速側がない）。
まとめの表と図の 2 枚目は、外挿の範囲のうち、低速側があれば低速側（通常版。前と同じ出力）、なければ高速側を使う。
2026-10-27：結果の新しい構成（学習結果フォルダをダイアログで選ぶ）に合わせた。10-26 までの結果（<材料フォルダ>/<番号>_学習結果/<材料フォルダ>_学習結果(5all学習データ)/<入力の取り方>/比較/。
2026-11-02 に「_学習結果」から名前を変え、2026-11-03 に <番号>_学習結果/ の下に移した。今は <材料フォルダ>/<番号>_学習結果/1回目_<材料の名前>_学習結果/〔2026-11-12 に確認。材料フォルダは「<材料の名前>(データと結果)」〕）の確認は、その時点のこのスクリプト（Git の履歴）で行ったもの。
2026-11-03：図の点の形と色を、モデルが 5 つより多くても足りるようにした（それまでは 4 つ分しかなく、シフトのあるモデルが 5 つ以上だと、図から抜けるモデルが出て、C の凡例も別のモデルに付いた）。
"""
import os
import sys
import time

os.environ["NB_REPO"] = "."
args = [a for a in sys.argv[1:] if "=" not in a]
given = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
mode = args[0] if len(args) > 0 else "cond"
n_seeds = int(args[1]) if len(args) > 1 else 5
n_restarts = int(args[2]) if len(args) > 2 else 20
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib
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
MATS, STUDY, TAB, CFG = [], {}, {}, {}
for mat in [m for m in ["PP", "MIX", "GPPS"] if m in given or not given]:     # 引数で学習結果フォルダを書いたときは、その材料だけ
    try:
        mc.choose_inputs([mat], {mat: {"結果": given[mat]}} if mat in given else {}, need_data=False)
    except RuntimeError:                                       # ダイアログでキャンセルした材料は飛ばす
        print(f"{mc.MAT_NAMES[mat]}：学習結果フォルダが選ばれなかったので、飛ばします", flush=True)
        continue
    STUDY[mat] = mc.result_dir(mat, mc.VERSION_DIRS[mode], mc.THICK_TOTAL, mc.STUDY_DIR, make=False)
    TAB[mat] = pd.read_csv(os.path.join(STUDY[mat], "シフトファクターの推定.csv"), encoding="utf-8-sig").set_index("モデル")
    obj = joblib.load(os.path.join(STUDY[mat], mc.MODEL_DIRS["C"], "学習済みモデル_C.joblib"))
    if obj["材料"] != mc.MAT_NAMES[mat] or obj["fit"].get("mode") != mode:
        raise ValueError(f"{STUDY[mat]} は、{mc.MAT_NAMES[mat]}・{mc.VERSION_DIRS[mode]} の結果ではありません（{obj['材料']}・{obj['入力の取り方']}）")
    mc.data_from_model(mat, obj)                               # 学習データ・検証データは、学習済みモデルに記録された場所から読む
    for key in list(TAB[mat].index) + ["C"]:                   # シフトファクターの設定は、モデルごとに、その学習済みモデルの記録から取る（型がモデルごとに違うため。2026-11-02）
        CFG[(mat, key)] = dict(joblib.load(os.path.join(STUDY[mat], mc.MODEL_DIRS[key], f"学習済みモデル_{key}.joblib"))["シフトファクター"])
    MATS.append(mat)
mc.D = mc.load_all(MATS)
MARKS = [("o", "#1f77b4"), ("s", "#d62728"), ("D", "#9467bd"), ("v", "#8c564b"), ("P", "#e377c2"), ("X", "#7f7f7f"), ("<", "#bcbd22"), (">", "#17becf")]   # シフトのあるモデルの点（C は緑の ▲）
for mat in MATS:
    tr = mc.D[(mat, "5data")]
    te = mc.add_groups(tr, mc.D[(mat, "3data")])
    sh = TAB[mat]
    keys = list(sh.index) + ["C"]
    groups = [g for g in ["低速側", "高速側", "範囲内"] if (te["group"] == g).any()]     # 検証データにある範囲（温度適切版は高速側だけ）
    side = next((g for g in ["低速側", "高速側"] if g in groups), None)                  # まとめと図の 2 枚目に使う外挿の範囲（低速側があれば低速側）
    rows = []
    for key in keys:
        th, tv = (sh.loc[key, "th_best"], sh.loc[key, "tv_best"]) if key != "C" else (0.0, 0.0)
        for seed in range(n_seeds):
            mc.SEED = seed
            t0 = time.time()
            f = mc.fit_gp(key, tr, CFG[(mat, key)], th, tv, restarts=n_restarts, theta0=None, mode=mode)
            mu, sd, sdf, nsd = mc.predict(f, te)
            r = {x["範囲"]: x for x in mc.metrics_rows(mat, key, te, mu, sd, sdf)}
            rows.append({"材料": mc.MAT_NAMES[mat], "入力": mc.VERSION_DIRS[mode], "モデル": key, "シード": seed, "LML": float(f["gp"].log_marginal_likelihood_value_),
                         "RMSE 全体": r["全体"]["RMSE"], **{f"RMSE {g}": r[g]["RMSE"] for g in groups}, "秒": round(time.time() - t0)})
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    mc.to_csv(t, os.path.join(STUDY[mat], "乱数シードの確認.csv"))
    s = t.groupby("モデル", sort=False).agg(LML最大=("LML", "max"), LML最小=("LML", "min"), RMSE全体の最小=("RMSE 全体", "min"), RMSE全体の最大=("RMSE 全体", "max"),
                                          **({f"RMSE{side}の最小": (f"RMSE {side}", "min"), f"RMSE{side}の最大": (f"RMSE {side}", "max")} if side else {})).reset_index()
    s.insert(0, "材料", mc.MAT_NAMES[mat])
    mc.to_csv(s, os.path.join(STUDY[mat], "乱数シードの確認_まとめ.csv"))
    print(s.round(4).to_string(), flush=True)
    # 図：横軸 LML、縦軸 3data の RMSE（点が 1 回の学習。同じ LML なのに RMSE が大きく違う点があれば、最適化の運に左右されている）
    panels = [("RMSE 全体", "3data 全体")]
    if side:
        vs = "・".join(f"{v:g}" for v in sorted(te.loc[te["group"] == side, "v"].unique()))
        panels.append((f"RMSE {side}", f"3data の{side}（{vs} mm/s。外挿）"))
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5.8), squeeze=False)
    for ax, (col, ttl) in zip(axes[0], panels):
        for key, (mk, c) in zip(keys, [MARKS[i % len(MARKS)] for i in range(len(keys) - 1)] + [("^", "#2ca02c")]):
            u = t[t["モデル"] == key]
            ax.scatter(u["LML"], u[col], marker=mk, s=110, color=c, alpha=0.75, edgecolors="k", linewidths=0.6, label=key)
        ax.set_xlabel("対数周辺尤度 LML（右ほど、データによく合う）")
        ax.set_ylabel(f"RMSE（ln η）：{ttl}")
        ax.set_title(ttl)
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend()
    fig.suptitle(f"{mc.MAT_NAMES[mat]}　乱数シード {n_seeds} 通り（最適化のやり直し {n_restarts} 回）で学習し直したときの、LML と 3data の RMSE（{mc.VERSION_DIRS[mode]}）")
    mc.save_fig(fig, STUDY[mat], "乱数シードの確認", show=False)
print("完了")
