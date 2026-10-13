# -*- coding: utf-8 -*-
"""手順3b の説明書（用語集と図の見方）を作る。

使い方（リポジトリ直下から）: uv run python 04_補助スクリプト/build_report_step3b.py
results/step3b の CSV から数値を取り出して report_step3b_template.tex に入れ、_tmp/report_3b で
LuaLaTeX を2回動かし、できた PDF と tex を results/学習結果説明_手順3b.pdf・.tex に置く。
"""
import os

import numpy as np
import pandas as pd

import report_common as rc
from report_common import MAT, MAT_JA, num, rd, rmse

S = "step3b"
met, ker, loco, eff, obs, sens = (rd(S, f) for f in ["metrics.csv", "kernels.csv", "loco_metrics.csv", "tset_effect.csv", "observed_ratio.csv", "sensitivity_judgement.csv"])
dsum = rd("step3a", "data_summary.csv")
ALL = ["L0", "L0T", "B0", "B0T", "B0F", "B0FT", "B2", "B2T", "B2F", "B2FT", "R0", "R2", "R3", "R3F", "R3P"]
GPR = [k for k in ALL if k[0] != "L"]
TSET = ["L0T", "B0T", "B0FT", "B2T", "B2FT", "R2", "R3", "R3F", "R3P"]
HI = "高速側（感度確認の条件を除く）"
V = {}


def M(mat, split, model, sub, col="RMSE", data="なし"):
    r = met[(met["material"] == MAT[mat]) & (met["split"] == split) & (met["model"] == model) & (met["subset"] == sub) & (met["data"] == data)]
    return float(r[col].iloc[0]) if len(r) == 1 else np.nan


def K(mat, model, col, split="S_V", data="なし"):
    r = ker[(ker["material"] == MAT[mat]) & (ker["split"] == split) & (ker["model"] == model) & (ker["data"] == data)]
    return r[col].iloc[0] if len(r) == 1 else np.nan


def ds(mat, grp, col, o="なし"):
    r = dsum[(dsum["材料"] == MAT[mat]) & (dsum["組"] == grp) & (dsum["外れ値"] == o)]
    return r[col].iloc[0]


def fm(x, nd=2):
    return "---" if x is None or (isinstance(x, float) and np.isnan(x)) else num(float(x), nd)


V["N_CSV"] = str(len([f for f in os.listdir(os.path.join(rc.R, S)) if f.endswith(".csv")]))
V["N_FIG"] = str(len([f for f in os.listdir(os.path.join(rc.R, S)) if f.endswith(".png")]))
for m in MAT:
    V[f"{m}_NTR"], V[f"{m}_NCTR"] = f"{int(ds(m, '5data', '行数')):,}", str(int(ds(m, "5data", "条件数")))
    V[f"{m}_NTE"], V[f"{m}_NCTE"] = f"{int(ds(m, '3data', '行数')):,}", str(int(ds(m, "3data", "条件数")))

# ---- 表：S_V
for m in MAT:
    rows = []
    for k in ALL:
        hi = M(m, "S_V", k, HI)
        rows.append(f"{k} & {rmse(M(m, 'S_V', k, '低速側'))} & {rmse(M(m, 'S_V', k, '高速側'))} & {rmse(hi)} & {M(m, 'S_V', k, '低速側', 'coverage95'):.2f} & {M(m, 'S_V', k, '高速側', 'coverage95'):.2f} & "
                    f"{fm(M(m, 'S_V', k, '低速側', 'NLPD'))} & {fm(M(m, 'S_V', k, '高速側', 'NLPD'))} \\\\")
    V[f"TABLE_SV_{m}"] = "\n".join(rows)

# ---- 表：S_T
rows = []
for k in ALL:
    c = []
    for m in MAT:
        for sp in ["S_T", "S_T3"]:
            c.append(f"{rmse(M(m, sp, k, '全体'))}／{fm(M(m, sp, k, '全体', 'NLPD'))}")
    rows.append(f"{k} & " + " & ".join(c) + " \\\\")
V["TABLE_ST"] = "\n".join(rows)

# ---- 表：設定温度の効果
rows = []
for m in MAT:
    o = obs[obs["material"] == MAT[m]].sort_values("H")["observed"].tolist()
    rows.append(f"{MAT_JA[m]} & 実測 & " + " & ".join(f"{x:.3f}" for x in o) + " \\\\")
    for k in TSET:
        e = eff[(eff["material"] == MAT[m]) & (eff["split"] == "S_V") & (eff["model"] == k)].sort_values("H")["ratio_190_230"].tolist()
        if e:
            rows.append(f" & {k} & " + " & ".join(f"{x:.3f}" for x in e) + " \\\\")
V["TABLE_EFF"] = "\n".join(rows)

# ---- 表：ノイズと純粋誤差
rows = []
for m in MAT:
    for k in GPR:
        flags = [sp for sp in ["S_V", "S_T"] if bool(K(m, k, "ls_at_bound", sp))]
        rows.append(f"{MAT_JA[m] if k == GPR[0] else ''} & {k} & {K(m, k, 'noise_sd'):.4f} & {K(m, k, 'pure_sd_ln_eta'):.4f} & {K(m, k, 'pure_sd_after_gd'):.4f} & "
                    f"{'、'.join(s.replace('_', chr(92) + '_') for s in flags) if flags else '---'} \\\\")
V["TABLE_NOISE"] = "\n".join(rows)

# ---- 表：感度確認
rows = []
for m in MAT:
    s = sens[(sens["material"] == MAT[m]) & sens["model"].isin(["B0T", "R3"]) & ~sens["split"].str.contains("予測比")]
    for _, r in s.iterrows():
        sub = r["subset"] if r["subset"] != "全体" else ""
        rng = f"{r['split'].replace('_', chr(92) + '_')} {sub}".strip()
        rows.append(f"{MAT_JA[m]} & {r['model']} & {rng} & {fm(r['RMSE_すべて'], 3)} & {fm(r['RMSE_感度確認の条件を除く'], 3)} & {fm(r['変化 [%]'], 0)} & {'大' if bool(r['影響が大きい']) else '小'} \\\\")
V["TABLE_SENS"] = "\n".join(rows)

# ---- 表：ランダムと LOCO
rows = []
for k in ["L0", "L0T", "B0", "B0T", "B0FT", "B2T", "R0", "R2", "R3"] + ["B0F", "R3F", "R3P"]:
    c = [M("PP", "S_R1", k, "全体"), M("PP", "S_R2", k, "全体"), M("MIX", "S_R1", k, "全体")]
    lo = [loco[(loco["material"] == MAT[m]) & (loco["model"] == k) & (loco["subset"] == "全体")]["RMSE"] for m in MAT]
    c += [float(x.iloc[0]) if len(x) else np.nan for x in lo]
    rows.append(f"{k} & " + " & ".join(rmse(x) for x in c) + " \\\\")
V["TABLE_RANDOM"] = "\n".join(rows)

exec(open(os.path.join(rc.HERE, "report_step3b_text.py"), encoding="utf-8").read())      # 文章（結果の数値を入れた説明）

rc.build_pdf("3b", S, "report_step3b_template.tex", V, "gakushu3b")
