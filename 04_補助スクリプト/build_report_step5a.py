# -*- coding: utf-8 -*-
"""手順5a の説明書（用語集と図の見方）を作る。

使い方（リポジトリ直下から）: uv run python 04_補助スクリプト/build_report_step5a.py
results/step5a の CSV から数値を取り出して report_step5a_template.tex に入れ、_tmp/report_5a で
LuaLaTeX を2回動かし、できた PDF と tex を results/学習結果説明_手順5a.pdf・.tex に置く。
"""
import os

import numpy as np
import pandas as pd

import report_common as rc
from report_common import MAT, num, pct, rd, rmse

S = "step5a"
best, met, eff, obs, sens = rd(S, "best_E.csv"), rd(S, "metrics.csv"), rd(S, "tset_effect.csv"), rd(S, "observed_ratio.csv"), rd(S, "sensitivity_E.csv")
prof = rd(S, "profile_likelihood.csv")
V = {}
MAT_JA_ = rc.MAT_JA


def B(mat, sname, st):
    r = best[(best["material"] == MAT[mat]) & (best["set"] == sname) & (best["structure"] == st)]
    assert len(r) == 1, (mat, sname, st)
    return r.iloc[0]


def M(mat, split, variant, sub, col="RMSE", mode="cm"):
    r = met[(met["material"] == MAT[mat]) & (met["split"] == split) & (met["mode"] == mode) & (met["variant"] == variant) & (met["subset"] == sub)]
    assert len(r) == 1, (mat, split, variant, sub, mode)
    return float(r[col].iloc[0])


def e1(x):
    return f"{x:.1f}".replace("-", "$-$") if x < 0 else f"{x:.1f}"


# ---- 基本の数値
V["SLOPE"] = "$-1.6$"
V["R40"] = f"{np.exp(0.0206 * 40):.1f}"
V["PER_KJ"] = f"{(np.exp(0.0206) - 1) * 100:.1f}"
V["R_LIT"] = V["R40"]
V["N_CSV"] = str(len([f for f in os.listdir(os.path.join(rc.R, S)) if f.endswith(".csv")]))
V["N_FIG"] = str(len([f for f in os.listdir(os.path.join(rc.R, S)) if f.endswith(".png")]))
ea = {m: float(B(m, "S_V", "A")["E_best"]) for m in MAT}
V["EA_PP"], V["EA_MIX"] = e1(ea["PP"]), e1(ea["MIX"])
V["EA_ALL"] = "、".join(f"{MAT_JA_[m]} " + "・".join(e1(float(B(m, s, "A")["E_best"])) for s in ["S_V", "S_T", "ALL"]) for m in MAT)
V["DA_PP"], V["DA_MIX"] = (f"{float(B(m, 'S_V', 'A')['dLML_vs_E0']):.0f}" for m in MAT)
V["EH_PP"], V["EH_MIX"] = (", ".join(e1(float(B(m, s, "H")["E_best"])) for s in ["S_V", "S_T", "ALL"]) for m in MAT)
V["EH_PP_ALL"] = "S\\_V・S\\_T・ALL の学習で $E$ = " + V["EH_PP"] + " kJ/mol"
V["DH_PP"], V["DH_MIX"] = (f"{float(B(m, 'S_V', 'H')['dLML_vs_E0']):.0f}" for m in MAT)
V["H40DROP_PP"], V["H40DROP_MIX"] = (f"{float(B(m, 'S_V', 'H')['LML_best'] - B(m, 'S_V', 'H')['LML_at_E40']):.0f}" for m in MAT)
V["RATIO_LIT"] = f"{40 / np.mean(list(ea.values())):.0f}"
V["RATIO_A"] = f"{np.exp(0.0206 * np.mean(list(ea.values()))):.2f}"
for m in MAT:
    r = B(m, "S_V", "B")
    V[f"BH_{m}"], V[f"BV_{m}"] = f"{r['Eh_best']:.0f}", f"{r['Ev_best']:.0f}"
    V[f"DB_{m}"] = f"{float(r['LML_best'] - B(m, 'S_V', 'A')['LML_best']):.0f}"
o = obs[obs["material"] == "PP"].set_index("H")["observed"]
V["OBS_05"], V["OBS_20"] = f"{o[0.5]:.2f}", f"{o[2.0]:.2f}"

# ---- 感度確認の文
big = sens["影響が大きい"].astype(bool).any()
move = sens["動き [kJ/mol]"].abs().max()
V["SENS_TXT"] = (f"A は変わらず（動き 0）、H も最大 {move:.1f} kJ/mol しか動かず、格子 1 目（2.5 kJ/mol）に届きません。P02 は、$E$ の推定に大きく影響しません" if not big
                 else "格子 1 目以上動いたものがあります。P02 だけを除いたデータファイルを作って、確かめます")

# ---- 表：最良の E
SET_JA = {"S_V": "S\\_V の学習", "S_T": "S\\_T の学習", "ALL": "ALL"}
rows = []
for m in MAT:
    for sname in ["S_V", "S_T", "ALL"]:
        for st in ["H", "A", "Hz", "B", "F"]:
            r = B(m, sname, st)
            if st == "B":
                est = f"($\\Eh$, $\\Ev$) = ({r['Eh_best']:.0f}, {r['Ev_best']:.0f})"
            else:
                est = f"{e1(r['E_best'])}（{e1(r['E_95_lo'])}〜{e1(r['E_95_hi'])}）"
            rows.append(f"{rc.MAT_JA[m] if (sname == 'S_V' and st == 'H') else ''} & {SET_JA[sname] if st == 'H' else ''} & {st} & {est} & {r['LML_best']:.1f} & {num(float(r['dLML_vs_E0']), 1)} \\\\")
V["TABLE_BEST"] = "\n".join(rows)

# ---- 表：検証の指標（cm）
VARS = ["C", "N", "H", "H40", "A", "Hz", "B", "F"]
cols = [("S_V", "低速側"), ("S_V", "高速側"), ("S_T", "全体"), ("S_T3", "全体")]
for m in MAT:
    rows = []
    for v in VARS:
        c = [M(m, s, v, sub, "RMSE") for s, sub in cols] + [M(m, s, v, sub, "NLPD") for s, sub in cols]
        rows.append(f"{v} & " + " & ".join(rmse(x) for x in c[:4]) + " & " + " & ".join(num(x, 2) for x in c[4:]) + " \\\\")
    V[f"TABLE_M_{m}"] = "\n".join(rows)

# ---- 表：設定温度の効果
rows = []
for m in MAT:
    o = obs[obs["material"] == MAT[m]].sort_values("H")["observed"].tolist()
    rows.append(f"{rc.MAT_JA[m]} & 実測 & " + " & ".join(f"{x:.3f}" for x in o) + " \\\\")
    for v in ["C", "A", "Hz", "B", "H", "H40"]:
        e = eff[(eff["material"] == MAT[m]) & (eff["split"] == "S_V") & (eff["mode"] == "cm") & (eff["variant"] == v)].sort_values("H")["ratio_190_230"].tolist()
        rows.append(f" & {v} & " + " & ".join(f"{x:.3f}" for x in e) + " \\\\")
V["TABLE_EFF"] = "\n".join(rows)

# ---- 表：感度確認
rows = []
for _, r in sens.iterrows():
    rows.append(f"{rc.MAT_JA['PP' if r['material'] == 'PP' else 'MIX'] if r['structure'] == 'H' else ''} & {r['structure']} & {e1(r['E_best_すべて'])} & {e1(r['E_best_P02を除く'])} & {num(float(r['動き [kJ/mol]']), 1)} \\\\")
V["TABLE_SENS"] = "\n".join(rows)

# ---- 本命と対案（main_vs_alt.csv）
cmp = rd(S, "main_vs_alt.csv")
V["MAIN"] = cmp[cmp["role"] == "本命"]["structure"].iloc[0]
V["ALT"] = cmp[cmp["role"] == "対案"]["structure"].iloc[0]
if (V["MAIN"], V["ALT"]) != ("B", "A"):
    print(f"★本命・対案が B・A ではありません（{V['MAIN']}・{V['ALT']}）。説明書の文章は、B・A を前提に書いてあるので、確認してください")
rows = []
for m in MAT:
    for role in ["本命", "対案", "基準"]:
        r = cmp[(cmp["material"] == MAT[m]) & (cmp["role"] == role)].iloc[0]
        e = "---" if np.isnan(r["Ev_ALL"]) else f"({r['Eh_ALL']:.1f}, {r['Ev_ALL']:.1f})".replace("(-", "($-$").replace(", -", ", $-$")
        lml = "---" if np.isnan(r["LML_S_V"]) else f"{r['LML_S_V']:.1f}"
        dl = "---" if np.isnan(r["dLML_vs_E0_S_V"]) else f"{r['dLML_vs_E0_S_V']:.1f}"
        rows.append(f"{rc.MAT_JA[m] if role == '本命' else ''} & {role} & {r['structure']} & {e} & {lml} & {dl} & " +
                    " & ".join(rmse(r[f"RMSE_{lab}"]) for lab in ["S_V_low", "S_V_high", "S_T", "S_T3"]) + " \\\\")
V["TABLE_MAIN"] = "\n".join(rows)
labs = ["S_V_low", "S_V_high", "S_T", "S_T3"]
rr = {}
for m in MAT:
    a = cmp[(cmp["material"] == MAT[m]) & (cmp["role"] == "本命")].iloc[0]
    b = cmp[(cmp["material"] == MAT[m]) & (cmp["role"] == "対案")].iloc[0]
    rr[m] = [a[f"RMSE_{lab}"] / b[f"RMSE_{lab}"] for lab in labs]
    V[f"RR_{m}"] = f"{min(rr[m]):.2f}〜{max(rr[m]):.2f}"
    dev = {}
    obs_m = cmp[(cmp["material"] == MAT[m]) & (cmp["role"] == "実測")].iloc[0]
    for role in ["本命", "対案"]:
        t = cmp[(cmp["material"] == MAT[m]) & (cmp["role"] == role)].iloc[0]
        dev[role] = max(abs(t[f"ratio_{h}"] - obs_m[f"ratio_{h}"]) for h in [0.5, 1.0, 1.5, 2.0])
    V[f"DEV_B_{m}"], V[f"DEV_A_{m}"] = f"{dev['本命']:.3f}", f"{dev['対案']:.3f}"
worst = max(max(v) for v in rr.values())
V["RR_TXT"] = "どの検証でも同等（1.05 倍以内）" if worst <= 1.05 else f"B のほうが当たる検証もあれば、A のほうが当たる検証もある（B の誤差が A の最大 {worst:.2f} 倍）。どちらが常に良い、とは言えない"
V["EB_ALL"] = "、".join(f"{MAT_JA_[m]} ($\\Eh$, $\\Ev$) = " + "・".join(f"({B(m, s, 'B')['Eh_best']:.0f}, {B(m, s, 'B')['Ev_best']:.0f})" for s in ["S_V", "S_T", "ALL"]) for m in MAT)
V["EHZ_PP"], V["EHZ_MIX"] = (", ".join(e1(float(B(m, s, "Hz")["E_best"])) for s in ["S_V", "S_T", "ALL"]) for m in MAT)

rc.build_pdf("5a", S, "report_step5a_template.tex", V, "gakushu5a")
