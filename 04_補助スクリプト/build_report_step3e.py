# -*- coding: utf-8 -*-
"""手順3e の説明書（用語集と図の見方）を作る。

使い方（リポジトリ直下から）: uv run python 04_補助スクリプト/build_report_step3e.py
results/step3e の CSV から数値を取り出して report_step3e_template.tex に入れ、_tmp/report_3e で
LuaLaTeX を2回動かし、できた PDF と tex を results/学習結果説明_手順3e.pdf・.tex に置く。
"""
import os

import numpy as np
import pandas as pd

import report_common as rc
from report_common import MAT, num, rd, rmse

S = "step3e"
form, ratio, met, bias, ker, proxy = (rd(S, f) for f in ["formula_check.csv", "thickness_ratio.csv", "metrics.csv", "bias_by_thickness.csv", "kernels.csv", "p32_proxy.csv"])
V = {}
MJ = rc.MAT_JA
PP, MX = "PP", "MIX"
KEYS = ["R3", "RFo", "W0", "W1", "W1T", "W2", "W3"]


def M(mat, ver, split, model, sub="全体", col="RMSE"):
    r = met[(met["material"] == MAT[mat]) & (met["version"] == ver) & (met["split"] == split) & (met["model"] == model) & (met["subset"] == sub)]
    assert len(r) == 1, (mat, ver, split, model, sub)
    return float(r[col].iloc[0])


def F(mat, inputs, col):
    r = form[(form["material"] == MAT[mat]) & (form["inputs"] == inputs)]
    assert len(r) == 1, (mat, inputs)
    return float(r[col].iloc[0])


V["N_FIG"] = str(len([f for f in os.listdir(os.path.join(rc.R, S)) if f.endswith(".png")]))
V["N_CSV"] = str(len([f for f in os.listdir(os.path.join(rc.R, S)) if f.endswith(".csv")]))
V["N_HR"] = str(len([f for f in os.listdir(os.path.join(rc.R, S, "highres")) if f.startswith("hr_")]))

# ---- 表：算出式の量での線形回帰
FORM_JA = {"ln γ̇ + ln H": "$\\ln\\gd$, $\\ln H$", "ln γ̇ + ln H + Tset": "$\\ln\\gd$, $\\ln H$, $\\Tset$", "ln Δt + ln H": "$\\ln\\Delta t$, $\\ln H$",
           "ln γ̇ + ln H + ln ΔP（算出式）": "$\\ln\\gd$, $\\ln H$, $\\ln\\Delta P$（算出式そのもの）", "ln γ̇ + ln ΔP": "$\\ln\\gd$, $\\ln\\Delta P$",
           "ln γ̇ + ln P3_2": "$\\ln\\gd$, $\\ln\\PII$（H なし）", "ln γ̇ + ln H + ln P3_2": "$\\ln\\gd$, $\\ln H$, $\\ln\\PII$（手順3a の L2）"}
rows = []
for m in MAT:
    for k, ja in FORM_JA.items():
        r = form[(form["material"] == MAT[m]) & (form["inputs"] == k)].iloc[0]
        ch = "---" if pd.isna(r.get("coef_ln_H", np.nan)) else num(float(r["coef_ln_H"]), 2)
        rows.append(f"{MJ[m] if k == 'ln γ̇ + ln H' else ''} & {ja} & {ch} & {r['RMSE_test']:.4f} & {r['RMSE_over_pure']:.2f} \\\\")
V["TABLE_FORM"] = "\n".join(rows)
V["PURE_PP"], V["PURE_MIX"] = f"{F(PP, 'ln γ̇ + ln H', 'pure_sd'):.3f}", f"{F(MX, 'ln γ̇ + ln H', 'pure_sd'):.3f}"
V["FORM_GH_PP"], V["FORM_GH_MIX"] = f"{F(PP, 'ln γ̇ + ln H', 'RMSE_test'):.2f}", f"{F(MX, 'ln γ̇ + ln H', 'RMSE_test'):.2f}"
V["FORM_GH_X"] = f"{min(F(PP, 'ln γ̇ + ln H', 'RMSE_over_pure'), F(MX, 'ln γ̇ + ln H', 'RMSE_over_pure')):.0f}"
V["COEF_H_PP"], V["COEF_H_MIX"] = num(F(PP, "ln γ̇ + ln H", "coef_ln_H"), 2), num(F(MX, "ln γ̇ + ln H", "coef_ln_H"), 2)
F_P32_X = f"{min(F(PP, 'ln γ̇ + ln P3_2', 'RMSE_over_pure'), F(MX, 'ln γ̇ + ln P3_2', 'RMSE_over_pure')):.0f}"
V["FORM_DP"] =f"{max(F(PP, 'ln γ̇ + ln H + ln ΔP（算出式）', 'RMSE_test'), F(MX, 'ln γ̇ + ln H + ln ΔP（算出式）', 'RMSE_test')):.4f}"

# ---- 表：厚みの比と温度の比
rows = []
for m in MAT:
    s = ratio[(ratio["material"] == MAT[m]) & (ratio["kind"] == "厚み")]
    for h in sorted(s["H"].unique()):
        g = s[np.isclose(s["H"], h)].sort_values("Tset")
        cells = []
        for t in [190, 210, 230]:
            r = g[g["Tset"] == t]
            cells.append(f"{r['ratio_mean'].iloc[0]:.2f}（{r['ratio_min'].iloc[0]:.2f}〜{r['ratio_max'].iloc[0]:.2f}）" if len(r) else "---")
        rows.append(f"{MJ[m] if h == sorted(s['H'].unique())[0] else ''} & {h} mm / 2.0 mm & " + " & ".join(cells) + " \\\\")
    tt = ratio[(ratio["material"] == MAT[m]) & (ratio["kind"] == "温度")]
    rows.append(f" & 190 ℃ / 230 ℃（参考：温度の効果） & \\multicolumn{{3}}{{l}}{{{tt['ratio_mean'].min():.2f}〜{tt['ratio_mean'].max():.2f}（厚みごとの平均の範囲）}} \\\\")
V["TABLE_RATIO"] = "\n".join(rows)
th = ratio[ratio["kind"] == "厚み"]
tt = ratio[ratio["kind"] == "温度"]
th05 = th[np.isclose(th["H"], 0.5)]
V["R05_LO"], V["R05_HI"] = f"{th05['ratio_mean'].min():.2f}", f"{th05['ratio_mean'].max():.2f}"
V["RT_LO"], V["RT_HI"] = f"{tt['ratio_mean'].min():.2f}", f"{tt['ratio_mean'].max():.2f}"

# ---- 表：モデルの RMSE（被覆率）
CATS = [("5data", "S_V", "低速側"), ("5data", "S_V", "高速側"), ("5data", "S_T", "全体"), ("Alldata", "CV", "全体")]
rows = []
for m in MAT:
    for k in KEYS:
        cells = [f"{rmse(M(m, v, s, k, sub))}（{M(m, v, s, k, sub, 'coverage95'):.2f}）" for v, s, sub in CATS]
        rows.append(f"{MJ[m] if k == 'R3' else ''} & {k} & " + " & ".join(cells) + " \\\\")
    rows.append(f" & 純粋誤差 & " + " & ".join(f"{M(m, v, s, 'R3', sub, 'pure_sd'):.3f}" for v, s, sub in CATS) + " \\\\")
V["TABLE_MET"] = "\n".join(rows)

# ---- 表：厚みごとの残差の平均（Alldata 版の CV）
rows = []
for m in MAT:
    b = bias[(bias["material"] == MAT[m]) & (bias["version"] == "Alldata") & (bias["split"] == "CV")]
    for k in KEYS:
        g = b[b["model"] == k].sort_values("H")
        rows.append(f"{MJ[m] if k == 'R3' else ''} & {k} & " + " & ".join(num(float(x), 3) for x in g["bias"]) + " \\\\")
V["TABLE_BIAS"] = "\n".join(rows)

# ---- 表：長さスケール（5data 版の S_V）
rows = []
for m in MAT:
    for k in KEYS:
        r = ker[(ker["material"] == MAT[m]) & (ker["version"] == "5data") & (ker["split"] == "S_V") & (ker["model"] == k)].iloc[0]
        ls = []
        for c, ja in [("ls_ln_gd_cm", "$\\gd$"), ("ls_ln_H", "$H$"), ("ls_ln_Fo_cm", "Fo"), ("ls_Tset", "$\\Tset$"), ("ls_ln_Tff_cm", "$\\Tff$"), ("ls_ln_P32_cm", "\\PII")]:
            if c in r and not pd.isna(r[c]):
                ls.append(f"{ja} {r[c]:.2f}")
        rows.append(f"{MJ[m] if k == 'R3' else ''} & {k} & {'，'.join(ls)} & {r['noise_sd']:.4f} \\\\")
V["TABLE_LS"] = "\n".join(rows)

# ---- 表：P3_2 は何の情報を持つか
pxr = rd(S, "p32_regression.csv")
rows = []
for m in MAT:
    for _, r in proxy[proxy["material"] == MAT[m]].iterrows():
        rows.append(f"{MJ[m] if r['base'] == 'γ̇, Tset' else ''} & $\\ln H$ を $\\ln\\gd$，{'$\\Tset$' if r['base'] == 'γ̇, Tset' else '$\\ln\\Tff$'} で回帰 & {r['R2_lnH_base']:.2f} & {r['R2_lnH_with_P32']:.2f} & {num(float(r['corr_lnH_lnP32']), 2)} \\\\")
V["TABLE_PROXY"] = "\n".join(rows)
rows = []
for m in MAT:
    for _, r in pxr[pxr["material"] == MAT[m]].iterrows():
        nm = {"ln P3_2 ~ ln η, ln γ̇, ln H": "$\\ln\\PII$ ～ $\\ln\\eta$，$\\ln\\gd$，$\\ln H$", "ln P3_2 ~ ln γ̇, ln H": "$\\ln\\PII$ ～ $\\ln\\gd$，$\\ln H$", "ln P3_2 ~ ln H": "$\\ln\\PII$ ～ $\\ln H$"}.get(r["regression"], "$\\ln\\PI$ ～ $\\ln\\eta$，$\\ln\\gd$，$\\ln H$（比較：\\PI）")
        def cf(c):
            return "---" if c not in r or pd.isna(r[c]) else num(float(r[c]), 2)
        rows.append(f"{MJ[m] if r['regression'] == 'ln P3_2 ~ ln η, ln γ̇, ln H' else ''} & {nm} & {r['R2']:.3f} & {cf('coef_le')} & {cf('coef_lg')} & {cf('coef_lH')} \\\\")
V["TABLE_P32REG"] = "\n".join(rows)


def PX(m, reg, col):
    r = pxr[(pxr["material"] == MAT[m]) & (pxr["regression"] == reg)]
    return float(r[col].iloc[0])


def PXY(m, base, col):
    r = proxy[(proxy["material"] == MAT[m]) & (proxy["base"] == base)]
    return float(r[col].iloc[0])


# 手順3b の R3P（H を入れたまま P3_2 を足したモデル）との比較
b3 = rd("step3b", "metrics.csv")
b3 = b3[(b3["data"] == "なし") & (b3["split"] == "S_T") & (b3["subset"] == "全体") & b3["model"].isin(["R3", "R3P"])]
V["R3_S_T_PP"], V["R3P_S_T_PP"] = (f"{float(b3[(b3['material'] == 'PP') & (b3['model'] == k)]['RMSE'].iloc[0]):.3f}" for k in ["R3", "R3P"])

# ---- 文章
r3 = {(m, c): M(m, *c) for m in MAT for c in [("5data", "S_V", "R3", "低速側"), ("5data", "S_T", "R3"), ("Alldata", "CV", "R3")]}
w0 = {m: (M(m, "5data", "S_T", "W0"), M(m, "Alldata", "CV", "W0")) for m in MAT}
w1 = {m: (M(m, "5data", "S_T", "W1"), M(m, "Alldata", "CV", "W1")) for m in MAT}
w2 = {m: (M(m, "5data", "S_T", "W2"), M(m, "Alldata", "CV", "W2"), M(m, "5data", "S_V", "W2", "低速側"), M(m, "5data", "S_V", "W2", "全体")) for m in MAT}
w3 = {m: (M(m, "5data", "S_T", "W3"), M(m, "Alldata", "CV", "W3"), M(m, "5data", "S_V", "W3", "低速側"), M(m, "5data", "S_V", "W3", "全体")) for m in MAT}
r3v = {m: (M(m, "5data", "S_V", "R3", "低速側"), M(m, "5data", "S_V", "R3", "全体")) for m in MAT}
rf = {m: (M(m, "Alldata", "CV", "RFo"), M(m, "Alldata", "CV", "R3")) for m in MAT}
st = {m: (M(m, "5data", "S_T", "R3"), M(m, "Alldata", "CV", "R3")) for m in MAT}
bw0 = bias[(bias["version"] == "Alldata") & (bias["split"] == "CV") & (bias["model"] == "W0")]
V["FIG1_TXT"] = (f"どの設定温度でも、薄い厚みの曲線ほど上（粘度が高い）にあり、4 本の曲線は重なりません。0.5 mm の曲線は、2.0 mm の曲線の約 {V['R05_LO']}〜{V['R05_HI']} 倍の高さです（図2）。"
                 "理想的なスリット流れなら重なるはずなので、このずれは、計算の都合ではなく、流れの中で起きている現象（厚みが薄いほど強く効く、冷却・固化層など）によるものです。")
V["FIG2_TXT"] = (f"厚みによる粘度の比（同じ $\\gd$・同じ設定温度）は、0.5 mm で {V['R05_LO']}〜{V['R05_HI']} 倍と大きく、厚くなるほど 1 に近づきます。"
                 f"灰色の帯（温度の効果。190 ℃ と 230 ℃ の比 {V['RT_LO']}〜{V['RT_HI']} 倍）より、厚みの効果のほうが大きいことが分かります。見かけの粘度を予測するには、温度より厚みのほうが効きます。")
V["FIG3_TXT"] = (f"H を入れた R3（黒）と、H を Fo に替えた RFo（灰）は、どこでも純粋誤差（破線）に近い誤差です。H も圧力も入れない W0（赤）・W1（橙）は、誤差が大きく、"
                 f"S\\_T で PP {rmse(w0[PP][0])}・{rmse(w1[PP][0])}（R3 は {rmse(st[PP][0])}）、Alldata 版の CV で PP {rmse(w0[PP][1])}・{rmse(w1[PP][1])}（R3 は {rmse(st[PP][1])}）。2 つの温度を入れた W1T（紫）も、R3 には届きません。"
                 f"一方、P3\\_2 を入れた W2（青）・W3（緑）は、H を入れていないのに、R3 に近い低さです（S\\_T の PP：{rmse(w2[PP][0])}・{rmse(w3[PP][0])}、Alldata 版の CV の PP：{rmse(w2[PP][1])}・{rmse(w3[PP][1])}）。"
                 f"ただし、5data 版の外挿（S\\_V の低速側）では、W2・W3 は R3 より悪く（PP：{rmse(w2[PP][2])}・{rmse(w3[PP][2])}、R3 は {rmse(r3v[PP][0])}）、学習の範囲の外では、P3\\_2 の値が学習の範囲を外れるためと考えられます。")
sd_r3 = bias[(bias["version"] == "Alldata") & (bias["split"] == "CV") & (bias["model"] == "R3")]["sd"]
sd_w0 = bw0["sd"]
V["FIG4_TXT"] = (f"R3・RFo・W2・W3 は、どの厚みでも残差の平均がほぼ 0 で、厚みの中のばらつき（残差の標準偏差）も R3 で {sd_r3.min():.2f}〜{sd_r3.max():.2f} と、純粋誤差と同じくらいです（W2・W3 も同程度）。"
                 f"W0・W1 は、残差の平均が 0 から $\\pm$0.1〜0.2 ずれ（W0 の PP：" + "、".join(f"{h} mm {num(float(v), 3)}" for h, v in zip(bw0[bw0['material'] == 'PP']['H'], bw0[bw0['material'] == 'PP']['bias'])) +
                 f"）、厚みの中のばらつきが {sd_w0.min():.2f}〜{sd_w0.max():.2f} と、R3 の数倍〜20 倍です。H がないと、GPR は $\\gd$ のわずかな違いで条件を見分けて、厚みの違いを無理に当てはめるので、"
                 "抜いた条件の予測が、上にも下にも大きく外れます（平均すると打ち消し合い、平均だけでは小さく見えます）。5data 版の S\\_V（外挿）では、W0 の残差の平均は、どの厚みでも $-$0.25〜$-$0.55 と、一方に偏ります（\\file{bias\\_by\\_thickness.csv}）。")

V["SUMMARY"] = f"""\\begin{{enumerate}}[itemsep=0.6ex,topsep=0.6ex]
\\item \\textbf{{H はリークではない（式の面）}}：$\\ln\\eta = \\ln\\Delta P + \\ln H - \\ln\\gd$ ＋ 定数 なので、H と $\\gd$ だけでは、$\\Delta P$（流れの結果）が分からず、粘度を計算できません。実際に、$\\ln\\gd$ と $\\ln H$ だけの線形回帰の検証 RMSE は PP {V['FORM_GH_PP']}、粉砕材混合 PP {V['FORM_GH_MIX']} で、純粋誤差（約 0.04）の {V['FORM_GH_X']} 倍以上です。$\\ln H$ の係数も、式の一部としてだけ効くときの ＋1 ではなく、{V['COEF_H_PP']}（PP）でした。$\\Delta P$（\\PI\\ $-$ 0.30 MPa）を足すと、RMSE は {V['FORM_DP']} と、ほぼ 0 になります（これがリーク。手順3a）。
\\item \\textbf{{H はリークではない（予測の面）}}：H は、金型の寸法で、予測するときに事前に分かります。同じ条件のショットで同じ値なので、ショットごとのゆれ（ばらつき）の情報も持っていません。実際に、H を入れた R3 の誤差は、純粋誤差とほぼ同じで、それより小さくはなりません（表\\ref{{tab:met}}）。リークなら、純粋誤差より明らかに小さくなります。
\\item \\textbf{{$\\gd$ も、H を使って計算している}}：$\\gd = 6L/(H\\,\\Delta t)$ なので、「目的変数の算出に使う値は入力にしない」とすると、Cross-WLF の入力である $\\gd$ も使えなくなります。$\\gd$ と H を入れることは、測った到達時間 $\\Delta t$ と H を入れることと、同じ情報です（表\\ref{{tab:form}} の 1 行目と 3 行目が同じ）。
\\item \\textbf{{見かけの粘度は、厚みで大きく違う}}：同じ $\\gd$・同じ設定温度で、0.5 mm は 2.0 mm の約 {V['R05_LO']}〜{V['R05_HI']} 倍の粘度です（図1・図2）。温度の効果（190 ℃ と 230 ℃ で {V['RT_LO']}〜{V['RT_HI']} 倍）より大きい効果です。理想的なスリット流れなら、この比は 1 です。
\\item \\textbf{{H も圧力も入れないと、見かけの粘度を予測できない}}：Cross-WLF の変数のうち、圧力を除いた（$\\gd$、$\\Tset$）の W0 は、R3 の数倍〜十数倍の誤差です（S\\_T で PP {rmse(w0[PP][0])}、R3 は {rmse(st[PP][0])}）。温度を、流れの中の樹脂温度 $\\Tff$ にした W1 も、同じくらい悪く（{rmse(w1[PP][0])}）、$\\Tff$ では、厚みの効果を説明できません。H を外した GPR は、$\\gd$ の長さスケールが下限（0.01）に張り付き、$\\gd$ のわずかな違いで条件を見分けて、厚みの違いを無理に説明しようとします（表\\ref{{tab:ls}}）。
\\item \\textbf{{Fo（構造 A・B の入力）は、H と同じ情報}}：$\\ln\\mathrm{{Fo}} = -(\\ln\\gd + 3\\ln H)$ ＋ 定数 なので、Fo を入れた RFo は、R3 とほぼ同じ精度です（CV：PP {rmse(rf[PP][0])}、R3 は {rmse(rf[PP][1])}）。手順5a・5本の構造 A・B も、Fo を通して H を使っています。
\\item \\textbf{{P3\\_2 のリークの確認（算出式による）}}：手順3a で済んでいます（H を入れたモデルで）。P3\\_2 は算出式に入っておらず、線形回帰に足しても学習の残差はほとんど下がりません（PP：{F(PP, 'ln γ̇ + ln H', 'resid_sd_train'):.3f} $\\to$ {F(PP, 'ln γ̇ + ln H + ln P3_2', 'resid_sd_train'):.3f}。$\\Delta P$（\\PI\\ $-$ 0.30 MPa）なら {F(PP, 'ln γ̇ + ln H + ln ΔP（算出式）', 'resid_sd_train'):.4f}）。3e では、\\textbf{{H を外した状態でも}}確かめました：$\\ln\\gd$ と $\\ln\\PII$ だけの線形回帰の検証 RMSE は、純粋誤差の {F_P32_X} 倍以上で、粘度を計算できません（表\\ref{{tab:form}}）。結論は、「算出式によるリークではない」です。
\\item \\textbf{{P3\\_2 を圧力として入れた W2・W3 は、R3 に近い精度になる。ただし、P3\\_2 が厚みを運んでいる}}：W2（$\\gd$，$\\Tset$，$\\PII$）・W3（$\\gd$，$\\Tff$，$\\PII$）は、H を入れていないのに、S\\_T で PP {rmse(w2[PP][0])}・{rmse(w3[PP][0])}（R3 は {rmse(st[PP][0])}）、Alldata 版の CV で PP {rmse(w2[PP][1])}・{rmse(w3[PP][1])}（R3 は {rmse(st[PP][1])}）と、W0・W1 より、はるかに良くなります（表\\ref{{tab:met}}）。理由は、\\textbf{{P3\\_2 が、粘度ではなく、厚みでほぼ決まる量}}だからです（表\\ref{{tab:p32reg}}）：$\\ln\\PII$ を $\\ln\\eta$・$\\ln\\gd$・$\\ln H$ で回帰すると、$\\ln\\eta$ の係数は {PX(PP, 'ln P3_2 ~ ln η, ln γ̇, ln H', 'coef_le'):+.2f}（PP）とほぼ 0 で、$\\ln H$ の係数は {PX(PP, 'ln P3_2 ~ ln η, ln γ̇, ln H', 'coef_lH'):+.2f}（R2 {PX(PP, 'ln P3_2 ~ ln η, ln γ̇, ln H', 'R2'):.2f}）です。つまり、P3\\_2 はおよそ $H^{{-1.5}}$ に比例します（厚いほど、圧力が低い）。そのため、入力に P3\\_2 を足すと、$\\gd$・$\\Tset$ から $\\ln H$ を当てる R2 が {PXY(PP, 'γ̇, Tset', 'R2_lnH_base'):.2f} から {PXY(PP, 'γ̇, Tset', 'R2_lnH_with_P32'):.2f} に上がります（表\\ref{{tab:proxy}}）。手順3b で、H を入れたまま P3\\_2 を足しても精度が変わらなかった（S\\_T：R3 {V['R3_S_T_PP']}、R3P {V['R3P_S_T_PP']}）ことと合います。\\textbf{{W2・W3 は、「H なし」のモデルではなく、「H を P3\\_2 の形で入れたモデル」}}です。
\\item \\textbf{{W2・W3 は、予測には使えない}}：P3\\_2 は、流れの結果として測る値で、成形条件から予測するときに、事前に分かりません（P3\\_1・$\\Tff$ と同じ）。また、5data 版の外挿（S\\_V の低速側）では、W2・W3 は R3 より悪くなります（PP：{rmse(w2[PP][2])}・{rmse(w3[PP][2])}、R3 は {rmse(r3v[PP][0])}）。さらに、P3\\_2 は「センサ4の出力が最大になった時刻」のセンサ3の圧力で、粘度を測った区間（センサ3 $\\to$ 4 の到達の間）の圧力ではありません。Cross-WLF の $P$（その場の圧力）とは、意味が違います。
\\end{{enumerate}}"""

rc.build_pdf("3e", S, "report_step3e_template.tex", V, "gakushu3e")
