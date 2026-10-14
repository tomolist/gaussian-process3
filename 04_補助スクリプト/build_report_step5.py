# -*- coding: utf-8 -*-
"""手順5 の説明書（用語集と図の見方）を作る。

使い方（リポジトリ直下から）: uv run python 04_補助スクリプト/build_report_step5.py
results/step5 の CSV から数値を取り出して report_step5_template.tex に入れ、_tmp/report_5 で
LuaLaTeX を2回動かし、できた PDF と tex を results/学習結果説明_手順5.pdf・.tex に置く。
"""
import os
import re

import numpy as np
import pandas as pd

import report_common as rc
from report_common import num, rmse

S = "step5"
R5 = os.path.join(rc.R, S)


def rd5(name):
    return pd.read_csv(os.path.join(R5, name), encoding="utf-8-sig", dtype={"hv": str})


met, best, ker, eff, obs, p32, p4, dec, cmp_ = (rd5(f) for f in ["metrics.csv", "best_E.csv", "kernels.csv", "tset_effect.csv", "observed_ratio.csv", "p32_pred.csv",
                                                                   "p32_actual_vs_predicted.csv", "decomposition.csv", "main_vs_alt.csv"])
V = {}
MJ = rc.MAT_JA
MATS = ["PP", "MIX"]
MN = rc.MAT
VERS = ["5data", "Alldata"]
HVS = ["1", "2", "4", "14", "24"]
HV_NAME = {"1": "(1)", "2": "(2)", "4": "(4)", "14": "(1+4)", "24": "(2+4)"}
PANELS = {"5data": [("S_V", "低速側", "S\\_V 低速側"), ("S_V", "高速側", "S\\_V 高速側"), ("S_T", "全体", "S\\_T"), ("S_T3", "全体", "S\\_T3")],
          "Alldata": [("CV", "全体", "CV 全体"), ("CV", "低速側", "CV 低速側"), ("S_T", "全体", "S\\_T"), ("S_T3", "全体", "S\\_T3")]}
MAIN_SET = {"5data": "S_V", "Alldata": "ALL"}


def mn(st, hv):
    return f"{st}\\_{hv}"


def M(mat, ver, split, st, hv, sub="全体", psrc=None, col="RMSE"):
    psrc = psrc or ("実測" if "4" in hv else "―")
    r = met[(met["material"] == MN[mat]) & (met["version"] == ver) & (met["split"] == split) & (met["struct"] == st) & (met["hv"] == hv) & (met["subset"] == sub) & (met["psrc"] == psrc)]
    assert len(r) == 1, (mat, ver, split, st, hv, sub, psrc)
    return float(r[col].iloc[0])


def B(mat, ver, st, hv, col, sset=None):
    sset = sset or MAIN_SET[ver]
    r = best[(best["material"] == MN[mat]) & (best["version"] == ver) & (best["set"] == sset) & (best["struct"] == st) & (best["hv"] == hv)]
    assert len(r) == 1, (mat, ver, st, hv, sset)
    return float(r[col].iloc[0])


V["N_FIG"] = str(len([f for f in os.listdir(R5) if f.endswith(".png")]))
V["N_CSV"] = str(len([f for f in os.listdir(R5) if f.endswith(".csv")]))
V["N_HR"] = str(len([f for f in os.listdir(os.path.join(R5, "highres")) if f.startswith("hr_")]))

# ノートの判定の数（実行済みのノートが _tmp にあれば数える）
nbp = os.path.join(rc.W, "_tmp", "手順5_シフト構造の本実装.ipynb")
n_ok = n_star = None
hours = "約 2.6"
if os.path.exists(nbp):
    import json
    nb = json.load(open(nbp, encoding="utf-8"))
    txt = "\n".join("".join(o.get("text", "")) for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []) if o["output_type"] == "stream")
    n_ok, n_star = len(re.findall(r">> 判定：OK", txt)), len(re.findall(r">> 判定：★要確認", txt))
V["N_OK"], V["N_STAR"] = (str(n_ok), str(n_star)) if n_ok is not None else ("---", "---")
V["RUN_HOURS"] = "2.6"

# ---- 表1：E
rows = []
for ver in VERS:
    for mat in MATS:
        for hv in HVS:
            lab = f"{MJ[mat] if hv == '1' else ''} & {ver if hv == '1' else ''} & {HV_NAME[hv]} & {B(mat, ver, 'A', hv, 'Ev_best'):.1f} & {B(mat, ver, 'B', hv, 'Eh_best'):.0f} & {B(mat, ver, 'B', hv, 'Ev_best'):.0f} & "
            lab += f"{B(mat, ver, 'B', hv, 'diff_best'):.0f}（{B(mat, ver, 'B', hv, 'diff_95_lo'):.0f}〜{B(mat, ver, 'B', hv, 'diff_95_hi'):.0f}） & {B(mat, ver, 'B', hv, 'LML_best') - B(mat, ver, 'A', hv, 'LML_best'):.0f} \\\\"
            rows.append(lab)
        rows.append("\\midrule" if not (ver == VERS[-1] and mat == MATS[-1]) else "")
V["TABLE_E"] = "\n".join(rows)

# ---- 表2：RMSE
FOOT = {"5data": "5data 版（学習 5data）", "Alldata": "Alldata 版（学習 Alldata）"}
blocks = []
for ver in VERS:
    for mat in MATS:
        rows = []
        for hv in HVS:
            for st in ["B", "A", "C"]:
                cells = []
                for sp, sub, _ in PANELS[ver]:
                    a = M(mat, ver, sp, st, hv, sub)
                    cells.append(f"{a:.3f}" + (f"（{M(mat, ver, sp, st, hv, sub, '予測'):.3f}）" if "4" in hv else ""))
                rows.append(f"{HV_NAME[hv] if st == 'B' else ''} & {mn(st, hv)} & " + " & ".join(cells) + " \\\\")
            rows.append("\\midrule")
        pure = float(met[(met["material"] == MN[mat]) & (met["version"] == ver) & (met["split"] == PANELS[ver][0][0])]["pure_sd"].iloc[0])
        rows[-1] = f"\\multicolumn{{2}}{{l}}{{純粋誤差（参考）}} & \\multicolumn{{4}}{{l}}{{約 {pure:.3f}}} \\\\"
        cap = f"{MJ[mat]}・{FOOT[ver]}：RMSE（$\\ln\\eta$）。(4) を含む行は、実測の \\PII\\ のとき（括弧の中が、予測した \\PII\\ のとき）"
        head = " & ".join(["扱い", "モデル"] + [p[2] for p in PANELS[ver]])
        blocks.append(f"\\begin{{table}}[H]\n\\caption{{{cap}}}\\label{{tab:met_{ver}_{mat}}}\n\\footnotesize\n\\begin{{tabularx}}{{\\linewidth}}{{@{{}}llXXXX@{{}}}}\n\\toprule\n{head} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}")
V["TABLE_MET"] = "\n\n".join(blocks)

# ---- 表3：設定温度の効果（Alldata 版）
rows = []
for mat in MATS:
    o = obs[obs["material"] == MN[mat]].sort_values("H")
    rows.append(f"{MJ[mat]} & 実測 & " + " & ".join(f"{x:.2f}" for x in o["observed"]) + " \\\\")
    for hv in ["1", "2", "4"]:
        for st in ["B", "A", "C"]:
            e = eff[(eff["version"] == "Alldata") & (eff["material"] == MN[mat]) & (eff["model"] == f"{st}_{hv}")].sort_values("H")
            rows.append(f" & {mn(st, hv)} & " + " & ".join(f"{x:.2f}" for x in e["ratio_190_230"]) + " \\\\")
    rows.append("\\midrule")
rows[-1] = ""
V["TABLE_TSET"] = "\n".join(rows)

# ---- 表4：P3_2 の実測と予測
rows = []
for ver in VERS:
    for mat in MATS:
        for st in ["B", "A", "C"]:
            cells = []
            for sp, sub, _ in PANELS[ver]:
                cells.append(f"{M(mat, ver, sp, st, '4', sub):.3f}→{M(mat, ver, sp, st, '4', sub, '予測'):.3f}")
            rows.append(f"{(MJ[mat] + '・' + ver) if st == 'B' else ''} & {mn(st, '4')} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
rows[-1] = ""
V["TABLE_P32"] = "\n".join(rows)
head32 = " & ".join(["", "モデル", "検証1", "検証2", "S\\_T", "S\\_T3"])
V["P32_HEAD"] = head32
pr = p32.assign(e=p32["pred"] - p32["actual"]).groupby(["material", "version", "split"])["e"].agg(lambda x: float(np.sqrt(np.mean(x ** 2))))


def P(mat, ver, split):
    return float(pr.loc[(MN[mat], ver, split)])


# ---- 表5：(2) の分解
rows = []
for ver in VERS:
    for mat in MATS:
        cells = []
        for st in ["B", "A", "C"]:
            cc = dec[(dec["material"] == MN[mat]) & (dec["version"] == ver) & (dec["struct"] == st) & (dec["kind"] == "c_cond")]["mean"]
            cells.append(f"{float(cc.max() - cc.min()):.2f}")
        rows.append(f"{(MJ[mat] + '・' + ver)} & " + " & ".join(cells) + f" & {np.log(obs[obs['material'] == MN[mat]]['observed'].max()):.2f} \\\\")
V["TABLE_DEC"] = "\n".join(rows)

# ---- 文章用の数値
cvp = {(m, st, hv): M(m, "Alldata", "CV", st, hv) for m in MATS for st in ["B", "A", "C"] for hv in HVS}
stp = {(m, st, hv): M(m, "5data", "S_T3", st, hv, "全体") for m in MATS for st in ["B", "A", "C"] for hv in HVS}
svl = {(m, st, hv): M(m, "5data", "S_V", st, hv, "低速側") for m in MATS for st in ["B", "A", "C"] for hv in HVS}
svh = {(m, st, hv): M(m, "5data", "S_V", st, hv, "高速側") for m in MATS for st in ["B", "A", "C"] for hv in HVS}
PP, MX = "PP", "MIX"


def pc(a, b):
    return f"{(a / b - 1) * 100:+.0f}\\%".replace("+", "$+$").replace("-", "$-$")


FIGS = []


def fig(title, fname, caption, what, axes, look, result, width="0.97"):
    what = what + f"（ファイル：\\file{{{fname}}}。\\file{{highres/}} に 600 dpi 版もあります）"
    FIGS.append(f"""\\subsection{{{title}}}
\\begin{{figure}}[H]
\\centering
\\includegraphics[width={width}\\linewidth]{{{fname}}}
\\caption{{{caption}}}
\\end{{figure}}
\\begin{{points}}
\\pt{{何を示す図か}}{what}
\\pt{{軸と記号}}{axes}
\\pt{{見るポイント}}{look}
\\pt{{今回の結果}}{result}
\\end{{points}}
""")


# 図1
e_a = {(m, v): B(m, v, "A", "1", "Ev_best") for m in MATS for v in VERS}
fig("図1：A のプロファイル尤度（Alldata 版）", "fig1_profile_A_Alldata.png", "A の $\\Ev$ と、対数周辺尤度の最大との差（上：PP、下：粉砕材混合 PP。左：S\\_T の学習、右：ALL の学習。色は $H$ の扱い）",
    "A（縦だけシフト）の $\\Ev$ を横軸に振り、各 $\\Ev$ で学習して求めた対数周辺尤度を、その最大との差で描いた図（尤度曲線）です。山の頂点（★）が、データに最も合う $\\Ev$ です。",
    "横軸は $\\Ev$ [kJ/mol]、縦軸は「対数周辺尤度 $-$ その曲線の最大」（0 が頂点）。灰色の破線が $-1.92$（おおよその 95\\% 区間の境）、縦線が $E=0$（温度の効果なし）。色は $H$ の扱い。",
    "山がはっきりしていて、頂点が $E=0$ から離れているか。山が鋭いほど、$E$ がデータで決まっている。$H$ の扱いで頂点がずれるか。",
    f"どの扱いでも山は鋭く、頂点は $E=0$ から離れています（温度の効果は、データで支持されます）。$\\Ev$ は、$H$ を残す (1)(2) で {B('PP', 'Alldata', 'A', '1', 'Ev_best'):.1f}（PP）・{B('MIX', 'Alldata', 'A', '1', 'Ev_best'):.1f}（粉砕材混合 PP）kJ/mol、\\PII\\ を入れる (4)(1+4)(2+4) で {B('PP', 'Alldata', 'A', '4', 'Ev_best'):.1f}〜{B('PP', 'Alldata', 'A', '14', 'Ev_best'):.1f} kJ/mol と、少し小さくなります。190/230 ℃ の粘度比は $\\exp(0.0207\\,\\Ev)$ で、約 1.15 倍です。")

# 図2
fig("図2：B の 2 次元プロファイル（PP・Alldata 版）", "fig2_profile_B_Alldata_PP.png", "B の $(\\Eh,\\Ev)$ の対数周辺尤度（最良の点の $\\pm10$ kJ/mol を拡大。列が $H$ の扱い）",
    "B（横と縦を別々にシフト）では、$\\Eh$（横）と $\\Ev$（縦）の 2 つを決めます。2 つの値の組ごとに学習して求めた対数周辺尤度を、色で描いた図です。",
    "横軸が $\\Eh$、縦軸が $\\Ev$（kJ/mol）。黄〜緑が尤度の高い所（最大に近い）、紫は最大より 40 以上低い所。赤い星が最良の点、白丸が最大から 1.92 以内の点、黒い小点が計算した格子の点。黒い破線が $\\Eh=\\Ev$（当初の構造 H）、赤い点線が、最良の点を通る $\\Ev-\\Eh=$ 一定の線。",
    "尤度の山が、星のまわりに、細い斜めの帯（赤い点線に沿った帯）になっていれば、「$\\Eh$ と $\\Ev$ は、差だけがよく決まる」ことを表します。帯の幅（白丸の範囲）が狭いほど、$\\Eh$・$\\Ev$ がデータで決まっています。",
    f"山は、赤い点線に沿った細い斜めの帯で、星のまわりの狭い範囲に絞られました。$H$ を残す (1) では、白丸の範囲は星の 1 点（$\\Eh$ は $\\pm$1 kJ/mol 程度）で、$\\Eh={B('PP', 'Alldata', 'B', '1', 'Eh_best'):.0f}$・$\\Ev={B('PP', 'Alldata', 'B', '1', 'Ev_best'):.0f}$ kJ/mol、差 $\\Ev-\\Eh={B('PP', 'Alldata', 'B', '1', 'diff_best'):.0f}$ です。(2)(4)(1+4)(2+4) では、帯が少し長く、斜めに伸びますが、差は 8〜11 kJ/mol の範囲に収まっています（表\\ref{{tab:E}}）。手順5a では、「$\\Eh$ と $\\Ev$ は別々には決めきれない」と書きましたが、今回は、前の格子点の解を初期値にして局所解を減らし、細かい格子で調べた結果、$\\Eh$ も数 kJ/mol の幅に絞れました（ただし、細かい格子の範囲は最良の点の $\\pm$4 kJ/mol までなので、帯の長さは、この範囲までしか分かりません）。", width="1.0")

# 図3
fig("図3：マスターカーブ（PP・Alldata 版）", "fig3_master_Alldata_PP.png", "温度シフトで換算したあとの条件平均（行が厚み。左から：シフトなし、A、B。色が設定温度）",
    "各条件の点（条件平均）を、温度シフトでずらしたあとの位置に描いた図です。シフトの目的は、3 つの温度の点を、1 本の曲線（マスターカーブ）に重ねることです。",
    "横軸は換算後の $\\ln\\gd$（B は $\\ln\\gd+\\ln a_T(\\Eh)$）、縦軸は換算後の $\\ln\\eta$（縦シフトを引いた値）。青 190 ℃、緑 210 ℃、赤 230 ℃。",
    "同じ厚みの中で、3 つの色の点が、1 本の曲線の上に重なっているか。",
    "シフトなし（左）では、3 つの温度が、縦に少しずれて並びます（低温ほど上）。A（中）は、縦のずれを一定の倍率で消し、点がよく重なります。B（右）は、横にもずらすため、同じ厚みの中で、温度ごとに点が少し横にずれて並びますが、3 つの温度の点は、ほぼ 1 本の曲線の上にあります。A も B も、3 つの温度のデータを、1 本の曲線で表せることを示します。")

# 図4
fig("図4：設定温度ごとの予測曲線（PP・Alldata 版、H の扱い (1)）", "fig4_curves_Alldata_PP_1.png", "予測曲線と 95\\% 予測区間（行が厚み、列が構造 B・A・C。最終モデル。線の色が設定温度）",
    "最終モデル（Alldata 全条件で学習）が、せん断速度と粘度の関係を、設定温度ごとにどう描くかを示した図です。",
    "横軸はせん断速度、縦軸は粘度（どちらも対数）。青 190 ℃・緑 210 ℃・赤 230 ℃。線が予測、薄い帯が 95\\% 予測区間、点が学習の条件平均。$\\times$ は問題データ一覧の「感度確認」の条件。(2)(4) の図は \\file{fig4_curves_Alldata_PP_2.png}・\\file{fig4_curves_Alldata_PP_4.png}（ほかの材料・版も同じ名前の形）。",
    "点が帯の中に入っているか。3 つの温度の曲線の間隔（温度の効果）が、厚みやせん断速度でどう変わるか。",
    "B・A・C のどれも、点が帯に入り、温度ごとの曲線が 190 ℃ > 210 ℃ > 230 ℃ の順に並びます。B と C は、厚いほど（また低せん断速度ほど）曲線の間隔が広がり、実測の「厚いほど温度の効果が大きい」傾向を表します。A は、どの厚みでも間隔が一定です。(4) を含む図では、P3\\_2 を、同じ厚み・温度の条件平均から内挿して描いています（予測のときには分からない値です）。", width="0.95")

# 図5
r_cv = {(m, st): cvp[(m, st, "1")] for m in MATS for st in ["B", "A", "C"]}
fig("図5：評価指標の比較（Alldata 版）", "fig5_metrics_Alldata.png", "15 モデルの RMSE と NLPD（左：PP、右：粉砕材混合 PP。色が検証の種類。(実)(予) は、実測／予測した P3\\_2）",
    "15 のモデルを、検証の種類ごとに、並べて比べた図です。",
    "横軸がモデル（構造\\_扱い）、縦軸が RMSE（上段。対数軸）と NLPD（下段）。どちらも小さいほどよい。色は検証：青 CV 全体など（凡例参照）。",
    "(1) の 3 本（B\\_1・A\\_1・C\\_1）の高さがそろっているか。(2)・(4) で、特定の検証だけ高くならないか。",
    f"CV 全体では、B\\_1 と C\\_1 がほぼ同じで（PP：{r_cv[(PP, 'B')]:.3f} と {r_cv[(PP, 'C')]:.3f}。粉砕材混合 PP：{r_cv[(MX, 'B')]:.3f} と {r_cv[(MX, 'C')]:.3f}）、A\\_1 はやや高く（{r_cv[(PP, 'A')]:.3f}・{r_cv[(MX, 'A')]:.3f}）なります。(2) は、CV 全体で (1) より {pc(cvp[(PP, 'B', '2')], cvp[(PP, 'B', '1')])}（PP の B）高く、(4) の一部（B\\_4 など）は、低速側の検証だけが高くなります。(1+4) は (1) と同じ高さです。5data 版の図（\\file{{fig5_metrics_5data.png}}）では、外挿（S\\_V 低速側）で、(2)(4) の高さが大きく違います（表\\ref{{tab:met_5data_PP}}）。", width="1.0")

# 図6
o_pp = obs[obs["material"] == "PP"].sort_values("H")["observed"].tolist()
fig("図6：設定温度の効果（Alldata 版）", "fig6_tset_effect_Alldata.png", "190 ℃ と 230 ℃ の粘度の比（同じ $\\gd$。厚みごと。太い黒線が実測、細い線が各モデルの予測）",
    "Tset だけを 190 ℃ から 230 ℃ に変えたときの粘度の比を、厚みごとに、実測と各モデルで比べた図です。",
    "横軸は厚み [mm]、縦軸は粘度の比（190 ℃ の粘度 ÷ 230 ℃ の粘度）。太い黒線が実測。線の色は構造（紫 B、青 A、黒 C）、線の種類は $H$ の扱い（実線 (1)、破線 (2)、点線 (4)）。",
    "実測の傾向（厚いほど比が大きい）を、モデルが表せているか。",
    f"実測の比は、厚み 0.5 → 2.0 mm で、PP {o_pp[0]:.2f} → {o_pp[-1]:.2f} と、厚いほど大きくなります。B\\_1・C\\_1 は、この傾向をよく表します（PP の B\\_1：{eff[(eff['version'] == 'Alldata') & (eff['material'] == 'PP') & (eff['model'] == 'B_1')].sort_values('H')['ratio_190_230'].iloc[0]:.2f} → {eff[(eff['version'] == 'Alldata') & (eff['material'] == 'PP') & (eff['model'] == 'B_1')].sort_values('H')['ratio_190_230'].iloc[-1]:.2f}）。A は、構造上、どの厚みでも一定（約 1.16）で、傾向を表せません。(4) の C\\_4 は、傾向が弱く、1.5 mm 以上で頭打ちになります（表\\ref{{tab:tset}}）。", width="0.97")

# 図7
fig("図7：(2) の分解 ―― 材料の部分 $m$ と、厚み・冷却の補正 $c$（PP・Alldata 版）", "fig7_decomposition_Alldata_PP.png", "上：$m$（材料の部分。設定温度ごと）、下：$c$（ln Fo の関数。点は学習の条件、色は厚み）。列は構造 B・A・C",
    "H の扱い (2)（$m$ と $c$ の足し算）の最終モデルを、2 つの成分に分けて見せた図です。",
    "上段：横軸は $\\gd$、縦軸は材料の部分の粘度（$c$ の平均を 0 にそろえたもの）。3 本の線が設定温度。下段：横軸は $\\ln\\mathrm{Fo}=-(\\ln\\gd+3\\ln H)$ ＋ 定数、縦軸は厚み・冷却の補正 $c$（$\\ln\\eta$ の単位）。灰色の帯が 95\\% の帯。",
    "上段が、せん断速度とともに下がる滑らかな曲線（Cross-WLF の粘度曲線のような形）で、温度で位置が変わるか。下段の補正 $c$ が、どれくらいの大きさか。",
    f"材料の部分 $m$ は、滑らかな右下がりの曲線で、温度で上下します（B は横にもずれます）。補正 $c$ は、$\\ln\\mathrm{{Fo}}$ とともにほぼ直線的に増え、最大と最小の差が約 {float(dec[(dec['material'] == 'PP') & (dec['version'] == 'Alldata') & (dec['struct'] == 'B') & (dec['kind'] == 'c_cond')]['mean'].max() - dec[(dec['material'] == 'PP') & (dec['version'] == 'Alldata') & (dec['struct'] == 'B') & (dec['kind'] == 'c_cond')]['mean'].min()):.1f}（粘度で約 3 倍）あり、温度の効果（$\\ln$ で 0.17 ほど）よりはるかに大きい補正です。つまり、「見かけの粘度」の変化の大部分は、冷却・固化に由来する厚みの補正が占めています（表\\ref{{tab:dec}}）。", width="0.97")

# 図8
fig("図8：P3\\_2 の予測（5data 版）", "fig8_p32_pred_5data.png", "(4) で使う P3\\_2 の、実測と予測（検証の条件。縦棒は 95\\% 区間。色は厚み）",
    "(4) の「予測した P3\\_2」を作るモデル（$(\\ln\\gd,\\ln H,\\Tset)$ から $\\ln$ P3\\_2 を予測する GPR）が、検証の条件の P3\\_2 を、どれだけ当てられるかを示した図です。",
    "横軸が実測の $\\ln$ P3\\_2（条件平均、Pa）、縦軸が予測。黒い破線が一致の線（対角線）。点の色が厚み。",
    "点が対角線の近くにあるか。外れる点は、どの厚み・条件か。",
    f"点はほぼ対角線に乗り、ln P3\\_2 の予測の RMSE は、5data 版の外挿で PP {P(PP, '5data', 'S_V'):.2f}、Alldata 版の CV で {P(PP, 'Alldata', 'CV'):.3f} と小さいです（粘度の ln η の誤差は 0.05 ほど）。外れるのは、外挿側の 1.5〜2.0 mm の厚い条件（予測が低め）です。つまり、\\PII\\ は、$H$・$\\gd$・$\\Tset$ から、ほぼ予測できます。\\PII\\ が厚みの情報を持つ（$H$ の代わりの目印）、手順3e の結論と一致します。", width="0.97")
V["FIGURES"] = "\n".join(FIGS)

# ---- 表の説明
V["TABLES"] = f"""
\\subsection{{表\\ref{{tab:E}}：最良の $E$}}
\\begin{{table}}[H]
\\caption{{構造 A・B の最良の $E$（kJ/mol）。5data 版は S\\_V の学習、Alldata 版は ALL の学習。差＝$\\Ev-\\Eh$（括弧：LML が最大から 1.92 以内の範囲）。ΔLML＝B の LML $-$ A の LML（同じ $H$ の扱い）}}\\label{{tab:E}}
\\footnotesize
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}llllllXr@{{}}}}
\\toprule
材料 & 版 & 扱い & A の $\\Ev$ & B の $\\Eh$ & B の $\\Ev$ & 差（範囲） & ΔLML \\\\
\\midrule
{V['TABLE_E']}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}
\\begin{{points}}
\\pt{{読み方}}A の $\\Ev$ は 7〜7.5 kJ/mol（$H$ を残すとき）。B は $(\\Eh,\\Ev)\\approx(6\\text{{〜}}7,\\ 15\\text{{〜}}16)$、差は約 9。B の LML が A より大きい（ΔLML は PP の Alldata 版 (1) で {B('PP', 'Alldata', 'B', '1', 'LML_best') - B('PP', 'Alldata', 'A', '1', 'LML_best'):.0f}）ので、「横のシフトも必要」とデータが言っています。\\PII\\ を入れる扱い (4)(1+4)(2+4) では、A の $\\Ev$ が 4.5〜5.5 に小さくなり、B の $\\Eh$・$\\Ev$ も少し動きます（\\PII\\ が、温度による粘度の違いの一部を持つため）。CV の 10 分割でも、A の $\\Ev$ のばらつき（標準偏差）は 0.2〜0.4、B の差は 0〜1.3 kJ/mol でした。
\\end{{points}}

\\subsection{{表\\ref{{tab:met_5data_PP}} ほか：検証の RMSE}}
\\begin{{points}}
\\pt{{表の種類}}5data 版と Alldata 版、PP と粉砕材混合 PP の 4 つの表です。行が 15 モデル（構造 B・A・C $\\times$ 扱い 5 通り）、列が検証の種類です。5data 版の S\\_V は外挿（低速側・高速側）、S\\_T は 210 ℃、S\\_T3 は 3data の 210 ℃（速度の外挿も重なる）。Alldata 版の CV は、条件単位の 10 分割です。
\\pt{{(4) を含む行}}「実測の \\PII\\ のときの値（予測した \\PII\\ のときの値）」の形で 2 つの値を書きました。
\\end{{points}}
{V['TABLE_MET']}

\\subsection{{表\\ref{{tab:tset}}：設定温度の効果}}
\\begin{{table}}[H]
\\caption{{190 ℃ と 230 ℃ の粘度の比（Alldata 版の最終モデル。列が厚み [mm]）}}\\label{{tab:tset}}
\\small
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}llXXXX@{{}}}}
\\toprule
材料 & モデル & 0.5 & 1.0 & 1.5 & 2.0 \\\\
\\midrule
{V['TABLE_TSET']}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}

\\subsection{{表\\ref{{tab:p32}}：(4) の P3\\_2 が、実測のときと、予測のとき}}
\\begin{{table}}[H]
\\caption{{(4) の RMSE（実測の \\PII\\ → 予測した \\PII）。5data 版の列は S\\_V 低速側・S\\_V 高速側・S\\_T・S\\_T3、Alldata 版の列は CV 全体・CV 低速側・S\\_T・S\\_T3}}\\label{{tab:p32}}
\\footnotesize
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}llXXXX@{{}}}}
\\toprule
 & モデル & 検証 1 & 検証 2 & 検証 3 & 検証 4 \\\\
\\midrule
{V['TABLE_P32']}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}

\\subsection{{表\\ref{{tab:dec}}：(2) の補正 $c$ の大きさ}}
\\begin{{table}}[H]
\\caption{{補正 $c$ の、学習の条件での最大と最小の差（$\\ln\\eta$ の単位。H の扱い (2) の最終モデル）。右端は、温度の効果（実測の 190/230 ℃ の粘度比の最大の ln）}}\\label{{tab:dec}}
\\small
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}lXXXX@{{}}}}
\\toprule
 & B\\_2 & A\\_2 & C\\_2 & 温度の効果 \\\\
\\midrule
{V['TABLE_DEC']}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}
補正 $c$ は、温度の効果の約 6 倍の大きさです。
"""

# ---- 結論
star_txt = ""
if n_ok is not None:
    star_txt = f"ノートの判定は、OK {n_ok} 個、★ {n_star} 個でした。"
v1 = (f"CV 全体の RMSE：B\\_1 {cvp[(PP, 'B', '1')]:.3f}・A\\_1 {cvp[(PP, 'A', '1')]:.3f}・C\\_1 {cvp[(PP, 'C', '1')]:.3f}（PP）、"
      f"B\\_1 {cvp[(MX, 'B', '1')]:.3f}・A\\_1 {cvp[(MX, 'A', '1')]:.3f}・C\\_1 {cvp[(MX, 'C', '1')]:.3f}（粉砕材混合 PP）")
V["CONCLUSION"] = f"""
{star_txt}以下、結果をまとめます。

\\begin{{enumerate}}[itemsep=0.8ex,topsep=0.6ex]
\\item \\textbf{{温度シフトの $E$ は、データで決まった}}：A の $\\Ev$ は約 7〜7.5 kJ/mol、B の $(\\Eh,\\Ev)$ は約 $(6\\text{{〜}}7,\\ 15\\text{{〜}}16)$ kJ/mol（差 約 9）。どちらも、$E=0$（温度の効果なし）より、対数周辺尤度が 40 以上大きく、学習データ（S\\_V・S\\_T・ALL・CV の各分割）を変えても、ほぼ動きません（CV の 10 分割で、A の $\\Ev$ の標準偏差は 0.2〜0.4）。文献値（40 kJ/mol 前後）とは別物で、「この見かけの粘度の温度依存性」を表す係数です。手順5a では「$\\Eh$ と $\\Ev$ は別々には決まらない」としましたが、今回は $\\Eh$ も数 kJ/mol の幅に絞れました。
\\item \\textbf{{構造 B は、通常の GPR（C）と同等。A は、やや劣る}}：$H$ を残す (1) で、{v1}。B は C と同じ精度で、5data 版の S\\_T3 では C より良く（PP：{stp[(PP, 'B', '1')]:.3f} 対 {stp[(PP, 'C', '1')]:.3f}）、外挿の低速側では、PP で C よりやや悪い（{svl[(PP, 'B', '1')]:.3f} 対 {svl[(PP, 'C', '1')]:.3f}）一方、粉砕材混合 PP では良い（{svl[(MX, 'B', '1')]:.3f} 対 {svl[(MX, 'C', '1')]:.3f}）、という結果です。A は、CV で C より {pc(cvp[(PP, 'A', '1')], cvp[(PP, 'C', '1')])}（PP）・{pc(cvp[(MX, 'A', '1')], cvp[(MX, 'C', '1')])}（粉砕材混合 PP）悪い。理由は、A が「温度の効果は、厚みによらず一定」としか表せないためで、実測の「厚いほど温度の効果が大きい」傾向（図6）を表せないからです。\\textbf{{温度シフト構造（B）は、データで支持されました}}。
\\item \\textbf{{(2)（$m$ と $c$ に分ける）は、解釈はできるが、精度が落ち、外挿に弱い}}：分解の図（図7）で、材料の部分 $m$ は滑らかな粘度曲線、厚み・冷却の補正 $c$ は、温度の効果の約 6 倍の大きさ（粘度で約 3 倍）になりました。ただし、$c$ を $\\ln\\mathrm{{Fo}}$ だけの関数にしたため、(1) より精度が落ちます（CV 全体：PP の B で {pc(cvp[(PP, 'B', '2')], cvp[(PP, 'B', '1')])}）。外挿の低速側では、学習の範囲外の $\\ln\\mathrm{{Fo}}$ になり、(1) の数倍悪くなります（PP の B：{svl[(PP, 'B', '2')]:.3f} 対 {svl[(PP, 'B', '1')]:.3f}）。精度が必要なら (1)、「材料の部分」を取り出して解釈したいときは (2)、という使い分けです。
\\item \\textbf{{(4)（P3\\_2 を入れる）は、「H の別の姿」で、予測では H を超えない}}：実測の P3\\_2 を使うと、CV では (1) に近い精度（B\\_4 {cvp[(PP, 'B', '4')]:.3f}）ですが、外挿の低速側では悪化します（B\\_4 {svl[(PP, 'B', '4')]:.3f}）。P3\\_2 を事前に分からないものとして予測して使うと、悪化せず、むしろ低速側では良くなります（B\\_4：{M(PP, '5data', 'S_V', 'B', '4', '低速側', '予測'):.3f}）。これは、P3\\_2 を $(\\ln\\gd,\\ln H,\\Tset)$ からほぼ当てられる（ln P3\\_2 の RMSE {P(PP, 'Alldata', 'CV'):.3f}、図8）ためで、P3\\_2 が、$H$ から作れる情報にすぎないことを示します。(1+4) は (1) と同じ精度で、P3\\_2 を足しても、$H$ があれば、改善しません。手順3e の結論（P3\\_2 は厚みを運ぶ）が、構造 B・A でも確認できました。
\\item \\textbf{{設定温度の効果}}：B\\_1・C\\_1 は、実測の「厚いほど 190/230 ℃ の粘度比が大きい」傾向を表します（図6）。A は一定（約 1.16）、C\\_4 は頭打ちです。
\\item \\textbf{{ノートの ★ の意味}}：★ は、主に次の 4 種類です。(a) 「A が C の 1.1 倍以内」を超えた（A が C より悪い。2 を参照）。(b) 「(2) が (1) の 1.1 倍以内」を超えた（3 を参照）。(c) 「被覆率が 0.90〜0.99」を外れた（B 系の PP の Alldata 版などで、0.87〜0.93 と、区間がやや狭い）。(d) 「予測した P3\\_2 が実測の 1.2 倍以内」を超えた（PP の 5data 版の S\\_T3 で 1.28 倍など）。エラーではなく、結果を読むときの注意点です。
\\end{{enumerate}}

\\subsection{{これからの選択肢}}
\\begin{{enumerate}}[itemsep=0.6ex,topsep=0.6ex]
\\item \\textbf{{最終候補の選び方}}：精度と、構造の支持と、予測のときに使える情報（成形条件だけ）をあわせると、\\textbf{{B\\_1（構造 B、$H$ を残す）}}が最有力です。C\\_1（R3）と同じ精度で、温度の効果を 2 つの $E$ で表せ、厚みごとの温度の効果の違いも表せます。簡単さを優先するなら A\\_1、解釈を優先するなら B\\_2。
\\item \\textbf{{次の手順（ユーザーの決定、10 月 14 日）}}：最終候補の MCMC は、この結果を見てから、別の手順で行います。B\\_1 と、比較用の C\\_1 が、候補です（$E$ を含むハイパーパラメータの不確かさを、サンプルで確かめる）。
\\item \\textbf{{その後}}：手順6（ばらつきの分析：純粋誤差との比較、必要なら異分散ノイズ）、手順7（GPPS）。
\\end{{enumerate}}
"""
rc.build_pdf("5", S, "report_step5_template.tex", V, "gakushu5")
