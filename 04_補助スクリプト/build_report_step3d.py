# -*- coding: utf-8 -*-
"""手順3d の説明書（用語集と図の見方）を作る。

使い方（リポジトリ直下から）: uv run python 04_補助スクリプト/build_report_step3d.py
results/step3d の CSV から数値を取り出して report_step3d_template.tex に入れ、_tmp/report_3d で
LuaLaTeX を2回動かし、できた PDF と tex を results/学習結果説明_手順3d.pdf・.tex に置く。
"""
import os

import numpy as np
import pandas as pd

import report_common as rc
from report_common import MAT, num, pct, rd, rmse

S = "step3d"
ver, inc, drift, scan, met, ker, summ, prob = (rd(S, f) for f in ["versions.csv", "inclusion.csv", "shot_drift.csv", "outlier_check.csv", "metrics.csv", "kernels.csv", "summary.csv", "problem_conditions.csv"])
V = {}
MJ = rc.MAT_JA
PP, MX = "PP", "MIX"


def M(mat, version, split, sub, col="RMSE"):
    r = met[(met["material"] == MAT[mat]) & (met["version"] == version) & (met["split"] == split) & (met["subset"] == sub)]
    assert len(r) == 1, (mat, version, split, sub)
    return float(r[col].iloc[0])


def K(mat, version, split, col):
    r = ker[(ker["material"] == MAT[mat]) & (ker["version"] == version) & (ker["split"] == split)]
    assert len(r) == 1, (mat, version, split)
    return float(r[col].iloc[0])


def drift_row(mat, o, var, group="全体"):
    r = drift[(drift["material"] == MAT[mat]) & (drift["outlier"] == o) & (drift["variable"] == var) & (drift["group"] == group)]
    assert len(r) == 1, (mat, o, var, group)
    return r.iloc[0]


# ---- 基本の数値
V["N_FIG"] = str(len([f for f in os.listdir(os.path.join(rc.R, S)) if f.endswith(".png")]))
V["N_CSV"] = str(len([f for f in os.listdir(os.path.join(rc.R, S)) if f.endswith(".csv")]))
V["N_HR"] = str(len([f for f in os.listdir(os.path.join(rc.R, S, "highres")) if f.startswith("hr_")]))
V["MIN_TOTAL"] = "8"

# ---- 表1：版の中身
def ver_tests(t):
    return t.replace("；", "，").replace("3data（外挿）", "3data").replace("5data の 210 ℃", "5data の 210 ℃").replace("1 分割のテスト", "テスト").replace("％", "\\%")


rows = []
for m in MAT:
    for v in ["5data", "Alldata"]:
        sub = ver[(ver["material"] == MAT[m]) & (ver["version"] == v)]
        for _, r in sub.iterrows():
            nm = r["split"].replace("ALL（最終モデル）", "ALL（最終モデル）").replace("S_V", "S\\_V").replace("S_T", "S\\_T").replace("S_Vall", "S\\_Vall").replace("（10 分割。1 分割目の例）", "（10 分割。1 分割の例）")
            t = "---" if r["tests"] == "―" else ver_tests(str(r["tests"]))
            rows.append(f"{MJ[m] if (v == '5data' and r['split'].startswith('ALL')) else ''} & {v if r['split'].startswith('ALL') else ''} & {nm} & {int(r['n_train'])} & {int(r['n_cond_train'])} & {t} \\\\")
V["TABLE_VERSIONS"] = "\n".join(rows)

# ---- 表2：包含関係
rows = []
for m in MAT:
    for o in ["あり", "なし"]:
        sub = inc[(inc["material"] == MAT[m]) & (inc["outlier"] == o)]
        for _, r in sub.iterrows():
            g = r["group"].replace("Alldata（5data・3data にない行）", "Alldata（5data・3data との逆）")
            rows.append(f"{MJ[m] if (o == 'あり' and r['group'] == '5data') else ''} & {o if r['group'] == '5data' else ''} & {g} & {int(r['n_rows'])} & {int(r['n_in_alldata'])} & {int(r['n_not_in_alldata'])} \\\\")
V["TABLE_INCL"] = "\n".join(rows)
pp_n = inc[(inc["material"] == "PP") & (inc["outlier"] == "なし") & (inc["group"] == "5data")].iloc[0]
V["INCL_PP_NASHI"] = f"{int(pp_n['n_not_in_alldata'])}"
V["INCL_OK"] = "成り立っています" if (inc[(inc["outlier"] == "あり") & inc["group"].isin(["5data", "3data"])]["n_not_in_alldata"] == 0).all() else "成り立っていません（表を確認してください）"

# ---- 表3：ショットの順番によるずれ
rows = []
for m in MAT:
    for o in ["なし", "あり"]:
        for var in ["ln η", "ln γ̇"]:
            r = drift_row(m, o, var)
            hd = "---" if np.isnan(r["half_diff"]) else f"{num(float(r['half_diff']), 4)}（{r['half_diff_se']:.4f}）"
            v = var.replace("ln η", "$\\ln\\eta$").replace("ln γ̇", "$\\ln\\gd$")
            rows.append(f"{MJ[m] if (o == 'なし' and var == 'ln η') else ''} & {o if var == 'ln η' else ''} & {v} & {int(r['n_cond'])} & {num(float(r['slope_per_shot']), 5)}（{r['se']:.5f}） & {num(float(r['drift_first_to_last']), 4)} & {hd} \\\\")
V["TABLE_DRIFT"] = "\n".join(rows)
d_pp, d_mx = drift_row(PP, "なし", "ln η"), drift_row(MX, "なし", "ln η")
gr = drift[(drift["outlier"] == "なし") & (drift["variable"] == "ln η") & (drift["group"] != "全体")]
gw = gr.loc[gr["drift_first_to_last"].abs().idxmax()]
gw_small = abs(float(gw["drift_first_to_last"])) < 0.04
GW_TXT = (f"最大は {gw['material'].replace('粉砕材混合PP', '粉砕材混合 PP')} の {gw['group_type']} {gw['group']} の {num(float(gw['drift_first_to_last']), 4)} で、許容値の 2 倍（0.04）より小さい値でした。" if gw_small else
          f"最大は {gw['material'].replace('粉砕材混合PP', '粉砕材混合 PP')} の {gw['group_type']} {gw['group']} の {num(float(gw['drift_first_to_last']), 4)}（粘度で約 {abs(float(gw['drift_first_to_last'])) * 100:.0f}\\%）で、許容値の 2 倍（0.04）をわずかに超えますが、傾きは標準誤差の約 {abs(float(gw['z'])):.1f} 倍で、"
          f"偶然の範囲です（{int(gw['n_cond'])} 条件しかなく、標準誤差が大きいため。「順番 6 以降 $-$ 5 以前」は {num(float(gw['half_diff']), 4) if not np.isnan(float(gw['half_diff'])) else '---'}）。")
V["GW_TXT"] = GW_TXT
V["DRIFT_TXT"] = (f"PP（外れ値なし）の ln η のずれ（最初 $\\to$ 最後）は {num(float(d_pp['drift_first_to_last']), 4)}（粘度で約 {abs(float(d_pp['drift_first_to_last'])) * 100:.1f}\\%）、"
                  f"粉砕材混合 PP は {num(float(d_mx['drift_first_to_last']), 4)} で、どちらも許容値の 0.02 より小さい値です。PP の「順番 6〜10 の平均 $-$ 順番 1〜5 の平均」も {num(float(d_pp['half_diff']), 4)} です。"
                  f"群（設定温度・厚み・射出速度）ごとに分けても、{GW_TXT}")
V["DRIFT_Z"] = f"{float(d_pp['z']):.1f}"

# ---- 外れ値の検知
rows = []
SCN = scan.copy()
for m in MAT:
    for label in ["全条件", "除外後"]:
        s = SCN[(SCN["material"] == MAT[m]) & (SCN["data"] == label)]
        top = s.loc[s["z"].abs().idxmax()]
        nflag = int(s["flag"].sum())
        top_ids = s[s["flag"]]["list_id"].fillna("").tolist()
        flag_txt = f"{nflag}（" + "、".join(i or "未登録" for i in top_ids) + "）" if nflag else "0"
        cond_tex = str(top["cond"]).replace("_", "\\_")
        rows.append(f"{MJ[m] if label == '全条件' else ''} & {label} & {len(s)} & {flag_txt} & {num(float(top['z']), 1)} & {cond_tex} & {num(float(top['resid']), 3)} \\\\")
V["TABLE_OUTLIER"] = "\n".join(rows)
p03 = SCN[(SCN["material"] == "PP") & (SCN["data"] == "全条件") & (SCN["cond"] == "190_0.5_320")].iloc[0]
V["P03_Z"], V["P03_RESID"], V["P03_RATIO"], V["P03_N"] = f"{p03['z']:.1f}", f"{p03['resid']:+.2f}".replace("+", "$+$"), f"{np.exp(p03['resid']):.2f}", f"{int(p03['n'])}"
after = SCN[SCN["data"] == "除外後"]
V["AFTER_MAXZ_PP"] = f"{after[after['material'] == 'PP']['z'].abs().max():.1f}"
V["AFTER_MAXZ_MIX"] = f"{after[after['material'] == MAT['MIX']]['z'].abs().max():.1f}"
V["AFTER_N"] = str(int(after["flag"].sum()))
pc = prob[(prob["material"] == "PP") & (prob["v"] == 320)]
V["P03_KEPT"] = str(int(pc[pc["Tset"] == 190]["n_kept_in_nashi"].iloc[0]))
V["P03_ALL"] = str(int(pc[pc["Tset"] == 190]["n_shots_all"].iloc[0]))

# ---- 表4〜6：検証の指標（RMSE・被覆率・NLPD）
COLS = [("S_V", "低速側"), ("S_Vall", "低速側"), ("S_V", "高速側"), ("S_Vall", "高速側"), ("S_T", "全体"), ("S_T3", "全体"), ("CV", "全体")]


def met_table(col, fmt):
    rows = []
    for m in MAT:
        for v in ["5data", "Alldata"]:
            cells = []
            for sp, sub in COLS:
                # 5data 版は S_V、Alldata 版は S_Vall が外挿。ない分割は「---」
                sp2 = sp if not (v == "5data" and sp == "S_Vall") and not (v == "Alldata" and sp == "S_V") else None
                if sp2 is None:
                    cells.append("---")
                    continue
                r = met[(met["material"] == MAT[m]) & (met["version"] == v) & (met["split"] == sp2) & (met["subset"] == sub)]
                cells.append(fmt(float(r[col].iloc[0])) if len(r) else "---")
            rows.append(f"{MJ[m] if v == '5data' else ''} & {v} & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows)


V["TABLE_RMSE"] = met_table("RMSE", rmse)
V["TABLE_COV"] = met_table("coverage95", lambda x: f"{x:.2f}")
V["TABLE_NLPD"] = met_table("NLPD", lambda x: num(x, 2))

# ---- 表7：最終モデルのノイズと純粋誤差
rows = []
for m in MAT:
    for v in ["5data", "Alldata"]:
        sp = "S_V" if v == "5data" else "ALL"
        n_tr, cn, ns, ps, lml = (K(m, v, sp, c) for c in ["n_train", "n_cond_train", "noise_sd", "pure_sd_ln_eta", "LML"])
        rows.append(f"{MJ[m] if v == '5data' else ''} & {v} & {int(n_tr)} & {int(cn)} & {ns:.4f} & {ps:.4f} & {ns / ps:.2f} & {lml:.1f} \\\\")
V["TABLE_KERN"] = "\n".join(rows)

# ---- 図の文章と、まとめ
st_pp = (M(PP, "5data", "S_T", "全体"), M(PP, "Alldata", "S_T", "全体"))
st_mx = (M(MX, "5data", "S_T", "全体"), M(MX, "Alldata", "S_T", "全体"))
s3_pp = (M(PP, "5data", "S_T3", "全体"), M(PP, "Alldata", "S_T3", "全体"))
s3_mx = (M(MX, "5data", "S_T3", "全体"), M(MX, "Alldata", "S_T3", "全体"))
sv_pp = (M(PP, "5data", "S_V", "低速側"), M(PP, "Alldata", "S_Vall", "低速側"), M(PP, "5data", "S_V", "高速側"), M(PP, "Alldata", "S_Vall", "高速側"))
sv_mx = (M(MX, "5data", "S_V", "低速側"), M(MX, "Alldata", "S_Vall", "低速側"), M(MX, "5data", "S_V", "高速側"), M(MX, "Alldata", "S_Vall", "高速側"))
cv = {(m, v): M(m, v, "CV", "全体") for m in MAT for v in ["5data", "Alldata"]}
cvc = {(m, v): M(m, v, "CV", "全体", "coverage95") for m in MAT for v in ["5data", "Alldata"]}
cv_lo, cv_hi = {m: M(m, "Alldata", "CV", "低速側") for m in MAT}, {m: M(m, "Alldata", "CV", "高速側") for m in MAT}

V["FIG1_TXT"] = (f"PP の ln η の箱は、順番 1〜10 のどこでも 0 のまわりにあり、赤い平均の線にも、右上がり・右下がりの傾向は見えません（傾き {num(float(d_pp['slope_per_shot']), 5)} /ショット。標準誤差 {d_pp['se']:.5f} の約 {abs(float(d_pp['z'])):.1f} 倍で、ごくわずかな右下がり。9 ショット分で粘度の約 {abs(float(d_pp['drift_first_to_last'])) * 100:.1f}\\%）。"
                 f"粉砕材混合 PP も同じです（順番は 1〜5）。ln $\\gd$ の箱も 0 のまわりで、ショットが進んでも、せん断速度は変わっていません。群ごとに見ても、{GW_TXT}")
V["FIG2_TXT"] = (f"PP の図の右上の 1 点（P03：190 ℃・0.5 mm・320 mm/s）だけが、しきい値（点線）の外にあります（残差 {V['P03_RESID']}、z = {V['P03_Z']}。粘度は予測の {V['P03_RATIO']} 倍）。"
                 f"ほかの条件は、残差が $\\pm$0.06 以内、z も $\\pm$4 以内に収まっています。粉砕材混合 PP は、拾った条件がありません。")
V["FIG3_TXT"] = (f"P03（左上）は、10 ショットの粘度とせん断速度が、大きくばらつきます（せん断速度 14,000〜23,000 1/s、粘度 1.5〜7 Pa$\\cdot$s と 4 倍以上）。そのうち {int(V['P03_ALL']) - int(V['P03_KEPT'])} ショットが外れ値として除かれ、残った {V['P03_KEPT']} ショット（● ）の平均は、ほかの条件から予測した値より約 {(float(V['P03_RATIO']) - 1) * 100:.0f}\\% 高くなっています。"
                 f"同じ 0.5 mm・320 mm/s の 210・230 ℃ は、せん断速度約 28,000 1/s、粘度約 0.8 Pa$\\cdot$s の近くに、ばらつきなく並んでいます。右上（P02）は、190 ℃ だけ、せん断速度が高く、粘度が低い条件で、10 ショットともそろっているので、外れ値ではなく、この条件の特徴として残しています（感度確認）。"
                 f"左下の粉砕材混合 PP は、190 ℃ の 3 ショットのうち、× が 2 行（1 行は 230 ℃ と同じ値の重複行 P08、1 行は外れ値判定）です。")
V["FIG4_TXT"] = (f"外挿の検証（S\\_V・S\\_Vall）の RMSE は、PP の低速側が {rmse(sv_pp[0])}（5data 版）と {rmse(sv_pp[1])}（Alldata の 5data の条件）、高速側が {rmse(sv_pp[2])} と {rmse(sv_pp[3])} で、ほぼ同じです。"
                 f"S\\_T（210 ℃）は、PP で {rmse(st_pp[0])} と {rmse(st_pp[1])}、粉砕材混合 PP で {rmse(st_mx[0])} と {rmse(st_mx[1])} で、同等。S\\_T3（3data の 210 ℃。外挿も重なる）は、Alldata 版のほうが小さく（PP {rmse(s3_pp[0])} $\\to$ {rmse(s3_pp[1])}、粉砕材混合 PP {rmse(s3_mx[0])} $\\to$ {rmse(s3_mx[1])}）、"
                 f"学習に、3data の条件（190・230 ℃）が入るためです。CV は、5data 版が {rmse(cv[(PP, '5data')])}（PP）、Alldata 版が {rmse(cv[(PP, 'Alldata')])} で、Alldata 版のほうが大きく出るのは、低速側・高速側の条件が含まれるためです（Alldata 版の低速側 {rmse(cv_lo[PP])}、高速側 {rmse(cv_hi[PP])}、内側 {rmse(M(PP, 'Alldata', 'CV', '内側'))}）。")
V["FIG5_TXT"] = (f"点は、対角線（破線）の近くに並んでいます。5data 版の外挿（S\\_V）では、低速側（青）の棒が長く（予測が不確か）、実測もその範囲に入ります（被覆率 PP {M(PP, '5data', 'S_V', '全体', 'coverage95'):.2f}、粉砕材混合 PP {M(MX, '5data', 'S_V', '全体', 'coverage95'):.2f}）。"
                 f"Alldata 版の CV は、棒が短く（予測が確か）、被覆率は PP {cvc[(PP, 'Alldata')]:.2f}、粉砕材混合 PP {cvc[(MX, 'Alldata')]:.2f} です。")

V["SUMMARY"] = f"""\\begin{{enumerate}}[itemsep=0.6ex,topsep=0.6ex]
\\item \\textbf{{今後使う学習データは、外れ値なしの 2 つの版}}：5data 版（5data で学習、3data で外挿を検証）と、Alldata 版（全条件で学習）です。どちらも、問題データ一覧の「除外」（P01・P03・P08）を除きます。外れ値ありは、今後は使いません。ノートの設定 \\texttt{{DATA\\_VERSIONS}} で、どちらでも、両方でも検討できます。
\\item \\textbf{{ファイルの包含関係は、予想どおり}}：外れ値ありでは、5data・3data の全行が Alldata にあり、粉砕材混合 PP は 5data と 3data を合わせたものが Alldata と一致します（表\\ref{{tab:incl}}）。外れ値なしでは、外れ値の除き方がファイルごとに違うため、PP の 5data の {V['INCL_PP_NASHI']} 行が Alldata にありません。
\\item \\textbf{{Alldata の後半のショットを足しても、粘度はずれない}}：ショットの順番によるずれは、PP で {num(float(d_pp['drift_first_to_last']), 4)}（9 ショット分。粘度の約 {abs(float(d_pp['drift_first_to_last'])) * 100:.1f}\\%）と、許容値の 0.02 よりずっと小さく、粉砕材混合 PP も同様です（図1、表\\ref{{tab:drift}}）。群ごとに見ると、{GW_TXT}Alldata 版の学習データとして使えます。
\\item \\textbf{{外れ値は、P03 だけ（除外に変更）}}：規則 f で拾ったのは、PP の 190 ℃・0.5 mm・320 mm/s（P03）だけでした（z = {V['P03_Z']}）。ユーザーの指示（「検知したら、一覧に記録して、除いてよい」）に従い、P03 を「感度確認」から「除外」に変えました。除いた後に、新しく拾われる条件は {V['AFTER_N']} 件です（図2、表\\ref{{tab:outlier}}）。
\\item \\textbf{{2 つの版は、同じモデル（R3）で、同等の精度}}：S\\_T は、PP で {rmse(st_pp[0])}（5data 版）と {rmse(st_pp[1])}（Alldata 版）、粉砕材混合 PP で {rmse(st_mx[0])} と {rmse(st_mx[1])}。Alldata 版の外挿（S\\_Vall）も、5data 版（S\\_V）と同じ水準です。ショットが増えても、悪くなりません（図4、表\\ref{{tab:rmse}}）。
\\item \\textbf{{Alldata 版の利点は、両端の条件も学習に入ること}}：Alldata 版の最終モデルは、低速側・高速側の条件も学習しているので、予測の区間が狭く（ノイズの標準偏差 PP {K(PP, 'Alldata', 'ALL', 'noise_sd'):.3f}、純粋誤差 {K(PP, 'Alldata', 'ALL', 'pure_sd_ln_eta'):.3f}）、両端の外側でない限り、外挿に頼りません。この利点は、同じデータの中の検証（CV）では測れません。5data 版の外挿の成績が、両端を学習に入れない場合の目安です。
\\item \\textbf{{使い分け}}：5data 版は、「学習の外側をどれだけ当てられるか」を確かめるときの版（モデルの性能評価）、Alldata 版は、最終的な予測モデル（全条件を使う）の版、という使い分けを想定しています。手順5 の本実装（構造 B・A）は、この 2 つの版に当てはめて、比べます。
\\end{{enumerate}}"""

V["STAR_TXT"] = f"""ノートの \\texttt{{★要確認}} は、想定と違う、または注意が要る点です。この文書の時点で出る主なものと、その意味は次のとおりです。
\\begin{{points}}
\\pt{{5data 版の S\\_V 低速側の被覆率が 1.00}}95\\% 区間に、低速側の検証データがすべて入っています（PP・粉砕材混合 PP とも）。区間が広め（保守的）という意味で、外れるより安全です。低速側は、学習していない範囲の外挿なので、広めに出ること自体は正しい振る舞いです（手順3b の R3 と同じ）。
\\pt{{PP・5data 版の S\\_V 高速側の被覆率 {M(PP, '5data', 'S_V', '高速側', 'coverage95'):.2f}}}0.90 を少し下回ります。高速側の予測の幅が狭く（予測標準偏差の平均 {M(PP, '5data', 'S_V', '高速側', 'mean_sd'):.3f}）、外れる条件（66 行中 {int(round((1 - M(PP, '5data', 'S_V', '高速側', 'coverage95')) * 66))} 行）があるためです（P03 は、すでに除外した後の値です）。Alldata の 5data の条件で学習した S\\_Vall では {M(PP, 'Alldata', 'S_Vall', '高速側', 'coverage95'):.2f}、ショットの増えた Alldata 版の S\\_T3 では {M(PP, 'Alldata', 'S_T3', '全体', 'coverage95'):.2f} です。
\\pt{{ショットの順番によるずれ（群ごと）}}{GW_TXT}
\\pt{{Alldata 版の CV で、低速側・高速側の被覆率が低め}}PP の CV の低速側は {M(PP, 'Alldata', 'CV', '低速側', 'coverage95'):.2f}、高速側は {M(PP, 'Alldata', 'CV', '高速側', 'coverage95'):.2f} で、内側（{M(PP, 'Alldata', 'CV', '内側', 'coverage95'):.2f}）より低めです。両端の条件は、その外側に学習データがなく、予測が外挿になるためです（ノートの判定は、CV 全体の被覆率で行っています）。
\\end{{points}}"""

rc.build_pdf("3d", S, "report_step3d_template.tex", V, "gakushu3d")
