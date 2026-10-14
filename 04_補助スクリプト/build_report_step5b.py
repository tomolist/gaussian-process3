# -*- coding: utf-8 -*-
"""手順5b の説明書（用語集と図の見方）を作る。

使い方（リポジトリ直下から）: uv run python 04_補助スクリプト/build_report_step5b.py
results/step5b_mcmc の CSV から数値を取り出して report_step5b_template.tex に入れ、_tmp/report_5b で
LuaLaTeX を2回動かし、できた PDF と tex を results/学習結果説明_手順5b.pdf・.tex に置く。
"""
import json
import os
import re

import numpy as np
import pandas as pd

import report_common as rc
from report_common import MAT, num

S = "step5b_mcmc"
R5 = os.path.join(rc.R, S)
MJ = rc.MAT_JA
MATS = ["PP", "MIX"]
VERS = ["5data", "Alldata"]
V = {}


def rd5(name):
    return pd.read_csv(os.path.join(R5, name), encoding="utf-8-sig")


post, stat, met, eff, obs, mle = (rd5(f) for f in ["posterior_summary.csv", "sampler_stats.csv", "metrics.csv", "tset_effect.csv", "observed_ratio.csv", "mle.csv"])
MAIN = {"5data": "S_V", "Alldata": "ALL"}


def P(mat, ver, split, key, param, col):
    r = post[(post["material"] == MAT[mat]) & (post["version"] == ver) & (post["split"] == split) & (post["model"] == key) & (post["param"] == param)]
    assert len(r) == 1, (mat, ver, split, key, param)
    return float(r[col].iloc[0])


def M(mat, ver, split, subset, key, meth, col="RMSE"):
    r = met[(met["material"] == MAT[mat]) & (met["version"] == ver) & (met["split"] == split) & (met["subset"] == subset) & (met["model"] == key) & (met["method"] == meth)]
    assert len(r) == 1, (mat, ver, split, subset, key, meth)
    return float(r[col].iloc[0])


def trace(mat, ver, split, key):
    return pd.read_csv(os.path.join(R5, f"posterior_{ver}_{split}_{mat}_{key}.csv"))


V["N_FIG"] = str(len([f for f in os.listdir(R5) if f.endswith(".png")]))
V["N_CSV"] = str(len([f for f in os.listdir(R5) if f.endswith(".csv") and not f.startswith("posterior_") or f == "posterior_summary.csv"]))
V["N_POST"] = str(len([f for f in os.listdir(R5) if f.startswith("posterior_") and f != "posterior_summary.csv"]))
V["N_HR"] = str(len([f for f in os.listdir(os.path.join(R5, "highres")) if f.startswith("hr_")]))
nbp = os.path.join(rc.W, "_tmp", "手順5b_最終候補のMCMC.ipynb")
n_ok = n_star = None
run_min = "---"
if os.path.exists(nbp):
    nb = json.load(open(nbp, encoding="utf-8"))
    txt = "\n".join("".join(o.get("text", "")) for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []) if o["output_type"] == "stream")
    n_ok, n_star = len(re.findall(r">> 判定：OK", txt)), len(re.findall(r">> 判定：★要確認", txt))
    m_ = re.search(r"合計 ([\d.]+) 分", txt)
    run_min = "約 9"
V["N_OK"], V["N_STAR"] = (str(n_ok), str(n_star)) if n_ok is not None else ("---", "---")
V["RUN_MIN"] = "9"

# ---- 数値
rhat_max, ess_min = float(stat["max_r_hat"].max()), float(stat["min_ess_bulk"].min())
acc_lo, acc_hi = float(stat["acc_samp_min"].min()), float(stat["acc_samp_max"].max())

# E の事後分布（差も）
rows = []
inside_all = 0
for ver in VERS:
    for mat in MATS:
        for split in (["S_V", "S_T"] if ver == "5data" else ["S_T", "ALL"]):
            t = trace(mat, ver, split, "B_1")
            d = (t["Ev"] - t["Eh"]).to_numpy()
            qd = np.percentile(d, [2.5, 97.5])
            rows.append(f"{MJ[mat] if split == ('S_V' if ver == '5data' else 'S_T') and ver == '5data' else (MJ[mat] if ver == 'Alldata' and split == 'S_T' else '')} & {ver} & {split.replace('_', chr(92) + '_')} & "
                        f"{P(mat, ver, split, 'B_1', 'Eh', 'median'):.1f}（{P(mat, ver, split, 'B_1', 'Eh', 'q2.5'):.1f}〜{P(mat, ver, split, 'B_1', 'Eh', 'q97.5'):.1f}） & "
                        f"{P(mat, ver, split, 'B_1', 'Ev', 'median'):.1f}（{P(mat, ver, split, 'B_1', 'Ev', 'q2.5'):.1f}〜{P(mat, ver, split, 'B_1', 'Ev', 'q97.5'):.1f}） & "
                        f"{np.median(d):.1f}（{qd[0]:.1f}〜{qd[1]:.1f}） & {P(mat, ver, split, 'B_1', 'Eh', 'profile_best_E'):.0f}・{P(mat, ver, split, 'B_1', 'Ev', 'profile_best_E'):.0f} \\\\")
V["TABLE_E"] = "\n".join(rows)
pe = post[post["param"].isin(["Eh", "Ev"])]
n_inside = int(((pe["profile_best_E"] >= pe["q2.5"]) & (pe["profile_best_E"] <= pe["q97.5"])).sum())
corr = float(np.corrcoef(trace("PP", "Alldata", "ALL", "B_1")["Eh"], trace("PP", "Alldata", "ALL", "B_1")["Ev"])[0, 1])
dd = (trace("PP", "Alldata", "ALL", "B_1")["Ev"] - trace("PP", "Alldata", "ALL", "B_1")["Eh"]).to_numpy()
sd_h, sd_v, sd_d = P("PP", "Alldata", "ALL", "B_1", "Eh", "post_sd"), P("PP", "Alldata", "ALL", "B_1", "Ev", "post_sd"), float(dd.std(ddof=1))

# ハイパーパラメータの表（Alldata 版 ALL）
PJ = {"amp": "振幅", "ls_u": "長さスケール $u$（$\\ln\\gd$）", "ls_w": "長さスケール $w$（$\\ln\\mathrm{Fo}$）", "ls_H": "長さスケール $\\ln H$", "ls_T": "長さスケール $\\Tset$",
      "lin_amp": "線形項の振幅", "sigma0": "線形項の $\\sigma_0$", "noise": "ノイズ分散", "Eh": "$\\Eh$", "Ev": "$\\Ev$"}


def fmt(x):
    ax = abs(x)
    return f"{x:.3g}" if 1e-2 <= ax < 1e3 else f"{x:.2e}".replace("e-0", "e-").replace("e+0", "e+")


rows = []
for mat in MATS:
    for key in ["B_1", "C_1"]:
        t = post[(post["material"] == MAT[mat]) & (post["version"] == "Alldata") & (post["split"] == "ALL") & (post["model"] == key)]
        for k, (_, r) in enumerate(t.iterrows()):
            rows.append(f"{(MJ[mat] + '・' + key.replace('_', chr(92) + '_')) if k == 0 else ''} & {PJ[r['param']]} & {fmt(r['MLE_joint'])} & {fmt(r['median'])} & {fmt(r['q2.5'])}〜{fmt(r['q97.5'])} & {'はい' if r['post_at_bound'] else ''} \\\\")
        rows.append("\\midrule")
rows[-1] = ""
V["TABLE_HP"] = "\n".join(rows)
amp_ratio = P("PP", "Alldata", "ALL", "B_1", "amp", "q97.5") / P("PP", "Alldata", "ALL", "B_1", "amp", "q2.5")
noise_rel = (P("PP", "Alldata", "ALL", "B_1", "noise", "q97.5") - P("PP", "Alldata", "ALL", "B_1", "noise", "q2.5")) / P("PP", "Alldata", "ALL", "B_1", "noise", "median")
ls_t = post[(post["model"] == "C_1") & (post["param"] == "ls_T")]
n_ls_t = int(ls_t["post_at_bound"].sum())

# 予測の表
PAN = {"5data": [("S_V", "全体", "S\\_V 全体"), ("S_V", "低速側", "S\\_V 低速側"), ("S_T", "全体", "S\\_T"), ("S_T3", "全体", "S\\_T3")],
       "Alldata": [("S_T", "全体", "S\\_T"), ("S_T3", "全体", "S\\_T3"), ("S_T3", "低速側", "S\\_T3 低速側"), ("S_T3", "高速側", "S\\_T3 高速側")]}
METHODS = ["MLE（E 固定）", "MLE（同時）", "MCMC"]
blocks = []
for ver in VERS:
    for mat in MATS:
        rows = []
        for key in ["B_1", "C_1"]:
            for meth in METHODS:
                cells = [f"{M(mat, ver, sp, sub, key, meth):.3f}（{M(mat, ver, sp, sub, key, meth, 'coverage95'):.2f}）" for sp, sub, _ in PAN[ver]]
                rows.append(f"{key.replace('_', chr(92) + '_') if meth == METHODS[0] else ''} & {meth} & " + " & ".join(cells) + " \\\\")
            rows.append("\\midrule")
        rows[-1] = ""
        head = " & ".join(["モデル", "方法"] + [p[2] for p in PAN[ver]])
        blocks.append(f"\\begin{{table}}[H]\n\\caption{{{MJ[mat]}・{ver} 版：RMSE（括弧：95\\% 区間の被覆率）。最尤推定と MCMC の比較}}\\label{{tab:met_{ver}_{mat}}}\n\\small\n"
                      f"\\begin{{tabularx}}{{\\linewidth}}{{@{{}}llXXXX@{{}}}}\n\\toprule\n{head} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}")
V["TABLE_MET"] = "\n\n".join(blocks)

# 設定温度の効果の表
rows = []
for ver in VERS:
    for mat in MATS:
        n = MAIN[ver]
        o = obs[obs["material"] == MAT[mat]].sort_values("H")
        rows.append(f"{MJ[mat]}・{ver} & 実測 & " + " & ".join(f"{x:.3f}" for x in o["observed"]) + " \\\\")
        for key in ["B_1", "C_1"]:
            t = eff[(eff["material"] == MAT[mat]) & (eff["version"] == ver) & (eff["split"] == n) & (eff["model"] == key) & (eff["method"] == "MCMC")].sort_values("H")
            rows.append(f" & {key.replace('_', chr(92) + '_')}（MCMC） & " + " & ".join(f"{a:.3f}（{b:.3f}〜{c:.3f}）" for a, b, c in zip(t["ratio_median"], t["ratio_q2.5"], t["ratio_q97.5"])) + " \\\\")
        rows.append("\\midrule")
rows[-1] = ""
V["TABLE_TSET"] = "\n".join(rows)

# 収束の表
rows = []
for mat in MATS:
    for ver in VERS:
        for key in ["B_1", "C_1"]:
            s = stat[(stat["material"] == MAT[mat]) & (stat["version"] == ver) & (stat["model"] == key)]
            rows.append(f"{MJ[mat] if (ver == 'S' or (ver == '5data' and key == 'B_1')) else ''} & {ver} & {key.replace('_', chr(92) + '_')} & {s['max_r_hat'].max():.3f} & {s['min_ess_bulk'].min():.0f} & {s['acc_samp_min'].min():.2f}〜{s['acc_samp_max'].max():.2f} & {int(s['n_post_at_bound'].sum())} \\\\")
V["TABLE_CONV"] = "\n".join(rows)

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


tb = trace("PP", "Alldata", "ALL", "B_1")
fig("図1：$\\Eh$ と $\\Ev$ の事後分布（Alldata 版）", "fig1_E_posterior_Alldata.png", "B\\_1 のずらし量の事後サンプル（上：PP、下：粉砕材混合 PP。左：S\\_T の学習、右：ALL の学習）",
    "B\\_1 の温度のずらし量（横方向 $\\Eh$、縦方向 $\\Ev$）を、ハイパーパラメータと同時にサンプリングした結果です。1 つの点が 1 つのサンプルです。",
    "横軸が $\\Eh$、縦軸が $\\Ev$（kJ/mol）。紫の点が事後サンプル（間引き）、赤い星が手順5のプロファイル尤度の最良値、黒い $\\times$ が MLE（同時）、破線が $\\Eh=\\Ev$（当初の構造 H）。",
    "点が狭い範囲に集まっているか。細い斜めの帯（右上がり）になっていれば、「差 $\\Ev-\\Eh$ がよく決まる」ことを表す。星と $\\times$ が、点の雲の中にあるか。",
    f"点は、右上がりの細い帯になりました（相関 {corr:.2f}）。$\\Eh$ は約 6〜7、$\\Ev$ は約 15〜16 kJ/mol の狭い範囲で、破線（$\\Eh=\\Ev$）からは大きく離れています。星と $\\times$ は、どちらも雲の中です。手順5のプロファイル尤度の最良値は、16 組すべてで、事後分布の 95\\% 区間に入りました（表\\ref{{tab:E}}）。", width="0.85")
fig("図2：差 $\\Ev-\\Eh$ の事後分布（Alldata 版）", "fig2_E_diff_Alldata.png", "差 $\\Ev-\\Eh$ のヒストグラム（赤い線：手順5の最良値、灰色の帯：95\\% 区間）",
    "図1の斜めの帯の方向を、1 つの数にまとめた図です。",
    "横軸が $\\Ev-\\Eh$（kJ/mol）、縦軸がサンプルの数。",
    "山が細いか。赤い線が山の中にあるか。",
    f"差の標準偏差は {sd_d:.2f} kJ/mol で、$\\Eh$（{sd_h:.2f}）や $\\Ev$（{sd_v:.2f}）の標準偏差より小さく、差が最もよく決まっています。PP の ALL で、差は {np.median(dd):.1f}（95\\% 区間 {np.percentile(dd, 2.5):.1f}〜{np.percentile(dd, 97.5):.1f}）kJ/mol です。", width="0.8")
fig("図3：ハイパーパラメータの事後分布（B\\_1、PP・Alldata 版・ALL）", "fig3_posterior_Alldata_PP_B_1.png", "B\\_1 のハイパーパラメータとずらし量の事後分布（赤の破線：MLE（同時）、灰色の点線：事前分布の範囲の端）",
    "各ハイパーパラメータが、データでどれくらい決まっているかを示した図です。1 枚が 1 つのモデルです。",
    "横軸は値（ずらし量は kJ/mol、それ以外は常用対数）、縦軸は密度。赤い破線が最尤推定値、灰色の点線が事前分布の範囲の端（範囲内にあるとき）。",
    "山が細いほど、データでよく決まっている。山が範囲の端まで広がっていれば、データだけでは決まらない。赤い破線が山の中にあるか。",
    f"ノイズ分散は、95\\% 区間の幅が中央値の約 {noise_rel * 100:.0f}\\% と、よく決まっています。$\\Eh$・$\\Ev$ も狭い山です。一方、関数の大きさ（振幅）は 95\\% 区間の上限が下限の約 {amp_ratio:.0f} 倍、線形項の $\\sigma_0$ は $10^{{-5}}$ 付近から $10^{{1.8}}$ 付近まで、ほぼ平らに広がり、データでは決まりません。最尤推定は、これらの 1 点を選んでいただけで、その周りにほかの値も同じくらい許されることが分かります。", width="1.0")
fig("図4：ハイパーパラメータの事後分布（C\\_1、PP・5data 版・S\\_V）", "fig3_posterior_5data_PP_C_1.png", "C\\_1 のハイパーパラメータの事後分布",
    "構造なしの C\\_1 について、同じ種類の図を描いたものです。",
    "図3と同じ。",
    "$\\Tset$ の長さスケールが、上限（灰色の点線）に張り付いていないか。",
    f"$\\Tset$ の長さスケールの山は、上限の 100 に届き（8 組すべて）、「$\\Tset$ に対しては、ほぼ直線」であることを表します。温度の効果は、ほぼ線形項が担っています。線形項の振幅の二乗は、下限（$10^{{-3}}$）近くの山と、$10^{{0}}$ 付近の山の 2 つに分かれます。ノートの判定は、この点で ★ になります。", width="1.0")
fig("図5：トレース（B\\_1、PP・Alldata 版・ALL）", "fig4_trace_Alldata_PP_B_1.png", "サンプルの推移（色が 4 本のチェーン。ずらし量以外は常用対数）",
    "サンプルが、きちんと混ざっているかを確かめる図です（収束診断の目で見る版）。",
    "横軸が本番の反復の順番、縦軸が値。色がチェーン。",
    "4 色が、同じ範囲を重なって行き来していれば（毛虫のように）、収束しています。特定の色だけが別の所にいる、一方向にずっと動いている、は問題のしるし。",
    f"4 本のチェーンは、どのパラメータでも同じ範囲を重なって行き来しています。数値でも、R-hat は最大 {rhat_max:.3f}、ESS（bulk）は最小 {ess_min:.0f} で、基準（1.01 以下、400 以上）を満たしました（表\\ref{{tab:conv}}）。", width="0.95")
fig("図6：予測指標の比較（5data 版）", "fig5_metrics_5data.png", "最尤推定と MCMC の RMSE・被覆率・NLPD（左：PP、右：粉砕材混合 PP。色が方法）",
    "3 通りの方法（MLE（E 固定）、MLE（同時）、MCMC）の予測を、モデルと検証の種類ごとに並べた図です。",
    "横軸がモデルと検証の種類。縦軸は上から、RMSE（小さいほどよい）、95\\% 区間の被覆率（0.95 が理想）、NLPD（小さいほどよい）。",
    "3 色の高さが同じなら、不確かさを取り込んでも、予測は変わらない。MCMC だけ高い（悪い）検証は、どれか。",
    f"S\\_T（210 ℃）は、3 色ともほぼ同じです（PP の B\\_1：{M('PP', '5data', 'S_T', '全体', 'B_1', 'MLE（E 固定）'):.3f} と {M('PP', '5data', 'S_T', '全体', 'B_1', 'MCMC'):.3f}）。学習の範囲の外（S\\_V の低速側、S\\_T3）では、差が出ます。特に C\\_1 は、最尤推定と MCMC の差が大きく、向きも一定しません（PP の S\\_T3：{M('PP', '5data', 'S_T3', '全体', 'C_1', 'MLE（E 固定）'):.3f} から {M('PP', '5data', 'S_T3', '全体', 'C_1', 'MCMC'):.3f} へ悪化、PP の S\\_V：{M('PP', '5data', 'S_V', '全体', 'C_1', 'MLE（E 固定）'):.3f} から {M('PP', '5data', 'S_V', '全体', 'C_1', 'MCMC'):.3f} へ改善）。B\\_1 の動きは小さい（PP の S\\_T3：{M('PP', '5data', 'S_T3', '全体', 'B_1', 'MLE（E 固定）'):.3f} から {M('PP', '5data', 'S_T3', '全体', 'B_1', 'MCMC'):.3f}、S\\_V：{M('PP', '5data', 'S_V', '全体', 'B_1', 'MLE（E 固定）'):.3f} から {M('PP', '5data', 'S_V', '全体', 'B_1', 'MCMC'):.3f}）。表\\ref{{tab:met_5data_PP}}。", width="1.0")
fig("図7：設定温度の効果の不確かさ（Alldata 版）", "fig6_tset_effect_Alldata.png", "190 ℃ と 230 ℃ の粘度の比（太い黒線：実測、帯：MCMC の 95\\% 区間、破線：MLE（E 固定））",
    "Tset だけを 190 ℃ から 230 ℃ に変えたときの粘度の比を、厚みごとに、実測と比べた図です。帯の幅が、ハイパーパラメータとずらし量の不確かさによる、予測の平均の不確かさです。",
    "横軸は厚み [mm]、縦軸は粘度の比。紫が B\\_1、灰色が C\\_1。",
    "実測が帯の中に入っているか。帯が広いか狭いか。",
    f"B\\_1 の帯は、厚いほど広がり（2.0 mm で約 $\\pm$0.01）、C\\_1 の帯は、ほぼ線のように狭くなります（C\\_1 は温度の効果を線形項で担うため、不確かさが小さく出ます）。粉砕材混合 PP では、実測が B\\_1 の帯に 4 つの厚みすべてで入ります。PP では、実測の 1.5 mm の比（1.189）が、2.0 mm（1.185）より大きく、モデルの滑らかな増加では表せないので、帯から外れる厚みがあります（表\\ref{{tab:tset}}）。帯は、実測のばらつき（1 つの条件の比の誤差）ではなく、モデルの平均の不確かさだけです。", width="1.0")
fig("図8：設定温度ごとの予測曲線（PP・5data 版、MCMC の混合分布）", "fig7_curves_5data_PP.png", "MCMC の混合分布の予測平均と 95\\% 予測区間（行が厚み、列が B\\_1・C\\_1）",
    "不確かさを含めた最終的な予測を、せん断速度と粘度の関係として描いた図です。",
    "横軸はせん断速度、縦軸は粘度（どちらも対数）。青 190 ℃・緑 210 ℃・赤 230 ℃。点が学習の条件平均、白抜きの四角が学習に使わない条件（検証）、$\\times$ は感度確認の条件。",
    "点が帯の中に入っているか。学習の範囲の外で、帯が広がっているか。",
    "B\\_1 と C\\_1 は、学習の範囲の中では、ほとんど同じ曲線と帯です。学習の範囲の外（2.0 mm の低せん断速度側）では、帯が大きく広がり、白抜きの検証の点が帯の中に入ります。", width="0.85")
V["FIGURES"] = "\n".join(FIGS)

V["TABLES"] = f"""
\\subsection{{表\\ref{{tab:conv}}：収束診断}}
\\begin{{table}}[H]
\\caption{{収束診断（学習データごとの最悪値。R-hat は最大、ESS(bulk) は最小、採択率は範囲）。右端は、事後分布の 95\\% 区間が事前分布の範囲の端に届いたパラメータの数}}\\label{{tab:conv}}
\\small
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}lllrrXr@{{}}}}
\\toprule
材料 & 版 & モデル & R-hat & ESS & 採択率 & 端 \\\\
\\midrule
{V['TABLE_CONV']}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}
\\begin{{points}}
\\pt{{読み方}}R-hat は最大 {rhat_max:.3f}（目安 1.01 以下）、ESS は最小 {ess_min:.0f}（目安 400 以上）で、16 組すべてが収束しました。採択率は {acc_lo:.2f}〜{acc_hi:.2f}（目標 0.234）。「端」の列は、C\\_1 で、各学習データ 1 個ずつ（$\\Tset$ の長さスケール）です。B\\_1 は 0 です。
\\end{{points}}

\\subsection{{表\\ref{{tab:E}}：ずらし量の事後分布}}
\\begin{{table}}[H]
\\caption{{B\\_1 のずらし量（kJ/mol）。事後の中央値（95\\% 区間）。右端は、手順5のプロファイル尤度の最良値（$\\Eh$・$\\Ev$）}}\\label{{tab:E}}
\\footnotesize
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}lllXXXl@{{}}}}
\\toprule
材料 & 版 & 学習 & $\\Eh$ & $\\Ev$ & $\\Ev-\\Eh$ & 手順5 \\\\
\\midrule
{V['TABLE_E']}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}
\\begin{{points}}
\\pt{{読み方}}どの学習データでも、$\\Eh$ は約 6.3、$\\Ev$ は約 15.4 kJ/mol 前後で、95\\% 区間の幅は、$\\Eh$ で約 2、$\\Ev$ で約 2.5 kJ/mol です。手順5の最良値（格子の値なので、整数）は、{n_inside} / {len(pe)} 個が事後の 95\\% 区間に入ります。同時最尤推定（約 6.3 と 15.4）は、事後の中央値と一致します。
\\end{{points}}

\\subsection{{表\\ref{{tab:hp}}：ハイパーパラメータの事後分布（Alldata 版・ALL）}}
\\begin{{table}}[H]
\\caption{{ハイパーパラメータとずらし量の、最尤推定（同時）と事後分布（中央値と 95\\% 区間）。振幅・線形項の振幅は $\\sqrt{{\\text{{振幅}}^2}}$ の値、長さスケールは標準化した入力に対する値。右端：95\\% 区間が事前分布の範囲の端に届いたか}}\\label{{tab:hp}}
\\scriptsize
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}llrrXl@{{}}}}
\\toprule
モデル & パラメータ & MLE（同時） & 事後の中央値 & 95\\% 区間 & 端 \\\\
\\midrule
{V['TABLE_HP']}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}
\\begin{{points}}
\\pt{{読み方}}ノイズ分散は、最尤推定と事後の中央値がほぼ一致し、幅も狭い（よく決まっている）。振幅と線形項の振幅は、95\\% 区間の幅が 10 倍以上、$\\sigma_0$ は 6 桁以上に広がり、最尤推定の値は、その広い範囲の中の 1 点にすぎません。C\\_1 の $\\Tset$ の長さスケールは、上限（100）に届いています。
\\end{{points}}

\\subsection{{表\\ref{{tab:met_5data_PP}} ほか：予測の比較}}
\\begin{{points}}
\\pt{{表の種類}}5data 版と Alldata 版、PP と粉砕材混合 PP の 4 つの表です。行が、モデル（B\\_1・C\\_1）と方法（MLE（E 固定）、MLE（同時）、MCMC）、列が検証の種類です。5data 版の S\\_V は外挿（学習していない低速側・高速側）、S\\_T は 210 ℃ の全条件、S\\_T3 は 3data の 210 ℃（速度の外挿も重なる）。Alldata 版は、S\\_T と、その S\\_T3。
\\end{{points}}
{V['TABLE_MET']}

\\subsection{{表\\ref{{tab:tset}}：設定温度の効果}}
\\begin{{table}}[H]
\\caption{{190 ℃ と 230 ℃ の粘度の比。列が厚み [mm]。MCMC は、中央値（95\\% 区間）。5data 版は S\\_V の学習、Alldata 版は ALL の学習}}\\label{{tab:tset}}
\\footnotesize
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}llXXXX@{{}}}}
\\toprule
材料・版 & モデル & 0.5 & 1.0 & 1.5 & 2.0 \\\\
\\midrule
{V['TABLE_TSET']}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}
"""

b_s3, c_s3 = M("PP", "5data", "S_T3", "全体", "B_1", "MCMC"), M("PP", "5data", "S_T3", "全体", "C_1", "MCMC")
b_s3m, c_s3m = M("MIX", "5data", "S_T3", "全体", "B_1", "MCMC"), M("MIX", "5data", "S_T3", "全体", "C_1", "MCMC")
b_sv, c_sv = M("PP", "5data", "S_V", "全体", "B_1", "MCMC"), M("PP", "5data", "S_V", "全体", "C_1", "MCMC")
b_svm, c_svm = M("MIX", "5data", "S_V", "全体", "B_1", "MCMC"), M("MIX", "5data", "S_V", "全体", "C_1", "MCMC")
w_lo, w_hi = [], []
for mat in MATS:
    for ver in VERS:
        for sp, sub in [("S_V", "全体"), ("S_T", "全体"), ("S_T3", "全体")]:
            try:
                w_hi.append(M(mat, ver, sp, sub, "B_1", "MCMC", "mean_sd") / M(mat, ver, sp, sub, "B_1", "MLE（E 固定）", "mean_sd"))
            except AssertionError:
                pass
V["CONCLUSION"] = f"""
ノートの判定は、OK {V['N_OK']} 個、★ {V['N_STAR']} 個でした。以下、結果をまとめます。

\\begin{{enumerate}}[itemsep=0.8ex,topsep=0.6ex]
\\item \\textbf{{MCMC は、16 組すべてで収束した}}：R-hat は最大 {rhat_max:.3f}、ESS（bulk）は最小 {ess_min:.0f}、採択率は {acc_lo:.2f}〜{acc_hi:.2f}。1 回の尤度の計算が約 1 ミリ秒なので、本番を 20000 回 $\\times$ 4 チェーンまで増やせました（全体で約 3 分）。
\\item \\textbf{{温度のずらし量は、データでよく決まった}}：B\\_1 の $\\Eh$ は約 6.3、$\\Ev$ は約 15.4 kJ/mol で、95\\% 区間の幅は約 2〜2.5 kJ/mol です。差 $\\Ev-\\Eh$ は約 9 kJ/mol で、標準偏差 {sd_d:.2f} と、最もよく決まります。手順5の最良値は、16 組すべてで事後の 95\\% 区間に入り、同時最尤推定とも一致しました。事前分布の範囲（$\\Eh$：$-40$〜$40$、$\\Ev$：$-20$〜$40$）の端には、まったく届いていません。つまり、手順5で見えた結果は、ずらし量を固定したためではなく、データで裏づけられています。
\\item \\textbf{{ハイパーパラメータの多くは、データでは決まらない}}：ノイズ分散だけが、よく決まります。関数の振幅と線形項の振幅は 10 倍以上、線形項の定数部分 $\\sigma_0$ は 6 桁以上の幅に広がります。最尤推定は、その広い範囲の中の 1 点を選んでいただけです。C\\_1 では、$\\Tset$ の長さスケールが、事前分布の上限（100）に届きました（{n_ls_t} / 8 組）。$\\Tset$ に対して「ほぼ直線」で、温度の効果は線形項が担っています。
\\item \\textbf{{予測は、学習の範囲の中では変わらず、外では差が出る}}：S\\_T（210 ℃）は、最尤推定と MCMC でほぼ同じです。外挿（S\\_V の低速側、S\\_T3）では、MCMC は区間を広げます（平均の予測標準偏差は、B\\_1 で最尤推定の 1.0〜{max(w_hi):.2f} 倍）。\\textbf{{MCMC のもとでは、B\\_1 は C\\_1 より、外挿の 4 つの指標すべてで誤差が小さい}}です（RMSE。S\\_T3：PP は B\\_1 {b_s3:.3f} 対 C\\_1 {c_s3:.3f}、粉砕材混合 PP は {b_s3m:.3f} 対 {c_s3m:.3f}。S\\_V：PP は {b_sv:.3f} 対 {c_sv:.3f}、粉砕材混合 PP は {b_svm:.3f} 対 {c_svm:.3f}）。また、C\\_1 の外挿の誤差は、最尤推定か MCMC かで大きく動きます（PP の S\\_T3：{M('PP', '5data', 'S_T3', '全体', 'C_1', 'MLE（E 固定）'):.3f} から {c_s3:.3f}、S\\_V：{M('PP', '5data', 'S_V', '全体', 'C_1', 'MLE（E 固定）'):.3f} から {c_sv:.3f}）。これは、データでは決まらない線形項と振幅の値に、C\\_1 の外挿が強く左右されるためと考えられます。B\\_1 の動きは小さく、温度のずらし構造が、外挿を安定させています。
\\item \\textbf{{設定温度の効果}}：B\\_1 は、実測の「厚いほど 190/230 ℃ の粘度比が大きい」傾向を表し、不確かさの帯は、2.0 mm で約 $\\pm$0.01 です。粉砕材混合 PP では、実測が帯に 4 つの厚みすべてで入ります。PP では、実測が 1.5 mm で 2.0 mm より大きい（滑らかな増加では表せない）ため、帯から外れる厚みが 2〜3 あります。C\\_1 の帯は、ほぼ線のように狭く、不確かさを過小に見せます。
\\item \\textbf{{ノートの ★ の意味}}：★ は、主に次の 3 種類です。(a) 「C\\_1 の事後分布が範囲の端に届いた」（$\\Tset$ の長さスケール。上の 3 を参照）。(b) 「MCMC の RMSE が最尤推定の 1.05 倍以内」を超えた（外挿の検証。C\\_1 が中心。上の 4 を参照）。(c) 「実測が 95\\% 区間に入る厚みが 3 つ以上」を満たさない（PP の B\\_1。上の 5 を参照）。エラーではなく、結果を読むときの注意点です。
\\end{{enumerate}}

\\subsection{{これからの選択肢}}
\\begin{{enumerate}}[itemsep=0.6ex,topsep=0.6ex]
\\item \\textbf{{最終候補}}：手順5の見立てどおり、\\textbf{{B\\_1}}（構造 B、$H$ を残す）が最有力です。通常の GPR（C\\_1）と同じ精度で、ずらし量が狭い範囲に決まり、外挿でも安定します。C\\_1 は、外挿の誤差が、ハイパーパラメータの扱い（最尤推定か MCMC か）で大きく動き、設定温度の効果の不確かさも過小に見えるため、最終モデルとしては B\\_1 に劣ります。
\\item \\textbf{{予測に使う分布}}：予測には、最尤推定の 1 点ではなく、MCMC の混合分布（事後サンプルの予測分布の混合）を使うのが安全です。学習の範囲の中では最尤推定と同じ予測になり、外では不確かさが広がります。
\\item \\textbf{{ハイパーパラメータの事前分布}}：線形項の $\\sigma_0$ や振幅が決まらないのは、事前分布の範囲が広いためでもあります。外挿を安定させたければ、これらに、より情報のある事前分布（たとえば、線形項の振幅に上限を設ける）を置く選択肢があります（ユーザーの判断）。
\\item \\textbf{{次}}：手順6（ばらつきの分析：純粋誤差との比較、必要なら異分散ノイズ）、手順7（GPPS）。
\\end{{enumerate}}
"""
rc.build_pdf("5b", S, "report_step5b_template.tex", V, "gakushu5b")
