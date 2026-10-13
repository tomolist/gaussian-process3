# -*- coding: utf-8 -*-
"""手順3c の説明書（用語集と図の見方）を作る。

使い方（リポジトリ直下から）: uv run python 04_補助スクリプト/build_report_step3c.py
results/step3c_mcmc の CSV から数値を取り出して report_step3c_template.tex に入れ、_tmp/report_3c で
LuaLaTeX を2回動かし、できた PDF と tex を results/学習結果説明_手順3c.pdf・.tex に置く。
"""
import os

import numpy as np
import pandas as pd

import report_common as rc
from report_common import MAT, MAT_JA, num, rd, rmse

S = "step3c_mcmc"
stats, cmp_, post, effm, info = (rd(S, f) for f in ["sampler_stats.csv", "mle_vs_mcmc.csv", "posterior_summary.csv", "tset_effect_mcmc.csv", "chain_info.csv"])
V = {}
SP = lambda s: s.replace("_", "\\_")
SUBS = {"低速側": "S\\_V 低速", "高速側": "S\\_V 高速", "全体": ""}


def st(mat, split, model):
    r = stats[(stats["material"] == MAT[mat]) & (stats["split"] == split) & (stats["model"] == model)]
    return r.iloc[0]


def cm(mat, split, sub, model):
    r = cmp_[(cmp_["material"] == MAT[mat]) & (cmp_["split"] == split) & (cmp_["subset"] == sub) & (cmp_["model"] == model)]
    return r.iloc[0]


V["N_CSV"] = str(len([f for f in os.listdir(os.path.join(rc.R, S)) if f.endswith(".csv") and not f.startswith("posterior_")]) )
V["N_FIG"] = str(len([f for f in os.listdir(os.path.join(rc.R, S)) if f.endswith(".png")]))
V["MIN_TOTAL"] = f"{info.groupby(['material', 'split', 'model'])['wall_seconds'].max().sum() / 60:.0f}"

# ---- 表：サンプラーの統計
rows = []
for _, r in stats.iterrows():
    mat = [k for k, v in MAT.items() if v == r["material"]][0]
    rows.append(f"{MAT_JA[mat]} & {SP(r['split'])} & {r['model']} & {int(r['n_samp_per_chain']):,} & {r['max_r_hat']:.3f} & {r['min_ess_bulk']:.0f} & {r['min_ess_tail']:.0f} & {int(r['n_post_at_bound'])} \\\\")
V["TABLE_STATS"] = "\n".join(rows)
ok = stats[(stats["max_r_hat"] <= 1.01) & (stats["min_ess_bulk"] >= 400)]
bad = stats[~stats.index.isin(ok.index)]
ext = stats[stats["n_samp_per_chain"] > 8000]

# ---- 表：最尤推定と MCMC の比較
rows = []
for mat in MAT:
    for split, sub in [("S_V", "低速側"), ("S_V", "高速側"), ("S_T", "全体"), ("S_T3", "全体")]:
        for model in ["B0T", "R3", "R2"]:
            r = cm(mat, split, sub, model)
            lab = (SUBS[sub] if split == "S_V" else SP(split))
            rows.append(f"{MAT_JA[mat] if (split == 'S_V' and sub == '低速側' and model == 'B0T') else ''} & {lab if model == 'B0T' else ''} & {model} & {rmse(r['RMSE_MLE'])} & {rmse(r['RMSE_MCMC'])} & "
                        f"{num(r['NLPD_MLE'], 2)} & {num(r['NLPD_MCMC'], 2)} & {r['sd_MLE']:.3f} & {r['sd_MCMC']:.3f} \\\\")
V["TABLE_CMP"] = "\n".join(rows)

# ---- 表：B0T の Tset の長さスケール
rows = []
for mat in MAT:
    r = post[(post["material"] == MAT[mat]) & (post["split"] == "S_T") & (post["model"] == "B0T") & (post["param"] == "ls_Tset")].iloc[0]
    rows.append(f"{MAT_JA[mat]} & {r['MLE']:.3g} & {r['median']:.3g} & {r['q2.5']:.3g} & {r['q97.5']:.3g} & {r['r_hat']:.3f} \\\\")
V["TABLE_B0T"] = "\n".join(rows)

# ---- 文章
pp_b = post[(post["material"] == "PP") & (post["split"] == "S_T") & (post["model"] == "B0T") & (post["param"] == "ls_Tset")].iloc[0]
mx_b = post[(post["material"] == "粉砕材混合PP") & (post["split"] == "S_T") & (post["model"] == "B0T") & (post["param"] == "ls_Tset")].iloc[0]
bad_txt = "、".join(f"{MAT_JA[[k for k, v in MAT.items() if v == r['material']][0]]} の {r['model']}（{SP(r['split'])} の学習）" for _, r in bad.iterrows()) or "なし"
V["TRACE_TXT"] = (
    f"12 組のうち {len(ok)} 組は、4 本のチェーンが混ざり合い、R-hat が 1.01 以下、ESS が 400 以上になりました（表\\ref{{tab:stats}}）。最初の 8,000 回で ESS が足りなかった {len(ext)} 組は、本番を 20,000 回に増やしました。"
    f"収束の目安を満たさなかったのは、{bad_txt} の {len(bad)} 組です。図のように、チェーン 3 が 10,000 回ごろに別の山（ノイズ分散が小さく、$\\gd$ の長さスケールが短い解）へ移り、R-hat が {float(bad['max_r_hat'].max()) if len(bad) else 1:.2f}、ESS が {float(bad['min_ess_bulk'].min()) if len(bad) else 0:.0f} になりました。"
    f"B0T の事後分布には、「ショットごとの $\\gd$ に細かく合わせる山」と「なめらかな山」があり、チェーンが行き来しにくいためです（多峰性）。この組の予測は、4 本をすべて混ぜて計算しているので、参考値として読みます。"
)
V["POST_TXT"] = (
    f"PP の B0T（S\\_T の学習）では、$\\Tset$ の長さスケールの最尤推定値が {pp_b['MLE']:.3f} で、事後分布の山（中央値 {pp_b['median']:.1f}、95\\% 区間 {pp_b['q2.5']:.1f}〜{pp_b['q97.5']:.1f}）から大きく離れています（表\\ref{{tab:b0t}}）。最尤推定は、確率のほとんどない谷に止まっていました。MCMC は、山の本体（長さスケール 約 14）を見つけます。"
    f"R3 では、$\\Tset$ の長さスケールの山が、上限（100）の近くにあります（中央値 約 70〜76）。$\\Tset$ の効果が、Matern の部分ではなく、線形カーネルの部分で、ほぼ直線的に表されていることを意味します。事前分布の範囲の端に届いた R3 の 4 組は、この理由です。"
    f"線形項の $\\sigma_0$（右から 2 番目の列）は、範囲いっぱいに広がっていて、データでは決まりません（予測には、ほとんど効きません）。ほかのハイパーパラメータは、最尤推定の値が、山の中にあります。"
)
r3l, r3m = cm("PP", "S_V", "低速側", "R3"), cm("MIX", "S_V", "低速側", "R3")
b0 = cm("PP", "S_T", "全体", "B0T")
r2h = cm("PP", "S_V", "高速側", "R2")
V["FIGA_TXT"] = (
    f"\\textbf{{R3}}：高速側・S\\_T は、最尤推定とほとんど同じです（NLPD の差 0.1 未満）。低速側（外挿）では、MCMC のほうが区間が約 {r3l['sd_ratio']:.1f} 倍広く（保守的）、NLPD が {r3l['NLPD_diff']:+.2f}（PP）・{r3m['NLPD_diff']:+.2f}（粉砕材混合 PP）と少し悪くなります。RMSE も、PP で {rmse(r3l['RMSE_MLE'])} から {rmse(r3l['RMSE_MCMC'])}、粉砕材混合 PP で {rmse(r3m['RMSE_MLE'])} から {rmse(r3m['RMSE_MCMC'])} と、わずかに大きくなります。事後の平均は、直線的な傾きの不確かさ（線形項）を平均するため、低速側で、最尤推定よりやや外れやすくなるためです。最尤推定でも十分な精度です。"
    f"\\textbf{{B0T}}：PP の S\\_T では、最尤推定が谷に止まっていたため、MCMC で RMSE が {rmse(b0['RMSE_MLE'])} から {rmse(b0['RMSE_MCMC'])}、NLPD が {num(b0['NLPD_MLE'], 2)} から {num(b0['NLPD_MCMC'], 2)} へ、大きく改善します（区間の広さは {b0['sd_MLE']:.3f} から {b0['sd_MCMC']:.3f}）。"
    f"\\textbf{{R2}}：MCMC の区間は、S\\_T で約 1.5〜2 倍と広くなり、PP の S\\_V の高速側の NLPD は {num(r2h['NLPD_MLE'], 2)} から {num(r2h['NLPD_MCMC'], 2)} へ改善します（P03 に引きずられる最尤推定より、広めの区間のため）。"
)
e = effm.copy()
def _rng(model, mat):
    g = e[(e["material"] == MAT[mat]) & (e["model"] == model)]
    return float((g["ratio_q97.5"] - g["ratio_q2.5"]).max())
V["FIGB_TXT"] = (
    f"B0T と R3 の分布は、非常に細く、95\\% 区間の幅が最大でも PP で {max(_rng('B0T', 'PP'), _rng('R3', 'PP')):.3f}、粉砕材混合 PP で {max(_rng('B0T', 'MIX'), _rng('R3', 'MIX')):.3f} です。ハイパーパラメータの不確かさは、設定温度の効果（比 1.07〜1.20）にほとんど影響しません。R2 は、少し幅が広くなります（最大 {max(_rng('R2', 'PP'), _rng('R2', 'MIX')):.3f}）。"
    f"どのモデル・厚みでも、事後サンプルの 95\\% 以上で、190 ℃ のほうが 230 ℃ より粘い（最小 {e['share_gt1'].min():.2f}）という結果です。なお、この図は「ハイパーパラメータの不確かさ」だけを表します。予測そのものの不確かさ（予測区間）は、図Cで見ます。"
)
V["FIGC_TXT"] = (
    "R3 では、MCMC の帯（青・緑・赤）と最尤推定の破線が、ほとんど重なります。低速側（左端）では、帯が破線より少し広がります。検証の四角（外挿）は、帯の中に入ります。B0T は、曲線が波打ち、帯の幅も場所で変わります（ショットごとの $\\gd$ を入力にしているため。手順3bの図1と同じ）。"
)


# ---- まとめ
def f2(x):
    return f"{x:.2f}"


V["SUMMARY"] = f"""\\begin{{enumerate}}[itemsep=0.6ex,topsep=0.6ex]
\\item \\textbf{{収束}}：12 組のうち {len(ok)} 組が、R-hat 1.01 以下・ESS 400 以上を満たしました。満たさなかったのは、{bad_txt}です。多峰性（複数の山）のためで、予測には 4 本のチェーンの全サンプルを使いました。
\\item \\textbf{{B0T は、MCMC で直る}}：手順3bで、PP の S\\_T の B0T は、最尤推定が谷に止まり、210 ℃ を外しました（RMSE {rmse(b0['RMSE_MLE'])}）。MCMC は、山の本体を見つけ、RMSE {rmse(b0['RMSE_MCMC'])}、NLPD {num(b0['NLPD_MCMC'], 2)} になりました（図A、表\\ref{{tab:cmp}}）。ただし、B0T の事後分布は多峰性で、扱いにくいモデルです。
\\item \\textbf{{R3 は、最尤推定で十分}}：R3 の MCMC と最尤推定は、高速側・S\\_T でほぼ同じ（NLPD の差 0.1 未満）です。低速側では、MCMC の区間が約 {r3l['sd_ratio']:.1f} 倍広く（外挿で、広めになるのは妥当）、NLPD は {r3l['NLPD_diff']:+.2f}（PP）・{r3m['NLPD_diff']:+.2f}（粉砕材混合 PP）と、少し悪くなります。ハイパーパラメータが 1 つの山に決まる R3 では、1 組で近似して問題ありません。
\\item \\textbf{{R2（設定値だけ）は、MCMC で区間が広がる}}：S\\_T で約 1.5〜2 倍。実測の $\\gd$ を使わない分、不確かさが大きいことを、区間が表しています。
\\item \\textbf{{決まらないパラメータ}}：線形項の $\\sigma_0$ は、データで決まらず、事前の範囲いっぱいに広がります（予測にはほとんど効きません）。R3 の $\\Tset$ の長さスケールは、上限（100）に近く、温度の効果が、ほぼ直線的（線形項）で表されています。
\\item \\textbf{{設定温度の効果は、ハイパーパラメータによらない}}：190/230 ℃ の粘度比の 95\\% 区間の幅は、B0T・R3 で 0.02 未満です（図B）。
\\item \\textbf{{結論}}：基本モデル（R3）は、最尤推定でよく、MCMC は、最適化が不安定なモデル（B0T）の確認と、区間を広めに見たいときに使います。手順5の本実装で、MCMC を使うかは、構造（5a）が決まってから相談します。
\\end{{enumerate}}"""

rc.build_pdf("3c", S, "report_step3c_template.tex", V, "gakushu3c")
