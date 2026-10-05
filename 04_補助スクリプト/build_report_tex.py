# -*- coding: utf-8 -*-
"""学習結果説明（用語集と図の見方）の LaTeX 原稿を、results の CSV とノートの出力から作る。

使い方: python build_report_tex.py <作業フォルダ（プロジェクトの直下）> <ビルド先フォルダ>
ビルド先に gakushu.tex を書き、results の図を step2/・step2_mcmc/ にコピーする（results と同じ構成）。
"""
import datetime
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

W, OUT = sys.argv[1], sys.argv[2]
R = os.path.join(W, "results")
NB_DIR = os.path.join(W, "01_機械学習コードフォルダ")


def rd(rel):
    return pd.read_csv(os.path.join(R, rel), encoding="utf-8-sig")


m2, k2, fc, te = rd("step2/metrics.csv"), rd("step2/kernels.csv"), rd("step2/formula_check.csv"), rd("step2/temperature_effect_190_vs_230.csv")
mm, ps, ss = rd("step2_mcmc/metrics.csv"), rd("step2_mcmc/posterior_summary.csv"), rd("step2_mcmc/sampler_stats.csv")
raw = pd.read_csv(os.path.join(W, "02_機械学習データ", "NR500", "00_バージンPP実験データ", "学習データ6月発表時.csv"), encoding="utf-8-sig")


def nb_stdout(name):
    nb = json.load(open(os.path.join(NB_DIR, name), encoding="utf-8"))
    return "\n".join("".join(o["text"]) for c in nb["cells"] for o in c.get("outputs", [])
                     if o["output_type"] == "stream" and o["name"] == "stdout")


o2 = nb_stdout("手順2_現行モデル再現とリーク確認.ipynb")
om = nb_stdout("手順2_MCMC版_現行モデル再現とリーク確認.ipynb")


def num(x, nd):
    """数値を文字列に。負の数は数式モードにして正しいマイナス記号にする"""
    s = f"{abs(x):,.{nd}f}"
    return f"$-{s}$" if x < 0 else s


V = {}
# ---- データ
parts = raw["元ファイル"].str.extract(r"^(\d{3})_(\d+)℃_(\d{3})_([\d.]+)mm\.csv$")
V["N_ROWS"] = f"{len(raw):,}"
V["N_COND"] = str(raw.assign(T=parts[1], H=raw["キャビティ厚み"], Q=raw["射出率 [㎤/s]"]).groupby(["T", "H", "Q"]).ngroup().nunique())
V["T_LEVELS"] = "・".join(sorted(parts[1].unique())) + " ℃"
V["H_LEVELS"] = "・".join(f"{h:.1f}" for h in sorted(raw["キャビティ厚み"].unique())) + " mm"
q = sorted(raw["射出率 [㎤/s]"].unique())
V["N_Q"], V["Q_MIN"], V["Q_MAX"] = str(len(q)), f"{q[0]:g}", f"{q[-1]:g}"
V["P_MED"] = f"{np.median(raw['流動中樹脂圧力P(Pa)']) / 1e6:.2f}"
# ---- 算出式
a = fc[fc["H_mm"].astype(str) == "all"].iloc[0]
V["L_MM"], V["P2_MPA"], V["RES_MAX"] = f"{a['L_mm']:.2f}", f"{a['P2_MPa']:.3f}", f"{a['max_abs_rel_resid_pct']:.3f}"
V["LN2L"] = num(-np.log(2 * a["L_mm"]), 3)
# ---- 分割（手順2のノートの出力）
s1 = re.search(r"S1: 学習 (\d+) 行 / テスト (\d+) 行、テストを含む条件 (\d+)、学習とテストの両方に入った条件 (\d+)", o2).groups()
s2 = re.search(r"S2: 学習 (\d+) 行 / テスト (\d+) 行、テストを含む条件 (\d+)、学習とテストの両方に入った条件 (\d+)", o2).groups()
V.update({"S1_NTR": f"{int(s1[0]):,}", "S1_NTE": s1[1], "S1_NCT": s1[2], "S2_NTR": f"{int(s2[0]):,}", "S2_NTE": s2[1], "S2_NCT": s2[2], "S2_BOTH": s2[3]})
V["M2_ALL_R2"] = f"{float(re.search(r'M2（全1290行で当てはめ）: R2\(ln η\) = ([\d.]+)', o2).group(1)):.3f}"
# ---- 指標（手順2）
for _, r in m2.iterrows():
    key = f"{r['model']}_{r['split']}"
    V[f"{key}_R2"] = num(r["R2_ln_eta"], 4 if r["R2_ln_eta"] > 0 else 2)
    V[f"{key}_RMSE"] = num(r["RMSE_ln_eta"], 4)
V["M1_S1_R2ETA"] = num(m2.query("split == 'S1' and model == 'M1'")["R2_eta"].iloc[0], 4)
# ---- 最尤推定値（S1・M1）
k = k2.query("split == 'S1' and model == 'M1'").iloc[0]
V.update({"MLE_AMP": f"{k['amplitude']:.2f}", "MLE_LST": f"{k['ls_lnT']:.1f}", "MLE_LSP": f"{k['ls_lnP']:.2f}",
          "MLE_LSSR": f"{k['ls_lnSR']:.2f}", "MLE_LML": f"{k['LML']:.2f}"})
k4 = k2.query("split == 'S1' and model == 'M4'").iloc[0]
V.update({"M4_LST": f"{k4['ls_lnT']:.2f}", "M4_LSSR": f"{k4['ls_lnSR']:.2f}"})
# ---- 温度の効果
V["TEMP_RATIO"] = f"{np.exp(te['ln_eta'].median()):.2f}"
# ---- M1 の予測曲線（手順2・MCMC 版のノートの出力）
sl2 = [float(x) for x in re.findall(r"T_flow = \d+ ℃: この範囲での ln η–ln γ̇ の傾き (-?[\d.]+)", o2)]
d2 = re.search(r"190 ℃ と 230 ℃ の曲線の差: ln η で最大 ([\d.]+)（粘度の比 ([\d.]+)）、平均 ([\d.]+)", o2).groups()
slm = [float(x) for x in re.findall(r"流動先端温度 \d+ ℃: ln η–ln γ̇ の傾き (-?[\d.]+)", om)]
dm = re.search(r"190 ℃ と 230 ℃ の曲線の差: ln η で最大 ([\d.]+)（粘度の比 ([\d.]+)）、平均 ([\d.]+)", om).groups()
V["M1C_SLOPE"] = num(np.mean(sl2), 2)
V["M1C_MAX_RATIO"], V["M1C_MEAN_RATIO"] = d2[1], f"{np.exp(float(d2[2])):.2f}"
V["M1CM_SLOPE"], V["M1CM_MAX_RATIO"] = num(np.mean(slm), 2), dm[1]
# ---- MCMC
V["ACC_MIN"], V["ACC_MAX"] = f"{ss['acc_samp_min'].min():.2f}", f"{ss['acc_samp_max'].max():.2f}"
V["RHAT_MAX"], V["ESS_MIN"] = f"{ss['max_r_hat'].max():.3f}", f"{ss['min_ess_bulk'].min():,.0f}"

# 表：手順2の指標
rows = []
for mdl, lab in [("M1", "M1 現行 GPR"), ("M2", "M2 線形回帰"), ("M3", "M3 線形回帰（算出式）"), ("M4", "M4 P を外した GPR")]:
    r1 = m2.query("split == 'S1' and model == @mdl").iloc[0]
    r2 = m2.query("split == 'S2' and model == @mdl").iloc[0]
    rows.append(f"{lab} & {num(r1['R2_ln_eta'], 3)} & {num(r1['RMSE_ln_eta'], 3)} & {num(r1['coverage95'], 2)} & "
                f"{num(r2['R2_ln_eta'], 3)} & {num(r2['RMSE_ln_eta'], 3)} & {num(r2['coverage95'], 2)} \\\\")
V["TABLE_STEP2"] = "\n".join(rows)

# 表：サンプラーの統計
rows = []
for _, r in ss.iterrows():
    rows.append(f"{r['split']}・{r['model']} & {int(r['n_samp_per_chain']):,} & {r['sampling_min']:.1f} & "
                f"{r['acc_samp_min']:.2f}〜{r['acc_samp_max']:.2f} & {r['max_r_hat']:.3f} & {r['min_ess_bulk']:,.0f} \\\\")
V["TABLE_SAMPLER"] = "\n".join(rows)

# 表：事後分布（S1・M1）
PJ = {"amplitude": "振幅", "ls_lnT": r"長さスケール（$\ln T$）", "ls_lnP": r"長さスケール（$\ln P$）",
      "ls_lnSR": r"長さスケール（$\ln\dot{\gamma}$）", "noise": "ノイズ分散"}
rows = []
for _, r in ps.query("split == 'S1' and model == 'M1'").iterrows():
    nd = 6 if r["param"] == "noise" else (1 if r["median"] > 10 else 2)
    rows.append(f"{PJ[r['param']]} & {r['MLE']:.{nd}f} & {r['median']:.{nd}f} & {r['q2.5']:.{nd}f}〜{r['q97.5']:.{nd}f} \\\\")
V["TABLE_POST"] = "\n".join(rows)
p = ps.query("split == 'S1' and model == 'M1'").set_index("param")
V["POST_AMP_CI"] = f"{p.loc['amplitude', 'q2.5']:.2f}〜{p.loc['amplitude', 'q97.5']:.2f}"

# 表：MCMC と最尤推定
rows = []
for s in ["S1", "S2"]:
    for mdl in ["M1", "M4"]:
        rm = mm.query("split == @s and model == @mdl and method == 'MCMC'").iloc[0]
        rl = mm.query("split == @s and model == @mdl and method == 'MLE'").iloc[0]
        rows.append(f"{s}・{mdl} & {num(rm['R2_ln_eta'], 4)} & {num(rl['R2_ln_eta'], 4)} & {num(rm['NLPD'], 3)} & {num(rl['NLPD'], 3)} & "
                    f"{rm['mean_pred_sd']:.4f} & {rl['mean_pred_sd']:.4f} \\\\")
V["TABLE_MCMC_MLE"] = "\n".join(rows)
widen = [(mm.query("split == @s and model == @m and method == 'MCMC'")['mean_pred_sd'].iloc[0]
          / mm.query("split == @s and model == @m and method == 'MLE'")['mean_pred_sd'].iloc[0] - 1) * 100
         for s in ["S1", "S2"] for m in ["M1", "M4"]]
V["WIDEN_MIN"], V["WIDEN_MAX"] = f"{min(widen):.0f}", f"{max(widen):.0f}"
today = datetime.date.today()
V["DATE"] = f"{today.year}年{today.month}月{today.day}日"

TEMPLATE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template.tex"), encoding="utf-8").read()
tex = TEMPLATE
for key, val in V.items():
    tex = tex.replace(f"@@{key}@@", val)
left = sorted(set(re.findall(r"@@([A-Z0-9_]+)@@", tex)))
assert not left, f"置き換えていない項目: {left}"

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "gakushu.tex"), "w", encoding="utf-8", newline="\n") as f:
    f.write(tex)
for sub in ["step2", "step2_mcmc"]:
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    for fn in os.listdir(os.path.join(R, sub)):
        if fn.endswith(".png"):
            shutil.copy2(os.path.join(R, sub, fn), os.path.join(OUT, sub, fn))
print("作成:", os.path.join(OUT, "gakushu.tex"), f"（置き換え {len(V)} 項目）")
for kk in ["N_ROWS", "N_COND", "L_MM", "P2_MPA", "M1_S1_R2", "M4_S1_R2", "M4_S2_R2", "TEMP_RATIO", "M1C_SLOPE", "RHAT_MAX", "ESS_MIN", "WIDEN_MIN", "WIDEN_MAX", "DATE"]:
    print(f"  {kk} = {V[kk]}")
