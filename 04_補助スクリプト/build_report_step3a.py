# -*- coding: utf-8 -*-
"""手順3a の説明書（用語集と図の見方）を作る。

使い方（リポジトリ直下から）: uv run python 04_補助スクリプト/build_report_step3a.py

results/step3a の CSV から数値を取り出して report_step3a_template.tex に入れ、_tmp/report_step3a で
LuaLaTeX を2回動かし、できた PDF と tex を results/学習結果説明_手順3a.pdf・.tex に置く。
"""
import datetime
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)                              # リポジトリ直下
R = os.path.join(W, "results")
S = os.path.join(R, "step3a")
OUT = os.path.join(W, "_tmp", "report_step3a")
TEX_BIN = r"C:\texlive\2026\bin\windows"
LUALATEX = shutil.which("lualatex") or os.path.join(TEX_BIN, "lualatex.exe")


def rd(name):
    return pd.read_csv(os.path.join(S, name), encoding="utf-8-sig")


met, ker, ols = rd("metrics.csv"), rd("kernels.csv"), rd("ols_coefficients.csv")
loco, dsum, trend_df, unit = rd("loco_metrics.csv"), rd("data_summary.csv"), rd("linear_trend_trial.csv"), rd("unit_check.csv")
COL_GD, COL_ETA = dsum.columns[6], dsum.columns[7]                 # γ̇ [1/s]、η [Pa·s]。列名の特殊文字に依存しないよう、位置で取る
assert COL_GD.endswith("[1/s]") and COL_ETA.endswith("s]"), (COL_GD, COL_ETA)
COL_DINT = [c for c in unit.columns if c.startswith("切片の差")][0]
COL_DCOEF = [c for c in unit.columns if c.startswith("圧力の係数")][0]
MAT = {"PP": "PP", "MIX": "粉砕材混合PP"}
MODELS = ["L0", "L1", "L2", "A0", "A1", "A2", "A1p"]
MLAB = {"L0": "L0", "L1": "L1", "L2": "L2", "A0": "A0", "A1": "A1", "A2": "A2", "A1p": "A1$'$"}


def num(x, nd):
    """数値を文字列に。負の数は数式モードにして正しいマイナス記号にする"""
    s = f"{abs(x):,.{nd}f}"
    return f"$-{s}$" if x < 0 else s


def rmse(x):
    return "$<$0.001" if x < 0.0005 else f"{x:.3f}"


def sci(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def sig(x):
    """比の表示：10 未満は整数、それ以上は2桁"""
    return f"{x:.0f}" if x < 10 else f"{int(float(f'{x:.2g}'))}"


def M(mat, split, model, subset, col, data="なし"):
    r = met[(met["material"] == MAT[mat]) & (met["split"] == split) & (met["model"] == model) & (met["subset"] == subset) & (met["data"] == data)]
    assert len(r) == 1, (mat, split, model, subset, data)
    return float(r[col].iloc[0])


def K(mat, model, col, split="S_V", data="なし"):
    r = ker[(ker["material"] == MAT[mat]) & (ker["split"] == split) & (ker["model"] == model) & (ker["data"] == data)]
    assert len(r) == 1, (mat, split, model)
    return r[col].iloc[0]


def ds(mat, grp, col, o="なし"):
    r = dsum[(dsum["材料"] == MAT[mat]) & (dsum["組"] == grp) & (dsum["外れ値"] == o)]
    assert len(r) == 1, (mat, grp, o)
    return r[col].iloc[0]


def rng(s):
    return [float(x) for x in re.findall(r"[\d.]+", s)]


def rng_fmt(s):
    """「0.86〜394」のような範囲を、3桁区切りで整える（10 未満の値は小数のまま）"""
    a, b = rng(s)
    one = lambda x: f"{x:,.0f}" if x >= 10 else f"{x:g}"
    return f"{one(a)}〜{one(b)}"


def trend(mat, model, subset, col):
    r = trend_df[(trend_df["material"] == MAT[mat]) & (trend_df["model"] == model) & (trend_df["subset"] == subset)]
    assert len(r) == 1, (mat, model, subset)
    return float(r[col].iloc[0])


V = {}
today = datetime.date.today()
V["DATE"] = f"{today.year}年{today.month}月{today.day}日"

# ---- データ
for mat in MAT:
    V[f"{mat}_NTR"], V[f"{mat}_NCTR"] = f"{int(ds(mat, '5data', '行数')):,}", str(int(ds(mat, "5data", "条件数")))
    V[f"{mat}_NTE"], V[f"{mat}_NCTE"] = f"{int(ds(mat, '3data', '行数')):,}", str(int(ds(mat, "3data", "条件数")))
    V[f"{mat}_GD_TR"], V[f"{mat}_GD_TE"] = rng_fmt(ds(mat, "5data", COL_GD)), rng_fmt(ds(mat, "3data", COL_GD))
    V[f"{mat}_ETA_TR"], V[f"{mat}_ETA_TE"] = rng_fmt(ds(mat, "5data", COL_ETA)), rng_fmt(ds(mat, "3data", COL_ETA))
g_tr, g_te = rng(ds("PP", "5data", COL_GD)), rng(ds("PP", "3data", COL_GD))
e_tr, e_te = rng(ds("PP", "5data", COL_ETA)), rng(ds("PP", "3data", COL_ETA))
V["GD_LOW_PP"], V["ETA_RATIO_PP"] = f"{g_tr[0] / g_te[0]:.1f}", f"{e_te[1] / e_tr[1]:.1f}"
rows = []
for mat in MAT:
    for grp, lab in [("5data", "5data（学習）"), ("3data", "3data（検証）"), ("Alldata", "Alldata")]:
        suffix = "（no1.5-3.0）" if mat == "PP" and grp != "5data" else ""
        mlab = "PP" if mat == "PP" else "粉砕材混合 PP"
        rows.append(f"{mlab} & {lab}{suffix} & {int(ds(mat, grp, '行数')):,} & {int(ds(mat, grp, '条件数'))} & "
                    f"{ds(mat, grp, 'ショット/条件')} & {rng_fmt(ds(mat, grp, COL_GD))} & {rng_fmt(ds(mat, grp, COL_ETA))} \\\\")
V["TABLE_DATA"] = "\n".join(rows)

# ---- 線形回帰（リーク）
for mat in MAT:
    for key, col in [("L1", "ln_P31"), ("L2", "ln_P32")]:
        r = ols[(ols["material"] == MAT[mat]) & (ols["split"] == "S_V") & (ols["data"] == "なし") & (ols["model"] == key)].iloc[0]
        V[f"{key}_{mat}_BP"], V[f"{key}_{mat}_BG"], V[f"{key}_{mat}_BH"] = num(r[col], 3), num(r["ln_gd"], 3), num(r["ln_H"], 3)
        V[f"{key}_{mat}_R2"] = num(M(mat, "S_V", key, "全体", "R2"), 4)
rows = []
for mat in MAT:
    for key in ["L0", "L1", "L2"]:
        r = ols[(ols["material"] == MAT[mat]) & (ols["split"] == "S_V") & (ols["data"] == "なし") & (ols["model"] == key)].iloc[0]
        col = {"L0": None, "L1": "ln_P31", "L2": "ln_P32"}[key]
        rows.append(f"{MAT[mat] if key == 'L0' else ''} & {key} & {num(r['intercept'], 2)} & {num(r[col], 3) if col else '---'} & "
                    f"{num(r['ln_gd'], 3)} & {num(r['ln_H'], 3)} & {num(M(mat, 'S_V', key, '全体', 'R2'), 4)} \\\\")
V["TABLE_OLS"] = "\n".join(rows)

# ---- S_V の指標
for mat in MAT:
    for key in MODELS:
        for sub, tag in [("低速側", "LOW"), ("高速側", "HIGH")]:
            V[f"{key.upper()}_{mat}_{tag}"] = rmse(M(mat, "S_V", key, sub, "RMSE"))
            V[f"{key.upper()}_{mat}_COV_{tag}"] = f"{M(mat, 'S_V', key, sub, 'coverage95'):.2f}"
            V[f"{key.upper()}_{mat}_SD_{tag}"] = f"{M(mat, 'S_V', key, sub, 'mean_sd'):.2f}"
for mat in MAT:
    rows = []
    for key in MODELS:
        c = [M(mat, "S_V", key, s, k) for k in ["RMSE", "coverage95", "NLPD"] for s in ["低速側", "高速側"]]
        rows.append(f"{MLAB[key]} & {rmse(c[0])} & {rmse(c[1])} & {c[2]:.2f} & {c[3]:.2f} & {num(c[4], 2)} & {num(c[5], 2)} \\\\")
    V[f"TABLE_SV_{mat}"] = "\n".join(rows)

# ---- ランダム分割
V["A0_PP_R1"], V["A0_MIX_R1"] = rmse(M("PP", "S_R1", "A0", "全体", "RMSE")), rmse(M("MIX", "S_R1", "A0", "全体", "RMSE"))
V["RATIO_HIGH"] = sig(M("PP", "S_V", "A0", "高速側", "RMSE") / M("PP", "S_R1", "A0", "全体", "RMSE"))
V["RATIO_LOW"] = sig(M("PP", "S_V", "A0", "低速側", "RMSE") / M("PP", "S_R1", "A0", "全体", "RMSE"))
FILES = {"PP": ("02_機械学習データ", "NR500", "01_バージンPP再々実験_alldata", "04_pp_max_Alldata(no1.5-3.0)", "外れ値なし", "結合_04_pp_max_Alldata(no1.5-3.0)_外れ値なし.csv"),
         "MIX": ("02_機械学習データ", "NR500", "03_粉砕材混合PP実験_alldata", "01_mixpp_max_Alldata", "外れ値なし", "結合_01_mixpp_max_Alldata_外れ値なし.csv")}
for mat, parts in FILES.items():
    df = pd.read_csv(os.path.join(W, *parts), encoding="utf-8-sig")
    cond = (df["加熱シリンダ温度[℃]"].astype(int).astype(str) + "_" + df["キャビ厚"].astype(str) + "_" + df["射出速度[mm/s]"].astype(int).astype(str)).to_numpy()
    i_tr, i_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)       # ノートと同じ分割
    c_te = set(cond[i_te])
    V[f"OVERLAP_{mat}"] = f"{len(c_te)} のうち {len(c_te & set(cond[i_tr]))}"
rows = []
for key in MODELS:
    cells = [M("PP", "S_V", key, "低速側", "RMSE"), M("PP", "S_V", key, "高速側", "RMSE"), M("PP", "S_R1", key, "全体", "RMSE"), M("PP", "S_R2", key, "全体", "RMSE"),
             M("MIX", "S_V", key, "低速側", "RMSE"), M("MIX", "S_V", key, "高速側", "RMSE"), M("MIX", "S_R1", key, "全体", "RMSE")]
    rows.append(f"{MLAB[key]} & " + " & ".join(rmse(c) for c in cells) + " \\\\")
V["TABLE_RANDOM"] = "\n".join(rows)

# ---- LOCO
rows = []
for key in MODELS:
    c = []
    for mat in MAT:
        r = loco[(loco["material"] == MAT[mat]) & (loco["model"] == key) & (loco["subset"] == "全体")].iloc[0]
        c += [rmse(r["RMSE"]), f"{r['coverage95']:.2f}"]
    rows.append(f"{MLAB[key]} & " + " & ".join(c) + " \\\\")
V["TABLE_LOCO"] = "\n".join(rows)

# ---- ARD
for mat in MAT:
    V[f"A0_{mat}_LSG"], V[f"A0_{mat}_LSH"], V[f"A0_{mat}_LST"] = (f"{K(mat, 'A0', c):.2f}" if K(mat, 'A0', c) < 10 else f"{K(mat, 'A0', c):.0f}"
                                                              for c in ["ls_ln_gd", "ls_ln_H", "ls_ln_Tff"])
    V[f"A2_{mat}_LSP"] = f"{K(mat, 'A2', 'ls_ln_P32'):.0f}"
V["A1_LS_MIN"] = f"{min(K(m, 'A1', c) for m in MAT for c in ['ls_ln_gd', 'ls_ln_H', 'ls_ln_Tff', 'ls_ln_P31']):.0f}"

# ---- ノイズ
for mat in MAT:
    V[f"A0_{mat}_NSD"] = f"{K(mat, 'A0', 'noise_sd'):.3f}"
    V[f"PURE_{mat}"] = f"{K(mat, 'A0', 'pure_sd_ln_eta'):.3f}"
rows = []
for mat in MAT:
    for key in ["A0", "A2", "A1", "A1p"]:
        rows.append(f"{MAT[mat] if key == 'A0' else ''} & {MLAB[key]} & {K(mat, key, 'noise_sd'):.4f} & {K(mat, key, 'pure_sd_ln_eta'):.4f} & "
                    f"{K(mat, key, 'pure_sd_after_gd'):.4f} & {'はい' if K(mat, key, 'noise_at_lower_bound') else 'いいえ'} \\\\")
V["TABLE_NOISE"] = "\n".join(rows)

# ---- 線形の傾向の試算
TR = {"A0": "定数平均（A0）", "A": "(a) 線形回帰の残差を GPR", "B": "(b) 線形カーネルを足す"}
for mat in MAT:
    for k, name in TR.items():
        V[f"TR_{mat}_{k}_LOW"] = rmse(trend(mat, name, "低速側", "RMSE"))
    V[f"TR_{mat}_B_COV_LOW"] = f"{trend(mat, TR['B'], '低速側', 'coverage95'):.2f}"
rows = []
for mat in MAT:
    for k, name in TR.items():
        c = [trend(mat, name, s, col) for col in ["RMSE", "coverage95", "NLPD"] for s in ["低速側", "高速側"]]
        rows.append(f"{MAT[mat] if k == 'A0' else ''} & {name} & {rmse(c[0])} & {rmse(c[1])} & {c[2]:.2f} & {c[3]:.2f} & {num(c[4], 2)} & {num(c[5], 2)} \\\\")
V["TABLE_TREND"] = "\n".join(rows)

# ---- 単位の確認
V["UNIT_MAXDIFF"] = sci(float(unit["予測平均の差の最大"].max()))
rows = []
for _, r in unit.iterrows():
    ols_row = not np.isnan(r[COL_DINT])
    rows.append(f"{r['材料'] if r['モデル'] == 'L1' else ''} & {MLAB[r['モデル']]} & {sci(float(r['予測平均の差の最大']))} & {sci(float(r['予測標準偏差の差の最大']))} & "
                f"{num(r[COL_DINT], 3) if ols_row else '---'} & {num(r[COL_DCOEF], 3) if ols_row else '---'} \\\\")
V["TABLE_UNIT"] = "\n".join(rows)

# ---- 気になる条件（PP の 190 ℃・0.5 mm・320 mm/s）
anom_c, anom = rd("anomaly_condition.csv"), rd("anomaly_check.csv")
pp_c = anom_c[anom_c.iloc[:, 0] == MAT["PP"]]
eta_by_t = dict(zip(pp_c.iloc[:, 1].astype(int), pp_c.iloc[:, 4]))                 # 設定温度 → η の平均
V["ANOM_N"] = str(int(pp_c[pp_c.iloc[:, 1].astype(int) == 190].iloc[0, 2]))
V["ANOM_RATIO"] = f"{eta_by_t[190] / np.mean([eta_by_t[210], eta_by_t[230]]):.1f}"
V["A0_PP_NLPD_HIGH"] = num(M("PP", "S_V", "A0", "高速側", "NLPD"), 2)
sel = anom[(anom["material"] == MAT["PP"]) & (anom["model"] == "A0") & (anom["subset"] == "高速側（この条件を除く）")].iloc[0]
V["A0_PP_NLPD_HIGH_X"], V["A0_PP_RMSE_HIGH_X"] = num(sel["NLPD"], 2), rmse(sel["RMSE"])
rows, prev = [], None
for _, r in anom_c.iterrows():
    rows.append(f"{r.iloc[0] if r.iloc[0] != prev else ''} & {int(r.iloc[1])} & {int(r.iloc[2])} & {r.iloc[3]:,.0f} & {r.iloc[4]:.2f} & {r.iloc[5]:.1f} \\\\")
    prev = r.iloc[0]
V["TABLE_ANOM_COND"] = "\n".join(rows)
rows = []
for mat in MAT:
    for sub in ["高速側すべて", "高速側（この条件を除く）", "この条件だけ"]:
        r = anom[(anom["material"] == MAT[mat]) & (anom["model"] == "A0") & (anom["subset"] == sub)].iloc[0]
        rows.append(f"{MAT[mat] if sub == '高速側すべて' else ''} & {sub} & {int(r['n_test'])} & {rmse(r['RMSE'])} & {r['coverage95']:.2f} & {num(r['NLPD'], 2)} \\\\")
V["TABLE_ANOM"] = "\n".join(rows)

# ---- 置き換え
TEMPLATE =open(os.path.join(HERE, "report_step3a_template.tex"), encoding="utf-8").read()
tex = TEMPLATE
for key, val in V.items():
    tex = tex.replace(f"@@{key}@@", str(val))
left = sorted(set(re.findall(r"@@([A-Za-z0-9_]+)@@", tex)))
assert not left, f"置き換えていない項目: {left}"

os.makedirs(os.path.join(OUT, "step3a"), exist_ok=True)
with open(os.path.join(OUT, "gakushu3a.tex"), "w", encoding="utf-8", newline="\n") as f:
    f.write(tex)
for fn in os.listdir(S):
    if fn.endswith(".png"):
        shutil.copy2(os.path.join(S, fn), os.path.join(OUT, "step3a", fn))
print("原稿を作成:", os.path.join(OUT, "gakushu3a.tex"), f"（置き換え {len(V)} 項目）")

for i in range(2):
    r = subprocess.run([LUALATEX, "-interaction=nonstopmode", "-halt-on-error", "gakushu3a.tex"], cwd=OUT, capture_output=True)
    if r.returncode != 0:
        print(r.stdout.decode("utf-8", "replace")[-3000:])
        sys.exit(f"lualatex が失敗しました（{i + 1} 回目）")
shutil.copy2(os.path.join(OUT, "gakushu3a.pdf"), os.path.join(R, "学習結果説明_手順3a.pdf"))
shutil.copy2(os.path.join(OUT, "gakushu3a.tex"), os.path.join(R, "学習結果説明_手順3a.tex"))
print("作成:", os.path.join(R, "学習結果説明_手順3a.pdf"), "と .tex")
