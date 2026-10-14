# -*- coding: utf-8 -*-
"""手順3e（キャビティ厚み H を入力にすることの検討：リークか、Cross-WLF と同じ変数で足りるか）のノートブック (未実行) を生成する。

使い方: python build_step3e_nb.py <出力する .ipynb のパス> [--force]
既存ファイルがある場合は、--force を付けない限り上書きしない。
"""
import json
import os
import random
import sys

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src})


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nb_parts import COMMON_CELL, HR_MD, hr_cell, settings_cell, usage_md  # noqa: E402

# ---------------------------------------------------------------- 0. タイトル
md(r'''# 手順3e：キャビティ厚み H を入力にすることの検討

ユーザーの問い（2026-10-14）：**H は、目的変数（見かけの粘度）を算出するときの流量の計算に使われる値なので、入力に使うとリークではないか。Cross-WLF モデルの入力（せん断速度・温度・圧力）には厚みがないので、厚みを入力にする学習方法には疑問がある。**

この手順では、次の 3 つを確かめる。

1. **リークか**（式とデータの両面）：H を入れると、算出式を使って粘度を「計算」できてしまうか。H は、予測するときに事前に分かるか。
2. **H を外すとどうなるか**：Cross-WLF と同じ変数（γ̇、温度）だけで、見かけの粘度を予測できるか。温度に、設定温度 Tset を使う場合と、流動先端温度 Tff（流れの中の樹脂温度。Cross-WLF の「T」に近い）を使う場合。
3. **厚みの効果は何か**：同じせん断速度・同じ設定温度で、厚みによって見かけの粘度がどれだけ違うか。理想的なスリット流れ（等温・固化層なし）なら、見かけの流動曲線は厚みによらず重なる（理由は 2 節）。

**算出式**（スリット近似。L = 17.5 mm はセンサ間距離、W は幅、Δt は 2 点への到達の時間差、ΔP は圧力差）

| 量 | 式 | ln をとると |
|---|---|---|
| 流量 | Q = W·H·L/Δt | |
| 見かけのせん断速度 | γ̇ = 6Q/(W·H²) = 6L/(H·Δt) | ln γ̇ = −ln H − ln Δt + 定数 |
| 見かけの壁面せん断応力 | τ = ΔP·H/(2L) | ln τ = ln ΔP + ln H + 定数 |
| 見かけの粘度 | η = τ/γ̇ = ΔP·H²·Δt/(12L²) | **ln η = ln ΔP + ln H − ln γ̇ + 定数** |

ln η は、ΔP（流れの結果。測る値）、H（金型の寸法。事前に決まる）、γ̇（Δt と H から計算）で決まる。**ΔP（＝ P3_1 − 0.30 MPa）を入力に入れると、ln η を計算できてしまう（リーク。手順3a）**。H と γ̇ だけでは、ΔP が分からないので計算できない。

| 記号 | モデル（GPR。ARD Matern 5/2 ＋ 線形 ＋ ノイズ。γ̇ は条件平均） | 入力 | 位置づけ |
|---|---|---|---|
| R3 | 基本のモデル（手順3b） | γ̇, ln H, Tset | 今まで |
| RFo | R3 の H をフーリエ数に替える | γ̇, ln Fo, Tset | ln Fo = −(ln γ̇ + 3 ln H) ＋ 定数。H と同じ情報（構造 A・B が使う入力） |
| W0 | **Cross-WLF と同じ変数**（T ＝ 設定温度） | γ̇, Tset | H を外す |
| W1 | **Cross-WLF と同じ変数**（T ＝ 流動先端温度） | γ̇, ln Tff | H を外し、温度を、流れの中の樹脂温度にする |
| W1T | 参考 | γ̇, ln Tff, Tset | H を外し、2 つの温度 |
| W2 | **Cross-WLF と同じ変数**（T ＝ 設定温度、P ＝ P3_2） | γ̇, Tset, ln P3_2 | H を外し、圧力を入れる（ユーザーの依頼、10-14） |
| W3 | **Cross-WLF と同じ変数**（T ＝ 流動先端温度、P ＝ P3_2） | γ̇, ln Tff, ln P3_2 | 同上 |

データは、手順3d の 2 つの版（外れ値なし。5data 版：S_V・S_T、Alldata 版：CV）。P3_2・Tff は、条件平均を使う。

**P3_2 について**（Cross-WLF の圧力 P として W2・W3 に入れる）
- 算出式によるリークの確認は、手順3a で済んでいる（H を入れたモデルで）。P3_2 は算出式に入っておらず、線形回帰に足しても残差はほとんど下がらない（PP：0.346 → 0.333。P3_1 なら 0.021）。結論は「リークではないが、役に立たない」（3a・3b）。
- ただし、流れの結果で、成形条件から予測するときには事前に分からない（P3_1・Tff と同じ）。
- H を外すと、P3_2 が厚みの代わりの目印になるおそれがある（薄いほど圧力が高い）。3a の線形回帰でも、P3_2 を足すと ln H の係数が −0.24 → ＋0.49 に変わった。そこで 3 節で、H なしでのリークの確認と、P3_2 が厚みをどれだけ表すかを確かめる。
- P3_2 は「センサ4の出力が最大になった時刻」のセンサ3の圧力で、粘度を測った区間（センサ3→4 の到達の間）の圧力ではない。Cross-WLF の P（その場の圧力）とは意味が違う。''')

# ---------------------------------------------------------------- 1. 設定
md(usage_md("3e", "results/step3e", "results/学習結果説明_手順3e.pdf", 20))
md(r'''## 1. 設定''')
code(settings_cell('''DATA_VERSIONS = ["5data", "Alldata"]   # 検討する版（手順3d）。片方だけにもできる
MAIN_DATA = "なし"            # 外れ値なし（外れ値ありは使わない）
TEMP_HOLDOUT = 210            # S_T でテストにする設定温度 [℃]
CV_K = 10                     # Alldata 版の交差検証の分割数（条件単位）
N_RESTARTS_CV = 1             # 交差検証の各分割での最適化のやり直し回数
N_JOBS = 6                    # 並列に計算する数（CPU の数に合わせる）
KEYS = ["R3", "RFo", "W0", "W1", "W1T", "W2", "W3"]   # 比べるモデル（記号はノート冒頭の表）
REF_H = 2.0                   # 厚みの効果を比べる基準の厚み [mm]
HR_ALL = True                 # すべての図を、高解像度でも results/step3e/highres/ に保存するか（Git の管理外）
HR_ALL_DPI = 600              # その解像度
'''))
code(r'''# セル1.5：ライブラリ、フォント、出力先
%matplotlib inline
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from joblib import Parallel, delayed

OUT_DIR = os.path.join(REPO, "results", "step3e")
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 80
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
try:
    fm.findfont("IPAexGothic", fallback_to_default=False)
except ValueError:
    raise RuntimeError("日本語フォント IPAexGothic が見つかりません。インストールしてからカーネルを再起動してください。")
plt.rcParams["font.family"] = ["IPAexGothic", "DejaVu Sans"]
print("Python:", sys.executable, "/", sys.version.split()[0])
print("出力先:", os.path.abspath(OUT_DIR))''')
md(r'''### 共通の部品

データの読み込み、問題データ一覧の「除外」、モデル、評価、版ごとの分割（`version_sets`）、高解像度の図（`hr_flow`、`save_fig`）の関数。中身は `04_補助スクリプト/nb_common.py` と同じ。''')
code(COMMON_CELL)

# ---------------------------------------------------------------- 2. 読み込み
md(r'''## 2. データの読み込み

全条件のファイルを読み、問題データ一覧の「除外」を除く。フーリエ数の列 `ln_Fo_cm` ＝ −(ln γ̇（条件平均）＋ 3 ln H) を足す（定数は GP に影響しないので省く）。

**理想的なスリット流れ**（等温で、固化層も壁面すべりもない）では、見かけの粘度と見かけのせん断速度の関係（見かけの流動曲線）は、**厚みによらず 1 本に重なる**。見かけの値と真の値の換算（ラビノビッチの補正）は、流動曲線の傾きだけで決まり、厚みに関係しないため（スリットやキャピラリーの寸法を変えて、流動曲線が重なるかで、壁面すべりなどを調べる方法〔Mooney の方法〕の前提）。**厚みで曲線がずれるなら、そのずれは、計算の都合ではなく、流れの中で起きている現象（冷却・固化層・せん断発熱など）による。**''')
code(r'''# セル2：読み込み
D = load_all()
for k in D:
    D[k]["ln_Fo_cm"] = -(D[k]["ln_gd_cm"] + 3.0 * D[k]["ln_H"])
leg = legacy_check(D)
judge(bool(leg["同じ行"].all()) or not USE_EXCLUSIONS, "一覧の「除外」を除いたデータが、no1.5-3.0 のファイル（P03 などの除外を同じように適用したもの）と同じ")
for mat in MATS:
    for g in ["5data", "3data", "Alldata"]:
        d = D[(mat, g, MAIN_DATA)]
        print(f"{MATS[mat]} {g}：{len(d)} 行 {d['cond'].nunique()} 条件")''')

# ---------------------------------------------------------------- 3. 式の確認
md(r'''## 3. 式の確認：算出式に入っている量だけで、ln η を再現できるか

ln η を、算出式に入っている量で、線形回帰する（ショットごと。5data で学習 → 3data で検証）。**リークなら、検証の RMSE が、同じ条件のショット間のばらつき（純粋誤差、約 0.04）よりずっと小さくなる**（計算で答えが出るため）。

| 回帰の入力 | 予想 |
|---|---|
| ln γ̇, ln H | H が式の一部としてだけ効くなら、ln H の係数は ＋1。実際は ΔP が分からないので、再現できないはず |
| ln γ̇, ln H, Tset | 同上 |
| ln Δt, ln H | ln γ̇ と同じ情報（γ̇ = 6L/(H·Δt)）。同じ結果になるはず |
| ln γ̇, ln H, ln ΔP | **算出式そのもの**。係数は (−1, ＋1, ＋1) で、完全に再現できるはず（リーク） |
| ln γ̇, ln ΔP | H なしでも、ΔP があれば大部分を再現できるか |
| ln γ̇, ln P3_2 | **H なしで、圧力 P3_2 を入れる**（W2・W3 の線形版）。算出式に入っていないので、再現できないはず（リークではない） |
| ln γ̇, ln H, ln P3_2 | 3a の L2。P3_2 を足すと、ln H の係数が変わるか（P3_2 が厚みの情報を持つ印） |''')
code(r'''# セル3：算出式の量での線形回帰
FORMS = {"ln γ̇ + ln H": ["ln_gd", "ln_H"], "ln γ̇ + ln H + Tset": ["ln_gd", "ln_H", "Tset"], "ln Δt + ln H": ["ln_dt", "ln_H"],
         "ln γ̇ + ln H + ln ΔP（算出式）": ["ln_gd", "ln_H", "ln_dP"], "ln γ̇ + ln ΔP": ["ln_gd", "ln_dP"],
         "ln γ̇ + ln P3_2": ["ln_gd", "ln_P32"], "ln γ̇ + ln H + ln P3_2": ["ln_gd", "ln_H", "ln_P32"]}
frows = []
for mat in MATS:
    tr, te = split_V(mat, MAIN_DATA)
    for d in (tr, te):
        d["ln_dP"] = np.log((d["P31"] - 0.30) * 1e6)                          # ΔP [Pa]（P3_1 − 0.30 MPa）
        d["ln_dt"] = np.log(6 * 17.5e-3 / (d["H"] * 1e-3 * d["gd"]))          # Δt [s]（γ̇ = 6L/(H·Δt) から）
    pure = within_stats(tr)["sd_ln_eta"]
    for name, cols in FORMS.items():
        m = fit_ols(tr[cols].to_numpy(), tr["ln_eta"].to_numpy())
        mu, _ = predict_ols(m, te[cols].to_numpy())
        rm = float(np.sqrt(np.mean((te["ln_eta"].to_numpy() - mu) ** 2)))
        frows.append({"material": MATS[mat], "inputs": name, **{f"coef_{c}": float(b) for c, b in zip(cols, m[0][1:])}, "resid_sd_train": float(np.sqrt(m[2])),
                      "RMSE_test": rm, "pure_sd": pure, "RMSE_over_pure": rm / pure})
form_df = pd.DataFrame(frows)
form_df.to_csv(os.path.join(OUT_DIR, "formula_check.csv"), index=False, encoding="utf-8-sig")
print(form_df.round(4).to_string(index=False))
for mat in MATS:
    s = form_df[form_df["material"] == MATS[mat]].set_index("inputs")
    a, b = s.loc["ln γ̇ + ln H", "RMSE_over_pure"], s.loc["ln γ̇ + ln H + ln ΔP（算出式）", "RMSE_over_pure"]
    judge(a > 2, f"{MATS[mat]}：ln γ̇ と ln H だけの回帰の検証 RMSE は、純粋誤差の {a:.1f} 倍（2 倍を超えれば、H と γ̇ からは ln η を計算できない ＝ 式によるリークではない）。ln H の係数は {s.loc['ln γ̇ + ln H', 'coef_ln_H']:+.2f}（式の一部としてだけ効くなら ＋1）")
    judge(b < 0.1, f"{MATS[mat]}：ΔP（P3_1 − 0.30 MPa）を足すと、検証 RMSE は純粋誤差の {b:.4f} 倍（ほぼ 0 ＝ ΔP はリーク。手順3a と同じ）")
    c = s.loc["ln γ̇ + ln P3_2", "RMSE_over_pure"]
    judge(c > 2, f"{MATS[mat]}：H なしで ln γ̇ と ln P3_2 だけの回帰の検証 RMSE は、純粋誤差の {c:.1f} 倍（2 倍を超えれば、P3_2 からも ln η を計算できない ＝ 算出式によるリークではない。手順3a の確認と同じ結論）")
    st = s.loc["ln γ̇ + ln H", "resid_sd_train"], s.loc["ln γ̇ + ln H + ln P3_2", "resid_sd_train"]
    print(f"   {MATS[mat]}：P3_2 を足したときの ln H の係数 {s.loc['ln γ̇ + ln H', 'coef_ln_H']:+.2f} → {s.loc['ln γ̇ + ln H + ln P3_2', 'coef_ln_H']:+.2f}、学習の残差 sd {st[0]:.3f} → {st[1]:.3f}（P3_1 なら 0.021）")''')
code(r'''# セル3.5：P3_2 が、厚みの代わりの目印になるか（条件平均。Alldata の外れ値なし）
def r2_of(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ beta
    return float(1 - (r @ r) / ((y - y.mean()) @ (y - y.mean())))


prows = []
for mat in MATS:
    d = D[(mat, "Alldata", MAIN_DATA)]
    g = d.groupby("cond").agg(H=("H", "first"), Tset=("Tset", "first"), lg=("ln_gd", "mean"), lp=("ln_P32", "mean"), lt=("ln_Tff", "mean")).reset_index()
    y = np.log(g["H"].to_numpy())
    for base_name, base in [("γ̇, Tset", [g["lg"], g["Tset"]]), ("γ̇, Tff", [g["lg"], g["lt"]])]:
        Xb = np.column_stack(base)
        Xp = np.column_stack(base + [g["lp"]])
        prows.append({"material": MATS[mat], "base": base_name, "R2_lnH_base": r2_of(Xb, y), "R2_lnH_with_P32": r2_of(Xp, y)})
    pc = np.corrcoef(np.log(g["H"]), g["lp"])[0, 1]
    prows[-1]["corr_lnH_lnP32"] = float(pc)
    prows[-2]["corr_lnH_lnP32"] = float(pc)
proxy_df = pd.DataFrame(prows)
proxy_df["gain"] = proxy_df["R2_lnH_with_P32"] - proxy_df["R2_lnH_base"]
proxy_df.to_csv(os.path.join(OUT_DIR, "p32_proxy.csv"), index=False, encoding="utf-8-sig")
# P3_2 は、粘度 η の情報を持つのか、厚み H の情報を持つのか：ln P3_2 を、ln η・ln γ̇・ln H（条件平均）で線形回帰する
qrows = []
for mat in MATS:
    d = D[(mat, "Alldata", MAIN_DATA)]
    g = d.groupby("cond").agg(H=("H", "first"), Tset=("Tset", "first"), lg=("ln_gd", "mean"), le=("ln_eta", "mean"), lp=("ln_P32", "mean"), l1=("ln_P31", "mean")).reset_index()
    g["lH"] = np.log(g["H"])
    for name, cols, ycol in [("ln P3_2 ~ ln η, ln γ̇, ln H", ["le", "lg", "lH"], "lp"), ("ln P3_2 ~ ln γ̇, ln H", ["lg", "lH"], "lp"), ("ln P3_2 ~ ln H", ["lH"], "lp"),
                             ("ln P3_1 ~ ln η, ln γ̇, ln H（比較：P3_1 は粘度の算出式に入っている）", ["le", "lg", "lH"], "l1")]:
        A = np.column_stack([np.ones(len(g)), g[cols].to_numpy()])
        beta, *_ = np.linalg.lstsq(A, g[ycol].to_numpy(), rcond=None)
        res = g[ycol].to_numpy() - A @ beta
        qrows.append({"material": MATS[mat], "regression": name, "R2": float(1 - res @ res / ((g[ycol] - g[ycol].mean()) ** 2).sum()), "resid_sd": float(np.sqrt(res @ res / (len(g) - A.shape[1]))),
                      **{f"coef_{c}": float(b) for c, b in zip(cols, beta[1:])}})
p32reg_df = pd.DataFrame(qrows)
p32reg_df.to_csv(os.path.join(OUT_DIR, "p32_regression.csv"), index=False, encoding="utf-8-sig")
print("\nln P3_2 を、条件平均の ln η・ln γ̇・ln H で回帰（P3_2 が何の情報を持つか）")
print(p32reg_df.round(3).to_string(index=False))
for mat in MATS:
    s = p32reg_df[(p32reg_df["material"] == MATS[mat]) & (p32reg_df["regression"] == "ln P3_2 ~ ln η, ln γ̇, ln H")].iloc[0]
    judge(abs(s["coef_le"]) < 0.2, f"{MATS[mat]}：ln P3_2 の回帰で、ln η の係数が {s['coef_le']:+.2f}（0 に近い ＝ P3_2 は、粘度そのものの情報をほとんど持たない。P3_1 は {p32reg_df[(p32reg_df['material'] == MATS[mat]) & p32reg_df['regression'].str.startswith('ln P3_1')].iloc[0]['coef_le']:+.2f}）。ln H の係数は {s['coef_lH']:+.2f}（厚みでほぼ決まる）")
print("ln H を、条件平均の入力で線形回帰したときの R2（1 なら、入力から H が完全に分かる）")
print(proxy_df.round(3).to_string(index=False))
for _, r in proxy_df.iterrows():
    judge(r["gain"] < 0.3, f"{r['material']}：入力（{r['base']}）に ln P3_2 を足したときの ln H の R2 が {r['R2_lnH_base']:.2f} → {r['R2_lnH_with_P32']:.2f}（増加 {r['gain']:.2f}。0.3 以上増えるなら、P3_2 が厚みの代わりの目印になりうる ＝ H を外して P3_2 を入れると、厚みの情報を P3_2 が運ぶ）")''')

# ---------------------------------------------------------------- 4. 厚みの効果
md(r'''## 4. 厚みの効果：同じ γ̇・同じ設定温度で、厚みによって粘度がどれだけ違うか

条件平均の見かけの流動曲線（ln γ̇ と ln η）を、厚みごとに、ln γ̇ について直線で補間し、**基準の厚み（`REF_H`）と γ̇ が重なる範囲で、粘度の比 η(H)/η(基準)** を出す。比べるために、**同じ厚み・同じ γ̇ での 190/230 ℃ の粘度の比**（温度の効果。手順3b・5a で 1.07〜1.21 倍）も同じ方法で出す。理想的なスリット流れなら、厚みの比は 1（曲線が重なる）。''')
code(r'''# セル4：厚みの比と、温度の比
def curve_ratio(cm, key_col, a_val, b_val, by_col):
    """by_col の値ごとに、key_col が a_val の曲線と b_val の曲線を、ln γ̇ が重なる範囲で比べる（ln η の差の平均・最小・最大）"""
    rows = []
    for bv, g in cm.groupby(by_col):
        A = g[np.isclose(g[key_col], a_val)].sort_values("ln_gd")
        B = g[np.isclose(g[key_col], b_val)].sort_values("ln_gd")
        if len(A) < 2 or len(B) < 2:
            continue
        lo, hi = max(A["ln_gd"].min(), B["ln_gd"].min()), min(A["ln_gd"].max(), B["ln_gd"].max())
        if hi <= lo:
            continue
        x = np.linspace(lo, hi, 50)
        d = np.interp(x, A["ln_gd"], A["ln_eta"]) - np.interp(x, B["ln_gd"], B["ln_eta"])
        rows.append({by_col: bv, "gd_lo": float(np.exp(lo)), "gd_hi": float(np.exp(hi)), "ratio_mean": float(np.exp(d.mean())), "ratio_min": float(np.exp(d.min())), "ratio_max": float(np.exp(d.max()))})
    return rows


trows = []
for mat in MATS:
    cm = cond_means(D[(mat, "Alldata", MAIN_DATA)])
    for h in sorted(cm["H"].unique()):
        if np.isclose(h, REF_H):
            continue
        for r in curve_ratio(cm, "H", h, REF_H, "Tset"):
            trows.append({"material": MATS[mat], "kind": "厚み", "compare": f"{h} mm / {REF_H} mm", "H": h, **r})
    for r in curve_ratio(cm, "Tset", 190.0, 230.0, "H"):
        trows.append({"material": MATS[mat], "kind": "温度", "compare": "190 ℃ / 230 ℃", **r})
ratio_df = pd.DataFrame(trows)
ratio_df.to_csv(os.path.join(OUT_DIR, "thickness_ratio.csv"), index=False, encoding="utf-8-sig")
print(ratio_df.round(3).to_string(index=False))
for mat in MATS:
    s = ratio_df[(ratio_df["material"] == MATS[mat])]
    th, tt = s[s["kind"] == "厚み"], s[s["kind"] == "温度"]
    th05 = th[np.isclose(th["H"], 0.5)]["ratio_mean"]
    judge(bool((th["ratio_mean"] - 1).abs().max() > 0.1), f"{MATS[mat]}：同じ γ̇・同じ設定温度で、厚みによる粘度の比が 1 から 10% 以上ずれる（範囲 {th['ratio_mean'].min():.2f}〜{th['ratio_mean'].max():.2f} 倍。0.5 mm / {REF_H} mm は {th05.min():.2f}〜{th05.max():.2f} 倍）→ 見かけの流動曲線は、厚みで重ならない（理想的なスリット流れではない）。温度の比（190/230 ℃）は {tt['ratio_mean'].min():.2f}〜{tt['ratio_mean'].max():.2f} 倍")''')
code(r'''# セル4.5：図1 厚みごとの見かけの流動曲線（条件平均。設定温度ごと）、図2 厚みの比と温度の比
HC = {0.5: "#d62728", 1.0: "#1f77b4", 1.5: "#f0b400", 2.0: "#2ca02c"}
fig, axes = plt.subplots(len(MATS), 3, figsize=(15, 4.6 * len(MATS)), squeeze=False)
for i, mat in enumerate(MATS):
    cm = cond_means(D[(mat, "Alldata", MAIN_DATA)])
    for j, t in enumerate(sorted(cm["Tset"].unique())):
        ax = axes[i, j]
        for h, g in cm[cm["Tset"] == t].groupby("H"):
            g = g.sort_values("ln_gd")
            ax.plot(np.exp(g["ln_gd"]), np.exp(g["ln_eta"]), "o-", color=HC[h], ms=5, label=f"{h} mm")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"見かけのせん断速度 $\dot{\gamma}$ (1/s)")
        ax.set_ylabel("見かけの粘度 η (Pa·s)")
        ax.set_title(f"{MATS[mat]}　設定温度 {int(t)} ℃")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title="厚み")
fig.suptitle("図1　厚みごとの見かけの流動曲線（条件平均）。理想的なスリット流れなら、厚みによらず 1 本に重なる", y=1.01)
fig.tight_layout()
print(save_fig(fig, OUT_DIR, "fig1_flow_curves_by_thickness"))
plt.show()

fig, axes = plt.subplots(1, len(MATS), figsize=(6.5 * len(MATS), 4.6), squeeze=False)
for i, mat in enumerate(MATS):
    ax = axes[0, i]
    s = ratio_df[(ratio_df["material"] == MATS[mat]) & (ratio_df["kind"] == "厚み")]
    for t, g in s.groupby("Tset"):
        ax.errorbar(g["H"] + (t - 210) / 400, g["ratio_mean"], yerr=[g["ratio_mean"] - g["ratio_min"], g["ratio_max"] - g["ratio_mean"]], fmt="o", capsize=3,
                    color={190: "tab:blue", 210: "tab:green", 230: "tab:red"}[int(t)], label=f"厚み：{int(t)} ℃（η(H)/η({REF_H} mm)）")
    tt = ratio_df[(ratio_df["material"] == MATS[mat]) & (ratio_df["kind"] == "温度")]
    ax.axhspan(tt["ratio_mean"].min(), tt["ratio_mean"].max(), color="gray", alpha=0.25, label="温度：η(190 ℃)/η(230 ℃) の範囲")
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xlabel("厚み H (mm)")
    ax.set_ylabel("同じ γ̇ での粘度の比")
    ax.set_title(MATS[mat])
    ax.legend(fontsize=8)
fig.suptitle(f"図2　同じ γ̇・同じ設定温度での、厚みによる粘度の比（点は平均、棒は重なる範囲での最小〜最大）。灰色の帯は、温度の効果", y=1.03)
fig.tight_layout()
print(save_fig(fig, OUT_DIR, "fig2_thickness_ratio"))
plt.show()''')

# ---------------------------------------------------------------- 5. GPR で H を外す
md(r'''## 5. GPR で、H を外す・替えると、どうなるか

モデル（冒頭の表）を、手順3d の 2 つの版で学習・評価する。5data 版：S_V（外挿）、S_T（210 ℃）。Alldata 版：CV（条件単位の 10 分割）。

**見る点**
- **リークの目印**：リークがあると、検証の RMSE が、純粋誤差（約 0.04）より明らかに小さくなる（ショットごとのゆれまで当ててしまう）。H は、同じ条件のショットで同じ値なので、ショットごとのゆれの情報を持たない。
- **H が要るか**：R3 と W0（H を外す）の差。W0 の誤差が、厚みごとに偏るか（厚みごとの残差の平均）。
- **Tff で代わりになるか**：W1（γ̇, Tff）、W1T（γ̇, Tff, Tset）。
- **Fo は H と同じ情報か**：RFo と R3 の差（構造 A・B の入力 ln Fo には、H が入っている）。
- **圧力 P3_2 で、H の代わりになるか**：W0 → W2（Tset ＋ P3_2）、W1 → W3（Tff ＋ P3_2）の差。良くなったとしても、(1) P3_2 は、予測のときに事前に分からない流れの結果で、(2) 薄いほど圧力が高いので、厚みの代わりの目印として使われているだけかもしれない（3 節の `p32_proxy.csv`）。P3_2 のリークの確認（算出式による）は、手順3a と、3 節の回帰で行っている。''')
code(r'''# セル5：学習（並列。15〜25 分）
SPECS = {"R3": ["ln_gd_cm", "ln_H", "Tset"], "RFo": ["ln_gd_cm", "ln_Fo_cm", "Tset"], "W0": ["ln_gd_cm", "Tset"], "W1": ["ln_gd_cm", "ln_Tff_cm"], "W1T": ["ln_gd_cm", "ln_Tff_cm", "Tset"],
         "W2": ["ln_gd_cm", "Tset", "ln_P32_cm"], "W3": ["ln_gd_cm", "ln_Tff_cm", "ln_P32_cm"]}
for k in KEYS:
    MODELS[k] = {"kind": "gpr", "cols": SPECS[k], "linear": True}
LABELS = {"R3": r"R3：［$\dot{\gamma}$, $H$, $T_{set}$］", "RFo": r"RFo：［$\dot{\gamma}$, Fo, $T_{set}$］", "W0": r"W0：［$\dot{\gamma}$, $T_{set}$］（Cross-WLF と同じ変数）",
          "W1": r"W1：［$\dot{\gamma}$, $T_{ff}$］（Cross-WLF と同じ変数）", "W1T": r"W1T：［$\dot{\gamma}$, $T_{ff}$, $T_{set}$］",
          "W2": r"W2：［$\dot{\gamma}$, $T_{set}$, P3_2］（Cross-WLF と同じ変数）", "W3": r"W3：［$\dot{\gamma}$, $T_{ff}$, P3_2］（Cross-WLF と同じ変数）"}


def job(mat, ver, name, tr, tests, key, restarts):
    warnings.simplefilter("ignore")
    t0 = time.time()
    fit = fit_model(key, tr, restarts)
    return mat, ver, name, key, fit, {tn: predict_model(key, fit, te)[:3] for tn, te in tests.items()}, time.time() - t0


SETS3E, JOBS = {}, []
for ver in DATA_VERSIONS:
    for mat in MATS:
        vs = version_sets(mat, ver, MAIN_DATA, TEMP_HOLDOUT)
        sp = {k: v for k, v in vs["splits"].items() if (k in ("S_V", "S_T") if ver == "5data" else k.startswith("CV"))}
        SETS3E[(mat, ver)] = {"pool": vs["pool"], "splits": sp}
        for key in KEYS:
            for name, (tr, tests) in sp.items():
                JOBS.append((mat, ver, name, tr, tests, key, N_RESTARTS_CV if name.startswith("CV") else N_RESTARTS))
            if ver == "Alldata":
                JOBS.append((mat, ver, "ALL", vs["pool"], {}, key, N_RESTARTS))      # 最終モデル（高解像度のグラフ用）。5data 版は S_V と同じ学習
JOBS.sort(key=lambda j: -len(j[3]))
t0 = time.time()
RES = Parallel(n_jobs=N_JOBS)(delayed(job)(*j) for j in JOBS)
print(f"{len(JOBS)} 回の学習を並列（{N_JOBS}）で計算：{(time.time() - t0) / 60:.1f} 分")''')
code(r'''# セル6：指標、厚みごとの残差、ハイパーパラメータ
FIT3E, acc, KROWS = {}, {}, []
for (mat, ver, name, key, fit, preds, sec), j in zip(RES, JOBS):
    tr, tests = j[3], j[4]
    FIT3E[(mat, ver, name, key)] = fit
    KROWS.append({"material": MATS[mat], "version": ver, "split": name, "model": key, "n_train": len(tr), **kernel_row(key, fit), "pure_sd_ln_eta": within_stats(tr)["sd_ln_eta"], "fit_sec": sec})
    for tn, te in tests.items():
        mu, sd, sdf = preds[tn]
        split = "CV" if tn.startswith("CV") else tn
        acc.setdefault((mat, ver, split, key), []).append(pd.DataFrame({"y": te["ln_eta"].to_numpy(), "mu": mu, "sd": sd, "H": te["H"].to_numpy(), "Tset": te["Tset"].to_numpy(),
                                                                        "v": te["v"].to_numpy(), "cond": te["cond"].to_numpy(), "group": te["group"].to_numpy()}))
for mat in MATS:
    for key in KEYS:
        if (mat, "5data", "S_V", key) in FIT3E:
            FIT3E[(mat, "5data", "ALL", key)] = FIT3E[(mat, "5data", "S_V", key)]
kern_df = pd.DataFrame(KROWS)
kern_df.to_csv(os.path.join(OUT_DIR, "kernels.csv"), index=False, encoding="utf-8-sig")
PRED3E, mrows, brows = {}, [], []
for (mat, ver, split, key), parts in acc.items():
    d = pd.concat(parts, ignore_index=True)
    if split == "CV":
        d["group"] = np.where(d["v"] <= 5, "低速側", np.where(d["v"] >= 280, "高速側", "内側"))
    PRED3E[(mat, ver, split, key)] = d
    k = kern_df[(kern_df["material"] == MATS[mat]) & (kern_df["version"] == ver) & (kern_df["model"] == key)]
    k = k[k["split"].str.fullmatch(r"CV\d+")] if split == "CV" else k[k["split"] == {"S_T3": "S_T"}.get(split, split)]
    subs = {"全体": np.ones(len(d), bool), **{g: (d["group"] == g).to_numpy() for g in d["group"].unique() if g != "全体"}}
    for sub, m in subs.items():
        if m.sum() < 2:
            continue
        ev = evaluate(d["y"].to_numpy()[m], d["mu"].to_numpy()[m], d["sd"].to_numpy()[m])
        mrows.append({"material": MATS[mat], "version": ver, "split": split, "subset": sub, "model": key, "n_test": int(m.sum()), **ev,
                      "noise_sd": float(k["noise_sd"].mean()), "pure_sd": float(k["pure_sd_ln_eta"].mean())})
    r = d.assign(r=d["y"] - d["mu"])
    for h, g in r.groupby("H"):
        brows.append({"material": MATS[mat], "version": ver, "split": split, "model": key, "H": h, "bias": float(g["r"].mean()), "sd": float(g["r"].std()), "n": len(g)})
met_df = pd.DataFrame(mrows)
bias_df = pd.DataFrame(brows)
met_df.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False, encoding="utf-8-sig")
bias_df.to_csv(os.path.join(OUT_DIR, "bias_by_thickness.csv"), index=False, encoding="utf-8-sig")
for mat in MATS:
    print(f"\n【{MATS[mat]}】RMSE（括弧は被覆率）")
    t = met_df[met_df["material"] == MATS[mat]].copy()
    t["v"] = t["RMSE"].map("{:.3f}".format) + "(" + t["coverage95"].map("{:.2f}".format) + ")"
    print(t.pivot_table(index=["version", "split", "subset"], columns="model", values="v", aggfunc="first").reindex(columns=KEYS).to_string())
print("\n【厚みごとの残差の平均（実測 − 予測。ln η）】Alldata 版の CV と、5data 版の S_V")
show = bias_df[((bias_df["version"] == "Alldata") & (bias_df["split"] == "CV")) | ((bias_df["version"] == "5data") & (bias_df["split"] == "S_V"))]
print(show.pivot_table(index=["material", "version", "split", "H"], columns="model", values="bias").reindex(columns=KEYS).round(3).to_string())
print("\n【長さスケール（標準化した入力。小さいほど、その入力で出力が大きく変わる。下限 0.01）】5data 版の S_V")
ls_cols = [c for c in kern_df.columns if c.startswith("ls_")]
print(kern_df[(kern_df["version"] == "5data") & (kern_df["split"] == "S_V")][["material", "model", "noise_sd", "pure_sd_ln_eta"] + ls_cols].round(3).to_string(index=False))''')
code(r'''# セル7：判定
def mv(mat, ver, split, model, sub="全体", col="RMSE"):
    r = met_df[(met_df["material"] == MATS[mat]) & (met_df["version"] == ver) & (met_df["split"] == split) & (met_df["model"] == model) & (met_df["subset"] == sub)]
    return float(r[col].iloc[0]) if len(r) else np.nan


CHECKS = [(v, s) for v, s in [("5data", "S_V"), ("5data", "S_T"), ("Alldata", "CV")] if v in DATA_VERSIONS]
for mat in MATS:
    n = MATS[mat]
    for ver, sp in CHECKS:
        r3, pure = mv(mat, ver, sp, "R3"), mv(mat, ver, sp, "R3", col="pure_sd")
        judge(r3 > 0.8 * pure, f"{n}・{ver} 版 {sp}：H を入れた R3 の RMSE {r3:.3f} が、純粋誤差 {pure:.3f} の 0.8 倍より大きい（ショットごとのゆれまで当ててはいない ＝ H によるリークの目印はない）")
    for ver, sp in CHECKS:
        if "W0" in KEYS and "R3" in KEYS:
            r3, w0 = mv(mat, ver, sp, "R3"), mv(mat, ver, sp, "W0")
            judge(w0 < 1.5 * r3, f"{n}・{ver} 版 {sp}：H を外した W0 の RMSE {w0:.3f} が、R3 {r3:.3f} の 1.5 倍未満（Cross-WLF と同じ変数でも足りる）。超えれば、見かけの粘度の予測に H が要る")
        if "W1" in KEYS and "R3" in KEYS:
            r3, w1 = mv(mat, ver, sp, "R3"), mv(mat, ver, sp, "W1")
            judge(w1 < 1.5 * r3, f"{n}・{ver} 版 {sp}：H の代わりに Tff を使った W1 の RMSE {w1:.3f} が、R3 {r3:.3f} の 1.5 倍未満（流れの中の樹脂温度で、厚みの効果を説明できる）")
    for key, nm in [("W2", "Tset と P3_2"), ("W3", "Tff と P3_2")]:
        if key not in KEYS or "R3" not in KEYS:
            continue
        for ver, sp in CHECKS:
            r3, wk, pure = mv(mat, ver, sp, "R3"), mv(mat, ver, sp, key), mv(mat, ver, sp, "R3", col="pure_sd")
            judge(wk < 1.5 * r3, f"{n}・{ver} 版 {sp}：H を外して P3_2 を入れた {key}（{nm}）の RMSE {wk:.3f} が、R3 {r3:.3f} の 1.5 倍未満（H なしでも、圧力で足りる）。超えれば、圧力を入れても、H の代わりにならない")
            judge(wk > 0.8 * pure, f"{n}・{ver} 版 {sp}：{key} の RMSE {wk:.3f} が、純粋誤差 {pure:.3f} の 0.8 倍より大きい（P3_2 によるリークの目印はない）")
    if "W2" in KEYS and "W0" in KEYS:
        for ver, sp in CHECKS:
            w0, w2 = mv(mat, ver, sp, "W0"), mv(mat, ver, sp, "W2")
            judge(w2 < 0.5 * w0, f"{n}・{ver} 版 {sp}：圧力 P3_2 を足すと（W0 {w0:.3f} → W2 {w2:.3f}）、RMSE が半分未満になる（P3_2 に、H の代わりになる情報がある）")
    if "RFo" in KEYS and "Alldata" in DATA_VERSIONS:
        r3, rf = mv(mat, "Alldata", "CV", "R3"), mv(mat, "Alldata", "CV", "RFo")
        judge(abs(rf / r3 - 1) < 0.05, f"{n}：Fo を使った RFo の CV の RMSE {rf:.3f} が、R3 {r3:.3f} とほぼ同じ（±5%。Fo は H と同じ情報を持つ ＝ 構造 A・B も、Fo を通して H を使っている）")
    if "W0" in KEYS and "Alldata" in DATA_VERSIONS:
        b = bias_df[(bias_df["material"] == n) & (bias_df["version"] == "Alldata") & (bias_df["split"] == "CV") & (bias_df["model"] == "W0")]
        print(f"   {n}：W0 の厚みごとの残差の平均（Alldata 版の CV）：" + "、".join(f"{h} mm {v:+.3f}" for h, v in zip(b["H"], b["bias"])))''')

# ---------------------------------------------------------------- 6. 図
md(r'''## 6. 図

- 図3：モデルごとの RMSE（分割ごと）。破線は純粋誤差。
- 図4：厚みごとの残差の平均（Alldata 版の CV）。0 に近いほど、厚みによる偏りがない。''')
code(r'''# セル8：図3 RMSE、図4 厚みごとの残差の平均
CATS = [(v, s, sub) for v, s, sub in [("5data", "S_V", "低速側"), ("5data", "S_V", "高速側"), ("5data", "S_T", "全体"), ("Alldata", "CV", "全体"), ("Alldata", "CV", "低速側"), ("Alldata", "CV", "高速側")] if v in DATA_VERSIONS]
MCOL = {"R3": "k", "RFo": "tab:gray", "W0": "tab:red", "W1": "tab:orange", "W1T": "tab:purple", "W2": "tab:blue", "W3": "tab:green"}
fig, axes = plt.subplots(1, len(MATS), figsize=(7.5 * len(MATS), 4.8), squeeze=False)
for i, mat in enumerate(MATS):
    ax = axes[0, i]
    w = 0.8 / len(KEYS)
    for k, key in enumerate(KEYS):
        ax.bar(np.arange(len(CATS)) + (k - (len(KEYS) - 1) / 2) * w, [mv(mat, v, s, key, sub) for v, s, sub in CATS], w, color=MCOL[key], label=LABELS[key])
    pure = np.nanmean([mv(mat, v, s, "R3", sub, "pure_sd") for v, s, sub in CATS])
    ax.axhline(pure, color="k", ls="--", lw=0.8, label=f"純粋誤差（約 {pure:.3f}）")
    ax.set_xticks(np.arange(len(CATS)))
    ax.set_xticklabels([f"{v} 版\n{s} {sub}" for v, s, sub in CATS], fontsize=8)
    ax.set_ylabel("RMSE（ln η）")
    ax.set_title(MATS[mat])
    ax.legend(fontsize=7)
fig.suptitle("図3　モデルごとの RMSE。H を外す（W0・W1）と、誤差が大きくなる", y=1.02)
fig.tight_layout()
print(save_fig(fig, OUT_DIR, "fig3_rmse_by_model"))
plt.show()

if "Alldata" in DATA_VERSIONS:
    fig, axes = plt.subplots(1, len(MATS), figsize=(6.5 * len(MATS), 4.6), squeeze=False)
    for i, mat in enumerate(MATS):
        ax = axes[0, i]
        b = bias_df[(bias_df["material"] == MATS[mat]) & (bias_df["version"] == "Alldata") & (bias_df["split"] == "CV")]
        for k, key in enumerate(KEYS):
            g = b[b["model"] == key].sort_values("H")
            ax.errorbar(g["H"] + (k - (len(KEYS) - 1) / 2) * 0.03, g["bias"], yerr=g["sd"] / np.sqrt(g["n"]), fmt="o-", color=MCOL[key], capsize=3, label=LABELS[key])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("厚み H (mm)")
        ax.set_ylabel("残差の平均（実測 − 予測、ln η）")
        ax.set_title(f"{MATS[mat]}（Alldata 版の CV）")
        ax.legend(fontsize=7)
    fig.suptitle("図4　厚みごとの残差の平均。H を入れたモデル（R3・RFo）は 0 の上に並び、H を外したモデルは厚みで偏る", y=1.03)
    fig.tight_layout()
    print(save_fig(fig, OUT_DIR, "fig4_bias_by_thickness"))
    plt.show()''')

# ---------------------------------------------------------------- 7. 高解像度のグラフ
md(HR_MD.replace("## 高解像度のグラフ", "## 7. 高解像度のグラフ") + "\n\nH を外したモデル（W0・W1）は、厚みが違っても同じ曲線になる（入力に H がないため）。設定温度ごとのレイアウトで、4 つの厚みの曲線が重なり、点が曲線から離れる様子が見える。")
code(hr_cell("# セル9：高解像度のグラフ（せん断速度 ― 粘度。GPR6月ノートのセル9-2 にならう）",
             '''HR_MODELS = ["R3", "W0", "W1", "W2", "W3"]     # 描くモデル（KEYS のどれでも）
HR_VERSIONS = ["Alldata"]          # 描く版（"5data"：5data で学習〔3data は白抜き〕／"Alldata"：全条件で学習）''',
             r'''hr_saved = []
for mat in HR_MATERIALS:
    for ver in HR_VERSIONS:
        for key in HR_MODELS:
            if (mat, ver, "ALL", key) not in FIT3E:
                print(f"{MATS[mat]}・{ver} 版・{key} は、学習していないので描かない（DATA_VERSIONS・KEYS に入れて、上から実行し直す）")
                continue
            t0 = time.time()
            fit = FIT3E[(mat, ver, "ALL", key)]
            pool = SETS3E[(mat, ver)]["pool"] if ver == "Alldata" else pool_of(mat, MAIN_DATA)
            hr_saved += hr_flow(lambda Xg, f=fit, k=key: predict_model(k, f, Xg.assign(ln_Fo_cm=-(Xg["ln_gd_cm"] + 3 * Xg["ln_H"])))[:3], mat, MAIN_DATA,
                                set(SETS3E[(mat, ver)]["pool"]["cond"]), f"{LABELS[key]}（{ver} 版）", f"hr_{mat}_{key}_{ver}", HR_DIR, pool=pool, **HR_KW)
            print(f"{MATS[mat]}・{ver} 版・{key}：保存したファイル 累計 {len(hr_saved)} 個（{time.time() - t0:.0f} 秒）")
judge(len(hr_saved) > 0, f"高解像度のグラフを {len(hr_saved)} 個保存した（{os.path.normpath(HR_DIR)}）。形式 {HR_FORMATS}、{HR_DPI} dpi")'''))

# ---------------------------------------------------------------- 8. 保存したファイル
code(r'''# セル10：保存したファイル
print("保存したファイル:")
for f in sorted(os.listdir(OUT_DIR)):
    print("  ", os.path.normpath(os.path.join(OUT_DIR, f)))''')


# ---------------------------------------------------------------- 書き出し
def to_lines(s):
    lines = s.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


rnd = random.Random(20261015)
for c in cells:
    c["id"] = "%08x" % rnd.getrandbits(32)
    c["source"] = to_lines(c["source"])

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "ガウス過程回帰3 (Python 3.12)", "language": "python", "name": "gaussian-process3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = sys.argv[1]
if os.path.exists(out) and "--force" not in sys.argv[2:]:
    sys.exit(f"既にファイルがあるため上書きしません（上書きするときは --force）: {out}")
assert len({c["id"] for c in cells}) == len(cells)
for c in cells:  # コードセルの構文チェック (%matplotlib の行は除く)
    if c["cell_type"] == "code":
        compile("".join(l for l in c["source"] if not l.startswith("%")), "<cell>", "exec")
with open(out, "wb") as f:
    f.write((json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
print(f"作成: {out}（{len(cells)} セル）")
