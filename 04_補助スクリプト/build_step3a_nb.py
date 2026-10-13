# -*- coding: utf-8 -*-
"""手順3a（新データ：圧力列の検討）のノートブック (未実行) を生成する。

使い方: python build_step3a_nb.py <出力する .ipynb のパス> [--force]
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


# 全ノートに共通の部分（使い方、設定エリア、共通の部品）
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nb_parts import COMMON_CELL, settings_cell, usage_md  # noqa: E402


# ---------------------------------------------------------------- 0. タイトル
md(r'''# 手順3a：圧力列の検討（新データ）

CLAUDE.md §7 の手順3a。新しいデータ（PP 再々実験・粉砕材混合 PP）には、流動中樹脂圧力が **P3_1**（センサ4に樹脂が届いた時刻のセンサ3の圧力）と **P3_2**（センサ4が最大値になった時刻のセンサ3の圧力）の2列ある。どちらかを説明変数に使えるかを、次の観点で確かめる（方針を変えるかは、結果を見てユーザーが決める）。

1. **算出式のリーク**：線形回帰で R² ≈ 1、係数が ±1 にならないか
2. **予測のときに使えるか**：成形条件から予測するとき、その値が事前に分かるか
3. **検証データでの精度**：低速側（3・5 mm/s）と高速側（280・320 mm/s）に分けた R²・RMSE（ln η）、95% 予測区間の被覆率、NLPD
4. **物理的な妥当さ**：ARD の長さスケール、学習範囲の外で予測の幅が広がるか
5. **ノイズ分散**：同じ条件のショット間のばらつき（純粋誤差）と比べてどうか

- データ：外れ値なしが主（外れ値ありは感度確認）。PP は厚み 1.5 mm・射出速度 3 mm/s の3条件を除いた no1.5-3.0 のファイル。新データの `NR500/` の下の `結合_*.csv`（既存の CSV は変更しない）
- 図・表の保存先：`results/step3a/`（図の文字は日本語。フォントは IPAexGothic）
- 計算は自然対数 ln で行う。η [Pa·s]、γ̇ [1/s]、H [mm]、Tff [K]（表示は ℃）。圧力は CSV では MPa だが、**Pa に直してから ln をとる**（3.1 で、MPa のままでも結果が変わらないことを確かめる）。乱数シードは 42
- 標準化と中心化は、**学習データだけ**で計算する（旧ノートは全行で計算していた）
- ノイズ分散（WhiteKernel）の下限は 1e-6（旧ノートは 1e-3）

| 記号 | モデル | 入力（ln） |
|---|---|---|
| L0 | 線形回帰（基準） | γ̇, H |
| L1 | 線形回帰 | P3_1, γ̇, H |
| L2 | 線形回帰 | P3_2, γ̇, H |
| A0 | GPR（圧力なし） | γ̇, H, Tff |
| A1 | GPR | γ̇, H, Tff, P3_1 |
| A2 | GPR | γ̇, H, Tff, P3_2 |
| A1′ | GPR（現行ノートの形） | γ̇, Tff, P3_1 |

GPR はすべて ARD Matern 5/2 ＋ WhiteKernel（`n_restarts_optimizer=5`）。付録（11 節）で、低速側への外挿のために線形の傾向を入れた GPR を試算する（3b 以降の準備）。

| 記号 | 分割 | 備考 |
|---|---|---|
| S_V | 5data で学習 → 3data で検証 | 射出速度の両端の外挿（低速側 3・5 mm/s、高速側 280・320 mm/s）。学習と検証で条件は重ならない |
| S_R1 | Alldata のランダム 20% をテスト | 従来の評価方法（行単位）。評価の甘さを見る |
| S_R2 | 5data＋3data のランダム 20% をテスト（PP のみ） | PP は Alldata が10ショット/条件なので、データ量を 5data＋3data にそろえた場合 |
| LOCO | 条件を1つずつ抜く交差検証 | ハイパーパラメータは全データで推定して固定し、閉じた式で計算（近似） |''')

# ---------------------------------------------------------------- 1. 設定
md(usage_md("3a", "results/step3a", "results/学習結果説明_手順3a.pdf", 11, hr=False))
md(r'''## 1. 設定''')
code(settings_cell())
code(r'''# セル1.5：ライブラリ、フォント、出力先
%matplotlib inline
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

OUT_DIR = os.path.join(REPO, "results", "step3a")
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 80    # ノート内の表示は小さめにする
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

# 図の日本語フォント（IPAexGothic。ない文字は DejaVu Sans で表示）
try:
    fm.findfont("IPAexGothic", fallback_to_default=False)
except ValueError:
    raise RuntimeError("日本語フォント IPAexGothic が見つかりません。インストールしてからカーネルを再起動してください。")
plt.rcParams["font.family"] = ["IPAexGothic", "DejaVu Sans"]

print("Python:", sys.executable, "/", sys.version.split()[0])
print("出力先:", os.path.abspath(OUT_DIR))''')
md(r'''### 共通の部品

データの読み込み、問題データ一覧の「除外」、モデル（GPR・線形回帰）、評価、データの分け方、条件を1つずつ抜く交差検証の関数。中身は `04_補助スクリプト/nb_common.py` と同じ（直すときは、そのファイルを直して、ノートを作り直す）。''')
code(COMMON_CELL)

# ---------------------------------------------------------------- 2. データ
md(r'''## 2. データの読み込みと確認

全条件の結合ファイルを読み、列名・欠損・元ファイル名と列の一致を確かめる。`02_機械学習データ/問題データ一覧.csv` の「除外」の条件（今は、PP の厚み 1.5 mm・射出速度 3 mm/s）を除く。「感度確認」の条件は残し、`sens` 列に ID を付ける。圧力は MPa なので、Pa に直してから ln をとる。条件番号は (設定温度, 厚み, 射出速度) の組で作る。''')
code(r'''# セル2：読み込み（全条件のファイルを読み、一覧の「除外」を除く）
D = load_all()
pl = read_problem_list()
print("問題データ一覧（除外・感度確認）")
print(pl[pl["区分"].isin(["除外", "感度確認"])][["ID", "区分", "材料", "設定温度", "厚み", "射出速度", "再測定", "今の扱い"]].to_string(index=False))
print("\n一覧の「除外」で除いた行数:", {f"{MATS[m]} {g} {o}": n for (m, g, o), n in REMOVED.items() if n})
leg = legacy_check(D)
print("\n全条件のファイルから一覧で除いたデータが、no1.5-3.0 のファイルと同じ行か")
print(leg.to_string(index=False))
judge(bool(leg["同じ行"].all()) or not USE_EXCLUSIONS, "一覧の「除外」を除いたデータが、これまでの no1.5-3.0 のファイルと同じ（取り直した後は違ってよい）")

rows = []
for (mat, grp, o), df in D.items():
    sh = df.groupby("cond").size()
    rows.append({"材料": MATS[mat], "組": grp, "外れ値": o, "行数": len(df), "条件数": df["cond"].nunique(),
                 "ショット/条件": f"{sh.min()}〜{sh.max()}", "γ̇ [1/s]": f"{df['gd'].min():.0f}〜{df['gd'].max():.0f}",
                 "η [Pa·s]": f"{df['eta'].min():.2f}〜{df['eta'].max():.0f}", "P3_1 [MPa]": f"{df['P31'].min():.2f}〜{df['P31'].max():.1f}",
                 "P3_2 [MPa]": f"{df['P32'].min():.1f}〜{df['P32'].max():.1f}", "P3_2/P3_1": f"{(df['P32'] / df['P31']).min():.1f}〜{(df['P32'] / df['P31']).max():.1f}"})
summary_df = pd.DataFrame(rows)
summary_df.to_csv(os.path.join(OUT_DIR, "data_summary.csv"), index=False, encoding="utf-8-sig")
print(summary_df.to_string(index=False))''')

md(r'''### 2.1 粘度の算出式と Δt の確認

η·γ̇ = H(P3_1 − c)/(2L) が成り立つか（c：センサ4の立ち上がり時の圧力）。γ̇ = 6L/(H·Δt) から Δt を逆算し、サンプリング間隔の整数倍になっているかも見る（5 µs で確認）。''')
code(r'''# セル3：算出式と Δt
rows = []
for (mat, grp, o), df in D.items():
    tau = df["eta"] * df["gd"] / 1e6                          # せん断応力 [MPa]
    A = np.column_stack([df["H"] * df["P31"], df["H"]])
    coef, *_ = np.linalg.lstsq(A, tau, rcond=None)
    rel_fix = np.abs(tau / (df["H"] * (df["P31"] - 0.30) / (2 * 17.5)) - 1)
    dt_us = 6 * 17.5e-3 / (df["H"] * 1e-3 * df["gd"]) * 1e6   # Δt [µs]
    k5 = dt_us / 5.0
    rows.append({"材料": MATS[mat], "組": grp, "外れ値": o, "L推定 [mm]": 1 / (2 * coef[0]), "c推定 [MPa]": -coef[1] / coef[0],
                 "最大相対誤差(L=17.5,c=0.30) [%]": rel_fix.max() * 100, "Δt [ms]": f"{dt_us.min() / 1e3:.1f}〜{dt_us.max() / 1e3:.0f}",
                 "5µs の倍数からのずれの最大": np.abs(k5 - np.round(k5)).max()})
formula_df = pd.DataFrame(rows)
formula_df.to_csv(os.path.join(OUT_DIR, "formula_check.csv"), index=False, encoding="utf-8-sig")
print(formula_df.round(4).to_string(index=False))''')

md(r'''### 2.2 同じ条件のショット間のばらつき（純粋誤差）

同じ条件（設定温度・厚み・射出速度）の中で、ln η がどれだけばらつくか。ショットごとの γ̇ と連動するので、ln γ̇ で説明した残りも出す。外れ値なしは、中央値から 20% の打ち切りを含むので、ばらつきが小さめに出る。''')
code(r'''# セル4：純粋誤差（within_stats は共通の部品）
rows = [{"材料": MATS[mat], "組": grp, "外れ値": o, **within_stats(df)} for (mat, grp, o), df in D.items()]
pure_df = pd.DataFrame(rows)
pure_df.to_csv(os.path.join(OUT_DIR, "pure_error.csv"), index=False, encoding="utf-8-sig")
print(pure_df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))''')

md(r'''### 2.3 データの全体像（図0）

学習（5data）と検証（3data）が、せん断速度と粘度の上でどこにあるか。検証データは、学習データの範囲の外（低速側と高速側）にある。''')
code(r'''# セル4.5：図0 データの全体像
T_COL = {190: "tab:blue", 210: "tab:green", 230: "tab:red"}
H_LEVELS = [0.5, 1.0, 1.5, 2.0]
fig, axes = plt.subplots(len(MATS), len(H_LEVELS), figsize=(20, 9), sharex=True, sharey=True)
for i, mat in enumerate(MATS):
    tr, te = D[(mat, "5data", "なし")], D[(mat, "3data", "なし")]
    for j, h in enumerate(H_LEVELS):
        ax = axes[i, j]
        a_all = tr[tr["H"] == h]
        ax.axvspan(a_all["gd"].min(), a_all["gd"].max(), color="gray", alpha=0.10, label="学習データのせん断速度の範囲")
        for t, col in T_COL.items():
            a = tr[(tr["H"] == h) & (tr["Tset"] == t)]
            b = te[(te["H"] == h) & (te["Tset"] == t)]
            ax.scatter(a["gd"], a["eta"], s=14, color=col, alpha=0.7, label=f"学習 {t} ℃")
            ax.scatter(b["gd"], b["eta"], s=26, facecolors="none", edgecolors=col, marker="s", linewidths=1.2, label=f"検証 {t} ℃")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{MATS[mat]}　厚み {h} mm")
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ [1/s]")
        if j == 0:
            ax.set_ylabel(r"見かけの粘度 $\eta$ [Pa·s]")
axes[0, 0].legend(fontsize=7, ncol=2)
fig.suptitle("学習データ（塗りつぶし）と検証データ（白抜き）の位置（外れ値なし）", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig0_data_overview.png"), dpi=FIG_DPI)
plt.show()''')

# ---------------------------------------------------------------- 3. モデル
md(r'''## 3. モデルと評価の関数

- GPR：`C * Matern(ARD, nu=2.5) + WhiteKernel`。標準化と中心化は学習データだけで計算する。
- GPR の予測標準偏差 `sd` には、ノイズ分散が含まれる。潜在関数の標準偏差 `sd_f` は、`sd² − ノイズ分散` から求める。
- 評価（ln η）：R²、RMSE、95% 予測区間の被覆率（|z| ≤ 1.96）、NLPD（各ショットの予測分布の負の対数密度の平均）。
- LOCO（条件を1つずつ抜く）：ハイパーパラメータを全データで推定して固定し、条件 g を除いた予測分布を `平均 = y_g − [K⁻¹_gg]⁻¹ (K⁻¹y)_g`、`共分散 = [K⁻¹_gg]⁻¹` で求める（閉じた式）。中心化と標準化は全データで計算するので、少し甘い評価になる。''')
code(r'''# セル5：モデルの定義（学習・予測・評価の関数は共通の部品）
MODELS.update({
    "L0": {"kind": "ols", "cols": ["ln_gd", "ln_H"]},
    "L1": {"kind": "ols", "cols": ["ln_P31", "ln_gd", "ln_H"]},
    "L2": {"kind": "ols", "cols": ["ln_P32", "ln_gd", "ln_H"]},
    "A0": {"kind": "gpr", "cols": ["ln_gd", "ln_H", "ln_Tff"]},
    "A1": {"kind": "gpr", "cols": ["ln_gd", "ln_H", "ln_Tff", "ln_P31"]},
    "A2": {"kind": "gpr", "cols": ["ln_gd", "ln_H", "ln_Tff", "ln_P32"]},
    "A1p": {"kind": "gpr", "cols": ["ln_gd", "ln_Tff", "ln_P31"]},
})
LABELS = {
    "L0": r"L0：線形［$\dot{\gamma}$, $H$］",
    "L1": r"L1：線形［P3_1, $\dot{\gamma}$, $H$］",
    "L2": r"L2：線形［P3_2, $\dot{\gamma}$, $H$］",
    "A0": r"A0：GPR［$\dot{\gamma}$, $H$, $T_{ff}$］",
    "A1": r"A1：GPR［$\dot{\gamma}$, $H$, $T_{ff}$, P3_1］",
    "A2": r"A2：GPR［$\dot{\gamma}$, $H$, $T_{ff}$, P3_2］",
    "A1p": r"A1′：GPR［$\dot{\gamma}$, $T_{ff}$, P3_1］",
}
OLS_KEYS = ["L0", "L1", "L2"]
GPR_KEYS = ["A0", "A1", "A2", "A1p"]
ALL_KEYS = OLS_KEYS + GPR_KEYS
INPUT_JA = {"ln_gd": "γ̇", "ln_H": "H", "ln_Tff": "Tff", "ln_P31": "P3_1", "ln_P32": "P3_2"}


for mat in MATS:
    tr, te = split_V(mat, "なし")
    print(f"{MATS[mat]} S_V：学習 {len(tr)} 行 {tr['cond'].nunique()} 条件 / 検証 {len(te)} 行 {te['cond'].nunique()} 条件"
          f"（低速側 {int((te['group'] == '低速側').sum())} 行、高速側 {int((te['group'] == '高速側').sum())} 行）")''')

md(r'''### 3.1 圧力の単位の確認（MPa と Pa）

圧力は CSV では MPa。ln をとって学習データで標準化するので、MPa のまま ln をとっても、Pa に直してから ln をとっても、**GPR の予測は同じ**になる（ln P が全行で一律に ln(10⁶) ≈ 13.8 だけずれ、標準化で平均を引くと消える）。線形回帰でも予測は同じで、変わるのは**切片だけ**（係数 × ln 10⁶ だけずれる）。S_V（外れ値なし）で確かめる。''')
code(r'''# セル5.1：圧力の単位の確認
for df in D.values():
    df["ln_P31_MPa"], df["ln_P32_MPa"] = np.log(df["P31"]), np.log(df["P32"])      # MPa のまま
MODELS["L1_MPa"] = {"kind": "ols", "cols": ["ln_P31_MPa", "ln_gd", "ln_H"]}
MODELS["L2_MPa"] = {"kind": "ols", "cols": ["ln_P32_MPa", "ln_gd", "ln_H"]}
MODELS["A2_MPa"] = {"kind": "gpr", "cols": ["ln_gd", "ln_H", "ln_Tff", "ln_P32_MPa"]}
rows = []
for mat in MATS:
    tr, te = split_V(mat, "なし")
    for key_pa, key_mpa in [("L1", "L1_MPa"), ("L2", "L2_MPa"), ("A2", "A2_MPa")]:
        f_pa, f_mpa = fit_model(key_pa, tr), fit_model(key_mpa, tr)
        mu_pa, sd_pa, _, _ = predict_model(key_pa, f_pa, te)
        mu_mpa, sd_mpa, _, _ = predict_model(key_mpa, f_mpa, te)
        row = {"材料": MATS[mat], "モデル": key_pa, "予測平均の差の最大": np.abs(mu_pa - mu_mpa).max(),
               "予測標準偏差の差の最大": np.abs(sd_pa - sd_mpa).max()}
        if MODELS[key_pa]["kind"] == "ols":
            b_pa, b_mpa = f_pa["model"][0], f_mpa["model"][0]
            row.update({"係数の差の最大": np.abs(b_pa[1:] - b_mpa[1:]).max(), "切片の差（MPa − Pa）": b_mpa[0] - b_pa[0],
                        "圧力の係数 × ln(10^6)": b_pa[1] * np.log(1e6)})
        rows.append(row)
unit_df = pd.DataFrame(rows)
unit_df.to_csv(os.path.join(OUT_DIR, "unit_check.csv"), index=False, encoding="utf-8-sig")
for k in ["L1_MPa", "L2_MPa", "A2_MPa"]:
    del MODELS[k]
print(unit_df.to_string(index=False, float_format=lambda x: f"{x:.3g}"))''')

# ---------------------------------------------------------------- 4. S_V
md(r'''## 4. 5data → 3data（S_V）

射出速度の両端の外挿。低速側（3・5 mm/s）と高速側（280・320 mm/s）に分けて評価する。''')
code(r'''# セル6：S_V（外れ値なし）の学習と評価（数分かかる）
t_all = time.time()
for mat in MATS:
    tr, te = split_V(mat, "なし")
    run_split(mat, "S_V", "なし", tr, te)
print(f"\n合計 {(time.time() - t_all) / 60:.1f} 分")''')

# ---------------------------------------------------------------- 5. ランダム
md(r'''## 5. ランダム 20%（S_R1・S_R2）

従来と同じ行単位の分割。同じ条件のショットが学習とテストの両方に入るので、評価が甘くなる。''')
code(r'''# セル7：ランダム 20%（外れ値なし）の学習と評価（PP の Alldata は 1,000 行近いので数分かかる）
t_all = time.time()
for mat in MATS:
    tr, te = split_random(D[(mat, "Alldata", "なし")].reset_index(drop=True))
    n_both = len(set(tr["cond"]) & set(te["cond"]))
    print(f"{MATS[mat]} S_R1：学習 {len(tr)} 行 / テスト {len(te)} 行、テストの条件 {te['cond'].nunique()} のうち学習にも入る条件 {n_both}")
    run_split(mat, "S_R1", "なし", tr, te)
tr, te = split_random(pool_of("PP", "なし"))
print(f"PP S_R2：学習 {len(tr)} 行 / テスト {len(te)} 行、テストの条件 {te['cond'].nunique()} のうち学習にも入る条件 {len(set(tr['cond']) & set(te['cond']))}")
run_split("PP", "S_R2", "なし", tr, te)
print(f"\n合計 {(time.time() - t_all) / 60:.1f} 分")''')

# ---------------------------------------------------------------- 6. LOCO
md(r'''## 6. 条件を1つずつ抜く交差検証（LOCO）

5data＋3data（PP は no1.5-3.0）の全条件を対象に、条件を1つずつ抜いて予測する。低速側・高速側は射出速度で分ける（3・5 mm/s、280・320 mm/s）。''')
code(r'''# セル8：LOCO（loco_gpr・loco_ols は共通の部品）
LOCO_ROWS, LOCO_PRED = [], {}
t_all = time.time()
for mat in MATS:
    pool = pool_of(mat, "なし")
    grp = np.where(pool["v"] <= 5, "低速側", np.where(pool["v"] >= 280, "高速側", "内側"))
    for key in ALL_KEYS:
        t0 = time.time()
        if MODELS[key]["kind"] == "gpr":
            mu, sd, gp = loco_gpr(key, pool)
            noise_sd = float(np.sqrt(gp.kernel_.k2.noise_level))
        else:
            mu, sd = loco_ols(key, pool)
            noise_sd = np.nan
        LOCO_PRED[(mat, key)] = pd.DataFrame({"y": pool["ln_eta"].to_numpy(), "mu": mu, "sd": sd, "v": pool["v"].to_numpy(), "group": grp})
        for sub in ["全体", "低速側", "内側", "高速側"]:
            mask = np.ones(len(pool), bool) if sub == "全体" else (grp == sub)
            LOCO_ROWS.append({"material": MATS[mat], "model": key, "subset": sub, "n_test": int(mask.sum()),
                              "n_cond": int(pool.loc[mask, "cond"].nunique()), **evaluate(pool["ln_eta"].to_numpy()[mask], mu[mask], sd[mask]),
                              "noise_sd": noise_sd})
        print(f"{MATS[mat]} LOCO {key}: {len(pool)} 行 {pool['cond'].nunique()} 条件、{time.time() - t0:.1f} 秒")
loco_df = pd.DataFrame(LOCO_ROWS)
loco_df.to_csv(os.path.join(OUT_DIR, "loco_metrics.csv"), index=False, encoding="utf-8-sig")
print(f"\n合計 {(time.time() - t_all) / 60:.1f} 分")''')

# ---------------------------------------------------------------- 7. 感度確認
md(r'''## 7. 感度確認：外れ値ありのデータ

外れ値ありのデータ（5data → 3data）で、主な結論が変わるかを確かめる。''')
code(r'''# セル9：S_V（外れ値あり）
t_all = time.time()
for mat in MATS:
    tr, te = split_V(mat, "あり")
    print(f"{MATS[mat]} S_V（外れ値あり）：学習 {len(tr)} 行 / 検証 {len(te)} 行")
    run_split(mat, "S_V", "あり", tr, te, keys=["L0", "L1", "L2", "A0", "A1", "A2"])
print(f"\n合計 {(time.time() - t_all) / 60:.1f} 分")''')

# ---------------------------------------------------------------- 8. 結果の表
md(r'''## 8. 結果の表

- 観点1（リーク）：線形回帰 L1・L2 の係数と R²
- 観点3（検証データでの精度）：低速側・高速側に分けた指標
- 観点5（ノイズ分散）：GPR のノイズの標準偏差と、同じ条件内の標準偏差（純粋誤差）''')
code(r'''# セル10：表と CSV
metrics_df = pd.DataFrame(METRICS)
kernels_df = pd.DataFrame(KERNELS)
ols_df = pd.DataFrame(OLSROWS)
metrics_df.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False, encoding="utf-8-sig")
kernels_df.to_csv(os.path.join(OUT_DIR, "kernels.csv"), index=False, encoding="utf-8-sig")
ols_df.to_csv(os.path.join(OUT_DIR, "ols_coefficients.csv"), index=False, encoding="utf-8-sig")

print("【観点1】線形回帰の係数（S_V の学習データ 5data で当てはめ。外れ値なし）")
print(ols_df.query("split == 'S_V' and data == 'なし'").drop(columns=["split", "data"]).round(4).to_string(index=False))


def show(mat, split, data="なし"):
    t = metrics_df[(metrics_df["material"] == MATS[mat]) & (metrics_df["split"] == split) & (metrics_df["data"] == data)]
    p = t.pivot(index="model", columns="subset", values=["RMSE", "R2", "coverage95", "NLPD"])
    order = [s for s in ["全体", "低速側", "高速側"] if s in p.columns.get_level_values(1)]
    p = p.reindex(columns=[(m, s) for m in ["RMSE", "R2", "coverage95", "NLPD"] for s in order]).reindex(ALL_KEYS if data == "なし" else ["L0", "L1", "L2", "A0", "A1", "A2"])
    print(p.round(3).to_string())


for mat in MATS:
    for split in ["S_V", "S_R1", "S_R2"]:
        if ((metrics_df["material"] == MATS[mat]) & (metrics_df["split"] == split)).any():
            n = metrics_df[(metrics_df["material"] == MATS[mat]) & (metrics_df["split"] == split) & (metrics_df["data"] == "なし")].iloc[0]
            print(f"\n【観点3】{MATS[mat]} {split}（学習 {n['n_train']} 行。外れ値なし。ln η の指標）")
            show(mat, split)
print("\n【感度確認】S_V（外れ値あり）")
for mat in MATS:
    print(f"\n{MATS[mat]}")
    show(mat, "S_V", "あり")''')
code(r'''# セル11：ハイパーパラメータ、ノイズ分散と純粋誤差、LOCO
cols = ["material", "split", "data", "model", "n_train", "amplitude", "ls_ln_gd", "ls_ln_H", "ls_ln_Tff", "ls_ln_P31", "ls_ln_P32",
        "noise_sd", "pure_sd_ln_eta", "pure_sd_after_gd", "LML", "noise_at_lower_bound", "ls_at_bound"]
print("【観点4・5】GPR のハイパーパラメータ（長さスケールは標準化後の入力に対する値。ARD：小さいほど効く）")
print(kernels_df[cols].query("data == 'なし'").round(4).to_string(index=False))
print("\n【観点5】noise_sd は GPR のノイズの標準偏差、pure_sd_ln_eta は学習データの同じ条件内の標準偏差、pure_sd_after_gd は ln γ̇ で説明した残り")

print("\n【LOCO】条件を1つずつ抜く交差検証（ln η の指標）")
for mat in MATS:
    t = loco_df[loco_df["material"] == MATS[mat]]
    p = t.pivot(index="model", columns="subset", values=["RMSE", "coverage95", "NLPD"])
    p = p.reindex(columns=[(m, s) for m in ["RMSE", "coverage95", "NLPD"] for s in ["全体", "低速側", "内側", "高速側"]]).reindex(ALL_KEYS)
    print(f"\n{MATS[mat]}（{int(t['n_test'].iloc[0])} 行 {int(t['n_cond'].iloc[0])} 条件）")
    print(p.round(3).to_string())''')

md(r'''### 8.1 検証データで気になる条件

PP の検証データの、190 ℃・厚み 0.5 mm・射出速度 320 mm/s の条件は、同じ速度の 210・230 ℃ に比べて、せん断速度が約 0.56 倍、粘度が約 5 倍で、外れています（外れ値の判定のあとでも、2〜3 ショットが残る）。この条件を除くと、高速側の指標がどう変わるかを見る。''')
code(r'''# セル11.5：気になる条件（190 ℃・0.5 mm・320 mm/s）の確認
ANOM = "190_0.5_320"
cond_rows = []
for mat in MATS:
    te = D[(mat, "3data", "なし")]
    s = te[(te["H"] == 0.5) & (te["v"] == 320)]
    for t, g in s.groupby("Tset"):
        cond_rows.append({"材料": MATS[mat], "設定温度 [℃]": int(t), "ショット数": len(g), "γ̇ の平均 [1/s]": g["gd"].mean(), "η の平均 [Pa·s]": g["eta"].mean(), "Tff の平均 [℃]": g["Tff"].mean()})
cond_df = pd.DataFrame(cond_rows)
print("検証データの 0.5 mm・320 mm/s（外れ値なし）")
print(cond_df.round(2).to_string(index=False))
cond_df.to_csv(os.path.join(OUT_DIR, "anomaly_condition.csv"), index=False, encoding="utf-8-sig")

rows = []
for mat in MATS:
    for key in ALL_KEYS:
        p = PRED[(mat, "S_V", "なし", key)]
        hi, an = (p["group"] == "高速側").to_numpy(), (p["cond"] == ANOM).to_numpy()
        for label, m in [("高速側すべて", hi), ("高速側（この条件を除く）", hi & ~an), ("この条件だけ", an)]:
            if m.sum() > 0:
                rows.append({"material": MATS[mat], "model": key, "subset": label, "n_test": int(m.sum()),
                             **evaluate(p["y"].to_numpy()[m], p["mu"].to_numpy()[m], p["sd"].to_numpy()[m])})
anom_df = pd.DataFrame(rows)
anom_df.to_csv(os.path.join(OUT_DIR, "anomaly_check.csv"), index=False, encoding="utf-8-sig")
print("\n高速側の指標（A0・A2。ln η）")
print(anom_df[anom_df["model"].isin(["A0", "A2"])][["material", "model", "subset", "n_test", "RMSE", "coverage95", "NLPD", "mean_sd"]].round(3).to_string(index=False))''')

# ---------------------------------------------------------------- 9. 図
md(r'''## 9. 図

- 図1：S_V のパリティ図（縦棒は 95% 予測区間）
- 図2：分割による RMSE・NLPD の違い（評価の甘さ）
- 図3：ARD の感度（長さスケールの逆数）
- 図4：予測標準偏差と射出速度（学習範囲の外で幅が広がるか）''')
code(r'''# セル12：図1 パリティ図（S_V）
KEYS_FIG = ["L1", "L2", "A0", "A1", "A2", "A1p"]
GCOL = {"低速側": "tab:blue", "高速側": "tab:red"}
fig, axes = plt.subplots(len(MATS), len(KEYS_FIG), figsize=(22, 8.8))
for i, mat in enumerate(MATS):
    for j, key in enumerate(KEYS_FIG):
        ax = axes[i, j]
        p = PRED[(mat, "S_V", "なし", key)]
        for g, col in GCOL.items():
            m = (p["group"] == g).to_numpy()
            ax.errorbar(p["y"][m], p["mu"][m], yerr=1.96 * p["sd"][m], fmt="o", ms=3, alpha=0.6, elinewidth=0.6, color=col, ecolor="lightgray", label=g)
        lo, hi = min(p["y"].min(), p["mu"].min()) - 0.2, max(p["y"].max(), p["mu"].max()) + 0.2
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        mt = metrics_df[(metrics_df["material"] == MATS[mat]) & (metrics_df["split"] == "S_V") & (metrics_df["data"] == "なし")
                        & (metrics_df["model"] == key)].set_index("subset")
        ax.set_title(f"{MATS[mat]}　{LABELS[key]}\nRMSE 低速側 {mt.loc['低速側', 'RMSE']:.3f}／高速側 {mt.loc['高速側', 'RMSE']:.3f}", fontsize=8.5)
        ax.set_xlabel(r"実測値 $\ln\eta$（$\eta$ は Pa·s）", fontsize=8)
        ax.set_ylabel(r"予測値 $\ln\eta$", fontsize=8)
        ax.grid(alpha=0.3)
        if i == 0 and j == 0:
            ax.legend(fontsize=8)
fig.suptitle("5data → 3data（S_V）のパリティ図（外れ値なし）", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_parity_S_V.png"), dpi=FIG_DPI)
plt.show()''')
code(r'''# セル13：図2 分割による RMSE・NLPD の違い
panels = [("S_V", "低速側", "S_V 低速側"), ("S_V", "高速側", "S_V 高速側"), ("S_R1", "全体", "ランダム20%（Alldata）"), ("S_R2", "全体", "ランダム20%（5data＋3data）")]
colors = ["tab:blue", "tab:red", "tab:green", "tab:olive"]
fig, axes = plt.subplots(2, len(MATS), figsize=(14, 8.5))
for j, mat in enumerate(MATS):
    for i, (metric, ylab) in enumerate([("RMSE", r"RMSE（$\ln\eta$）"), ("NLPD", "NLPD（小さいほど良い）")]):
        ax = axes[i, j]
        w = 0.8 / len(panels)
        for k, (split, sub, lab) in enumerate(panels):
            t = metrics_df[(metrics_df["material"] == MATS[mat]) & (metrics_df["split"] == split) & (metrics_df["data"] == "なし") & (metrics_df["subset"] == sub)]
            if t.empty:
                continue
            t = t.set_index("model").reindex(ALL_KEYS)
            ax.bar(np.arange(len(ALL_KEYS)) + (k - (len(panels) - 1) / 2) * w, t[metric], w, color=colors[k], label=lab)
        ax.set_xticks(np.arange(len(ALL_KEYS)))
        ax.set_xticklabels(["L0", "L1", "L2", "A0", "A1", "A2", "A1′"])
        ax.set_ylabel(ylab)
        ax.set_title(f"{MATS[mat]}：{ylab}")
        ax.grid(alpha=0.3, axis="y")
        if metric == "RMSE":
            ax.set_yscale("log")
        if i == 0 and j == 0:
            ax.legend(fontsize=8)
fig.suptitle("分割による評価の違い（ランダム分割は条件ごと学習に入るので、検証より甘い）", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_metrics_by_split.png"), dpi=FIG_DPI)
plt.show()''')
code(r'''# セル14：図3 ARD の感度、図4 予測標準偏差と射出速度
fig, axes = plt.subplots(1, len(MATS), figsize=(14, 4.8))
base = kernels_df.query("split == 'S_V' and data == 'なし'")
inputs = ["ln_gd", "ln_H", "ln_Tff", "ln_P31", "ln_P32"]
for ax, mat in zip(axes, MATS):
    t = base[base["material"] == MATS[mat]].set_index("model").reindex(GPR_KEYS)
    w = 0.8 / len(GPR_KEYS)
    for k, key in enumerate(GPR_KEYS):
        vals = [1.0 / t.loc[key, f"ls_{c}"] if not np.isnan(t.loc[key, f"ls_{c}"]) else 0.0 for c in inputs]
        ax.bar(np.arange(len(inputs)) + (k - (len(GPR_KEYS) - 1) / 2) * w, vals, w, label=LABELS[key])
    ax.set_xticks(np.arange(len(inputs)))
    ax.set_xticklabels([INPUT_JA[c] for c in inputs])
    ax.set_ylabel("感度 = 1／長さスケール（標準化後）")
    ax.set_title(f"{MATS[mat]}：ARD の感度（S_V の学習）")
    ax.grid(alpha=0.3, axis="y")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_ard_sensitivity.png"), dpi=FIG_DPI)
plt.show()

fig, axes = plt.subplots(1, len(MATS), figsize=(14, 4.8), sharey=True)
for ax, mat in zip(axes, MATS):
    for key in GPR_KEYS:
        p = PRED[(mat, "S_V", "なし", key)]
        g = p.groupby("v")["sd"].mean()
        ax.plot(g.index, g.values, "o-", label=LABELS[key])
    ax.axvspan(10, 240, color="gray", alpha=0.12, label="学習範囲（10〜240 mm/s）")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("射出速度 [mm/s]")
    ax.set_title(f"{MATS[mat]}：検証データの予測標準偏差（ノイズ込み）の平均")
    ax.grid(alpha=0.3, which="both")
axes[0].set_ylabel(r"予測標準偏差（$\ln\eta$）")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_predictive_sd_vs_speed.png"), dpi=FIG_DPI)
plt.show()''')

# ---------------------------------------------------------------- 10. まとめ
md(r'''## 10. まとめ（観点ごとの数値）''')
code(r'''# セル15：観点ごとの要点（数値は上の表と同じ）
print("【観点1 リーク】S_V の学習データで ln η を線形回帰したときの検証 R²と係数")
for mat in MATS:
    for key in ["L1", "L2"]:
        r = ols_df[(ols_df["split"] == "S_V") & (ols_df["data"] == "なし") & (ols_df["model"] == key) & (ols_df["material"] == MATS[mat])].iloc[0]
        cname = "ln_P31" if key == "L1" else "ln_P32"
        te_r2 = metrics_df[(metrics_df["split"] == "S_V") & (metrics_df["data"] == "なし") & (metrics_df["model"] == key)
                           & (metrics_df["subset"] == "全体") & (metrics_df["material"] == MATS[mat])]["R2"].iloc[0]
        print(f"  {MATS[mat]} {key}：検証 R² {te_r2:.4f}、係数 ln P = {r[cname]:.3f}、ln γ̇ = {r['ln_gd']:.3f}、ln H = {r['ln_H']:.3f}")
print("\n【観点2 予測のときに使えるか】射出速度・厚み・設定温度は成形条件として事前に決まる。γ̇（ショットごとの実測）、Tff、P3_1、P3_2 は流動の結果で、事前には分からない。")
print("\n【観点5 ノイズ分散】S_V の学習で、GPR のノイズの標準偏差（noise_sd）と、同じ条件内の標準偏差（純粋誤差）")
print(kernels_df.query("split == 'S_V' and data == 'なし'")[["material", "model", "noise_sd", "pure_sd_ln_eta", "pure_sd_after_gd", "noise_at_lower_bound"]].round(4).to_string(index=False))

# ---- 判定（基準は、文の中に書いてある）
def mval(mat, model, sub, col="RMSE", split="S_V"):
    r = metrics_df[(metrics_df["material"] == MATS[mat]) & (metrics_df["split"] == split) & (metrics_df["data"] == "なし")
                   & (metrics_df["model"] == model) & (metrics_df["subset"] == sub)]
    return float(r[col].iloc[0])


print("\n【判定】（OK＝想定どおり／★要確認＝想定と違う、または注意が要る）")
for mat in MATS:
    n = MATS[mat]
    judge(mval(mat, "L1", "全体", "R2") >= 0.999, f"{n}：P3_1 を入れた線形回帰（L1）の検証 R² = {mval(mat, 'L1', '全体', 'R2'):.4f}（0.999 以上なら、粘度の算出式のリークを確認 → P3_1 は説明変数に使えない）")
    judge(mval(mat, "L2", "全体") >= 0.9 * mval(mat, "L0", "全体"), f"{n}：P3_2 を足した線形回帰（L2）の検証 RMSE {mval(mat, 'L2', '全体'):.3f} が、圧力なし（L0）{mval(mat, 'L0', '全体'):.3f} の 0.9 倍以上（改善が 1 割未満 → P3_2 は効果が小さい）")
    judge(mval(mat, "A0", "低速側") >= 1.0, f"{n}：定数平均の GPR（A0）の低速側 RMSE = {mval(mat, 'A0', '低速側'):.3f}（1.0 以上なら、低速側へ外挿できない。想定どおり。線形カーネルで解決する → 手順3b）")
    judge(mval(mat, "A0", "高速側") < 0.1, f"{n}：A0 の高速側 RMSE = {mval(mat, 'A0', '高速側'):.3f}（0.1 未満なら、高速側は当たっている）")
    cov = mval(mat, "A0", "高速側", "coverage95")
    judge(0.9 <= cov <= 0.99, f"{n}：A0 の高速側の 95% 予測区間の被覆率 = {cov:.2f}（0.90〜0.99 なら、区間の幅が適切。外れるなら、区間が狭すぎる）")
    ks = kernels_df[(kernels_df["split"] == "S_V") & (kernels_df["data"] == "なし") & (kernels_df["material"] == MATS[mat]) & (kernels_df["model"] == "A0")].iloc[0]
    judge(ks["noise_sd"] < 0.5 * ks["pure_sd_ln_eta"], f"{n}：A0 のノイズの標準偏差 {ks['noise_sd']:.4f} が、同じ条件内の標準偏差（純粋誤差）{ks['pure_sd_ln_eta']:.4f} の半分未満（ショットごとの γ̇ を入力にしたため小さい。想定どおり）")
    rr = mval(mat, "A0", "全体", "RMSE", "S_R1")
    judge(rr < 0.5 * mval(mat, "A0", "高速側"), f"{n}：ランダム 20% の RMSE {rr:.3f} が、検証（高速側 {mval(mat, 'A0', '高速側'):.3f}）の半分未満（ランダム分割は評価が甘い。想定どおり）")''')

# ---------------------------------------------------------------- 11. 付録
md(r'''## 11. 付録：線形の傾向を入れた試算（3b 以降の準備）

定数平均の GPR（A0）は、学習範囲の外（低速側）で平均値に戻り、大きく外れた（4 節）。粘度は ln γ̇ に対して強い傾向を持つので、次の2つを S_V（外れ値なし。入力は A0 と同じ）で試す。

- **(a) 線形回帰の残差を GPR で予測**：線形回帰（ln γ̇, ln H）を学習データで当てはめ、その**残差**を GPR で予測する。予測 ＝ 線形回帰 ＋ GPR。線形の傾きは固定され、傾きの不確かさは予測区間に入らない。
- **(b) カーネルに線形項を足す**：GPR のカーネルに線形項（DotProduct）を足す。傾きも GPR の中で、データから同時に推定され、傾きの不確かさも予測区間に入る。ユーザーの決定（10-12）で、3b 以降はこの (b) を使う。

(a) は二重学習にはならない。GPR が学ぶのは、線形回帰で説明しきれなかった残りだけで、テストデータは学習に使わない。ただし、傾きと残りを別々に決めるので、全体として最適とは限らない。''')
code(r'''# セル16：線形の傾向を入れた試算（(b) は、共通の部品の make_gpr(d, linear=True)）
TREND = {"A0": "定数平均（A0）", "a": "(a) 線形回帰の残差を GPR", "b": "(b) 線形カーネルを足す"}
TREND_PRED, trend_rows = {}, []
for mat in MATS:
    tr, te = split_V(mat, "なし")
    cols = MODELS["A0"]["cols"]
    sc = StandardScaler().fit(tr[cols])
    Xtr, Xte = sc.transform(tr[cols]), sc.transform(te[cols])
    y_tr, y_te = tr["ln_eta"].to_numpy(), te["ln_eta"].to_numpy()
    out = {}
    gp = make_gpr(3).fit(Xtr, y_tr - y_tr.mean())                       # 定数平均（4 節の A0 と同じ）
    mu, sd = gp.predict(Xte, return_std=True)
    out["A0"] = (mu + y_tr.mean(), sd, gp)
    A = np.column_stack([np.ones(len(tr)), tr[["ln_gd", "ln_H"]]])      # (a) 線形回帰 ＋ 残差の GPR
    beta = np.linalg.lstsq(A, y_tr, rcond=None)[0]
    gp = make_gpr(3).fit(Xtr, y_tr - A @ beta)
    mu, sd = gp.predict(Xte, return_std=True)
    out["a"] = (np.column_stack([np.ones(len(te)), te[["ln_gd", "ln_H"]]]) @ beta + mu, sd, gp)
    gp = make_gpr(3, linear=True).fit(Xtr, y_tr - y_tr.mean())          # (b) 線形カーネルを足す
    mu, sd = gp.predict(Xte, return_std=True)
    out["b"] = (mu + y_tr.mean(), sd, gp)
    for k, (mu, sd, gp) in out.items():
        TREND_PRED[(mat, k)] = pd.DataFrame({"y": y_te, "mu": mu, "sd": sd, "group": te["group"].to_numpy()})
        for sub in ["低速側", "高速側"]:
            m = (te["group"] == sub).to_numpy()
            trend_rows.append({"material": MATS[mat], "model": TREND[k], "subset": sub, "n_test": int(m.sum()), **evaluate(y_te[m], mu[m], sd[m]),
                               "noise_sd": float(np.sqrt(gp.kernel_.k2.noise_level)), "kernel": str(gp.kernel_)})
trend_df = pd.DataFrame(trend_rows)
trend_df.to_csv(os.path.join(OUT_DIR, "linear_trend_trial.csv"), index=False, encoding="utf-8-sig")
print(trend_df.drop(columns=["kernel"]).round(3).to_string(index=False))
print()
for _, r in trend_df[trend_df["subset"] == "低速側"].iterrows():
    print(f"{r['material']} {r['model']}: {r['kernel']}")''')
code(r'''# セル17：図5 線形の傾向の試算（パリティ図）
GCOL = {"低速側": "tab:blue", "高速側": "tab:red"}
fig, axes = plt.subplots(len(MATS), 3, figsize=(15, 9))
for i, mat in enumerate(MATS):
    for j, k in enumerate(["A0", "a", "b"]):
        ax = axes[i, j]
        p = TREND_PRED[(mat, k)]
        for g, col in GCOL.items():
            m = (p["group"] == g).to_numpy()
            ax.errorbar(p["y"][m], p["mu"][m], yerr=1.96 * p["sd"][m], fmt="o", ms=3, alpha=0.6, elinewidth=0.6, color=col, ecolor="lightgray", label=g)
        lo, hi = min(p["y"].min(), p["mu"].min()) - 0.2, max(p["y"].max(), p["mu"].max()) + 0.2
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        t = trend_df[(trend_df["material"] == MATS[mat]) & (trend_df["model"] == TREND[k])].set_index("subset")
        ax.set_title(f"{MATS[mat]}　{TREND[k]}\nRMSE 低速側 {t.loc['低速側', 'RMSE']:.3f}／高速側 {t.loc['高速側', 'RMSE']:.3f}、"
                     f"被覆率 {t.loc['低速側', 'coverage95']:.2f}／{t.loc['高速側', 'coverage95']:.2f}", fontsize=9)
        ax.set_xlabel(r"実測値 $\ln\eta$（$\eta$ は Pa·s）", fontsize=8)
        ax.set_ylabel(r"予測値 $\ln\eta$", fontsize=8)
        ax.grid(alpha=0.3)
        if i == 0 and j == 0:
            ax.legend(fontsize=8)
fig.suptitle("線形の傾向を入れた試算（5data → 3data。縦棒は 95% 予測区間）", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig5_linear_trend_trial.png"), dpi=FIG_DPI)
plt.show()''')
code(r'''# セル18：保存したファイル
print("保存したファイル:")
for f in sorted(os.listdir(OUT_DIR)):
    print("  ", os.path.normpath(os.path.join(OUT_DIR, f)))''')


# ---------------------------------------------------------------- 書き出し
def to_lines(s):
    lines = s.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


rnd = random.Random(20261012)
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
