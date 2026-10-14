# -*- coding: utf-8 -*-
"""手順3d（学習データの確認：5data 版と Alldata 版）のノートブック (未実行) を生成する。

使い方: python build_step3d_nb.py <出力する .ipynb のパス> [--force]
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
md(r'''# 手順3d：学習データの確認（5data 版と Alldata 版）

CLAUDE.md §3.2・§6。**今後の検討（手順5 以降）で使うデータを、2 つの版にそろえる**。外れ値ありのデータは、今後は使わない（ユーザー決定、2026-10-14）。

| 版 | 学習 | 検証 | 最終モデル（ALL） |
|---|---|---|---|
| **5data 版**（外挿） | 5data の外れ値なし（93 条件） | 3data の外れ値なし（射出速度の両端の外挿）。S_T：5data の 190・230 ℃ → 210 ℃ | 5data（3data は検証専用） |
| **Alldata 版** | Alldata の外れ値なし（全条件。PP は 1 条件 10 ショット） | 条件単位の交差検証（CV）。S_T：Alldata の 190・230 ℃ → 210 ℃（テストは 5data 版と同じ行）。S_Vall：Alldata のうち 5data の条件だけで学習 → 3data を検証（PP のみ意味がある） | Alldata（全条件） |

- どちらも、問題データ一覧の「除外」を除く（P01：1.5 mm・3 mm/s、**P03：PP の 190 ℃・0.5 mm・320 mm/s**〔この手順で「感度確認」から「除外」へ〕、P08：重複行）。
- **外れ値の検知（規則 f）**を足した。条件平均の GPR で、条件を 1 つずつ抜いて予測し、残差が大きい条件を拾う。拾った条件は、一覧に記録して除いてよい（ユーザー、10-14）。
- モデルは **R3**（条件平均の ln γ̇、ln H、Tset。ARD Matern 5/2 ＋ 線形カーネル ＋ ノイズ。CLAUDE.md §6 の比較用 GPR）だけ。構造あり（B・A）は、手順5 の本実装で、この 2 つの版に当てはめる。
- この手順の目的は、**2 つの版のデータの中身と、データの注意点（ショットの順番によるずれ、外れ値）を確かめ、同じモデルで精度を比べること**。
- **図は、すべて高解像度（600 dpi）でも `results/step3d/highres/` に保存される**（設定 `HR_ALL`・`HR_ALL_DPI`。報告書・発表資料用。Git の管理外）。せん断速度 ― 粘度のグラフは、最後の節のセルで描く（そのセルだけを実行し直せる）。
- ノートは、設定エリアの `DATA_VERSIONS` で、5data 版だけ・Alldata 版だけ・両方を選べる。外れ値の検知（規則 f）で拾った条件は、`02_機械学習データ/問題データ一覧.csv` に「除外」で記録すれば、ノートを作り直さずに、次の実行から除かれる。

| 節 | 内容 | 出力 |
|---|---|---|
| 2 | 版の中身（行数・条件数・分け方） | `versions.csv` |
| 3 | ファイルの包含関係（5data・3data が Alldata の一部か） | `inclusion.csv` |
| 4 | ショットの順番によるずれ（Alldata の注意点） | `shot_drift.csv`、図1 |
| 5 | 外れ値の検知（規則 f）、問題の条件の確認 | `outlier_check.csv`、図2・図3 |
| 6〜7 | R3 で 2 つの版を学習・評価 | `metrics.csv`、`kernels.csv`、図4・図5 |
| 8 | 高解像度のグラフ（せん断速度 ― 粘度） | `highres/` |

**注意**：Alldata 版の利点（両端の条件も学習に入る）は、同じデータの中の検証（CV）では測れない。CV で両端の条件を抜いたときの成績は、5data 版の外挿より楽になる（残りの条件が両端の近くにあるため）。''')

# ---------------------------------------------------------------- 1. 設定
md(usage_md("3d", "results/step3d", "results/学習結果説明_手順3d.pdf", 20))
md(r'''## 1. 設定''')
code(settings_cell('''DATA_VERSIONS = ["5data", "Alldata"]   # 検討する版。"5data"：5data で学習し、3data で外挿を検証する版／"Alldata"：全条件で学習する版。片方だけにもできる
MAIN_DATA = "なし"            # 外れ値なしのファイルを使う（外れ値ありは、今後は使わない方針。ユーザー、2026-10-14）
TEMP_HOLDOUT = 210            # S_T でテストにする設定温度 [℃]
CV_K = 10                     # 交差検証の分割数（条件単位）
N_RESTARTS_CV = 1             # 交差検証の各分割での最適化のやり直し回数（少なくてよい：他の回数でも LML は同じだった）
N_JOBS = 6                    # 並列に計算する数（CPU の数に合わせる）
DRIFT_TOL = 0.02              # ショットの順番によるずれの許容値（ln η。同じ条件内の標準偏差 約 0.04 の半分）
OUTLIER_Z = 3.5               # 規則 f（外れ値の検知）：z がこれを超え、かつ
OUTLIER_RESID = 0.1           #   |残差 ln η| がこれを超える条件を候補にする（0.1 は粘度で約 10%）
HR_ALL = True                 # すべての図を、高解像度でも results/step3d/highres/ に保存するか（Git の管理外）
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

OUT_DIR = os.path.join(REPO, "results", "step3d")
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

データの読み込み、問題データ一覧の「除外」、モデル、評価、データの分け方（版ごとの分割 `version_sets`、条件単位の交差検証 `split_cv`）、外れ値の検知（`outlier_scan`）、高解像度の図（`hr_flow`、`save_fig`）の関数。中身は `04_補助スクリプト/nb_common.py` と同じ。''')
code(COMMON_CELL)

# ---------------------------------------------------------------- 2. データ
md(r'''## 2. 版の中身

全条件のファイルを読み、問題データ一覧の「除外」を除く。版ごとに、学習データと検証データの行数・条件数を出す。''')
code(r'''# セル2：読み込みと、版の中身
D_RAW = load_all(exclude=False)                          # 除外する前（外れ値の確認用）
D = load_all()                                           # 問題データ一覧の「除外」を除いた後（REMOVED は、こちらで除いた行数になる）
plist = read_problem_list()
print(plist[plist["区分"].isin(["除外", "感度確認"])][["ID", "区分", "材料", "組", "設定温度", "厚み", "射出速度", "ショット", "再測定"]].to_string(index=False))
leg = legacy_check(D)
judge(bool(leg["同じ行"].all()) or not USE_EXCLUSIONS, "一覧の「除外」を除いたデータが、no1.5-3.0 のファイル（P03 などの除外を同じように適用したもの）と同じ（取り直した後は違ってよい）")

TESTS_JA = {"S_V": "3data（外挿）", "S_T": "5data の 210 ℃", "S_T3": "3data の 210 ℃", "S_Vall": "3data（外挿）"}
vrows, SETS3D = [], {}
for ver in DATA_VERSIONS:
    for mat in MATS:
        vs = version_sets(mat, ver, MAIN_DATA, TEMP_HOLDOUT)
        SETS3D[(mat, ver)] = vs
        pool = vs["pool"]
        sh = pool.groupby("cond").size()
        vrows.append({"material": MATS[mat], "version": ver, "split": "ALL（最終モデル）", "n_train": len(pool), "n_cond_train": pool["cond"].nunique(), "shots_per_cond": f"{sh.min()}〜{sh.max()}",
                      "gd_range": f"{pool['gd'].min():.0f}〜{pool['gd'].max():.0f}", "tests": "―", "removed_by_list": REMOVED[(mat, "5data" if ver == "5data" else "Alldata", MAIN_DATA)]})
        for name, (tr, tests) in vs["splits"].items():
            if name.startswith("CV") and name != "CV1":
                continue
            vrows.append({"material": MATS[mat], "version": ver, "split": name if not name.startswith("CV") else f"CV（{CV_K} 分割。1 分割目の例）", "n_train": len(tr),
                          "n_cond_train": tr["cond"].nunique(), "shots_per_cond": "", "gd_range": "",
                          "tests": "；".join(f"{TESTS_JA.get(tn, '1 分割のテスト')}：{len(te)} 行 {te['cond'].nunique()} 条件" for tn, te in tests.items()), "removed_by_list": ""})
versions_df = pd.DataFrame(vrows)
versions_df.to_csv(os.path.join(OUT_DIR, "versions.csv"), index=False, encoding="utf-8-sig")
print(versions_df.to_string(index=False))
print("\nremoved_by_list：問題データ一覧の「除外」で、そのファイルから除いた行数")''')

# ---------------------------------------------------------------- 3. 包含関係
md(r'''## 3. ファイルの包含関係

5data（学習）と 3data（検証）は、Alldata の一部のはず（PP：各条件の最初の 5・3 ショット。粉砕材混合 PP：5data ＋ 3data がそのまま Alldata）。元ファイル名と、η・γ̇・P3_1・P3_2・Tff の値が一致する行で数える。**外れ値あり**では完全に一致するはず。**外れ値なし**は、ファイルごとに別々に外れ値を除いたので、一致しない行があってよい（その数を見る）。''')
code(r'''# セル3：包含関係
KEYV = ["file", "eta", "gd", "P31", "P32", "Tff"]


def row_keys(df):
    return df[KEYV].round(9).astype(str).agg("|".join, axis=1)


irows = []
for mat in MATS:
    for o in ["あり", "なし"]:
        ka = row_keys(D[(mat, "Alldata", o)])
        sa = set(ka)
        union = set()
        for g in ["5data", "3data"]:
            kg = row_keys(D[(mat, g, o)])
            union |= set(kg)
            irows.append({"material": MATS[mat], "outlier": o, "group": g, "n_rows": len(kg), "n_in_alldata": int(kg.isin(sa).sum()), "n_not_in_alldata": int((~kg.isin(sa)).sum())})
        irows.append({"material": MATS[mat], "outlier": o, "group": "Alldata（5data・3data にない行）", "n_rows": len(ka), "n_in_alldata": int(ka.isin(union).sum()), "n_not_in_alldata": int((~ka.isin(union)).sum())})
incl_df = pd.DataFrame(irows)
incl_df.to_csv(os.path.join(OUT_DIR, "inclusion.csv"), index=False, encoding="utf-8-sig")
print(incl_df.rename(columns={"n_in_alldata": "Alldata（または 5data・3data）に同じ行がある", "n_not_in_alldata": "ない"}).to_string(index=False))
print("\n最後の行（Alldata）は、「同じ行がある」＝ 5data・3data のどちらかにある行、「ない」＝ Alldata だけにある行")
for mat in MATS:
    s = incl_df[(incl_df["material"] == MATS[mat]) & (incl_df["outlier"] == "あり")]
    judge(bool((s[s["group"].isin(["5data", "3data"])]["n_not_in_alldata"] == 0).all()), f"{MATS[mat]}（外れ値あり）：5data・3data の全行が Alldata にある")
mx = incl_df[(incl_df["material"] == MATS["MIX"]) & (incl_df["outlier"] == "あり") & incl_df["group"].str.startswith("Alldata")]
judge(bool((mx["n_not_in_alldata"] == 0).all()), "粉砕材混合PP（外れ値あり）：Alldata は 5data ＋ 3data だけ（Alldata だけにある行がない）")
pn = incl_df[(incl_df["material"] == MATS["PP"]) & (incl_df["outlier"] == "なし") & incl_df["group"].isin(["5data", "3data"])]
print(f"（参考）PP の外れ値なしで、Alldata にない行：5data {int(pn[pn['group'] == '5data']['n_not_in_alldata'].iloc[0])} 行、3data {int(pn[pn['group'] == '3data']['n_not_in_alldata'].iloc[0])} 行。外れ値の除き方がファイルごとに違うため")''')

# ---------------------------------------------------------------- 4. ショットの順番によるずれ
md(r'''## 4. ショットの順番によるずれ

Alldata は 1 条件 10 ショット（PP）。ショットの順番が進むと、金型や樹脂の状態が変わって、粘度がずれていくことはないか。元ファイル名の先頭 3 桁の番号は、設定温度・厚みごとの通し番号（速度の条件ごとに、連続した番号が割り当てられている）なので、**条件の中での順番 `pos`**（最初のショット ＝ 1。外れ値で除かれたショットの番号は欠ける）を使う。**条件ごとの平均からの偏差**を、`pos` に回帰して、傾き（1 ショットあたりのずれ）を出す。

- 「ずれ（最初→最後）」＝ 傾き × （最大の `pos` − 最小の `pos`）。PP は 9 ショット分、粉砕材混合 PP は 4 ショット分。許容値 `DRIFT_TOL` 未満なら OK。
- 3 ショット以上ある条件だけを使う。PP の外れ値なしは、外れ値で除かれた行があるので、ショット数が条件で違う。
- あわせて、PP は「`pos` 6〜10 の平均 − `pos` 1〜5 の平均」を条件ごとに出して、条件にわたって平均する（`half_diff`。5data が「最初の 5 ショット」なので、後半の 5 ショットが加わると平均がどれだけずれるかにあたる）。
- 外れたからといって、外れ値として除くわけではない。Alldata 版の**注意点**として報告する。''')
code(r'''# セル4：ショットの順番によるずれ
DR_VARS = {"ln_eta": "ln η", "ln_gd": "ln γ̇", "ln_P31": "ln P3_1", "Tff": "Tff [℃]"}


def drift_stats(df, tag, var_cols=DR_VARS):
    """条件内の偏差（各ショット − 条件平均）を、ショットの順番 pos の偏差に回帰した傾き。3 ショット以上ある条件だけ。(行数, 条件数, {列: (傾き, 標準誤差)}, 順番の幅)"""
    d = df[df.groupby("cond")["pos"].transform("size") >= 3].copy()
    if len(d) == 0:
        return 0, 0, {}, 0
    ds = d["pos"] - d.groupby("cond")["pos"].transform("mean")
    n, n_c = len(d), d["cond"].nunique()
    sxx = float((ds ** 2).sum())
    out = {}
    for c in var_cols:
        dy = d[c] - d.groupby("cond")[c].transform("mean")
        b = float((ds * dy).sum() / sxx)
        out[c] = (b, float(np.sqrt(((dy - b * ds) ** 2).sum() / max(n - n_c - 1, 1) / sxx)))
    return n, n_c, out, int(d["pos"].max() - d["pos"].min())


def half_diff(df, c):
    """PP：条件ごとの「順番 6 以降の平均 − 順番 5 以前の平均」（両方がある条件だけ）の、条件にわたる平均と標準誤差"""
    a = df[df["pos"] <= 5].groupby("cond")[c].mean()
    b = df[df["pos"] >= 6].groupby("cond")[c].mean()
    x = (b - a).dropna()
    return (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), len(x)) if len(x) >= 3 else (np.nan, np.nan, len(x))


def speed_class(v):
    return np.where(v <= 5, "低速側（3・5 mm/s）", np.where(v >= 280, "高速側（280・320 mm/s）", "内側"))


drows = []
for mat in MATS:
    for o in ["なし", "あり"]:
        df = D[(mat, "Alldata", o)].copy()
        df["sp"] = speed_class(df["v"].to_numpy())
        groups = [("全体", "全体", df)] + [("設定温度", f"{int(t)} ℃", df[df["Tset"] == t]) for t in sorted(df["Tset"].unique())] \
            + [("厚み", f"{h} mm", df[np.isclose(df["H"], h)]) for h in sorted(df["H"].unique())] + [("射出速度", s, df[df["sp"] == s]) for s in ["低速側（3・5 mm/s）", "内側", "高速側（280・320 mm/s）"]]
        for gt, gl, sub in groups:
            n, n_c, st, span = drift_stats(sub, gl)
            for c, (b, se) in st.items():
                hd = half_diff(sub, c) if mat == "PP" else (np.nan, np.nan, 0)
                drows.append({"material": MATS[mat], "outlier": o, "group_type": gt, "group": gl, "variable": DR_VARS[c], "n_rows": n, "n_cond": n_c, "slope_per_shot": b, "se": se,
                              "drift_first_to_last": b * span, "z": b / se if se > 0 else np.nan, "half_diff": hd[0], "half_diff_se": hd[1], "half_diff_n_cond": hd[2]})
drift_df = pd.DataFrame(drows)
drift_df.to_csv(os.path.join(OUT_DIR, "shot_drift.csv"), index=False, encoding="utf-8-sig")
tot = drift_df[(drift_df["group"] == "全体") & drift_df["variable"].isin(["ln η", "ln γ̇"])]
print(tot[["material", "outlier", "variable", "n_cond", "slope_per_shot", "se", "drift_first_to_last", "z", "half_diff", "half_diff_se"]].round(4).to_string(index=False))
print("\n（ずれ＝最初のショット → 最後のショットの変化量。ln η で 0.02 ＝ 粘度の約 2%）")
for mat in MATS:
    s = drift_df[(drift_df["material"] == MATS[mat]) & (drift_df["outlier"] == MAIN_DATA) & (drift_df["variable"] == "ln η")]
    a = s[s["group"] == "全体"].iloc[0]
    judge(abs(a["drift_first_to_last"]) < DRIFT_TOL, f"{MATS[mat]}（外れ値{MAIN_DATA}）：ショットの順番によるずれ（ln η）= {a['drift_first_to_last']:+.4f}（許容 {DRIFT_TOL}。傾き {a['slope_per_shot']:+.5f} ± {a['se']:.5f} /ショット）")
    g = s[s["group"] != "全体"]
    worst = g.loc[g["drift_first_to_last"].abs().idxmax()]
    judge(abs(worst["drift_first_to_last"]) < 2 * DRIFT_TOL or abs(worst["z"]) < 2, f"{MATS[mat]}：群（設定温度・厚み・射出速度）ごとの最大のずれ = {worst['drift_first_to_last']:+.4f}（{worst['group_type']} {worst['group']}。傾き / 標準誤差 = {worst['z']:+.1f}）が、許容の 2 倍（{2 * DRIFT_TOL}）未満、または標準誤差の 2 倍以内（偶然の範囲）。外れたら、その群の注意点にする")''')
code(r'''# セル4.5：図1 ショットの順番ごとの、条件平均からの偏差（Alldata の外れ値なし）
fig, axes = plt.subplots(len(MATS), 2, figsize=(12, 4.2 * len(MATS)), squeeze=False)
for i, mat in enumerate(MATS):
    df = D[(mat, "Alldata", MAIN_DATA)]
    df = df[df.groupby("cond")["pos"].transform("size") >= 3]
    for j, (c, lab) in enumerate([("ln_eta", "ln η"), ("ln_gd", r"ln $\dot{\gamma}$")]):
        ax = axes[i, j]
        dev = df[c] - df.groupby("cond")[c].transform("mean")
        shots = sorted(df["pos"].unique())
        data = [dev[df["pos"] == s].to_numpy() for s in shots]
        ax.boxplot(data, positions=shots, widths=0.6, showfliers=True, flierprops=dict(markersize=2, alpha=0.4))
        ax.plot(shots, [d.mean() for d in data], "o-", color="tab:red", label="平均")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("条件の中でのショットの順番（1 ＝ 最初）")
        ax.set_ylabel(f"{lab} の、条件平均からの偏差")
        ax.set_title(f"{MATS[mat]}（外れ値{MAIN_DATA}）：{lab}")
        ax.legend()
fig.suptitle("図1　ショットの順番ごとの偏差（条件ごとの平均を引いたもの）。順番によるずれがなければ、箱は 0 のまわりに並ぶ", y=1.01)
fig.tight_layout()
print(save_fig(fig, OUT_DIR, "fig1_shot_drift"))
plt.show()''')

# ---------------------------------------------------------------- 5. 外れ値の検知
md(r'''## 5. 外れ値の検知（規則 f）と、問題の条件

**規則 f**：条件ごとの平均（ln η と ln γ̇ をショットで平均した点）を、GPR（入力 ln γ̇（条件平均）・ln H・Tset。R3 と同じ形）で、**条件を 1 つずつ抜いて予測**する。実測 − 予測（残差）を予測標準偏差で割った **z** が `OUTLIER_Z` を超え、かつ |残差| が `OUTLIER_RESID`（ln η で 0.1 ＝ 粘度で約 10%）を超える条件を、候補にする。

- 拾った条件は、問題データ一覧（`02_機械学習データ/問題データ一覧.csv`）に区分「除外」で記録して除いてよい（ユーザー、2026-10-14）。拾った条件が 4 つ以上のときは、登録の前に相談する。
- 「全条件」は除外する前のデータ、「除外後」は一覧の「除外」を除いた後（除いた条件の影に隠れていた外れ値を見つけるため）。
- Alldata の外れ値なしで確認する（5data・3data は Alldata の一部）。''')
code(r'''# セル5：規則 f
def reg_id(mat, grp, r):
    """この条件が、問題データ一覧の「除外」「感度確認」に登録されているか（ID）"""
    one = pd.DataFrame({"Tset": [float(r.Tset)], "H": [float(r.H)], "v": [float(r.v)], "shot": [np.nan]})
    _, ids = problem_mask(one, mat, grp, plist, ("除外", "感度確認"))
    return ids[0]


srows = []
for mat in MATS:
    for label, src in (("全条件", D_RAW), ("除外後", D)):
        df = src[(mat, "Alldata", MAIN_DATA)]
        t = outlier_scan(df)
        t = t.merge(cond_means(df)[["cond", "ln_gd", "ln_eta"]], on="cond")
        t.insert(0, "material", MATS[mat])
        t.insert(1, "data", label)
        t["list_id"] = [reg_id(mat, "Alldata", r) for r in t.itertuples()]
        srows.append(t)
scan_df = pd.concat(srows, ignore_index=True)
scan_df.to_csv(os.path.join(OUT_DIR, "outlier_check.csv"), index=False, encoding="utf-8-sig")
for mat in MATS:
    for label in ["全条件", "除外後"]:
        s = scan_df[(scan_df["material"] == MATS[mat]) & (scan_df["data"] == label)]
        print(f"{MATS[mat]}（{label}）：{len(s)} 条件、|z| の最大 {s['z'].abs().max():.2f}、拾った条件 {int(s['flag'].sum())} 件")
        if s["flag"].any():
            print(s[s["flag"]][["cond", "n", "resid", "sd", "z", "list_id"]].round(3).to_string(index=False))
        top = s.reindex(s["z"].abs().sort_values(ascending=False).index).head(3)
        print("  z の大きい順（参考）：" + "、".join(f"{r.cond}（z={r.z:+.1f}、残差 {r.resid:+.3f}）" for r in top.itertuples()))
for mat in MATS:
    s = scan_df[(scan_df["material"] == MATS[mat]) & scan_df["flag"]]
    a = s[s["data"] == "全条件"]
    judge(bool((a["list_id"] != "").all()), f"{MATS[mat]}：規則 f で拾った {len(a)} 条件は、すべて問題データ一覧に登録済み（未登録なら、一覧に「除外」で記録する。4 件以上なら相談）")
    b = s[s["data"] == "除外後"]
    judge(len(b) == 0, f"{MATS[mat]}：一覧の「除外」を除いた後に、規則 f で拾う条件が 0 件（残っていれば、新しい外れ値の候補）")''')
code(r'''# セル5.5：図2 規則 f の結果（全条件。条件ごとの残差と z）
fig, axes = plt.subplots(len(MATS), 2, figsize=(13, 4.6 * len(MATS)), squeeze=False)
TC = {190: "tab:blue", 210: "tab:green", 230: "tab:red"}
for i, mat in enumerate(MATS):
    s = scan_df[(scan_df["material"] == MATS[mat]) & (scan_df["data"] == "全条件")]
    for j, (col, thr, lab) in enumerate([("resid", OUTLIER_RESID, "残差 ln η（実測 − 予測）"), ("z", OUTLIER_Z, "z（残差 / 予測標準偏差）")]):
        ax = axes[i, j]
        for t, g in s.groupby("Tset"):
            ax.scatter(np.exp(g["ln_gd"]), g[col], s=18, color=TC[int(t)], alpha=0.75, label=f"{int(t)} ℃")
        f = s[s["flag"]]
        ax.scatter(np.exp(f["ln_gd"]), f[col], s=160, facecolors="none", edgecolors="k", linewidths=1.8, label="規則 f で拾った条件")
        for r in f.itertuples():
            ax.annotate(f"{r.list_id or '未登録'}（{r.cond}）", (np.exp(r.ln_gd), getattr(r, col)), textcoords="offset points", xytext=(8, 4), fontsize=8)
        ax.axhline(thr, color="gray", ls="--", lw=0.8)
        ax.axhline(-thr, color="gray", ls="--", lw=0.8)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel(r"条件平均のせん断速度 $\dot{\gamma}$ (1/s)")
        ax.set_ylabel(lab)
        ax.set_title(f"{MATS[mat]}：{lab}")
        ax.legend(fontsize=8)
fig.suptitle("図2　規則 f（条件を 1 つずつ抜いた予測との違い）。点線がしきい値", y=1.01)
fig.tight_layout()
print(save_fig(fig, OUT_DIR, "fig2_outlier_scan"))
plt.show()''')
code(r'''# セル5.6：図3 問題の条件の、外れ値ありの全ショット（● 外れ値なしにも残った行、× 外れ値として除かれた行）
PANELS = [("PP", 320, "P03：PP 0.5 mm・320 mm/s"), ("PP", 20, "P02：PP 0.5 mm・20 mm/s"), ("MIX", 320, "粉砕材混合PP 0.5 mm・320 mm/s（P08 の重複行を含む）"), ("MIX", 20, "P02：粉砕材混合PP 0.5 mm・20 mm/s")]
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
prow = []
for ax, (mat, v0, ttl) in zip(axes.ravel(), PANELS):
    raw = D_RAW[(mat, "Alldata", "あり")]
    kept_files = set(D_RAW[(mat, "Alldata", "なし")]["file"])
    sub = raw[np.isclose(raw["H"], 0.5) & (raw["v"] == v0)]
    for t, g in sub.groupby("Tset"):
        k = g["file"].isin(kept_files).to_numpy()
        ax.scatter(g["gd"][k], g["eta"][k], s=60, color=TC[int(t)], edgecolors="k", linewidths=0.5, label=f"{int(t)} ℃：残った {int(k.sum())} 行")
        ax.scatter(g["gd"][~k], g["eta"][~k], s=60, color=TC[int(t)], marker="x", linewidths=1.6, label=f"{int(t)} ℃：除かれた {int((~k).sum())} 行")
        prow.append({"material": MATS[mat], "v": v0, "Tset": int(t), "n_shots_all": len(g), "n_kept_in_nashi": int(k.sum()), "gd_mean_kept": float(np.exp(np.log(g["gd"][k]).mean())) if k.any() else np.nan,
                     "eta_mean_kept": float(np.exp(np.log(g["eta"][k]).mean())) if k.any() else np.nan})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ (1/s)")
    ax.set_ylabel("粘度 η (Pa·s)")
    ax.set_title(ttl)
    ax.legend(fontsize=7)
fig.suptitle("図3　問題の条件の全ショット（外れ値あり）。● は外れ値なしにも残った行、× は外れ値として除かれた行", y=1.0)
fig.tight_layout()
print(save_fig(fig, OUT_DIR, "fig3_problem_conditions"))
plt.show()
pd.DataFrame(prow).to_csv(os.path.join(OUT_DIR, "problem_conditions.csv"), index=False, encoding="utf-8-sig")
print(pd.DataFrame(prow).round(3).to_string(index=False))''')

# ---------------------------------------------------------------- 6. 学習と評価
md(r'''## 6. 学習と評価（R3 で、2 つの版を比べる）

モデルは R3（入力：条件平均の ln γ̇、ln H、Tset）。ARD Matern 5/2 ＋ 線形カーネル ＋ ノイズ（下限 1e-6）。標準化と中心化は学習データだけで計算する。

| 分割 | 5data 版 | Alldata 版 |
|---|---|---|
| S_V | 5data → 3data（低速側 3・5 mm/s、高速側 280・320 mm/s の外挿） | ― |
| S_Vall | ― | Alldata のうち 5data の条件（全ショット）→ 3data。PP では、同じ 93 条件のショットが増えると、外挿が良くなるかを見る（粉砕材混合 PP は 5data と同じデータなので、S_V と同じになる） |
| S_T | 5data の 190・230 ℃ → 5data の 210 ℃、S_T3：3data の 210 ℃ | Alldata の 190・230 ℃ → **同じテスト行**（5data ファイルの 210 ℃、3data ファイルの 210 ℃） |
| CV | 5data の中の条件単位の 10 分割 | Alldata の条件単位の 10 分割 |

- 並列に計算する（`N_JOBS`）。PP の Alldata は 1 回の学習が 1〜3 分かかる。
- CV は、各分割の予測をまとめて指標を出す（全体と、低速側・内側・高速側）。''')
code(r'''# セル6：学習（並列）
MODELS["R3"] = {"kind": "gpr", "cols": ["ln_gd_cm", "ln_H", "Tset"], "linear": True}
MODEL_JA = r"R3：［$\dot{\gamma}$（条件平均）, $H$, $T_{set}$］"
VER_JA = {"5data": "5data 版：5data で学習", "Alldata": "Alldata 版：全条件で学習"}


def job(mat, ver, name, tr, tests, restarts):
    warnings.simplefilter("ignore")
    t0 = time.time()
    fit = fit_model("R3", tr, restarts)
    return mat, ver, name, fit, {tn: predict_model("R3", fit, te)[:3] for tn, te in tests.items()}, time.time() - t0


JOBS = []
for ver in DATA_VERSIONS:
    for mat in MATS:
        vs = SETS3D[(mat, ver)]
        for name, (tr, tests) in vs["splits"].items():
            JOBS.append((mat, ver, name, tr, tests, N_RESTARTS_CV if name.startswith("CV") else N_RESTARTS))
        if ver == "Alldata":                                   # 最終モデル（全条件）。5data 版の最終モデルは S_V と同じ学習
            JOBS.append((mat, ver, "ALL", vs["pool"], {}, N_RESTARTS))
JOBS.sort(key=lambda j: -len(j[3]))                            # 重い仕事から先に
t0 = time.time()
RES = Parallel(n_jobs=N_JOBS)(delayed(job)(*j) for j in JOBS)
print(f"{len(JOBS)} 回の学習を並列（{N_JOBS}）で計算：{(time.time() - t0) / 60:.1f} 分（学習 1 回の最長 {max(r[5] for r in RES):.0f} 秒）")''')
code(r'''# セル7：指標の計算
FIT3D, PRED3D, KROWS, cv_acc = {}, {}, [], {}
for (mat, ver, name, fit, preds, sec), j in zip(RES, JOBS):
    tr, tests = j[3], j[4]
    FIT3D[(mat, ver, name)] = fit
    KROWS.append({"material": MATS[mat], "version": ver, "split": name, "n_train": len(tr), "n_cond_train": tr["cond"].nunique(), **kernel_row("R3", fit),
                  "pure_sd_ln_eta": within_stats(tr)["sd_ln_eta"], "fit_sec": sec})
    for tn, te in tests.items():
        mu, sd, sdf = preds[tn]
        df = pd.DataFrame({"y": te["ln_eta"].to_numpy(), "mu": mu, "sd": sd, "v": te["v"].to_numpy(), "H": te["H"].to_numpy(), "Tset": te["Tset"].to_numpy(),
                           "cond": te["cond"].to_numpy(), "group": te["group"].to_numpy()})
        if name.startswith("CV"):
            cv_acc.setdefault((mat, ver), []).append(df)
        else:
            PRED3D[(mat, ver, tn)] = df
for (mat, ver), parts in cv_acc.items():
    d = pd.concat(parts, ignore_index=True)
    d["group"] = np.where(d["v"] <= 5, "低速側", np.where(d["v"] >= 280, "高速側", "内側"))
    PRED3D[(mat, ver, "CV")] = d
for mat in MATS:                                                # 5data 版の最終モデルは S_V と同じ学習
    if (mat, "5data", "S_V") in FIT3D:
        FIT3D[(mat, "5data", "ALL")] = FIT3D[(mat, "5data", "S_V")]
kern_df = pd.DataFrame(KROWS)
kern_df["_a"] = kern_df["split"].str.startswith("CV").astype(int)
kern_df["_b"] = kern_df["split"].str.extract(r"(\d+)")[0].fillna(0).astype(int)
kern_df = kern_df.sort_values(["material", "version", "_a", "_b", "split"]).drop(columns=["_a", "_b"]).reset_index(drop=True)
kern_df.to_csv(os.path.join(OUT_DIR, "kernels.csv"), index=False, encoding="utf-8-sig")


def noise_of(mat, ver, split):
    """その分割の学習のノイズの標準偏差と学習の行数（CV は分割の平均。S_T3 は S_T と同じ学習）"""
    k = kern_df[(kern_df["material"] == MATS[mat]) & (kern_df["version"] == ver)]
    s = k[k["split"].str.fullmatch(r"CV\d+")] if split == "CV" else k[k["split"] == {"S_T3": "S_T"}.get(split, split)]
    return float(s["noise_sd"].mean()), float(s["n_train"].mean())


mrows = []
for (mat, ver, split), d in PRED3D.items():
    nz, ntr = noise_of(mat, ver, split)
    subsets = {"全体": np.ones(len(d), bool)}
    for g in d["group"].unique():
        if g != "全体":
            subsets[g] = (d["group"] == g).to_numpy()
    for sub, m in subsets.items():
        if m.sum() < 2:
            continue
        ev = evaluate(d["y"].to_numpy()[m], d["mu"].to_numpy()[m], d["sd"].to_numpy()[m])
        mrows.append({"material": MATS[mat], "version": ver, "split": split, "subset": sub, "n_train": ntr, "n_test": int(m.sum()), "n_cond_test": int(d.loc[m, "cond"].nunique()), **ev, "noise_sd": nz})
met_df = pd.DataFrame(mrows)
ORDER = ["S_V", "S_Vall", "S_T", "S_T3", "CV"]
met_df["_o"] = met_df["split"].map({k: i for i, k in enumerate(ORDER)})
met_df = met_df.sort_values(["material", "_o", "subset", "version"]).drop(columns="_o").reset_index(drop=True)
met_df.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False, encoding="utf-8-sig")
view = met_df[["material", "version", "split", "subset", "n_test", "n_cond_test", "RMSE", "coverage95", "NLPD", "mean_sd", "noise_sd"]].round(3)
print(view.to_string(index=False))
print("\nn_train は、CV では分割の平均。RMSE・mean_sd・noise_sd は ln η の単位")''')
code(r'''# セル8：判定（2 つの版の比較）
def mv(mat, ver, split, sub, col="RMSE"):
    r = met_df[(met_df["material"] == MATS[mat]) & (met_df["version"] == ver) & (met_df["split"] == split) & (met_df["subset"] == sub)]
    return float(r[col].iloc[0]) if len(r) else np.nan


have_both = "5data" in DATA_VERSIONS and "Alldata" in DATA_VERSIONS
for mat in MATS:
    n = MATS[mat]
    if have_both:
        for sp in ["S_T", "S_T3"]:
            a, b = mv(mat, "Alldata", sp, "全体"), mv(mat, "5data", sp, "全体")
            judge(a <= 1.05 * b, f"{n}：{sp}（同じテスト行）で、Alldata 版の RMSE {a:.3f} が 5data 版 {b:.3f} の 1.05 倍以内（悪くならない）")
        for sub in ["低速側", "高速側"]:
            a, b = mv(mat, "Alldata", "S_Vall", sub), mv(mat, "5data", "S_V", sub)
            judge(a <= 1.05 * b, f"{n}：外挿（{sub}）で、Alldata のうち 5data の条件だけで学習した S_Vall の RMSE {a:.3f} が S_V {b:.3f} の 1.05 倍以内（ショットが増えても悪くならない。{'粉砕材混合PP は 5data と同じデータなので同じ値になる' if mat == 'MIX' else 'PP は 1 条件のショットが増える'}）")
    for ver in DATA_VERSIONS:
        for sp, sub in [("S_V", "低速側"), ("S_V", "高速側"), ("S_T", "全体"), ("CV", "全体")]:
            c = mv(mat, ver, sp, sub, "coverage95")
            if not np.isnan(c):
                judge(0.90 <= c <= 0.99, f"{n}・{ver} 版：{sp} {sub} の 95% 予測区間の被覆率 = {c:.2f}（0.90〜0.99 なら、区間の幅が適切。低いと狭すぎる）")
    if "Alldata" in DATA_VERSIONS:
        k = kern_df[(kern_df["material"] == n) & (kern_df["version"] == "Alldata") & (kern_df["split"] == "ALL")].iloc[0]
        ratio = k["noise_sd"] / k["pure_sd_ln_eta"]
        judge(0.8 <= ratio <= 1.25, f"{n}・Alldata 版：最終モデルのノイズの標準偏差 {k['noise_sd']:.3f} が、同じ条件内の ln η の標準偏差（純粋誤差）{k['pure_sd_ln_eta']:.3f} の 0.8〜1.25 倍（{ratio:.2f} 倍。成形条件から見たばらつきとして整合している）")''')

# ---------------------------------------------------------------- 7. 図
md(r'''## 7. 図

- 図4：分割ごとの RMSE と NLPD（版を色分け。小さいほどよい）。
- 図5：予測と実測の ln η（パリティ図）。5data 版は S_V（外挿）、Alldata 版は CV（条件単位の 10 分割）。''')
code(r'''# セル9：図4 版ごとの RMSE・NLPD
CATS = [("S_V", "低速側"), ("S_V", "高速側"), ("S_Vall", "低速側"), ("S_Vall", "高速側"), ("S_T", "全体"), ("S_T3", "全体"), ("CV", "全体"), ("CV", "低速側"), ("CV", "内側"), ("CV", "高速側")]
VCOL = {"5data": "tab:blue", "Alldata": "tab:orange"}
fig, axes = plt.subplots(len(MATS), 2, figsize=(14, 4.6 * len(MATS)), squeeze=False)
for i, mat in enumerate(MATS):
    for j, metric in enumerate(["RMSE", "NLPD"]):
        ax = axes[i, j]
        w = 0.38
        for k, ver in enumerate(DATA_VERSIONS):
            vals = [mv(mat, ver, sp, sub, metric) for sp, sub in CATS]
            ax.bar(np.arange(len(CATS)) + (k - (len(DATA_VERSIONS) - 1) / 2) * w, vals, w, color=VCOL[ver], label=VER_JA[ver])
        ax.set_xticks(np.arange(len(CATS)))
        ax.set_xticklabels([f"{sp}\n{sub}" for sp, sub in CATS], fontsize=8)
        ax.set_ylabel(metric + (" (ln η)" if metric == "RMSE" else "（小さいほどよい）"))
        ax.set_title(f"{MATS[mat]}：{metric}")
        ax.axhline(0, color="gray", lw=0.6)
        ax.legend(fontsize=8)
fig.suptitle("図4　分割ごとの RMSE・NLPD。棒がない所は、その版にない分割", y=1.01)
fig.tight_layout()
print(save_fig(fig, OUT_DIR, "fig4_version_metrics"))
plt.show()''')
code(r'''# セル10：図5 パリティ図（予測と実測の ln η）
fig, axes = plt.subplots(len(MATS), len(DATA_VERSIONS), figsize=(6.2 * len(DATA_VERSIONS), 5.6 * len(MATS)), squeeze=False)
for i, mat in enumerate(MATS):
    for j, ver in enumerate(DATA_VERSIONS):
        ax = axes[i, j]
        key = (mat, ver, "S_V" if ver == "5data" else "CV")
        d = PRED3D[key]
        for g, c in [("低速側", "tab:blue"), ("内側", "gray"), ("高速側", "tab:red")]:
            m = (d["group"] == g).to_numpy()
            if m.any():
                ax.errorbar(d["y"][m], d["mu"][m], yerr=1.96 * d["sd"][m], fmt="o", ms=3, color=c, alpha=0.5, elinewidth=0.6, label=g)
        lim = [d["y"].min() - 0.2, d["y"].max() + 0.2]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel("実測 ln η")
        ax.set_ylabel("予測 ln η（棒は 95% 予測区間）")
        r = met_df[(met_df["material"] == MATS[mat]) & (met_df["version"] == ver) & (met_df["split"] == key[2]) & (met_df["subset"] == "全体")].iloc[0]
        ax.set_title(f"{MATS[mat]}・{VER_JA[ver]}\n{key[2]}：RMSE {r['RMSE']:.3f}、被覆率 {r['coverage95']:.2f}", fontsize=10)
        ax.legend(fontsize=8)
fig.suptitle("図5　予測と実測（5data 版は S_V の外挿、Alldata 版は CV）", y=1.01)
fig.tight_layout()
print(save_fig(fig, OUT_DIR, "fig5_parity"))
plt.show()''')
code(r'''# セル11：版のまとめ（summary.csv）
srows2 = []
for ver in DATA_VERSIONS:
    for mat in MATS:
        pool = SETS3D[(mat, ver)]["pool"]
        k = kern_df[(kern_df["material"] == MATS[mat]) & (kern_df["version"] == ver) & (kern_df["split"] == ("S_V" if ver == "5data" else "ALL"))].iloc[0]
        srows2.append({"material": MATS[mat], "version": ver, "n_rows": len(pool), "n_cond": pool["cond"].nunique(), "shots_per_cond": f"{pool.groupby('cond').size().min()}〜{pool.groupby('cond').size().max()}",
                       "RMSE_S_V_low": mv(mat, ver, "S_V", "低速側"), "RMSE_S_V_high": mv(mat, ver, "S_V", "高速側"), "RMSE_S_Vall_low": mv(mat, ver, "S_Vall", "低速側"), "RMSE_S_Vall_high": mv(mat, ver, "S_Vall", "高速側"),
                       "RMSE_S_T": mv(mat, ver, "S_T", "全体"), "RMSE_S_T3": mv(mat, ver, "S_T3", "全体"), "RMSE_CV": mv(mat, ver, "CV", "全体"), "coverage_CV": mv(mat, ver, "CV", "全体", "coverage95"),
                       "noise_sd_final": float(k["noise_sd"]), "pure_sd_ln_eta": float(k["pure_sd_ln_eta"]), "LML_final": float(k["LML"])})
summary_df = pd.DataFrame(srows2)
summary_df.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False, encoding="utf-8-sig")
print(summary_df.round(3).to_string(index=False))''')

# ---------------------------------------------------------------- 8. 高解像度のグラフ
md(HR_MD.replace("## 高解像度のグラフ", "## 8. 高解像度のグラフ"))
code(hr_cell("# セル12：高解像度のグラフ（せん断速度 ― 粘度。GPR6月ノートのセル9-2 にならう）",
             '''HR_VERSIONS = ["5data", "Alldata"]  # 描く版（"5data"：5data で学習〔3data は白抜きの検証の点〕／"Alldata"：全条件で学習〔全部が学習の点〕）''',
             r'''hr_saved = []
for mat in HR_MATERIALS:
    for ver in HR_VERSIONS:
        if (mat, ver, "ALL") not in FIT3D:
            print(f"{MATS[mat]}・{ver} 版は、学習していないので描かない（DATA_VERSIONS に入れて、上から実行し直す）")
            continue
        t0 = time.time()
        fit = FIT3D[(mat, ver, "ALL")]
        pool = SETS3D[(mat, ver)]["pool"] if ver == "Alldata" else pool_of(mat, MAIN_DATA)        # 点に使うデータ（5data 版は、検証の 3data も重ねる）
        hr_saved += hr_flow(lambda Xg, f=fit: predict_model("R3", f, Xg)[:3], mat, MAIN_DATA, set(SETS3D[(mat, ver)]["pool"]["cond"]),
                            f"{MODEL_JA}（{VER_JA[ver]}）", f"hr_{mat}_R3_{ver}", HR_DIR, pool=pool, **HR_KW)
        print(f"{MATS[mat]}・{ver} 版：保存したファイル 累計 {len(hr_saved)} 個（{time.time() - t0:.0f} 秒）")
judge(len(hr_saved) > 0, f"高解像度のグラフを {len(hr_saved)} 個保存した（{os.path.normpath(HR_DIR)}）。形式 {HR_FORMATS}、{HR_DPI} dpi")'''))

# ---------------------------------------------------------------- 9. 保存したファイル
code(r'''# セル13：保存したファイル
print("保存したファイル:")
for f in sorted(os.listdir(OUT_DIR)):
    print("  ", os.path.normpath(os.path.join(OUT_DIR, f)))''')


# ---------------------------------------------------------------- 書き出し
def to_lines(s):
    lines = s.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


rnd = random.Random(20261014)
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
