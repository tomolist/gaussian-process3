# -*- coding: utf-8 -*-
"""手順3b（新データ：設定温度の比較）のノートブック (未実行) を生成する。

使い方: python build_step3b_nb.py <出力する .ipynb のパス> [--force]
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
md(r'''# 手順3b：設定温度の比較（新データ）

CLAUDE.md §7 の手順3b。粘度の予測に、**加熱シリンダ設定温度 Tset** を入力として足すと、予測がどう変わるかを確かめる。あわせて、**流動先端温度 Tff** と**圧力 P3_2** が、本当に要らないかを確かめる（ユーザーの決定、2026-10-12：どちらも「あり」と「なし」の両方で比べ、グラフを目で見て判断する）。

| 記号 | モデル | 入力（ln は自然対数。Tset は ℃ のまま） |
|---|---|---|
| L0／L0T | 線形回帰（基準） | γ̇, H ／ ＋Tset |
| B0／B0T | GPR | γ̇, H ／ ＋Tset |
| B0F／B0FT | GPR（Tff あり） | γ̇, H, Tff ／ ＋Tset |
| B2／B2T | GPR（P3_2 あり） | γ̇, H, P3_2 ／ ＋Tset |
| B2F／B2FT | GPR（Tff・P3_2 あり） | γ̇, H, Tff, P3_2 ／ ＋Tset |
| R0 | GPR（条件ごとの値） | 条件平均の γ̇, H |
| R2 | GPR（設定値だけ） | 射出速度 v, H, Tset |
| R3 | GPR（条件ごとの値） | 条件平均の γ̇, H, Tset |
| R3F | GPR（条件ごとの値、Tff あり） | 条件平均の γ̇, H, Tset, Tff（条件平均） |
| R3P | GPR（条件ごとの値、P3_2 あり） | 条件平均の γ̇, H, Tset, P3_2（条件平均） |

- GPR は、すべて **ARD Matern 5/2 ＋ 線形カーネル（DotProduct）＋ ノイズ**（低速側への外挿のため。ユーザーの決定）。ノイズ分散の下限は 1e-6。標準化と中心化は学習データだけで計算する。圧力は Pa に直して ln をとる。
- **ショットごとの γ̇ を入力にする B 系**と、**条件ごとの平均を入力にする R 系**の 2 通りを並べる。ショットごとの γ̇ を入れると、同じ条件内の γ̇ のゆれ（η と連動している）を GPR が使えてしまい、ノイズ分散が極端に小さくなる（5e-5 ほど）。R 系は、成形条件から見たばらつきを調べるための形で、ノイズ分散は純粋誤差に近くなる（3a の発見）。R3 が、CLAUDE.md §6 の「比較用の通常の GPR（R1）」にあたる。R2 は、成形条件（事前に決まる射出速度）だけを使う。
- Tff と P3_2 の「あり・なし」は、B 系（B0T／B0FT／B2T／B2FT）と R 系（R3／R3F／R3P）の両方で比べる。
- データは、外れ値なしが主。PP は、問題データ一覧の「除外」（1.5 mm・3 mm/s）を除いた全条件のファイル。

| 記号 | 分割 | 備考 |
|---|---|---|
| S_V | 5data で学習 → 3data で検証 | 低速側（3・5 mm/s）と高速側（280・320 mm/s）に分けて評価 |
| S_T | 5data の 190・230 ℃ で学習 → 5data の 210 ℃ をテスト | 温度の内挿の検証。3data の 210 ℃（速度の外挿も重なる）は S_T3 として別に評価 |
| S_R1・S_R2 | ランダム 20%（Alldata／PP の 5data＋3data） | 従来の評価方法。評価の甘さを見る |
| LOCO | 条件を1つずつ抜く交差検証 | ハイパーパラメータは全データで固定して、閉じた式で計算（近似） |

- **感度確認**：問題データ一覧の P02（190 ℃・0.5 mm・20 mm/s）と P03（PP の 190 ℃・0.5 mm・320 mm/s）を除いたときに、結果がどれだけ変わるかを見る。影響が大きいときは、その条件だけ除いたデータファイルを新しく作る（ユーザーの決定）。外れ値ありのデータでも確かめる。''')

# ---------------------------------------------------------------- 1. 設定
md(usage_md("3b", "results/step3b", "results/学習結果説明_手順3b.pdf", 25))
md(r'''## 1. 設定''')
code(settings_cell('''MAIN_DATA = "なし"            # 主に使う外れ値の扱い（"なし"：外れ値を除いたファイル／"あり"）
TEMP_HOLDOUT = 210            # S_T でテストにする設定温度 [℃]（学習は、5data の残りの温度）
RUN_SENS = True               # 感度確認（外れ値あり、感度確認の条件を学習から除く）を行うか
'''))
code(r'''# セル1.5：ライブラリ、フォント、出力先
%matplotlib inline
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

OUT_DIR = os.path.join(REPO, "results", "step3b")
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

データの読み込み、問題データ一覧の「除外」、モデル、評価、データの分け方、条件を1つずつ抜く交差検証の関数。中身は `04_補助スクリプト/nb_common.py` と同じ。''')
code(COMMON_CELL)

# ---------------------------------------------------------------- 2. データ
md(r'''## 2. データの読み込み

全条件のファイルを読み、問題データ一覧の「除外」を除く。「感度確認」の条件は残し、`sens` 列に ID を付ける。''')
code(r'''# セル2：読み込み
D = load_all()
pl = read_problem_list()
print(pl[pl["区分"].isin(["除外", "感度確認"])][["ID", "区分", "材料", "設定温度", "厚み", "射出速度", "再測定", "今の扱い"]].to_string(index=False))
leg = legacy_check(D)
judge(bool(leg["同じ行"].all()) or not USE_EXCLUSIONS, "一覧の「除外」を除いたデータが、これまでの no1.5-3.0 のファイルと同じ（取り直した後は違ってよい）")
for mat in MATS:
    tr, te = D[(mat, "5data", MAIN_DATA)], D[(mat, "3data", MAIN_DATA)]
    print(f"{MATS[mat]}：学習（5data）{len(tr)} 行 {tr['cond'].nunique()} 条件、検証（3data）{len(te)} 行 {te['cond'].nunique()} 条件、"
          f"感度確認の条件（学習 {int(tr['is_sens'].sum())} 行、検証 {int(te['is_sens'].sum())} 行）")''')

# ---------------------------------------------------------------- 3. モデル
md(r'''## 3. モデルの定義

入力の列：`ln_gd`（ln γ̇）、`ln_H`、`ln_Tff`（流動先端温度 [K]）、`ln_P32`（P3_2 [Pa]）、`Tset`（設定温度 [℃]）、`ln_v`（射出速度）、`ln_gd_cm`（条件ごとの ln γ̇ の平均）。''')
code(r'''# セル3：モデルの定義
def spec(kind, cols, linear=True):
    return {"kind": kind, "cols": cols, "linear": linear}


BASE = ["ln_gd", "ln_H"]
MODELS.update({
    "L0": spec("ols", BASE, False), "L0T": spec("ols", BASE + ["Tset"], False),
    "B0": spec("gpr", BASE), "B0T": spec("gpr", BASE + ["Tset"]),
    "B0F": spec("gpr", BASE + ["ln_Tff"]), "B0FT": spec("gpr", BASE + ["ln_Tff", "Tset"]),
    "B2": spec("gpr", BASE + ["ln_P32"]), "B2T": spec("gpr", BASE + ["ln_P32", "Tset"]),
    "B2F": spec("gpr", BASE + ["ln_Tff", "ln_P32"]), "B2FT": spec("gpr", BASE + ["ln_Tff", "ln_P32", "Tset"]),
    "R0": spec("gpr", ["ln_gd_cm", "ln_H"]), "R2": spec("gpr", ["ln_v", "ln_H", "Tset"]), "R3": spec("gpr", ["ln_gd_cm", "ln_H", "Tset"]),
    "R3F": spec("gpr", ["ln_gd_cm", "ln_H", "Tset", "ln_Tff_cm"]), "R3P": spec("gpr", ["ln_gd_cm", "ln_H", "Tset", "ln_P32_cm"]),
})
ALL_KEYS = list(MODELS)
GPR_KEYS = [k for k in ALL_KEYS if MODELS[k]["kind"] == "gpr"]
TSET_KEYS = [k for k in ALL_KEYS if "Tset" in MODELS[k]["cols"]]
KEYS_RANDOM = ["L0", "L0T", "B0", "B0T", "B0FT", "B2T", "R0", "R2", "R3"]            # ランダム分割は、計算が重いので主なモデルだけ
KEYS_LOCO = ["L0", "L0T", "B0", "B0T", "B0F", "B0FT", "B2T", "R0", "R2", "R3", "R3F", "R3P"]
KEYS_SENS = ["L0T", "B0T", "B2T", "R3", "R3F", "R3P"]
LABELS = {
    "L0": r"L0：線形［$\dot{\gamma}$, $H$］", "L0T": r"L0T：線形［$\dot{\gamma}$, $H$, $T_{set}$］",
    "B0": r"B0：［$\dot{\gamma}$, $H$］", "B0T": r"B0T：［$\dot{\gamma}$, $H$, $T_{set}$］",
    "B0F": r"B0F：［$\dot{\gamma}$, $H$, $T_{ff}$］", "B0FT": r"B0FT：［$\dot{\gamma}$, $H$, $T_{ff}$, $T_{set}$］",
    "B2": r"B2：［$\dot{\gamma}$, $H$, P3_2］", "B2T": r"B2T：［$\dot{\gamma}$, $H$, P3_2, $T_{set}$］",
    "B2F": r"B2F：［$\dot{\gamma}$, $H$, $T_{ff}$, P3_2］", "B2FT": r"B2FT：［$\dot{\gamma}$, $H$, $T_{ff}$, P3_2, $T_{set}$］",
    "R0": r"R0：［$\dot{\gamma}$（条件平均）, $H$］", "R2": r"R2：［$v$, $H$, $T_{set}$］", "R3": r"R3：［$\dot{\gamma}$（条件平均）, $H$, $T_{set}$］",
    "R3F": r"R3F：［$\dot{\gamma}$（条件平均）, $H$, $T_{set}$, $T_{ff}$（条件平均）］", "R3P": r"R3P：［$\dot{\gamma}$（条件平均）, $H$, $T_{set}$, P3_2（条件平均）］",
}
INPUT_JA = {"ln_gd": r"$\dot{\gamma}$", "ln_H": "H", "ln_Tff": "Tff", "ln_P32": "P3_2", "Tset": "Tset", "ln_v": "v", "ln_gd_cm": r"$\dot{\gamma}$（条件平均）",
            "ln_Tff_cm": "Tff（条件平均）", "ln_P32_cm": "P3_2（条件平均）"}
T_COL = {190: "tab:blue", 210: "tab:green", 230: "tab:red"}
H_LEVELS = [0.5, 1.0, 1.5, 2.0]
print("GPR:", GPR_KEYS)
print("Tset を入力にするモデル:", TSET_KEYS)''')

# ---------------------------------------------------------------- 4. 学習と評価
md(r'''## 4. 学習と評価

- S_V：5data で学習し、3data で検証する。低速側・高速側に分けて評価する。
- S_T：5data の 190・230 ℃ で学習し、5data の 210 ℃ をテストにする（温度の内挿）。同じ学習で、3data の 210 ℃（S_T3）も評価する。''')
code(r'''# セル4：S_V の学習と評価
t_all = time.time()
for mat in MATS:
    tr, te = split_V(mat, MAIN_DATA)
    print(f"--- {MATS[mat]} S_V：学習 {len(tr)} 行 / 検証 {len(te)} 行（低速側 {int((te['group'] == '低速側').sum())}、高速側 {int((te['group'] == '高速側').sum())}）")
    run_split(mat, "S_V", MAIN_DATA, tr, te, ALL_KEYS)
print(f"\n合計 {(time.time() - t_all) / 60:.1f} 分")''')
code(r'''# セル5：S_T（210 ℃ のホールドアウト）の学習と評価
t_all = time.time()
for mat in MATS:
    tr, te, te3 = split_T(mat, MAIN_DATA, TEMP_HOLDOUT)
    print(f"--- {MATS[mat]} S_T：学習 {len(tr)} 行（{sorted(tr['Tset'].unique().tolist())} ℃）/ テスト {len(te)} 行（{TEMP_HOLDOUT} ℃）、S_T3：3data の {TEMP_HOLDOUT} ℃ {len(te3)} 行")
    run_split(mat, "S_T", MAIN_DATA, tr, te, ALL_KEYS, extra_te={"S_T3": te3})
print(f"\n合計 {(time.time() - t_all) / 60:.1f} 分")''')
md(r'''### ランダム 20%（S_R1・S_R2）と、条件を1つずつ抜く交差検証（LOCO）

従来の評価方法（行単位のランダム分割）は、同じ条件のショットが学習にも入るので、評価が甘くなる。LOCO は、条件を1つずつ抜く。同じ厚み・射出速度の他の温度の条件が学習に残るので、設定温度の内挿の検証にはなるが、速度の外挿には厳しくない。''')
code(r'''# セル6：ランダム 20%（数分かかる）
t_all = time.time()
for mat in MATS:
    tr, te = split_random(D[(mat, "Alldata", MAIN_DATA)].reset_index(drop=True))
    print(f"--- {MATS[mat]} S_R1：学習 {len(tr)} 行 / テスト {len(te)} 行")
    run_split(mat, "S_R1", MAIN_DATA, tr, te, KEYS_RANDOM)
tr, te = split_random(pool_of("PP", MAIN_DATA))
print(f"--- PP S_R2：学習 {len(tr)} 行 / テスト {len(te)} 行")
run_split("PP", "S_R2", MAIN_DATA, tr, te, KEYS_RANDOM)
print(f"\n合計 {(time.time() - t_all) / 60:.1f} 分")''')
code(r'''# セル7：LOCO
LOCO_ROWS, LOCO_PRED = [], {}
t_all = time.time()
for mat in MATS:
    pool = pool_of(mat, MAIN_DATA)
    grp = np.where(pool["v"] <= 5, "低速側", np.where(pool["v"] >= 280, "高速側", "内側"))
    for key in KEYS_LOCO:
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
            LOCO_ROWS.append({"material": MATS[mat], "model": key, "subset": sub, "n_test": int(mask.sum()), "n_cond": int(pool.loc[mask, "cond"].nunique()),
                              **evaluate(pool["ln_eta"].to_numpy()[mask], mu[mask], sd[mask]), "noise_sd": noise_sd})
        print(f"{MATS[mat]} LOCO {key}: {len(pool)} 行 {pool['cond'].nunique()} 条件、{time.time() - t0:.1f} 秒")
loco_df = pd.DataFrame(LOCO_ROWS)
print(f"\n合計 {(time.time() - t_all) / 60:.1f} 分")''')

# ---------------------------------------------------------------- 5. 感度確認
md(r'''## 5. 感度確認

- 外れ値ありのデータ（S_V）で、主な結論が変わらないかを見る。
- 問題データ一覧の「感度確認」の条件（P02・P03）を、学習から除いて学習し直す（S_V、S_T）。検証側の P03 は、評価から除いた指標（subset 名の後ろに「（感度確認の条件を除く）」）が、上の評価に入っている。''')
code(r'''# セル8：感度確認
t_all = time.time()
if RUN_SENS:
    for mat in MATS:
        tr, te = split_V(mat, "あり")
        run_split(mat, "S_V", "あり", tr, te, KEYS_SENS)
        tr, te = split_V(mat, MAIN_DATA)
        run_split(mat, "S_V", MAIN_DATA, tr[~tr["is_sens"]], te, KEYS_SENS, tag="_感度除く")
        tr, te, te3 = split_T(mat, MAIN_DATA, TEMP_HOLDOUT)
        run_split(mat, "S_T", MAIN_DATA, tr[~tr["is_sens"]], te[~te["is_sens"]], KEYS_SENS, tag="_感度除く", extra_te={"S_T3": te3[~te3["is_sens"]]})
print(f"\n合計 {(time.time() - t_all) / 60:.1f} 分")''')

# ---------------------------------------------------------------- 6. 結果の表
md(r'''## 6. 結果の表

分け方ごとの指標（ln η）。subset は、全体・低速側・高速側。「（感度確認の条件を除く）」は、検証側の P03 を除いた指標。''')
code(r'''# セル9：表と CSV
metrics_df, kernels_df, ols_df = pd.DataFrame(METRICS), pd.DataFrame(KERNELS), pd.DataFrame(OLSROWS)
metrics_df.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False, encoding="utf-8-sig")
kernels_df.to_csv(os.path.join(OUT_DIR, "kernels.csv"), index=False, encoding="utf-8-sig")
ols_df.to_csv(os.path.join(OUT_DIR, "ols_coefficients.csv"), index=False, encoding="utf-8-sig")
loco_df.to_csv(os.path.join(OUT_DIR, "loco_metrics.csv"), index=False, encoding="utf-8-sig")


def mval(mat, split, model, sub, col="RMSE", data=None):
    r = metrics_df[(metrics_df["material"] == MATS[mat]) & (metrics_df["split"] == split) & (metrics_df["data"] == (data or MAIN_DATA))
                   & (metrics_df["model"] == model) & (metrics_df["subset"] == sub)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def show(mat, split, subs, cols=("RMSE", "coverage95", "NLPD"), data=None):
    t = metrics_df[(metrics_df["material"] == MATS[mat]) & (metrics_df["split"] == split) & (metrics_df["data"] == (data or MAIN_DATA)) & metrics_df["subset"].isin(subs)]
    p = t.pivot(index="model", columns="subset", values=list(cols))
    p = p.reindex(columns=[(c, s) for c in cols for s in subs if (c, s) in p.columns]).reindex([k for k in ALL_KEYS if k in p.index])
    print(p.round(3).to_string())


for mat in MATS:
    print(f"\n【S_V】{MATS[mat]}（5data → 3data）")
    show(mat, "S_V", ["低速側", "高速側"])
    print(f"\n【S_T】{MATS[mat]}（{TEMP_HOLDOUT} ℃ をテスト）：5data の {TEMP_HOLDOUT} ℃ ／ 3data の {TEMP_HOLDOUT} ℃（S_T3）")
    show(mat, "S_T", ["全体"])
    show(mat, "S_T3", ["全体"])
print("\n【ランダム 20%】RMSE")
print(metrics_df[(metrics_df["split"].isin(["S_R1", "S_R2"])) & (metrics_df["subset"] == "全体")].pivot_table(index="model", columns=["material", "split"], values="RMSE").round(4).to_string())''')

# ---------------------------------------------------------------- 7. 設定温度の効果
md(r'''## 7. 設定温度の効果

粘度は、設定温度が高いほど下がるはず（同じ γ̇ で、190 ℃ の粘度は 230 ℃ の 1.1〜1.2 倍）。

- **実測の比**：同じ厚み・同じ γ̇ で、190 ℃ の粘度 ÷ 230 ℃ の粘度（230 ℃ の条件平均を、ln γ̇ に対して内挿）。感度確認の条件は除く。
- **モデルの比**：5data＋3data の 210 ℃ のショットについて、ほかの入力（γ̇、Tff、P3_2 など）はそのまま、Tset だけを 190 ℃ と 230 ℃ にして予測した粘度の比。S_V の学習で作ったモデルを使う。Tset を入力にしないモデルは、比が 1 になるので対象外。''')
code(r'''# セル10：設定温度の効果（実測の比と、モデルの比）
def observed_ratio(mat, data=None):
    """厚みごとの、実測の 190/230 ℃ の粘度比（同じ γ̇。230 ℃ の条件平均を内挿）。感度確認の条件は除く"""
    d = pool_of(mat, data or MAIN_DATA)
    d = d[~d["is_sens"]]
    cm = d.groupby(["Tset", "H", "v"]).agg(lg=("ln_gd", "mean"), le=("ln_eta", "mean")).reset_index()
    rows = []
    for h in H_LEVELS:
        a, c = cm[(cm["H"] == h) & (cm["Tset"] == 190)], cm[(cm["H"] == h) & (cm["Tset"] == 230)].sort_values("lg")
        ins = ((a["lg"] >= c["lg"].min()) & (a["lg"] <= c["lg"].max())).to_numpy()
        r = np.exp(a["le"].to_numpy()[ins] - np.interp(a["lg"].to_numpy()[ins], c["lg"], c["le"]))
        rows.append({"material": MATS[mat], "H": h, "observed": float(np.median(r)), "n": int(ins.sum())})
    return pd.DataFrame(rows)


def tset_effect(name, data=None, models=None):
    """S_V などの学習で作ったモデルが予測する、Tset だけを 190 → 230 ℃ にしたときの粘度の比（厚みごとの中央値）"""
    rows = []
    for mat in MATS:
        base = pool_of(mat, data or MAIN_DATA)
        base = base[(base["Tset"] == 210) & ~base["is_sens"]]
        for key in (models or TSET_KEYS):
            fit = FITS.get((mat, name, data or MAIN_DATA, key))
            if fit is None:
                continue
            lo, hi = base.copy(), base.copy()
            lo["Tset"], hi["Tset"] = 190.0, 230.0
            r = np.exp(predict_model(key, fit, lo)[0] - predict_model(key, fit, hi)[0])
            for h, g in base.assign(r=r).groupby("H"):
                rows.append({"material": MATS[mat], "split": name, "model": key, "H": h, "ratio_190_230": float(np.median(g["r"])), "n": len(g)})
    return pd.DataFrame(rows)


obs_df = pd.concat([observed_ratio(m) for m in MATS], ignore_index=True)
eff_df = pd.concat([tset_effect("S_V"), tset_effect("S_T")], ignore_index=True)
if RUN_SENS:
    eff_df = pd.concat([eff_df, tset_effect("S_V_感度除く", models=KEYS_SENS[1:]), tset_effect("S_T_感度除く", models=KEYS_SENS[1:])], ignore_index=True)
obs_df.to_csv(os.path.join(OUT_DIR, "observed_ratio.csv"), index=False, encoding="utf-8-sig")
eff_df.to_csv(os.path.join(OUT_DIR, "tset_effect.csv"), index=False, encoding="utf-8-sig")
print("実測の 190/230 ℃ の粘度比（同じ γ̇、厚みごと）")
print(obs_df.pivot(index="material", columns="H", values="observed").round(3).to_string())
for name in ["S_V", "S_T"]:
    print(f"\nモデルの比（{name} の学習。Tset だけを 190 → 230 ℃）")
    print(eff_df[eff_df["split"] == name].pivot_table(index=["material", "model"], columns="H", values="ratio_190_230").round(3).to_string())''')

# ---------------------------------------------------------------- 8. 図
md(r'''## 8. 図

- 図1：設定温度ごとの予測曲線（Tff・P3_2 の有無を目で比べる）。材料ごとに1枚（1a：PP、1b：粉砕材混合 PP）
- 図2：S_V のパリティ図
- 図3：S_T（210 ℃）のパリティ図
- 図4：分け方ごとの RMSE・NLPD
- 図5：設定温度の効果（190/230 ℃ の粘度比）
- 図6：ARD の感度
- 図7：ノイズの標準偏差と、同じ条件内のばらつき（純粋誤差）''')
code(r'''# セル11：図1 設定温度ごとの予測曲線（S_V の学習。材料ごと。列がモデル、行が厚み）
CURVE_KEYS = ["B0T", "R3", "R3F", "R3P"]


def curve_fig(mat, tag):
    n = len(CURVE_KEYS)
    fig, axes = plt.subplots(4, n, figsize=(5.2 * n, 15), sharex="row", sharey="row")
    te_cond = set(D[(mat, "3data", MAIN_DATA)]["cond"])
    allp = pool_of(mat, MAIN_DATA)
    for j, key in enumerate(CURVE_KEYS):
        fit = FITS[(mat, "S_V", MAIN_DATA, key)]
        for i, h in enumerate(H_LEVELS):
            ax = axes[i, j]
            for t, col in T_COL.items():
                sub = allp[(allp["H"] == h) & (allp["Tset"] == t)]
                if sub.empty:
                    continue
                cm = sub.groupby("cond").agg(lg=("ln_gd", "mean"), le=("ln_eta", "mean"), tff=("ln_Tff", "mean"), p32=("ln_P32", "mean"), sens=("is_sens", "max")).sort_values("lg")
                lg = np.linspace(sub["ln_gd"].min(), sub["ln_gd"].max(), 120)
                tff_i, p32_i = np.interp(lg, cm["lg"], cm["tff"]), np.interp(lg, cm["lg"], cm["p32"])
                Xg = pd.DataFrame({"ln_gd": lg, "ln_H": np.log(h), "Tset": float(t), "ln_Tff": tff_i, "ln_P32": p32_i, "ln_v": np.nan, "ln_gd_cm": lg,
                                   "ln_Tff_cm": tff_i, "ln_P32_cm": p32_i})
                mu, sd, _, _ = predict_model(key, fit, Xg)
                ax.plot(np.exp(lg), np.exp(mu), color=col, lw=1.4)
                ax.fill_between(np.exp(lg), np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd), color=col, alpha=0.12)
                is_te = np.array([c in te_cond for c in cm.index])
                ax.scatter(np.exp(cm["lg"][~is_te]), np.exp(cm["le"][~is_te]), s=16, color=col, alpha=0.8, zorder=3)
                ax.scatter(np.exp(cm["lg"][is_te]), np.exp(cm["le"][is_te]), s=28, facecolors="none", edgecolors=col, marker="s", linewidths=1.2, zorder=3)
                ax.scatter(np.exp(cm["lg"][cm["sens"].astype(bool)]), np.exp(cm["le"][cm["sens"].astype(bool)]), s=45, color="k", marker="x", zorder=4)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
            if j == 0:
                ax.set_ylabel(f"厚み {h} mm\n" + r"見かけの粘度 $\eta$ [Pa·s]")
            if i == 0:
                ax.set_title(LABELS[key], fontsize=10)
            if i == 3:
                ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ [1/s]")
    handles = [plt.Line2D([], [], color=c, label=f"{t} ℃") for t, c in T_COL.items()]
    handles += [plt.Line2D([], [], color="gray", marker="o", ls="", label="学習の条件平均"), plt.Line2D([], [], color="gray", marker="s", mfc="none", ls="", label="検証の条件平均"),
                plt.Line2D([], [], color="k", marker="x", ls="", label="感度確認の条件（P02・P03）")]
    axes[0, 0].legend(handles=handles, fontsize=7, loc="upper right")
    fig.suptitle(f"{MATS[mat]}：設定温度ごとの予測曲線（5data で学習。線：予測、帯：95% 予測区間）。Tff・P3_2 は、同じ設定温度・厚みの条件平均を " + r"$\ln\dot{\gamma}$" + " で内挿して入力に使う", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(OUT_DIR, f"fig1{tag}_tset_curves_{mat}.png"), dpi=FIG_DPI)
    plt.show()


curve_fig("PP", "a")
curve_fig("MIX", "b")''')
code(r'''# セル12：図2 S_V のパリティ図、図3 S_T（210 ℃）のパリティ図
PAR_KEYS = ["B0", "B0T", "R0", "R2", "R3", "R3F"]
GCOL = {"低速側": "tab:blue", "高速側": "tab:red"}


def parity(ax, p, groups, key, title):
    for g, col in groups.items():
        m = (p["group"] == g).to_numpy() if g else np.ones(len(p), bool)
        ax.errorbar(p["y"][m], p["mu"][m], yerr=1.96 * p["sd"][m], fmt="o", ms=3, alpha=0.6, elinewidth=0.6, color=col, ecolor="lightgray", label=g or "全体")
    s = p["is_sens"].to_numpy()
    if s.any():
        ax.scatter(p["y"][s], p["mu"][s], s=40, color="k", marker="x", zorder=5)
    lo, hi = min(p["y"].min(), p["mu"].min()) - 0.2, max(p["y"].max(), p["mu"].max()) + 0.2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel(r"実測値 $\ln\eta$", fontsize=8)
    ax.set_ylabel(r"予測値 $\ln\eta$", fontsize=8)
    ax.grid(alpha=0.3)


fig, axes = plt.subplots(len(MATS), len(PAR_KEYS), figsize=(22, 8.8))
for i, mat in enumerate(MATS):
    for j, key in enumerate(PAR_KEYS):
        p = PRED[(mat, "S_V", MAIN_DATA, key)]
        parity(axes[i, j], p, GCOL, key, f"{MATS[mat]}　{LABELS[key]}\nRMSE 低速側 {mval(mat, 'S_V', key, '低速側'):.3f}／高速側 {mval(mat, 'S_V', key, '高速側'):.3f}")
axes[0, 0].legend(fontsize=8)
fig.suptitle("5data → 3data（S_V）のパリティ図（青：低速側、赤：高速側、×：感度確認の条件）", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_parity_S_V.png"), dpi=FIG_DPI)
plt.show()

H_COL = {0.5: "tab:red", 1.0: "tab:blue", 1.5: "tab:orange", 2.0: "tab:green"}
PAR_T_KEYS = ["B0", "B0T", "B0F", "R0", "R3", "R3F"]
fig, axes = plt.subplots(len(MATS), len(PAR_T_KEYS), figsize=(22, 8.8))
for i, mat in enumerate(MATS):
    for j, key in enumerate(PAR_T_KEYS):
        ax = axes[i, j]
        p = PRED[(mat, "S_T", MAIN_DATA, key)]
        for h, col in H_COL.items():
            m = (p["H"] == h).to_numpy()
            ax.errorbar(p["y"][m], p["mu"][m], yerr=1.96 * p["sd"][m], fmt="o", ms=3, alpha=0.6, elinewidth=0.6, color=col, ecolor="lightgray", label=f"{h} mm")
        lo, hi = p["y"].min() - 0.2, p["y"].max() + 0.2
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{MATS[mat]}　{LABELS[key]}\nRMSE {mval(mat, 'S_T', key, '全体'):.3f}、被覆率 {mval(mat, 'S_T', key, '全体', 'coverage95'):.2f}", fontsize=9)
        ax.set_xlabel(r"実測値 $\ln\eta$", fontsize=8)
        ax.set_ylabel(r"予測値 $\ln\eta$", fontsize=8)
        ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=8)
fig.suptitle(f"S_T：{TEMP_HOLDOUT} ℃ を全部テストにして、残りの温度で学習した場合のパリティ図（色：厚み）", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_parity_S_T.png"), dpi=FIG_DPI)
plt.show()''')
code(r'''# セル13：図4 分け方ごとの RMSE・NLPD
PANELS = [("S_V", "低速側", "S_V 低速側"), ("S_V", "高速側", "S_V 高速側"), ("S_T", "全体", f"S_T（{TEMP_HOLDOUT} ℃）"),
          ("S_T3", "全体", f"S_T3（3data の {TEMP_HOLDOUT} ℃）"), ("S_R1", "全体", "ランダム 20%（Alldata）")]
COLORS = ["tab:blue", "tab:red", "tab:purple", "tab:brown", "tab:green"]
fig, axes = plt.subplots(2, len(MATS), figsize=(17, 9))
for j, mat in enumerate(MATS):
    for i, (metric, ylab) in enumerate([("RMSE", r"RMSE（$\ln\eta$）"), ("NLPD", "NLPD（小さいほど良い）")]):
        ax = axes[i, j]
        w = 0.8 / len(PANELS)
        for k, (split, sub, lab) in enumerate(PANELS):
            vals = [mval(mat, split, key, sub, metric) for key in ALL_KEYS]
            ax.bar(np.arange(len(ALL_KEYS)) + (k - (len(PANELS) - 1) / 2) * w, vals, w, color=COLORS[k], label=lab)
        ax.set_xticks(np.arange(len(ALL_KEYS)))
        ax.set_xticklabels(ALL_KEYS, rotation=0)
        ax.set_ylabel(ylab)
        ax.set_title(f"{MATS[mat]}：{ylab}")
        ax.grid(alpha=0.3, axis="y")
        if metric == "RMSE":
            ax.set_yscale("log")
        if i == 0 and j == 0:
            ax.legend(fontsize=8)
fig.suptitle("分け方ごとの評価（ランダム分割は、条件のほとんどが学習にも入るので甘い）", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_metrics_by_split.png"), dpi=FIG_DPI)
plt.show()''')
code(r'''# セル14：図5 設定温度の効果、図6 ARD の感度、図7 ノイズと純粋誤差
fig, axes = plt.subplots(1, len(MATS), figsize=(15, 5.2), sharey=True)
MK = {"L0T": "s", "B0T": "o", "B0FT": "^", "B2T": "v", "B2FT": "D", "R2": "P", "R3": "X", "R3F": "*", "R3P": "h"}
for ax, mat in zip(axes, MATS):
    o = obs_df[obs_df["material"] == MATS[mat]]
    ax.plot(o["H"], o["observed"], "k-o", lw=2.5, ms=9, label=r"実測（同じ $\dot{\gamma}$）", zorder=10)
    for key in TSET_KEYS:
        e = eff_df[(eff_df["split"] == "S_V") & (eff_df["material"] == MATS[mat]) & (eff_df["model"] == key)]
        ax.plot(e["H"], e["ratio_190_230"], marker=MK[key], ls="--", alpha=0.85, label=key)
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xlabel("厚み [mm]")
    ax.set_title(f"{MATS[mat]}：190 ℃ と 230 ℃ の粘度の比（Tset だけを変えた予測）")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("粘度の比（190 ℃ ÷ 230 ℃）")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig5_temperature_effect.png"), dpi=FIG_DPI)
plt.show()

inputs = list(INPUT_JA)
fig, axes = plt.subplots(1, len(MATS), figsize=(18, 5.2), sharey=True)
base = kernels_df[(kernels_df["split"] == "S_V") & (kernels_df["data"] == MAIN_DATA)]
for ax, mat in zip(axes, MATS):
    t = base[base["material"] == MATS[mat]].set_index("model").reindex(GPR_KEYS)
    w = 0.85 / len(GPR_KEYS)
    for k, key in enumerate(GPR_KEYS):
        vals = [1.0 / t.loc[key, f"ls_{c}"] if (f"ls_{c}" in t.columns and not np.isnan(t.loc[key, f"ls_{c}"])) else np.nan for c in inputs]
        ax.bar(np.arange(len(inputs)) + (k - (len(GPR_KEYS) - 1) / 2) * w, vals, w, label=key, color=plt.cm.tab20(k / 20))
    ax.set_yscale("log")
    ax.set_ylim(5e-3, 300)
    ax.set_xticks(np.arange(len(inputs)))
    ax.set_xticklabels([INPUT_JA[c] for c in inputs], rotation=25, ha="right")
    ax.set_title(f"{MATS[mat]}：ARD の感度（S_V の学習）")
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("感度 = 1／長さスケール（対数軸）")
fig.suptitle("感度が大きいほどよく効く。0.01 は、長さスケールが上限 100 に張り付いた（ほぼ無視）こと", fontsize=10)
axes[0].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig6_ard_sensitivity.png"), dpi=FIG_DPI)
plt.show()

fig, axes = plt.subplots(1, len(MATS), figsize=(15, 5), sharey=True)
for ax, mat in zip(axes, MATS):
    t = base[base["material"] == MATS[mat]].set_index("model").reindex(GPR_KEYS)
    ax.bar(np.arange(len(GPR_KEYS)), t["noise_sd"], color="tab:blue", label="モデルのノイズの標準偏差")
    ax.axhline(t["pure_sd_ln_eta"].iloc[0], color="tab:red", ls="--", label="純粋誤差（同じ条件内の標準偏差）")
    ax.axhline(t["pure_sd_after_gd"].iloc[0], color="tab:orange", ls=":", label=r"純粋誤差のうち、$\ln\dot{\gamma}$ で説明した残り")
    ax.set_xticks(np.arange(len(GPR_KEYS)))
    ax.set_xticklabels(GPR_KEYS)
    ax.set_title(f"{MATS[mat]}：ノイズの標準偏差（S_V の学習）")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel(r"標準偏差（$\ln\eta$）")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig7_noise_vs_pure_error.png"), dpi=FIG_DPI)
plt.show()''')

# ---------------------------------------------------------------- 9. 感度確認の判定
md(r'''## 9. 感度確認の判定

学習から感度確認の条件（P02・P03）を除いたとき、結果がどれだけ変わったか。**「影響が大きい」の基準**：検証の RMSE が 10% 以上、かつ 0.01（粘度で約 1%）以上変わる（RMSE が 0.01 前後の小さな値のときの、割合だけの変化は数えない）／0.5 mm での 190/230 ℃ の予測比が 0.05 以上変わる。どちらかに当たれば、その条件だけ除いたデータファイルを新しく作る（`04_補助スクリプト/make_excluded_files.py`）。''')
code(r'''# セル15：感度確認の判定
rows = []
if RUN_SENS:
    for mat in MATS:
        for key in KEYS_SENS:
            for split, sub in [("S_V", "低速側"), ("S_V", "高速側"), ("S_T", "全体"), ("S_T3", "全体")]:
                a, b = mval(mat, split, key, sub), mval(mat, split + "_感度除く", key, sub)
                rows.append({"material": MATS[mat], "model": key, "split": split, "subset": sub, "RMSE_すべて": a, "RMSE_感度確認の条件を除く": b,
                             "変化 [%]": (b / a - 1) * 100 if a else np.nan, "影響が大きい": abs(b / a - 1) >= 0.10 and abs(b - a) >= 0.01})
            a = mval(mat, "S_V", key, "高速側")
            b = mval(mat, "S_V", key, "高速側" + SENS_SUFFIX)               # 検証側の P03 を評価から除いた指標
            rows.append({"material": MATS[mat], "model": key, "split": "S_V", "subset": "高速側（検証の P03 を評価から除く）", "RMSE_すべて": a, "RMSE_感度確認の条件を除く": b,
                         "変化 [%]": (b / a - 1) * 100 if a else np.nan, "影響が大きい": abs(b / a - 1) >= 0.10 and abs(b - a) >= 0.01})
    ratio_rows = []
    for mat in MATS:
        for key in KEYS_SENS[1:]:
            r = {}
            for sp in ["S_V", "S_V_感度除く"]:
                e = eff_df[(eff_df["material"] == MATS[mat]) & (eff_df["split"] == sp) & (eff_df["model"] == key) & (eff_df["H"] == 0.5)]
                r[sp] = float(e["ratio_190_230"].iloc[0]) if len(e) else np.nan
            ratio_rows.append({"material": MATS[mat], "model": key, "split": "0.5 mm の 190/230 ℃ の予測比", "subset": "（S_V の学習）", "RMSE_すべて": r["S_V"],
                               "RMSE_感度確認の条件を除く": r["S_V_感度除く"], "変化 [%]": (r["S_V_感度除く"] - r["S_V"]) * 100, "影響が大きい": abs(r["S_V_感度除く"] - r["S_V"]) >= 0.05})
    sens_df = pd.concat([pd.DataFrame(rows), pd.DataFrame(ratio_rows)], ignore_index=True)
    sens_df.to_csv(os.path.join(OUT_DIR, "sensitivity_judgement.csv"), index=False, encoding="utf-8-sig")
    print(sens_df.round(3).to_string(index=False))
    for mat in MATS:
        s = sens_df[sens_df["material"] == MATS[mat]]
        is_ratio, is_eval = s["split"].str.contains("予測比"), s["subset"].str.contains("P03")
        learn, evl, ratio = s[~is_ratio & ~is_eval], s[is_eval], s[is_ratio]
        r3 = learn[learn["model"] == "R3"]
        others = int(learn[learn["model"] != "R3"]["影響が大きい"].sum())
        judge(not r3["影響が大きい"].any() and not ratio[ratio["model"] == "R3"]["影響が大きい"].any(),
              f"{MATS[mat]}：学習から P02・P03 を除いても、R3 の検証の RMSE（{len(r3)} 項目）と設定温度の効果（0.5 mm の予測比）は大きく変わらない。ほかのモデルで大きく変わった項目 {others} 個は、最適化が不安定なモデル（B0T の S_T など）の揺れ")
        e3 = evl[evl["model"] == "R3"]
        if len(e3) and e3["RMSE_感度確認の条件を除く"].notna().any():
            r = e3.iloc[0]
            judge(not bool(r["影響が大きい"]), f"{MATS[mat]}：検証から P03（PP の 190 ℃・0.5 mm・320 mm/s）を評価で除くと、R3 の高速側の RMSE が {r['RMSE_すべて']:.3f} → {r['RMSE_感度確認の条件を除く']:.3f}（{r['変化 [%]']:.0f}%）。10% 以上変わるなら、P03 は評価を左右する → その条件だけを除いたファイルを作る")
        else:
            judge(True, f"{MATS[mat]}：検証データに感度確認の条件（P03）がない")''')

# ---------------------------------------------------------------- 10. 判定のまとめ
md(r'''## 10. 判定のまとめ

観点ごとに、結果の判定を表示する。基準は、判定の文の中に書いてある。''')
code(r'''# セル16：判定
print("【判定】（OK＝条件を満たす／★要確認＝満たさない。基準は、文の中に書いてある）")
HI = "高速側" + SENS_SUFFIX                      # 高速側から、検証の P03 を除いた指標（PP の高速側は P03 に引きずられるので、比べるときはこちらを使う）
for mat in MATS:
    n = MATS[mat]
    o = obs_df[obs_df["material"] == n]["observed"]
    print(f"--- {n}")
    # 1. 設定温度の効果
    for key in ["B0T", "R3", "R3F", "R3P"]:
        e = eff_df[(eff_df["split"] == "S_V") & (eff_df["material"] == n) & (eff_df["model"] == key)]["ratio_190_230"]
        judge(bool((e > 1.0).all()), f"{key}：Tset だけを 190 → 230 ℃ にした予測の粘度比が、全厚みで 1 を超える（190 ℃ のほうが粘い）。範囲 {e.min():.2f}〜{e.max():.2f}（実測 {o.min():.2f}〜{o.max():.2f}）")
    e3 = eff_df[(eff_df["split"] == "S_V") & (eff_df["material"] == n) & (eff_df["model"] == "R3")].sort_values("H")["ratio_190_230"].to_numpy()
    ob = obs_df[obs_df["material"] == n].sort_values("H")["observed"].to_numpy()
    judge(float(np.abs(e3 - ob).max()) <= 0.05, f"R3：粘度比が、実測と 0.05 以内で一致する（厚みごとの差の最大 {float(np.abs(e3 - ob).max()):.3f}）")
    # 2. Tset は要るか（条件平均の形で、Tset なし R0 と、あり R3 を比べる）
    j0, j3 = mval(mat, "S_T", "R0", "全体"), mval(mat, "S_T", "R3", "全体")
    judge(j3 < 0.8 * j0, f"Tset を足す（R0 → R3）と、{TEMP_HOLDOUT} ℃ のホールドアウト（S_T）の RMSE が 2 割以上小さくなる（R0 {j0:.3f} → R3 {j3:.3f}）→ Tset は要る")
    # 3. Tff・P3_2 は要るか（R3 に足して比べる。S_T、S_V 低速側）
    r3_t, r3_l = mval(mat, "S_T", "R3", "全体"), mval(mat, "S_V", "R3", "低速側")
    for key, nm in [("R3F", "Tff"), ("R3P", "P3_2")]:
        t_, l_ = mval(mat, "S_T", key, "全体"), mval(mat, "S_V", key, "低速側")
        judge(t_ > 0.9 * r3_t and l_ > 0.9 * r3_l, f"{nm} を足しても（R3 → {key}）、S_T の RMSE（{r3_t:.3f} → {t_:.3f}）と S_V 低速側の RMSE（{r3_l:.3f} → {l_:.3f}）が、1 割以上は良くならない → {nm} は要らない")
    # 4. 低速側への外挿と、区間の幅
    lo = mval(mat, "S_V", "R3", "低速側")
    judge(lo < 0.1, f"R3 の S_V 低速側の RMSE = {lo:.3f}（0.1 未満なら、線形カーネルで低速側へ外挿できている）")
    cv = mval(mat, "S_V", "R3", "低速側", "coverage95")
    judge(0.9 <= cv <= 0.99, f"R3 の S_V 低速側の 95% 被覆率 = {cv:.2f}（0.90〜0.99 なら、区間の幅が適切。1.00 なら、広すぎる）")
    b0t = mval(mat, "S_V", "B0T", "低速側")
    judge(lo < 0.5 * b0t, f"条件平均の γ̇（R3 {lo:.3f}）が、ショットごとの γ̇（B0T {b0t:.3f}）より、低速側へ外挿できる（半分未満）")
    # 5. ばらつき
    k = kernels_df[(kernels_df["split"] == "S_V") & (kernels_df["data"] == MAIN_DATA) & (kernels_df["material"] == n)].set_index("model")
    judge(k.loc["R3", "noise_sd"] > 0.5 * k.loc["R3", "pure_sd_ln_eta"], f"R3 のノイズの標準偏差 {k.loc['R3', 'noise_sd']:.4f} が、純粋誤差 {k.loc['R3', 'pure_sd_ln_eta']:.4f} の半分以上（成形条件から見たばらつきを表している。B0T は {k.loc['B0T', 'noise_sd']:.4f} で、ショットごとの γ̇ を使うため小さい）")
    # 6. 設定値だけ（R2）で足りるか
    def hi_val(key):                                  # 高速側から P03 を除いた RMSE（検証に P03 がない材料は、高速側そのまま）
        v = mval(mat, "S_V", key, HI)
        return v if not np.isnan(v) else mval(mat, "S_V", key, "高速側")
    r2, r3h = hi_val("R2"), hi_val("R3")
    judge(r2 < 2 * r3h, f"設定値だけの R2 の S_V 高速側（P03 を除く）の RMSE {r2:.3f} が、R3 {r3h:.3f} の 2 倍未満（実測の γ̇ を使わなくても予測できる）")
    # 7. 設定温度の長さスケールが、下限に張り付いたモデル（Tset が「区別するだけの目印」になり、学習にない温度を内挿できなくなる）
    kb = kernels_df[(kernels_df["data"] == MAIN_DATA) & (kernels_df["material"] == n) & kernels_df["model"].isin(TSET_KEYS) & (kernels_df["ls_Tset"] < 0.05)]
    judge(kb.empty, f"Tset の長さスケールが 0.05 未満（下限 0.01 に張り付き）のモデル: {sorted(set(kb['split'] + ':' + kb['model'])) or 'なし'}（張り付くと、そのモデルは設定温度を内挿できない。3c の MCMC で確かめる）")
print("\n（Tff・P3_2 の判定は、条件平均を入力にした R3 系で行った。ショットごとの γ̇ を入力にした B 系は、ノイズが極端に小さく、予測が不安定になるため）")''')

# ---------------------------------------------------------------- 11. 高解像度のグラフ
md(HR_MD.replace("## 高解像度のグラフ", "## 11. 高解像度のグラフ"))
code(hr_cell("# セル17：高解像度のグラフ（せん断速度 ― 粘度。GPR6月ノートのセル9-2 にならう）",
             '''HR_MODELS = ["R3"]                 # 描くモデル（GPR のどれでも。例：["R3", "B0T", "R3F"]。記号の意味は、ノート冒頭の表）
HR_SPLIT = "S_V"                   # どの学習のモデルか："S_V"（5data で学習）／"S_T"（190・230 ℃ で学習）／"ALL"（5data＋3data で学習し直す。点は全部が学習の点）''',
             r'''HR_SPLIT_JA = {"S_V": "5data で学習", "S_T": "190・230 ℃ で学習", "ALL": "5data＋3data で学習"}


def hr_fit(mat, key):
    """描くモデルの学習済み予測器。S_V・S_T は上で学習したもの、ALL は 5data＋3data で学習し直す"""
    return fit_model(key, pool_of(mat, MAIN_DATA)) if HR_SPLIT == "ALL" else FITS[(mat, HR_SPLIT, MAIN_DATA, key)]


hr_saved = []
for mat in HR_MATERIALS:
    for key in HR_MODELS:
        if MODELS[key]["kind"] != "gpr":
            print(f"{key} は線形回帰なので、描かない（GPR のモデルを指定する）")
            continue
        t0 = time.time()
        fit = hr_fit(mat, key)
        hr_saved += hr_flow(lambda Xg, f=fit, k=key: predict_model(k, f, Xg)[:3], mat, MAIN_DATA, hr_split_conds(mat, MAIN_DATA, HR_SPLIT, TEMP_HOLDOUT),
                            f"{LABELS[key]}（{HR_SPLIT_JA[HR_SPLIT]}）", f"hr_{mat}_{key}_{HR_SPLIT}", HR_DIR, **HR_KW)
        print(f"{MATS[mat]} {key}：保存したファイル 累計 {len(hr_saved)} 個（{time.time() - t0:.0f} 秒）")
judge(len(hr_saved) > 0, f"高解像度のグラフを {len(hr_saved)} 個保存した（{os.path.normpath(HR_DIR)}）。形式 {HR_FORMATS}、{HR_DPI} dpi")'''))

# ---------------------------------------------------------------- 12. 保存したファイル
code(r'''# セル18：保存したファイル
print("保存したファイル:")
for f in sorted(os.listdir(OUT_DIR)):
    print("  ", os.path.normpath(os.path.join(OUT_DIR, f)))''')


# ---------------------------------------------------------------- 書き出し
def to_lines(s):
    lines = s.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


rnd = random.Random(20261013)
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
