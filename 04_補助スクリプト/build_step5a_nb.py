# -*- coding: utf-8 -*-
"""手順5a（シフト構造の見直し）のノートブック (未実行) を生成する。

使い方: python build_step5a_nb.py <出力する .ipynb のパス> [--force]
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
md(r'''# 手順5a：温度シフト構造の見直し（新データ）

CLAUDE.md §5「見直し中：シフト構造」の確認。当初の構造（時間–温度換算則を、設定温度 Tset で組み込む）が、新しいデータで成り立つかを調べ、代わりになる構造を比べる。

**なぜ見直すのか**：当初の構造では、横方向（せん断速度の軸）と縦方向（粘度の軸）を**同じ活性化エネルギー E** でずらす。ところが新データの見かけの流動曲線は、傾き s ≈ −1.6（せん断速度が上がると、せん断応力が下がる）。そのため、同じせん断速度での粘度の差は (1 + s)·Δln aT になり、E > 0 にすると「低温ほど粘度が低い」予測になってしまう。データは「低温ほど粘度が高い」（190 ℃ は 230 ℃ の 1.09〜1.21 倍）。試算では、最適な E が 0 付近か負になった（−5〜−2.5 kJ/mol）。

**比べる構造**（f は GP。入力は u = ln γ̇ ＋ ln aT_h、w = ln Fo = −(ln γ̇ ＋ 3 ln H) ＋ 定数。シフトは ln a(T; E) = (E/R)·(1/T − 1/Tref)、Tref = 483.15 K（210 ℃））

| 記号 | 構造 | 式 | 推定するもの |
|---|---|---|---|
| H | 当初（§5）：横も縦も同じ E | ln η = ln a(E) + f(ln γ̇ + ln a(E), ln Fo) | E |
| A | 縦方向だけシフト | ln η = ln a(E_v) + f(ln γ̇, ln Fo) | E_v |
| Hz | 横方向だけシフト | ln η = f(ln γ̇ + ln a(E_h), ln Fo) | E_h |
| B | 横と縦を別々 | ln η = ln a(E_v) + f(ln γ̇ + ln a(E_h), ln Fo) | E_h, E_v |
| N | 温度なし（H の E = 0） | ln η = f(ln γ̇, ln Fo) | ― |
| H40 | H で E = 40 kJ/mol に固定（文献値） | ― | ― |
| F | 参考：Tff でシフト（H と同じ式で、T に Tff を使う） | ― | E |
| C | 比較用：構造なし。Tset を入力に並べた通常の GPR（R1） | ln η = f(ln γ̇, ln H, Tset) | ― |

**推定（プロファイル尤度）**：E を格子で振り、各 E で入力を (u, ln Fo)、出力を ln η − ln a(E_v) に変換して GPR（ARD Matern 5/2 ＋ 線形カーネル ＋ ノイズ）を学習し、対数周辺尤度が最大になる E を採用する。出力の自動標準化は使わない（E ごとに尺度が変わり、尤度を比べられなくなるため）。入力の標準化も、E に依存しない定数（E = 0 のときの学習データの平均・標準偏差）を使う。**尤度の比較は条件平均**（条件ごとに、ショットの ln η と ln γ̇ を平均した点）で行う。ショットごとの γ̇ を入力にすると、E ≠ 0 のとき設定温度が入力の上で少し分かれ、GP がそれを温度の目印に使えてしまうため（CLAUDE.md §5）。

**予測の評価**：尤度で決めた E を使って、ショット単位の GPR を学習し、S_V（5data → 3data）と S_T（190・230 ℃ → 210 ℃）で評価する。入力の γ̇ は、条件平均（cm：成形条件から見た予測）と、ショットごと（shot）の両方。

- 「A」「B」の E_v が 7 kJ/mol 程度なら、190 ℃ と 230 ℃ の粘度の比は約 1.15 倍（exp(0.0206·E_v)）。

**本命と対案（ユーザー決定、2026-10-12）**：本命は **B**（横・縦を別々）、対案は **A**（縦だけ）。後で自分で実行して比べられるように、この 2 つを並べて見る節（「8. 本命と対案の比較」）を設けた。比べる構造は、設定エリアの `MAIN_STRUCT`・`ALT_STRUCT` で変えられる（H・A・Hz・B・N・H40・F・C のどれでもよい）。比較の基準は C（構造なしの通常の GPR）。

**それぞれの形で、何が表せるか（なぜ採用したか）**

| 記号 | この形で表せること | 採用したときの期待 |
|---|---|---|
| H | 温度を変えると、流動曲線が形を変えずに、横（せん断速度）と縦（粘度）に、同じ割合でずれる。1 本の基準の曲線（マスターカーブ）から、どの温度の曲線も作れる。 | 当初の構造（CLAUDE.md §5）。温度ごとのデータを 1 本にまとめられ、E を文献値（PP は 40 kJ/mol 前後）と比べられる、と期待した。 |
| A | 曲線の形は温度で変わらず、全体が一定の倍率だけ上下に動く。同じ γ̇ で、低温ほど粘度が一定の倍率だけ高い。 | 実測の「190 ℃ は 230 ℃ の 1.1〜1.2 倍」を、パラメータ 1 個（E_v）でいちばん簡単に表せる、と期待した。 |
| Hz | 曲線の形は温度で変わらず、全体が横にだけ動く。同じ粘度になる γ̇ が、温度で変わる。 | 横のずれだけで温度の効果を表せるかを確かめる対照。 |
| B | A の「一定の倍率の上下」に、横のずれを足す。曲線の傾きは γ̇ の場所で違うので、横にずらすと、粘度の差が場所ごとに変わる。 | 実測の「厚いほど（γ̇ が小さいほど）温度の効果が大きい」傾向を、A のわかりやすさ（一定の倍率が土台）を残したまま表せる、と期待した。 |
| N | 温度を無視する（E = 0）。 | 温度の効果がないときの基準。 |
| H40 | H で E を文献値 40 kJ/mol に固定。 | 文献値が、このデータに合うかの確認。 |
| F | H と同じ形で、T に流動先端温度 Tff を使う。 | 参考。Tff は流れの結果で、成形条件から事前には分からない。 |
| C | 構造なし。Tset を、ほかの入力と並べた通常の GPR（R1）。 | 構造が要るかの基準。構造あり（A・B）が C と同等なら、構造が支持された、とみなす。 |

図0に、これらの形の違いの模式図（実データではない）を付けた。''')

# ---------------------------------------------------------------- 1. 設定
md(usage_md("5a", "results/step5a", "results/学習結果説明_手順5a.pdf", 40))
md(r'''## 1. 設定''')
code(settings_cell('''import numpy as np             # 格子の指定に使う
MAIN_STRUCT = "B"             # 本命の構造（ユーザー決定、2026-10-12）。H・A・Hz・B・N・H40・F・C のどれでも
ALT_STRUCT = "A"              # 対案の構造（同上）。「8. 本命と対案の比較」と、高解像度のグラフの既定で使う
MAIN_DATA = "なし"            # 主に使う外れ値の扱い
TEMP_HOLDOUT = 210            # S_T でテストにする設定温度 [℃]
E_GRID = np.arange(-20.0, 62.5, 2.5)        # 1 次元の E の格子 [kJ/mol]（H・A・Hz・F）
EH_GRID = np.arange(-40.0, 45.0, 5.0)       # B の横方向の E_h の格子 [kJ/mol]
EV_GRID = np.arange(-20.0, 45.0, 5.0)       # B の縦方向の E_v の格子 [kJ/mol]
N_RESTARTS_PROFILE = 3        # 尤度の格子の各点での最適化のやり直し回数
N_RESTARTS_FINAL = 3          # 予測の評価のための学習でのやり直し回数
N_JOBS = 6                    # 並列に計算する数（CPU の数に合わせる）
E_LIT = 40.0                  # 文献値の目安 [kJ/mol]（PP の流動の活性化エネルギー）
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

OUT_DIR = os.path.join(REPO, "results", "step5a")
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

データの読み込み、問題データ一覧の「除外」、評価、データの分け方の関数。中身は `04_補助スクリプト/nb_common.py` と同じ。''')
code(COMMON_CELL)

# ---------------------------------------------------------------- 2. データ
md(r'''## 2. データの読み込み''')
code(r'''# セル2：読み込み
D = load_all()
leg = legacy_check(D)
judge(bool(leg["同じ行"].all()) or not USE_EXCLUSIONS, "一覧の「除外」を除いたデータが、これまでの no1.5-3.0 のファイルと同じ（取り直した後は違ってよい）")
for mat in MATS:
    tr, te = D[(mat, "5data", MAIN_DATA)], D[(mat, "3data", MAIN_DATA)]
    print(f"{MATS[mat]}：学習（5data）{len(tr)} 行 {tr['cond'].nunique()} 条件、検証（3data）{len(te)} 行 {te['cond'].nunique()} 条件、感度確認の条件（学習 {int(tr['is_sens'].sum())} 行、検証 {int(te['is_sens'].sum())} 行）")''')

# ---------------------------------------------------------------- 3. 構造の定義
md(r'''## 3. シフト構造の定義と学習の関数

- `design`：データ → (標準化の定数を決めるための入力, 実際の入力, 出力のシフト量)。
  - 構造あり（shift）：入力は (u, w) = (ln γ̇ ＋ ln a(T; E_h), −(ln γ̇ ＋ 3 ln H))。出力のシフト量は ln a(T; E_v)。T は Tset [K]（基準が Tff のときは Tff [K]）。
  - 構造なし（free）：入力は (ln γ̇, ln H, Tset)。シフトなし。
- `fit_variant`・`predict_variant`：学習と予測。標準化・中心化は学習データだけで計算し、E に依存しない定数を使う。''')
code(r'''# セル3：構造の定義、学習・予測の関数
R_GAS, TREF_K = 8.314, 483.15                     # 気体定数 [J/mol/K]、基準温度 210 ℃ [K]
MODE_COLS = {"cm": ("ln_gd_cm", "ln_Tff_cm"), "shot": ("ln_gd", "ln_Tff")}    # cm：条件平均の γ̇、shot：ショットごとの γ̇


def shift_ln(Tk, E):
    """ln a(T; E) = (E/R)(1/T − 1/Tref)。E は kJ/mol、Tk は K"""
    return (E * 1000.0 / R_GAS) * (1.0 / np.asarray(Tk) - 1.0 / TREF_K)


def design(df, v, mode):
    gcol, fcol = MODE_COLS[mode]
    lng = df[gcol].to_numpy()
    if v["kind"] == "free":
        X = np.column_stack([lng, df["ln_H"].to_numpy(), df["Tset"].to_numpy()])
        return X, X, np.zeros(len(df))
    Tk = df["Tset"].to_numpy() + 273.15 if v.get("basis", "Tset") == "Tset" else np.exp(df[fcol].to_numpy())
    w = -(lng + 3.0 * df["ln_H"].to_numpy())                       # ln Fo（シフト前の γ̇ で計算。定数は省く）
    return np.column_stack([lng, w]), np.column_stack([lng + shift_ln(Tk, v["Eh"]), w]), shift_ln(Tk, v["Ev"])


def fit_variant(v, tr, mode, restarts):
    X0, Xs_, ysh = design(tr, v, mode)
    mu_x, sd_x = X0.mean(axis=0), X0.std(axis=0)                   # E に依存しない定数（シフト前の入力の平均・標準偏差）
    y = tr["ln_eta"].to_numpy()
    y0 = y.mean()                                                   # E に依存しない固定の定数
    gp = GaussianProcessRegressor(kernel=kernel_for(X0.shape[1], True), n_restarts_optimizer=restarts, random_state=SEED)
    gp.fit((Xs_ - mu_x) / sd_x, y - ysh - y0)
    return {"gp": gp, "mu_x": mu_x, "sd_x": sd_x, "y0": y0, "v": v, "mode": mode}


def predict_variant(fit, df):
    """予測平均、予測標準偏差（ノイズ込み）、潜在関数の標準偏差、ノイズの標準偏差"""
    X0, Xs_, ysh = design(df, fit["v"], fit["mode"])
    mu, sd = fit["gp"].predict((Xs_ - fit["mu_x"]) / fit["sd_x"], return_std=True)
    noise = fit["gp"].kernel_.k2.noise_level
    return mu + fit["y0"] + ysh, sd, np.sqrt(np.maximum(sd ** 2 - noise, 0.0)), float(np.sqrt(noise))


def cond_table(df):
    """条件ごとの平均（ln η と ln γ̇ と ln Tff）。尤度の比較に使う"""
    g = df.groupby("cond", sort=False)
    t = g.agg(Tset=("Tset", "first"), H=("H", "first"), v=("v", "first"), ln_eta=("ln_eta", "mean"), ln_gd=("ln_gd", "mean"),
              ln_Tff=("ln_Tff", "mean"), n=("ln_eta", "size"), is_sens=("is_sens", "max")).reset_index()
    t["ln_H"], t["ln_gd_cm"], t["ln_Tff_cm"] = np.log(t["H"]), t["ln_gd"], t["ln_Tff"]
    return t


STRUCT = {   # 1 次元の構造：E → 構造の指定
    "H": lambda e: {"kind": "shift", "Eh": e, "Ev": e, "basis": "Tset"},
    "A": lambda e: {"kind": "shift", "Eh": 0.0, "Ev": e, "basis": "Tset"},
    "Hz": lambda e: {"kind": "shift", "Eh": e, "Ev": 0.0, "basis": "Tset"},
    "F": lambda e: {"kind": "shift", "Eh": e, "Ev": e, "basis": "Tff"},
}
STRUCT_JA = {"H": "H：横・縦を同じ E（当初の §5）", "A": "A：縦だけシフト", "Hz": "Hz：横だけシフト", "B": "B：横・縦を別々", "F": "F：Tff でシフト（参考）",
             "N": "N：温度なし（E = 0）", "H40": "H40：E = 40 kJ/mol", "C": "C：通常の GPR［γ̇, H, Tset］"}
STRUCT_COL = {"H": "tab:red", "A": "tab:blue", "Hz": "tab:green", "B": "tab:purple", "F": "tab:orange", "N": "gray", "H40": "tab:brown", "C": "k"}
print("構造:", list(STRUCT_JA))''')

# ---------------------------------------------------------------- 4. プロファイル尤度
md(r'''## 4. プロファイル尤度（条件平均）

各構造について、E の格子の各点で GPR を学習し、対数周辺尤度（LML）を求める。**LML が大きいほど、データに合っている**。学習データは 3 通り：

- **S_V**：5data（190・210・230 ℃、93 条件）
- **S_T**：5data の 190・230 ℃（210 ℃ を除く）
- **ALL**：5data ＋ 3data（129 条件。最終的な E の推定値用）

LML の最大値から 1.92 以内（尤度比検定の 95% 区間にあたる）の E の範囲も求める。B は 2 次元の格子（E_h × E_v）。''')
code(r'''# セル4：プロファイルの計算（並列。数分かかる）。まず粗い格子で全体を見て、最良の点の近くを細かい格子で計算し直す
def prof_job(cdf, v):
    warnings.simplefilter("ignore")                  # 並列の子プロセスでも、下限への張り付きの警告を出さない（kernels.csv に記録する）
    f = fit_variant(v, cdf, "cm", N_RESTARTS_PROFILE)
    return float(f["gp"].log_marginal_likelihood_value_), float(np.sqrt(f["gp"].kernel_.k2.noise_level))


def profile_sets(mat):
    tr_v, _ = split_V(mat, MAIN_DATA)
    tr_t, _, _ = split_T(mat, MAIN_DATA, TEMP_HOLDOUT)
    return {"S_V": tr_v, "S_T": tr_t, "ALL": pool_of(mat, MAIN_DATA)}


def grid_jobs(cdf, mat, sname, grid_1d, structs, eh_grid=(), ev_grid=()):
    """計算の仕事の一覧 [(条件平均のデータ, 構造の指定, (材料, 学習データ, 構造, E, E_h, E_v))]。構造 B は (E_h, E_v) の格子"""
    jl = []
    for st in structs:
        for e in grid_1d:
            e = float(e)
            jl.append((cdf, STRUCT[st](e), (mat, sname, st, e, e if st in ("H", "Hz", "F") else 0.0, e if st in ("H", "A", "F") else 0.0)))
    for eh in eh_grid:
        for ev in ev_grid:
            jl.append((cdf, {"kind": "shift", "Eh": float(eh), "Ev": float(ev), "basis": "Tset"}, (mat, sname, "B", np.nan, float(eh), float(ev))))
    return jl


def run_profile(jl, stage):
    out = Parallel(n_jobs=N_JOBS, batch_size=4)(delayed(prof_job)(c, v) for c, v, _ in jl)
    return pd.DataFrame([{"material": MATS[k[0]], "mat": k[0], "set": k[1], "structure": k[2], "E": k[3], "Eh": k[4], "Ev": k[5], "LML": o[0], "noise_sd": o[1], "stage": stage}
                         for (_, _, k), o in zip(jl, out)])


def best_table(pdf):
    """構造ごとの最良の E（B は E_h と E_v）と、LML が最大から 1.92 以内の範囲"""
    rows = []
    for (mat, sname, st), g in pdf.groupby(["mat", "set", "structure"]):
        r = g.loc[g["LML"].idxmax()]
        h = pdf[(pdf["mat"] == mat) & (pdf["set"] == sname) & (pdf["structure"] == "H") & (pdf["E"] == 0.0)]
        n0 = h["LML"].iloc[0]                                                                 # 温度なし（E = 0）の LML
        inside = g[g["LML"] >= r["LML"] - 1.92]
        row = {"material": MATS[mat], "mat": mat, "set": sname, "structure": st, "LML_best": r["LML"], "LML_at_E0": n0, "dLML_vs_E0": r["LML"] - n0}
        if st == "B":
            row.update({"Eh_best": r["Eh"], "Ev_best": r["Ev"], "n_in_95": len(inside), "Eh_95_lo": inside["Eh"].min(), "Eh_95_hi": inside["Eh"].max(),
                        "Ev_95_lo": inside["Ev"].min(), "Ev_95_hi": inside["Ev"].max()})
        else:
            g40 = g[np.isclose(g["E"], E_LIT)]
            row.update({"E_best": r["E"], "E_95_lo": inside["E"].min(), "E_95_hi": inside["E"].max(), "at_grid_edge": bool(r["E"] in (g["E"].min(), g["E"].max())),
                        "LML_at_E40": g40["LML"].iloc[0] if len(g40) else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def refine(coarse, sets_cdf):
    """粗い格子で最良の点の近くを、細かい格子（1 次元は ±2.5 kJ/mol を 0.5 刻み、B は ±5 kJ/mol を 1 刻み）で計算し直して、まとめる"""
    best0, jl = best_table(coarse), []
    for _, r in best0.iterrows():
        cdf = sets_cdf[(r["mat"], r["set"])]
        if r["structure"] == "B":
            jl += grid_jobs(cdf, r["mat"], r["set"], [], [], eh_grid=r["Eh_best"] + np.arange(-5.0, 5.01, 1.0), ev_grid=r["Ev_best"] + np.arange(-5.0, 5.01, 1.0))
        else:
            jl += grid_jobs(cdf, r["mat"], r["set"], r["E_best"] + np.arange(-2.5, 2.51, 0.5), [r["structure"]])
    fine = run_profile(jl, "細かい格子")
    allp = pd.concat([coarse, fine], ignore_index=True)
    return allp.drop_duplicates(subset=["mat", "set", "structure", "Eh", "Ev"], keep="first").reset_index(drop=True)


SETS = {(mat, sname): cond_table(tr) for mat in MATS for sname, tr in profile_sets(mat).items()}
t0 = time.time()
jl = [j for (mat, sname), cdf in SETS.items() for j in grid_jobs(cdf, mat, sname, E_GRID, ["H", "A", "Hz", "F"], EH_GRID, EV_GRID)]
coarse = run_profile(jl, "粗い格子")
print(f"粗い格子：{len(jl)} 点、{(time.time() - t0) / 60:.1f} 分")
prof_df = refine(coarse, SETS)
print(f"細かい格子を足して、全部で {len(prof_df)} 点、{(time.time() - t0) / 60:.1f} 分")
prof_df.drop(columns="mat").to_csv(os.path.join(OUT_DIR, "profile_likelihood.csv"), index=False, encoding="utf-8-sig")''')

code(r'''# セル5：最良の E と、その区間
best_df = best_table(prof_df)
best_df.drop(columns="mat").to_csv(os.path.join(OUT_DIR, "best_E.csv"), index=False, encoding="utf-8-sig")
print("【最良の E（kJ/mol）と LML】E_95_lo〜hi：LML が最大から 1.92 以内の範囲。dLML_vs_E0：温度なし（E = 0）との LML の差（正なら、シフトのほうがデータに合う）")
print(best_df.drop(columns="mat").round(2).to_string(index=False))''')

# ---------------------------------------------------------------- 5. 感度確認（P02）
md(r'''## 5. 感度確認：0.5 mm・20 mm/s・190 ℃（P02）を除いた場合

問題データ一覧の P02 は、他の温度と傾向が逆になっている条件。学習から除いて、H と A の最良の E がどれだけ動くかを見る（S_V の学習）。**粗い格子 1 目（2.5 kJ/mol）以上動いたら、影響が大きい**と判断する。''')
code(r'''# セル6：P02 を除いたプロファイル
SETS2 = {(mat, "S_V（P02 を除く）"): cond_table(split_V(mat, MAIN_DATA)[0].pipe(lambda t: t[~t["is_sens"]])) for mat in MATS}
jl2 = [j for (mat, sname), cdf in SETS2.items() for j in grid_jobs(cdf, mat, sname, E_GRID, ["H", "A"])]
sens_prof = refine(run_profile(jl2, "粗い格子"), SETS2)
best2 = best_table(sens_prof)
srows = []
for _, r in best2.iterrows():
    e_all = float(best_df[(best_df["mat"] == r["mat"]) & (best_df["set"] == "S_V") & (best_df["structure"] == r["structure"])]["E_best"].iloc[0])
    srows.append({"material": r["material"], "structure": r["structure"], "E_best_すべて": e_all, "E_best_P02を除く": float(r["E_best"]), "動き [kJ/mol]": float(r["E_best"]) - e_all,
                  "影響が大きい": abs(float(r["E_best"]) - e_all) >= 2.5 - 1e-9})
sens_df = pd.DataFrame(srows)
sens_prof.drop(columns="mat").to_csv(os.path.join(OUT_DIR, "profile_likelihood_P02_excluded.csv"), index=False, encoding="utf-8-sig")
sens_df.to_csv(os.path.join(OUT_DIR, "sensitivity_E.csv"), index=False, encoding="utf-8-sig")
print(sens_df.round(2).to_string(index=False))
for mat in MATS:
    s = sens_df[sens_df["material"] == MATS[mat]]
    judge(not s["影響が大きい"].any(), f"{MATS[mat]}：P02 を除いても、H・A の最良の E が 2.5 kJ/mol（粗い格子 1 目）以上は動かない（動き {s['動き [kJ/mol]'].round(1).tolist()}）。動いたなら、P02 だけを除いたファイルを作る")''')

# ---------------------------------------------------------------- 6. 予測の評価
md(r'''## 6. 予測の評価（S_V・S_T）

4 の最良の E（その学習データのもの）を使い、ショット単位の GPR を学習して、検証データを予測する。S_V の学習には S_V の E、S_T の学習には S_T の E を使う（検証データの情報は使わない）。

- 構造：C・N・H・H40・A・Hz・B・F（上の表）
- 入力の γ̇：`cm`（条件平均。成形条件から見た予測）と `shot`（ショットごと）
- 評価：S_V（低速側・高速側）、S_T（5data の 210 ℃）、S_T3（3data の 210 ℃：速度の外挿も重なる）。「（感度確認の条件を除く）」は、検証の P03（PP の 190 ℃・0.5 mm・320 mm/s）を評価から除いた指標。''')
code(r'''# セル7：最良の E で学習し直して、予測を評価する
def e_of(mat, sname, st):
    r = best_df[(best_df["mat"] == mat) & (best_df["set"] == sname) & (best_df["structure"] == st)].iloc[0]
    return r


def variant_defs(mat, sname):
    eH, eA, eZ, eF, rB = e_of(mat, sname, "H")["E_best"], e_of(mat, sname, "A")["E_best"], e_of(mat, sname, "Hz")["E_best"], e_of(mat, sname, "F")["E_best"], e_of(mat, sname, "B")
    return {"C": {"kind": "free"}, "N": STRUCT["H"](0.0), "H": STRUCT["H"](float(eH)), "H40": STRUCT["H"](E_LIT), "A": STRUCT["A"](float(eA)),
            "Hz": STRUCT["Hz"](float(eZ)), "B": {"kind": "shift", "Eh": float(rB["Eh_best"]), "Ev": float(rB["Ev_best"]), "basis": "Tset"}, "F": STRUCT["F"](float(eF))}


def fit_job(tr, v, mode):
    warnings.simplefilter("ignore")
    return fit_variant(v, tr, mode, N_RESTARTS_FINAL)


tasks, tkeys = [], []
for mat in MATS:
    for split in ["S_V", "S_T"]:
        tr = split_V(mat, MAIN_DATA)[0] if split == "S_V" else split_T(mat, MAIN_DATA, TEMP_HOLDOUT)[0]
        for name, v in variant_defs(mat, split).items():
            for mode in ["cm", "shot"]:
                tasks.append((tr, v, mode))
                tkeys.append((mat, split, name, mode))
t0 = time.time()
fits = Parallel(n_jobs=N_JOBS)(delayed(fit_job)(*t) for t in tasks)
FIT5 = dict(zip(tkeys, fits))
print(f"{len(fits)} 個のモデルを学習。{(time.time() - t0) / 60:.1f} 分")

rows, krows = [], []
for (mat, split, name, mode), fit in FIT5.items():
    tests = {"S_V": {"S_V": split_V(mat, MAIN_DATA)[1]}, "S_T": {"S_T": split_T(mat, MAIN_DATA, TEMP_HOLDOUT)[1], "S_T3": split_T(mat, MAIN_DATA, TEMP_HOLDOUT)[2]}}[split]
    for tname, te in tests.items():
        subsets = {"全体": np.ones(len(te), bool)}
        for g in te["group"].unique():
            if g != "全体":
                subsets[g] = (te["group"] == g).to_numpy()
        if te["is_sens"].any():
            subsets.update({s + SENS_SUFFIX: m & ~te["is_sens"].to_numpy() for s, m in list(subsets.items())})
        mu, sd, sd_f, noise_sd = predict_variant(fit, te)
        y = te["ln_eta"].to_numpy()
        for sub, mask in subsets.items():
            if mask.sum():
                rows.append({"material": MATS[mat], "split": tname, "mode": mode, "variant": name, "subset": sub, "n_test": int(mask.sum()), **evaluate(y[mask], mu[mask], sd[mask]), "noise_sd": noise_sd})
    v = fit["v"]
    k = fit["gp"].kernel_
    krows.append({"material": MATS[mat], "split": split, "mode": mode, "variant": name, "Eh": v.get("Eh", np.nan), "Ev": v.get("Ev", np.nan), "LML": float(fit["gp"].log_marginal_likelihood_value_),
                  "noise_sd": float(np.sqrt(k.k2.noise_level)), "noise_at_lower_bound": bool(k.k2.noise_level < NOISE_LB * 1.05)})
metrics5 = pd.DataFrame(rows)
kern5 = pd.DataFrame(krows)
metrics5.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False, encoding="utf-8-sig")
kern5.to_csv(os.path.join(OUT_DIR, "kernels.csv"), index=False, encoding="utf-8-sig")
VARS = ["C", "N", "H", "H40", "A", "Hz", "B", "F"]


def m5(mat, split, variant, sub, col="RMSE", mode="cm"):
    r = metrics5[(metrics5["material"] == MATS[mat]) & (metrics5["split"] == split) & (metrics5["mode"] == mode) & (metrics5["variant"] == variant) & (metrics5["subset"] == sub)]
    return float(r[col].iloc[0]) if len(r) else np.nan


for mode in ["cm", "shot"]:
    for mat in MATS:
        print(f"\n【{MATS[mat]}】入力の γ̇ = {'条件平均（cm）' if mode == 'cm' else 'ショットごと（shot）'}。RMSE（ln η）／NLPD")
        cols = [("S_V", "低速側"), ("S_V", "高速側"), ("S_T", "全体"), ("S_T3", "全体")]
        t = pd.DataFrame({f"{s}{'' if sub == '全体' else ' ' + sub}": {v: f"{m5(mat, s, v, sub, 'RMSE', mode):.3f}／{m5(mat, s, v, sub, 'NLPD', mode):.2f}" for v in VARS} for s, sub in cols})
        print(t.to_string())''')

# ---------------------------------------------------------------- 7. 設定温度の効果
md(r'''## 7. 設定温度の効果

実測と、各構造の予測を比べる（3b の図5と同じ見方）。

- **実測の比**：同じ厚み・同じ γ̇ で、190 ℃ の粘度 ÷ 230 ℃ の粘度（感度確認の条件は除く）。
- **モデルの比**：5data＋3data の 210 ℃ のショットについて、ほかの入力はそのまま、Tset だけを 190 ℃・230 ℃ にして予測した粘度の比（S_V の学習、cm）。N と F は Tset に反応しないので、比が 1 になる。''')
code(r'''# セル8：設定温度の効果
H_LEVELS = [0.5, 1.0, 1.5, 2.0]


def observed_ratio(mat):
    d = pool_of(mat, MAIN_DATA)
    d = d[~d["is_sens"]]
    cm = d.groupby(["Tset", "H", "v"]).agg(lg=("ln_gd", "mean"), le=("ln_eta", "mean")).reset_index()
    rows = []
    for h in H_LEVELS:
        a, c = cm[(cm["H"] == h) & (cm["Tset"] == 190)], cm[(cm["H"] == h) & (cm["Tset"] == 230)].sort_values("lg")
        ins = ((a["lg"] >= c["lg"].min()) & (a["lg"] <= c["lg"].max())).to_numpy()
        r = np.exp(a["le"].to_numpy()[ins] - np.interp(a["lg"].to_numpy()[ins], c["lg"], c["le"]))
        rows.append({"material": MATS[mat], "H": h, "observed": float(np.median(r)), "n": int(ins.sum())})
    return pd.DataFrame(rows)


def model_ratio(split="S_V", mode="cm"):
    rows = []
    for mat in MATS:
        base = pool_of(mat, MAIN_DATA)
        base = base[(base["Tset"] == 210) & ~base["is_sens"]]
        for name in VARS:
            fit = FIT5[(mat, split, name, mode)]
            lo, hi = base.copy(), base.copy()
            lo["Tset"], hi["Tset"] = 190.0, 230.0
            r = np.exp(predict_variant(fit, lo)[0] - predict_variant(fit, hi)[0])
            for h, g in base.assign(r=r).groupby("H"):
                rows.append({"material": MATS[mat], "split": split, "mode": mode, "variant": name, "H": h, "ratio_190_230": float(np.median(g["r"]))})
    return pd.DataFrame(rows)


obs5 = pd.concat([observed_ratio(m) for m in MATS], ignore_index=True)
eff5 = pd.concat([model_ratio("S_V"), model_ratio("S_T")], ignore_index=True)
obs5.to_csv(os.path.join(OUT_DIR, "observed_ratio.csv"), index=False, encoding="utf-8-sig")
eff5.to_csv(os.path.join(OUT_DIR, "tset_effect.csv"), index=False, encoding="utf-8-sig")
print("実測の 190/230 ℃ の粘度比（同じ γ̇、厚みごと）")
print(obs5.pivot(index="material", columns="H", values="observed").round(3).to_string())
for sp in ["S_V", "S_T"]:
    print(f"\nモデルの比（{sp} の学習、Tset だけを 190 → 230 ℃）")
    print(eff5[eff5["split"] == sp].pivot_table(index=["material", "variant"], columns="H", values="ratio_190_230").reindex(VARS, level=1).round(3).to_string())''')

# ---------------------------------------------------------------- 8. 本命と対案の比較
md(r'''## 8. 本命と対案の比較（ユーザー決定、2026-10-12）

本命（`MAIN_STRUCT`、既定は B）と対案（`ALT_STRUCT`、既定は A）を、基準 C（構造なしの通常の GPR）と並べて比べる。後で自分で実行して、どちらを使うか判断するための節。

- **E**：S_V・S_T・ALL の学習での最良の E（E_h・E_v。A は E_h = 0、H は E_h = E_v）。学習データを変えても動かないか。
- **LML**：S_V の学習での、最良の E の対数周辺尤度と、温度なし（E = 0）との差。**大きいほどデータに合う**。差が 2 以上なら、はっきりした違い。
- **検証の指標**：RMSE（ln η）・95% 被覆率・NLPD。入力の γ̇ は条件平均（cm）。S_V の低速側・高速側、S_T（210 ℃）、S_T3（3data の 210 ℃）。
- **設定温度の効果**：190 ℃ と 230 ℃ の粘度の比（同じ γ̇。厚みごと）と、実測。

結果は `main_vs_alt.csv` に保存する。''')
code(r'''# セル13.5：本命と対案の比較
PANELS5 = [("S_V", "低速側", "S_V_low"), ("S_V", "高速側", "S_V_high"), ("S_T", "全体", "S_T"), ("S_T3", "全体", "S_T3")]
ROLES = [("本命", MAIN_STRUCT), ("対案", ALT_STRUCT), ("基準", "C")]
cmp_rows = []
for mat in MATS:
    for role, name in ROLES:
        row = {"material": MATS[mat], "role": role, "structure": name}
        for sname in ["S_V", "S_T", "ALL"]:
            v = variant_defs(mat, sname)[name]
            row[f"Eh_{sname}"], row[f"Ev_{sname}"] = v.get("Eh", np.nan), v.get("Ev", np.nan)
        b = best_df[(best_df["mat"] == mat) & (best_df["set"] == "S_V") & (best_df["structure"] == name)]
        row["LML_S_V"], row["dLML_vs_E0_S_V"] = (float(b["LML_best"].iloc[0]), float(b["dLML_vs_E0"].iloc[0])) if len(b) else (np.nan, np.nan)
        for split, sub, lab in PANELS5:
            for metric in ["RMSE", "coverage95", "NLPD"]:
                row[f"{metric}_{lab}"] = m5(mat, split, name, sub, metric)
        e = eff5[(eff5["split"] == "S_V") & (eff5["material"] == MATS[mat]) & (eff5["variant"] == name)].sort_values("H")
        for h, r in zip(e["H"], e["ratio_190_230"]):
            row[f"ratio_{h}"] = float(r)
        cmp_rows.append(row)
    o = obs5[obs5["material"] == MATS[mat]].sort_values("H")
    cmp_rows.append({"material": MATS[mat], "role": "実測", "structure": "―", **{f"ratio_{h}": float(r) for h, r in zip(o["H"], o["observed"])}})
cmp_df = pd.DataFrame(cmp_rows)
cmp_df.to_csv(os.path.join(OUT_DIR, "main_vs_alt.csv"), index=False, encoding="utf-8-sig")
RATIO_COLS = [f"ratio_{h}" for h in H_LEVELS]
for mat in MATS:
    t = cmp_df[cmp_df["material"] == MATS[mat]].set_index("role")
    print(f"\n【{MATS[mat]}】本命 {MAIN_STRUCT}、対案 {ALT_STRUCT}、基準 C")
    print("E [kJ/mol]（E_h／E_v。S_V・S_T・ALL の学習）")
    print(t.loc[[r for r, _ in ROLES], [f"{p}_{s}" for s in ["S_V", "S_T", "ALL"] for p in ["Eh", "Ev"]]].round(1).to_string())
    print("LML（S_V の学習）と、検証の RMSE")
    print(t.loc[[r for r, _ in ROLES], ["LML_S_V", "dLML_vs_E0_S_V"] + [f"RMSE_{lab}" for _, _, lab in PANELS5]].round(3).to_string())
    print("190/230 ℃ の粘度の比（同じ γ̇。厚みごと。実測を含む）")
    print(t.loc[[r for r, _ in ROLES] + ["実測"], RATIO_COLS].round(3).to_string())
for mat in MATS:
    n = MATS[mat]
    t = cmp_df[cmp_df["material"] == n].set_index("role")
    lm, la = t.loc["本命", "LML_S_V"], t.loc["対案", "LML_S_V"]
    if np.isnan(lm) or np.isnan(la):
        print(f"（{n}：本命か対案が N・H40・C なので、LML は比べない）")
    else:
        judge(lm - la >= 2.0, f"{n}：本命 {MAIN_STRUCT} の LML（{lm:.1f}）が、対案 {ALT_STRUCT}（{la:.1f}）より 2 以上大きい（差 {lm - la:.1f}）→ 本命のほうがデータに合う")
    bad = [lab for _, _, lab in PANELS5 if t.loc["本命", f"RMSE_{lab}"] > 1.05 * t.loc["対案", f"RMSE_{lab}"]]
    judge(not bad, f"{n}：本命の検証の RMSE が、対案の 1.05 倍以内（超えた検証: {bad or 'なし'}）。4 つの検証：" + "、".join(f"{lab} {t.loc['本命', f'RMSE_{lab}']:.3f}／{t.loc['対案', f'RMSE_{lab}']:.3f}" for _, _, lab in PANELS5))
    bad = [lab for _, _, lab in PANELS5 if t.loc["本命", f"RMSE_{lab}"] > 1.1 * t.loc["基準", f"RMSE_{lab}"]]
    judge(not bad, f"{n}：本命の検証の RMSE が、基準 C の 1.1 倍以内（超えた検証: {bad or 'なし'}。同等なら、構造がデータで支持されている）")
    dev = {r: float(np.abs(t.loc[r, RATIO_COLS].astype(float).to_numpy() - t.loc["実測", RATIO_COLS].astype(float).to_numpy()).max()) for r in ["本命", "対案"]}
    judge(dev["本命"] <= dev["対案"], f"{n}：190/230 ℃ の粘度比が、本命のほうが実測に近い（実測との差の最大：本命 {dev['本命']:.3f}、対案 {dev['対案']:.3f}）。対案が A のように厚みによらず一定だと、厚みごとの違いを表せない")
    for role in ["本命", "対案"]:
        spread = {c: float(t.loc[role, [f"{c}_S_V", f"{c}_S_T", f"{c}_ALL"]].astype(float).max() - t.loc[role, [f"{c}_S_V", f"{c}_S_T", f"{c}_ALL"]].astype(float).min()) for c in ["Eh", "Ev"]}
        if not any(np.isnan(v) for v in spread.values()):
            judge(max(spread.values()) <= 5.0, f"{n}：{role}（{t.loc[role, 'structure']}）の E が、学習データ（S_V・S_T・ALL）を変えても 5 kJ/mol 以内（E_h の動き {spread['Eh']:.1f}、E_v の動き {spread['Ev']:.1f}）")''')

# ---------------------------------------------------------------- 9. 図
md(r'''## 9. 図

- 図0：構造の考え方の模式図（実データではない）
- 図1：プロファイル尤度（E と LML の関係）。構造ごと・学習データごと
- 図2：B の 2 次元プロファイル（E_h × E_v）。材料ごとに 1 枚（2a：PP、2b：粉砕材混合 PP）
- 図3：マスターカーブ（換算後のデータ）。重なっていれば、その構造でシフトできている
- 図4：設定温度ごとの予測曲線（構造ごと）
- 図5：評価指標の比較
- 図6：設定温度の効果（190/230 ℃ の粘度比）''')
code(r'''# セル8.9：図0 構造の考え方（模式図。実データではない）
E_DEMO = {"H": (30.0, 30.0), "A": (0.0, 30.0), "Hz": (30.0, 0.0), "B": (15.0, 30.0)}      # (E_h, E_v) [kJ/mol]。形が見やすいように、実測より大きな値にした例
T0_COL = {190: "tab:blue", 210: "tab:green", 230: "tab:red"}
TK0 = {190: 463.15, 210: 483.15, 230: 503.15}
ug = np.linspace(np.log(3.0), np.log(600000.0), 1500)                                      # ln γ̇（基準の 210 ℃ の曲線を描く格子）
slope0 = -0.4 - 1.5 / (1.0 + np.exp(-(ug - np.log(3000.0)) / 1.2))                         # 基準の曲線の傾き：低せん断で −0.4、高せん断で −1.9（見かけの流動曲線は、せん断速度とともに急になる）
f_ref = np.r_[0.0, np.cumsum((slope0[1:] + slope0[:-1]) / 2 * np.diff(ug))] + np.log(2000.0)    # ln η の基準の曲線（任意の単位）
xs0 = np.linspace(np.log(30.0), np.log(60000.0), 300)
fig, axes = plt.subplots(2, len(E_DEMO), figsize=(5.0 * len(E_DEMO), 9.5), sharex=True)
for j, (name, (eh, ev)) in enumerate(E_DEMO.items()):
    curves0 = {t: shift_ln(TK0[t], ev) + np.interp(xs0 + shift_ln(TK0[t], eh), ug, f_ref) for t in TK0}          # ln η = ln a(E_v) + f(ln γ̇ + ln a(E_h))
    for t, col in T0_COL.items():
        axes[0, j].plot(np.exp(xs0), np.exp(curves0[t]), color=col, lw=2, label=f"{t} ℃")
    axes[0, j].set_xscale("log")
    axes[0, j].set_yscale("log")
    axes[0, j].set_title(f"{STRUCT_JA[name]}\nE_h = {eh:g}、E_v = {ev:g} kJ/mol", fontsize=10)
    axes[0, j].grid(alpha=0.3, which="both")
    axes[1, j].plot(np.exp(xs0), np.exp(curves0[190] - curves0[230]), color="k", lw=2)
    axes[1, j].axhline(1.0, color="gray", lw=0.8, ls="--")
    axes[1, j].set_xscale("log")
    axes[1, j].set_yscale("log")
    axes[1, j].set_ylim(0.25, 3.5)
    axes[1, j].set_yticks([0.3, 0.5, 0.7, 1, 1.5, 2, 3])
    axes[1, j].set_yticklabels(["0.3", "0.5", "0.7", "1", "1.5", "2", "3"])
    axes[1, j].yaxis.set_minor_formatter(plt.NullFormatter())
    axes[1, j].set_xlabel(r"せん断速度 $\dot{\gamma}$ [1/s]")
    axes[1, j].grid(alpha=0.3, which="both")
axes[0, 0].set_ylabel(r"粘度 $\eta$（任意の単位）")
axes[1, 0].set_ylabel("粘度の比（190 ℃ ÷ 230 ℃）")
axes[0, 0].legend(fontsize=9)
fig.suptitle("構造の考え方の模式図（実データではない。E は、形が見やすいように大きめの例）。上段：温度ごとの流動曲線、下段：同じ γ̇ での粘度の比（190 ℃ ÷ 230 ℃）\n"
             "H は、傾きが −1 より急な所で比が 1 を下回る（向きが逆になる）。A は比が一定。B は、A の一定の比に、場所で変わる分が加わる", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(OUT_DIR, "fig0_shift_concept.png"), dpi=FIG_DPI)
plt.show()''')
code(r'''# セル9：図1 プロファイル尤度（1 次元）
SET_JA = {"S_V": "S_V の学習（5data）", "S_T": "S_T の学習（190・230 ℃）", "ALL": "ALL（5data＋3data）"}
fig, axes = plt.subplots(len(MATS), 3, figsize=(17, 9), sharex=True)
for i, mat in enumerate(MATS):
    for j, sname in enumerate(["S_V", "S_T", "ALL"]):
        ax = axes[i, j]
        for st in ["H", "A", "Hz", "F"]:
            g = prof_df[(prof_df["mat"] == mat) & (prof_df["set"] == sname) & (prof_df["structure"] == st)].sort_values("E")
            ax.plot(g["E"], g["LML"], "-o", ms=2.5, lw=1.2, color=STRUCT_COL[st], label=STRUCT_JA[st])
            b = g.loc[g["LML"].idxmax()]
            ax.plot([b["E"]], [b["LML"]], "*", ms=14, color=STRUCT_COL[st], mec="k")
        ax.axvline(0, color="gray", lw=0.8)
        ax.axvline(E_LIT, color="k", ls=":", lw=1, label="文献値の目安 40 kJ/mol")
        ax.set_title(f"{MATS[mat]}：{SET_JA[sname]}", fontsize=10)
        ax.set_xlabel("E [kJ/mol]")
        ax.set_ylabel("対数周辺尤度（大きいほど良い）")
        ax.grid(alpha=0.3)
        top = prof_df[(prof_df["mat"] == mat) & (prof_df["set"] == sname) & prof_df["structure"].isin(["H", "A", "Hz", "F"])]["LML"].max()
        ax.set_ylim(top - 130, top + 10)
        if i == 0 and j == 0:
            ax.legend(fontsize=7, loc="lower right")
fig.suptitle("プロファイル尤度（条件平均）：★ が最良の E。E = 0 が温度なし。A・Hz は鋭い山、H・F は平らで、折れ線がでこぼこ（最適化が局所解に止まった点を含む）。B は図2", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_profile_likelihood.png"), dpi=FIG_DPI)
plt.show()''')

code(r'''# セル10：図2 B の 2 次元プロファイル（材料ごとに 1 枚。上段：粗い格子の全体、下段：最良の点の近くの細かい格子）
for mat in MATS:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for j, sname in enumerate(["S_V", "S_T", "ALL"]):
        g = prof_df[(prof_df["mat"] == mat) & (prof_df["set"] == sname) & (prof_df["structure"] == "B")]
        b = g.loc[g["LML"].idxmax()]
        top = b["LML"]
        for i, (sub, ttl) in enumerate([(g[g["stage"] == "粗い格子"], "全体（粗い格子。5 kJ/mol 刻み）"),
                                         (g[(abs(g["Eh"] - b["Eh"]) <= 5.001) & (abs(g["Ev"] - b["Ev"]) <= 5.001)], "最良の点の近く（細かい格子。1 kJ/mol 刻み）")]):
            ax = axes[i, j]
            Z = sub.pivot(index="Ev", columns="Eh", values="LML")
            im = ax.pcolormesh(Z.columns, Z.index, np.clip(Z.to_numpy(), top - 40, top), shading="nearest", cmap="viridis")
            if i == 1:
                ax.contour(Z.columns, Z.index, Z.to_numpy(), levels=[top - 1.92], colors="w", linewidths=1.2)
            ax.plot([b["Eh"]], [b["Ev"]], "*", ms=14, color="r", mec="w", label=f"最良（E_h = {b['Eh']:.0f}、E_v = {b['Ev']:.0f}）")
            lim = [min(Z.columns.min(), Z.index.min()), max(Z.columns.max(), Z.index.max())]
            ax.plot(lim, lim, "w--", lw=0.8, label="E_h = E_v（H）")
            ax.axvline(0, color="w", lw=0.8, ls=":")
            ax.axhline(0, color="w", lw=0.8, ls=":")
            ax.set_xlim(Z.columns.min(), Z.columns.max())
            ax.set_ylim(Z.index.min(), Z.index.max())
            ax.set_title(f"{MATS[mat]}：{SET_JA[sname]}\n{ttl}", fontsize=9)
            ax.set_xlabel("横方向 E_h [kJ/mol]")
            ax.set_ylabel("縦方向 E_v [kJ/mol]")
            ax.legend(fontsize=7, loc="upper left")
            fig.colorbar(im, ax=ax, label="対数周辺尤度")
    fig.suptitle(f"{MATS[mat]}：B（横・縦を別々）の対数周辺尤度。白の輪郭：最大から 1.92 以内。破線：H（E_h = E_v）、点線：A（E_h = 0）・Hz（E_v = 0）", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"fig2{'a' if mat == 'PP' else 'b'}_profile_2d_B_{mat}.png"), dpi=FIG_DPI)
    plt.show()''')

code(r'''# セル11：図3 マスターカーブ（ALL の学習データを、構造ごとの最良の E で換算）
T_COL = {190: "tab:blue", 210: "tab:green", 230: "tab:red"}
MASTER = [("N", "N：温度なし"), ("H", "H：横・縦同じ E"), ("A", "A：縦だけ"), ("B", "B：横・縦別々")]
for mat in MATS:
    pool = pool_of(mat, MAIN_DATA)
    cdf = cond_table(pool)
    defs = variant_defs(mat, "ALL")
    fig, axes = plt.subplots(4, len(MASTER), figsize=(5.0 * len(MASTER), 14), sharey="row")
    for j, (name, lab) in enumerate(MASTER):
        v = defs[name]
        X0, Xs_, ysh = design(cdf, v, "cm")
        for i, h in enumerate(H_LEVELS):
            ax = axes[i, j]
            for t, col in T_COL.items():
                m = ((cdf["H"] == h) & (cdf["Tset"] == t)).to_numpy()
                ax.scatter(Xs_[m, 0], cdf["ln_eta"].to_numpy()[m] - ysh[m], s=16, color=col, alpha=0.8, label=f"{t} ℃")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(f"厚み {h} mm\n" + r"換算後の $\ln\eta$（縦シフトを引いた値）")
            if i == 0:
                ax.set_title(f"{lab}\n(E_h, E_v) = ({v['Eh']:.1f}, {v['Ev']:.1f}) kJ/mol", fontsize=10)
            if i == 3:
                ax.set_xlabel(r"換算後の $\ln\dot{\gamma}$ ＋ 横シフト")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{MATS[mat]}：マスターカーブ（条件平均。3 つの温度の点が重なっていれば、その構造でシフトできている）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(OUT_DIR, f"fig3{'a' if mat == 'PP' else 'b'}_master_curve_{mat}.png"), dpi=FIG_DPI)
    plt.show()''')

code(r'''# セル12：図4 設定温度ごとの予測曲線（S_V の学習、cm。列が構造、行が厚み）
CURVE_VARS = ["C", "N", "A", "B", "H"]
for mat in MATS:
    allp = pool_of(mat, MAIN_DATA)
    te_cond = set(D[(mat, "3data", MAIN_DATA)]["cond"])
    fig, axes = plt.subplots(4, len(CURVE_VARS), figsize=(4.6 * len(CURVE_VARS), 14), sharex="row", sharey="row")
    for j, name in enumerate(CURVE_VARS):
        fit = FIT5[(mat, "S_V", name, "cm")]
        for i, h in enumerate(H_LEVELS):
            ax = axes[i, j]
            for t, col in T_COL.items():
                sub = allp[(allp["H"] == h) & (allp["Tset"] == t)]
                if sub.empty:
                    continue
                cm = cond_table(sub).sort_values("ln_gd")
                lg = np.linspace(sub["ln_gd"].min(), sub["ln_gd"].max(), 120)
                Xg = pd.DataFrame({"ln_gd": lg, "ln_gd_cm": lg, "ln_H": np.log(h), "Tset": float(t), "ln_Tff": np.interp(lg, cm["ln_gd"], cm["ln_Tff"]), "ln_Tff_cm": np.interp(lg, cm["ln_gd"], cm["ln_Tff"])})
                mu, sd, _, _ = predict_variant(fit, Xg)
                ax.plot(np.exp(lg), np.exp(mu), color=col, lw=1.4)
                ax.fill_between(np.exp(lg), np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd), color=col, alpha=0.12)
                is_te = np.array([c in te_cond for c in cm["cond"]])
                ax.scatter(np.exp(cm["ln_gd"][~is_te]), np.exp(cm["ln_eta"][~is_te]), s=14, color=col, alpha=0.8, zorder=3)
                ax.scatter(np.exp(cm["ln_gd"][is_te]), np.exp(cm["ln_eta"][is_te]), s=26, facecolors="none", edgecolors=col, marker="s", linewidths=1.2, zorder=3)
                sens = cm["is_sens"].to_numpy().astype(bool)
                ax.scatter(np.exp(cm["ln_gd"][sens]), np.exp(cm["ln_eta"][sens]), s=45, color="k", marker="x", zorder=4)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
            if j == 0:
                ax.set_ylabel(f"厚み {h} mm\n" + r"$\eta$ [Pa·s]")
            if i == 0:
                v = fit["v"]
                ax.set_title(STRUCT_JA[name] + ("" if v["kind"] == "free" else f"\n(E_h, E_v) = ({v['Eh']:.1f}, {v['Ev']:.1f})"), fontsize=9)
            if i == 3:
                ax.set_xlabel(r"$\dot{\gamma}$ [1/s]")
    handles = [plt.Line2D([], [], color=c, label=f"{t} ℃") for t, c in T_COL.items()]
    handles += [plt.Line2D([], [], color="gray", marker="o", ls="", label="学習の条件平均"), plt.Line2D([], [], color="gray", marker="s", mfc="none", ls="", label="検証の条件平均"),
                plt.Line2D([], [], color="k", marker="x", ls="", label="感度確認の条件")]
    axes[0, 0].legend(handles=handles, fontsize=7, loc="upper right")
    fig.suptitle(f"{MATS[mat]}：設定温度ごとの予測曲線（5data で学習。線：予測、帯：95% 予測区間）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(OUT_DIR, f"fig4{'a' if mat == 'PP' else 'b'}_curves_{mat}.png"), dpi=FIG_DPI)
    plt.show()''')

code(r'''# セル13：図5 評価指標の比較、図6 設定温度の効果
PANELS = [("S_V", "低速側", "S_V 低速側"), ("S_V", "高速側", "S_V 高速側"), ("S_T", "全体", f"S_T（{TEMP_HOLDOUT} ℃）"), ("S_T3", "全体", f"S_T3（3data の {TEMP_HOLDOUT} ℃）")]
COLORS = ["tab:blue", "tab:red", "tab:purple", "tab:brown"]
fig, axes = plt.subplots(2, len(MATS), figsize=(16, 9))
for j, mat in enumerate(MATS):
    for i, (metric, ylab) in enumerate([("RMSE", r"RMSE（$\ln\eta$）"), ("NLPD", "NLPD（小さいほど良い）")]):
        ax = axes[i, j]
        w = 0.8 / len(PANELS)
        for k, (split, sub, lab) in enumerate(PANELS):
            ax.bar(np.arange(len(VARS)) + (k - (len(PANELS) - 1) / 2) * w, [m5(mat, split, v, sub, metric) for v in VARS], w, color=COLORS[k], label=lab)
        ax.set_xticks(np.arange(len(VARS)))
        ax.set_xticklabels(VARS)
        ax.set_title(f"{MATS[mat]}：{ylab}（入力の " + r"$\dot{\gamma}$" + " は条件平均）")
        ax.grid(alpha=0.3, axis="y")
        if metric == "RMSE":
            ax.set_yscale("log")
        if i == 0 and j == 0:
            ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig5_metrics.png"), dpi=FIG_DPI)
plt.show()

fig, axes = plt.subplots(1, len(MATS), figsize=(15, 5.2), sharey=True)
MK = {"C": "s", "N": "x", "H": "o", "H40": "^", "A": "v", "Hz": "D", "B": "P", "F": "X"}
for ax, mat in zip(axes, MATS):
    o = obs5[obs5["material"] == MATS[mat]]
    ax.plot(o["H"], o["observed"], "k-o", lw=2.5, ms=9, label=r"実測（同じ $\dot{\gamma}$）", zorder=10)
    for name in VARS:
        e = eff5[(eff5["split"] == "S_V") & (eff5["material"] == MATS[mat]) & (eff5["variant"] == name)]
        ax.plot(e["H"], e["ratio_190_230"], marker=MK[name], ls="--", alpha=0.85, color=STRUCT_COL[name], label=name)
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xlabel("厚み [mm]")
    ax.set_title(f"{MATS[mat]}：190 ℃ と 230 ℃ の粘度の比（Tset だけを変えた予測、S_V の学習）")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("粘度の比（190 ℃ ÷ 230 ℃）")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig6_temperature_effect.png"), dpi=FIG_DPI)
plt.show()''')

# ---------------------------------------------------------------- 9. 判定
md(r'''## 10. 判定のまとめ

OK は想定どおり、★ は想定と違う、または注意が要る点。基準は、判定の文の中に書いてある。''')
code(r'''# セル14：判定
def bE(mat, sname, st, col="E_best"):
    return float(e_of(mat, sname, st)[col])


print("【判定】")
for mat in MATS:
    n = MATS[mat]
    eH = bE(mat, "S_V", "H")
    dA, dH = float(e_of(mat, "S_V", "A")["dLML_vs_E0"]), float(e_of(mat, "S_V", "H")["dLML_vs_E0"])
    judge(dA - dH >= 20.0, f"{n}：A の LML の改善（温度なし E = 0 との差）{dA:.1f} が、H の改善 {dH:.1f} より 20 以上大きい（H は、E = −20〜12.5 でほぼ平坦で、E がデータで決まらない。10-12 の試算でも、H の最良は 0 付近か負）")
    judge(eH >= 20.0, f"{n}：H の最良の E = {eH:.1f} kJ/mol が、文献値の目安（40）に近い（20 以上）なら、§5 の構造（H）はデータで支持される")
    dh = float(e_of(mat, "S_V", "H")["LML_best"] - e_of(mat, "S_V", "H")["LML_at_E40"])
    judge(dh < 10.0, f"{n}：H の LML は、最良の E に比べて、E = 40 では {dh:.1f} 下がる（10 未満なら、40 でも許される。10 以上なら、40 はデータと合わない）")
    lA, lH, lB, lZ = (float(e_of(mat, "S_V", s)["LML_best"]) for s in ["A", "H", "B", "Hz"])
    judge(lA - lH >= 2.0, f"{n}：A（縦だけ）の LML が H より 2 以上大きい（A {lA:.1f}、H {lH:.1f}）なら、縦だけのシフトのほうがデータに合う")
    judge(lB - max(lA, lH, lZ) >= 2.0, f"{n}：B（横・縦別々）の LML が A・H・Hz のどれより 2 以上大きい（B {lB:.1f}、最大の他 {max(lA, lH, lZ):.1f}）なら、横と縦の両方が必要")
    d0 = float(e_of(mat, "S_V", "A")["dLML_vs_E0"])
    judge(d0 >= 2.0, f"{n}：A の最良の E_v = {bE(mat, 'S_V', 'A'):.1f} kJ/mol（95% 区間 {bE(mat, 'S_V', 'A', 'E_95_lo'):.1f}〜{bE(mat, 'S_V', 'A', 'E_95_hi'):.1f}）。温度なし（E = 0）より LML が 2 以上大きい（差 {d0:.1f}）なら、温度の効果がデータで支持される")
    for st in ["H", "A"]:
        es = [bE(mat, s, st) for s in ["S_V", "S_T", "ALL"]]
        judge(max(es) - min(es) <= 5.0, f"{n}：{st} の最良の E が、学習データ（S_V・S_T・ALL）を変えても 5 kJ/mol 以内（{', '.join(f'{x:.1f}' for x in es)}）")
    edge = [st for st in ["H", "A", "Hz", "F"] if bool(e_of(mat, "S_V", st)["at_grid_edge"])]
    judge(not edge, f"{n}：最良の E が格子の端に張り付いている構造: {edge or 'なし'}（張り付いたら、格子の範囲を広げる）")
    for split, sub, lab in [("S_V", "低速側", "S_V 低速側"), ("S_V", "高速側", "S_V 高速側"), ("S_T", "全体", "S_T")]:
        c = m5(mat, split, "C", sub)
        best = min(["A", "B", "H"], key=lambda s: m5(mat, split, s, sub))
        judge(m5(mat, split, best, sub) <= 1.1 * c, f"{n} {lab}：構造ありの最良（{best}、RMSE {m5(mat, split, best, sub):.3f}）が、通常の GPR（C、{c:.3f}）の 1.1 倍以内（同等以上なら、構造がデータで支持される）")
    r = eff5[(eff5["split"] == "S_V") & (eff5["material"] == n)]
    o = obs5[obs5["material"] == n]["observed"]
    for st in ["A", "B", "H"]:
        e_ = r[r["variant"] == st]["ratio_190_230"]
        judge(bool((e_ > 1.0).all()), f"{n} {st}：Tset だけを 190 → 230 ℃ にした予測の粘度比が、全厚みで 1 を超える（範囲 {e_.min():.2f}〜{e_.max():.2f}、実測 {o.min():.2f}〜{o.max():.2f}）")
print("\n保存したファイル:")
for f in sorted(os.listdir(OUT_DIR)):
    print("  ", os.path.normpath(os.path.join(OUT_DIR, f)))''')

# ---------------------------------------------------------------- 11. 高解像度のグラフ
md(HR_MD.replace("## 高解像度のグラフ", "## 11. 高解像度のグラフ").replace("**レイアウト**", "**構造**：既定は、本命（`MAIN_STRUCT`）と対案（`ALT_STRUCT`）。`HR_STRUCTS` で、H・A・Hz・B・N・H40・F・C のどれでも選べる。\n- **レイアウト**"))
code(hr_cell("# セル15：高解像度のグラフ（せん断速度 ― 粘度。本命と対案の予測。GPR6月ノートのセル9-2 にならう）",
             '''HR_STRUCTS = [MAIN_STRUCT, ALT_STRUCT]   # 描く構造（既定は、本命と対案。H・A・Hz・B・N・H40・F・C のどれでも。記号の意味は、ノート冒頭の表）
HR_SPLIT = "S_V"                   # どの学習のモデルか："S_V"（5data で学習）／"S_T"（190・230 ℃ で学習）／"ALL"（5data＋3data で学習し直す。点は全部が学習の点）''',
             r'''HR_SPLIT_JA = {"S_V": "5data で学習", "S_T": "190・230 ℃ で学習", "ALL": "5data＋3data で学習"}


def hr_fit5(mat, name):
    """描く構造の学習済み予測器（入力の γ̇ は条件平均）。S_V・S_T は上で学習したもの、ALL は 5data＋3data で学習し直す（E は、ALL のプロファイルの最良の値）"""
    if HR_SPLIT == "ALL":
        return fit_variant(variant_defs(mat, "ALL")[name], pool_of(mat, MAIN_DATA), "cm", N_RESTARTS_FINAL)
    return FIT5[(mat, HR_SPLIT, name, "cm")]


hr_saved = []
for mat in HR_MATERIALS:
    for name in HR_STRUCTS:
        t0 = time.time()
        fit = hr_fit5(mat, name)
        v = fit["v"]
        label = STRUCT_JA[name] + ("" if v["kind"] == "free" else f"　E_h = {v['Eh']:.1f}、E_v = {v['Ev']:.1f} kJ/mol") + f"（{HR_SPLIT_JA[HR_SPLIT]}）"
        hr_saved += hr_flow(lambda Xg, f=fit: predict_variant(f, Xg)[:3], mat, MAIN_DATA, hr_split_conds(mat, MAIN_DATA, HR_SPLIT, TEMP_HOLDOUT),
                            label, f"hr_{mat}_{name}_{HR_SPLIT}", HR_DIR, **HR_KW)
        print(f"{MATS[mat]} {name}：保存したファイル 累計 {len(hr_saved)} 個（{time.time() - t0:.0f} 秒）")
judge(len(hr_saved) > 0, f"高解像度のグラフを {len(hr_saved)} 個保存した（{os.path.normpath(HR_DIR)}）。形式 {HR_FORMATS}、{HR_DPI} dpi")'''))


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
        "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python",
                          "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.12.12"},
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
