# -*- coding: utf-8 -*-
"""手順5（シフト構造の本実装：構造 B・A・C × H の扱い (1)(2)(4)）のノートブック (未実行) を生成する。

使い方: python build_step5_nb.py <出力する .ipynb のパス> [--force]
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

_core = open(os.path.join(HERE, "step5_core.py"), encoding="utf-8").read().split('"""', 2)[2].lstrip("\n")
CORE_CELL = "# セル：手順5の部品（構造・H の扱い・プロファイル尤度・P3_2 の予測・学習の仕事）。04_補助スクリプト/step5_core.py の中身。直すときは、そのファイルを直して、ノートを作り直す\n" + _core

# ---------------------------------------------------------------- 0. タイトル
md(r'''# 手順5：シフト構造の本実装（構造 B・A・C × H の扱い (1)(2)(4)）

CLAUDE.md §5・§7。手順5a で決まった構造（**本命 B**：横・縦を別々にシフト、**対案 A**：縦だけシフト、基準 **C**：構造なしの通常の GPR）を、手順3e で決めた **H の扱い (1)・(2)・(4) の組み合わせ**で、同じ形で学習・評価する。データは、手順3d の 2 つの版（**5data 版**：5data で学習して 3data で外挿を検証／**Alldata 版**：全条件で学習して交差検証）を `DATA_VERSIONS` で切り替える。

**モデルの表**（記号は「構造_扱い」。例：B_1、A_2、C_4。手順3b の B0・B2 とは別物）

u = ln γ̇（条件平均）＋ ln a(Tset; E_h)、w = ln Fo = −(ln γ̇ ＋ 3 ln H) ＋ 定数、p = ln P3_2（条件平均、Pa）。ln a(T; E) = (E/R)(1/T − 1/Tref)、Tref = 483.15 K（210 ℃）。f・m・c は GP。

| H の扱い | 構造 B・A の形（A は E_h = 0） | 構造 C の形（Tset を入力に並べる） |
|---|---|---|
| (1) H を残す（`"1"`） | ln η = ln a(E_v) ＋ f(u, w) | f(ln γ̇, ln H, Tset)　※手順3b・3d の R3 |
| (2) 分ける（`"2"`） | ln a(E_v) ＋ **m(u)** ＋ **c(w)** | m(ln γ̇, Tset) ＋ c(w) |
| (4) P3_2 を入れる・H なし（`"4"`） | ln a(E_v) ＋ f(u, p) | f(ln γ̇, Tset, p)　※手順3e の W2 |
| (1+4)（`"14"`） | ln a(E_v) ＋ f(u, w, p) | f(ln γ̇, ln H, Tset, p) |
| (2+4)（`"24"`） | ln a(E_v) ＋ m(u, p) ＋ c(w) | m(ln γ̇, Tset, p) ＋ c(w) |

**(2) の意味**：Cross-WLF のように「材料の部分 m（せん断速度と温度の関数）」と、「厚み・冷却の補正 c(Fo)」を足し算で分ける。足し算のカーネル（m の部分と c の部分に、それぞれ Matern ＋ 線形項）で表す。m と c は定数だけ入れ替えられるので、図では c の平均を 0 にそろえる。

**(4) の注意**：P3_2 は流れの結果で、成形条件から予測するときは**事前に分からない**。しかも**厚みでほぼ決まる量**（ln P3_2 ≈ −1.5 ln H）なので、H を P3_2 の形で入れただけ、かもしれない（手順3e）。そこで (4) を含むモデルは 2 通りで評価する。
- **実測の P3_2**：検証の条件の、実測の条件平均を使う（P3_2 が分かっている場合の上限。手順3e と同じ）。
- **予測した P3_2**：事前に分からないとして、ln P3_2 を (ln γ̇, ln H, Tset) から GPR で予測し（学習の条件の点で学習）、その予測分布から `P32_DRAWS` 回引いて ln η を予測し、平均と分散にまとめる。P3_2 を予測するのに H が要るので、「実測」との差が、「(4) の良さが、P3_2 が分かっているおかげか」を表す。

**推定**（これまでどおり scikit-learn）
1. **E はプロファイル尤度で決める**：条件平均の点で、E の格子の各点について GPR を学習し、対数周辺尤度（LML）が最大の E を採る。LML が最大から 1.92 以内の範囲を、おおよその 95% 区間（B は斜めの帯）とする。粗い格子で全体を見て、最良の点の近くを細かい格子で計算し直す。前の格子点の解を初期値にして、局所解でのがたつきを減らす。出力の自動標準化は使わない（E ごとに尺度が変わり、尤度が比べられなくなるため）。
2. **評価用のモデルは、ショットごとに学習し直す**：最良の E で、入力の γ̇ を条件平均にして学習する（成形条件から見たばらつきを出すため）。E は、学習データごとに、その学習データだけで推定する（CV の分割ごとにも推定する。検証の情報は使わない）。
3. **B の E_h と E_v は、別々には決まりにくい**（手順5a：差 E_v − E_h だけが決まる）。両方を推定し、斜めの帯と差を示す。設定 `B_EH_FIX` に数を入れると、E_h を固定して実行し直せる。

**MCMC は、結果を見てから別の手順で行う**（ユーザー決定、10-14）。''')

# ---------------------------------------------------------------- 1. 設定
md(usage_md("5", "results/step5", "results/学習結果説明_手順5.pdf", 150))
md(r'''### 実行時間の目安と、短くする方法

全部（2 つの版、3 構造 × 5 通り）で約 2〜3 時間（プロファイル尤度が約 1 時間、ショットごとの学習が約 1〜1.5 時間）。短くするには、設定エリアで次を減らす。
- `DATA_VERSIONS = ["5data"]`：約 20 分。
- `SPLITS`：Alldata 版の `"CV"`（10 分割）を外す。
- `H_VARIANTS`（例：`["1", "2"]`）、`STRUCTS`（例：`["B", "C"]`）。

「すべて実行」でなく、版ごとに分かれたセル（プロファイル・学習）を、順に実行してもよい。1 セルが長いので、メニューの「すべて実行」では時間の上限（ノートの設定）に注意する。''')
md(r'''## 1. 設定''')
code(settings_cell('''import numpy as np             # 格子の指定に使う
N_RESTARTS = 3                # 最終モデルの学習でのやり直し回数
MAIN_STRUCT = "B"             # 本命の構造（ユーザー決定、2026-10-12）
ALT_STRUCT = "A"              # 対案の構造
STRUCTS = ["B", "A", "C"]     # 比べる構造（C：構造なし）
H_VARIANTS = ["1", "2", "4", "14", "24"]   # H の扱い：1＝(1) H を残す／2＝(2) m＋c(Fo) に分ける／4＝(4) P3_2（H なし）／14＝(1+4)／24＝(2+4)
DATA_VERSIONS = ["5data", "Alldata"]       # データの版（手順3d）。"5data"：外挿の検証／"Alldata"：全条件で学習し、交差検証
SPLITS = {"5data": ["S_V", "S_T"],         # 版ごとの評価の分け方。5data 版：S_V（外挿）・S_T（210 ℃）・CV（条件単位の交差検証。時間がかかる）
          "Alldata": ["S_T", "CV"]}        # Alldata 版：S_T・CV（10 分割）・S_Vall（5data の条件で学習 → 3data を検証）
MAIN_DATA = "なし"            # 主に使う外れ値の扱い（外れ値なし）
TEMP_HOLDOUT = 210            # S_T でテストにする設定温度 [℃]
E_GRID_A = np.arange(-20.0, 45.01, 2.5)    # A の E_v の粗い格子 [kJ/mol]
EH_GRID = np.arange(-40.0, 40.01, 5.0)     # B の E_h の粗い格子
EV_GRID = np.arange(-20.0, 40.01, 5.0)     # B の E_v の粗い格子
EH_GRID_CV = np.arange(-10.0, 25.01, 5.0)  # CV の各分割での B の粗い格子（時間を抑えるため狭める。最良が端なら ★）
EV_GRID_CV = np.arange(0.0, 35.01, 5.0)
FINE_A = 2.5                  # 細かい格子：A は最良の点の ±FINE_A（0.5 刻み）
FINE_B = 4.0                  # 細かい格子：B は最良の点の ±FINE_B（1 刻み）。CV の分割では FINE_B_CV
FINE_B_CV = 3.0
N_RESTARTS_PROFILE = 3        # 尤度の格子の各点での最適化のやり直し回数
N_RESTARTS_CV = 1             # CV の各分割での学習のやり直し回数（Alldata は行数が多く遅いので少なくする）
B_EH_FIX = None               # None：B の E_h・E_v を両方推定／数（kJ/mol）：E_h をその値に固定（E_v だけ推定）
P32_DRAWS = 20                # (4) で P3_2 を予測して使うときの、予測分布からの引き直しの回数
FIG_VARIANTS = ["1", "2", "4"]  # 曲線などの図に出す H の扱い（全部の指標の表は、すべての扱いで出る）
N_JOBS = 8                    # 並列に計算する数（CPU の数に合わせる）
E_LIT = 40.0                  # 文献値の目安 [kJ/mol]（PP の流動の活性化エネルギー。このデータの「見かけのシフトエネルギー」とは別物）
'''))
code(r'''# セル1.5：ライブラリ、フォント、出力先
%matplotlib inline
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from joblib import Parallel, delayed

OUT_DIR = os.path.join(REPO, "results", "step5")
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["figure.dpi"] = 80
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
try:
    fm.findfont("IPAexGothic", fallback_to_default=False)
except ValueError:
    raise RuntimeError("日本語フォント IPAexGothic が見つかりません。インストールしてからカーネルを再起動してください。")
plt.rcParams["font.family"] = ["IPAexGothic", "DejaVu Sans"]
print("Python:", sys.executable, "/", sys.version.split()[0])
print("出力先:", os.path.abspath(OUT_DIR))''')
md(r'''### 共通の部品

データの読み込み、問題データ一覧の「除外」、評価、データの分け方、列を選んで当てるカーネル `OnDims` など。中身は `04_補助スクリプト/nb_common.py` と同じ。''')
code(COMMON_CELL)
md(r'''### 手順5の部品

構造とHの扱いの定義（`layout5`・`design5`・`kernel5`）、学習と予測（`fit5`・`predict5`・`rebuild5`）、足し算の成分の予測（`component_predict`）、プロファイル尤度の計算（`prof_chain`）、P3_2 の予測（`p32_model`・`predict5_p`）、並列の仕事（`job5`）。中身は `04_補助スクリプト/step5_core.py` と同じ。''')
code(CORE_CELL)

# ---------------------------------------------------------------- 2. データ
md(r'''## 2. データの読み込みと、部品の確認

手順3d の 2 つの版の分け方（`version_sets`）で、学習データとテストデータを作る。学習データの名前は、`S_V`・`S_T`・`CV1`〜（分け方）と、`ALL`（最終モデル用の全部）。5data 版の `ALL` は `S_V` と同じ学習データなので、計算を共有する。

**部品の確認**（判定）
- `OnDims` に全部の列を渡したカーネルが、ふつうのカーネルと、カーネル行列・勾配で一致する。
- C_1 と C_4 が、手順3d・3e の R3・W2 と同じ RMSE になる（同じ入力・同じ分け方・同じ乱数なので）。''')
code(r'''# セル2：読み込みと版の確認
D = load_all()
leg = legacy_check(D)
judge(bool(leg["同じ行"].all()) or not USE_EXCLUSIONS, "一覧の「除外」を除いたデータが、これまでの no1.5-3.0 のファイルと同じ（取り直した後は違ってよい）")
VS = {(mat, ver): version_sets(mat, ver, MAIN_DATA, TEMP_HOLDOUT) for ver in DATA_VERSIONS for mat in MATS}
TRAIN_DF, TESTS, ALIAS = {}, {}, {}
for (mat, ver), vs in VS.items():
    names = []
    for s in SPLITS[ver]:
        names += [k for k in vs["splits"] if k.startswith("CV")] if s == "CV" else ([s] if s in vs["splits"] else [])
        if s != "CV" and s not in vs["splits"]:
            print(f"（{MATS[mat]}・{ver} 版に、分け方 {s} はない。飛ばす）")
    for n in names:
        TRAIN_DF[(mat, ver, n)], TESTS[(mat, ver, n)] = vs["splits"][n]
    if ver == "5data" and "S_V" in names:
        ALIAS[(mat, ver)] = "S_V"                    # 5data 版の最終モデル（ALL）は、S_V と同じ学習データ
    else:
        TRAIN_DF[(mat, ver, "ALL")], TESTS[(mat, ver, "ALL")] = vs["pool"], {}
    pool = vs["pool"]
    print(f"{MATS[mat]}・{ver} 版：最終モデルの学習 {len(pool)} 行 {pool['cond'].nunique()} 条件。分け方 {names}"
          + "".join(f"\n    {n}：学習 {len(TRAIN_DF[(mat, ver, n)])} 行、テスト " + "、".join(f"{k} {len(t)} 行" for k, t in TESTS[(mat, ver, n)].items()) for n in names[:3]))


def tr_of(mat, ver, sname):
    """学習データ。5data 版の ALL は S_V と同じ"""
    return TRAIN_DF[(mat, ver, ALIAS[(mat, ver)] if (sname == "ALL" and (mat, ver) in ALIAS) else sname)]


def sets_of(mat, ver):
    """計算する学習データの名前（5data 版の ALL は S_V と共有するので入れない）。最終モデルは ALL（別名を含む）"""
    return [n for (m, v, n) in TRAIN_DF if m == mat and v == ver]


def mname(st, hv):
    return f"{st}_{hv}"


print("構造:", STRUCTS, "／H の扱い:", H_VARIANTS)''')
code(r'''# セル3：部品の確認（OnDims、C_1 = R3、C_4 = W2）
rng = np.random.RandomState(0)
mat0, ver0 = list(VS)[0]
tr0 = tr_of(mat0, ver0, [n for n in sets_of(mat0, ver0) if not n.startswith("CV")][0])
X0_ = StandardScaler().fit_transform(tr0[["ln_gd_cm", "ln_H", "Tset"]].to_numpy())
k_plain = kernel_for(3, True)
k_on = OnDims(C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0] * 3, length_scale_bounds=(1e-2, 1e2), nu=2.5) + C(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-5, 1e3)),
              [0, 1, 2], "f") + WhiteKernel(noise_level=0.1, noise_level_bounds=(NOISE_LB, 1e1))
th_ = rng.uniform(-1, 1, len(k_plain.theta))
(Ka, Ga), (Kb, Gb) = k_plain.clone_with_theta(th_)(X0_, eval_gradient=True), k_on.clone_with_theta(th_)(X0_, eval_gradient=True)
judge(max(np.abs(Ka - Kb).max(), np.abs(Ga - Gb).max()) < 1e-8, f"OnDims（全部の列）とふつうのカーネルの、カーネル行列・勾配の最大の差が 1e-8 未満（差 {max(np.abs(Ka - Kb).max(), np.abs(Ga - Gb).max()):.1e}）")
# 学習データが最小の分け方で、C_1 と R3、C_4 と W2 を比べる
mat1, ver1, n1 = min(((m, v, n) for (m, v, n) in TRAIN_DF if n != "ALL" and TESTS[(m, v, n)]), key=lambda k: len(TRAIN_DF[k]))
tr1, te1 = TRAIN_DF[(mat1, ver1, n1)], list(TESTS[(mat1, ver1, n1)].values())[0]
MODELS["R3"] = {"kind": "gpr", "cols": ["ln_gd_cm", "ln_H", "Tset"], "linear": True}
MODELS["W2"] = {"kind": "gpr", "cols": ["ln_gd_cm", "Tset", "ln_P32_cm"], "linear": True}
for hv_, key_ in [("1", "R3"), ("4", "W2")]:
    mu_a = predict5(fit5({"struct": "C", "hv": hv_, "pcol": "ln_P32_cm"}, tr1, N_RESTARTS), te1)[0]
    mu_b = predict_model(key_, fit_model(key_, tr1, N_RESTARTS), te1)[0]
    ra, rb = float(np.sqrt(np.mean((te1["ln_eta"].to_numpy() - mu_a) ** 2))), float(np.sqrt(np.mean((te1["ln_eta"].to_numpy() - mu_b) ** 2)))
    judge(abs(ra - rb) <= 0.003, f"C_{hv_} の RMSE が、{key_} と ±0.003 で一致（{MATS[mat1]}・{ver1} 版・{n1}：C_{hv_} {ra:.4f}、{key_} {rb:.4f}）")''')

# ---------------------------------------------------------------- 3. プロファイル尤度
md(r'''## 3. プロファイル尤度（条件平均）

各学習データ（S_V・S_T・CV の各分割・ALL）、各構造（A・B）、各 H の扱いについて、E の格子の各点で GPR を学習し、LML を求める。**LML が大きいほど、データに合っている**。H の扱いが違うと入力の次元が違うので、LML の絶対値は扱いの間では比べられない（扱いごとに、最大との差で見る）。

- A：E_v の 1 次元の格子（E_h = 0）。
- B：(E_h, E_v) の 2 次元の格子。**斜めの帯**になる（差 E_v − E_h だけが決まる）。
- 最良の点と、LML が最大から 1.92 以内の範囲（おおよその 95% 区間）を `best_E.csv` に保存する。
- 判定：①格子の端に張り付いていないか。②E = 0（温度の効果なし）より LML が 2 以上大きいか。③学習データを変えても、A の E_v と B の差 E_v − E_h が 5 kJ/mol 以内か。④B が A より LML が 2 以上大きいか。

版ごとにセルを分けてある（長いので）。''')
code(r'''# セル4：プロファイルの計算の関数（並列）
PROF_STRUCTS = [s for s in STRUCTS if s != "C"]
PROF_LIST = []


def grid_for(sname):
    cv = sname.startswith("CV")
    return {"A": E_GRID_A, "B_eh": EH_GRID_CV if cv else EH_GRID, "B_ev": EV_GRID_CV if cv else EV_GRID}


def run_chains(tasks, stage):
    """tasks：[(キー(材料, 版, 学習データ, 構造, 扱い), 条件平均の表, 基本の指定, 格子の点の列)] → 結果の表"""
    tasks = sorted(tasks, key=lambda t: -len(t[3]))
    out = Parallel(n_jobs=N_JOBS)(delayed(prof_chain)(t[1], t[2], t[3], N_RESTARTS_PROFILE) for t in tasks)
    rows = []
    for (key, _, _, pts), res in zip(tasks, out):
        for (eh, ev), (lml, nz) in zip(pts, res):
            rows.append({"mat": key[0], "version": key[1], "set": key[2], "struct": key[3], "hv": key[4], "Eh": round(eh, 6), "Ev": round(ev, 6), "LML": lml, "noise_sd": nz, "stage": stage})
    return pd.DataFrame(rows)


KEYC = ["mat", "version", "set", "struct", "hv"]


def profile_version(ver):
    """版 ver の全材料・全学習データのプロファイルを計算する（粗い格子 → 最良の点の近くの細かい格子）。結果を PROF_LIST に足す"""
    t0 = time.time()
    CD = {(mat, ver, n): cond_table5(tr_of(mat, ver, n)) for mat in MATS for n in sets_of(mat, ver)}
    tasks = []
    for (mat, _, n), cdf in CD.items():
        for st in PROF_STRUCTS:
            for hv in H_VARIANTS:
                for pts in chains_for(st, grid_for(n), B_EH_FIX):
                    tasks.append(((mat, ver, n, st, hv), cdf, {"struct": st, "hv": hv, "pcol": "ln_P32_cm"}, pts))
    coarse = run_chains(tasks, "粗い格子")
    print(f"{ver} 版：粗い格子 {len(coarse)} 点、{(time.time() - t0) / 60:.1f} 分")
    best0 = profile_best(coarse, KEYC)
    tasks = []
    for _, r in best0.iterrows():
        key = (r["mat"], r["version"], r["set"], r["struct"], r["hv"])
        cdf = CD[(r["mat"], r["version"], r["set"])]
        bv = {"struct": r["struct"], "hv": r["hv"], "pcol": "ln_P32_cm"}
        if r["struct"] == "A":
            tasks.append((key, cdf, bv, [(0.0, float(e)) for e in r["Ev_best"] + np.arange(-FINE_A, FINE_A + 0.01, 0.5)]))
        else:
            half = FINE_B_CV if r["set"].startswith("CV") else FINE_B
            ehs = [B_EH_FIX] if B_EH_FIX is not None else list(r["Eh_best"] + np.arange(-half, half + 0.01, 1.0))
            for eh in ehs:
                tasks.append((key, cdf, bv, [(float(eh), float(e)) for e in r["Ev_best"] + np.arange(-half, half + 0.01, 1.0)]))
    fine = run_chains(tasks, "細かい格子")
    allp = pd.concat([coarse, fine], ignore_index=True).drop_duplicates(subset=KEYC + ["Eh", "Ev"], keep="first")
    PROF_LIST.append(allp)
    print(f"{ver} 版：細かい格子を足して、全部で {len(allp)} 点、{(time.time() - t0) / 60:.1f} 分")''')
code(r'''# セル5：プロファイルの計算（5data 版）
if "5data" in DATA_VERSIONS and PROF_STRUCTS:
    profile_version("5data")''')
code(r'''# セル6：プロファイルの計算（Alldata 版。CV の 10 分割を含むので、1 時間ほどかかる）
if "Alldata" in DATA_VERSIONS and PROF_STRUCTS:
    profile_version("Alldata")''')
code(r'''# セル7：最良の E と区間、判定
if PROF_STRUCTS:
    prof_df = pd.concat(PROF_LIST, ignore_index=True)
    prof_df.insert(0, "material", prof_df["mat"].map(MATS))
    prof_df.drop(columns="mat").to_csv(os.path.join(OUT_DIR, "profile_likelihood.csv"), index=False, encoding="utf-8-sig")
    best_df = profile_best(prof_df, KEYC)
    best_df.insert(0, "material", best_df["mat"].map(MATS))
    best_df.drop(columns="mat").to_csv(os.path.join(OUT_DIR, "best_E.csv"), index=False, encoding="utf-8-sig")
else:
    prof_df, best_df = pd.DataFrame(), pd.DataFrame()


def best_row(mat, ver, sname, st, hv):
    """最良の E の行（5data 版の ALL は S_V の行）"""
    sname = ALIAS[(mat, ver)] if (sname == "ALL" and (mat, ver) in ALIAS) else sname
    return best_df[(best_df["mat"] == mat) & (best_df["version"] == ver) & (best_df["set"] == sname) & (best_df["struct"] == st) & (best_df["hv"] == hv)].iloc[0]


if len(best_df):
    show =best_df[best_df["set"].isin(["S_V", "S_T", "ALL"]) & ~best_df["set"].str.startswith("CV")].copy()
    show["Eh"], show["Ev"] = show["Eh_best"], show["Ev_best"]
    print("【最良の E（kJ/mol）。diff＝E_v − E_h、diff_95＝LML が最大から 1.92 以内の範囲の差。dLML_vs_E0＝E = 0 との LML の差】")
    cols_show = ["material", "version", "set", "struct", "hv", "Eh_best", "Ev_best", "diff_best", "diff_95_lo", "diff_95_hi", "dLML_vs_E0", "at_edge"]
    print(show[cols_show].round(2).to_string(index=False))
    for ver in DATA_VERSIONS:
        for mat in MATS:
            n_ = MATS[mat]
            g = best_df[(best_df["mat"] == mat) & (best_df["version"] == ver)]
            full = g[~g["set"].str.startswith("CV")]
            judge(not g["at_edge"].any() if len(g) else True, f"{n_}・{ver} 版：最良の E が格子の端に張り付いていない（張り付いた: {int(g['at_edge'].sum())} / {len(g)} 件。CV の狭い格子で張り付いたら、EH_GRID_CV・EV_GRID_CV を広げる）")
            for st in PROF_STRUCTS:
                h1 = full[(full["struct"] == st) & (full["hv"] == H_VARIANTS[0])]
                judge(bool((h1["dLML_vs_E0"] >= 2.0).all()), f"{n_}・{ver} 版・{st}（H の扱い {H_VARIANTS[0]}）：E = 0 より LML が 2 以上大きい（差 {h1['dLML_vs_E0'].round(1).tolist()}）。温度の効果がデータで支持される")
                for hv in H_VARIANTS:
                    s = full[(full["struct"] == st) & (full["hv"] == hv)]
                    col = "Ev_best" if st == "A" else "diff_best"
                    sp_ = float(s[col].max() - s[col].min()) if len(s) > 1 else 0.0
                    if hv in ("1", "2") or hv == H_VARIANTS[0]:
                        judge(sp_ <= 5.0, f"{n_}・{ver} 版・{st}（扱い {hv}）：{'E_v' if st == 'A' else '差 E_v − E_h'} が、学習データ（{', '.join(s['set'])}）を変えても 5 kJ/mol 以内（{', '.join(f'{x:.1f}' for x in s[col])}）")
            if "A" in PROF_STRUCTS and "B" in PROF_STRUCTS:
                ms_ = "S_V" if ver == "5data" else "ALL"
                for hv in H_VARIANTS:
                    a_, b_ = full[(full["struct"] == "A") & (full["hv"] == hv) & (full["set"] == ms_)], full[(full["struct"] == "B") & (full["hv"] == hv) & (full["set"] == ms_)]
                    if len(a_) and len(b_):
                        d_ = float(b_["LML_best"].iloc[0] - a_["LML_best"].iloc[0])
                        judge(d_ >= 2.0, f"{n_}・{ver} 版・扱い {hv}：B の LML が A より 2 以上大きい（差 {d_:.1f}）なら、横のシフトも必要")''')

# ---------------------------------------------------------------- 4. 学習と評価
md(r'''## 4. 学習と評価

3 構造 × 5 通りのモデルを、最良の E（その学習データのもの）で、ショットごとに学習し直して評価する。入力の γ̇ は条件平均。

- 評価の分け方：5data 版は **S_V**（5data → 3data。低速側・高速側）と **S_T**（190・230 ℃ → 210 ℃。5data の 210 ℃ と、3data の 210 ℃〔S_T3〕）。Alldata 版は **CV**（条件単位の 10 分割。全分割の予測をまとめて評価）と **S_T**（テストは 5data 版と同じ行）、設定すれば **S_Vall**。
- (4) を含むモデルは、P3_2 を「実測」と「予測」の 2 通りで評価する（冒頭の説明）。
- 指標：log η の RMSE、R²、95% 予測区間の被覆率、NLPD。`noise_sd`（モデルのノイズの標準偏差）と `pure_sd`（同じ条件内のショット間のばらつき＝純粋誤差）も並べる。入力が条件平均なので、ノイズは純粋誤差と同程度かそれ以上になるはず。
- 結果は `metrics.csv`、`kernels.csv`（ハイパーパラメータと、下限・上限への張り付き）。''')
code(r'''# セル8：学習の仕事の関数
RES5 = {}


def eval_version(ver):
    """版 ver の全材料・全学習データ・全モデルを並列で学習して、予測を RES5 に入れる"""
    t0 = time.time()
    jobs = []
    for mat in MATS:
        for n in sets_of(mat, ver):
            tr = TRAIN_DF[(mat, ver, n)]
            for st in STRUCTS:
                for hv in H_VARIANTS:
                    v = {"struct": st, "hv": hv, "pcol": "ln_P32_cm"}
                    if st != "C":
                        r = best_row(mat, ver, n, st, hv)
                        v["Eh"], v["Ev"] = float(r["Eh_best"]), float(r["Ev_best"])
                    jobs.append(((mat, ver, n, st, hv), tr, TESTS[(mat, ver, n)], v, N_RESTARTS_CV if n.startswith("CV") else N_RESTARTS))
    jobs.sort(key=lambda j: -len(j[1]))
    out = Parallel(n_jobs=N_JOBS)(delayed(job5)(j[0][0], j[0][1], j[0][2], j[1], j[2], j[3], j[4], P32_DRAWS) for j in jobs)
    for j, o in zip(jobs, out):
        RES5[j[0]] = o
        RES5[j[0]]["v"] = j[3]
    for mat in MATS:
        if (mat, ver) in ALIAS:
            for st in STRUCTS:
                for hv in H_VARIANTS:
                    RES5[(mat, ver, "ALL", st, hv)] = RES5[(mat, ver, ALIAS[(mat, ver)], st, hv)]
    print(f"{ver} 版：{len(jobs)} 回の学習を並列（{N_JOBS}）で計算：{(time.time() - t0) / 60:.1f} 分")''')
code(r'''# セル9：学習（5data 版）
if "5data" in DATA_VERSIONS:
    eval_version("5data")''')
code(r'''# セル10：学習（Alldata 版。CV の 10 分割を含むので、1 時間ほどかかる）
if "Alldata" in DATA_VERSIONS:
    eval_version("Alldata")''')
code(r'''# セル11：指標とハイパーパラメータの表
FIT_CACHE = {}


def get_fit(mat, ver, sname, st, hv):
    """学習済みの予測器（並列の仕事が返した theta から、最適化なしで作り直す）"""
    key = (mat, ver, sname, st, hv)
    if key not in FIT_CACHE:
        o = RES5[key]
        FIT_CACHE[key] = rebuild5(o["v"], tr_of(mat, ver, sname), o["theta"])
    return FIT_CACHE[key]


KROWS, acc, nzs = [], {}, {}
for (mat, ver, sname, st, hv), o in RES5.items():
    if sname == "ALL" and (mat, ver) in ALIAS:
        continue
    tr = TRAIN_DF[(mat, ver, sname)]
    KROWS.append({"material": MATS[mat], "version": ver, "split": sname, "struct": st, "hv": hv, "model": mname(st, hv), "n_train": len(tr), **o["kernel"],
                  "pure_sd_ln_eta": within_stats(tr)["sd_ln_eta"], "fit_sec": o["sec"]})
    for (tn, psrc), (mu, sd, sdf) in o["preds"].items():
        te = TESTS[(mat, ver, sname)][tn]
        split = "CV" if tn.startswith("CV") else tn
        acc.setdefault((mat, ver, split, st, hv, psrc), []).append(pd.DataFrame({"y": te["ln_eta"].to_numpy(), "mu": mu, "sd": sd, "sdf": sdf, "H": te["H"].to_numpy(), "Tset": te["Tset"].to_numpy(),
                                                                                  "v": te["v"].to_numpy(), "cond": te["cond"].to_numpy(), "group": te["group"].to_numpy(), "is_sens": te["is_sens"].to_numpy()}))
        nzs.setdefault((mat, ver, split, st, hv), {})[sname] = (o["kernel"]["noise_sd"], within_stats(tr)["sd_ln_eta"])
kern_df = pd.DataFrame(KROWS)
kern_df.to_csv(os.path.join(OUT_DIR, "kernels.csv"), index=False, encoding="utf-8-sig")
PRED5, mrows = {}, []
for (mat, ver, split, st, hv, psrc), parts in acc.items():
    d = pd.concat(parts, ignore_index=True)
    if split == "CV":
        d["group"] = np.where(d["v"] <= 5, "低速側", np.where(d["v"] >= 280, "高速側", "内側"))
    PRED5[(mat, ver, split, st, hv, psrc)] = d
    subs = {"全体": np.ones(len(d), bool), **{g: (d["group"] == g).to_numpy() for g in d["group"].unique() if g != "全体"}}
    if d["is_sens"].any():
        subs.update({s + SENS_SUFFIX: m & ~d["is_sens"].to_numpy() for s, m in list(subs.items())})
    nz = np.array(list(nzs[(mat, ver, split, st, hv)].values()))
    for sub, m in subs.items():
        if m.sum() < 2:
            continue
        ev = evaluate(d["y"].to_numpy()[m], d["mu"].to_numpy()[m], d["sd"].to_numpy()[m])
        mrows.append({"material": MATS[mat], "mat": mat, "version": ver, "split": split, "subset": sub, "struct": st, "hv": hv, "model": mname(st, hv), "psrc": psrc,
                      "n_test": int(m.sum()), "n_cond_test": int(d["cond"][m].nunique()), **ev, "noise_sd": float(nz[:, 0].mean()), "pure_sd": float(nz[:, 1].mean())})
met_df = pd.DataFrame(mrows)
met_df.drop(columns="mat").to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False, encoding="utf-8-sig")
PANELS = {"5data": [("S_V", "低速側", "S_V 低速側"), ("S_V", "高速側", "S_V 高速側"), ("S_T", "全体", "S_T（210 ℃）"), ("S_T3", "全体", "S_T3（3data の 210 ℃）")],
          "Alldata": [("CV", "全体", "CV 全体"), ("CV", "低速側", "CV 低速側"), ("CV", "高速側", "CV 高速側"), ("S_T", "全体", "S_T（210 ℃）"), ("S_T3", "全体", "S_T3（3data の 210 ℃）"),
                      ("S_Vall", "低速側", "S_Vall 低速側"), ("S_Vall", "高速側", "S_Vall 高速側")]}


def mv(mat, ver, split, st, hv, sub="全体", psrc=None, col="RMSE"):
    """指標。psrc を省くと、P3_2 を使わないモデルは "―"、使うモデルは "実測" の行"""
    psrc = psrc or ("実測" if has_p(hv) else "―")
    r = met_df[(met_df["mat"] == mat) & (met_df["version"] == ver) & (met_df["split"] == split) & (met_df["struct"] == st) & (met_df["hv"] == hv) & (met_df["subset"] == sub) & (met_df["psrc"] == psrc)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def panels_of(ver):
    return [p for p in PANELS[ver] if ((met_df["version"] == ver) & (met_df["split"] == p[0]) & (met_df["subset"] == p[1])).any()]


for ver in DATA_VERSIONS:
    for mat in MATS:
        print(f"\n【{MATS[mat]}・{ver} 版】RMSE（ln η）／被覆率。P3_2 を使うモデルは、実測の P3_2（括弧の中が、予測した P3_2）")
        rows_ = {}
        for hv in H_VARIANTS:
            for st in STRUCTS:
                cells_ = {}
                for sp_, sub_, lab_ in panels_of(ver):
                    s = f"{mv(mat, ver, sp_, st, hv, sub_):.3f}/{mv(mat, ver, sp_, st, hv, sub_, col='coverage95'):.2f}"
                    if has_p(hv):
                        s += f"（{mv(mat, ver, sp_, st, hv, sub_, '予測'):.3f}）"
                    cells_[lab_] = s
                rows_[mname(st, hv)] = cells_
        print(pd.DataFrame(rows_).T.to_string())''')

# ---------------------------------------------------------------- 5. 設定温度の効果
md(r'''## 5. 設定温度の効果

実測と、各モデルの予測を比べる（手順5a の図6と同じ見方）。
- **実測の比**：同じ厚み・同じ γ̇ で、190 ℃ の粘度 ÷ 230 ℃ の粘度（感度確認の条件は除く）。
- **モデルの比**：5data＋3data の 210 ℃ のショットについて、ほかの入力はそのまま、Tset だけを 190 ℃・230 ℃ にして予測した粘度の比。最終モデル（ALL）を使う。P3_2 を使うモデルは、P3_2 を実測の条件平均に固定する。

実測は、「厚いほど温度の効果が大きい」傾向（0.5 mm で 1.09 倍、2.0 mm で 1.2 倍ほど）。縦だけシフトする A は、厚みによらず一定の比になる。''')
code(r'''# セル12：設定温度の効果
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


obs5 = pd.concat([observed_ratio(m) for m in MATS], ignore_index=True)
obs5.to_csv(os.path.join(OUT_DIR, "observed_ratio.csv"), index=False, encoding="utf-8-sig")
erows = []
for ver in DATA_VERSIONS:
    for mat in MATS:
        base = pool_of(mat, MAIN_DATA)
        base = base[(base["Tset"] == 210) & ~base["is_sens"]]
        lo, hi = base.copy(), base.copy()
        lo["Tset"], hi["Tset"] = 190.0, 230.0
        for st in STRUCTS:
            for hv in H_VARIANTS:
                fit = get_fit(mat, ver, "ALL", st, hv)
                r = np.exp(predict5(fit, lo)[0] - predict5(fit, hi)[0])
                for h, g in base.assign(r=r).groupby("H"):
                    erows.append({"material": MATS[mat], "version": ver, "model": mname(st, hv), "struct": st, "hv": hv, "H": h, "ratio_190_230": float(np.median(g["r"]))})
eff5 = pd.DataFrame(erows)
eff5.to_csv(os.path.join(OUT_DIR, "tset_effect.csv"), index=False, encoding="utf-8-sig")
print("実測の 190/230 ℃ の粘度比（同じ γ̇、厚みごと）")
print(obs5.pivot(index="material", columns="H", values="observed").round(3).to_string())
for ver in DATA_VERSIONS:
    print(f"\nモデルの比（{ver} 版の最終モデル、Tset だけを 190 → 230 ℃）")
    print(eff5[eff5["version"] == ver].pivot_table(index=["material", "model"], columns="H", values="ratio_190_230").round(3).to_string())
for ver in DATA_VERSIONS:
    for mat in MATS:
        e = eff5[(eff5["version"] == ver) & (eff5["material"] == MATS[mat])]
        for st in STRUCTS:
            r_ = e[(e["struct"] == st) & (e["hv"] == "1")]["ratio_190_230"]
            if len(r_):
                judge(bool((r_ > 1.0).all()), f"{MATS[mat]}・{ver} 版・{st}_1：Tset だけを 190 → 230 ℃ にした予測の粘度比が、全厚みで 1 を超える（範囲 {r_.min():.2f}〜{r_.max():.2f}、実測 {obs5[obs5['material'] == MATS[mat]]['observed'].min():.2f}〜{obs5[obs5['material'] == MATS[mat]]['observed'].max():.2f}）")''')

# ---------------------------------------------------------------- 6. (2) の分解
md(r'''## 6. (2) の分解：材料の部分 m と、厚み・冷却の補正 c(Fo)

H の扱い (2) では、ln η = ln a(E_v) ＋ m ＋ c を、足し算のカーネルで表した。最終モデル（ALL）で、2 つの成分の事後平均と 95% の帯を取り出す。
- **m（材料の部分）**：せん断速度（と温度）の関数。Cross-WLF にあたる。図では、定数 y0・縦シフト ln a(E_v)・c の平均の分を足して、「粘度 η_m [Pa·s]」として、設定温度ごとに描く。
- **c（厚み・冷却の補正）**：ln Fo の関数。学習の条件での平均を 0 にそろえてある。0 に近いほど補正が小さい。点は学習の条件で、厚みごとに色分け。
- m と c は定数だけ入れ替えられるので、c の平均を 0 にそろえた（η_m ＋ c の合計は、予測と同じ）。帯は、成分の事後分散（平均をそろえる定数の不確かさは含まない）。

判定：c の大きさ（最大と最小の差）が、温度の効果（190/230 ℃ の粘度比の対数 ≈ 0.1〜0.2）より大きい＝厚み・冷却の補正は、材料の部分とは別に、無視できない。結果は `decomposition.csv`。''')
code(r'''# セル13：(2) の分解
drows, DEC = [], {}
if "2" in H_VARIANTS:
    for ver in DATA_VERSIONS:
        for mat in MATS:
            tr = tr_of(mat, ver, "ALL")
            for st in STRUCTS:
                fit = get_fit(mat, ver, "ALL", st, "2")
                c_tr, _ = component_predict(fit, tr, "c")
                c_off = float(c_tr.mean())                                           # c の平均を 0 にそろえるための定数（m に足す）
                cdf = cond_table5(tr)
                c_c, c_s = component_predict(fit, cdf, "c")
                w_c = -(cdf["ln_gd_cm"] + 3.0 * cdf["ln_H"]).to_numpy()
                for w_, h_, c_, s_ in zip(w_c, cdf["H"], c_c - c_off, c_s):
                    drows.append({"material": MATS[mat], "version": ver, "struct": st, "kind": "c_cond", "H": h_, "Tset": np.nan, "x": float(w_), "mean": float(c_), "sd": float(s_)})
                wg = np.linspace(w_c.min() - 0.2, w_c.max() + 0.2, 80)
                gdf = pd.DataFrame({"ln_gd_cm": 0.0, "ln_H": -wg / 3.0, "Tset": 210.0})
                cg, sg = component_predict(fit, gdf, "c")
                for w_, c_, s_ in zip(wg, cg - c_off, sg):
                    drows.append({"material": MATS[mat], "version": ver, "struct": st, "kind": "c_curve", "H": np.nan, "Tset": np.nan, "x": float(w_), "mean": float(c_), "sd": float(s_)})
                lg_lo, lg_hi = float(tr["ln_gd"].min()), float(tr["ln_gd"].max())
                lgg = np.linspace(lg_lo, lg_hi, 80)
                for t in [190, 210, 230]:
                    mdf = pd.DataFrame({"ln_gd_cm": lgg, "ln_H": 0.0, "Tset": float(t)})
                    mm, ms = component_predict(fit, mdf, "m")
                    ysh = design5(mdf, fit["v"])[2]
                    for x_, m_, s_ in zip(lgg, mm + fit["y0"] + ysh + c_off, ms):
                        drows.append({"material": MATS[mat], "version": ver, "struct": st, "kind": "m_curve", "H": np.nan, "Tset": float(t), "x": float(x_), "mean": float(m_), "sd": float(s_)})
                DEC[(mat, ver, st)] = c_off
dec_df = pd.DataFrame(drows)
dec_df.to_csv(os.path.join(OUT_DIR, "decomposition.csv"), index=False, encoding="utf-8-sig")
if len(dec_df):
    for ver in DATA_VERSIONS:
        for mat in MATS:
            for st in STRUCTS:
                cc = dec_df[(dec_df["material"] == MATS[mat]) & (dec_df["version"] == ver) & (dec_df["struct"] == st) & (dec_df["kind"] == "c_cond")]["mean"]
                eo = obs5[obs5["material"] == MATS[mat]]["observed"]
                judge(float(cc.max() - cc.min()) > float(np.log(eo.max())), f"{MATS[mat]}・{ver} 版・{st}_2：c(Fo) の大きさ（最大 − 最小 = {float(cc.max() - cc.min()):.2f}）が、温度の効果（ln の実測比の最大 {float(np.log(eo.max())):.2f}）より大きい＝厚み・冷却の補正が、材料の部分と分けて必要")
else:
    print("H の扱い (2) を含めていないので、分解は飛ばす")''')

# ---------------------------------------------------------------- 7. P3_2
md(r'''## 7. (4) の P3_2：実測と予測

P3_2 を使うモデルの評価は、P3_2 が分かっている場合（実測）と、事前に分からず予測して使う場合（予測）で、どれだけ違うか。
- **P3_2 の予測の精度**：検証の条件の ln P3_2（条件平均）を、(ln γ̇, ln H, Tset) から予測したときの RMSE（`p32_pred.csv`）。
- **ln η の RMSE の比**：予測した P3_2 を使ったときの RMSE ÷ 実測の P3_2 を使ったときの RMSE。1 に近いほど、P3_2 が分からなくても良い。大きければ、(4) の良さは「P3_2 が分かっているおかげ」。
- **H を残したモデルとの比較**：(1+4) が (1) より良くなるか。良くならなければ、H があれば P3_2 は要らない（手順3b の R3P と同じ）。''')
code(r'''# セル14：P3_2 の予測の精度と、実測／予測の比
p32rows = []
for (mat, ver, sname), tests in TESTS.items():
    if sname == "ALL" or sname not in [n for n in sets_of(mat, ver)] or not tests:
        continue
    pm = p32_model(TRAIN_DF[(mat, ver, sname)])
    for tn, te in tests.items():
        ptab, _ = p32_predict(pm, te)
        ptab.insert(0, "split", "CV" if tn.startswith("CV") else tn)
        ptab.insert(0, "version", ver)
        ptab.insert(0, "material", MATS[mat])
        p32rows.append(ptab)
p32_df = pd.concat(p32rows, ignore_index=True) if p32rows else pd.DataFrame()
if len(p32_df):
    p32_df.to_csv(os.path.join(OUT_DIR, "p32_pred.csv"), index=False, encoding="utf-8-sig")
    s = p32_df.assign(e=p32_df["pred"] - p32_df["actual"]).groupby(["material", "version", "split"])["e"].agg(lambda x: float(np.sqrt(np.mean(x ** 2)))).rename("RMSE_ln_P32")
    print("【ln P3_2（条件平均）の予測の RMSE。ln η の RMSE と比べて見る】")
    print(s.round(3).to_string())
    P4 = [hv for hv in H_VARIANTS if has_p(hv)]
    rr = []
    for ver in DATA_VERSIONS:
        for mat in MATS:
            for sp_, sub_, lab_ in panels_of(ver):
                for st in STRUCTS:
                    for hv in P4:
                        a, b = mv(mat, ver, sp_, st, hv, sub_, "実測"), mv(mat, ver, sp_, st, hv, sub_, "予測")
                        rr.append({"material": MATS[mat], "version": ver, "panel": lab_, "model": mname(st, hv), "RMSE_actual": a, "RMSE_predicted": b, "ratio": b / a,
                                   "RMSE_with_H_(1)": mv(mat, ver, sp_, st, "1", sub_) if "1" in H_VARIANTS else np.nan})
    p4_df = pd.DataFrame(rr)
    p4_df.to_csv(os.path.join(OUT_DIR, "p32_actual_vs_predicted.csv"), index=False, encoding="utf-8-sig")
    print("\n【P3_2 を予測して使ったときの ln η の RMSE ÷ 実測の P3_2 のときの RMSE】（1 に近いほど、P3_2 が分からなくても同じ）")
    print(p4_df.pivot_table(index=["material", "version", "panel"], columns="model", values="ratio").round(2).to_string())
    for ver in DATA_VERSIONS:
        for mat in MATS:
            q = p4_df[(p4_df["version"] == ver) & (p4_df["material"] == MATS[mat]) & (p4_df["model"].str.endswith("_4"))]
            judge(bool((q["ratio"] <= 1.2).all()), f"{MATS[mat]}・{ver} 版：(4) の予測した P3_2 での RMSE が、実測の P3_2 の 1.2 倍以内（最大 {q['ratio'].max():.2f}）。超えるなら、(4) の良さは「P3_2 が分かっているおかげ」")
else:
    print("(4) を含むモデルがないか、テストがないので飛ばす")''')

# ---------------------------------------------------------------- 8. 本命と対案
md(r'''## 8. 本命と対案の比較（ユーザー決定、2026-10-12）

本命（`MAIN_STRUCT`、既定は B）と対案（`ALT_STRUCT`、既定は A）を、基準 C と、H の扱いごとに並べる。結果は `main_vs_alt.csv`。

- **E**：最終モデル（ALL）の最良の E（E_h・E_v、差）と、LML、E = 0 との LML の差。
- **検証の指標**：RMSE（ln η）・被覆率・NLPD（版ごとの検証の分け方）。
- **設定温度の効果**：190/230 ℃ の粘度比（厚みごと）と、実測。

判定：①本命・対案の RMSE が、基準 C の 1.1 倍以内か（同等なら、構造がデータで支持された）。②(2) の RMSE が (1) の 1.1 倍以内か（分けても精度が落ちない）。③(1+4) が (1) より 10% 以上は良くならないか（H があれば P3_2 は要らない）。④被覆率が 0.90〜0.99。⑤ノイズが純粋誤差の 0.8 倍以上。''')
code(r'''# セル15：本命と対案の比較
ROLES = [("本命", MAIN_STRUCT), ("対案", ALT_STRUCT), ("基準", "C")]
cmp_rows = []
for ver in DATA_VERSIONS:
    pn = panels_of(ver)
    for mat in MATS:
        for hv in H_VARIANTS:
            for role, st in ROLES:
                if st not in STRUCTS:
                    continue
                row = {"material": MATS[mat], "version": ver, "hv": hv, "role": role, "struct": st, "model": mname(st, hv)}
                if st != "C" and len(best_df):
                    b = best_row(mat, ver, "ALL", st, hv)
                    row.update({"Eh": b["Eh_best"], "Ev": b["Ev_best"], "diff": b["diff_best"], "LML": b["LML_best"], "dLML_vs_E0": b["dLML_vs_E0"], "diff_95_lo": b["diff_95_lo"], "diff_95_hi": b["diff_95_hi"]})
                for sp_, sub_, lab_ in pn:
                    for metric in ["RMSE", "coverage95", "NLPD"]:
                        row[f"{metric}_{lab_}"] = mv(mat, ver, sp_, st, hv, sub_, col=metric)
                    if has_p(hv):
                        row[f"RMSE_{lab_}（P3_2 予測）"] = mv(mat, ver, sp_, st, hv, sub_, "予測")
                e = eff5[(eff5["version"] == ver) & (eff5["material"] == MATS[mat]) & (eff5["model"] == mname(st, hv))].sort_values("H")
                for h, r_ in zip(e["H"], e["ratio_190_230"]):
                    row[f"ratio_{h}"] = float(r_)
                cmp_rows.append(row)
cmp_df = pd.DataFrame(cmp_rows)
cmp_df.to_csv(os.path.join(OUT_DIR, "main_vs_alt.csv"), index=False, encoding="utf-8-sig")
for ver in DATA_VERSIONS:
    for mat in MATS:
        n_ = MATS[mat]
        pn = panels_of(ver)
        t = cmp_df[(cmp_df["version"] == ver) & (cmp_df["material"] == n_)]
        print(f"\n【{n_}・{ver} 版】E（ALL）と、検証の RMSE（P3_2 を使うモデルは実測の P3_2）")
        print(t[["hv", "role", "struct", "Eh", "Ev", "diff", "dLML_vs_E0"] + [f"RMSE_{lab}" for _, _, lab in pn]].round(3).to_string(index=False))
        for hv in H_VARIANTS:
            for role, st in ROLES[:2]:
                if st not in STRUCTS or "C" not in STRUCTS:
                    continue
                bad = [lab for sp_, sub_, lab in pn if mv(mat, ver, sp_, st, hv, sub_) > 1.1 * mv(mat, ver, sp_, "C", hv, sub_)]
                judge(not bad, f"{n_}・{ver} 版・扱い {hv}：{role} {st} の RMSE が、基準 C の 1.1 倍以内（超えた検証: {bad or 'なし'}）")
        for st in STRUCTS:
            if "1" in H_VARIANTS and "2" in H_VARIANTS:
                bad = [lab for sp_, sub_, lab in pn if mv(mat, ver, sp_, st, "2", sub_) > 1.1 * mv(mat, ver, sp_, st, "1", sub_)]
                judge(not bad, f"{n_}・{ver} 版・{st}：(2) の RMSE が (1) の 1.1 倍以内（超えた検証: {bad or 'なし'}）。分けても精度が落ちない")
            if "1" in H_VARIANTS and "14" in H_VARIANTS:
                good = [lab for sp_, sub_, lab in pn if mv(mat, ver, sp_, st, "14", sub_) < 0.9 * mv(mat, ver, sp_, st, "1", sub_)]
                judge(not good, f"{n_}・{ver} 版・{st}：(1+4) が (1) より 10% 以上は良くならない（良くなった検証: {good or 'なし'}）。H があれば P3_2 は要らない")
        m_ = met_df[(met_df["material"] == n_) & (met_df["version"] == ver) & (met_df["subset"] == "全体") & (met_df["psrc"] != "予測")]
        judge(bool(m_["coverage95"].between(0.90, 0.99).all()), f"{n_}・{ver} 版：被覆率が全モデルで 0.90〜0.99（外れたモデル: {sorted(set(m_[~m_['coverage95'].between(0.90, 0.99)]['model']))}）")
        judge(bool((m_["noise_sd"] >= 0.8 * m_["pure_sd"]).all()), f"{n_}・{ver} 版：ノイズの標準偏差が、純粋誤差の 0.8 倍以上（下回ったモデル: {sorted(set(m_[m_['noise_sd'] < 0.8 * m_['pure_sd']]['model']))}）")''')

# ---------------------------------------------------------------- 9. 図
md(r'''## 9. 図

- 図1：A のプロファイル尤度（E_v と、最大との LML の差。H の扱いごとの線）
- 図2：B の 2 次元プロファイル（E_h × E_v。斜めの帯）
- 図3：マスターカーブ（換算後のデータ）
- 図4：設定温度ごとの予測曲線と予測区間（B・A・C、H の扱いごと）
- 図5：評価指標の比較
- 図6：設定温度の効果（190/230 ℃ の粘度比）
- 図7：(2) の分解（m と c）
- 図8：(4) の P3_2 の実測と予測

すべて `save_fig` で保存する（通常の PNG と、`highres/` の 600 dpi）。''')
code(r'''# セル16：図の共通の設定
ST_COL = {"B": "tab:purple", "A": "tab:blue", "C": "k"}
HV_COL = {"1": "tab:red", "2": "tab:green", "4": "tab:blue", "14": "tab:orange", "24": "tab:brown"}
T_COL = {190: "tab:blue", 210: "tab:green", 230: "tab:red"}
H_COL = {0.5: "#d62728", 1.0: "#1f77b4", 1.5: "#f0b400", 2.0: "#2ca02c"}
FIG_HV = [hv for hv in FIG_VARIANTS if hv in H_VARIANTS]


def points_pool(mat, ver):
    """曲線の図の点に使うデータ。5data 版：5data＋3data（3data は検証の条件）／Alldata 版：Alldata"""
    return pool_of(mat, MAIN_DATA) if ver == "5data" else D[(mat, "Alldata", MAIN_DATA)]


def train_conds_of(mat, ver):
    return set(tr_of(mat, ver, "ALL")["cond"])


def figname(stem, *parts):
    return "_".join([stem, *[str(p) for p in parts]])


def main_set(ver):
    return "S_V" if ver == "5data" else "ALL"''')
code(r'''# セル17：図1 A のプロファイル尤度（E_v と、最大との LML の差）
if len(prof_df) and "A" in PROF_STRUCTS:
    for ver in DATA_VERSIONS:
        sets_ = ["S_V", "S_T", "ALL"] if ver == "5data" else ["S_T", "ALL"]
        fig, axes = plt.subplots(len(MATS), len(sets_), figsize=(5.6 * len(sets_), 4.6 * len(MATS)), squeeze=False)
        for i, mat in enumerate(MATS):
            for j, sname in enumerate(sets_):
                ax = axes[i, j]
                sn = ALIAS[(mat, ver)] if (sname == "ALL" and (mat, ver) in ALIAS) else sname
                for hv in H_VARIANTS:
                    g = prof_df[(prof_df["mat"] == mat) & (prof_df["version"] == ver) & (prof_df["set"] == sn) & (prof_df["struct"] == "A") & (prof_df["hv"] == hv)].sort_values("Ev")
                    if g.empty:
                        continue
                    ax.plot(g["Ev"], g["LML"] - g["LML"].max(), "-o", ms=2.5, lw=1.2, color=HV_COL[hv], label=HV_JA[hv])
                    b = g.loc[g["LML"].idxmax()]
                    ax.plot([b["Ev"]], [0.0], "*", ms=13, color=HV_COL[hv], mec="k")
                ax.axhline(-1.92, color="gray", ls="--", lw=0.8)
                ax.axvline(0, color="gray", lw=0.8)
                ax.set_ylim(-30, 2)
                ax.set_title(f"{MATS[mat]}・{ver} 版：{sname} の学習", fontsize=10)
                ax.set_xlabel("A の E_v [kJ/mol]")
                ax.set_ylabel("対数周辺尤度 − その最大")
                ax.grid(alpha=0.3)
                if i == 0 and j == 0:
                    ax.legend(fontsize=7, loc="lower right")
        fig.suptitle(f"{ver} 版：A のプロファイル尤度。★＝最良、破線＝最大から 1.92（おおよその 95% 区間の境）、縦線＝E = 0（温度の効果なし）", fontsize=11)
        fig.tight_layout()
        save_fig(fig, OUT_DIR, figname("fig1_profile_A", ver))
        plt.show()''')
code(r'''# セル18：図2 B の 2 次元プロファイル（斜めの帯）
if len(prof_df) and "B" in PROF_STRUCTS:
    for ver in DATA_VERSIONS:
        for mat in MATS:
            sn = ALIAS[(mat, ver)] if (main_set(ver) == "ALL" and (mat, ver) in ALIAS) else main_set(ver)
            fig, axes = plt.subplots(1, len(H_VARIANTS), figsize=(4.6 * len(H_VARIANTS), 4.8), squeeze=False)
            for j, hv in enumerate(H_VARIANTS):
                ax = axes[0, j]
                g = prof_df[(prof_df["mat"] == mat) & (prof_df["version"] == ver) & (prof_df["set"] == sn) & (prof_df["struct"] == "B") & (prof_df["hv"] == hv)]
                if g.empty:
                    continue
                b = g.loc[g["LML"].idxmax()]
                w_ = 10.0                                   # 最良の点の近く（±10 kJ/mol）を拡大して見せる。全体は、粗い格子の点が、ほぼ一様に低い（LML が最大より 40 以上低い）
                gz = g[(abs(g["Eh"] - b["Eh"]) <= w_) & (abs(g["Ev"] - b["Ev"]) <= w_)]
                z = np.clip(gz["LML"].to_numpy() - b["LML"], -40, 0)
                sc_ = ax.tricontourf(gz["Eh"], gz["Ev"], z, levels=np.linspace(-40, 0, 21), cmap="viridis")
                ax.scatter(gz["Eh"], gz["Ev"], s=5, color="k", alpha=0.35)
                ins = gz["LML"] >= b["LML"] - 1.92
                ax.scatter(gz.loc[ins, "Eh"], gz.loc[ins, "Ev"], s=40, facecolors="none", edgecolors="w", linewidths=1.2)
                ax.plot([b["Eh"]], [b["Ev"]], "*", ms=14, color="r", mec="w", label=f"最良（{b['Eh']:.0f}, {b['Ev']:.0f}）")
                lim = [b["Eh"] - w_ - 2, b["Eh"] + w_ + 2]
                ax.plot(lim, lim, "k--", lw=0.8, label="E_h = E_v")
                ax.plot(lim, [x + (b["Ev"] - b["Eh"]) for x in lim], "r:", lw=0.8, label=f"E_v − E_h = {b['Ev'] - b['Eh']:.0f}")
                ax.axvline(0, color="gray", lw=0.6, ls=":")
                ax.set_xlim(b["Eh"] - w_, b["Eh"] + w_)
                ax.set_ylim(b["Ev"] - w_, b["Ev"] + w_)
                ax.set_title(HV_JA[hv], fontsize=10)
                ax.set_xlabel("横方向 E_h [kJ/mol]")
                ax.set_ylabel("縦方向 E_v [kJ/mol]")
                ax.legend(fontsize=6, loc="upper left")
                if j == len(H_VARIANTS) - 1:
                    fig.colorbar(sc_, ax=ax, label="LML − 最大")
            fig.suptitle(f"{MATS[mat]}・{ver} 版（{sn} の学習）：B の対数周辺尤度（最良の点の ±10 kJ/mol を拡大）。白丸：最大から 1.92 以内。尤度の山は E_v − E_h が一定の斜めの帯", fontsize=11)
            fig.tight_layout()
            save_fig(fig, OUT_DIR, figname("fig2_profile_B", ver, mat))
            plt.show()''')
code(r'''# セル19：図3 マスターカーブ（ALL の学習データを、最良の E で換算）
if len(best_df):
    hv_m = "1" if "1" in H_VARIANTS else H_VARIANTS[0]
    cols_m = [("N", "シフトなし（E = 0）")] + [(s, f"{s}：{STRUCT_JA5[s]}") for s in PROF_STRUCTS[::-1]]
    for ver in DATA_VERSIONS:
        for mat in MATS:
            cdf = cond_table5(tr_of(mat, ver, "ALL"))
            fig, axes = plt.subplots(4, len(cols_m), figsize=(4.8 * len(cols_m), 13), sharey="row", squeeze=False)
            for j, (name, lab) in enumerate(cols_m):
                if name == "N":
                    v = {"struct": "A", "hv": hv_m, "Eh": 0.0, "Ev": 0.0}
                else:
                    r = best_row(mat, ver, "ALL", name, hv_m)
                    v = {"struct": name, "hv": hv_m, "Eh": float(r["Eh_best"]), "Ev": float(r["Ev_best"])}
                X0_, Xs_, ysh_ = design5(cdf, v)
                for i, h in enumerate(H_LEVELS):
                    ax = axes[i, j]
                    for t, col in T_COL.items():
                        m = ((cdf["H"] == h) & (cdf["Tset"] == t)).to_numpy()
                        ax.scatter(Xs_[m, 0], cdf["ln_eta"].to_numpy()[m] - ysh_[m], s=16, color=col, alpha=0.8, label=f"{t} ℃")
                    ax.grid(alpha=0.3)
                    if j == 0:
                        ax.set_ylabel(f"厚み {h} mm\n" + r"換算後の $\ln\eta$（縦シフトを引いた値）")
                    if i == 0:
                        ax.set_title(f"{lab}\n(E_h, E_v) = ({v['Eh']:.1f}, {v['Ev']:.1f}) kJ/mol", fontsize=10)
                    if i == 3:
                        ax.set_xlabel(r"換算後の $\ln\dot{\gamma}$ ＋ 横シフト")
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(f"{MATS[mat]}・{ver} 版：マスターカーブ（条件平均。H の扱い {hv_m}。3 つの温度の点が重なっていれば、その構造でシフトできている）", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            save_fig(fig, OUT_DIR, figname("fig3_master", ver, mat))
            plt.show()''')
code(r'''# セル20：図4 設定温度ごとの予測曲線と予測区間（最終モデル ALL。列が構造、行が厚み）
for ver in DATA_VERSIONS:
    for mat in MATS:
        pool, tconds = points_pool(mat, ver), train_conds_of(mat, ver)
        for hv in FIG_HV:
            fig, axes = plt.subplots(4, len(STRUCTS), figsize=(4.8 * len(STRUCTS), 14), sharex="row", sharey="row", squeeze=False)
            for j, st in enumerate(STRUCTS):
                fit = get_fit(mat, ver, "ALL", st, hv)
                for i, h in enumerate(H_LEVELS):
                    ax = axes[i, j]
                    for t, col in T_COL.items():
                        sub = pool[(pool["H"] == h) & (pool["Tset"] == t)]
                        if sub.empty:
                            continue
                        cm = sub.groupby("cond").agg(lg=("ln_gd", "mean"), le=("ln_eta", "mean"), sens=("is_sens", "max")).reset_index()
                        lg = np.linspace(sub["ln_gd"].min(), sub["ln_gd"].max(), 120)
                        mu, sd, _, _ = predict5(fit, hr_inputs(sub, lg, h, t))
                        ax.plot(np.exp(lg), np.exp(mu), color=col, lw=1.4)
                        ax.fill_between(np.exp(lg), np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd), color=col, alpha=0.12)
                        is_tr = cm["cond"].isin(tconds).to_numpy()
                        ax.scatter(np.exp(cm["lg"][is_tr]), np.exp(cm["le"][is_tr]), s=14, color=col, alpha=0.8, zorder=3)
                        ax.scatter(np.exp(cm["lg"][~is_tr]), np.exp(cm["le"][~is_tr]), s=26, facecolors="none", edgecolors=col, marker="s", linewidths=1.2, zorder=3)
                        se = cm["sens"].to_numpy().astype(bool)
                        ax.scatter(np.exp(cm["lg"][se]), np.exp(cm["le"][se]), s=45, color="k", marker="x", zorder=4)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))      # 範囲が狭い軸でも、目盛りの数字が重ならないように
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda x_, _: f"{x_:g}"))
                    ax.xaxis.set_minor_formatter(NullFormatter())
                    ax.grid(alpha=0.3, which="both")
                    if j == 0:
                        ax.set_ylabel(f"厚み {h} mm\n" + r"$\eta$ [Pa·s]")
                    if i == 0:
                        v = fit["v"]
                        ax.set_title(mname(st, hv) + ("" if st == "C" else f"\n(E_h, E_v) = ({v['Eh']:.1f}, {v['Ev']:.1f})"), fontsize=10)
                    if i == 3:
                        ax.set_xlabel(r"$\dot{\gamma}$ [1/s]")
            handles = [plt.Line2D([], [], color=c, label=f"{t} ℃") for t, c in T_COL.items()]
            handles += [plt.Line2D([], [], color="gray", marker="o", ls="", label="学習の条件平均"), plt.Line2D([], [], color="gray", marker="s", mfc="none", ls="", label="学習に使わない条件"),
                        plt.Line2D([], [], color="k", marker="x", ls="", label="感度確認の条件")]
            axes[0, 0].legend(handles=handles, fontsize=7, loc="upper right")
            note = "　※ (4) を含む曲線の P3_2 は、同じ厚み・温度の条件平均を内挿した値（予測のときには分からない）" if has_p(hv) else ""
            fig.suptitle(f"{MATS[mat]}・{ver} 版・H の扱い {HV_JA[hv]}：設定温度ごとの予測曲線（線：予測、帯：95% 予測区間）{note}", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            save_fig(fig, OUT_DIR, figname("fig4_curves", ver, mat, hv))
            plt.show()''')
code(r'''# セル21：図5 評価指標の比較（RMSE と NLPD）
for ver in DATA_VERSIONS:
    pn = panels_of(ver)
    labels = [(st, hv, ps) for hv in H_VARIANTS for st in STRUCTS for ps in (["実測", "予測"] if has_p(hv) else ["―"])]
    fig, axes = plt.subplots(2, len(MATS), figsize=(9.5 * len(MATS), 9.5), squeeze=False)
    colors = ["tab:blue", "tab:red", "tab:purple", "tab:brown", "tab:green", "tab:orange", "gray"]
    for j, mat in enumerate(MATS):
        for i, (metric, ylab) in enumerate([("RMSE", r"RMSE（$\ln\eta$）"), ("NLPD", "NLPD（小さいほど良い）")]):
            ax = axes[i, j]
            w = 0.8 / len(pn)
            for k, (sp_, sub_, lab_) in enumerate(pn):
                ax.bar(np.arange(len(labels)) + (k - (len(pn) - 1) / 2) * w, [mv(mat, ver, sp_, st, hv, sub_, ps if ps != "―" else None, metric) for st, hv, ps in labels], w, color=colors[k], label=lab_)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels([mname(st, hv) + {"実測": "(実)", "予測": "(予)", "―": ""}[ps] for st, hv, ps in labels], rotation=75, fontsize=7)
            ax.set_title(f"{MATS[mat]}・{ver} 版：{ylab}（入力の " + r"$\dot{\gamma}$" + " は条件平均。(実)＝実測の P3_2、(予)＝予測した P3_2）", fontsize=9)
            ax.grid(alpha=0.3, axis="y")
            if metric == "RMSE":
                ax.set_yscale("log")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, OUT_DIR, figname("fig5_metrics", ver))
    plt.show()''')
code(r'''# セル22：図6 設定温度の効果
MK = {"B": "P", "A": "v", "C": "s"}
LS = {"1": "-", "2": "--", "4": ":", "14": "-.", "24": (0, (3, 1, 1, 1))}
for ver in DATA_VERSIONS:
    fig, axes = plt.subplots(1, len(MATS), figsize=(8.5 * len(MATS), 5.4), sharey=True, squeeze=False)
    for ax, mat in zip(axes[0], MATS):
        o = obs5[obs5["material"] == MATS[mat]]
        ax.plot(o["H"], o["observed"], "k-o", lw=2.8, ms=9, label=r"実測（同じ $\dot{\gamma}$）", zorder=10)
        for st in STRUCTS:
            for hv in FIG_HV:
                e = eff5[(eff5["version"] == ver) & (eff5["material"] == MATS[mat]) & (eff5["model"] == mname(st, hv))]
                ax.plot(e["H"], e["ratio_190_230"], marker=MK[st], ls=LS[hv], alpha=0.85, color=ST_COL[st], label=mname(st, hv))
        ax.axhline(1.0, color="gray", lw=0.8)
        ax.set_xlabel("厚み [mm]")
        ax.set_title(f"{MATS[mat]}・{ver} 版：190 ℃ と 230 ℃ の粘度の比（Tset だけを変えた予測）")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("粘度の比（190 ℃ ÷ 230 ℃）")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_fig(fig, OUT_DIR, figname("fig6_tset_effect", ver))
    plt.show()''')
code(r'''# セル23：図7 (2) の分解（m と c）、図8 (4) の P3_2 の実測と予測
if len(dec_df):
    for ver in DATA_VERSIONS:
        for mat in MATS:
            fig, axes = plt.subplots(2, len(STRUCTS), figsize=(5.2 * len(STRUCTS), 9), squeeze=False)
            for j, st in enumerate(STRUCTS):
                d = dec_df[(dec_df["material"] == MATS[mat]) & (dec_df["version"] == ver) & (dec_df["struct"] == st)]
                ax = axes[0, j]
                for t, col in T_COL.items():
                    g = d[(d["kind"] == "m_curve") & (d["Tset"] == t)]
                    ax.plot(np.exp(g["x"]), np.exp(g["mean"]), color=col, lw=1.6, label=f"{t} ℃")
                    ax.fill_between(np.exp(g["x"]), np.exp(g["mean"] - 1.96 * g["sd"]), np.exp(g["mean"] + 1.96 * g["sd"]), color=col, alpha=0.15)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel(r"$\dot{\gamma}$ [1/s]")
                ax.set_ylabel(r"材料の部分 $\eta_m$ [Pa·s]（c を平均 0 にそろえた）")
                ax.set_title(f"{mname(st, '2')}：m（材料の部分）", fontsize=10)
                ax.grid(alpha=0.3, which="both")
                if j == 0:
                    ax.legend(fontsize=8)
                ax = axes[1, j]
                g = d[d["kind"] == "c_curve"]
                ax.plot(g["x"], g["mean"], "k-", lw=1.6)
                ax.fill_between(g["x"], g["mean"] - 1.96 * g["sd"], g["mean"] + 1.96 * g["sd"], color="gray", alpha=0.25)
                for h, col in H_COL.items():
                    gg = d[(d["kind"] == "c_cond") & (d["H"] == h)]
                    ax.scatter(gg["x"], gg["mean"], s=14, color=col, label=f"{h} mm", zorder=3)
                ax.axhline(0.0, color="gray", lw=0.8)
                ax.set_xlabel(r"$\ln$ Fo $= -(\ln\dot{\gamma} + 3\ln H)$ ＋ 定数")
                ax.set_ylabel(r"厚み・冷却の補正 $c$（$\ln\eta$ の単位）")
                ax.set_title(f"{mname(st, '2')}：c(Fo)（厚み・冷却の補正）", fontsize=10)
                ax.grid(alpha=0.3)
                if j == 0:
                    ax.legend(fontsize=8, title="学習の条件（厚み）")
            fig.suptitle(f"{MATS[mat]}・{ver} 版：(2) の分解。上：材料の部分 m、下：厚み・冷却の補正 c（帯は成分の 95% の帯）", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            save_fig(fig, OUT_DIR, figname("fig7_decomposition", ver, mat))
            plt.show()
if len(p32_df):
    for ver in DATA_VERSIONS:
        q = p32_df[(p32_df["version"] == ver) & (p32_df["split"].isin(["S_V", "CV"]))]
        if q.empty:
            q = p32_df[p32_df["version"] == ver]
        fig, axes = plt.subplots(1, len(MATS), figsize=(6.2 * len(MATS), 5.6), squeeze=False)
        for ax, mat in zip(axes[0], MATS):
            g = q[q["material"] == MATS[mat]]
            for h, col in H_COL.items():
                gg = g[g["H"] == h]
                ax.errorbar(gg["actual"], gg["pred"], yerr=1.96 * gg["pred_sd"], fmt="o", ms=4, color=col, alpha=0.7, elinewidth=0.6, label=f"{h} mm")
            lim = [g["actual"].min() - 0.2, g["actual"].max() + 0.2]
            ax.plot(lim, lim, "k--", lw=0.8)
            ax.set_xlabel(r"実測の $\ln P_{3\_2}$（条件平均、Pa）")
            ax.set_ylabel(r"予測した $\ln P_{3\_2}$（95% 区間つき）")
            ax.set_title(f"{MATS[mat]}・{ver} 版：検証の条件の P3_2（RMSE {float(np.sqrt(np.mean((g['pred'] - g['actual']) ** 2))):.3f}）", fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("(4) で使う P3_2 の予測（(ln γ̇, ln H, Tset) から GPR で予測。検証の条件）。対角線に乗るほど、P3_2 を事前に予測できる", fontsize=10)
        fig.tight_layout()
        save_fig(fig, OUT_DIR, figname("fig8_p32_pred", ver))
        plt.show()''')

# ---------------------------------------------------------------- 10. 判定のまとめ
md(r'''## 10. 判定のまとめ

OK は想定どおり、★ は想定と違う、または注意が要る点。基準は、判定の文の中に書いてある。上の各節の判定を、ここに集めて数える。''')
code(r'''# セル24：保存したファイルの一覧
print("保存したファイル:")
for f in sorted(os.listdir(OUT_DIR)):
    if os.path.isfile(os.path.join(OUT_DIR, f)):
        print("  ", os.path.normpath(os.path.join(OUT_DIR, f)))''')

# ---------------------------------------------------------------- 11. 高解像度のグラフ
md(HR_MD.replace("## 高解像度のグラフ", "## 11. 高解像度のグラフ").replace("**レイアウト**", "**モデル**：`HR_MODELS` に「構造_扱い」（例：B_1、A_2、C_4）を並べる。最終モデル（ALL）の曲線。(4) を含むモデルの曲線は、P3_2 を同じ厚み・温度の条件平均から内挿して描く（予測には使えない値。タイトルに書く）。\n- **レイアウト**"))
code(hr_cell("# セル25：高解像度のグラフ（せん断速度 ― 粘度。最終モデルの予測。GPR6月ノートのセル9-2 にならう）",
             '''HR_VERSION = DATA_VERSIONS[-1]     # どの版のモデルか："5data"（5data で学習。3data の条件は白抜き）／"Alldata"（全条件で学習）
HR_MODELS = ["B_1", "A_1", "B_2", "A_2"]   # 描くモデル（「構造_扱い」。STRUCTS・H_VARIANTS に入れて学習したもの）''',
             r'''hr_saved = []
for mat in HR_MATERIALS:
    pool, tconds = points_pool(mat, HR_VERSION), train_conds_of(mat, HR_VERSION)
    for name in HR_MODELS:
        st, hv = name.split("_")
        if (mat, HR_VERSION, "ALL", st, hv) not in RES5:
            print(f"{MATS[mat]}・{HR_VERSION} 版・{name} は、学習していないので描かない（設定の DATA_VERSIONS・STRUCTS・H_VARIANTS に入れて、上から実行し直す）")
            continue
        t0 = time.time()
        fit = get_fit(mat, HR_VERSION, "ALL", st, hv)
        v = fit["v"]
        label = f"{name}（{HV_JA[hv]}）" + ("" if st == "C" else f"　E_h = {v['Eh']:.1f}、E_v = {v['Ev']:.1f} kJ/mol") + f"（{HR_VERSION} 版）" + ("　※P3_2 は条件平均の内挿（予測には使えない）" if has_p(hv) else "")
        hr_saved += hr_flow(lambda Xg, f=fit: predict5(f, Xg)[:3], mat, MAIN_DATA, tconds, label, f"hr_{mat}_{HR_VERSION}_{name}", HR_DIR, pool=pool, **HR_KW)
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
