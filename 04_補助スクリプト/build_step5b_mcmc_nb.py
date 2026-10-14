# -*- coding: utf-8 -*-
"""手順5b（最終候補 B_1・C_1 の MCMC）のノートブック (未実行) を生成する。

使い方: python build_step5b_mcmc_nb.py <出力する .ipynb のパス> [--force]
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


def core_cell(fname, title):
    body = open(os.path.join(HERE, fname), encoding="utf-8").read().split('"""', 2)[2].lstrip("\n")
    return f"# セル：{title}。04_補助スクリプト/{fname} の中身。直すときは、そのファイルを直して、ノートを作り直す\n" + body


# ---------------------------------------------------------------- 0. タイトル
md(r'''# 手順5b：最終候補（B_1・C_1）を MCMC で推定

CLAUDE.md §5・§7。手順5 で決まった最終候補 **B_1**（構造 B：温度で横にも縦にもずらす。厚み H を残す）と、比較用の **C_1**（構造なし。手順3b・3d の R3）を、MCMC（自作の適応型ランダムウォーク・メトロポリス法。手順2・3c と同じ）でも推定し、最尤推定と比べる。B_1 は、温度のずらし量 E_h・E_v も、ハイパーパラメータと**同時にサンプリング**する（ユーザー決定、10-14）。データは、手順3d の 2 つの版（5data 版・Alldata 版）。

- **何を確かめるか**：(1) E_h・E_v の事後分布（手順5 のプロファイル尤度の結果と合うか。斜めの帯〔差 E_v − E_h が決まる〕になるか）。(2) ハイパーパラメータが、探索範囲の端に張り付いていないか。(3) ハイパーパラメータと E の不確かさを予測に含めたとき、予測区間の広さ・被覆率・NLPD が、最尤推定とどれだけ違うか。(4) 設定温度の効果（190/230 ℃ の粘度比）の不確かさと、実測との関係。

## モデルの構成（手順5 と同じ）

| 項目 | 内容 |
|---|---|
| 学習データ | 外れ値なし。問題データ一覧の「除外」（P01・P03・P08）を除く。1 行＝1 ショット。5data 版：5data（PP 460 行・混合 459 行）／Alldata 版：Alldata（PP 1234 行・混合 560 行） |
| 目的変数 | ln η（η：見かけの粘度 [Pa·s]）。学習データの平均 y0 を引く（固定の定数。自動標準化はしない） |
| 入力：C_1 | 条件平均の ln γ̇、ln H、Tset［℃］（3 列。学習データの平均・標準偏差で標準化） |
| 入力：B_1 | u = 条件平均の ln γ̇ ＋ ln a(Tset; E_h)、w = ln Fo = −(ln γ̇ ＋ 3 ln H)（2 列）。出力から ln a(Tset; E_v) を引く。ln a(T; E) = (E/R)(1/T − 1/Tref)、Tref = 483.15 K、R = 8.314 |
| 平均関数 | 明示的なものはない（ガウス過程の平均はゼロ）。定数 y0、B_1 の縦シフト ln a(Tset; E_v)、カーネルの線形項が、平均の役目をする |
| カーネル | C1·Matern(ν = 5/2、入力ごとの長さスケール〔ARD〕) ＋ C2·DotProduct(σ0) ＋ ノイズ（WhiteKernel、分散 σn²） |
| 事前分布（カーネル） | 手順3c と同じ。探索範囲の中で、対数スケールの一様分布（C1・C2：1e-3〜1e3、長さスケール：1e-2〜1e2、σ0：1e-5〜1e3、σn²：1e-6〜10）。範囲の外は確率 0 |
| 事前分布（E） | 範囲の中で一様分布（`E_PRIOR`。E_h：−40〜40、E_v：−20〜40 kJ/mol。手順5 のプロファイル尤度の格子と同じ範囲） |
| 関数の事前分布 | f ~ ガウス過程（平均 0、共分散 上のカーネル）。ノイズ ε ~ N(0, σn²) |

**条件ごとにまとめた尤度（厳密）**：入力が条件平均なので、同じ条件のショットは入力が同じ。ショットごとの尤度は、条件ごとの平均・ショット数・条件内の平方和を使って、厳密に「条件内のばらつきの項」と「条件平均の GP の項」に分けられる（`step5b_core.py`）。条件の数（93〜129）の大きさの行列だけで済むので、MCMC の 1 回の評価が約 1 ミリ秒になる（Alldata 版でも同じ）。ノートの最初の節で、scikit-learn のショットごとの GPR と、対数周辺尤度・予測が一致することを確かめる。

**予測**：事後サンプルを等間隔に `N_PRED_DRAWS` 個選び、各サンプルでの予測分布（ノイズ込みの正規分布）を等しい重みで混ぜた混合分布。P3_2 を使うモデル（H の扱い 4・14・24）を指定したときは、実測の P3_2（条件平均）で予測する。

### 自作の適応型ランダムウォーク・メトロポリス法（手順3c と同じ）
1. 今いる点 φ（カーネルの対数パラメータと E）の近くに、正規分布 N(φ, λ·Σ) から候補を出す。
2. 候補の対数事後密度（範囲内なら対数尤度、範囲外なら −∞）を計算し、確率 min(1, exp(候補 − 現在)) で移動する。
3. **調整期間**：前半 75% では、それまでの足跡から Σ（パラメータどうしの相関を含む）を学ぶ。調整期間の間ずっと、採択率が 0.234 に近づくよう λ を合わせる。調整期間のサンプルは捨てる。
4. **本番**：Σ と λ を固定して歩く。R-hat が 1.01 を超えるか、有効サンプル数（ESS）が 400 に届かないときは、本番の回数を増やして、サンプリングし直す（最大 `N_SAMP_MAX` 回）。
5. 4 本のチェーンを別々の初期値（最尤推定値に、ゆらぎを足したもの）から回し、R-hat と ESS で収束を確かめる。''')

# ---------------------------------------------------------------- 1. 設定
md(usage_md("5b", "results/step5b_mcmc", "results/学習結果説明_手順5b.pdf", 30) + r'''
8. サンプリングには時間がかかる（約 30 分）。`USE_SAVED_SAMPLES = True` のときは、保存済みの事後サンプル（`results/step5b_mcmc/posterior_*.csv`）があれば読み込み、サンプリングを省く。データを取り直したときは `False` にする（`rerun_all.py` は、実行の前に保存済みのサンプルを消す）。''')
md(r'''## 1. 設定''')
code(settings_cell('''import numpy as np             # 範囲の指定に使う
MCMC_MODELS = ["B_1", "C_1"]  # MCMC で推定するモデル（「構造_H の扱い」。手順5 の 15 モデルのどれでもよい。H の扱い：1＝H を残す、2＝m＋c(Fo)、4＝P3_2、14、24）
SETS5B = {"5data": ["S_V", "S_T"],          # 版ごとの学習データ。5data 版：S_V（5data で学習。3data を検証）・S_T（190・230 ℃ で学習。210 ℃ を検証）
          "Alldata": ["S_T", "ALL"]}        # Alldata 版：S_T・ALL（全条件で学習した最終モデル。検証データはない）
DATA_VERSIONS = ["5data", "Alldata"]       # データの版（手順3d）
MAIN_DATA = "なし"            # 主に使う外れ値の扱い（外れ値なし）
TEMP_HOLDOUT = 210            # S_T でテストにする設定温度 [℃]
E_PRIOR = {"Eh": (-40.0, 40.0), "Ev": (-20.0, 40.0)}   # E の事前分布（一様分布の範囲）[kJ/mol]
E_START = {"B": (7.0, 16.0), "A": (0.0, 7.0)}           # 手順5 の結果（best_E.csv）がないときの、E の出発点 [kJ/mol]
N_WARM, N_SAMP, CHAINS = 5000, 20000, 4    # 調整の反復数（捨てる）、本番の反復数、チェーン数（1 回の評価が約 1 ミリ秒なので、手順3c より多くできる）
N_SAMP_MAX = 100000           # R-hat が 1.01 を超えたときに増やす本番の回数の上限
N_JOBS = 8                    # 並列に回す数（チェーンと最尤推定。CPU の数に合わせる）
N_PRED_DRAWS = 200            # 予測に使う事後サンプル数（間引き後）
USE_SAVED_SAMPLES = True      # 保存済みの事後サンプルがあれば読み込む（サンプリングを省く）
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
import arviz as az   # 収束診断（R-hat・有効サンプル数）に使う

OUT_DIR = os.path.join(REPO, "results", "step5b_mcmc")
STEP5_DIR = os.path.join(REPO, "results", "step5")      # 手順5 の結果（最良の E を読む）
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
print("ArviZ :", az.__version__)
print("出力先:", os.path.abspath(OUT_DIR))''')
md(r'''### 共通の部品

データの読み込み、問題データ一覧の「除外」、評価、データの分け方、列を選んで当てるカーネル `OnDims` など。中身は `04_補助スクリプト/nb_common.py` と同じ。''')
code(COMMON_CELL)
md(r'''### 手順5 の部品

構造と H の扱い（`layout5`・`design5`・`kernel5`）、学習と予測（`fit5`・`predict5`）、条件平均の表（`cond_table5`）など。中身は `04_補助スクリプト/step5_core.py` と同じ。''')
code(core_cell("step5_core.py", "手順5の部品（構造・H の扱い・プロファイル尤度・P3_2 の予測・学習の仕事）"))
md(r'''### 手順5b の部品

条件ごとにまとめた尤度（`make_grouped`・`grouped_lml`）、予測（`grouped_predict`・`predict_mixture`）、対数事後密度と事前分布（`make_logpost`）、最尤推定（`fit_mle5`）、適応型メトロポリス法（`run_chain5`）、評価・収束診断（`evaluate_mixture`・`diag_summary`）。中身は `04_補助スクリプト/step5b_core.py` と同じ。''')
code(core_cell("step5b_core.py", "手順5bの部品（条件ごとにまとめた尤度・MCMC・最尤推定・混合予測）"))

# ---------------------------------------------------------------- 2. データと部品の確認
md(r'''## 2. データの読み込みと、部品の確認

手順3d の 2 つの版の分け方（`version_sets`）で、学習データとテストデータを作る。`SETS5B` で選んだ学習データごと・モデルごとに、MCMC を行う。

**部品の確認**（判定）：条件ごとにまとめた対数尤度と予測が、scikit-learn のショットごとの GPR（同じカーネルの値）と、1e-6 以内で一致する。''')
code(r'''# セル2：読み込みと、学習データ・テストデータの用意
D = load_all()
VS = {(mat, ver): version_sets(mat, ver, MAIN_DATA, TEMP_HOLDOUT) for ver in DATA_VERSIONS for mat in MATS}
TRAIN_DF, TESTS = {}, {}
for (mat, ver), vs in VS.items():
    for n in SETS5B[ver]:
        if n == "ALL":
            TRAIN_DF[(mat, ver, n)], TESTS[(mat, ver, n)] = vs["pool"], {}
        elif n in vs["splits"]:
            TRAIN_DF[(mat, ver, n)], TESTS[(mat, ver, n)] = vs["splits"][n]
        else:
            print(f"（{MATS[mat]}・{ver} 版に、分け方 {n} はない。飛ばす）")
MODEL_SPEC = {key: tuple(key.split("_")) for key in MCMC_MODELS}          # "B_1" → ("B", "1")
COMBOS = [(mat, ver, n, key) for mat in MATS for ver in DATA_VERSIONS for n in SETS5B[ver] if (mat, ver, n) in TRAIN_DF for key in MCMC_MODELS]
GD = {c: make_grouped(TRAIN_DF[c[:3]], *MODEL_SPEC[c[3]]) for c in COMBOS}       # 条件ごとにまとめた学習データ
for (mat, ver, n), tr in TRAIN_DF.items():
    print(f"{MATS[mat]}・{ver} 版・{n}：学習 {len(tr)} 行 {tr['cond'].nunique()} 条件、テスト " + ("、".join(f"{k} {len(t)} 行" for k, t in TESTS[(mat, ver, n)].items()) or "なし"))
print(f"モデル {MCMC_MODELS}、組の数 {len(COMBOS)}（材料 × 版 × 学習データ × モデル）")

BEST5 = pd.read_csv(os.path.join(STEP5_DIR, "best_E.csv"), encoding="utf-8-sig", dtype={"hv": str}) if os.path.exists(os.path.join(STEP5_DIR, "best_E.csv")) else None
print("手順5 の最良の E：" + ("読み込んだ" if BEST5 is not None else "結果がないので、E_START を使う"))


def e0_of(mat, ver, n, struct, hv):
    """手順5 のプロファイル尤度の最良の E（そのモデルの学習データのもの）。なければ E_START"""
    if struct == "C":
        return []
    if BEST5 is not None:
        sn = "S_V" if (n == "ALL" and ver == "5data") else n
        r = BEST5[(BEST5["material"] == MATS[mat]) & (BEST5["version"] == ver) & (BEST5["set"] == sn) & (BEST5["struct"] == struct) & (BEST5["hv"] == hv)]
        if len(r):
            return [float(r["Eh_best"].iloc[0]), float(r["Ev_best"].iloc[0])] if struct == "B" else [float(r["Ev_best"].iloc[0])]
    return list(E_START[struct])''')
code(r'''# セル3：部品の確認（条件ごとにまとめた尤度・予測が、scikit-learn のショットごとの GPR と一致するか）
mat1, ver1, n1 = min(((m, v, n) for (m, v, n) in TRAIN_DF if TESTS[(m, v, n)]), key=lambda k: len(TRAIN_DF[k]))
tr1, te1 = TRAIN_DF[(mat1, ver1, n1)], list(TESTS[(mat1, ver1, n1)].values())[0]
for key in MCMC_MODELS:
    st, hv = MODEL_SPEC[key]
    e0 = e0_of(mat1, ver1, n1, st, hv)
    v = {"struct": st, "hv": hv, "pcol": "ln_P32_cm", "Eh": e0[0] if st == "B" else 0.0, "Ev": e0[-1] if st != "C" else 0.0}
    f = fit5(v, tr1, 1)                                                  # scikit-learn のショットごとの GPR（やり直し 1 回）
    th = f["gp"].kernel_.theta
    gd1 = make_grouped(tr1, st, hv)
    l_sk, l_gr = float(f["gp"].log_marginal_likelihood_value_), grouped_lml(gd1, v, kernel5(st, hv), th)
    phi = np.r_[th, [v[nm] for nm in E_NAMES[st]]]
    mu_sk, sd_sk, _, _ = predict5(f, te1)
    mu_gr, sd_gr, _ = grouped_predict(gd1, st, hv, phi, te1)
    diff = max(abs(l_sk - l_gr), float(np.abs(mu_sk - mu_gr).max()), float(np.abs(sd_sk - sd_gr).max()))
    judge(diff < 1e-6, f"{key}：条件ごとにまとめた尤度・予測が、ショットごとの GPR と一致（{MATS[mat1]}・{ver1} 版・{n1}：対数尤度 {l_gr:.4f} と {l_sk:.4f}、最大の差 {diff:.1e}）")
c0 = COMBOS[0]
st0, hv0 = MODEL_SPEC[c0[3]]
k0 = kernel5(st0, hv0)
v0 = model_v(st0, hv0, np.r_[k0.theta, e0_of(c0[0], c0[1], c0[2], st0, hv0)])
t0 = time.time()
for _ in range(100):
    grouped_lml(GD[c0], v0, k0, k0.theta)
print(f"尤度 1 回あたり {(time.time() - t0) * 10:.2f} ミリ秒")''')

# ---------------------------------------------------------------- 3. 最尤推定
md(r'''## 3. 最尤推定（比べる基準）

同じ尤度を、scipy の L-BFGS-B で最大化する（8 通りの出発点）。2 通りの最尤推定を作る。
- **MLE（E 固定）**：E を、手順5 のプロファイル尤度の最良値に固定して、カーネルだけを推定（手順5 の評価と同じ。C_1 は E がないので、同時と同じ）。
- **MLE（同時）**：E も動かして、カーネルと同時に推定。MCMC の初期値にもなる。

結果は `mle.csv` に保存する。''')
code(r'''# セル4：最尤推定（並列）
def mle_job(gd, struct, hv, e_prior, e0, seed):
    warnings.simplefilter("ignore")
    phi_f, l_f = fit_mle5(gd, struct, hv, e_prior, e0, True, 8, seed)
    phi_j, l_j = fit_mle5(gd, struct, hv, e_prior, e0, False, 8, seed)
    return phi_f, l_f, phi_j, l_j


t_all = time.time()
res = Parallel(n_jobs=N_JOBS)(delayed(mle_job)(GD[c], *MODEL_SPEC[c[3]], E_PRIOR, e0_of(c[0], c[1], c[2], *MODEL_SPEC[c[3]]), SEED + i) for i, c in enumerate(COMBOS))
MLE_FIX = {c: r[0] for c, r in zip(COMBOS, res)}
MLE_JOINT = {c: r[2] for c, r in zip(COMBOS, res)}
rows = []
for c, r in zip(COMBOS, res):
    names = theta_names5(*MODEL_SPEC[c[3]])
    for meth, phi, ll in [("E 固定", r[0], r[1]), ("同時", r[2], r[3])]:
        rows.append({"material": MATS[c[0]], "version": c[1], "split": c[2], "model": c[3], "method": meth, "logLik": ll, **dict(zip(names, phi))})
mle_df = pd.DataFrame(rows)
mle_df.to_csv(os.path.join(OUT_DIR, "mle.csv"), index=False, encoding="utf-8-sig")
print(f"最尤推定：{len(COMBOS)} 組、{(time.time() - t_all) / 60:.1f} 分")
print(mle_df[mle_df["model"].str.startswith("B")][["material", "version", "split", "model", "method", "logLik", "Eh", "Ev"]].round(2).to_string(index=False))''')

# ---------------------------------------------------------------- 4. MCMC
md(r'''## 4. MCMC によるサンプリング

- 4 本のチェーンの初期値は、MLE（同時）に、ノイズ以外のカーネルの対数パラメータへ N(0, 0.3²)、E へ N(0, 0.5²) [kJ/mol] のずれを加えたもの。
- 提案分布の最初の大きさは、対数パラメータ 0.05、E 0.5 kJ/mol（調整期間に、足跡から学び直す）。
- 本番のサンプルは `results/step5b_mcmc/posterior_<版>_<学習>_<材料>_<モデル>.csv` に、チェーンごとの採択率などは `chain_info.csv` に保存する。''')
code(r'''# セル5：サンプリング（保存済みのサンプルがあれば読み込む）
def post_path(c):
    mat, ver, n, key = c
    return os.path.join(OUT_DIR, f"posterior_{ver}_{n}_{mat}_{key}.csv")


def starts_seeds(i, c):
    """チェーンの初期値（MLE（同時）にゆらぎを足す）とシード"""
    st, hv = MODEL_SPEC[c[3]]
    names = theta_names5(st, hv)
    _, lo, hi = make_logpost(GD[c], st, hv, E_PRIOR)
    rng = np.random.default_rng(SEED + 100 * i)
    scale = np.array([0.0 if nm == "log_noise" else (0.5 if nm in E_NAMES[st] else 0.3) for nm in names])
    starts = [np.clip(MLE_JOINT[c] + rng.normal(0, 1, len(names)) * scale, lo + 1e-6, hi - 1e-6) for _ in range(CHAINS)]
    return starts, [SEED + 1000 * i + k for k in range(CHAINS)]


info_path = os.path.join(OUT_DIR, "chain_info.csv")
chain_info = pd.read_csv(info_path) if os.path.exists(info_path) else pd.DataFrame()
traces = {}
for c in COMBOS:
    if USE_SAVED_SAMPLES and os.path.exists(post_path(c)):
        saved = pd.read_csv(post_path(c))
        per_chain = saved.groupby("chain").size()
        if len(per_chain) == CHAINS and (per_chain >= N_SAMP).all():
            traces[c] = saved
            print(f"{MATS[c[0]]}・{c[1]}・{c[2]}・{c[3]}: 保存済みのサンプルを読み込み（1チェーン {int(per_chain.iloc[0])} 回）")
pending, n_samp = [c for c in COMBOS if c not in traces], N_SAMP
t_all = time.time()
while pending:
    tasks = []
    for c in pending:
        st, hv = MODEL_SPEC[c[3]]
        starts, seeds = starts_seeds(COMBOS.index(c), c)
        init_sd = np.array([0.5 if nm in E_NAMES[st] else 0.05 for nm in theta_names5(st, hv)])
        tasks += [(c, k, st, hv, starts[k], init_sd, seeds[k]) for k in range(CHAINS)]
    t0 = time.time()
    outs = Parallel(n_jobs=N_JOBS)(delayed(run_chain5)(GD[t[0]], t[2], t[3], E_PRIOR, t[4], t[5], N_WARM, n_samp, t[6]) for t in tasks)
    print(f"本番 {n_samp:,}回 × {CHAINS}チェーン × {len(pending)} 組：{(time.time() - t0) / 60:.1f} 分")
    still = []
    for c in pending:
        mat, ver, n, key = c
        res_c = [o for (t, o) in zip(tasks, outs) if t[0] == c]
        names = theta_names5(*MODEL_SPEC[key])
        trace = pd.concat([pd.DataFrame({"chain": k, "draw": np.arange(n_samp), **{nm: r["draws"][:, j] for j, nm in enumerate(names)},
                                         "lp": r["lp"], "accepted": r["accepted"].astype(int)}) for k, r in enumerate(res_c)], ignore_index=True)
        dg = diag_summary(trace, names)
        rhat, ess = float(dg["r_hat"].max()), float(dg["ess_bulk"].min())
        print(f"  {MATS[mat]}・{ver}・{n}・{key}: 最大 R-hat {rhat:.3f}、最小 ESS(bulk) {ess:.0f}、採択率 " + ", ".join(f"{r['acc_samp']:.2f}" for r in res_c))
        if (rhat > 1.01 or ess < 400) and n_samp < N_SAMP_MAX:
            still.append(c)                                           # 収束していない（または有効サンプル数が少ない）ので、本番を増やしてやり直す
            continue
        trace.to_csv(post_path(c), index=False, float_format="%.7g")
        traces[c] = pd.read_csv(post_path(c))                         # 保存した値（7桁）で以降を計算する
        rows_ = pd.DataFrame([{"material": MATS[mat], "version": ver, "split": n, "model": key, "chain": k, "acc_warm": r["acc_warm"], "acc_samp": r["acc_samp"],
                               "lambda": r["lam"], "chain_seconds": r["seconds"], "wall_seconds": max(x["seconds"] for x in res_c)} for k, r in enumerate(res_c)])
        if len(chain_info):
            chain_info = chain_info[~((chain_info["material"] == MATS[mat]) & (chain_info["version"] == ver) & (chain_info["split"] == n) & (chain_info["model"] == key))]
        chain_info = pd.concat([chain_info, rows_], ignore_index=True)
        chain_info.to_csv(info_path, index=False)
    pending, n_samp = still, min(int(n_samp * 2.5), N_SAMP_MAX)
RUN_INFO = {c: {"chains": int(t["chain"].nunique()), "n_warm": N_WARM, "n_samp_per_chain": int(len(t) // t["chain"].nunique()), "draws_total": int(len(t))} for c, t in traces.items()}
print(f"\n合計 {(time.time() - t_all) / 60:.1f} 分")
print(pd.DataFrame([{"material": MATS[c[0]], "version": c[1], "split": c[2], "model": c[3], **i} for c, i in RUN_INFO.items()]).to_string(index=False))''')

# ---------------------------------------------------------------- 5. 収束診断と事後分布
md(r'''## 5. 収束診断と事後分布

- R-hat（1.01 以下が目安）と有効サンプル数 ESS（bulk・tail。400 以上が目安）は ArviZ で計算する。
- 事後分布の要約は、振幅（√振幅²）・長さスケール・σ0・ノイズ分散の元の単位と、E［kJ/mol］で示す（長さスケールは標準化後の入力に対する値）。
- 判定：①全組で収束（R-hat ≤ 1.01、ESS ≥ 400）。②事後分布の 95% 区間が、事前分布の範囲の端（下限・上限の 5% 以内）に届いたパラメータ（届いたら、データだけでは決まらない）。③E の事後分布が、E の事前分布の範囲の端に届いていない。④E の事後分布の 95% 区間に、手順5 のプロファイル尤度の最良値が入る。結果は `posterior_summary.csv`・`sampler_stats.csv`。''')
code(r'''# セル6：収束診断と事後分布の要約
def to_orig(name, x):
    """対数パラメータ → 元の単位（振幅² は振幅にする）。E はそのまま。(ラベル, 値)"""
    if name in ("Eh", "Ev"):
        return name, np.asarray(x, float)
    x = np.exp(np.asarray(x, float))
    if name.startswith("log_c_lin"):
        return name.replace("log_c_lin", "lin_amp"), np.sqrt(x)
    if name.startswith("log_c"):
        return name.replace("log_c", "amp"), np.sqrt(x)
    return name.replace("log_", ""), x


rows = []
for c, trace in traces.items():
    mat, ver, n, key = c
    st, hv = MODEL_SPEC[key]
    names = theta_names5(st, hv)
    dg = diag_summary(trace, names)
    _, lo, hi = make_logpost(GD[c], st, hv, E_PRIOR)
    e0 = e0_of(mat, ver, n, st, hv)
    for j, nm in enumerate(names):
        lab, v = to_orig(nm, trace[nm].to_numpy())
        _, lo_b = to_orig(nm, lo[j])
        _, hi_b = to_orig(nm, hi[j])
        _, m_f = to_orig(nm, MLE_FIX[c][j])
        _, m_j = to_orig(nm, MLE_JOINT[c][j])
        q = np.percentile(v, [2.5, 50, 97.5])
        span = float(hi_b - lo_b) if nm in E_NAMES[st] else None
        at_bound = bool(q[0] < lo_b + 0.05 * span or q[2] > hi_b - 0.05 * span) if span else bool(q[0] < lo_b * 1.05 or q[2] > hi_b * 0.95)
        prof = e0[E_NAMES[st].index(nm)] if nm in E_NAMES[st] and e0 else np.nan
        rows.append({"material": MATS[mat], "version": ver, "split": n, "model": key, **RUN_INFO[c], "param": lab, "MLE_E_fixed": float(m_f), "MLE_joint": float(m_j), "profile_best_E": prof,
                     "post_mean": float(v.mean()), "post_sd": float(v.std(ddof=1)), "q2.5": q[0], "median": q[1], "q97.5": q[2], "r_hat": float(dg.loc[nm, "r_hat"]),
                     "ess_bulk": float(dg.loc[nm, "ess_bulk"]), "ess_tail": float(dg.loc[nm, "ess_tail"]), "post_at_bound": at_bound})
post_df = pd.DataFrame(rows)
post_df.to_csv(os.path.join(OUT_DIR, "posterior_summary.csv"), index=False, encoding="utf-8-sig")
show_ = post_df[post_df["split"].isin(["S_V", "ALL"])]
print(show_[["material", "version", "split", "model", "param", "MLE_joint", "median", "q2.5", "q97.5", "r_hat", "ess_bulk", "post_at_bound"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
stat_rows = []
for c in traces:
    ci = chain_info[(chain_info["material"] == MATS[c[0]]) & (chain_info["version"] == c[1]) & (chain_info["split"] == c[2]) & (chain_info["model"] == c[3])]
    p = post_df[(post_df["material"] == MATS[c[0]]) & (post_df["version"] == c[1]) & (post_df["split"] == c[2]) & (post_df["model"] == c[3])]
    stat_rows.append({"material": MATS[c[0]], "version": c[1], "split": c[2], "model": c[3], **RUN_INFO[c], "acc_samp_min": ci["acc_samp"].min(), "acc_samp_max": ci["acc_samp"].max(),
                      "sampling_min": ci["wall_seconds"].iloc[0] / 60, "max_r_hat": p["r_hat"].max(), "min_ess_bulk": p["ess_bulk"].min(), "min_ess_tail": p["ess_tail"].min(),
                      "n_post_at_bound": int(p["post_at_bound"].sum()), "params_at_bound": ", ".join(p[p["post_at_bound"]]["param"])})
stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv(os.path.join(OUT_DIR, "sampler_stats.csv"), index=False, encoding="utf-8-sig")
print("\n【サンプラーの統計】")
print(stat_df.round(3).to_string(index=False))
n_bad = int(((stat_df["max_r_hat"] > 1.01) | (stat_df["min_ess_bulk"] < 400)).sum())
judge(n_bad == 0, f"全 {len(stat_df)} 組で R-hat ≤ 1.01、ESS(bulk) ≥ 400（満たさないのは {n_bad} 組。満たさないなら、N_SAMP_MAX を増やして再実行）")
for key in MCMC_MODELS:
    s = stat_df[stat_df["model"] == key]
    judge(int((s["n_post_at_bound"] > 0).sum()) == 0, f"{key}：事後分布の 95% 区間が、事前分布の範囲の端に届いた組は {int((s['n_post_at_bound'] > 0).sum())} / {len(s)}（届いたパラメータ: {sorted(set(', '.join(s['params_at_bound']).split(', ')) - {''})}。届いたら、そのパラメータは、データだけでは決まらない）")
pe = post_df[post_df["param"].isin(["Eh", "Ev"])]
if len(pe):
    judge(not bool(pe["post_at_bound"].any()), f"E の事後分布が、E の事前分布の範囲の端（5% 以内）に届いていない（届いた: {int(pe['post_at_bound'].sum())} / {len(pe)}）")
    inside = (pe["profile_best_E"] >= pe["q2.5"]) & (pe["profile_best_E"] <= pe["q97.5"])
    judge(bool(inside.all()), f"E の事後分布の 95% 区間に、手順5 のプロファイル尤度の最良値が入る（入った: {int(inside.sum())} / {len(pe)}。外れたもの: {sorted(set(pe[~inside]['model'] + ' ' + pe[~inside]['split'] + ' ' + pe[~inside]['param']))}）")''')

# ---------------------------------------------------------------- 6. 図
md(r'''## 6. 事後分布の図

- 図1：B_1 の E_h と E_v の事後分布（散布図。手順5 の最良値と MLE を重ねる）。斜めの帯になるか。
- 図2：差 E_v − E_h の事後分布（ヒストグラム）。
- 図3：ハイパーパラメータの事後分布（ヒストグラム。log10。赤の破線が MLE（同時）、灰色の点線が事前分布の範囲の端）。版ごとの主な学習データ（5data 版は S_V、Alldata 版は ALL）のもの。
- 図4：トレース（チェーンごとの推移）。同じく主な学習データのもの。

すべて `save_fig` で保存する（通常の PNG と、`highres/` の 600 dpi）。''')
code(r'''# セル7：図の共通の設定、図1・図2（E の事後分布）
CHAIN_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
T_COL = {190: "tab:blue", 210: "tab:green", 230: "tab:red"}
H_LEVELS = [0.5, 1.0, 1.5, 2.0]
H_COL = {0.5: "#d62728", 1.0: "#1f77b4", 1.5: "#f0b400", 2.0: "#2ca02c"}


def main_set(ver):
    n = "S_V" if ver == "5data" else "ALL"
    return n if n in SETS5B[ver] else SETS5B[ver][-1]


def thinned(c, n=N_PRED_DRAWS):
    names = theta_names5(*MODEL_SPEC[c[3]])
    return thin_draws(traces[c], names, n)


bkeys = [k for k in MCMC_MODELS if MODEL_SPEC[k][0] == "B"]
if bkeys:
    key = bkeys[0]
    for ver in DATA_VERSIONS:
        sets_ = [n for n in SETS5B[ver] if (list(MATS)[0], ver, n, key) in traces]
        fig, axes = plt.subplots(len(MATS), len(sets_), figsize=(5.4 * len(sets_), 4.8 * len(MATS)), squeeze=False)
        fig2, axes2 = plt.subplots(len(MATS), len(sets_), figsize=(5.4 * len(sets_), 3.6 * len(MATS)), squeeze=False)
        for i, mat in enumerate(MATS):
            for j, n in enumerate(sets_):
                c = (mat, ver, n, key)
                t = traces[c]
                ax, ax2 = axes[i, j], axes2[i, j]
                ax.scatter(t["Eh"][::10], t["Ev"][::10], s=3, alpha=0.25, color="tab:purple", label="事後サンプル（間引き）")
                e0 = e0_of(mat, ver, n, "B", MODEL_SPEC[key][1])
                ax.plot([e0[0]], [e0[1]], "r*", ms=14, mec="k", label=f"手順5 の最良値（{e0[0]:.0f}, {e0[1]:.0f}）")
                mj = MLE_JOINT[c]
                ax.plot([mj[-2]], [mj[-1]], "kx", ms=9, mew=2, label="MLE（同時）")
                lim = [min(t["Eh"].min(), t["Ev"].min()) - 1, max(t["Eh"].max(), t["Ev"].max()) + 1]
                ax.plot(lim, lim, "k--", lw=0.8, label="E_h = E_v")
                ax.set_xlabel("横方向 E_h [kJ/mol]")
                ax.set_ylabel("縦方向 E_v [kJ/mol]")
                ax.set_title(f"{MATS[mat]}・{ver} 版・{n} の学習", fontsize=10)
                ax.grid(alpha=0.3)
                if i == 0 and j == 0:
                    ax.legend(fontsize=7, loc="upper left")
                d = (t["Ev"] - t["Eh"]).to_numpy()
                ax2.hist(d, bins=40, color="tab:purple", alpha=0.7)
                ax2.axvline(e0[1] - e0[0], color="r", lw=2, label=f"手順5 の最良値（{e0[1] - e0[0]:.0f}）")
                q = np.percentile(d, [2.5, 97.5])
                ax2.axvspan(q[0], q[1], color="gray", alpha=0.15, label=f"95% 区間（{q[0]:.1f}〜{q[1]:.1f}）")
                ax2.set_xlabel("E_v − E_h [kJ/mol]")
                ax2.set_title(f"{MATS[mat]}・{ver} 版・{n}", fontsize=10)
                ax2.legend(fontsize=7)
        fig.suptitle(f"{key}：E_h と E_v の事後分布（{ver} 版）。斜めの帯 ＝ 差だけがよく決まる", fontsize=11)
        fig.tight_layout()
        save_fig(fig, OUT_DIR, f"fig1_E_posterior_{ver}")
        plt.show()
        fig2.suptitle(f"{key}：差 E_v − E_h の事後分布（{ver} 版）", fontsize=11)
        fig2.tight_layout()
        save_fig(fig2, OUT_DIR, f"fig2_E_diff_{ver}")
        plt.show()''')
code(r'''# セル8：図3 ハイパーパラメータの事後分布（版ごとの主な学習データ。材料×モデルごとに 1 枚）
INPUT_JA = {"u": "u（ln γ̇。B は横シフト後）", "H": "ln H", "T": "Tset", "w": "w（ln Fo）", "p": "ln P3_2"}


def param_ja(nm):
    if nm in ("Eh", "Ev"):
        return {"Eh": "横方向のずらし量 E_h", "Ev": "縦方向のずらし量 E_v"}[nm] + "（kJ/mol）"
    base = {"log_c": "振幅²", "log_noise": "ノイズ分散", "log_sigma0": "線形項の σ0", "log_c_lin": "線形項の振幅²"}
    if nm in base:
        return base[nm] + "（log10）"
    if nm.startswith("log_ls_"):
        rest = nm[len("log_ls_"):]
        return "長さスケール：" + INPUT_JA.get(rest.split("_")[-1], rest) + "（log10）"
    return nm


for ver in DATA_VERSIONS:
    for mat in MATS:
        for key in MCMC_MODELS:
            c = (mat, ver, main_set(ver), key)
            if c not in traces:
                continue
            st, hv = MODEL_SPEC[key]
            names = theta_names5(st, hv)
            _, lo, hi = make_logpost(GD[c], st, hv, E_PRIOR)
            ncol = 4
            nrow = (len(names) + ncol - 1) // ncol
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.9 * nrow), squeeze=False)
            for j, nm in enumerate(names):
                ax = axes[j // ncol, j % ncol]
                if nm in E_NAMES[st]:
                    v, mle, b0, b1 = traces[c][nm].to_numpy(), MLE_JOINT[c][j], lo[j], hi[j]
                else:
                    v, mle, b0, b1 = traces[c][nm].to_numpy() / np.log(10), MLE_JOINT[c][j] / np.log(10), lo[j] / np.log(10), hi[j] / np.log(10)
                ax.set_title(param_ja(nm), fontsize=9)
                ax.hist(v, bins=50, color="tab:blue", alpha=0.7, density=True)
                ax.axvline(mle, color="r", ls="--", lw=1.5, label="MLE（同時）")
                for b in (b0, b1):
                    if v.min() - 0.1 * (np.ptp(v) + 1e-9) <= b <= v.max() + 0.1 * (np.ptp(v) + 1e-9) or abs(b - np.median(v)) < 3 * (np.ptp(v) + 1e-9):
                        ax.axvline(b, color="gray", ls=":", lw=1.5)
                ax.tick_params(labelsize=7)
            for j in range(len(names), nrow * ncol):
                axes[j // ncol, j % ncol].axis("off")
            axes[0, 0].legend(fontsize=7)
            fig.suptitle(f"{MATS[mat]}・{ver} 版・{main_set(ver)}・{key}：ハイパーパラメータの事後分布。赤の破線：MLE（同時）、灰色の点線：事前分布の範囲の端（範囲内にあるとき）", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            save_fig(fig, OUT_DIR, f"fig3_posterior_{ver}_{mat}_{key}")
            plt.show()

# 図4 トレース（版ごとの主な学習データ）
for ver in DATA_VERSIONS:
    for mat in MATS:
        for key in MCMC_MODELS:
            c = (mat, ver, main_set(ver), key)
            if c not in traces:
                continue
            names = theta_names5(*MODEL_SPEC[key])
            fig, axes = plt.subplots(len(names), 1, figsize=(11, 1.5 * len(names)), sharex=True, squeeze=False)
            for j, nm in enumerate(names):
                ax = axes[j, 0]
                for k in range(CHAINS):
                    tk = traces[c][traces[c]["chain"] == k]
                    ax.plot(tk["draw"], tk[nm] if nm in E_NAMES[MODEL_SPEC[key][0]] else tk[nm] / np.log(10), lw=0.4, color=CHAIN_COLORS[k % 4], alpha=0.8)
                ax.set_ylabel(nm, fontsize=7)
                ax.tick_params(labelsize=7)
            axes[-1, 0].set_xlabel("本番の反復")
            fig.suptitle(f"{MATS[mat]}・{ver} 版・{main_set(ver)}・{key}：トレース（色がチェーン。E 以外は log10）", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            save_fig(fig, OUT_DIR, f"fig4_trace_{ver}_{mat}_{key}")
            plt.show()''')

# ---------------------------------------------------------------- 7. 予測の比較
md(r'''## 7. 予測の比較（最尤推定と MCMC）

学習データごとのテストを、3 通りで予測する。
- **MLE（E 固定）**：手順5 と同じ（E は最良値に固定。カーネルは最尤推定）。
- **MLE（同時）**：E もカーネルと同時に最尤推定。
- **MCMC**：事後サンプル `N_PRED_DRAWS` 個の予測分布を、等しい重みで混ぜた混合分布（ハイパーパラメータと E の不確かさを含む）。

指標は RMSE・R²・被覆率（95% 予測区間）・NLPD・予測標準偏差の平均（区間の広さの目安）。結果は `metrics.csv`。テストのない学習データ（ALL）は、評価しない。

判定：①MCMC の被覆率が 0.90〜0.99。②MCMC の RMSE が MLE（E 固定）の 1.05 倍以内。③MCMC の NLPD が MLE（E 固定）より 0.05 を超えて悪くない。④不確かさを含めた分、MCMC の区間（平均の予測標準偏差）が、MLE より広い（ただし 1.5 倍以内）。''')
code(r'''# セル9：予測の指標
mrows = []
for c in COMBOS:
    mat, ver, n, key = c
    if not TESTS[(mat, ver, n)]:
        continue
    st, hv = MODEL_SPEC[key]
    th_draws = thinned(c)
    for tn, te in TESTS[(mat, ver, n)].items():
        y = te["ln_eta"].to_numpy()
        preds = {}
        for meth, phi in [("MLE（E 固定）", MLE_FIX[c]), ("MLE（同時）", MLE_JOINT[c])]:
            mu, sd, _ = grouped_predict(GD[c], st, hv, phi, te)
            preds[meth] = (mu[None], sd[None])
        mus, sds, _, _, _ = predict_mixture(GD[c], st, hv, th_draws, te)
        preds["MCMC"] = (mus, sds)
        subs = {"全体": np.ones(len(te), bool), **{g: (te["group"] == g).to_numpy() for g in te["group"].unique() if g != "全体"}}
        for meth, (m_, s_) in preds.items():
            for sub, mask in subs.items():
                if mask.sum() < 2:
                    continue
                ev = evaluate_mixture(y[mask], m_[:, mask], s_[:, mask])
                mrows.append({"material": MATS[mat], "version": ver, "split": tn, "train_set": n, "subset": sub, "model": key, "method": meth, "n_test": int(mask.sum()), **ev})
met_df = pd.DataFrame(mrows)
met_df.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False, encoding="utf-8-sig")
METHODS = ["MLE（E 固定）", "MLE（同時）", "MCMC"]
PANELS = [("S_V", "低速側"), ("S_V", "高速側"), ("S_T", "全体"), ("S_T3", "全体")]
for mat in MATS:
    for ver in DATA_VERSIONS:
        t = met_df[(met_df["material"] == MATS[mat]) & (met_df["version"] == ver)]
        if t.empty:
            continue
        print(f"\n【{MATS[mat]}・{ver} 版】RMSE／被覆率／NLPD／平均の予測標準偏差")
        t = t.assign(panel=t["split"] + " " + t["subset"])
        t = t.assign(v=t["RMSE"].map("{:.3f}".format) + "／" + t["coverage95"].map("{:.2f}".format) + "／" + t["NLPD"].map("{:.2f}".format) + "／" + t["mean_sd"].map("{:.3f}".format))
        print(t.pivot_table(index=["model", "method"], columns="panel", values="v", aggfunc="first").reindex(index=[(k, m) for k in MCMC_MODELS for m in METHODS]).to_string())


def mm(mat, ver, split, sub, key, meth, col):
    r = met_df[(met_df["material"] == MATS[mat]) & (met_df["version"] == ver) & (met_df["split"] == split) & (met_df["subset"] == sub) & (met_df["model"] == key) & (met_df["method"] == meth)]
    return float(r[col].iloc[0]) if len(r) else np.nan


for mat in MATS:
    for ver in DATA_VERSIONS:
        for key in MCMC_MODELS:
            t = met_df[(met_df["material"] == MATS[mat]) & (met_df["version"] == ver) & (met_df["model"] == key) & (met_df["subset"] == "全体")]
            if t.empty:
                continue
            a = t[t["method"] == "MCMC"].set_index("split")
            b = t[t["method"] == "MLE（E 固定）"].set_index("split")
            tag = f"{MATS[mat]}・{ver} 版・{key}"
            judge(bool(a["coverage95"].between(0.90, 0.99).all()), f"{tag}：MCMC の被覆率が 0.90〜0.99（{', '.join(f'{s} {v:.2f}' for s, v in a['coverage95'].items())}）")
            judge(bool((a["RMSE"] <= 1.05 * b["RMSE"]).all()), f"{tag}：MCMC の RMSE が、MLE（E 固定）の 1.05 倍以内（{', '.join(f'{s} {a.loc[s, 'RMSE']:.3f} 対 {b.loc[s, 'RMSE']:.3f}' for s in a.index)}）")
            judge(bool((a["NLPD"] <= b["NLPD"] + 0.05).all()), f"{tag}：MCMC の NLPD が、MLE（E 固定）より 0.05 を超えて悪くない（{', '.join(f'{s} {a.loc[s, 'NLPD']:.2f} 対 {b.loc[s, 'NLPD']:.2f}' for s in a.index)}）")
            ratio = a["mean_sd"] / b["mean_sd"]
            judge(bool(((ratio >= 1.0 - 1e-9) & (ratio <= 1.5)).all()), f"{tag}：MCMC の区間（平均の予測標準偏差）が MLE より広い（1.0〜1.5 倍。{', '.join(f'{s} {r:.2f} 倍' for s, r in ratio.items())}）")''')
code(r'''# セル10：図5 予測指標の比較（MLE と MCMC）
if len(met_df):
    colors = {"MLE（E 固定）": "tab:blue", "MLE（同時）": "tab:green", "MCMC": "tab:red"}
    for ver in DATA_VERSIONS:
        t = met_df[met_df["version"] == ver]
        panels = [(s, g) for s, g in PANELS if ((t["split"] == s) & (t["subset"] == g)).any()]
        if not panels:
            continue
        fig, axes = plt.subplots(3, len(MATS), figsize=(8.5 * len(MATS), 12), squeeze=False)
        for j, mat in enumerate(MATS):
            for i, (metric, ylab) in enumerate([("RMSE", r"RMSE（$\ln\eta$）"), ("coverage95", "95% 区間の被覆率"), ("NLPD", "NLPD（小さいほど良い）")]):
                ax = axes[i, j]
                groups = [(key, p) for key in MCMC_MODELS for p in panels]
                w = 0.8 / len(METHODS)
                for k, meth in enumerate(METHODS):
                    ax.bar(np.arange(len(groups)) + (k - 1) * w, [mm(mat, ver, p[0], p[1], key, meth, metric) for key, p in groups], w, color=colors[meth], label=meth)
                ax.set_xticks(np.arange(len(groups)))
                ax.set_xticklabels([f"{key}\n{p[0]} {p[1]}" for key, p in groups], fontsize=7)
                ax.set_title(f"{MATS[mat]}・{ver} 版：{ylab}", fontsize=10)
                ax.grid(alpha=0.3, axis="y")
                if metric == "coverage95":
                    ax.axhline(0.95, color="k", lw=0.8, ls="--")
                    ax.set_ylim(0.6, 1.02)
                if i == 0 and j == 0:
                    ax.legend(fontsize=8)
        fig.tight_layout()
        save_fig(fig, OUT_DIR, f"fig5_metrics_{ver}")
        plt.show()''')

# ---------------------------------------------------------------- 8. 設定温度の効果
md(r'''## 8. 設定温度の効果の不確かさ

最終モデルでなく、各学習データのモデルで、5data＋3data の 210 ℃ のショットについて、ほかの入力はそのまま、Tset だけを 190 ℃・230 ℃ にして予測した粘度の比（190 ℃ ÷ 230 ℃）を、厚みごとに中央値でまとめる。MCMC は、事後サンプルごとに計算して、95% 区間を出す。実測の比（同じ γ̇、厚みごと）と比べる。結果は `tset_effect.csv`・`observed_ratio.csv`。

判定：B 系のモデルで、実測の比が、MCMC の 95% 区間に入る厚みが 3 つ以上（4 つのうち）。''')
code(r'''# セル11：設定温度の効果
def observed_ratio(mat):
    d = pool_of(mat, MAIN_DATA)
    d = d[~d["is_sens"]]
    cm = d.groupby(["Tset", "H", "v"]).agg(lg=("ln_gd", "mean"), le=("ln_eta", "mean")).reset_index()
    rows = []
    for h in H_LEVELS:
        a, c_ = cm[(cm["H"] == h) & (cm["Tset"] == 190)], cm[(cm["H"] == h) & (cm["Tset"] == 230)].sort_values("lg")
        ins = ((a["lg"] >= c_["lg"].min()) & (a["lg"] <= c_["lg"].max())).to_numpy()
        r = np.exp(a["le"].to_numpy()[ins] - np.interp(a["lg"].to_numpy()[ins], c_["lg"], c_["le"]))
        rows.append({"material": MATS[mat], "H": h, "observed": float(np.median(r)), "n": int(ins.sum())})
    return pd.DataFrame(rows)


obs_df = pd.concat([observed_ratio(m) for m in MATS], ignore_index=True)
obs_df.to_csv(os.path.join(OUT_DIR, "observed_ratio.csv"), index=False, encoding="utf-8-sig")
trows = []
for c in COMBOS:
    mat, ver, n, key = c
    st, hv = MODEL_SPEC[key]
    base = pool_of(mat, MAIN_DATA)
    base = base[(base["Tset"] == 210) & ~base["is_sens"]]
    lo_, hi_ = base.copy(), base.copy()
    lo_["Tset"], hi_["Tset"] = 190.0, 230.0

    def ratio_by_h(phis):
        m_lo = predict_mixture(GD[c], st, hv, phis, lo_)[0]
        m_hi = predict_mixture(GD[c], st, hv, phis, hi_)[0]
        r = np.exp(m_lo - m_hi)                                                    # (サンプル数, 行数)
        return np.array([np.median(r[:, (base["H"] == h).to_numpy()], axis=1) for h in H_LEVELS])    # (厚み, サンプル数)

    for meth, phis in [("MLE（E 固定）", MLE_FIX[c][None]), ("MLE（同時）", MLE_JOINT[c][None]), ("MCMC", thinned(c))]:
        rb = ratio_by_h(phis)
        for i, h in enumerate(H_LEVELS):
            q = np.percentile(rb[i], [2.5, 50, 97.5])
            trows.append({"material": MATS[mat], "version": ver, "split": n, "model": key, "method": meth, "H": h, "ratio_median": q[1], "ratio_q2.5": q[0], "ratio_q97.5": q[2]})
eff_df = pd.DataFrame(trows)
eff_df.to_csv(os.path.join(OUT_DIR, "tset_effect.csv"), index=False, encoding="utf-8-sig")
print("実測の 190/230 ℃ の粘度比（同じ γ̇、厚みごと）")
print(obs_df.pivot(index="material", columns="H", values="observed").round(3).to_string())
for ver in DATA_VERSIONS:
    for mat in MATS:
        n = main_set(ver)
        t = eff_df[(eff_df["material"] == MATS[mat]) & (eff_df["version"] == ver) & (eff_df["split"] == n)]
        if t.empty:
            continue
        print(f"\n【{MATS[mat]}・{ver} 版・{n} の学習】中央値（95% 区間）")
        t = t.assign(v=t["ratio_median"].map("{:.3f}".format) + "（" + t["ratio_q2.5"].map("{:.3f}".format) + "〜" + t["ratio_q97.5"].map("{:.3f}".format) + "）")
        print(t.pivot_table(index=["model", "method"], columns="H", values="v", aggfunc="first").to_string())
        for key in [k for k in MCMC_MODELS if MODEL_SPEC[k][0] == "B"]:
            tt = t[(t["model"] == key) & (t["method"] == "MCMC")].sort_values("H")
            o = obs_df[obs_df["material"] == MATS[mat]].sort_values("H")["observed"].to_numpy()
            inside = (o >= tt["ratio_q2.5"].to_numpy()) & (o <= tt["ratio_q97.5"].to_numpy())
            judge(int(inside.sum()) >= 3, f"{MATS[mat]}・{ver} 版・{key}：実測の 190/230 ℃ の粘度比が、MCMC の 95% 区間に入る厚みが {int(inside.sum())} / 4（3 以上なら OK）")''')
code(r'''# セル12：図6 設定温度の効果（区間と実測）
for ver in DATA_VERSIONS:
    fig, axes = plt.subplots(1, len(MATS), figsize=(8.5 * len(MATS), 5.4), sharey=True, squeeze=False)
    for ax, mat in zip(axes[0], MATS):
        o = obs_df[obs_df["material"] == MATS[mat]].sort_values("H")
        ax.plot(o["H"], o["observed"], "k-o", lw=2.8, ms=9, label="実測（同じ γ̇）", zorder=10)
        n = main_set(ver)
        for k, key in enumerate(MCMC_MODELS):
            t = eff_df[(eff_df["material"] == MATS[mat]) & (eff_df["version"] == ver) & (eff_df["split"] == n) & (eff_df["model"] == key)]
            if t.empty:
                continue
            col = ["tab:purple", "tab:gray", "tab:blue", "tab:orange"][k % 4]
            m = t[t["method"] == "MCMC"].sort_values("H")
            ax.fill_between(m["H"], m["ratio_q2.5"], m["ratio_q97.5"], color=col, alpha=0.25, label=f"{key}：MCMC の 95% 区間")
            ax.plot(m["H"], m["ratio_median"], "-", color=col, lw=1.6, label=f"{key}：MCMC の中央値")
            ml = t[t["method"] == "MLE（E 固定）"].sort_values("H")
            ax.plot(ml["H"], ml["ratio_median"], "--", color=col, lw=1.2, label=f"{key}：MLE（E 固定）")
        ax.axhline(1.0, color="gray", lw=0.8)
        ax.set_xlabel("厚み [mm]")
        ax.set_title(f"{MATS[mat]}・{ver} 版（{main_set(ver)} の学習）：190 ℃ と 230 ℃ の粘度の比", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("粘度の比（190 ℃ ÷ 230 ℃）")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, OUT_DIR, f"fig6_tset_effect_{ver}")
    plt.show()''')
code(r'''# セル13：図7 設定温度ごとの予測曲線（MCMC の混合分布。行が厚み、列がモデル。版ごとの主な学習データ）
for ver in DATA_VERSIONS:
    for mat in MATS:
        n = main_set(ver)
        if (mat, ver, n, MCMC_MODELS[0]) not in traces:
            continue
        pool = pool_of(mat, MAIN_DATA) if ver == "5data" else D[(mat, "Alldata", MAIN_DATA)]
        tconds = set(TRAIN_DF[(mat, ver, n)]["cond"])
        fig, axes = plt.subplots(4, len(MCMC_MODELS), figsize=(5.2 * len(MCMC_MODELS), 14), sharex="row", sharey="row", squeeze=False)
        for j, key in enumerate(MCMC_MODELS):
            c = (mat, ver, n, key)
            st, hv = MODEL_SPEC[key]
            phis = thinned(c, 60)
            for i, h in enumerate(H_LEVELS):
                ax = axes[i, j]
                for t, col in T_COL.items():
                    sub = pool[(pool["H"] == h) & (pool["Tset"] == t)]
                    if sub.empty:
                        continue
                    cm = sub.groupby("cond").agg(lg=("ln_gd", "mean"), le=("ln_eta", "mean"), sens=("is_sens", "max")).reset_index()
                    lg = np.linspace(sub["ln_gd"].min(), sub["ln_gd"].max(), 100)
                    _, _, mu, sd, _ = predict_mixture(GD[c], st, hv, phis, hr_inputs(sub, lg, h, t))
                    ax.plot(np.exp(lg), np.exp(mu), color=col, lw=1.4)
                    ax.fill_between(np.exp(lg), np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd), color=col, alpha=0.15)
                    is_tr = cm["cond"].isin(tconds).to_numpy()
                    ax.scatter(np.exp(cm["lg"][is_tr]), np.exp(cm["le"][is_tr]), s=14, color=col, alpha=0.8, zorder=3)
                    ax.scatter(np.exp(cm["lg"][~is_tr]), np.exp(cm["le"][~is_tr]), s=26, facecolors="none", edgecolors=col, marker="s", linewidths=1.2, zorder=3)
                    se = cm["sens"].to_numpy().astype(bool)
                    ax.scatter(np.exp(cm["lg"][se]), np.exp(cm["le"][se]), s=45, color="k", marker="x", zorder=4)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x_, _: f"{x_:g}"))
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.grid(alpha=0.3, which="both")
                if j == 0:
                    ax.set_ylabel(f"厚み {h} mm\n" + r"$\eta$ [Pa·s]")
                if i == 0:
                    ax.set_title(key, fontsize=10)
                if i == 3:
                    ax.set_xlabel(r"$\dot{\gamma}$ [1/s]")
        handles = [plt.Line2D([], [], color=c_, label=f"{t} ℃") for t, c_ in T_COL.items()]
        handles += [plt.Line2D([], [], color="gray", marker="o", ls="", label="学習の条件平均"), plt.Line2D([], [], color="gray", marker="s", mfc="none", ls="", label="学習に使わない条件")]
        axes[0, 0].legend(handles=handles, fontsize=7, loc="upper right")
        fig.suptitle(f"{MATS[mat]}・{ver} 版（{n} の学習）：設定温度ごとの予測曲線（線：MCMC の混合分布の平均、帯：95% 予測区間）", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        save_fig(fig, OUT_DIR, f"fig7_curves_{ver}_{mat}")
        plt.show()''')

# ---------------------------------------------------------------- 9. 判定のまとめ・高解像度
md(r'''## 7b. 保存したファイル''')
code(r'''# セル14：保存したファイルの一覧
print("保存したファイル:")
for f in sorted(os.listdir(OUT_DIR)):
    if os.path.isfile(os.path.join(OUT_DIR, f)) and (not f.startswith("posterior_") or f == "posterior_summary.csv"):
        print("  ", os.path.normpath(os.path.join(OUT_DIR, f)))
print(f"  （事後サンプル posterior_<版>_<学習>_<材料>_<モデル>.csv：{len([f for f in os.listdir(OUT_DIR) if f.startswith('posterior_') and f != 'posterior_summary.csv'])} 個）")''')
md(HR_MD.replace("## 高解像度のグラフ", "## 8. 高解像度のグラフ").replace("**レイアウト**", "**モデル**：`HR_MODELS` に、MCMC で推定したモデル（`MCMC_MODELS` のもの。例：B_1、C_1）を並べる。曲線は、事後サンプルの予測分布の混合分布（平均と 95% 予測区間）。\n- **レイアウト**"))
code(hr_cell("# セル15：高解像度のグラフ（せん断速度 ― 粘度。MCMC の混合分布の予測。GPR6月ノートのセル9-2 にならう）",
             '''HR_VERSION = DATA_VERSIONS[-1]     # どの版か："5data"（5data で学習。3data の条件は白抜き）／"Alldata"（全条件で学習）
HR_SET = None                      # 学習データ。None なら、版ごとの主な学習データ（5data 版：S_V、Alldata 版：ALL）
HR_MODELS = list(MCMC_MODELS)      # 描くモデル（MCMC_MODELS のもの）
HR_DRAWS = 60                      # 曲線に使う事後サンプルの数（増やすと遅い）''',
             r'''hr_saved = []
for mat in HR_MATERIALS:
    n = HR_SET or main_set(HR_VERSION)
    pool = pool_of(mat, MAIN_DATA) if HR_VERSION == "5data" else D[(mat, "Alldata", MAIN_DATA)]
    tconds = set(TRAIN_DF[(mat, HR_VERSION, n)]["cond"]) if (mat, HR_VERSION, n) in TRAIN_DF else set()
    for key in HR_MODELS:
        c = (mat, HR_VERSION, n, key)
        if c not in traces:
            print(f"{MATS[mat]}・{HR_VERSION} 版・{n}・{key} は、サンプリングしていないので描かない（設定の DATA_VERSIONS・SETS5B・MCMC_MODELS に入れて、上から実行し直す）")
            continue
        t0 = time.time()
        st, hv = MODEL_SPEC[key]
        phis = thinned(c, HR_DRAWS)
        label = f"{key}：MCMC の混合分布（{HR_VERSION} 版・{n} の学習）"
        hr_saved += hr_flow(lambda Xg, c=c, st=st, hv=hv, phis=phis: tuple(predict_mixture(GD[c], st, hv, phis, Xg)[2:]), mat, MAIN_DATA, tconds, label, f"hr_{mat}_{HR_VERSION}_{n}_{key}_mcmc", HR_DIR, pool=pool, **HR_KW)
        print(f"{MATS[mat]} {key}：保存したファイル 累計 {len(hr_saved)} 個（{time.time() - t0:.0f} 秒）")
judge(len(hr_saved) > 0, f"高解像度のグラフを {len(hr_saved)} 個保存した（{os.path.normpath(HR_DIR)}）。形式 {HR_FORMATS}、{HR_DPI} dpi")'''))


# ---------------------------------------------------------------- 書き出し
def to_lines(s):
    lines = s.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


rnd = random.Random(20261016)
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
