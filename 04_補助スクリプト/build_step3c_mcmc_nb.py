# -*- coding: utf-8 -*-
"""手順3c（最終候補のモデルを MCMC で推定）のノートブック (未実行) を生成する。

使い方: python build_step3c_mcmc_nb.py <出力する .ipynb のパス> [--force]
既存ファイルがある場合は、--force を付けない限り上書きしない。
手順3b の結果（results/step3b/metrics.csv）から候補のモデルを自動で選ぶ（設定エリアで手動にもできる）。
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
md(r'''# 手順3c：最終候補のモデルを MCMC で推定（新データ）

CLAUDE.md §5・§7 の手順3c。手順3b の最尤推定（ハイパーパラメータを 1 点に決める）で選んだ**最終候補のモデル**だけを、MCMC（自作の適応型ランダムウォーク・メトロポリス法。手順2と同じ）でも推定し、最尤推定と比べる。

- **何を確かめるか**：ハイパーパラメータ（振幅・長さスケール・ノイズ・線形項）の不確かさを予測に含めたとき、予測区間の広さ・被覆率・NLPD が最尤推定とどれだけ違うか。ハイパーパラメータが下限・上限に張り付いていないか。設定温度の効果（190/230 ℃ の粘度比）の不確かさ。
- **候補のモデル**：材料ごとに、B0T（ショットごとの γ̇）・R3（条件平均の γ̇）・R2（設定値だけ）の 3 つ。Tff・P3_2 を足したモデルは、手順3b で要らないと分かったので入れない。設定エリアで変えられる（手順3b の NLPD が良いモデルを足す設定もある）。
- **分割**：S_V（5data → 3data）と S_T（5data の 190・230 ℃ → 210 ℃）。
- **GPR**：ARD Matern 5/2 ＋ 線形カーネル（DotProduct）＋ ノイズ（手順3b と同じ）。標準化と中心化は学習データだけで計算する。
- **事前分布**：カーネルの探索範囲（`nb_common.py` の `kernel_for`）と同じ範囲の、対数スケールの一様分布（振幅² 1e-3〜1e3、長さスケール 1e-2〜1e2、線形項の振幅² 1e-3〜1e3、σ₀ 1e-5〜1e3、ノイズ分散 NOISE_LB〜10）。事後分布は範囲内で exp(対数周辺尤度) に比例するので、最頻値は最尤推定と一致する。
- **予測**：事後サンプルを等間隔に N_PRED_DRAWS 個選び、各サンプルでの GPR の予測分布（ノイズ込みの正規分布）を等しい重みで混ぜた混合分布を使う。

### 自作の適応型ランダムウォーク・メトロポリス法

1. 今いる点 θ（ハイパーパラメータの対数）の近くに、正規分布 N(θ, λ·Σ) から候補を出す。
2. 候補の対数事後密度（範囲内なら対数周辺尤度、範囲外なら −∞）を計算し、確率 min(1, exp(候補 − 現在)) で移動する。移動しなければその場にとどまる。
3. **調整期間**：前半 75% では、それまでの足跡から提案のばらつき Σ（パラメータどうしの相関も含む）を学ぶ。調整期間の間ずっと、採択率が 0.234 に近づくよう大きさ λ を合わせる。調整期間のサンプルは捨てる。
4. **本番**：Σ と λ を固定した、ふつうのメトロポリス法として歩く。R-hat が 1.01 を超えるか、有効サンプル数（ESS）が 400 に届かないときは、本番の回数を増やして、サンプリングし直す（最大 N_SAMP_MAX 回）。
5. 4 本のチェーンを別々の初期値から並列に回し、R-hat と有効サンプル数（ESS）で収束を確かめる。''')

# ---------------------------------------------------------------- 1. 設定
md(usage_md("3c", "results/step3c_mcmc", "results/学習結果説明_手順3c.pdf", 40) + r'''
8. サンプリングには時間がかかる。`USE_SAVED_SAMPLES = True` のときは、保存済みの事後サンプル（`results/step3c_mcmc/posterior_*.csv`）があれば読み込み、サンプリングを省く。データを取り直したときは `False` にする（`rerun_all.py` は、実行の前に保存済みのサンプルを消すので、そのままでよい。ノートだけを実行し直すときは、自分で `False` にするか、`posterior_*.csv` を消す）。''')
md(r'''## 1. 設定''')
code(settings_cell('''CANDIDATES = None             # 候補のモデル。None なら手順3b の結果から自動で選ぶ。手動なら {"PP": ["B0T", "R3"], "MIX": ["B0T", "R3", "R2"]} のように書く
CAND_FIXED = ["B0T", "R3", "R2"]   # 自動のとき、必ず入れるモデル（B0T：ショットごとの γ̇、R3：条件平均の γ̇、R2：設定値だけ。どれも Tset あり。Tff・P3_2 を足したモデルは、要らないと分かったので入れない）
N_CAND_EXTRA = 0              # 自動のとき、必ず入れるもののほかに、S_V の NLPD が良い順に選ぶ数（0 なら選ばない）
CAND_POOL = ["B0FT", "B2T", "R3F", "R3P"]   # 追加で選ぶ対象
N_WARM, N_SAMP, CHAINS = 3000, 8000, 4     # 調整の反復数（捨てる）、本番の反復数、チェーン数
N_SAMP_MAX = 20000            # R-hat が 1.01 を超えたときに増やす本番の回数の上限
N_JOBS = 12                   # 並列に回す数（チェーンと最尤推定。CPU の数に合わせる）
N_PRED_DRAWS = 200            # 予測に使う事後サンプル数（間引き後）
USE_SAVED_SAMPLES = True      # 保存済みの事後サンプルがあれば読み込む（サンプリングを省く）
MAIN_DATA = "なし"            # 主に使う外れ値の扱い
TEMP_HOLDOUT = 210            # S_T でテストにする設定温度 [℃]
'''))
code(r'''# セル1.5：ライブラリ、フォント、出力先
%matplotlib inline
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator
from joblib import Parallel, delayed
from scipy.special import logsumexp
from scipy.stats import norm
import arviz as az   # 収束診断（R-hat・有効サンプル数）に使う

OUT_DIR = os.path.join(REPO, "results", "step3c_mcmc")
STEP3B_DIR = os.path.join(REPO, "results", "step3b")
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
print("ArviZ :", az.__version__)
print("出力先:", os.path.abspath(OUT_DIR))''')
md(r'''### 共通の部品

データの読み込み、問題データ一覧の「除外」、モデル、評価、データの分け方の関数。中身は `04_補助スクリプト/nb_common.py` と同じ。''')
code(COMMON_CELL)

# ---------------------------------------------------------------- 2. データ・モデル・候補
md(r'''## 2. データ・モデル・候補の選び方''')
code(r'''# セル2：データの読み込みと、モデルの定義（手順3b と同じ）
D = load_all()


def spec(kind, cols, linear=True):
    return {"kind": kind, "cols": cols, "linear": linear}


BASE = ["ln_gd", "ln_H"]
MODELS.update({
    "B0": spec("gpr", BASE), "B0T": spec("gpr", BASE + ["Tset"]),
    "B0F": spec("gpr", BASE + ["ln_Tff"]), "B0FT": spec("gpr", BASE + ["ln_Tff", "Tset"]),
    "B2": spec("gpr", BASE + ["ln_P32"]), "B2T": spec("gpr", BASE + ["ln_P32", "Tset"]),
    "B2F": spec("gpr", BASE + ["ln_Tff", "ln_P32"]), "B2FT": spec("gpr", BASE + ["ln_Tff", "ln_P32", "Tset"]),
    "R0": spec("gpr", ["ln_gd_cm", "ln_H"]), "R2": spec("gpr", ["ln_v", "ln_H", "Tset"]), "R3": spec("gpr", ["ln_gd_cm", "ln_H", "Tset"]),
    "R3F": spec("gpr", ["ln_gd_cm", "ln_H", "Tset", "ln_Tff_cm"]), "R3P": spec("gpr", ["ln_gd_cm", "ln_H", "Tset", "ln_P32_cm"]),
})
LABELS = {
    "B0": r"B0：［$\dot{\gamma}$, $H$］", "B0T": r"B0T：［$\dot{\gamma}$, $H$, $T_{set}$］",
    "B0F": r"B0F：［$\dot{\gamma}$, $H$, $T_{ff}$］", "B0FT": r"B0FT：［$\dot{\gamma}$, $H$, $T_{ff}$, $T_{set}$］",
    "B2": r"B2：［$\dot{\gamma}$, $H$, P3_2］", "B2T": r"B2T：［$\dot{\gamma}$, $H$, P3_2, $T_{set}$］",
    "B2F": r"B2F：［$\dot{\gamma}$, $H$, $T_{ff}$, P3_2］", "B2FT": r"B2FT：［$\dot{\gamma}$, $H$, $T_{ff}$, P3_2, $T_{set}$］",
    "R0": r"R0：［$\dot{\gamma}$（条件平均）, $H$］", "R2": r"R2：［$v$, $H$, $T_{set}$］", "R3": r"R3：［$\dot{\gamma}$（条件平均）, $H$, $T_{set}$］",
    "R3F": r"R3F：［$\dot{\gamma}$（条件平均）, $H$, $T_{set}$, $T_{ff}$（条件平均）］", "R3P": r"R3P：［$\dot{\gamma}$（条件平均）, $H$, $T_{set}$, P3_2（条件平均）］",
}
INPUT_JA = {"ln_gd": r"$\ln\dot{\gamma}$", "ln_H": "ln H", "ln_Tff": "ln Tff", "ln_P32": "ln P3_2", "Tset": "Tset", "ln_v": "ln v", "ln_gd_cm": r"$\ln\dot{\gamma}$（条件平均）",
            "ln_Tff_cm": "ln Tff（条件平均）", "ln_P32_cm": "ln P3_2（条件平均）"}
SPLIT_JA = {"S_V": "S_V（5data → 3data）", "S_T": f"S_T（190・230 ℃ → {TEMP_HOLDOUT} ℃）"}
for mat in MATS:
    tr, te = D[(mat, "5data", MAIN_DATA)], D[(mat, "3data", MAIN_DATA)]
    print(f"{MATS[mat]}：学習（5data）{len(tr)} 行 {tr['cond'].nunique()} 条件、検証（3data）{len(te)} 行 {te['cond'].nunique()} 条件")''')

code(r'''# セル3：候補のモデルを決める（手順3b の S_V の NLPD が良い順。B0T は必ず入れる）
def pick_candidates():
    if CANDIDATES is not None:
        return {m: list(v) for m, v in CANDIDATES.items()}, "設定エリアで指定"
    path = os.path.join(STEP3B_DIR, "metrics.csv")
    if not os.path.exists(path):
        return {m: CAND_FIXED + ["R2"] for m in MATS}, "手順3b の結果がないので、B0T・R3・R2"
    mt = pd.read_csv(path)
    out = {}
    for mat in MATS:
        t = mt[(mt["material"] == MATS[mat]) & (mt["split"] == "S_V") & (mt["data"] == MAIN_DATA) & (mt["model"].isin(CAND_POOL)) & mt["subset"].isin(["低速側", "高速側" + SENS_SUFFIX])]
        score = t.pivot_table(index="model", columns="subset", values="NLPD").mean(axis=1).drop(CAND_FIXED, errors="ignore").sort_values()
        out[mat] = CAND_FIXED + list(score.index[:N_CAND_EXTRA])
    return out, f"必ず入れる {CAND_FIXED} と、手順3b の S_V の NLPD（低速側と、高速側から感度確認の条件を除いたものの平均）が良い {N_CAND_EXTRA} 個"


CAND, CAND_HOW = pick_candidates()
for mat, keys in CAND.items():
    assert all(k in MODELS for k in keys), f"未定義のモデルがある: {keys}"
    print(f"{MATS[mat]}：{keys}")
print("選び方：", CAND_HOW)
pd.DataFrame([{"material": MATS[m], "model": k, "how": CAND_HOW} for m, ks in CAND.items() for k in ks]).to_csv(os.path.join(OUT_DIR, "candidates.csv"), index=False, encoding="utf-8-sig")''')

# ---------------------------------------------------------------- 3. サンプラー
md(r'''## 3. サンプラーと予測の関数

ハイパーパラメータは対数で扱う。θ は scikit-learn の `kernel.theta`（固定していないハイパーパラメータの対数）と同じ順番：[ln 振幅², ln 長さスケール…, ln 線形項の振幅², ln σ₀, ln ノイズ分散]。''')
code(r'''# セル4：MCMC（適応型ランダムウォーク・メトロポリス法）、予測、評価の関数
TARGET_ACC = 0.234


def theta_names(key):
    return ["log_c"] + [f"log_ls_{c}" for c in MODELS[key]["cols"]] + ["log_c_lin", "log_sigma0", "log_noise"]


PARAM_JA = {"log_c": "振幅²", "log_c_lin": "線形項の振幅²", "log_sigma0": "線形項の σ₀", "log_noise": "ノイズ分散"}


def param_ja(name):
    if name.startswith("log_ls_"):
        return "長さスケール（" + INPUT_JA[name[len("log_ls_"):]] + "）"
    return PARAM_JA[name]


def run_chain(X_tr, y_tr, d, theta0, n_warm, n_samp, seed):
    """適応型ランダムウォーク・メトロポリス法の1チェーン。
    調整中：提案分布 N(θ, λ·Σ) の Σ を足跡から学び（前半75%）、λ を採択率が TARGET_ACC に近づくよう合わせる。本番：Σ と λ を固定。"""
    rng = np.random.default_rng(seed)
    gp = GaussianProcessRegressor(kernel=kernel_for(d, True), optimizer=None).fit(X_tr, y_tr)
    lo, hi = gp.kernel_.bounds[:, 0], gp.kernel_.bounds[:, 1]

    def logpost(th):
        if np.any(th <= lo) or np.any(th >= hi):
            return -np.inf                       # 事前分布の範囲の外は確率0
        return gp.log_marginal_likelihood(th)    # 範囲内では一様なので、対数周辺尤度そのもの

    k = len(theta0)
    th = np.array(theta0, dtype=float)
    lp = logpost(th)
    cov, log_lam = np.eye(k) * 0.05 ** 2, 0.0
    warm = np.empty((n_warm, k))
    draws, lps, acc = np.empty((n_samp, k)), np.empty(n_samp), np.zeros(n_warm + n_samp, dtype=bool)
    t0 = time.time()
    for it in range(n_warm + n_samp):
        L = np.linalg.cholesky(np.exp(log_lam) * cov)
        prop = th + L @ rng.standard_normal(k)
        lp_prop = logpost(prop)
        a = np.exp(min(0.0, lp_prop - lp)) if np.isfinite(lp_prop) else 0.0
        if rng.uniform() < a:
            th, lp, acc[it] = prop, lp_prop, True
        if it < n_warm:                           # 調整（本番では行わない）
            warm[it] = th
            log_lam += (it + 1) ** -0.6 * (a - TARGET_ACC)
            if it >= 199 and (it + 1) % 50 == 0 and it < 0.75 * n_warm:
                cov = np.cov(warm[(it + 1) // 2: it + 1].T) * (2.38 ** 2 / k) + np.eye(k) * 1e-10
        else:
            draws[it - n_warm], lps[it - n_warm] = th, lp
    return {"draws": draws, "lp": lps, "accepted": acc[n_warm:], "acc_warm": acc[:n_warm].mean(),
            "acc_samp": acc[n_warm:].mean(), "lam": np.exp(log_lam), "seconds": time.time() - t0}


def gp_predict_thetas(X_tr, y_tr, X_new, d, thetas):
    """θ ごとの GPR の予測平均と標準偏差（ノイズ込み）。形は (θ の数, 点の数)"""
    base = kernel_for(d, True)
    mus, sds = [], []
    for th in np.atleast_2d(thetas):
        gp = GaussianProcessRegressor(kernel=base.clone_with_theta(th), optimizer=None).fit(X_tr, y_tr)
        mu, sd = gp.predict(X_new, return_std=True)
        mus.append(mu)
        sds.append(sd)
    return np.array(mus), np.array(sds)


def evaluate_mixture(y, mus, sds):
    """正規分布を等しい重みで混ぜた予測分布での指標。mus, sds は (成分数, 点の数)。成分が 1 つなら evaluate と同じ"""
    mus, sds = np.atleast_2d(mus), np.atleast_2d(sds)
    mu = mus.mean(axis=0)
    cdf = norm.cdf((y - mus) / sds).mean(axis=0)
    log_dens = logsumexp(norm.logpdf(y, mus, sds), axis=0) - np.log(len(mus))
    sd_mix = np.sqrt((sds ** 2 + mus ** 2).mean(axis=0) - mu ** 2)
    return {"R2": float(r2_score(y, mu)) if len(y) > 1 else np.nan, "RMSE": float(np.sqrt(mean_squared_error(y, mu))),
            "coverage95": float(np.mean((cdf >= 0.025) & (cdf <= 0.975))), "NLPD": float(-np.mean(log_dens)), "mean_sd": float(np.mean(sd_mix))}, mu, sd_mix


def to_arrays(trace, cols):
    t = trace.sort_values(["chain", "draw"])
    n_chain = t["chain"].nunique()
    return {c: t[c].to_numpy().reshape(n_chain, -1) for c in cols}


def diag_of(trace, cols):
    """最大の R-hat と、最小の有効サンプル数（bulk）"""
    arrs = to_arrays(trace, cols)
    diag = az.summary(az.from_dict({"posterior": arrs}), kind="diagnostics", round_to="none")
    return float(diag["r_hat"].max()), float(diag["ess_bulk"].min())


print("関数を定義した")''')

# ---------------------------------------------------------------- 4. サンプリング
md(r'''## 4. 学習と MCMC によるサンプリング

- 最尤推定（手順3b と同じ設定の `fit_model`）で θ の初期値を決める。4 本のチェーンの初期値は、最尤推定値に、ノイズ以外だけ N(0, 0.3²) のずれを加えたもの。
- 本番のサンプル（対数スケール）は `results/step3c_mcmc/posterior_<分割>_<材料>_<モデル>.csv` に、チェーンごとの採択率などは `chain_info.csv` に保存する。''')
code(r'''# セル5：学習データの用意と、サンプリング（保存済みのサンプルがあれば読み込む）
def get_split(mat, split):
    if split == "S_V":
        tr, te = split_V(mat, MAIN_DATA)
        return tr, {"S_V": te}
    tr, te, te3 = split_T(mat, MAIN_DATA, TEMP_HOLDOUT)
    return tr, {"S_T": te, "S_T3": te3}


def prepare(mat, split, key):
    """学習データだけで標準化・中心化した入力と目的変数"""
    tr, tests = get_split(mat, split)
    spec_ = MODELS[key]
    sc = StandardScaler().fit(tr[spec_["cols"]].to_numpy())
    y0 = tr["ln_eta"].mean()
    return tr, tests, sc, y0, sc.transform(tr[spec_["cols"]].to_numpy()), tr["ln_eta"].to_numpy() - y0


info_path = os.path.join(OUT_DIR, "chain_info.csv")
chain_info = pd.read_csv(info_path) if os.path.exists(info_path) else pd.DataFrame()
combos = [(mat, split, key) for mat in MATS for split in ["S_V", "S_T"] for key in CAND[mat]]
PREP = {c: prepare(*c) for c in combos}                           # (tr, tests, sc, y0, Xs, yc)
t_all = time.time()


def fit_job(key, tr):
    warnings.simplefilter("ignore")                  # 並列の子プロセスでも、下限への張り付きの警告を出さない
    return fit_model(key, tr)


fit_list = Parallel(n_jobs=N_JOBS)(delayed(fit_job)(key, PREP[(mat, split, key)][0]) for (mat, split, key) in combos)    # 最尤推定（並列）
MLE = dict(zip(combos, fit_list))
print(f"最尤推定：{len(combos)} 個、{(time.time() - t_all) / 60:.1f} 分")


def post_path(combo):
    mat, split, key = combo
    return os.path.join(OUT_DIR, f"posterior_{split}_{mat}_{key}.csv")


def starts_seeds(idx, combo):
    """チェーンの初期値（最尤推定値に、ノイズ以外だけ N(0, 0.3²) のずれを加える）とシード"""
    mat, split, key = combo
    d = PREP[combo][4].shape[1]
    cols = theta_names(key)
    bnd = kernel_for(d, True).bounds
    th0 = np.clip(MLE[combo]["gp"].kernel_.theta, bnd[:, 0] + 1e-6, bnd[:, 1] - 1e-6)
    assert len(th0) == len(cols), "θ の長さが想定と違う"
    rng = np.random.default_rng(SEED + 100 * idx)
    shift = np.r_[np.ones(len(cols) - 1), 0.0]
    starts = [np.clip(th0 + rng.normal(0, 0.3, len(cols)) * shift, bnd[:, 0] + 1e-6, bnd[:, 1] - 1e-6) for _ in range(CHAINS)]
    return starts, [SEED + 1000 * idx + c for c in range(CHAINS)]


traces = {}
for combo in combos:                                              # 保存済みのサンプルがあれば読み込む
    if USE_SAVED_SAMPLES and os.path.exists(post_path(combo)):
        saved = pd.read_csv(post_path(combo))
        per_chain = saved.groupby("chain").size()
        if len(per_chain) == CHAINS and (per_chain >= N_SAMP).all():
            traces[combo] = saved
            print(f"{MATS[combo[0]]} {combo[1]} {combo[2]}: 保存済みのサンプルを読み込み（1チェーン {int(per_chain.iloc[0])} 回）")

pending, n_samp = [c for c in combos if c not in traces], N_SAMP
while pending:
    tasks = []
    for combo in pending:
        starts, seeds = starts_seeds(combos.index(combo), combo)
        _, _, _, _, Xs, yc = PREP[combo]
        tasks += [(combo, c, Xs, yc, starts[c], seeds[c]) for c in range(CHAINS)]
    t0 = time.time()
    outs = Parallel(n_jobs=N_JOBS)(delayed(run_chain)(Xs, yc, Xs.shape[1], st, N_WARM, n_samp, sd) for (_, _, Xs, yc, st, sd) in tasks)
    print(f"本番 {n_samp:,}回 × {CHAINS}チェーン × {len(pending)} 組：{(time.time() - t0) / 60:.1f} 分")
    still = []
    for combo in pending:
        mat, split, key = combo
        res = [o for (t, o) in zip(tasks, outs) if t[0] == combo]
        cols = theta_names(key)
        trace = pd.concat([pd.DataFrame({"chain": c, "draw": np.arange(n_samp), **{col: r["draws"][:, j] for j, col in enumerate(cols)},
                                         "lp": r["lp"], "accepted": r["accepted"].astype(int)}) for c, r in enumerate(res)], ignore_index=True)
        rhat, ess = diag_of(trace, cols)
        print(f"  {MATS[mat]} {split} {key}: 最大 R-hat {rhat:.3f}、最小 ESS(bulk) {ess:.0f}、採択率 " + ", ".join(f"{r['acc_samp']:.2f}" for r in res))
        if (rhat > 1.01 or ess < 400) and n_samp < N_SAMP_MAX:
            still.append(combo)                                       # R-hat が大きい、または有効サンプル数が少ないので、本番を増やしてやり直す
            continue
        trace.to_csv(post_path(combo), index=False, float_format="%.7g")
        traces[combo] = pd.read_csv(post_path(combo))                 # 保存した値（7桁）で以降を計算する
        rows = pd.DataFrame([{"material": MATS[mat], "split": split, "model": key, "chain": c, "acc_warm": r["acc_warm"], "acc_samp": r["acc_samp"],
                              "lambda": r["lam"], "chain_seconds": r["seconds"], "wall_seconds": max(x["seconds"] for x in res)} for c, r in enumerate(res)])
        if len(chain_info):
            chain_info = chain_info[~((chain_info["material"] == MATS[mat]) & (chain_info["split"] == split) & (chain_info["model"] == key))]
        chain_info = pd.concat([chain_info, rows], ignore_index=True)
        chain_info.to_csv(info_path, index=False)
    pending, n_samp = still, min(int(n_samp * 2.5), N_SAMP_MAX)

RUN_INFO = {k: {"chains": int(t["chain"].nunique()), "n_warm": N_WARM, "n_samp_per_chain": int(len(t) // t["chain"].nunique()), "draws_total": int(len(t))}
            for k, t in traces.items()}
run_df = pd.DataFrame([{"material": MATS[m], "split": s, "model": k, **info} for (m, s, k), info in RUN_INFO.items()])
print(f"\n合計 {(time.time() - t_all) / 60:.1f} 分")
print(run_df.to_string(index=False))''')

# ---------------------------------------------------------------- 5. 収束診断
md(r'''## 5. 収束診断と事後分布

- R-hat（1.01 以下が目安）と有効サンプル数 ESS（bulk・tail。400 以上が目安）は ArviZ で計算する。
- 事後分布の要約は、振幅（√振幅²）・長さスケール・σ₀・ノイズ分散の元の単位で示す（長さスケールは標準化後の入力に対する値）。MLE は最尤推定値。''')
code(r'''# セル6：収束診断と事後分布の要約
rows = []
for (mat, split, key), trace in traces.items():
    cols = theta_names(key)
    arrs = to_arrays(trace, cols)
    diag = az.summary(az.from_dict({"posterior": arrs}), kind="diagnostics", round_to="none")
    th_mle = MLE[(mat, split, key)]["gp"].kernel_.theta
    bnd = kernel_for(len(MODELS[key]["cols"]), True).bounds
    for j, col in enumerate(cols):
        v = np.exp(arrs[col].ravel())
        label, mle, lo_b, hi_b = col.replace("log_", ""), np.exp(th_mle[j]), np.exp(bnd[j, 0]), np.exp(bnd[j, 1])
        if col in ("log_c", "log_c_lin"):                                  # 振幅² → 振幅
            v, mle, lo_b, hi_b = np.sqrt(v), np.sqrt(mle), np.sqrt(lo_b), np.sqrt(hi_b)
            label = "amplitude" if col == "log_c" else "lin_amplitude"
        q = np.percentile(v, [2.5, 50, 97.5])
        rows.append({"material": MATS[mat], "split": split, "model": key, **RUN_INFO[(mat, split, key)], "param": label, "MLE": mle,
                     "post_mean": v.mean(), "post_sd": v.std(ddof=1), "q2.5": q[0], "median": q[1], "q97.5": q[2],
                     "r_hat": diag.loc[col, "r_hat"], "ess_bulk": diag.loc[col, "ess_bulk"], "ess_tail": diag.loc[col, "ess_tail"],
                     "post_at_bound": bool(q[0] < lo_b * 1.05 or q[2] > hi_b * 0.95)})
post_df = pd.DataFrame(rows)
post_df.to_csv(os.path.join(OUT_DIR, "posterior_summary.csv"), index=False, encoding="utf-8-sig")
print(post_df[["material", "split", "model", "param", "MLE", "median", "q2.5", "q97.5", "r_hat", "ess_bulk", "post_at_bound"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

stat_rows = []
for (mat, split, key), trace in traces.items():
    ci = chain_info[(chain_info["material"] == MATS[mat]) & (chain_info["split"] == split) & (chain_info["model"] == key)]
    p = post_df[(post_df["material"] == MATS[mat]) & (post_df["split"] == split) & (post_df["model"] == key)]
    stat_rows.append({"material": MATS[mat], "split": split, "model": key, **RUN_INFO[(mat, split, key)],
                      "acc_samp_min": ci["acc_samp"].min(), "acc_samp_max": ci["acc_samp"].max(), "sampling_min": ci["wall_seconds"].iloc[0] / 60,
                      "max_r_hat": p["r_hat"].max(), "min_ess_bulk": p["ess_bulk"].min(), "min_ess_tail": p["ess_tail"].min(), "n_post_at_bound": int(p["post_at_bound"].sum())})
stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv(os.path.join(OUT_DIR, "sampler_stats.csv"), index=False, encoding="utf-8-sig")
print("\n【サンプラーの統計】")
print(stat_df.round(3).to_string(index=False))
n_bad = int(((stat_df["max_r_hat"] > 1.01) | (stat_df["min_ess_bulk"] < 400)).sum())
judge(n_bad == 0, f"全 {len(stat_df)} 組で R-hat ≤ 1.01、ESS(bulk) ≥ 400（満たさないのは {n_bad} 組。満たさないなら、N_SAMP を増やして再実行）")
n_bnd = int((stat_df["n_post_at_bound"] > 0).sum())
judge(n_bnd == 0, f"事後分布の 95% 区間が、事前分布の範囲の端（下限・上限の 5% 以内）に届いたモデルは {n_bnd} 組（届いたら、そのハイパーパラメータは、データだけでは決まらない）")''')

code(r'''# セル7：トレース図（左：チェーンごとの分布、右：本番の推移。値は log10）。S_V の学習のものを材料×モデルごとに保存する
CHAIN_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
for (mat, split, key), trace in traces.items():
    cols = theta_names(key)
    fig, axes = plt.subplots(len(cols), 2, figsize=(12, 1.9 * len(cols)), gridspec_kw={"width_ratios": [1, 3]})
    for i, col in enumerate(cols):
        for ch, t in trace.groupby("chain"):
            v = t[col].to_numpy() / np.log(10)
            axes[i, 0].hist(v, bins=40, histtype="step", density=True, color=CHAIN_COLORS[ch % 4], label=f"チェーン {ch + 1}")
            axes[i, 1].plot(t["draw"], v, lw=0.3, alpha=0.7, color=CHAIN_COLORS[ch % 4])
        axes[i, 0].set_ylabel(f"log10 {param_ja(col)}", fontsize=8)
        axes[i, 1].set_ylabel(f"log10 {param_ja(col)}", fontsize=8)
        axes[i, 0].xaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[0, 0].legend(fontsize=7)
    axes[-1, 0].set_xlabel("値（log10）")
    axes[-1, 1].set_xlabel("本番の反復回数")
    r = RUN_INFO[(mat, split, key)]
    fig.suptitle(f"トレース図：{MATS[mat]}　{LABELS[key]}、{SPLIT_JA[split]}\n本番 {r['n_samp_per_chain']:,}回 × {r['chains']}チェーン（調整 {r['n_warm']:,}回）", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"fig_trace_{split}_{mat}_{key}.png"), dpi=FIG_DPI)
    plt.close(fig)
print("トレース図を保存した（fig_trace_*.png）")''')

code(r'''# セル8：事後分布と最尤推定値（赤の破線）、事前分布の範囲の端（灰色の点線）。材料×分割ごとに 1 枚
for mat in MATS:
    for split in ["S_V", "S_T"]:
        keys = CAND[mat]
        n_col = max(len(theta_names(k)) for k in keys)
        fig, axes = plt.subplots(len(keys), n_col, figsize=(2.9 * n_col, 2.5 * len(keys)), squeeze=False)
        for i, key in enumerate(keys):
            cols, th_mle = theta_names(key), MLE[(mat, split, key)]["gp"].kernel_.theta
            bnd = kernel_for(len(MODELS[key]["cols"]), True).bounds
            for j in range(n_col):
                ax = axes[i, j]
                if j >= len(cols):
                    ax.axis("off")
                    continue
                v = traces[(mat, split, key)][cols[j]].to_numpy() / np.log(10)
                ax.hist(v, bins=50, density=True, color="tab:blue", alpha=0.6, label="事後分布（MCMC）")
                ax.axvline(th_mle[j] / np.log(10), color="red", ls="--", lw=1.2, label="最尤推定値")
                for b in bnd[j] / np.log(10):
                    if v.min() - 0.3 < b < v.max() + 0.3:
                        ax.axvline(b, color="gray", ls=":", lw=1.2, label="事前分布の範囲の端")
                ax.set_title(f"{key}：{param_ja(cols[j])}", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                if i == 0 and j == 0:
                    ax.legend(fontsize=6)
        fig.suptitle(f"事後分布（MCMC）と最尤推定値：{MATS[mat]}、{SPLIT_JA[split]}（横軸は log10）", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"fig_posterior_vs_mle_{split}_{mat}.png"), dpi=FIG_DPI)
        plt.show()''')

# ---------------------------------------------------------------- 6. 予測と評価
md(r'''## 6. 予測と評価（最尤推定と MCMC の比較）

- **MCMC**：事後サンプルを等間隔に N_PRED_DRAWS 個選び、各サンプルの GPR の予測分布（ノイズ込み）を混ぜる。予測平均は混合の平均、95%区間の被覆率は混合分布の累積分布、NLPD は混合分布の密度で計算する。
- **MLE**：最尤推定値 1 つで予測する（手順3b と同じ）。
- `mean_sd` は、テストデータでの予測標準偏差（ln η）の平均。予測区間の広さの目安。
- 「（感度確認の条件を除く）」は、検証の P03（PP の 190 ℃・0.5 mm・320 mm/s）を評価から除いた指標。''')
code(r'''# セル9：予測と評価
res_rows, PREDS = [], {}
t_all = time.time()
for (mat, split, key), trace in traces.items():
    tr, tests, sc, y0, Xs, yc = prepare(mat, split, key)
    d = Xs.shape[1]
    tr_ = trace.sort_values(["chain", "draw"])
    idx = np.linspace(0, len(tr_) - 1, N_PRED_DRAWS).round().astype(int)
    thetas = tr_[theta_names(key)].to_numpy()[idx]
    th_mle = MLE[(mat, split, key)]["gp"].kernel_.theta
    for tname, te in tests.items():
        Xt = sc.transform(te[MODELS[key]["cols"]].to_numpy())
        y = te["ln_eta"].to_numpy()
        mus, sds = gp_predict_thetas(Xs, yc, Xt, d, thetas)
        mu1, sd1 = gp_predict_thetas(Xs, yc, Xt, d, th_mle)
        subsets = {"全体": np.ones(len(te), bool)}
        if "group" in te.columns:
            for g in te["group"].unique():
                if g != "全体":
                    subsets[g] = (te["group"] == g).to_numpy()
        if te["is_sens"].any():
            subsets.update({s + SENS_SUFFIX: m & ~te["is_sens"].to_numpy() for s, m in list(subsets.items())})
        for method, (m_, s_) in {"MCMC": (mus + y0, sds), "MLE": (mu1 + y0, sd1)}.items():
            for sub, mask in subsets.items():
                if mask.sum() == 0:
                    continue
                ev, _, _ = evaluate_mixture(y[mask], m_[:, mask], s_[:, mask])
                res_rows.append({"material": MATS[mat], "split": tname, "model": key, "method": method, "subset": sub, "n_test": int(mask.sum()), **ev})
            ev_all, mu_mix, sd_mix = evaluate_mixture(y, m_, s_)
            PREDS[(mat, tname, key, method)] = pd.DataFrame({"y": y, "mu": mu_mix, "sd": sd_mix, "H": te["H"].to_numpy(), "v": te["v"].to_numpy(), "Tset": te["Tset"].to_numpy(),
                                                             "group": te["group"].to_numpy(), "is_sens": te["is_sens"].to_numpy()})
    print(f"{MATS[mat]} {split} {key}: 予測 完了")
mcmc_df = pd.DataFrame(res_rows)
mcmc_df.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False, encoding="utf-8-sig")
print(f"\n合計 {(time.time() - t_all) / 60:.1f} 分")


def show_cmp(mat, tname, subs, cols=("RMSE", "coverage95", "NLPD", "mean_sd")):
    t = mcmc_df[(mcmc_df["material"] == MATS[mat]) & (mcmc_df["split"] == tname) & mcmc_df["subset"].isin(subs)]
    p = t.pivot_table(index=["model", "method"], columns="subset", values=list(cols))
    p = p.reindex(columns=[(c, s) for c in cols for s in subs if (c, s) in p.columns])
    print(p.round(3).to_string())


for mat in MATS:
    print(f"\n【S_V】{MATS[mat]}")
    show_cmp(mat, "S_V", ["低速側", "高速側", "高速側" + SENS_SUFFIX])
    print(f"\n【S_T】{MATS[mat]}（{TEMP_HOLDOUT} ℃ をテスト）")
    show_cmp(mat, "S_T", ["全体"])
    show_cmp(mat, "S_T3", ["全体"])''')

code(r'''# セル10：MLE と MCMC の差（被覆率・NLPD・区間の広さ）の判定
diffs = []
for mat in MATS:
    for tname, sub in [("S_V", "低速側"), ("S_V", "高速側"), ("S_T", "全体"), ("S_T3", "全体")]:
        for key in CAND[mat]:
            a = mcmc_df[(mcmc_df["material"] == MATS[mat]) & (mcmc_df["split"] == tname) & (mcmc_df["model"] == key) & (mcmc_df["subset"] == sub)].set_index("method")
            if len(a) < 2:
                continue
            diffs.append({"material": MATS[mat], "split": tname, "subset": sub, "model": key,
                          "RMSE_MLE": a.loc["MLE", "RMSE"], "RMSE_MCMC": a.loc["MCMC", "RMSE"],
                          "cover_MLE": a.loc["MLE", "coverage95"], "cover_MCMC": a.loc["MCMC", "coverage95"],
                          "NLPD_MLE": a.loc["MLE", "NLPD"], "NLPD_MCMC": a.loc["MCMC", "NLPD"],
                          "sd_MLE": a.loc["MLE", "mean_sd"], "sd_MCMC": a.loc["MCMC", "mean_sd"],
                          "NLPD_diff": a.loc["MCMC", "NLPD"] - a.loc["MLE", "NLPD"], "sd_ratio": a.loc["MCMC", "mean_sd"] / a.loc["MLE", "mean_sd"]})
diff_df = pd.DataFrame(diffs)
diff_df.to_csv(os.path.join(OUT_DIR, "mle_vs_mcmc.csv"), index=False, encoding="utf-8-sig")
print(diff_df.round(3).to_string(index=False))
for mat in MATS:
    s = diff_df[diff_df["material"] == MATS[mat]]
    big = s[s["NLPD_diff"].abs() >= 0.2]
    judge(big.empty, f"{MATS[mat]}：MCMC と最尤推定の NLPD の差が、すべて 0.2 未満（最大 {s['NLPD_diff'].abs().max():.2f}）。差が小さければ、最尤推定で十分。0.2 以上のもの：{[f'{a} {b} {c}（MCMC − 最尤 = {d:+.2f}）' for a, b, c, d in zip(big['split'], big['subset'], big['model'], big['NLPD_diff'])] or 'なし'}。負の大きな値は、最尤推定が局所解に止まっていた（MCMC のほうが良い）ことを表す")
    wide = s[~s["sd_ratio"].between(0.8, 1.5)]
    judge(wide.empty, f"{MATS[mat]}：MCMC の予測区間の広さ（平均の標準偏差）が、最尤推定の 0.8〜1.5 倍（範囲 {s['sd_ratio'].min():.2f}〜{s['sd_ratio'].max():.2f}）。外れたもの：{[f'{a} {b} {c}（{d:.2f} 倍）' for a, b, c, d in zip(wide['split'], wide['subset'], wide['model'], wide['sd_ratio'])] or 'なし'}")''')

# ---------------------------------------------------------------- 7. 図
md(r'''## 7. 図

- 図 A：最尤推定と MCMC の、被覆率・NLPD・予測区間の広さの比較
- 図 B：設定温度の効果（190/230 ℃ の粘度比）の事後分布。ハイパーパラメータの不確かさによる、粘度比のばらつき
- 図 C：R3・B0T の予測曲線（S_V の学習）。MCMC の混合予測の 95% 区間と、最尤推定の 95% 区間''')
code(r'''# セル11：図A 最尤推定と MCMC の比較
PANELS = [("S_V", "低速側"), ("S_V", "高速側"), ("S_T", "全体"), ("S_T3", "全体")]
fig, axes = plt.subplots(3, len(MATS), figsize=(15, 12))
for j, mat in enumerate(MATS):
    for i, (col, ylab) in enumerate([("coverage95", "95%区間の被覆率"), ("NLPD", "NLPD（小さいほど良い）"), ("mean_sd", "予測区間の広さ（標準偏差の平均）")]):
        ax = axes[i, j]
        items = [(p, k) for p in PANELS for k in CAND[mat]]
        x = np.arange(len(items))
        for method, off, colr in [("MLE", -0.2, "tab:red"), ("MCMC", 0.2, "tab:blue")]:
            vals = []
            for (tname, sub), key in items:
                a = mcmc_df[(mcmc_df["material"] == MATS[mat]) & (mcmc_df["split"] == tname) & (mcmc_df["model"] == key) & (mcmc_df["subset"] == sub) & (mcmc_df["method"] == method)]
                vals.append(float(a[col].iloc[0]) if len(a) else np.nan)
            ax.bar(x + off, vals, 0.4, color=colr, label="最尤推定" if method == "MLE" else "MCMC")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}{'' if s == '全体' else chr(10) + s}\n{k}" for (t, s), k in items], fontsize=6.5)
        ax.set_title(f"{MATS[mat]}：{ylab}")
        if col == "coverage95":
            ax.axhline(0.95, color="k", ls="--", lw=0.8)
        ax.grid(alpha=0.3, axis="y")
        if i == 0 and j == 0:
            ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figA_mle_vs_mcmc.png"), dpi=FIG_DPI)
plt.show()''')

code(r'''# セル12：図B 設定温度の効果（Tset だけを 190 → 230 ℃）の事後分布
TSET_CAND = {mat: [k for k in CAND[mat] if "Tset" in MODELS[k]["cols"]] for mat in MATS}
eff_rows = []
fig, axes = plt.subplots(len(MATS), 4, figsize=(18, 4.6 * len(MATS)), squeeze=False)
for i, mat in enumerate(MATS):
    pool = pool_of(mat, MAIN_DATA)
    base = pool[(pool["Tset"] == 210) & ~pool["is_sens"]]
    obs = None
    for j, h in enumerate([0.5, 1.0, 1.5, 2.0]):
        ax = axes[i, j]
        g = base[base["H"] == h]
        data_, labels_ = [], []
        for key in TSET_CAND[mat]:
            tr, tests, sc, y0, Xs, yc = prepare(mat, "S_V", key)
            trace = traces[(mat, "S_V", key)].sort_values(["chain", "draw"])
            idx = np.linspace(0, len(trace) - 1, N_PRED_DRAWS).round().astype(int)
            thetas = trace[theta_names(key)].to_numpy()[idx]
            lo_, hi_ = g.copy(), g.copy()
            lo_["Tset"], hi_["Tset"] = 190.0, 230.0
            m_lo, _ = gp_predict_thetas(Xs, yc, sc.transform(lo_[MODELS[key]["cols"]].to_numpy()), Xs.shape[1], thetas)
            m_hi, _ = gp_predict_thetas(Xs, yc, sc.transform(hi_[MODELS[key]["cols"]].to_numpy()), Xs.shape[1], thetas)
            ratio = np.exp(np.median(m_lo - m_hi, axis=1))                                  # θ ごとの、210 ℃ のショットの粘度比の中央値
            f1 = MLE[(mat, "S_V", key)]
            r_mle = float(np.exp(np.median(predict_model(key, f1, lo_)[0] - predict_model(key, f1, hi_)[0])))
            eff_rows.append({"material": MATS[mat], "H": h, "model": key, "ratio_mle": r_mle, "ratio_mcmc_median": float(np.median(ratio)),
                             "ratio_q2.5": float(np.percentile(ratio, 2.5)), "ratio_q97.5": float(np.percentile(ratio, 97.5)), "share_gt1": float(np.mean(ratio > 1))})
            data_.append(ratio)
            labels_.append(key)
        if data_:
            ax.violinplot(data_, showmedians=True)
            ax.set_xticks(np.arange(1, len(labels_) + 1))
            ax.set_xticklabels(labels_)
            ax.axhline(1.0, color="gray", lw=0.8)
        ax.set_title(f"{MATS[mat]}　厚み {h} mm", fontsize=10)
        ax.set_ylabel("粘度の比（190 ℃ ÷ 230 ℃）")
        ax.grid(alpha=0.3, axis="y")
fig.suptitle("設定温度の効果（Tset だけを 190 → 230 ℃）の事後分布：ハイパーパラメータの不確かさ（S_V の学習）", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figB_tset_effect_posterior.png"), dpi=FIG_DPI)
plt.show()
eff_df = pd.DataFrame(eff_rows)
eff_df.to_csv(os.path.join(OUT_DIR, "tset_effect_mcmc.csv"), index=False, encoding="utf-8-sig")
print(eff_df.round(3).to_string(index=False))
for mat in MATS:
    e = eff_df[eff_df["material"] == MATS[mat]]
    judge(bool((e["share_gt1"] > 0.95).all()), f"{MATS[mat]}：どの厚み・モデルでも、事後サンプルの 95% 以上で、190 ℃ のほうが 230 ℃ より粘い（最小 {e['share_gt1'].min():.2f}）")''')

code(r'''# セル13：図C 予測曲線（S_V の学習。MCMC と最尤推定の 95% 区間）。R3（条件平均の γ̇）と B0T（ショットごとの γ̇）の 2 枚
T_COL = {190: "tab:blue", 210: "tab:green", 230: "tab:red"}
H_LEVELS = [0.5, 1.0, 1.5, 2.0]


def curve_fig(key):
    fig, axes = plt.subplots(len(H_LEVELS), len(MATS), figsize=(6.5 * len(MATS), 14), squeeze=False)
    for j, mat in enumerate(MATS):
        tr, tests, sc, y0, Xs, yc = prepare(mat, "S_V", key)
        trace = traces[(mat, "S_V", key)].sort_values(["chain", "draw"])
        idx = np.linspace(0, len(trace) - 1, N_PRED_DRAWS).round().astype(int)
        thetas = trace[theta_names(key)].to_numpy()[idx]
        allp = pool_of(mat, MAIN_DATA)
        te_cond = set(D[(mat, "3data", MAIN_DATA)]["cond"])
        for i, h in enumerate(H_LEVELS):
            ax = axes[i, j]
            for t, col in T_COL.items():
                sub = allp[(allp["H"] == h) & (allp["Tset"] == t)]
                if sub.empty:
                    continue
                lg = np.linspace(sub["ln_gd"].min(), sub["ln_gd"].max(), 100)
                Xg = pd.DataFrame({"ln_gd": lg, "ln_gd_cm": lg, "ln_H": np.log(h), "Tset": float(t)})
                Xn = sc.transform(Xg[MODELS[key]["cols"]].to_numpy())
                mus, sds = gp_predict_thetas(Xs, yc, Xn, Xs.shape[1], thetas)
                _, mu, sd = evaluate_mixture(np.zeros(len(lg)), mus + y0, sds)
                mu1, sd1, _, _ = predict_model(key, MLE[(mat, "S_V", key)], Xg)
                ax.plot(np.exp(lg), np.exp(mu), color=col, lw=1.4)
                ax.fill_between(np.exp(lg), np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd), color=col, alpha=0.15)
                ax.plot(np.exp(lg), np.exp(mu1 - 1.96 * sd1), color=col, lw=0.7, ls="--")
                ax.plot(np.exp(lg), np.exp(mu1 + 1.96 * sd1), color=col, lw=0.7, ls="--")
                cm = sub.groupby("cond").agg(lg=("ln_gd", "mean"), le=("ln_eta", "mean"))
                is_te = np.array([c in te_cond for c in cm.index])
                ax.scatter(np.exp(cm["lg"][~is_te]), np.exp(cm["le"][~is_te]), s=14, color=col, alpha=0.8, zorder=3)
                ax.scatter(np.exp(cm["lg"][is_te]), np.exp(cm["le"][is_te]), s=26, facecolors="none", edgecolors=col, marker="s", linewidths=1.2, zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
            ax.set_ylabel(f"厚み {h} mm\n" + r"$\eta$ [Pa·s]")
            if i == 0:
                ax.set_title(f"{MATS[mat]}：{key}（帯：MCMC の 95% 区間、破線：最尤推定の 95% 区間）", fontsize=9)
            if i == len(H_LEVELS) - 1:
                ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ [1/s]")
    handles = [plt.Line2D([], [], color=c, label=f"{t} ℃") for t, c in T_COL.items()]
    handles += [plt.Line2D([], [], color="gray", marker="o", ls="", label="学習の条件平均"), plt.Line2D([], [], color="gray", marker="s", mfc="none", ls="", label="検証の条件平均")]
    axes[0, 0].legend(handles=handles, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"figC_{key}_curves.png"), dpi=FIG_DPI)
    plt.show()


for key_ in ["R3", "B0T"]:
    curve_fig(key_)''')

# ---------------------------------------------------------------- 8. 判定のまとめ
md(r'''## 8. 判定のまとめ''')
code(r'''# セル14：保存したファイルと、判定のまとめ
print("【判定のまとめ】（上の各節の判定を、観点ごとにもう一度）")
judge(n_bad == 0, f"収束：R-hat ≤ 1.01、ESS(bulk) ≥ 400（満たさないのは {n_bad} 組）")
judge(n_bnd == 0, f"事前分布の範囲の端に届いたモデルは {n_bnd} 組")
for mat in MATS:
    s = diff_df[diff_df["material"] == MATS[mat]]
    judge(bool((s["NLPD_diff"].abs() < 0.2).all()), f"{MATS[mat]}：MCMC と最尤推定の NLPD の差 最大 {s['NLPD_diff'].abs().max():.2f}（0.2 未満なら、最尤推定で十分。大きいのは、最尤推定が局所解に止まったモデル）")
print("\n保存したファイル:")
for f in sorted(os.listdir(OUT_DIR)):
    print("  ", os.path.normpath(os.path.join(OUT_DIR, f)))''')

# ---------------------------------------------------------------- 9. 高解像度のグラフ
md(HR_MD.replace("## 高解像度のグラフ", "## 9. 高解像度のグラフ").replace("線は予測の平均、帯は 95% 区間", "線は MCMC の予測の平均（事後サンプルの予測を混ぜた分布）、帯は 95% 区間"))
code(hr_cell("# セル15：高解像度のグラフ（せん断速度 ― 粘度。MCMC の予測。GPR6月ノートのセル9-2 にならう）",
             '''HR_MODELS = ["R3"]                 # 描くモデル（このノートで MCMC をかけたモデル。例：["R3", "B0T", "R2"]。記号の意味は、ノート冒頭の表）
HR_SPLIT = "S_V"                   # どの学習のモデルか："S_V"（5data で学習）／"S_T"（190・230 ℃ で学習）。MCMC のサンプルがあるものだけ''',
             r'''HR_SPLIT_JA = {"S_V": "5data で学習", "S_T": "190・230 ℃ で学習"}


def hr_predict_mcmc(mat, split, key):
    """MCMC の事後サンプル（N_PRED_DRAWS 個）で予測した分布を等しい重みで混ぜた、予測の平均・標準偏差（ノイズ込み）・標準偏差（ノイズを除く）"""
    _, _, sc, y0, Xs, yc = PREP[(mat, split, key)]
    trace = traces[(mat, split, key)].sort_values(["chain", "draw"])
    idx = np.linspace(0, len(trace) - 1, N_PRED_DRAWS).round().astype(int)
    thetas = trace[theta_names(key)].to_numpy()[idx]
    noise = np.exp(thetas[:, -1])                                         # θ の最後は ln ノイズ分散

    def predict(Xg):
        Xn = sc.transform(Xg[MODELS[key]["cols"]].to_numpy())
        mus, sds = gp_predict_thetas(Xs, yc, Xn, Xs.shape[1], thetas)
        mus = mus + y0
        mu = mus.mean(axis=0)
        sd = np.sqrt((sds ** 2 + mus ** 2).mean(axis=0) - mu ** 2)
        sdf = np.sqrt((np.maximum(sds ** 2 - noise[:, None], 0.0) + mus ** 2).mean(axis=0) - mu ** 2)
        return mu, sd, sdf
    return predict


hr_saved = []
for mat in HR_MATERIALS:
    for key in HR_MODELS:
        if (mat, HR_SPLIT, key) not in traces:
            print(f"{MATS[mat]} {HR_SPLIT} {key}：MCMC のサンプルがないので、描かない（上の候補のモデルを指定する）")
            continue
        t0 = time.time()
        hr_saved += hr_flow(hr_predict_mcmc(mat, HR_SPLIT, key), mat, MAIN_DATA, hr_split_conds(mat, MAIN_DATA, HR_SPLIT, TEMP_HOLDOUT),
                            f"{LABELS[key]}（{HR_SPLIT_JA[HR_SPLIT]}、MCMC）", f"hr_{mat}_{key}_{HR_SPLIT}_mcmc", HR_DIR, **HR_KW)
        print(f"{MATS[mat]} {key}：保存したファイル 累計 {len(hr_saved)} 個（{time.time() - t0:.0f} 秒）")
judge(len(hr_saved) > 0, f"高解像度のグラフを {len(hr_saved)} 個保存した（{os.path.normpath(HR_DIR)}）。形式 {HR_FORMATS}、{HR_DPI} dpi")'''))


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
