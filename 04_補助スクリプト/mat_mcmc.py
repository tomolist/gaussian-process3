# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR の学習用のノートの MCMC 版（02_MCMC(適応型RWM)/）で使う部品。

2026-11-17 のユーザーの依頼：学習用のノートで、対数周辺尤度の扱いを、勾配法（L-BFGS-B）から適応型 RWM の MCMC に変える。
ユーザーの決定：温度シフトの係数（E_h・E_v など）も、カーネルのハイパーパラメータと一緒にサンプリングする（格子は使わない）。
予測は事後分布の混合（間引いたサンプルごとの予測の平均と分散を混ぜる）。
ノート（build_mat_shift_mcmc_nb.py が作る）の「MCMC の部品」のセルに、この中身がそのまま入る（共通の部品 mat_common.py の後）。
モジュールとして import しても使える（そのときは mat_common の中身を読み込む）。
適応型 RWM は、本体の手順5b（step5b_core.py の run_chain5）・手順3c と同じ方式：調整期間に、提案 N(φ, λΣ) の Σ を足跡から学び（Haario 型）、
λ を採択率が MCMC_TARGET_ACC に近づくように合わせる。本番は Σ と λ を固定した、ふつうのメトロポリス法。
事前分布：カーネルのハイパーパラメータは kernel.bounds の中で対数一様（手順3c・5b と同じ）、係数は SHIFT_PRIOR の範囲で一様。
範囲の中では、対数事後確率は対数周辺尤度（LML）と定数の差だけ。
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve

try:                                                         # ノートの中では、共通の部品（mat_common の中身）がすでにある
    design
except NameError:                                            # モジュールとして import したとき
    from mat_common import *  # noqa: F401,F403
    from mat_common import choose_inputs, predict            # noqa: F401（下で包むため、名前を確かめて読む）

# ---------------------------------------------------------------- 設定（ノートの設定エリアの値があれば、それを使う）
MCMC_CHAINS = globals().get("MCMC_CHAINS", 4)                   # チェーンの数（並列に回す）
MCMC_WARM = globals().get("MCMC_WARM", 5000)                    # 調整期間の反復数（提案の分布を合わせる。捨てる）
MCMC_SAMPLES = globals().get("MCMC_SAMPLES", 20000)             # 本番の反復数（1 チェーンあたり）
MCMC_MAX_SAMPLES = globals().get("MCMC_MAX_SAMPLES", 100000)    # 収束の基準に届かないとき、本番を続きから足す上限（1 チェーンあたり）
MCMC_RHAT = globals().get("MCMC_RHAT", 1.01)                    # 収束の基準：R-hat がこれ以下
MCMC_ESS = globals().get("MCMC_ESS", 400)                       # 収束の基準：有効サンプル数（bulk）がこれ以上
MCMC_TARGET_ACC = globals().get("MCMC_TARGET_ACC", 0.234)       # 目標の採択率（調整期間に歩幅を合わせる）
MCMC_INIT = globals().get("MCMC_INIT", "mle")                   # 初期値："mle"＝係数 0 の最尤推定（L-BFGS-B）のカーネルから／"default"＝カーネルの既定値から（係数はどちらも 0 から）
MCMC_INIT_RESTARTS = globals().get("MCMC_INIT_RESTARTS", 5)     # "mle" の最尤推定のやり直し回数
MCMC_INIT_SD = globals().get("MCMC_INIT_SD", (0.3, 1.0))        # チェーンごとの初期値のずれの標準偏差（カーネルの対数, 係数）。ノイズにはずれを足さない
MCMC_PROP_SD = globals().get("MCMC_PROP_SD", (0.05, 0.5))       # 提案の分布の最初の標準偏差（カーネルの対数, 係数）
SHIFT_PRIOR = globals().get("SHIFT_PRIOR", {"arrhenius": {"h": (-20.0, 40.0), "v": (-20.0, 60.0)},
                                            "wlf": {"h": (-20.0, 40.0), "v": (-20.0, 40.0)}})   # 係数の事前分布（一様）の範囲
N_PRED_DRAWS = globals().get("N_PRED_DRAWS", 200)               # 予測に使う事後サンプルの数（全チェーンから等間隔に選ぶ）
MCMC_SAVE_ROWS = globals().get("MCMC_SAVE_ROWS", 10000)         # 事後サンプルの CSV に書く行の数の上限（等間隔に間引く）
REQUIRE_MCMC_FOLDER = globals().get("REQUIRE_MCMC_FOLDER", True)   # True：学習結果フォルダの名前に「MCMC」がなければ止める（勾配法版の結果の上書きを防ぐ）
GP_ALPHA = 1e-10                                                # 対角に足す小さな値（scikit-learn の GaussianProcessRegressor の既定と同じ）


# ---------------------------------------------------------------- 結果のフォルダの名前の確かめ（choose_inputs を包む）
_choose_inputs_point = globals().get("_choose_inputs_point", choose_inputs)   # 共通の部品の choose_inputs（このセルを実行し直しても、包んだものを包まない）


def _mcmc_name_ok(p):
    return "MCMC" in os.path.basename(os.path.normpath(p))


def choose_inputs(mats, files=None, need_data=True):
    """共通の部品の choose_inputs と同じ。ただし REQUIRE_MCMC_FOLDER が True なら、学習結果フォルダの名前に「MCMC」があるかを確かめる
    （勾配法版の学習結果フォルダを選んで、同じ名前の図・表を上書きするのを防ぐ）。設定 FILES に書いた名前は、フォルダを作る前に確かめる"""
    msg = "MCMC 版の結果は、名前に「MCMC」を含む学習結果フォルダ（例：4回目_01_バージンPP再々実験_学習結果_MCMC）に置きます。選び直してください：{}"
    if REQUIRE_MCMC_FOLDER:
        bad = [str(f["結果"]) for f in (files or {}).values() if f.get("結果") and not _mcmc_name_ok(f["結果"])]
        if bad:
            raise ValueError(msg.format("、".join(bad)))
    out = _choose_inputs_point(mats, files, need_data)
    if REQUIRE_MCMC_FOLDER:
        bad = [SOURCES[m]["結果"] for m in mats if not _mcmc_name_ok(SOURCES[m]["結果"])]
        if bad:
            raise ValueError(msg.format("、".join(bad)))
    return out


# ---------------------------------------------------------------- サンプリングする形
def mcmc_space(key, tr, cfg, mode=None):
    """MCMC に使う形。パラメータ φ = [カーネルの theta（対数）…, θ_h（B だけ）, θ_v（A・B）]（C はカーネルだけ）。
    標準化と中心化の定数は fit_gp と同じ（シフト前の入力の平均・標準偏差、ln η の平均。θ によらない）。
    シフトの式は θ に比例する（ln a(T; θ) = θ × ln a(T; 1)）ので、ln a(T; 1) を先に計算しておく"""
    mode = INPUT_MODE if mode is None else mode
    X0, _, _ = design(tr, key, cfg, 0.0, 0.0, mode)
    mu_x, sd_x = X0.mean(axis=0), X0.std(axis=0)
    y = tr["ln_eta"].to_numpy(float)
    _, tcol = layout(key)
    s1 = shift_ln(tr[mode_col(tr, tcol, mode)].to_numpy(float), 1.0, cfg) if tcol else np.zeros(len(tr))
    kern = kernel_for(X0.shape[1])
    hp = [h for h in kern.hyperparameters if not h.fixed]
    names = []
    for h in hp:                                             # カーネルの theta の順（対数）に、読める名前を付ける
        if h.name.endswith("length_scale"):
            names += [f"ln_ls_{c}" for c in input_labels(key)]
        elif h.name.endswith("noise_level"):
            names.append("ln_noise")
        elif h.name.endswith("sigma_0"):
            names.append("ln_lin_sigma0")
        elif h.name == "k1__k1__k1__constant_value" or h.name == "k1__k1__constant_value":
            names.append("ln_amp2")
        elif h.name == "k1__k2__k1__constant_value":
            names.append("ln_lin_c")
        else:
            raise ValueError(f"カーネルのハイパーパラメータの名前が想定と違う：{h.name}")
    nk = len(kern.theta)
    if len(names) != nk:
        raise ValueError(f"カーネルのハイパーパラメータの数（{nk}）と名前（{len(names)}）が合わない")
    thn, tvn, _ = theta_names(cfg)
    shift = [] if key == "C" else (["th", "tv"] if key.split("_")[0] == "B" else ["tv"])
    pr = SHIFT_PRIOR[cfg["type"]]
    lo = np.r_[kern.bounds[:, 0], [pr[s[1]][0] for s in shift]]
    hi = np.r_[kern.bounds[:, 1], [pr[s[1]][1] for s in shift]]
    return {"key": key, "cfg": dict(cfg), "mode": mode, "Xn0": (X0 - mu_x) / sd_x, "mu_x": mu_x, "sd_x": sd_x, "y": y, "y0": float(y.mean()),
            "s1": s1, "kern": kern, "nk": nk, "shift": shift, "names": names + [{"th": thn, "tv": tvn}[s] for s in shift],
            "lo": lo, "hi": hi, "cols": layout(key)[0]}


def split_phi(phi, sp):
    """φ → (カーネルの theta, θ_h, θ_v)。A は θ_h = 0、C は両方 0"""
    nk = sp["nk"]
    th = float(phi[nk]) if sp["shift"] == ["th", "tv"] else 0.0
    tv = float(phi[-1]) if sp["shift"] else 0.0
    return np.asarray(phi[:nk], float), th, tv


def shifted_xy(sp, th, tv):
    """係数 θ でずらした、標準化した入力と、中心化した出力（fit_gp の gp.fit に渡すものと同じ）"""
    Xn = sp["Xn0"]
    if th != 0.0:
        Xn = Xn.copy()
        Xn[:, 0] = Xn[:, 0] + th * sp["s1"] / sp["sd_x"][0]
    return Xn, sp["y"] - sp["y0"] - tv * sp["s1"]


def log_post(phi, sp):
    """対数事後確率（定数を除く）。範囲の外は −inf、中は対数周辺尤度（GaussianProcessRegressor.log_marginal_likelihood と同じ式）"""
    phi = np.asarray(phi, float)
    if np.any(phi <= sp["lo"]) or np.any(phi >= sp["hi"]):
        return -np.inf
    kt, th, tv = split_phi(phi, sp)
    Xn, yt = shifted_xy(sp, th, tv)
    K = sp["kern"].clone_with_theta(kt)(Xn)
    K[np.diag_indices_from(K)] += GP_ALPHA
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return -np.inf
    a = cho_solve((L, True), yt, check_finite=False)
    return float(-0.5 * yt @ a - np.log(np.diag(L)).sum() - 0.5 * len(yt) * np.log(2 * np.pi))


# ---------------------------------------------------------------- 適応型 RWM
def run_chain(sp, phi0, prop_sd, n_warm, n_samp, seed, target_acc=None, state=None):
    """1 チェーン（step5b_core.run_chain5 と同じ方式）。state を渡すと、調整をせずに、その続きから本番だけを n_samp 回足す。
    返す値：本番のサンプル（draws）と対数事後確率（lp）、調整期間の対数事後確率（warm_lp）、採択率、続きを回すための状態（state）"""
    import warnings
    warnings.simplefilter("ignore")
    target_acc = MCMC_TARGET_ACC if target_acc is None else target_acc
    rng = np.random.default_rng(seed)
    if state is None:
        th = np.array(phi0, dtype=float)
        lp = log_post(th, sp)
        cov, log_lam = np.diag(np.asarray(prop_sd, float) ** 2), 0.0
    else:
        th, lp, cov, log_lam = np.array(state["th"]), state["lp"], np.array(state["cov"]), state["log_lam"]
        rng.bit_generator.state = state["rng"]
        n_warm = 0
    k = len(th)
    warm = np.empty((n_warm, k))
    warm_lp = np.empty(n_warm)
    draws, lps = np.empty((n_samp, k)), np.empty(n_samp)
    acc = np.zeros(n_warm + n_samp, dtype=bool)
    t0 = time.time()
    L = np.linalg.cholesky(np.exp(log_lam) * cov)
    for it in range(n_warm + n_samp):
        if it < n_warm:
            L = np.linalg.cholesky(np.exp(log_lam) * cov)
        prop = th + L @ rng.standard_normal(k)
        lp_prop = log_post(prop, sp)
        a = np.exp(min(0.0, lp_prop - lp)) if np.isfinite(lp_prop) else 0.0
        if rng.uniform() < a:
            th, lp, acc[it] = prop, lp_prop, True
        if it < n_warm:
            warm[it], warm_lp[it] = th, lp
            log_lam += (it + 1) ** -0.6 * (a - target_acc)
            if it >= 199 and (it + 1) % 50 == 0 and it < 0.75 * n_warm:
                cov = np.cov(warm[(it + 1) // 2: it + 1].T) * (2.38 ** 2 / k) + np.eye(k) * 1e-10
            if it == n_warm - 1:
                L = np.linalg.cholesky(np.exp(log_lam) * cov)   # 本番の提案（固定）
        else:
            draws[it - n_warm], lps[it - n_warm] = th, lp
    return {"draws": draws, "lp": lps, "warm_lp": warm_lp, "acc_warm": float(acc[:n_warm].mean()) if n_warm else np.nan,
            "acc_samp": float(acc[n_warm:].mean()), "seconds": time.time() - t0,
            "state": {"th": th, "lp": lp, "cov": cov, "log_lam": log_lam, "rng": rng.bit_generator.state}}


def init_points(sp, tr, n_chains, seed):
    """チェーンごとの初期値。MCMC_INIT が "mle" なら、係数 0 で最尤推定（L-BFGS-B。勾配法は出発点を決めるためだけに使う）したカーネルに、
    "default" ならカーネルの既定値に、ずれ（カーネルの対数は MCMC_INIT_SD[0]、係数は [1]。ノイズにはずれを足さない）を足す。範囲の内側に切り詰める"""
    if MCMC_INIT == "mle":
        base = fit_gp(sp["key"], tr, sp["cfg"], 0.0, 0.0, MCMC_INIT_RESTARTS, None, False, sp["mode"])["gp"].kernel_.theta
    elif MCMC_INIT == "default":
        base = sp["kern"].theta
    else:
        raise ValueError(f"MCMC_INIT は \"mle\" か \"default\" です（指定：{MCMC_INIT}）")
    rng = np.random.default_rng(seed)
    nk, ns = sp["nk"], len(sp["shift"])
    pts = []
    for c in range(n_chains):
        jit = rng.normal(0.0, MCMC_INIT_SD[0], nk)
        jit[sp["names"].index("ln_noise")] = 0.0
        p = np.r_[base + jit, rng.normal(0.0, MCMC_INIT_SD[1], ns)]
        pts.append(np.clip(p, sp["lo"] + 1e-6, sp["hi"] - 1e-6))
    return pts


def diag_table(draws, names):
    """収束の診断（ArviZ。本体の手順5b の diag_summary と同じ）：パラメータごとの R-hat、有効サンプル数（bulk・tail）。draws は (チェーン, 反復, パラメータ)"""
    import arviz as az
    post = {f"p{i}": draws[:, :, i] for i in range(draws.shape[2])}
    t = az.summary(az.from_dict({"posterior": post}), kind="diagnostics", round_to="none")
    t = t.reindex([f"p{i}" for i in range(draws.shape[2])])
    t.index = names
    return t[["r_hat", "ess_bulk", "ess_tail"]]


def run_mcmc(sp, tr, seed, n_jobs=None, log=print):
    """チェーンを並列に回し、収束の基準（R-hat ≤ MCMC_RHAT、ESS(bulk) ≥ MCMC_ESS）に届かなければ、本番を続きから足す（上限 MCMC_MAX_SAMPLES）"""
    from joblib import Parallel, delayed
    n_jobs = min(N_JOBS if n_jobs is None else n_jobs, MCMC_CHAINS)
    t0 = time.time()
    init = init_points(sp, tr, MCMC_CHAINS, seed)
    psd = np.r_[np.full(sp["nk"], MCMC_PROP_SD[0]), np.full(len(sp["shift"]), MCMC_PROP_SD[1])]
    ch = Parallel(n_jobs=n_jobs)(delayed(run_chain)(sp, p0, psd, MCMC_WARM, MCMC_SAMPLES, seed + 1000 + c) for c, p0 in enumerate(init))
    n_add = 0
    while True:
        draws = np.stack([c["draws"] for c in ch])
        dg = diag_table(draws, sp["names"])
        ok = bool(dg["r_hat"].max() <= MCMC_RHAT and dg["ess_bulk"].min() >= MCMC_ESS)
        n = draws.shape[1]
        if ok or n >= MCMC_MAX_SAMPLES:
            break
        more = int(min(MCMC_SAMPLES, MCMC_MAX_SAMPLES - n))
        log(f"  収束の基準に届かない（R-hat 最大 {dg['r_hat'].max():.3f}、ESS 最小 {dg['ess_bulk'].min():.0f}）ので、本番を {more} 回足す（いま {n} 回）")
        add = Parallel(n_jobs=n_jobs)(delayed(run_chain)(sp, None, None, 0, more, seed + 2000 + 100 * n_add + i, state=c["state"]) for i, c in enumerate(ch))
        for c, a in zip(ch, add):
            c["draws"] = np.vstack([c["draws"], a["draws"]])
            c["lp"] = np.r_[c["lp"], a["lp"]]
            c["acc_samp"] = (c["acc_samp"] * (n) + a["acc_samp"] * more) / (n + more)
            c["seconds"] += a["seconds"]
            c["state"] = a["state"]
        n_add += 1
    return {"draws": draws, "lp": np.stack([c["lp"] for c in ch]), "warm_lp": np.stack([c["warm_lp"] for c in ch]),
            "acc_warm": np.array([c["acc_warm"] for c in ch]), "acc_samp": np.array([c["acc_samp"] for c in ch]),
            "chain_seconds": np.array([c["seconds"] for c in ch]), "seconds": time.time() - t0, "n_samples": int(draws.shape[1]),
            "n_extend": n_add, "converged": ok, "diag": dg, "init": np.array(init)}


# ---------------------------------------------------------------- 事後分布の要約
def post_summary(sp, res):
    """パラメータごとの事後の要約（中央値・2.5%・97.5%・平均・標準偏差・範囲の端への張り付き）と、係数の支持の判定。
    カーネルは対数のまま（列 scale = "ln"）と、元の値（exp。振幅・線形項の c とノイズは分散の値）も出す"""
    flat = res["draws"].reshape(-1, res["draws"].shape[2])
    rows = []
    for i, nm in enumerate(sp["names"]):
        v = flat[:, i]
        q = np.percentile(v, [2.5, 50, 97.5])
        w = sp["hi"][i] - sp["lo"][i]
        row = {"パラメータ": nm, "中央値": q[1], "2.5%": q[0], "97.5%": q[2], "平均": v.mean(), "標準偏差": v.std(),
               "事前分布の下限": sp["lo"][i], "事前分布の上限": sp["hi"][i],
               "範囲の端に届く": bool(q[0] < sp["lo"][i] + 0.05 * w or q[2] > sp["hi"][i] - 0.05 * w)}
        if i < sp["nk"]:
            row.update({"中央値（元の値）": float(np.exp(q[1])), "2.5%（元の値）": float(np.exp(q[0])), "97.5%（元の値）": float(np.exp(q[2]))})
        else:
            row["P(>0)"] = float((v > 0).mean())
        rows.append(row)
    tab = pd.DataFrame(rows)
    out = {"table": tab}
    if sp["shift"]:
        tv = flat[:, -1]
        th = flat[:, sp["nk"]] if sp["shift"] == ["th", "tv"] else np.zeros(len(tv))
        pct = lambda v: np.percentile(v, [2.5, 50, 97.5])   # noqa: E731
        (tvl, tvm, tvh), (thl, thm, thh), (dl, dm, dh) = pct(tv), pct(th), pct(tv - th)
        out.update({"th_best": thm, "th_lo": thl, "th_hi": thh, "tv_best": tvm, "tv_lo": tvl, "tv_hi": tvh,
                    "diff_best": dm, "diff_lo": dl, "diff_hi": dh, "P_tv_pos": float((tv > 0).mean())})
        if sp["shift"] == ["th", "tv"]:                       # 2 つの係数を合わせて：(0, 0) が事後分布の 95% の楕円の外か（マハラノビス距離² > 5.99）
            m = np.array([th.mean(), tv.mean()])
            d2 = float(m @ np.linalg.solve(np.cov(np.vstack([th, tv])), m))
            out.update({"d2_zero": d2, "supported": d2 > 5.991, "support_rule": f"({sp['names'][sp['nk']]}, {sp['names'][-1]}) = (0, 0) が 95% の楕円の外（マハラノビス距離² > 5.99）"})
        else:
            out.update({"d2_zero": np.nan, "supported": bool(tvl > 0 or tvh < 0), "support_rule": "縦の係数の 95% 区間が 0 を含まない"})
        idx = [len(sp["names"]) - 1] + ([sp["nk"]] if sp["shift"] == ["th", "tv"] else [])
        out["theta_at_edge"] = bool(tab.loc[idx, "範囲の端に届く"].any())
    return out


# ---------------------------------------------------------------- 最終モデル（事後分布の混合）と予測
def fit_at(sp, phi):
    """パラメータ φ に固定した GPR（最適化しない）。fit_gp と同じ形の辞書を返す"""
    import warnings
    warnings.simplefilter("ignore")
    kt, th, tv = split_phi(phi, sp)
    Xn, yt = shifted_xy(sp, th, tv)
    gp = GaussianProcessRegressor(kernel=sp["kern"].clone_with_theta(kt), optimizer=None, alpha=GP_ALPHA)
    gp.fit(Xn, yt)
    return {"gp": gp, "mu_x": sp["mu_x"], "sd_x": sp["sd_x"], "y0": sp["y0"], "key": sp["key"], "cfg": dict(sp["cfg"]),
            "th": th, "tv": tv, "cols": sp["cols"], "mode": sp["mode"]}


def mixture_fit(sp, res, n_draws=None):
    """最終モデル：代表点（事後サンプルの中で対数事後確率が最大の点）の GPR に、予測に使う事後サンプル（draws。全チェーンから等間隔に n_draws 組）と、
    サンプルごとの GPR を作り直すための材料（mix_base）を足したもの。predict は、draws があれば混合分布で予測する"""
    n_draws = N_PRED_DRAWS if n_draws is None else n_draws
    flat = res["draws"].reshape(-1, res["draws"].shape[2])
    lpf = res["lp"].reshape(-1)
    f = fit_at(sp, flat[int(np.argmax(lpf))])
    f["draws"] = flat[np.linspace(0, len(flat) - 1, min(n_draws, len(flat))).astype(int)]
    f["mix_base"] = {k: sp[k] for k in ("Xn0", "y", "y0", "s1", "sd_x", "mu_x", "nk", "shift", "names", "lo", "hi", "kern", "key", "cfg", "mode", "cols")}
    f["rep_phi"] = flat[int(np.argmax(lpf))]
    f["rep_lp"] = float(lpf.max())
    return f


_predict_point = globals().get("_predict_point", predict)   # 共通の部品の predict（1 点の予測。このセルを実行し直しても、包んだものを包まない）
_MIX_CACHE = {}
MIX_CACHE_MAX = 6                                            # 覚えておくモデルの数の上限（1 つで、学習データ 460 行・200 組なら約 0.35 GB）


def mixture_members(fit):
    """混合分布の成分（事後サンプルごとの GPR）。作ったものはメモリに覚えておく（学習済みモデルの joblib には入れない）。
    覚える名前は、事後サンプルと学習データの中身の指紋（同じ中身なら作り直さない。MCMC のセルを実行し直して中身が変われば、作り直す）"""
    import hashlib
    h = hashlib.sha1()
    for a in (fit["draws"], fit["mix_base"]["Xn0"], fit["mix_base"]["y"], fit["mix_base"]["s1"]):
        h.update(np.ascontiguousarray(a, dtype=float).tobytes())
    k = h.hexdigest()
    if k not in _MIX_CACHE:
        while len(_MIX_CACHE) >= MIX_CACHE_MAX:              # 上限なら、いちばん古いものを忘れる
            _MIX_CACHE.pop(next(iter(_MIX_CACHE)))
        _MIX_CACHE[k] = [fit_at(fit["mix_base"], phi) for phi in fit["draws"]]
    return _MIX_CACHE[k]


def predict(fit, df):
    """予測平均（ln η）、予測標準偏差（ノイズ込み）、潜在関数の標準偏差、ノイズの標準偏差。
    fit に事後サンプル（draws）があれば、サンプルごとの予測を等しい重みで混ぜる：平均 = mean(μ_i)、分散 = mean(σ_i² + μ_i²) − 平均²。
    潜在関数の分散も同じように混ぜ、ノイズは分散の平均。なければ共通の部品の predict と同じ（1 点の予測）"""
    if "draws" not in fit:
        return _predict_point(fit, df)
    out = [_predict_point(f, df) for f in mixture_members(fit)]
    mus = np.array([o[0] for o in out])
    sds = np.array([o[1] for o in out])
    lat = np.array([o[2] for o in out])
    noise = np.array([o[3] for o in out])
    mu = mus.mean(axis=0)
    var = (sds ** 2 + mus ** 2).mean(axis=0) - mu ** 2
    var_lat = (lat ** 2 + mus ** 2).mean(axis=0) - mu ** 2
    return mu, np.sqrt(np.maximum(var, 0.0)), np.sqrt(np.maximum(var_lat, 0.0)), float(np.sqrt((noise ** 2).mean()))
