# -*- coding: utf-8 -*-
"""手順5b（最終候補の MCMC）の部品。ノートのセルに、この中身がそのまま入る（build_step5b_mcmc_nb.py）。
nb_common.py・step5_core.py の部品と、ノートの設定（SEED など）を前提にする。

考え方：入力が条件平均なので、同じ条件のショットは入力が同じ。ショットごとの尤度は、条件ごとの平均・ショット数・条件内の平方和を使って、
厳密に「条件内のばらつきの項」と「条件平均の GP の項」に分けられる（条件数の大きさの行列だけで済むので、MCMC の 1 回の評価が速い）。
  log p(y) = −(N−m)/2·log(2π σn²) − ½ Σ log nᵢ − Σ SSᵢ/(2σn²) ＋ log N(ȳ | 0, K ＋ diag(σn²/nᵢ))
（N：ショット数、m：条件数、nᵢ：条件 i のショット数、SSᵢ：条件内の平方和、K：ノイズなしのカーネル行列）
"""
from scipy.linalg import LinAlgError, cholesky
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

E_NAMES = {"B": ["Eh", "Ev"], "A": ["Ev"], "C": []}
TARGET_ACC = 0.234


def theta_names5(struct, hv):
    """MCMC で動かすパラメータの名前（カーネルの対数パラメータ → ずらし量 E）。順番は kernel5 の theta と同じ"""
    cols, m, c = layout5(struct, hv)
    names = []
    if m is None:
        names += ["log_c"] + [f"log_ls_{x}" for x in cols] + ["log_c_lin", "log_sigma0"]
    else:
        for part, idx in (("m", m), ("c", c)):
            names += [f"log_c_{part}"] + [f"log_ls_{part}_{cols[i]}" for i in idx] + [f"log_c_lin_{part}", f"log_sigma0_{part}"]
    names += ["log_noise"] + E_NAMES[struct]
    assert len(names) == len(kernel5(struct, hv).theta) + len(E_NAMES[struct]), "パラメータの名前の数が合わない"
    return names


def model_v(struct, hv, phi):
    """パラメータのベクトル phi（カーネルの対数パラメータ → E）から、構造の指定 v を作る"""
    ne = len(E_NAMES[struct])
    ev_ = list(phi[len(phi) - ne:]) if ne else []
    named = dict(zip(E_NAMES[struct], ev_))
    return {"struct": struct, "hv": hv, "pcol": "ln_P32_cm", "Eh": float(named.get("Eh", 0.0)), "Ev": float(named.get("Ev", 0.0))}


def make_grouped(tr, struct, hv):
    """学習データ（ショットの表）から、条件ごとにまとめた量を作る。標準化の定数と y0 は、手順5と同じくショットの行から計算する"""
    ct = cond_table5(tr)
    dev = tr["ln_eta"] - tr.groupby("cond")["ln_eta"].transform("mean")
    ss = (dev ** 2).groupby(tr["cond"], sort=False).sum().reindex(ct["cond"]).to_numpy()
    X0 = design5(tr, {"struct": struct, "hv": hv, "pcol": "ln_P32_cm", "Eh": 0.0, "Ev": 0.0})[0]
    return {"ct": ct, "n": ct["n"].to_numpy().astype(float), "ss": ss, "y0": float(tr["ln_eta"].mean()), "mu_x": X0.mean(axis=0), "sd_x": X0.std(axis=0),
            "N": int(len(tr)), "m": int(len(ct))}


def _train_parts(gd, v):
    X0, Xs, ysh = design5(gd["ct"], v)
    return (Xs - gd["mu_x"]) / gd["sd_x"], gd["ct"]["ln_eta"].to_numpy() - ysh - gd["y0"]


def grouped_lml(gd, v, kern, theta_k):
    """条件ごとにまとめた対数尤度（ショットごとの GPR の対数周辺尤度と一致する）。kern：kernel5 で作ったカーネル（theta を書き換える）"""
    kern.theta = theta_k
    Xt, yb = _train_parts(gd, v)
    noise = kern.k2.noise_level
    K = kern.k1(Xt)
    K[np.diag_indices_from(K)] += noise / gd["n"]
    try:
        L = cholesky(K, lower=True)
    except LinAlgError:
        return -np.inf
    alpha = cho_solve((L, True), yb)
    ll_cond = -0.5 * float(yb @ alpha) - float(np.log(np.diag(L)).sum()) - 0.5 * gd["m"] * np.log(2.0 * np.pi)
    ll_within = -0.5 * (gd["N"] - gd["m"]) * np.log(2.0 * np.pi * noise) - 0.5 * float(np.log(gd["n"]).sum()) - 0.5 * float(gd["ss"].sum()) / noise
    return ll_cond + ll_within


def grouped_predict(gd, struct, hv, phi, te, kern=None):
    """予測平均、予測標準偏差（ノイズ込み）、潜在関数の標準偏差。条件平均だけで厳密に計算できる（ショットごとの GPR の予測と一致する）"""
    kern = kern or kernel5(struct, hv)
    nk = len(kern.theta)
    kern.theta = phi[:nk]
    v = model_v(struct, hv, phi)
    Xt, yb = _train_parts(gd, v)
    noise = kern.k2.noise_level
    K = kern.k1(Xt)
    K[np.diag_indices_from(K)] += noise / gd["n"]
    L = cholesky(K, lower=True)
    alpha = cho_solve((L, True), yb)
    X0, Xs, yshT = design5(te, v)
    Xq = (Xs - gd["mu_x"]) / gd["sd_x"]
    Ks = kern.k1(Xq, Xt)
    var_f = np.maximum(kern.k1.diag(Xq) - np.einsum("ij,ji->i", Ks, cho_solve((L, True), Ks.T)), 0.0)
    return Ks @ alpha + gd["y0"] + yshT, np.sqrt(var_f + noise), np.sqrt(var_f)


def predict_mixture(gd, struct, hv, phis, te):
    """事後サンプル phis（行ごとに 1 サンプル）の予測分布を、等しい重みで混ぜる。戻り値：(成分ごとの平均, 成分ごとの標準偏差〔ノイズ込み〕, 混合の平均, 混合の標準偏差, 混合の潜在関数の標準偏差)"""
    kern = kernel5(struct, hv)
    out = [grouped_predict(gd, struct, hv, ph, te, kern) for ph in np.atleast_2d(phis)]
    mus, sds, sdf = (np.array([o[i] for o in out]) for i in range(3))
    mu = mus.mean(axis=0)
    return mus, sds, mu, np.sqrt((sds ** 2 + mus ** 2).mean(axis=0) - mu ** 2), np.sqrt((sdf ** 2 + mus ** 2).mean(axis=0) - mu ** 2)


def make_logpost(gd, struct, hv, e_prior):
    """対数事後密度の関数と、パラメータの範囲（事前分布）。範囲の中は一様（カーネルは対数スケール、E はそのまま）なので、範囲内では対数尤度そのもの"""
    kern = kernel5(struct, hv)
    nk = len(kern.theta)
    lo = np.r_[kern.bounds[:, 0], [e_prior[n][0] for n in E_NAMES[struct]]]
    hi = np.r_[kern.bounds[:, 1], [e_prior[n][1] for n in E_NAMES[struct]]]

    def logpost(phi):
        if np.any(phi <= lo) or np.any(phi >= hi):
            return -np.inf
        return grouped_lml(gd, model_v(struct, hv, phi), kern, phi[:nk])

    return logpost, lo, hi


def fit_mle5(gd, struct, hv, e_prior, e0, e_fixed, n_starts=8, seed=0):
    """最尤推定（同じ尤度を scipy の L-BFGS-B で最大化）。e_fixed が True なら E を e0 に固定してカーネルだけ、False なら E も動かす（e0 から出発）。"""
    logpost, lo, hi = make_logpost(gd, struct, hv, e_prior)
    ne = len(E_NAMES[struct])
    nk = len(lo) - ne
    rng = np.random.default_rng(seed)
    base = kernel5(struct, hv).theta
    best_x, best_f = None, np.inf
    free = np.arange(nk) if (e_fixed or ne == 0) else np.arange(nk + ne)

    def neg(x):
        phi = np.r_[x[:nk], np.asarray(e0, float)] if (e_fixed and ne) else np.array(x, float)
        v = logpost(phi)
        return -v if np.isfinite(v) else 1e10

    bnd = [(lo[i] + 1e-6, hi[i] - 1e-6) for i in free]
    for s in range(n_starts):
        th = np.clip(base if s == 0 else rng.uniform(lo[:nk] + 0.5, hi[:nk] - 0.5), lo[:nk] + 1e-3, hi[:nk] - 1e-3)
        e_start = np.asarray(e0, float) if (s == 0 or ne == 0) else np.asarray(e0, float) + rng.normal(0, 1.0, ne)
        x0 = th if (e_fixed or ne == 0) else np.r_[th, np.clip(e_start, lo[nk:] + 1e-3, hi[nk:] - 1e-3)]
        r = minimize(neg, x0, method="L-BFGS-B", bounds=bnd)
        if r.fun < best_f:
            best_x, best_f = r.x, r.fun
    phi = np.r_[best_x[:nk], np.asarray(e0, float)] if (e_fixed and ne) else np.array(best_x, float)
    return phi, -best_f


def run_chain5(gd, struct, hv, e_prior, phi0, init_sd, n_warm, n_samp, seed):
    """適応型ランダムウォーク・メトロポリス法の 1 チェーン（手順3c と同じ。調整中：提案 N(φ, λΣ) の Σ を足跡から学び、λ を採択率が 0.234 に近づくよう合わせる。本番：固定）"""
    rng = np.random.default_rng(seed)
    logpost, lo, hi = make_logpost(gd, struct, hv, e_prior)
    k = len(phi0)
    th = np.array(phi0, dtype=float)
    lp = logpost(th)
    cov, log_lam = np.diag(np.asarray(init_sd, float) ** 2), 0.0
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
        if it < n_warm:
            warm[it] = th
            log_lam += (it + 1) ** -0.6 * (a - TARGET_ACC)
            if it >= 199 and (it + 1) % 50 == 0 and it < 0.75 * n_warm:
                cov = np.cov(warm[(it + 1) // 2: it + 1].T) * (2.38 ** 2 / k) + np.eye(k) * 1e-10
        else:
            draws[it - n_warm], lps[it - n_warm] = th, lp
    return {"draws": draws, "lp": lps, "accepted": acc[n_warm:], "acc_warm": acc[:n_warm].mean(), "acc_samp": acc[n_warm:].mean(), "lam": np.exp(log_lam), "seconds": time.time() - t0}


def evaluate_mixture(y, mus, sds):
    """正規分布を等しい重みで混ぜた予測分布での指標。mus, sds は (成分数, 点の数)。成分が 1 つなら evaluate と同じ"""
    mus, sds = np.atleast_2d(mus), np.atleast_2d(sds)
    mu = mus.mean(axis=0)
    cdf = norm.cdf((y - mus) / sds).mean(axis=0)
    log_dens = logsumexp(norm.logpdf(y, mus, sds), axis=0) - np.log(len(mus))
    sd_mix = np.sqrt((sds ** 2 + mus ** 2).mean(axis=0) - mu ** 2)
    return {"R2": float(r2_score(y, mu)) if len(y) > 1 else np.nan, "RMSE": float(np.sqrt(mean_squared_error(y, mu))),
            "coverage95": float(np.mean((cdf >= 0.025) & (cdf <= 0.975))), "NLPD": float(-np.mean(log_dens)), "mean_sd": float(np.mean(sd_mix))}


def to_arrays(trace, cols):
    """サンプルの表 → {パラメータ: (チェーン数, 反復数) の配列}"""
    t = trace.sort_values(["chain", "draw"])
    n_chain = t["chain"].nunique()
    return {c: t[c].to_numpy().reshape(n_chain, -1) for c in cols}


def diag_summary(trace, cols):
    """ArviZ による収束診断の表（R-hat、ESS の bulk・tail）"""
    import arviz as az
    return az.summary(az.from_dict({"posterior": to_arrays(trace, cols)}), kind="diagnostics", round_to="none")


def thin_draws(trace, cols, n):
    """全チェーンのサンプルから、等間隔に n 個選んだパラメータの行列（行ごとに 1 サンプル）"""
    t = trace.sort_values(["chain", "draw"])
    a = t[cols].to_numpy()
    return a[np.linspace(0, len(a) - 1, min(n, len(a))).astype(int)]
