# -*- coding: utf-8 -*-
"""手順5（シフト構造の本実装）の部品。ノートのセルに、この中身がそのまま入る（build_step5_nb.py）。
nb_common.py の部品（kernel_for、OnDims、StandardScaler など）と、ノートの設定（SEED、NOISE_LB など）を前提にする。

モデルの記号は「構造_扱い」（例：B_1、A_2、C_4）。
- 構造：B（横・縦を別々にシフト）、A（縦だけ）、C（構造なし。Tset を入力に並べた通常の GPR）
- H の扱い："1" H（Fo）を残す、"2" 材料の部分 m ＋ 厚み・冷却の補正 c(Fo) に分ける、"4" P3_2 を入れる（H なし）、"14" (1)＋(4)、"24" (2)＋(4)
"""
import time as _time

from scipy.linalg import cho_solve

R_GAS, TREF_K = 8.314, 483.15                     # 気体定数 [J/mol/K]、基準温度 210 ℃ [K]
HV_JA = {"1": "(1) Hを残す", "2": "(2) m＋c(Fo)に分ける", "4": "(4) P3_2（Hなし）", "14": "(1+4)", "24": "(2+4)"}
STRUCT_JA5 = {"B": "B（横・縦を別々）", "A": "A（縦だけ）", "C": "C（構造なし）"}
COL_JA = {"u": "ln γ̇", "H": "ln H", "T": "Tset", "w": "ln Fo", "p": "ln P3_2"}


def shift_ln(Tk, E):
    """ln a(T; E) = (E/R)(1/T − 1/Tref)。E は kJ/mol、Tk は K"""
    return (E * 1000.0 / R_GAS) * (1.0 / np.asarray(Tk) - 1.0 / TREF_K)


def has_p(hv):
    return "4" in hv


def layout5(struct, hv):
    """入力の列の並び cols と、(2) の足し算の分け方 (m の列番号, c の列番号)（足し算でなければ None, None）。
    列の記号：u＝ln γ̇（B・A はシフト後）、H＝ln H、T＝Tset、w＝ln Fo、p＝ln P3_2（条件平均）"""
    if struct == "C":
        base = {"1": ["u", "H", "T"], "2": ["u", "T", "w"], "4": ["u", "T", "p"], "14": ["u", "H", "T", "p"], "24": ["u", "T", "p", "w"]}
    else:
        base = {"1": ["u", "w"], "2": ["u", "w"], "4": ["u", "p"], "14": ["u", "w", "p"], "24": ["u", "p", "w"]}
    cols = base[hv]
    if hv in ("2", "24"):
        return cols, [i for i, c in enumerate(cols) if c != "w"], [cols.index("w")]
    return cols, None, None


def kernel5(struct, hv):
    """(1)(4)(1+4)：ARD Matern 5/2 ＋ 線形 ＋ ノイズ（kernel_for と同じ）。(2)(2+4)：m の部分と c の部分を足し算（それぞれ Matern ＋ 線形）＋ ノイズ"""
    cols, m, c = layout5(struct, hv)
    if m is None:
        return kernel_for(len(cols), True)

    def part(d):
        return C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0] * d, length_scale_bounds=(1e-2, 1e2), nu=2.5) + C(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-5, 1e3))

    return OnDims(part(len(m)), m, "m") + OnDims(part(len(c)), c, "c") + WhiteKernel(noise_level=0.1, noise_level_bounds=(NOISE_LB, 1e1))


def design5(df, v):
    """データ → (標準化の定数を決めるための入力 X0、実際の入力 Xs、出力のシフト量 ysh)。v：{"struct", "hv", "Eh", "Ev", "pcol"}。
    入力の γ̇ は条件平均（ln_gd_cm）。B・A は u = ln γ̇ + ln a(Tset; E_h)、ysh = ln a(Tset; E_v)。P3_2 は pcol の列（既定 ln_P32_cm）"""
    cols = layout5(v["struct"], v["hv"])[0]
    lng, lnH, T = df["ln_gd_cm"].to_numpy(), df["ln_H"].to_numpy(), df["Tset"].to_numpy()
    raw = {"u": lng, "H": lnH, "T": T, "w": -(lng + 3.0 * lnH)}            # ln Fo = −(ln γ̇ + 3 ln H) + 定数（シフト前の γ̇ で計算）
    if has_p(v["hv"]):
        raw["p"] = df[v.get("pcol", "ln_P32_cm")].to_numpy()
    shifted = dict(raw)
    ysh = np.zeros(len(df))
    if v["struct"] != "C":
        Tk = T + 273.15
        shifted["u"] = lng + shift_ln(Tk, v["Eh"])
        ysh = shift_ln(Tk, v["Ev"])
    return np.column_stack([raw[c] for c in cols]), np.column_stack([shifted[c] for c in cols]), ysh


def fit5(v, tr, restarts, theta0=None):
    """学習。標準化・中心化は E に依存しない定数（シフト前の入力の平均・標準偏差、ln η の平均）。theta0：カーネルの初期値（前の格子点の解）"""
    X0, Xs, ysh = design5(tr, v)
    mu_x, sd_x = X0.mean(axis=0), X0.std(axis=0)
    y = tr["ln_eta"].to_numpy()
    y0 = float(y.mean())
    k = kernel5(v["struct"], v["hv"])
    if theta0 is not None:
        k = k.clone_with_theta(theta0)
    gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=restarts, random_state=SEED).fit((Xs - mu_x) / sd_x, y - ysh - y0)
    return {"gp": gp, "mu_x": mu_x, "sd_x": sd_x, "y0": y0, "v": v}


def rebuild5(v, tr, theta):
    """学習済みのカーネルの値 theta から、学習をやり直さずに予測器を作り直す（最適化なし。並列の仕事から theta だけを受け取る）"""
    X0, Xs, ysh = design5(tr, v)
    mu_x, sd_x = X0.mean(axis=0), X0.std(axis=0)
    y = tr["ln_eta"].to_numpy()
    y0 = float(y.mean())
    gp = GaussianProcessRegressor(kernel=kernel5(v["struct"], v["hv"]).clone_with_theta(theta), optimizer=None).fit((Xs - mu_x) / sd_x, y - ysh - y0)
    return {"gp": gp, "mu_x": mu_x, "sd_x": sd_x, "y0": y0, "v": v}


def predict5(fit, df):
    """予測平均、予測標準偏差（ノイズ込み）、潜在関数の標準偏差、ノイズの標準偏差"""
    X0, Xs, ysh = design5(df, fit["v"])
    mu, sd = fit["gp"].predict((Xs - fit["mu_x"]) / fit["sd_x"], return_std=True)
    noise = fit["gp"].kernel_.k2.noise_level
    return mu + fit["y0"] + ysh, sd, np.sqrt(np.maximum(sd ** 2 - noise, 0.0)), float(np.sqrt(noise))


def component_predict(fit, df, part):
    """足し算のモデル (2)(2+4) の成分 part（"m" か "c"）の事後平均と事後標準偏差（ln η の単位。定数は m と c の間で入れ替えられる）"""
    gp = fit["gp"]
    k = gp.kernel_
    kc = k.k1.k1 if part == "m" else k.k1.k2
    assert kc.tag == part
    X0, Xs, ysh = design5(df, fit["v"])
    Xt = (Xs - fit["mu_x"]) / fit["sd_x"]
    Ks = kc(Xt, gp.X_train_)
    mean = Ks @ gp.alpha_
    var = kc.diag(Xt) - np.einsum("ij,ji->i", Ks, cho_solve((gp.L_, True), Ks.T))
    return mean, np.sqrt(np.maximum(var, 0.0))


def kernel_row5(fit, tr=None):
    """ハイパーパラメータの一覧（部品ごとの振幅・長さスケール・線形項、ノイズ、対数周辺尤度、下限・上限への張り付き）"""
    v, gp = fit["v"], fit["gp"]
    k = gp.kernel_
    cols = layout5(v["struct"], v["hv"])[0]
    if hasattr(k.k1.k1, "tag"):
        parts = [("m", k.k1.k1.kernel, k.k1.k1.dims), ("c", k.k1.k2.kernel, k.k1.k2.dims)]
    else:
        parts = [("f", k.k1, list(range(len(cols))))]
    row = {"Eh": v.get("Eh", np.nan), "Ev": v.get("Ev", np.nan), "LML": float(gp.log_marginal_likelihood_value_), "noise_var": float(k.k2.noise_level),
           "noise_sd": float(np.sqrt(k.k2.noise_level)), "noise_at_lower_bound": bool(k.k2.noise_level < NOISE_LB * 1.05)}
    at_bound = False
    for name, kk, dims in parts:
        prod, lin = kk.k1, kk.k2
        ls = np.atleast_1d(prod.k2.length_scale)
        at_bound = at_bound or bool(np.any(ls > 99) or np.any(ls < 0.0105))
        row[f"amp_{name}"] = float(np.sqrt(prod.k1.constant_value))
        row[f"lin_amp_{name}"] = float(np.sqrt(lin.k1.constant_value))
        row[f"lin_sigma0_{name}"] = float(lin.k2.sigma_0)
        for d, l in zip(dims, ls):
            row[f"ls_{name}_{cols[d]}"] = float(l)
    row["ls_at_bound"] = at_bound
    return row


def cond_table5(df):
    """条件ごとの平均（ln η、ln γ̇、ln P3_2）。尤度の比較に使う"""
    g = df.groupby("cond", sort=False)
    t = g.agg(Tset=("Tset", "first"), H=("H", "first"), v=("v", "first"), ln_eta=("ln_eta", "mean"), ln_gd=("ln_gd", "mean"), ln_P32=("ln_P32", "mean"),
              n=("ln_eta", "size"), is_sens=("is_sens", "max")).reset_index()
    t["ln_H"], t["ln_gd_cm"], t["ln_P32_cm"] = np.log(t["H"]), t["ln_gd"], t["ln_P32"]
    return t


# ---------------------------------------------------------------- プロファイル尤度
def prof_chain(cdf, base_v, points, restarts):
    """格子の点 points = [(E_h, E_v), …] を順に学習して LML を返す。前の点の解を初期値にする（局所解でがたつくのを減らす）"""
    warnings.simplefilter("ignore")
    out, theta = [], None
    for eh, ev in points:
        f = fit5({**base_v, "Eh": float(eh), "Ev": float(ev)}, cdf, restarts, theta)
        theta = f["gp"].kernel_.theta
        out.append((float(f["gp"].log_marginal_likelihood_value_), float(np.sqrt(f["gp"].kernel_.k2.noise_level))))
    return out


def chains_for(struct, grid, eh_fix=None):
    """構造ごとの格子を、順に計算する点の列（chain）に分ける。A：E_v の 1 列／B：E_h ごとに E_v の列（eh_fix が数なら、E_h をその値に固定した 1 列）"""
    if struct == "A":
        return [[(0.0, float(e)) for e in grid["A"]]]
    if eh_fix is not None:
        return [[(float(eh_fix), float(e)) for e in grid["B_ev"]]]
    return [[(float(eh), float(ev)) for ev in grid["B_ev"]] for eh in grid["B_eh"]]


def profile_best(pdf, key_cols):
    """格子の点の表 → 条件（key_cols）ごとの最良の E と、LML が最大から 1.92 以内の範囲"""
    rows = []
    for key, g in pdf.groupby(key_cols):
        r = g.loc[g["LML"].idxmax()]
        inside = g[g["LML"] >= r["LML"] - 1.92]
        z = g[(g["Eh"] == 0.0) & (g["Ev"] == 0.0)]
        row = dict(zip(key_cols, key))
        diff = inside["Ev"] - inside["Eh"]
        row.update({"Eh_best": float(r["Eh"]), "Ev_best": float(r["Ev"]), "diff_best": float(r["Ev"] - r["Eh"]), "LML_best": float(r["LML"]),
                    "LML_at_E0": float(z["LML"].iloc[0]) if len(z) else np.nan, "n_in_95": int(len(inside)),
                    "Eh_95_lo": float(inside["Eh"].min()), "Eh_95_hi": float(inside["Eh"].max()), "Ev_95_lo": float(inside["Ev"].min()), "Ev_95_hi": float(inside["Ev"].max()),
                    "diff_95_lo": float(diff.min()), "diff_95_hi": float(diff.max()), "n_points": int(len(g)),
                    "at_edge": bool(r["Ev"] in (g["Ev"].min(), g["Ev"].max()) or (g["Eh"].nunique() > 1 and r["Eh"] in (g["Eh"].min(), g["Eh"].max())))})
        row["dLML_vs_E0"] = row["LML_best"] - row["LML_at_E0"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- P3_2 の予測（(4) の「予測した P3_2」）
def p32_model(tr):
    """条件平均の ln P3_2 を (ln γ̇, ln H, Tset) から予測する GPR（学習の条件の点で学習）"""
    ct = cond_table5(tr)
    X = ct[["ln_gd_cm", "ln_H", "Tset"]].to_numpy()
    sc = StandardScaler().fit(X)
    y0 = float(ct["ln_P32_cm"].mean())
    gp = GaussianProcessRegressor(kernel=kernel_for(3, True), n_restarts_optimizer=N_RESTARTS, random_state=SEED).fit(sc.transform(X), ct["ln_P32_cm"].to_numpy() - y0)
    return {"gp": gp, "sc": sc, "y0": y0}


def p32_predict(pm, te):
    """テストの条件ごとに ln P3_2 を予測する。条件ごとの表（cond、実測の条件平均、予測平均、予測標準偏差〔ノイズ込み〕）と、行ごとの条件の番号を返す"""
    ct = cond_table5(te)
    mu, sd = pm["gp"].predict(pm["sc"].transform(ct[["ln_gd_cm", "ln_H", "Tset"]].to_numpy()), return_std=True)
    out = pd.DataFrame({"cond": ct["cond"], "H": ct["H"], "Tset": ct["Tset"], "v": ct["v"], "actual": ct["ln_P32_cm"], "pred": mu + pm["y0"], "pred_sd": sd})
    idx = pd.Series(np.arange(len(ct)), index=ct["cond"]).loc[te["cond"]].to_numpy()
    return out, idx


def predict5_p(fit, te, pm, n_draw, seed):
    """P3_2 を予測して使う場合：P3_2 の予測分布から n_draw 回引き、そのつど ln η を予測して、平均と分散にまとめる（全分散の法則）"""
    ptab, idx = p32_predict(pm, te)
    z = np.random.RandomState(seed).standard_normal((n_draw, len(ptab)))
    mus, vs, vfs = [], [], []
    for d in range(n_draw):
        pdraw = ptab["pred"].to_numpy() + ptab["pred_sd"].to_numpy() * z[d]
        mu, sd, sdf, _ = predict5(fit, te.assign(ln_P32_cm=pdraw[idx]))
        mus.append(mu)
        vs.append(sd ** 2)
        vfs.append(sdf ** 2)
    mus = np.array(mus)
    between = mus.var(axis=0)
    return mus.mean(axis=0), np.sqrt(np.mean(vs, axis=0) + between), np.sqrt(np.mean(vfs, axis=0) + between)


# ---------------------------------------------------------------- 学習と予測の仕事（並列）
def job5(mat, ver, sname, tr, tests, v, restarts, n_draw):
    """1 つのモデル（v）を学習し、tests の各テストを予測する。戻り値は、theta と予測（受け渡しやすいように、モデルそのものは返さない）。
    P3_2 を使うモデルは、実測の P3_2（条件平均）と、予測した P3_2 の両方で評価する"""
    warnings.simplefilter("ignore")
    t0 = _time.time()
    fit = fit5(v, tr, restarts)
    out = {"theta": fit["gp"].kernel_.theta, "kernel": kernel_row5(fit), "preds": {}}
    pm = p32_model(tr) if has_p(v["hv"]) else None
    for tn, te in tests.items():
        mu, sd, sdf, _ = predict5(fit, te)
        out["preds"][(tn, "実測" if pm is not None else "―")] = (mu, sd, sdf)
        if pm is not None:
            out["preds"][(tn, "予測")] = predict5_p(fit, te, pm, n_draw, SEED)
    out["sec"] = _time.time() - t0
    return out
