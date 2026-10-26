# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR の「厚みごとの比較」の部品。

ノート（01_機械学習コードフォルダ/材料別_温度シフトファクター検討/材料別_温度シフトGPR_ARD対応Matern52＋線形＋ホワイトノイズ_厚みごとの比較.ipynb）の
「厚みごとの比較の部品」のセルに、この中身（下の import の行を除く）がそのまま入る。直すときは、このファイルを直して、
ノートを作り直す（04_補助スクリプト/build_mat_thick_nb.py）。ノートでは、この前のセルに mat_common の中身がある。
モジュールとして import しても使える（mat_common の部品を、下の import で読み込む）。

調べること（ユーザー依頼、2026-10-22）：
  1. 厚みごとに、学習済みのモデルで、温度シフトファクターが効いているか（温度の順序、温度による粘度の比 R、換算前後のずれ）
  2. 厚みごとに、シフトに使う温度は、流動中樹脂温度 Tff と加熱シリンダ温度 Tset のどちらが適しているか
     (a) 厚みごとに係数を決め直したときの重ね合わせのずれ、(b) 厚みごとに学習した GPR の対数周辺尤度、(c) 検証データ（3data）の誤差
  3. 厚みごとに学習したモデルと、全厚みの総合データで学習したモデルの、検証データの誤差（学習 5data・検証 3data のまま）
"""
from mat_common import *  # noqa: F401,F403  （ノートに入れるときは、この行を除く）

THICK_KEYS = ["B_Tset", "B_Tff"]                              # シフトに使う温度が違う 2 つのモデル
TEMP_JA = {"B_Tset": "加熱シリンダ温度 Tset", "B_Tff": "流動中樹脂温度 Tff", None: "換算前"}
TEMP_SHORT = {"B_Tset": "Tset", "B_Tff": "Tff"}
A_CLEAR = globals().get("A_CLEAR", 0.5)                       # (a) 重ね合わせのずれ [%] の差が、これ以上なら「違う」（ノートの設定エリアの値があれば、それを使う）
DL_CLEAR = globals().get("DL_CLEAR", 3.0)                     # (b) モデル間の対数周辺尤度の差が、これ以上なら「違う」
ERR_CLEAR = globals().get("ERR_CLEAR", 1.0)                   # (c)・3：予測誤差 [%] の差が、これ以上なら「違う」


def pct_err(rmse):
    """ln η の RMSE → 粘度での誤差 [%]（exp(RMSE) − 1）"""
    return (np.exp(rmse) - 1.0) * 100.0


def thicknesses(df):
    return [float(h) for h in sorted(df["H"].unique())]


def sub_h(df, h):
    return df[np.isclose(df["H"], h)].reset_index(drop=True)


def shot_sd(df):
    """同じ成形条件のショット間の ln η の標準偏差（純粋誤差。条件ごとの分散を、自由度で重み付けしてまとめたもの）。
    比 R の「大きさが一致」の許容差に使う（入力の取り方によらない。ショットごとの入力では GP のノイズが小さく出るため、ノイズは使わない）"""
    g = df.groupby("cond")["ln_eta"]
    dof = g.count() - 1
    return float(np.sqrt((g.var(ddof=1) * dof).sum() / dof.sum()))


# ---------------------------------------------------------------- 重ね合わせのずれ（厚みごと）
def cubic_rmse(t):
    """条件平均の点（x = 換算した ln γ̇、y = 換算した ln η）に 3 次式を当てはめた残差の RMSE（superpose_rmse と同じ。1 つの厚みの分）"""
    if len(t) < 6:
        return np.nan
    c = np.polyfit(t["x"], t["y"], 3)
    return float(np.sqrt(np.mean((t["y"] - np.polyval(c, t["x"])) ** 2)))


def superpose_by_h(t):
    """厚みごとの重ね合わせのずれ：{厚み: (RMSE, 条件数)}（t は shifted_cond_means の表）"""
    return {float(h): (cubic_rmse(g), len(g)) for h, g in t.groupby("H")}


def best_superpose(pool_h, key, cfg, fine=None, max_refine=None):
    """(a) 1 つの厚みのデータ（pool_h。5data＋3data）で、重ね合わせのずれが最小になる係数（横・縦）を、元のノートのプロファイル尤度と同じ格子で探す
    （粗い格子 → 最良の点の周りの細かい格子 → 最良の点が動いたら取り直す）。at_edge：最良の点が粗い格子の端"""
    fine = fine or FINE_B
    max_refine = MAX_REFINE if max_refine is None else max_refine
    hg, vg = grid_of(*cfg["h_grid"]), grid_of(*cfg["v_grid"])
    vals = {}

    def ev(points):
        for th, tv in points:
            k = (round(float(th), 6), round(float(tv), 6))
            if k not in vals:
                vals[k] = cubic_rmse(shifted_cond_means(pool_h, key, cfg, k[0], k[1]))

    ev([(a, b) for a in hg for b in vg])
    for _ in range(max_refine):
        b = min(vals, key=vals.get)
        ev([(a, c) for a in grid_of(b[0] - fine[0], b[0] + fine[0], fine[1]) for c in grid_of(b[1] - fine[0], b[1] + fine[0], fine[1])])
        if min(vals, key=vals.get) == b:
            break
    b = min(vals, key=vals.get)
    return {"th": b[0], "tv": b[1], "rmse": vals[b], "rmse_0": cubic_rmse(shifted_cond_means(pool_h, None, cfg, 0.0, 0.0)),
            "at_edge": bool(b[0] in (hg.min(), hg.max()) or b[1] in (vg.min(), vg.max())), "n_points": len(vals)}


# ---------------------------------------------------------------- 1. 温度シフトの確認（学習済みのモデル）
def cond_pred_table(fit, pool_h):
    """1 つの厚みの、成形条件ごとの平均（実測の ln η、ln γ̇）と、モデルの予測平均の条件ごとの平均（ショットごとの入力なら、ショットの予測の平均）"""
    mu, _, _, _ = predict(fit, pool_h)
    t = pool_h.assign(lp=mu).groupby("cond").agg(Tset=("Tset", "first"), H=("H", "first"), lg=("ln_gd", "mean"), ly=("ln_eta", "mean"),
                                                 lp=("lp", "mean"), grp=("grp", "first")).reset_index()
    return t


def temp_ratio(cm, col):
    """温度による粘度の比 R：最低と最高の加熱シリンダ温度の条件平均を、ln γ̇ の重なる範囲の共通の格子（50 点）で内挿し、
    R = exp(平均[ln η_低温 − ln η_高温])。重なる範囲がなければ NaN。返り値：(R, 重なる範囲の γ̇ の下限, 上限)"""
    lv = sorted(cm["Tset"].unique())
    a = cm[cm["Tset"] == lv[0]].sort_values("lg")
    b = cm[cm["Tset"] == lv[-1]].sort_values("lg")
    x0, x1 = max(a["lg"].min(), b["lg"].min()), min(a["lg"].max(), b["lg"].max())
    if len(a) < 2 or len(b) < 2 or x1 <= x0:
        return np.nan, np.nan, np.nan
    g = np.linspace(x0, x1, 50)
    d = np.interp(g, a["lg"], a[col]) - np.interp(g, b["lg"], b[col])
    return float(np.exp(d.mean())), float(np.exp(x0)), float(np.exp(x1))


def thick_check(fit, pool):
    """1 の表：厚みごとに、温度による粘度の比 R（実測・予測）、温度の順序、学習済みの係数で換算する前後のずれ、判定（○・△・×）。
    ○：順序が一致し、|ln R_予測 − ln R_実測| がショット間の標準偏差（純粋誤差、shot_sd）以内で、換算後のずれ < 換算前。×：予測の順序が実測と逆。△：それ以外"""
    key, cfg = fit["key"], fit["cfg"]
    noise_sd = float(np.sqrt(fit["gp"].kernel_.k2.noise_level))
    tol = shot_sd(pool)
    before = superpose_by_h(shifted_cond_means(pool, None, cfg, 0.0, 0.0))
    after = superpose_by_h(shifted_cond_means(pool, key, cfg, fit["th"], fit["tv"]))
    rows = []
    for h in thicknesses(pool):
        cm = cond_pred_table(fit, sub_h(pool, h))
        r_obs, g0, g1 = temp_ratio(cm, "ly")
        r_pred, _, _ = temp_ratio(cm, "lp")
        order_ok = bool(np.isfinite(r_obs) and np.isfinite(r_pred) and (r_obs > 1) == (r_pred > 1))
        size_ok = bool(np.isfinite(r_obs) and np.isfinite(r_pred) and abs(np.log(r_pred) - np.log(r_obs)) <= tol)
        shift_ok = bool(after[h][0] < before[h][0])
        mark = "○" if (order_ok and size_ok and shift_ok) else ("×" if not order_ok else "△")
        rows.append({"材料": None, "モデル": key, "厚み": h, "条件数": len(cm), "R_実測": r_obs, "R_予測": r_pred,
                     "lnR の差": (np.log(r_pred) - np.log(r_obs)) if np.isfinite(r_obs) and np.isfinite(r_pred) else np.nan, "許容差（ショット間の標準偏差）": tol,
                     "ノイズの標準偏差": noise_sd,
                     "比べた γ̇ の範囲": f"{g0:.0f}〜{g1:.0f}" if np.isfinite(g0) else "なし", "順序が一致": order_ok, "比の大きさが一致": size_ok,
                     "ずれ_換算前": before[h][0], "ずれ_換算後": after[h][0], "換算でずれが減る": shift_ok, "判定": mark})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2(b)・3：厚みごとに学習した GPR
def thick_profile(key, tr, cfg, h, n_jobs=None, restarts=None, mode="shot"):
    """(b) 1 つの厚みの 5data だけで、モデル key の係数をプロファイル尤度で決める（元のノートと同じ run_profile）。返り値：(格子の表, profile_best の辞書)"""
    pts, conv = run_profile(key, sub_h(tr, h), cfg, n_jobs=n_jobs, restarts=restarts, mode=mode)
    return pts, profile_best(pts, key, cfg, conv)


def thick_model(key, tr, te, cfg, h, best=None, restarts=None, mode="shot"):
    """3：1 つの厚みの 5data で最終モデルを学習し（B は best の係数。C はシフトなし）、同じ厚みの 3data を予測する。返り値：(fit, 予測の表)"""
    trh, teh = sub_h(tr, h), sub_h(te, h)
    th, tv, theta0 = (best["th_best"], best["tv_best"], best.get("theta_best")) if best else (0.0, 0.0, None)
    fit = fit_gp(key, trh, cfg, th, tv, restarts=N_RESTARTS if restarts is None else restarts, theta0=theta0, mode=mode)
    mu, sd, sdf, _ = predict(fit, teh)
    return fit, teh.assign(予測_ln_eta=mu, 予測標準偏差=sd, 潜在関数の標準偏差=sdf)


def err_rows(pred, labels):
    """予測の表（ln_eta・予測_ln_eta・予測標準偏差・group）から、全体・低速側・高速側の指標（RMSE、誤差 %、95% 予測区間に入った割合）"""
    rows = []
    y, mu, sd = pred["ln_eta"].to_numpy(), pred["予測_ln_eta"].to_numpy(), pred["予測標準偏差"].to_numpy()
    for rng in ["全体", "低速側", "高速側"]:
        m = np.ones(len(pred), bool) if rng == "全体" else (pred["group"] == rng).to_numpy()
        if m.sum() == 0:
            continue
        e = evaluate(y[m], mu[m], sd[m])
        rows.append({**labels, "範囲": rng, "行数": int(m.sum()), "条件数": int(pred.loc[m, "cond"].nunique()), "RMSE": e["RMSE"],
                     "誤差_%": pct_err(e["RMSE"]), "区間に入った割合": e["coverage95"]})
    return rows


def decide(v_ff, v_set, clear, larger_is_better):
    """2 つの値から、適した温度を「Tff」「Tset」「差なし」で返す"""
    if not (np.isfinite(v_ff) and np.isfinite(v_set)):
        return "—"
    d = (v_ff - v_set) if larger_is_better else (v_set - v_ff)
    return "Tff" if d >= clear else ("Tset" if d <= -clear else "差なし")


def majority(votes):
    """基準ごとの判定の多数決（「—」は数えない）。同数なら「差なし」"""
    v = [x for x in votes if x in ("Tff", "Tset", "差なし")]
    n_ff, n_set = v.count("Tff"), v.count("Tset")
    return "Tff" if n_ff > n_set and n_ff >= len(v) / 2 else ("Tset" if n_set > n_ff and n_set >= len(v) / 2 else "差なし")


# ---------------------------------------------------------------- 図
def _flow_axes(ax, fit, pool_h, train_conds, h, points="shot", extend=1.3, n=200):
    """1 つの厚みの学習結果（設定温度ごとの曲線と 95% 予測区間。入力は flow_inputs で内挿）を ax に描く。
    points："shot"＝ショットごとの計測値の点（既定）／"cond"＝成形条件ごとの平均の点。曲線の範囲は、どちらも条件平均の ln γ̇ の範囲を extend 倍に延ばしたもの"""
    ts = t_style(pool_h["Tset"].unique())
    xs, ys = [], []
    for t, sub in pool_h.groupby("Tset"):
        col, mk = ts[t]
        a = sub.groupby("cond").agg(lg=("ln_gd", "mean"), le=("ln_eta", "mean")).reset_index()
        lg = np.linspace(a["lg"].min() - np.log(extend), a["lg"].max() + np.log(extend), n)
        mu, sd, _, _ = predict(fit, flow_inputs(sub, lg, h, t))
        ax.fill_between(np.exp(lg), np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd), color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(np.exp(lg), np.exp(mu), color=col, lw=2.0, zorder=4, label=f"{int(t)} ℃")
        if points == "cond":
            px, py, trm, ms, lw = np.exp(a["lg"].to_numpy()), np.exp(a["le"].to_numpy()), a["cond"].isin(train_conds).to_numpy(), 55, (0.5, 1.8)
        else:
            px, py, trm, ms, lw = sub["gd"].to_numpy(), sub["eta"].to_numpy(), sub["cond"].isin(train_conds).to_numpy(), 20, (0.4, 1.2)
        ax.scatter(px[trm], py[trm], s=ms, color=col, marker=mk, edgecolors="k", linewidths=lw[0], zorder=6)
        ax.scatter(px[~trm], py[~trm], s=ms, facecolors="white", edgecolors=col, marker=mk, linewidths=lw[1], zorder=6)
        xs += list(np.exp(lg)) + list(px)
        ys += list(py) + list(np.exp(mu))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys) / 1.3, max(ys) * 1.3)
    log_ticks(ax)
    ax.grid(True, which="major", ls="-", lw=0.6, color="0.8")
    ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ (1/s)")


def thick_flow_pair(fits, mat, pool, train_conds, h, out_dir, points="shot"):
    """厚みごとの学習結果を、モデルごと（B_Tset・B_Tff）に左右に並べる。塗り＝学習（5data）、白抜き＝検証（3data）。
    points："shot"＝ショットごとの計測値の点（既定）／"cond"＝成形条件ごとの平均の点"""
    import matplotlib.pyplot as plt
    keys = [k for k in THICK_KEYS if k in fits]
    fig, axes = plt.subplots(1, len(keys), figsize=(7.2 * len(keys), 6.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        _flow_axes(ax, fits[k], sub_h(pool, h), train_conds, h, points=points)
        ax.set_title(f"{k}（{TEMP_JA[k]} でシフト）", fontsize=14)
    axes[0].set_ylabel(r"樹脂粘度 $\eta$ (Pa·s)")
    axes[-1].legend(title="加熱シリンダ温度", loc="upper right", fontsize=11, title_fontsize=11)
    ptxt = "成形条件ごとの平均" if points == "cond" else "ショットごとの計測値"
    fig.suptitle(f"{MAT_NAMES[mat]}　厚み {h:g} mm の学習結果（点：{ptxt}。塗り＝学習、白抜き＝検証）", fontsize=14)
    fig.tight_layout()
    return save_fig(fig, out_dir, f"学習結果_厚み{h:g}mm")


def thick_master(mat, pool, cfg, h, thetas, out_dir, stem):
    """1 つの厚みの重ね合わせ：換算前・加熱シリンダ温度で換算・流動中樹脂温度で換算の 3 枚。thetas：{key: (th, tv)}。破線は当てはめた 3 次式"""
    import matplotlib.pyplot as plt
    ph = sub_h(pool, h)
    ts = t_style(ph["Tset"].unique())
    cases = [(None, 0.0, 0.0)] + [(k, *thetas[k]) for k in THICK_KEYS if k in thetas]
    fig, axes = plt.subplots(1, len(cases), figsize=(5.6 * len(cases), 5.0))
    for ax, (k, th, tv) in zip(axes, cases):
        t = shifted_cond_means(ph, k, cfg, th, tv)
        for T, g in t.groupby("Tset"):
            ax.scatter(np.exp(g["x"]), np.exp(g["y"]), s=50, color=ts[T][0], marker=ts[T][1], edgecolors="k", linewidths=0.5, label=f"{int(T)} ℃", zorder=4)
        xx = np.linspace(t["x"].min(), t["x"].max(), 200)
        if len(t) >= 6:
            ax.plot(np.exp(xx), np.exp(np.polyval(np.polyfit(t["x"], t["y"], 3), xx)), ls="--", color="0.3", lw=1.3, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        log_ticks(ax)
        ax.grid(True, which="major", ls="-", lw=0.6, color="0.8")
        sub = "" if k is None else f"（係数 {th:g}, {tv:g}）"
        ax.set_title(f"{TEMP_JA[k]}{'で換算' if k else ''}{sub}\nずれ {pct_err(cubic_rmse(t)):.1f}%", fontsize=12)
        ax.set_xlabel(r"換算せん断速度 (1/s)" if k else r"せん断速度 $\dot{\gamma}$ (1/s)")
    axes[0].set_ylabel(r"樹脂粘度（換算後は換算粘度）(Pa·s)")
    axes[0].legend(fontsize=10, loc="lower left")
    fig.suptitle(f"{MAT_NAMES[mat]}　厚み {h:g} mm の重ね合わせ", fontsize=13)
    fig.tight_layout()
    return save_fig(fig, out_dir, f"{stem}_厚み{h:g}mm")


def thick_bars(df, value, labels, title, ylabel, out_dir, stem, fmt="{:.1f}", hline=None):
    """厚みごとの棒グラフ：df の列「厚み」ごとに、labels（{列名: 凡例}）の列を並べる"""
    import matplotlib.pyplot as plt
    hs = df["厚み"].tolist()
    cols = list(labels)
    w = 0.8 / len(cols)
    colors = ["#B8C0C8", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    fig, ax = plt.subplots(figsize=(1.6 + 1.9 * len(hs), 4.6))
    x = np.arange(len(hs))
    for i, c in enumerate(cols):
        v = df[c].to_numpy(float)
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, v, w * 0.95, color=colors[i % len(colors)], edgecolor="k", lw=0.5, label=labels[c])
        for xi, vi in zip(x, v):
            if np.isfinite(vi):
                ax.text(xi + (i - (len(cols) - 1) / 2) * w, vi, fmt.format(vi), ha="center", va="bottom", fontsize=9)
    if hline is not None:
        ax.axhline(hline, color="0.4", lw=1.0, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:g} mm" for h in hs])
    ax.set_xlabel("キャビティ厚み")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, ncol=min(len(cols), 3), loc="upper left", bbox_to_anchor=(0, -0.18))
    ax.grid(True, axis="y", lw=0.5, color="0.85")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save_fig(fig, out_dir, stem)
