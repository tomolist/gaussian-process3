# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR の 2 つのノート（学習用・シフトファクター検討用）で、同じになるコードセルの中身（先頭の「# セルN：…」の行を除く）。

build_mat_shift_nb.py（学習用）と build_mat_shiftfactor_nb.py（シフトファクター検討用）が読み、それぞれのセルの番号の行を付けてノートに入れる。
どのセルも、ノートの「モデルと結果の置き場所」のセルで決めた KEYS（学習するモデルの一覧）・SHIFT_KEYS（そのうちシフトのあるモデル）・
MAIN_KEY（図をノートに表示するモデル）と、結果の置き場所の関数 out_dir(mat, key=None, sub=None) を使う。
学習用ノートはモデルが 1 つ（C だけのこともある）、シフトファクター検討用ノートは、シフトのあるモデルと基準の C。どちらでも動くように書いてある。
判定の「節」には、2 つのノートで番号が違うので、節の名前だけを書く（2026-10-27）。
"""

# ---------------------------------------------------------------- ライブラリ、フォント
LIBS = r'''%matplotlib inline
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.tri as mtri
from IPython.display import display

plt.rcParams["figure.dpi"] = 80
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
try:
    fm.findfont("IPAexGothic", fallback_to_default=False)
except ValueError:
    raise RuntimeError("日本語フォント IPAexGothic が見つかりません。インストールしてからカーネルを再起動してください。")
plt.rcParams["font.family"] = ["IPAexGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
print("Python:", sys.executable, "/", sys.version.split()[0])'''

# ---------------------------------------------------------------- ファイルを選んで読み込む
READ = r'''JUDGES.clear()
CHOSEN = choose_inputs(MATERIALS, FILES)            # 材料ごとに、学習データ・検証データの CSV と、学習結果フォルダを決める（設定 FILES に書いていなければ、ダイアログで選ぶ）
display(CHOSEN)
D = load_all(MATERIALS)
INFO = pd.DataFrame([FILE_INFO[(m, g)] for m in MATERIALS for g in ["5data", "3data"]])
display(INFO)
TRAIN, TEST, CFG = {}, {}, {}
for mat in MATERIALS:
    chk_dir = result_dir(mat, "データ確認")         # データの確認の結果（入力の取り方によらないので、学習結果フォルダの直下）
    to_csv(INFO[INFO["材料"] == MAT_NAMES[mat]], os.path.join(chk_dir, "データの概要.csv"))
    TRAIN[mat] = D[(mat, "5data")]
    TEST[mat] = add_groups(TRAIN[mat], D[(mat, "3data")])
    cfg = dict(SHIFT_CFG[mat])
    if cfg.get("Tref") is None:
        lv = sorted(TRAIN[mat]["Tset"].unique())
        cfg["Tref"] = float(lv[len(lv) // 2])
    CFG[mat] = cfg
    rem = pd.concat([REMOVED[(mat, g)].assign(組=g) for g in ["5data", "3data"]], ignore_index=True)
    inv = pd.concat([INVALID[(mat, g)].assign(組=g) for g in ["5data", "3data"]], ignore_index=True)
    to_csv(rem, os.path.join(chk_dir, "一覧で除いた行.csv"))
    grp_txt = "、".join(f"{g} {int((TEST[mat]['group'] == g).sum())} 行" for g in ["低速側", "高速側", "範囲内"] if (TEST[mat]["group"] == g).any())
    print(f"\n【{MAT_NAMES[mat]}】学習 {len(TRAIN[mat])} 行・{TRAIN[mat]['cond'].nunique()} 条件、検証 {len(TEST[mat])} 行・{TEST[mat]['cond'].nunique()} 条件（{grp_txt}）")
    if len(rem):
        print("  問題データ一覧の「除外」で除いた行：" + "、".join(f"{r['元ファイル']}（{r['組']}、{r['ID']}）" for _, r in rem.iterrows()))
    if len(inv):
        print("  ★ 0 以下の値で除いた行（一覧に未登録）：" + "、".join(inv["file"]))
    n_t = (TRAIN[mat]["Tset"].nunique(), TEST[mat]["Tset"].nunique())
    how = f"学習は{FILE_INFO[(mat, '5data')]['設定温度の取り方']}、検証は{FILE_INFO[(mat, '3data')]['設定温度の取り方']}から"
    judge(min(n_t) >= 3, f"学習・検証とも、設定温度が 3 水準以上ある（学習 {n_t[0]}・検証 {n_t[1]} 水準。設定温度は、{how}）", mat, "データ")
    judge(len(inv) == 0, f"η・γ̇・P3_3 が 0 以下の行（対数がとれない）は、すべて問題データ一覧に登録済み（未登録のまま自動で除いた行 {len(inv)} 行）。未登録なら、一覧に「除外」で登録する", mat, "データ")
    judge(TRAIN[mat]["Tset"].min() <= cfg["Tref"] <= TRAIN[mat]["Tset"].max(),
          f"基準温度 Tref = {cfg['Tref']:g} ℃ が、学習データの設定温度の範囲の中にある（シフトファクターは{'アレニウス型' if cfg['type'] == 'arrhenius' else 'WLF 型'}）", mat, "データ")
    judge((TEST[mat]["group"] != "範囲内").any(), f"検証データに、学習データの射出速度の範囲（{TRAIN[mat]['v'].min():g}〜{TRAIN[mat]['v'].max():g} mm/s）の外の条件がある"
          "（すべて範囲の中なら、学習データと検証データのファイルを逆に選んだおそれがある）", mat, "データ")'''

# ---------------------------------------------------------------- プロファイル尤度
PROFILE = r'''PROF, BEST = {}, {}
t_all = time.time()
for mat in MATERIALS:
    thn, tvn, un = theta_names(CFG[mat])
    for key in SHIFT_KEYS:
        t0 = time.time()
        pts, conv = run_profile(key, TRAIN[mat], CFG[mat])
        PROF[(mat, key)] = pts
        BEST[(mat, key)] = b = profile_best(pts, key, CFG[mat], conv)
        to_csv(pts.drop(columns="theta"), os.path.join(out_dir(mat, key), "プロファイル尤度_格子.csv"))
        print(f"{MAT_NAMES[mat]} {key}：{len(pts)} 点、{time.time() - t0:.0f} 秒（細かい格子の取り直しが{'収束' if conv else '収束せず'}）。最良 {thn} = {b['th_best']:g}、{tvn} = {b['tv_best']:g} {un}"
              f"（差 {tvn} − {thn} = {b['diff_best']:g}〔{b['diff_lo']:g}〜{b['diff_hi']:g}〕）、格子の LML {b['LML_best']:.1f}")
print(f"合計 {(time.time() - t_all) / 60:.1f} 分" + ("" if SHIFT_KEYS else "（シフトのあるモデルがないので、計算なし）"))
for mat in MATERIALS:
    thn, tvn, un = theta_names(CFG[mat])
    for key in SHIFT_KEYS:
        b = BEST[(mat, key)]
        judge((not b["at_edge"]) and b["refine_converged"], f"{key}：最良の θ が、粗い格子の端にも、細かい格子の端にもない（{thn} = {b['th_best']:g}、{tvn} = {b['tv_best']:g} {un}。取り直しが{'収束' if b['refine_converged'] else '収束せず'}）。端なら SHIFT_CFG の範囲を広げる。収束していなければ MAX_REFINE を増やす", mat, "尤度")'''

# ---------------------------------------------------------------- 最良の θ と θ = 0 の念入りな最適化
CAREFUL = r'''from joblib import Parallel, delayed
t0 = time.time()
FITS, ZERO, LOSS = {}, {}, {}
jobs = [(mat, key, kind) for mat in MATERIALS for key in SHIFT_KEYS for kind in ("best", "zero")]
res = Parallel(n_jobs=min(N_JOBS, len(jobs)))(
    delayed(fit_job)(key, TRAIN[mat], CFG[mat], (BEST[(mat, key)]["th_best"] if kind == "best" else 0.0), (BEST[(mat, key)]["tv_best"] if kind == "best" else 0.0),
                     N_RESTARTS, (BEST[(mat, key)]["theta_best"] if kind == "best" else BEST[(mat, key)]["theta_zero"]), INPUT_MODE, kind == "best")
    for mat, key, kind in jobs) if jobs else []
for (mat, key, kind), f in zip(jobs, res):
    if kind == "best":
        FITS[(mat, key)] = f
        LOSS[(mat, key)] = f["loss_runs"]
    else:
        ZERO[(mat, key)] = f
print(f"念入りな最適化：{len(jobs)} 回、{(time.time() - t0) / 60:.1f} 分")
for mat in (MATERIALS if SHIFT_KEYS else []):       # シフトのあるモデルがなければ（C だけなら）、飛ばす
    thn, tvn, un = theta_names(CFG[mat])
    for key in SHIFT_KEYS:
        b = BEST[(mat, key)]
        lml_b, lml_z = float(FITS[(mat, key)]["gp"].log_marginal_likelihood_value_), float(ZERO[(mat, key)]["gp"].log_marginal_likelihood_value_)
        b["LML_best_grid"], b["LML_at_0_grid"], b["dLML_vs_0_grid"] = b["LML_best"], b["LML_at_0"], b["dLML_vs_0"]
        b["LML_best"], b["LML_at_0"] = max(lml_b, b["LML_best_grid"]), max(lml_z, b["LML_at_0_grid"])
        b["dLML_vs_0"] = b["LML_best"] - b["LML_at_0"]
    t = pd.DataFrame([{"材料": MAT_NAMES[mat], "モデル": key, "入力": VERSION_DIRS[INPUT_MODE], "θ の名前": f"{thn}・{tvn}", "単位": un,
                       **{k: v for k, v in BEST[(mat, key)].items() if k not in ("theta_best", "theta_zero")}} for key in SHIFT_KEYS])
    to_csv(t, os.path.join(out_dir(mat), "シフトファクターの推定.csv"))
    display(t[["材料", "モデル", "θ の名前", "単位", "th_best", "th_lo", "th_hi", "tv_best", "tv_lo", "tv_hi", "diff_best", "diff_lo", "diff_hi", "LML_best", "LML_at_0", "dLML_vs_0", "dLML_vs_0_grid", "at_edge", "refine_converged"]].round(2))
    for key in SHIFT_KEYS:
        b = BEST[(mat, key)]
        other = "Tff・P3_3" if key.endswith("Tset") else "Tset・P3_3"
        print(f"{MAT_NAMES[mat]} {key}：念入りな最適化の LML {b['LML_best']:.1f}（格子 {b['LML_best_grid']:.1f}）、θ = 0 は {b['LML_at_0']:.1f}（格子 {b['LML_at_0_grid']:.1f}）→ ΔLML {b['dLML_vs_0']:+.1f}（格子では {b['dLML_vs_0_grid']:+.1f}）")
        judge(b["dLML_vs_0"] >= LRT_DLML, f"{key}：温度のシフトが尤度で支持される（念入りに最適化した LML が、θ = 0〔{key.split('_')[1]} を使わない〕より {b['dLML_vs_0']:.1f} 大きい。基準 {LRT_DLML:g} 以上＝尤度比検定の 5% 基準）。小さければ、温度の効果はほかの入力（{other}）が担っている", mat, "尤度")'''

# ---------------------------------------------------------------- 尤度の面
SURFACE = r'''for mat in MATERIALS:
    thn, tvn, un = theta_names(CFG[mat])
    for key in SHIFT_KEYS:
        pts, b = PROF[(mat, key)], BEST[(mat, key)]
        val = (pts["LML"] - b["LML_best"]).to_numpy()
        if key.startswith("A"):
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.plot(pts["tv"], val, "o-", color="#1f77b4")
            ax.axhline(-1.92, color="red", ls="--", label="最大 − 1.92（95% 区間の目安）")
            ax.axvline(0, color="k", lw=0.6)
            ax.set_xlabel(f"縦のシフト {tvn} ({un})")
            ax.set_ylabel("LML − 最大値")
            ax.set_ylim(max(val.min(), -60), 2)
            ax.legend()
            ax.grid(True, ls=":", alpha=0.5)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
            rng = (CFG[mat].get("fine") or FINE_B)[0]
            near = pts[pts["th"].between(b["th_best"] - rng, b["th_best"] + rng) & pts["tv"].between(b["tv_best"] - rng, b["tv_best"] + rng)]
            for ax, sub, lo, ttl in [(axes[0], pts, -60.0, "全体（粗い格子＋細かい格子）"), (axes[1], near, -15.0, "最良の点の近く（細かい格子）")]:
                v = np.maximum((sub["LML"] - b["LML_best"]).to_numpy(), lo)
                tri = mtri.Triangulation(sub["th"].to_numpy(), sub["tv"].to_numpy())
                cs = ax.tricontourf(tri, v, levels=np.linspace(lo, 0, 25), cmap="viridis")
                ax.tricontour(tri, v, levels=[-1.92], colors="red", linewidths=2)
                fig.colorbar(cs, ax=ax, label="LML − 最大値")
                ax.plot(b["th_best"], b["tv_best"], marker="*", color="white", ms=20, mec="k", ls="none", label=f"最良（{b['th_best']:g}, {b['tv_best']:g}）")
                if sub["th"].min() <= 0 <= sub["th"].max() and sub["tv"].min() <= 0 <= sub["tv"].max():
                    ax.plot(0, 0, marker="x", color="k", ms=12, mew=2.5, ls="none", label="θ = 0（シフトなし）")
                ax.set_xlabel(f"横のシフト {thn} ({un})")
                ax.set_ylabel(f"縦のシフト {tvn} ({un})")
                ax.set_title(f"{ttl}。赤線の内側：95% 区間の目安")
                ax.legend(loc="upper left", fontsize=10)
        fig.suptitle(f"{MAT_NAMES[mat]}　{model_title(key)}　プロファイル尤度（{'アレニウス型' if CFG[mat]['type'] == 'arrhenius' else 'WLF 型'}）")
        save_fig(fig, out_dir(mat, key), "プロファイル尤度")'''

# ---------------------------------------------------------------- シフトのないモデル（C）の学習と、ハイパーパラメータの表
NOSHIFT = r'''t0 = time.time()
NOSHIFT_KEYS = [k for k in KEYS if k not in SHIFT_KEYS]      # シフトのないモデル（C）
jobs = [(mat, key) for mat in MATERIALS for key in NOSHIFT_KEYS]
res = Parallel(n_jobs=min(N_JOBS, len(jobs)))(delayed(fit_job)(key, TRAIN[mat], CFG[mat], 0.0, 0.0, N_RESTARTS, None, INPUT_MODE, True) for mat, key in jobs) if jobs else []
for (mat, key), f in zip(jobs, res):
    FITS[(mat, key)] = f
    LOSS[(mat, key)] = f["loss_runs"]
print(f"シフトのないモデルの学習：{len(jobs)} 回、{(time.time() - t0) / 60:.1f} 分")
KROWS = []
for mat in MATERIALS:
    for key in KEYS:
        f = FITS[(mat, key)]
        KROWS.append({"材料": MAT_NAMES[mat], **kernel_row(f)})
        print(f"{MAT_NAMES[mat]} {key}：{f['gp'].kernel_}")
KT = pd.DataFrame(KROWS)
display(KT.round(4))
for mat in MATERIALS:
    to_csv(KT[KT["材料"] == MAT_NAMES[mat]], os.path.join(out_dir(mat), "ハイパーパラメータ.csv"))
    for key in KEYS:
        r = KT[(KT["材料"] == MAT_NAMES[mat]) & (KT["model"] == key)].iloc[0]
        if key in SHIFT_KEYS:
            lb = BEST[(mat, key)]["LML_best"]
            judge(r["LML"] >= lb - 0.5, f"{key}：最終モデルの LML（{r['LML']:.1f}）が、尤度の格子の最良（{lb:.1f}）以上", mat, "学習")
        judge(not r["noise_at_lower_bound"], f"{key}：ノイズ分散（{r['noise_var']:.2e}）が下限 {NOISE_LB:g} に張り付いていない", mat, "学習")
        at = [c[3:] for c in KT.columns if c.startswith("ls_") and c != "ls_at_bound" and pd.notna(r[c]) and (r[c] > 99 or r[c] < 0.0105)]
        if at:
            print(f"  （参考）{MAT_NAMES[mat]} {key}：長さスケールが上限・下限に届いた入力：{'、'.join(at)}")'''

# ---------------------------------------------------------------- シフトファクターとマスターカーブ
SHIFT = r'''SHIFT_ROWS = []
for mat in MATERIALS:
    cfg = CFG[mat]
    thn, tvn, un = theta_names(cfg)
    pool = pd.concat([TRAIN[mat], TEST[mat]], ignore_index=True)
    train_c = set(TRAIN[mat]["cond"])
    tl, th_ = pool["Tset"].min(), pool["Tset"].max()
    if SHIFT_KEYS:                                  # シフトファクターの図（シフトのあるモデルがあるときだけ）
        Tg = np.linspace(min(pool["Tff"].min(), tl) - 5, max(pool["Tff"].max(), th_) + 5, 200)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
        cols_ = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
        for key, col in zip(SHIFT_KEYS, cols_):
            pts, b = PROF[(mat, key)], BEST[(mat, key)]
            inside = pts[pts["LML"] >= b["LML_best"] - 1.92]
            for ax, w, nm in [(axes[0], "th", thn), (axes[1], "tv", tvn)]:
                if w == "th" and key.startswith("A"):
                    continue
                cur = np.array([shift_ln(Tg, v, cfg) for v in inside[w]])
                ax.fill_between(Tg, cur.min(axis=0), cur.max(axis=0), color=col, alpha=0.18)
                ax.plot(Tg, shift_ln(Tg, b[f"{w}_best"], cfg), color=col, lw=2.5, label=f"{key}（{nm} = {b[f'{w}_best']:g} {un}、95%：{b[f'{w}_lo']:g}〜{b[f'{w}_hi']:g}）")
            SHIFT_ROWS.append({"材料": MAT_NAMES[mat], "モデル": key, "型": "アレニウス" if cfg["type"] == "arrhenius" else "WLF", "Tref [℃]": cfg["Tref"],
                               f"横 {thn}": b["th_best"], f"縦 {tvn}": b["tv_best"], "単位": un,
                               f"縦の倍率 a_v({tl:g})/a_v({th_:g})": float(np.exp(shift_ln(tl, b["tv_best"], cfg) - shift_ln(th_, b["tv_best"], cfg))),
                               f"横の倍率 a_h({tl:g})/a_h({th_:g})": float(np.exp(shift_ln(tl, b["th_best"], cfg) - shift_ln(th_, b["th_best"], cfg)))})
        for ax, ttl in [(axes[0], "横のシフト ln a_h(T)（せん断速度の向き）"), (axes[1], "縦のシフト ln a_v(T)（粘度の向き）")]:
            for t in sorted(pool["Tset"].unique()):
                ax.axvline(t, color="gray", ls=":", lw=1)
            ax.axvspan(pool["Tff"].min(), pool["Tff"].max(), color="orange", alpha=0.07, label="Tff の範囲（データ）")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xlabel("温度 T (℃)（B_Tset は設定温度、B_Tff は流動中樹脂温度）")
            ax.set_ylabel(ttl.split("（")[0].replace("のシフト ", " "))
            ax.set_title(ttl)
            ax.legend(fontsize=9, loc="best")
            ax.grid(True, ls=":", alpha=0.5)
        typ = "アレニウス型" if cfg["type"] == "arrhenius" else f"WLF 型（D2 = Tg = {cfg['Tg']:g} ℃、A2 = {cfg['A2']:g} K）"
        fig.suptitle(f"{MAT_NAMES[mat]}　温度のシフトファクター（{typ}、Tref = {cfg['Tref']:g} ℃。帯：尤度の 95% 区間）")
        save_fig(fig, out_dir(mat), "シフトファクター")
    # マスターカーブ（シフト前と、モデルごと）
    Hs = sorted(pool["H"].unique())
    ncol = 2 if len(Hs) == 4 else len(Hs)
    nrow = int(np.ceil(len(Hs) / ncol))
    ts = t_style(pool["Tset"].unique())
    sup = {}
    for key in [None] + SHIFT_KEYS:
        b = BEST.get((mat, key), {"th_best": 0.0, "tv_best": 0.0})
        t = shifted_cond_means(pool, key, cfg, b["th_best"] if (key or "").startswith("B") else 0.0, b["tv_best"])
        sup[key] = superpose_rmse(t)
        fig, axes = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, 5.6 * nrow), squeeze=False)
        for ax, h in zip(axes.flat, Hs):
            for tt, (col, mk) in ts.items():
                m = (t["H"] == h) & (t["Tset"] == tt)
                trm, tem = m & t["cond"].isin(train_c), m & ~t["cond"].isin(train_c)
                ax.scatter(np.exp(t.loc[trm, "x"]), np.exp(t.loc[trm, "y"]), color=col, marker=mk, s=60, edgecolors="k", linewidths=0.5, label=f"{int(tt)} ℃", zorder=3)
                ax.scatter(np.exp(t.loc[tem, "x"]), np.exp(t.loc[tem, "y"]), facecolors="white", edgecolors=col, marker=mk, s=60, linewidths=1.8, zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            log_ticks(ax)
            ax.set_title(f"厚み {h:g} mm")
            ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ (1/s)" if key is None else r"換算せん断速度 $\dot{\gamma}\cdot a_h$ (1/s)")
            ax.set_ylabel(r"粘度 $\eta$ (Pa·s)" if key is None else r"換算粘度 $\eta / a_v$ (Pa·s)")
            ax.grid(True, which="both", ls=":", alpha=0.5)
            ax.legend(fontsize=9, title="設定温度")
        for ax in list(axes.flat)[len(Hs):]:
            ax.axis("off")
        ttl = "シフト前（実測の条件平均）" if key is None else f"マスターカーブ：{model_title(key)}（最良の θ でずらした条件平均）"
        fig.suptitle(f"{MAT_NAMES[mat]}　{ttl}\n塗り：学習（5data）、白抜き：検証（3data）。重ね合わせの残差 {sup[key]:.3f}")
        save_fig(fig, out_dir(mat) if key is None else out_dir(mat, key), "流動曲線_シフト前" if key is None else "マスターカーブ",
                 show=SHOW_FIGS and key in (None, MAIN_KEY))
    for r in SHIFT_ROWS:
        if r["材料"] == MAT_NAMES[mat]:
            r["重ね合わせの残差（シフト後）"], r["重ね合わせの残差（シフト前）"] = sup[r["モデル"]], sup[None]
    for key in SHIFT_KEYS:
        b = BEST[(mat, key)]
        judge(shift_ln(tl, b["tv_best"], cfg) > 0, f"{key}：縦のシフトの向きが「低温ほど粘い」（ln a_v({tl:g} ℃) = {float(shift_ln(tl, b['tv_best'], cfg)):+.3f}。{tvn} = {b['tv_best']:g} {un}）", mat, "シフト")
        judge(sup[key] < sup[None], f"{key}：シフトで温度ごとの曲線が近づく（重ね合わせの残差 {sup[None]:.3f} → {sup[key]:.3f}）", mat, "シフト")
SHIFT_TAB = pd.DataFrame(SHIFT_ROWS)
display(SHIFT_TAB.round(3))
for mat in (MATERIALS if len(SHIFT_TAB) else []):   # シフトのあるモデルがなければ（C だけなら）、表はない
    to_csv(SHIFT_TAB[SHIFT_TAB["材料"] == MAT_NAMES[mat]].dropna(axis=1, how="all"), os.path.join(out_dir(mat), "シフトファクターの倍率.csv"))'''

# ---------------------------------------------------------------- 3data による予測検証
VALID = r'''PRED, MROWS = {}, []
for mat in MATERIALS:
    te = TEST[mat]
    for key in KEYS:
        mu, sd, sdf, nsd = predict(FITS[(mat, key)], te)
        PRED[(mat, key)] = te[["file", "cond", "Tset", "H", "v", "group", "eta", "gd", "P33", "Tff", "ln_eta"]].assign(
            予測_ln_eta=mu, 予測標準偏差=sd, 潜在関数の標準偏差=sdf, 予測η=np.exp(mu), 下限95=np.exp(mu - 1.96 * sd), 上限95=np.exp(mu + 1.96 * sd))
        to_csv(PRED[(mat, key)], os.path.join(out_dir(mat, key), "予測検証_3data_予測値.csv"))
        MROWS += metrics_rows(mat, key, te, mu, sd, sdf, {"ノイズの標準偏差": nsd})
MT = pd.DataFrame(MROWS)
GCOL = {"低速側": ("#1f77b4", "o"), "高速側": ("#d62728", "s"), "範囲内": ("#2ca02c", "^")}
for mat in MATERIALS:
    t = MT[MT["材料"] == MAT_NAMES[mat]]
    to_csv(t, os.path.join(out_dir(mat), "予測検証_3data_指標.csv"))
    print(f"\n【{MAT_NAMES[mat]}】3data の ln η の RMSE")
    display(t.pivot_table(index="モデル", columns="範囲", values="RMSE", sort=False).round(4))
    print("95% 予測区間に入る割合（被覆率）")
    display(t.pivot_table(index="モデル", columns="範囲", values="coverage95", sort=False).round(3))
    display(t[t["範囲"] == "全体"][["モデル", "R2", "R2_real", "NLPD", "mean_sd", "ノイズの標準偏差"]].round(4))
    for key in KEYS:
        p = PRED[(mat, key)]
        r = t[(t["モデル"] == key) & (t["範囲"] == "全体")].iloc[0]
        fig, ax = plt.subplots(figsize=(9.5, 9))
        for g, (col, mk) in GCOL.items():
            m = (p["group"] == g).to_numpy()
            if m.any():
                ax.errorbar(p.loc[m, "eta"], p.loc[m, "予測η"], yerr=[p.loc[m, "予測η"] - p.loc[m, "下限95"], p.loc[m, "上限95"] - p.loc[m, "予測η"]],
                            fmt=mk, color=col, ms=6, elinewidth=1.2, capsize=3, alpha=0.8, label=f"{g}（{int(m.sum())} ショット）")
        lim = [min(p["eta"].min(), p["下限95"].min()) / 1.3, max(p["eta"].max(), p["上限95"].max()) * 1.3]
        ax.plot(lim, lim, "k--", lw=1, label="予測 = 実測")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        log_ticks(ax)
        ax.set_xlabel(r"実測の粘度 $\eta$ (Pa·s)")
        ax.set_ylabel(r"予測の粘度 $\eta$ (Pa·s)")
        ax.set_title(f"{MAT_NAMES[mat]}　{model_title(key)}\n3data：RMSE(ln η) {r['RMSE']:.3f}、R²(ln η) {r['R2']:.4f}、R²(実数) {r['R2_real']:.4f}、被覆率 {r['coverage95']:.2f}")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(loc="upper left")
        save_fig(fig, out_dir(mat, key), "予測検証_3data_パリティ", show=SHOW_FIGS and key == MAIN_KEY)
    if "C" in KEYS:                                 # 基準の C（シフトなし）と比べる（C とシフトのあるモデルの両方を学習したとき）
        rc = float(t[(t["モデル"] == "C") & (t["範囲"] == "全体")]["RMSE"].iloc[0])
        for key in SHIFT_KEYS:
            rk = float(t[(t["モデル"] == key) & (t["範囲"] == "全体")]["RMSE"].iloc[0])
            judge(rk <= 1.1 * rc, f"{key} の 3data の RMSE（全体）{rk:.4f} が、C（{rc:.4f}）の 1.1 倍以内", mat, "検証")
    cov = t[t["範囲"] == "全体"].set_index("モデル")["coverage95"]
    judge(bool((cov >= 0.8).all()), "95% 予測区間に入る割合が、どのモデルも 0.8 以上（" + "、".join(f"{k} {v:.2f}" for k, v in cov.items()) + "）", mat, "検証")'''

# ---------------------------------------------------------------- 学習済みモデルの保存
SAVE = r'''import joblib
import sklearn
for mat in MATERIALS:
    for key in KEYS:
        path = os.path.join(out_dir(mat, key), f"学習済みモデル_{key}.joblib")
        joblib.dump({"fit": FITS[(mat, key)], "材料": MAT_NAMES[mat], "モデル": key, "説明": model_title(key), "入力": input_labels(key), "入力の取り方": INPUT_MODE,
                     "シフトファクター": CFG[mat], "学習データのファイル": show_path(SOURCES[mat]["5data"]), "検証データのファイル": show_path(SOURCES[mat]["3data"]),
                     "scikit-learn": sklearn.__version__, "作成日": time.strftime("%Y-%m-%d")}, path)
        print("保存:", show_path(path))'''
