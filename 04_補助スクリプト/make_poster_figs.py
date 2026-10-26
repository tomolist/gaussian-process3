# -*- coding: utf-8 -*-
"""ポスター（05_…/学習結果報告書作成_261015/ポスター/報告書Bから作成.pptx）の図を作る（日本語。学習はやり直さない）。

使うモデルは報告書B と同じ B_Tff・条件平均の入力（保存済みの 学習済みモデル_B_Tff.joblib と、予測検証の表）。
  見本_*.png        ：見本のポスター（報告書Aから作成2.pdf）から、写真・説明図を文字や矢印ごと切り出したもの（300 dpi）
  算出式.png         ：樹脂粘度とせん断速度の算出式
  GPR概念図.png      ：ガウス過程回帰の考え方（データの多い所は予測の幅が狭く、ない所は広い）。人工のデータの例
  温度シフト概念図.png：温度シフトファクターの考え方（横・縦のずれ）
  温度シフトの効果.png：温度シフトファクターで換算する前と後の、温度によるデータのずれ（材料ごと）
  学習結果_<材料>.png ：樹脂粘度とせん断速度の関係の学習結果（報告書B の図5 と同じ中身。軸の範囲は拡張版と同じ）
  予測検証_<材料>.png ：学習に用いていない射出率条件の、実測と予測の粘度（報告書B の図6 と同じ中身）
  予測誤差の比較.png  ：本モデルと Cross-WLF モデル（参考）の予測誤差
  ポスターの数値.csv  ：ポスターに書いた数値の元（表から計算した値）

使い方（リポジトリ直下で）: uv run python 04_補助スクリプト/make_poster_figs.py [--crops]
  --crops：見本のポスターの切り出しを作り直す（なければ作る）
図の大きさは、ポスターに貼る大きさ（mm）で作る（文字の大きさがポスター上の pt になる）。
"""
import os
import sys

os.environ.setdefault("NB_REPO", ".")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch  # noqa: E402

import mat_common as mc  # noqa: E402

MATS = ["PP", "MIX", "GPPS"]
MAT_JA = {"PP": "PP", "MIX": "粉砕材混合PP", "GPPS": "GPPS"}
KEY = "B_Tff"
MODE = "cond"            # 読むモデルの入力の取り方（報告書B・ポスターは条件平均の入力で作った。発表資料の make_slide_figs.py は "shot" に変えて使う）
POSTER = os.path.join(mc.DATA_ROOT, "学習結果報告書作成_261015", "ポスター")
OUT = os.path.join(POSTER, "図")
SAMPLE_PDF = os.path.join(POSTER, "報告書Aから作成2.pdf")
DPI = 300
BLUE = "#006699"                              # 見本のポスターの見出しの色
T_COLORS = ["#d62728", "#1f77b4", "#2ca02c"]  # 加熱シリンダ温度ごとの線・帯（低い温度から赤・青・緑。ノートの総合図と同じ）

fm.findfont("IPAexGothic", fallback_to_default=False)
plt.rcParams.update({"font.family": ["IPAexGothic", "DejaVu Sans"], "axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
                     "font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12.5,
                     "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                     "xtick.major.width": 1.0, "ytick.major.width": 1.0, "xtick.major.size": 6, "ytick.major.size": 6,
                     "xtick.minor.size": 3, "ytick.minor.size": 3})


def mm(w, h):
    return (w / 25.4, h / 25.4)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=DPI)       # 大きさを変えない（bbox_inches は使わない）
    plt.close(fig)


def pct(r):
    """ln η の RMSE → 粘度での誤差 [%]（exp(RMSE) − 1。報告書B と同じ換算）"""
    return (np.exp(r) - 1.0) * 100.0


def r0(x):
    """四捨五入して整数に（Python の round は偶数への丸めなので使わない）"""
    return int(np.floor(x + 0.5))


# ---------------------------------------------------------------- 見本のポスターからの切り出し
CROPS = {  # 見本のページの範囲（pt。左上が原点：左 上 右 下）
    "見本_計測金型": (25.9, 849.6, 691.2, 1315.8),
    "見本_キャビティ入れ子": (708.1, 838.8, 1166.0, 1116.0),
    "見本_入れ子の挿入": (703.8, 1146.6, 1166.0, 1423.8),
    "見本_センサの設置位置": (21.0, 1544.4, 1182.6, 1854.0),
    "見本_成形品の形状": (39.6, 2037.6, 583.2, 2409.0),
    "見本_成形品の外観": (686.5, 2045.9, 1131.5, 2309.4),
    "見本_キャビティの模式図": (1249.2, 480.6, 1893.6, 698.4),
    "見本_ショートショット": (1890.7, 838.4, 2350.4, 1170.0),
    "見本_樹脂温度の経時変化": (588.5, 2845.8, 1168.2, 3249.0),
}


def crop_sample():
    """見本のポスターを Ghostscript（TeX Live の rungs）で 300 dpi の画像にして（_tmp/poster/sample300.png）、写真・説明図を切り出す"""
    import shutil
    import subprocess

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    gs = shutil.which("rungs")
    assert gs, "rungs（TeX Live の Ghostscript）が見つからない"
    work = os.path.join(mc.REPO, "_tmp", "poster")
    os.makedirs(work, exist_ok=True)
    png = os.path.join(work, "sample300.png")
    if not os.path.exists(png):
        subprocess.run([gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", f"-r{DPI}", "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
                        f"-sOutputFile={png}", SAMPLE_PDF], check=True)
    page = Image.open(png).convert("RGB")
    s = DPI / 72.0
    os.makedirs(OUT, exist_ok=True)
    for name, (l, t, r, b) in CROPS.items():
        im = page.crop((round(l * s), round(t * s), round(r * s), round(b * s)))
        im.save(os.path.join(OUT, f"{name}.png"), dpi=(DPI, DPI))
        print(f"{name}：{im.width}×{im.height} px")


# ---------------------------------------------------------------- 説明の図
def fig_formula():
    fig = plt.figure(figsize=mm(190, 62))
    fs = 27
    fig.text(0.02, 0.70, r"$\eta = \dfrac{\Delta P_1\, B\, H^3}{12\, L\, Q}$", fontsize=fs, va="center")
    fig.text(0.53, 0.70, r"$\dot{\gamma} = \dfrac{6\,Q}{B\, H^2}$", fontsize=fs, va="center")
    fig.text(0.02, 0.20, r"流量：$Q = V H B$", fontsize=fs * 0.82, va="center")
    fig.text(0.53, 0.20, r"流速：$V = L\,/\,\Delta t$", fontsize=fs * 0.82, va="center")
    save(fig, "算出式.png")


def fig_gpr_concept():
    """人工のデータ（x が 2〜6.5 に多く、7 より右にはない）でガウス過程回帰を当てはめ、予測平均と 95% 予測区間を描く"""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    rng = np.random.default_rng(3)
    f = lambda x: 2.2 - 0.32 * x + 0.35 * np.sin(1.1 * x)
    x = np.sort(np.concatenate([rng.uniform(1.6, 6.6, 26), [0.6]]))
    y = f(x) + rng.normal(0, 0.06, len(x))
    gp = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(1.5) + WhiteKernel(0.01), normalize_y=True, random_state=0).fit(x[:, None], y)
    xs = np.linspace(0, 10, 400)
    mu, sd = gp.predict(xs[:, None], return_std=True)
    fig, ax = plt.subplots(figsize=mm(195, 108))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.86)
    ax.fill_between(xs, mu - 1.96 * sd, mu + 1.96 * sd, color="#e8212b", alpha=0.18, lw=0, label="95% 予測区間（予測の幅）")
    ax.plot(xs, mu, color="#e8212b", lw=2.5, label="予測平均値")
    ax.scatter(x, y, s=55, color="#1f4fe0", edgecolors="k", linewidths=0.6, zorder=5, label="学習データ")
    for xc in (4.0, 9.0):                                            # 予測の分布（正規分布）を横向きに描く
        i = np.argmin(np.abs(xs - xc))
        yy = np.linspace(mu[i] - 3.2 * sd[i], mu[i] + 3.2 * sd[i], 200)
        pdf = np.exp(-0.5 * ((yy - mu[i]) / sd[i]) ** 2)
        ax.plot(xc + 0.9 * pdf, yy, color="0.15", lw=1.6)
        ax.plot([xc, xc], [yy[0], yy[-1]], color="0.15", lw=1.0, ls=":")
    i4, i9 = np.argmin(np.abs(xs - 4.0)), np.argmin(np.abs(xs - 9.0))
    ax.annotate("データが多い\n→ 予測の幅が狭い", xy=(4.15, mu[i4] - 1.0 * sd[i4]), xytext=(3.3, -1.05), fontsize=15, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", lw=1.4, color="0.15"))
    ax.annotate("データがない\n→ 予測の幅が広い", xy=(8.4, mu[i9] - 1.75 * sd[i9]), xytext=(6.6, -1.05), fontsize=15, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", lw=1.4, color="0.15"))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.6, 3.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("入力（例：せん断速度）", labelpad=4)
    ax.set_ylabel("出力（例：樹脂粘度）", labelpad=4)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, frameon=False, fontsize=13.5, borderaxespad=0.0, columnspacing=1.2)
    save(fig, "GPR概念図.png")


def fig_shift_concept():
    """Cross 型の 2 本の曲線（低温・高温）と、横（せん断速度）・縦（粘度）方向のずれの矢印"""
    g = np.logspace(0, 5, 300)
    kh, kv = 5.0, 3.5                                                     # 高温の曲線：横に kh 倍・縦に 1/kv にずらした曲線
    cross = lambda g: 3000.0 / (1 + (3000.0 * g / 3e4) ** 0.72)
    lo, hi = cross(g), cross(g * kh) / kv
    fig, ax = plt.subplots(figsize=mm(170, 100))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.97)
    ax.plot(g, lo, color="#1f77b4", lw=3, label="低温")
    ax.plot(g, hi, color="#d62728", lw=3, ls="--", label="高温")
    gx = 12.0
    p0 = (gx, cross(gx * kh) / kv)                                        # 高温の曲線の点
    p1 = (gx * kh, p0[1])                                                 # 横にずらした点
    p2 = (gx * kh, cross(gx * kh))                                        # さらに縦にずらすと、低温の曲線に乗る
    for a, b in [(p0, p1), (p1, p2)]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=26, lw=2.8, color="#333333", zorder=6, shrinkA=0, shrinkB=2))
    arrow = dict(arrowstyle="->", lw=1.3, color="0.25", shrinkA=2, shrinkB=4)
    ax.annotate("横のずれ（せん断速度方向）", xy=(np.sqrt(p0[0] * p1[0]), p0[1]), xytext=(22.0, 11.0), ha="center", va="center", fontsize=14,
                arrowprops=arrow, zorder=7)
    ax.annotate("縦のずれ\n（粘度方向）", xy=(p1[0], np.sqrt(p1[1] * p2[1])), xytext=(260.0, 620.0), ha="left", va="center", fontsize=14,
                arrowprops=arrow, zorder=7)
    ax.text(4e3, 2.6e3, "温度シフトファクターで\n横・縦にずらすと\n1 本の曲線に重なる", ha="center", va="center", fontsize=14,
            bbox=dict(boxstyle="round,pad=0.4", fc="#eef6fb", ec=BLUE, lw=1.2), zorder=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e5)
    ax.set_ylim(2, 8000)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.minorticks_off()
    ax.set_xlabel(r"せん断速度 $\dot{\gamma}$（対数）", labelpad=4)
    ax.set_ylabel(r"樹脂粘度 $\eta$（対数）", labelpad=4)
    ax.legend(loc="upper left", fontsize=14, frameon=True, framealpha=1.0, ncol=2)
    save(fig, "温度シフト概念図.png")


# ---------------------------------------------------------------- 学習結果の図
def model_dir(mat):
    folder = mc.MAT_FOLDERS[mat]
    return os.path.join(mc.DATA_ROOT, folder, f"{folder}_学習結果", mc.MODE_JA[MODE], KEY)


def compare_dir(mat):
    folder = mc.MAT_FOLDERS[mat]
    return os.path.join(mc.DATA_ROOT, folder, f"{folder}_学習結果", mc.MODE_JA[MODE], "比較")


def log_axes(ax, xlim, ylim):
    import matplotlib.ticker as ticker
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))
        a.set_major_formatter(ticker.LogFormatterSciNotation())
        a.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, which="major", ls="-", lw=0.6, color="0.75", zorder=0)
    ax.grid(True, which="minor", ls=":", lw=0.4, color="0.85", zorder=0)


def fig_flow(mat, tr, fit):
    """学習データの条件平均の点（色・形は厚み）と、加熱シリンダ温度ごとの予測平均値・95% 予測区間（ΔP2 は学習データの中央値、
    金型内樹脂最高温度は加熱シリンダ温度ごとの中央値に固定。ノートの拡張版の総合図と同じ）"""
    xl, yl = (1.0, 1e5), (0.5, 3000.0)
    p_med = float(np.exp(tr["ln_P33"].median()) / 1e6)
    fig, ax = plt.subplots(figsize=mm(129, 112))
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.165, top=0.975)
    log_axes(ax, xl, yl)
    lg = np.linspace(np.log(xl[0]), np.log(xl[1]), 500)
    h_t, info = [], []
    for i, (t, sub) in enumerate(tr.groupby("Tset")):
        tff = float(np.exp(sub["ln_Tff"].median()) - 273.15)
        mu, sd, _, _ = mc.predict(fit, mc.model_inputs(np.exp(lg), p_med, tff, float(t)))
        c = T_COLORS[i]
        ax.fill_between(np.exp(lg), np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd), color=c, alpha=0.22, lw=0, zorder=2)
        ax.plot(np.exp(lg), np.exp(mu), color=c, lw=2.0, zorder=4)
        h_t.append((Line2D([], [], color=c, lw=2.0), Patch(facecolor=c, alpha=0.22)))
        info.append((int(t), tff))
    h_h = []
    for h in sorted(tr["H"].unique()):
        a = tr[np.isclose(tr["H"], h)].groupby("cond").agg(lg=("ln_gd", "mean"), le=("ln_eta", "mean"))
        col, mk = mc.H_COLORS[float(h)], mc.H_MARKS[float(h)]
        ax.scatter(np.exp(a["lg"]), np.exp(a["le"]), s=46, color=col, marker=mk, alpha=0.85, edgecolors="k", linewidths=0.5, zorder=5)
        h_h.append(Line2D([], [], ls="", marker=mk, ms=7.5, color=col, mec="k", mew=0.5, label=f"{h:g} mm"))
    lg1 = ax.legend(h_t, [f"{t} ℃" for t, _ in info], handler_map={tuple: HandlerTuple(ndivide=None)}, title="加熱シリンダ温度",
                    loc="upper right", fontsize=12, title_fontsize=12, frameon=True, framealpha=0.9, borderaxespad=0.4, handlelength=2.2)
    ax.add_artist(lg1)
    ax.legend(handles=h_h, title="キャビティ厚み", loc="lower left", fontsize=12, title_fontsize=12, frameon=True, framealpha=0.9,
              borderaxespad=0.4, handletextpad=0.3)
    ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ (1/s)")
    ax.set_ylabel(r"樹脂粘度 $\eta$ (Pa·s)")
    save(fig, f"学習結果_{mat}.png")
    return p_med, info


def fig_parity(mat):
    p = pd.read_csv(os.path.join(model_dir(mat), "予測検証_3data_予測値.csv"), encoding="utf-8-sig")
    rmse = float(np.sqrt(np.mean((p["ln_eta"] - p["予測_ln_eta"]) ** 2)))
    cover = float(np.mean((p["eta"] >= p["下限95"]) & (p["eta"] <= p["上限95"])))
    lo = 10.0 ** np.floor(np.log10(min(p["eta"].min(), p["下限95"].min()) / 1.3))
    hi = 10.0 ** np.ceil(np.log10(max(p["eta"].max(), p["上限95"].max()) * 1.3))
    fig, ax = plt.subplots(figsize=mm(129, 116))
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.16, top=0.975)
    log_axes(ax, (lo, hi), (lo, hi))
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.3, color="0.3", zorder=1, label="予測値 ＝ 実測値")
    for g, col, mk, lab in [("低速側", "#1f4fe0", "o", "低射出率側"), ("高速側", "#e8212b", "s", "高射出率側")]:
        q = p[p["group"] == g]
        yerr = np.vstack([q["予測η"] - q["下限95"], q["上限95"] - q["予測η"]])
        ax.errorbar(q["eta"], q["予測η"], yerr=yerr, ls="", marker=mk, ms=6.5, mfc=col, mec="k", mew=0.5, ecolor=col,
                    elinewidth=1.2, capsize=0, alpha=0.85, label=lab, zorder=3)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=12.5, frameon=True, framealpha=0.9, borderaxespad=0.4, handletextpad=0.3)
    ax.text(0.97, 0.04, f"平均的な誤差：約 {r0(pct(rmse))}%\n95% 予測区間に入った割合：{cover * 100:.0f}%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=12.5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5"))
    ax.set_xlabel(r"実測値 $\eta$ (Pa·s)")
    ax.set_ylabel(r"予測値 $\eta$ (Pa·s)")
    save(fig, f"予測検証_{mat}.png")
    return rmse, cover, len(p)


def fig_shift_effect(rows):
    """温度シフトファクターで換算する前と後の、温度によるデータのずれ（厚みごとに 1 本の曲線を当てはめたときの、点のずれ。粘度での %）"""
    fig, ax = plt.subplots(figsize=mm(120, 76))
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.14, top=0.82)
    x = np.arange(len(rows))
    b0 = [pct(r["残差（換算前）"]) for r in rows]
    b1 = [pct(r["残差（換算後）"]) for r in rows]
    ax.bar(x - 0.19, b0, 0.36, color="0.70", edgecolor="k", lw=0.6, label="換算前")
    ax.bar(x + 0.19, b1, 0.36, color=BLUE, edgecolor="k", lw=0.6, label="換算後")
    for i in range(len(rows)):
        ax.text(x[i] - 0.19, b0[i] + 0.25, f"{b0[i]:.1f}", ha="center", va="bottom", fontsize=13)
        ax.text(x[i] + 0.19, b1[i] + 0.25, f"{b1[i]:.1f}", ha="center", va="bottom", fontsize=13, color=BLUE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MAT_JA[r["mat"]] for r in rows], fontsize=14)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.tick_params(axis="y", right=False)
    ax.set_ylabel("温度によるずれ (%)")
    ax.set_ylim(0, max(b0) * 1.22)
    ax.legend(loc="upper left", fontsize=13, frameon=False, ncol=2, bbox_to_anchor=(0.0, 1.17))
    ax.grid(True, axis="y", ls="-", lw=0.5, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    save(fig, "温度シフトの効果.png")


def fig_error_compare(rows):
    fig, ax = plt.subplots(figsize=mm(148, 98))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.12, top=0.86)
    x = np.arange(len(rows))
    g = [pct(r["RMSE 全体"]) for r in rows]
    c = [pct(r["Cross-WLF RMSE 全体"]) for r in rows]
    ax.bar(x - 0.19, g, 0.36, color=BLUE, edgecolor="k", lw=0.6, label="本モデル")
    ax.bar(x + 0.19, c, 0.36, color="0.70", edgecolor="k", lw=0.6, label="Cross-WLF モデル（参考）")
    for i in range(len(rows)):
        ax.text(x[i] - 0.19, g[i] + 1.5, f"{r0(g[i])}", ha="center", va="bottom", fontsize=13, color=BLUE, fontweight="bold")
        ax.text(x[i] + 0.19, c[i] + 1.5, f"{r0(c[i])}", ha="center", va="bottom", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([MAT_JA[r["mat"]] for r in rows], fontsize=14)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.tick_params(axis="y", right=False)
    ax.set_ylabel("平均的な予測誤差 (%)")
    ax.set_ylim(0, max(c) * 1.2)
    ax.legend(loc="upper left", fontsize=13, frameon=False, ncol=2, bbox_to_anchor=(0.0, 1.17))
    ax.grid(True, axis="y", ls="-", lw=0.5, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    save(fig, "予測誤差の比較.png")


def check_conditions(mat):
    """学習（5data）と検証（3data）の射出速度・厚みの組（ポスターの「学習データと検証データ」の記述の確認用）"""
    for grp in ("5data", "3data"):
        d = mc.D[(mat, grp)]
        v = sorted(d["v"].unique())
        lows = sorted(d.loc[d["v"] <= 5, "H"].unique())
        shots = d.groupby("cond").size()
        print(f"  {MAT_JA[mat]} {grp}：射出速度 {[int(a) for a in v]} mm/s（射出率 {[round(0.531 * a, 1) for a in v]} cm³/s）、"
              f"厚み {sorted(d['H'].unique())}、5 mm/s 以下の厚み {lows}、温度 {sorted(int(t) for t in d['Tset'].unique())}、"
              f"条件 {len(shots)}、1 条件のショット {shots.min()}〜{shots.max()}")


def main():
    if "--crops" in sys.argv or not all(os.path.exists(os.path.join(OUT, f"{n}.png")) for n in CROPS):
        crop_sample()
    fig_formula()
    fig_gpr_concept()
    fig_shift_concept()
    mc.INPUT_MODE = MODE
    mc.SHOW_FIGS = False
    mc.D = mc.load_all(MATS)
    rows = []
    for mat in MATS:
        check_conditions(mat)
        tr = mc.D[(mat, "5data")]
        te = mc.add_groups(tr, mc.D[(mat, "3data")])
        pool = pd.concat([tr, te], ignore_index=True)
        obj = joblib.load(os.path.join(model_dir(mat), f"学習済みモデル_{KEY}.joblib"))
        fit = obj["fit"]
        assert fit.get("mode") == MODE, f"{mc.MODE_JA[MODE]}のモデルのはず"
        cfg, th, tv = fit["cfg"], fit["th"], fit["tv"]
        r_before = mc.superpose_rmse(mc.shifted_cond_means(pool, None, cfg, 0.0, 0.0))
        r_after = mc.superpose_rmse(mc.shifted_cond_means(pool, KEY, cfg, th, tv))
        p_med, info = fig_flow(mat, tr, fit)
        rmse, cover, n = fig_parity(mat)
        met = pd.read_csv(os.path.join(compare_dir(mat), "予測検証_3data_指標.csv"), encoding="utf-8-sig")
        get = lambda m, rng: float(met[(met["モデル"] == m) & (met["範囲"] == rng)]["RMSE"].iloc[0])
        assert np.isclose(rmse, get(KEY, "全体")), f"{mat}：RMSE が指標の表と違う"
        summ = pd.read_csv(os.path.join(compare_dir(mat), "まとめ.csv"), encoding="utf-8-sig")
        dl = float(summ[summ["モデル"] == KEY]["θ=0 との LML の差"].iloc[0])
        dl_set = float(summ[summ["モデル"] == "B_Tset"]["θ=0 との LML の差"].iloc[0])
        rows.append({"mat": mat, "材料": MAT_JA[mat], "シフトの型": cfg["type"], "横の係数": th, "縦の係数": tv,
                     "残差（換算前）": r_before, "残差（換算後）": r_after, "比": r_after / r_before,
                     "LML の差（Tff）": dl, "LML の差（Tset）": dl_set,
                     "RMSE 全体": rmse, "RMSE 低速側": get(KEY, "低速側"), "RMSE 高速側": get(KEY, "高速側"),
                     "被覆率": cover, "検証の行数": n, "Cross-WLF RMSE 全体": get("Cross-WLF", "全体"),
                     "ΔP2 の中央値 [MPa]": p_med, "Tff の中央値 [℃]": "・".join(f"{t}:{v:.1f}" for t, v in info)})
    fig_shift_effect(rows)
    fig_error_compare(rows)
    tab = pd.DataFrame(rows)
    for c in ["RMSE 全体", "RMSE 低速側", "RMSE 高速側", "Cross-WLF RMSE 全体"]:
        tab[c.replace("RMSE", "誤差 [%]")] = [pct(v) for v in tab[c]]
    tab["ずれ 換算前 [%]"] = [pct(v) for v in tab["残差（換算前）"]]
    tab["ずれ 換算後 [%]"] = [pct(v) for v in tab["残差（換算後）"]]
    tab.drop(columns="mat").to_csv(os.path.join(OUT, "ポスターの数値.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(tab.drop(columns=["mat", "Tff の中央値 [℃]"]).round(3).T)
    print("保存先:", OUT)


if __name__ == "__main__":
    main()
