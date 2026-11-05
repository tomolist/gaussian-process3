# -*- coding: utf-8 -*-
"""簡略版のポスター（ポスター/報告書Bから作成_簡略版.pptx。A0 × 2 枚）の図を作る（日本語。学習はしない）。

2026-11-04 のユーザーの指摘（グラフの凡例・軸名や、図の中の文字が小さい など）を受けて、図は「ポスターに貼る大きさ（mm）」で作り、
図の中の文字の大きさが、そのままポスター上の pt になるようにした（凡例・注 22 pt 以上、目盛り 24 pt、軸名 28 pt）。
  樹脂圧力の経時変化.png ：報告書B の図3（作成中報告書B.pdf から 1000 dpi で切り出す。中身は変えない）
  実験結果_<材料>.png    ：報告書B の図4 の 3 枚のグラフを、材料ごとに切り出したもの（1000 dpi。下の英語の題は除く。中身は変えない）
  算出式.png             ：樹脂粘度とせん断速度の算出式
  温度シフト概念図.png   ：温度シフトファクターの考え方（横・縦のずれ）
  学習結果_<材料>.png    ：樹脂粘度とせん断速度の関係の学習結果（今のポスターの図と同じ中身。make_poster_figs.fig_flow と同じ描き方）
  パリティ_<材料>.png    ：学習に用いていない射出率条件（3data）の実測値と予測値（今のポスターの 予測検証_<材料>.png と同じ中身）
  予測誤差の比較.png     ：本モデルと Cross-WLF モデル（参考）の予測誤差
  学習検証の条件.csv     ：材料ごとの成形条件と、学習・検証の条件数と射出率（ポスターの表と「射出率による学習・検証の分け方」の図形の元）
モデルと数値は、今のポスター・報告書B と同じ（B_Tff・条件平均の入力）。数値が ポスター/図/ポスターの数値.csv と同じになることを確かめる。
（11-04 の最初の版の 成形条件と学習検証.png〔材料 × 厚み × 射出率のマスの図〕は、ユーザーの指摘〔場所を取りすぎ・文字が小さい〕で、
  ポスターの表と図形に替えたので、作らない。）
すでにあるファイルは上書きしない（1 つでもあれば止まる）。作り直すときは、前の図を _to_delete/ へ移すか、--out で別のフォルダに出す。

使い方（リポジトリ直下で）:
  uv run python 04_補助スクリプト/make_poster_simple_figs.py [--out <フォルダ>]   （既定：ポスター/図_簡略版/）
"""
import argparse
import os
import shutil
import subprocess
import sys

os.environ.setdefault("NB_REPO", ".")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch  # noqa: E402
from PIL import Image  # noqa: E402

import make_poster_figs as mpf  # noqa: E402  （model_dir・compare_dir・pct・r0。import しても図は作らない）
import make_report_B_figs as rb  # noqa: E402  （報告書B の元の PDF と、図3・図4 の範囲）
import mat_common as mc  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
MATS = ["PP", "MIX", "GPPS"]
MAT_JA = {"PP": "PP", "MIX": "粉砕材混合 PP", "GPPS": "GPPS"}
POSTER = os.path.join(mc.DATA_ROOT, "学習結果報告書作成_261015", "ポスター")
OUT_DEFAULT = os.path.join(POSTER, "図_簡略版")
DPI = 300
CROP_DPI = 1000
PAGE_DIR = os.path.join(mc.REPO, "_tmp", "poster_simple", "pages1000")    # 作成中報告書B.pdf の 2 ページ目の画像（1000 dpi）を置く作業フォルダ
RATE = 0.531                                   # 射出率 [cm³/s] = 0.531 × 射出速度 [mm/s]
BLUE = mpf.BLUE
T_COLORS = mpf.T_COLORS

# 図の大きさ [mm]（ポスターに貼る大きさ。build_poster_simple.py は、この大きさのまま貼る）
SIZE = {"flow": (262.0, 320.0), "parity": (262.0, 262.0), "error": (400.0, 300.0), "shift": (258.0, 150.0), "formula": (226.0, 84.0)}
AX = {"flow": [0.175, 0.115, 0.78, 0.87], "parity": [0.175, 0.145, 0.78, 0.835]}   # 軸の位置（図に対する割合。右端の目盛りの数字が切れないように右を空ける）
# 報告書B の図4 の 3 枚の範囲（作成中報告書B.pdf の 2 ページ目。左下が原点の bp：左 下 右 上）。
# 図4 全体の範囲（make_report_B_figs.CROPS）の中で、3 枚の間の白い列と、下の英語の題の上の白い行で区切った（_tmp/poster_simple/fig4_panels.py で探した）
FIG4_PANELS = {"PP": (68.6, 269.1, 222.7, 381.9), "MIX": (222.8, 269.1, 374.6, 381.9), "GPPS": (374.7, 269.1, 528.0, 381.9)}

plt.rcParams.update({"font.family": ["IPAexGothic", "DejaVu Sans"], "axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
                     "axes.labelweight": "normal", "font.size": 22, "axes.labelsize": 28, "xtick.labelsize": 24, "ytick.labelsize": 24, "legend.fontsize": 22,
                     "legend.title_fontsize": 22, "axes.linewidth": 1.6, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "xtick.major.width": 1.6, "ytick.major.width": 1.6,
                     "xtick.minor.width": 1.0, "ytick.minor.width": 1.0, "xtick.major.size": 10, "ytick.major.size": 10,
                     "xtick.minor.size": 5, "ytick.minor.size": 5, "xtick.major.pad": 8, "ytick.major.pad": 8})


def mm(w, h):
    return (w / 25.4, h / 25.4)


def fig_of(kind):
    return plt.figure(figsize=mm(*SIZE[kind]))


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=DPI)       # 大きさを変えない（bbox_inches は使わない）
    plt.close(fig)


# ---------------------------------------------------------------- 報告書B の図3・図4 の切り出し
def page_image():
    """作成中報告書B.pdf の 2 ページ目を、Ghostscript（TeX Live の rungs）で 1000 dpi の画像にする（あれば使う）"""
    png = os.path.join(PAGE_DIR, "p2.png")
    if not os.path.exists(png):
        gs = shutil.which("rungs")
        assert gs, "rungs（TeX Live の Ghostscript）が見つからない"
        os.makedirs(PAGE_DIR, exist_ok=True)
        subprocess.run([gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", f"-r{CROP_DPI}", "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
                        "-dFirstPage=2", "-dLastPage=2", f"-sOutputFile={png}", rb.SRC_PDF], check=True)
    return Image.open(png).convert("RGB")


def crop_trim(page, box_bp, path):
    """bp の範囲を切り出し、まわりの白い余白を 12 px を残して切り詰める"""
    s = CROP_DPI / 72.0
    llx, lly, urx, ury = box_bp
    im = page.crop((round(llx * s), round((rb.PAGE_H_BP - ury) * s), round(urx * s), round((rb.PAGE_H_BP - lly) * s)))
    g = np.asarray(im.convert("L"))
    ys, xs = np.where(g < 245)
    pad = 12
    im = im.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad + 1, im.width), min(ys.max() + pad + 1, im.height)))
    im.save(path, dpi=(CROP_DPI, CROP_DPI))
    print(f"  {os.path.basename(path)}：{im.width}×{im.height} px（元の大きさ {im.width / s * 25.4 / 72:.1f}×{im.height / s * 25.4 / 72:.1f} mm）")


def crops(out):
    page = page_image()
    crop_trim(page, rb.CROPS["Fig3"][1], os.path.join(out, "樹脂圧力の経時変化.png"))
    for mat, box in FIG4_PANELS.items():
        crop_trim(page, box, os.path.join(out, f"実験結果_{mat}.png"))


# ---------------------------------------------------------------- 説明の図
def fig_formula(out):
    fig = fig_of("formula")
    fs = 34
    fig.text(0.02, 0.70, r"$\eta = \dfrac{\Delta P_1\, B\, H^3}{12\, L\, Q}$", fontsize=fs, va="center")
    fig.text(0.56, 0.70, r"$\dot{\gamma} = \dfrac{6\,Q}{B\, H^2}$", fontsize=fs, va="center")
    fig.text(0.02, 0.18, r"流量：$Q = V H B$", fontsize=28, va="center")
    fig.text(0.56, 0.18, r"流速：$V = L\,/\,\Delta t$", fontsize=28, va="center")
    save(fig, out, "算出式.png")


def fig_shift_concept(out):
    """Cross 型の 2 本の曲線（低温・高温）と、横（せん断速度）・縦（粘度）方向のずれの矢印（make_poster_figs.fig_shift_concept と同じ中身で、文字を大きく。
    「1 本の曲線に重なる」の囲みは、文字を大きくすると「縦のずれ」と重なるので、図に入れず、ポスターの図の説明に書く）"""
    g = np.logspace(0, 5, 300)
    kh, kv = 5.0, 3.5
    cross = lambda g: 3000.0 / (1 + (3000.0 * g / 3e4) ** 0.72)
    lo, hi = cross(g), cross(g * kh) / kv
    fig = fig_of("shift")
    ax = fig.add_axes([0.075, 0.155, 0.915, 0.83])
    ax.plot(g, lo, color="#1f77b4", lw=4, label="低温")
    ax.plot(g, hi, color="#d62728", lw=4, ls="--", label="高温")
    gx = 12.0
    p0 = (gx, cross(gx * kh) / kv)
    p1 = (gx * kh, p0[1])
    p2 = (gx * kh, cross(gx * kh))
    for a, b in [(p0, p1), (p1, p2)]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=34, lw=3.5, color="#333333", zorder=6, shrinkA=0, shrinkB=2))
    arrow = dict(arrowstyle="->", lw=1.8, color="0.25", shrinkA=2, shrinkB=4)
    ax.annotate("横のずれ（せん断速度方向）", xy=(np.sqrt(p0[0] * p1[0]), p0[1]), xytext=(30.0, 6.5), ha="center", va="center", fontsize=24,
                arrowprops=arrow, zorder=7)
    ax.annotate("縦のずれ\n（粘度方向）", xy=(p1[0], np.sqrt(p1[1] * p2[1])), xytext=(330.0, 1100.0), ha="left", va="center", fontsize=24,
                arrowprops=arrow, zorder=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e5)
    ax.set_ylim(2, 9000)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.minorticks_off()
    ax.set_xlabel(r"せん断速度 $\dot{\gamma}$（対数）", fontsize=26, labelpad=6)
    ax.set_ylabel(r"樹脂粘度 $\eta$（対数）", fontsize=26, labelpad=6)
    ax.legend(loc="upper left", fontsize=24, frameon=True, framealpha=1.0, ncol=2, handlelength=2.6)
    save(fig, out, "温度シフト概念図.png")


# ---------------------------------------------------------------- 学習結果・予測の図
def log_axes(ax, xlim, ylim):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))
        a.set_major_formatter(ticker.LogFormatterSciNotation())
        a.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, which="major", ls="-", lw=1.0, color="0.75", zorder=0)
    ax.grid(True, which="minor", ls=":", lw=0.7, color="0.85", zorder=0)


def fig_flow(out, mat, tr, fit):
    """学習データの条件平均の点（色・形は厚み）と、加熱シリンダ温度ごとの予測平均値・95% 予測区間（ΔP2 は学習データの中央値、
    金型内樹脂最高温度は加熱シリンダ温度ごとの中央値に固定。make_poster_figs.fig_flow と同じ）"""
    xl, yl = (1.0, 1e5), (0.5, 3000.0)
    p_med = float(np.exp(tr["ln_P33"].median()) / 1e6)
    fig = fig_of("flow")
    ax = fig.add_axes(AX["flow"])
    log_axes(ax, xl, yl)
    lg = np.linspace(np.log(xl[0]), np.log(xl[1]), 500)
    h_t, info = [], []
    for i, (t, sub) in enumerate(tr.groupby("Tset")):
        tff = float(np.exp(sub["ln_Tff"].median()) - 273.15)
        mu, sd, _, _ = mc.predict(fit, mc.model_inputs(np.exp(lg), p_med, tff, float(t)))
        c = T_COLORS[i]
        ax.fill_between(np.exp(lg), np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd), color=c, alpha=0.22, lw=0, zorder=2)
        ax.plot(np.exp(lg), np.exp(mu), color=c, lw=3.0, zorder=4)
        h_t.append((Line2D([], [], color=c, lw=3.0), Patch(facecolor=c, alpha=0.22)))
        info.append((int(t), tff))
    h_h = []
    for h in sorted(tr["H"].unique()):
        a = tr[np.isclose(tr["H"], h)].groupby("cond").agg(lg=("ln_gd", "mean"), le=("ln_eta", "mean"))
        col, mk = mc.H_COLORS[float(h)], mc.H_MARKS[float(h)]
        ax.scatter(np.exp(a["lg"]), np.exp(a["le"]), s=130, color=col, marker=mk, alpha=0.85, edgecolors="k", linewidths=0.8, zorder=5)
        h_h.append(Line2D([], [], ls="", marker=mk, ms=13, color=col, mec="k", mew=0.8, label=f"{h:g} mm"))
    lg1 = ax.legend(h_t, [f"{t} ℃" for t, _ in info], handler_map={tuple: HandlerTuple(ndivide=None)}, title="加熱シリンダ温度",
                    loc="upper right", frameon=True, framealpha=0.92, borderaxespad=0.5, handlelength=2.4, labelspacing=0.3)
    ax.add_artist(lg1)
    ax.legend(handles=h_h, title="キャビティ厚み", loc="lower left", frameon=True, framealpha=0.92, borderaxespad=0.5, handletextpad=0.3,
              labelspacing=0.3)
    ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ (1/s)")
    ax.set_ylabel(r"樹脂粘度 $\eta$ (Pa·s)")
    save(fig, out, f"学習結果_{mat}.png")


def fig_parity(out, mat):
    """学習に用いていない射出率条件（3data）の実測値と予測値（縦線＝95% 予測区間。make_poster_figs.fig_parity と同じ中身）"""
    p = pd.read_csv(os.path.join(mpf.model_dir(mat), "予測検証_3data_予測値.csv"), encoding="utf-8-sig")
    rmse = float(np.sqrt(np.mean((p["ln_eta"] - p["予測_ln_eta"]) ** 2)))
    cover = float(np.mean((p["eta"] >= p["下限95"]) & (p["eta"] <= p["上限95"])))
    lo = 10.0 ** np.floor(np.log10(min(p["eta"].min(), p["下限95"].min()) / 1.3))
    hi = 10.0 ** np.ceil(np.log10(max(p["eta"].max(), p["上限95"].max()) * 1.3))
    fig = fig_of("parity")
    ax = fig.add_axes(AX["parity"])
    log_axes(ax, (lo, hi), (lo, hi))
    ax.plot([lo, hi], [lo, hi], ls="--", lw=2.0, color="0.3", zorder=1, label="予測値 ＝ 実測値")
    for g, col, mk, lab in [("低速側", "#1f4fe0", "o", "低射出率側"), ("高速側", "#e8212b", "s", "高射出率側")]:
        q = p[p["group"] == g]
        yerr = np.vstack([q["予測η"] - q["下限95"], q["上限95"] - q["予測η"]])
        ax.errorbar(q["eta"], q["予測η"], yerr=yerr, ls="", marker=mk, ms=12, mfc=col, mec="k", mew=0.8, ecolor=col,
                    elinewidth=2.0, capsize=0, alpha=0.85, label=lab, zorder=3)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, borderaxespad=0.5, handletextpad=0.3, labelspacing=0.3)
    ax.text(0.97, 0.04, f"平均的な誤差：約 {mpf.r0(mpf.pct(rmse))}%\n95% 予測区間に入った割合：{cover * 100:.0f}%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=22, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.5", lw=1.2))
    ax.set_xlabel(r"実測値 $\eta$ (Pa·s)")
    ax.set_ylabel(r"予測値 $\eta$ (Pa·s)")
    save(fig, out, f"パリティ_{mat}.png")
    return rmse, cover


def fig_error_compare(out, rows):
    fig = fig_of("error")
    ax = fig.add_axes([0.115, 0.09, 0.87, 0.79])
    x = np.arange(len(rows))
    g = [mpf.pct(r["RMSE 全体"]) for r in rows]
    c = [mpf.pct(r["Cross-WLF RMSE 全体"]) for r in rows]
    ax.bar(x - 0.19, g, 0.36, color=BLUE, edgecolor="k", lw=1.0, label="本モデル")
    ax.bar(x + 0.19, c, 0.36, color="0.70", edgecolor="k", lw=1.0, label="Cross-WLF モデル（参考）")
    for i in range(len(rows)):
        ax.text(x[i] - 0.19, g[i] + 1.5, f"{mpf.r0(g[i])}", ha="center", va="bottom", fontsize=26, color=BLUE, fontweight="bold")
        ax.text(x[i] + 0.19, c[i] + 1.5, f"{mpf.r0(c[i])}", ha="center", va="bottom", fontsize=26)
    ax.set_xticks(x)
    ax.set_xticklabels([MAT_JA[r["mat"]] for r in rows], fontsize=26)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.tick_params(axis="y", right=False)
    ax.set_ylabel("平均的な予測誤差 (%)")
    ax.set_ylim(0, max(c) * 1.2)
    ax.legend(loc="lower left", fontsize=24, frameon=False, ncol=2, bbox_to_anchor=(0.0, 1.01), borderaxespad=0.0, handlelength=1.6)
    ax.grid(True, axis="y", ls="-", lw=0.8, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    save(fig, out, "予測誤差の比較.png")


# ---------------------------------------------------------------- 成形条件と学習・検証の条件
def rate_label(v):
    r = RATE * v
    return f"{r:.1f}" if r < 100 else f"{r:.0f}"


def conditions_table(D):
    """材料ごとの成形条件と、学習・検証の条件数・射出率。射出率の分け方（両端が検証）が 3 材料で同じことも確かめる"""
    rows, splits = [], set()
    for mat in MATS:
        tr, te = D[(mat, "5data")], D[(mat, "3data")]
        v_tr, v_te = sorted(tr["v"].unique()), sorted(te["v"].unique())
        lo = [v for v in v_te if v < min(v_tr)]
        hi = [v for v in v_te if v > max(v_tr)]
        assert len(lo) + len(hi) == len(v_te), f"{mat}：検証の射出速度が学習の範囲の中にある"
        splits.add((tuple(lo), tuple(v_tr), tuple(hi)))
        n_tr, n_te = tr.groupby("cond").size(), te.groupby("cond").size()
        h_lo = sorted(float(h) for h in te.loc[te["v"].isin(lo), "H"].unique())
        rows.append({"材料": MAT_JA[mat], "加熱シリンダ温度": "・".join(str(int(t)) for t in sorted(tr["Tset"].unique())) + " ℃",
                     "キャビティ厚み": "・".join(f"{h:.1f}" for h in sorted(pd.concat([tr["H"], te["H"]]).unique())) + " mm",
                     "学習の条件数": int(tr["cond"].nunique()), "検証の条件数": int(te["cond"].nunique()),
                     "学習のショット": f"{n_tr.min()}〜{n_tr.max()}", "検証のショット": f"{n_te.min()}〜{n_te.max()}",
                     "射出率（学習）": ";".join(rate_label(v) for v in v_tr), "射出率（検証・低）": ";".join(rate_label(v) for v in lo),
                     "射出率（検証・高）": ";".join(rate_label(v) for v in hi),
                     "低射出率の検証の厚み": "・".join(f"{h:.1f}" for h in h_lo) + " mm"})
        print(f"  {MAT_JA[mat]}：学習 {rows[-1]['学習の条件数']} 条件（{rows[-1]['学習のショット']} ショット）、検証 {rows[-1]['検証の条件数']} 条件"
              f"（{rows[-1]['検証のショット']} ショット）、射出率 学習 {rows[-1]['射出率（学習）']}・検証 {rows[-1]['射出率（検証・低）']} | "
              f"{rows[-1]['射出率（検証・高）']}、低射出率の検証の厚み {rows[-1]['低射出率の検証の厚み']}")
    assert len(splits) == 1, "射出率の学習・検証の分け方が、材料によって違う（ポスターの図形は 3 材料共通で描いている）"
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- まとめて作る
NAMES = (["樹脂圧力の経時変化.png", "算出式.png", "温度シフト概念図.png", "予測誤差の比較.png", "学習検証の条件.csv"]
         + [f"{k}_{m}.png" for k in ("実験結果", "学習結果", "パリティ") for m in MATS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT, help="図を出すフォルダ（既定：ポスター/図_簡略版/）")
    a = ap.parse_args()
    there = [n for n in NAMES if os.path.exists(os.path.join(a.out, n))]
    if there:
        sys.exit(f"すでにある：{there}（上書きしない。前の図を _to_delete/ へ移すか、--out で別のフォルダを指定）")
    os.makedirs(a.out, exist_ok=True)
    print("報告書B の図3・図4 の切り出し")
    crops(a.out)
    fig_formula(a.out)
    fig_shift_concept(a.out)
    mc.INPUT_MODE = "cond"
    mc.SHOW_FIGS = False
    mc.D = mc.load_all(MATS)
    print("成形条件と学習・検証の条件")
    conditions_table(mc.D).to_csv(os.path.join(a.out, "学習検証の条件.csv"), index=False, encoding="utf-8-sig")
    ref = pd.read_csv(os.path.join(mpf.OUT, "ポスターの数値.csv"), encoding="utf-8-sig")
    rows = []
    for mat in MATS:
        tr = mc.D[(mat, "5data")]
        fit = joblib.load(os.path.join(mpf.model_dir(mat), f"学習済みモデル_{mpf.KEY}.joblib"))["fit"]
        assert fit.get("mode") == "cond", "条件平均の入力のモデルのはず"
        fig_flow(a.out, mat, tr, fit)
        rmse, cover = fig_parity(a.out, mat)
        met = pd.read_csv(os.path.join(mpf.compare_dir(mat), "予測検証_3data_指標.csv"), encoding="utf-8-sig")
        cw = float(met[(met["モデル"] == "Cross-WLF") & (met["範囲"] == "全体")]["RMSE"].iloc[0])
        r = ref[ref["材料"] == mpf.MAT_JA[mat]].iloc[0]
        assert np.isclose(rmse, r["RMSE 全体"]) and np.isclose(cover, r["被覆率"]) and np.isclose(cw, r["Cross-WLF RMSE 全体"]), \
            f"{mat}：数値が ポスターの数値.csv と違う"
        rows.append({"mat": mat, "RMSE 全体": rmse, "Cross-WLF RMSE 全体": cw})
        print(f"  {MAT_JA[mat]}：誤差 約 {mpf.r0(mpf.pct(rmse))}%（Cross-WLF 約 {mpf.r0(mpf.pct(cw))}%）、95% 予測区間に入った割合 {cover * 100:.0f}%")
    fig_error_compare(a.out, rows)
    for k, (w, h) in SIZE.items():
        print(f"  図の大きさ {k}：{w:.0f}×{h:.0f} mm")
    print("保存先:", a.out)


if __name__ == "__main__":
    main()
