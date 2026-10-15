# -*- coding: utf-8 -*-
"""報告書B（05_…/学習結果報告書作成_261015/報告書B_完成版.tex）の図を作る（英語。学習はやり直さない）。

図1〜5：作成中報告書B.pdf から 600 dpi の PNG で切り出す（図/Fig1.png〜Fig5.png。なければ作る。--crops で作り直す）
検証の図（報告書の図6）：学習に用いていない射出率条件（3data）の、実測と予測の粘度（B_Tff・条件平均の入力。縦線＝95% 予測区間）
マスターカーブの図（報告書では既定で載せない）：温度シフトファクターで換算した条件平均の点（B_Tff・条件平均の入力。塗り＝学習 5data、白抜き＝検証 3data）

使い方（リポジトリ直下で）: uv run python 04_補助スクリプト/make_report_B_figs.py [--crops]
出力: 05_材料ごとの機械学習データと予測検証データ/学習結果報告書作成_261015/図/ の Fig1〜5.png、Parity_<材料>.pdf、MasterCurve_<材料>.pdf
（PDF には確認用の PNG も付ける）
報告書B の図に合わせて、図の文字は英語にする（CLAUDE.md §8 の「図は日本語」の例外。ユーザーの報告書の書式に合わせるため）。
"""
import os
import sys

os.environ.setdefault("NB_REPO", ".")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import mat_common as mc  # noqa: E402

MATS = ["PP", "MIX", "GPPS"]
KEY = "B_Tff"
MODE_DIR = "条件平均の入力"
OUT = os.path.join(mc.DATA_ROOT, "学習結果報告書作成_261015", "図")
PANEL = (2.15, 1.80)                      # 1 枚の大きさ（インチ）。報告書では 3 枚を横に並べる
H_STYLE = {0.5: "#e8212b", 1.0: "#1f4fe0", 1.5: "#f2b705", 2.0: "#22c32e"}   # 厚みの色（報告書B の図4 と同じ並び：赤・青・黄・緑）
T_MARKS = ["o", "^", "s"]                 # 加熱シリンダ温度の形（報告書B の図4 と同じ：低い順に ●・▲・■）

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7.5, "axes.labelweight": "bold",
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 5.6, "axes.linewidth": 0.6,
                     "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
                     "mathtext.fontset": "dejavusans", "pdf.fonttype": 42})


def model_dir(mat):
    folder = mc.MAT_FOLDERS[mat]
    return os.path.join(mc.DATA_ROOT, folder, f"{folder}_学習結果", MODE_DIR, KEY)


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{stem}.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, f"{stem}.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def decade_lim(lo, hi):
    """データの範囲を少し広げて、10 の累乗で切った軸の範囲（点が軸の端に掛からないように）"""
    return 10.0 ** np.floor(np.log10(lo / 1.3)), 10.0 ** np.ceil(np.log10(hi * 1.3))


SRC_PDF = os.path.join(mc.DATA_ROOT, "学習結果報告書作成_261015", "作成中報告書B.pdf")
CROP_DPI = 600
PAGE_H_BP = 841.89                        # A4 の高さ（bp）
CROPS = {                                 # 図1〜5 の範囲（作成中報告書B.pdf のページ、左下が原点の bp：左 下 右 上。図の題は含めない）
    "Fig1": (1, (310.0, 256.9, 520.0, 405.4)), "Fig2": (1, (310.5, 84.9, 523.0, 237.9)), "Fig3": (2, (306.0, 399.4, 522.5, 545.4)),
    "Fig4": (2, (68.5, 254.4, 528.0, 381.9)), "Fig5": (2, (68.5, 83.9, 527.0, 233.4))}


def crop_source_figures():
    """作成中報告書B.pdf の図1〜5 を、600 dpi の PNG として切り出す（ページをそのまま貼ると、黒塗りの下の文字などページ全体の中身が
    PDF に残るため、画像にする）。ページの画像は Ghostscript（TeX Live の rungs）で _tmp/report_B/pages600/ に作る"""
    import shutil
    import subprocess

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    gs = shutil.which("rungs")
    assert gs, "rungs（TeX Live の Ghostscript）が見つからない"
    work = os.path.join(mc.REPO, "_tmp", "report_B", "pages600")
    os.makedirs(work, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    pages = {}
    for p in sorted({v[0] for v in CROPS.values()}):
        png = os.path.join(work, f"p{p}.png")
        subprocess.run([gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", f"-r{CROP_DPI}", "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
                        f"-dFirstPage={p}", f"-dLastPage={p}", f"-sOutputFile={png}", SRC_PDF], check=True)
        pages[p] = Image.open(png).convert("RGB")
    s = CROP_DPI / 72.0
    for name, (p, (llx, lly, urx, ury)) in CROPS.items():
        box = (round(llx * s), round((PAGE_H_BP - ury) * s), round(urx * s), round((PAGE_H_BP - lly) * s))
        im = pages[p].crop(box)
        im.save(os.path.join(OUT, f"{name}.png"), dpi=(CROP_DPI, CROP_DPI))
        print(f"{name}：{im.width}×{im.height} px（{im.width / s * 25.4 / 72:.1f} × {im.height / s * 25.4 / 72:.1f} mm）")


def main():
    if "--crops" in sys.argv or not all(os.path.exists(os.path.join(OUT, f"{n}.png")) for n in CROPS):
        crop_source_figures()
    mc.INPUT_MODE = "cond"
    mc.SHOW_FIGS = False
    mc.D = mc.load_all(MATS)
    shift_csv = []
    for mat in MATS:
        tr = mc.D[(mat, "5data")]
        te = mc.add_groups(tr, mc.D[(mat, "3data")])
        pool = pd.concat([tr, te], ignore_index=True)
        fit = joblib.load(os.path.join(model_dir(mat), f"学習済みモデル_{KEY}.joblib"))["fit"]
        cfg, th, tv = fit["cfg"], fit["th"], fit["tv"]
        assert fit.get("mode") == "cond", "条件平均の入力のモデルのはず"

        # ---------------- 図6：マスターカーブ
        t0 = mc.shifted_cond_means(pool, None, cfg, 0.0, 0.0)
        t1 = mc.shifted_cond_means(pool, KEY, cfg, th, tv)
        r0, r1 = mc.superpose_rmse(t0), mc.superpose_rmse(t1)
        ref = pd.read_csv(os.path.join(mc.DATA_ROOT, mc.MAT_FOLDERS[mat], f"{mc.MAT_FOLDERS[mat]}_学習結果", MODE_DIR, "比較",
                                       "シフトファクターの倍率.csv"), encoding="utf-8-sig")
        ref = ref[ref["モデル"] == KEY].iloc[0]
        assert np.isclose(r0, ref["重ね合わせの残差（シフト前）"]) and np.isclose(r1, ref["重ね合わせの残差（シフト後）"]), \
            f"{mat}：重ね合わせの残差が説明書の値と違う（{r0:.4f}→{r1:.4f}）"
        shift_csv.append({"材料": mc.MAT_NAMES[mat], "横の係数": th, "縦の係数": tv, "残差（換算前）": r0, "残差（換算後）": r1, "比": r1 / r0})
        train_c = set(tr["cond"])
        temps = sorted(pool["Tset"].unique())
        Hs = sorted(pool["H"].unique())
        fig, ax = plt.subplots(figsize=PANEL)
        for i, T in enumerate(temps):
            for h in Hs:
                m = (t1["Tset"] == T) & (t1["H"] == h)
                trm, tem = m & t1["cond"].isin(train_c), m & ~t1["cond"].isin(train_c)
                ax.scatter(np.exp(t1.loc[trm, "x"]), np.exp(t1.loc[trm, "y"]), s=11, marker=T_MARKS[i], color=H_STYLE[h],
                           edgecolors="k", linewidths=0.3, zorder=3)
                ax.scatter(np.exp(t1.loc[tem, "x"]), np.exp(t1.loc[tem, "y"]), s=11, marker=T_MARKS[i], facecolors="white",
                           edgecolors=H_STYLE[h], linewidths=0.8, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*decade_lim(np.exp(t1["x"]).min(), np.exp(t1["x"]).max()))
        ax.set_ylim(*decade_lim(np.exp(t1["y"]).min(), np.exp(t1["y"]).max()))
        ax.set_xlabel(r"Reduced shear rate  $\dot{\gamma}\,a_h$ (1/s)")
        ax.set_ylabel(r"Reduced viscosity  $\eta\,/\,a_v$ (Pa・s)".replace("・", "·"))
        ax.grid(True, which="major", ls="-", lw=0.3, color="0.8")
        ax.grid(True, which="minor", ls=":", lw=0.25, color="0.88")
        h_t = [Line2D([], [], ls="", marker=T_MARKS[i], ms=3.2, mfc="0.35", mec="k", mew=0.3, label=f"{int(T)} °C") for i, T in enumerate(temps)]
        h_h = [Line2D([], [], ls="", marker="s", ms=3.2, mfc=H_STYLE[h], mec=H_STYLE[h], label=f"{h:.1f} mm") for h in Hs]
        lg1 = ax.legend(handles=h_t, title="Cylinder temperature", loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False,
                        title_fontsize=5.6, handletextpad=0.2, borderaxespad=0.3, labelspacing=0.15)
        ax.add_artist(lg1)
        ax.legend(handles=h_h, title="Cavity thickness", loc="upper right", bbox_to_anchor=(1.0, 0.70), frameon=False,
                  title_fontsize=5.6, handletextpad=0.2, borderaxespad=0.3, labelspacing=0.15)
        ax.text(0.04, 0.05, f"Superposition residual\n{r0:.3f} (before) → {r1:.3f} (after)", transform=ax.transAxes, fontsize=5.6,
                va="bottom", ha="left")
        save(fig, f"MasterCurve_{mat}")

        # ---------------- 図7：検証データの実測と予測
        p = pd.read_csv(os.path.join(model_dir(mat), "予測検証_3data_予測値.csv"), encoding="utf-8-sig")
        rmse = float(np.sqrt(np.mean((p["ln_eta"] - p["予測_ln_eta"]) ** 2)))
        met = pd.read_csv(os.path.join(os.path.dirname(model_dir(mat)), "比較", "予測検証_3data_指標.csv"), encoding="utf-8-sig")
        ref_rmse = float(met[(met["モデル"] == KEY) & (met["範囲"] == "全体")]["RMSE"].iloc[0])
        assert np.isclose(rmse, ref_rmse), f"{mat}：RMSE が指標の表と違う（{rmse:.4f} と {ref_rmse:.4f}）"
        cover = float(np.mean((p["eta"] >= p["下限95"]) & (p["eta"] <= p["上限95"])))
        fig, ax = plt.subplots(figsize=PANEL)
        lo, hi = decade_lim(min(p["eta"].min(), p["下限95"].min()), max(p["eta"].max(), p["上限95"].max()))
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.7, color="0.3", zorder=1)
        for g, col, mk, lab in [("低速側", "#1f4fe0", "o", "Low injection rate"), ("高速側", "#e8212b", "s", "High injection rate")]:
            q = p[p["group"] == g]
            yerr = np.vstack([q["予測η"] - q["下限95"], q["上限95"] - q["予測η"]])
            ax.errorbar(q["eta"], q["予測η"], yerr=yerr, ls="", marker=mk, ms=2.8, mfc=col, mec="k", mew=0.3, ecolor=col,
                        elinewidth=0.6, capsize=0, alpha=0.9, label=lab, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("Measured viscosity (Pa·s)")
        ax.set_ylabel("Predicted viscosity (Pa·s)")
        ax.grid(True, which="major", ls="-", lw=0.3, color="0.8")
        ax.grid(True, which="minor", ls=":", lw=0.25, color="0.88")
        ax.legend(loc="upper left", frameon=False, handletextpad=0.2, borderaxespad=0.3, labelspacing=0.2)
        ax.text(0.97, 0.04, f"RMSE (ln η) = {rmse:.3f}\nIn 95% interval: {cover * 100:.0f}%", transform=ax.transAxes,
                fontsize=5.6, va="bottom", ha="right")
        save(fig, f"Parity_{mat}")
        print(f"{mc.MAT_NAMES[mat]}：θ_h {th:g}・θ_v {tv:g}、重ね合わせの残差 {r0:.3f} → {r1:.3f}（比 {r1 / r0:.2f}）、"
              f"3data の RMSE {rmse:.3f}、95% 予測区間に入った割合 {cover:.2f}（{len(p)} 行）")
    print("保存先:", OUT)


if __name__ == "__main__":
    main()
