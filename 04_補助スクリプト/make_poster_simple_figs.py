# -*- coding: utf-8 -*-
"""簡略版のポスター（ポスター/報告書Bから作成_簡略版.pptx）の新しい図を作る（日本語。学習はしない）。

  成形条件と学習検証.png：材料ごとの、キャビティ厚み × 射出率の成形条件と、学習（5data）・検証（3data）の分け方
                          （マスの色：学習・検証・データなし。データは報告書B・今のポスターと同じ通常版で、問題データ一覧の「除外」を除く）

ほかの図は、今のポスターの図（ポスター/図/。make_poster_figs.py が作ったもの）を、そのまま使う。
図の大きさは、ポスターに貼る大きさ（mm）で作る（文字の大きさがポスター上の pt になる）。
すでにあるファイルは上書きしない（あれば止まる）。作り直すときは、前の図を _to_delete/ へ移すか、--out で別のフォルダに出す。

使い方（リポジトリ直下で）:
  uv run python 04_補助スクリプト/make_poster_simple_figs.py [--out <フォルダ>]   （既定：ポスター/図_簡略版/）
"""
import argparse
import os
import sys

os.environ.setdefault("NB_REPO", ".")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import mat_common as mc  # noqa: E402

MATS = ["PP", "MIX", "GPPS"]
MAT_JA = {"PP": "PP", "MIX": "粉砕材混合 PP", "GPPS": "GPPS"}
POSTER = os.path.join(mc.DATA_ROOT, "学習結果報告書作成_261015", "ポスター")
OUT_DEFAULT = os.path.join(POSTER, "図_簡略版")
NAME = "成形条件と学習検証.png"
DPI = 300
RATE = 0.531                                   # 射出率 [cm³/s] = 0.531 × 射出速度 [mm/s]
C_TRAIN, C_TEST, C_NONE = "#9DC3E6", "#F4B183", "#E7E6E6"     # 学習（ポスターの「学習」と同じ薄い青）・検証（橙）・データなし（灰）
C_TRAIN_T, C_TEST_T = "#1F4E79", "#843C0C"                    # 文字の色

fm.findfont("IPAexGothic", fallback_to_default=False)
plt.rcParams.update({"font.family": ["IPAexGothic", "DejaVu Sans"], "axes.unicode_minus": False})


def rate_label(v):
    r = RATE * v
    return f"{r:.1f}" if r < 100 else f"{r:.0f}"


def conditions(D):
    """材料ごとの {(厚み, 射出速度): (組, 温度の数)} と、射出速度・厚みの一覧"""
    cells, speeds = {}, set()
    for mat in MATS:
        c = {}
        for grp in ("5data", "3data"):
            d = D[(mat, grp)]
            speeds |= set(d["v"].unique())
            for (h, v), sub in d.groupby(["H", "v"]):
                key = (float(h), float(v))
                assert key not in c, f"{mat}：厚み {h}・射出速度 {v} が学習と検証の両方にある"
                c[key] = (grp, sub["Tset"].nunique())
        cells[mat] = c
    return cells, sorted(speeds)


def report(D, cells):
    """図の中身の確かめ（表示だけ）"""
    for mat in MATS:
        for grp in ("5data", "3data"):
            d = D[(mat, grp)]
            n = d.groupby("cond").size()
            print(f"{MAT_JA[mat]} {grp}：条件 {d['cond'].nunique()}、温度 {sorted(int(t) for t in d['Tset'].unique())}、"
                  f"厚み {sorted(float(h) for h in d['H'].unique())}、射出速度 {sorted(int(v) for v in d['v'].unique())}、"
                  f"1 条件のショット {n.min()}〜{n.max()}（平均 {n.mean():.2f}）")
        part = {k: v for k, v in cells[mat].items() if v[1] < D[(mat, "5data")]["Tset"].nunique()}
        print(f"  温度がそろわないマス：{part if part else 'なし'}")


def draw(cells, speeds, path):
    W, LAB, CELL_H, GAP, TOP, BOT = 396.0, 96.0, 12.5, 4.0, 28.0, 16.0      # 図の幅・左の見出しの幅・マスの高さ・材料の間・上（射出率）・下（凡例）[mm]
    rows = [(mat, h) for mat in MATS for h in sorted({k[0] for k in cells[mat]} | ({0.5, 1.0, 1.5, 2.0} if mat != "GPPS" else {1.0, 1.5, 2.0}))]
    H = TOP + len(rows) * CELL_H + (len(MATS) - 1) * GAP + BOT
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)                                   # 上から下へ mm
    ax.axis("off")
    cw = (W - LAB) / len(speeds)
    fs, fs_s = 23, 19.5
    # 上：検証・学習の範囲と、射出率
    lo, hi = 2, 2                                       # 両端の検証の列の数（図の中身から決める）
    test_cols = [j for j, v in enumerate(speeds) if any(c.get((h, v), ("", 0))[0] == "3data" for c in cells.values() for h in (0.5, 1.0, 1.5, 2.0))]
    lo = sum(1 for j in test_cols if j < len(speeds) / 2)
    hi = len(test_cols) - lo
    for x0, n, lab, col in [(LAB, lo, "検証", C_TEST_T), (LAB + lo * cw, len(speeds) - lo - hi, "学習", C_TRAIN_T),
                            (LAB + (len(speeds) - hi) * cw, hi, "検証", C_TEST_T)]:
        ax.text(x0 + n * cw / 2, 5.0, lab, ha="center", va="center", fontsize=fs, color=col, fontweight="bold")
        ax.plot([x0 + 1.0, x0 + n * cw - 1.0], [10.0, 10.0], color=col, lw=2.0)
    for j, v in enumerate(speeds):
        ax.text(LAB + (j + 0.5) * cw, 18.5, rate_label(v), ha="center", va="center", fontsize=fs_s)
    ax.text(LAB - 3.0, 18.5, "射出率 (cm³/s)", ha="right", va="center", fontsize=fs_s)
    # マス
    y = TOP
    for i, mat in enumerate(MATS):
        hs = [h for m, h in rows if m == mat]
        ax.text(2.0, y + len(hs) * CELL_H / 2, MAT_JA[mat], ha="left", va="center", fontsize=fs, fontweight="bold")
        n_t = max(t for _, t in cells[mat].values())
        for h in hs:
            ax.text(LAB - 3.0, y + CELL_H / 2, f"{h:.1f} mm", ha="right", va="center", fontsize=fs_s)
            for j, v in enumerate(speeds):
                grp, nt = cells[mat].get((h, v), ("", 0))
                col = {"5data": C_TRAIN, "3data": C_TEST}.get(grp, C_NONE)
                ax.add_patch(Rectangle((LAB + j * cw, y), cw, CELL_H, facecolor=col, edgecolor="white", lw=2.0))
                if grp and nt < n_t:
                    ax.text(LAB + (j + 0.5) * cw, y + CELL_H / 2, f"{nt}温度", ha="center", va="center", fontsize=16)
            y += CELL_H
        if i < len(MATS) - 1:
            y += GAP
    # 下：凡例
    yl = y + 4.0
    x = LAB
    for col, lab in [(C_TRAIN, "学習データ"), (C_TEST, "検証データ"), (C_NONE, "データなし（薄肉・低射出率で成形できない条件など）")]:
        ax.add_patch(Rectangle((x, yl), 9.0, 7.5, facecolor=col, edgecolor="0.4", lw=1.0))
        ax.text(x + 11.0, yl + 3.75, lab, ha="left", va="center", fontsize=fs_s)
        x += 11.0 + mc_text_width(lab, fs_s) + 9.0
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"図の大きさ：{W:.0f}×{H:.1f} mm")


def mc_text_width(text, size):
    """文字列の幅の目安 [mm]（全角 1 字 = 1 em、半角 = 0.55 em）"""
    return sum(1.0 if ord(c) > 0x2000 else 0.55 for c in text) * size * 25.4 / 72.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT, help="図を出すフォルダ（既定：ポスター/図_簡略版/）")
    a = ap.parse_args()
    path = os.path.join(a.out, NAME)
    if os.path.exists(path):
        sys.exit(f"すでにある：{path}（上書きしない。前の図を _to_delete/ へ移すか、--out で別のフォルダを指定）")
    os.makedirs(a.out, exist_ok=True)
    mc.INPUT_MODE = "cond"
    mc.SHOW_FIGS = False
    D = mc.load_all(MATS)
    cells, speeds = conditions(D)
    report(D, cells)
    draw(cells, speeds, path)
    print("保存:", path)


if __name__ == "__main__":
    main()
