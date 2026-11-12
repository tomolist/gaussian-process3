# -*- coding: utf-8 -*-
"""A0 1 枚の簡略版ポスター（報告書Bから作成/報告書Bから作成_簡略版2.pptx。build_poster_one.py）の 4 節のグラフを作る（日本語。学習はしない）。

2026-11-05 のユーザーの依頼：文字はグラフの中も 28 pt 以上。4 節は報告書Bから作成の 4 節（箇条書きを除く）にし、
樹脂粘度とせん断速度のグラフ 3 つは、グラフエリアを正方形にして横に並べる。縦軸の名前と目盛りの数字は左端だけ（ユーザーの回答）。
グラフの下の題は日本語の材料名（バージンPP・リサイクルPP・GPPS）。
図は「ポスターに貼る大きさ（mm）」・300 dpi で作るので、図の中の文字の pt が、そのままポスター上の pt になる（28 pt 未満の文字があれば止まる。
上付き・下付き〔10ⁿ の指数、ΔP₁ など〕は matplotlib が自動で小さく描くので、例外）。

  樹脂圧力の経時変化.png        ：報告書B の図3 を作り直したもの。作成中報告書B.pdf の 2 ページ目（ベクター）から、pdftocairo の SVG を通して
                                  波形と注記（ΔP2・Δt・ΔP1）の位置を取り出して描く。拡大図は、軸の名前を省き、目盛りの数字を 3 つに減らした。
                                  28 pt の文字が線に重ならないように、ΔP2 の文字は縦書き、ΔP1 の矢印は②の立ち上がりの時刻の縦の線の上に置いた
  樹脂温度の経時変化.png        ：見本（報告書Aから作成2）の樹脂温度のグラフを作り直したもの。見本の PDF の中でも画像なので、
                                  ポスター/図/見本_樹脂温度の経時変化.png から黒い曲線を読み取って描く。中の断面図（文字 約 10 pt）は外した
  算出式.png                    ：樹脂粘度とせん断速度の算出式
  樹脂粘度とせん断速度の関係.png ：報告書B の図4 を作り直したもの（3 材料。グラフエリアは正方形）。点とキャピラリーレオメータの値は、
                                  作成中報告書B.pdf の図4 から取り出したもの（PP は 1 条件 10 ショットの全データで、05 フォルダの 5data・3data より多い）。
                                  取り出した点は、リポジトリのデータ（5data・3data のショットごとの値）と照合する（印の中心：丸・四角は外接する四角の中心、
                                  三角は重心）。横軸の数字は 10^0・10^2・10^4 だけ（目盛りの線は 1 桁ごと）
  図データ_*.csv                ：取り出したデータ（あとで確かめられるように）
すでにあるファイルは上書きしない（1 つでもあれば止まる）。作り直すときは、前の図を _to_delete/ へ移すか、--out で別のフォルダに出す。

使い方（リポジトリ直下で）:
  uv run python 04_補助スクリプト/make_poster_one_figs.py [--out <フォルダ>]   （既定：ポスター/報告書Bから作成/図_簡略版_A0一枚/）
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("NB_REPO", ".")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text as mtext  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

import make_report_B_figs as rb  # noqa: E402  （作成中報告書B.pdf の場所）
import mat_common as mc  # noqa: E402

POSTER = os.path.join(mc.DATA_ROOT, rb.REPORT_DIR, "ポスター")   # 2026-11-12 にユーザーが 06_学習結果報告書作成_261015 に名前を変えた
B_DIR = os.path.join(POSTER, "報告書Bから作成")
OUT_DEFAULT = os.path.join(B_DIR, "図_簡略版_A0一枚")
SAMPLE_T = os.path.join(POSTER, "図", "見本_樹脂温度の経時変化.png")
WORK = os.path.join(mc.REPO, "_tmp", "poster_one")          # SVG を置く作業フォルダ
DPI = 300
MIN_PT = 28                                                   # 図の中の文字の大きさの下限（ユーザーの指示）
TICK, LABEL = 28, 30                                          # 目盛りの数字、軸の名前

# 図の大きさ [mm]（ポスターに貼る大きさ。build_poster_one.py は、この大きさのまま貼る。列の中身の幅は 399.5 mm）
COL_W = 399.5
# 圧力と温度は横に並べる（間 7.5 mm）。高さは、ポスターの 3 節と 4 節の下端をそろえるために 140 → 130 mm にした（2026-11-05）
SIZE = {"pressure": (224.0, 130.0), "temperature": (168.0, 130.0), "formula": (190.0, 72.0)}

MATS = ["PP", "MIX", "GPPS"]
MAT_TITLE = {"PP": "(1) バージンPP", "MIX": "(2) リサイクルPP", "GPPS": "(3) GPPS"}       # グラフの下の題（ユーザーの指定の材料名）
# 報告書B の図4 の 3 枚の範囲（作成中報告書B.pdf の 2 ページ目。左下が原点の bp：左 下 右 上。make_poster_simple_figs.FIG4_PANELS と同じ）
FIG4_PANELS = {"PP": (68.6, 269.1, 222.7, 381.9), "MIX": (222.8, 269.1, 374.6, 381.9), "GPPS": (374.7, 269.1, 528.0, 381.9)}
FIG3_BOX = rb.CROPS["Fig3"][1]                                # 報告書B の図3 の範囲（同じ書き方）
# 図4 の点の色 → キャビティ厚み [mm]、形 → 加熱シリンダ温度の水準（0・1・2 番目）
H_OF_COLOR = {"rgb(100%,0%,0%)": 0.5, "rgb(0%,0%,100%)": 1.0, "rgb(94.099426%,73.298645%,0.782776%)": 1.5, "rgb(0%,100%,0%)": 2.0}
H_COLOR = {0.5: "#FF0000", 1.0: "#0000FF", 1.5: "#F0BB02", 2.0: "#00FF00"}                 # 報告書B の図4 の色（build_poster_simple.FIG4_THICK と同じ）
SHAPES = ["circle", "triangle", "square"]
MARK = {"circle": "o", "triangle": "^", "square": "s"}
TEMPS = {"PP": [190, 210, 230], "MIX": [190, 210, 230], "GPPS": [200, 220, 240]}
CAP_T = {"PP": 210, "MIX": 210, "GPPS": 220}                                                 # キャピラリーレオメータの温度（報告書B の図4 の凡例）

plt.rcParams.update({"font.family": ["IPAexGothic", "DejaVu Sans"], "axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
                     "font.size": TICK, "axes.labelsize": LABEL, "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": TICK,
                     "legend.title_fontsize": TICK, "axes.linewidth": 1.6, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "xtick.major.width": 1.6, "ytick.major.width": 1.6,
                     "xtick.minor.width": 1.0, "ytick.minor.width": 1.0, "xtick.major.size": 10, "ytick.major.size": 10,
                     "xtick.minor.size": 5, "ytick.minor.size": 5, "xtick.major.pad": 8, "ytick.major.pad": 8})


def mm(w, h):
    return (w / 25.4, h / 25.4)


def axes_mm(fig, w, h, left, bottom, aw, ah):
    """図の左下から left・bottom [mm] の位置に、幅 aw・高さ ah [mm] の軸を置く"""
    return fig.add_axes([left / w, bottom / h, aw / w, ah / h])


def check_fonts(fig, name):
    """図の中の見えている文字が、すべて 28 pt 以上かを確かめる"""
    fig.canvas.draw()
    small = sorted({(t.get_text(), t.get_fontsize()) for t in fig.findobj(mtext.Text)
                    if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_PT - 1e-9})
    assert not small, f"{name}：{MIN_PT} pt 未満の文字がある {small}"


def save(fig, out, name):
    check_fonts(fig, name)
    fig.savefig(os.path.join(out, name), dpi=DPI)       # 大きさを変えない（bbox_inches は使わない）
    plt.close(fig)


# ================================================================ PDF（SVG）からの取り出し
NS = "{http://www.w3.org/2000/svg}"


def _mul(a, b):
    return (a[0] * b[0] + a[2] * b[1], a[1] * b[0] + a[3] * b[1], a[0] * b[2] + a[2] * b[3], a[1] * b[2] + a[3] * b[3],
            a[0] * b[4] + a[2] * b[5] + a[4], a[1] * b[4] + a[3] * b[5] + a[5])


def _tf(s):
    m = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    for name, args in re.findall(r"(\w+)\(([^)]*)\)", s or ""):
        v = [float(x) for x in re.split(r"[ ,]+", args.strip()) if x]
        t = {"matrix": lambda: tuple(v), "translate": lambda: (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0),
             "scale": lambda: (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)}[name]()
        m = _mul(m, t)
    return m


def _cmds(d):
    """path の d を [(命令, (x, y)), ...] にする（pdftocairo の出力は M・L・C・Z だけ）"""
    toks = re.findall(r"[MLCZ]|-?[\d.]+(?:e-?\d+)?", d)
    out, i, cmd = [], 0, None
    while i < len(toks):
        t = toks[i]
        if t in "MLCZ":
            cmd = t
            i += 1
            if cmd == "Z":
                out.append(("Z", None))
            continue
        if cmd in ("M", "L"):
            out.append((cmd, (float(toks[i]), float(toks[i + 1]))))
            i += 2
        elif cmd == "C":
            out.append(("C", (float(toks[i + 4]), float(toks[i + 5]))))
            i += 6
        else:
            raise ValueError(f"想定にない path の命令：{cmd}")
    return out


def svg_paths(svg):
    """SVG の本文の path を、ページの座標（pt。左上が原点）の点と、線・塗りの書式にして返す（defs の中の字形などは除く）"""
    root = ET.parse(svg).getroot()
    page_h = float(root.get("viewBox").split()[3])
    out = []

    def walk(el, m):
        tag = el.tag.replace(NS, "")
        if tag in ("defs", "clipPath", "symbol", "mask", "pattern"):
            return
        m2 = _mul(m, _tf(el.get("transform")))
        if tag == "path":
            st = dict(kv.split(":", 1) for kv in el.get("style", "").split(";") if ":" in kv)
            cm = _cmds(el.get("d", ""))
            pts = np.array([(m2[0] * x + m2[2] * y + m2[4], m2[1] * x + m2[3] * y + m2[5]) for c, p in cm if p is not None for x, y in [p]])
            if len(pts):
                out.append({"stroke": st.get("stroke", "none"), "fill": st.get("fill", "none"), "width": st.get("stroke-width", ""),
                            "cmds": [c for c, _ in cm], "pts": pts})
        for c in el:
            walk(c, m2)

    for c in root:
        walk(c, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    return out, page_h


def page_svg(pdf, page, name):
    """PDF の 1 ページを、pdftocairo（TeX Live）で SVG にする（作業フォルダにあれば使う）"""
    path = os.path.join(WORK, name)
    if not os.path.exists(path):
        tool = shutil.which("pdftocairo")
        assert tool, "pdftocairo（TeX Live）が見つからない"
        os.makedirs(WORK, exist_ok=True)
        subprocess.run([tool, "-svg", "-f", str(page), "-l", str(page), pdf, path], check=True)
    return path


def in_box(items, page_h, box):
    l, b, r, t = box
    y0, y1 = page_h - t, page_h - b
    return [it for it in items if it["pts"][:, 0].min() >= l and it["pts"][:, 0].max() <= r and it["pts"][:, 1].min() >= y0 and it["pts"][:, 1].max() <= y1]


def straight_lines(items, stroke, width):
    """指定の線の、2 点の水平・垂直の直線：横は (x0, x1, y)、縦は (x, y0, y1)"""
    hl, vl = [], []
    for it in items:
        if it["stroke"] != stroke or it["width"] != width or len(it["pts"]) != 2:
            continue
        (xa, ya), (xb, yb) = it["pts"]
        if abs(ya - yb) < 0.01:
            hl.append((min(xa, xb), max(xa, xb), ya))
        elif abs(xa - xb) < 0.01:
            vl.append((xa, min(ya, yb), max(ya, yb)))
    return hl, vl


def frame_of(hl, vl, rank=0):
    """長い順に rank 番目の枠（左 右 上 下。ページの座標）"""
    lens = sorted({round(x1 - x0, 2) for x0, x1, _ in hl}, reverse=True)
    L = lens[rank]
    hs = sorted([h for h in hl if abs((h[1] - h[0]) - L) < 0.05], key=lambda h: h[2])
    top, bot = hs[0][2], hs[-1][2]
    left, right = hs[0][0], hs[0][1]
    vs = [v for v in vl if abs(v[1] - top) < 0.05 and abs(v[2] - bot) < 0.05]
    assert len(vs) >= 2, "枠の縦の線が見つからない"
    return left, right, top, bot


# ---------------------------------------------------------------- 報告書B の図4（樹脂粘度とせん断速度の関係）
def extract_fig4(svg):
    """材料ごとに、点（せん断速度・粘度・厚み・温度の水準）と、キャピラリーレオメータの値と、軸の範囲を取り出す"""
    items, page_h = svg_paths(svg)
    pts_all, cap_all, lims = [], [], {}
    for mat, box in FIG4_PANELS.items():
        its = in_box(items, page_h, box)
        hl, vl = straight_lines(its, "rgb(0%,0%,0%)", "1.1241")
        left, right, top, bot = frame_of(hl, vl)
        # 横軸：下の枠の目盛り（長い＝10 の累乗）。左端が 10^0（報告書B の図4 の目盛りの数字）
        xmaj = sorted(x for x, y0, y1 in vl if abs(y1 - bot) < 0.05 and (y1 - y0) > 1.0 and (y1 - y0) < 5.0)
        assert len(xmaj) == 6 and abs(xmaj[0] - left) < 0.05 and abs(xmaj[-1] - right) < 0.05, f"{mat}：横軸の目盛りが想定（10^0〜10^5）と違う"
        dx = (right - left) / 5.0                                       # 1 桁の長さ（枠の幅から）
        # 縦軸：左の枠の目盛り（長い＝10 の累乗）。いちばん下の長い目盛りが 10^0（同）。1 桁の長さは、目盛りの位置の直線の当てはめから
        ymaj = np.array(sorted((y for x0, x1, y in hl if abs(x0 - left) < 0.05 and 1.0 < (x1 - x0) < 5.0), reverse=True))
        k = np.arange(len(ymaj))
        dy, y_one = -np.polyfit(k, ymaj, 1)[0], float(np.polyval(np.polyfit(k, ymaj, 1), 0))
        to_x = lambda px: 10.0 ** ((px - left) / dx)
        to_y = lambda py: 10.0 ** ((y_one - py) / dy)
        lims[mat] = ((to_x(left), to_x(right)), (to_y(bot), to_y(top)))
        n0 = len(pts_all)
        # 点：線だけの色つきの印（凡例の印は塗りつぶしなので入らない）
        for it in its:
            if it["fill"] != "none" or it["stroke"] not in H_OF_COLOR or it["width"] != "2.2482":
                continue
            c = it["cmds"]
            shape = "circle" if "C" in c else {2: "triangle", 3: "square"}.get(c.count("L"))
            assert shape, f"{mat}：想定にない形の点（{c}）"
            p = it["pts"]
            if shape == "triangle":                                     # 三角は重心がデータの位置（リポジトリのデータとの照合で確かめた）
                cx, cy = p[:3, 0].mean(), p[:3, 1].mean()
            else:                                                       # 丸・四角は、外接する四角の中心
                cx, cy = (p[:, 0].min() + p[:, 0].max()) / 2, (p[:, 1].min() + p[:, 1].max()) / 2
            pts_all.append({"材料": mat, "せん断速度": to_x(cx), "樹脂粘度": to_y(cy), "厚み": H_OF_COLOR[it["stroke"]],
                            "加熱シリンダ温度": TEMPS[mat][SHAPES.index(shape)], "形": shape})
        # キャピラリーレオメータ：白で塗った黒い丸と、それを結ぶ黒い線（凡例の見本は、水平の線の真ん中に丸がある組）
        circ = [((it["pts"][:, 0].min() + it["pts"][:, 0].max()) / 2, (it["pts"][:, 1].min() + it["pts"][:, 1].max()) / 2)
                for it in its if it["fill"] == "rgb(100%,100%,100%)" and it["stroke"] == "rgb(0%,0%,0%)" and "C" in it["cmds"]]
        segs = [it["pts"] for it in its if it["fill"] == "none" and it["stroke"] == "rgb(0%,0%,0%)" and it["width"] == "3.3723" and len(it["pts"]) == 2]
        legend = []
        for s in segs:
            if abs(s[0, 1] - s[1, 1]) < 0.05:
                mid = s.mean(axis=0)
                legend += [i for i, (cx, cy) in enumerate(circ) if np.hypot(cx - mid[0], cy - mid[1]) < 0.8]
        assert len(legend) == 1, f"{mat}：キャピラリーの凡例の見本が見つからない（{len(legend)}）"
        cap = sorted((to_x(cx), to_y(cy)) for i, (cx, cy) in enumerate(circ) if i not in legend)
        cap_all += [{"材料": mat, "温度": CAP_T[mat], "せん断速度": x, "樹脂粘度": y} for x, y in cap]
        print(f"  図4 {MAT_TITLE[mat]}：点 {len(pts_all) - n0} 個、キャピラリー {len(cap)} 点、"
              f"軸 横 {lims[mat][0][0]:.3g}〜{lims[mat][0][1]:.3g}、縦 {lims[mat][1][0]:.3g}〜{lims[mat][1][1]:.3g}")
    return pd.DataFrame(pts_all), pd.DataFrame(cap_all), lims


def match_repo(pts):
    """取り出した点と、リポジトリのデータ（05 フォルダの 5data・3data。除外なし。ショットごとの値）を照合する。
    リポジトリの各行に、同じ厚み・同じ温度の点が、対数で 0.005 桁（約 1%）以内にあるかを数える"""
    D = mc.load_all(MATS, exclude=False)
    rows = []
    for mat in MATS:
        rep = pd.concat([D[(mat, "5data")], D[(mat, "3data")]])
        p = pts[pts["材料"] == mat]
        dist = []
        for (h, t), g in rep.groupby(["H", "Tset"]):
            q = p[np.isclose(p["厚み"], h) & np.isclose(p["加熱シリンダ温度"], t)]
            if q.empty:
                dist += [np.inf] * len(g)
                continue
            a = np.log10(np.c_[g["gd"].to_numpy(), g["eta"].to_numpy()])
            b = np.log10(q[["せん断速度", "樹脂粘度"]].to_numpy())
            dist += list(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)).min(axis=1))
        dist = np.array(dist)
        r = {"材料": mat, "図の点": len(p), "リポジトリの行": len(rep), "0.005桁以内": int((dist < 0.005).sum()),
             "0.02桁以内": int((dist < 0.02).sum()), "距離の中央値[桁]": float(np.median(dist))}
        rows.append(r)
        print(f"  照合 {MAT_TITLE[mat]}：図の点 {r['図の点']}、リポジトリの行 {r['リポジトリの行']} のうち、0.005 桁以内 {r['0.005桁以内']}、"
              f"0.02 桁以内 {r['0.02桁以内']}（距離の中央値 {r['距離の中央値[桁]']:.4f} 桁）")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 報告書B の図3（樹脂圧力の経時変化）
def extract_fig3(svg):
    """本図と拡大図の波形（センサ位置①：赤、②：青）と、注記（ΔP2・Δt・ΔP1）の位置を取り出す"""
    items, page_h = svg_paths(svg)
    its = in_box(items, page_h, FIG3_BOX)
    hl, vl = straight_lines(its, "rgb(0%,0%,0%)", "1.1241")
    L, R, T, B = frame_of(hl, vl, 0)                         # 本図の枠
    l2, r2, t2, b2 = frame_of(hl, vl, 1)                     # 拡大図の枠
    # 本図：長い目盛り（下は 0 s と 1 s、左は 0 MPa と 10 MPa）
    xl = sorted(x for x, y0, y1 in vl if abs(y1 - B) < 0.05 and (y1 - y0) > 3.0 and (y1 - y0) < 10.0)
    yl = sorted(y for x0, x1, y in hl if abs(x0 - L) < 0.05 and (x1 - x0) > 3.0 and (x1 - x0) < 10.0)
    assert len(xl) == 2 and len(yl) == 2, f"本図の長い目盛りが想定と違う（{xl}、{yl}）"
    to_t = lambda px: (px - xl[0]) / (xl[1] - xl[0]) * 1.0
    to_p = lambda py: (yl[1] - py) / (yl[1] - yl[0]) * 10.0
    # 拡大図：下のいちばん長い目盛りが 0.20 s、主な目盛り（長さ 1〜3 pt）の間は 0.01 s。左のいちばん長い目盛りが 0 MPa、主な目盛りの間は 1 MPa
    xs2 = sorted(x for x, y0, y1 in vl if abs(y1 - b2) < 0.05 and 1.0 < (y1 - y0) < 3.0)
    xl2 = [x for x, y0, y1 in vl if abs(y1 - b2) < 0.05 and 1.5 < (y1 - y0) < 3.0]
    ys2 = sorted(y for x0, x1, y in hl if abs(x0 - l2) < 0.05 and 1.0 < (x1 - x0) < 3.0)
    yl2 = [y for x0, x1, y in hl if abs(x0 - l2) < 0.05 and 1.5 < (x1 - x0) < 3.0]
    assert len(xl2) == 1 and len(yl2) == 1, "拡大図の長い目盛りが想定と違う"
    ddx = float(np.median(np.diff(xs2)))
    ddy = float(np.median(np.diff(ys2)))
    to_t2 = lambda px: 0.20 + (px - xl2[0]) / ddx * 0.01
    to_p2 = lambda py: (yl2[0] - py) / ddy * 1.0
    print(f"  図3：本図 横 {to_t(L):.3f}〜{to_t(R):.3f} s、縦 {to_p(B):.2f}〜{to_p(T):.2f} MPa。拡大図 横 {to_t2(l2):.3f}〜{to_t2(r2):.3f} s、"
          f"縦 {to_p2(b2):.2f}〜{to_p2(t2):.2f} MPa")

    def curve(color, width, tt, pp, legend=False):
        segs = [it["pts"] for it in its if it["stroke"] == color and it["width"] == width and it["fill"] == "none"]
        p = np.vstack(segs)
        d = pd.DataFrame({"t": tt(p[:, 0]), "P": pp(p[:, 1])}).round(7).drop_duplicates().sort_values("t")
        if legend:                                                # 本図の右上の凡例（Sensor position）の線は、同じ書式なので除く（0.7 s より後の波形は 1 MPa 未満）
            d = d[~((d["t"] > 0.7) & (d["P"] > 3.0))]
        return d.reset_index(drop=True)

    wave = []
    for s, col in [("①", "rgb(100%,0%,0%)"), ("②", "rgb(0%,0%,100%)")]:
        a = curve(col, "3.3723", to_t, to_p, legend=True)
        a["センサ位置"], a["図"] = s, "本図"
        b = curve(col, "5.6449", to_t2, to_p2)
        b["センサ位置"], b["図"] = s, "拡大図"
        wave += [a, b]
    wave = pd.concat(wave, ignore_index=True)
    # 注記：細い黒の線（0.5 pt）。本図の 2 本の水平線（ΔP2 の上下）、拡大図の 2 本の水平線（ΔP1 の上下）と 2 本の縦線（Δt の始めと終わり）
    hn, vn = straight_lines(its, "rgb(0%,0%,0%)", "0.5")
    vn += [(it["pts"][0, 0], it["pts"][:, 1].min(), it["pts"][:, 1].max()) for it in its
           if it["stroke"] == "rgb(0%,0%,0%)" and it["width"] == "0.5" and len(it["pts"]) == 2 and 0.01 <= abs(it["pts"][0, 0] - it["pts"][1, 0]) < 0.1]
    main_h = sorted(y for x0, x1, y in hn if x0 < l2)
    ins_h = sorted(y for x0, x1, y in hn if x0 >= l2)
    ins_v = sorted(x for x, y0, y1 in vn if x >= l2)
    gray = [it["pts"][0, 0] for it in its if it["stroke"].startswith("rgb(49.") and len(it["pts"]) == 2]
    assert len(main_h) == 2 and len(ins_h) == 2 and len(ins_v) == 2 and len(gray) == 1, "注記の線が想定と違う"
    ann = {"ΔP2_上": to_p(main_h[0]), "ΔP2_下": to_p(main_h[1]), "最大の時刻": to_t(gray[0]),
           "ΔP1_上": to_p2(ins_h[0]), "ΔP1_下": to_p2(ins_h[1]), "ΔP1_時刻": to_t2(min(x0 for x0, x1, y in hn if x0 >= l2)),
           "Δt_始め": to_t2(ins_v[0]), "Δt_終わり": to_t2(ins_v[1])}
    # 拡大する範囲の四角（紺の線）
    zb = [it["pts"] for it in its if it["stroke"].startswith("rgb(9.01")][0]
    ann.update({"四角_左": to_t(zb[:, 0].min()), "四角_右": to_t(zb[:, 0].max()), "四角_下": to_p(zb[:, 1].max()), "四角_上": to_p(zb[:, 1].min())})
    print("  図3 の注記：" + "、".join(f"{k} {v:.4f}" for k, v in ann.items()))
    return wave, ann


# ---------------------------------------------------------------- 見本の樹脂温度のグラフ（画像から曲線を読み取る）
def digitize_temperature(png):
    """見本_樹脂温度の経時変化.png の黒い曲線を読み取る。枠は、長い黒の線（縦・横）から探す。
    軸：横 2〜4 s、縦 0〜250 ℃（見本の目盛りの数字）。凡例・断面図・注記の範囲は除く（範囲は、枠に対する割合で決めた）"""
    g = np.asarray(Image.open(png).convert("RGB")).astype(int)
    dark = (g.max(axis=2) < 110)                                   # 黒（赤の注記は入らない）
    H, W = dark.shape
    cols = np.flatnonzero(dark.sum(axis=0) > 0.6 * H)
    rows = np.flatnonzero(dark.sum(axis=1) > 0.6 * W)
    left, right = cols[cols < W / 2].max(), cols[cols > W / 2].min()
    top, bot = rows[rows < H / 2].max(), rows[rows > H / 2].min()
    fw, fh = right - left, bot - top
    to_t = lambda px: 2.0 + (px - left) / fw * 2.0
    to_T = lambda py: (bot - py) / fh * 250.0
    m = dark.copy()
    edge = 70                                                      # 枠と目盛りの近く [px]（目盛りの線は約 50 px）
    m[:, : left + edge] = False
    m[:, right - edge:] = False
    m[: top + edge, :] = False
    m[bot - edge:, :] = False
    excl = [(0.62, 1.00, 0.00, 0.28),                              # 凡例（Sensor position ②）：枠の横の割合 左 右、縦の割合 上 下
            (0.50, 0.93, 0.56, 1.00)]                              # 断面図
    for a, b, c, d in excl:
        m[top + int(c * fh): top + int(d * fh), left + int(a * fw): left + int(b * fw)] = False
    pts = []
    for x in range(left + edge + 1, right - edge):
        ys = np.flatnonzero(m[:, x])
        if ys.size == 0:
            continue
        runs = np.split(ys, np.flatnonzero(np.diff(ys) > 3) + 1)
        r = runs[0]                                                # いちばん上のかたまり（曲線）
        if r[-1] - r[0] > 40:                                      # 立ち上がりの縦の部分は、下と上の端の 2 点
            pts += [(to_t(x), to_T(r[-1] - 6)), (to_t(x), to_T(r[0] + 6))]
        else:
            pts.append((to_t(x), to_T((r[0] + r[-1]) / 2)))
    d = pd.DataFrame(pts, columns=["t", "T"])
    print(f"  樹脂温度：枠 横 {left}〜{right} px、縦 {top}〜{bot} px、{len(d)} 点、最高 {d['T'].max():.1f} ℃（{d.loc[d['T'].idxmax(), 't']:.3f} s）")
    return d


# ================================================================ 図
def fig_pressure(out, wave, ann):
    W, H = SIZE["pressure"]
    fig = plt.figure(figsize=mm(W, H))
    ax = axes_mm(fig, W, H, 30.0, 25.0, W - 30.0 - 9.0, H - 25.0 - 4.0)
    main = wave[wave["図"] == "本図"]
    sty = {"①": dict(color="#FF0000", lw=3.2, ls="-"), "②": dict(color="#0000FF", lw=3.2, ls=(0, (4, 1.6)))}
    for s in ["①", "②"]:
        d = main[main["センサ位置"] == s]
        ax.plot(d["t"], d["P"], zorder=3, **sty[s])
    ax.set_xlim(0, 1.2)
    ax.set_ylim(-1, 17)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1", "1.2"])
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.set_xlabel("時間 (s)", labelpad=2)
    ax.set_ylabel("樹脂圧力 (MPa)", labelpad=4)
    # ΔP2：①と②の最大の時刻の圧力差（元の図と同じく、線はピークから左へ延ばし、矢印と縦書きの文字はその左に置いた）
    tp, up, lo = ann["最大の時刻"], ann["ΔP2_上"], ann["ΔP2_下"]
    ax.axvline(tp, color="0.45", lw=1.4, ls=(0, (5, 3)), zorder=1)
    xa = 0.155
    for v in (up, lo):
        ax.plot([xa - 0.03, tp], [v, v], color="k", lw=1.4, zorder=2)
    ax.add_patch(FancyArrowPatch((xa, lo), (xa, up), arrowstyle="<|-|>", mutation_scale=22, lw=1.6, color="k", shrinkA=0, shrinkB=0, zorder=4))
    ax.text(xa - 0.06, (up + lo) / 2, r"$\Delta P_2$", fontsize=LABEL, ha="center", va="center", rotation=90)
    # 凡例（センサ位置）
    h = [Line2D([], [], **sty["①"]), Line2D([], [], **sty["②"])]
    ax.legend(h, ["①", "②"], title="センサ位置", loc="upper right", frameon=False, handlelength=2.6, borderaxespad=0.3, labelspacing=0.25,
              ncol=2, columnspacing=1.0, handletextpad=0.5)
    # 拡大する範囲の四角と、拡大図
    zl, zr, zb, zt = ann["四角_左"], ann["四角_右"], ann["四角_下"], ann["四角_上"]
    ax.add_patch(Rectangle((zl, zb), zr - zl, zt - zb, fill=False, ec="#172C51", lw=1.6, zorder=5))
    # 拡大図の大きさ [mm]。本図の右下の空き（凡例〔軸の上から約 30 mm〕の下、3 MPa より上）に置く
    aw, ah = 85.0, min(56.0, (H - 29.0) * (1.0 - 4.0 / 18.0) - 33.0)
    axl, axb = W - 9.0 - 2.0 - aw, 25.0 + (3.0 + 1.0) / 18.0 * (H - 29.0)
    ins = axes_mm(fig, W, H, axl, axb, aw, ah)
    zoom = wave[wave["図"] == "拡大図"]
    for s in ["①", "②"]:
        d = zoom[zoom["センサ位置"] == s]
        ins.plot(d["t"], d["P"], zorder=3, **{**sty[s], "lw": 2.6})
    ins.set_xlim(0.18, 0.24)
    ins.set_ylim(-0.5, 5)
    ins.set_xticks([0.19, 0.21, 0.23])                         # 28 pt の数字が重ならないように、3 つだけ（両端の 0.18・0.24 は図の端からはみ出すので省いた）
    ins.set_xticklabels(["0.19", "0.21", "0.23"])
    ins.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ins.set_yticks([0, 2, 4])
    ins.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ins.tick_params(which="both", labelsize=TICK, pad=4)
    ins.tick_params(which="major", length=7)
    ins.set_facecolor("white")
    # Δt と ΔP1
    t0, t1 = ann["Δt_始め"], ann["Δt_終わり"]
    ya = 2.0
    for x in (t0, t1):
        ins.plot([x, x], [-0.1, ya + 0.4], color="k", lw=1.2, zorder=2)
    ins.add_patch(FancyArrowPatch((t0, ya), (t1, ya), arrowstyle="<|-|>", mutation_scale=18, lw=1.4, color="k", shrinkA=0, shrinkB=0, zorder=4))
    ins.text(t0 + 0.0103, ya + 0.12, r"$\Delta t$", fontsize=LABEL, ha="center", va="bottom")
    # ΔP1：②の立ち上がりの時刻の①と②の圧力差。28 pt の文字が線に重ならないように、矢印は その時刻の縦の線の上に置き（元の図は、右に延ばした線の間）、
    # 文字は矢印の右の空き（①と②の線の間）に置いた
    tq, pu, pl = ann["ΔP1_時刻"], ann["ΔP1_上"], ann["ΔP1_下"]
    ins.add_patch(FancyArrowPatch((tq, pl), (tq, pu), arrowstyle="<|-|>", mutation_scale=18, lw=1.4, color="k", shrinkA=0, shrinkB=0, zorder=4))
    ins.text(tq + 0.0009, 2.35, r"$\Delta P_1$", fontsize=MIN_PT, ha="left", va="bottom")
    # 四角から拡大図への矢印
    fig.add_artist(ConnectionPatch((zr, zt), (0.0, 0.0), coordsA=ax.transData, coordsB=ins.transAxes, arrowstyle="-|>", mutation_scale=22,
                                   lw=1.4, color="k", shrinkA=2, shrinkB=4, zorder=6))
    save(fig, out, "樹脂圧力の経時変化.png")


def fig_temperature(out, temp):
    W, H = SIZE["temperature"]
    fig = plt.figure(figsize=mm(W, H))
    ax = axes_mm(fig, W, H, 34.0, 25.0, W - 34.0 - 9.0, H - 25.0 - 4.0)
    ax.plot(temp["t"], temp["T"], color="k", lw=3.2, zorder=3)
    ax.set_xlim(2, 4)
    ax.set_ylim(0, 250)
    ax.set_xticks([2, 2.5, 3, 3.5, 4])
    ax.set_xticklabels(["2", "2.5", "3", "3.5", "4"])
    ax.set_yticks([0, 50, 100, 150, 200, 250])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
    ax.set_xlabel("時間 (s)", labelpad=2)
    ax.set_ylabel("樹脂温度 (℃)", labelpad=4)
    i = temp["T"].idxmax()
    tp, Tp = float(temp.loc[i, "t"]), float(temp.loc[i, "T"])
    ax.annotate("学習データ\nに使用", xy=(tp + 0.01, Tp + 2), xytext=(3.0, 247), ha="left", va="top", color="#FF0000", fontsize=TICK,
                linespacing=1.05, arrowprops=dict(arrowstyle="-|>", mutation_scale=22, lw=2.0, color="#FF0000", shrinkA=4, shrinkB=2))
    ax.legend([Line2D([], [], color="k", lw=3.2)], ["②"], title="センサ位置", loc="lower right", frameon=False, handlelength=2.6,
              borderaxespad=0.6)
    save(fig, out, "樹脂温度の経時変化.png")


def fig_formula(out):
    W, H = SIZE["formula"]
    fig = plt.figure(figsize=mm(W, H))
    fig.text(0.02, 0.66, r"$\eta = \dfrac{\Delta P_1\, B\, H^3}{12\, L\, Q}$", fontsize=34, va="center")
    fig.text(0.60, 0.66, r"$\dot{\gamma} = \dfrac{6\,Q}{B\, H^2}$", fontsize=34, va="center")
    fig.text(0.02, 0.15, r"流量：$Q = V H B$", fontsize=30, va="center")
    fig.text(0.60, 0.15, r"流速：$V = L\,/\,\Delta t$", fontsize=30, va="center")
    save(fig, out, "算出式.png")


FLOW = {"left": 34.0, "right": 4.0, "gap": 10.0, "top": 4.0, "bottom": 44.0}      # 樹脂粘度のグラフの余白 [mm]（横軸の数字は 10^0・10^2・10^4 だけなので、右端に数字がない）


def flow_size():
    s = (COL_W - FLOW["left"] - FLOW["right"] - 2 * FLOW["gap"]) / 3
    return COL_W, FLOW["top"] + s + FLOW["bottom"], s


def fig_flow3(out, pts, cap, lims):
    """3 材料の樹脂粘度とせん断速度の関係を横に並べる（グラフエリアは正方形。縦軸の名前と目盛りの数字は左端だけ。軸の範囲は 3 つで共通）"""
    W, H, s = flow_size()
    xl, yl = lims["PP"]
    for mat in MATS:
        assert np.allclose(np.log10(lims[mat][0]), np.log10(xl), atol=0.01) and np.allclose(np.log10(lims[mat][1]), np.log10(yl), atol=0.01), \
            f"{mat}：軸の範囲が PP と違う（{lims[mat]}）"
    fig = plt.figure(figsize=mm(W, H))
    axs = []
    for i, mat in enumerate(MATS):
        x0 = FLOW["left"] + i * (s + FLOW["gap"])
        ax = axes_mm(fig, W, H, x0, FLOW["bottom"], s, s)
        axs.append(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        for a in (ax.xaxis, ax.yaxis):
            a.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=20))     # 目盛りの線は 1 桁ごと（報告書B の図4 と同じ）
            a.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 1.0, numticks=20))
            a.set_major_formatter(ticker.LogFormatterSciNotation())
            a.set_minor_formatter(ticker.NullFormatter())
        # 横軸の数字は 10^0・10^2・10^4 だけ（28 pt で 1 桁ごとに書くと、となりのグラフの数字とくっつくため）
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, pos: rf"$\mathdefault{{10^{{{int(round(np.log10(v)))}}}}}$" if int(round(np.log10(v))) % 2 == 0 else ""))
        p = pts[pts["材料"] == mat]
        for (shape, h), g in p.groupby(["形", "厚み"], sort=False):
            ax.plot(g["せん断速度"], g["樹脂粘度"], ls="", marker=MARK[shape], ms=7.5, mfc="none", mec=H_COLOR[h], mew=1.3, zorder=3)
        c = cap[cap["材料"] == mat].sort_values("せん断速度")
        ax.plot(c["せん断速度"], c["樹脂粘度"], color="k", lw=2.6, marker="o", ms=10, mfc="white", mec="k", mew=2.2, zorder=5)
        ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ (1/s)", labelpad=2)
        if i == 0:
            ax.set_ylabel(r"樹脂粘度 $\eta$ (Pa·s)", labelpad=4)
        else:
            ax.tick_params(axis="y", which="both", labelleft=False)
        fig.text((x0 + s / 2) / W, 4.0 / H, MAT_TITLE[mat], ha="center", va="bottom", fontsize=LABEL)
    save(fig, out, "樹脂粘度とせん断速度の関係.png")
    for ax in axs:                                              # グラフエリアが正方形か（図を作ったときの大きさで）
        bb = ax.get_position()
        assert abs(bb.width * W - bb.height * H) < 0.5, "グラフエリアが正方形でない"
    print(f"  樹脂粘度のグラフ：図 {W:.1f}×{H:.1f} mm、グラフエリア {s:.1f} mm 角 × 3")


# ================================================================ まとめて作る
NAMES = ["樹脂圧力の経時変化.png", "樹脂温度の経時変化.png", "算出式.png", "樹脂粘度とせん断速度の関係.png",
         "図データ_図4の点.csv", "図データ_キャピラリー.csv", "図データ_図4の照合.csv", "図データ_樹脂圧力.csv", "図データ_図3の注記.csv",
         "図データ_樹脂温度.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT, help="図を出すフォルダ（既定：ポスター/報告書Bから作成/図_簡略版_A0一枚/）")
    a = ap.parse_args()
    there = [n for n in NAMES if os.path.exists(os.path.join(a.out, n))]
    if there:
        sys.exit(f"すでにある：{there}（上書きしない。前の図を _to_delete/ へ移すか、--out で別のフォルダを指定）")
    os.makedirs(a.out, exist_ok=True)
    w = lambda df, n: df.to_csv(os.path.join(a.out, n), index=False, encoding="utf-8-sig")
    svg = page_svg(rb.SRC_PDF, 2, "B_p2.svg")
    print("報告書B の図4 から取り出す")
    pts, cap, lims = extract_fig4(svg)
    mc.SHOW_FIGS = False
    chk = match_repo(pts)
    print("報告書B の図3 から取り出す")
    wave, ann = extract_fig3(svg)
    print("見本の樹脂温度のグラフを読み取る")
    temp = digitize_temperature(SAMPLE_T)
    fig_pressure(a.out, wave, ann)
    fig_temperature(a.out, temp)
    fig_formula(a.out)
    fig_flow3(a.out, pts, cap, lims)
    w(pts, "図データ_図4の点.csv")
    w(cap, "図データ_キャピラリー.csv")
    w(chk, "図データ_図4の照合.csv")
    w(wave, "図データ_樹脂圧力.csv")
    w(pd.DataFrame([ann]), "図データ_図3の注記.csv")
    w(temp, "図データ_樹脂温度.csv")
    for k, (wd, ht) in SIZE.items():
        print(f"  図の大きさ {k}：{wd:.0f}×{ht:.0f} mm")
    print("保存先:", a.out)


if __name__ == "__main__":
    main()
