# -*- coding: utf-8 -*-
"""A0 縦のポスターの簡略版（3〜5 分の発表用。PowerPoint）を作る：05_…/学習結果報告書作成_261015/ポスター/報告書Bから作成_簡略版.pptx

今のポスター（build_poster.py が作る 報告書Bから作成.pptx）を、2026-11-04 のユーザーの依頼で簡略にしたもの。
  ・文は、発表者がいなくてもポスターだけで「何を目指して、何を行い、どうなったか」が伝わる長さにする。各節の冒頭に「何をしたか・なぜか」の文、
    各図に説明の文、結果とまとめに数値つきの結論。細かい定義・手法の注・補足の数値は、台本（build_poster_simple_script.py）に回す。
  ・図は 12 枚（今のポスターは 19 枚）。「キャビティ入れ子の外観」の写真は使わない（ユーザーの指示）。
  ・材料ごとの成形条件と、学習・検証の分け方は、表と新しい図（make_poster_simple_figs.py の 成形条件と学習検証.png）で示す。
  ・温度シフトファクターは、いつ決めたか（学習の前・学習のとき・学習の後）と、学習したモデルの中での使われ方（予測の流れ）を図形で示す。
  ・上のタイトルと発表者名のスペースは、見本（報告書Aから作成2.pdf）と同じ高さ 138.7 mm の黒の帯（何も書かない）。
モデルと数値は、今のポスター・報告書B と同じ（B_Tff・条件平均の入力。数値は ポスター/図/ポスターの数値.csv から読む）。
部品（文字・図形・図・表）は build_poster.py のものを使う。すでにあるファイルは上書きしない（あれば止まる）。

使い方（リポジトリ直下で。先に make_poster_simple_figs.py で条件の図を作る）:
  uv run python 04_補助スクリプト/build_poster_simple.py [--out <フォルダ>] [--map-dir <条件の図のフォルダ>]
  （既定：--out はポスターのフォルダ、--map-dir は ポスター/図_簡略版/）
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.util import Mm  # noqa: E402

import build_poster as bp  # noqa: E402
from build_poster import BLUE, FRAME, GREEN, GREEN_LINE, LIGHT, PT_MM, box, est_width, frame, pic, shape, table, tbox, text_in  # noqa: E402

POSTER = bp.POSTER
FIG = bp.FIG                                            # 今のポスターの図（ポスター/図/）
REPORT_FIG = os.path.join(bp.REPORT, "図")              # 報告書B の図（Fig3・Fig4）
MAP_DIR = os.path.join(POSTER, "図_簡略版")             # 新しい図（make_poster_simple_figs.py）
MAP_NAME = "成形条件と学習検証.png"
OUT_NAME = "報告書Bから作成_簡略版.pptx"

PAGE_W, PAGE_H = 841.0, 1189.0   # A0 縦 [mm]
BAND_H = 138.7                    # 上のタイトル・発表者名のスペース（見本の黒い帯の高さ。_tmp/poster_simple/measure_band.py で測った）
TOP = 142.7                       # 最初の節の上端（見本と同じ）
M = 7.0                           # 外側の余白（見本は 6.9 mm）
GAP = 8.0                         # 列の間
CW = (PAGE_W - 2 * M - GAP) / 2   # 列の幅
XL, XR = M, M + CW + GAP
PAD = 5.0                         # 枠の内側の余白
IW = CW - 2 * PAD                 # 列の中身の幅
SEC_GAP = 5.0                     # 節の間
HEAD_H, HEAD_PT = 24.0, 54        # 節の見出しの帯（見本は 72 pt）
SUB_H, SUB_PT = 13.0, 30          # 小見出し
BODY = 28                         # 本文（見本は 32 pt）
NOTE = 23                         # 図の説明・表
SMALL = 20                        # 図の中の補足
LINE = 1.2                        # 行の高さ（文字の大きさの倍）


# ---------------------------------------------------------------- 文字の高さの見積もりと、大きい文字の見出し
def n_lines(text, size, width, hang=False):
    n = 0
    for p in text.split("\n"):
        w = width - (size * PT_MM if hang and p[:1] in "・①②③④" else 0.0)
        n += max(1, math.ceil(est_width(p, size) * 1.0 / w))
    return n


def text_h(text, size, width, spacing=1.0, margin=1.5, hang=False):
    return n_lines(text, size, width - 2 * margin, hang) * size * PT_MM * LINE * spacing + 2 * margin + 0.5


def para(s, x, y, w, text, size=BODY, gap=2.0, **kw):
    """文字を書いて、その下端（＋ gap）を返す"""
    h = text_h(text, size, w, kw.get("spacing", 1.0), kw.get("margin", 1.5), kw.get("hang") is not None)
    tbox(s, x, y, w, h, text, size, **kw)
    return y + h + gap


def boxed(s, x, y, w, text, size, gap=2.5, **kw):
    """囲みの中に文字を書いて、その下端（＋ gap）を返す"""
    kw.setdefault("margin", 2.5)
    h = text_h(text, size, w, kw.get("spacing", 1.0), kw["margin"], kw.get("hang") is not None)
    box(s, x, y, w, h, text, size, **kw)
    return y + h + gap


def header(s, x, y, w, title):
    sh = shape(s, MSO_SHAPE.RECTANGLE, x, y, w, HEAD_H, fill=BLUE)
    text_in(sh, title, HEAD_PT, color="FFFFFF", bold=True, anchor="middle", margin=4.0)
    return y + HEAD_H + 3.0


def sub(s, x, y, title, size=SUB_PT):
    w = est_width(title, size) + 8.0
    sh = shape(s, MSO_SHAPE.RECTANGLE, x, y, w, SUB_H, fill=GREEN, line=GREEN_LINE, lw=1.25)
    text_in(sh, title, size, bold=True, anchor="middle", margin=3.0)
    return y + SUB_H + 3.0


def japanese_breaks(prs):
    """すべての文字に日本語の設定（lang="ja-JP"）と、日本語の行の折り返し（禁則：行の頭に「、」「。」を置かない）を付ける"""
    for sl in prs.slides:
        for shp in sl.shapes:
            tfs = [shp.text_frame] if shp.has_text_frame else []
            if getattr(shp, "has_table", False) and shp.has_table:
                tfs += [c.text_frame for row in shp.table.rows for c in row.cells]
            for tf in tfs:
                for p in tf.paragraphs:
                    pPr = p._p.get_or_add_pPr()
                    pPr.set("eaLnBrk", "1")
                    pPr.set("latinLnBrk", "0")
                    pPr.set("hangingPunct", "1")
                    for r in p.runs:
                        r._r.get_or_add_rPr().set("lang", "ja-JP")


def ratio(name, folder=FIG):
    """図の 縦 / 横"""
    with Image.open(os.path.join(folder, name)) as im:
        return im.height / im.width


def right_arrow(s, x, y, w, h):
    shape(s, MSO_SHAPE.RIGHT_ARROW, x, y, w, h, fill="FFFFFF", line=BLUE, lw=2.25)


def down_arrow(s, x, y, w, h):
    shape(s, MSO_SHAPE.DOWN_ARROW, x, y, w, h, fill="FFFFFF", line=BLUE, lw=2.25)


# ---------------------------------------------------------------- 数値（今のポスターと同じ元）
def numbers():
    t = pd.read_csv(os.path.join(FIG, "ポスターの数値.csv"), encoding="utf-8-sig")
    r = lambda v: int(math.floor(v + 0.5))
    n = {"err": [r(v) for v in t["誤差 [%] 全体"]], "cw": [r(v) for v in t["Cross-WLF 誤差 [%] 全体"]],
         "cover": [r(v * 100) for v in t["被覆率"]], "ratio": [r(v * 10) for v in t["比"]]}
    rng = lambda xs: f"{min(xs)}〜{max(xs)}"
    return {k: rng(v) for k, v in n.items()}


# ---------------------------------------------------------------- 左の列
def sec_overview(s, x, y):
    xi = x + PAD
    y = header(s, x, y, CW, "1. 研究概要")
    t = ("　射出成形 CAE で用いられる樹脂粘度モデル^{1)} では、樹脂のロット間の違いや成形工程の熱履歴による粘度のばらつきが考慮されていない。"
         "本研究では、金型内で計測した樹脂圧力・樹脂温度のデータと機械学習を用いて、[r]樹脂粘度の予測値と、予測のばらつきの範囲を同時に求められる[/r]"
         "データ駆動型樹脂粘度モデルの構築を目指している。")
    y = para(s, xi, y, IW, t, BODY, gap=1.5)
    pts = ("[b]本報で行ったこと[/b]\n"
           "① PP・粉砕材混合 PP・GPPS の 3 材料でモデルを構築\n"
           "② 温度による粘度曲線のずれ（[r]温度シフトファクター[/r]）をモデルに組み込み\n"
           "③ [r]学習に用いていない射出率[/r]の条件で、予測の精度を検証")
    return boxed(s, xi, y, IW, pts, 26, fill=LIGHT, hang=1.0, gap=3.0)


def sec_mold(s, x, y):
    xi = x + PAD
    y = header(s, x, y, CW, "2. 計測金型およびセンサ")
    y = para(s, xi, y, IW, "　キャビティ内の 2 か所（センサ位置①②）で、流動中の樹脂の圧力と温度を計測できる計測金型を用いた。", BODY)
    n1, n2 = "見本_計測金型.png", "見本_成形品の形状.png"
    gi = 8.0
    h = (IW - gi) / (1 / ratio(n1) + 1 / ratio(n2))
    w1, w2 = h / ratio(n1), h / ratio(n2)
    x0 = xi + (IW - w1 - w2 - gi) / 2
    pic(s, n1, x0, y, h=h)
    pic(s, n2, x0 + w1 + gi, y, h=h)
    y += h + 1.0
    tbox(s, x0, y, w1, 10.5, "計測金型の外観", NOTE, align="center")
    tbox(s, x0 + w1 + gi, y, w2, 10.5, "キャビティの形状・寸法（mm）", NOTE, align="center")
    y += 12.0
    return para(s, xi, y, IW, "①②：水晶圧電式樹脂圧力センサ　②：赤外線放射樹脂温度センサ（圧力と温度は別々に計測）", NOTE, gap=3.0)


def sec_conditions(s, x, y, map_dir):
    xi = x + PAD
    y = header(s, x, y, CW, "3. 成形条件と学習・検証データ")
    t = ("　3 材料で、加熱シリンダ温度・キャビティ厚み・射出率（12 水準）を変えて成形した。[r]射出率の両端の条件は学習に用いず[/r]、"
         "モデルの予測の答え合わせ（検証）に用いた。")
    y = para(s, xi, y, IW, t, BODY)
    rows = [["材料", "樹脂", "加熱シリンダ温度", "キャビティ厚み"],
            ["PP", "プライムポリプロ J707EG", "190・210・230 ℃", "0.5〜2.0 mm の 4 水準"],
            ["粉砕材混合 PP", "同 PP ＋ 粉砕材 25 wt%", "190・210・230 ℃", "0.5〜2.0 mm の 4 水準"],
            ["GPPS", "ディックスチレン CR-2500", "200・220・240 ℃", "1.0〜2.0 mm の 3 水準"]]
    y = table(s, xi, y, [68.0, 128.0, 95.0, IW - 291.0], rows, 22) + 3.0
    wm, hm = pic(s, MAP_NAME, xi + (IW - 396.0) / 2, y, w=396.0, folder=map_dir)
    y += hm + 1.0
    t = "学習データは 1 条件 5 ショット、検証データは 1 条件 3 ショット。外れ値・計測不良のデータは除いた（「2温度」は、1 温度を除いた条件）"
    return para(s, xi, y, IW, t, NOTE, gap=3.0)


# ---------------------------------------------------------------- 右の列
def sec_experiment(s, x, y):
    xi = x + PAD
    y = header(s, x, y, CW, "4. 実験結果")
    t = "　2 つのセンサの圧力の差 ΔP_{1} と、樹脂が①から②へ届くまでの時間差 Δt から、樹脂粘度 η とせん断速度 γ̇ を求めた^{2)}。"
    y = para(s, xi, y, IW, t, BODY)
    w3 = 190.0
    _, h3 = pic(s, "Fig3.png", xi, y, w=w3, folder=REPORT_FIG)
    xr, wr = xi + w3 + 5.0, IW - w3 - 5.0
    _, hf = pic(s, "算出式.png", xr, y + 4.0, w=wr)
    notes = ("ΔP_{1}：②の立ち上がり時の①②の圧力差\nΔt：①②の圧力の立ち上がりの時間差\nΔP_{2}：最大圧力時の①②の圧力差（機械学習の入力）\n"
             "L：センサ間距離、B：流路幅、H：キャビティ厚み、Q：樹脂流量")
    yn = para(s, xr, y + 4.0 + hf + 1.0, wr, notes, SMALL, gap=0.0)
    y = max(y + h3, yn) + 0.5
    y = para(s, xi, y, w3, "樹脂圧力の経時変化の一例（キャビティ厚み 1.0 mm、加熱シリンダ温度条件Ⅱ、射出率 42.5 cm^{3}/s）", SMALL, gap=2.0)
    _, h4 = pic(s, "Fig4.png", xi, y, w=IW, folder=REPORT_FIG)
    y += h4 + 2.0
    bw = IW - 130.0
    yb = boxed(s, xi, y, bw, "高せん断速度側は、キャピラリーレオメータの値（白丸）とおおむね一致", 23, anchor="middle", gap=0.0)
    hb = yb - y
    right_arrow(s, xi + bw + 3.0, y + hb / 2 - 7.0, 13.0, 14.0)
    box(s, xi + bw + 19.0, y, IW - bw - 19.0, hb, "[r]学習データとして使用可能[/r]", 23, anchor="middle", align="center", margin=1.5)
    return yb + 3.0


def sec_ml(s, x, y):
    xi = x + PAD
    y = header(s, x, y, CW, "5. 機械学習と温度シフトファクター")
    # ---- ガウス過程回帰
    y = sub(s, xi, y, "ガウス過程回帰（GPR）")
    wg = 178.0
    _, hg = pic(s, "GPR概念図.png", xi, y, w=wg)
    t = ("粘度式の形（Cross 式など）をあらかじめ決めずに、入力と粘度の関係をデータから学習する手法^{3)}。\n"
         "[r]予測値と、そのばらつきの範囲（95% 予測区間）[/r]が同時に求まる。データの少ない所ほど、範囲が広くなる。")
    yt = para(s, xi + wg + 5.0, y + 2.0, IW - wg - 5.0, t, 25, gap=0.0)
    y = max(y + hg, yt) + 3.0
    # ---- 温度シフトファクター
    y = sub(s, xi, y, "温度シフトファクター")
    t = ("　Cross-WLF モデルと同様に、温度によって粘度曲線が横（せん断速度の方向）と縦（粘度の方向）にずれる性質を、"
         "[r]金型内樹脂最高温度 T に基づく温度シフトファクター a_{T}[/r] としてモデルに組み込んだ。")
    y = para(s, xi, y, IW, t, BODY)
    wc = 152.0
    _, hc = pic(s, "温度シフト概念図.png", xi, y, w=wc)
    xt, wt = xi + wc + 5.0, IW - wc - 5.0
    tag_w = 50.0
    steps = [("学習の前", "式の形を決めた（PP・粉砕材混合 PP：アレニウス型、GPPS：WLF 型）"),
             ("学習のとき", "ずれの大きさ（横・縦の 2 つの係数）を、GPR と同時に、学習データに最もよく合うように決めた"),
             ("学習の後\n（予測）", "係数は固定。予測のたびに、T から a_{T} を計算して使う（下の流れ）")]
    yy = y
    for i, (tag, body) in enumerate(steps):
        h = text_h(body, 22, wt - tag_w - 3.0, margin=2.5)
        h = max(h, text_h(tag, 22, tag_w, margin=1.5))
        tg = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, xt, yy, tag_w, h, fill=BLUE, radius=0.12)
        text_in(tg, tag, 22, color="FFFFFF", bold=True, align="center", anchor="middle", margin=1.0)
        box(s, xt + tag_w + 3.0, yy, wt - tag_w - 3.0, h, body, 22, anchor="middle")
        yy += h
        if i < len(steps) - 1:
            down_arrow(s, xt + tag_w / 2 - 6.0, yy + 0.5, 12.0, 6.0)
            yy += 7.0
    y = max(y + hc, yy) + 3.0
    # ---- 予測の流れ
    y = sub(s, xi, y, "学習したモデルでの予測の流れ")
    items = [("[b]入力[/b]\nせん断速度 γ̇\n樹脂圧力差 ΔP_{2}\n金型内樹脂最高温度 T\n加熱シリンダ温度", 84.0, LIGHT, "000000"),
             ("① T から a_{T} を計算\n② γ̇ を、基準温度での値に換算（横のずれ）", 76.0, "FFFFFF", "000000"),
             ("[b]GPR[/b]\n基準温度での粘度を予測", 64.0, BLUE, "FFFFFF"),
             ("③ a_{T} で、縦のずれを戻す", 58.0, "FFFFFF", "000000"),
             ("[b]出力[/b]\n樹脂粘度 η の\n[r]予測値と 95% 予測区間[/r]", None, LIGHT, "000000")]
    aw, sp = 9.0, 1.5
    fixed = sum(w for _, w, _, _ in items if w)
    last = IW - fixed - (len(items) - 1) * (aw + 2 * sp)
    widths = [w if w else last for _, w, _, _ in items]
    hb = max(text_h(t, 20, w, margin=2.0) for (t, _, _, _), w in zip(items, widths))
    xx = xi
    for i, ((t, _, fill, col), w) in enumerate(zip(items, widths)):
        b = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, xx, y, w, hb, fill=fill, line=BLUE, lw=2.0, radius=0.08)
        text_in(b, t, 20, color=col, align="center", anchor="middle", margin=2.0)
        xx += w
        if i < len(items) - 1:
            right_arrow(s, xx + sp, y + hb / 2 - 6.0, aw, 12.0)
            xx += aw + 2 * sp
    y += hb + 1.5
    return para(s, xi, y, IW, "a_{T}：基準温度（PP・粉砕材混合 PP は 210 ℃、GPPS は 220 ℃）からの、温度によるずれの大きさ", SMALL, gap=3.0)


# ---------------------------------------------------------------- 全幅
def sec_results(s, y):
    x, w = M, PAGE_W - 2 * M
    xs = x + PAD
    y0 = y
    y = header(s, x, y, w, "6. 結果")
    num = numbers()
    # ---- 左：学習結果
    wf, gf = 147.0, 4.0
    wl = 3 * wf + 2 * gf
    yl = sub(s, xs, y, "学習結果（樹脂粘度とせん断速度の関係）")
    hmax = 0.0
    for i, (f, lab) in enumerate(zip(["学習結果_PP.png", "学習結果_MIX.png", "学習結果_GPPS.png"], ["(1) PP", "(2) 粉砕材混合 PP", "(3) GPPS"])):
        xf = xs + i * (wf + gf)
        tbox(s, xf, yl, wf, 9.5, lab, 22, bold=True, align="center", anchor="middle", margin=0.2)
        _, hf = pic(s, f, xf, yl + 9.5, w=wf)
        hmax = max(hmax, hf + 9.5)
    yl += hmax + 1.0
    t = ("点：学習データ（成形条件ごとの平均）、線：加熱シリンダ温度ごとの予測値、帯：95% 予測区間（線と帯は、ΔP_{2} と T を代表値に固定して描いたため、"
         "薄肉の点は線からずれる）。データが密な範囲では帯が狭く、データのない低せん断速度側（約 150 1/s 以下）では広い → [r]データの量に応じた予測のばらつきが表せている[/r]")
    yl = para(s, xs, yl, wl, t, NOTE, gap=0.0)
    # ---- 右：予測精度
    xr = xs + wl + 9.0
    wr = x + w - PAD - xr
    yr = sub(s, xr, y, "予測精度（学習に用いていない射出率）")
    yr = para(s, xr, yr, wr, "検証データ（射出率の両端の条件）の粘度を予測し、実測値と比べた。", NOTE, gap=1.0)
    we = 166.0
    _, he = pic(s, "予測誤差の比較.png", xr, yr, w=we)
    t = (f"・予測誤差は約 {num['err']}%（Cross-WLF モデルは約 {num['cw']}%）\n"
         f"・検証データの [r]{num['cover']}% が 95% 予測区間に入った[/r]\n"
         f"・温度シフトファクターで、温度によるデータのずれが換算前の約 {num['ratio']} 割に減った")
    yt = para(s, xr + we + 4.0, yr + 2.0, wr - we - 4.0, t, 23, hang=1.0, spacing=1.0, gap=0.0)
    yr = max(yr + he, yt)
    yr = para(s, xr, yr + 0.5, wr, "予測誤差：粘度の対数の二乗平均平方根誤差を、粘度の % に換算した値。Cross-WLF モデルは厚みによる粘度の違いを表せないため誤差が大きい",
              SMALL, gap=0.0)
    y = max(yl, yr) + 3.0
    frame(s, x, y0, w, y)
    return y


def sec_summary(s, y, num):
    x, w = M, PAGE_W - 2 * M
    y0 = y
    y = header(s, x, y, w, "7. まとめと今後の課題")
    t = ("・金型内で計測したデータとガウス過程回帰を用いて、PP・粉砕材混合 PP・GPPS の 3 材料で、[r]樹脂粘度の予測値と予測のばらつきの範囲を"
         "同時に求められる[/r]データ駆動型樹脂粘度モデルを構築した。\n"
         f"・温度シフトファクターを組み込み、学習に用いていない射出率の条件でも、予測誤差は約 {num['err']}%、検証データの {num['cover']}% が 95% 予測区間に入った。\n"
         "・[b]今後の課題[/b]：計測データのない低せん断速度側の予測（計測データの追加や、Cross 式などの粘度式とガウス過程回帰の組み合わせを検討する）")
    y = para(s, x + PAD, y, w - 2 * PAD, t, 26, hang=1.0, gap=1.0)
    refs = "【参考文献】1) 後藤：成形加工’31, 8, 297 (2019)　2) 村田 他：成形加工’シンポジア 25, 24 (2025)　3) 持橋, 大羽：ガウス過程と機械学習, 講談社 (2019)"
    y = para(s, x + PAD, y, w - 2 * PAD, refs, 17, gap=1.0)
    frame(s, x, y0, w, y)
    return y


# ---------------------------------------------------------------- まとめて作る
def build(map_dir):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Mm(PAGE_W), Mm(PAGE_H)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    shape(s, MSO_SHAPE.RECTANGLE, 0, 0, PAGE_W, BAND_H, fill="000000")         # タイトル・発表者名のスペース（見本と同じ黒の帯。何も書かない）
    cols = []
    for x, secs in [(XL, [sec_overview, sec_mold, lambda s, x, y: sec_conditions(s, x, y, map_dir)]), (XR, [sec_experiment, sec_ml])]:
        y, spans = TOP, []
        for f in secs:
            y1 = f(s, x, y)
            spans.append((y, y1))
            y = y1 + SEC_GAP
        cols.append((x, spans))
    bottom = max(spans[-1][1] for _, spans in cols)
    for x, spans in cols:
        for i, (y0, y1) in enumerate(spans):
            frame(s, x, y0, CW, bottom if i == len(spans) - 1 else y1)
    yl, yr = (cols[0][1][-1][1], cols[1][1][-1][1])
    y = sec_results(s, bottom + SEC_GAP)
    y = sec_summary(s, y + SEC_GAP, numbers())
    print(f"左の列の下端 {yl:.1f} mm、右の列の下端 {yr:.1f} mm、最後の下端 {y:.1f} mm（ページの高さ {PAGE_H:.0f} mm、余り {PAGE_H - y:.1f} mm）")
    if y > PAGE_H - 4.0:
        print("★ページからはみ出している")
    japanese_breaks(prs)
    prs.core_properties.title = "金型内センサデータを用いたデータ駆動型樹脂粘度モデルの開発Ⅲ（簡略版）"
    return prs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=POSTER, help="pptx を出すフォルダ（既定：ポスターのフォルダ）")
    ap.add_argument("--map-dir", default=MAP_DIR, help="条件の図（成形条件と学習検証.png）のフォルダ（既定：ポスター/図_簡略版/）")
    a = ap.parse_args()
    out = os.path.join(a.out, OUT_NAME)
    if os.path.exists(out):
        sys.exit(f"すでにある：{out}（上書きしない。前のファイルを _to_delete/ へ移すか、--out で別のフォルダを指定）")
    assert os.path.exists(os.path.join(a.map_dir, MAP_NAME)), f"条件の図がない：{a.map_dir}（先に make_poster_simple_figs.py を実行）"
    prs = build(a.map_dir)
    os.makedirs(a.out, exist_ok=True)
    prs.save(out)
    print("保存:", out)


if __name__ == "__main__":
    main()
