# -*- coding: utf-8 -*-
"""A0 縦 2 枚のポスターの簡略版（3〜5 分の発表用。PowerPoint の 2 スライド）を作る：05_…/06_学習結果報告書作成_261015/ポスター/報告書Bから作成/報告書Bから作成_簡略版.pptx

今のポスター（build_poster.py が作る 報告書Bから作成.pptx）を、2026-11-04 のユーザーの依頼で簡略にしたもの。
  ・文は、発表者がいなくてもポスターだけで「何を目指して、何を行い、どうなったか」が伝わる長さにする。細かい定義・手法の注・補足の数値は、
    台本（build_poster_simple_script.py）に回す。「キャビティ入れ子の外観」の写真は使わない。
  ・同じ日の 2 回目の指摘で、次のように直した（A0 1 枚に収まらない分は、2 枚目の A0 に載せる。ユーザーの指示）：
    - 図の中の文字（凡例・軸名）と、図の説明・注の文字を大きくした（図は make_poster_simple_figs.py で、貼る大きさのまま作る）。
    - 1. 研究概要：行の間と余白を広げた。
    - 2. 計測金型およびセンサ：下にセンサの設置位置の 3D 図（見本_センサの設置位置.png）を足した。
    - 3. 成形条件と学習・検証データ：材料 × 厚み × 射出率のマスの図をやめ、表（材料ごとの条件と、学習・検証の条件数）と、
      射出率の 12 水準を学習・検証で色分けした図形に替えた。
    - 4. 実験結果：報告書B の図4 を材料ごとに切り出し、2 枚を横に並べ、その下に 1 枚と、その横に図の見方と結論の文を置いた。
    - 5. 機械学習モデルと温度シフトファクターの検討（題を変えた）：GPR の図をやめ、見本（報告書Aから作成2）と同じ文だけで説明。
      温度シフトの図の文字を大きくし、「決めた時期」と「予測の流れ」を、幅を広げて縦に並べた。
    - 6. 学習結果と予測精度検証（題を変えた）：学習結果の図を大きくし、材料ごとのパリティプロットを足した。
    - 7. まとめと今後の課題：参考文献を消し（本文の文献番号も消した）、2 枚目の下端に置いた。
  ・1 枚目の上のタイトルと発表者名のスペースは、見本（報告書Aから作成2.pdf）と同じ高さ 138.7 mm の黒の帯（何も書かない）。2 枚目には帯を置かない。
モデルと数値は、今のポスター・報告書B と同じ（B_Tff・条件平均の入力。数値は ポスター/図/ポスターの数値.csv から読む）。
部品（文字・図形・図・表）は build_poster.py のものを使う。すでにあるファイルは上書きしない（あれば止まる）。

使い方（リポジトリ直下で。先に make_poster_simple_figs.py で図を作る）:
  uv run python 04_補助スクリプト/build_poster_simple.py [--out <フォルダ>] [--fig-dir <簡略版の図のフォルダ>]
  （既定：--out は ポスター/報告書Bから作成/、--fig-dir は ポスター/報告書Bから作成/図_簡略版/。2026-11-04 にユーザーが 報告書Bから作成/ の下に移した）
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.util import Mm  # noqa: E402

import build_poster as bp  # noqa: E402
from build_poster import BLUE, GREEN, GREEN_LINE, LIGHT, PT_MM, box, est_width, frame, pic, shape, table, tbox, text_in  # noqa: E402

POSTER = bp.POSTER
FIG = bp.FIG                                            # 今のポスターの図（ポスター/図/。見本から切り出した写真・説明図）
B_DIR = os.path.join(POSTER, "報告書Bから作成")         # 報告書Bから作成のポスターの一式（2026-11-04 にユーザーが移した）
FIG_DIR = os.path.join(B_DIR, "図_簡略版")              # 簡略版の図（make_poster_simple_figs.py）
OUT_NAME = "報告書Bから作成_簡略版.pptx"

PAGE_W, PAGE_H = 841.0, 1189.0   # A0 縦 [mm]
BAND_H = 138.7                    # 1 枚目の上のタイトル・発表者名のスペース（見本の黒い帯の高さ。_tmp/poster_simple/measure_band.py で測った）
TOP = 142.7                       # 1 枚目の最初の節の上端（見本と同じ）
M = 7.0                           # 外側の余白（見本は 6.9 mm）
TOP2 = M                          # 2 枚目の最初の節の上端
BOTTOM = PAGE_H - M               # 節の下端の限り
GAP = 8.0                         # 列の間
CW = (PAGE_W - 2 * M - GAP) / 2   # 列の幅
XL, XR = M, M + CW + GAP
PAD = 5.0                         # 枠の内側の余白
IW = CW - 2 * PAD                 # 列の中身の幅
FW = PAGE_W - 2 * M               # 全幅の節の幅
FIW = FW - 2 * PAD                # 全幅の節の中身の幅
SEC_GAP = 5.0                     # 節の間
HEAD_H, HEAD_PT = 24.0, 54        # 節の見出しの帯（見本は 72 pt）
SUB_H, SUB_PT = 13.0, 30          # 小見出し
BODY = 28                         # 本文（見本は 32 pt）
NOTE = 26                         # 図の説明・表・注（11-04 の最初の版は 23 pt。小さいとの指摘で大きくした）
SMALL = 24                        # 細かい注（同 20 pt）
LINE = 1.2                        # 行の高さ（文字の大きさの倍）
C_TRAIN, C_TEST = "9DC3E6", "F4B183"                    # 学習（薄い青）・検証（橙）
C_TRAIN_T, C_TEST_T = "1F4E79", "843C0C"                # その文字の色


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


def nat(name, folder):
    """図のもとの大きさ [mm]（make_poster_simple_figs.py は、貼る大きさ・300 dpi で作っている）"""
    with Image.open(os.path.join(folder, name)) as im:
        dpi = float(im.info.get("dpi", (300, 300))[0])
        return im.width / dpi * 25.4, im.height / dpi * 25.4


def right_arrow(s, x, y, w, h):
    shape(s, MSO_SHAPE.RIGHT_ARROW, x, y, w, h, fill="FFFFFF", line=BLUE, lw=2.25)


def down_arrow(s, x, y, w, h):
    shape(s, MSO_SHAPE.DOWN_ARROW, x, y, w, h, fill="FFFFFF", line=BLUE, lw=2.25)


def scratch():
    """高さを測るための、捨てるスライド"""
    prs = Presentation()
    prs.slide_width, prs.slide_height = Mm(PAGE_W), Mm(PAGE_H)
    return prs.slides.add_slide(prs.slide_layouts[6])


def recolor_cell(cell, fill, color):
    cell.fill.solid()
    cell.fill.fore_color.rgb = RGBColor.from_string(fill)
    for p in cell.text_frame.paragraphs:
        for r in p.runs:
            r.font.color.rgb = RGBColor.from_string(color)


# ---------------------------------------------------------------- 数値（今のポスターと同じ元）
def numbers():
    t = pd.read_csv(os.path.join(FIG, "ポスターの数値.csv"), encoding="utf-8-sig")
    r = lambda v: int(math.floor(v + 0.5))
    n = {"err": [r(v) for v in t["誤差 [%] 全体"]], "cw": [r(v) for v in t["Cross-WLF 誤差 [%] 全体"]],
         "cover": [r(v * 100) for v in t["被覆率"]], "ratio": [r(v * 10) for v in t["比"]]}
    rng = lambda xs: f"{min(xs)}〜{max(xs)}"
    return {k: rng(v) for k, v in n.items()}


# ---------------------------------------------------------------- 1 枚目・左の列
def sec_overview(s, x, y, fig_dir):
    xi = x + PAD
    y = header(s, x, y, CW, "1. 研究概要") + 2.0
    t = ("　射出成形 CAE で用いられる樹脂粘度モデルでは、樹脂のロット間の違いや成形工程の熱履歴による粘度のばらつきが考慮されていない。"
         "本研究では、金型内で計測した樹脂圧力・樹脂温度のデータと機械学習を用いて、[r]樹脂粘度の予測値と、予測のばらつきの範囲を同時に求められる[/r]"
         "データ駆動型樹脂粘度モデルの構築を目指している。")
    y = para(s, xi, y, IW, t, BODY, gap=5.0, spacing=1.15)
    pts = ("[b]本報で行ったこと[/b]\n"
           "① PP・粉砕材混合 PP・GPPS の 3 材料でモデルを構築\n"
           "② 温度による粘度曲線のずれ（[r]温度シフトファクター[/r]）をモデルに組み込み\n"
           "③ [r]学習に用いていない射出率[/r]の条件で、予測の精度を検証")
    return boxed(s, xi, y, IW, pts, 26, fill=LIGHT, hang=1.0, spacing=1.15, margin=4.0, gap=5.0)


def sec_mold(s, x, y, fig_dir):
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
    hc = text_h("計測金型の外観", NOTE, w1)
    tbox(s, x0, y, w1, hc, "計測金型の外観", NOTE, align="center")
    tbox(s, x0 + w1 + gi, y, w2, hc, "キャビティの形状・寸法（mm）", NOTE, align="center")
    y += hc + 3.0
    _, h3 = pic(s, "見本_センサの設置位置.png", xi, y, w=IW)
    y += h3 + 1.0
    t = "センサの設置位置（3D 図）。左：樹脂圧力センサ（①②）、右：樹脂温度センサ（②）。圧力と温度は、キャビティブロックを交換して別々に計測"
    return para(s, xi, y, IW, t, NOTE, gap=3.0, align="center")


def conditions(fig_dir):
    return pd.read_csv(os.path.join(fig_dir, "学習検証の条件.csv"), encoding="utf-8-sig", dtype=str)


def sec_conditions(s, x, y, fig_dir):
    xi = x + PAD
    y = header(s, x, y, CW, "3. 成形条件と学習・検証データ")
    t = ("　3 材料で、加熱シリンダ温度・キャビティ厚み・射出率（12 水準）を変えて成形した。材料によって、加熱シリンダ温度とキャビティ厚みの条件が異なる。"
         "射出率の両端の条件は[r]学習に用いず[/r]、モデルの予測の答え合わせ（検証）に用いた。")
    y = para(s, xi, y, IW, t, BODY, gap=3.0)
    # ---- 表：材料ごとの条件と、学習・検証の条件数（PP と違う条件は赤）
    c = conditions(fig_dir)
    pp = c.iloc[0]
    rows = [["材料", "加熱シリンダ温度", "キャビティ厚み", "学習データ", "検証データ"]]
    for _, r in c.iterrows():
        red = lambda k: f"[r]{r[k]}[/r]" if r[k] != pp[k] else r[k]
        rows.append([r["材料"], red("加熱シリンダ温度"), red("キャビティ厚み"), f"{r['学習の条件数']} 条件", f"{r['検証の条件数']} 条件"])
    TAB = NOTE
    need = [max(est_width(row[j], TAB) for row in rows) + 6.0 for j in range(len(rows[0]))]
    extra = IW - sum(need)
    assert extra >= 0, f"表が列の幅に入らない（{-extra:.1f} mm はみ出す）"
    widths = [w + extra * w / sum(need) for w in need]
    y1 = table(s, xi, y, widths, rows, TAB, row_h=TAB * PT_MM * 1.2 + 3.0)
    tbl = s.shapes[-1].table
    recolor_cell(tbl.cell(0, 3), C_TRAIN, C_TRAIN_T)
    recolor_cell(tbl.cell(0, 4), C_TEST, C_TEST_T)
    y = y1 + 1.0
    t = "樹脂：PP はプライムポリプロ J707EG、粉砕材混合 PP は同じ PP に粉砕材を 25 wt% 混合、GPPS はディックスチレン CR-2500"
    y = para(s, xi, y, IW, t, SMALL, gap=4.0)
    # ---- 図形：射出率による学習・検証の分け方（3 材料で共通。make_poster_simple_figs.py が確かめている）
    lo, tr, hi = (pp[k].split(";") for k in ("射出率（検証・低）", "射出率（学習）", "射出率（検証・高）"))
    rates = lo + tr + hi
    y = para(s, xi, y, IW, "[b]射出率による学習・検証の分け方[/b]（3 材料で共通）", NOTE, gap=1.0)
    lab_w = 44.0
    cw = (IW - lab_w) / len(rates)
    x0 = xi + lab_w
    hl = 11.5
    for i0, n, lab, col in [(0, len(lo), "検証", C_TEST_T), (len(lo), len(tr), "学習", C_TRAIN_T), (len(lo) + len(tr), len(hi), "検証", C_TEST_T)]:
        tbox(s, x0 + i0 * cw, y, n * cw, hl, f"[b]{lab}[/b]", NOTE, color=col, align="center", anchor="middle", margin=0.2)
        shape(s, MSO_SHAPE.RECTANGLE, x0 + i0 * cw + 1.0, y + hl + 0.3, n * cw - 2.0, 1.4, fill=col)
    y += hl + 3.0
    hc = 15.0
    tbox(s, xi, y - 2.5, lab_w - 1.0, hc + 5.0, "射出率\n(cm^{3}/s)", SMALL, align="right", anchor="middle", margin=0.5)
    for i, rv in enumerate(rates):
        fill, col = (C_TRAIN, C_TRAIN_T) if len(lo) <= i < len(lo) + len(tr) else (C_TEST, C_TEST_T)
        b = shape(s, MSO_SHAPE.RECTANGLE, x0 + i * cw, y, cw, hc, fill=fill, line="FFFFFF", lw=2.0)
        text_in(b, rv, SMALL, color="000000", align="center", anchor="middle", margin=0.3)
    y += hc + 3.0
    t = (f"・低射出率側の検証は、キャビティ厚み {pp['低射出率の検証の厚み']} の条件だけ（薄肉は成形できないため）\n"
         "・学習データは 1 条件 5 ショット、検証データは 1 条件 3 ショット（外れ値などは除いた）")
    return para(s, xi, y, IW, t, NOTE, hang=1.0, gap=3.0)


# ---------------------------------------------------------------- 1 枚目・右の列
def sec_experiment(s, x, y, fig_dir, w3=None, extra=0.0):
    """w3：図3（樹脂圧力の経時変化）の幅（左右の列の下端をそろえるために変える）。extra：図の間に足す余白の合計"""
    xi = x + PAD
    w3 = IW if w3 is None else w3
    eg = extra / 3.0
    y = header(s, x, y, CW, "4. 実験結果")
    t = "　2 つのセンサの圧力の差 ΔP_{1} と、樹脂が①から②へ届くまでの時間差 Δt から、樹脂粘度 η とせん断速度 γ̇ を求めた。"
    y = para(s, xi, y, IW, t, BODY)
    _, h3 = pic(s, "樹脂圧力の経時変化.png", xi + (IW - w3) / 2, y, w=w3, folder=fig_dir)
    y += h3 + 0.5
    y = para(s, xi, y, IW, "樹脂圧力の経時変化の一例（キャビティ厚み 1.0 mm、加熱シリンダ温度条件Ⅱ、射出率 42.5 cm^{3}/s）", NOTE, gap=2.0 + eg,
             align="center")
    wf, hf = nat("算出式.png", fig_dir)
    pic(s, "算出式.png", xi, y, w=wf, folder=fig_dir)
    notes = ("ΔP_{1}：②の立ち上がり時の①②の圧力差\nΔt：①②の圧力の立ち上がりの時間差\nΔP_{2}：最大圧力時の①②の圧力差\n　　（機械学習の入力）\n"
             "L：センサ間距離、B：流路幅\nH：キャビティ厚み、Q：樹脂流量")
    xn, wn = xi + wf + 2.0, IW - wf - 2.0
    hn = text_h(notes, SMALL, wn)
    tbox(s, xn, y + max(0.0, (hf - hn) / 2), wn, hn, notes, SMALL)
    y += max(hf, hn) + 3.0 + eg
    # ---- 図4 を材料ごとに：上に 2 枚、下に 1 枚とその横に図の見方と結論
    wp = (IW - 8.0) / 2
    hl = text_h("(1) PP", NOTE, wp)
    labs = {"PP": "(1) PP", "MIX": "(2) 粉砕材混合 PP", "GPPS": "(3) GPPS"}

    def panel(mat, xp, yp):
        tbox(s, xp, yp, wp, hl, f"[b]{labs[mat]}[/b]", NOTE, align="center", anchor="middle", margin=0.2)
        _, hp = pic(s, f"実験結果_{mat}.png", xp, yp + hl, w=wp, folder=fig_dir)
        return yp + hl + hp

    y = max(panel("PP", xi, y), panel("MIX", xi + wp + 8.0, y)) + 3.0 + eg
    yb = panel("GPPS", xi, y)
    xt = xi + wp + 8.0
    h_leg = fig4_legend(scratch(), xt, 0.0, wp)
    fig4_legend(s, xt, y + max(0.0, (yb - y - h_leg) / 2), wp)
    y = max(yb, y + h_leg) + 3.0
    # 結論（図4 の下に全幅で）
    bw = IW - 150.0
    concl = "高せん断速度側は、キャピラリーレオメータの\n値（白丸）とおおむね一致"
    yb2 = boxed(s, xi, y, bw, concl, NOTE, anchor="middle", gap=0.0)
    hb = yb2 - y
    right_arrow(s, xi + bw + 3.0, y + hb / 2 - 7.0, 14.0, 14.0)
    box(s, xi + bw + 20.0, y, IW - bw - 20.0, hb, "[r]学習データとして使用可能[/r]", NOTE, anchor="middle", align="center", margin=1.5)
    return yb2 + 3.0


FIG4_THICK = [("0.5", "FF0000"), ("1.0", "0000FF"), ("1.5", "F0BB02"), ("2.0", "00FF00")]   # 報告書B の図4 の厚みの色（_tmp/poster_simple/fig4_colors.py で図から読んだ）


def fig4_legend(s, x, y, w):
    """報告書B の図4 の凡例を、大きい日本語の文字と図形で描く（図の中の凡例は英語で小さいため。図そのものは変えない）。返り値：高さ"""
    y0 = y
    lh = NOTE * PT_MM * LINE + 3.5                         # 1 行の高さ（文字の箱）

    def label(xx, yy, ww, t, size=NOTE):
        tbox(s, xx, yy, ww, lh, t, size, anchor="middle", margin=0.5)

    label(x, y, w, "[b]図の見方[/b]")
    y += lh + 1.0
    # キャピラリーレオメータ（白丸と黒線）
    sw = 26.0
    yc = y + lh / 2
    shape(s, MSO_SHAPE.RECTANGLE, x + 1.0, yc - 0.6, sw - 4.0, 1.2, fill="000000")
    shape(s, MSO_SHAPE.OVAL, x + sw / 2 - 4.5, yc - 3.5, 7.0, 7.0, fill="FFFFFF", line="000000", lw=2.25)
    label(x + sw, y, w - sw, "キャピラリーレオメータ")
    y += lh
    label(x + sw, y, w - sw, "（210 ℃。GPPS は 220 ℃）", SMALL)
    y += lh + 3.0
    # 加熱シリンダ温度（形）
    label(x, y, w, "形：加熱シリンダ温度")
    y += lh
    xx = x + 4.0
    for kind, t in [(MSO_SHAPE.OVAL, "190"), (MSO_SHAPE.ISOSCELES_TRIANGLE, "210"), (MSO_SHAPE.RECTANGLE, "230 ℃")]:
        shape(s, kind, xx, y + lh / 2 - 3.5, 7.0, 7.0, fill="404040", line="000000", lw=1.5)
        tw = est_width(t, NOTE) * 1.15 + 5.0                # 狭いと数字の途中で折り返すので、広めにとる
        label(xx + 8.5, y, tw, t)
        xx += 8.5 + tw + 6.0
    y += lh
    label(x + 4.0, y, w - 4.0, "（GPPS は 200・220・240 ℃）", SMALL)
    y += lh + 3.0
    # キャビティ厚み（色）
    label(x, y, w, "色：キャビティ厚み")
    y += lh
    xx = x + 4.0
    for i, (t, col) in enumerate(FIG4_THICK):
        t = t + (" mm" if i == len(FIG4_THICK) - 1 else "")
        shape(s, MSO_SHAPE.RECTANGLE, xx, y + lh / 2 - 2.5, 11.0, 5.0, fill=col)
        tw = est_width(t, NOTE) * 1.15 + 5.0
        label(xx + 12.5, y, tw, t)
        xx += 12.5 + tw + 4.0
    assert xx - x < w + 4.0, "図の見方の厚みの行が、幅に入らない"
    y += lh
    return y - y0


# ---------------------------------------------------------------- 1 枚目・下の全幅
def sec_ml(s, y, fig_dir, bottom=None):
    """bottom：節の下端（下まで広げるとき。3 つの囲みの間の余白を広げて、そろえる）"""
    x = M
    xs = x + PAD
    y0 = y
    y = header(s, x, y, FW, "5. 機械学習モデルと温度シフトファクターの検討")
    t = ("　学習データに対して取り得る関数のガウス分布が得られるため、樹脂粘度の予測を平均値と予測幅で求めることが可能な[r]ガウス過程回帰モデルを選定[/r]"
         "して、機械学習を行った。\n"
         "　さらに、Cross-WLF モデルと同様に、温度による粘度曲線の横・縦のずれを、[r]金型内樹脂最高温度 T に基づく温度シフトファクター a_{T}[/r] "
         "で表し、モデルに組み込んだ。")
    y = para(s, xs, y, FIW, t, BODY, gap=3.0)
    # ---- 3 つの囲み（考え方・決めた時期・予測の流れ）
    wa, ha = nat("温度シフト概念図.png", fig_dir)
    gap3 = 12.0
    wb = (FIW - wa - 2 * gap3) * 0.47
    wc = FIW - wa - 2 * gap3 - wb
    xa, xb, xc = xs, xs + wa + gap3, xs + wa + gap3 + wb + gap3
    cap = "横・縦にずらすと、温度の違う曲線が 1 本に重なる"
    h_cap = text_h(cap, NOTE, wa)
    ya = sub(s, xa, y, "温度シフトファクターの考え方")
    hA_nat = (ya - y) + ha + 1.0 + h_cap
    hA = hA_nat if bottom is None else max(hA_nat, bottom - 3.0 - y)
    # 考え方（図と説明。下まで広げたときは、上下の真ん中に置く）
    ya += (hA - hA_nat) / 2
    pic(s, "温度シフト概念図.png", xa, ya, w=wa, folder=fig_dir)
    tbox(s, xa, ya + ha + 1.0, wa, h_cap, cap, NOTE, align="center")
    # 決めた時期
    yb = sub(s, xb, y, "温度シフトファクターを決めた時期")
    tag_w = 60.0
    steps = [("学習の前", "式の形（PP・粉砕材混合 PP：アレニウス型、GPPS：WLF 型）と、基準温度を決めた"),
             ("学習の\nとき", "ずれの大きさ（横・縦の 2 つの係数）を、GPR と同時に、学習データに最もよく合うように決めた"),
             ("学習の後\n（予測）", "係数は固定。予測のたびに、T から a_{T} を計算して使う（右の流れ）")]
    note = "a_{T}：基準温度（PP・粉砕材混合 PP は 210 ℃、GPPS は 220 ℃）からの、温度によるずれの大きさ"
    hs = [max(text_h(b, SMALL, wb - tag_w - 3.0, margin=2.5), text_h(t_, SMALL, tag_w, margin=1.5)) for t_, b in steps]
    h_note = text_h(note, SMALL, wb)
    ag_b = 14.0                                          # 矢印の段の高さ。余った高さは、囲みを高くして使う
    room_b = y + hA - yb - h_note - 2.0
    hs = [h + max(0.0, room_b - sum(hs) - ag_b * (len(steps) - 1)) / len(steps) for h in hs]
    yy = yb
    for i, ((tag, body), h) in enumerate(zip(steps, hs)):
        tg = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, xb, yy, tag_w, h, fill=BLUE, radius=0.12)
        text_in(tg, tag, SMALL, color="FFFFFF", bold=True, align="center", anchor="middle", margin=1.0)
        box(s, xb + tag_w + 3.0, yy, wb - tag_w - 3.0, h, body, SMALL, anchor="middle")
        yy += h
        if i < len(steps) - 1:
            down_arrow(s, xb + tag_w / 2 - 8.0, yy + 2.0, 16.0, ag_b - 4.0)
            yy += ag_b
    tbox(s, xb, yy + 2.0, wb, h_note, note, SMALL)
    yy += 2.0 + h_note
    # 予測の流れ
    yc = sub(s, xc, y, "学習したモデルでの予測の流れ")
    items = [("[b]入力[/b]：せん断速度 γ̇、樹脂圧力差 ΔP_{2}、金型内樹脂最高温度 T、加熱シリンダ温度", LIGHT, "000000"),
             ("① T から a_{T} を計算\n② γ̇ を、基準温度での値に換算（横のずれ）", "FFFFFF", "000000"),
             ("[b]GPR[/b]：基準温度での粘度を予測", BLUE, "FFFFFF"),
             ("③ a_{T} で、縦のずれを戻す", "FFFFFF", "000000"),
             ("[b]出力[/b]：樹脂粘度 η の[r]予測値と 95% 予測区間[/r]", LIGHT, "000000")]
    hb = [text_h(t_, SMALL, wc, margin=2.5) for t_, _, _ in items]
    ag_c = 12.0
    room_c = y + hA - yc
    hb = [h + max(0.0, room_c - sum(hb) - ag_c * (len(items) - 1)) / len(items) for h in hb]
    yy2 = yc
    for i, ((t_, fill, col), h) in enumerate(zip(items, hb)):
        b = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, xc, yy2, wc, h, fill=fill, line=BLUE, lw=2.0, radius=0.08)
        text_in(b, t_, SMALL, color=col, align="center", anchor="middle", margin=2.5)
        yy2 += h
        if i < len(items) - 1:
            down_arrow(s, xc + wc / 2 - 8.0, yy2 + 2.0, 16.0, ag_c - 4.0)
            yy2 += ag_c
    y_end = max(y + hA, yy, yy2) + 3.0
    if bottom is not None:
        y_end = max(y_end, bottom)
    frame(s, x, y0, FW, y_end)
    return y_end


# ---------------------------------------------------------------- 2 枚目
def sec_results(s, y, fig_dir, bottom=None):
    """bottom：節の下端（まとめの上まで広げる。余った高さは、図の段の間に分けて足す）"""
    x = M
    xs = x + PAD
    y0 = y
    num = numbers()
    mats = [("PP", "(1) PP"), ("MIX", "(2) 粉砕材混合 PP"), ("GPPS", "(3) GPPS")]

    def body(s, y, eg):
        y = header(s, x, y, FW, "6. 学習結果と予測精度検証")
        # ---- 学習結果
        y = sub(s, xs, y, "学習結果（樹脂粘度とせん断速度の関係）")
        wf, hf = nat("学習結果_PP.png", fig_dir)
        gf = (FIW - 3 * wf) / 2
        hl = text_h("(1) PP", NOTE, wf)
        for i, (mat, lab) in enumerate(mats):
            xf = xs + i * (wf + gf)
            tbox(s, xf, y, wf, hl, f"[b]{lab}[/b]", NOTE, align="center", anchor="middle", margin=0.2)
            pic(s, f"学習結果_{mat}.png", xf, y + hl, w=wf, folder=fig_dir)
        y += hl + hf + 1.0
        t = ("点：学習データ（成形条件ごとの平均）、線：加熱シリンダ温度ごとの予測値、帯：95% 予測区間（線と帯は、ΔP_{2} と T を代表値に固定して描いたため、"
             "薄肉の点は線からずれる）。データが密な範囲では帯が狭く、データのない低せん断速度側（約 150 1/s 以下）では広い → "
             "[r]データの量に応じた予測のばらつきが表せている[/r]")
        y = para(s, xs, y, FIW, t, NOTE, gap=4.0 + eg)
        # ---- 予測精度検証
        y = sub(s, xs, y, "予測精度検証（学習に用いていない射出率）")
        y = para(s, xs, y, FIW, "　「3. 成形条件と学習・検証データ」の検証データ（射出率の両端の条件）の粘度を予測し、実測値と比べた。", BODY, gap=1.0)
        wp, hp = nat("パリティ_PP.png", fig_dir)
        gp = (FIW - 3 * wp) / 2
        for i, (mat, lab) in enumerate(mats):
            xf = xs + i * (wp + gp)
            tbox(s, xf, y, wp, hl, f"[b]{lab}[/b]", NOTE, align="center", anchor="middle", margin=0.2)
            pic(s, f"パリティ_{mat}.png", xf, y + hl, w=wp, folder=fig_dir)
        y += hl + hp + 1.0
        t = "横軸：実測値、縦軸：予測値、縦線：95% 予測区間。点が破線（予測値 ＝ 実測値）に近いほど、予測が当たっている"
        y = para(s, xs, y, FIW, t, NOTE, gap=3.0 + eg, align="center")
        y = para(s, xs, y, FIW, "[b]予測誤差の比較[/b]（3 材料の検証データ全体。本モデルと Cross-WLF モデル）", NOTE, gap=1.0)
        we, he = nat("予測誤差の比較.png", fig_dir)
        pic(s, "予測誤差の比較.png", xs, y, w=we, folder=fig_dir)
        xr, wr = xs + we + 12.0, FIW - we - 12.0
        t = (f"・予測誤差は約 {num['err']}%（Cross-WLF モデルは約 {num['cw']}%）\n"
             f"・検証データの [r]{num['cover']}% が 95% 予測区間に入った[/r]\n"
             f"・温度シフトファクターで、温度によるデータのずれが換算前の約 {num['ratio']} 割に減った")
        note = "予測誤差：粘度の対数の二乗平均平方根誤差を、粘度の % に換算した値。Cross-WLF モデルは、厚みによる粘度の違いを表せないため誤差が大きい"
        h_t = text_h(t, BODY, wr, hang=True, spacing=1.15)
        h_n = text_h(note, SMALL, wr)
        yt = y + max(0.0, (he - h_t - 6.0 - h_n) / 2)
        tbox(s, xr, yt, wr, h_t, t, BODY, hang=1.0, spacing=1.15)
        tbox(s, xr, yt + h_t + 6.0, wr, h_n, note, SMALL)
        return max(y + he, yt + h_t + 6.0 + h_n) + 3.0

    eg = 0.0
    if bottom is not None:
        eg = max(0.0, (bottom - body(scratch(), y, 0.0)) / 2.0)
    y = body(s, y, eg)
    y_end = max(y, bottom) if bottom is not None else y
    frame(s, x, y0, FW, y_end)
    return y_end


def sec_summary(s, y, num):
    x, w = M, FW
    y0 = y
    y = header(s, x, y, w, "7. まとめと今後の課題")
    t = ("・金型内で計測したデータとガウス過程回帰を用いて、PP・粉砕材混合 PP・GPPS の 3 材料で、[r]樹脂粘度の予測値と予測のばらつきの範囲を"
         "同時に求められる[/r]データ駆動型樹脂粘度モデルを構築した。\n"
         f"・温度シフトファクターを組み込み、学習に用いていない射出率の条件でも、予測誤差は約 {num['err']}%、検証データの {num['cover']}% が 95% 予測区間に入った。\n"
         "・[b]今後の課題[/b]：計測データのない低せん断速度側の予測（計測データの追加や、Cross 式などの粘度式とガウス過程回帰の組み合わせを検討する）")
    y = para(s, x + PAD, y, w - 2 * PAD, t, BODY, hang=1.0, spacing=1.1, gap=3.0)
    frame(s, x, y0, w, y)
    return y


# ---------------------------------------------------------------- まとめて作る
def build(fig_dir):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Mm(PAGE_W), Mm(PAGE_H)
    # ================= 1 枚目
    s = prs.slides.add_slide(prs.slide_layouts[6])
    shape(s, MSO_SHAPE.RECTANGLE, 0, 0, PAGE_W, BAND_H, fill="000000")         # タイトル・発表者名のスペース（見本と同じ黒の帯。何も書かない）
    spans_l, y = [], TOP
    for f in (sec_overview, sec_mold, sec_conditions):
        y1 = f(s, XL, y, fig_dir)
        spans_l.append((y, y1))
        y = y1 + SEC_GAP
    yl = spans_l[-1][1]
    # 右の列：図3 の幅を、左の列の下端にそろうように決める（狭くても 300 mm。足りなければ、図の間に余白を足す）
    r3 = ratio("樹脂圧力の経時変化.png", fig_dir)
    full = sec_experiment(scratch(), XR, TOP, fig_dir)
    w3 = IW if full <= yl else max(300.0, IW - (full - yl) / r3)
    yr0 = sec_experiment(scratch(), XR, TOP, fig_dir, w3)
    extra = max(0.0, yl - yr0)
    yr = sec_experiment(s, XR, TOP, fig_dir, w3, extra)
    bottom = max(yl, yr)
    for i, (y0, y1) in enumerate(spans_l):
        frame(s, XL, y0, CW, bottom if i == len(spans_l) - 1 else y1)
    frame(s, XR, TOP, CW, bottom)
    y5_nat = sec_ml(scratch(), bottom + SEC_GAP, fig_dir)
    y5 = sec_ml(s, bottom + SEC_GAP, fig_dir, bottom=BOTTOM if y5_nat <= BOTTOM else None)
    print(f"1 枚目：左の列の下端 {yl:.1f} mm、右の列の下端 {yr:.1f} mm（図3 の幅 {w3:.1f} mm、足した余白 {extra:.1f} mm）、"
          f"5. の下端 {y5:.1f} mm（広げる前 {y5_nat:.1f} mm。限り {BOTTOM:.0f} mm）")
    if y5 > BOTTOM + 0.5:
        print("★1 枚目がページからはみ出している")
    # ================= 2 枚目
    s2 = prs.slides.add_slide(prs.slide_layouts[6])
    num = numbers()
    h7 = sec_summary(scratch(), 0.0, num)
    y7 = BOTTOM - h7
    y6_nat = sec_results(scratch(), TOP2, fig_dir)
    y6 = sec_results(s2, TOP2, fig_dir, bottom=y7 - SEC_GAP if y6_nat <= y7 - SEC_GAP else None)
    y_end = sec_summary(s2, max(y7, y6 + SEC_GAP), num)
    print(f"2 枚目：6. の下端 {y6:.1f} mm（広げる前 {y6_nat:.1f} mm）、7. は {max(y7, y6 + SEC_GAP):.1f}〜{y_end:.1f} mm（限り {BOTTOM:.0f} mm）")
    if y_end > BOTTOM + 0.5:
        print("★2 枚目がページからはみ出している")
    japanese_breaks(prs)
    prs.core_properties.title = "金型内センサデータを用いたデータ駆動型樹脂粘度モデルの開発Ⅲ（簡略版）"
    return prs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=B_DIR, help="pptx を出すフォルダ（既定：ポスター/報告書Bから作成/）")
    ap.add_argument("--fig-dir", default=FIG_DIR, help="簡略版の図（make_poster_simple_figs.py）のフォルダ（既定：ポスター/報告書Bから作成/図_簡略版/）")
    a = ap.parse_args()
    out = os.path.join(a.out, OUT_NAME)
    if os.path.exists(out):
        sys.exit(f"すでにある：{out}（上書きしない。前のファイルを _to_delete/ へ移すか、--out で別のフォルダを指定）")
    assert os.path.exists(os.path.join(a.fig_dir, "学習検証の条件.csv")), f"簡略版の図がない：{a.fig_dir}（先に make_poster_simple_figs.py を実行）"
    prs = build(a.fig_dir)
    os.makedirs(a.out, exist_ok=True)
    prs.save(out)
    print("保存:", out)


if __name__ == "__main__":
    main()
