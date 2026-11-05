# -*- coding: utf-8 -*-
"""A0 縦 1 枚の簡略版ポスター（1〜4 節だけ。5 節以降の場所は空白）を作る：ポスター/報告書Bから作成/報告書Bから作成_簡略版2.pptx

2026-11-05 のユーザーの依頼：今の簡略版（A0 2 枚。build_poster_simple.py）を、最終的に A0 1 枚に収めたい。とりあえず次だけを直し、
1〜4 節だけを残して、それ以外を空白にした版を作る。もとの簡略版（報告書Bから作成_簡略版.pptx）は残し、この版は「簡略版2」とする（ユーザーの指示）。
  ・文字は、グラフの中も含めて 28 pt 以上（図は make_poster_one_figs.py が、貼る大きさのまま作る）。
    例外：報告書A から持ってきた写真・説明図（金型の写真・寸法図・3D 図・キャビティの模式図）の中の文字（ユーザーの回答）と、上付き・下付きの文字。
  ・1. 研究概要：見本（報告書Aから作成2）と全く同じ内容（赤字の位置も同じ）。
  ・3. 成形条件：見本に寄せて情報を減らす。表をやめ、各条件の水準の数を文で書く。題は「成形条件」だけ。学習・検証の分け方は 1 文だけ残す（ユーザーの回答）。
  ・4. 実験結果：報告書Bから作成の 4 節から、箇条書き（「・樹脂圧力は、センサ位置①→②の順に立ち上がる」などの 5 行）を除いた版。
    樹脂粘度とせん断速度の算出は、経時変化の隣（下）に移す。樹脂粘度とせん断速度のグラフ 3 つは、グラフエリアを正方形にして横に並べ、
    縦軸の名前と目盛りの数字は左端だけ（ユーザーの回答）、グラフの下の題は日本語の材料名（バージンPP・リサイクルPP・GPPS）。
  ・並べ方（2026-11-05 のユーザーの指示で改めた）：1〜4 は縦に 2 列に分ける（左の列に 1→2→3、右の列に 4）。3 の終わりと 4 の終わりをそろえる。
    その下に、機械学習結果・予測精度検証結果・まとめを、分けずに横長（全幅）で載せる予定なので、そこは空白にしておく。
    下端をそろえるため、2 つの列の高さを測り、短いほうの列の図と文の間（左の列は 4 か所、右の列は 4 か所）を、同じ幅ずつ空ける。
2. 計測金型およびセンサは、今の簡略版のまま（図の説明の文字だけ 28 pt に）。上のタイトル・発表者名のスペース（黒の帯）も同じ。
部品は build_poster.py・build_poster_simple.py のものを使う。すでにあるファイルは上書きしない（あれば止まる）。

使い方（リポジトリ直下で。先に make_poster_one_figs.py で図を作る）:
  uv run python 04_補助スクリプト/build_poster_one.py [--out <フォルダ>] [--fig-dir <図のフォルダ>]
  （既定：--out は ポスター/報告書Bから作成/、--fig-dir は ポスター/報告書Bから作成/図_簡略版_A0一枚/）
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from pptx import Presentation  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.util import Mm, Pt  # noqa: E402

import build_poster as bp  # noqa: E402
import build_poster_simple as bs  # noqa: E402
from build_poster import PT_MM, box, frame, pic, shape, tbox  # noqa: E402
from build_poster_simple import header, japanese_breaks, nat, para, ratio, right_arrow, scratch, sub, text_h  # noqa: E402

POSTER = bp.POSTER
FIG = bp.FIG                                                      # 写真・説明図（ポスター/図/。見本から切り出したもの）
B_DIR = os.path.join(POSTER, "報告書Bから作成")
FIG_DIR = os.path.join(B_DIR, "図_簡略版_A0一枚")                  # make_poster_one_figs.py の図
OUT_NAME = "報告書Bから作成_簡略版2.pptx"

PAGE_W, PAGE_H = bs.PAGE_W, bs.PAGE_H                             # A0 縦 [mm]
BAND_H, TOP, M, GAP = bs.BAND_H, bs.TOP, bs.M, bs.GAP             # 黒の帯の高さ・最初の節の上端・外側の余白・列の間（今の簡略版と同じ）
BOTTOM = PAGE_H - M
CW, XL, XR, PAD, IW = bs.CW, bs.XL, bs.XR, bs.PAD, bs.IW          # 列の幅・左右の列の左端・枠の内側の余白・列の中身の幅（399.5 mm）
SEC_GAP = bs.SEC_GAP
MIN = 28                                                          # 文字の大きさの下限 [pt]（ユーザーの指示）
BODY = NOTE = MIN                                                 # 本文・図の説明・凡例
FIG4_THICK = bs.FIG4_THICK                                        # 報告書B の図4 の厚みの色
SLOTS = 4                                                         # 下端をそろえるときに空ける場所の数（左の列・右の列とも）


# ---------------------------------------------------------------- 左の列（1〜3）
def sec_overview(s, x, y):
    """1. 研究概要：見本（報告書Aから作成2）と同じ文（赤字の位置も同じ）"""
    xi = x + PAD
    y = header(s, x, y, CW, "1. 研究概要") + 2.0
    t = ("　本研究では、各種レオメータに代わって、金型内において[r]樹脂圧力と温度を計測することができる計測金型を設計・製作[/r]し、"
         "それにより得られた計測データと機械学習を用いて、樹脂粘度の予測値に加えて、予測値のばらつきの範囲も同時に推定することができる"
         "[r]新たなデータ駆動型樹脂粘度モデルの構築[/r]を目的としている。")
    return para(s, xi, y, IW, t, BODY, gap=3.0, spacing=1.1)


def sec_mold(s, x, y, eg=0.0):
    """2. 計測金型およびセンサ：今の簡略版（build_poster_simple.sec_mold）と同じ。図の説明の文字だけ 28 pt。eg：図と文の間に足す幅（3 か所）"""
    xi = x + PAD
    y = header(s, x, y, CW, "2. 計測金型およびセンサ")
    y = para(s, xi, y, IW, "　キャビティ内の 2 か所（センサ位置①②）で、流動中の樹脂の圧力と温度を計測できる計測金型を用いた。", BODY) + eg
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
    y += hc + 3.0 + eg
    _, h3 = pic(s, "見本_センサの設置位置.png", xi, y, w=IW)
    y += h3 + 1.0 + eg
    t = "センサの設置位置（3D 図）。左：樹脂圧力センサ（①②）、右：樹脂温度センサ（②）。圧力と温度は、キャビティブロックを交換して別々に計測"
    return para(s, xi, y, IW, t, NOTE, gap=3.0, align="center")


def hanging(s, x, y, w, lines, size, head_em):
    """1 行目の頭の head_em 字分（「成形材料：」）を出し、2 行目以降の段落をその右にそろえて書く。下端を返す"""
    hang = head_em * size * PT_MM
    h = sum(bs.n_lines(t, size, w - hang - 3.0) for t in lines) * size * PT_MM * bs.LINE + 3.5
    tb = tbox(s, x, y, w, h, "\n".join(lines), size)
    for i, p in enumerate(tb.text_frame.paragraphs):
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(head_em * size * 12700)))
        pPr.set("indent", str(-int(head_em * size * 12700)) if i == 0 else "0")
    return y + h


def sec_conditions(s, x, y, eg=0.0):
    """3. 成形条件：見本（報告書Aから作成2）の形。材料と、各条件の水準の数を文で書く（表と射出率の帯はやめた）。eg：材料と文の間に足す幅（1 か所）"""
    xi = x + PAD
    y = header(s, x, y, CW, "3. 成形条件")
    mats = ["成形材料：バージンPP（プライムポリプロ J707EG、㈱プライムポリマー）",
            "リサイクルPP（バージンPP に一次成形品の粉砕材を 25 wt% 混合）",
            "GPPS（ディックスチレン CR-2500、DIC㈱）"]
    y = hanging(s, xi, y, IW, mats, BODY, 4.45) + 1.0 + eg      # 4.45 字：「成形材料：」の幅（BIZ UDPゴシックの「：」は狭い。PDF の画像で測った）
    t = ("　加熱シリンダ温度を 3 条件、射出率を 12 条件、キャビティ厚みを 4 条件（GPPS は 3 条件）変化させて成形し、"
         "機械学習に用いる流動中樹脂におけるデータ収集を行った。射出率の両端の条件は[r]学習に用いず[/r]、予測の検証に用いた。")
    return para(s, xi, y, IW, t, BODY, gap=3.0)


def left_column(s, eg=0.0):
    """左の列（1〜3）を書き、各節の (上端, 下端) を返す（枠は build で描く）。eg：空ける幅（2 節に 3 か所、3 節に 1 か所）"""
    spans, y = [], TOP
    for f in (sec_overview, lambda s_, x_, y_: sec_mold(s_, x_, y_, eg), lambda s_, x_, y_: sec_conditions(s_, x_, y_, eg)):
        y1 = f(s, XL, y)
        spans.append((y, y1))
        y = y1 + SEC_GAP
    return spans


# ---------------------------------------------------------------- 右の列（4）
def flow_legend(s, x, y, w):
    """樹脂粘度とせん断速度のグラフの凡例（3 つに共通。図形と 28 pt の文字）。返り値：高さ"""
    y0 = y
    lh = NOTE * PT_MM * bs.LINE + 1.5                                 # 1 行の高さ（行の間は詰めた）

    def label(xx, yy, t):
        ww = sum(1.0 if ord(c) > 0x2000 else 0.65 for c in t) * NOTE * PT_MM + 3.0     # 文字の箱の幅（BIZ UDPゴシックの半角は 0.55 em より広いので 0.65 em とした）
        tbox(s, xx, yy, ww, lh, t, NOTE, anchor="middle", margin=0.5)
        return xx + ww

    # キャピラリーレオメータ（白丸と黒線）
    sw = 26.0
    yc = y + lh / 2
    shape(s, MSO_SHAPE.RECTANGLE, x + 1.0, yc - 0.6, sw - 4.0, 1.2, fill="000000")
    shape(s, MSO_SHAPE.OVAL, x + sw / 2 - 4.5, yc - 3.5, 7.0, 7.0, fill="FFFFFF", line="000000", lw=2.25)
    label(x + sw, y, "キャピラリーレオメータ（210 ℃。GPPS は 220 ℃）")
    y += lh
    # 形：加熱シリンダ温度（報告書B の図4 と同じく、線だけの印）
    xx = label(x, y, "形：加熱シリンダ温度") + 2.0
    for kind, t in [(MSO_SHAPE.OVAL, "190"), (MSO_SHAPE.ISOSCELES_TRIANGLE, "210"), (MSO_SHAPE.RECTANGLE, "230 ℃")]:
        shape(s, kind, xx, y + lh / 2 - 3.5, 7.0, 7.0, fill="FFFFFF", line="000000", lw=2.0)
        xx = label(xx + 8.5, y, t) + 3.0
    xx = label(xx, y, "（GPPS は 200・220・240 ℃）")
    assert xx - x < w + 2.0, f"凡例の形の行が、幅に入らない（{xx - x:.0f} mm）"
    y += lh
    # 色：キャビティ厚み
    xx = label(x, y, "色：キャビティ厚み") + 2.0
    for i, (t, col) in enumerate(FIG4_THICK):
        t = t + (" mm" if i == len(FIG4_THICK) - 1 else "")
        shape(s, MSO_SHAPE.RECTANGLE, xx, y + lh / 2 - 2.5, 11.0, 5.0, fill=col)
        xx = label(xx + 12.5, y, t) + 3.0
    assert xx - x < w + 2.0, f"凡例の色の行が、幅に入らない（{xx - x:.0f} mm）"
    y += lh
    return y - y0


def sec_experiment(s, x, y, fig_dir, eg=0.0):
    """4. 実験結果（右の列にまとめた）：経時変化 → 算出 → 樹脂粘度とせん断速度の関係。eg：小見出しの前などに足す幅（4 か所）。下端を返す"""
    xi = x + PAD
    y = header(s, x, y, CW, "4. 実験結果")
    # ---- 樹脂圧力と樹脂温度の経時変化の一例（報告書Bから作成の 4 節から、箇条書きを除いた）
    y = sub(s, xi, y, "樹脂圧力と樹脂温度の経時変化の一例")
    wp, hp = nat("樹脂圧力の経時変化.png", fig_dir)
    wt, ht = nat("樹脂温度の経時変化.png", fig_dir)
    g = IW - wp - wt
    assert g >= 0, f"圧力と温度のグラフが列の幅に入らない（{-g:.1f} mm はみ出す）"
    pic(s, "樹脂圧力の経時変化.png", xi, y, w=wp, folder=fig_dir)
    pic(s, "樹脂温度の経時変化.png", xi + wp + g, y, w=wt, folder=fig_dir)
    y += max(hp, ht) + 1.0
    t = "キャビティ厚み 1.0 mm、加熱シリンダ温度条件Ⅱ、射出率 42.5 cm^{3}/s\n（左：樹脂圧力、右：樹脂温度）"
    y = para(s, xi, y, IW, t, NOTE, gap=3.0, align="center") + eg
    # ---- 樹脂粘度とせん断速度の算出
    y = sub(s, xi, y, "樹脂粘度とせん断速度の算出（ニュートン流体の計算式）")
    ws = 200.0
    wf, hf = nat("算出式.png", fig_dir)
    _, hs = pic(s, "見本_キャビティの模式図.png", xi, y, w=ws)
    pic(s, "算出式.png", xi + IW - wf, y + max(0.0, (hs - hf) / 2), w=wf, folder=fig_dir)
    y += max(hs, hf) + 1.0
    # 流路幅 B・流路高さ H は模式図の中に書いてあるので、注は 1 行に（3 節と 4 節の下端をそろえるため、2 行から縮めた）
    y = para(s, xi, y, IW, "L：センサ間距離（17.5 mm）、H：キャビティ厚み、Q：樹脂流量、V：流速", NOTE, gap=4.0) + eg
    # ---- 樹脂粘度とせん断速度の関係
    y = sub(s, xi, y, "樹脂粘度とせん断速度の関係")
    wg, hg = nat("樹脂粘度とせん断速度の関係.png", fig_dir)
    assert wg <= IW + 0.5, "樹脂粘度のグラフが列の幅に入らない"
    pic(s, "樹脂粘度とせん断速度の関係.png", xi, y, w=wg, folder=fig_dir)
    y += hg + 2.0 + eg
    y += flow_legend(s, xi + 2.0, y, IW - 2.0) + 4.0 + eg
    # 報告書Bから作成と同じ 3 つの囲み
    t1 = "せん断速度の増加、\n加熱シリンダ温度の上昇に\n伴い、粘度が低下"                       # 改行の位置は、語の途中で折り返さないように決めた
    t2 = "高せん断速度側のプロットは、\nキャピラリーレオメータの計測\nデータとおおむね一致"
    t3 = "[r]学習データとして[/r]\n[r]使用可能[/r]"                                              # 赤の指定は段落（行）ごと
    aw, gg = 14.0, 3.0
    w1, w2 = 132.0, 152.0
    w3 = IW - w1 - w2 - aw - 4 * gg
    hb = max(text_h(t1, NOTE, w1, margin=2.5), text_h(t2, NOTE, w2, margin=2.5), text_h(t3, NOTE, w3, margin=2.5))
    box(s, xi, y, w1, hb, t1, NOTE, anchor="middle")
    box(s, xi + w1 + gg, y, w2, hb, t2, NOTE, anchor="middle")
    right_arrow(s, xi + w1 + w2 + 2 * gg, y + hb / 2 - 7.0, aw, 14.0)
    box(s, xi + w1 + w2 + aw + 3 * gg, y, w3, hb, t3, NOTE, anchor="middle", align="center")
    return y + hb + 3.0


# ---------------------------------------------------------------- まとめて作る
def build(fig_dir):
    # 下端をそろえる：2 つの列の高さを、捨てるスライドで測り、短いほうの列の図と文の間を空ける
    yl0 = left_column(scratch())[-1][1]
    yr0 = sec_experiment(scratch(), XR, TOP, fig_dir)
    bottom = max(yl0, yr0)
    egl, egr = (bottom - yl0) / SLOTS, (bottom - yr0) / SLOTS
    prs = Presentation()
    prs.slide_width, prs.slide_height = Mm(PAGE_W), Mm(PAGE_H)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    shape(s, MSO_SHAPE.RECTANGLE, 0, 0, PAGE_W, BAND_H, fill="000000")       # タイトル・発表者名のスペース（見本と同じ黒の帯。何も書かない）
    spans = left_column(s, egl)
    yr = sec_experiment(s, XR, TOP, fig_dir, egr)
    assert abs(spans[-1][1] - bottom) < 0.5 and abs(yr - bottom) < 0.5, f"下端がそろっていない（左 {spans[-1][1]:.1f}、右 {yr:.1f}、目標 {bottom:.1f} mm）"
    for y0, y1 in spans:
        frame(s, XL, y0, CW, y1)
    frame(s, XR, TOP, CW, bottom)
    names = ["1. 研究概要", "2. 計測金型およびセンサ", "3. 成形条件"]
    print("左の列：" + "、".join(f"{n} {a:.0f}〜{b:.0f} mm" for n, (a, b) in zip(names, spans)) + f"（そろえる前の下端 {yl0:.0f} mm、空けた幅 {egl:.1f} mm × {SLOTS}）")
    print(f"右の列：4. 実験結果 {TOP:.0f}〜{bottom:.0f} mm（そろえる前の下端 {yr0:.0f} mm、空けた幅 {egr:.1f} mm × {SLOTS}）")
    print(f"空白（5 節以降を全幅で載せる場所）：{bottom + SEC_GAP:.0f}〜{BOTTOM:.0f} mm（高さ {BOTTOM - bottom - SEC_GAP:.0f} mm、幅 {PAGE_W - 2 * M:.0f} mm）")
    assert bottom <= BOTTOM + 0.5, "ページからはみ出している"
    japanese_breaks(prs)
    small = sorted({(r.text, r.font.size.pt) for shp in s.shapes if shp.has_text_frame for p in shp.text_frame.paragraphs for r in p.runs
                    if r.text.strip() and r.font.size is not None and r.font.size < Pt(MIN)})
    assert not small, f"{MIN} pt 未満の文字がある：{small}"
    prs.core_properties.title = "金型内センサデータを用いたデータ駆動型樹脂粘度モデルの開発Ⅲ（簡略版2・A0 1 枚）"
    return prs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=B_DIR, help="pptx を出すフォルダ（既定：ポスター/報告書Bから作成/）")
    ap.add_argument("--fig-dir", default=FIG_DIR, help="make_poster_one_figs.py の図のフォルダ（既定：ポスター/報告書Bから作成/図_簡略版_A0一枚/）")
    a = ap.parse_args()
    out = os.path.join(a.out, OUT_NAME)
    if os.path.exists(out):
        sys.exit(f"すでにある：{out}（上書きしない。前のファイルを _to_delete/ へ移すか、--out で別のフォルダを指定）")
    assert os.path.exists(os.path.join(a.fig_dir, "樹脂粘度とせん断速度の関係.png")), f"図がない：{a.fig_dir}（先に make_poster_one_figs.py を実行）"
    prs = build(a.fig_dir)
    os.makedirs(a.out, exist_ok=True)
    prs.save(out)
    print("保存:", out)


if __name__ == "__main__":
    main()
