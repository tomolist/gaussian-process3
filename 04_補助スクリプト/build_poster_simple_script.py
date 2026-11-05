# -*- coding: utf-8 -*-
"""簡略版のポスター（報告書Bから作成_簡略版.pptx）の台本（A4 の PDF と tex）を作る：ポスター/報告書Bから作成_簡略版_台本.pdf・.tex

2026-11-04 のユーザーの依頼：ポスターの文は簡潔にし、それ以外は口頭で説明する。各節の補足説明（台本）を、ポスターと同じフォルダに作る（PDF ＋ tex）。
節ごとに「話すこと」（そのまま読める文。【 】は図を指す合図）と「質問されたときの補足」（ポスターから消した詳しい内容）を書く。
話すことの字数から、話す時間の目安（1 分 300 字）を数えて、最初に時間配分の表を置く。
同じ日の 2 回目の指摘でポスターを A0 × 2 枚に直したのに合わせて、節の題・図を指す合図（3D 図、表と射出率の帯、パリティプロットなど）を直した。
数値は、ポスターと同じ ポスター/図/ポスターの数値.csv から読む。すでにあるファイルは上書きしない（あれば止まる）。

使い方（リポジトリ直下で）: uv run python 04_補助スクリプト/build_poster_simple_script.py [--out <フォルダ>]   （既定：ポスターのフォルダ）
"""
import argparse
import datetime
import math
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402

import build_poster as bp  # noqa: E402
import report_common as rc  # noqa: E402

POSTER = bp.POSTER
STEM = "報告書Bから作成_簡略版_台本"
CHARS_PER_MIN = 300          # 話す速さの目安（1 分あたりの字数）


def r0(x):
    return int(math.floor(x + 0.5))


def load_numbers():
    t = pd.read_csv(os.path.join(bp.FIG, "ポスターの数値.csv"), encoding="utf-8-sig")
    m = {r["材料"]: r for _, r in t.iterrows()}
    return t, m


def spoken_len(tex):
    """話すことの字数（【 】の合図・LaTeX の命令・空白を除く。数式は 1 字ずつ数える）"""
    s = re.sub(r"\\cue\{[^}]*\}", "", tex)
    s = re.sub(r"\\[A-Za-z]+", "", s)
    s = re.sub(r"[{}$_^\\\s]", "", s)
    return len(s)


def sections(t, m):
    P, X, G = m["PP"], m["粉砕材混合PP"], m["GPPS"]
    rng = lambda col, f=lambda v: v: f"{min(r0(f(v)) for v in t[col])}〜{max(r0(f(v)) for v in t[col])}"
    err, cw, cov = rng("誤差 [%] 全体"), rng("Cross-WLF 誤差 [%] 全体"), rng("被覆率", lambda v: v * 100)
    S = []
    # ---------------------------------------------------------------- 1
    S.append(("1. 研究概要", r"""
射出成形の CAE で使われる樹脂粘度モデルは、樹脂のロットの違いや、成形中の熱の履歴による粘度のばらつきを考慮していません。
そこで、金型の中で計測した樹脂の圧力と温度を機械学習に使い、粘度の予測値と、その予測のばらつきを同時に出せるモデルを作っています。
今回は、\cue{青い囲みを指す}3 つの材料でモデルを作ったこと、温度シフトファクターを組み込んだこと、学習に使っていない条件で精度を確かめたこと、の 3 点を報告します。
""", [
        r"前報（見本のポスター「報告書Aから作成2」）との違い：前報は PP だけで学習し、ランダムに選んだ 2 割のデータで検証した（決定係数 0.9666）。ランダムに選ぶと、同じ成形条件の別のショットが学習にも入るので、評価が甘くなる。今回は、射出率の両端の条件をまるごと学習から外して検証した。",
        r"「予測のばらつき」には、データが少ない所でのモデルの不確かさと、同じ条件で成形を繰り返したときのショット間のばらつきの両方が入る（95\% 予測区間）。",
    ]))
    # ---------------------------------------------------------------- 2
    S.append(("2. 計測金型およびセンサ", r"""
\cue{写真を指す}こちらが計測金型です。\cue{右の図の①②を指す}キャビティの中の 2 か所、①と②で圧力を、②で樹脂の温度を測ります。
\cue{下の 3D 図を指す}圧力センサと温度センサは、この図のように別々に付けて測りました。
""", [
        r"樹脂圧力センサ：水晶圧電式樹脂圧力センサ（6157CAUMRKF0.4、日本キスラー合同会社）。樹脂温度センサ：赤外線放射樹脂温度センサ（EPSSZL$\phi$4、双葉電子工業㈱）。",
        r"圧力と温度は、専用の固定側キャビティブロックを交換して、別々に計測した。",
        r"キャビティの厚みは、キャビティブロック（入れ子）を交換して、0.5・1.0・1.5・2.0 mm に変えた。センサ間の距離は $L$ = 17.5 mm。",
    ]))
    # ---------------------------------------------------------------- 3
    S.append(("3. 成形条件と学習・検証データ", r"""
材料は、PP、PP に粉砕材を混ぜたもの、GPPS の 3 つです。\cue{表を指す}赤字のように、GPPS は加熱シリンダ温度と厚みが違います。
\cue{射出率の帯を指す}射出率は 12 水準で、青を学習に、両端のオレンジを答え合わせに使いました。学習の範囲の外をどこまで予測できるかを確かめるためです。
""", [
        r"成形機：全電動式射出成形機 ROBOSHOT $\alpha$-50iAP（ファナック㈱、スクリュ直径 26 mm）。保持圧力は 0 MPa。",
        r"射出率は 1.6〜170 cm$^3$/s の 12 水準（射出率 [cm$^3$/s] ＝ 0.531 × 射出速度 [mm/s]）。3 材料で共通。",
        r"学習データは 1 条件 5 ショット、検証データは 1 条件 3 ショット（外れ値を除いた後は、1 条件 1〜5 ショット）。低射出率側の検証は、成形できる厚肉（1.5・2.0 mm）だけ。薄肉（0.5 mm は 10.6 cm$^3$/s より低い射出率、1.0 mm は 5.3 cm$^3$/s より低い射出率）は成形できない。",
        rf"検証データの数：PP {int(P['検証の行数'])}・粉砕材混合 PP {int(X['検証の行数'])}・GPPS {int(G['検証の行数'])} ショット。",
        r"除いたデータ：PP の 1.5 mm・1.6 cm$^3$/s は、計測ミスで圧力の最大値を検出できなかったので除いた。PP の 190 ℃・0.5 mm・170 cm$^3$/s は、ほかの条件からの予測と大きく違う外れ値として除いた。表の条件の数は、除いた後の数。",
    ]))
    # ---------------------------------------------------------------- 4
    S.append(("4. 実験結果", r"""
\cue{上のグラフを指す}これは、2 つのセンサの圧力の波形です。①と②の圧力の差 $\Delta P_1$ と、樹脂が届くまでの時間差 $\Delta t$ から、\cue{式を指す}下の式で粘度とせん断速度を計算しました。
\cue{下の 3 つのグラフを指す}下は、材料ごとの粘度とせん断速度の関係です。\cue{白丸を指す}高せん断速度側では、キャピラリーレオメータで測った白丸とおおむね一致したので、このデータを学習データとして使いました。
""", [
        r"$\Delta P_1$：センサ②の圧力が立ち上がった時刻の、①と②の圧力差。粘度の計算に使う。$\Delta P_2$：圧力が最大のときの①と②の圧力差。機械学習の入力に使う。",
        r"金型内樹脂最高温度 $T$：樹脂の先端（フローフロント）が②に届いたときに、樹脂温度センサが示す最高値。機械学習の入力と、温度シフトファクターに使う。",
        r"計算式は、ニュートン流体のスリット流れの式（$\eta = \Delta P_1 B H^3 / (12 L Q)$、$\dot{\gamma} = 6 Q / (B H^2)$）。せん断発熱などを含んだ「見かけの粘度」で、補正（Rabinowitsch 補正など）はしていない。",
    ]))
    # ---------------------------------------------------------------- 5
    S.append(("5. 機械学習モデルと温度シフトファクターの検討", r"""
機械学習には、粘度の予測値と予測の幅を同時に出せる、ガウス過程回帰を使いました。
\cue{左の図を指す}温度が変わると、粘度の曲線は横と縦にずれます。Cross-WLF モデルと同じように、このずれを温度シフトファクターとしてモデルに組み込みました。
\cue{真ん中の 3 段を指す}式の形は学習の前に決め、ずれの大きさは、学習のときにガウス過程回帰と同時に決めました。学習後のモデルでは固定です。
\cue{右の流れを指す}予測では、上から順に、温度からずれを計算し、せん断速度を換算してから粘度を予測し、最後に縦のずれを戻します。
""", [
        r"ガウス過程回帰：粘度式の形（Cross 式など）をあらかじめ決めずに、入力と粘度の関係をデータから学習する手法。予測の幅（95\% 予測区間）は、データの多い所では狭く、データのない所では広くなる（6. の学習結果のグラフの帯）。",
        r"入力は、せん断速度 $\dot{\gamma}$、樹脂圧力差 $\Delta P_2$、金型内樹脂最高温度 $T$、加熱シリンダ温度の 4 つ。出力は樹脂粘度 $\eta$。$\eta$・$\dot{\gamma}$・$\Delta P_2$ は対数をとった。$\dot{\gamma}$・$\Delta P_2$・$T$ は、同じ成形条件のショットの平均値を使った（ショットごとの値では、計測値の小さなゆらぎに予測曲線が過敏に反応し、学習範囲の外の精度が下がったため）。",
        r"カーネル関数（データどうしの似ている度合い）：入力ごとに影響の大きさを自動で調整する ARD Matern 5/2 カーネルに、学習範囲の外で予測値が平均値に戻るのを防ぐ直線的な成分（線形カーネル）と、ショット間のばらつきの成分（ホワイトノイズ）を足した。",
        r"式の形：PP・粉砕材混合 PP はアレニウス型 $\ln a_T = (E/R)(1/T - 1/T_\mathrm{ref})$、GPPS は WLF 型（$A_2$ = 51.6 K、ガラス転移温度は 100 ℃ と仮定）。基準温度 $T_\mathrm{ref}$ は、PP・粉砕材混合 PP が 210 ℃、GPPS が 220 ℃。",
        rf"係数の決め方：係数の候補ごとにガウス過程回帰を学習し、学習データへの当てはまり（対数周辺尤度）が最もよい値を採った。値は、横・縦の順に、PP {r0(P['横の係数'])}・{r0(P['縦の係数'])} kJ/mol、粉砕材混合 PP {r0(X['横の係数'])}・{r0(X['縦の係数'])} kJ/mol、GPPS（$A_1$）{r0(G['横の係数'])}・{r0(G['縦の係数'])}。",
        rf"温度シフトの効果の確かさ：係数を 0 にした（ずれなし）場合より、対数周辺尤度が PP {P['LML の差（Tff）']:.1f}・粉砕材混合 PP {X['LML の差（Tff）']:.1f}・GPPS {G['LML の差（Tff）']:.1f} 大きく、偶然の差とはいえない（目安 3.0 以上）。",
        rf"加熱シリンダ温度でずらした場合は、この差が PP {P['LML の差（Tset）']:.1f}・粉砕材混合 PP {X['LML の差（Tset）']:.1f}・GPPS {G['LML の差（Tset）']:.1f} と小さかったので、金型内樹脂最高温度を使った。ただし、金型内樹脂最高温度はせん断発熱で射出率とともに変わるので、ずれにはせん断発熱の影響も含まれている可能性がある。加熱シリンダ温度は、ガウス過程回帰の入力としても使っている。",
    ]))
    # ---------------------------------------------------------------- 6
    pc = lambda r, c: r0(r[c])
    S.append(("6. 学習結果と予測精度検証", rf"""
\cue{{上の 3 つのグラフを指す}}点が学習データ、線が予測値、帯が 95\% 予測区間です。データのない低せん断速度側で帯が広がり、データの量に応じたばらつきを表せています。
\cue{{中の 3 つのグラフを指す}}学習に使っていない両端の射出率を予測すると、どの材料も点が破線の近くに並びました。
\cue{{下の棒グラフを指す}}誤差は約 {err}\% で、Cross-WLF モデルの約 {cw}\% よりずっと小さく、検証データの {cov}\% が予測区間に入りました。
""", [
        r"パリティプロット：横軸が実測の粘度、縦軸が予測の粘度。青は低射出率側（1.6・2.7 cm$^3$/s）、赤は高射出率側（149・170 cm$^3$/s）の検証データ。縦線は 95\% 予測区間。",
        rf"予測誤差：粘度の対数の二乗平均平方根誤差（RMSE）を、粘度の \% に換算した値（$e^{{\mathrm{{RMSE}}}} - 1$）。材料ごと（全体／低射出率側／高射出率側）：PP {pc(P, '誤差 [%] 全体')}／{pc(P, '誤差 [%] 低速側')}／{pc(P, '誤差 [%] 高速側')}\%、粉砕材混合 PP {pc(X, '誤差 [%] 全体')}／{pc(X, '誤差 [%] 低速側')}／{pc(X, '誤差 [%] 高速側')}\%、GPPS {pc(G, '誤差 [%] 全体')}／{pc(G, '誤差 [%] 低速側')}／{pc(G, '誤差 [%] 高速側')}\%。",
        rf"95\% 予測区間に入った割合：PP {P['被覆率'] * 100:.0f}\%、粉砕材混合 PP {X['被覆率'] * 100:.0f}\%、GPPS {G['被覆率'] * 100:.0f}\%（図の表示と同じ丸め方）。",
        r"Cross-WLF モデルの誤差が大きいのは、キャビティ厚みによる見かけの粘度の違いを表せないため（同じせん断速度でも、薄肉ほど見かけの粘度が高い）。",
        rf"温度によるずれ（同じ厚みのデータを 1 本の曲線で表したときの、温度によるずれ）は、温度シフトファクターで換算すると、PP {P['ずれ 換算前 [%]']:.1f} → {P['ずれ 換算後 [%]']:.1f}\%、粉砕材混合 PP {X['ずれ 換算前 [%]']:.1f} → {X['ずれ 換算後 [%]']:.1f}\%、GPPS {G['ずれ 換算前 [%]']:.1f} → {G['ずれ 換算後 [%]']:.1f}\% に減った（換算前の約 4〜5 割）。",
        r"学習結果のグラフの線と帯は、$\Delta P_2$ を学習データの中央値、金型内樹脂最高温度を加熱シリンダ温度ごとの中央値に固定して描いた。そのため、$\Delta P_2$ が中央値と大きく違う薄肉（0.5 mm）の点は、線からずれる。",
    ]))
    # ---------------------------------------------------------------- 7
    S.append(("7. まとめと今後の課題", rf"""
まとめです。金型内の計測データとガウス過程回帰で、3 つの材料について、粘度の予測値とばらつきの範囲を同時に出せるモデルを作りました。学習に使っていない条件でも、誤差は約 {err}\% でした。
今後は、データのない低せん断速度側の予測をよくするため、計測データの追加や、Cross 式との組み合わせを検討します。以上です。
""", [
        r"低せん断速度側（約 150 1/s より小さい範囲）は計測データがなく、予測区間が大きく広がる。粉砕材混合 PP では、低せん断速度側で予測値が下がるなど、物理的に妥当でない曲線の形も見られた。",
        r"対策の案：低せん断速度側の計測データを足す。Cross 式などの粘度式をガウス過程回帰の平均の形として使い、データのない所で式の形に近づける方法を検討する。",
        r"GPPS のガラス転移温度（100 ℃）は仮定の値。",
    ]))
    return S


def build_tex(S):
    rows, total = [], 0
    for title, speech, _ in S:
        n = spoken_len(speech)
        total += n
        rows.append(f"{title} & {n} & 約 {r0(n / CHARS_PER_MIN * 60 / 5) * 5} 秒 \\\\")
    sec_total = total / CHARS_PER_MIN * 60
    body = []
    for title, speech, notes in S:
        paras = "\n\n".join(p.strip() for p in speech.strip().split("\n") if p.strip())
        items = "\n".join(f"  \\item {x}" for x in notes)
        body.append(f"\\section*{{{title}}}\n\\subsection*{{話すこと（約 {r0(spoken_len(speech) / CHARS_PER_MIN * 60 / 5) * 5} 秒）}}\n{paras}\n\n"
                    f"\\subsection*{{質問されたときの補足}}\n\\begin{{itemize}}\n{items}\n\\end{{itemize}}\n")
    tex = r"""%% 生成：04_補助スクリプト/build_poster_simple_script.py（直接は直さない。直すときはスクリプトを直して作り直す）
\documentclass[paper=a4paper,fontsize=12pt,line_length=40zw,number_of_lines=36]{jlreq}
\usepackage[haranoaji]{luatexja-preset}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\newcommand{\cue}[1]{{\small\textcolor{blue!70!black}{【#1】}}}
\setlist[itemize]{leftmargin=1.5\zw,itemsep=0.3ex,topsep=0.5ex}
\begin{document}
\begin{center}
{\Large 簡略版ポスターの台本}\\[0.8ex]
「報告書Bから作成\_簡略版」ポスター（A0 × 2 枚）の発表用（3〜5 分）\\
@@DATE@@
\end{center}

\section*{使い方}
\begin{itemize}
  \item ポスターの文は、発表者がいなくても読めるように書いてある。発表では読み上げず、図を指しながら、下の「話すこと」を話す。\cue{ }の中は、図を指す合図（読まない）。
  \item 「質問されたときの補足」は、ポスターから省いた詳しい内容。質問への答えに使う。
  \item 話す時間は、1 分 @@CPM@@ 字で数えた目安。合計は約 @@MIN@@ 分（@@TOTAL@@ 字）。
\end{itemize}

\begin{center}
\begin{tabular}{lrr}
\toprule
節 & 話す字数 & 時間の目安 \\
\midrule
@@ROWS@@
\midrule
合計 & @@TOTAL@@ & 約 @@MIN@@ 分 \\
\bottomrule
\end{tabular}
\end{center}

@@BODY@@
\end{document}
"""
    tex = (tex.replace("@@DATE@@", rc.today_ja()).replace("@@CPM@@", str(CHARS_PER_MIN)).replace("@@TOTAL@@", str(total))
           .replace("@@MIN@@", f"{sec_total / 60:.1f}").replace("@@ROWS@@", "\n".join(rows)).replace("@@BODY@@", "\n".join(body)))
    return tex, total, sec_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=POSTER, help="tex と PDF を出すフォルダ（既定：ポスターのフォルダ）")
    a = ap.parse_args()
    tex_path, pdf_path = os.path.join(a.out, f"{STEM}.tex"), os.path.join(a.out, f"{STEM}.pdf")
    for p in (tex_path, pdf_path):
        if os.path.exists(p):
            sys.exit(f"すでにある：{p}（上書きしない。前のファイルを _to_delete/ へ移すか、--out で別のフォルダを指定）")
    t, m = load_numbers()
    S = sections(t, m)
    tex, total, sec = build_tex(S)
    os.makedirs(a.out, exist_ok=True)
    with open(tex_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(tex)
    work = os.path.join(rc.W, "_tmp", "poster_script", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))   # LuaLaTeX の作業フォルダ（実行ごとに新しく）
    os.makedirs(work, exist_ok=True)
    rel = os.path.relpath(os.path.abspath(tex_path), rc.W).replace("\\", "/")      # リポジトリ直下から動かす（ポスターのフォルダを作業フォルダにすると、LuaLaTeX が途中で止まったため）
    for i in range(2):
        r = subprocess.run([rc.LUALATEX, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={os.path.relpath(work, rc.W)}", rel],
                           cwd=rc.W, capture_output=True)
        if r.returncode != 0:
            print(r.stdout.decode("utf-8", "replace")[-3000:])
            sys.exit(f"lualatex が失敗しました（{i + 1} 回目）")
    shutil.copy2(os.path.join(work, f"{STEM}.pdf"), pdf_path)
    for title, speech, _ in S:
        print(f"{title}：{spoken_len(speech)} 字")
    print(f"話す字数の合計 {total} 字 → 約 {sec / 60:.1f} 分（1 分 {CHARS_PER_MIN} 字）")
    print("保存:", tex_path, "と", pdf_path)


if __name__ == "__main__":
    main()
