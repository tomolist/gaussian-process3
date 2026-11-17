# -*- coding: utf-8 -*-
"""温度シフトファクターと学習の流れについての、ユーザーの 8 つの疑問への回答を、A4 の資料（PDF と tex）にする（2026-11-17 のユーザーの依頼）。

- 疑問ごとに「短い答え → 説明 → 関係する質問対策（ポスター4_質問対策.pdf の番号）」の順に書く。最初に結論の一覧と、モデルの形の図を置く。
- 数値は、台本・質問対策と同じく build_poster4_talk.load() で、3 材料の最新の学習結果の CSV から読む（文の前提の確かめも、そこで行う）。
  質問対策の番号は build_poster4_talk.QA の質問の文から探す（質問対策を直しても、番号がずれない）。
- コードの事実（ガウス過程の入力の列、学習用のノートの節）は、mat_common.layout と学習用のノートの中身で確かめる。
- 樹脂温度最高値 − 加熱シリンダ温度の範囲と、せん断速度との相関（疑問 7）は、学習データ（5data）の条件ごとの平均から計算する。
- 同じ名前のファイルがあれば止める（上書きしない）。LuaLaTeX は _tmp/poster4_shiftqa/<日時>/ で動かし、できた tex と PDF をコピーする。

使い方（リポジトリ直下で）: uv run python 04_補助スクリプト/build_poster4_shift_qa.py [--out <フォルダ>]
  （既定：build_poster4_talk.OUT_DEFAULT〔05_…/07_学習結果報告書作成_261015/ポスター/完成版の添削と複数デザイン/ポスター4_発表報告/〕）
"""
import argparse
import datetime
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import build_poster4_talk as bt  # noqa: E402
import report_common as rc  # noqa: E402

STEM = "ポスター4_温度シフトファクターと学習の流れの疑問への回答"
NB_LEARN = "01_機械学習コードフォルダ/材料別_温度シフトファクター検討/02_材料別_温度シフト機械学習コード/01_勾配法（L-BFGS-B）/材料別_温度シフトGPR_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb"
NB_HEADINGS = ["## 1. 設定", "## 2. ファイルを選んで、データを読み込む", "### 5.1 プロファイル尤度", "### 5.2 最終モデルの学習",
               "## 6. 温度のシフトファクターと、マスターカーブ", "## 8. 3data による予測検証", "## 12. 学習済みモデルの保存"]
NB_SETTINGS = ["SHIFT_COL", "SHIFT_STRUCT", "SHIFT_FORM"]
MAT_JA = {"PP": "バージンPP", "MIX": "リサイクルPP", "GPPS": "バージンGPPS"}
COL_JA = {"ln_gd": r"せん断速度 $\dot{\gamma}$", "ln_P33": r"樹脂圧力差 $\Delta P_2$", "ln_Tff": r"樹脂温度最高値 $T_\mathrm{ff}$",
          "Tset": r"加熱シリンダ温度 $T_\mathrm{set}$", "Tff": r"樹脂温度最高値 $T_\mathrm{ff}$"}
# 質問対策の質問（質問の文の一部）。番号は build_poster4_talk.QA から探す
QS = {"hyp": "ハイパーパラメータは、どう決めた", "grid": "なぜ格子で", "form": "機械学習を行う前に定義", "need": "温度シフトファクターなしで学習した場合",
      "overfit": "過学習", "flow": "学習が終わるまで", "when": "学習のどの段階", "predcalc": "どう計算して予測", "tset": "ガウス過程の入力に入っている",
      "tff": "なぜ樹脂温度ではなく", "inp": "どの値を使ったのか", "imp": "どの入力が重要", "ho": "本当に効いているのか", "what": "温度シフトファクターとは",
      "alone": "平均関数とその予測幅", "pred": "樹脂圧力差と樹脂温度はどう与える", "e3": "係数の値は？"}
# 疑問ごとの関係する質問（キー, 扱い）と、質問対策に正面からの答えがあるか
REL = {1: ([("hyp", ""), ("grid", "")], "一部"), 2: ([("form", ""), ("need", ""), ("overfit", "")], "一部"),
       3: ([("flow", ""), ("when", ""), ("form", ""), ("predcalc", "")], "あり"), 4: ([("tset", ""), ("tff", ""), ("inp", "")], "一部"),
       5: ([("imp", ""), ("ho", ""), ("tset", "")], "一部"), 6: ([("what", ""), ("when", ""), ("tset", "")], "一部"),
       7: ([("tff", ""), ("pred", "")], "一部"), 8: ([("predcalc", ""), ("alone", ""), ("when", "")], "あり")}


def check(cond, msg):
    if not cond:
        sys.exit(f"文の前提が結果と合わない：{msg}（スクリプトの文を見直す）")


def qa_numbers(N):
    """質問対策の質問の番号と文（<<…>> は値に置き換え、印は除く）"""
    out = {}
    for key, part in QS.items():
        hit = [(i, q) for i, (_, _, q, _, _) in enumerate(bt.QA, 1) if part in q]
        check(len(hit) == 1, f"質問対策の質問「{part}」がちょうど 1 つ")
        out[key] = (hit[0][0], bt.sub(hit[0][1], N))
    return out


def dt_stats(m):
    """学習データ（5data）の条件ごとの平均で、樹脂温度最高値 − 加熱シリンダ温度の範囲 [℃]、条件の数、厚みごとの ln γ̇ との相関 [(厚み, 相関)]。
    全厚みをまとめた相関は、厚みの違いで薄まるので使わない（PP 0.59・リサイクルPP 0.71。_tmp/talk_pp4_1117/dt_check.py）"""
    tr = bt.rd(bt.TRAIN_CSV[m])
    tr = tr[(tr[bt.C_ETA] > 0) & (tr[bt.C_GD] > 0) & (tr[bt.C_P33] > 0)].copy()
    tcol = [c for c in tr.columns if "加熱シリンダ温度" in c][0]
    vcol = [c for c in tr.columns if c.startswith("射出速度")][0]
    tr["dT"] = tr[bt.C_TFF] - tr[tcol]
    tr["lg"] = np.log(tr[bt.C_GD])
    g = tr.groupby([tcol, "キャビ厚", vcol])[["dT", "lg"]].mean().reset_index()
    corr = [(float(h), float(np.corrcoef(s["dT"], s["lg"])[0, 1])) for h, s in g.groupby("キャビ厚")]
    return float(g["dT"].min()), float(g["dT"].max()), len(g), corr


def code_facts():
    """コードの事実を確かめる：ガウス過程の入力の列（mat_common.layout）と、学習用のノートの節・設定の名前"""
    os.environ.setdefault("NB_REPO", rc.W)
    import mat_common as mc
    lay = {k: mc.layout(k) for k in ("A_Tset", "B_Tset", "A_Tff", "C")}
    check(lay["A_Tset"] == (["ln_gd", "ln_P33", "ln_Tff"], "Tset") and lay["B_Tset"] == lay["A_Tset"], "加熱シリンダ温度でシフトするモデルの入力")
    check(lay["A_Tff"] == (["ln_gd", "ln_P33", "Tset"], "Tff"), "樹脂温度最高値でシフトするモデルの入力")
    check(lay["C"] == (["ln_gd", "ln_P33", "ln_Tff", "Tset"], None), "C の入力")
    nb = open(os.path.join(rc.W, NB_LEARN), encoding="utf-8").read()
    for h in NB_HEADINGS + NB_SETTINGS:
        check(h in nb, f"学習用のノートに「{h}」がある")
    return lay


def build(N, raw, lay, Q):
    P, X, G = raw["PP"], raw["MIX"], raw["GPPS"]
    V = {}                                            # 文に入れる値（@@名前@@）
    for i, (rel, cover) in REL.items():
        V[f"REL{i}"] = "、".join(f"Q{Q[k][0]}「{Q[k][1]}」" for k, _ in rel) + f"（質問対策での答え：{'あり' if cover == 'あり' else '一部だけ'}）"
        V[f"RELS{i}"] = "・".join(f"Q{Q[k][0]}" for k, _ in rel)
        V[f"COV{i}"] = cover
    lay_rows = []
    for k, name in (("A_Tset", r"\textbf{本番}（加熱シリンダ温度でシフト。PP・リサイクルPP は縦だけ、GPPS は縦と横）"),
                    ("A_Tff", r"比べた案（樹脂温度最高値でシフト）"), ("C", r"C（シフトなし。4 つを並べた通常のガウス過程回帰）")):
        cols, tcol = lay[k]
        lay_rows.append(f"{name} & {COL_JA[tcol] if tcol else 'なし'} & " + "・".join(COL_JA[c] for c in cols) + r" \\")
    V["LAYROWS"] = "\n".join(lay_rows)
    t_rows = []
    for m in bt.MATS:
        r = raw[m]
        t_rows.append(f"{MAT_JA[m]} & {r['VR']:.2f} 倍（{r['VR_T']}） & {bt.sgn(r['SLOPE'][2], 3)} \\\\")
    V["TROWS"] = "\n".join(t_rows)
    dts = {m: dt_stats(m) for m in bt.MATS}
    pm = [c for m in ("PP", "MIX") for _, c in dts[m][3]]
    gc = dts["GPPS"][3]
    check(min(pm) > 0.8, "PP・リサイクルPP は、同じ厚みの中で、樹脂温度最高値 − 加熱シリンダ温度がせん断速度と強く連動する（相関 0.8 超）")
    check(gc[0][1] > 0.6 and min(c for _, c in gc) < 0.3, "GPPS は、薄い厚みだけで連動し、ほかの厚みでは弱いか逆（文の前提）")
    check(all(s[0] < 0 < s[1] for s in dts.values()), "樹脂温度最高値は、加熱シリンダ温度より低い条件も高い条件もある")
    V["DTROWS"] = "、".join(f"{MAT_JA[m]} {bt.sgn(s[0], 0)}〜$+{s[1]:.0f}$ ℃" for m, s in dts.items())
    V["DTC_PM"] = f"{min(pm):.2f}〜{max(pm):.2f}"
    V["DTC_G"] = "、".join(f"{h:g} mm {bt.sgn(c, 2)}" for h, c in gc)
    check(not P["SUP"] and abs(P["VR"] - 1) < 0.02, "PP（直した後）は係数がほぼ 0（倍率 1.02 未満）")
    V.update({"VRPCT": f"{(P['VR'] - 1) * 100:.0f}", "EV_PP": N["EV_PP"], "GAP": N["GAP_PP"], "GAPSH": N["GAPSH_PP"], "SDT": N["SDT_RANGE"],
              "CA_PP": N["CA_PP"], "CA_MIX": N["CA_MIX"], "CB_GPPS": N["CB_GPPS"], "TFFA_PP": N["TFFA_PP"], "TFFA_MIX": N["TFFA_MIX"],
              "TFFA_GPPS": N["TFFA_GPPS"], "DLML_PP": N["DLML_PP"], "DLMLPRE_PP": N["DLMLPRE_PP"], "DLML_MIX": N["DLML_MIX"], "DLML_GPPS": N["DLML_GPPS"],
              "TREF_PP": N["TREF_PP"], "TREF_GPPS": N["TREF_GPPS"], "DATE": rc.today_ja(), "QA": bt.STEM_QA.replace("_", r"\_")})
    check(all(raw[m]["TFFA"] < 1.92 for m in bt.MATS), "樹脂温度最高値でのシフトは、3 材料とも縦だけの基準 1.92 に届かない")
    tex = TEMPLATE
    for k, v in V.items():
        tex = tex.replace(f"@@{k}@@", str(v))
    check("@@" not in tex, "資料の @@…@@ をすべて置き換えた")
    return tex, dts


TEMPLATE = r"""%% 生成：04_補助スクリプト/build_poster4_shift_qa.py（直接は直さない。直すときはスクリプトを直して作り直す）
\documentclass[paper=a4paper,fontsize=11pt,line_length=44zw,number_of_lines=38]{jlreq}
\usepackage[haranoaji]{luatexja-preset}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{array}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{needspace}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,calc}
\usepackage[hidelinks]{hyperref}
\newcommand{\kak}[1]{\textcolor{orange!75!black}{（要確認：#1）}}
\newcommand{\Qd}[2]{\par\Needspace{8\baselineskip}\vspace{1.4ex}\noindent{\large\bfseries 疑問 #1}\par\vspace{0.3ex}\noindent\textcolor{black!70}{#2}\par\nopagebreak[4]\vspace{0.6ex}}
\newcommand{\Short}[1]{\par\noindent\colorbox{blue!6}{\parbox{\dimexpr\linewidth-2\fboxsep\relax}{\textcolor{blue!60!black}{\bfseries 短い答え}\quad #1}}\par\vspace{0.4ex}}
\newcommand{\Rel}[1]{\par\noindent{\small\textcolor{black!60}{\bfseries 関係する質問対策}\quad #1}\par}
\setlist[itemize]{leftmargin=1.5\zw,itemsep=0.2ex,topsep=0.4ex}
\begin{document}
\begin{center}
{\Large 温度シフトファクターと学習の流れ ― 8 つの疑問への回答}\\[0.6ex]
{\small @@DATE@@　関係する資料：「@@QA@@.pdf」（同じフォルダ。質問の番号は、その資料の番号）}
\end{center}

{\small
\begin{itemize}
  \item 疑問ごとに、「短い答え」→「説明」→「関係する質問対策」の順に書いた。最初に、結論の一覧と、モデルの形の図を置いた。
  \item 言葉：加熱シリンダ温度 $T_\mathrm{set}$（設定値）、樹脂温度最高値 $T_\mathrm{ff}$（計測値。データの列名は「流動中樹脂温度」）、樹脂圧力差 $\Delta P_2$（データの列名は「流動中樹脂圧力P3\_3」）。
  \item 数値は、ポスター4\_添削版と同じ学習結果（バージンPP は、樹脂圧力差を直して学習し直した 4回目）から読んだ。$^{\dagger}$ はバージンPP の直す前の樹脂圧力差での結果（その検討は、直した値ではまだしていない）。
\end{itemize}
}

\section*{結論の一覧}
{\small
\begin{tabular}{>{\raggedright\arraybackslash}p{12\zw}>{\raggedright\arraybackslash}p{21\zw}>{\raggedright\arraybackslash}p{7\zw}}
\toprule
疑問 & 一言で & 質問対策 \\
\midrule
1. シフトの検証に、なぜガウス過程回帰が要るか & 周辺尤度は、ずらした後の曲線のモデル（ガウス過程）があって初めて計算できる & @@RELS1@@（@@COV1@@） \\
2. 検討用と学習用で二重学習にならないか & ならない。検討用で決めたのは式の形だけ。学習用は同じデータで学習し直す & @@RELS2@@（@@COV2@@） \\
3. どこでシフトを定義しているか & 形は学習の前、係数は学習の中、予測でも使う & @@RELS3@@（@@COV3@@） \\
4. 2 つの温度は学習に使われるか & どちらも使われる。シフトに使った温度はシフトの式だけ、もう一方はガウス過程の入力 & @@RELS4@@（@@COV4@@） \\
5. 温度の影響はシフトだけか & いいえ。樹脂温度最高値の効果もガウス過程で学習され、予測に入る & @@RELS5@@（@@COV5@@） \\
6. 温度はシフト、ほかは機械学習か & 一部だけ正しい。シフトは加熱シリンダ温度だけ。樹脂温度最高値はガウス過程が学習 & @@RELS6@@（@@COV6@@） \\
7. $T_\mathrm{set}$ を基準、$T_\mathrm{ff}$ を変化温度にしたシフト & 作れるが、比べ済みの「$T_\mathrm{ff}$ でシフト」とほぼ同じ。それは支持されなかった & @@RELS7@@（@@COV7@@） \\
8. 予測と予測精度検証でも働くか & 働く。予測は、いつもシフトを足し戻す & @@RELS8@@（@@COV8@@） \\
\bottomrule
\end{tabular}

「あり」は質問対策に正面からの答えがある、「一部」は関係する答えはあるが、この疑問そのものへの答えはない。
}

\section*{まず：このモデルの形}
本番のモデルは、次の式です（$f$ がガウス過程、$\varepsilon$ がノイズ）。PP・リサイクルPP は縦だけ（$a_h = 1$）、GPPS は縦と横です。
\[
\ln\eta \;=\; \underbrace{\ln a_v(T_\mathrm{set})}_{\mbox{\scriptsize 温度シフトファクター}} \;+\; \underbrace{f\bigl(\ln\dot{\gamma} + \ln a_h(T_\mathrm{set}),\ \ln\Delta P_2,\ \ln T_\mathrm{ff}\bigr)}_{\mbox{\scriptsize ガウス過程（形を決めずに学習）}} \;+\; \varepsilon,
\qquad \ln a(T) = \frac{E}{R}\Bigl(\frac{1}{T} - \frac{1}{T_\mathrm{ref}}\Bigr)
\]

\begin{center}
\begin{tikzpicture}[box/.style={draw, rounded corners, align=center, inner sep=4pt, font=\small}, arr/.style={-{Stealth[length=2.2mm]}, thick}]
\node[box, fill=blue!6, minimum width=36mm] (g) {せん断速度 $\dot{\gamma}$};
\node[box, fill=blue!6, minimum width=36mm, below=2mm of g] (p) {樹脂圧力差 $\Delta P_2$};
\node[box, fill=blue!6, minimum width=36mm, below=2mm of p] (t) {樹脂温度最高値 $T_\mathrm{ff}$};
\node[box, fill=green!8, minimum width=42mm, right=9mm of p, minimum height=22mm] (gp) {\textbf{ガウス過程 $f$}\\[2pt]\footnotesize 形を決めずに学習\\\footnotesize ハイパーパラメータは最尤推定};
\node[box, fill=orange!10, minimum width=36mm, below=10mm of t] (ts) {加熱シリンダ温度 $T_\mathrm{set}$};
\node[box, fill=orange!18, minimum width=42mm, right=9mm of ts] (sf) {\textbf{温度シフトファクター}\\$\ln a_v(T_\mathrm{set})$\\\footnotesize アレニウス式。係数 $E_v$ を学習};
\node[draw, circle, inner sep=1pt, font=\large] (plus) at ($(gp.east)!0.5!(sf.east) + (9mm,0)$) {$+$};
\node[box, fill=black!6, minimum width=26mm, right=6mm of plus] (out) {\textbf{$\ln\eta$}\\\footnotesize 予測値と\\\footnotesize 95\% 予測区間};
\draw[arr] (g.east) -- (gp.west |- g.east);
\draw[arr] (p.east) -- (gp.west |- p.east);
\draw[arr] (t.east) -- (gp.west |- t.east);
\draw[arr] (ts) -- (sf);
\draw[arr] (gp.east) -- (plus);
\draw[arr] (sf.east) -- (plus);
\draw[arr] (plus) -- (out);
\draw[arr, dashed, black!50] (ts.west) -- ++(-5mm,0) |- (g.west);
\node[font=\scriptsize, black!60, anchor=west] at ($(t.south west)!0.5!(ts.north west) + (-3.5mm,0)$) {GPPS だけ：横のずれ $\ln a_h$};
\end{tikzpicture}
\end{center}
{\small
\begin{itemize}
  \item \textbf{学習の前に決める}：式の形（アレニウス型、ずらす向き〔縦だけ／縦と横〕、ずらす温度〔加熱シリンダ温度〕、基準温度 $T_\mathrm{ref}$〔PP・リサイクルPP @@TREF_PP@@ ℃、GPPS @@TREF_GPPS@@ ℃〕）。
  \item \textbf{学習で決める}：シフトの係数（縦の $E_v$、GPPS は横の $E_h$ も）と、ガウス過程のハイパーパラメータ（長さスケール・振幅・ノイズなど）。どちらも、学習データ（5data）の尤度が最大になるように決める。
\end{itemize}
}

%% ------------------------------------------------------------------
\Qd{1}{なぜシフトファクターの検証にガウス過程回帰が必要なのか？ それぞれのシフトファクターの式の定義と、その周辺尤度をデータから決めるだけではないのか？}
\Short{周辺尤度は、ガウス過程があって初めて計算できる量です。シフトファクターの式だけでは粘度を予測できないので、「ずらした後のデータを、どれだけよく説明できるか」を測るには、ずらした後の曲線のモデル（ガウス過程）が要ります。}
\begin{itemize}
  \item シフトファクターが表すのは、「温度ごとの曲線を、どれだけずらすか」だけ。粘度の値そのものは、せん断速度・樹脂圧力差・樹脂温度最高値で決まる曲線のほうにある。
  \item 周辺尤度 $p(\mathbf{y}\mid X) = \int p(\mathbf{y}\mid f)\,p(f)\,df$ は、ガウス過程の関数 $f$ について積分した（周辺化した）尤度。「周辺」は、この積分のこと。
  \item そのため、「式ごとに周辺尤度を求める」ことは、「式と係数ごとに、ずらしたデータでガウス過程を学習する」ことと同じになる（プロファイル尤度。係数の格子の点ごとに 1 回学習する）。
  \item ずらした後の曲線を Cross 式などの決まった形にしても尤度は計算できるが、結果がその式の形に左右される。最終のモデルと同じガウス過程で比べると、「このモデルにとって、シフトが役に立つか」を、そのまま判定できる。
  \item 従来の時間–温度換算（曲線を重ねて、ずらす量を読む方法）は、粘度がせん断速度と温度だけで決まるときの方法。このデータは、樹脂圧力差・樹脂温度最高値でも変わるので、曲線を単純には重ねられない。
\end{itemize}
\Rel{@@REL1@@}

%% ------------------------------------------------------------------
\Qd{2}{温度シフトファクターの検証コードでも機械学習を行い、その結果を用いて学習用コードでどのシフトファクターを使うかを決めたら、二重学習になるのではないか？}
\Short{なりません。検証（検討用）のコードで決めたのは「式の形」だけで、学習用のコードは、同じ学習データで、係数とガウス過程を最初から学習し直します。検証データ（3data）は、形を選ぶのに使っていません。}
\begin{itemize}
  \item 2 つのノートは、同じ学習データ（5data）で、同じ計算（プロファイル尤度）をする。学習用のノートは、検討用のノートの係数を引き継がず、推定し直す。結果は一致した（係数 0 との尤度の差：PP @@CA_PP@@、リサイクルPP @@CA_MIX@@、GPPS @@CB_GPPS@@。本番の学習と 0.01 以内で同じことを確かめた）。
  \item 学習データの周辺尤度で式の形を選ぶのは、カーネルの種類を選ぶのと同じく、学習の一部（モデルの選択）。周辺尤度には複雑さの罰（$-\tfrac12\log|K|$）が入っているので、尤度で選ぶこと自体では過学習しにくい。
  \item 検証データ（3data）は、形を選ぶのには使っていない。選んだモデルの誤差が、加熱シリンダ温度を入力に並べたモデル（C）の 1.1 倍以内かの確かめにだけ使った。そのため、予測精度の評価は、学習から独立している。
  \item \textbf{注意}：8 つの組み合わせ（型 2 × 向き 2 × 温度 2）から最も良いものを選んでいるので、選んだモデルの尤度の差は、少し良く見える（選んだことによる効果）。それを、学習に使っていないデータで確かめるのが、3data の予測精度検証と、温度ホールドアウト検証。
\end{itemize}
\Rel{@@REL2@@}

%% ------------------------------------------------------------------
\Qd{3}{機械学習ノート全体の流れ（事前分布や推論モデルの定義、データの定義などの事前準備から、機械学習、予測精度検証まで）のどのタイミングで、温度シフトファクターを定義しているのか？ 事前に定義してから学習しているのか、学習後の予測の計算に用いるモデルにしているのか？}
\Short{両方です。式の形は学習の前（設定）に決め、係数は学習の中（プロファイル尤度）で決め、学習後の予測でも同じ式を使います。シフトは、モデルの一部（ガウス過程の平均の一部）として組み込んであります。}
{\small
\begin{center}
\begin{tabular}{>{\raggedright\arraybackslash}p{11\zw}>{\raggedright\arraybackslash}p{29\zw}}
\toprule
学習用のノートの節 & 温度シフトファクターの役目 \\
\midrule
\textbf{1. 設定} & \textbf{式の形を決める}：ずらす温度 \texttt{SHIFT\_COL}、向き \texttt{SHIFT\_STRUCT}（縦だけ A／縦と横 B）、型 \texttt{SHIFT\_FORM}（アレニウス型／WLF 型） \\
2〜4b. データの読み込みと確認 & 使わない（外れ値・樹脂圧力差の性質・厚み） \\
\textbf{5.1 プロファイル尤度} & \textbf{係数を決める}：係数の格子の点ごとに、データをずらし（$\ln\eta$ から $\ln a_v$ を引き、GPPS は $\ln\dot{\gamma}$ に $\ln a_h$ を足す）、ガウス過程を学習して尤度を記録。尤度が最大の係数を選ぶ \\
5.1b・\textbf{5.2 最終モデル} & 選んだ係数で、初期値を変えて念入りに学習し直す（係数 0 とも比べる） \\
6. シフトファクターとマスターカーブ & 係数と、ずらした後のデータの図 \\
\textbf{8. 3data による予測検証} & \textbf{予測でも使う}：ガウス過程の出力に $\ln a_v(T_\mathrm{set})$ を足し戻して予測し、誤差を出す \\
\textbf{12. 保存} & 係数も、学習済みモデル（.joblib）に保存する \\
\bottomrule
\end{tabular}
\end{center}
}
\begin{itemize}
  \item ガウス過程の言葉で言うと、$\ln\eta$ の平均を「定数 ＋ $\ln a_v(T_\mathrm{set})$」とし、そこからのずれを、平均 0 のガウス過程で表す形。GPPS は、入力のせん断速度もずらす。
  \item 事前分布：ずらした後のデータに、平均 0 のガウス過程（Matern 5/2 ＋ 線形 ＋ ノイズのカーネル）を置く。ハイパーパラメータと係数は最尤推定で、それらに事前分布は置いていない（探索の範囲だけ。係数は、縦は $-20$〜60 kJ/mol の格子）。
  \item 推論：係数は外側の格子で、ハイパーパラメータは内側の勾配法（L-BFGS-B）で決める。
\end{itemize}
\Rel{@@REL3@@}

%% ------------------------------------------------------------------
\Qd{4}{説明変数にある 2 つの樹脂温度値において、シフトファクターに用いられた方は学習にも使われているのか？ シフトファクターに用いられなかった方は学習に使われているのか？}
\Short{どちらも学習に使われていますが、入り方が違います。シフトに使った温度は、ガウス過程の入力から外れ、シフトの式を通してだけ学習に入ります（係数をデータから決める）。使わなかった温度は、ガウス過程の入力として、形を決めずに学習されます。}
{\small
\begin{center}
\begin{tabular}{>{\raggedright\arraybackslash}p{15\zw}>{\raggedright\arraybackslash}p{9\zw}>{\raggedright\arraybackslash}p{16\zw}}
\toprule
モデル & シフトに使う温度 & ガウス過程の入力 \\
\midrule
@@LAYROWS@@
\bottomrule
\end{tabular}
\end{center}
}
\begin{itemize}
  \item 本番では、加熱シリンダ温度がシフトに、樹脂温度最高値がガウス過程の入力に使われる。樹脂温度最高値でシフトする案では、役割が入れ替わる（加熱シリンダ温度がガウス過程の入力になる）。
  \item シフトに使った温度の効果は、式の形（アレニウス式：絶対温度の逆数に比例するずれ）に限られ、データから決めるのは係数 1 つ（GPPS は 2 つ）。
\end{itemize}
\Rel{@@REL4@@}

%% ------------------------------------------------------------------
\Qd{5}{目的変数である樹脂粘度に対する温度の影響はシフトファクターのみに限定していて、学習によって得られた樹脂粘度と樹脂温度の関係性や影響度は無視されるのか？}
\Short{無視されません。温度の効果は 2 つの道で入り、どちらも予測に使われます。形が決まっているのは、加熱シリンダ温度の効果（シフトファクター）だけです。}
{\small
\begin{center}
\begin{tabular}{lcc}
\toprule
材料 & 加熱シリンダ温度 → シフト & 樹脂温度最高値 → ガウス過程 \\
 & （低い温度 ÷ 高い温度の粘度） & （1 標準偏差あたりの $\ln\eta$ の変化） \\
\midrule
@@TROWS@@
\bottomrule
\end{tabular}
\end{center}
}
\begin{itemize}
  \item 右の列は、樹脂温度最高値を 1 標準偏差（約 @@SDT@@ ℃）動かしたときの $\ln\eta$ の変化（学習データの点での平均）。ガウス過程が学習した関係で、予測にもそのまま入る。
  \item \textbf{PP（樹脂圧力差を直した後）では、温度の効果の大部分を、樹脂温度最高値の側が受け持っている}。学習結果のグラフで、190 ℃ の線は 230 ℃ の線の @@GAP@@ 倍だが、そのうちシフトの分は @@GAPSH@@ 倍だけ。
  \item PP・リサイクルPP は縦だけなので、加熱シリンダ温度の効果は、せん断速度によらず同じ倍率。せん断速度などによる温度の効果の違いは、樹脂温度最高値の入力を通して表される。
\end{itemize}
\Rel{@@REL5@@}

%% ------------------------------------------------------------------
\Qd{6}{樹脂粘度に対する温度の影響度をシフトファクターにしていて、樹脂温度以外の説明変数と樹脂粘度の関係性を、機械学習によって定義しているのか？}
\Short{一部だけ正しいです。シフトファクターが受け持つのは加熱シリンダ温度だけで、ガウス過程は、せん断速度・樹脂圧力差に加えて、樹脂温度最高値との関係も学習します。シフトファクターの係数も、データから学習で決めます。}
\begin{itemize}
  \item 式で書くと、$\ln\eta = \ln a_v(T_\mathrm{set}) + f(\ln\dot{\gamma} + \ln a_h(T_\mathrm{set}),\ \ln\Delta P_2,\ \ln T_\mathrm{ff}) + \varepsilon$。温度は、$T_\mathrm{set}$ がシフトファクターに、$T_\mathrm{ff}$ がガウス過程 $f$ に入る（最初の図）。
  \item 「温度＝シフトファクター、ほかの変数＝機械学習」と分けたのではなく、「設定温度＝決まった形の式（係数は学習）、計測した量（せん断速度・樹脂圧力差・樹脂温度最高値）＝形を決めない学習」と分けた形。
\end{itemize}
\Rel{@@REL6@@}

%% ------------------------------------------------------------------
\Qd{7}{設定値である加熱シリンダ温度を基準温度として、計測値である流動中樹脂温度を変化温度とするとき、シフトファクターはできるのか？}
\Short{作れます。ただし、式を分けると、すでに比べた「樹脂温度最高値でシフトするモデル」とほぼ同じになります。そのモデルは、3 材料とも尤度で支持されませんでした。}
\[
\ln a \;=\; \frac{E}{R}\Bigl(\frac{1}{T_\mathrm{ff}} - \frac{1}{T_\mathrm{set}}\Bigr)
\;=\; \underbrace{\frac{E}{R}\Bigl(\frac{1}{T_\mathrm{ff}} - \frac{1}{T_\mathrm{ref}}\Bigr)}_{\mbox{\scriptsize 樹脂温度最高値でのシフト}}
\;-\; \underbrace{\frac{E}{R}\Bigl(\frac{1}{T_\mathrm{set}} - \frac{1}{T_\mathrm{ref}}\Bigr)}_{\mbox{\scriptsize 加熱シリンダ温度だけの項}}
\]
\begin{itemize}
  \item 後ろの項は、加熱シリンダ温度だけの関数。樹脂温度最高値でシフトするモデルでは、加熱シリンダ温度がガウス過程の入力に入っている（疑問 4 の表）ので、この項はガウス過程が吸収できる。WLF 型も、同じように 2 つの項に分けられる。
  \item そのため、この形は、すでに比べた「樹脂温度最高値でシフトするモデル」とほぼ同じと考えられる\kak{直接は計算していない}。そのモデルの、係数 0 との尤度の差（縦だけ。2 つの型の大きいほう。基準 1.92）は、PP @@TFFA_PP@@、リサイクルPP @@TFFA_MIX@@、GPPS @@TFFA_GPPS@@ で、どれも基準に届かない。
  \item 中身の注意 1：樹脂温度最高値と加熱シリンダ温度の差は、学習データの条件ごとの平均で、@@DTROWS@@。PP・リサイクルPP では、同じ厚みの中で、差はせん断速度とともに大きくなる（厚みごとの、せん断速度の対数との相関 @@DTC_PM@@）。主に、せん断発熱と金型による冷却の差と考えられ\kak{考察}、このシフトは、せん断速度の効果を二重に拾うおそれがある。GPPS は厚みで違う（相関 @@DTC_G@@）。
  \item 中身の注意 2：樹脂温度最高値は計測値で、成形の前には分からない。加熱シリンダ温度は、成形の前に決まる設定値。
\end{itemize}
\Rel{@@REL7@@}

%% ------------------------------------------------------------------
\Qd{8}{温度シフトファクターは、完成したモデルを実際に使う（樹脂粘度の予測を行う）ときにも働くのか？ 学習コード終盤の樹脂粘度の予測精度検証の時点で、温度シフトファクターを使っているのか？}
\Short{どちらでも働きます。予測の計算は、いつもガウス過程の出力に $\ln a_v(T_\mathrm{set})$ を足し戻します（GPPS は、せん断速度の入力もずらします）。3data による予測精度検証も、保存した学習済みモデルでの予測も、同じ計算です。}
\begin{itemize}
  \item 予測の式：$\ln\hat{\eta} = \mu_f + \bar{y} + \ln a_v(T_\mathrm{set})$（$\mu_f$ はガウス過程の予測の平均、$\bar{y}$ は学習データの $\ln\eta$ の平均）。95\% 予測区間は $\ln\hat{\eta} \pm 1.96\,\sigma$ を指数で粘度に戻す。縦のずれは平均に足すだけなので、区間の幅（対数）はそのまま上下に動く。
  \item 学習済みモデル（.joblib）に、係数・標準化の定数・ガウス過程（学習データとハイパーパラメータ）が保存されているので、モデルだけで温度シフトを考えた予測ができる。ただし、樹脂圧力差と樹脂温度最高値は、計測した値を与える必要がある。
  \item PP（樹脂圧力差を直した後）は、係数がほぼ 0（@@EV_PP@@ kJ/mol。係数 0 との尤度の差 @@DLML_PP@@ で、支持されない。直す前は @@DLMLPRE_PP@@）なので、シフトの効きは約 @@VRPCT@@\%。リサイクルPP（@@DLML_MIX@@）と GPPS（@@DLML_GPPS@@）は支持されている。
\end{itemize}
\Rel{@@REL8@@}

\section*{数値とコードの出どころ}
{\small
\begin{itemize}
  \item モデルの形・ガウス過程の入力の列：04\_補助スクリプト/mat\_common.py の \texttt{layout}・\texttt{design}・\texttt{fit\_gp}・\texttt{predict}・\texttt{shift\_ln}（このスクリプトで、入力の列が表のとおりであることを確かめた）。
  \item 学習用のノートの節と設定の名前：01\_機械学習コードフォルダ/材料別\_温度シフトファクター検討/02\_材料別\_温度シフト機械学習コード/01\_勾配法（L-BFGS-B）/材料別\_温度シフトGPR\_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb（節の名前があることを確かめた）。
  \item 係数・倍率・尤度の差・傾き・グラフの線のずれ：「@@QA@@.pdf」と同じ（数値の出どころは、その資料の最後の節）。
  \item 樹脂温度最高値 − 加熱シリンダ温度：各材料の学習データ（5data）の、成形条件ごとの平均。
\end{itemize}
}
\end{document}
"""


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=bt.OUT_DEFAULT, help="tex と PDF を置くフォルダ（リポジトリ直下から）")
    a = ap.parse_args()
    out = os.path.join(rc.W, a.out)
    for e in ("tex", "pdf"):
        p = os.path.join(out, f"{STEM}.{e}")
        if os.path.exists(p):
            sys.exit(f"すでにある：{p}（上書きしない。前のファイルを _to_delete/ へ移すか、--out で別のフォルダを指定）")
    lay = code_facts()
    N, raw = bt.load()
    Q = qa_numbers(N)
    tex, dts = build(N, raw, lay, Q)
    work = os.path.join(rc.W, "_tmp", "poster4_shiftqa", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(work, exist_ok=False)
    tex_p, pdf_p, missing, over = bt.compile_tex(tex, STEM, work)
    os.makedirs(out, exist_ok=True)
    for src in (tex_p, pdf_p):
        dst = os.path.join(out, os.path.basename(src))
        if os.path.exists(dst):
            sys.exit(f"すでにある：{dst}")
        shutil.copy2(src, dst)
    print(f"{os.path.basename(pdf_p)}：フォントにない文字 {missing}、はみ出し（Overfull）{over}")
    print("質問対策の番号：" + "、".join(f"{k} Q{v[0]}" for k, v in Q.items()))
    print("樹脂温度最高値 − 加熱シリンダ温度（条件平均）：" + "、".join(f"{m} {s[0]:.1f}〜{s[1]:.1f} ℃・{s[2]} 条件・厚みごとの相関 "
                                                               + "/".join(f"{h:g}:{c:.2f}" for h, c in s[3]) for m, s in dts.items()))
    print("作業フォルダ:", os.path.relpath(work, rc.W))
    print("保存:", a.out)


if __name__ == "__main__":
    main()
