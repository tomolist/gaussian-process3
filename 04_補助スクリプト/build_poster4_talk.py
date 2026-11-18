# -*- coding: utf-8 -*-
"""ポスター4（05_…/07_学習結果報告書作成_261015/ポスター/完成版の添削と複数デザイン/ポスター4.pptx・.pdf。11-17 からは添削版）を使った口頭の報告会のための、
台本と質問対策を、A4 の PDF と tex で作る（2026-11-16 のユーザーの依頼。台本は 3 分 → 11-17 から 4 分 30 秒）。

- 台本（ポスター4_台本_4分30秒。11-17 までは ポスター4_台本_3分）：ポスターの節（1〜6）の順に、そのまま読める文。【 】は図を指す合図（読まない）。
  灰色の【省略可】の文は、時間が足りないときに省く。話す字数から、時間の目安（1 分 300 字）と、節ごとの経過時間を数える。
- 質問対策（ポスター4_質問対策）：数値の早見表、答えにくい質問への構え、ポスターの表記で指摘されそうな点、
  想定される質問と答え（まず言うこと・聞かれたら言う補足）、数値の出どころ。
- 数値は、ポスター4 の図と同じ学習結果（3 材料の最新の学習結果フォルダ。全ショット版・全厚み総合。11-09 の報告ノート）の CSV から読む。
  ポスターに書かれた値（決定係数・誤差）と CSV が合うこと、文の前提（低射出率側の誤差が大きい、など）を確かめ、違えば止める。
- 同じ名前のファイルがあれば止める（上書きしない）。LuaLaTeX は _tmp/poster4_talk/<日時>/ で動かし、できた tex と PDF をコピーする。
- 同じ日の 2 回目の依頼で、台本を「初めて聞く大人が 3 分で概要とやったことを理解できる」形に書き直し（専門用語の言い換え、
  伝えること 3 つ、言い換えの表）、質問対策から射出成形そのものの基礎の質問（成形機・センサ、成形条件の範囲）を外した。
- 2026-11-17 のユーザーの指定で、台本の冒頭を短くした：挨拶と名乗りを外し、前提を「射出成形 → 製品や金型の設計に樹脂流動解析 →
  それに欠かせない粘度のモデルの改良 → 具体的には（研究概要）」の流れでつなぎ、すぐ研究概要に入る。
- 2026-11-17 の 2 回目のユーザーの依頼で、質問対策に、温度シフトファクターと機械学習の質問を足した（いつ・どう組み込むか、
  目的変数と説明変数がそろってから学習が終わるまでの流れ、カーネルの定義、ARD の長さスケールの結果と読み方、線形カーネル、過学習、
  予測の計算、係数の決め方と不確かさなど）。「カーネルと長さスケールの早見表」も足した。長さスケールを元の単位に直した値・
  線形カーネルの部分の大きさ・予測の傾きは、学習済みモデル（.joblib）から計算する。縦だけか縦と横か・樹脂温度でのシフト・C との比べは、
  シフトファクター検討用のノートの結果（検討結果フォルダ）から読む。
- 2026-11-17 の 3 回目のユーザーの依頼で、ポスター4_添削版（完成版の添削と複数デザイン/ポスター4_発表報告/ポスター4_添削版.pdf）に合わせた。
  台本は 4 分 30 秒（名前も ポスター4_台本_4分30秒）にし、実験結果・学習結果・予測精度検証の結果を具体的な数値で言い、
  グラフの所では縦軸と横軸が何かを言い、入力の樹脂圧力差（ΔP2）と樹脂温度最高値がどの値かを言う（定義は報告書B の文）。
  入力の範囲は学習済みモデルの学習データから、せん断速度と粘度の範囲は学習データ（5data）の CSV と検証の予測値の CSV から、
  予測幅は検証の予測値の CSV から読む。質問対策は、引用の文と「ポスターの表記で指摘されそうな点」を添削版に合わせ、入力の定義の質問を足した。
  台本の時間の目安には、話す時間（1 分 300 字）に、読み上げ音声と同じ間（図を指す合図・段落・節。PAUSE_*）を足す
  （発表時間 4 分 30 秒は、間を含めた長さで合わせる）。
- 同じ日の 4 回目のユーザーの依頼で、せん断速度と粘度の範囲、キャピラリーとの差、温度による粘度の比の 3 つは言わず、台本の中に
  （ ）で示すだけにした（\\memo。時間・読み上げ音声に入れない）。また、ユーザーが PP の P3_3 を直した学習データ（このときの TRAIN_CSV_FIX。5 回目から TRAIN_CSV）から、
  入力の範囲を出すようにした（ポスターの PP のモデルは直す前の値で学習しているので、そのことを台本の前書きと質問対策に書く）。
- 同じ日の 5 回目のユーザーの依頼で、PP の数値を、直した P3_3（1117修正_）で学習し直した 4回目の学習結果（ポスター4_添削版の PP の図と同じ）にした
  （ポスターの文は変えない。ユーザーの指示）。4回目の PP は、温度シフトが尤度で支持されない（θ=0 との LML の差 0.05）ので、質問対策の温度シフトの答えを直した。
  温度ホールドアウト検証とシフトファクター検討用のノートは、PP では直す前のデータ（3回目）でしか実行していないので、その値には † を付けて、
  直す前のデータの結果だと書く（PRE）。PP の MAPE は予測値の CSV から計算し（ポスター/図 の表は 3回目）、線形カーネルの有無の比は 4回目と
  4回目_線形項なし のフォルダから出す（NOLIN_ROOT）。学習結果のグラフの線のずれのうち、温度シフトファクターの分と樹脂温度最高値の分も出す（curve_gap）。
- 同じ日の 6 回目のユーザーの依頼で、質問対策に、ユーザーの 6 つの質問への答えを足した：外れ値を含めて学習したか・除いた基準（前の外れ値の質問を
  書き直した。除く前の行数は OUTLIER_ROWS、規則 f はデータ確認/外れ値の検知_規則f.csv）、アレニウスの式と WLF 式（前の GPPS の WLF 式の質問を
  まとめた。2 つの型の尤度は検討用のノートの 候補の選び方.csv）、式を学習の前に決めたか、学習後のモデルだけで温度シフトを考えた予測ができるか、
  温度シフトファクターなしとの比べ（温度ホールドアウト検証の全温度の回の 3data の誤差）。
- 同じ日の 7 回目のユーザーの依頼「ポスターの文章をやっぱり誤解のない形に変更」で、ポスター4_添削版の 5. の 2 か所の文を直した
  （モデルの説明の 4 行目「…温度シフトファクターを組み込んだ。温度の影響は、入力の樹脂温度最高値でも表される。」と、学習結果よりの
  1 つ目の点「…予測平均値と予測幅が変化している（バージンPPでは、主に樹脂温度最高値の入力の違いによる）。」。添削版の tex を直接直した）。
  それに合わせて、台本の 5. の文（温度の影響は樹脂温度最高値の入力でも表す、線と帯が「ずれる」→「変わる」、PP は主に樹脂温度最高値の
  違い〔省略可〕）と、質問対策の引用・「ポスターの表記で指摘されそうな点」を直した。
- 同じ日の 8 回目のユーザーの依頼で、質問対策の D 節に、温度シフトファクターの係数 E とは何か・E の求め方（今の格子の方法）の 2 問を足した
  （横の係数の 95% 区間 EH_LO・EH_HI と、粗い格子の点の数 NCOARSE を読む）。

使い方（リポジトリ直下で）: uv run python 04_補助スクリプト/build_poster4_talk.py [--out <フォルダ>] [--only talk|qa]
  （既定：05_…/07_学習結果報告書作成_261015/ポスター/完成版の添削と複数デザイン/ポスター4_発表報告/。--out はリポジトリ直下からの相対パス。
   --only を付けると、台本〔talk〕か質問対策〔qa〕の一方だけを作る。付けなければ両方）
"""
import argparse
import datetime
import math
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import report_common as rc  # noqa: E402

D = "05_材料ごとの機械学習データと予測検証データ"
ROOTS = {"PP": D + "/01_バージンPP再々実験(データと結果)/01_学習結果/4回目_01_バージンPP再々実験_学習結果",
         "MIX": D + "/02_粉砕材混合PP実験(データと結果)/02_学習結果/5回目_02_粉砕材混合PP実験_学習結果",
         "GPPS": D + "/03_GPPS実験(データと結果)/03_学習結果/3回目_03_GPPS実験_学習結果"}
# ↑ PP は 2026-11-17 の 5 回目から 4回目（ユーザーが P3_3 を直した 1117修正_ のデータで学習し直したもの。ポスター4_添削版の PP の図と同じ）
PRE = {"PP": D + "/01_バージンPP再々実験(データと結果)/01_学習結果/3回目_01_バージンPP再々実験_学習結果"}
# ↑ 直す前の P3_3 で学習した結果。温度ホールドアウト検証（HOLD）とシフトファクター検討用のノートの結果（STUDY）は、PP ではまだ直す前のデータでしか
#   実行していないので、PP のそれらの値は、ここ（3回目）に対応するもの（文では † を付ける）。直す前からの変化を言うのにも使う
NOLIN_ROOT = {"PP": ROOTS["PP"] + "_線形項なし"}   # PP の線形カーネルなし（4回目）。リサイクルPP・GPPS は LINEAR_CSV（11-11）
DAG = r"$^{\dagger}$"                              # 直す前のデータの結果の印
SUB = "全ショット版/全厚み総合"
MODEL_DIR = {"PP": "A_Tset_Arr_アレニウス型の縦だけの温度シフトに加熱シリンダ温度採用",
             "MIX": "A_Tset_Arr_アレニウス型の縦だけの温度シフトに加熱シリンダ温度採用",
             "GPPS": "B_Tset_Arr_アレニウス型の温度シフトに加熱シリンダ温度採用"}
KEY = {"PP": "A_Tset_Arr", "MIX": "A_Tset_Arr", "GPPS": "B_Tset_Arr"}
HOLD = SUB + "/温度ホールドアウト検証"
POSTER_DIR = D + "/07_学習結果報告書作成_261015/ポスター"
OUT_DEFAULT = POSTER_DIR + "/完成版の添削と複数デザイン/ポスター4_発表報告"   # 2026-11-17 にユーザーが台本・質問対策・添削版をここへ移した
TENSAKU_PDF = "07_学習結果報告書作成_261015/ポスター/完成版の添削と複数デザイン/ポスター4_発表報告/ポスター4_添削版.pdf"   # 文に書く場所（05_… から）
MAPE_CSV = POSTER_DIR + "/図/予測誤差の指標_MAE_MSE_MAPE.csv"
STEP5 = "results/step5/metrics.csv"
STEP5_MAT = {"PP": "PP", "MIX": "粉砕材混合PP"}
POSTER_VALUES = {"PP": (0.992, 27), "MIX": (0.995, 22), "GPPS": (0.998, 13)}   # ポスター4_添削版のパリティ図に書かれた決定係数と RMSE(%)（PP は 11-17 に 4回目の値に直した。元のポスター4 は 0.991・28）
TRAIN_CSV = {"PP": D + "/01_バージンPP再々実験(データと結果)/01_学習・検証データ/学習データ_pp_max_5data/1117修正_結合_03_pp_max_5data_外れ値なし.csv",
             "MIX": D + "/02_粉砕材混合PP実験(データと結果)/02_学習・検証データ/学習データ_mixpp_max_5data/結合_03_mixpp_max_5data_外れ値なし.csv",
             "GPPS": D + "/03_GPPS実験(データと結果)/03_学習・検証データ/学習データ_gpps_max_5data/結合_gpps_5data_20261014_060353.csv"}   # 学習データ（5data。通常版）
# ↑ PP は、2026-11-17 にユーザーが P3_3（樹脂圧力差）を直した学習データ（厚み 0.5 mm の 104 行）。4回目の学習データ
TRAIN_CSV_OLD = {"PP": D + "/01_バージンPP再々実験(データと結果)/01_学習・検証データ/学習データ_pp_max_5data/結合_03_pp_max_5data_外れ値なし.csv"}
# ↑ 直す前の PP の学習データ（3回目の学習データ）。直した行の数・厚み・直す前 ÷ 直した値を言うのに使う（p33_fix）
OUTLIER_ROWS = {"PP": {"5data": (465, 460), "3data": (99, 95)}, "MIX": {"5data": (465, 459), "3data": (108, 101)}}
# ↑ ショットごとの外れ値を除く前 → 除いた後の行数（本体の CLAUDE.md §3.2。PP の 3data は no1.5-3.0 版）。除いた後の行数が、学習結果フォルダの
#   データの概要.csv の「読んだ行数」と同じことを確かめる（11-17 の 6 回目。外れ値の質問）。GPPS は、除く前の行数の記録がない
C_ETA, C_GD, C_P33, C_TFF ="樹脂粘度η34(Pa・s)", "せん断速度γ34(1/s)", "流動中樹脂圧力P3_3(MPa)", "流動中樹脂温度(℃)"   # 学習データの列（3 材料で同じ名前）
MATS = ["PP", "MIX", "GPPS"]
CPM = 300                     # 話す速さの目安（1 分あたりの字数）
PAUSE_SECTION, PAUSE_PARA, PAUSE_CUE = 1.0, 0.6, 0.4   # 間の秒数：節の間・段落の間・図を指す合図（読み上げ音声と同じ。時間の目安にも足す。2026-11-17 の 3 回目）
STEM_TALK = "ポスター4_台本_4分30秒"   # 2026-11-17 に 3 分 → 4 分 30 秒（ユーザーの指定）
STEM_QA = "ポスター4_質問対策"
GOOD = 1 / 1.1                # 温度ホールドアウト検証の「良い」の境目（ノートと同じ）
GPPS_LML = {"Arr": 936.2, "WLF": 933.2}          # 本番学習の設定の決定.pdf の 5.3 節（GPPS の 2 つの型の LML）
GPPS_RMSE_WLF = 0.1117                            # 同じ節：B_Tset_WLF の 3data の RMSE（ln η）
STUDY = {"PP": D + "/01_バージンPP再々実験(データと結果)/01_検討結果/01_バージンPP再々実験_シフト検討結果(5all学習データ)",
         "MIX": D + "/02_粉砕材混合PP実験(データと結果)/02_検討結果/02_粉砕材混合PP実験_検討結果(5all学習データ)",
         "GPPS": D + "/03_GPPS実験(データと結果)/03_検討結果/03_GPPS実験_検討結果(5all学習データ)"}   # シフトファクター検討用のノートの結果（通常版。本番学習の設定の決定.pdf の元）
STUDY_SUB = SUB + "/シフトファクターの検討"
LINEAR_CSV = D + "/08_カーネル関数の違いによる結果の違い/線形項あり・なしの比較表_範囲別_通常版.csv"   # 11-11 の線形項あり・なしの比較
LIN_MAT = {"PP": "PP", "MIX": "粉砕材混合PP", "GPPS": "GPPS"}
EV_LIT = 40.0                 # 文献の流動の活性化エネルギー（PP で 40 kJ/mol 前後）
R_GAS = 8.314                 # 気体定数 [J/mol/K]（mat_common と同じ）


def r0(x):
    return int(math.floor(x + 0.5))


def pct(rmse_ln):
    """ln η の RMSE を % に換算（e^RMSE − 1。ポスターのパリティ図と同じ）"""
    return r0((math.exp(rmse_ln) - 1) * 100)


def rd(path):
    p = os.path.join(rc.W, path)
    if not os.path.exists(p):
        sys.exit(f"読めない：{path}")
    return pd.read_csv(p, encoding="utf-8-sig")


def span(vals, fmt=str):
    a, b = min(vals), max(vals)
    return fmt(a) if fmt(a) == fmt(b) else f"{fmt(a)}〜{fmt(b)}"


def num(x):
    """係数の表示（整数なら小数点なし）。負の数は数式にする（文字の負号だと、前に空きが入るため）"""
    s = f"{x:g}"
    return f"${s}$" if x < 0 else s


def check(cond, msg):
    if not cond:
        sys.exit(f"文の前提が結果と合わない：{msg}（スクリプトの文を見直す）")


def sgn(x, d):
    """小数 d 桁の表示。負の数は数式にする（文字の負号だと、前に空きが入るため）。−0.00 は 0.00 にする"""
    s = f"{x:.{d}f}"
    if float(s) == 0:
        return s.lstrip("-")
    return f"${s}$" if s.startswith("-") else s


def sci(v):
    """小さな正の数の表示（0.01 未満は a×10^n）"""
    if v >= 0.01:
        return f"{v:.2f}"
    e = int(math.floor(math.log10(v)))
    return f"${v / 10 ** e:.1f}\\times10^{{{e}}}$"


def sig2(v):
    """有効数字 2 桁の表示（台本で読む値。100 以上は 3 桁ごとにカンマ）"""
    d = int(math.floor(math.log10(abs(v))))
    x = round(v, 1 - d)
    if x >= 100:
        return f"{int(round(x)):,}"
    if x >= 10:
        return str(int(round(x)))
    return f"{x:.{1 - d}f}"


def rng2(a, b, fmt=sig2):
    """範囲「a〜b」（台本で読む値）"""
    return f"{fmt(a)}〜{fmt(b)}"


def ln_arr(E, T_c, Tref_c):
    """アレニウス型の ln a(T)（E は kJ/mol、温度は ℃。mat_common.shift_ln と同じ式）"""
    return E * 1000.0 / R_GAS * (1.0 / (T_c + 273.15) - 1.0 / (Tref_c + 273.15))


def p33_fix(m, tr):
    """ユーザーが直した P3_3（2026-11-17。TRAIN_CSV が直した学習データ、TRAIN_CSV_OLD が直す前）を比べる（11-17 の 5 回目。それまでの fixed_p33 の代わり）。
    tr は直した学習データ（正の行）。直す前の学習データの行が同じ並びで、P3_3 のほかの列（粘度・せん断速度・樹脂温度）が同じことを確かめる。
    返す値：直した行の数、直す前 ÷ 直した値の範囲、直した行の厚み、直す前の P3_3 の上限 [MPa]"""
    old = rd(TRAIN_CSV_OLD[m])
    old = old[(old[C_ETA] > 0) & (old[C_GD] > 0) & (old[C_P33] > 0)]
    check(list(old["元ファイル"]) == list(tr["元ファイル"]), f"{m}：直す前と直した学習データの行が、同じ並び")
    check(np.allclose(old[[C_ETA, C_GD, C_TFF]].to_numpy(float), tr[[C_ETA, C_GD, C_TFF]].to_numpy(float)), f"{m}：直したのは P3_3 だけ")
    o, n = old[C_P33].to_numpy(float), tr[C_P33].to_numpy(float)
    ch = ~np.isclose(o, n)
    check(ch.sum() > 0, f"{m}：直した行がある")
    ratio = o[ch] / n[ch]
    return {"N_P33FIX": int(ch.sum()), "P33FIX_RATIO": (float(ratio.min()), float(ratio.max())),
            "P33FIX_H": sorted(set(tr.loc[ch, "キャビ厚"])), "DP_OLD_MAX": float(o.max())}


def curve_gap(m, base, tr):
    """学習結果のグラフ（全厚み共通・全温度。mat_common.flow_plot_common）と同じ条件で、低い温度の線が高い温度の線の何倍かを、
    温度シフトファクターの分と、樹脂温度最高値の入力の分に分ける（2026-11-17 の 5 回目。_tmp/talk_pp4_1117/gap_split.py と同じ計算）。
    条件：せん断速度は学習データの範囲（60 点）、P3_3 は学習データの中央値（対数）、樹脂温度最高値はその設定温度の学習データの中央値、Tset はその温度。
    全体 ＝ η(低い温度; その Tff) ÷ η(高い温度; その Tff)、シフトの分 ＝ Tff を高い温度の値にそろえたときの比、Tff の分 ＝ 全体 ÷ シフトの分。
    返す値：全体の範囲、シフトの分（せん断速度によらず同じことを確かめる）、Tff の分の範囲、設定温度ごとの Tff の中央値 [℃]"""
    os.environ.setdefault("NB_REPO", rc.W)
    import joblib
    import mat_common as mc
    fit = joblib.load(os.path.join(rc.W, base, f"学習済みモデル_{KEY[m]}.joblib"))["fit"]
    tcol = [c for c in tr.columns if "加熱シリンダ温度" in c]
    tset = tr[tcol[0]].astype(float) if tcol else tr["元ファイル"].str.extract(r"_(\d+)℃_")[0].astype(float)
    p_med = float(np.exp(np.log(tr[C_P33]).median()))
    gd = np.exp(np.linspace(np.log(tr[C_GD].min()), np.log(tr[C_GD].max()), 60))
    ts = sorted(tset.unique())
    tff = {t: float(np.exp(np.log(tr.loc[tset == t, C_TFF] + 273.15).median()) - 273.15) for t in ts}
    mu = lambda t, tf: mc.predict(fit, mc.model_inputs(gd, p_med, tf, float(t)))[0]   # noqa: E731
    tot = np.exp(mu(ts[0], tff[ts[0]]) - mu(ts[-1], tff[ts[-1]]))
    sh = np.exp(mu(ts[0], tff[ts[-1]]) - mu(ts[-1], tff[ts[-1]]))
    check(np.ptp(sh) < 1e-9, f"{m}：縦だけの温度シフトの分は、せん断速度によらず同じ")
    tp = tot / sh
    dist = float((np.log(tff[ts[-1]] + 273.15) - np.log(tff[ts[0]] + 273.15)) / np.log(tr[C_TFF] + 273.15).std())   # Tff の中央値の差（標準偏差の何倍か）
    return {"GAP": (float(tot.min()), float(tot.max())), "GAP_SH": float(sh[0]), "GAP_TFF": (float(tp.min()), float(tp.max())),
            "TFF_MED": [tff[t] for t in ts], "TFF_DIST": dist}


def same_b(b, v):
    return np.allclose(np.asarray(b, float), np.asarray(v, float))


def kernel_facts(m, base, hp):
    """学習済みモデル（joblib）から、質問対策の ARD・線形カーネル・過学習の質問に使う値を出す（2026-11-17）。
    カーネルの形・探索範囲・やり直しの回数・乱数シードが文と同じこと、長さスケールが ハイパーパラメータ.csv と同じことも確かめる。
    返す値：長さスケール（樹脂圧力差・樹脂温度）、長さスケールを元の単位に直した値、学習の範囲が長さスケールの何個分か、
    予測の平均の傾き（学習データの点で、ほかの入力を固定して 1 つだけ動かしたときの平均。標準化した入力 1 あたりの ln η）、
    入力の 1 標準偏差の大きさ、予測の平均の線形カーネルの部分と Matern の部分の大きさ（標準偏差）、学習データの残差の標準偏差"""
    import joblib
    obj = joblib.load(os.path.join(rc.W, base, f"学習済みモデル_{KEY[m]}.joblib"))
    f = obj["fit"]
    gp = f["gp"]
    k = gp.kernel_
    mk, lk, wk = k.k1.k1, k.k1.k2, k.k2
    check(f["key"] == KEY[m] and f.get("mode") == "shot", f"{m} の学習済みモデルが、ポスターのモデル（全ショット版）")
    check(same_b(mk.k1.constant_value_bounds, (1e-3, 1e3)) and same_b(mk.k2.length_scale_bounds, (1e-2, 1e2)) and mk.k2.nu == 2.5
          and same_b(lk.k1.constant_value_bounds, (1e-3, 1e3)) and same_b(lk.k2.sigma_0_bounds, (1e-5, 1e3))
          and same_b(wk.noise_level_bounds, (1e-6, 1e1)), f"{m} のカーネルの形と探索範囲が、文に書いたものと同じ")
    check(gp.n_restarts_optimizer == 20 and gp.random_state == 42 and not gp.normalize_y, f"{m}：やり直し 20 回（合計 21 回）、乱数シード 42、出力の自動標準化なし")
    X, a = gp.X_train_, gp.alpha_
    ls = np.atleast_1d(mk.k2.length_scale)
    lsc = [c for c in hp.columns if c.startswith("ls_") and c != "ls_at_bound"]
    check(len(lsc) == 3 and np.allclose(ls, [float(hp[c].iloc[0]) for c in lsc], rtol=1e-6), f"{m} の学習済みモデルの長さスケールが、ハイパーパラメータ.csv と同じ")
    sd, mu = np.asarray(f["sd_x"], float), np.asarray(f["mu_x"], float)
    X0 = X * sd + mu                                  # 元の単位（ln）。GPPS の 1 列目は温度でずらした値
    width = ls * sd                                   # 長さスケールを ln の幅に直したもの
    nls = (X0.max(axis=0) - X0.min(axis=0)) / width   # 学習の範囲が、長さスケールの何個分か
    t_mean = float(np.exp(mu[2]))                     # 樹脂温度 [K]（ln の平均）
    h = 1e-4
    slope = []
    for i in range(X.shape[1]):
        Xp, Xm = X.copy(), X.copy()
        Xp[:, i] += h
        Xm[:, i] -= h
        slope.append(float(np.mean((gp.predict(Xp) - gp.predict(Xm)) / (2 * h))))
    fm, fl = mk(X, X) @ a, lk(X, X) @ a               # 予測の平均（学習データの点）の、Matern の部分と線形カーネルの部分
    return {"LS_P": float(ls[1]), "LS_T": float(ls[2]), "NLS": [float(v) for v in nls],
            "WG": float(np.exp(width[0])), "WP": float(np.exp(width[1])), "WT_C": t_mean * (math.exp(width[2]) - 1),
            "SLOPE": slope, "SDG": float(np.exp(sd[0])), "SDT_C": t_mean * (math.exp(sd[2]) - 1),
            "PART_M": float(fm.std()), "PART_L": float(fl.std()), "RES": float((gp.y_train_ - fm - fl).std()),
            "NKF": int(X.shape[0]), "NOISE_KF": float(np.sqrt(wk.noise_level)), "TREF": float(f["cfg"]["Tref"]),
            "EV_KF": float(f["tv"]), "EH_KF": float(f["th"]),
            # 学習データの入力の範囲（2026-11-17 の 3 回目。台本で言う）：樹脂圧力差 [MPa]（入力は Pa の ln）、樹脂温度最高値 [℃]（入力は K の ln）
            "DP_MIN": float(np.exp(X0[:, 1].min()) / 1e6), "DP_MAX": float(np.exp(X0[:, 1].max()) / 1e6),
            "T_MIN": float(np.exp(X0[:, 2].min()) - 273.15), "T_MAX": float(np.exp(X0[:, 2].max()) - 273.15)}


# ---------------------------------------------------------------- 数値を読む
def load():
    N, raw = {}, {}
    for m in MATS:
        base = f"{ROOTS[m]}/{SUB}/{MODEL_DIR[m]}"
        ind = rd(base + "/予測検証_3data_指標.csv")
        g = ind[ind["モデル"] == KEY[m]].set_index("範囲")
        c = ind[ind["モデル"] == "Cross-WLF"].set_index("範囲")
        r = {}
        for rng, tag in (("全体", "all"), ("低速側", "low"), ("高速側", "high")):
            r[f"E_{tag}"], r[f"CW_{tag}"] = pct(g.loc[rng, "RMSE"]), pct(c.loc[rng, "RMSE"])
            r[f"COV_{tag}"] = r0(g.loc[rng, "coverage95"] * 100)
            r[f"N_{tag}"], r[f"NC_{tag}"] = int(g.loc[rng, "行数"]), int(g.loc[rng, "条件数"])
        r["R2"], r["R2R"], r["NOISE"] = float(g.loc["全体", "R2"]), float(g.loc["全体", "R2_real"]), float(g.loc["全体", "ノイズの標準偏差"])
        r["RMSE_all"] = float(g.loc["全体", "RMSE"])
        r2p, ep = POSTER_VALUES[m]
        check(round(r["R2"], 3) == r2p and r["E_all"] == ep, f"{m} のポスターの値 R² {r2p}・{ep}% と CSV（{r['R2']:.4f}・{r['E_all']}%）")
        mt = rd(base + "/まとめ.csv")
        mt = mt[mt["モデル"] == KEY[m]].iloc[0]
        r["DLML"], r["CRIT"], r["LML"] = float(mt["θ=0 との LML の差"]), float(mt["支持の基準"]), float(mt["LML"])
        r["SUP"] = bool(mt["尤度で支持"])              # 11-17 の 5 回目から、PP（4回目）は支持されない。文の前提は下でまとめて確かめる
        sf = rd(base + "/シフトファクターの推定.csv").iloc[0]
        r["EV"], r["EV_LO"], r["EV_HI"], r["EH"] = float(sf["tv_best"]), float(sf["tv_lo"]), float(sf["tv_hi"]), float(sf["th_best"])
        r["EH_LO"], r["EH_HI"] = float(sf["th_lo"]), float(sf["th_hi"])   # 横の係数の 95% 区間（係数 E の質問。11-17 の 8 回目）
        bai = rd(base + "/シフトファクターの倍率.csv")
        vcol = [k for k in bai.columns if k.startswith("縦の倍率")][0]
        r["VR"] = float(bai[vcol].iloc[0])
        r["VR_T"] = "/".join(re.findall(r"a_v\((\d+)\)", vcol)) + " ℃"
        hp = rd(base + "/ハイパーパラメータ.csv")
        r["LS"] = float(hp[[k for k in hp.columns if k.startswith("ls_ln γ̇")][0]].iloc[0])
        cw = rd(base + "/CrossWLF_係数.csv")
        r["CW_EDGE"] = bool(cw["扱い"].str.contains("範囲の端").any())
        dc = rd(ROOTS[m] + "/データ確認/データの概要.csv").set_index("組")
        r["NTR"], r["NTRC"] = int(dc.loc["5data", "使う行数"]), int(dc.loc["5data", "条件数"])
        check(int(dc.loc["3data", "使う行数"]) == r["N_all"], f"{m} の検証の行数（データの概要と指標）")
        r["DC"] = {g: (int(dc.loc[g, "読んだ行数"]), int(dc.loc[g, "一覧で除外"])) for g in ("5data", "3data")}   # 外れ値の質問（11-17 の 6 回目）
        for g, (_, n_after) in OUTLIER_ROWS.get(m, {}).items():
            check(r["DC"][g][0] == n_after, f"{m}：{g} の外れ値なしの行数が、本体の CLAUDE.md §3.2 の値と同じ")
        rf = rd(ROOTS[m] + "/データ確認/外れ値の検知_規則f.csv")
        rf = rf[~rf["除外後に再計算"].astype(bool)]
        r["RULEF"] = rf[rf["flag"].astype(bool)][["cond", "grp", "z", "resid"]].to_dict("records")   # 規則 f で拾った条件
        p3 = rd(ROOTS[m] + "/データ確認/P3_3の性質.csv").iloc[0]
        r["P33ERR"] = float(p3["算出式の形で表したときの最大誤差 [%]"])
        r["P33R2"], r["GDR2"] = float(p3["検証 R²（ln P3_3・ln γ̇）"]), float(p3["検証 R²（ln γ̇ だけ）"])
        r["CORR32"] = float(p3.get("相関 ln P3_3 と ln P3_2", np.nan))   # GPPS は P3_2 の列がないので、この列もない（NaN）
        if m in PRE:                                   # 直す前のデータの学習（11-17 の 5 回目）。検討用のノート・温度ホールドアウト検証は、こちらに対応する
            pb = f"{PRE[m]}/{SUB}/{MODEL_DIR[m]}"
            pm = rd(pb + "/まとめ.csv")
            pm = pm[pm["モデル"] == KEY[m]].iloc[0]
            pi = rd(pb + "/予測検証_3data_指標.csv")
            pi = pi[pi["モデル"] == KEY[m]].set_index("範囲")
            r["PRE"] = {"DLML": float(pm["θ=0 との LML の差"]), "LML": float(pm["LML"]), "SUP": bool(pm["尤度で支持"]),
                        "RMSE": float(pi.loc["全体", "RMSE"]), "R2": float(pi.loc["全体", "R2"]),
                        "E_all": pct(pi.loc["全体", "RMSE"]), "E_low": pct(pi.loc["低速側", "RMSE"]),
                        "EV": float(rd(pb + "/シフトファクターの推定.csv").iloc[0]["tv_best"])}
            check((round(r["PRE"]["R2"], 3), r["PRE"]["E_all"]) == (0.991, 28), f"{m}：直す前の学習の決定係数と誤差が、元のポスター4 の値（0.991・28%）")
            r["PRE"]["SLOPE_T"] = kernel_facts(m, pb, rd(pb + "/ハイパーパラメータ.csv"))["SLOPE"][2]   # 直す前のモデルの樹脂温度の傾き
            ref = r["PRE"]
        else:
            ref = {"DLML": r["DLML"], "LML": r["LML"], "RMSE": r["RMSE_all"]}
        hs = rd(f"{PRE.get(m, ROOTS[m])}/{HOLD}/まとめ.csv")
        hs = hs[hs["モデル"] == KEY[m]].iloc[0]
        r["HO"] = [float(hs[f"5data の RMSE の比（÷ C_noTset）{p}"]) for p in ("低い温度", "中の温度", "高い温度")]
        r["HO_C"] = [float(hs[f"5data の RMSE の比（÷ C）{p}"]) for p in ("低い温度", "中の温度", "高い温度")]
        r["HO_CONC"] = str(hs["結論"])
        hf = rd(f"{PRE.get(m, ROOTS[m])}/{HOLD}/シフトファクターの推定_回ごと.csv")
        hf = hf[hf["回"].str.startswith("除いた温度")].sort_values("除いた温度 [℃]")
        r["HO_T"] = [r0(t) for t in hf["除いた温度 [℃]"]]
        r["HO_EV"] = [float(v) for v in hf["tv_best"]]
        ha = rd(f"{PRE.get(m, ROOTS[m])}/{HOLD}/予測の指標_回ごと.csv")   # 全温度の回の 3data の誤差（温度シフトなしとの比べ。11-17 の 6 回目）
        ha = ha[(ha["回"] == "全温度") & (ha["範囲"] == "3data 全体")].set_index("モデル")
        r["ALL3"] = {k: float(ha.loc[k, "RMSE"]) for k in (KEY[m], "C", "C_noTset")}
        check(abs(r["ALL3"][KEY[m]] - ref["RMSE"]) < 5e-4, f"{m}：温度ホールドアウト検証の全温度の回が、本番の学習（PP は直す前のデータの学習）と同じ誤差")
        # ---- 温度シフトファクターと機械学習の質問のための値（2026-11-17）
        is_a = KEY[m].startswith("A")
        check(not bool(hp["ls_at_bound"].iloc[0]) and not bool(hp["noise_at_lower_bound"].iloc[0]), f"{m}：長さスケールとノイズが、探索範囲の端に張り付いていない")
        r["AMP"], r["LIN_AMP"], r["SIGMA0"] = float(hp["amplitude"].iloc[0]), float(hp["lin_amplitude"].iloc[0]), float(hp["lin_sigma0"].iloc[0])
        r.update(kernel_facts(m, base, hp))
        check(r["NKF"] == r["NTR"], f"{m}：学習済みモデルの学習データの行数が、データの概要と同じ")
        # ---- 台本で言う数値（2026-11-17 の 3 回目）：計測データの範囲と、検証データの予測幅
        tr = rd(TRAIN_CSV[m])
        tr = tr[(tr[C_ETA] > 0) & (tr[C_GD] > 0) & (tr[C_P33] > 0)]
        check(len(tr) == r["NTR"], f"{m}：学習データの CSV の行数（{len(tr)}）が、学習に使った行数（{r['NTR']}）と同じ")
        check(abs(tr[C_P33].min() - r["DP_MIN"]) < 1e-6 * r["DP_MIN"] and abs(tr[C_P33].max() - r["DP_MAX"]) < 1e-6 * r["DP_MAX"]
              and abs(tr[C_TFF].min() - r["T_MIN"]) < 1e-6 and abs(tr[C_TFF].max() - r["T_MAX"]) < 1e-6,
              f"{m}：学習済みモデルの入力の範囲が、学習データの CSV の樹脂圧力差（P3_3）・樹脂温度最高値の範囲と同じ")
        r["N_P33FIX"] = 0
        if m in TRAIN_CSV_OLD:                         # 2026-11-17 にユーザーが直した P3_3（4回目は、直した値で学習した。11-17 の 5 回目）
            r.update(p33_fix(m, tr))
        pv = rd(base + "/予測検証_3data_予測値.csv")
        check(len(pv) == r["N_all"] and set(pv["group"]) == {"低速側", "高速側"}, f"{m}：検証の予測値の行数と範囲の名前")
        mape = float((np.abs(pv["予測η"] - pv["eta"]) / pv["eta"]).mean() * 100)   # 11-17 の 5 回目から、予測値の CSV で計算（ポスター/図 の表と同じ式）
        if m not in PRE:                               # PP の表の値は直す前（3回目）のもの
            mp = rd(MAPE_CSV)
            mp = mp[(mp["モデル"] == KEY[m]) & (mp["範囲"] == "全体") & (mp["行数"] == r["N_all"])]
            check(len(mp) == 1 and abs(float(mp["MAPE_%"].iloc[0]) - mape) < 1e-6, f"{m} の MAPE が、ポスター/図 の表と同じ")
        r["MAPE"] = r0(mape)
        if m in PRE:                                   # 学習結果のグラフの線のずれの内訳（11-17 の 5 回目）
            r.update(curve_gap(m, base, tr))
        r["G_MIN"], r["G_MAX"] = float(min(tr[C_GD].min(), pv["gd"].min())), float(max(tr[C_GD].max(), pv["gd"].max()))
        r["ETA_MIN"], r["ETA_MAX"] = float(min(tr[C_ETA].min(), pv["eta"].min())), float(max(tr[C_ETA].max(), pv["eta"].max()))
        for grp, tag in (("低速側", "low"), ("高速側", "high")):
            q = pv[pv["group"] == grp]
            check(len(q) == r[f"N_{tag}"], f"{m}：検証の予測値の {grp} の行数")
            r[f"BAND_{tag}"] = float(((q["上限95"] / q["予測η"]) - 1).median() * 100)   # 95% 予測区間の上限が、予測値の何 % 上か（中央値）
        check(abs(r["NOISE_KF"] - r["NOISE"]) < 1e-4 and r["EV_KF"] == r["EV"] and r["EH_KF"] == r["EH"], f"{m}：学習済みモデルのノイズと係数が、CSV と同じ")
        pg = rd(base + "/プロファイル尤度_格子.csv")
        check(pg["tv"].min() == -20 and pg["tv"].max() == 60
              and (bool((pg["th"] == 0).all()) if is_a else (pg["th"].min() == -20 and pg["th"].max() == 40)), f"{m}：粗い格子の範囲（縦 −20〜60、横 −20〜40 kJ/mol）")
        near = np.sort(pg.loc[((pg["tv"] - r["EV"]).abs() <= 4.5) & ((pg["th"] - r["EH"]).abs() < 1e-9), "tv"].unique())
        check(abs(np.diff(near).min() - (0.5 if is_a else 1.0)) < 1e-9, f"{m}：細かい格子の刻み（縦だけ 0.5、縦と横 1 kJ/mol）")
        check(len(pg) == int(sf["n_points"]), f"{m}：格子の点の数")
        r["NPTS"] = len(pg)
        r["NCOARSE"] = int((pg["stage"] == "粗い").sum())   # 粗い格子の点の数（係数 E の求め方の質問。11-17 の 8 回目）
        check(r["NCOARSE"] == (17 if is_a else 13 * 17), f"{m}：粗い格子の点の数（5 kJ/mol 刻み。縦 17 点、縦と横 13 × 17 点）")
        lit = float(pg.loc[(pg["tv"] - EV_LIT).abs() < 1e-9, "LML"].max())
        r["LIT_DROP"] = r["LML"] - lit
        check(lit < float(sf["LML_at_0"]), f"{m}：縦の係数 40 kJ/mol は、係数 0 より尤度が小さい")
        t_lo, t_hi = (int(t) for t in re.findall(r"\d+", r["VR_T"]))
        check(t_lo < r["TREF"] < t_hi, f"{m}：縦の倍率の温度（{r['VR_T']}）と基準温度")
        lv = lambda E, T: ln_arr(E, T, r["TREF"])   # noqa: E731
        check(abs(math.exp(lv(r["EV"], t_lo) - lv(r["EV"], t_hi)) - r["VR"]) < 0.006, f"{m}：縦の倍率をアレニウス式で計算し直すと、CSV と同じ")
        r["TLO"], r["THI"] = t_lo, t_hi
        r["SHL"] = (1 - math.exp(-lv(r["EV"], t_lo))) * 100             # 低い温度のデータの粘度を何 % 小さくして基準温度にそろえるか
        r["SHH"] = (math.exp(-lv(r["EV"], t_hi)) - 1) * 100             # 高い温度のデータを何 % 大きくするか
        r["HSH"] = (math.exp(lv(r["EH"], t_lo) - lv(r["EH"], t_hi)) - 1) * 100   # 横のずれ（低い温度と高い温度の間）
        r["VR_LO"] = math.exp(lv(r["EV_LO"], t_lo) - lv(r["EV_LO"], t_hi))
        r["VR_HI"] = math.exp(lv(r["EV_HI"], t_lo) - lv(r["EV_HI"], t_hi))
        cd = rd(f"{STUDY[m]}/{STUDY_SUB}/候補の選び方.csv")
        ca = cd[(cd["シフトに使う温度"] == "Tset") & (cd["型"] == "アレニウス型")].iloc[0]
        check(str(ca["この組の結論"]) == KEY[m], f"{m}：検討用のノートの候補が、ポスターのモデル")
        r["CA"], r["CB"], r["BA"] = float(ca["A の ΔLML"]), float(ca["B の ΔLML"]), float(ca["B − A の LML"])
        cw_row = cd[(cd["シフトに使う温度"] == "Tset") & (cd["型"] == "WLF型")].iloc[0]   # WLF 型との比べ（11-17 の 6 回目）
        dcol = "A の ΔLML" if is_a else "B の ΔLML"
        r["ARR_D"], r["WLF_D"] = float(ca[dcol]), float(cw_row[dcol])   # 係数 0 のモデルは 2 つの型で同じなので、差がそのまま LML の差
        check(abs((r["CA"] if is_a else r["CB"]) - ref["DLML"]) < 0.01, f"{m}：検討用のノートの ΔLML が、本番の学習（PP は直す前のデータの学習）と同じ")
        tf = cd[cd["シフトに使う温度"] == "Tff"]
        check(not (tf["A は支持（1.92 以上）"].astype(str) == "True").any() and not (tf["B は支持（3.0 以上）"].astype(str) == "True").any(),
              f"{m}：樹脂温度でのシフトは、どの型でも支持されない")
        r["TFFA"], r["TFFB"] = float(tf["A の ΔLML"].max()), float(tf["B の ΔLML"].max())
        check(r["TFFA"] < 1.92 and r["TFFB"] < 3.0, f"{m}：樹脂温度でのシフトの ΔLML は基準に届かない")
        sm = rd(f"{STUDY[m]}/{STUDY_SUB}/まとめ.csv").set_index("モデル")
        check(abs(float(sm.loc[KEY[m], "RMSE 全体"]) - ref["RMSE"]) < 5e-4 and abs(float(sm.loc[KEY[m], "LML"]) - ref["LML"]) < 0.06,
              f"{m}：検討用のノートの候補の結果が、本番の学習（PP は直す前のデータの学習）と同じ")
        r["DLC"] = ref["LML"] - float(sm.loc["C", "LML"])
        r["RC"] = ref["RMSE"] / float(sm.loc["C", "RMSE 全体"])
        raw[m] = r
    st = rd(STEP5)
    for m, sm in STEP5_MAT.items():
        s = st[(st["material"] == sm) & (st["version"] == "5data") & (st["split"] == "S_V") & (st["model"] == "B_1")].set_index("subset")
        check(int(s.loc["全体", "n_test"]) == raw[m]["N_all"] and int(s.loc["全体", "n_cond_test"]) == raw[m]["NC_all"],
              f"{m} の本体の B_1 の検証データが、ポスターと同じ数")
        raw[m]["B1"] = {t: pct(s.loc[k, "RMSE"]) for k, t in (("全体", "all"), ("低速側", "low"), ("高速側", "high"))}
        raw[m]["B1COV"] = r0(s.loc["全体", "coverage95"] * 100)
        raw[m]["B1NOISE"], raw[m]["PURE"] = float(s.loc["全体", "noise_sd"]), float(s.loc["全体", "pure_sd"])
    lc = rd(LINEAR_CSV)
    for m in MATS:
        if m in NOLIN_ROOT:                            # PP（4回目）：線形カーネルあり・なしの 2 つのフォルダから出す（11-17 の 5 回目。比較表は 3回目）
            nb = f"{NOLIN_ROOT[m]}/{SUB}/{MODEL_DIR[m]}"
            check("lin_amplitude" not in rd(nb + "/ハイパーパラメータ.csv").columns, f"{m}：線形項なしのフォルダのモデルに、線形カーネルがない")
            ia, ib = (x[x["モデル"] == KEY[m]].set_index("範囲") for x in (rd(f"{ROOTS[m]}/{SUB}/{MODEL_DIR[m]}/予測検証_3data_指標.csv"), rd(nb + "/予測検証_3data_指標.csv")))
            raw[m]["NOLIN"] = {tag: float(ib.loc[rng, "RMSE"] / ia.loc[rng, "RMSE"]) for rng, tag in (("全体", "all"), ("低速側", "low"), ("高速側", "high"))}
            continue
        t = lc[(lc["材料"] == LIN_MAT[m]) & (lc["厚み [mm]"].astype(str) == "全厚み")].set_index("範囲")
        check(abs(float(t.loc["全体", "RMSE（線形項あり）"]) - raw[m]["RMSE_all"]) < 5e-4, f"{m}：線形項あり・なしの比較表の「あり」が、ポスターのモデルと同じ")
        raw[m]["NOLIN"] = {tag: float(t.loc[rng, "RMSEの比（線形項なし÷線形項あり）"]) for rng, tag in (("全体", "all"), ("低速側", "low"), ("高速側", "high"))}

    # ---- 文の前提を確かめる
    P, X, G = raw["PP"], raw["MIX"], raw["GPPS"]
    for m in MATS:
        r = raw[m]
        check(r["E_low"] > 2 * r["E_high"], f"{m}：誤差の大部分は低射出率側")
        check(r["COV_low"] == 100, f"{m}：低射出率側は全部が 95% 区間に入る")
        check(r["COV_high"] < 95, f"{m}：高射出率側の被覆率は 95% より小さい")
        check(r["E_high"] < r["CW_high"], f"{m}：高射出率側はガウス過程回帰のほうが小さい")
        check(r["E_all"] < r["CW_all"], f"{m}：全体はガウス過程回帰のほうが小さい")
        check(r["R2"] >= 0.99, f"{m}：決定係数（対数）0.99 以上")
        check(r["E_all"] <= 30, f"{m}：全体の誤差 30% 以内")
        check(r["EV"] < 20, f"{m}：係数は文献の活性化エネルギー（40 kJ/mol 前後）よりずっと小さい")
        check(r["CW_EDGE"], f"{m}：Cross-WLF の係数の一部が探索範囲の端")
    check(P["CW_low"] < P["E_low"] and X["CW_low"] < X["E_low"] and G["E_low"] < G["CW_low"], "低射出率側：PP・リサイクルPP は Cross-WLF が小さく、GPPS はガウス過程回帰が小さい")
    check(P["E_low"] > 30, "PP の低射出率側は 30% を超える")
    check(G["E_low"] < X["E_low"] < P["E_low"], "低射出率側の誤差：GPPS < リサイクルPP < PP")
    check(G["HO_CONC"] == "要る" and X["HO_CONC"] == "要る" and P["HO_CONC"] == "どちらかといえば要らない", "温度ホールドアウト検証の結論")
    check(all(v < GOOD for v in G["HO"]) and all(v < GOOD for v in X["HO"]), "GPPS・リサイクルPP は 3 回とも「良い」")
    check(sum(v > 1.1 for v in P["HO"]) == 2 and P["HO"][0] < 1.1, "PP は 3 回中 2 回「悪い」")
    check(P["HO_T"] == [190, 210, 230] and P["HO_EV"][0] > 0 and P["HO_EV"][1] < 0 and P["HO_EV"][2] < 0, "PP は 210・230 ℃ を外すと係数が負")
    check(abs(G["LML"] - GPPS_LML["Arr"]) < 0.05, "GPPS のアレニウス型の LML が報告書と同じ")
    check(abs(G["RMSE_all"] - 0.1213) < 5e-5 and GPPS_RMSE_WLF < G["RMSE_all"], "GPPS の誤差が報告書と同じで、WLF 型のほうが少し小さい")
    for m in ("PP", "MIX"):
        check(raw[m]["B1"]["all"] < raw[m]["E_all"] and raw[m]["B1"]["low"] < raw[m]["E_low"], f"{m}：本体の B_1 のほうが誤差が小さい")
        check(raw[m]["NOISE"] * 4 < raw[m]["PURE"], f"{m}：ノイズの標準偏差はショット間のばらつきよりずっと小さい")
    check(G["EH"] > 0 and P["EH"] == 0 and X["EH"] == 0, "横の係数は GPPS だけ")
    check(X["EV_LO"] > 0 and G["EV_LO"] > 0 and X["VR"] > 1.05, "係数 E の質問の例：リサイクルPP・GPPS の縦の係数の 95% 区間は 0 を含まず、リサイクルPP の倍率は 1.05 倍より大きい（11-17 の 8 回目）")
    for m in MATS:                                     # 温度シフトファクターと機械学習の質問の前提（2026-11-17）
        r = raw[m]
        check(r["LS"] < 0.3 and 1 < r["LS_P"] < 4 and r["LS_T"] > 10, f"{m}：長さスケールの大小（せん断速度 < 0.3、1 < 樹脂圧力差 < 4、樹脂温度 > 10）")
        check(r["NLS"][0] > 10 and r["NLS"][2] < 0.5, f"{m}：学習の範囲は、せん断速度で長さスケール 10 個分超、樹脂温度で 0.5 個分未満")
        check(r["SLOPE"][0] < 0 < r["SLOPE"][1] and abs(r["SLOPE"][2]) < 0.1 and abs(r["SLOPE"][2]) < abs(r["SLOPE"][1]) < abs(r["SLOPE"][0]),
              f"{m}：予測の傾き（せん断速度で下がり、樹脂圧力差で上がる。大きさは せん断速度 > 樹脂圧力差 > 樹脂温度〔0.1 未満〕）")
        # ↑ 11-17 の 5 回目：PP の 4回目は樹脂温度の傾きが −0.047（直す前 −0.021）で、それまでの「0.03 未満」を超えたので、0.1 未満と大小の順にした
        check(r["PART_L"] > 3 * r["PART_M"], f"{m}：予測の平均は、線形カーネルの部分が Matern の部分よりずっと大きい")
        check(r["NOLIN"]["all"] > 5 and r["NOLIN"]["low"] > 5 and 0.8 < r["NOLIN"]["high"] < 1.6,
              f"{m}：線形カーネルなしは、全体・低射出率側で 5 倍超、高射出率側は大きくは変わらない（0.8〜1.6 倍。PP の 4回目は 0.85）")
        check(r["LIT_DROP"] > 50, f"{m}：縦の係数 40 kJ/mol では、尤度が大きく下がる")
        check(r["RES"] < 1.5 * r["NOISE"], f"{m}：学習データの残差は、ノイズの標準偏差と同じくらい")
        check(r["DLC"] > 0, f"{m}：温度シフトのモデルの LML は、加熱シリンダ温度を入力に並べた C より大きい")
    check(G["BA"] >= 1.92 and P["BA"] < 1.92 and X["BA"] < 1.92, "縦と横 − 縦だけの尤度の差が 1.92 以上なのは GPPS だけ")
    # ---- PP を直した P3_3 で学習し直した 4回目の文の前提（11-17 の 5 回目）
    check(X["SUP"] and G["SUP"] and not P["SUP"] and P["PRE"]["SUP"], "温度シフトの尤度の支持：リサイクルPP・GPPS は支持、PP は直す前は支持・直した後は支持されない")
    check(P["EV_LO"] < 0 < P["EV_HI"], "PP の縦の係数の 95% 区間は 0 をまたぐ")
    check(P["PRE"]["EV"] > P["EV"] and P["PRE"]["E_low"] > P["E_low"], "PP：直す前より、縦の係数と低射出率側の誤差が小さい")
    check(math.log(P["GAP_SH"]) < 0.5 * math.log(P["GAP"][0]) and P["GAP_TFF"][0] > 1, "PP：グラフの線のずれの大部分は、樹脂温度最高値の入力の分")
    check(abs(P["GAP_SH"] - P["VR"]) < 0.006, "PP：線のずれの温度シフトファクターの分が、縦の倍率と同じ")
    check(math.log(P["GAP_TFF"][1]) > abs(P["SLOPE"][2]) * P["TFF_DIST"],
          "PP：グラフの樹脂温度最高値の分（最大）は、学習データの点での平均の傾き × 中央値の差から見込むより大きい")
    check(abs(P["SLOPE"][2]) > 1.8 * abs(P["PRE"]["SLOPE_T"]) and abs(P["SLOPE"][2]) > 1.8 * max(abs(raw[m]["SLOPE"][2]) for m in ("MIX", "GPPS")),
          "PP：樹脂温度の傾きは、直す前とほかの 2 材料の約 2 倍以上")
    check(all(raw[m]["CORR32"] >= 0.945 for m in ("PP", "MIX")), "樹脂圧力差と P3_2 の対数の相関が 0.95 以上（PP・リサイクルPP）")
    check(P["DP_OLD_MAX"] > 2 * P["DP_MAX"], "PP：直す前の樹脂圧力差の上限は、直した値の上限よりずっと大きい（文の前提）")
    # ---- ユーザーの 6 つの質問（外れ値・アレニウスと WLF・式を学習の前に決めたか・学習後のモデルだけで予測できるか・温度シフトなしとの比べ）の前提（11-17 の 6 回目）
    check(all(raw[m]["DC"]["5data"][1] == 0 for m in MATS), "学習データ（5data）は、問題データ一覧で除いた行がない（除いたのはショットごとの外れ値だけ）")
    check(P["DC"]["3data"][1] > 0 and G["DC"]["3data"][1] > 0 and X["DC"]["3data"][1] == 0, "一覧で除いたのは、PP と GPPS の検証データ（3data）だけ")
    check(len(P["RULEF"]) == 1 and P["RULEF"][0]["cond"] == "190_0.5_320" and P["RULEF"][0]["grp"] == "3data" and not X["RULEF"] and not G["RULEF"],
          "規則 f で拾ったのは、PP の 190 ℃・0.5 mm・320 mm/s（検証データ）の 1 条件だけ")
    check(G["NTR"] < 5 * G["NTRC"], "GPPS の学習データは、1 条件 5 ショットより少ない条件がある")
    check(all(raw[m]["ARR_D"] > raw[m]["WLF_D"] for m in MATS), "3 材料とも、アレニウス型のほうが WLF 型より尤度が大きい")
    check(abs(G["ARR_D"] - G["WLF_D"] - (GPPS_LML["Arr"] - GPPS_LML["WLF"])) < 0.06, "GPPS の 2 つの型の尤度の差が、報告書の値と同じ")
    check(all(raw[m]["RC"] < 1.1 for m in MATS), "3data の誤差は、3 材料とも C の 1.1 倍以内")
    ns3 = {m: raw[m]["ALL3"][KEY[m]] / raw[m]["ALL3"]["C_noTset"] for m in MATS}
    check(ns3["PP"] < 1 and ns3["MIX"] < 1 and ns3["GPPS"] > 1, "3data の誤差：温度シフトなしより、PP・リサイクルPP は小さく、GPPS は大きい")
    check(min(raw[m]["BAND_low"] for m in MATS) > 2 * max(raw[m]["BAND_high"] for m in MATS), "台本：予測幅は、低射出率側のほうがずっと広い（3 材料とも）")
    check(all(raw[m]["VR"] > 1 for m in MATS), "台本：低い温度の粘度は、高い温度より大きい")
    check(all(raw[m]["G_MIN"] < 600 and raw[m]["G_MAX"] > 2500 for m in MATS), "台本：計測データのせん断速度は、キャピラリーと比べた 600〜2,500 1/s を含む")

    # ---- 文に入れる値
    for m in MATS:
        r = raw[m]
        for k in ("E_all", "E_low", "E_high", "CW_all", "CW_low", "CW_high", "COV_all", "COV_low", "COV_high",
                  "N_all", "NC_all", "NTR", "NTRC", "MAPE"):
            N[f"{k}_{m}"] = r[k]
        N[f"R2_{m}"], N[f"R2R_{m}"] = f"{r['R2']:.3f}", f"{r['R2R']:.2f}"
        N[f"EV_{m}"], N[f"EVLO_{m}"], N[f"EVHI_{m}"] = num(r["EV"]), num(r["EV_LO"]), num(r["EV_HI"])
        N[f"VR_{m}"], N[f"VRT_{m}"] = f"{r['VR']:.2f}", r["VR_T"]
        N[f"DLML_{m}"] = f"{r['DLML']:.2f}" if r["DLML"] < 1 else f"{r['DLML']:.1f}"   # 1 未満は小数 2 桁（PP の 4回目 0.05。11-17 の 5 回目）
        N[f"CRIT_{m}"] = "1.92" if abs(r["CRIT"] - 1.92) < 1e-9 else f"{r['CRIT']:.1f}"
        N[f"P33R2_{m}"], N[f"GDR2_{m}"] = f"{r['P33R2']:.3f}", f"{r['GDR2']:.3f}"
        N[f"HO_{m}"] = "・".join(f"{v:.2f}" for v in r["HO"])
        N[f"HOC_{m}"] = "・".join(f"{v:.2f}" for v in r["HO_C"])
    N["EH_GPPS"] = num(G["EH"])
    N["EHCI_GPPS"] = (f"（95\\% 区間 {num(G['EH_LO'])}〜{num(G['EH_HI'])}）" if G["EH_LO"] < G["EH_HI"] else
                      r"（95\% 区間に入った格子の点は、この値の列だけ。区間の幅は、細かい格子の刻み 1 kJ/mol より狭い）")   # 係数 E の質問（11-17 の 8 回目）。3回目の GPPS は 1〜1
    for m in MATS:
        N[f"NCOARSE_{m}"] = raw[m]["NCOARSE"]
    N["E_ALL_RANGE"] = span([raw[m]["E_all"] for m in MATS])
    N["E_LOW_RANGE"] = span([raw[m]["E_low"] for m in MATS])
    N["E_HIGH_RANGE"] = span([raw[m]["E_high"] for m in MATS])
    N["CW_HIGH_RANGE"] = span([raw[m]["CW_high"] for m in MATS])
    N["COV_ALL_RANGE"] = span([raw[m]["COV_all"] for m in MATS])
    N["COV_HIGH_RANGE"] = span([raw[m]["COV_high"] for m in MATS])
    N["MAPE_RANGE"] = span([raw[m]["MAPE"] for m in MATS])
    N["P33ERR_RANGE"] = span([r0(raw[m]["P33ERR"]) for m in MATS])
    N["LS_RANGE"] = span([raw[m]["LS"] for m in MATS], lambda v: f"{v:.2f}")
    N["NOISE_RANGE"] = span([raw[m]["NOISE"] for m in MATS], lambda v: f"{v:.3f}")
    N["R2_MIN_TXT"] = "0.99"
    N["HO_GPPS_GAIN"] = span([r0((1 - v) * 100) for v in G["HO"]])
    N["HOEV_PP"] = "、".join(f"{t} ℃ を外すと {num(v)}" for t, v in zip(P["HO_T"], P["HO_EV"]))
    for m in ("PP", "MIX"):
        for t in ("all", "low", "high"):
            N[f"B1_{m}_{t}"] = raw[m]["B1"][t]
        N[f"B1COV_{m}"] = raw[m]["B1COV"]
    N["PURE_SD"] = span([raw[m]["PURE"] for m in ("PP", "MIX")], lambda v: f"{v:.2f}")
    N["PURE_PCT"] = span([pct(raw[m]["PURE"]) for m in ("PP", "MIX")])
    N["B1NOISE"] = span([raw[m]["B1NOISE"] for m in ("PP", "MIX")], lambda v: f"{v:.2f}")
    N["NOISE_FRAC"] = span([r0(raw[m]["PURE"] / raw[m]["NOISE"]) for m in ("PP", "MIX")])
    N["GPPS_LML_ARR"], N["GPPS_LML_WLF"] = f"{GPPS_LML['Arr']:.1f}", f"{GPPS_LML['WLF']:.1f}"
    N["GPPS_RMSE_WLF"], N["GPPS_RMSE_ARR"] = f"{GPPS_RMSE_WLF:.3f}", f"{G['RMSE_all']:.3f}"
    # ---- 温度シフトファクターと機械学習の質問の値（2026-11-17）
    for m in MATS:
        r = raw[m]
        N[f"LS3_{m}"] = f"{r['LS']:.2f}／{r['LS_P']:.1f}／{r['LS_T']:.1f}"
        N[f"SL3_{m}"] = "／".join(sgn(v, 2) for v in r["SLOPE"])
        N[f"NPTS_{m}"] = r["NPTS"]
        N[f"LIT_{m}"] = r0(r["LIT_DROP"])
        N[f"CA_{m}"], N[f"CB_{m}"], N[f"BA_{m}"] = sgn(r["CA"], 2), sgn(r["CB"], 2), sgn(r["BA"], 2)
        N[f"TFFA_{m}"], N[f"TFFB_{m}"] = f"{r['TFFA']:.2f}", f"{r['TFFB']:.2f}"
        N[f"DLC_{m}"], N[f"RC_{m}"] = f"{r['DLC']:.1f}", f"{r['RC']:.2f}"
        N[f"TLOW_{m}"], N[f"THIGH_{m}"] = r["TLO"], r["THI"]
        N[f"SHL_{m}"], N[f"SHH_{m}"] = f"{r['SHL']:.1f}", f"{r['SHH']:.1f}"
        N[f"VRLO_{m}"], N[f"VRHI_{m}"] = f"{r['VR_LO']:.2f}", f"{r['VR_HI']:.2f}"
        N[f"TREF_{m}"] = f"{r['TREF']:g}"
    for m in PRE:                                      # 直す前のデータの結果（検討用のノート・温度ホールドアウト検証）に † を付ける（11-17 の 5 回目）
        for k in ("CA", "CB", "BA", "TFFA", "TFFB", "DLC", "RC", "HO", "HOC"):
            N[f"{k}_{m}"] += DAG
    N["HOEV_PP"] += DAG
    pre = P["PRE"]
    N["DLMLPRE_PP"], N["R2PRE_PP"], N["EPRE_PP"], N["ELPRE_PP"] = f"{pre['DLML']:.1f}", f"{pre['R2']:.3f}", pre["E_all"], pre["E_low"]
    N["EVPRE_PP"] = num(pre["EV"])
    N["GAP_PP"] = rng2(*P["GAP"], fmt=lambda v: f"{v:.2f}")
    N["GAPSH_PP"] = f"{P['GAP_SH']:.2f}"
    N["TFFMED_PP"] = "・".join(str(r0(v)) for v in P["TFF_MED"])
    N["GAPTFF_PP"] = rng2(*P["GAP_TFF"], fmt=lambda v: f"{v:.2f}")
    N["GAPEXP_PP"] = f"{math.exp(abs(P['SLOPE'][2]) * P['TFF_DIST']):.2f}"
    N["SLT_PP"], N["SLTPRE_PP"] = sgn(P["SLOPE"][2], 3), sgn(P["PRE"]["SLOPE_T"], 3)
    N["SLT_OTHER"] = "・".join(sgn(raw[m]["SLOPE"][2], 3) for m in ("MIX", "GPPS"))
    # ---- ユーザーの 6 つの質問の値（11-17 の 6 回目）。PRE の材料（PP）の検討用のノート・温度ホールドアウト検証の値には † を付ける
    for m in MATS:
        r, dg = raw[m], (DAG if m in PRE else "")
        N[f"ARRD_{m}"], N[f"WLFD_{m}"] = f"{r['ARR_D']:.2f}{dg}", f"{r['WLF_D']:.2f}{dg}"
        N[f"AWD_{m}"] = f"{r['ARR_D'] - r['WLF_D']:.1f}{dg}"
        N[f"NS3_{m}"] = f"{r['ALL3'][KEY[m]] / r['ALL3']['C_noTset']:.2f}{dg}"
        N[f"EX3_{m}"] = r["DC"]["3data"][1]
    N["OUT_ROWS"] = "、".join(f"{lab} {OUTLIER_ROWS[m][g][0]} → {OUTLIER_ROWS[m][g][1]}" for g, gl in (("5data", "学習"), ("3data", "検証"))
                             for m, lab in (("PP", gl + " PP"), ("MIX", "リサイクルPP")))
    rf0 = P["RULEF"][0]
    N["RF_PP"] = f"ずれは予測の標準偏差の {rf0['z']:.1f} 倍、粘度で約 {math.exp(rf0['resid']):.2f} 倍"
    N["GPPS_FULL"] = 5 * G["NTRC"]
    N["HSH_GPPS"] = f"{G['HSH']:.1f}"
    col = lambda k, i=None: [raw[m][k] if i is None else raw[m][k][i] for m in MATS]   # noqa: E731
    N["LSP_RANGE"] = span(col("LS_P"), lambda v: f"{v:.1f}")
    N["LST_RANGE"] = span(col("LS_T"), lambda v: f"{v:.1f}")
    N["WG_RANGE"] = span(col("WG"), lambda v: f"{v:.2f}")
    N["WP_RANGE"] = span(col("WP"), lambda v: f"{v:.1f}")
    N["WT_RANGE"] = span(col("WT_C"), lambda v: str(r0(v)))
    N["NLSG_RANGE"] = span(col("NLS", 0), lambda v: str(r0(v)))
    N["NLSP_RANGE"] = span(col("NLS", 1), lambda v: f"{v:.1f}")
    N["NLST_RANGE"] = span(col("NLS", 2), lambda v: f"{v:.2f}")
    N["INVG_RANGE"] = span([1 / v for v in col("LS")], lambda v: f"{v:.1f}")
    N["INVP_RANGE"] = span([1 / v for v in col("LS_P")], lambda v: f"{v:.2f}")
    N["INVT_RANGE"] = span([1 / v for v in col("LS_T")], lambda v: f"{v:.3f}")
    N["SLG_RANGE"] = span([abs(v) for v in col("SLOPE", 0)], lambda v: f"{v:.1f}")
    N["SLP_RANGE"] = span(col("SLOPE", 1), lambda v: f"{v:.2f}")
    N["SLT_MAX"] = f"{max(abs(v) for v in col('SLOPE', 2)):.2f}"
    N["SLT_PCT"] = r0(max((math.exp(abs(v)) - 1) * 100 for v in col("SLOPE", 2)))
    N["SLG_FAC"] = span([math.exp(-v) for v in col("SLOPE", 0)], lambda v: str(r0(v)))
    N["SDG_RANGE"] = span(col("SDG"), lambda v: f"{v:.1f}")
    N["SDT_RANGE"] = span(col("SDT_C"), lambda v: str(r0(v)))
    N["PL_RANGE"] = span(col("PART_L"), lambda v: f"{v:.2f}")
    N["PM_RANGE"] = span(col("PART_M"), lambda v: f"{v:.2f}")
    N["RES_RANGE"] = span(col("RES"), lambda v: f"{v:.3f}")
    N["NOLIN_ALL_RANGE"] = span([raw[m]["NOLIN"]["all"] for m in MATS], lambda v: f"{v:.1f}")
    N["NOLIN_LOW_RANGE"] = span([raw[m]["NOLIN"]["low"] for m in MATS], lambda v: f"{v:.1f}")
    N["NOLIN_HIGH_RANGE"] = span([raw[m]["NOLIN"]["high"] for m in MATS], lambda v: f"{v:.2f}")
    # ---- 台本の数値と、入力の定義の質問の値（2026-11-17 の 3 回目）
    t0 = lambda v: str(r0(v))   # noqa: E731
    for m in MATS:
        r = raw[m]
        N[f"DP_{m}"], N[f"TFF_{m}"] = rng2(r["DP_MIN"], r["DP_MAX"]), rng2(r["T_MIN"], r["T_MAX"], t0)
        N[f"BANDH_{m}"], N[f"BANDL_{m}"] = r0(r["BAND_high"]), r0(r["BAND_low"])
        N[f"DPMAX_{m}"] = sig2(r["DP_MAX"])
    N["P33FIX_N"] = P["N_P33FIX"]                      # 2026-11-17 にユーザーが直した PP の P3_3
    N["P33FIX_RATIO"] = rng2(*P["P33FIX_RATIO"], fmt=lambda v: f"{v:.1f}")
    N["P33FIX_H"] = "・".join(f"{h:g}" for h in P["P33FIX_H"])
    N["DPOLD_PP"] = sig2(P["DP_OLD_MAX"])
    N["DP_RANGE"] = rng2(min(col("DP_MIN")), max(col("DP_MAX")))
    N["TFF_RANGE"] = rng2(min(col("T_MIN")), max(col("T_MAX")), t0)
    N["G_RANGE"] = rng2(min(col("G_MIN")), max(col("G_MAX")))
    N["ETA_RANGE"] = rng2(min(col("ETA_MIN")), max(col("ETA_MAX")))
    N["VR_RANGE"] = span(col("VR"), lambda v: f"{v:.2f}")
    N["BAND_HIGH"] = span(col("BAND_high"), t0)
    N["BAND_LOW"] = span(col("BAND_low"), t0)
    for lab, key in QREF.items():                      # 質問どうしの参照（Q の番号）
        hit = [i for i, row in enumerate(QA, 1) if key in row[2]]
        check(len(hit) == 1, f"参照する質問「{key}」がちょうど 1 つ")
        N[f"Q_{lab}"] = f"Q{hit[0]}"
    return N, raw


def sub(text, N):
    def rep(mo):
        k = mo.group(1)
        if k not in N:
            sys.exit(f"文の中の値 <<{k}>> がない")
        return str(N[k])
    return re.sub(r"<<([A-Za-z0-9_]+)>>", rep, text)


# ---------------------------------------------------------------- 台本
# 段落の中の [[ … ]] は【省略可】の文（時間が足りないときに省く）
TALK = [   # 聞き手：初めて聞く大人（2026-11-16 のユーザーの指定）。専門用語は、ポスターの言葉に短い言い換えを添える（WORDS）
    # 2026-11-17 の 3 回目（ユーザーの指定）：約 4 分 30 秒。添削版の節の順。実験結果・学習結果・予測精度検証は具体的な数値で言い、
    # グラフの所では縦軸と横軸が何かを言い、入力の樹脂圧力差と樹脂温度最高値がどの値かを言う（定義は報告書B の文）
    ("1. 研究概要", [   # 挨拶なし。前提（射出成形 → 樹脂流動解析 → 粘度モデルの改良）から、すぐ研究概要に入る（2026-11-17 のユーザーの指定）
        r"プラスチックの射出成形では、製品や金型の設計に、樹脂の流れを計算する樹脂流動解析が使われています。本研究では、この計算に欠かせない、樹脂の流れにくさ、つまり粘度のモデルを改良しています。",
        r"\cue{1. 研究概要の文}具体的には、金型の中で樹脂の圧力と温度を測れる計測金型をつくり、そのデータと機械学習で、粘度の予測値と、そのばらつきの幅を同時に推定するモデルを作りました。",
    ]),
    ("2. 計測金型およびセンサ", [
        r"\cue{計測金型の外観・キャビティの図}こちらが計測金型です。キャビティの 2 か所、①と②で樹脂の圧力を、②で温度を測ります。①と②の間は 17.5 mm です。",
    ]),
    ("3. 成形条件", [
        r"\cue{3. 成形条件}樹脂は、バージンのポリプロピレン、それに粉砕材を 25\% 混ぜたリサイクル材、ポリスチレンの 3 種類です。加熱シリンダ温度を 3 通り、流し込む速さである射出率を 1.6〜170 cm³/s の 12 通り、キャビティの厚みを 0.5〜2.0 mm の 4 通り、ポリスチレンでは 3 通りに変えて成形しました。",
    ]),
    ("4. 実験結果", [
        r"\cue{樹脂圧力と樹脂温度の経時変化}左のグラフは、縦軸が樹脂圧力、横軸が時間です。樹脂は①、②の順に届き、\cue{算出式}その時間の差と、②に届いた時刻の圧力の差から、せん断速度と粘度を計算します。右のグラフは、縦軸が樹脂温度、横軸が時間です。",
        r"\cue{「入力データに使用」の矢印}機械学習の入力には、②の圧力が最大のときの①と②の圧力の差である樹脂圧力差と、樹脂の先端が②に届いたときの温度の最高値である樹脂温度最高値を使います。学習データでは、それぞれ約 <<DP_RANGE>> MPa、約 <<TFF_RANGE>> ℃です。",
        r"\cue{樹脂粘度とせん断速度の関係の 3 つのグラフ}こちらは、縦軸が樹脂粘度、横軸が、流れの速さの目安であるせん断速度で、どちらも対数の目盛りです。\memo{せん断速度 約 <<G_RANGE>> 1/s、粘度 約 <<ETA_RANGE>> Pa$\cdot$s}速く流れるほど、粘度が下がります。\cue{白丸}白丸のキャピラリーレオメータの値とは、中くらいから速い側で、おおむね一致しました。\memo{600〜2,500 1/s で差 10\% 以内}",
    ]),
    ("5. 機械学習と学習結果", [
        r"\cue{入力→出力の流れ図}機械学習には、予測の値と一緒に予測の幅も出せる、ガウス過程回帰を使いました。入力は、せん断速度、樹脂圧力差、樹脂温度最高値の 3 つで、出力は粘度です。さらに、加熱シリンダ温度に基づくアレニウス式の温度シフトファクターで、温度による粘度の曲線のずれを表しました。温度の影響は、樹脂温度最高値の入力でも表されます。[[学習には、1 条件につき 5 回、ポリプロピレンでは <<NTR_PP>> 回分の成形のデータを使いました。]]",
        r"\cue{学習結果の 3 つのグラフ}学習結果のグラフも、縦軸が樹脂粘度、横軸がせん断速度です。線が予測平均値、帯が予測幅で、加熱シリンダ温度ごとに線と帯が変わっています。\memo{温度シフトファクターの分：低い温度の粘度は、高い温度の <<VR_RANGE>> 倍。PP の線のずれ <<GAP_PP>> 倍のうち、シフトの分は <<GAPSH_PP>> 倍}[[バージンのポリプロピレンでは、主に樹脂温度最高値の違いによります。]][[帯は、データのない遅い側ほど広がります。]]",   # 11-17 の 7 回目：ポスターの文に合わせて「ずれています」→「変わっています」、PP の文（省略可）を足した
        r"\cue{データの分割}射出率の遅い側の 1.6 と 2.7、速い側の 149 と 170 cm³/s は、学習に使わず、検証に回しました。\cue{3 つのパリティ図}パリティ図は、縦軸が樹脂粘度の予測値、横軸が実測値です。決定係数は 3 材料で <<R2_PP>>、<<R2_MIX>>、<<R2_GPPS>>、予測誤差率は <<E_all_PP>>、<<E_all_MIX>>、<<E_all_GPPS>>\% でした。",
        r"速い側の誤差は <<E_HIGH_RANGE>>\% と小さく、学習範囲の外の遅い側は <<E_LOW_RANGE>>\% と大きくなりました。ただ、遅い側では予測幅が、予測値の約 <<BAND_LOW>>\% 上まで広がり、検証データはすべて 95\% 予測区間に入りました。",
    ]),
    ("6. まとめ", [
        r"\cue{6. まとめ}まとめです。計測金型のデータと機械学習で、3 種類の樹脂の粘度の予測値とばらつきの幅を同時に推定するモデルを作り、学習に使っていない射出率でも、予測誤差率 30\% 以内で予測できました。今後は、計測方法の改良と、樹脂流動解析への組み込みを検討します。以上です。",
    ]),
]
MESSAGES = [   # この発表で伝えること（3 つ）
    r"\textbf{目的}：射出成形の製品や金型の設計に使う樹脂流動解析に欠かせない、樹脂の流れにくさ（粘度）のモデルを改良する。金型の中で測ったデータから、粘度の予測値と、そのばらつきの幅を出したい。",
    r"\textbf{やったこと}：計測金型で 3 種類の樹脂を、加熱シリンダ温度 3 通り・射出率 12 通り・厚み 4 通り（ポリスチレンは 3 通り）で成形してデータを集め、せん断速度・樹脂圧力差・樹脂温度最高値から粘度と予測幅を出すガウス過程回帰に、加熱シリンダ温度の温度シフトファクターを組み込んで学習した。",
    r"\textbf{結果}：学習に使っていない射出率でも、決定係数 <<R2_PP>>〜<<R2_GPPS>>、予測誤差率 <<E_ALL_RANGE>>\% で予測できた。データのない遅い側では予測幅が広がり、その検証データはすべて 95\% 予測区間に入った。",
]
WORDS = [      # ポスターの言葉 → 台本での言い方（台本では、ポスターの言葉に、右の言い換えを添える）
    (r"射出成形", r"溶かした樹脂を金型に流し込んでつくる方法"),
    (r"樹脂流動解析技術", r"樹脂流動解析（樹脂の流れをコンピュータで計算する技術）"),
    (r"樹脂粘度 $\eta$", r"樹脂の流れにくさ（粘度）"),
    (r"各種レオメータ・キャピラリーレオメータ", r"粘度を測る専用の装置"),
    (r"バージンPP／リサイクルPP／バージンGPPS", r"バージンのポリプロピレン／リサイクル材／ポリスチレン"),
    (r"加熱シリンダ温度", r"溶かす温度（である加熱シリンダ温度）"),
    (r"射出率", r"流し込む速さ（である射出率）"),
    (r"キャビティ厚み", r"キャビティの厚み"),
    (r"せん断速度 $\dot{\gamma}$", r"せん断速度（流れの速さの目安）"),
    (r"樹脂圧力差", r"②の圧力が最大になった時刻の①と②の圧力の差"),
    (r"樹脂温度最高値", r"樹脂の先端が②に届いたときの温度の最高値"),
    (r"ガウス過程回帰", r"予測の値と一緒に、その予測の幅も出せる方法"),
    (r"温度シフトファクター", r"温度の変化で粘度の曲線がずれる性質を表すもの"),
    (r"予測平均値／予測幅（95\% 予測区間）", r"線／帯"),
    (r"パリティ図", r"縦軸が予測値、横軸が実測値の図。斜めの線に乗れば予測どおり"),
    (r"低射出率側と高射出率側の計4条件", r"遅い側の 1.6 と 2.7 cm³/s、速い側の 149 と 170 cm³/s の計 4 通り"),
    (r"予測誤差率 RMSE(\%)", r"予測誤差率（予測と実測のずれ）"),
]
KEYWORDS = [
    ("1", r"射出成形 → 製品・金型の設計に樹脂流動解析 → 欠かせない粘度（流れにくさ）のモデルを改良 → 具体的には（研究概要）計測金型のデータ＋機械学習で、予測値＋ばらつきの幅"),
    ("2", r"その計測金型 → ①②で圧力、②で温度、①②の間 17.5 mm"),
    ("3", r"3 種類の樹脂 → 加熱シリンダ温度 190・210・230 ℃、射出率 1.6〜170 cm³/s の 12 通り、厚み 0.5〜2.0 mm の 4 通り（ポリスチレンは 3 通り）"),
    ("4", r"圧力のグラフ（縦：樹脂圧力、横：時間）→ 時間の差＋圧力の差でせん断速度と粘度 → 温度のグラフ（縦：樹脂温度、横：時間）→ 入力：樹脂圧力差（②が最大の時刻の①と②の差。約 <<DP_RANGE>> MPa）と樹脂温度最高値（先端が②に届いたときの最高値。約 <<TFF_RANGE>> ℃）→ 粘度のグラフ（縦：樹脂粘度、横：せん断速度、両対数）\memo{約 <<G_RANGE>> 1/s・約 <<ETA_RANGE>> Pa$\cdot$s} → キャピラリーと中〜速い側でおおむね一致\memo{600〜2,500 1/s で差 10\% 以内}"),
    ("5", r"ガウス過程回帰（入力 3 つ）＋アレニウス式の温度シフトファクター（温度の影響は樹脂温度最高値の入力でも）→ 学習結果（縦：樹脂粘度、横：せん断速度）線と帯が温度ごとに変わる（PP は主に樹脂温度最高値の違い）\memo{シフトの分：低い温度は <<VR_RANGE>> 倍} →1.6・2.7・149・170 cm³/s で検証 → パリティ図（縦：予測値、横：実測値）決定係数 <<R2_PP>>・<<R2_MIX>>・<<R2_GPPS>>、誤差 <<E_all_PP>>・<<E_all_MIX>>・<<E_all_GPPS>>\% → 速い側 <<E_HIGH_RANGE>>\%、遅い側 <<E_LOW_RANGE>>\%、遅い側は予測幅が約 <<BAND_LOW>>\% に広がり、すべて区間の中"),
    ("6", r"3 種類でモデル → 予測誤差率 <<E_ALL_RANGE>>\%（30\% 以内）→ 今後：ばらつきをより多く測れる計測方法と、樹脂流動解析への組み込み方"),
]


def spoken_len(tex):
    """話す字数（【 】の合図・言わない（ ）の数値〔\\memo。2026-11-17 の 4 回目〕・LaTeX の命令・空白を除く）"""
    s = re.sub(r"\\(?:cue|memo)\{[^}]*\}", "", tex)
    s = re.sub(r"\\[A-Za-z]+", "", s)
    s = re.sub(r"[{}$_^\\\s]", "", s)
    return len(s)


def split_opt(par):
    """段落 → (必ず話す部分, 省略可の部分)"""
    opt = "".join(re.findall(r"\[\[(.*?)\]\]", par))
    return re.sub(r"\[\[.*?\]\]", "", par), opt


def mmss(sec):
    sec = r0(sec)
    return f"{sec // 60}:{sec % 60:02d}"


def pause_sec(i, pars):
    """節の中の間の秒数（読み上げ音声と同じ：節の前の間〔最初の節は除く〕・段落の間・図を指す合図。読み始めの合図の間は除く）"""
    s = (PAUSE_SECTION if i else 0.0) + PAUSE_PARA * (len(pars) - 1) + PAUSE_CUE * sum(p.count(r"\cue{") for p in pars)
    if i == 0 and pars[0].startswith(r"\cue{"):
        s -= PAUSE_CUE
    return s


def build_talk(N):
    rows, body, t_all, t_short = [], [], 0.0, 0.0
    n_all = n_short = 0
    t_pause = 0.0
    for i, (title, pars) in enumerate(TALK):
        pars = [sub(p, N) for p in pars]
        must = sum(spoken_len(split_opt(p)[0]) for p in pars)
        opt = sum(spoken_len(split_opt(p)[1]) for p in pars)
        ps = pause_sec(i, pars)
        t_pause += ps
        sec = (must + opt) / CPM * 60 + ps
        start, t_all = t_all, t_all + sec
        t_short += must / CPM * 60 + ps
        n_all, n_short = n_all + must + opt, n_short + must
        rows.append(f"{title} & {must + opt}" + (f"（{opt}）" if opt else "") + f" & 約 {r0(sec / 5) * 5} 秒 & {mmss(start)}〜{mmss(t_all)} \\\\")
        tex_pars = "\n\n".join(re.sub(r"\[\[(.*?)\]\]", r"\\opt{\1}", p) for p in pars)
        body.append(f"\\section*{{{title}\\quad{{\\normalsize\\textcolor{{black!55}}{{{mmss(start)}〜{mmss(t_all)}（約 {r0(sec / 5) * 5} 秒）}}}}}}\n{tex_pars}\n")
    kw = "\n".join(f"  \\item[{a}] {sub(b, N)}" for a, b in KEYWORDS)
    msg = "\n".join(f"  \\item {sub(x, N)}" for x in MESSAGES)
    words = "\n".join(f"{a} & {b} \\\\" for a, b in WORDS)
    tex = r"""%% 生成：04_補助スクリプト/build_poster4_talk.py（直接は直さない。直すときはスクリプトを直して作り直す）
\documentclass[paper=a4paper,fontsize=13pt,line_length=36zw,number_of_lines=30]{jlreq}
\usepackage[haranoaji]{luatexja-preset}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\newcommand{\cue}[1]{{\small\textcolor{blue!70!black}{【#1】}}}
\newcommand{\opt}[1]{\textcolor{black!45}{{\scriptsize【省略可】}#1}}
\newcommand{\kak}[1]{\textcolor{orange!75!black}{（要確認：#1）}}
\newcommand{\memo}[1]{{\small\textcolor{black!50}{（#1）}}}
\setlist[itemize]{leftmargin=1.5\zw,itemsep=0.2ex,topsep=0.4ex}
\setlength{\parskip}{0.5ex}
\begin{document}
\begin{center}
{\Large ポスター4（添削版） 口頭発表の台本（約 4 分 30 秒）}\\[0.6ex]
{\small @@DATE@@　@@TENSAKU@@}
\end{center}

{\small
\begin{itemize}
  \item 聞き手は、この研究を初めて聞く大人を想定した。4 分 30 秒で「何のために、何をして、どうなったか」が分かることを優先し、専門用語には短い言い換えを添えた（対応は最後の表）。
  \item 実験結果・学習結果・予測精度検証の結果は、具体的な数値で言う。グラフの所では、縦軸と横軸が何かを先に言う。入力の樹脂圧力差と樹脂温度最高値がどの値かは、4. 実験結果で言う（11-17 のご指定）。
  \item 灰色の小さい\memo{ }の数値は、言わない（11-17 のご指定：せん断速度と粘度の範囲、キャピラリーとの差、温度による粘度の比）。聞かれたときに答える数値として、台本に残した。時間にも数えていない。
  \item 文はそのまま読める形にした。\cue{ }は、ポスターの図を指す合図（読まない）。添削版の図と並びは、元のポスター4 と同じ。
  \item 灰色の\opt{ }の文は、時間が足りないときに省く。時間は、1 分 @@CPM@@ 字で数えた話す時間に、図を指す合図（@@PCUE@@ 秒）・段落の間（@@PPARA@@ 秒）・節の間（@@PSEC@@ 秒）の間（計 約 @@PAUSE@@ 秒。読み上げ音声と同じ）を足した目安：全部読むと約 @@ALL@@（@@NALL@@ 字）、省略可を省くと約 @@SHORT@@（@@NSHORT@@ 字）。練習で計って、超えるなら省略可の文を省く。
  \item 挨拶と名乗りは入れず、研究の前提（射出成形 → 樹脂流動解析 → 粘度のモデルの改良）から、すぐ研究概要に入る。数値の内訳と、質問への答えは、別の資料「ポスター4\_質問対策.pdf」。
\end{itemize}
}

\section*{この発表で伝えること（3 つ）}
\begin{itemize}
@@MSG@@
\end{itemize}

\begin{center}\small
\begin{tabular}{lrrr}
\toprule
節 & 字数（うち省略可） & 時間 & 経過 \\
\midrule
@@ROWS@@
\midrule
合計 & @@NALL@@ & 約 @@ALL@@ & \\
\bottomrule
\end{tabular}
\end{center}

@@BODY@@

\section*{要点だけの版}
覚えるときや、文を読まずにポスターを見ながら話すときに使う。
\begin{itemize}[leftmargin=4\zw,labelwidth=3.5\zw,align=left]
@@KW@@
\end{itemize}

\section*{ポスターの言葉と、台本での言い方}
ポスターには専門用語のまま書いてあるので、指しながら話すときは、ポスターの言葉に右の言い換えを添える。
{\small
\begin{longtable}{>{\raggedright\arraybackslash}p{14\zw}>{\raggedright\arraybackslash}p{20\zw}}
\toprule
ポスターの言葉 & 台本での言い方 \\
\midrule
\endhead
@@WORDS@@
\bottomrule
\end{longtable}
}

\section*{台本の数値の出どころ}
{\small
\begin{itemize}
  \item ポスターに書かれた値：成形条件（3 材料、粉砕材 25 wt\%、加熱シリンダ温度 3 条件、射出率 12 条件、キャビティ厚み 4 条件）、キャビティの寸法（①と②の間 17.5 mm、厚み 0.5〜2.0 mm）、決定係数と予測誤差率 RMSE(\%)。射出率の値（1.6〜170 cm³/s。検証の 1.6・2.7・149・170 cm³/s）は、射出速度 3〜320 mm/s に 0.531 を掛けた値（本体の CLAUDE.md §3.1）。
  \item 圧力と温度のグラフの例の条件（台本では言わない。聞かれたら）：バージンPP、キャビティ厚み 1.0 mm、加熱シリンダ温度 210 ℃、射出率 42.5 cm³/s（添削版の注記）。
  \item 入力の定義：報告書B（07\_学習結果報告書作成\_261015/報告書B\_完成版.tex）の「②の最大値時における①と②の樹脂圧力差 $\Delta P_2$」と、「樹脂温度は、溶融樹脂のフローフロントがセンサ位置②に到達した時点で最高値を示し」。学習データの列は「流動中樹脂圧力P3\_3」と「流動中樹脂温度」。
  \item 入力の範囲（樹脂圧力差 @@DP3@@ MPa、樹脂温度最高値 @@TFF3@@ ℃。バージンPP・リサイクルPP・バージンGPPS の順）：学習済みモデルの学習データ（5data。学習データの CSV の範囲と同じことを確かめた）。バージンPP の樹脂圧力差は、11-17 にご本人が直した値（1117修正\_結合\_03\_pp\_max\_5data\_外れ値なし.csv。厚み @@FIXH@@ mm の @@FIXN@@ 行）。
  \item ポスター（添削版）のバージンPP の図と決定係数・予測誤差率は、直した樹脂圧力差で学習し直したモデル（4回目\_01\_バージンPP再々実験\_学習結果。11-17）のもの。台本のバージンPP の数値も、このモデルの値。直す前の樹脂圧力差（上限 @@DPOLD@@ MPa。直した値の @@FIXR@@ 倍）で学習したモデル（3回目）からは、決定係数 @@R2PRE@@ → @@R2NEW@@、予測誤差率 @@EPRE@@ → @@ENEW@@\%、遅い側の誤差 @@ELPRE@@ → @@ELNEW@@\% に変わった。
  \item せん断速度と粘度の範囲：学習データ（5data）と検証データ（3data。除いた行を除く）を合わせた範囲（3 材料）。
  \item 低い温度 ÷ 高い温度の粘度の比（@@VR3@@ 倍）：各モデルのフォルダの シフトファクターの倍率.csv（温度シフトファクターの分）。バージンPP の学習結果のグラフの線は、温度ごとの樹脂温度最高値の違いの分も入って、@@GAP@@ 倍ずれている（質問対策の「本当に効いているのか」の補足）。
  \item 誤差（全体・速い側・遅い側）、予測幅、95\% 予測区間に入った割合：各モデルのフォルダの 予測検証\_3data\_指標.csv・予測検証\_3data\_予測値.csv。予測幅は、検証データの「95\% 予測区間の上の端 ÷ 予測値」の中央値（速い側 @@BH3@@\%、遅い側 @@BL3@@\% 上）。
  \item キャピラリーレオメータとの差（600〜2,500 1/s で 10\% 以内）：11-15 の報告ノート（3 材料のキャピラリーデータとの比べ）。
\end{itemize}
}
\end{document}
"""
    rep = {"@@DATE@@": rc.today_ja(), "@@CPM@@": str(CPM), "@@ALL@@": mmss(t_all).replace(":", " 分 ") + " 秒",
           "@@PCUE@@": f"{PAUSE_CUE:g}", "@@PPARA@@": f"{PAUSE_PARA:g}", "@@PSEC@@": f"{PAUSE_SECTION:g}", "@@PAUSE@@": str(r0(t_pause)),
           "@@SHORT@@": mmss(t_short).replace(":", " 分 ") + " 秒", "@@NALL@@": str(n_all), "@@NSHORT@@": str(n_short),
           "@@ROWS@@": "\n".join(rows), "@@BODY@@": "\n".join(body), "@@KW@@": kw, "@@MSG@@": msg, "@@WORDS@@": words,
           "@@TENSAKU@@": TENSAKU_PDF.replace("_", r"\_"),
           "@@DP3@@": "・".join(N[f"DP_{m}"] for m in MATS), "@@TFF3@@": "・".join(N[f"TFF_{m}"] for m in MATS),
           "@@VR3@@": "・".join(N[f"VR_{m}"] for m in MATS),
           "@@BH3@@": "・".join(str(N[f"BANDH_{m}"]) for m in MATS), "@@BL3@@": "・".join(str(N[f"BANDL_{m}"]) for m in MATS),
           "@@FIXH@@": N["P33FIX_H"], "@@FIXN@@": str(N["P33FIX_N"]), "@@DPOLD@@": N["DPOLD_PP"], "@@FIXR@@": N["P33FIX_RATIO"],
           "@@R2PRE@@": N["R2PRE_PP"], "@@R2NEW@@": N["R2_PP"], "@@EPRE@@": str(N["EPRE_PP"]), "@@ENEW@@": str(N["E_all_PP"]),
           "@@ELPRE@@": str(N["ELPRE_PP"]), "@@ELNEW@@": str(N["E_low_PP"]), "@@GAP@@": N["GAP_PP"]}
    for k, v in rep.items():
        tex = tex.replace(k, v)
    check("@@" not in tex, "台本の @@…@@ をすべて置き換えた")
    return tex, n_all, n_short, t_all, t_short


# ---------------------------------------------------------------- 質問対策
# (分類, 印, 質問, 答え, [補足])。印：◎ よく聞かれそう、★ 弱点（特に準備）。\kak{…} は要確認（ご自身の言葉・事実の確かめが要る所）
QA = [
    ("A. 研究の目的と位置づけ", "◎", r"レオメータで測れば十分では？ 金型の中で測る意味は？",
     r"実際の成形では、樹脂は冷えた金型の中を、温度が変わりながら短い時間で流れます。金型の中で測ると、その状態での見かけの粘度と、ショットごとのばらつきが得られます。キャピラリーレオメータは温度が一定の定常の流れなので、この効果は入りません。",
     [r"研究の背景（報告書B の「はじめに」）：CAE の樹脂粘度モデルは、樹脂のロットによる分子量分布のばらつきや、成形工程の熱履歴によるばらつきを考慮していない。",
      r"本研究の粘度は、せん断発熱や冷却などの非定常・非等温の効果を、あえて含めた「見かけの粘度」。補正（Rabinowitsch 補正など）はしていない。",
      r"3 材料のキャピラリーデータとの比べ（11-15 の調査）：同じせん断速度・同じ温度で、成形の見かけの粘度はキャピラリーの 0.33〜4.6 倍。600〜2,500 1/s あたりでは ±10\% で一致する（<<Q_c2>>）。"]),
    ("A. 研究の目的と位置づけ", "", r"予測の「ばらつき」が分かると、何に役立つのか？",
     r"流動解析の結果に、粘度のばらつきから来る幅を付けられるようになります\kak{想定している使い道は、ご自身の言葉で}。また、データの少ない条件ほど予測の幅が広くなるので、モデルをどこまで信用してよいかが分かります。",
     [r"学習結果のグラフでは、データのない低せん断速度側ほど帯が広い。学習の範囲の外の低射出率側では、検証データが 3 材料とも 100\% 予測区間に入った（<<Q_cov>>）。"]),
    ("A. 研究の目的と位置づけ", "◎", r"Cross-WLF 式などの従来の粘度式と比べて、何が違うのか？ 精度は？",
     r"従来の粘度式は式の形を決めて係数を当てはめますが、ガウス過程回帰は式の形を決めずにデータから関係を学び、予測の幅も出します。同じ学習データに Cross-WLF 式を当てはめて比べると、検証データ全体の誤差は、ガウス過程回帰のほうが小さくなりました。",
     [r"検証データ全体の誤差：ガウス過程回帰 PP <<E_all_PP>>・リサイクルPP <<E_all_MIX>>・GPPS <<E_all_GPPS>>\%、Cross-WLF <<CW_all_PP>>・<<CW_all_MIX>>・<<CW_all_GPPS>>\%。高射出率側は 3 材料ともガウス過程回帰がずっと小さい（<<E_HIGH_RANGE>>\% 対 <<CW_HIGH_RANGE>>\%）。",
      r"\textbf{★低射出率側だけでは、PP とリサイクルPP は Cross-WLF のほうが小さい}（PP <<CW_low_PP>> 対 <<E_low_PP>>\%、リサイクルPP <<CW_low_MIX>> 対 <<E_low_MIX>>\%）。GPPS は低射出率側もガウス過程回帰が小さい（<<E_low_GPPS>> 対 <<CW_low_GPPS>>\%）。",
      r"Cross-WLF は、学習データ（5data）に、せん断速度と樹脂温度の最高値で当てはめた参考の値（$D_3 = 0$、係数の一部は固定）。3 材料とも係数の一部が探索範囲の端で止まっており、よく当てはまった式とは言えない。"]),
    # ---- B
    ("B. 実験データとキャピラリーとの比較", "◎", r"圧力と温度を別々に測っているのに、組み合わせて使ってよいのか？",
     r"圧力と温度は、どちらも②の位置で測るため、固定側のキャビティブロックを交換し、同じ成形条件で別々に測って組み合わせました\kak{別々に測った理由と、組み合わせ方〔同じショット番号どうしか、など〕は、データを作ったご本人が答える}。",
     [r"同じ成形条件のショット間の粘度のばらつきは、PP・リサイクルPP で約 <<PURE_PCT>>\%（粘度の対数の標準偏差 <<PURE_SD>>。本体の検討の 5data）。"]),
    ("B. 実験データとキャピラリーとの比較", "", r"データの数は？ 1 条件で何ショット使ったか？",
     r"学習には 1 条件 5 ショット、検証には 1 条件 3 ショットを使いました。外れ値などを除いた後で、学習は PP <<NTR_PP>>・リサイクルPP <<NTR_MIX>>・GPPS <<NTR_GPPS>> ショット、検証は <<N_all_PP>>・<<N_all_MIX>>・<<N_all_GPPS>> ショットです。",
     [r"条件の数：学習 <<NTRC_PP>>・<<NTRC_MIX>>・<<NTRC_GPPS>>、検証 <<NC_all_PP>>・<<NC_all_MIX>>・<<NC_all_GPPS>> 条件。",
      r"学習は射出率 5.3〜127 cm$^3$/s。検証は低射出率側 1.6・2.7 cm$^3$/s（厚み 1.5・2.0 mm だけ）と、高射出率側 149・170 cm$^3$/s。"]),
    ("B. 実験データとキャピラリーとの比較", "◎", r"計測データに外れ値があった場合、それも含めて学習したのか？ 除いたなら、どんな基準で除いたのか？",   # 2026-11-17 の 6 回目：ユーザーの質問に合わせて書き直した
     r"外れ値は含めずに学習しました。外れ値は 3 つの段階で除いています。1 つ目は、データを作る段階で、計測値のうち中央値から 20\% 以上離れた値（樹脂温度は、修正 z スコアが 3.5 を超える値）を、ショットごとに除きました。2 つ目は、ほかの条件からの予測と大きく違う成形条件を、条件ごとに除きました。3 つ目は、計測ミスで値が使えない条件とショットを除きました。",
     [r"1 つ目（ショットごと）：除く前 → 除いた後のショット数は、<<OUT_ROWS>>。圧力などの列と樹脂温度の列で別々に判定し、両方に残ったショットだけを使った。修正 z スコアは $0.6745\,|x - \tilde{x}|/\mathrm{MAD}$（$\tilde{x}$ は中央値、MAD は中央値からの絶対偏差の中央値）。\kak{判定の単位（成形条件ごとか）と、GPPS でも同じ基準で除いたか〔GPPS の学習データは <<NTRC_GPPS>> 条件 × 5 = <<GPPS_FULL>> ショットのところ <<NTR_GPPS>>〕は、データを作ったご本人が答える}",
      r"2 つ目（条件ごと）：学習と検証の全条件の平均を、その条件を抜いて学習したガウス過程回帰（入力はせん断速度・厚み・加熱シリンダ温度）で予測し、ずれが予測の標準偏差の 3.5 倍を超え、かつ粘度で約 10\% を超える条件を外れ値とした。拾ったのは、PP の 190 ℃・0.5 mm・170 cm$^3$/s の 1 条件だけ（検証データ。<<RF_PP>>）。リサイクルPP と GPPS では拾っていない。",
      r"3 つ目（計測ミス）：PP の 1.5 mm・1.6 cm$^3$/s（3 温度。圧力の最大値を検出できなかった。検証データのファイルの段階で除いてある）と、GPPS の 200 ℃・1.5 mm・1.6 cm$^3$/s の 2 ショット（樹脂圧力差が負で、対数がとれない）。どちらも検証データ。",
      r"そのため、学習データ（5data）で除いたのは 1 つ目だけ。2・3 つ目で検証データから除いた行は、PP <<EX3_PP>>、GPPS <<EX3_GPPS>> 行。除いた条件とその理由は、問題データ一覧（05\_材料ごとの機械学習データと予測検証データ/問題データ一覧.csv）に記録した。",
      r"★中央値から 20\% 以上離れた値を除いているので、同じ条件のショット間のばらつきは小さめに出る。高射出率側で 95\% 区間に入った割合が 95\% より低い一因かもしれない（<<Q_cov>>）\kak{考察}。"]),
    # ---- C
    ("B. 実験データとキャピラリーとの比較", "", r"粘度とせん断速度の算出式は？ 補正はしていないのか？",
     r"スリット流れのニュートン流体の式で、$\eta = \Delta P B H^3/(12 L Q)$、$\dot{\gamma} = 6Q/(B H^2)$ です。流量 $Q$ は、①と②への到達の時間差 $\Delta t$ から求めた流速 $V = L/\Delta t$ に、断面積を掛けたものです。Rabinowitsch 補正などはせず、せん断発熱や冷却の効果を含めた「見かけの粘度」として扱っています。",
     [r"この $\Delta P$ は、②に樹脂が届いた時刻の①と②の圧力差（$\Delta P_1$）。入力に使う樹脂圧力差（$\Delta P_2$）とは別（<<Q_leak>>）。"]),
    ("B. 実験データとキャピラリーとの比較", "◎", r"入力の「樹脂圧力差」と「樹脂温度最高値」は、どの値を使ったのか？",   # 2026-11-17 の 3 回目
     r"樹脂圧力差は、②の圧力が最大値になった時刻の、①と②の樹脂圧力の差（$\Delta P_2$）です。樹脂温度最高値は、樹脂の先端が②に届いた時点で、②の温度センサが示す最高値です。どちらも、1 回の成形ごとの値を、そのまま入力にしました。",
     [r"学習データ（5data）の範囲：樹脂圧力差 PP <<DP_PP>>、リサイクルPP <<DP_MIX>>、GPPS <<DP_GPPS>> MPa。樹脂温度最高値 <<TFF_PP>>、<<TFF_MIX>>、<<TFF_GPPS>> ℃。PP の樹脂圧力差は、11-17 に直した値（厚み <<P33FIX_H>> mm の <<P33FIX_N>> 行）。",
      r"粘度の計算に使う圧力差（②に樹脂が届いた時刻の①と②の差 $\Delta P_1$）とは、別の時刻の値（<<Q_leak>>）。",
      r"樹脂温度は、樹脂の先端が②に届いた時点で最高値を示し、その後ゆるやかに下がる（報告書B）。ポスターの温度のグラフの「入力データに使用」の矢印の点。",
      r"定義は報告書B の文による。学習データの列は「流動中樹脂圧力P3\_3」と「流動中樹脂温度」。",
      r"ポスター（添削版）の PP のモデルは、直した樹脂圧力差で学習し直したもの（4回目、11-17）。直す前は、厚み <<P33FIX_H>> mm で直した値の <<P33FIX_RATIO>> 倍（上限 <<DPOLD_PP>> MPa）だった。学習し直して、決定係数は <<R2PRE_PP>> → <<R2_PP>>、誤差は <<EPRE_PP>> → <<E_all_PP>>\%、縦の係数は <<EVPRE_PP>> → <<EV_PP>> kJ/mol になった（<<Q_e3>>）。"]),
    ("B. 実験データとキャピラリーとの比較", "★", r"ポスターの「キャピラリーレオメータの計測データと概ね一致する」は、どのくらい合っているのか？",
     r"両対数のグラフで見て、中くらいから速いせん断速度の側で重なる、という意味です。数値では、成形とキャピラリーの曲線はせん断速度 600〜2,500 1/s あたりで交わり（そこでは ±10\% 以内）、それより遅い側では成形のほうが粘度が高く、速い側では低くなります。",
     [r"3 材料のキャピラリーデータとの比べ（11-15 の調査。同じ温度、厚み 2.0 mm）：成形 ÷ キャピラリーは、60.8 1/s で 4.1〜4.6 倍、2,430 1/s で 0.7〜0.8 倍。厚みでも違う。",
      r"キャピラリーの値も、補正なしの見かけの値（直径 1 mm、$L/D$ = 40。Bagley 補正・Rabinowitsch 補正なし）。",
      r"キャピラリーデータは、測っている量が違い、モデルの入力（樹脂圧力差・樹脂温度）もないので、検証データ（3data）のような予測精度の検証には使えない（11-15 の調査）。"]),
    # ---- D
    ("C. 機械学習モデル", "◎", r"なぜガウス過程回帰を選んだのか？ ニューラルネットワークではだめか？",
     r"予測値と同時に、予測の幅（不確かさ）を出せるからです。また、学習データが 1 材料あたり数百ショットと少なくても使えます。ニューラルネットワークは、そのままでは予測の幅を出せず、多くのデータが要ります。",
     [r"学習データ：PP <<NTR_PP>>、リサイクルPP <<NTR_MIX>>、GPPS <<NTR_GPPS>> ショット。"]),
    ("C. 機械学習モデル", "◎", r"目的変数と説明変数がそろってから、学習が終わるまで、どんな流れで計算したのか？",
     r"まず、粘度と入力の対数をとり、学習データだけで標準化の定数を決めます。次に、温度シフトの係数を 1 つ決めてデータを基準温度にそろえ、ガウス過程を学習して尤度を記録する、という計算を、係数を変えながら繰り返します。最後に、尤度が最大の係数で、初期値を変えて念入りに学習し直したものを、最終のモデルにしました。",
     [r"(1) データ：学習データ（5data）の、ショットごとの値をそのまま使う（PP <<NTR_PP>>・リサイクルPP <<NTR_MIX>>・GPPS <<NTR_GPPS>> 行）。問題のある条件と、対数がとれない行は除く。目的変数は $\ln\eta$、入力は $\ln\dot{\gamma}$、$\ln\Delta P_2$（Pa に直して）、$\ln T$（樹脂温度、K）。加熱シリンダ温度は入力に入れず、温度シフトファクターにだけ使う（<<Q_tset>>）。",
      r"(2) 標準化：入力（ずらす前）の平均と標準偏差、$\ln\eta$ の平均を、学習データだけで計算する。係数によらない定数にして、係数を変えた学習どうしで尤度を比べられるようにした（出力の自動標準化は使わない）。",
      r"(3) 係数を 1 つ決めて、データをずらす：$y' = \ln\eta - \ln a_v(T_\mathrm{set}) -$（$\ln\eta$ の平均）。GPPS は、入力の $\ln\dot{\gamma}$ にも $\ln a_h(T_\mathrm{set})$ を足す。ずらしたデータでガウス過程を学習し（ハイパーパラメータを最尤推定。<<Q_hyp>>）、そのときの対数周辺尤度を記録する（<<Q_when>>）。",
      r"(4) (3) を、係数の格子で繰り返す（プロファイル尤度。<<Q_grid>>）：5 kJ/mol 刻みの粗い格子（縦 $-20$〜60、横 $-20$〜40）→ 最良の点の周りの細かい格子（縦だけは 0.5、縦と横は 1 kJ/mol 刻み）→ 全点の見直し → 95\% 区間の端まで点を足す。学習した回数は PP <<NPTS_PP>>、リサイクルPP <<NPTS_MIX>>、GPPS <<NPTS_GPPS>> 回。",
      r"(5) 尤度が最大の係数で、初期値を変えて 21 回、念入りに学習し直す（係数 0 でも同じ）。前者が最終のモデル。2 つの尤度の差で、温度シフトが支持されるかを判定し、最大から 1.92 以内を係数の 95\% 区間とした。",
      r"(6) 最終のモデル（係数、標準化の定数、ガウス過程）を保存し、検証データ（3data）を予測して、誤差と 95\% 区間に入った割合を出した（<<Q_predcalc>>）。"]),
    ("C. 機械学習モデル", "★◎", r"粘度は圧力差から計算しているのに、入力にも樹脂圧力差を入れている。答えを入力に入れていないか？",
     r"粘度の計算に使う圧力差（②に樹脂が届いた時刻の差 $\Delta P_1$）と、入力の樹脂圧力差（②の圧力が最大のときの差 $\Delta P_2$）は、別の時刻の量です。確かめとして、入力の樹脂圧力差で粘度の算出式の形を作ると、誤差が最大 <<P33ERR_RANGE>>\% と大きく、算出式の代わりにはなりません。また、粘度を樹脂圧力差とせん断速度で直線回帰しても、検証の決定係数は、せん断速度だけの場合とほぼ同じです。",
     [r"検証の決定係数（粘度の対数の直線回帰。樹脂圧力差＋せん断速度／せん断速度だけ）：PP <<P33R2_PP>>／<<GDR2_PP>>、リサイクルPP <<P33R2_MIX>>／<<GDR2_MIX>>、GPPS <<P33R2_GPPS>>／<<GDR2_GPPS>>。",
      r"参考：算出式そのものの圧力差（$\Delta P_1$）を入れると、厚みも入れた直線回帰だけで決定係数 0.9999 になり、モデルは算出式を覚えるだけになる（本体の検討）。",
      r"★ただし、樹脂圧力差も流れの結果として決まる量で、成形の前には分からない（<<Q_pred>>）。"]),
    ("C. 機械学習モデル", "", r"キャビティの厚みを、モデルの入力に入れないのか？",
     r"厚みは入力に入れていませんが、入力の樹脂圧力差が厚みでほぼ決まる量なので、厚みの情報は樹脂圧力差を通してモデルに入っています。",
     [r"同じせん断速度・同じ温度で、0.5 mm の見かけの粘度は 2.0 mm の約 2.3 倍（PP。本体の検討）。厚みの影響は大きい。",
      r"樹脂圧力差は、②が最大の時刻の①の圧力とほぼ比例する（対数の相関 0.95 以上）。本体の検討で、その圧力は厚みとせん断速度でほぼ決まる（決定係数 0.92）。",
      r"\kak{厚みを入力に入れなかった理由（ご自身の決定）は、ご自身の言葉で}"]),
    ("C. 機械学習モデル", "★◎", r"予測するとき、樹脂圧力差と樹脂温度はどう与えるのか？ 成形の前には分からないのでは？",
     r"その通りで、今のモデルは「計測した樹脂圧力差と樹脂温度が分かっているときの粘度」を出すモデルです。成形条件だけから予測するには、この 2 つも求める必要があります。流動解析の中では圧力と温度も計算されるので、解析と組み合わせて繰り返し計算する方法を、今後検討します\kak{今後の方針の言い方は、ご自身で}。",
     [r"加熱シリンダ温度は成形の前に決まる設定値。せん断速度も、射出率と厚みからおおよそ決まる。",
      r"参考（ポスターにない別の検討。言うかはご自身で判断）：樹脂圧力差と樹脂温度を使わず、せん断速度（同じ条件の平均）・厚み・加熱シリンダ温度だけを入力にしたモデル（本体の手順5 の B\_1）では、同じ検証データで、誤差が PP <<B1_PP_all>>\%（低射出率側 <<B1_PP_low>>\%）、リサイクルPP <<B1_MIX_all>>\%（<<B1_MIX_low>>\%）だった。"]),
    ("C. 機械学習モデル", "◎", r"カーネル関数は何を使い、どう定義したのか？",
     r"3 つのカーネルの和です。入力ごとに長さスケールを持つ ARD 付きの Matern 5/2 カーネルで曲がった関係を、線形カーネルで全体の直線的な傾きを、ホワイトノイズでショット間のばらつきを表します。前報（6月）のモデルの Matern 5/2 ＋ ホワイトノイズに、線形カーネルを足した形です。",
     [r"式：$k(\mathbf{x},\mathbf{x}') = \sigma_f^2\,k_{5/2}(r) + c\,(\sigma_0^2 + \mathbf{x}\cdot\mathbf{x}') + \sigma_n^2\,\delta_{\mathbf{x}\mathbf{x}'}$。$k_{5/2}(r) = (1 + \sqrt{5}\,r + \tfrac{5}{3}r^2)\,e^{-\sqrt{5}\,r}$、$r = \sqrt{\textstyle\sum_i (x_i - x_i')^2/\ell_i^2}$（$\ell_i$ は入力 $i$ の長さスケール）。$\delta$ は同じ点なら 1、違えば 0。",
      r"入力 $\mathbf{x}$ は、標準化した 3 つ（$\ln\dot{\gamma}$〔GPPS は温度で横にずらした値〕、$\ln\Delta P_2$、$\ln T$）。出力は、温度で縦にずらして平均を引いた $\ln\eta$（<<Q_flow>>）。平均関数は 0 で、全体の傾きは線形カーネルが受け持つ（<<Q_lin>>）。",
      r"探索範囲（初期値）：$\sigma_f^2$ と $c$ は $10^{-3}$〜$10^{3}$（1）、$\ell_i$ は $10^{-2}$〜$10^{2}$（1）、$\sigma_0$ は $10^{-5}$〜$10^{3}$（1）、$\sigma_n^2$ は $10^{-6}$〜10（0.1）。3 材料とも、長さスケールとノイズは範囲の端に張り付いていない。学習した値は「カーネルと長さスケールの早見表」。",
      r"Matern 5/2 は、関数が 2 回微分できる程度の滑らかさを仮定する。よく使われる RBF（ガウス）カーネルは無限回微分できる滑らかさを仮定するので、実データの急な曲がりには Matern のほうが合わせやすい\kak{前報で Matern 5/2 を選んだ理由は、ご自身の言葉で}。",
      r"線形カーネルは ARD ではない（全入力で共通の形。傾きそのものは、入力ごとにデータで決まる）。",
      r"scikit-learn の表記：ConstantKernel × Matern(nu=2.5) ＋ ConstantKernel × DotProduct ＋ WhiteKernel（GaussianProcessRegressor で学習）。"]),
    ("C. 機械学習モデル", "", r"ハイパーパラメータは、どう決めたのか？",
     r"学習データの対数周辺尤度が最大になる値を、勾配法（L-BFGS-B）で探しました。局所解に止まらないように、初期値を変えて 21 回探し、最も尤度の大きい解を採りました。乱数シードは 42 に固定しています。",
     [r"対数周辺尤度：$\log p(\mathbf{y}\mid X) = -\tfrac12\mathbf{y}^{\top}K^{-1}\mathbf{y} - \tfrac12\log|K| - \tfrac{n}{2}\log 2\pi$（$K$ はカーネル行列。ノイズを含む）。第 1 項が当てはまりの良さ、第 2 項が複雑さの罰で、その釣り合いで決まる（<<Q_overfit>>）。",
      r"21 回の内訳：1 回は、係数の格子で最もよかった点の解から。20 回は、探索範囲の中の乱数の初期値から（scikit-learn の n\_restarts\_optimizer = 20）。",
      r"温度シフトの係数は、この最適化には入れず、外側で格子を使って決めた（<<Q_grid>>）。",
      r"ハイパーパラメータは 1 点の推定値で、その不確かさ（事後分布）は予測の幅に入れていない（<<Q_unc>>）。"]),
    ("C. 機械学習モデル", "◎", r"ARD とは何か？ 各材料の長さスケールは、どんな結果になったか？",
     r"ARD は、入力ごとに別々の長さスケールを持たせ、データから決める仕組みです。長さスケールは「その入力をどのくらい動かすと、関数の形が変わるか」の目安で、短いほど細かく変わります。3 材料とも、せん断速度が <<LS_RANGE>> と最も短く、樹脂圧力差が <<LSP_RANGE>>、樹脂温度が <<LST_RANGE>> と長くなりました（標準化した入力での値）。",
     [r"材料ごと（せん断速度／樹脂圧力差／樹脂温度）：PP <<LS3_PP>>、リサイクルPP <<LS3_MIX>>、GPPS <<LS3_GPPS>>。GPPS のせん断速度は、温度で横にずらした値。",
      r"元の単位に直すと：せん断速度は約 <<WG_RANGE>> 倍、樹脂圧力差は <<WP_RANGE>> 倍、樹脂温度は <<WT_RANGE>> ℃ 動かすと、関数の形が変わりうる、という長さ。",
      r"学習データの範囲は、長さスケールの何個分か：せん断速度 <<NLSG_RANGE>> 個分（範囲の中で何度も曲がれる）、樹脂圧力差 <<NLSP_RANGE>> 個分、樹脂温度 <<NLST_RANGE>> 個分（範囲の中では、Matern の部分はほぼ一定）。",
      r"各モデルのフォルダの 特徴スケール.png は、1/長さスケールを「重要度」として描いた図（せん断速度 <<INVG_RANGE>>、樹脂圧力差 <<INVP_RANGE>>、樹脂温度 <<INVT_RANGE>>）。ただし、これは曲がりの細かさで、効きの大きさそのものではない（<<Q_imp>>）。",
      r"前報（6月）のモデルでは、樹脂圧力の重要度（1/長さスケール）が突出していた。粘度の算出式の圧力を入力に入れていたため（<<Q_leak>>）。"]),
    ("C. 機械学習モデル", "★", r"長さスケールから、どの入力が重要と言えるか？ 樹脂温度は効いていないのか？",
     r"長さスケールは曲がりの細かさで、効きの大きさそのものではありません。直線的な効きは、全入力に共通の線形カーネルが受け持つからです。そこで予測の傾きで比べると、入力を 1 標準偏差動かしたときの粘度の対数の変化は、せん断速度で <<SLG_RANGE>> 下がり、樹脂圧力差で <<SLP_RANGE>> 上がり、樹脂温度では <<SLT_MAX>> 以下でした。せん断速度が圧倒的に大きく、樹脂温度の効きは小さいです。",
     [r"材料ごとの傾き（せん断速度／樹脂圧力差／樹脂温度。標準化した入力 1 あたりの $\ln\eta$。学習データの点での平均）：PP <<SL3_PP>>、リサイクルPP <<SL3_MIX>>、GPPS <<SL3_GPPS>>。",
      r"1 標準偏差の大きさ：せん断速度 約 <<SDG_RANGE>> 倍、樹脂温度 約 <<SDT_RANGE>> ℃。せん断速度を 1 標準偏差上げると、粘度は約 <<SLG_FAC>> 分の 1。樹脂温度を 1 標準偏差上げても、粘度の変化は約 <<SLT_PCT>>\% 以下。",
      r"★ほかの入力を固定して、1 つだけ動かしたときの傾き。実際には入力どうしが関係している（樹脂温度は射出速度とともに上がり、樹脂圧力差は厚みでほぼ決まる）ので、樹脂温度の効果の一部は、せん断速度や樹脂圧力差を通して入っているかもしれない\kak{考察}。",
      r"温度の効果は、加熱シリンダ温度の縦のずれとしても入っている（低い温度と高い温度の粘度の比 PP <<VR_PP>>、リサイクルPP <<VR_MIX>>、GPPS <<VR_GPPS>> 倍。<<Q_tset>>）。",
      r"★PP（直した樹脂圧力差で学習し直したもの）の樹脂温度の傾きは <<SLT_PP>> で、直す前のモデル（<<SLTPRE_PP>>）やほかの 2 材料（リサイクルPP・GPPS <<SLT_OTHER>>）の約 2 倍以上。PP では温度シフトが支持されなくなった（<<Q_e3>>）ことと合わせると、温度の効果を、温度シフトファクターの代わりに樹脂温度最高値の入力が受け持つようになったと考えられる\kak{考察}。",
      r"樹脂温度を入力から外したモデルとの比べは、ポスターの学習結果ではしていない\kak{「Tff の入力の有無の比較」のノートを実行したか}。"]),
    ("C. 機械学習モデル", "", r"線形カーネルは、何のために足したのか？ どのくらい効いているか？",
     r"学習の範囲の外で、予測が平均値に戻ってしまうのを防ぐためです。粘度はせん断速度とともに大きく下がるので、その全体の傾きを線形カーネルが受け持ち、Matern カーネルは直線からの曲がりを受け持っています。線形カーネルを外すと、検証データの誤差（対数の RMSE）は、全体で <<NOLIN_ALL_RANGE>> 倍、低射出率側で <<NOLIN_LOW_RANGE>> 倍になりました。",
     [r"学習データの点での、予測の平均の大きさ（標準偏差。粘度の対数）：線形カーネルの部分 <<PL_RANGE>>、Matern の部分 <<PM_RANGE>>。",
      r"高射出率側（学習の範囲のすぐ外）は、線形カーネルの有無で大きくは変わらない（<<NOLIN_HIGH_RANGE>> 倍）。違いは、範囲から遠い低射出率側で出る。",
      r"線形カーネルなしの結果：リサイクルPP・GPPS は 08\_カーネル関数の違いによる結果の違い/ の比較表（11-11 の実行）、PP は 4回目\_…\_学習結果\_線形項なし/（11-17 の実行）。どちらも、同じデータ・同じ設定で、線形カーネルだけを外した。",
      r"線形カーネル $c(\sigma_0^2 + \mathbf{x}\cdot\mathbf{x}')$ は、傾きに正規分布の事前分布を置いた線形回帰と同じ。"]),
    ("C. 機械学習モデル", "★", r"学習結果のグラフで、予測曲線が細かく波打っているのはなぜ？",
     r"入力に、ショットごとの計測値をそのまま使ったため、モデルがショットごとの小さなゆらぎにも合わせようとして、曲線が波打っています。また、グラフの線は、樹脂圧力差と樹脂温度をある値に固定して描いた断面なので、点と線は少しずれます。",
     [r"せん断速度の長さスケール（標準化した入力で）は <<LS_RANGE>> と短い（せん断速度が約 <<WG_RANGE>> 倍変わると、曲がりうる長さ。<<Q_ard>>）。同じ条件の平均を入力にすると、曲線は滑らかになる（長さスケール 約 4〜6。10-23 までの検討）。",
      r"\kak{ショットごとの入力に決めた理由（10-23）は、ご自身の言葉で}"]),
    ("C. 機械学習モデル", "★", r"過学習していないか？",
     r"ハイパーパラメータは対数周辺尤度で決めていて、これは当てはまりの良さと、モデルの複雑さの罰を合わせた量なので、必要以上に複雑なモデルは選ばれにくくなっています。また、学習に使っていない射出率の条件（3data）で、精度を確かめました。ただし、ショットごとの値を入力にしたため、ショットごとの小さなゆらぎにも合わせている面はあります（<<Q_wavy>>）。",
     [r"学習データでの残差（粘度の対数の標準偏差）は <<RES_RANGE>> で、ノイズの標準偏差 <<NOISE_RANGE>> と同じくらい。検証データの誤差は、高射出率側（範囲のすぐ外）<<E_HIGH_RANGE>>\%、低射出率側（範囲から遠い外挿）<<E_LOW_RANGE>>\%（<<Q_err>>）。",
      r"高射出率側で 95\% 区間に入った割合は <<COV_HIGH_RANGE>>\% で、区間がやや狭い。ノイズが小さく見積もられた影響と考えている（<<Q_cov>>）。",
      r"同じ条件のショット間では、粘度とせん断速度が同じ到達時間差から計算されるので、連動して動く。ショットごとのせん断速度を入力にすると、そのゆれもモデルが説明し、ノイズが小さく見積もられる（本体の検討）。"]),
    ("C. 機械学習モデル", "", r"新しい条件の粘度は、どう計算して予測するのか？",
     r"新しい条件のせん断速度・樹脂圧力差・樹脂温度を、学習と同じ係数と定数でずらして標準化し、ガウス過程で予測の平均と標準偏差を出します。平均に、加熱シリンダ温度の縦のずれ $\ln a_v$ と、学習データの粘度の対数の平均を足し戻し、平均 ±1.96 標準偏差を 95\% 予測区間とします。最後に指数をとって粘度に戻します。",
     [r"ガウス過程の予測：平均 $\mu_* = \mathbf{k}_*^{\top}(K + \sigma_n^2 I)^{-1}\mathbf{y}'$、分散 $\sigma_*^2 = k(\mathbf{x}_*,\mathbf{x}_*) - \mathbf{k}_*^{\top}(K + \sigma_n^2 I)^{-1}\mathbf{k}_* + \sigma_n^2$（$K$ は学習データの点どうしのカーネル〔ノイズを除く〕、$\mathbf{k}_*$ は新しい点と学習データの点のカーネル）。最後の $\sigma_n^2$ がショット間のばらつき、残りがモデルの不確かさ。",
      r"区間は粘度の対数で上下対称なので、粘度に戻すと上下非対称（上側が広い）。",
      r"予測に使う情報：学習データの全点（ガウス過程は学習データを持ち歩く）、ハイパーパラメータ、係数、標準化の定数。保存した学習済みモデル（.joblib）にすべて入っている。"]),
    ("C. 機械学習モデル", "◎", r"学習後のモデルには温度シフトファクターが入っているので、モデル（平均関数とその予測幅）だけで、温度シフトを考えた粘度の予測ができるのか？",   # 2026-11-17 の 6 回目（ユーザーの質問）
     r"はい、できます。保存した学習済みモデルには、温度シフトの係数も入っています。新しい条件のせん断速度・樹脂圧力差・樹脂温度最高値と加熱シリンダ温度を与えると、モデルが加熱シリンダ温度から温度シフトファクターを計算し、ガウス過程の予測の平均に足して、粘度の予測値と 95\% 予測区間を出します。縦のずれは平均に足すだけなので、予測幅は、対数の目盛りで同じ幅のまま上下に動きます。",
     [r"ただし、樹脂圧力差と樹脂温度最高値は、計測した値を与える必要がある（<<Q_pred>>）。加熱シリンダ温度だけを変えて、樹脂温度最高値をそのままにすると、温度の効果の一部しか入らない。特に PP では、温度による粘度の違いの大部分を、樹脂温度最高値の入力が受け持っている（<<Q_ho>>）。",
      r"GPPS は横のずれもあるので、せん断速度の入力を加熱シリンダ温度でずらしてから、ガウス過程に入れる。この場合は、予測幅も加熱シリンダ温度で少し変わる。",
      r"★縦だけの PP・リサイクルPP では、加熱シリンダ温度はガウス過程の入力に入らない。そのため、学習にない加熱シリンダ温度（190〜230 ℃ の外）でも計算はできるが、アレニウスの式をそのまま延ばした値になり、温度の外挿の不確かさは予測幅に入らない（樹脂温度最高値などのほかの入力が学習の範囲の中なら、予測幅は広がらない）。係数の不確かさも入っていない（<<Q_unc>>）。",
      r"「平均関数」は式 1 本ではなく、学習データの全点とハイパーパラメータから計算する（ガウス過程は学習データを持ち歩く）。必要なものは、すべて学習済みモデル（.joblib）に入っている（<<Q_predcalc>>）。"]),
    ("C. 機械学習モデル", "", r"95\% 予測区間には、何が含まれるのか？",
     r"2 つの不確かさの合計です。1 つはデータの少ない所でのモデルの不確かさ、もう 1 つは同じ条件で成形を繰り返したときのショット間のばらつき（ノイズ）です。データのない低せん断速度側では前者が大きくなり、帯が広がります。",
     [r"温度シフトの係数とハイパーパラメータの不確かさは入っていない（<<Q_unc>>）。"]),
    # ---- E
    ("D. 温度シフトファクター", "◎", r"温度シフトファクターとは？ 式は？",
     r"温度が変わると粘度曲線が縦や横にずれる、そのずれの量を温度の式で表したものです。加熱シリンダ温度からアレニウス式で計算し、粘度の縦方向のずれ（GPPS は横方向も）としてモデルに組み込みました。係数は、学習データの尤度が最大になる値を探して決めました（係数 $E$ の意味は <<Q_ewhat>>、求め方は <<Q_ecalc>>。組み込む段階は <<Q_when>>）。",
     [r"式：$\ln a_T = (E/R)(1/T - 1/T_\mathrm{ref})$（$T$ は加熱シリンダ温度の絶対温度、$R$ = 8.314 J/(mol·K)）。基準温度 $T_\mathrm{ref}$ は PP・リサイクルPP <<TREF_PP>> ℃、GPPS <<TREF_GPPS>> ℃（3 水準の真ん中）。基準温度では $a_T = 1$。",
      r"モデル：$\ln\eta = \ln a_v(T_\mathrm{set}) + f(\ln\dot{\gamma} + \ln a_h(T_\mathrm{set}),\ \ln\Delta P_2,\ \ln T)$。$f$ がガウス過程。縦の係数 $E_v$ と横の係数 $E_h$ は別々。PP・リサイクルPP は縦だけ（$a_h = 1$。<<Q_vh>>）。",
      r"係数 0（ずれなし）より尤度が十分大きければ、温度シフトがデータで支持されたとする（目安：縦だけ 1.92、縦と横 3.0）。"]),
    ("D. 温度シフトファクター", "◎", r"温度シフトファクターの係数 E（$E_v$・$E_h$）とは何か？ 何を表すのか？",   # 2026-11-17 の 8 回目（ユーザーの依頼：E そのものとは何か）
     r"アレニウス式 $\ln a_T = (E/R)(1/T - 1/T_\mathrm{ref})$ の中の係数で、加熱シリンダ温度が変わったときに、粘度の曲線をどれだけずらすかを決める数です（単位 kJ/mol）。$E$ が大きいほど温度で粘度が大きく変わり、$E = 0$ ならずらしません。縦の係数 $E_v$ は粘度の向きのずれ、横の係数 $E_h$ はせん断速度の向きのずれを決めます。例えばリサイクルPP は $E_v$ = <<EV_MIX>> kJ/mol で、<<TLOW_MIX>> ℃ の粘度は <<THIGH_MIX>> ℃ の <<VR_MIX>> 倍になります。",
     [r"値（縦の係数 $E_v$。（ ）は 95\% 区間）：PP <<EV_PP>>（<<EVLO_PP>>〜<<EVHI_PP>>）、リサイクルPP <<EV_MIX>>（<<EVLO_MIX>>〜<<EVHI_MIX>>）、GPPS <<EV_GPPS>>（<<EVLO_GPPS>>〜<<EVHI_GPPS>>）kJ/mol。横の係数 $E_h$ は GPPS だけで <<EH_GPPS>> kJ/mol<<EHCI_GPPS>>（<<Q_vh>>）。\textbf{PP の区間は 0 を含み、温度シフトは支持されない}（<<Q_e3>>）。",
      r"式の読み方：$\ln a_T$ は、温度 $T$ のデータを基準温度 $T_\mathrm{ref}$（PP・リサイクルPP <<TREF_PP>> ℃、GPPS <<TREF_GPPS>> ℃）にそろえるための、粘度の対数のずれ。$T$ は絶対温度、$R$ = 8.314 J/(mol·K)（気体定数）。$E/R$ が、温度の逆数 $1/T$ に対する $\ln a_T$ の傾きで、基準温度ではずれが 0。",
      r"低い温度と高い温度の粘度の比（縦の倍率）：PP <<VR_PP>> 倍（<<VRT_PP>>）、リサイクルPP <<VR_MIX>> 倍（<<VRT_MIX>>）、GPPS <<VR_GPPS>> 倍（<<VRT_GPPS>>）。$E_v$ が 0 なら 1 倍。",
      r"式の形は、化学反応や流動の「活性化エネルギー」と同じだが、ここでの $E$ は、このモデルの見かけの粘度から決めた「見かけのシフトの係数」で、物性値の活性化エネルギーそのものではない（文献の PP の流動の活性化エネルギーは 40 kJ/mol 前後。<<Q_e3>>）。入力の樹脂温度最高値も温度の効果を受け持つので、$E$ は加熱シリンダ温度の残りの効果を表すと考えられる\kak{考察}。",
      r"$E$ は、データから決めた値（<<Q_ecalc>>）。式の形（アレニウス型、ずらす向き、基準温度）は学習の前に決めた（<<Q_form>>）。"]),
    ("D. 温度シフトファクター", "◎", r"係数 E は、どうやって求めたのか？",   # 2026-11-17 の 8 回目（ユーザーの依頼：E の算出方法。今の格子の方法）
     r"格子を使って求めました（プロファイル尤度）。$E$ の値を少しずつ変えながら、そのたびに、データを $E$ でずらしてガウス過程を学習し、学習データへの当てはまりの良さ（対数周辺尤度）を計算します。尤度が最も大きい $E$ を採り、尤度が最大から 1.92 以内の範囲を、$E$ の 95\% 区間としました。最後に、$E = 0$（ずらさない）と尤度を比べて、温度シフトが支持されるかを判定しました。",
     [r"(1) 粗い格子：$E$ を 5 kJ/mol 刻みで、縦 $-20$〜60、横 $-20$〜40 kJ/mol の範囲で変える（縦だけの PP・リサイクルPP は <<NCOARSE_PP>> 点、縦と横の GPPS は <<NCOARSE_GPPS>> 点）。",
      r"(2) 各点の学習：その $E$ でデータをずらし（<<Q_when>>）、カーネルのハイパーパラメータを勾配法（L-BFGS-B）で最尤推定して、対数周辺尤度を記録する（<<Q_hyp>>）。",
      r"(3) 細かい格子：最良の点の周りを、縦だけは 0.5、縦と横は 1 kJ/mol 刻みで計算し直す。最良の点が動けば、そこを中心に取り直す。",
      r"(4) 見直しと延長：すべての点を、最良の点の解から学習し直し（局所解に止まった点を直す）、95\% 区間の端が決まるまで、外側に点を足す。学習した回数は PP <<NPTS_PP>>、リサイクルPP <<NPTS_MIX>>、GPPS <<NPTS_GPPS>> 回。",
      r"(5) 念入りな学習：最良の $E$ と $E = 0$ の 2 点を、初期値を変えて 21 回ずつ学習し直し、その尤度の差（$\Delta$LML）で支持を判定する。基準は縦だけ 1.92、縦と横 3.0（尤度比検定の 5\% 基準）。PP <<DLML_PP>>、リサイクルPP <<DLML_MIX>>、GPPS <<DLML_GPPS>>。",
      r"図と表：各モデルのフォルダの プロファイル尤度.png（横軸 $E$、縦軸 尤度 − 最大値。赤の破線が $-1.92$。GPPS は $E_h$–$E_v$ の面）と プロファイル尤度\_格子.csv。",
      r"ハイパーパラメータと一緒に最適化しなかった理由は <<Q_grid>>、学習の流れの全体は <<Q_flow>>。"]),
    ("D. 温度シフトファクター", "◎", r"アレニウスの式とは何か？ なぜ、広く使われている Cross-WLF 式と同じ WLF 式を、温度シフトファクターに使わなかったのか？",   # 2026-11-17 の 6 回目（ユーザーの質問。GPPS の WLF 式の質問をまとめた）
     r"アレニウスの式は、もともと化学反応の速さが温度でどう変わるかを表す式で、溶融した樹脂の粘度の温度による変化にもよく使われます。粘度の対数が絶対温度の逆数に比例して変わると考え、ずれの量を $\ln a_T = (E/R)(1/T - 1/T_\mathrm{ref})$ で表します。WLF 式も候補にして、3 材料すべてに、アレニウス型と WLF 型の両方を同じ条件で当てはめて比べました。学習データの尤度が、3 材料ともアレニウス型のほうが大きかったので、アレニウス型を採りました。",
     [r"WLF 型の式：$\ln a_T = -A_1\left[\dfrac{T - D_2}{A_2 + T - D_2} - \dfrac{T_\mathrm{ref} - D_2}{A_2 + T_\mathrm{ref} - D_2}\right]$（Cross-WLF 式の温度の部分と同じ形）。$A_2$ = 51.6 K、$D_2$ = ガラス転移温度（PP・リサイクルPP は $-10$ ℃〔Moldflow の PP でよく使われる値〕、GPPS は仮置きの 100 ℃）に固定し、$A_1$ だけを推定した。アレニウス型も、推定する係数は $E$ の 1 つ（GPPS は縦と横で 2 つ）で、数をそろえて比べた。",
      r"係数 0 との尤度の差（アレニウス型／WLF 型。PP・リサイクルPP は縦だけ、GPPS は縦と横）：PP <<ARRD_PP>>／<<WLFD_PP>>、リサイクルPP <<ARRD_MIX>>／<<WLFD_MIX>>、GPPS <<ARRD_GPPS>>／<<WLFD_GPPS>>。係数 0 のモデルは 2 つの型で同じなので、差（PP <<AWD_PP>>、リサイクルPP <<AWD_MIX>>、GPPS <<AWD_GPPS>>）が、そのまま 2 つの型の尤度の差（シフトファクター検討用のノート。本番学習の設定の決定.pdf。† は PP の直す前の樹脂圧力差での結果）。",
      r"一般に、WLF 式はガラス転移温度から約 100 ℃ 上までの範囲で、アレニウスの式はそれより十分高い温度で使われる。PP の成形温度 190〜230 ℃ はガラス転移温度（約 $-10$ ℃）より 200 ℃ 以上、GPPS の 200〜240 ℃ はガラス転移温度（約 100 ℃）より 100 ℃ 以上高い\kak{引用する文献はご自身で}。",
      r"GPPS の検証データ（3data）の誤差（対数の RMSE）は、WLF 型のほうが少し小さい（<<GPPS_RMSE_WLF>> 対 <<GPPS_RMSE_ARR>>）。尤度（アレニウス型 <<GPPS_LML_ARR>>、WLF 型 <<GPPS_LML_WLF>>）で選んだ。\kak{GPPS をどちらにするかは検討中}",
      r"Cross-WLF 式は、横（せん断速度の向き）と縦（粘度の向き）を同じ量だけずらす。このデータでは、同じ量でずらす形は合わなかったので、どちらの型でも、縦と横を別々の係数にしている（<<Q_vh>>）。"]),
    ("D. 温度シフトファクター", "◎", r"温度シフトファクターは、学習のどの段階で組み込んでいるのか？",
     r"ガウス過程を学習する前に、データの側を温度でずらす形で組み込みました。係数を決めると、各データの粘度の対数から $\ln a_v(T_\mathrm{set})$ を引き（GPPS は、せん断速度の対数にも $\ln a_h(T_\mathrm{set})$ を足し）、すべてのデータを基準温度（PP・リサイクルPP <<TREF_PP>> ℃、GPPS <<TREF_GPPS>> ℃）相当にそろえます。ガウス過程は、このそろえたデータを学習します。予測のときは逆に、ガウス過程の出力に $\ln a_v$ を足し戻します。",
     [r"学習する形：$\ln\eta - \ln a_v(T_\mathrm{set}) = f(\ln\dot{\gamma} + \ln a_h(T_\mathrm{set}),\ \ln\Delta P_2,\ \ln T) + \varepsilon$。$f$ がガウス過程、$\varepsilon$ がノイズ（ショット間のばらつき）。",
      r"そろえる量（縦。最終のモデルの係数で）：PP は <<TLOW_PP>> ℃ のデータの粘度を約 <<SHL_PP>>\% 小さく、<<THIGH_PP>> ℃ のデータを約 <<SHH_PP>>\% 大きくする。リサイクルPP は <<SHL_MIX>>・<<SHH_MIX>>\%、GPPS は <<SHL_GPPS>>・<<SHH_GPPS>>\%（<<TLOW_GPPS>>・<<THIGH_GPPS>> ℃）。基準温度のデータは動かさない。",
      r"係数の値は、ずらし方を変えて学習を繰り返し、尤度で選ぶ（<<Q_grid>>）。学習の流れの全体は <<Q_flow>>。",
      r"標準化の定数は、ずらす前のデータで決めている。ずらし方によらず同じ物差しにして、尤度を比べられるようにするため。"]),
    ("D. 温度シフトファクター", "", r"シフトファクターの式は、機械学習を行う前に定義しているのか？",   # 2026-11-17 の 6 回目（ユーザーの質問）
     r"はい。式の形（アレニウス型、ずらす向き〔PP・リサイクルPP は縦だけ、GPPS は縦と横〕、基準温度）は、学習の前に決めました。データから決めたのは、式の中の係数（縦の $E_v$、GPPS は横の $E_h$ も）だけです。係数は、ガウス過程と同じ学習データで、尤度が最大になる値を探して決めました。",
     [r"式の形は、本番の学習の前の検討（シフトファクター検討用のノート）で選んだ。型（アレニウス型・WLF 型。<<Q_wlf>>）、ずらす向き（縦だけ・縦と横。<<Q_vh>>）、ずらす温度（加熱シリンダ温度・樹脂温度最高値。<<Q_tff>>）の組み合わせを、同じ学習データ（5data）の尤度で比べた。",
      r"検証データ（3data）は、形を選ぶのには使っていない。選んだ形の誤差が、加熱シリンダ温度を入力に並べたモデル（C）の 1.1 倍以内かを確かめるのに使った（<<Q_need>>）。",
      r"基準温度 $T_\mathrm{ref}$ は 3 水準の真ん中（PP・リサイクルPP <<TREF_PP>> ℃、GPPS <<TREF_GPPS>> ℃）。基準温度のデータは動かさない。",
      r"係数は、ガウス過程のハイパーパラメータとは別に、外側の格子で決めた（<<Q_grid>>）。学習の流れの全体は <<Q_flow>>。"]),
    ("D. 温度シフトファクター", "", r"シフトの係数は、ハイパーパラメータと一緒に最適化したのか？ なぜ格子で調べたのか？",
     r"一緒ではなく、入れ子で決めました。外側で係数を格子状に変え、その値ごとに、内側でガウス過程のハイパーパラメータを最尤推定して、尤度が最大の係数を選びます（プロファイル尤度）。係数はデータのずらし方を変えるので、scikit-learn のハイパーパラメータの最適化には入れられないことと、尤度の曲線全体を見て、係数の 95\% 区間と、係数 0（ずらさない）との差を確かめたかったためです。",
     [r"格子：5 kJ/mol 刻みの粗い格子（縦 $-20$〜60、横 $-20$〜40）で全体を見てから、最良の点の周りを細かく（縦だけは 0.5、縦と横は 1 kJ/mol 刻み）。学習した回数は PP <<NPTS_PP>>、リサイクルPP <<NPTS_MIX>>、GPPS <<NPTS_GPPS>> 回。",
      r"判定：係数 0 との対数周辺尤度の差が、縦だけは 1.92、縦と横は 3.0 以上なら支持（尤度比検定の 5\% 基準。$2\Delta\mathrm{LML}$ が、自由度 1・2 のカイ二乗分布の 95\% 点 3.84・5.99 を超える）。95\% 区間は、尤度が最大から 1.92 以内の範囲。",
      r"尤度の曲線（PP・リサイクルPP）と面（GPPS）の図：各モデルのフォルダの プロファイル尤度.png。",
      r"係数とハイパーパラメータを同時に推定する方法（MCMC）は、本体の検討（別のモデル）で試し、格子で決めた係数と一致した。このポスターのモデルでは試していない。"]),
    ("D. 温度シフトファクター", "", r"加熱シリンダ温度は、ガウス過程の入力に入っているのか？",
     r"入っていません。加熱シリンダ温度は、温度シフトファクターを通してだけモデルに入ります。ガウス過程の入力は、せん断速度（GPPS は温度でずらした値）、樹脂圧力差、樹脂温度の 3 つです。そのため、加熱シリンダ温度の効果は、アレニウス式の形（絶対温度の逆数に比例するずれ）に限られます。",
     [r"比べたモデル：加熱シリンダ温度を、そのまま 4 つ目の入力に並べた通常のガウス過程回帰（C）。対数周辺尤度は温度シフトのモデルのほうが PP <<DLC_PP>>、リサイクルPP <<DLC_MIX>>、GPPS <<DLC_GPPS>> 大きい。検証データ全体の誤差（対数の RMSE）は、C の <<RC_PP>>・<<RC_MIX>>・<<RC_GPPS>> 倍で、同じくらい（シフトファクター検討用のノート。† は PP の直す前の樹脂圧力差での結果で、直した値ではまだ比べていない）。",
      r"温度シフトの形にすると、温度の効果が、なめらかな式に縛られるので、学習にない温度の予測に強くなると期待した\kak{考察}。温度を 1 つ外した検証の結果は <<Q_ho>>。"]),
    ("D. 温度シフトファクター", "", r"縦のずれと横のずれは、何が違うのか？ なぜ GPPS だけ縦と横なのか？",
     r"縦のずれは、同じせん断速度で粘度を何倍にするか、横のずれは、粘度曲線をせん断速度の向きにずらす量です。両方の形を当てはめ、縦と横にしたときの尤度が、縦だけより 1.92 以上大きいときだけ、縦と横を採りました。GPPS だけがこれを満たし（<<BA_GPPS>>）、PP とリサイクルPP は、横を足しても尤度が増えませんでした（<<BA_PP>>・<<BA_MIX>>）。",
     [r"係数 0 との尤度の差（縦だけ／縦と横）：PP <<CA_PP>>／<<CB_PP>>、リサイクルPP <<CA_MIX>>／<<CB_MIX>>、GPPS <<CA_GPPS>>／<<CB_GPPS>>（シフトファクター検討用のノート。本番学習の設定の決定.pdf の表）。",
      r"★† は PP の直す前の樹脂圧力差での結果。直した値で学習し直したポスターの PP のモデル（縦だけ）では、係数 0 との尤度の差は <<DLML_PP>> に下がった（<<Q_e3>>）。縦と横の形は、直した値ではまだ比べていない。",
      r"縦と横は、縦だけに係数を 1 つ足したもの。足した係数が効いているかを、尤度比検定（自由度 1）の 5\% 基準 1.92 で判定した。",
      r"GPPS の横の係数は <<EH_GPPS>> kJ/mol と小さく、<<TLOW_GPPS>> ℃ と <<THIGH_GPPS>> ℃ の間で、せん断速度の向きに約 <<HSH_GPPS>>\% ずらすだけ。主は縦（<<EV_GPPS>> kJ/mol）。",
      r"通常の換算則（Cross-WLF の $a_T$ など）は、横と縦を同じ量だけずらす。見かけの流動曲線の傾きが $-1$ より急なこのデータでは、その形だと「低温ほど粘度が低い」向きになり、合わなかった（本体の検討）。そのため、縦と横を別々の係数にした。"]),
    ("D. 温度シフトファクター", "", r"マスターカーブ（時間–温度換算則）と同じ考え方か？",
     r"考え方は同じです。温度の違いを曲線のずれとして表し、ずらした後のデータを 1 本の曲線にまとめます。Cross-WLF 式の温度シフト $a_T$ と同じ役目です。違いは、まとめた後の曲線の形を式で決めずにガウス過程で学ぶことと、ずれの量を尤度で決めていることです。",
     [r"ずらした後のデータと曲線の図：各モデルのフォルダの マスターカーブ.png（ずらす前は 流動曲線\_シフト前.png）。",
      r"通常の換算則とは、縦と横を別々の係数にした点が違う（<<Q_vh>>）。",
      r"★温度による粘度の差は小さい（低い温度と高い温度の粘度の比 PP <<VR_PP>>、リサイクルPP <<VR_MIX>>、GPPS <<VR_GPPS>> 倍）ので、図では、ずらす前と後の違いが見えにくい。"]),
    ("D. 温度シフトファクター", "", r"なぜ樹脂温度ではなく、加熱シリンダ温度でずらしたのか？",
     r"加熱シリンダ温度は成形の前に決まる設定値で、予測のときにも分かるからです。樹脂温度の最高値でずらす形も比べましたが、3 材料とも尤度で支持されませんでした。樹脂温度の最高値は、入力として別に入れています。",
     [r"樹脂温度でずらしたときの、係数 0 との尤度の差の最大（縦だけ／縦と横。基準は 1.92／3.0）：PP <<TFFA_PP>>／<<TFFB_PP>>、リサイクルPP <<TFFA_MIX>>／<<TFFB_MIX>>、GPPS <<TFFA_GPPS>>／<<TFFB_GPPS>>。どれも基準に届かない（† は PP の直す前の樹脂圧力差での結果。直した値では、まだ比べていない）。",
      r"比べた結果は「05\_本番学習の設定(検討結果説明書)/本番学習の設定の決定.pdf」。"]),
    ("D. 温度シフトファクター", "★", r"係数の値は？ 文献の活性化エネルギーより小さくないか？",
     r"縦方向の係数は PP <<EV_PP>>、リサイクルPP <<EV_MIX>>、GPPS <<EV_GPPS>> kJ/mol で、文献の流動の活性化エネルギー（PP で 40 kJ/mol 前後）よりずっと小さいです。見かけの粘度に、冷却やせん断発熱の効果が入っていることが主な理由と考えています\kak{考察}。そのため、物性値ではなく、このモデルの中での「見かけのシフトの係数」として扱っています。",
     [r"95\% 区間：PP <<EVLO_PP>>〜<<EVHI_PP>>、リサイクルPP <<EVLO_MIX>>〜<<EVHI_MIX>>、GPPS <<EVLO_GPPS>>〜<<EVHI_GPPS>> kJ/mol。GPPS の横方向の係数は <<EH_GPPS>> kJ/mol。",
      r"低い温度と高い温度の粘度の比（縦の倍率）：PP <<VR_PP>> 倍（<<VRT_PP>>）、リサイクルPP <<VR_MIX>> 倍（<<VRT_MIX>>）、GPPS <<VR_GPPS>> 倍（<<VRT_GPPS>>）。",
      r"係数 0 との対数周辺尤度の差：PP <<DLML_PP>>（基準 <<CRIT_PP>>）、リサイクルPP <<DLML_MIX>>（<<CRIT_MIX>>）、GPPS <<DLML_GPPS>>（<<CRIT_GPPS>>）。リサイクルPP と GPPS は基準を超え、支持された。",
      r"\textbf{★PP は、直した樹脂圧力差で学習し直すと、基準に届かず、温度シフトは支持されなくなった}（直す前は <<DLMLPRE_PP>> で支持。縦の係数は <<EVPRE_PP>> → <<EV_PP>> kJ/mol、95\% 区間 <<EVLO_PP>>〜<<EVHI_PP>> で 0 を含む）。PP では、加熱シリンダ温度による粘度の違いを、温度シフトファクターがほとんど受け持っていない（<<Q_ho>>）。\kak{理由の見当（樹脂圧力差と樹脂温度最高値の入力が、温度の効果を担っているなど）は、ご自身で}",
      r"樹脂圧力差と樹脂温度を入力にしない本体のモデル（同じ PP・リサイクルPP の成形データ。入力はせん断速度〔条件の平均〕と厚み。縦だけのずれ）でも、係数は 7〜7.5 kJ/mol で、やはり 40 よりずっと小さい。ここでの値がそれより小さいのは、入力の樹脂圧力差が温度の効果の一部を担っているためかもしれない\kak{考察}。入力の樹脂温度の効きは小さい（<<Q_imp>>）。",
      r"縦の係数を文献の 40 kJ/mol にすると、対数周辺尤度が PP で <<LIT_PP>>、リサイクルPP で <<LIT_MIX>>、GPPS で約 <<LIT_GPPS>>（横の係数は格子の中で最もよい値）下がり、係数 0 よりずっと悪い（各モデルの プロファイル尤度\_格子.csv）。文献値は、このデータには合わない。"]),
    ("D. 温度シフトファクター", "★", r"係数の不確かさは、予測の幅に入っているか？",
     r"入っていません。予測の幅は、係数とハイパーパラメータを、尤度が最大の値に固定したときのガウス過程の幅（データの少なさによる不確かさと、ショット間のばらつき）です。係数の 95\% 区間の中で係数を変えたときに、予測がどれだけ変わるかは、まだ確かめていません\kak{今後の方針}。",
     [r"縦の係数の 95\% 区間：PP <<EVLO_PP>>〜<<EVHI_PP>>、リサイクルPP <<EVLO_MIX>>〜<<EVHI_MIX>>、GPPS <<EVLO_GPPS>>〜<<EVHI_GPPS>> kJ/mol。",
      r"区間の両端の係数で、低い温度と高い温度の粘度の比（縦の倍率）は、PP <<VRLO_PP>>〜<<VRHI_PP>> 倍、リサイクルPP <<VRLO_MIX>>〜<<VRHI_MIX>> 倍、GPPS <<VRLO_GPPS>>〜<<VRHI_GPPS>> 倍。PP の区間は 0 を含み、比は 1 をまたぐ（温度シフトなしと区別できない。<<Q_e3>>）。",
      r"本体の検討（別のモデル）では、係数もハイパーパラメータと一緒に MCMC でサンプリングし、学習の範囲の中の予測はほぼ変わらず、外挿で 95\% 区間が 1.0〜1.1 倍に広がった。"]),
    ("D. 温度シフトファクター", "★", r"温度シフトファクターは本当に効いているのか？ 入れなくても同じでは？",
     r"材料によって違います。加熱シリンダ温度を 1 つずつ学習から外して、その温度を予測する検証をしたところ、GPPS は温度シフトを入れたほうが誤差が <<HO_GPPS_GAIN>>\% 小さく、はっきり効いていました。リサイクルPP は小さいながら効果があり、PP では効果が見られませんでした。PP は、直した樹脂圧力差で学習し直したポスターのモデルでも、係数 0 との尤度の差が <<DLML_PP>> で、温度シフトは支持されていません。",
     [r"\textbf{ポスターにない新しい結果（11-16）。言うかはご自身で判断}。資料は 09\_温度ホールドアウト検証結果/ の HTML。\textbf{PP の温度ホールドアウト検証（†）は、直す前の樹脂圧力差で学習したもの}で、直した値では、まだ実行していない。",
      r"誤差の比（温度シフトあり ÷ 加熱シリンダ温度を使わないモデル。除いた温度の低・中・高の順）：GPPS <<HO_GPPS>>、リサイクルPP <<HO_MIX>>、PP <<HO_PP>>。1/1.1 より小さければ「良い」、1.1 より大きければ「悪い」。",
      r"PP は、外した温度によって縦の係数が大きく変わる（<<HOEV_PP>> kJ/mol）。",
      r"加熱シリンダ温度をそのまま入力に並べたモデルとの比：GPPS <<HOC_GPPS>>、リサイクルPP <<HOC_MIX>>、PP <<HOC_PP>>（リサイクルPP は、シフトの形でもただの入力でも同じくらい）。",
      r"ポスターの学習（3 温度すべて）では、リサイクルPP と GPPS の係数は尤度で支持され、PP（直した樹脂圧力差で学習し直したもの）は支持されない（<<DLML_PP>>。直す前は <<DLMLPRE_PP>>。<<Q_e3>>）。",
      r"\textbf{★PP の学習結果のグラフでも、線は温度ごとにずれて見える}（<<TLOW_PP>> ℃ の線は <<THIGH_PP>> ℃ の <<GAP_PP>> 倍。学習データのせん断速度の範囲）。グラフは、温度ごとに、その温度の樹脂温度最高値の中央値（<<TFFMED_PP>> ℃）を入力にして描いた断面で（<<Q_wavy>>）、ずれのうち温度シフトファクターの分は <<GAPSH_PP>> 倍だけ。残りは、樹脂温度最高値の入力の違いによる。ポスターの学習結果の「バージンPPでは、主に樹脂温度最高値の入力の違いによる」は、このこと（11-17 にポスターの文を直した）。",
      r"この樹脂温度最高値の分は <<GAPTFF_PP>> 倍で、せん断速度によって変わる。学習データの点での平均の傾き（<<Q_imp>>）から見込む値（約 <<GAPEXP_PP>> 倍）より大きい所があるのは、グラフの断面が、学習データの少ない組み合わせ（遅い側で樹脂温度最高値が高い、など）を通るためかもしれない\kak{考察}。"]),
    ("D. 温度シフトファクター", "★◎", r"温度シフトファクターなしで学習した場合と比べたのか？ 比べたなら、何の値を比べて、温度シフトファクターが必要と判断したのか？",   # 2026-11-17 の 6 回目（ユーザーの質問）
     r"比べました。判断に主に使ったのは、学習データの対数周辺尤度です。係数を 0 にした、温度シフトファクターなしのモデルとの尤度の差が基準（縦だけ 1.92、縦と横 3.0）を超えたら、温度シフトがデータで支持されたとしました。そのうえで、検証データの誤差が、加熱シリンダ温度をそのまま入力に並べたモデルの 1.1 倍以内かを確かめました。ポスターの後に、加熱シリンダ温度を 1 つずつ外して予測する検証もしました。",
     [r"(1) 係数 0 との尤度の差（基準 1.92／3.0）：PP <<DLML_PP>>（直す前は <<DLMLPRE_PP>>）、リサイクルPP <<DLML_MIX>>、GPPS <<DLML_GPPS>>。リサイクルPP と GPPS は支持。PP は、直す前は支持されたが、直した樹脂圧力差で学習し直すと支持されない（<<Q_e3>>）。",
      r"(2) 検証データ（3data）の誤差の比（温度シフトあり ÷ 加熱シリンダ温度を入力に並べたモデル C）：PP <<RC_PP>>、リサイクルPP <<RC_MIX>>、GPPS <<RC_GPPS>>。3 材料とも 1.1 倍以内で、同じくらい（<<Q_tset>>）。",
      r"(3) 検証データの誤差の比（温度シフトあり ÷ 加熱シリンダ温度をまったく使わないモデル）：PP <<NS3_PP>>、リサイクルPP <<NS3_MIX>>、GPPS <<NS3_GPPS>>。\textbf{★GPPS は、温度シフトなしのほうが誤差が小さい}。\kak{理由は、まだ調べていない}",
      r"(4) 加熱シリンダ温度を 1 つずつ外した検証（<<Q_ho>>）：GPPS とリサイクルPP は、温度シフトありのほうが誤差が小さく、PP は小さくならなかった（PP は†）。",
      r"★検証データ（3data）は、射出率の両端（せん断速度の外挿）で、温度の外挿ではない。そのため (2)(3) では、温度シフトが要るかを直接は測れない。温度について直接確かめたのは (4)。",
      r"比べたモデルは、ガウス過程の入力（せん断速度・樹脂圧力差・樹脂温度最高値）とカーネルをそろえ、温度の入れ方だけを変えた。係数 0 のモデルと、加熱シリンダ温度をまったく使わないモデルは同じもの。(2)〜(4) は、シフトファクター検討用のノートと温度ホールドアウト検証の結果（† は PP の直す前の樹脂圧力差での結果）。"]),
    # ---- F
    ("E. 予測精度", "★◎", r"誤差 <<E_ALL_RANGE>>\% は大きくないか？ 何の誤差か？",
     r"検証データ全体の誤差で、その大部分は低射出率側（学習の範囲より低い側の外挿）から来ています。高射出率側は <<E_HIGH_RANGE>>\%、低射出率側は <<E_LOW_RANGE>>\% です。",
     [r"誤差は、粘度の対数の二乗平均平方根誤差（RMSE）を \% に換算した値（$e^{\mathrm{RMSE}} - 1$）。",
      r"材料ごと（全体／低射出率側／高射出率側）：PP <<E_all_PP>>／<<E_low_PP>>／<<E_high_PP>>\%、リサイクルPP <<E_all_MIX>>／<<E_low_MIX>>／<<E_high_MIX>>\%、GPPS <<E_all_GPPS>>／<<E_low_GPPS>>／<<E_high_GPPS>>\%。",
      r"平均の誤差（MAPE）では、全体 <<MAPE_RANGE>>\%。RMSE は大きく外れた点を重く数える。",
      r"\textbf{★まとめの「予測誤差率30\%以内」は全体の値}。低射出率側だけでは、PP は <<E_low_PP>>\% で 30\% を超える。"]),
    ("E. 予測精度", "★", r"「決定係数R$^2$が1に極めて近く、本学習が正常に行われた」と言えるのか？ 評価が甘くないか？",
     r"決定係数は、粘度の対数で計算しています。検証データの粘度は約 3 桁の範囲に広がっているので、対数の決定係数は高く出やすく、精度は誤差（RMSE）のほうで見ています。また、検証には学習と重ならない射出率の条件を使ったので、ランダムに分けるより厳しい評価です。",
     [r"粘度の実数値で計算した決定係数：PP <<R2R_PP>>、リサイクルPP <<R2R_MIX>>、GPPS <<R2R_GPPS>>（低射出率側の大きな粘度の誤差に引っ張られる）。"]),
    ("E. 予測精度", "◎", r"低射出率側の誤差が大きいのはなぜ？ どうすれば改善できるか？",
     r"低射出率側は学習データの範囲の外（外挿）で、データがないためです。ただ、モデルは予測の幅を広げていて、検証データはすべて 95\% 予測区間に入りました。改善には、低せん断速度側のデータを足すことや、Cross 式などの粘度式をモデルの平均の形に使うことを考えています\kak{今後の方針}。",
     [r"低射出率側の被覆率：3 材料とも 100\%。",
      r"参考：同じ条件の平均を入力にしたモデルでは、低射出率側の誤差が小さかった（<<Q_pred>> の B\_1）。"]),
    ("E. 予測精度", "★", r"95\% 予測区間に、検証データの 95\% が入っているか？",
     r"検証データ全体では PP <<COV_all_PP>>、リサイクルPP <<COV_all_MIX>>、GPPS <<COV_all_GPPS>>\% でした。低射出率側は 100\% ですが、高射出率側は <<COV_HIGH_RANGE>>\% で、区間がやや狭すぎます。ショットごとの値を入力にしたため、ショット間のばらつき（ノイズ）が小さく見積もられたことが原因と考えています\kak{考察}。",
     [r"ノイズの標準偏差（粘度の対数）は <<NOISE_RANGE>> で、同じ条件のショット間のばらつき（PP・リサイクルPP で <<PURE_SD>>）の <<NOISE_FRAC>> 分の 1。",
      r"同じ条件の平均を入力にしたモデル（本体の B\_1）では、ノイズの標準偏差は <<B1NOISE>> で、被覆率は PP <<B1COV_PP>>、リサイクルPP <<B1COV_MIX>>\%。"]),
    ("E. 予測精度", "", r"材料によって精度が違う（GPPS がよい）のはなぜ？",
     r"違いの大部分は低射出率側の誤差です（GPPS <<E_low_GPPS>>\%、リサイクルPP <<E_low_MIX>>\%、PP <<E_low_PP>>\%）。高射出率側は 3 材料とも <<E_HIGH_RANGE>>\% で同じくらいです。低射出率側で差が出る理由は、まだ調べていません\kak{理由の見当があれば}。",
     []),
    ("E. 予測精度", "", r"なぜランダムに分けて検証しないのか？",
     r"同じ成形条件の別のショットは値がほとんど同じなので、ランダムに分けると、同じ条件のショットが学習にも検証にも入り、評価が甘くなります。今回は、射出率の両端の条件を丸ごと学習から外し、学習の範囲の外をどこまで予測できるかを確かめました。",
     [r"前報（6月の PP のモデル）は、全データの 2 割をランダムに選んで検証していた（決定係数 0.9 超。CLAUDE.md の本体の §4）。"]),
    # ---- G
    ("F. 今後と応用", "", r"流動解析にどう使うのか？",
     r"粘度の予測値と予測の幅を流動解析に渡し、充填の圧力などの予測に、ばらつきの幅を付けることを考えています\kak{ご自身の言葉で}。そのためには、解析の中で計算される圧力と温度を、モデルの入力に与える方法が課題です（<<Q_pred>>）。",
     []),
    ("F. 今後と応用", "", r"新しい材料にも使えるか？ リサイクルPP とバージンPP の違いは？",
     r"今のモデルは材料ごとに学習しているので、新しい材料では計測金型での成形実験が必要です（今回は PP で 129 条件）。材料の違いを入力に加えてまとめて学習することは、今後の課題です\kak{今後の方針}。リサイクルPP とバージンPP の粘度の差は、ポスターでは比べていません。",
     [r"温度による粘度の変化は、リサイクルPP のほうがやや大きい（低い温度と高い温度の粘度の比 PP <<VR_PP>> 倍、リサイクルPP <<VR_MIX>> 倍）。"]),
    ("F. 今後と応用", "", r"まとめの「計測値のばらつきをより多く得られる射出成形金型」とは何か？",
     r"\kak{ご自身の言葉で準備する（ポスターの 6. まとめの文）}",
     [r"例（Claude の推測。そのまま使わない）：1 回の成形で、より多くの位置の圧力・温度を測れる金型、圧力と温度を同じ位置で同時に測れる金型、など。"]),
]


QREF = {"c2": "どのくらい合っているのか", "cov": "検証データの 95", "leak": "答えを入力に入れていないか",
        "pred": "樹脂圧力差と樹脂温度はどう与える", "e3": "係数の値は？", "r2": "本学習が正常に行われた", "err": "何の誤差か",
        "inp": "どの値を使ったのか",
        "flow": "学習が終わるまで", "hyp": "ハイパーパラメータは、どう決めた", "ard": "各材料の長さスケール", "imp": "どの入力が重要",
        "lin": "線形カーネルは、何のために", "wavy": "波打っている", "overfit": "過学習", "predcalc": "どう計算して予測",
        "when": "学習のどの段階", "grid": "なぜ格子で", "tset": "ガウス過程の入力に入っている", "vh": "縦のずれと横のずれ",
        "unc": "係数の不確かさ", "ho": "本当に効いているのか", "ker": "カーネル関数は何を使い",
        "out": "外れ値があった場合", "wlf": "アレニウスの式とは何か", "form": "機械学習を行う前に定義", "alone": "モデル（平均関数とその予測幅）だけで",
        "need": "温度シフトファクターなしで学習した場合", "tff": "なぜ樹脂温度ではなく",
        "ewhat": "係数 E（", "ecalc": "係数 E は、どうやって求めた"}   # 質問どうしの参照（質問の文の一部 → <<Q_…>>）。11-17 の 6 回目に out〜tff、8 回目に ewhat・ecalc を足した


def build_qa(N, raw):
    P, X, G = raw["PP"], raw["MIX"], raw["GPPS"]
    f3 = lambda k, fmt=str: " & ".join(fmt(raw[m][k]) for m in MATS)
    rows = [
        r"温度シフトのモデル & A\_Tset\_Arr（縦） & A\_Tset\_Arr（縦） & B\_Tset\_Arr（縦・横） \\",
        r"加熱シリンダ温度 [℃] & 190・210・230 & 190・210・230 & 200・220・240 \\",
        r"キャビティ厚み [mm] & 0.5〜2.0（4） & 0.5〜2.0（4） & 1.0〜2.0（3） \\",
        "学習（ショット／条件） & " + " & ".join(f"{raw[m]['NTR']}／{raw[m]['NTRC']}" for m in MATS) + r" \\",
        "検証（ショット／条件） & " + " & ".join(f"{raw[m]['N_all']}／{raw[m]['NC_all']}" for m in MATS) + r" \\",
        r"\midrule",
        "決定係数（対数／実数値） & " + " & ".join(f"{raw[m]['R2']:.3f}／{raw[m]['R2R']:.2f}" for m in MATS) + r" \\",
        r"誤差 [\%] 全体／低／高 & " + " & ".join(f"{raw[m]['E_all']}／{raw[m]['E_low']}／{raw[m]['E_high']}" for m in MATS) + r" \\",
        r"MAPE [\%] 全体 & " + f3("MAPE") + r" \\",
        r"95\% 区間に入った割合 [\%] 全体／低／高 & " + " & ".join(f"{raw[m]['COV_all']}／{raw[m]['COV_low']}／{raw[m]['COV_high']}" for m in MATS) + r" \\",
        r"Cross-WLF の誤差 [\%] 全体／低／高 & " + " & ".join(f"{raw[m]['CW_all']}／{raw[m]['CW_low']}／{raw[m]['CW_high']}" for m in MATS) + r" \\",
        r"\midrule",
        r"縦の係数 $E_v$ [kJ/mol]（95\% 区間） & " + " & ".join(f"{num(raw[m]['EV'])}（{num(raw[m]['EV_LO'])}〜{num(raw[m]['EV_HI'])}）" for m in MATS) + r" \\",
        r"横の係数 $E_h$ [kJ/mol] & ― & ― & " + num(G["EH"]) + r" \\",
        "低温 ÷ 高温の粘度の比 & " + " & ".join(f"{raw[m]['VR']:.2f}" for m in MATS) + r" \\",
        r"係数 0 との $\Delta$LML（基準） & " + " & ".join(f"{N['DLML_' + m]}（{N['CRIT_' + m]}）" for m in MATS) + r" \\",
        r"尤度で支持されたか & " + " & ".join("支持" if raw[m]["SUP"] else r"\textbf{支持されない}" for m in MATS) + r" \\",
        r"\midrule",
        r"3data の誤差の比（÷ C／÷ 温度なし）$^{*9}$ & " + " & ".join(f"{N['RC_' + m]}／{N['NS3_' + m]}" for m in MATS) + r" \\",
        r"温度ホールドアウトの誤差の比$^{*1}$ & " + " & ".join(N["HO_" + m] for m in MATS) + r" \\",
        r"本体の B\_1 の誤差 [\%] 全体／低$^{*2}$ & " + f"{P['B1']['all']}／{P['B1']['low']} & {X['B1']['all']}／{X['B1']['low']} & ― " + r"\\",
    ]
    fk = lambda fn: " & ".join(fn(raw[m]) for m in MATS)   # noqa: E731
    rows2 = [   # カーネルと長さスケールの早見表（2026-11-17）
        r"学習データ [行] & " + fk(lambda r: str(r["NTR"])) + r" \\",
        r"Matern の振幅 $\sigma_f$ & " + fk(lambda r: f"{r['AMP']:.3f}") + r" \\",
        r"長さスケール $\ell$ & " + fk(lambda r: f"{r['LS']:.2f}／{r['LS_P']:.2f}／{r['LS_T']:.1f}") + r" \\",
        r"$1/\ell$（特徴スケール.png） & " + fk(lambda r: f"{1 / r['LS']:.2f}／{1 / r['LS_P']:.2f}／{1 / r['LS_T']:.3f}") + r" \\",
        r"$\ell$ を元の単位で$^{*3}$ & " + fk(lambda r: f"×{r['WG']:.2f}／×{r['WP']:.1f}／{r0(r['WT_C'])} ℃") + r" \\",
        r"学習の範囲 ÷ $\ell$ & " + fk(lambda r: f"{r['NLS'][0]:.1f}／{r['NLS'][1]:.1f}／{r['NLS'][2]:.2f}") + r" \\",
        r"予測の傾き$^{*4}$ & " + fk(lambda r: "／".join(sgn(v, 2) for v in r["SLOPE"])) + r" \\",
        r"\midrule",
        r"線形カーネル $\sqrt{c}$／$\sigma_0$ & " + fk(lambda r: f"{r['LIN_AMP']:.3f}／{sci(r['SIGMA0'])}") + r" \\",
        r"予測の平均の大きさ$^{*5}$ & " + fk(lambda r: f"{r['PART_L']:.2f}／{r['PART_M']:.2f}") + r" \\",
        r"線形なしの誤差の比$^{*6}$ & " + fk(lambda r: f"{r['NOLIN']['all']:.1f}／{r['NOLIN']['low']:.1f}／{r['NOLIN']['high']:.2f}") + r" \\",
        r"ノイズの標準偏差 $\sigma_n$ & " + fk(lambda r: f"{r['NOISE']:.4f}") + r" \\",
        r"学習データの残差の標準偏差 & " + fk(lambda r: f"{r['RES']:.4f}") + r" \\",
        r"\midrule",
        r"対数周辺尤度 LML & " + fk(lambda r: f"{r['LML']:.1f}") + r" \\",
        r"係数の格子で学習した回数 & " + fk(lambda r: str(r["NPTS"])) + r" \\",
        r"$\Delta$LML（縦だけ／縦と横）$^{*7}$ & " + " & ".join(f"{N['CA_' + m]}／{N['CB_' + m]}" for m in MATS) + r" \\",
        r"縦と横 − 縦だけの LML & " + " & ".join(N["BA_" + m] for m in MATS) + r" \\",
        r"$E_v$ = 40 kJ/mol での LML の低下 & " + fk(lambda r: str(r0(r["LIT_DROP"]))) + r" \\",
        r"C との LML の差$^{*8}$ & " + " & ".join(N["DLC_" + m] for m in MATS) + r" \\",
    ]
    cats, body, n = [], [], 0
    for cat, mark, q, a, notes in QA:
        if cat not in cats:
            cats.append(cat)
            body.append(f"\\Needspace{{12\\baselineskip}}\\section*{{{cat}}}")       # 見出しだけがページの下に残らないように
        n += 1
        mk = f"\\ \\Mark{{{mark}}}" if mark else ""
        items = "\n".join(f"  \\item {sub(x, N)}" for x in notes)
        hos = f"\\begin{{Hosoku}}\n{items}\n\\end{{Hosoku}}" if notes else ""
        body.append(f"\\Q{{{sub(q, N)}{mk}}}\n\\Ans{{{sub(a, N)}}}\n{hos}\n")
    n_kak = sum(len(re.findall(r"\\kak\{", f"{q}{a}{''.join(nt)}")) for _, _, q, a, nt in QA) + 1   # ＋1：使い方の PP の P3_3 の直しの項目（2026-11-17）
    tex = r"""%% 生成：04_補助スクリプト/build_poster4_talk.py（直接は直さない。直すときはスクリプトを直して作り直す）
\documentclass[paper=a4paper,fontsize=11pt,line_length=44zw,number_of_lines=37]{jlreq}
\usepackage[haranoaji]{luatexja-preset}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{needspace}
\usepackage[hidelinks]{hyperref}
\newcounter{qn}
\newcommand{\Mark}[1]{\textcolor{red!75!black}{\small #1}}
\newcommand{\kak}[1]{\textcolor{orange!75!black}{（要確認：#1）}}
\newcommand{\Q}[1]{\par\Needspace{6\baselineskip}\vspace{1.2ex}\refstepcounter{qn}\noindent{\bfseries Q\theqn.\quad #1}\par\nopagebreak[4]\vspace{0.3ex}}
\newcommand{\Ans}[1]{\noindent\textcolor{blue!60!black}{\bfseries 答え}\quad #1\par}
\newenvironment{Hosoku}{\par\noindent{\small\textcolor{black!60}{\bfseries 補足（聞かれたら）}}\begin{itemize}[leftmargin=1.5\zw,itemsep=0.1ex,topsep=0.1ex]\small}{\end{itemize}}
\setlist[itemize]{leftmargin=1.5\zw,itemsep=0.2ex,topsep=0.4ex}
\begin{document}
\begin{center}
{\Large ポスター4（添削版） 口頭発表の質問対策}\\[0.6ex]
{\small @@DATE@@　ポスターは「ポスター4\_添削版.pdf」、台本は「@@STEMTALK@@.pdf」（同じフォルダ）}
\end{center}

\section*{使い方}
\begin{itemize}
  \item 質問には、まず「答え」の 1〜3 文で答え、さらに聞かれたら「補足」を使う。1 つの答えは 30 秒以内を目安にする。
  \item 印：\Mark{◎} よく聞かれそうな質問、\Mark{★} 弱点になりうる質問（答えを特に準備しておく）。
  \item \kak{…} は、ご自身の言葉で答える所や、事実を確かめてほしい所（@@NKAK@@ か所）。Claude の考察・推測を含む文にも付けた。
  \item 射出成形そのものの基礎（成形機・センサの種類、成形条件の決め方、薄肉で成形できない理由、金型の中の冷却など）の質問は、入れていない（ご自身で答えられるため。11-16 のご指示）。
  \item 材料の呼び方はポスターに合わせた（リサイクルPP ＝ データの「粉砕材混合 PP」）。「誤差」は、粘度の対数の RMSE を \% に換算した値（ポスターのパリティ図の RMSE(\%)）。
  \item 数値は、ポスター4\_添削版の図と同じ学習結果（3 材料の最新の学習結果フォルダ。全ショット版・全厚み総合。バージンPP は 11-17 に学習し直した 4回目）の CSV から、このスクリプトが読んだ（最後の節）。
  \item 温度シフトファクターと機械学習の細かい質問（いつ・どう組み込むか、学習の流れ、カーネルの定義、ARD の長さスケールなど）は、C・D の節にある（11-17 に追加）。値は「カーネルと長さスケールの早見表」。
  \item 11-17 に、ポスター4\_添削版に合わせて、ポスターの文の引用と「ポスターの表記で指摘されそうな点」を直し、入力の樹脂圧力差と樹脂温度最高値の定義の質問（@@QINP@@）を足した。
  \item \Mark{★} PP の学習データの樹脂圧力差（P3\_3）は、11-17 にご本人が直した（厚み @@FIXH@@ mm の @@FIXN@@ 行。直す前の値は、直した値の @@FIXR@@ 倍）。ポスター（添削版）の PP の図と数値、この資料の PP の学習結果の数値は、直した値で学習し直したもの（4回目）。直す前（3回目）から、決定係数 @@R2PRE@@ → @@R2NEW@@、誤差 @@EPRE@@ → @@ENEW@@\%（低射出率側 @@ELPRE@@ → @@ELNEW@@\%）、縦の係数 @@EVPRE@@ → @@EVNEW@@ kJ/mol、係数 0 との $\Delta$LML @@DLPRE@@ → @@DLNEW@@ に変わり、\textbf{PP の温度シフトは尤度で支持されなくなった}（@@QE3@@・@@QHO@@）。
  \item $^{\dagger}$ を付けた PP の値（温度ホールドアウト検証、縦だけと縦と横の比べ、樹脂温度でのシフト、C との比べ）は、直す前の樹脂圧力差での結果。直した値では、まだ調べていない。\kak{† の検討を、直した値でやり直すか}
  \item 11-17 に、ご質問 6 つへの答えを足した：外れ値を含めて学習したか・除いた基準（@@QOUT@@。前の「外れ値はどう除いたのか」を書き直した）、アレニウスの式と WLF 式（@@QWLF@@。前の「GPPS は WLF 式のほうがよいのでは」をまとめた）、式を学習の前に決めたか（@@QFORM@@）、学習後のモデルだけで温度シフトを考えた予測ができるか（@@QALONE@@）、温度シフトファクターなしとの比べ（@@QNEED@@）。
  \item 11-17 に、温度シフトファクターの係数 $E$ とは何か（@@QEWHAT@@）と、$E$ の求め方（今の格子の方法。@@QECALC@@）の質問を足した（ご依頼）。
  \item 11-17 のご指示で、ポスター（添削版）の 5. の 2 か所の文を、誤解のない形に直した（モデルの説明の 4 行目と、学習結果よりの 1 つ目の点。「ポスターの表記で指摘されそうな点」の ★）。台本の 5. も、それに合わせた。
\end{itemize}

\section*{数値の早見表}
\begin{center}\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccc}
\toprule
項目 & バージンPP & リサイクルPP & バージンGPPS \\
\midrule
@@ROWS@@
\bottomrule
\end{tabular}
\end{center}
{\footnotesize
「低」「高」は検証データの低射出率側（1.6・2.7 cm$^3$/s）と高射出率側（149・170 cm$^3$/s）。
$^{*1}$ 加熱シリンダ温度を 1 つ外して学習し、外した温度を予測したときの誤差の比（温度シフトあり ÷ 加熱シリンダ温度を使わないモデル。外した温度の低・中・高の順。1/1.1 より小さければ「良い」）。ポスターにない結果（11-16）。
$^{*2}$ 樹脂圧力差と樹脂温度を使わず、せん断速度（条件の平均）・厚み・加熱シリンダ温度で学習した本体のモデル（同じ検証データ）。ポスターにない結果。
$^{*9}$ 検証データの誤差（対数の RMSE）の比。温度シフトあり ÷ 加熱シリンダ温度を入力に並べたモデル（C）／温度シフトあり ÷ 加熱シリンダ温度をまったく使わないモデル（@@QNEED@@）。
$^{\dagger}$ PP の直す前の樹脂圧力差での結果（直した値では、まだ実行していない）。\par}

\section*{答えにくい質問への構え}
\begin{itemize}
  \item 弱点は先に認めてから、分かっていることを言う。例：「低射出率側の誤差は大きいです。ただ、その範囲ではモデルが予測の幅を広げていて、検証データはすべて予測区間に入りました。」
  \item 調べていないことは、推測で言い切らない。例：「そこはまだ確かめていません。今後の課題として検討します。」
  \item 質問の意味が分からないときは、言い換えて確かめる。例：「〜ということでしょうか。」
  \item 式や計算の手順を聞かれたら、まず言葉で 1〜2 文で答え、さらに聞かれたら補足の式を示す。
  \item 数値は早見表から答える。覚えておくと安心な数：決定係数 @@R2@@、誤差 全体 @@EALL@@\%（低射出率側 @@ELOW@@\%、高射出率側 @@EHIGH@@\%）、95\% 区間に入った割合 @@COV@@\%。
\end{itemize}

\section*{ポスターの表記で指摘されそうな点（添削版）}
\begin{itemize}
  \item 添削版で直した点：図の中の誤記（学習結果の左のグラフの縦軸「樹脂粘度度」、単位「(Pa・S)」「(1/S)」、GPPS のパリティ図の横軸の右端の目盛り「$10^3$」→「$10^4$」）、3. 成形条件の GPPS の厚み（3 条件）、文の言い回し。元のポスター4 や案3 を見た人に違いを聞かれたら、添削版が正しい。
  \item 右下の枠の「低射出率側と高射出率側の計4条件」：射出率 1.6・2.7 cm$^3$/s（低射出率側。厚み 1.5・2.0 mm だけ）と 149・170 cm$^3$/s（高射出率側）の 4 つ。加熱シリンダ温度と厚みとの組み合わせでは、PP @@NC_PP@@・リサイクルPP @@NC_MIX@@・GPPS @@NC_GPPS@@ 条件。
  \item 右下の枠の「決定係数R$^2$が1に極めて近く、本学習が正常に行われたことが示された」：決定係数は学習に使っていない検証データで計算した、対数の値（@@QR2@@）。
  \item 6. まとめの「予測誤差率30\%以内」：全体の値。低射出率側だけでは PP が @@ELOWPP@@\%（@@QERR@@）。
  \item 5. の入力の「樹脂圧力差」「樹脂温度最高値」：ポスターには定義が書かれていないので、聞かれやすい（@@QINP@@）。
  \item \Mark{★} 5. の温度シフトファクターの文と、学習結果の「条件ごとに予測平均値と予測幅が変化している（バージンPPでは、主に樹脂温度最高値の入力の違いによる）」：PP では、直した樹脂圧力差で学習し直すと、温度シフトは尤度で支持されず（$\Delta$LML @@DLNEW@@）、温度シフトファクターの分は @@GAPSH@@ 倍だけ。PP のグラフの線のずれ（@@GAP@@ 倍）は、主に樹脂温度最高値の入力の違いによる（@@QHO@@）。11-17 のご指示で、この 2 か所を誤解のない形に直した（前は「温度シフトファクターを学習モデルに組み込んで機械学習を行った」「予測平均値と予測幅がシフトしている」。今は「温度シフトファクターを組み込んだ。温度の影響は、入力の樹脂温度最高値でも表される」）。「温度シフトファクターは効いているのか」と聞かれやすい。
\end{itemize}

\section*{カーネルと長さスケールの早見表}
\begin{center}\small\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{lccc}
\toprule
項目 & バージンPP & リサイクルPP & バージンGPPS \\
\midrule
@@ROWS2@@
\bottomrule
\end{tabular}
\end{center}
{\footnotesize
カーネル：$\sigma_f^2\,\mathrm{Matern}_{5/2}$（ARD）$+\ c\,(\sigma_0^2 + \mathbf{x}\cdot\mathbf{x}')$ ＋ ホワイトノイズ $\sigma_n^2$（@@QKER@@）。
「／」で 3 つ並べた値は、せん断速度／樹脂圧力差／樹脂温度の順（ガウス過程の入力。GPPS のせん断速度は、温度で横にずらした値）。
長さスケールと傾きは、標準化した入力（学習データの平均を引き、標準偏差で割った値）での値。
$^{*3}$ 長さスケールの分だけ入力を動かしたときの、せん断速度・樹脂圧力差の倍率と、樹脂温度の変化（℃）。
$^{*4}$ 入力を 1 標準偏差動かしたときの $\ln\eta$ の変化（学習データの点で、ほかの入力を固定して 1 つだけ動かした傾きの平均）。
$^{*5}$ 学習データの点での、予測の平均の線形カーネルの部分／Matern の部分の標準偏差（$\ln\eta$）。
$^{*6}$ 線形カーネルを外したときの、検証データの誤差（対数の RMSE）の比（全体／低射出率側／高射出率側。リサイクルPP・GPPS は 11-11、PP は 11-17 の実行）。
$^{*7}$ 係数 0（ずらさない）との対数周辺尤度の差。
$^{*8}$ 加熱シリンダ温度を、ずらす形でなく、ガウス過程の 4 つ目の入力に並べたモデル（C）との対数周辺尤度の差。
長さスケール・傾き・予測の平均の大きさ・残差は、学習済みモデル（.joblib）から計算した。$^{*7}$・$^{*8}$ と「縦と横 − 縦だけ」は、シフトファクター検討用のノートの結果（$^{\dagger}$ は PP の直す前の樹脂圧力差での結果。直した値で学習し直した PP のモデルの「係数 0 との $\Delta$LML」は @@DLNEW@@）。\par}

@@BODY@@

\section*{数値の出どころ}
{\small
\begin{itemize}
  \item 学習結果（ポスター4\_添削版の図と同じ）：05\_材料ごとの機械学習データと予測検証データ/ の、01\_バージンPP再々実験(データと結果)/01\_学習結果/4回目\_…（11-17。直した樹脂圧力差で学習し直したもの）、02\_粉砕材混合PP実験(データと結果)/02\_学習結果/5回目\_…、03\_GPPS実験(データと結果)/03\_学習結果/3回目\_…（11-09 の報告ノート）の、全ショット版/全厚み総合/〈モデルのフォルダ〉/ の 予測検証\_3data\_指標.csv・まとめ.csv・シフトファクターの推定.csv・シフトファクターの倍率.csv・ハイパーパラメータ.csv・CrossWLF\_係数.csv。データの数と樹脂圧力差の性質は、同じフォルダの データ確認/。PP の直す前の値は、同じ場所の 3回目\_…。
  \item MAPE：07\_学習結果報告書作成\_261015/ポスター/図/予測誤差の指標\_MAE\_MSE\_MAPE.csv（11-09）と同じ式で、上の各モデルのフォルダの 予測検証\_3data\_予測値.csv から計算した（リサイクルPP・GPPS は表と同じ値。表の PP は直す前の値）。
  \item 入力の樹脂圧力差・樹脂温度最高値の範囲：上の各モデルのフォルダの 学習済みモデル（学習データ）。各材料の 0N\_学習・検証データ/学習データ\_…\_5data/ の CSV（PP は 1117修正\_結合\_03\_pp\_max\_5data\_外れ値なし.csv）の列「流動中樹脂圧力P3\_3(MPa)」「流動中樹脂温度(℃)」の範囲と同じことを確かめた。定義は報告書B（07\_学習結果報告書作成\_261015/報告書B\_完成版.tex）。
  \item 温度ホールドアウト検証（11-16）：上の学習結果フォルダ（PP は直す前の 3回目\_…）の 全ショット版/全厚み総合/温度ホールドアウト検証/、まとめの資料は 09\_温度ホールドアウト検証結果/。
  \item PP の学習結果のグラフの線のずれの内訳：上の PP のフォルダの 学習済みモデル\_A\_Tset\_Arr.joblib で、グラフ（せん断速度\_粘度\_全厚み共通\_全温度）と同じ条件（樹脂圧力差は学習データの中央値、樹脂温度最高値は温度ごとの中央値）で計算した。
  \item カーネルの値と長さスケール：上の各モデルのフォルダの ハイパーパラメータ.csv。長さスケールを元の単位に直した値・予測の傾き・予測の平均の大きさ・残差は、同じフォルダの 学習済みモデル\_〈モデル〉.joblib（Git の管理外）から計算した。係数の格子：同じフォルダの プロファイル尤度\_格子.csv・シフトファクターの推定.csv。
  \item 縦だけと縦と横の比べ、樹脂温度でのシフト、C との比べ：各材料フォルダの 0N\_検討結果/〈材料〉\_…検討結果(5all学習データ)/全ショット版/全厚み総合/シフトファクターの検討/ の 候補の選び方.csv・まとめ.csv（本番学習の設定の決定.pdf の元。PP は直す前の樹脂圧力差で実行したもの）。
  \item 線形カーネルの有無：08\_カーネル関数の違いによる結果の違い/線形項あり・なしの比較表\_範囲別\_通常版.csv（11-11 の実行）。PP は 4回目\_…\_学習結果 と 4回目\_…\_学習結果\_線形項なし の 予測検証\_3data\_指標.csv（11-17 の実行）。
  \item 本体の B\_1：リポジトリ直下の results/step5/metrics.csv（10-14。5data 版、S\_V）。
  \item 文から引いた値：キャピラリーとの比べ（reports/2026-11-15 の報告ノート）、GPPS の型の比べと樹脂温度でのシフトの比べ（05\_本番学習の設定(検討結果説明書)/本番学習の設定の決定.pdf）、研究の背景（報告書B）、厚みによる粘度の違いと算出式の圧力差のリーク（CLAUDE.md の本体の検討）。
\end{itemize}
}
\end{document}
"""
    rep = {"@@DATE@@": rc.today_ja(), "@@ROWS@@": "\n".join(rows), "@@ROWS2@@": "\n".join(rows2), "@@QKER@@": N["Q_ker"],
           "@@BODY@@": "\n".join(body), "@@NKAK@@": str(n_kak),
           "@@R2@@": "・".join(f"{raw[m]['R2']:.3f}" for m in MATS), "@@EALL@@": N["E_ALL_RANGE"], "@@ELOW@@": N["E_LOW_RANGE"],
           "@@EHIGH@@": N["E_HIGH_RANGE"], "@@COV@@": N["COV_ALL_RANGE"], "@@ELOWPP@@": str(P["E_low"]),
           "@@QR2@@": N["Q_r2"], "@@QERR@@": N["Q_err"], "@@QINP@@": N["Q_inp"], "@@STEMTALK@@": STEM_TALK.replace("_", r"\_"),
           "@@NC_PP@@": str(P["NC_all"]), "@@NC_MIX@@": str(X["NC_all"]), "@@NC_GPPS@@": str(G["NC_all"]),
           "@@FIXH@@": N["P33FIX_H"], "@@FIXN@@": str(N["P33FIX_N"]), "@@FIXR@@": N["P33FIX_RATIO"],
           "@@R2PRE@@": N["R2PRE_PP"], "@@R2NEW@@": N["R2_PP"], "@@EPRE@@": str(N["EPRE_PP"]), "@@ENEW@@": str(N["E_all_PP"]),
           "@@ELPRE@@": str(N["ELPRE_PP"]), "@@ELNEW@@": str(N["E_low_PP"]), "@@EVPRE@@": N["EVPRE_PP"], "@@EVNEW@@": N["EV_PP"],
           "@@DLPRE@@": N["DLMLPRE_PP"], "@@DLNEW@@": N["DLML_PP"], "@@QE3@@": N["Q_e3"], "@@QHO@@": N["Q_ho"],
           "@@GAPSH@@": N["GAPSH_PP"], "@@GAP@@": N["GAP_PP"],
           "@@QOUT@@": N["Q_out"], "@@QWLF@@": N["Q_wlf"], "@@QFORM@@": N["Q_form"], "@@QALONE@@": N["Q_alone"], "@@QNEED@@": N["Q_need"],
           "@@QEWHAT@@": N["Q_ewhat"], "@@QECALC@@": N["Q_ecalc"]}
    for k, v in rep.items():
        tex = tex.replace(k, v)
    check("@@" not in tex, "質問対策の @@…@@ をすべて置き換えた")
    return tex, n, n_kak


def compile_tex(tex, stem, work):
    path = os.path.join(work, f"{stem}.tex")
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(tex)
    rel_tex = os.path.relpath(path, rc.W).replace("\\", "/")
    rel_out = os.path.relpath(work, rc.W).replace("\\", "/")
    for i in range(2):
        r = subprocess.run([rc.LUALATEX, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={rel_out}", rel_tex],
                           cwd=rc.W, capture_output=True)
        if r.returncode != 0:
            print(r.stdout.decode("utf-8", "replace")[-3000:])
            sys.exit(f"lualatex が失敗しました（{stem}、{i + 1} 回目）")
    log = open(os.path.join(work, f"{stem}.log"), encoding="utf-8", errors="replace").read()
    missing = len(re.findall(r"Missing character", log))
    over = len(re.findall(r"Overfull \\hbox", log))
    return path, os.path.join(work, f"{stem}.pdf"), missing, over


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT, help="tex と PDF を置くフォルダ（リポジトリ直下から）")
    ap.add_argument("--only", choices=["talk", "qa"], default=None, help="台本（talk）か質問対策（qa）の一方だけを作る")
    a = ap.parse_args()
    do_talk, do_qa = a.only in (None, "talk"), a.only in (None, "qa")
    out = os.path.join(rc.W, a.out)
    stems = [s for s, ok in ((STEM_TALK, do_talk), (STEM_QA, do_qa)) if ok]
    targets = [os.path.join(out, f"{s}.{e}") for s in stems for e in ("tex", "pdf")]
    for p in targets:
        if os.path.exists(p):
            sys.exit(f"すでにある：{p}（上書きしない。前のファイルを _to_delete/ へ移すか、--out で別のフォルダを指定）")
    N, raw = load()
    talk, n_all, n_short, t_all, t_short = build_talk(N)
    qa, n_q, n_kak = build_qa(N, raw)
    work = os.path.join(rc.W, "_tmp", "poster4_talk", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(work, exist_ok=False)
    res = ([compile_tex(talk, STEM_TALK, work)] if do_talk else []) + ([compile_tex(qa, STEM_QA, work)] if do_qa else [])
    os.makedirs(out, exist_ok=True)
    for tex_p, pdf_p, missing, over in res:
        for src in (tex_p, pdf_p):
            dst = os.path.join(out, os.path.basename(src))
            if os.path.exists(dst):
                sys.exit(f"すでにある：{dst}")
            shutil.copy2(src, dst)
        print(f"{os.path.basename(pdf_p)}：フォントにない文字 {missing}、はみ出し（Overfull）{over}")
    for title, pars in TALK:
        pars = [sub(p, N) for p in pars]
        print(f"  {title}：{sum(spoken_len(split_opt(p)[0]) for p in pars)} 字 ＋ 省略可 {sum(spoken_len(split_opt(p)[1]) for p in pars)} 字")
    print(f"台本：全部 {n_all} 字 → 約 {mmss(t_all)}、省略可を省くと {n_short} 字 → 約 {mmss(t_short)}（1 分 {CPM} 字）")
    print(f"質問対策：質問 {n_q} 個、要確認 {n_kak} か所" + ("" if do_qa else "（今回は作っていない）"))
    print("作業フォルダ:", os.path.relpath(work, rc.W))
    print("保存:", a.out)


if __name__ == "__main__":
    main()
