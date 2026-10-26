# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR（05 フォルダ）の説明書（用語集と図の見方）を作る。

使い方（リポジトリ直下から）: uv run python 04_補助スクリプト/build_report_mat_shift.py [--run-min 分]
前提：ノート（条件平均の入力・ショットごとの入力の両方）を実行済みで、mat_seed_check.py（cond）も実行済みであること。
各材料の <材料フォルダ>_学習結果/ の CSV から数値を取り出して report_mat_shift_template.tex に入れ、
05_材料ごとの機械学習データと予測検証データ/学習結果説明_材料別温度シフトGPR.tex を書き、そのフォルダで LuaLaTeX を2回動かす
（補助ファイルは _tmp/report_mat/ に出す）。図は、tex から相対パスで読む。
"""
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import report_common as rc

ROOT = os.path.join(rc.W, "05_材料ごとの機械学習データと予測検証データ")
FOLD = {"PP": "01_バージンPP再々実験", "MIX": "02_粉砕材混合PP実験", "GPPS": "03_GPPS実験"}
JA = {"PP": "PP", "MIX": "粉砕材混合 PP", "GPPS": "GPPS"}
MODES = {"cond": "条件平均の入力", "shot": "ショットごとの入力"}
KEYS = ["B_Tset", "B_Tff", "C"]
STEM = "学習結果説明_材料別温度シフトGPR"
V = {}
run_min = sys.argv[sys.argv.index("--run-min") + 1] if "--run-min" in sys.argv else "---"
C, Sh = "条件平均の入力", "ショットごとの入力"


def base(mat):
    return os.path.join(ROOT, FOLD[mat], f"{FOLD[mat]}_学習結果")


def rel(mat, *parts):
    """tex から見た図の相対パス（ROOT から）"""
    return "/".join([FOLD[mat], f"{FOLD[mat]}_学習結果", *parts])


def rd(mat, mode, sub, name):
    p = os.path.join(base(mat), MODES[mode], sub, name) if mode else os.path.join(base(mat), sub, name)
    return pd.read_csv(p, encoding="utf-8-sig")


def esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%")


MET = {(m, md): rd(m, md, "比較", "予測検証_3data_指標.csv") for m in FOLD for md in MODES}
SH = {(m, md): rd(m, md, "比較", "シフトファクターの推定.csv") for m in FOLD for md in MODES}
SC = {(m, md): rd(m, md, "比較", "シフトファクターの倍率.csv").set_index("モデル") for m in FOLD for md in MODES}
JD = {(m, md): rd(m, md, "比較", "判定の一覧.csv") for m in FOLD for md in MODES}
CW = {(m, md): rd(m, md, "比較", "CrossWLF_係数.csv") for m in FOLD for md in MODES}
HP = {(m, md): rd(m, md, "比較", "ハイパーパラメータ.csv") for m in FOLD for md in MODES}
SEED = {m: rd(m, "cond", "比較", "乱数シードの確認.csv") for m in FOLD}
INFO = {m: rd(m, None, "データ確認", "データの概要.csv") for m in FOLD}
OUTF = {m: rd(m, None, "データ確認", "外れ値の検知_規則f.csv") for m in FOLD}
P33 = {m: rd(m, None, "データ確認", "P3_3の性質.csv").iloc[0] for m in FOLD}
CAND = {m: rd(m, None, "データ確認", "問題の候補.csv") for m in FOLD}


def M(mat, mode, key, sub="全体", col="RMSE"):
    t = MET[(mat, mode)]
    r = t[(t["モデル"] == key) & (t["範囲"] == sub)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def S(mat, mode, key, col):
    t = SH[(mat, mode)]
    return t[t["モデル"] == key][col].iloc[0]


# ---- ファイルの数・判定の数
figs = csvs = 0
for m in FOLD:
    for dp, dn, fn in os.walk(base(m)):
        if "highres" in dp.split(os.sep) or not (MODES["cond"] in dp or MODES["shot"] in dp or "データ確認" in dp):
            continue                                              # 試運転の古い結果（入力の取り方のフォルダの外）は数えない
        figs += sum(f.endswith(".png") for f in fn)
        csvs += sum(f.endswith(".csv") for f in fn)
V["N_FIG"], V["N_CSV"] = str(figs), str(csvs)
nok = {md: sum(int((JD[(m, md)]["判定"] == "OK").sum()) for m in FOLD) for md in MODES}
nst = {md: sum(int((JD[(m, md)]["判定"] != "OK").sum()) for m in FOLD) for md in MODES}
V["N_OK"] = f"ショットごとの入力で {nok['shot']}・条件平均の入力で {nok['cond']}"
V["N_STAR"] = f"それぞれ {nst['shot']}・{nst['cond']}"
V["RUN_MIN"] = run_min
V["INPUT_MODE_TEXT"] = ("$\\gd$・\\PIII・$\\Tff$ の入れ方は 2 通り。\\textbf{ショットごとの入力}（\\file{INPUT_MODE = \"shot\"}。\\textbf{既定}。6月のノートと同じく、1 ショットの値をそのまま入れ、"
                        "1 ショットを 1 つのデータとして学習する。ユーザーの決定、10 月 23 日）と、\\textbf{条件平均の入力}（\\file{\"cond\"}。同じ条件のショットの平均。本体の手順3b の R3 と同じ考え方。"
                        "10 月 15〜22 日の既定。参考）。この版の説明書は、ショットごとの入力の結果を主に説明する。6 章に、2 つの比較を書いた")
P, R = "shot", "cond"                               # 主に説明する入力の取り方（P）と、参考に比べる取り方（R）
MAIN = "B_Tset"                                     # 主に説明するモデル（ショットごとの入力では、加熱シリンダ温度のシフトが支持された。発表資料の主のモデル）

# ---- データの表
rows = []
for m in FOLD:
    t = INFO[m].set_index("組")
    rows.append(f"{JA[m]} & {int(t.loc['5data', '使う行数'])} & {int(t.loc['5data', '条件数'])} & {int(t.loc['3data', '使う行数'])} & {int(t.loc['3data', '条件数'])} & "
                f"{t.loc['5data', '設定温度']} ℃、{t.loc['5data', '厚み']} mm \\\\")
V["TABLE_DATA"] = "\n".join(rows)

# ---- 外れ値の文
fl = OUTF["PP"][OUTF["PP"]["flag"]].drop_duplicates(subset=["cond"])
pp_z = float(fl["z"].iloc[0]) if len(fl) else np.nan
pp_r = float(fl["resid"].iloc[0]) if len(fl) else np.nan
n_f = {m: int(OUTF[m][OUTF[m]["flag"]]["cond"].nunique()) for m in FOLD}
n_c = {m: int(CAND[m]["規則"].str.startswith(("c ", "g ")).sum()) for m in FOLD}
V["OUTLIER_TEXT"] = (
    f"規則 f で拾った条件は、PP の 190 ℃・0.5 mm・320 mm/s（3data。$z={pp_z:.1f}$、$\\ln\\eta$ の差 ${pp_r:+.2f}$。粘度で約 {np.exp(pp_r):.2f} 倍）の 1 つだけでした"
    f"（粉砕材混合 PP {n_f['MIX']} 件、GPPS {n_f['GPPS']} 件）。この条件は、本体の検討でも同じ規則で拾い、「除外」にしたもの（P03）です。05 フォルダの問題データ一覧は 10 月 12 日時点の写しで、P03 が「感度確認」のままだったので、「除外」に直しました（本体の一覧は変えていません）。\n\n"
    "また、GPPS の 3data の 200 ℃・1.5 mm・3 mm/s で、3 ショット中 2 ショット（002・003）の \\PIII\\ が負の値（$-4.09$・$-17.50$ MPa）でした。対数がとれないので、\\textbf{P09} として「除外」に登録しました"
    "（そのショットだけ。残るショット 001 の \\PIII\\ も、同じ厚み・温度の 5 mm/s より大きく、計測が疑わしい。条件全体を除くかは、ユーザーが決めます）。PP の P01（1.5 mm・3 mm/s の圧力の計測ミス）と同じ厚み・射出速度です。"
    f"\n\nこのほか、見直しの候補（外れ値と判定したものではなく、一覧には登録していない）があります：同じ条件内のばらつきが大きい条件（規則 c）と、\\PIII\\ が条件の中央値から 20\\% 以上離れたショット（規則 g）が、"
    f"PP {n_c['PP']} 件、粉砕材混合 PP {n_c['MIX']} 件、GPPS {n_c['GPPS']} 件（\\file{{データ確認/問題の候補.csv}}）。PP の規則 c の 3 件は、10 月 12 日にユーザーが「登録しない」と決めた候補と同じです。")

# ---- 図
FIGS = []


def fig(title, paths, caption, what, axes, look, result, widths=("0.9",)):
    if isinstance(paths, str):
        paths = [paths]
    inc = "\n".join(f"\\includegraphics[width={w}\\linewidth]{{{p}}}" for p, w in zip(paths, widths))
    FIGS.append(f"""\\subsection{{{title}}}
\\begin{{figure}}[H]
\\centering
{inc}
\\caption{{{caption}}}
\\end{{figure}}
\\begin{{points}}
\\pt{{何を示す図か}}{what}
\\pt{{軸と記号}}{axes}
\\pt{{見るポイント}}{look}
\\pt{{今回の結果}}{result}
\\end{{points}}
""")


def sh(m, md, k):
    return S(m, md, k, "th_best"), S(m, md, k, "tv_best"), S(m, md, k, "diff_best"), S(m, md, k, "dLML_vs_0"), S(m, md, k, "diff_lo"), S(m, md, k, "diff_hi")


fig("図1：外れ値の検知（規則 f）", rel("PP", "データ確認", "外れ値の検知_規則f.png"), "PP の条件ごとの $z$（除く前の全条件）",
    "条件を 1 つずつ抜いて、ほかの条件から予測したときの、実測との差の大きさ（$z$）を、条件ごとに並べた図です（ファイル：\\file{データ確認/外れ値の検知_規則f.png}）。",
    "横軸は条件（厚み → 射出速度 → 設定温度の順）、縦軸は $z$。色と形は設定温度。破線が $\\pm3.5$。赤い丸が、拾った条件。",
    "破線の外に大きく飛び出す点があるか。",
    f"PP では、190 ℃・0.5 mm・320 mm/s だけが $z={pp_z:.1f}$ と大きく外れ、問題データ一覧の P03 として除きました。ほかの点は、ほぼ $\\pm3$ の中です。粉砕材混合 PP と GPPS は、拾った条件がありません。")
fig("図2：P3\\_3 の性質", rel("PP", "データ確認", "P3_3の性質.png"), "左：見かけのせん断応力 $\\eta\\gd$ と \\PIII、右：\\PIII\\ とせん断速度（色は厚み）",
    "\\PIII\\ が、粘度の算出式の圧力（リークになる量）かどうかを見る図です（ファイル：\\file{データ確認/P3_3の性質.png}）。",
    "どちらも両対数。左の縦軸は $\\eta\\gd$（kPa）、横軸は \\PIII（MPa）。右の横軸はせん断速度、縦軸は \\PIII。",
    "算出式の圧力なら、左の図で、厚みごとに点が 1 本の直線に乗る。",
    f"点は直線に乗らず、算出式の形で表すと誤差は最大 {P33['PP']['算出式の形で表したときの最大誤差 [%]']:.0f}\\%（粉砕材混合 PP {P33['MIX']['算出式の形で表したときの最大誤差 [%]']:.0f}\\%、GPPS {P33['GPPS']['算出式の形で表したときの最大誤差 [%]']:.0f}\\%）でした。\\PIII\\ は算出式の圧力ではありません。"
    f"右の図のとおり、\\PIII\\ はせん断速度と厚みでほぼ決まり（厚みが薄いほど大きい）、ln \\PII\\ との相関は {P33['PP'].get('相関 ln P3_3 と ln P3_2', np.nan):.2f}（PP）と、本体で調べた \\PII\\ に近い量です。\\PIII\\ は、流れの結果として測られる量で、成形の前には分かりません。")
pp_m, pp_o = sh("PP", P, MAIN), sh("PP", P, "B_Tff")
fig(f"図3：プロファイル尤度（PP・{esc(MAIN)}・ショットごとの入力）", rel("PP", Sh, MAIN, "プロファイル尤度.png"), "横のシフト $\\Eh$ と縦のシフト $\\Ev$ の尤度の面（左：全体、右：最良の点の近く）",
    f"シフトファクターの係数 $(\\Eh,\\Ev)$ を格子で振り、それぞれでデータへの合い具合（LML）を求めた図です（ファイル：\\file{{ショットごとの入力/{MAIN}/プロファイル尤度.png}}。他のモデル・材料・入力の取り方も同じ名前）。",
    "横軸が $\\Eh$、縦軸が $\\Ev$（kJ/mol）。色は LML − 最大値（明るいほどよく合う）。赤線の内側が、おおよその 95\\% 区間。★ が最良、× が $\\theta=0$（シフトなし）。",
    "赤線の内側が狭いか（係数がよく決まるか）。斜めに伸びているか（尾根）。× が赤線の外にあるか（シフトを入れた効果があるか）。",
    f"最良は $\\Eh={pp_m[0]:g}$、$\\Ev={pp_m[1]:g}$ kJ/mol で、横のシフトは 0（ずらさない）になりました。念入りに最適化した $\\theta=0$ との LML の差（$\\Delta$LML）は {pp_m[3]:.1f} で、基準の 3.0 を超えます。"
    f"B\\_Tff の $\\Delta$LML は {pp_o[3]:.1f} で、基準に届きません（5.1 節の表）。条件平均の入力（参考）では逆に、B\\_Tff の $\\Delta$LML が {S('PP', R, 'B_Tff', 'dLML_vs_0'):.1f}、"
    f"B\\_Tset が {S('PP', R, 'B_Tset', 'dLML_vs_0'):.1f} でした。", widths=("1.0",))
sc_p = SC[("PP", P)]
fig("図4：シフトファクター（PP・ショットごとの入力）", rel("PP", Sh, "比較", "シフトファクター.png"), "横のシフト $\\ln a_h(T)$（左）と縦のシフト $\\ln a_v(T)$（右）。線が最良、帯が 95\\% 区間",
    "推定したシフトファクターを、温度の関数として描いた図です（ファイル：\\file{<入力>/比較/シフトファクター.png}）。",
    "横軸が温度（B\\_Tset は設定温度、B\\_Tff は流動中樹脂温度）。縦軸が $\\ln a$。点線は設定温度、薄い橙は $\\Tff$ の範囲。$\\ln a_v>0$ は、その温度で、同じせん断速度の粘度が $T_{\\mathrm{ref}}$ より高いこと。",
    "線の傾き（温度による変化の大きさ）と、帯の幅（決まり方）。右肩下がりなら「低温ほど粘い」。",
    f"横のシフトは 0（線が水平）で、縦のシフトだけが右肩下がり（低温ほど粘い）です。PP の 190 ℃ と 230 ℃ の縦の倍率 $a_v(190)/a_v(230)$ は、B\\_Tset で {sc_p.loc['B_Tset', '縦の倍率 a_v(190)/a_v(230)']:.2f}、"
    f"B\\_Tff で {sc_p.loc['B_Tff', '縦の倍率 a_v(190)/a_v(230)']:.2f} です。5.1 節の表に、全材料・全モデルの値をまとめました。")
fig(f"図5：マスターカーブ（PP・{esc(MAIN)}・ショットごとの入力）", rel("PP", Sh, MAIN, "マスターカーブ.png"), f"{esc(MAIN)} の最良の係数でずらした条件平均の点（厚みごと。シフト前は \\file{{比較/流動曲線_シフト前.png}}）",
    f"温度ごとの曲線を、シフトファクターでずらして重ねた図です（ファイル：\\file{{ショットごとの入力/{MAIN}/マスターカーブ.png}}）。点は、見やすさのため成形条件ごとの平均です。",
    "横軸が換算せん断速度 $\\gd a_h$、縦軸が換算粘度 $\\eta/a_v$（両対数）。色が設定温度。塗りが学習、白抜きが検証。",
    "温度の違う点が、1 本の曲線に重なるか。図の上の「重ね合わせの残差」が、シフト前より小さいか。",
    f"PP の {esc(MAIN)} で、重ね合わせの残差は {sc_p.loc[MAIN, '重ね合わせの残差（シフト前）']:.3f} → {sc_p.loc[MAIN, '重ね合わせの残差（シフト後）']:.3f} に減り、温度ごとの点が 1 本の曲線に近づきました。"
    f"B\\_Tff では {sc_p.loc['B_Tff', '重ね合わせの残差（シフト前）']:.3f} → {sc_p.loc['B_Tff', '重ね合わせの残差（シフト後）']:.3f} です（5.1 節の表）。")
fig(f"図6：せん断速度 ― 粘度の学習結果（PP・厚み 2.0 mm。ショットごとの入力）", rel("PP", Sh, MAIN, "せん断速度_粘度", "せん断速度_粘度_厚みごと_2mm.png"),
    f"{esc(MAIN)} の予測曲線と 95\\% 予測区間（色が設定温度）",
    "6月のノートのセル9-2 と同じ見せ方の図です（ファイル：\\file{<入力>/<モデル>/せん断速度_粘度/}。設定温度ごと・厚みごと・全厚み共通の図があります）。",
    "横軸がせん断速度、縦軸が粘度（両対数）。線が予測の平均、帯が 95\\% 予測区間。塗りの点が学習（5data）、白抜きが検証（3data）で、どちらもショットごとの計測値（1 つの成形条件に、ショットの数だけ点があります）。"
    "目盛りは、軸の範囲が 2.5 桁以下なら 1・2・5 × $10^n$、それより広いとき（または目盛りの数字が重なるとき）は $10^n$ だけです（10 月 26 日から。この図は両軸とも $10^n$）。",
    "白抜きの点（学習していない低速側・高速側）が、線の近く・帯の中にあるか。曲線が滑らかか。",
    f"ショットごとの入力では、曲線が条件の間で細かく波打ち、低速側（左端の白抜き）の予測は実測から外れやすくなります（PP の {esc(MAIN)} の低速側の RMSE：ショットごと {M('PP', P, MAIN, '低速側'):.3f}、"
    f"条件平均 {M('PP', R, MAIN, '低速側'):.3f}）。ショットごとの $\\gd$ には、同じ条件の中のゆれ（センサへの到達時刻のゆれ。$\\ln\\eta$ と $\\ln\\gd$ が傾き約 $-2$ で一緒に動く）が入っていて、"
    f"GP がそれに合わせて、$\\ln\\gd$ の長さスケールを短くする（C で {float(HP[('PP', P)].set_index('model').loc['C', 'ls_ln γ̇']):.2f}。条件平均では {float(HP[('PP', R)].set_index('model').loc['C', 'ls_ln γ̇']):.1f}）ためです。次の図7（条件平均の入力）と比べてください。")
fig("図7：同じ図（条件平均の入力。参考）", rel("PP", C, MAIN, "せん断速度_粘度", "せん断速度_粘度_厚みごと_2mm.png"), f"条件平均の入力の {esc(MAIN)}（PP・厚み 2.0 mm）",
    "図6と同じ図を、条件平均の入力（10 月 15〜22 日の既定）で描いたものです。",
    "図6と同じ。ただし、点は成形条件ごとの平均です（このときの学習に合わせた点です）。目盛りは、10 月 26 日の変更の前の付け方（1・2・5 × $10^n$）のままです。",
    "曲線の形と、低速側の白抜きの点と線のずれ。",
    "条件平均の入力では、曲線は滑らかで、低速側の点も帯の近くにあります。本体の手順3b で、$\\gd$ を条件平均で入れると決めた理由と同じです。10 月 23 日に、ユーザーの決定で、ノートの既定をショットごとに変えました。")
fig(f"図8：全厚み共通の平均関数と予測幅（PP・{esc(MAIN)}・ショットごとの入力）",
    [rel("PP", Sh, MAIN, "せん断速度_粘度", "せん断速度_粘度_全厚み共通_全温度.png"), rel("PP", Sh, MAIN, "せん断速度_粘度", "せん断速度_粘度_全厚み共通_全温度_凡例.png")],
    "加熱シリンダ温度ごとの、全厚み共通の平均関数と 95\\% 予測区間（左：グラフ、右：凡例の画像。線・帯の色は 190 ℃ が赤、210 ℃ が青、230 ℃ が緑）",
    "6月のノートのセル9-2 と同じ見せ方で、全キャビティ厚みの学習データの点（ショットごとの計測値。色・形は厚み）を重ね、温度ごとの平均関数と予測幅を、1 つのグラフで比べる図です（ファイル：\\file{せん断速度_粘度_全厚み共通_全温度.png} と \\file{…_凡例.png}。温度ごとの図 \\file{…_全厚み共通_190C.png} なども同じフォルダにあります）。",
    "両軸とも 10 の累乗の目盛り（6月と同じ）。曲線は、せん断速度だけを動かし、\\PIII\\ は学習データの中央値、$\\Tff$ は、その設定温度の学習データの中央値に固定したもの（固定した値は、凡例の画像に書いてあります）。",
    "温度ごとの曲線と帯が、どれくらい離れているか。点（厚みごと）が曲線のどちら側に散らばるか。",
    "\\textbf{読むときの注意}：入力が 4 つあるので、1 本の曲線にするには、ほかの入力を固定する必要があります。厚みの違いは \\PIII\\ を通して入るので、厚み 0.5 mm の点は、共通の曲線の上に離れて並びます。"
    "また、温度ごとの曲線の違いには、$\\Tset$ の効果と、固定した $\\Tff$ の違い（約 20 ℃ ずつ）の両方が入っています。\\textbf{この図だけでは、シフトの証拠にはなりません}。"
    "シフトの根拠は、図3〜5（尤度、シフトファクター、マスターカーブ）と 5.1 節の表で見ます。", widths=("0.62", "0.3"))
fig(f"図9：3data の予測検証（PP・{esc(MAIN)}・ショットごとの入力）", rel("PP", Sh, MAIN, "予測検証_3data_パリティ.png"), "横軸が実測、縦軸が予測の粘度。縦線が 95\\% 予測区間",
    "学習していない 3data（射出速度の両端）を予測した結果です。6月のノートのセル10 と同じ図です（ファイル：\\file{<入力>/<モデル>/予測検証_3data_パリティ.png}）。",
    "両対数。青が低速側（3・5 mm/s）、赤が高速側（280・320 mm/s）。破線が「予測 = 実測」。",
    "点が破線の近くにあるか。縦線が破線をまたぐか（予測区間に入っているか）。",
    f"ショットごとの入力の {esc(MAIN)} で、RMSE は {M('PP', P, MAIN):.3f}、粘度の実数の $R^2$ は {M('PP', P, MAIN, col='R2_real'):.3f}、95\\% 予測区間に入る割合は {M('PP', P, MAIN, col='coverage95'):.2f} です。"
    f"条件平均の入力では、それぞれ {M('PP', R, MAIN):.3f}・{M('PP', R, MAIN, col='R2_real'):.3f}・{M('PP', R, MAIN, col='coverage95'):.2f} です（5.2 節の表）。低速側（左上の青）が、予測で大きく外れます。", widths=("0.7",))
fig("図10：3data の予測誤差の比較（PP・ショットごとの入力）", rel("PP", Sh, "比較", "予測検証_RMSEの比較.png"), "モデルごとの RMSE（全体・低速側・高速側）",
    "全モデルと Cross-WLF の 3data の誤差を、範囲ごとに並べた図です（ファイル：\\file{<入力>/比較/予測検証_RMSEの比較.png}）。",
    "縦軸が RMSE（$\\ln\\eta$。小さいほどよい）。",
    "シフトのあるモデル（B\\_Tset・B\\_Tff）が、基準の C と比べてどうか。",
    f"高速側（学習範囲の外の 280・320 mm/s）は、3 モデルとも同じくらい（{min(M('PP', P, k, '高速側') for k in KEYS):.3f}〜{max(M('PP', P, k, '高速側') for k in KEYS):.3f}）です。"
    f"低速側（3・5 mm/s。外挿）は、3 モデルとも誤差が大きく（{min(M('PP', P, k, '低速側') for k in KEYS):.3f}〜{max(M('PP', P, k, '低速側') for k in KEYS):.3f}）、モデルの差は小さいです（5.2 節の表）。", widths=("0.8",))
fig("図11：乱数シードを変えて学習し直したときの、LML と 3data の RMSE（PP・条件平均の入力）", rel("PP", C, "比較", "乱数シードの確認.png"), "点が 1 回の学習（乱数シード 5 通り、最適化のやり直し 20 回。係数 $\\theta$ は固定）",
    "最適化の初期値（乱数シード）を変えて、モデルを学習し直したときの、対数周辺尤度（LML）と 3data の RMSE の関係です（補助スクリプト \\file{mat\\_seed\\_check.py}。ファイル：\\file{<入力>/比較/乱数シードの確認.png}）。\\textbf{条件平均の入力でだけ実施}しました（ショットごとの入力では未実施）。",
    "横軸が LML（右ほど、データによく合う）、縦軸が RMSE（左：全体、右：低速側。小さいほどよい）。色と形がモデル。",
    "同じモデルの点が、同じ LML・同じ RMSE に集まるか。LML がほぼ同じなのに、RMSE が大きく違う点があるか。",
    "同じモデルでも、最適化が着く解によって、LML が 1 前後違うだけで、3data の（特に低速側の）RMSE が大きく変わる場合があります（5.5 節の表）。"
    "したがって、モデルどうしの 3data の RMSE の差は、この変動の範囲と比べて読む必要があります。", widths=("1.0",))
fig(f"図12：依存性グラフ（PP・{esc(MAIN)}・ショットごとの入力）", rel("PP", Sh, MAIN, "依存性グラフ.png"), "1 つの説明変数だけを動かしたときの予測（ほかは学習データの中央値、設定温度は $T_{\\mathrm{ref}}$）",
    "6月のノートのセル9 と同じ図です。各説明変数の効果の向きと大きさを見ます（ファイル：\\file{<入力>/<モデル>/依存性グラフ.png}）。",
    "縦軸は粘度（対数）。赤線が予測の平均、帯が 95\\% 予測区間、点は実測（色は厚み）。右下から 2 番目の図は、$\\Tset$ と $\\Tff$ を同時に動かしたもの。",
    "せん断速度を上げると粘度が下がるか、温度を上げると粘度が下がるか。",
    "ほかの変数を固定しているので、実際には起こらない組み合わせも含みます。$\\Tset$ だけ・$\\Tff$ だけを動かすと、もう一方が固定のため、効果が小さく・逆向きに出ることがあります。$\\Tset$ と $\\Tff$ を同時に動かした図で、「温度を上げると粘度が下がる」ことを確かめます（ノートの判定）。", widths=("1.0",))
fig(f"図13：3次元グラフ（PP・{esc(MAIN)}・ショットごとの入力）", rel("PP", Sh, MAIN, "3次元グラフ", "3次元_gd_Tset.png"), "せん断速度と設定温度を動かしたときの予測の粘度",
    "6月のノートのセル12〜15 と同じ図です（ファイル：\\file{<入力>/<モデル>/3次元グラフ/}。せん断速度と、設定温度・\\PIII・$\\Tff$ の組み合わせの 3 枚）。",
    "横の 2 軸が動かした変数（せん断速度は常用対数）、縦軸が粘度の常用対数。",
    "温度の向きの傾きが、どれくらいあるか。", "依存性グラフと同じく、ほかの変数は固定です。", widths=("0.75",))
fig("図14：Cross-WLF との比較（PP・ショットごとの入力）", rel("PP", Sh, "比較", "CrossWLFとの比較_パリティ.png"), "同じ 3data での、Cross-WLF（青）と GPR（赤）の予測",
    "6月のノートのセル11 と同じ比較です（ファイル：\\file{<入力>/比較/CrossWLFとの比較_パリティ.png}）。",
    "横軸が実測、縦軸が予測の粘度（両対数）。",
    "どちらが破線に近いか。",
    f"Cross-WLF の RMSE は {M('PP', P, 'Cross-WLF'):.3f} で、GPR（{esc(MAIN)}、{M('PP', P, MAIN):.3f}）より大きい誤差です。Cross-WLF は、温度とせん断速度だけの式で、厚みの違い（\\PIII\\ に入る）を表せません。初期値を 12 通り変えても、係数は探索範囲の端に着き、同じ結果でした。あくまで参考です。", widths=("0.7",))
g_m, g_o = sh("GPPS", P, MAIN), sh("GPPS", P, "B_Tff")
fig("図15：GPPS のシフトファクター（WLF 型。ショットごとの入力）", rel("GPPS", Sh, "比較", "シフトファクター.png"), "GPPS の $\\ln a_h(T)$（左）と $\\ln a_v(T)$（右）",
    "非晶性の GPPS は、WLF 型でシフトファクターを表しました（$A_2=51.6$ K、$D_2=T_g$〔仮置き 100 ℃〕固定）。",
    "図4と同じ。係数は $A_{1,h}$・$A_{1,v}$。",
    "図4と同じ。",
    f"B\\_Tset は $A_{{1,h}}={g_m[0]:g}$、$A_{{1,v}}={g_m[1]:g}$（$\\Delta$LML {g_m[3]:.1f}）、"
    f"B\\_Tff は $A_{{1,h}}={g_o[0]:g}$、$A_{{1,v}}={g_o[1]:g}$（$\\Delta$LML {g_o[3]:.1f}）でした。$T_g$ は仮置きなので、実際の値が分かれば、設定 \\file{{SHIFT_CFG}} で直して実行し直します。")
fig(f"図16：特徴スケール（PP・{esc(MAIN)}・ショットごとの入力）", rel("PP", Sh, MAIN, "特徴スケール.png"), "特徴スケール（重要度 = 1／長さスケール）。損失関数の推移とカーネル行列は、同じフォルダの \\file{損失関数の推移.png}・\\file{カーネル行列.png}",
    "6月のノートのセル7・7.5・8 と同じ図です。学習がうまく進んだか、どの入力に粘度が敏感かを見ます。",
    "棒の長さが重要度（大きいほど、その入力の変化に粘度が敏感）。",
    "どの入力の重要度が大きいか。シフトのあるモデルでは、シフトに使った温度の効果は係数 $\\theta$ が担うので、ここには出ない。",
    "ショットごとの入力では、換算せん断速度の重要度が、ほかの入力よりずっと大きくなります（長さスケールが短い）。5.3 節の表に、全モデルのハイパーパラメータをまとめました。", widths=("0.7",))
V["FIGURES"] = "\n".join(FIGS)

# ---- 表
rows = []
for md in ["shot", "cond"]:
    for m in FOLD:
        for k in ["B_Tset", "B_Tff"]:
            nm, un = S(m, md, k, "θ の名前"), S(m, md, k, "単位")
            rows.append(f"{MODES[md] if (m == 'PP' and k == 'B_Tset') else ''} & {JA[m] if k == 'B_Tset' else ''} & {esc(k)} & {S(m, md, k, 'th_best'):g}（{S(m, md, k, 'th_lo'):g}〜{S(m, md, k, 'th_hi'):g}） & "
                        f"{S(m, md, k, 'tv_best'):g}（{S(m, md, k, 'tv_lo'):g}〜{S(m, md, k, 'tv_hi'):g}） & {S(m, md, k, 'diff_best'):g}（{S(m, md, k, 'diff_lo'):g}〜{S(m, md, k, 'diff_hi'):g}） & "
                        f"{S(m, md, k, 'dLML_vs_0'):.1f} & {SC[(m, md)].loc[k, '重ね合わせの残差（シフト前）']:.3f}→{SC[(m, md)].loc[k, '重ね合わせの残差（シフト後）']:.3f} \\\\")
    rows.append("\\midrule")
rows[-1] = ""
V_TAB_SHIFT = "\n".join(rows)
rows = []
for md in ["shot", "cond"]:
    for m in FOLD:
        for k in KEYS + ["Cross-WLF"]:
            rows.append(f"{MODES[md] if (m == 'PP' and k == 'B_Tset') else ''} & {JA[m] if k == 'B_Tset' else ''} & {esc(k)} & {M(m, md, k):.3f} & {M(m, md, k, '低速側'):.3f} & {M(m, md, k, '高速側'):.3f} & "
                        f"{M(m, md, k, col='coverage95'):.2f} & {M(m, md, k, col='R2_real'):.3f} \\\\")
        rows.append("\\midrule")
rows[-1] = ""
V_TAB_MET = "\n".join(rows)
rows = []
for m in FOLD:
    h = HP[(m, P)]
    for _, r in h.iterrows():
        ls = "、".join(f"{c[3:]} {r[c]:.3g}" for c in h.columns if c.startswith("ls_") and c != "ls_at_bound" and pd.notna(r[c]))
        rows.append(f"{JA[m] if r['model'] == 'B_Tset' else ''} & {esc(r['model'])} & {r['LML']:.1f} & {r['amplitude']:.3g} & {r.get('lin_amplitude', np.nan):.3g} & {r['noise_sd']:.3f} & {esc(ls)} \\\\")
    rows.append("\\midrule")
rows[-1] = ""
V_TAB_HP = "\n".join(rows)
rows = []
for m in FOLD:
    t = CW[(m, P)]
    rows.append(f"{JA[m]} & " + "、".join(f"{esc(r['係数'])} = {r['値']:.4g}（{r['扱い']}）" for _, r in t.iterrows()) + " \\\\")
V_TAB_CW = "\n".join(rows)
rows = []
for m in FOLD:
    g = SEED[m].groupby("モデル", sort=False).agg(a=("LML", "max"), b=("LML", "min"), c=("RMSE 全体", "min"), d=("RMSE 全体", "max"), e=("RMSE 低速側", "min"), f=("RMSE 低速側", "max"))
    for i, (k, r) in enumerate(g.iterrows()):
        used = HP[(m, "cond")].set_index("model").loc[k, "LML"]
        rows.append(f"{JA[m] if i == 0 else ''} & {esc(k)} & {r['b']:.1f}〜{r['a']:.1f} & {r['c']:.3f}〜{r['d']:.3f} & {r['e']:.3f}〜{r['f']:.3f} & {M(m, 'cond', k):.3f}（{M(m, 'cond', k, '低速側'):.3f}） \\\\")
    rows.append("\\midrule")
rows[-1] = ""
V_TAB_SEED = "\n".join(rows)
V["TABLES"] = f"""
\\subsection{{シフトファクターの推定}}
\\begin{{table}}[H]
\\caption{{最良の係数（括弧：おおよその 95\\% 区間）、差 $\\theta_v-\\theta_h$、念入りに最適化した LML の、$\\theta=0$ との差 $\\Delta$LML、重ね合わせの残差（シフト前→後）。係数は、PP・粉砕材混合 PP が $\\Eh$・$\\Ev$（kJ/mol）、GPPS が $A_{{1,h}}$・$A_{{1,v}}$}}\\label{{tab:shift}}
\\scriptsize
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}lllXXXrl@{{}}}}
\\toprule
入力 & 材料 & モデル & 横 $\\theta_h$ & 縦 $\\theta_v$ & 差 & $\\Delta$LML & 重ね合わせ \\\\
\\midrule
{V_TAB_SHIFT}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}
\\begin{{points}}
\\pt{{読み方}}$\\Delta$LML が 3.0 以上なら、その温度をシフトの形で入れる効果が、尤度比検定の 5\\% 基準で認められます。小さければ、シフトなし（その温度を使わない）でも、ほとんど同じくらいデータに合います。尾根があるので、個々の係数の区間は、計算した範囲に左右されます。差 $\\theta_v-\\theta_h$ が、最もよく決まる量です。ショットごとの入力（既定）では、$\\theta_h$ はほぼすべて 0 で、B\\_Tset の縦のシフトだけが支持されました。ショットごとの尤度は、設定温度の目印が入力に入る影響も受けます（本体の手順5a の注意）。条件平均の入力（参考）では、B\\_Tff が支持されました。
\\end{{points}}

\\subsection{{3data の予測検証}}
\\begin{{table}}[H]
\\caption{{3data の RMSE（$\\ln\\eta$）、95\\% 予測区間に入る割合（被覆率）、粘度の実数の $R^2$（6月のノートと同じ）}}\\label{{tab:met}}
\\scriptsize
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}lllrrrrr@{{}}}}
\\toprule
入力 & 材料 & モデル & 全体 & 低速側 & 高速側 & 被覆率 & $R^2$（実数） \\\\
\\midrule
{V_TAB_MET}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}
\\begin{{points}}
\\pt{{読み方}}RMSE は小さいほどよい（0.1 は粘度で約 10\\%）。被覆率は 0.95 に近いほどよい。$R^2$（実数）は、粘度の大きい低速側の誤差に強く左右されます。Cross-WLF は、6月のノートと同じく温度に $\\Tff$ を使い、一部の係数を固定して 5data に当てはめたもの（5.4 節）。\\textbf{{ショットごとの入力は、低速側の誤差が大きい}}（条件平均の 2〜5 倍）です。
\\end{{points}}

\\subsection{{ハイパーパラメータ（ショットごとの入力）}}
\\begin{{table}}[H]
\\caption{{最終モデルのハイパーパラメータ（振幅は $\\sqrt{{\\text{{振幅}}^2}}$、長さスケールは標準化した入力に対する値）}}\\label{{tab:hp}}
\\scriptsize
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}llrrrrX@{{}}}}
\\toprule
材料 & モデル & LML & 振幅 & 線形項の振幅 & ノイズ $\\sigma_n$ & 長さスケール \\\\
\\midrule
{V_TAB_HP}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}
\\begin{{points}}
\\pt{{読み方}}ノイズの標準偏差（約 0.006〜0.007）は、同じ条件のショット間のばらつき（約 0.04〜0.05）よりずっと小さく出ます。ショットごとの $\\gd$ が、同じ条件の中の粘度のゆれの大部分を説明するためです（条件平均の入力では、ノイズの標準偏差がショット間のばらつきとほぼ同じ）。長さスケールが 100（上限）に届いた入力は、粘度にほとんど効いていない（線形項が担う）入力です。振幅と線形項の振幅は、本体の手順5b の MCMC でも、データでは決まりにくいことが分かっています。
\\end{{points}}

\\subsection{{Cross-WLF の係数（ショットごとの入力のノートで当てはめたもの）}}
\\begin{{table}}[H]
\\caption{{Cross-WLF の係数（$D_3=0$。$D_2$ は K。「範囲の端」は、探索の範囲の端に届いた係数）}}\\label{{tab:cw}}
\\scriptsize
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}lX@{{}}}}
\\toprule
材料 & 係数 \\\\
\\midrule
{V_TAB_CW}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}

\\subsection{{乱数シードの確認（条件平均の入力）}}
\\begin{{table}}[H]
\\caption{{乱数シード 5 通り（最適化のやり直し 20 回。係数 $\\theta$ は固定）で学習し直したときの、LML と 3data の RMSE の範囲。右端は、ノートの最終モデルの値（括弧：低速側）}}\\label{{tab:seed}}
\\scriptsize
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}llXXXX@{{}}}}
\\toprule
材料 & モデル & LML & RMSE 全体 & RMSE 低速側 & ノートの最終モデル \\\\
\\midrule
{V_TAB_SEED}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}
\\begin{{points}}
\\pt{{読み方}}LML の範囲が狭く、RMSE の範囲も狭ければ、最適化は安定しています。LML の範囲が 1 前後でも RMSE の範囲が広ければ、「LML がほぼ同じ解が複数あり、予測（特に外挿）が違う」ことを表します。
\\end{{points}}
"""

# ---- 結論（主に説明するのはショットごとの入力 P。条件平均の入力 R は参考）
LOW = {md: {m: {k: M(m, md, k, "低速側") for k in KEYS} for m in FOLD} for md in MODES}
ALL = {md: {m: {k: M(m, md, k) for k in KEYS} for m in FOLD} for md in MODES}
HIGH = {m: [M(m, P, k, "高速側") for k in KEYS] for m in FOLD}
dl = {md: {m: {k: S(m, md, k, "dLML_vs_0") for k in ["B_Tset", "B_Tff"]} for m in FOLD} for md in MODES}
best = {md: {m: min(KEYS, key=lambda k: ALL[md][m][k]) for m in FOLD} for md in MODES}
seed_rng = {m: SEED[m].groupby("モデル", sort=False)["RMSE 全体"].agg(["min", "max"]) for m in FOLD}
all_ratio = [seed_rng[m].loc[k, "max"] / seed_rng[m].loc[k, "min"] for m in FOLD for k in KEYS]
hp_c = {md: {m: HP[(m, md)].set_index("model").loc["C"] for m in FOLD} for md in MODES}
ls_gd = {md: [float(hp_c[md][m]["ls_ln γ̇"]) for m in FOLD] for md in MODES}
nsd = {md: [float(hp_c[md][m]["noise_sd"]) for m in FOLD] for md in MODES}
f3 = lambda d, k, fmt=".1f": "・".join(f"{d[m][k]:{fmt}}" for m in FOLD)
no_h = all(float(S(m, P, MAIN, "th_best")) == 0 for m in FOLD)
star_txt = "・".join(sorted(set(pd.concat([JD[(m, P)][JD[(m, P)]["判定"] != "OK"]["節"] for m in FOLD]))))
V["CONCLUSION"] = f"""
ノートの判定は、OK {V['N_OK']} 個、★ {V['N_STAR']} 個でした。以下、主にショットごとの入力（既定）の結果をまとめ、条件平均の入力（参考）と比べます。

\\begin{{enumerate}}[itemsep=0.8ex,topsep=0.6ex]
\\item \\textbf{{プログラムと結果}}：6月のノートを基準に、材料ごとのデータ（5data で学習、3data で検証）を一人で実行できるノートを作りました。結果は、各材料フォルダの \\file{{<材料フォルダ>_学習結果/<入力の取り方>/}} にあります。ノートは 3 材料 $\\times$ 3 モデルを学習し、ショットごとの入力で約 {run_min} 分（そのうち、大部分はプロファイル尤度）かかります。
\\item \\textbf{{データと外れ値}}：規則 f で拾ったのは、PP の P03 の 1 条件だけで、除外しました。GPPS の 3data には \\PIII\\ が負のショットがあり（P09）、除外しました。\\PIII\\ は、粘度の算出式の圧力ではありませんが、流れの結果として測られる量で、成形の前には分かりません。このモデルは、「測った流れの状態（$\\gd$・\\PIII・$\\Tff$）と設定温度 $\\rightarrow$ 粘度」のモデルです。
\\item \\textbf{{入力の取り方（ユーザーの決定、10 月 23 日）}}：ショットごとの値で学習する仕様にしました（既定 \\file{{"shot"}}）。ショットごとの入力では、同じ条件の中のせん断速度と粘度のゆれ（センサへの到達時刻のゆれ）に GP が合わせるので、$\\ln\\gd$ の長さスケールが短く（C で {'・'.join(f'{v:.2f}' for v in ls_gd[P])}。条件平均では {'・'.join(f'{v:.1f}' for v in ls_gd[R])}）、ノイズの標準偏差も小さく出ます（{'・'.join(f'{v:.3f}' for v in nsd[P])}。条件平均では {'・'.join(f'{v:.3f}' for v in nsd[R])}）。その結果、曲線がデータの範囲の中でも波打ち、3data の低速側の誤差が大きくなりました（C の RMSE：ショットごと {f3(LOW[P], 'C', '.3f')}、条件平均 {f3(LOW[R], 'C', '.3f')}。PP・粉砕材混合 PP・GPPS）。
\\item \\textbf{{温度のシフト}}：ショットごとの入力では、\\textbf{{加熱シリンダ温度 $\\Tset$ でずらす B\\_Tset のシフトが支持され}}（$\\theta=0$ との LML の差：{f3(dl[P], 'B_Tset')}。基準 3.0）、\\textbf{{流動中樹脂温度 $\\Tff$ でずらす B\\_Tff は支持されませんでした}}（{f3(dl[P], 'B_Tff')}）。{'B\\_Tset の横の係数は 3 材料とも 0 で、縦のシフト（同じせん断速度での粘度の倍率）だけが支持されました。' if no_h else ''}条件平均の入力では逆で、B\\_Tff が支持され（{f3(dl[R], 'B_Tff')}）、B\\_Tset は基準に届かないか、ぎりぎりでした（{f3(dl[R], 'B_Tset')}）。\\textbf{{どちらの温度がシフトに適するかは、入力の取り方に左右されます}}。ショットごとの入力では、ショットごとに変わる $\\Tff$ が GP の入力として粘度の違いを担うため、と考えられます。
\\item \\textbf{{シフトの解釈の注意}}：$\\Tff$ は、せん断速度と連動して動きます（低速ほど低く、高速ほど高い。せん断発熱）。$\\Tff$ でずらすシフトは、「設定温度の違い」だけでなく「見かけの粘度に含まれる、せん断発熱の効果」も拾いうるので、どちらの入力の取り方でも、温度の効果をシフトとして取り出すのは難しい面があります。総合グラフ（図8）は、\\PIII\\ と $\\Tff$ を固定した断面で、シフトの証拠にはなりません。
\\item \\textbf{{3data の予測}}：高速側（280・320 mm/s の外挿）は、3 モデルともほぼ同じ精度です（RMSE {min(min(v) for v in HIGH.values()):.3f}〜{max(max(v) for v in HIGH.values()):.3f}）。低速側（3・5 mm/s の外挿）は誤差が大きく、全体の RMSE が最小のモデルは、PP {esc(best[P]['PP'])}（{ALL[P]['PP'][best[P]['PP']]:.3f}）、粉砕材混合 PP {esc(best[P]['MIX'])}（{ALL[P]['MIX'][best[P]['MIX']]:.3f}）、GPPS {esc(best[P]['GPPS'])}（{ALL[P]['GPPS'][best[P]['GPPS']]:.3f}）でした。条件平均の入力の最小（{'・'.join(f"{ALL[R][m][best[R][m]]:.3f}" for m in FOLD)}）より大きい誤差です。乱数シードを変えた確認（5.5 節。条件平均の入力で実施）では、同じモデルでも全体の RMSE が最大で最小の {max(all_ratio):.1f} 倍まで変わったので、モデルどうしの差は、この変動と比べて読む必要があります（ショットごとの入力では、まだ実施していません）。
\\item \\textbf{{Cross-WLF との比較}}：参考の Cross-WLF は、RMSE が {M('PP', P, 'Cross-WLF'):.2f}（PP）・{M('MIX', P, 'Cross-WLF'):.2f}（粉砕材混合 PP）・{M('GPPS', P, 'Cross-WLF'):.2f}（GPPS）で、GPR（{min(min(ALL[P][m].values()) for m in FOLD):.2f}〜{max(max(ALL[P][m].values()) for m in FOLD):.2f}）より大きい誤差です。厚みの違いを表せないためと考えられます。
\\item \\textbf{{ノートの ★ の意味}}：ショットごとの入力では、主に {star_txt} の節で、B\\_Tff の $\\Delta$LML が基準に届かない、縦のシフトの向きや重ね合わせの残差が期待どおりでない（B\\_Tff の係数がほぼ 0 のため）、という注意が出ました。エラーではなく、結果を読むときの注意点です。
\\end{{enumerate}}

\\subsection{{これからの選択肢（ユーザーの判断）}}
\\begin{{enumerate}}[itemsep=0.6ex,topsep=0.6ex]
\\item \\textbf{{入力の取り方}}：ノートの既定はショットごとです（10 月 23 日）。3data の予測誤差は、条件平均の入力のほうが小さくなりました。条件平均に戻すときは、設定 \\file{{INPUT_MODE}} を \\file{{"cond"}} にします（結果は \\file{{条件平均の入力/}} に残っています）。
\\item \\textbf{{乱数シードの確認}}：ショットごとの入力でも、\\file{{mat_seed_check.py shot}} で、最適化の運による RMSE の変動を確かめられます（未実施）。
\\item \\textbf{{P09 の扱い}}：GPPS の 3data、200 ℃・1.5 mm・3 mm/s は、残るショット 001 も疑わしいので、条件全体を除くか決めてください（一覧の「ショット」列を空欄にします）。
\\item \\textbf{{GPPS の $T_g$}}：仮置きの 100 ℃ です。実際の値が分かれば、\\file{{SHIFT_CFG}} で直して実行し直します。
\\item \\textbf{{\\PIII\\ の定義}}：センサ・時点の定義を教えてください（説明書に書くため）。算出式の圧力ではないことは確かめました。
\\item \\textbf{{モデルの比較を確かにするには}}：本体の手順5b と同じく、ハイパーパラメータと $\\theta$ を MCMC でサンプリングして、予測の不確かさに取り込む方法があります（実施するかは、ご判断ください）。
\\item \\textbf{{見直しの候補}}：規則 c・g で挙がった条件・ショットを、一覧に登録するかを決めてください（\\file{{データ確認/問題の候補.csv}}）。
\\end{{enumerate}}
"""


# ---- 書き出しと PDF
tex = open(os.path.join(rc.HERE, "report_mat_shift_template.tex"), encoding="utf-8").read()
V = {"DATE": rc.today_ja(), **V}
for key, val in V.items():
    tex = tex.replace(f"@@{key}@@", str(val))
left = sorted(set(re.findall(r"@@([A-Za-z0-9_]+)@@", tex)))
assert not left, f"置き換えていない項目: {left}"
with open(os.path.join(ROOT, f"{STEM}.tex"), "w", encoding="utf-8", newline="\n") as f:
    f.write(tex)
out = os.path.join(rc.W, "_tmp", "report_mat")
os.makedirs(out, exist_ok=True)
for ext in ("aux", "toc", "out"):
    p = os.path.join(out, f"{STEM}.{ext}")
    if os.path.exists(p):
        os.remove(p)
for i in range(2):
    r = subprocess.run([rc.LUALATEX, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={out}", f"{STEM}.tex"], cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        print(r.stdout.decode("utf-8", "replace")[-3000:])
        sys.exit(f"lualatex が失敗しました（{i + 1} 回目）")
shutil.copy2(os.path.join(out, f"{STEM}.pdf"), os.path.join(ROOT, f"{STEM}.pdf"))
print("作成:", os.path.join(ROOT, f"{STEM}.pdf"), "と .tex")
