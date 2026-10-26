# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR の「厚みごとの比較」の説明書（用語集と図の見方）を作る。

使い方（リポジトリ直下から）: uv run python 04_補助スクリプト/build_report_mat_thick.py [--run-min 分]
前提：ノート「材料別_温度シフトGPR_ARD対応Matern52＋線形＋ホワイトノイズ_厚みごとの比較.ipynb」を実行済みであること。
各材料の <材料フォルダ>_学習結果/ショットごとの入力/厚みごとの比較/ の CSV から数値を取り出して report_mat_thick_template.tex に入れ、
05_材料ごとの機械学習データと予測検証データ/学習結果説明_厚みごとの温度シフト.tex を書き、そのフォルダで LuaLaTeX を2回動かす
（補助ファイルは _tmp/report_mat_thick/ に出す）。図は、tex から相対パスで読む。
2026-10-23：ユーザーの決定で、ショットごとの入力の結果を説明する版に変えた（条件平均の入力の結果は、参考として比べる）。
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
TNAME = {"PP": "PP", "MIX": "粉砕材混合PP", "GPPS": "GPPS"}           # CSV の「材料」列の名前
MODE_DIR = "ショットごとの入力"                       # 説明する結果の入力の取り方（ユーザー決定、2026-10-23）
REF_DIR = "条件平均の入力"                            # 参考に比べる結果（10-22 の版）
SUB = os.path.join(MODE_DIR, "厚みごとの比較")
MAIN = "B_Tset"                                    # 発表資料の主のモデル（ショットごとの入力では、加熱シリンダ温度のシフトが支持された）
STEM = "学習結果説明_厚みごとの温度シフト"
KEYS = ["C", "B_Tset", "B_Tff"]
TEMP = {"B_Tset": "加熱シリンダ温度", "B_Tff": "流動中樹脂温度"}
WIN = {"Tff": "$\\Tff$", "Tset": "$\\Tset$", "差なし": "差なし", "—": "—"}
V = {}
run_min = sys.argv[sys.argv.index("--run-min") + 1] if "--run-min" in sys.argv else "---"


def tdir(mat, mode=MODE_DIR):
    return os.path.join(ROOT, FOLD[mat], f"{FOLD[mat]}_学習結果", mode, "厚みごとの比較")


def rel(mat, name):
    """tex から見た図の相対パス（ROOT から）"""
    return "/".join([FOLD[mat], f"{FOLD[mat]}_学習結果", MODE_DIR, "厚みごとの比較", name])


def rd(mat, name, mode=MODE_DIR):
    return pd.read_csv(os.path.join(tdir(mat, mode), name), encoding="utf-8-sig")


def rcomp(mat, name, mode=MODE_DIR):
    """元のノートの比較の表（<入力>/比較/）"""
    return pd.read_csv(os.path.join(ROOT, FOLD[mat], f"{FOLD[mat]}_学習結果", mode, "比較", name), encoding="utf-8-sig")


def esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%")


def pe(v):
    """ln の差 → 粘度の %"""
    return (np.exp(v) - 1.0) * 100.0


def hl(hs):
    return "・".join(f"{h:.1f}" for h in sorted(hs)) + " mm"


CHK = {m: rd(m, "温度シフトの確認_厚みごと.csv") for m in FOLD}
SUP = {m: rd(m, "重ね合わせ_厚みごと.csv").sort_values("厚み") for m in FOLD}
PROF = {m: rd(m, "プロファイル尤度_厚みごと.csv") for m in FOLD}
SUMM = {m: rd(m, "厚みごとの比較_まとめ.csv").sort_values("厚み") for m in FOLD}
TT = {m: rd(m, "厚みごとと総合の比較.csv") for m in FOLD}
JD = {m: rd(m, "判定の一覧.csv") for m in FOLD}
SH = {m: rcomp(m, "シフトファクターの推定.csv").set_index("モデル") for m in FOLD}
SUMT = {m: rd(m, "厚みごとと総合の比較_まとめ.csv").set_index("モデル") for m in FOLD}
DLML = {m: rcomp(m, "まとめ.csv").set_index("モデル")["θ=0 との LML の差"] for m in FOLD}
DLML_REF = {m: rcomp(m, "まとめ.csv", REF_DIR).set_index("モデル")["θ=0 との LML の差"] for m in FOLD}
SUMT_REF = {m: rd(m, "厚みごとと総合の比較_まとめ.csv", REF_DIR).set_index("モデル") for m in FOLD}


def tt(mat, key, learn, rng="全体", h="全厚み", col="誤差_%"):
    """厚みごとと総合の比較の値（h は「全厚み」か厚み [mm]）"""
    t = TT[mat]
    t = t[(t["モデル"] == key) & (t["学習"] == learn) & (t["範囲"] == rng)]
    names = t["厚みの名前"].astype(str)
    sel = (names == "全厚み") if h == "全厚み" else names.map(lambda s: s != "全厚み" and bool(np.isclose(float(s), float(h))))
    r = t[sel]
    return float(r[col].iloc[0]) if len(r) else np.nan


# ---- ファイルの数・判定の数
figs = csvs = 0
for m in FOLD:
    fn = os.listdir(tdir(m))
    figs += sum(f.endswith(".png") for f in fn)
    csvs += sum(f.endswith(".csv") for f in fn)
V["N_FIG"], V["N_CSV"] = str(figs), str(csvs)
V["N_OK"] = str(sum(int((JD[m]["判定"] == "OK").sum()) for m in FOLD))
V["N_STAR"] = str(sum(int((JD[m]["判定"] != "OK").sum()) for m in FOLD))
V["RUN_MIN"] = run_min
V["A_CLEAR"], V["DL_CLEAR"], V["ERR_CLEAR"] = "0.5", "3.0", "1.0"


def coef(m, k):
    unit = "kJ/mol" if m != "GPPS" else ""
    nm = ("$\\Eh$", "$\\Ev$") if m != "GPPS" else ("$A_{1,h}$", "$A_{1,v}$")
    return f"{nm[0]} {SH[m].loc[k, 'th_best']:g}・{nm[1]} {SH[m].loc[k, 'tv_best']:g}{(' ' + unit) if unit else ''}"


V["SHIFT_TEXT"] = "、".join(f"{JA[m]} は B\\_Tset が {coef(m, 'B_Tset')}、B\\_Tff が {coef(m, 'B_Tff')}" for m in FOLD)
V["DLML_TEXT"] = ("θ = 0（シフトなし）との対数周辺尤度の差は、B\\_Tset が " + "・".join(f"{JA[m]} {DLML[m]['B_Tset']:.1f}" for m in FOLD)
                  + "、B\\_Tff が " + "・".join(f"{JA[m]} {DLML[m]['B_Tff']:.1f}" for m in FOLD)
                  + "（基準 3.0）。ショットごとの入力では、加熱シリンダ温度のシフトが支持され、流動中樹脂温度のシフトは支持されませんでした。"
                  "このため、発表資料の主のモデルは B\\_Tset です（ユーザーの決定、10 月 23 日）")

# ---- よく使う数
CK = {k: {m: CHK[m][CHK[m]["モデル"] == k].sort_values("厚み") for m in FOLD} for k in ["B_Tset", "B_Tff"]}
st, ff = CK["B_Tset"], CK["B_Tff"]
n_h = sum(len(ff[m]) for m in FOLD)
n_ok = {k: sum(int((CK[k][m]["判定"] == "○").sum()) for m in FOLD) for k in CK}
n_red = {k: sum(int(CK[k][m]["換算でずれが減る"].sum()) for m in FOLD) for k in CK}


def bad_list(k):
    """全厚み共通の係数で換算しても、ずれが減らない厚み（材料ごと）"""
    return [f"{JA[m]} の{'全厚み' if (~CK[k][m]['換算でずれが減る']).all() else ' ' + hl(CK[k][m][~CK[k][m]['換算でずれが減る']]['厚み'])}"
            for m in FOLD if (~CK[k][m]["換算でずれが減る"]).any()]


bad = {k: bad_list(k) for k in CK}
good = [k for k in CK if not bad[k]]
lnr_max = max(float(CHK[m]["lnR の差"].abs().max()) for m in FOLD)
tol = [float(CHK[m]["許容差（ショット間の標準偏差）"].iloc[0]) for m in FOLD]
votes = pd.concat([SUMM[m][f"適した温度({c})"] for m in FOLD for c in "abc"]).value_counts()
over = pd.concat([SUMM[m]["総合の判定"] for m in FOLD]).value_counts()
p0 = pd.concat([PROF[m] for m in FOLD])
n0 = {k: int((p0[p0["モデル"] == k]["係数0との差"] >= 3.0).sum()) for k in CK}
n_cond = (int(p0["学習の条件数"].min()), int(p0["学習の条件数"].max()))
tot_cond = {m: int(PROF[m][PROF[m]["モデル"] == MAIN]["学習の条件数"].sum()) for m in FOLD}
pairs = [(m, k, tt(m, k, "厚みごと"), tt(m, k, "総合")) for m in FOLD for k in KEYS]
n_tot = sum(b < a for _, _, a, b in pairs)
best_tot = {m: min((tt(m, k, "総合"), k) for k in KEYS) for m in FOLD}
best_th = {m: min((tt(m, k, "厚みごと"), k) for k in KEYS) for m in FOLD}
rec = {m: ("厚みごと" if SUMT[m].loc[MAIN, "勧める学習"] == "厚みごと" else "総合データ") for m in FOLD}
r_thin = {m: (float(st[m]["R_実測"].iloc[0]), float(st[m]["R_実測"].iloc[-1])) for m in FOLD}
n_tot_ref = sum(float(SUMT_REF[m].loc[k, "全厚みの誤差_総合_%"]) < float(SUMT_REF[m].loc[k, "全厚みの誤差_厚みごと_%"]) for m in FOLD for k in KEYS)
best_ref = {m: float(SUMT_REF[m]["全厚みの誤差_総合_%"].min()) for m in FOLD}


def names(ms):
    return "・".join(JA[m] for m in ms)


def win_groups(rng):
    """主のモデルで、範囲 rng の誤差が小さい学習のしかたを、材料ごとにまとめた文"""
    w = {m: ("厚みごと" if tt(m, MAIN, "厚みごと", rng) < tt(m, MAIN, "総合", rng) else "総合データ") for m in FOLD}
    return "、".join(f"{names([m for m in FOLD if w[m] == lab])} で{lab}" for lab in ["厚みごと", "総合データ"] if any(w[m] == lab for m in FOLD))


# ---- 図
FIGS = []


def fig(title, paths, caption, what, axes, look, result, widths=("0.95",)):
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


r_st, r_ff = st["PP"].set_index("厚み"), ff["PP"].set_index("厚み")
fig("図1：厚みごとの学習結果（PP・厚み 2.0 mm）", rel("PP", "学習結果_厚み2mm.png"), "左が B\\_Tset、右が B\\_Tff。点はショットごとの計測値（塗り＝学習、白抜き＝検証）、線は加熱シリンダ温度ごとの予測平均、帯は 95\\% 予測区間",
    "全厚み共通の係数で学習したモデル（ショットごとの入力）の予測を、1 つの厚みだけ取り出して描いた図です（ファイル：\\file{学習結果_厚み<厚み>mm.png}。厚みごとに 1 枚）。点は、学習に使ったのと同じショットごとの計測値です（1 つの成形条件に、ショットの数だけ点があります。ノートの設定 \\texttt{FLOW\\_POINTS = \"cond\"} で、成形条件ごとの平均の点にもできます）。",
    "両対数。横軸がせん断速度 $\\gd$、縦軸が粘度 $\\eta$。色と形が加熱シリンダ温度（青 190 ℃・緑 210 ℃・赤 230 ℃。GPPS は 200・220・240 ℃）。曲線の入力（\\PIII・$\\Tff$）は、同じ厚み・温度の成形条件の値を、せん断速度で内挿したもの。",
    "3 本の線が、実測の点と同じ順序（低温が上）・同じ間隔で並ぶか。白抜きの点（学習していない低速側・高速側）が帯の中にあるか。",
    f"PP の 2.0 mm では、どちらのモデルも 3 本の線が実測と同じ順序で並び、間隔も合います（温度による粘度の比 $R$：実測 {r_st.loc[2.0, 'R_実測']:.2f}、B\\_Tset の予測 {r_st.loc[2.0, 'R_予測']:.2f}、"
    f"B\\_Tff の予測 {r_ff.loc[2.0, 'R_予測']:.2f}）。ショットごとの入力なので、曲線はデータの範囲の中でも細かく波打ちます（同じ条件のショットのゆれに合わせるため）。低速側（左端の白抜き）は外挿なので、帯が広がります。")
fig("図2：温度による粘度の比 $R$（PP）", rel("PP", "温度による粘度の比.png"), "厚みごとの $R$（灰：実測、青：B\\_Tset の予測、赤：B\\_Tff の予測）",
    "同じせん断速度で、最低温度の粘度が最高温度の何倍かを、厚みごとに実測と予測で比べた図です（ファイル：\\file{温度による粘度の比.png}）。",
    "縦軸が $R$（1 より大きければ低温ほど高粘度。点線が 1）。横軸がキャビティ厚み。",
    "予測の棒が、実測の棒と同じ高さか。",
    f"3 材料・全厚みで、予測の $R$ は実測とよく合いました（$|\\ln R_{{\\text{{予測}}}}-\\ln R_{{\\text{{実測}}}}|$ は最大 {lnr_max:.3f}。許容差〔同じ成形条件のショット間の標準偏差、"
    f"{min(tol):.3f}〜{max(tol):.3f}〕より小さい）。PP では、$R$ は厚いほど大きく（{r_st['R_実測'].min():.2f}〜{r_st['R_実測'].max():.2f}）、モデルもこの傾向を表しています。"
    "B\\_Tset と B\\_Tff の差は、この図ではほとんどありません。")
m3 = max(FOLD, key=lambda m: abs(pe(ff[m]["ずれ_換算後"].iloc[0]) - pe(st[m]["ずれ_換算後"].iloc[0])))   # 最も薄い厚みで、2 つのモデルの差が大きい材料
s3, f3 = st[m3].iloc[0], ff[m3].iloc[0]
h3 = float(s3["厚み"])
txt3 = (f"{JA[m3]} の {h3:.1f} mm（換算前のずれ {pe(s3['ずれ_換算前']):.1f}\\%）では、加熱シリンダ温度で換算すると {pe(s3['ずれ_換算後']):.1f}\\%、"
        f"流動中樹脂温度で換算すると {pe(f3['ずれ_換算後']):.1f}\\% になりました。")
if pe(f3["ずれ_換算後"]) >= pe(f3["ずれ_換算前"]):
    txt3 += f"B\\_Tff の全厚み共通の係数は {coef(m3, 'B_Tff')} とほぼ 0 で、ほとんど換算されず、ずれも減りません（ショットごとの入力では、流動中樹脂温度のシフトが支持されなかったため）。"
fig(f"図3：重ね合わせ（{JA[m3]}・厚み {h3:.1f} mm・全厚み共通の係数）", rel(m3, f"重ね合わせ_学習済みの係数_厚み{h3:g}mm.png"), "左から、換算前、加熱シリンダ温度で換算、流動中樹脂温度で換算。破線は 3 次式",
    "全厚み共通の係数で、温度ごとの点を換算（シフト）したとき、1 本の曲線に重なるかを見る図です（ファイル：\\file{重ね合わせ_学習済みの係数_厚み<厚み>mm.png}。厚みごとに決め直した係数の図は \\file{重ね合わせ_厚みごとの係数_厚み<厚み>mm.png}）。",
    "両対数。横軸が（換算）せん断速度、縦軸が（換算）粘度。色と形が加熱シリンダ温度。図の上の「ずれ」が、3 次式からのずれ（粘度の \\%）。",
    "換算後に、温度の違う点が破線の近くに集まるか。ずれが換算前より小さいか。", txt3)
txt4 = (f"加熱シリンダ温度で換算すると、3 材料の {n_h} 厚みのうち {n_red['B_Tset']} 厚みでずれが減りました" + (f"（{'、'.join(bad['B_Tset'])}では減らない）" if bad["B_Tset"] else "")
        + f"。流動中樹脂温度では {n_red['B_Tff']} 厚みで減り" + (f"、{'、'.join(bad['B_Tff'])}では減りませんでした" if bad["B_Tff"] else "ました")
        + f"（5 章の表 3）。このため、全厚み共通の係数での判定（問い 1）は、B\\_Tset が {n_ok['B_Tset']} 厚みで ○、B\\_Tff が {n_ok['B_Tff']} 厚みで ○ です。"
        "なお、どちらのモデルでも、予測の $R$ は実測と合っています（図2）。シフトで表せない温度の効果は、GP の入力に入れた、もう一方の温度が担うためです。")
fig(f"図4：重ね合わせのずれ（{JA[m3]}・全厚み共通の係数）", rel(m3, "重ね合わせのずれ_学習済みの係数.png"), "厚みごとの、換算前（灰）・加熱シリンダ温度で換算（青）・流動中樹脂温度で換算（赤）のずれ",
    "図3 のずれを、全厚みについて並べた図です（ファイル：\\file{重ね合わせのずれ_学習済みの係数.png}）。",
    "縦軸がずれ（粘度の \\%）。横軸がキャビティ厚み。",
    "赤・青の棒が、灰の棒より低いか（換算でずれが減るか）。", txt4)
a_win = [(JA[m], float(h), w) for m in FOLD for h, w in zip(SUMM[m]["厚み"], SUMM[m]["適した温度(a)"]) if w in ("Tff", "Tset")]
txt5 = ("係数を決め直すと、どちらの温度でも、ずれは換算前の 1/2〜1/4 に減ります。"
        + ("差が 0.5 ポイント以上あったのは、" + "、".join(f"{n} の {h:.1f} mm（{WIN[w]}）" for n, h, w in a_win) + "だけで、ほかの厚みは差なしです。" if a_win else "どの厚みも差なしです。")
        + "この基準はデータの点だけで決まるので、入力の取り方（ショットごと・条件平均）によりません。")
fig("図5：(a) 厚みごとに係数を決め直したときの重ね合わせのずれ（PP）", rel("PP", "重ね合わせのずれ_厚みごとの係数.png"), "厚みごとに、ずれが最小になる係数を探したときのずれ",
    "厚みごとに係数（横・縦）を決め直し、どちらの温度で換算したほうがよく重なるかを比べる図です（基準 (a)。ファイル：\\file{重ね合わせのずれ_厚みごとの係数.png}）。データの点だけで決まり、GP のモデルを使いません。",
    "図4 と同じ。",
    f"青と赤の棒の差が {V['A_CLEAR']} ポイント以上あるか（あれば、低いほうの温度が適する）。", txt5)
dl_sig = [(JA[m], float(h), d) for m in FOLD for h, d in zip(SUMM[m]["厚み"], SUMM[m]["ΔL（Tff − Tset）"]) if abs(d) >= 3.0]
pv = PROF["PP"][PROF["PP"]["モデル"] == MAIN].sort_values("厚み")
txt6 = (("$|\\Delta L|$ が 3 以上だったのは、" + "、".join(f"{n} の {h:.1f} mm（{d:+.1f}。{'$\\Tff$' if d > 0 else '$\\Tset$'}）" for n, h, d in dl_sig) + "だけで、ほかの厚みは差なしです。")
        if dl_sig else "3 材料・全厚みで $|\\Delta L|$ は 3 に届きません（すべて差なし）。")
txt6 += (f"係数 0 との差が 3 以上だったのは、B\\_Tset で {n0['B_Tset']}/{n_h}、B\\_Tff で {n0['B_Tff']}/{n_h} 厚みでした。"
         f"厚みごとの学習データは成形条件 {n_cond[0]}〜{n_cond[1]} しかなく、決まった係数も厚みで大きく変わります"
         f"（例：PP の B\\_Tset の縦の係数 $\\Ev$ は、薄い順に {'・'.join(f'{v:g}' for v in pv['縦の係数'])} kJ/mol）。")
fig("図6：(b) 厚みごとの対数周辺尤度の差（PP）", rel("PP", "対数周辺尤度の差_厚みごと.png"), "灰：B\\_Tset の係数 0 との差、青：B\\_Tff の係数 0 との差、赤：$\\Delta L$（B\\_Tff − B\\_Tset）",
    "厚みごとの 5data だけで、2 つのモデルの係数をプロファイル尤度で決め、データへの合い具合（LML）を比べた図です（基準 (b)。ファイル：\\file{対数周辺尤度の差_厚みごと.png}）。",
    "縦軸が LML の差。点線が 3.0（効果が認められる目安）。",
    "赤の棒が $+3$ を超えるか（$\\Tff$ が適する）、$-3$ を下回るか（$\\Tset$ が適する）。灰・青の棒が 3 を超えるか（その厚みだけで、シフトの効果が認められるか）。", txt6, widths=("0.9",))
e2 = {m: SUMM[m].set_index("厚み").loc[2.0] for m in FOLD}
fig("図7：(c) 検証データの予測誤差（PP・厚みごと）", rel("PP", "検証誤差_厚みごと.png"), "元のノート（総合データで学習）の 3 モデルの、3data の予測誤差を厚みごとに集計したもの",
    "学習に用いていない 3data（射出速度の両端）の予測誤差を、厚みごとに比べる図です（基準 (c)。ファイル：\\file{検証誤差_厚みごと.png}）。",
    "縦軸が予測誤差（粘度の \\%）。灰が C（シフトなし）、青が B\\_Tset、赤が B\\_Tff。",
    f"青と赤の差が {V['ERR_CLEAR']} ポイント以上あるか。",
    "薄肉（0.5・1.0 mm。3data は高速側だけ）は、3 モデルとも近い値です。差が出るのは厚肉（1.5・2.0 mm。低速側の外挿を含む）で、2.0 mm の誤差（B\\_Tset／B\\_Tff）は "
    + "、".join(f"{JA[m]} {e2[m]['誤差_Tset_%']:.1f}／{e2[m]['誤差_Tff_%']:.1f}\\%" for m in FOLD)
    + "でした。ショットごとの入力では、厚肉の低速側の外挿の誤差が大きく、2 つのモデルの差は、この外れ方の違いに左右されます。")
lp, lt = tt("MIX", MAIN, "厚みごと", "低速側"), tt("MIX", MAIN, "総合", "低速側")
hp_, ht_ = tt("MIX", MAIN, "厚みごと", "高速側", h=2.0), tt("MIX", MAIN, "総合", "高速側", h=2.0)
fig(f"図8：厚みごとのモデルと総合データのモデル（粉砕材混合 PP・{esc(MAIN)}）", rel("MIX", f"厚みごとと総合_{MAIN}.png"),
    "灰：総合データ（全厚みの 5data）で学習、青：その厚みの 5data だけで学習。3data の予測誤差",
    "学習のしかただけを変えて、同じ 3data の予測誤差を比べる図です（問い 3。ファイル：\\file{厚みごとと総合_<モデル>.png}。モデルごとに 1 枚）。",
    "縦軸が予測誤差（粘度の \\%）。横軸がキャビティ厚み。",
    "青の棒（厚みごと）が灰の棒（総合）より低いか。",
    f"粉砕材混合 PP の {esc(MAIN)} では、厚肉の低速側（外挿）で、厚みごとのモデルの誤差が{'小さく' if lp < lt else '大きく'}なりました（低速側の全厚み：厚みごと {lp:.0f}\\%、総合 {lt:.0f}\\%）。"
    f"一方、2.0 mm の高速側は、厚みごと {hp_:.0f}\\%、総合 {ht_:.0f}\\% です。3 材料 $\\times$ 3 モデルの 9 組のうち、全厚みの誤差は {n_tot} 組で総合データ、{9 - n_tot} 組で厚みごとのほうが小さくなりました（5 章の表 6）。",
    widths=("0.9",))
V["FIGURES"] = "\n".join(FIGS)

# ---- 表
rows = []
for m in FOLD:
    a, b = st[m].set_index("厚み"), ff[m].set_index("厚み")
    for i, h in enumerate(a.index):
        rows.append(f"{JA[m] if i == 0 else ''} & {h:.1f} & {int(b.loc[h, '条件数'])} & {b.loc[h, 'R_実測']:.3f} & {a.loc[h, 'R_予測']:.3f} & {b.loc[h, 'R_予測']:.3f} & "
                    f"{pe(b.loc[h, 'ずれ_換算前']):.1f} & {pe(a.loc[h, 'ずれ_換算後']):.1f} & {pe(b.loc[h, 'ずれ_換算後']):.1f} & {a.loc[h, '判定']} & {b.loc[h, '判定']} \\\\")
    rows.append("\\midrule")
rows[-1] = ""
T_CHK = "\n".join(rows)
rows = []
for m in FOLD:
    s, u = SUMM[m].set_index("厚み"), SUP[m].set_index("厚み")
    for i, h in enumerate(s.index):
        rows.append(f"{JA[m] if i == 0 else ''} & {h:.1f} & {s.loc[h, 'ずれ_Tset_%']:.1f}／{s.loc[h, 'ずれ_Tff_%']:.1f} & {u.loc[h, '横の係数_Tset']:g}, {u.loc[h, '縦の係数_Tset']:g}／"
                    f"{u.loc[h, '横の係数_Tff']:g}, {u.loc[h, '縦の係数_Tff']:g} & {WIN[s.loc[h, '適した温度(a)']]} & {s.loc[h, 'ΔL（Tff − Tset）']:+.1f} & {WIN[s.loc[h, '適した温度(b)']]} & "
                    f"{s.loc[h, '誤差_Tset_%']:.1f}／{s.loc[h, '誤差_Tff_%']:.1f} & {WIN[s.loc[h, '適した温度(c)']]} & \\textbf{{{WIN[s.loc[h, '総合の判定']]}}} \\\\")
    rows.append("\\midrule")
rows[-1] = ""
T_WIN = "\n".join(rows)
rows = []
for m in FOLD:
    p = PROF[m]
    for i, h in enumerate(sorted(p["厚み"].unique())):
        a = p[(p["厚み"] == h) & (p["モデル"] == "B_Tset")].iloc[0]
        b = p[(p["厚み"] == h) & (p["モデル"] == "B_Tff")].iloc[0]
        mark = lambda r: "" if (r["取り直しが収束"] and not r["格子の端"]) else "$^{*}$"
        rows.append(f"{JA[m] if i == 0 else ''} & {h:.1f} & {int(a['学習の条件数'])} & {a['横の係数']:g}, {a['縦の係数']:g}{mark(a)} & {a['係数0との差']:.1f} & "
                    f"{b['横の係数']:g}, {b['縦の係数']:g}{mark(b)} & {b['係数0との差']:.1f} & {b['LML'] - a['LML']:+.1f} \\\\")
    rows.append("\\midrule")
rows[-1] = ""
T_PROF = "\n".join(rows)
rows = []
for m in FOLD:
    for i, k in enumerate(KEYS):
        cells = []
        for rng in ["全体", "低速側", "高速側"]:
            a, b = tt(m, k, "厚みごと", rng), tt(m, k, "総合", rng)
            bs = f"{b:.1f}"
            cells.append(f"{a:.1f}／" + (f"\\textbf{{{bs}}}" if b < a else bs))
        ca, cb = tt(m, k, "厚みごと", col="区間に入った割合"), tt(m, k, "総合", col="区間に入った割合")
        rows.append(f"{JA[m] if i == 0 else ''} & {esc(k)} & {' & '.join(cells)} & {ca:.2f}／{cb:.2f} \\\\")
    rows.append("\\midrule")
rows[-1] = ""
T_TT = "\n".join(rows)
V["TABLES"] = f"""
\\subsection{{温度シフトの確認（問い 1。全厚み共通の係数）}}
\\begin{{table}}[H]
\\caption{{厚みごとの温度による粘度の比 $R$ と、重ね合わせのずれ（粘度の \\%）。「予測」「換算後」「判定」は、左が B\\_Tset、右が B\\_Tff}}\\label{{tab:check}}
\\scriptsize
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}llrrrrrrrcc@{{}}}}
\\toprule
材料 & 厚み (mm) & 条件数 & $R$ 実測 & $R$ 予測 & $R$ 予測 & 換算前 & 換算後 & 換算後 & 判定 & 判定 \\\\
 & & & & B\\_Tset & B\\_Tff & & B\\_Tset & B\\_Tff & B\\_Tset & B\\_Tff \\\\
\\midrule
{T_CHK}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}
\\begin{{points}}
\\pt{{読み方}}条件数は、5data と 3data を合わせた成形条件の数。$R$ の予測が実測に近く、換算後のずれが換算前より小さければ ○ です。B\\_Tset は {n_ok['B_Tset']} 厚み、B\\_Tff は {n_ok['B_Tff']} 厚みで ○ です（$R$ はどちらも実測と合う）。
\\end{{points}}

\\subsection{{シフトに使う温度（問い 2）}}
\\begin{{table}}[H]
\\caption{{厚みごとの 3 つの基準。数値は「加熱シリンダ温度／流動中樹脂温度」。(a) の係数は「横, 縦」（PP・粉砕材混合 PP は kJ/mol、GPPS は $A_1$）。右端が多数決}}\\label{{tab:win}}
\\scriptsize
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}llrXcrcrcc@{{}}}}
\\toprule
材料 & 厚み & (a) ずれ (\\%) & (a) 係数 & (a) & (b) $\\Delta L$ & (b) & (c) 誤差 (\\%) & (c) & 多数決 \\\\
\\midrule
{T_WIN}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}
\\begin{{points}}
\\pt{{読み方}}(a) は 5data＋3data の点で、(b) は 5data だけで、(c) は元のノート（総合データで学習）の 3data の予測で決めています。「—」は、比べられなかったもの。
\\end{{points}}

\\begin{{table}}[H]
\\caption{{(b) 厚みごとのプロファイル尤度。係数は「横, 縦」。$^{{*}}$ は、細かい格子の取り直しが収束しなかった（または格子の端に当たった）もの。$\\Delta L$ は B\\_Tff − B\\_Tset}}\\label{{tab:prof}}
\\scriptsize
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}llrXrXrr@{{}}}}
\\toprule
材料 & 厚み & 学習の条件数 & B\\_Tset の係数 & 係数 0 との差 & B\\_Tff の係数 & 係数 0 との差 & $\\Delta L$ \\\\
\\midrule
{T_PROF}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}
\\begin{{points}}
\\pt{{読み方}}係数 0 との差が 3 未満なら、その厚みのデータだけでは、シフトを入れた効果がはっきりしません。全厚み共通の係数（2.1 節）と比べると、厚みごとの係数はばらつきが大きく、負の値（低温ほど粘度が低い向き）も出ます。
\\end{{points}}

\\subsection{{厚みごとに学習するか、総合データで学習するか（問い 3）}}
\\begin{{table}}[H]
\\caption{{全厚みの 3data の予測誤差（粘度の \\%）と、95\\% 予測区間に入った割合。数値は「厚みごとに学習／総合データで学習」。総合が小さいものを太字}}\\label{{tab:tt}}
\\scriptsize
\\begin{{tabularx}}{{\\linewidth}}{{@{{}}llXXXX@{{}}}}
\\toprule
材料 & モデル & 全体 & 低速側 & 高速側 & 区間に入った割合 \\\\
\\midrule
{T_TT}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}
\\begin{{points}}
\\pt{{読み方}}厚みごとの値は、各厚みのモデルの予測を、全厚みについてまとめて計算したもの。厚みごとの内訳は \\file{{厚みごとと総合の比較.csv}} にあります。低速側（3・5 mm/s）は厚肉（1.5・2.0 mm）だけのデータです。
\\end{{points}}
"""

# ---- 結論
st_range = "、".join(f"{JA[m]} {pe(st[m]['ずれ_換算前'].min()):.1f}〜{pe(st[m]['ずれ_換算前'].max()):.1f}\\% → {pe(st[m]['ずれ_換算後'].min()):.1f}〜{pe(st[m]['ずれ_換算後'].max()):.1f}\\%" for m in FOLD)
star = "・".join(sorted(set(pd.concat([JD[m][JD[m]["判定"] != "OK"]["節"] for m in FOLD]))))
th_ms, to_ms = [m for m in FOLD if rec[m] == "厚みごと"], [m for m in FOLD if rec[m] == "総合データ"]
rec_text = ("、".join(t for t in [f"{names(th_ms)} は\\textbf{{厚みごとに}}" if th_ms else "", f"{names(to_ms)} は\\textbf{{全厚みの総合データで}}" if to_ms else ""] if t)
            + "学習するのがよい")
low_cov = [m for m in FOLD if tt(m, MAIN, "厚みごと", col="区間に入った割合") < tt(m, MAIN, "総合", col="区間に入った割合")]
V["CONCLUSION"] = f"""
ノートの判定は、OK {V['N_OK']} 個、★ {V['N_STAR']} 個でした（★ は {star} の節）。以下、結果をまとめます。この版は、\\textbf{{ショットごとの入力}}（10 月 23 日に既定を変えた）の結果です。

\\begin{{enumerate}}[itemsep=0.8ex,topsep=0.6ex]
\\item \\textbf{{問い 1：温度シフトは効いているか}}：発表資料の主のモデル B\\_Tset（加熱シリンダ温度でシフト）は、3 材料の {n_h} 厚みのうち {n_ok['B_Tset']} 厚みで ○ でした。温度の順序は実測と同じで、温度による粘度の比 $R$ も実測と合い（差は最大 {lnr_max:.3f}。許容差より小さい）、全厚み共通の係数で換算すると、重ね合わせのずれが減ります（{st_range}）。B\\_Tff は {n_ok['B_Tff']} 厚みで ○ で、{'、'.join(bad['B_Tff']) if bad['B_Tff'] else 'なし'}では、換算でずれが減りませんでした（△。全厚み共通の係数がほぼ 0 のため）。
\\item \\textbf{{問い 2：シフトに使う温度}}：厚みごとに係数を決め直して比べると、3 つの基準の多数決は、{n_h} 厚みのうち差なし {over.get('差なし', 0)}・流動中樹脂温度 {over.get('Tff', 0)}・加熱シリンダ温度 {over.get('Tset', 0)} でした（{votes.sum()} 判定の内訳：流動中樹脂温度 {votes.get('Tff', 0)}・加熱シリンダ温度 {votes.get('Tset', 0)}・差なし {votes.get('差なし', 0)}）。\\textbf{{厚みごとに見れば、どちらの温度でもほぼ同じ}}です。一方、\\textbf{{全厚み共通の 1 組の係数で表すなら、{'・'.join(TEMP[k] for k in good) if good else 'どちらとも言えない'}{'のほうが適しています' if len(good) == 1 else ''}}}（問い 1）。ショットごとの入力では、流動中樹脂温度は GP の入力として粘度の違いを担い、シフトとしては支持されませんでした（元のノートの $\\theta=0$ との LML の差：B\\_Tff {'・'.join(f'{DLML[m]["B_Tff"]:.1f}' for m in FOLD)}、B\\_Tset {'・'.join(f'{DLML[m]["B_Tset"]:.1f}' for m in FOLD)}）。\\textbf{{条件平均の入力（10 月 22 日の版）では逆に、流動中樹脂温度のシフトが支持されていた}}（B\\_Tff {'・'.join(f'{DLML_REF[m]["B_Tff"]:.1f}' for m in FOLD)}）ので、この結論は入力の取り方に左右されます。
\\item \\textbf{{解釈の注意}}：流動中樹脂温度は、せん断発熱で射出速度とともに変わります。また、ショットごとの入力では、同じ条件の中のせん断速度と粘度のゆれ（センサへの到達時刻のゆれ）に GP が合わせるので、曲線がデータの範囲の中でも波打ちます。どちらも、温度の効果をシフトとして取り出すのを難しくしています。
\\item \\textbf{{問い 3：厚みごとか、総合データか}}：学習用 5data・検証用 3data のまま、ショットごとに学習するなら、{rec_text}と判断しました（主のモデル {esc(MAIN)} の全厚みの予測誤差で判断。ノートの 9 節の規則）。
\\begin{{enumerate}}[label=(\\roman*),itemsep=0.3ex,topsep=0.3ex]
\\item 予測誤差（{esc(MAIN)}、厚みごと／総合）：{'、'.join(f"{JA[m]} {tt(m, MAIN, '厚みごと'):.1f}／{tt(m, MAIN, '総合'):.1f}\\%" for m in FOLD)}。3 材料 $\\times$ 3 モデルの 9 組では、{n_tot} 組で総合データ、{9 - n_tot} 組で厚みごとのほうが小さい。
\\item 差が大きいのは、学習範囲の外の低速側（厚肉）で、{win_groups('低速側')}のほうが誤差が小さい。高速側は、{win_groups('高速側')}のほうが小さい。
\\item 95\\% 予測区間に入った割合は、{names(low_cov) + ' で、厚みごとのほうが低い（区間が狭すぎる）' if low_cov else '厚みごとと総合で同程度'}。また、厚みごとのモデルは、学習していない厚みを予測できず、学習データの成形条件も {n_cond[0]}〜{n_cond[1]}（総合データの約 1/4）しかありません。
\\end{{enumerate}}
参考：条件平均の入力では、9 組中 {n_tot_ref} 組で総合データのほうが小さく、最もよいモデルの誤差も {'・'.join(f'{best_ref[m]:.1f}' for m in FOLD)}\\%（PP・粉砕材混合 PP・GPPS）と、ショットごとの入力（総合データで {'・'.join(f'{best_tot[m][0]:.1f}' for m in FOLD)}\\%、厚みごとで {'・'.join(f'{best_th[m][0]:.1f}' for m in FOLD)}\\%）より小さくなっていました。
\\item \\textbf{{ノートの ★ の意味}}：4 節の ★ は、全厚み共通の係数で換算しても、ずれが減らない厚みがあること（問い 1 の △）を示します。エラーではなく、結果を読むときの注意点です。
\\end{{enumerate}}

\\subsection{{これからの選択肢（ユーザーの判断）}}
\\begin{{enumerate}}[itemsep=0.6ex,topsep=0.6ex]
\\item \\textbf{{入力の取り方}}：ショットごとの入力では、3data の予測誤差が、条件平均の入力より大きくなりました（上の参考）。条件平均に戻すかは、ご判断ください（ノートの設定 \\file{{INPUT_MODE}} の 1 行で戻せ、結果は \\file{{条件平均の入力/}} に残っています）。
\\item \\textbf{{厚みごとに学習する場合}}：学習していない厚みは予測できず、95\\% 予測区間も狭すぎる傾向があります。厚みごとに係数をよく決めるには、成形条件（射出速度・温度の水準）を増やす必要があります（ショット数を増やしても、成形条件の数は増えません）。
\\end{{enumerate}}
"""

# ---- 書き出しと PDF
tex = open(os.path.join(rc.HERE, "report_mat_thick_template.tex"), encoding="utf-8").read()
V = {"DATE": rc.today_ja(), **V}
for key, val in V.items():
    tex = tex.replace(f"@@{key}@@", str(val))
left = sorted(set(re.findall(r"@@([A-Za-z0-9_]+)@@", tex)))
assert not left, f"置き換えていない項目: {left}"
with open(os.path.join(ROOT, f"{STEM}.tex"), "w", encoding="utf-8", newline="\n") as f:
    f.write(tex)
out = os.path.join(rc.W, "_tmp", "report_mat_thick")
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
