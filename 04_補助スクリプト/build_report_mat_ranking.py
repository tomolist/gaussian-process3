# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR（05 フォルダ）：本番学習の設定の根拠を、材料 × 学習データ（通常版・温度適切版）ごとに、すべてのモデルのランキング表にした PDF を作る
（2026-11-06 のユーザー依頼。「本番学習の設定の決定.pdf」と同じ場所に置く）。あわせて、流動中樹脂温度 Tff を GP の入力に使ったほうがよいかの根拠の表も載せる。

使い方（リポジトリ直下から）: uv run python 04_補助スクリプト/build_report_mat_ranking.py [--out <フォルダ>]
- 読むもの（学習はしない。CSV を読むだけ）：
  - 6 つの検討結果フォルダの 全ショット版/全厚み総合/シフトファクターの検討/ の まとめ.csv・候補の選び方.csv・シフトファクターの推定.csv・ハイパーパラメータ.csv
  - 同じフォルダの 全ショット版/厚みごと/ の プロファイル尤度_厚みごと.csv・温度シフトの確認_厚みごと.csv・厚みごとの比較_まとめ.csv・厚みごとと総合の比較.csv
  - C から Tff を外したモデル（C_noTff）の 3data の指標：2026-11-06 の確かめの実行の結果 _tmp/tff_input/run3/<版>/<材料>/…/Tffの入力の有無の比較/予測検証_3data_指標_全モデル.csv
    （ユーザーが本物の学習結果フォルダで比較のノートを実行したら、そちらを読むように直す。C_noTff の LML は、検討の結果の LML_at_0 から取る）
- 書くもの：--out のフォルダ（省くと 05_材料ごとの機械学習データと予測検証データ/04_本番学習の設定(検討結果説明書)/。2026-11-12 にユーザーが移した場所）に 本番学習の設定の根拠_ランキング表.tex と .pdf。
  同じ名前のファイルがあれば止まる（上書きしない）。LuaLaTeX の補助ファイルは _tmp/report_mat_ranking/<日時>/。
"""
import datetime
import glob
import os
import re
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mat_common as mc  # noqa: E402
import report_common as rc  # noqa: E402

W = rc.W
ROOT_REL = "05_材料ごとの機械学習データと予測検証データ"
ROOT = os.path.join(W, ROOT_REL)
MAT_BASE = {"PP": "01_バージンPP再々実験", "MIX": "02_粉砕材混合PP実験", "GPPS": "03_GPPS実験"}   # 材料の名前
FOLD = {k: f"{v}(データと結果)" for k, v in MAT_BASE.items()}   # 05 フォルダの中の材料フォルダ（2026-11-12 のユーザーのフォルダの整理で「(データと結果)」が付いた）
MJ = {"PP": "PP", "MIX": "粉砕材混合 PP", "GPPS": "GPPS"}
VERS = {"通常版": "(5all学習データ)", "温度適切版": "(5適切学習データ)"}
STEM = "本番学習の設定の根拠_ランキング表"
OUT_DEFAULT = os.path.join(ROOT, "04_本番学習の設定(検討結果説明書)")   # 書く場所の既定（2026-11-12 にユーザーが 05 フォルダの直下から移した）
SF = ("全ショット版", "全厚み総合", "シフトファクターの検討")
TH = ("全ショット版", "厚みごと")
CASES = [(v, m) for v in VERS for m in FOLD]
NO_TFF = "C_noTff"
TMP_NOTFF = os.path.join(W, "_tmp", "tff_input", "run3")      # C_noTff の 3data の指標（2026-11-06 の確かめの実行）
DAG = "\\textsuperscript{†}"
LML_CLEAR, RMSE_RATIO = 1.92, 1.1                              # Tff の入力の判断の基準（比較のノートと同じ）

out_dir = os.path.join(W, sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else OUT_DEFAULT
TEX, PDF = os.path.join(out_dir, STEM + ".tex"), os.path.join(out_dir, STEM + ".pdf")
for _p in (TEX, PDF):
    if os.path.exists(_p):
        sys.exit(f"同じ名前のファイルがあるので止めました（上書きしない）: {os.path.relpath(_p, W)}")


# ---------------------------------------------------------------- 小さな部品
def base(ver, mat):
    gl = glob.glob(os.path.join(ROOT, FOLD[mat], FOLD[mat][:2] + "_検討結果", "*" + VERS[ver]))
    assert len(gl) == 1, (ver, mat, gl)
    return gl[0]


def rd(*parts):
    return pd.read_csv(os.path.join(*parts), encoding="utf-8-sig")


def rel(p):
    return os.path.relpath(p, W).replace("\\", "/").replace(ROOT_REL + "/", "05_…/")


def g(x):
    x = float(x)
    s = f"{abs(x):g}"
    return f"$-{s}$" if x < 0 else s


def n(x, nd):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    return rc.num(float(x), nd)


def M(k):
    return f"\\mdl{{{k}}}"


def B(s, cond):
    return f"\\textbf{{{s}}}" if cond else s


def tt(s):
    """\\file{…}・\\mdl{…} の中身 → 等幅の文字（特殊文字を逃がし、/ と _ の後ろで改行できるようにする）"""
    out = []
    for ch in s:
        out.append({"_": "\\_", "%": "\\%", "&": "\\&", "#": "\\#", "$": "\\$", "\\": "/", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}"}.get(ch, ch))
        if ch in "/_\\":
            out.append("\\allowbreak{}")
    return "\\texttt{" + "".join(out) + "}"


def n_params(key):
    """パラメータの数（カーネル：振幅 1 ＋ 長さスケール〔入力の数〕＋ 線形項 2 ＋ ノイズ 1、シフトの係数：A 1・B 2）"""
    if key == NO_TFF:
        return 7
    if key == "C":
        return 8
    return 7 + {"A": 1, "B": 2}[key.split("_")[0]]


def theta_str(row):
    """まとめ.csv の θ の列 → 表の書き方（構造 A は横の係数を書かない）"""
    s = str(row["θ"])
    if s in ("nan", ""):
        return "---"
    unit = " kJ/mol" if "kJ/mol" in s else ""
    out = []
    for sym, hv, best, lo, hi in re.findall(r"(E|A1)_(h|v) ([-\d.]+)［([-\d.]+)〜([-\d.]+)］", s):
        if hv == "h" and str(row["SHIFT_STRUCT"]) == "A":
            continue
        name = f"$E_{hv}$" if sym == "E" else f"$A_{{1,{hv}}}$"
        out.append(f"{name} {g(best)}［{g(lo)}〜{g(hi)}］")
    return "・".join(out) + unit


def coef_thick(key, th, tv, unit):
    s = (f"横 {g(th)}・" if key.startswith("B") else "") + f"縦 {g(tv)}"
    return s + (" kJ/mol" if unit == "kJ/mol" else "")


def rank(values, ascending=False):
    """順位（同じ値は同じ順位。NaN は空欄）"""
    r = pd.Series(values).rank(method="min", ascending=ascending)
    return ["" if np.isnan(x) else str(int(x)) for x in r]


# ---------------------------------------------------------------- 読み込み
D = {}
for ver, mat in CASES:
    b = base(ver, mat)
    d = {"base": b, "sum": rd(b, *SF, "まとめ.csv"), "ctab": rd(b, *SF, "候補の選び方.csv"), "est": rd(b, *SF, "シフトファクターの推定.csv"),
         "hyp": rd(b, *SF, "ハイパーパラメータ.csv"),
         "prof": rd(b, *TH, "シフトに使う温度", "プロファイル尤度_厚みごと.csv"), "tchk": rd(b, *TH, "温度シフトの確認", "温度シフトの確認_厚みごと.csv"),
         "tsum": rd(b, *TH, "シフトに使う温度", "厚みごとの比較_まとめ.csv"), "cmp": rd(b, *TH, "厚みごとのモデルと総合の比較", "厚みごとと総合の比較.csv")}
    rows = [{"モデル": r["モデル"], "LML": r["LML"], "dLML_vs_0": r["θ=0 との LML の差"]} for _, r in d["sum"].iterrows() if r["モデル"] != "C"]
    win, _, notes = mc.shift_candidates(rows)
    d["key"] = win or "C"
    d["S"] = d["sum"].set_index("モデル")
    est = d["est"]
    d["lml_notff"] = float(est[est["モデル"].str.contains("_Tff_")]["LML_at_0"].max())      # C_noTff ＝ A_Tff・B_Tff の θ = 0
    d["lml_tffonly"] = float(est[est["モデル"].str.contains("_Tset_")]["LML_at_0"].max())   # Tff だけを入力（Tset なし）＝ A_Tset・B_Tset の θ = 0
    mt = rd(TMP_NOTFF, ver, mat, *SF[:2], "Tffの入力の有無の比較", "予測検証_3data_指標_全モデル.csv")
    mtc = mt[mt["範囲"].isin(["全体", "低速側", "高速側"])].pivot_table(index="モデル", columns="範囲", values="RMSE", sort=False)
    assert abs(mtc.loc["C", "全体"] - d["S"].loc["C", "RMSE 全体"]) < 1e-4, (ver, mat, "確かめの実行の C の RMSE が、まとめ.csv と違う")   # まとめ.csv は小数 4 桁に丸めてある
    d["notff_rmse"] = mtc.loc[NO_TFF].to_dict()
    d["notff_cov"] = float(mt[(mt["モデル"] == NO_TFF) & (mt["範囲"] == "全体")]["coverage95"].iloc[0])
    lt = rd(TMP_NOTFF, ver, mat, *SF[:2], "Tffの入力の有無の比較", "対数周辺尤度の比較.csv").set_index("モデル")
    assert abs(lt.loc[NO_TFF, "LML"] - d["lml_notff"]) < 0.05, (ver, mat, "確かめの実行の C_noTff の LML が、LML_at_0 と違う")
    D[(ver, mat)] = d


# ---------------------------------------------------------------- 1. Tff を GP の入力に使うか
def tff_summary():
    rows = []
    for ver, mat in CASES:
        d = D[(ver, mat)]
        S = d["S"]
        dc = float(S.loc["C", "LML"]) - d["lml_notff"]
        q = d["notff_rmse"]["全体"] / float(S.loc["C", "RMSE 全体"])
        lik = "支持される" if dc >= LML_CLEAR else "支持されない"
        err = "Tff ありが良い" if q > RMSE_RATIO else ("Tff なしが良い" if q < 1 / RMSE_RATIO else "同等")
        concl = {("支持される", "同等"): "入れたほうがよい", ("支持される", "Tff ありが良い"): "入れたほうがよい",
                 ("支持されない", "Tff ありが良い"): "入れたほうがよい（誤差で）", ("支持されない", "同等"): "どちらでもよい",
                 ("支持されない", "Tff なしが良い"): "外したほうがよい", ("支持される", "Tff なしが良い"): "決まらない"}[(lik, err)]
        key = d["key"]
        dk = float(S.loc[key, "LML"]) - d["lml_notff"]
        crit = {7: 0.0, 8: 1.92, 9: 3.0}[n_params(key)]
        h = d["hyp"].set_index("model")
        ls = h.loc[key, "ls_ln Tff"] if key in h.index and "ls_ln Tff" in h.columns and pd.notna(h.loc[key, "ls_ln Tff"]) else np.nan
        rows.append(f"{ver} & {MJ[mat]} & {n(S.loc['C', 'LML'], 2)} & {n(d['lml_notff'], 2)} & {B(n(dc, 2), dc >= LML_CLEAR)} & "
                    f"{n(S.loc['C', 'RMSE 全体'], 4)} & {n(d['notff_rmse']['全体'], 4)}{DAG} & {B(n(q, 3), not (1 / RMSE_RATIO <= q <= RMSE_RATIO))} & {concl} & "
                    f"{M(key)} & {n(dk, 2) if key != 'C' else '---'} & {('{:g}'.format(crit)) if key != 'C' else '---'} & "
                    f"{n(ls, 1) if pd.notna(ls) else '---'} & {n(d['lml_tffonly'] - d['lml_notff'], 2)} \\\\")
    return "\n".join(rows)


def tff_thick():
    """厚みごと：Tff だけを入力にしたモデル（A_Tset・B_Tset の θ = 0）と、Tset だけを入力にしたモデル（A_Tff・B_Tff の θ = 0 ＝ C_noTff）の LML の差"""
    hs = sorted({float(h) for c in CASES for h in D[c]["prof"]["厚み"]})
    out = []
    global L0_SPREAD
    L0_SPREAD = []      # 同じモデル（係数 0）の LML の、A・B・型による違い（厚みごとの格子の局所解の大きさの目安）
    for ver, mat in CASES:
        p = D[(ver, mat)]["prof"].assign(L0=lambda x: x["LML"] - x["係数0との差"])
        for h, q in p.groupby("厚み"):
            for t in ("_Tset_", "_Tff_"):
                s = q[q["モデル"].str.contains(t)]["L0"]
                L0_SPREAD.append({"ver": ver, "mat": mat, "h": float(h), "t": t.strip("_"), "spread": float(s.max() - s.min())})
    for ver, mat in CASES:
        p = D[(ver, mat)]["prof"].assign(L0=lambda x: x["LML"] - x["係数0との差"])
        cells = []
        for h in hs:
            q = p[np.isclose(p["厚み"], h)]
            if len(q) == 0:
                cells.append("---")
                continue
            a = q[q["モデル"].str.contains("_Tset_")]["L0"].max()
            b = q[q["モデル"].str.contains("_Tff_")]["L0"].max()
            cells.append(n(a - b, 2))
        out.append(f"{ver} & {MJ[mat]} & " + " & ".join(cells) + " \\\\")
    return hs, "\n".join(out)


# ---------------------------------------------------------------- 2. 全厚みのランキング
def rule_label(ver, mat, key):
    d = D[(ver, mat)]
    if key == NO_TFF:
        return "参考（Tff を使わない）"
    if key == "C":
        return B("本番（どのシフトも支持されない）", True) if d["key"] == "C" else "基準（シフトなし）"
    if key == d["key"]:
        return B("本番", True)
    s, t, f = key.split("_")
    form = "アレニウス型" if f == "Arr" else "WLF型"
    c = d["ctab"][(d["ctab"]["シフトに使う温度"] == t) & (d["ctab"]["型"] == form)]["この組の結論"].iloc[0]
    if c == key:
        return "組で残る（LML で負け）"
    if c == "支持されない":
        return "支持されない"
    if str(c).startswith("決まらない"):
        return "決まらない（★）"
    return f"組では {M(c)}"


def tab_total(ver, mat):
    d = D[(ver, mat)]
    S = d["sum"].copy()
    keys = list(S["モデル"]) + [NO_TFF]
    lml = list(S["LML"]) + [d["lml_notff"]]
    rm = {k: (S.set_index("モデル").loc[k, "RMSE 全体"] if k != NO_TFF else d["notff_rmse"]["全体"]) for k in keys}
    order = np.argsort(-np.array(lml), kind="stable")
    rk = rank(lml)
    rr = dict(zip(keys, rank([rm[k] for k in keys], ascending=True)))
    best = max(lml)
    out = []
    for i in order:
        k = keys[i]
        if k == NO_TFF:
            row = {"θ": "", "SHIFT_STRUCT": "", "θ=0 との LML の差": np.nan, "支持の基準": np.nan, "尤度で支持": None,
                   "RMSE 全体": d["notff_rmse"]["全体"], "RMSE 低速側": d["notff_rmse"].get("低速側", np.nan), "RMSE 高速側": d["notff_rmse"].get("高速側", np.nan),
                   "被覆率 全体": d["notff_cov"]}
        else:
            row = S.set_index("モデル").loc[k].to_dict()
        for c in ("RMSE 低速側", "RMSE 高速側"):        # 温度適切版の検証データは高速側だけ（低速側の列がない）
            row.setdefault(c, np.nan)
        sup = row.get("尤度で支持")
        sup_s = "---" if k in ("C", NO_TFF) else ("○" if bool(sup) else "×")
        dag = DAG if k == NO_TFF else ""
        out.append(f"{rk[i]} & {B(M(k), k == d['key'])} & {n_params(k)} & {theta_str(row) if k not in ('C', NO_TFF) else '---'} & {n(lml[i], 2)} & "
                   f"{n(lml[i] - best, 2)} & {n(row['θ=0 との LML の差'], 2) if k not in ('C', NO_TFF) else '---'} & "
                   f"{'{:g}'.format(row['支持の基準']) if k not in ('C', NO_TFF) else '---'} & {sup_s} & {rule_label(ver, mat, k)} & "
                   f"{n(row['RMSE 全体'], 4)}{dag} & {n(row['RMSE 低速側'], 4)}{dag if pd.notna(row['RMSE 低速側']) else ''} & {n(row['RMSE 高速側'], 4)}{dag} & "
                   f"{rr[k]} & {n(row['被覆率 全体'], 2)}{dag} \\\\")
    return "\n".join(out)


def total_sections():
    out = []
    for ver, mat in CASES:
        d = D[(ver, mat)]
        out.append(f"""\\subsection{{{ver}・{MJ[mat]}（本番：{M(d['key'])}）}}
\\begin{{table}}[H]\\centering\\footnotesize
\\caption{{全厚みのランキング：{ver}・{MJ[mat]}（LML の大きい順。\\file{{{rel(os.path.join(d['base'], *SF))}/}}）}}
\\begin{{tabular}}{{rlrp{{12\\zw}}rrrrcp{{8.5\\zw}}rrrrr}}
\\toprule
順位 & モデル & 数 & 係数（最良［95\\% 区間］） & LML & 最大との差 & $\\Delta$LML & 基準 & 支持 & 候補の選び方での扱い & RMSE 全体 & 低速側 & 高速側 & RMSE 順位 & 被覆率 \\\\
\\midrule
{tab_total(ver, mat)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")
    return "\n".join(out)


# ---------------------------------------------------------------- 3. 厚みごとのランキング
def tab_thick(ver, mat, h):
    d = D[(ver, mat)]
    p = d["prof"][np.isclose(d["prof"]["厚み"], h)].copy()
    rws = [{"モデル": r["モデル"], "LML": r["LML"], "dLML_vs_0": r["係数0との差"]} for _, r in p.iterrows()]
    w2, _, _ = mc.shift_candidates(rws)
    w2 = w2 or "C"
    p["L0"] = p["LML"] - p["係数0との差"]
    tc = d["tchk"][np.isclose(d["tchk"]["厚み"], h)].set_index("モデル")
    cm = d["cmp"][(d["cmp"]["範囲"] == "全体") & (d["cmp"]["厚み"].astype(str) != "全厚み")].copy()
    cm = cm[np.isclose(cm["厚み"].astype(float), h)]
    err_t = cm[cm["学習"] == "厚みごと"].set_index("モデル")["誤差_%"]
    err_a = cm[cm["学習"] == "総合"].set_index("モデル")["誤差_%"]
    keys = list(p["モデル"]) + [NO_TFF, "C"]
    lml = list(p["LML"]) + [float(p[p["モデル"].str.contains("_Tff_")]["L0"].max()), np.nan]
    rk = rank(lml)
    re_t = dict(zip(keys, rank([err_t.get(k, np.nan) for k in keys], ascending=True)))
    best = np.nanmax(lml)
    P = p.set_index("モデル")
    order = list(np.argsort(-np.nan_to_num(np.array(lml, float), nan=-1e18), kind="stable"))
    out = []
    for i in order:
        k = keys[i]
        if k in P.index:
            r = P.loc[k]
            coef = coef_thick(k, r["横の係数"], r["縦の係数"], r["単位"]) + ("\\textsuperscript{§}" if r["縦の係数"] < 0 else "")
            dl, lim = r["係数0との差"], r["支持の基準"]
            sup = "○" if dl >= lim else "×"
            conv = "収束" if bool(r["取り直しが収束"]) else B("未収束", True)
            dls, lims = n(dl, 2), "{:g}".format(lim)
        else:
            coef, dls, lims, sup, conv = "---", "---", "---", "---", "---"
        mark = []
        if k == d["key"]:
            mark.append(B("本番", True))
        if k == w2:
            mark.append("厚みの規則")
        out.append(f"{rk[i]} & {B(M(k), k == d['key'])} & {coef} & {n(lml[i], 2)} & {n(lml[i] - best, 2) if not np.isnan(lml[i]) else '---'} & {dls} & {lims} & {sup} & {conv} & "
                   f"{tc.loc[k, '判定'] if k in tc.index else '---'} & {n(err_t.get(k, np.nan), 1)} & {n(err_a.get(k, np.nan), 1)} & {re_t[k]} & {'・'.join(mark)} \\\\")
    return "\n".join(out), int(p["学習の条件数"].iloc[0])


def tab_group(ver, mat):
    t = D[(ver, mat)]["tsum"]
    out = []
    for _, r in t.iterrows():
        out.append(f"{float(r['厚み']):g} mm & {M(r['組'])} & {n(r['ずれ_Tset_%'], 1)} & {n(r['ずれ_Tff_%'], 1)} & {r['適した温度(a)']} & {n(r['ΔL（Tff − Tset）'], 2)} & "
                   f"{r['適した温度(b)']} & {n(r['誤差_Tset_%'], 1)} & {n(r['誤差_Tff_%'], 1)} & {r['適した温度(c)']} & {B(r['総合の判定'], r['総合の判定'] != '差なし')} \\\\")
    return "\n".join(out)


def thick_sections():
    out = []
    for ver, mat in CASES:
        d = D[(ver, mat)]
        out.append(f"\\subsection{{{ver}・{MJ[mat]}（本番：{M(d['key'])}）}}")
        for h in sorted({float(x) for x in d["prof"]["厚み"]}):
            body, nc = tab_thick(ver, mat, h)
            out.append(f"""\\begin{{table}}[H]\\centering\\footnotesize
\\caption{{厚みごとのランキング：{ver}・{MJ[mat]}・厚み {h:g} mm（学習の条件 {nc}。LML の大きい順）}}
\\begin{{tabular}}{{rlp{{11\\zw}}rrrrccc rrrl}}
\\toprule
順位 & モデル & 係数（厚みごと） & LML & 最大との差 & $\\Delta$LML & 基準 & 支持 & 取り直し & 確認 & 誤差 厚みごと & 誤差 総合 & 誤差順位 & 印 \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")
        out.append(f"""\\begin{{table}}[H]\\centering\\footnotesize
\\caption{{Tff か Tset か（組ごと・厚みごと）：{ver}・{MJ[mat]}（厚みごとの比較のノートの 5〜8 節。\\file{{{rel(os.path.join(d['base'], *TH, 'シフトに使う温度', '厚みごとの比較_まとめ.csv'))}}}）}}
\\begin{{tabular}}{{llrrlrlrrll}}
\\toprule
厚み & 組 & (a) ずれ Tset \\% & ずれ Tff \\% & (a) & (b) $\\Delta L$（Tff − Tset） & (b) & (c) 誤差 Tset \\% & 誤差 Tff \\% & (c) & 多数決 \\\\
\\midrule
{tab_group(ver, mat)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
\\clearpage""")
    return "\n".join(out)


def tab_files():
    out = []
    for ver, mat in CASES:
        out.append(f"{ver} & {MJ[mat]} & \\file{{{rel(D[(ver, mat)]['base'])}/}} \\\\")
    return "\n".join(out)


# ---------------------------------------------------------------- 原稿
PRE = r"""\documentclass[paper=a4paper,landscape,fontsize=10pt,line_length=62zw,number_of_lines=34]{jlreq}
\usepackage[haranoaji]{luatexja-preset}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{float}
\usepackage[hidelinks,bookmarksnumbered]{hyperref}
\newcommand{\file}[1]{\texttt{\detokenize{#1}}}
\newcommand{\mdl}[1]{\texttt{\detokenize{#1}}}
\let\origtextbf\textbf
\renewcommand{\textbf}[1]{\origtextbf{\gtfamily #1}}   % 日本語の強調はゴシック体（明朝の太字の書体がないため）
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{3pt}
\title{材料別の温度シフト GPR\\本番学習の設定の根拠（ランキング表）}
\author{シフトファクター検討用のノート（11 月 4 日実行）と厚みごとの比較のノート（11 月 5 日実行）の結果から}
\date{@@DATE@@}
\begin{document}
\maketitle
"""

BODY = r"""
\section*{この資料について}
\begin{itemize}[itemsep=0.2ex,topsep=0.3ex]
\item 「本番学習の設定の決定.pdf」の根拠を、\textbf{材料 × 学習データ（通常版・温度適切版）ごとに、すべてのモデルのランキング表}にしたものです（2026 年 11 月 6 日のユーザー依頼）。1 章に、流動中樹脂温度 Tff を GP の入力に使ったほうがよいかの表も載せました。
\item 数値は、検討結果フォルダの CSV から、生成スクリプト \file{04_補助スクリプト/build_report_mat_ranking.py} が読んだものです（学習はしていません）。読んだフォルダは付録の表~\ref{tab:files}。
\item \textbf{LML（対数周辺尤度）は、同じ材料・同じ学習データ（・同じ厚み）の中でだけ比べます}。材料や学習データが違うと、LML の大きさは、データの行数とばらつきで決まるためです。順位も、その中だけのものです。
\item パラメータの数（表の「数」）が多いモデルほど、LML は大きく出やすくなります。そのため、差には基準を当てます。係数 1 つで 1.92、2 つで 3.0 です（\textbf{$\Delta$LML} は、シフトの係数を 0 にしたモデルとの LML の差）。本番のモデルは、LML の順位だけでなく、この基準と「候補の選び方」（2026 年 11 月 3 日のユーザー決定）で決めています。そのため、LML が 1 位でないモデルが本番になることがあります。
\item モデルの記号は「構造\_シフトに使う温度\_型」です。構造 A は縦だけ、B は横・縦のシフト、温度は Tset（加熱シリンダ温度）・Tff（流動中樹脂温度）、型は Arr（アレニウス型）・WLF（WLF 型）。C はシフトなしで、4 変数（$\ln\dot\gamma$・$\ln$ P3\_3・$\ln$ Tff・Tset）を並べた通常の GPR。\mdl{C_noTff} は C から Tff を外したもの（入力 $\ln\dot\gamma$・$\ln$ P3\_3・Tset）で、A\_Tff・B\_Tff の係数を 0 にしたものと同じモデルです。
\item {}†：\mdl{C_noTff} の 3data の誤差は、2026 年 11 月 6 日の確かめの実行（\file{_tmp/tff_input/run3/}。Tff の入力の有無の比較のノート）の値です。同じノートを本物の学習結果フォルダで実行すれば、同じ値になります。LML は、検討の結果の \mdl{LML_at_0} から取りました（確かめの実行の値と一致）。
\end{itemize}

\section{流動中樹脂温度 Tff を GP の入力に使ったほうがよいか}\label{sec:tff}
\begin{table}[H]\centering\scriptsize
\caption{Tff を入力に使うかの根拠（全厚み）。太字は基準を超えたもの}\label{tab:tff}
\begin{tabular}{llrrrrrrlllrrr}
\toprule
 & & \multicolumn{3}{c}{LML} & \multicolumn{3}{c}{3data の RMSE（全体）} & & \multicolumn{3}{c}{本番のモデル} & & \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){10-12}
学習データ & 材料 & C & \mdl{C_noTff} & 差 & C & \mdl{C_noTff} & 比 & C の結論 & モデル & \mdl{C_noTff} との差 & 基準 & Tff の長さスケール & Tff だけ − Tset だけ \\
\midrule
@@TAB_TFF@@
\bottomrule
\end{tabular}
\end{table}
\begin{itemize}[itemsep=0.2ex,topsep=0.3ex]
\item \textbf{C（シフトなし）}：Tff を入力に足しても、6 つとも LML は基準（$+1.92$）まで上がりません（差 $= $ LML(C) $-$ LML(\mdl{C_noTff})）。3data の誤差は 5 つで同等（比が 1/1.1〜1.1）です。温度適切版の粉砕材混合 PP だけ、Tff を外したほうが誤差が小さくなります（比 0.68。検証は高速側だけ）。
\item \textbf{Tff でシフトするモデル（A\_Tff・B\_Tff）}：Tff は入力ではなく、シフトに使います（GP の入力は $\ln\dot\gamma$・$\ln$ P3\_3・Tset）。そのシフトは、6 つとも支持されません（2 章の表）。係数 0 では \mdl{C_noTff} と同じモデルです。
\item \textbf{Tset でシフトするモデル（A\_Tset・B\_Tset。本番の 5 つ）}：Tff を GP の入力に使います。\textbf{Tff だけを外した同じ構造のモデル（Tset のシフト ＋ $f(\ln\dot\gamma, \ln$ P3\_3$)$）は計算していないので、直接の答えはありません}。手がかりは 3 つです。① Tff の長さスケールは 8〜32 と長く（入力は標準化していて、データの幅は約 $\pm2$）、GP の Matern の部分は Tff の向きにほとんど変わりません。② Tff だけを入力にしたモデル（A\_Tset の係数 0）は、Tset だけを入力にしたモデル（\mdl{C_noTff}）より LML が 2〜16 低く、温度の情報は Tset が持っています（表の右端）。③ 本番のモデルが \mdl{C_noTff} より基準以上大きいのは、GPPS の 2 つだけで、これは Tset のシフトの効果です。
\item \textbf{まとめ}：今の結果には、Tff を入力に入れて良くなったという結果は、どの材料・学習データ・モデルにもありません。C では、入れなくてよい（温度適切版の粉砕材混合 PP は、外したほうがよい）と言えます。Tset でシフトするモデルは、上の①〜③から入れなくてよい見込みですが、確かめるには、Tff の入力の有無の比較のノートに「Tset のシフトのモデルから Tff を外したもの」を足して計算する必要があります。Tff はユーザー指定の説明変数なので、外すかはユーザーが決めます。
\end{itemize}

\begin{table}[H]\centering\footnotesize
\caption{厚みごと：Tff だけを入力にしたモデル（A\_Tset・B\_Tset の係数 0）と、Tset だけを入力にしたモデル（A\_Tff・B\_Tff の係数 0 ＝ \mdl{C_noTff}）の LML の差（厚みごとのプロファイル尤度の値から。負なら Tset のほうがよく説明する）}\label{tab:tffthick}
\begin{tabular}{ll@@THCOLS@@}
\toprule
学習データ & 材料 & @@THHEAD@@ \\
\midrule
@@TAB_TFFTH@@
\bottomrule
\end{tabular}
\end{table}
厚みごとの C の LML は、厚みごとの比較のノートが残していないので、厚みごとの「C $-$ \mdl{C_noTff}」は出せません。また、厚みごとの係数 0 の LML は、同じモデルでも、A・B・型のどの格子から取るかで最大 @@SPREADMAX@@ 違います（@@SPREADN@@ 組のうち @@SPREAD1@@ 組で 1 以上。厚みごとのプロファイル尤度は格子の各点で最適化のやり直しが 1 回なので、局所解に止まった点があるため）。この表と 3 章の \mdl{C_noTff} の LML は、その中の最大の値を使いました。厚みごとの LML の差は、この大きさの不確かさを含みます。

\clearpage
\section{全厚みの根拠（ランキング表）}\label{sec:total}
\begin{itemize}[itemsep=0.2ex,topsep=0.3ex]
\item 読んだ CSV：\file{まとめ.csv}（LML・$\Delta$LML・基準・支持・3data の RMSE・被覆率・係数）、\file{候補の選び方.csv}（候補の選び方での扱い）。LML の大きい順。本番のモデルは太字。
\item 列：「数」はパラメータの数（\mdl{C_noTff} 7、C・A 8、B 9）。「支持」は $\Delta$LML が基準以上なら ○。「候補の選び方での扱い」は、列と型の組ごとの結論（本番／組で残ったが LML で負け／組では別の構造〔A か B〕／支持されない）。「RMSE 順位」は 3data の RMSE（全体）の小さい順。被覆率は 3data が 95\% 予測区間に入った割合（全体）。温度適切版の検証データは高速側だけなので、低速側は「---」。
\end{itemize}
@@TOTAL@@

\clearpage
\section{厚みごとの根拠（ランキング表）}\label{sec:thick}
\begin{itemize}[itemsep=0.2ex,topsep=0.3ex]
\item 読んだ CSV（厚みごとの比較のノート）：\file{プロファイル尤度_厚みごと.csv}（その厚みの 5data だけで係数を決め直したときの LML・係数・$\Delta$LML・取り直しが収束したか）、\file{温度シフトの確認_厚みごと.csv}（全厚みのモデルで見た、その厚みの温度シフトの判定 ○・△・×）、\file{厚みごとと総合の比較.csv}（3data のその厚みの誤差〔粘度の \%〕。「厚みごと」はその厚みだけで学習したモデル、「総合」は全厚みのモデル）、\file{厚みごとの比較_まとめ.csv}（組ごとの Tff か Tset か）。
\item LML は、その厚みの中だけで比べます。C の厚みごとの LML は残っていないので「---」です。\mdl{C_noTff} の LML は、A\_Tff・B\_Tff の係数 0 の値です（LML $-$ $\Delta$LML の最大）。厚みごとの LML は、局所解のため最大 @@SPREADMAX@@ ほどの不確かさを含みます（1 章の表~\ref{tab:tffthick} の下）。差が 1〜3 程度の順位の入れ替わりは、意味のある差とは言えません。
\item 「印」：本番＝全厚みで選んだモデル（厚みごとの学習にも、このモデルを使う）。厚みの規則＝その厚みのプロファイル尤度に「候補の選び方」を当てたときの候補（参考）。§：縦の係数が負（低温ほど粘度が低いという、逆の向き）。
\item 「確認」：○＝温度の順序と温度による粘度の比が実測と合い、換算でずれが減る。×＝予測の温度の順序が実測と逆。△＝それ以外。
\end{itemize}
@@THICK@@

\appendix
\section{読んだ検討結果フォルダ}
\begin{table}[H]\centering\footnotesize
\caption{検討結果フォルダ（この下の \file{全ショット版/全厚み総合/シフトファクターの検討/} と \file{全ショット版/厚みごと/} の CSV を読んだ）}\label{tab:files}
\begin{tabular}{lll}
\toprule
学習データ & 材料 & フォルダ \\
\midrule
@@TAB_FILES@@
\bottomrule
\end{tabular}
\end{table}
\end{document}
"""

hs, tff_th = tff_thick()
SP = pd.DataFrame(L0_SPREAD)
V = {"DATE": rc.today_ja(), "TAB_TFF": tff_summary(), "TAB_TFFTH": tff_th,
     "SPREADMAX": n(SP["spread"].max(), 1), "SPREADN": len(SP), "SPREAD1": int((SP["spread"] >= 1).sum()), "THCOLS": "r" * len(hs), "THHEAD": " & ".join(f"{h:g} mm" for h in hs),
     "TOTAL": total_sections(), "THICK": thick_sections(), "TAB_FILES": tab_files()}
body = BODY
for k, v in V.items():
    body = body.replace(f"@@{k}@@", str(v))
body = re.sub(r"\\(?:file|mdl)\{([^{}]*)\}", lambda m: tt(m.group(1)), body)
tex = PRE.replace("@@DATE@@", V["DATE"]) + body
left = sorted(set(re.findall(r"@@([A-Za-z0-9_]+)@@", tex)))
assert not left, f"置き換えていない項目: {left}"

for ver, mat in CASES:      # 確かめ用に画面にも
    d = D[(ver, mat)]
    print(f"{ver} {MJ[mat]}: 本番 {d['key']}、LML(C) − LML(C_noTff) = {float(d['S'].loc['C', 'LML']) - d['lml_notff']:+.2f}、"
          f"RMSE 比 C_noTff/C = {d['notff_rmse']['全体'] / float(d['S'].loc['C', 'RMSE 全体']):.3f}")

os.makedirs(out_dir, exist_ok=True)
with open(TEX, "x", encoding="utf-8", newline="\n") as f:     # "x"：同じ名前があれば止まる
    f.write(tex)
aux = os.path.join(W, "_tmp", "report_mat_ranking", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(aux, exist_ok=False)
for i in range(2):
    r = subprocess.run([rc.LUALATEX, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={aux}", f"{STEM}.tex"], cwd=out_dir, capture_output=True)
    if r.returncode != 0:
        print(r.stdout.decode("utf-8", "replace")[-3000:])
        sys.exit(f"lualatex が失敗しました（{i + 1} 回目）。tex は {rel(TEX)}")
if os.path.exists(PDF):
    sys.exit(f"同じ名前の PDF ができていたので止めました（上書きしない）: {rel(PDF)}")
shutil.copy2(os.path.join(aux, f"{STEM}.pdf"), PDF)
print("作成:", rel(TEX), "と", rel(PDF), "（補助ファイル:", rel(aux), "）")
