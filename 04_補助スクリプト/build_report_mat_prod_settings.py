# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR（05 フォルダ）：検討の結果から、学習用のノート（本番）の設定を決め、その結論と根拠を PDF の報告書にする（2026-11-05 のユーザー依頼）。

使い方（リポジトリ直下から）: uv run python 04_補助スクリプト/build_report_mat_prod_settings.py [--out <フォルダ>]
- 読むもの：3 材料 × 2 種類の学習データ（通常版＝名前の末尾 (5all学習データ)、温度適切版＝(5適切学習データ)）の検討結果フォルダ
  （<材料フォルダ>/<番号>_検討結果/*(5all学習データ)/ など）の CSV。シフトファクター検討用のノートの 全ショット版/全厚み総合/シフトファクターの検討/ と、
  厚みごとの比較のノートの 全ショット版/厚みごと/。ほかに、候補のモデルの学習済みモデル（.joblib）に記録された設定。
- 全厚みの設定は、検討用のノートと同じ「候補の選び方」（mat_common.shift_candidates。2026-11-03 のユーザー決定）で決め、候補の選び方.csv の結論と一致することを確かめる。
- 厚みごとの設定は、全厚みと同じモデルにする（理由は報告書の 4 章）。参考に、厚みごとのプロファイル尤度に同じ規則を当てた結果も出す。
- 書くもの：--out のフォルダ（リポジトリ直下からの相対パス。省くと 05_材料ごとの機械学習データと予測検証データ/）に 本番学習の設定の決定.tex と .pdf。
  同じ名前のファイルがあれば止まる（上書きしない）。LuaLaTeX の補助ファイルは _tmp/report_mat_prod/<日時>/ に出す。
"""
import datetime
import glob
import os
import re
import shutil
import subprocess
import sys
import warnings

import pandas as pd

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mat_common as mc  # noqa: E402
import report_common as rc  # noqa: E402

W = rc.W
ROOT_REL = "05_材料ごとの機械学習データと予測検証データ"
ROOT = os.path.join(W, ROOT_REL)
FOLD = {"PP": "01_バージンPP再々実験", "MIX": "02_粉砕材混合PP実験", "GPPS": "03_GPPS実験"}
MJ = {"PP": "PP", "MIX": "粉砕材混合 PP", "GPPS": "GPPS"}
VERS = {"通常版": "(5all学習データ)", "温度適切版": "(5適切学習データ)"}
STEM = "本番学習の設定の決定"
NB = "01_機械学習コードフォルダ/材料別_温度シフトファクター検討/材料別_温度シフト機械学習コード/材料別_温度シフトGPR_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb"
SF = ("全ショット版", "全厚み総合", "シフトファクターの検討")
TH = ("全ショット版", "厚みごと")
COLS = {"Tset": "加熱シリンダ温度[℃]", "Tff": "流動中樹脂温度(℃)"}
FORM_SET = {"Arr": "arrhenius", "WLF": "wlf"}
CASES = [(v, m) for v in VERS for m in FOLD]
HS = [0.5, 1.0, 1.5, 2.0]
DAG, DDAG = "\\textsuperscript{†}", "\\textsuperscript{‡}"

out_dir = os.path.join(W, sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else ROOT
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
    """リポジトリ直下からのパス（/ 区切り。05 フォルダは 05_…/ と短くする）"""
    return os.path.relpath(p, W).replace("\\", "/").replace(ROOT_REL + "/", "05_…/")


def g(x):
    """係数などを、余分な 0 を付けずに（負の数は数式のマイナス）"""
    x = float(x)
    s = f"{abs(x):g}"
    return f"$-{s}$" if x < 0 else s


def n(x, nd):
    return rc.num(float(x), nd)


def M(k):
    return f"\\mdl{{{k}}}"


def B(s, cond):
    return f"\\textbf{{{s}}}" if cond else s


def settings_of(key):
    """モデルの記号 → 学習用のノートの (SHIFT_COL, SHIFT_STRUCT, SHIFT_FORM) の書き方"""
    if key == "C":
        return "None", "（使わない）", "（使わない）"
    s, t, f = key.split("_")
    return f'"{COLS[t]}"', f'"{s}"', f'"{FORM_SET[f]}"'


def theta_str(row):
    """まとめ.csv の θ の列 → 報告書の書き方（構造 A は横の係数を書かない）"""
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


def ver_row(ver, ncol):
    return f"\\multicolumn{{{ncol}}}{{l}}{{\\textbf{{{ver}}}}} \\\\"


def tt(s):
    """\\file{…}・\\mdl{…} の中身 → 等幅の文字（LaTeX の特殊文字を逃がし、/ と _ の後ろで改行できるようにする。
    \\detokenize だと、日本語の字間が開いてしまうため）"""
    out = []
    for ch in s:
        out.append({"_": "\\_", "%": "\\%", "&": "\\&", "#": "\\#", "$": "\\$", "\\": "/", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}"}.get(ch, ch))
        if ch in "/_\\":
            out.append("\\allowbreak{}")
    return "\\texttt{" + "".join(out) + "}"


def mat_rel(mat, p):
    """材料フォルダからのパス（/ 区切り）"""
    return os.path.relpath(p, os.path.join(ROOT, FOLD[mat])).replace("\\", "/")


# ---------------------------------------------------------------- 読み込みと、全厚みの候補
D = {}
for ver, mat in CASES:
    b = base(ver, mat)
    d = {"base": b, "data": rd(b, "データ確認", "データの概要.csv"), "sum": rd(b, *SF, "まとめ.csv"), "ctab": rd(b, *SF, "候補の選び方.csv"),
         "jd": rd(b, *SF, "判定の一覧.csv"), "prof": rd(b, *TH, "シフトに使う温度", "プロファイル尤度_厚みごと.csv"),
         "tchk": rd(b, *TH, "温度シフトの確認", "温度シフトの確認_厚みごと.csv"), "tsum": rd(b, *TH, "シフトに使う温度", "厚みごとの比較_まとめ.csv"),
         "cmp": rd(b, *TH, "厚みごとのモデルと総合の比較", "厚みごとと総合の比較.csv"),
         "cmps": rd(b, *TH, "厚みごとのモデルと総合の比較", "厚みごとと総合の比較_まとめ.csv")}
    rows = [{"モデル": r["モデル"], "LML": r["LML"], "dLML_vs_0": r["θ=0 との LML の差"]} for _, r in d["sum"].iterrows() if r["モデル"] != "C"]
    win, _, notes = mc.shift_candidates(rows)
    surv = [c for c in d["ctab"]["この組の結論"] if c != "支持されない" and not str(c).startswith("決まらない")]
    lml = d["sum"].set_index("モデル")["LML"]
    win_csv = max(surv, key=lambda k: lml[k]) if surv else None
    assert win == win_csv, f"{ver} {mat}：規則の当てはめ直し {win} と 候補の選び方.csv の結論 {win_csv} が違う"
    assert not notes, notes
    d["key"] = win or "C"
    d["S"] = d["sum"].set_index("モデル")
    D[(ver, mat)] = d


def model_record(ver, mat, key):
    """学習済みモデル（検討用のノートが保存したもの）に記録された設定"""
    import __main__
    import joblib
    for k, v in vars(mc).items():      # 学習済みモデルの中の部品（ノートで定義したクラス）を読めるようにする
        if isinstance(v, type) and not hasattr(__main__, k):
            setattr(__main__, k, v)
    p = glob.glob(os.path.join(D[(ver, mat)]["base"], *SF, mc.MODEL_DIRS[key], f"学習済みモデル_{key}.joblib"))
    assert len(p) == 1, p
    return joblib.load(p[0])


for ver, mat in CASES:
    key = D[(ver, mat)]["key"]
    r = model_record(ver, mat, key)
    want = mc.shift_cfg(mat, FORM_SET[key.split("_")[2]] if key != "C" else "arrhenius")
    got = r["シフトファクター"]
    assert r["入力の取り方"] == "shot", r["入力の取り方"]
    assert all(got.get(k) == v for k, v in want.items()), (ver, mat, got, want)
    files = D[(ver, mat)]["data"].set_index("組")["ファイル"]
    assert r["学習データのファイル"].replace("\\", "/") == files["5data"].replace("\\", "/"), (r["学習データのファイル"], files["5data"])

# ---------------------------------------------------------------- 厚みごと：選んだモデルの値と、厚みごとに規則を当てた候補
TROWS = []
for ver, mat in CASES:
    d = D[(ver, mat)]
    key = d["key"]
    for h, gp in d["prof"].groupby("厚み"):
        rws = [{"モデル": r["モデル"], "LML": r["LML"], "dLML_vs_0": r["係数0との差"]} for _, r in gp.iterrows()]
        w2, _, n2 = mc.shift_candidates(rws)
        P = gp.set_index("モデル")
        rec = {"ver": ver, "mat": mat, "h": float(h), "n_cond": int(gp["学習の条件数"].iloc[0]), "rule": w2 or "C",
               "rule_v": P.loc[w2, "縦の係数"] if w2 else float("nan"), "rule_note": bool(n2),
               "n_unconv": int((~gp["取り直しが収束"].astype(bool)).sum()), "n_models": len(gp), "tv": float("nan")}
        if key != "C":
            r = P.loc[key]
            rec.update({"th": r["横の係数"], "tv": r["縦の係数"], "unit": r["単位"], "dl": r["係数0との差"], "lim": r["支持の基準"],
                        "conv": bool(r["取り直しが収束"])})
        TROWS.append(rec)
T = pd.DataFrame(TROWS)


# ---------------------------------------------------------------- 表
def tab_conclusion():
    out = []
    for ver, mat in CASES:
        key = D[(ver, mat)]["key"]
        c, s, f = settings_of(key)
        out.append(f"{ver} & {MJ[mat]} & {M(key)} & \\mdl{{{c}}} & \\mdl{{{s}}} & \\mdl{{{f}}} \\\\")
        if (ver, mat) == ("通常版", "GPPS"):
            out.append("\\midrule")
    return "\n".join(out)


def runs():
    """実行の分け方：SHIFT_COL・SHIFT_STRUCT・SHIFT_FORM は MATERIALS のすべての材料に同じものを使うので、同じモデルの材料を 1 回にまとめる"""
    out = []
    for ver in VERS:
        by = {}
        for mat in FOLD:
            by.setdefault(D[(ver, mat)]["key"], []).append(mat)
        out += [(ver, key, mats) for key, mats in by.items()]
    return out


RUNS = runs()
CIRC = "①②③④⑤⑥⑦⑧"


def tab_runs():
    out = []
    for i, (ver, key, mats) in enumerate(RUNS):
        c, s, f = settings_of(key)
        ml = "[" + ", ".join(f'"{m}"' for m in mats) + "]"
        out.append(f"{CIRC[i]} & {ver} & \\mdl{{{ml}}} & {M(key)} & \\mdl{{{c}}} & \\mdl{{{s}}} & \\mdl{{{f}}} \\\\")
    return "\n".join(out)


def run_files():
    out = []
    for i, (ver, key, mats) in enumerate(RUNS):
        out.append(f"\\item[{CIRC[i]}] {ver}・{'、'.join(MJ[m] for m in mats)}（結果は \\file{{全ショット版/全厚み総合/{mc.MODEL_DIRS[key]}/}}）")
        out.append("\\begin{itemize}[itemsep=0.1ex,topsep=0.2ex]")
        for m in mats:
            dg = D[(ver, m)]["data"].set_index("組")
            out.append(f"\\item {MJ[m]}（\\file{{05_…/{FOLD[m]}/}} から）：学習 \\file{{{mat_rel(m, os.path.join(W, dg.loc['5data', 'ファイル']))}}}、"
                       f"検証 \\file{{{mat_rel(m, os.path.join(W, dg.loc['3data', 'ファイル']))}}}、"
                       f"学習結果フォルダ \\file{{{mat_rel(m, D[(ver, m)]['base'])}/}}")
        out.append("\\end{itemize}")
    return "\n".join(out)


def tab_candidates():
    out = []
    for ver in VERS:
        out.append(ver_row(ver, 7))
        for mat in FOLD:
            d = D[(ver, mat)]
            for j, (_, r) in enumerate(d["ctab"].iterrows()):
                a, b, ba = r["A の ΔLML"], r["B の ΔLML"], r["B − A の LML"]
                concl = r["この組の結論"]
                concl = M(concl) + ("（候補）" if concl == d["key"] else "") if "_" in str(concl) else concl
                out.append(f"{MJ[mat] if j == 0 else ''} & {r['シフトに使う温度']} & {r['型']} & {B(n(a, 2), a >= mc.LRT_DLML_1)} & "
                           f"{B(n(b, 2), b >= mc.LRT_DLML)} & {B(n(ba, 2), ba >= mc.LRT_DLML_1)} & {concl} \\\\")
        if ver != list(VERS)[-1]:
            out.append("\\midrule")
    return "\n".join(out)


def tab_cand_detail():
    out = []
    for ver in VERS:
        out.append(ver_row(ver, 9))
        for mat in FOLD:
            d = D[(ver, mat)]
            key, S = d["key"], d["S"]
            r, c = S.loc[key], S.loc["C"]
            low = n(r["RMSE 低速側"], 4) if "RMSE 低速側" in S.columns else "---"
            out.append(f"{MJ[mat]} & {M(key)} & {theta_str(r) if key != 'C' else '---'} & {n(r['LML'], 1)} & {n(r['RMSE 全体'], 4)} & {low} & "
                       f"{n(r['RMSE 高速側'], 4)} & {n(r['RMSE 全体'] / c['RMSE 全体'], 2)} & {n(r['被覆率 全体'], 2)} \\\\")
        if ver != list(VERS)[-1]:
            out.append("\\midrule")
    return "\n".join(out)


def tab_rmse_rank():
    out = []
    for ver in VERS:
        out.append(ver_row(ver, 7))
        for mat in FOLD:
            d = D[(ver, mat)]
            key, S = d["key"], d["S"]
            srt = S.sort_values("RMSE 全体")
            rank = list(srt.index).index(key) + 1
            ties = [k for k in srt.index if S.loc[k, "RMSE 全体"] - srt["RMSE 全体"].iloc[0] < 5e-5]    # 4 桁で同じ RMSE のモデル（B の横の係数が 0 で A と同じになったものなど）
            sup = "---" if ties == ["C"] else ("支持" if any(k != "C" and bool(S.loc[k, "尤度で支持"]) for k in ties) else "支持されない")
            out.append(f"{MJ[mat]} & {M(key)} & {n(S.loc[key, 'RMSE 全体'], 4)} & {rank} / {len(S)} & {'・'.join(M(k) for k in ties)} & "
                       f"{n(srt['RMSE 全体'].iloc[0], 4)} & {sup} \\\\")
        if ver != list(VERS)[-1]:
            out.append("\\midrule")
    return "\n".join(out)


def tab_data_h():
    out = []
    for ver in VERS:
        out.append(ver_row(ver, 6))
        for mat in FOLD:
            d = D[(ver, mat)]
            cm = d["cmp"]
            cells = []
            for h in HS:
                t = T[(T["ver"] == ver) & (T["mat"] == mat) & (T["h"] == h)]
                v = cm[(cm["モデル"] == "C") & (cm["学習"] == "厚みごと") & (cm["範囲"] == "全体") & (cm["厚みの名前"].astype(str) == f"{h:g}")]
                cells.append(f"{int(t['n_cond'].iloc[0])} / {int(v['条件数'].iloc[0])}" if len(t) else "---")
            dg = d["data"].set_index("組")
            out.append(f"{MJ[mat]} & {' & '.join(cells)} & {int(dg.loc['5data', '条件数'])} / {int(dg.loc['3data', '条件数'])} \\\\")
        if ver != list(VERS)[-1]:
            out.append("\\midrule")
    return "\n".join(out)


def temp_counts():
    """8 節の多数決（総合の判定）の数と、4 節の判定（○・△・×）の数"""
    tot, rows = {"Tff": 0, "Tset": 0, "差なし": 0}, {}
    for ver, mat in CASES:
        d = D[(ver, mat)]
        vc = d["tsum"]["総合の判定"].value_counts().to_dict()
        for k in tot:
            tot[k] += vc.get(k, 0)
        tc = d["tchk"]
        is_set = tc["モデル"].str.contains("Tset")
        rows[(ver, mat)] = (vc, tc[is_set]["判定"].value_counts().to_dict(), tc[~is_set]["判定"].value_counts().to_dict(), len(d["tsum"]),
                            int(is_set.sum()), int((~is_set).sum()))
    return tot, rows


TOT, TCR = temp_counts()
assert all(set(TCR[c][1]) == {"○"} for c in CASES), "Tset のモデルの 4 節の判定に ○ でないものがある（報告書の文を見直す）"
FF_OK = {c: TCR[c][2].get("○", 0) for c in CASES}
FF_N = {c: TCR[c][5] for c in CASES}


def tab_temp_choice():
    out = []
    for ver in VERS:
        out.append(ver_row(ver, 4))
        for mat in FOLD:
            vc, tset, tff, nall, nset, nff = TCR[(ver, mat)]
            f8 = "・".join(f"{k} {vc.get(k, 0)}" for k in ["差なし", "Tff", "Tset"])
            f4s = "・".join(f"{k} {tset.get(k, 0)}" for k in ["○", "△", "×"])
            f4f = "・".join(f"{k} {tff.get(k, 0)}" for k in ["○", "△", "×"])
            out.append(f"{MJ[mat]} & {f8}（{nall}） & {f4s}（{nset}） & {f4f}（{nff}） \\\\")
        if ver != list(VERS)[-1]:
            out.append("\\midrule")
    return "\n".join(out)


def coef_str(key, th, tv, unit):
    if unit == "kJ/mol":
        s = f"$E_v$ {g(tv)}" if key.startswith("A") else f"$E_h$ {g(th)}・$E_v$ {g(tv)}"
    else:
        s = f"$A_{{1,v}}$ {g(tv)}" if key.startswith("A") else f"$A_{{1,h}}$ {g(th)}・$A_{{1,v}}$ {g(tv)}"
    return s + (DAG if tv < 0 else "")


def tab_thick_profile():
    out = []
    for i, (ver, mat) in enumerate(CASES):
        key = D[(ver, mat)]["key"]
        out.append(f"\\multicolumn{{6}}{{l}}{{\\textbf{{{ver}・{MJ[mat]}}}（選んだモデル {M(key)}）}} \\\\")
        for _, r in T[(T["ver"] == ver) & (T["mat"] == mat)].iterrows():
            if key == "C":
                mine = "--- & --- & ---"
            else:
                mine = f"{coef_str(key, r['th'], r['tv'], r['unit'])} & {B(n(r['dl'], 2), r['dl'] >= r['lim'])} & {'収束' if r['conv'] else B('未収束', True)}"
            if r["rule"] == "C":
                rule = "C（どの組も支持されない）"
            else:
                rule = f"{M(r['rule'])}（縦 {g(r['rule_v'])}{DAG if r['rule_v'] < 0 else ''}）" + (DDAG if r["rule_note"] else "")
            out.append(f"{r['h']:g} mm & {mine} & {rule} & {r['n_unconv']} / {r['n_models']} \\\\")
        if i < len(CASES) - 1:
            out.append("\\midrule")
    return "\n".join(out)


def tvt_values(ver, mat):
    d = D[(ver, mat)]
    cm = d["cmp"]
    c = cm[(cm["モデル"] == d["key"]) & (cm["範囲"] == "全体")]
    return {h: c[c["厚みの名前"].astype(str) == h].set_index("学習")["誤差_%"] for h in ["0.5", "1", "1.5", "2", "全厚み"]}


def tab_thick_vs_total():
    out = []
    for ver in VERS:
        out.append(ver_row(ver, 8))
        for mat in FOLD:
            d = D[(ver, mat)]
            cells = []
            for h, x in tvt_values(ver, mat).items():
                if len(x) == 0:
                    cells.append("---")
                else:
                    a, b = x["厚みごと"], x["総合"]
                    cells.append(f"{B(f'{a:.1f}', a < b)} / {B(f'{b:.1f}', b < a)}")
            verdict = d["cmps"].set_index("モデル").loc[d["key"], "勧める学習"]
            out.append(f"{MJ[mat]} & {M(d['key'])} & {' & '.join(cells)} & {verdict} \\\\")
        if ver != list(VERS)[-1]:
            out.append("\\midrule")
    return "\n".join(out)


def tvt_mismatch():
    """「勧める学習」と、全厚みの誤差の大小が食い違う材料"""
    out = []
    for ver, mat in CASES:
        d = D[(ver, mat)]
        v = d["cmps"].set_index("モデル").loc[d["key"]]
        tot = tvt_values(ver, mat)["全厚み"]
        better = "厚みごと" if tot["厚みごと"] < tot["総合"] else "総合"
        if not str(v["勧める学習"]).startswith(better):
            out.append(f"{ver}の{MJ[mat]}（「勧める学習」は{v['勧める学習']}〔厚みごとがよい厚み {int(v['厚みごとがよい厚み'])}、総合がよい厚み {int(v['総合がよい厚み'])}〕、"
                       f"全厚みの誤差は厚みごと {tot['厚みごと']:.1f}\\%・総合 {tot['総合']:.1f}\\%）")
    return "、".join(out) if out else ""


def tab_files():
    return "\n".join(f"\\item {ver}・{MJ[mat]}：\\file{{{rel(D[(ver, mat)]['base'])}/全ショット版/}}" for ver, mat in CASES)


def thick_runs():
    """厚みごとの実行：表 runs の実行ごとに、THICKNESS に書く値と、その厚みで学習する材料（その厚みのデータがある材料）"""
    out = []
    for i, (ver, key, mats) in enumerate(RUNS):
        hs = sorted({h for m in mats for h in T[(T["ver"] == ver) & (T["mat"] == m)]["h"]})
        by_h = {h: [m for m in mats if ((T["ver"] == ver) & (T["mat"] == m) & (T["h"] == h)).any()] for h in hs}
        out.append((i, ver, key, mats, hs, by_h))
    return out


TRUNS = thick_runs()


def tab_truns():
    out = []
    for i, ver, key, mats, hs, by_h in TRUNS:
        groups = {}
        for h, ms in by_h.items():
            groups.setdefault(tuple(ms), []).append(h)
        who = "／".join(f"{'・'.join(f'{h:g}' for h in hl)} mm：{'・'.join(MJ[m] for m in ms)}" for ms, hl in groups.items()) if len(groups) > 1 \
            else "・".join(MJ[m] for m in mats)
        out.append(f"{CIRC[i]} & {ver} & {M(key)} & {'・'.join(f'{h:g}' for h in hs)}（{len(hs)} 回） & {who} \\\\")
    return "\n".join(out)


def vis_len(s):
    """等幅の文字での見た目の幅（全角は 2）"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def split_path(p, width=46):
    """長いパスを、\\ の前で区切って、見た目の幅が width 以下のかたまりに分ける（2 つ目からは \\ で始まる。r"…" の最後が \\ にならないように）"""
    parts = p.split("\\")
    chunks, cur = [], parts[0]
    for s in parts[1:]:
        if vis_len(cur) + 1 + vis_len(s) <= width:
            cur += "\\" + s
        else:
            chunks.append(cur)
            cur = "\\" + s
    return chunks + [cur]


def files_snippet(ver):
    """設定 FILES の書き方の例（その学習データの 3 材料。学習データ・検証データは データの概要.csv、学習結果フォルダは検討と同じ）。
    Python として読めることと、書いた場所がすべてあることを確かめてから返す"""
    import ast
    lines = ["FILES = {"]
    for mat in FOLD:
        dg = D[(ver, mat)]["data"].set_index("組")
        items = [("学習", dg.loc["5data", "ファイル"]), ("検証", dg.loc["3data", "ファイル"]), ("結果", os.path.relpath(D[(ver, mat)]["base"], W))]
        lines.append(f'    "{mat}": {{')
        for k, p in items:
            head = f'        "{k}": '
            for j, c in enumerate(split_path(p.replace("/", "\\"))):
                lines.append((head if j == 0 else " " * vis_len(head)) + f'r"{c}"')
            lines[-1] += ","
        lines.append("    },")
    lines.append("}")
    code = "\n".join(lines)
    obj = ast.literal_eval(code.split("=", 1)[1].strip())
    for mat, d in obj.items():
        for k, p in d.items():
            assert os.path.exists(os.path.join(W, p)), f"FILES の例の場所がない：{mat} {k} {p}"
    return code


def test_check():
    """試しの実行で確かめた値（厚みごとの比較のノートの値と同じだったもの）の文"""
    a = T[(T["ver"] == "温度適切版") & (T["mat"] == "MIX") & (T["h"] == 0.5)].iloc[0]
    b = T[(T["ver"] == "通常版") & (T["mat"] == "GPPS") & (T["h"] == 1.0)].iloc[0]
    s1 = (f"温度適切版の粉砕材混合 PP の 0.5 mm（{M(D[('温度適切版', 'MIX')]['key'])}）：$E_v$ {g(a['tv'])} kJ/mol・$\\Delta$LML {n(a['dl'], 2)}、"
          f"通常版の GPPS の 1.0 mm（{M(D[('通常版', 'GPPS')]['key'])}）：$E_h$ {g(b['th'])}・$E_v$ {g(b['tv'])} kJ/mol・$\\Delta$LML {n(b['dl'], 2)}")
    s = D[("温度適切版", "GPPS")]["S"].loc[D[("温度適切版", "GPPS")]["key"]]
    s0 = f"温度適切版の GPPS（{M(D[('温度適切版', 'GPPS')]['key'])}）：{theta_str(s)}・LML {n(s['LML'], 1)}・3data の RMSE {n(s['RMSE 全体'], 4)}"
    return s1, s0


# ---------------------------------------------------------------- 材料ごとの文
def case_text(ver, mat):
    d = D[(ver, mat)]
    key, S = d["key"], d["S"]
    C = d["ctab"].set_index(["シフトに使う温度", "型"])
    L = []
    tset, tff = C.loc["Tset"], C.loc["Tff"]
    a_txt = "、".join(f"{f} {n(tset.loc[f, 'A の ΔLML'], 2)}" for f in tset.index)
    ba_txt = "、".join(f"{f} {n(tset.loc[f, 'B − A の LML'], 2)}" for f in tset.index)
    L.append(f"Tset のシフト：A の $\\Delta$LML は {a_txt}（基準 1.92）。B − A は {ba_txt}（基準 1.92）。")
    L.append(f"Tff のシフト：A の $\\Delta$LML は最大 {n(tff['A の ΔLML'].max(), 2)}、B は最大 {n(tff['B の ΔLML'].max(), 2)} で、どの組も支持されない。")
    surv = [c for c in d["ctab"]["この組の結論"] if "_" in str(c)]
    if surv:
        L.append("残った組の LML は " + "、".join(f"{M(k)} {n(S.loc[k, 'LML'], 1)}" for k in surv) + f"。最も大きい {M(key)} を候補にした。")
    else:
        L.append("残った組がないので、シフトなしの C（\\mdl{SHIFT_COL = None}）にした。")
    r, c = S.loc[key], S.loc["C"]
    if key == "C":
        L.append(f"確かめ：3data の RMSE（全体）は {n(r['RMSE 全体'], 4)}、95\\% 予測区間に入る割合は {n(r['被覆率 全体'], 2)}（基準 0.8 以上）。")
    else:
        L.append(f"確かめ：3data の RMSE（全体）{n(r['RMSE 全体'], 4)} は C（{n(c['RMSE 全体'], 4)}）の {n(r['RMSE 全体'] / c['RMSE 全体'], 2)} 倍（基準 1.1 倍以内）、"
                 f"95\\% 予測区間に入る割合は {n(r['被覆率 全体'], 2)}（基準 0.8 以上）。")
    if key != "C":
        sh = d["jd"][(d["jd"]["節"] == "シフト") & (d["jd"]["内容"].str.startswith(key + "："))]
        assert len(sh) == 2, sh
        L.append("縦のシフトの向きは「低温ほど粘い」で、シフトで温度ごとの曲線が近づく（判定の一覧の「シフト」の 2 行が OK）。" if (sh["判定"] == "OK").all() else
                 "★ 判定の一覧の「シフト」の行に ★ がある。")
    return " ".join(L)


def note_text(ver, mat):
    d = D[(ver, mat)]
    key, S = d["key"], d["S"]
    if (ver, mat) == ("通常版", "PP"):
        b = S.loc["B_Tff_Arr"]
        z = b["LML"] - b["θ=0 との LML の差"]
        return (f"Tff のモデルの LML（{M('B_Tff_Arr')} {n(b['LML'], 1)} など）は、候補より大きい。しかし、$\\theta = 0$（Tff を使わないモデル）との差は {n(b['θ=0 との LML の差'], 2)} しかない。"
                f"LML の大きさは、Tff のシフトではなく、Tff を使わずに Tset を GP の入力にしたモデル（LML {n(z, 1)}）から来ている。"
                "説明変数に Tff を入れる指定（CLAUDE.md §9）があるので、Tff を使わないモデルは選べない。規則どおり、Tset のシフトにした。"
                f"低速側の RMSE は、どのモデルも {n(S['RMSE 低速側'].min(), 2)}〜{n(S['RMSE 低速側'].max(), 2)} と大きい（外挿。設定では変わらない）。")
    if (ver, mat) == ("通常版", "MIX"):
        wl = [k for k in ["A_Tset_WLF", "B_Tset_WLF"] if S.loc[k, "RMSE 全体"] > 1.1 * S.loc["C", "RMSE 全体"]]
        return (f"3data の RMSE は C（{n(S.loc['C', 'RMSE 全体'], 4)}）が最も小さいが、候補との差は 1.1 倍以内。"
                + (f"WLF 型の {'・'.join(M(k) for k in wl)} は、C の 1.1 倍を少し超えた（★）。" if wl else ""))
    if (ver, mat) == ("通常版", "GPPS"):
        w = S.loc["B_Tset_WLF"]
        return (f"B が A より 1.92 以上大きいので、B（横・縦のシフト）になった。WLF 型の {M('B_Tset_WLF')} は、LML は {n(S.loc[key, 'LML'] - w['LML'], 1)} 小さいが、"
                f"3data の RMSE は {n(w['RMSE 全体'], 4)} で、候補（{n(S.loc[key, 'RMSE 全体'], 4)}）より小さい（5 章）。")
    if (ver, mat) == ("温度適切版", "PP"):
        best = S.sort_values("RMSE 全体").index[0]
        return (f"Tset・Tff のどちらのシフトも、$\\Delta$LML は最大 {n(S['θ=0 との LML の差'].max(), 2)} で、支持されない。"
                + (f"C の 3data の RMSE（{n(S.loc['C', 'RMSE 全体'], 4)}）は、9 つのモデルで最も小さい。" if best == "C" else "")
                + "温度適切版の検証データは、高速側（280・320 mm/s）だけ。")
    if (ver, mat) == ("温度適切版", "MIX"):
        best = S.sort_values("RMSE 全体").index[0]
        return (f"3data の RMSE は、Tff のモデル（{M(best)} {n(S.loc[best, 'RMSE 全体'], 4)}）のほうが、候補（{n(S.loc[key, 'RMSE 全体'], 4)}）より小さい。"
                "しかし、Tff のシフトは尤度で支持されず、縦のシフトの向きも逆（判定の一覧で ★）。検証データは高速側だけなので、誤差の差だけでは選ばなかった（5 章）。")
    if (ver, mat) == ("温度適切版", "GPPS"):
        ba = d["ctab"].set_index(["シフトに使う温度", "型"]).loc[("Tset", "アレニウス型"), "B − A の LML"]
        return (f"B − A は {n(ba, 2)} で基準 1.92 に届かないので、A（縦だけ）になった。95\\% 予測区間に入る割合は {n(S.loc[key, '被覆率 全体'], 2)} で、"
                f"基準 0.8 に届かない（★。9 つのモデルとも {n(S['被覆率 全体'].min(), 2)}〜{n(S['被覆率 全体'].max(), 2)}）。予測区間が、やや狭い。")
    return ""


def case_sections():
    out = []
    for ver, mat in CASES:
        out.append(f"\\subsection{{{ver}・{MJ[mat]}：{M(D[(ver, mat)]['key'])}}}")
        out.append(case_text(ver, mat))
        nt = note_text(ver, mat)
        if nt:
            out.append(f"\n\n補足：{nt}")
    return "\n".join(out)


# ---------------------------------------------------------------- 注意の文に使う値
GN = D[("通常版", "GPPS")]["S"]
MT = D[("温度適切版", "MIX")]["S"]
GT = D[("温度適切版", "GPPS")]["S"]
NOTE_FORM = (f"通常版の GPPS は {M('B_Tset_WLF')}（{n(GN.loc['B_Tset_WLF', 'RMSE 全体'], 3)} 対 {n(GN.loc[D[('通常版', 'GPPS')]['key'], 'RMSE 全体'], 3)}）、"
             f"温度適切版の粉砕材混合 PP は Tff のモデル（{M(MT.sort_values('RMSE 全体').index[0])} {n(MT['RMSE 全体'].min(), 3)} 対 "
             f"{n(MT.loc[D[('温度適切版', 'MIX')]['key'], 'RMSE 全体'], 3)}）")
NOTE_COV = f"{n(GT.loc[D[('温度適切版', 'GPPS')]['key'], '被覆率 全体'], 2)}（どのモデルも {n(GT['被覆率 全体'].min(), 2)}〜{n(GT['被覆率 全体'].max(), 2)}）"
SUBT = T[T["tv"].notna()]
NEG = SUBT[SUBT["tv"] < 0]
NEG_LIST = "、".join(f"{r['ver']}の{MJ[r['mat']]}の {r['h']:g} mm（{M(D[(r['ver'], r['mat'])]['key'])}、縦の係数 {g(r['tv'])}、$\\Delta$LML {n(r['dl'], 2)}）"
                    for _, r in NEG.iterrows())
EX = T[(T["ver"] == "通常版") & (T["mat"] == "PP")]
NEG_RULE = int((T["rule_v"] < 0).sum())
CHG = sum(int((T[(T["ver"] == v) & (T["mat"] == m)]["rule"] != D[(v, m)]["key"]).sum()) for v, m in CASES)
FF_OTHER = [c for c in CASES if c != ("通常版", "PP")]

# ---------------------------------------------------------------- tex
PRE = r"""\documentclass[paper=a4paper,fontsize=10.5pt,line_length=44zw,number_of_lines=40]{jlreq}
\usepackage[haranoaji]{luatexja-preset}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{float}
\usepackage{fancyvrb}
\usepackage[hidelinks,bookmarksnumbered]{hyperref}
\newcommand{\file}[1]{\texttt{\detokenize{#1}}}
\newcommand{\mdl}[1]{\texttt{\detokenize{#1}}}
\newcommand{\Tff}{T_{\mathrm{ff}}}
\newcommand{\Tset}{T_{\mathrm{set}}}
\let\origtextbf\textbf
\renewcommand{\textbf}[1]{\origtextbf{\gtfamily #1}}   % 日本語の強調はゴシック体（明朝の太字の書体がないため）
\renewcommand{\arraystretch}{1.2}
\setlength{\tabcolsep}{4pt}
\title{材料別の温度シフト GPR\\本番の学習の設定と、その根拠}
\author{シフトファクター検討用のノート（11 月 4 日実行）と\\厚みごとの比較のノート（11 月 5 日実行）の結果から}
\date{@@DATE@@}
\begin{document}
\maketitle
"""

BODY = r"""
\section*{この報告書について}
\begin{itemize}[itemsep=0.3ex,topsep=0.4ex]
\item 学習用のノート（本番。\file{@@NB@@}）を、3 材料 × 2 種類の学習データ（通常版・温度適切版）× 学習の範囲 2 つ（全厚み・厚みごと）で実行する前に、設定エリアの \mdl{SHIFT_COL}・\mdl{SHIFT_STRUCT}・\mdl{SHIFT_FORM} を決めました。
\item 根拠は、ユーザーが実行したシフトファクター検討用のノートと、厚みごとの比較のノートの結果の CSV です（場所は付録 A）。報告書の数値は、生成スクリプト \file{04_補助スクリプト/build_report_mat_prod_settings.py} が CSV から読んだものです。
\item 用語：LML は対数周辺尤度（学習データの説明のよさ。大きいほどよい）。$\Delta$LML は、シフトの係数を 0 にしたモデル（$\theta = 0$）との LML の差です。モデルの記号は「構造\_シフトに使う温度\_型」です。構造は A（縦だけのシフト）・B（横・縦のシフト）、温度は Tset（加熱シリンダ温度）・Tff（流動中樹脂温度）、型は Arr（アレニウス型）・WLF（WLF 型）。C はシフトなし（4 変数を並べた通常の GPR）です。
\item この版（11 月 5 日の 2 つ目の版）：ユーザーの指示で学習用のノートに厚みの設定 \mdl{THICKNESS} を足したので、\ref{sec:thickrun} 節を書き直し、厚みごとの実行の一覧（表~\ref{tab:truns}）と、ダイアログを使わないときの \mdl{FILES} の例（付録 B）を足しました。また、最初の版で 5 章に書いた「★ が出た厚みだけ C で学習し直す」という選択肢を取り下げ、その問題点を \ref{sec:noc} 節に書きました。
\end{itemize}

\section{結論}\label{sec:concl}
\begin{table}[H]\centering\footnotesize
\caption{本番の学習の設定（学習用のノートの設定エリア。全厚み・厚みごとで同じ）}\label{tab:concl}
\begin{tabular}{llllll}
\toprule
学習データ & 材料 & モデル & \mdl{SHIFT_COL} & \mdl{SHIFT_STRUCT} & \mdl{SHIFT_FORM} \\
\midrule
@@TAB_CONCL@@
\bottomrule
\end{tabular}
\end{table}
\begin{itemize}[itemsep=0.3ex,topsep=0.4ex]
\item 6 つのうち 5 つで、\textbf{加熱シリンダ温度 Tset のアレニウス型のシフト}を使います。縦だけ（A）が 4 つ、横・縦（B）が通常版の GPPS の 1 つです。温度適切版の PP だけは、どのシフトも支持されないので、\textbf{シフトなし（C）}です。
\item 流動中樹脂温度 Tff のシフトは、どの材料・学習データでも、尤度で支持されません（\ref{sec:total} 章）。
\item \textbf{厚みごとの学習も、全厚みと同じモデル}にします。厚みごとの検討の結果は、どの厚みでも Tset のままでよいことを示しています。一方で、厚みごとに係数を決め直すと、係数が厚みで大きく動き、符号が逆になる厚みもあります。厚みごとに別のモデルを選ぶ根拠にはなりません（\ref{sec:thick} 章）。
\item ほかの設定は、既定のままにします（検討のときと同じ。表~\ref{tab:other}）。
\end{itemize}

\begin{table}[H]\centering\footnotesize
\caption{そのほかの設定（既定のまま）}\label{tab:other}
\begin{tabular}{lp{25\zw}}
\toprule
設定 & 値と理由 \\
\midrule
\mdl{THICKNESS} & 全厚み：\mdl{None}（既定）、厚みごと：0.5・1.0・1.5・2.0（1 回に 1 つ。\ref{sec:thickrun} 節） \\
\mdl{INPUT_MODE} & \mdl{"shot"}（ショットごとの入力。10 月 23 日のユーザー決定） \\
\mdl{SEED}・\mdl{NOISE_LB}・\mdl{USE_LINEAR} & 42・1e-6・\mdl{True}（検討と同じ） \\
\mdl{N_RESTARTS}・\mdl{PROFILE_RESTARTS} & 20・1（検討と同じ） \\
\mdl{USE_EXCLUSIONS} & \mdl{True}（問題データ一覧の「除外」を除く） \\
\mdl{SHIFT_CFG} & PP・粉砕材混合 PP：Tref 210 ℃・Tg $-10$ ℃、GPPS：Tref 220 ℃・Tg 100 ℃ \\
\mdl{FORM_CFG}・\mdl{FINE_*}・\mdl{MAX_*} & 既定（プロファイル尤度の格子。検討と同じ） \\
\bottomrule
\end{tabular}
\end{table}
候補のモデルの学習済みモデル（\file{学習済みモデル_<モデル>.joblib}）の記録で、6 つとも、入力の取り方が \mdl{"shot"}、シフトファクターの設定（基準温度・Tg・格子）が既定と同じこと、学習データのファイルが \file{データ確認/データの概要.csv} と同じことを確かめました。やり直しの回数などは記録にないので、既定で実行したと考えています（要確認）。

\subsection{全厚みの実行の分け方}\label{sec:runs}
\mdl{SHIFT_COL}・\mdl{SHIFT_STRUCT}・\mdl{SHIFT_FORM} は、\mdl{MATERIALS} のすべての材料に同じものを使います。そのため、同じモデルの材料をまとめて、全厚みは次の @@NRUNS@@ 回に分けて実行します。
\begin{table}[H]\centering\footnotesize
\caption{全厚みの実行の分け方}\label{tab:runs}
\begin{tabular}{lllllll}
\toprule
実行 & 学習データ & \mdl{MATERIALS} & モデル & \mdl{SHIFT_COL} & \mdl{SHIFT_STRUCT} & \mdl{SHIFT_FORM} \\
\midrule
@@TAB_RUNS@@
\bottomrule
\end{tabular}
\end{table}
ダイアログで選ぶファイル（検討のときと同じ）：
\begin{itemize}[itemsep=0.3ex,topsep=0.4ex]
@@RUN_FILES@@
\end{itemize}
学習結果フォルダに検討と同じフォルダを選ぶと、ノートの設計どおり \file{全ショット版/全厚み総合/<モデルのフォルダ>/} に入ります。検討の結果（\file{全厚み総合/シフトファクターの検討/}・\file{厚みごと/}）とは別のフォルダなので、消えません。\file{データ確認/} の表は、同じデータなので同じ内容で書き直されます。

\subsection{厚みごとの実行}\label{sec:thickrun}
ユーザーの指示で、学習用のノートに、学習する厚みを選ぶ設定 \mdl{THICKNESS} を足しました（11 月 5 日）。\mdl{None}（既定）なら全厚み、厚みの数（例：\mdl{0.5}）を書くと、その厚みの 5data だけで学習し、同じ厚みの 3data で検証します。厚みごとの本番は、表~\ref{tab:runs} の 4 つの実行のそれぞれで、\mdl{THICKNESS} だけを変えて、1 回ずつ実行します（列・形・型は表~\ref{tab:concl} のまま）。
\begin{table}[H]\centering\footnotesize
\caption{厚みごとの実行の一覧（\mdl{MATERIALS} と列・形・型は表~\ref{tab:runs} と同じ。全部で @@NTRUNS@@ 回）}\label{tab:truns}
\begin{tabular}{lllp{9\zw}p{15\zw}}
\toprule
実行 & 学習データ & モデル & \mdl{THICKNESS} に書く値 & 学習する材料 \\
\midrule
@@TAB_TRUNS@@
\bottomrule
\end{tabular}
\end{table}
\begin{itemize}[itemsep=0.3ex,topsep=0.4ex]
\item その厚みのデータがない材料は、その回は学習しません（GPPS の 0.5 mm。④の 0.5 mm は、粉砕材混合 PP だけを学習します）。
\item 結果は、全厚みと同じ学習結果フォルダの \file{全ショット版/厚みごと/厚み<厚み>mm/<モデルのフォルダ>/} です（例：\file{厚み0.5mm/}。1 mm は \file{厚み1mm/}）。全厚みの結果（\file{全厚み総合/}）とも、厚みごとの比較のノートの結果（\file{厚みごと/} の直下の 4 つのフォルダ）とも別のフォルダです。
\item データの確認（外れ値・P3\_3 の性質）は、厚みを選んでも、全厚みのデータで行います。検証の低速側・高速側は、その厚みの学習データの射出速度の範囲で分け直します（通常版の 0.5・1.0 mm の検証は、高速側だけ）。厚みが 1 つなので、全厚み共通の図は作りません。
\item 試しの実行（結果は \file{_tmp/} の中。本物の学習結果フォルダには書いていない）で、厚みごとの係数・$\Delta$LML・3data の誤差が、厚みごとの比較のノート（6・9 節）と同じになることを確かめました（@@TESTCHK@@）。全厚み（\mdl{THICKNESS = None}）の結果も、検討用のノートと同じでした（@@TESTCHK0@@）。
\item 厚みごとの学習のデータは少ない（学習の条件 @@NCMIN@@〜@@NCMAX@@ 個。表~\ref{tab:datah}）ので、ノートの判定に ★ が出やすくなります。★ が出ても、モデルは替えずに、結果として扱います（\ref{sec:noc} 節）。
\item ダイアログを使わないときは、付録 B の \mdl{FILES} を設定エリアに書きます（学習データ・検証データ・学習結果フォルダは、全厚み・厚みごととも同じ）。
\end{itemize}

\section{決め方}
\subsection{全厚み：候補の選び方（ユーザー決定、11 月 3 日）}
シフトファクター検討用のノートのまとめと同じ規則（\file{mat_common.shift_candidates}）です。この報告書のスクリプトで当てはめ直し、6 つとも \file{候補の選び方.csv} の結論と一致することを確かめました。
\begin{enumerate}[itemsep=0.3ex,topsep=0.4ex]
\item 列（Tset・Tff）と型（Arr・WLF）の組ごとに、A の $\Delta$LML が 1.92 以上なら、その組の温度シフトは支持される（尤度比検定の 5\% 基準。係数 1 つ）。支持されたら、B が A より 1.92 以上大きいときだけ B、そうでなければ A。A が支持されず、B だけが 3.0 以上（係数 2 つの基準）のときは、B − A が 1.92 以上なら B、そうでなければ「決まらない」。
\item 残った組どうしは、LML の大きいものを選ぶ（$\theta = 0$ のモデルが列ごとに違うので、$\Delta$LML では比べない）。
\item どの組も残らなければ、シフトなしの C。
\end{enumerate}
あわせて、検討用のノートの判定（3data の RMSE が C の 1.1 倍以内、95\% 予測区間に入る割合が 0.8 以上、縦のシフトの向きが「低温ほど粘い」）も確かめました。3data の誤差は、選ぶ基準には使っていません（\ref{sec:note} 章）。

\subsection{厚みごと：全厚みと同じモデル（Claude の判断）}
厚みごとの比較のノートの結果を、次の 4 つの面から見て決めました（詳しくは \ref{sec:thick} 章）。
\begin{enumerate}[itemsep=0.3ex,topsep=0.4ex]
\item シフトに使う温度：厚みごとに Tff と Tset を比べた多数決（ノートの 8 節）と、全厚みの係数が厚みごとに効いているか（4 節）。
\item 形と型：厚みごとのデータだけで決め直した係数（6 節）が、厚みによらず安定しているか。
\item 厚みごとに同じ規則を当てた候補が、厚みによらずそろうか（参考）。
\item 厚みごとのモデルと総合データのモデルの予測誤差（9 節。設定ではなく、本番の結果を比べるときの参考）。
\end{enumerate}

\section{全厚みの根拠}\label{sec:total}
表~\ref{tab:cand} は \file{全厚み総合/シフトファクターの検討/候補の選び方.csv} の値です（太字は基準を満たしたもの）。表~\ref{tab:detail}・\ref{tab:rank} は、同じフォルダの \file{まとめ.csv} の値です。
\begin{table}[H]\centering\footnotesize
\caption{候補の選び方（\file{候補の選び方.csv}。A の基準 1.92、B の基準 3.0、B − A の基準 1.92）}\label{tab:cand}
\begin{tabular}{lllrrrl}
\toprule
材料 & 温度 & 型 & A の $\Delta$LML & B の $\Delta$LML & B − A & この組の結論 \\
\midrule
@@TAB_CAND@@
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]\centering\footnotesize
\caption{候補のモデルの係数と 3data の誤差（\file{まとめ.csv}。RMSE は ln $\eta$、C 比は C の RMSE との比、被覆率は 95\% 予測区間に入る割合。係数の単位は、アレニウス型が kJ/mol、WLF 型は無次元）}\label{tab:detail}
\begin{tabular}{llp{13\zw}rrrrrr}
\toprule
材料 & モデル & 係数［95\% 区間］ & LML & RMSE & 低速側 & 高速側 & C 比 & 被覆率 \\
\midrule
@@TAB_DETAIL@@
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]\centering\footnotesize
\caption{3data の RMSE（全体）で 9 つのモデルを並べたときの、候補の順位と、RMSE が最小のモデル（右端は、そのモデルが尤度で支持されたか）}\label{tab:rank}
\begin{tabular}{lllllll}
\toprule
材料 & 候補 & RMSE & 順位 & 最小のモデル & RMSE & 尤度で \\
\midrule
@@TAB_RANK@@
\bottomrule
\end{tabular}
\end{table}

@@CASES@@

\section{厚みごとの根拠}\label{sec:thick}
\subsection{データの量}
\begin{table}[H]\centering\small
\caption{厚みごとの条件の数（学習 5data の条件 / 検証 3data の条件。\file{プロファイル尤度_厚みごと.csv} の「学習の条件数」と、\file{厚みごとと総合の比較.csv} の「条件数」。全厚みは \file{データの概要.csv}）}\label{tab:datah}
\begin{tabular}{lccccc}
\toprule
材料 & 0.5 mm & 1.0 mm & 1.5 mm & 2.0 mm & 全厚み \\
\midrule
@@TAB_DATA_H@@
\bottomrule
\end{tabular}
\end{table}
厚みごとの学習の条件は @@NCMIN@@〜@@NCMAX@@ 個で、全厚み（@@NTMIN@@〜@@NTMAX@@ 個）の 3〜4 分の 1 です。温度適切版は、1 つの厚みで 12 個（射出速度 4 水準 × 設定温度 3 水準）しかありません。

\subsection{シフトに使う温度：Tset のまま}
\begin{table}[H]\centering\small
\caption{厚みごとの比較の判定の数（8 節：\file{シフトに使う温度/厚みごとの比較_まとめ.csv} の「総合の判定」。4 節：\file{温度シフトの確認/温度シフトの確認_厚みごと.csv} の「判定」。かっこは数えた数）}\label{tab:temp}
\begin{tabular}{llll}
\toprule
材料 & 8 節：適した温度（厚み × 組） & 4 節：Tset のモデル & 4 節：Tff のモデル \\
\midrule
@@TAB_TEMP@@
\bottomrule
\end{tabular}
\end{table}
\begin{itemize}[itemsep=0.3ex,topsep=0.4ex]
\item 8 節（厚みごとに、(a) 重ね合わせのずれ・(b) 尤度・(c) 検証誤差で Tff と Tset を比べた多数決）では、全部で @@T8ALL@@ の「厚み × 組」のうち、差なしが @@T8NONE@@、Tff が @@T8FF@@、Tset が @@T8SET@@ でした。厚みごとに温度を替える根拠はありません。
\item 4 節（全厚みで学習した係数で、厚みごとの温度による粘度の比 R を表せるか）では、Tset のシフトのモデルは、6 つとも、すべての厚みで ○ でした。Tff のシフトのモデルは、通常版の PP（すべて ○）を除くと、○ は @@FFN@@ 個中 @@FFOK@@ 個で、残りは △（換算しても、温度の違う曲線のずれが減らない）でした。
\item 全厚みの結論（\ref{sec:total} 章：Tset のシフトが支持され、Tff は支持されない）と食い違わないので、厚みごとも Tset のままにします。
\end{itemize}

\subsection{形と型：厚みごとの係数は不安定}\label{sec:tprof}
表~\ref{tab:tprof} は、全厚みで選んだモデルを、厚みごとの 5data だけで学習したときの係数（6 節。\file{シフトに使う温度/プロファイル尤度_厚みごと.csv}）です。右の 2 列は参考で、厚みごとに \ref{sec:total} 章と同じ規則を当てた候補と、その厚みで細かい格子の取り直しが収束しなかったモデルの数です。
\begin{table}[H]\centering\footnotesize
\caption{厚みごとのプロファイル尤度（$\Delta$LML の太字は基準を満たしたもの）。†：縦の係数が負（低温ほど粘度が低い向き）。‡：規則の注意が出た（係数の数が違う組が残った、など）}\label{tab:tprof}
\begin{tabular}{lllllc}
\toprule
厚み & 選んだモデルの係数 & $\Delta$LML & 取り直し & 厚みごとに規則を当てた候補 & 未収束 \\
\midrule
@@TAB_TPROF@@
\bottomrule
\end{tabular}
\end{table}
\begin{itemize}[itemsep=0.3ex,topsep=0.4ex]
\item 係数は、厚みで大きく動きます（例：通常版の PP の \mdl{A_Tset_Arr} の $E_v$ は、0.5〜2.0 mm で @@EXPP@@ kJ/mol。全厚みでは @@EXPPALL@@）。選んだモデルの @@NALL@@ 個の「材料・厚み」のうち @@NNEG@@ 個で、縦の係数が負（全厚みの係数や、実測の「低温ほど粘い」と逆の向き）になりました：@@NEGLIST@@。$\Delta$LML が基準を満たしたのは @@NSUP@@ 個です。
\item 厚みごとに規則を当てると、@@NROWS@@ 個の「材料・厚み」のうち @@NCHG@@ 個で、全厚みと違うモデルになります。その中には、縦の係数が負のモデル（@@NRNEG@@ 個）や Tff のモデルも入り、厚みによって列・形・型がばらばらです。厚みごとのモデル @@NMOD@@ 個のうち @@NUNC@@ 個は、細かい格子の取り直しが収束していません（尤度の面が平らで、係数が決まりにくい）。
\item 条件の数が少ないために尤度の差が小さく、たまたまの違いで候補が入れ替わっていると考えられます。そこで、厚みごとにモデルを選び直さず、全厚みで選んだモデルを使います。本番の学習でも、係数は厚みごとのデータで決め直されるので、厚みによる違いは係数に表れます。
\item 通常版の GPPS の \mdl{B_Tset_Arr} は、横の係数が 0 なら A と同じ形になるので、A のほうが合う厚みがあっても、B のままで困りません。
\end{itemize}

\subsection{厚みごとのモデルと総合データのモデル（参考）}
\begin{table}[H]\centering\footnotesize
\caption{3data の予測誤差 [\%]（厚みごとのモデル / 総合データのモデル。太字は小さいほう。\file{厚みごとのモデルと総合の比較/厚みごとと総合の比較.csv} の「誤差\_\%」、範囲は全体。右端は \file{厚みごとと総合の比較_まとめ.csv} の「勧める学習」）}\label{tab:tvt}
\begin{tabular}{llccccc l}
\toprule
材料 & モデル & 0.5 mm & 1.0 mm & 1.5 mm & 2.0 mm & 全厚み & 勧める学習 \\
\midrule
@@TAB_TVT@@
\bottomrule
\end{tabular}
\end{table}
これは、厚みごとの比較のノートが、6 節の係数で厚みごとに学習し直したモデルの結果で、本番の厚みごとの学習の目安になります。「勧める学習」は、誤差の差が 1 ポイント以上ある厚みの数の多数決です。@@TVTMIS@@

\section{注意と、ユーザーが決めること}\label{sec:note}
\begin{enumerate}[itemsep=0.5ex,topsep=0.4ex]
\item \textbf{型やモデルを 3data の誤差で選ぶ場合}：規則は、学習データの LML で選びます。3data の RMSE だけで選ぶと、@@NOTEFORM@@ になります（表~\ref{tab:rank}）。ただし、Tff のシフトは尤度で支持されず、温度適切版の検証は高速側だけです。誤差を優先するなら、通常版の GPPS だけ \mdl{SHIFT_FORM = "wlf"} にする選択肢があります。
\item \textbf{温度適切版の検証}：検証データは高速側（280・320 mm/s）だけで、低速側の外挿は確かめられません。
\item \textbf{温度適切版の GPPS の予測区間}：95\% 予測区間に入る割合が @@NOTECOV@@ で、基準 0.8 に届きません。
\item \textbf{厚みごとで、縦の係数が負になりそうな厚み}：@@NEGLIST@@。本番の厚みごとの学習も、同じデータを同じ方法で計算するので、同じように負になる見込みです（ノートの「シフト」の判定で ★ が出ます）。\textbf{表~\ref{tab:concl} のモデルのまま学習し、★ は結果として扱います}（\ref{sec:noc} 節）。
\item \textbf{規則に入っていないこと}：規則は、縦のシフトの向きを見ません。全厚みの候補は 6 つとも向きが正しいことを確かめましたが、厚みごとに規則を当てるときは、向きも見る必要があります（\ref{sec:tprof} 節）。
\end{enumerate}

\subsection{★ が出た厚みだけ C（シフトなし）に替えることの問題点}\label{sec:noc}
11 月 5 日の最初の版では、5 章に「★ が出たら、その厚みだけシフトなし（C）で学習し直す」という選択肢を書きました。ユーザーの指摘を受けて見直し、\textbf{この選択肢は取り下げます}。理由は次の 5 つです。
\begin{enumerate}[itemsep=0.5ex,topsep=0.4ex]
\item \textbf{結果を見てからモデルを替えることになる}：替える理由が「★ を消すため」なので、判定に合う結果が出るまでモデルを選び直すことになります。学習を無理に成功させるのと同じで、選んだ後の結果は、実際よりよく見えます。
\item \textbf{問題は消えずに、見えなくなるだけ}：負の係数は、「$\Tff$ と P3\_3 を入れたうえでの $\Tset$ の効果が、その厚みでは逆向きに出た」というデータの性質です（$\Tset$ の効果の多くは $\Tff$ と P3\_3 が担うので、残りは小さく、条件の少ない厚みでは向きが決まりにくい）。C は $\Tset$ を GP の普通の入力にするので、同じ逆向きの効果を、そのまま GP の中に取り込みます。C には係数がないので、★ で気づけなくなるだけです。
\item \textbf{比べられなくなる}：その厚みだけモデルが違うと、厚みごとの係数の表に穴があきます。厚みごとと全厚みの比較、ほかの材料や学習データ（通常版・温度適切版）との比較で、「データの分け方の違い」と「モデルの違い」が混ざり、差の理由が分からなくなります。
\item \textbf{判定の基準の意味が崩れる}：ふつうの意味の過学習（学習データの細かいゆれまで覚えて、予測が悪くなること）とは、少し違う問題です。C は、学習データへの当てはまりを上げるために選ぶわけではなく、自由度はむしろ大きい（$\Tset$ を自由な入力にする）ので、過学習が減るわけでもありません。しかし、選び方が結果を見て後から決まるので、$\Delta$LML の基準（5\%）の意味が崩れます。@@NROWS@@ の「材料・厚み」で調べれば、基準を偶然に超えるものが 1 つ前後出るのは、ふつうのことです。また、3data の誤差を見て替えるなら、検証データをモデル選びに使うことになり、3data の誤差が楽観的になります（過学習と同じ種類の問題）。
\item \textbf{決まりを後から足すことになる}：公平にするには、「どの材料・学習データ・厚みでも、★ が出たら C」という決まりを、結果を見る前に決めて、すべてに当てはめる必要があります。今回は結果を見た後なので、その条件を満たしません。
\end{enumerate}
\textbf{勧める扱い}：表~\ref{tab:concl} のモデルのまま、すべての厚みを学習します。★ は、「この厚みでは、$\Tff$ と P3\_3 を入れたうえでの $\Tset$ の効果が逆向きに出た（条件が少なく、向きが決まりにくい）」という結果として、そのまま報告します。厚みごとの係数の向きをそろえたいときは、結果を見る前の決まりとして、すべての厚みに同じ扱い（例：全厚みで決めた係数に固定して、GP だけを厚みごとに学習する）を当てはめる方法があります（ノートの直しが要るので、ユーザーの判断です）。

\appendix
\section{見た結果ファイル}
次のフォルダの CSV を読みました（学習結果フォルダの \file{全ショット版/}。ほかに、\file{データ確認/データの概要.csv} と、候補のモデルの学習済みモデル）。
\begin{itemize}[itemsep=0.2ex,topsep=0.4ex]
@@TAB_FILES@@
\end{itemize}
その中の、次のファイルです。
\begin{itemize}[itemsep=0.2ex,topsep=0.4ex]
\item \file{全厚み総合/シフトファクターの検討/候補の選び方.csv}：組ごとの A・B の $\Delta$LML と結論（表~\ref{tab:cand}）
\item \file{全厚み総合/シフトファクターの検討/まとめ.csv}：モデルごとの係数・LML・3data の誤差（表~\ref{tab:detail}・\ref{tab:rank}、\ref{sec:total} 章の文）
\item \file{全厚み総合/シフトファクターの検討/判定の一覧.csv}：シフトの向き・重ね合わせ・C との比などの判定（\ref{sec:total} 章の文）
\item \file{厚みごと/シフトに使う温度/プロファイル尤度_厚みごと.csv}：厚みごとの係数・LML・取り直し（表~\ref{tab:datah}・\ref{tab:tprof}）
\item \file{厚みごと/シフトに使う温度/厚みごとの比較_まとめ.csv}：Tff か Tset かの多数決（表~\ref{tab:temp}）
\item \file{厚みごと/温度シフトの確認/温度シフトの確認_厚みごと.csv}：全厚みの係数の、厚みごとの判定（表~\ref{tab:temp}）
\item \file{厚みごと/厚みごとのモデルと総合の比較/厚みごとと総合の比較.csv}・\file{厚みごとと総合の比較_まとめ.csv}：厚みごとのモデルと総合データのモデルの誤差（表~\ref{tab:datah}・\ref{tab:tvt}）
\end{itemize}

\section{ダイアログを使わないときの \mdl{FILES}}\label{sec:files}
設定エリアの \mdl{FILES = {}} を、学習データの種類に合わせて、次のどちらかに書き換えると、ダイアログは出ません。3 材料分を書いてあり、\mdl{MATERIALS} に書かない材料の行は使われないので、全厚み・厚みごとの全部の実行で同じものを使えます。学習データ・検証データは検討のときと同じファイル、学習結果フォルダは検討と同じフォルダです（スクリプトで、Python として読めることと、場所がすべてあることを確かめました）。PDF から写すときは、字下げと \mdl{r"…"} が崩れていないかを確かめてください。
\subsection*{通常版}
\begin{Verbatim}[fontsize=\footnotesize,frame=single]
@@FILES_V1@@
\end{Verbatim}
\subsection*{温度適切版}
\begin{Verbatim}[fontsize=\footnotesize,frame=single]
@@FILES_V2@@
\end{Verbatim}
\end{document}
"""

TVTM = tvt_mismatch()
V = {"DATE": rc.today_ja(), "NB": NB, "NRUNS": len(RUNS),
     "TAB_CONCL": tab_conclusion(), "TAB_RUNS": tab_runs(), "RUN_FILES": run_files(), "TAB_CAND": tab_candidates(),
     "TAB_DETAIL": tab_cand_detail(), "TAB_RANK": tab_rmse_rank(), "CASES": case_sections(), "TAB_DATA_H": tab_data_h(),
     "TAB_TEMP": tab_temp_choice(), "TAB_TPROF": tab_thick_profile(), "TAB_TVT": tab_thick_vs_total(), "TAB_FILES": tab_files(),
     "NCMIN": int(T["n_cond"].min()), "NCMAX": int(T["n_cond"].max()),
     "NTMIN": min(int(D[c]["data"].set_index("組").loc["5data", "条件数"]) for c in CASES),
     "NTMAX": max(int(D[c]["data"].set_index("組").loc["5data", "条件数"]) for c in CASES),
     "T8ALL": sum(TOT.values()), "T8NONE": TOT["差なし"], "T8FF": TOT["Tff"], "T8SET": TOT["Tset"],
     "FFN": sum(FF_N[c] for c in FF_OTHER), "FFOK": sum(FF_OK[c] for c in FF_OTHER),
     "EXPP": "・".join(g(x) for x in EX["tv"]),
     "EXPPALL": g(re.search(r"E_v ([-\d.]+)", str(D[("通常版", "PP")]["S"].loc[D[("通常版", "PP")]["key"], "θ"])).group(1)), "NALL": len(SUBT), "NNEG": int((SUBT["tv"] < 0).sum()),
     "NSUP": int((SUBT["dl"] >= SUBT["lim"]).sum()), "NROWS": len(T), "NCHG": CHG, "NRNEG": NEG_RULE,
     "NMOD": int(T["n_models"].sum()), "NUNC": int(T["n_unconv"].sum()),
     "TVTMIS": (f"{TVTM}は、「勧める学習」と全厚みの誤差の大小が食い違います。本番の結果を比べるときは、厚みごとの誤差と全厚みの誤差の両方を見てください。" if TVTM else ""),
     "NOTEFORM": NOTE_FORM, "NOTECOV": NOTE_COV, "NEGLIST": NEG_LIST,
     "NTRUNS": sum(len(r[4]) for r in TRUNS), "TAB_TRUNS": tab_truns(), "TESTCHK": test_check()[0], "TESTCHK0": test_check()[1],
     "FILES_V1": files_snippet("通常版"), "FILES_V2": files_snippet("温度適切版")}
body = BODY
for k, v in V.items():
    body = body.replace(f"@@{k}@@", str(v))
body = re.sub(r"\\(?:file|mdl)\{([^{}]*)\}", lambda m: tt(m.group(1)), body)     # 本文の \file{…}・\mdl{…} を、逃がした等幅の文字に
tex = PRE.replace("@@DATE@@", V["DATE"]) + body
left = sorted(set(re.findall(r"@@([A-Za-z0-9_]+)@@", tex)))
assert not left, f"置き換えていない項目: {left}"

# 結論を画面にも出す（確かめ用）
for ver, mat in CASES:
    print(f"{ver} {MJ[mat]}: {D[(ver, mat)]['key']}  {settings_of(D[(ver, mat)]['key'])}")
print("実行:", [(v, k, m) for v, k, m in RUNS])
print({k: V[k] for k in ["NCMIN", "NCMAX", "T8ALL", "T8NONE", "T8FF", "T8SET", "FFN", "FFOK", "EXPP", "NALL", "NNEG", "NSUP", "NROWS", "NCHG", "NRNEG", "NMOD", "NUNC"]})

os.makedirs(out_dir, exist_ok=True)
with open(TEX, "x", encoding="utf-8", newline="\n") as f:     # "x"：同じ名前があれば止まる
    f.write(tex)
aux = os.path.join(W, "_tmp", "report_mat_prod", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
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
