# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR（05 フォルダ）：温度ホールドアウト検証の本番の結果から、「3 材料で温度シフトファクターを使ったほうがよいか」を
簡潔にまとめた HTML の資料を作る（2026-11-16 のユーザー依頼）。

使い方（リポジトリ直下から）: uv run python 04_補助スクリプト/build_report_mat_loto.py [--out <フォルダ>]
- 読むもの：3 材料の最新の学習結果フォルダ（PP 3 回目・粉砕材混合 PP 5 回目・GPPS 3 回目。線形項あり・通常版）の
  全ショット版/全厚み総合/温度ホールドアウト検証/ の CSV（まとめ.csv・予測の指標_回ごと.csv・シフトファクターの推定_回ごと.csv・分け方.csv）と図。
  2026-11-16 に、材料ごとの複製（PP_・MIXPP_・GPPS_温度ホールドアウト検証_…ipynb）を本番の設定で実行した結果（コミット be31c30）。
- 書くもの：05_材料ごとの機械学習データと予測検証データ/09_温度ホールドアウト検証結果/ に
  - 温度シフトファクターを使うべきか_温度ホールドアウト検証のまとめ.html（1 つのファイル。外部の読み込みなし。アニメーションは素の JavaScript・SVG・CSS）
  - 引用した図と表/<PP|粉砕材混合PP|GPPS>/ に、引用した図 2 枚と表 4 個の原本のコピー（元の名前のまま。shutil.copy2）
  同じ名前のファイルがあれば、何も書かずに止める（上書きしない）。
- 資料の文は、結果の CSV の値から作る。結論の文が結果と食い違わないよう、文の前提（ノートの目安・係数の符号など）を確かめ、違えば止める。
"""
import filecmp
import html
import json
import os
import shutil
import sys
import urllib.parse

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)                                   # リポジトリ直下
D = "05_材料ごとの機械学習データと予測検証データ"
ROOTS = {"PP": D + "/01_バージンPP再々実験(データと結果)/01_学習結果/3回目_01_バージンPP再々実験_学習結果",
         "MIX": D + "/02_粉砕材混合PP実験(データと結果)/02_学習結果/5回目_02_粉砕材混合PP実験_学習結果",
         "GPPS": D + "/03_GPPS実験(データと結果)/03_学習結果/3回目_03_GPPS実験_学習結果"}
HOLD = "全ショット版/全厚み総合/温度ホールドアウト検証"
OUT_DIR = os.path.join(W, sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else os.path.join(W, D, "09_温度ホールドアウト検証結果")   # --out：書く場所（リポジトリ直下から。確かめ用）
COPY_DIR = "引用した図と表"
MAT_DIR = {"PP": "PP", "MIX": "粉砕材混合PP", "GPPS": "GPPS"}
MJ = {"PP": "PP", "MIX": "粉砕材混合 PP", "GPPS": "GPPS"}
FIGS = ["予測のRMSE_回ごと.png", "シフトファクターの係数_回ごと.png"]
TABS = ["まとめ.csv", "予測の指標_回ごと.csv", "シフトファクターの推定_回ごと.csv", "分け方.csv"]
HTML_NAME = "温度シフトファクターを使うべきか_温度ホールドアウト検証のまとめ.html"
EXPECT = {"PP": ("A_Tset_Arr", "どちらかといえば要らない"), "MIX": ("A_Tset_Arr", "要る"), "GPPS": ("B_Tset_Arr", "要る")}   # 資料の文の前提
COMMIT = "be31c30"
RUN_DATE = "2026-11-16"
GOOD, BAD = 1 / 1.1, 1.1                                    # ノートの設定 RMSE_RATIO = 1.1 と同じ
POS = ["低い温度", "中の温度", "高い温度"]


def src(mat, name):
    return os.path.join(W, ROOTS[mat], HOLD, name)


def rd(mat, name):
    return pd.read_csv(src(mat, name), encoding="utf-8-sig")


def esc(s):
    return html.escape(str(s), quote=True)


def g(x):
    """係数などを、余分な 0 を付けずに（負の数は − で）"""
    x = float(x)
    s = f"{abs(x):g}"
    return f"−{s}" if x < 0 else s


def st_of(r):
    return "good" if r < GOOD else ("bad" if r > BAD else "same")


ST = {"good": ("✓", "良い"), "same": ("＝", "同等"), "bad": ("✕", "悪い")}


def badge(r):
    s = st_of(r)
    return f'<span class="st st-{s}">{ST[s][0]} {ST[s][1]}</span>'


def is_true(x):
    return str(x).strip() == "True"


# ---------------------------------------------------------------- 書く場所の確かめ（同じ名前があれば止める）
copies = [(mat, name) for mat in ROOTS for name in FIGS + TABS]
out_html = os.path.join(OUT_DIR, HTML_NAME)
targets = [out_html] + [os.path.join(OUT_DIR, COPY_DIR, MAT_DIR[m], n) for m, n in copies]
exist = [os.path.relpath(p, W) for p in targets if os.path.exists(p)]
if exist:
    sys.exit("同じ名前のファイルがあるので止めました（上書きしない）:\n  " + "\n  ".join(exist))
missing = [src(m, n) for m, n in copies if not os.path.exists(src(m, n))]
assert not missing, f"原本がない: {missing}"

# ---------------------------------------------------------------- 結果を読む
MATS = []
for mat in ["PP", "MIX", "GPPS"]:
    key, verdict = EXPECT[mat]
    S, MT, E, P = rd(mat, "まとめ.csv"), rd(mat, "予測の指標_回ごと.csv"), rd(mat, "シフトファクターの推定_回ごと.csv"), rd(mat, "分け方.csv")
    srow = S[S["モデル"] == key].iloc[0]
    crow = S[S["モデル"] == "C"].iloc[0]
    assert srow["結論"] == verdict, (mat, srow["結論"], "資料の文の前提（ノートの目安）と違う")
    B = key.startswith("B_")

    def rm(fold, model, rng):
        v = MT[(MT["回"] == fold) & (MT["モデル"] == model) & (MT["範囲"] == rng)]["RMSE"]
        assert len(v) == 1, (mat, fold, model, rng)
        return float(v.iloc[0])

    def theta(fold):
        e = E[(E["回"] == fold) & (E["モデル"] == key)]
        assert len(e) == 1, (mat, fold)
        e = e.iloc[0]
        return {"v": [float(e["tv_best"]), float(e["tv_lo"]), float(e["tv_hi"])],
                "h": [float(e["th_best"]), float(e["th_lo"]), float(e["th_hi"])] if B else None,
                "dLML": float(e["dLML_vs_0"]), "lim": float(e["支持の基準"]), "sup": is_true(e["尤度で支持"])}

    folds = []
    PF = P[P["回"] != "全温度"].sort_values("除いた温度 [℃]").reset_index(drop=True)
    assert len(PF) == 3, mat
    for i, p in PF.iterrows():
        fold = p["回"]
        r5 = {"shift": rm(fold, key, "5data"), "C": rm(fold, "C", "5data"), "noT": rm(fold, "C_noTset", "5data")}
        r3 = {"shift": rm(fold, key, "3data 全体"), "C": rm(fold, "C", "3data 全体"), "noT": rm(fold, "C_noTset", "3data 全体")}
        r3l = {"shift": rm(fold, key, "3data 低速側"), "C": rm(fold, "C", "3data 低速側"), "noT": rm(fold, "C_noTset", "3data 低速側")}
        ratio = {"shift": r5["shift"] / r5["noT"], "C": r5["C"] / r5["noT"], "vsC": r5["shift"] / r5["C"]}
        # まとめ.csv の比と一致するか（同じ値から計算しているはずのもの）
        assert abs(ratio["shift"] - float(srow[f"5data の RMSE の比（÷ C_noTset）{POS[i]}"])) < 1e-9, (mat, fold)
        assert abs(ratio["C"] - float(crow[f"5data の RMSE の比（÷ C_noTset）{POS[i]}"])) < 1e-9, (mat, fold)
        folds.append({"fold": fold, "T": int(p["除いた温度 [℃]"]), "pos": p["温度の位置"], "train": p["学習の温度 [℃]"],
                      "nTrain": int(p["学習 行数"]), "nVal5": int(p["検証5 行数"]), "nVal3": int(p["検証3 行数"]),
                      "rmse5": r5, "rmse3": r3, "rmse3low": r3l, "ratio": ratio, "ratio3": r3["shift"] / r3["noT"], "theta": theta(fold)})
    th_all = theta("全温度")
    a_lo, a_hi = th_all["v"][1], th_all["v"][2]
    for f in folds:
        f["inBand"] = a_lo <= f["theta"]["v"][0] <= a_hi
    cnt = {s: sum(st_of(f["ratio"]["shift"]) == s for f in folds) for s in ST}
    assert (cnt["good"], cnt["same"], cnt["bad"]) == (int(srow["良い回"]), int(srow["同等の回"]), int(srow["悪い回"])), mat
    MATS.append({"id": mat, "name": MJ[mat], "shift": key, "struct": "B" if B else "A", "verdict": verdict, "verdictC": crow["結論"],
                 "folds": folds, "all": {"theta": th_all, "rmse3": {"shift": rm("全温度", key, "3data 全体"), "noT": rm("全温度", "C_noTset", "3data 全体")}},
                 "cnt": cnt, "nSup": sum(f["theta"]["sup"] for f in folds), "nBand": sum(f["inBand"] for f in folds),
                 "rmseMax": max(v for f in folds for v in f["rmse5"].values()) * 1.08})
M = {m["id"]: m for m in MATS}

# ---------------------------------------------------------------- 結論の文の前提を確かめる（違えば、文を直す必要がある）
pp, mix, gp = M["PP"], M["MIX"], M["GPPS"]
assert gp["cnt"]["good"] == 3 and gp["nSup"] == 3 and all(f["ratio"]["vsC"] < 1 for f in gp["folds"]), "GPPS の文の前提"
assert all(f["ratio3"] > 1 for f in gp["folds"]), "GPPS の 3data の文の前提（3data では C_noTset のほうが小さい）"
assert mix["cnt"]["good"] == 3 and sum(0.88 < f["ratio"]["shift"] < GOOD for f in mix["folds"]) == 2, "粉砕材混合 PP の文の前提（2 回が境目の近く）"
assert all(f["ratio3"] < 1 for f in mix["folds"]), "粉砕材混合 PP の 3data の文の前提"
neg = [f for f in pp["folds"] if f["theta"]["v"][0] < 0]
assert [f["T"] for f in neg] == [210, 230] and pp["cnt"]["bad"] == 2 and all(f["ratio"]["C"] > 1 for f in pp["folds"]), "PP の文の前提"
assert all(0.95 < f["ratio3"] < 1.05 for f in pp["folds"]), "PP の 3data の文の前提（ほぼ同じ）"


def pct_less(m):
    v = sorted((1 - f["ratio"]["shift"]) * 100 for f in m["folds"])
    return f"{v[0]:.0f}〜{v[-1]:.0f}%"


def ratios(m, k="shift"):
    return "・".join(f"{f['ratio'][k]:.3f}" for f in m["folds"])


def evs(m):
    return "・".join(g(f["theta"]["v"][0]) for f in m["folds"])


def band_txt(m):
    a = m["all"]["theta"]["v"]
    return f"{g(a[0])}［{g(a[1])}〜{g(a[2])}］"


def temps(m):
    return "・".join(str(f["T"]) for f in m["folds"])


def dlmls(m):
    return "・".join(f"{f['theta']['dLML']:.2f}" for f in m["folds"])


# 結論のカード（材料ごと）
CARDS = [
    {"m": gp, "use": True, "head": "使ったほうがよい",
     "pts": [f"除いた温度（{temps(gp)} ℃）の誤差が、3 回とも C_noTset（設定温度を使わない）より <b>{pct_less(gp)} 小さい</b>（比 {ratios(gp)}）。設定温度を GP の普通の入力にした C よりも小さい。",
             f"3 回とも尤度で支持された（ΔLML {dlmls(gp)}、基準 {gp['folds'][0]['theta']['lim']:g}）。係数 E_v は {evs(gp)} kJ/mol で、全温度の {band_txt(gp)} に近い。",
             "ただし、射出速度の外挿（3data）では、C_noTset のほうが誤差が小さかった（3.3 節）。"]},
    {"m": mix, "use": True, "head": "使ったほうがよい（差は小さい）",
     "pts": [f"除いた温度の誤差が、3 回とも C_noTset より <b>{pct_less(mix)} 小さい</b>（比 {ratios(mix)}）。ただし 2 回は、「良い」の境目（0.909）のすぐ下。",
             f"C とはほぼ同じ（C との比 {ratios(mix, 'vsC')}）。シフトの形でも GP の入力でも、設定温度を使うと良くなる。",
             f"係数 E_v は {evs(mix)} kJ/mol で、3 回とも全温度の区間（{band_txt(mix)}）に入る。尤度の支持は 3 回中 {mix['nSup']} 回。"]},
    {"m": pp, "use": False, "head": "使わなくてよい",
     "pts": [f"除いた温度の誤差が、C_noTset より <b>大きい回が {pp['cnt']['bad']} 回</b>、同等が {pp['cnt']['same']} 回（比 {ratios(pp)}）。C も 3 回とも C_noTset より大きい（比 {ratios(pp, 'C')}）。",
             f"{'・'.join(str(f['T']) for f in neg)} ℃ を除くと、係数 E_v が<b>負</b>になる（{'・'.join(g(f['theta']['v'][0]) for f in neg)} kJ/mol。高温ほど粘度が高い向き）。全温度の {band_txt(pp)} から外れ、係数が安定しない。",
             "PP では、設定温度の効果は、Tff（流動中樹脂温度）と P3_3 でほぼ説明できると読める。"]},
]


def card_html(c):
    m = c["m"]
    cls = "use" if c["use"] else "nouse"
    icon = "✓" if c["use"] else "—"
    lis = "".join(f"<li>{p}</li>" for p in c["pts"])
    return (f'<article class="card {cls}"><header><span class="mat">{esc(m["name"])}</span>'
            f'<span class="model">{esc(m["shift"])}</span></header>'
            f'<p class="verdict"><span class="vicon" aria-hidden="true">{icon}</span>{esc(c["head"])}</p>'
            f'<p class="nb">ノートの目安：{esc(m["verdict"])}（良い {m["cnt"]["good"]}・同等 {m["cnt"]["same"]}・悪い {m["cnt"]["bad"]} 回）</p>'
            f"<ul>{lis}</ul></article>")


# ---------------------------------------------------------------- 表
def tab5():
    rows = []
    for m in MATS:
        for i, f in enumerate(m["folds"]):
            head = f'<th rowspan="3" scope="rowgroup">{esc(m["name"])}<br><span class="sub">{esc(m["shift"])}</span></th>' if i == 0 else ""
            r = f["rmse5"]
            rows.append(f"<tr>{head}<td>{f['T']} ℃<span class='sub'>（{esc(f['pos'])}）</span></td>"
                        f"<td class='num'>{r['shift']:.4f}</td><td class='num'>{r['C']:.4f}</td><td class='num'>{r['noT']:.4f}</td>"
                        f"<td class='num strong'>{f['ratio']['shift']:.3f}</td><td>{badge(f['ratio']['shift'])}</td>"
                        f"<td class='num'>{f['ratio']['C']:.3f}</td></tr>")
    return "\n".join(rows)


def tab_theta():
    rows = []
    for m in MATS:
        cats = m["folds"] + [dict(T=None, theta=m["all"]["theta"], inBand=None)]
        for i, f in enumerate(cats):
            head = f'<th rowspan="4" scope="rowgroup">{esc(m["name"])}<br><span class="sub">{esc(m["shift"])}</span></th>' if i == 0 else ""
            t = f["theta"]
            v = f"{g(t['v'][0])}<span class='sub'>［{g(t['v'][1])}〜{g(t['v'][2])}］</span>"
            h = f"{g(t['h'][0])}<span class='sub'>［{g(t['h'][1])}〜{g(t['h'][2])}］</span>" if t["h"] else "―"
            sup = "○" if t["sup"] else "<span class='warn'>×</span>"
            band = "―（基準）" if f["T"] is None else ("○" if f["inBand"] else "<span class='warn'>× 区間の外</span>")
            name = "全温度（参考）" if f["T"] is None else f"{f['T']} ℃ を除く"
            rows.append(f"<tr>{head}<td>{name}</td><td class='num'>{v}</td><td class='num'>{h}</td>"
                        f"<td class='num'>{t['dLML']:.2f}<span class='sub'>（{t['lim']:g}）</span></td><td>{sup}</td><td>{band}</td></tr>")
    return "\n".join(rows)


def tab3():
    rows = []
    for m in MATS:
        for i, f in enumerate(m["folds"]):
            head = f'<th rowspan="3" scope="rowgroup">{esc(m["name"])}</th>' if i == 0 else ""
            r, l = f["rmse3"], f["rmse3low"]
            rows.append(f"<tr>{head}<td>{f['T']} ℃</td><td class='num'>{r['shift']:.3f}</td><td class='num'>{r['C']:.3f}</td><td class='num'>{r['noT']:.3f}</td>"
                        f"<td class='num strong'>{f['ratio3']:.2f}</td><td class='num'>{l['shift']:.3f}</td><td class='num'>{l['noT']:.3f}</td></tr>")
    return "\n".join(rows)


def rel_repo(p):
    return os.path.relpath(p, W).replace("\\", "/")


def url(p):
    return urllib.parse.quote(p.replace("\\", "/"))


def tab_files():
    rows = []
    for mat in ROOTS:
        for n in FIGS + TABS:
            cp = f"{COPY_DIR}/{MAT_DIR[mat]}/{n}"
            rows.append(f"<tr><td>{'図' if n.endswith('.png') else '表'}</td><td>{esc(MJ[mat])}</td>"
                        f"<td><a href=\"{url(cp)}\">{esc(cp)}</a></td><td class='path'>{esc(ROOTS[mat] + '/' + HOLD + '/' + n)}</td></tr>")
    return "\n".join(rows)


def fig_tabs(fname, cap, gid):
    btns, panels = [], []
    for i, mat in enumerate(ROOTS):
        cp = f"{COPY_DIR}/{MAT_DIR[mat]}/{fname}"
        sel = "true" if i == 0 else "false"
        btns.append(f'<button type="button" role="tab" aria-selected="{sel}" aria-controls="{gid}-{i}" data-idx="{i}">{esc(MJ[mat])}</button>')
        panels.append(f'<figure class="tabpanel{" active" if i == 0 else ""}" id="{gid}-{i}" role="tabpanel" data-idx="{i}">'
                      f'<a href="{url(cp)}" target="_blank" rel="noopener"><img src="{url(cp)}" alt="{esc(MJ[mat])}：{esc(cap)}" loading="lazy"></a>'
                      f'<figcaption>{esc(MJ[mat])}：{esc(cap)}（クリックで原寸。原本のコピー <code>{esc(cp)}</code>）</figcaption></figure>')
    return (f'<div class="figtabs" data-figtabs><div class="controls tablist" role="tablist" aria-label="材料">{"".join(btns)}</div>'
            f'{"".join(panels)}</div>')


# ---------------------------------------------------------------- HTML
DATA = {"thr": {"good": GOOD, "bad": BAD}, "materials": MATS}
summary = ("温度シフトファクター（加熱シリンダ温度 T<sub>set</sub> のシフト）は、<b>GPPS と粉砕材混合 PP では使ったほうがよく、PP では使わなくてよい</b>。"
           "ただし、粉砕材混合 PP は差が小さく、GPPS は射出速度の外挿（3data）では逆の結果だった。")

PAGE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>温度シフトファクターの要否</title>
<style>
:root{--bg:#f7f7f5;--card:#ffffff;--ink:#1f2328;--ink2:#4a525b;--muted:#77808a;--line:#d9dde1;--grid:#eceef0;
--shift:#1f77b4;--c:#7f7f7f;--notset:#ff7f0e;
--good:#2e7d32;--goodbg:#e7f2e8;--same:#5f6670;--samebg:#eef0f2;--bad:#c62828;--badbg:#fbe9e9;
--train:#3d5a80;--trainink:#ffffff;--hold:#fff1cc;--holdline:#a86b00;--band:#e3e6ea;--track:#eef0f3;--accent:#3d5a80}
@media (prefers-color-scheme: dark){:root{--bg:#16181b;--card:#1f2226;--ink:#e8eaed;--ink2:#b9bfc6;--muted:#8d949c;--line:#373c42;--grid:#2b2f34;
--shift:#5aa9e6;--c:#a3a3a3;--notset:#ffa14d;
--good:#6cc070;--goodbg:#1f3322;--same:#a7adb5;--samebg:#2a2e33;--bad:#f07070;--badbg:#3a2020;
--train:#5b7ba3;--trainink:#ffffff;--hold:#4a3b10;--holdline:#e0a63a;--band:#30353b;--track:#2a2e33;--accent:#8fb3e0}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Yu Gothic UI","Yu Gothic","Meiryo","Hiragino Sans","Noto Sans JP",sans-serif;line-height:1.75;font-size:15px}
main{max-width:980px;margin:0 auto;padding:28px 16px 72px}
h1{font-size:1.55rem;line-height:1.4;margin:0 0 .3rem}
h2{font-size:1.2rem;margin:2.4rem 0 .6rem;padding-bottom:.25rem;border-bottom:2px solid var(--line)}
h3{font-size:1.02rem;margin:1.6rem 0 .4rem}
p{margin:.45rem 0}
.lead{color:var(--ink2);font-size:.92rem;margin-bottom:1rem}
.summary{background:var(--card);border:1px solid var(--line);border-left:6px solid var(--accent);border-radius:10px;padding:14px 18px;font-size:1.05rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:14px;margin-top:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;opacity:1;transform:none;transition:opacity .6s ease,transform .6s ease}
.card.pending{opacity:0;transform:translateY(14px)}
.card header{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.card .mat{font-weight:700;font-size:1.1rem}
.card .model{font-family:Consolas,"Courier New",monospace;font-size:.82rem;color:var(--muted)}
.card .verdict{font-size:1.2rem;font-weight:700;margin:.4rem 0 .1rem;display:flex;align-items:center;gap:.5rem}
.card.use .verdict{color:var(--good)} .card.nouse .verdict{color:var(--same)}
.vicon{display:inline-flex;width:1.7rem;height:1.7rem;border-radius:50%;align-items:center;justify-content:center;font-size:1rem;color:#fff}
.card.use .vicon{background:var(--good)} .card.nouse .vicon{background:var(--same)}
.card .nb{font-size:.82rem;color:var(--muted);margin:0 0 .3rem}
.card ul{margin:.3rem 0 0;padding-left:1.1rem;font-size:.9rem}
.card li{margin:.25rem 0}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:.8rem 0}
.controls{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:6px}
.controls .label{font-size:.8rem;color:var(--muted);margin-right:2px}
button{font:inherit;font-size:.85rem;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:999px;padding:3px 12px;cursor:pointer}
button[aria-selected="true"],button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
button:focus-visible,[tabindex]:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
.spacer{flex:1}
.grp,.fold-btns{display:inline-flex;flex-wrap:wrap;gap:6px;align-items:center}
svg{width:100%;height:auto;display:block;overflow:visible}
svg text{fill:var(--ink);font-family:inherit}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:.82rem;color:var(--ink2);margin:.2rem 0 .4rem}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:14px;height:10px;border-radius:3px;display:inline-block}
.sw.good{background:var(--good)} .sw.same{background:var(--same)} .sw.bad{background:var(--bad)}
.sw.shift{background:var(--shift)} .sw.c{background:var(--c)} .sw.notset{background:var(--notset)}
.sw.band{background:var(--band);border:1px solid var(--line)}
.grid{stroke:var(--grid);stroke-width:1}
.axis-one{stroke:var(--ink);stroke-width:1.5}
.axis-zero{stroke:var(--ink2);stroke-width:1.2}
.thr{stroke:var(--ink2);stroke-width:1;stroke-dasharray:4 4}
.zone-good{fill:var(--goodbg)} .zone-bad{fill:var(--badbg)}
.zone-label{font-size:12px;fill:var(--ink2)}
.tick{font-size:12px;fill:var(--muted)}
.axis-title{font-size:12.5px;fill:var(--ink2)}
.row-head{font-size:14px;font-weight:700}
.row-label{font-size:12.5px;fill:var(--ink2)}
.val{font-size:12.5px;font-variant-numeric:tabular-nums}
.note-small{font-size:11.5px;fill:var(--muted)}
.bar{transform-box:fill-box;transition:transform .9s cubic-bezier(.2,.75,.2,1)}
.bar.grow-left{transform-origin:right center} .bar.grow-right{transform-origin:left center}
.bar.pending{transform:scaleX(0)}
.st-good{fill:var(--good)} .st-same{fill:var(--same)} .st-bad{fill:var(--bad)}
.fade{transition:opacity .6s ease} .fade.pending{opacity:0}
.hit{fill:transparent;cursor:default}
.hit:hover,.hit:focus{fill:var(--ink);fill-opacity:.04}
.tb-rect{transition:fill .5s ease,stroke .5s ease}
.tblock.train .tb-rect{fill:var(--train);stroke:var(--train);stroke-width:2}
.tblock.train text{fill:var(--trainink)}
.tblock.held .tb-rect{fill:var(--hold);stroke:var(--holdline);stroke-width:2.5;stroke-dasharray:7 5;animation:pulse 1.8s ease-in-out infinite}
.tblock.held text{fill:var(--ink)}
@keyframes pulse{50%{stroke-width:5}}
.tb-temp{font-size:22px;font-weight:700} .tb-role{font-size:12.5px}
.m-title{font-size:15px;font-weight:700} .m-flow{font-size:13px;fill:var(--ink2)} .m-sub{font-size:12.5px;fill:var(--ink2)}
.m-ratio{font-size:14px;font-weight:700}
.st-text-good{fill:var(--good)} .st-text-same{fill:var(--same)} .st-text-bad{fill:var(--bad)}
.track{fill:var(--track)}
.mbar{transform-box:fill-box;transform-origin:left center;transition:transform .8s cubic-bezier(.2,.75,.2,1)}
.mbar-shift{fill:var(--shift)} .mbar-C{fill:var(--c)} .mbar-noT{fill:var(--notset)}
.tband{fill:var(--band)}
.whisker{stroke:var(--shift);stroke-width:2}
.tdot{fill:var(--shift);stroke:var(--card);stroke-width:2}
.tpt.isall .tdot{fill:var(--card);stroke:var(--shift);stroke-width:2.5}
.tpt.out .tdot{fill:var(--bad)} .tpt.out .whisker{stroke:var(--bad)}
.out-label{font-size:11.5px;fill:var(--bad)}
.tpt{transition:opacity .5s ease,transform .7s cubic-bezier(.2,.75,.2,1)}
.tpt.pending{opacity:0;transform:translateY(-36px)}
table{border-collapse:collapse;width:100%;font-size:.86rem;background:var(--card)}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:middle}
thead th{background:var(--samebg);font-weight:600;font-size:.8rem;color:var(--ink2);white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.strong{font-weight:700}
.sub{color:var(--muted);font-size:.78rem;font-weight:400;margin-left:2px}
.warn{color:var(--bad);font-weight:600}
.st{display:inline-block;border-radius:999px;padding:0 9px;font-size:.78rem;font-weight:700;white-space:nowrap}
.st.st-good{background:var(--goodbg);color:var(--good)} .st.st-same{background:var(--samebg);color:var(--same)} .st.st-bad{background:var(--badbg);color:var(--bad)}
.tablewrap{overflow-x:auto;margin:.6rem 0}
.figtabs .tabpanel{display:none;margin:.4rem 0 0}
.figtabs .tabpanel.active{display:block}
.figtabs img{width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:8px}
figcaption{font-size:.8rem;color:var(--muted);margin-top:4px}
code{font-family:Consolas,"Courier New",monospace;font-size:.85em;word-break:break-all}
td.path{font-family:Consolas,"Courier New",monospace;font-size:.72rem;color:var(--muted);word-break:break-all}
dl.terms{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;font-size:.88rem;margin:.6rem 0}
dl.terms dt{font-weight:700} dl.terms dd{margin:0;color:var(--ink2)}
ul.notes li{margin:.3rem 0}
.foot{margin-top:2.6rem;font-size:.78rem;color:var(--muted);border-top:1px solid var(--line);padding-top:.6rem}
#tip{position:fixed;z-index:10;pointer-events:none;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:7px 10px;font-size:12.5px;line-height:1.55;box-shadow:0 4px 16px rgba(0,0,0,.18);max-width:300px}
#tip b{font-weight:700}
#tip .k{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}
@media (prefers-reduced-motion: reduce){*,*::before,*::after{transition:none!important;animation:none!important}}
@media (max-width:640px){body{font-size:14px} .tb-temp{font-size:20px}}
@media print{
  @page{size:A4;margin:12mm}
  body{background:#fff;font-size:11.5px}
  main{max-width:none;padding:0}
  .controls,.noprint{display:none!important}
  .figtabs .tabpanel{display:block!important;break-inside:avoid}
  .card,.panel,figure,table{break-inside:avoid}
  .pending{opacity:1!important;transform:none!important}
  .tblock.held .tb-rect{animation:none}
  #tip{display:none}
}
</style>
<noscript><style>.figtabs .tabpanel{display:block}.pending{opacity:1!important;transform:none!important}</style></noscript>
</head>
<body>
<main>
<h1>温度シフトファクターを使ったほうがよいか<br><span style="font-size:.75em;font-weight:600;color:var(--ink2)">3 材料の温度ホールドアウト検証の結果から</span></h1>
<p class="lead">@@RUNINFO@@</p>

<h2>1. 結論</h2>
<div class="summary">@@SUMMARY@@</div>
<div class="cards" id="cards">@@CARDS@@</div>

<h2>2. 調べ方</h2>
<p>温度シフトファクターの本来の役目は、<b>学習にない温度の粘度を当てること</b>です。そこで、材料ごとに 3 つの加熱シリンダ温度のうち 1 つを学習から除き、残りの 2 温度の 5data で学習して、除いた温度の 5data を予測しました。これを 3 つの温度すべてで行いました（3 回）。下の図は、その 3 回を順に見せています。</p>
<div class="panel" id="method">
  <div class="controls">
    <span class="label">材料</span>
    <span role="tablist" aria-label="材料" class="mat-tabs">@@METHOD_TABS@@</span>
    <span class="spacer"></span>
    <span class="grp"><span class="label">回</span><span class="fold-btns"></span>
    <button type="button" class="play">⏸ 一時停止</button></span>
  </div>
  <div class="method-svg"></div>
  <div class="legend"><span><i class="sw shift"></i>温度シフトのモデル</span><span><i class="sw c"></i>C（設定温度を GP の普通の入力にする）</span><span><i class="sw notset"></i>C_noTset（設定温度を使わない）</span></div>
</div>
<p>比べたモデルは 3 つです。比べの基準は C_noTset で、<b>比（モデルの誤差 ÷ C_noTset の誤差）が 1/1.1 ≒ 0.909 より小さければ「良い」、1.1 より大きければ「悪い」</b>、その間を「同等」としました。</p>
<dl class="terms">
  <dt>温度シフトのモデル</dt><dd>設定温度 T<sub>set</sub> による粘度のずれを、アレニウス型のシフトファクターで表すモデル。A_Tset_Arr は縦（粘度の向き）だけ、B_Tset_Arr は横（せん断速度の向き）と縦にずらす。係数 E_v・E_h［kJ/mol］はデータから決める。</dd>
  <dt>C・C_noTset</dt><dd>シフトのない通常の GPR。C は T<sub>set</sub> を GP の普通の入力にする。C_noTset は T<sub>set</sub> を使わない（入力は ln γ̇・ln P3_3・ln Tff。シフトの係数を 0 にしたモデルと同じ）。</dd>
  <dt>RMSE</dt><dd>予測の誤差（ln η の二乗平均平方根）。小さいほど良い。0.01 は粘度で約 1%。</dd>
  <dt>ΔLML・尤度で支持</dt><dd>シフトの係数を 0 にしたときより、学習データの説明（対数周辺尤度）がどれだけ良くなるか。基準（A は 1.92、B は 3.0）以上で「支持」。</dd>
  <dt>95% 区間</dt><dd>係数の、おおよその 95% の範囲（プロファイル尤度から）。</dd>
</dl>

<h2>3. 結果</h2>
<h3>3.1 除いた温度の誤差（5data。主の比べ）</h3>
<p>棒は、温度シフトのモデルの誤差が C_noTset の何倍かを表します。<b>左（緑の範囲）に伸びるほど、シフトを使うと良い</b>ことを表します。棒に指を当てると、値が出ます。</p>
<div class="panel">
  <div class="legend"><span><i class="sw good"></i>✓ 良い（比 &lt; 0.909）</span><span><i class="sw same"></i>＝ 同等</span><span><i class="sw bad"></i>✕ 悪い（比 &gt; 1.1）</span></div>
  <div id="ratio-chart"></div>
</div>
<div class="tablewrap"><table>
<thead><tr><th>材料</th><th>除いた温度</th><th>RMSE：シフト</th><th>C</th><th>C_noTset</th><th>比（シフト ÷ C_noTset）</th><th>判定</th><th>比（C ÷ C_noTset）</th></tr></thead>
<tbody>
@@TAB5@@
</tbody></table></div>
<p>原本の図（ノートが作ったもの。左が除いた温度の 5data、右が 3data）：</p>
@@FIG_RMSE@@

<h3>3.2 シフトの係数は、どの温度を除いても安定しているか</h3>
<p>シフトファクターが温度の効果を正しく表していれば、どの温度を除いても、係数はほぼ同じになるはずです。灰色の帯は、全温度で学習したときの E_v の 95% 区間です。<b>赤い点は、帯の外に出た回</b>です。</p>
<div class="panel">
  <div class="legend"><span><i class="sw shift"></i>その回の E_v（点）と 95% 区間（縦線）</span><span><i class="sw band"></i>全温度の 95% 区間</span><span><i class="sw bad"></i>帯の外</span></div>
  <div id="theta-chart"></div>
</div>
<div class="tablewrap"><table>
<thead><tr><th>材料</th><th>回</th><th>E_v［95% 区間］kJ/mol</th><th>E_h［95% 区間］kJ/mol</th><th>ΔLML（基準）</th><th>尤度で支持</th><th>全温度の区間に入る</th></tr></thead>
<tbody>
@@TABTH@@
</tbody></table></div>
<p>原本の図：</p>
@@FIG_THETA@@

<h3>3.3 参考：3data（温度と射出速度の両方の外挿）</h3>
<p>除いた温度の 3data（射出速度の両端。学習の範囲の外）の誤差です。<b>GPPS では、3 回とも C_noTset のほうが小さく</b>（比 1 より大）、特に低速側で差が大きくなりました。設定温度を使うと、除いた温度の予測は良くなりますが、射出速度の外挿は良くなりませんでした（理由は未確認）。粉砕材混合 PP はシフトのモデルが小さく、PP はほぼ同じでした。</p>
<div class="tablewrap"><table>
<thead><tr><th>材料</th><th>除いた温度</th><th>3data 全体：シフト</th><th>C</th><th>C_noTset</th><th>比（シフト ÷ C_noTset）</th><th>低速側：シフト</th><th>低速側：C_noTset</th></tr></thead>
<tbody>
@@TAB3@@
</tbody></table></div>

<h2>4. 読むときの注意</h2>
<ul class="notes">
<li><b>学習は 2 温度だけ</b>です。温度の効果は 2 点の差でしか決まらないので、全温度で学習するときより係数がぶれやすくなります。両端の温度を除く回は温度の外挿、中の温度を除く回は内挿です。</li>
<li>C_noTset は設定温度を使いませんが、<b>流動中樹脂温度 Tff と P3_3 を入力に使う</b>ので、温度の情報の一部はそれらが運びます。「温度をまったく使わない」モデルではありません。</li>
<li>粉砕材混合 PP の「良い」は、3 回中 2 回が境目（0.909）のすぐ下です。乱数や最適化の局所解で、判定が 1 段変わりうる大きさです（短い設定の試しの実行では、230 ℃ の回が 0.915 で「同等」でした）。</li>
<li>どれも 1 回の実行（乱数シード 42）で、<b>線形項あり・通常版・全ショット版</b>の結果です。線形項なし・温度適切版では確かめていません。</li>
<li>決めるのはユーザーです。この資料は、温度ホールドアウト検証のノートの「まとめ」の目安と、その根拠の数値をまとめたものです。</li>
</ul>

<h2>5. 引用した図と表</h2>
<p>原本のコピーを、このフォルダの <code>@@COPYDIR@@/</code> に置きました（元の名前のまま。原本は変えていません）。</p>
<div class="tablewrap"><table>
<thead><tr><th>種類</th><th>材料</th><th>コピー（このフォルダから）</th><th>原本（リポジトリ直下から）</th></tr></thead>
<tbody>
@@TABFILES@@
</tbody></table></div>
<p class="foot">@@FOOT@@</p>
</main>
<div id="tip" role="tooltip" hidden></div>
<script type="application/json" id="data">@@DATA@@</script>
<script>
(function () {
  'use strict';
  var DATA = JSON.parse(document.getElementById('data').textContent);
  var reduce = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var NS = 'http://www.w3.org/2000/svg';
  var ST = {good: ['✓', '良い'], same: ['＝', '同等'], bad: ['✕', '悪い']};
  function el(tag, attrs, parent, text) {
    var e = document.createElementNS(NS, tag);
    if (attrs) { Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); }); }
    if (text !== undefined) { e.textContent = text; }
    if (parent) { parent.appendChild(e); }
    return e;
  }
  function status(r) { return r < DATA.thr.good ? 'good' : (r > DATA.thr.bad ? 'bad' : 'same'); }
  function g(x) { var s = String(Math.round(Math.abs(x) * 100) / 100); return x < 0 ? '−' + s : s; }
  function later(fn, ms) { if (reduce) { fn(); } else { setTimeout(fn, ms); } }

  // ---- 説明の吹き出し（指を当てる・キーボードで選ぶ）
  var tip = document.getElementById('tip');
  function place(x, y) {
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var left = x + 14, top = y - h - 12;
    if (left + w > window.innerWidth - 8) { left = Math.max(8, x - w - 14); }
    if (top < 8) { top = y + 18; }
    tip.style.left = left + 'px'; tip.style.top = top + 'px';
  }
  function bindTip(node, html) {
    node.setAttribute('tabindex', '0');
    node.addEventListener('pointerenter', function (e) { tip.innerHTML = html(); tip.hidden = false; place(e.clientX, e.clientY); });
    node.addEventListener('pointermove', function (e) { place(e.clientX, e.clientY); });
    node.addEventListener('pointerleave', function () { tip.hidden = true; });
    node.addEventListener('focus', function () { tip.innerHTML = html(); tip.hidden = false; var r = node.getBoundingClientRect(); place(r.left + r.width / 2, r.top); });
    node.addEventListener('blur', function () { tip.hidden = true; });
  }
  function onVisible(node, fn) {
    if (reduce || !('IntersectionObserver' in window)) { fn(); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { io.disconnect(); fn(); } });
    }, {threshold: 0.2});
    io.observe(node);
  }
  function key(color) { return '<span class="k" style="background:var(' + color + ')"></span>'; }
  function rmseRows(f, m) {
    return key('--shift') + m.shift + '：' + f.rmse5.shift.toFixed(4) + '<br>' +
           key('--c') + 'C：' + f.rmse5.C.toFixed(4) + '<br>' +
           key('--notset') + 'C_noTset：' + f.rmse5.noT.toFixed(4);
  }

  // ---- 1. 結論のカード（画面に入ったら順に出す）
  var cards = Array.prototype.slice.call(document.querySelectorAll('#cards .card'));
  cards.forEach(function (c) { c.classList.add('pending'); });
  onVisible(document.getElementById('cards'), function () {
    cards.forEach(function (c, k) { later(function () { c.classList.remove('pending'); }, k * 180); });
  });

  // ---- 2. 調べ方の図（3 回を順に回す）
  (function method() {
    var host = document.getElementById('method');
    var Wd = 760, H = 330;
    var svg = el('svg', {viewBox: '0 0 ' + Wd + ' ' + H, role: 'img', 'aria-label': '1 つの温度を除いて学習し、除いた温度を予測する 3 回の分け方と、その回の誤差'}, host.querySelector('.method-svg'));
    var title = el('text', {x: Wd / 2, y: 24, class: 'm-title', 'text-anchor': 'middle'}, svg, '');
    var bx = [70, 290, 510], bw = 180, bh = 74, by = 42;
    var blocks = bx.map(function (x) {
      var g0 = el('g', {class: 'tblock train'}, svg);
      var r = el('rect', {x: x, y: by, width: bw, height: bh, rx: 10, class: 'tb-rect'}, g0);
      var t1 = el('text', {x: x + bw / 2, y: by + 33, class: 'tb-temp', 'text-anchor': 'middle'}, g0, '');
      var t2 = el('text', {x: x + bw / 2, y: by + 57, class: 'tb-role', 'text-anchor': 'middle'}, g0, '');
      return {g: g0, r: r, t1: t1, t2: t2};
    });
    var flow = el('text', {x: Wd / 2, y: by + bh + 28, class: 'm-flow', 'text-anchor': 'middle'}, svg, '');
    var bL = 270, bR = 92, bTop = by + bh + 64, bRow = 32, bW = Wd - bL - bR;
    el('text', {x: 20, y: bTop - 12, class: 'm-sub'}, svg, '除いた温度の 5data の誤差（RMSE、ln η）');
    var models = [['shift', ''], ['C', 'C（T_set を GP の入力に）'], ['noT', 'C_noTset（T_set を使わない）']];
    var bars = models.map(function (mk, j) {
      var y = bTop + j * bRow;
      var lab = el('text', {x: bL - 10, y: y + 16, class: 'row-label', 'text-anchor': 'end'}, svg, mk[1]);
      el('rect', {x: bL, y: y + 3, width: bW, height: 18, rx: 3, class: 'track'}, svg);
      var r = el('rect', {x: bL, y: y + 3, width: bW, height: 18, rx: 3, class: 'mbar mbar-' + mk[0]}, svg);
      r.style.transform = 'scaleX(0)';
      var v = el('text', {x: bL + bW + 8, y: y + 17, class: 'val'}, svg, '');
      return {k: mk[0], lab: lab, r: r, v: v};
    });
    var ratioTxt = el('text', {x: Wd / 2, y: bTop + 3 * bRow + 24, class: 'm-ratio', 'text-anchor': 'middle'}, svg, '');
    var tabs = Array.prototype.slice.call(host.querySelectorAll('[data-mat]'));
    var foldBox = host.querySelector('.fold-btns');
    var playBtn = host.querySelector('.play');
    var mi = 0, fi = 0, playing = !reduce, timer = null, shown = false, foldBtns = [];

    function buildFolds() {
      foldBox.innerHTML = '';
      foldBtns = DATA.materials[mi].folds.map(function (f, k) {
        var b = document.createElement('button');
        b.type = 'button';
        b.textContent = f.T + ' ℃';
        b.setAttribute('aria-label', f.T + ' ℃ を除く回');
        b.addEventListener('click', function () { fi = k; stop(); });
        foldBox.appendChild(b);
        return b;
      });
    }
    function render() {
      var m = DATA.materials[mi], f = m.folds[fi];
      title.textContent = m.name + '　回 ' + (fi + 1) + '／3：' + f.T + ' ℃ を除く（' + f.pos + '）';
      m.folds.forEach(function (ff, k) {
        var b = blocks[k], held = (k === fi);
        b.g.setAttribute('class', 'tblock ' + (held ? 'held' : 'train'));
        b.t1.textContent = ff.T + ' ℃';
        b.t2.textContent = held ? '除く → 予測して確かめる' : '学習に使う（5data）';
      });
      flow.textContent = f.train + ' ℃ の 5data（' + f.nTrain + ' 行）で学習　→　' + f.T + ' ℃ の 5data（' + f.nVal5 + ' 行）を予測';
      bars.forEach(function (b) {
        var v = f.rmse5[b.k];
        if (shown) { b.r.style.transform = 'scaleX(' + (v / m.rmseMax) + ')'; }
        b.v.textContent = v.toFixed(4);
        if (b.k === 'shift') { b.lab.textContent = m.shift + '（温度シフト）'; }
      });
      var r = f.ratio.shift, st = status(r);
      ratioTxt.textContent = '比（' + m.shift + ' ÷ C_noTset）＝ ' + r.toFixed(3) + '　→　' + ST[st][0] + ' ' + ST[st][1];
      ratioTxt.setAttribute('class', 'm-ratio st-text-' + st);
      tabs.forEach(function (t, k) { t.setAttribute('aria-selected', k === mi ? 'true' : 'false'); });
      foldBtns.forEach(function (b, k) { b.setAttribute('aria-pressed', k === fi ? 'true' : 'false'); });
      playBtn.textContent = playing ? '⏸ 一時停止' : '▶ 自動で回す';
    }
    function tick() { fi = (fi + 1) % 3; render(); }
    function start() { playing = true; clearInterval(timer); timer = setInterval(tick, 3600); render(); }
    function stop() { playing = false; clearInterval(timer); timer = null; render(); }
    tabs.forEach(function (t, k) {
      t.addEventListener('click', function () { mi = k; fi = 0; buildFolds(); if (playing) { start(); } else { render(); } });
    });
    playBtn.addEventListener('click', function () { if (playing) { stop(); } else { start(); } });
    buildFolds();
    render();
    onVisible(host, function () { shown = true; render(); if (playing) { start(); } });
  })();

  // ---- 3.1 誤差の比の横棒（1 から左右へ伸びる）
  (function ratioChart() {
    var host = document.getElementById('ratio-chart');
    var Wd = 760, left = 200, right = 24, top = 40, rowH = 30, headH = 26;
    var x0 = 0.6, x1 = 2.2;
    var rows = [], y = top;
    DATA.materials.forEach(function (m) {
      rows.push({head: true, m: m, y: y}); y += headH;
      m.folds.forEach(function (f) { rows.push({head: false, m: m, f: f, y: y}); y += rowH; });
      y += 8;
    });
    var plotBottom = y, H = y + 44;
    function X(v) { return left + (v - x0) / (x1 - x0) * (Wd - left - right); }
    var svg = el('svg', {viewBox: '0 0 ' + Wd + ' ' + H, role: 'img', 'aria-label': '除いた温度の 5data の誤差の比（温度シフトのモデル ÷ C_noTset）。表と同じ値'}, host);
    el('rect', {x: X(x0), y: top - 8, width: X(DATA.thr.good) - X(x0), height: plotBottom - top + 8, class: 'zone-good'}, svg);
    el('rect', {x: X(DATA.thr.bad), y: top - 8, width: X(x1) - X(DATA.thr.bad), height: plotBottom - top + 8, class: 'zone-bad'}, svg);
    for (var t = 0.6; t <= 2.2001; t += 0.2) {
      var tt = Math.round(t * 10) / 10;
      el('line', {x1: X(tt), x2: X(tt), y1: top - 8, y2: plotBottom, class: Math.abs(tt - 1) < 1e-9 ? 'axis-one' : 'grid'}, svg);
      el('text', {x: X(tt), y: plotBottom + 16, class: 'tick', 'text-anchor': 'middle'}, svg, tt.toFixed(1));
    }
    [DATA.thr.good, DATA.thr.bad].forEach(function (v) { el('line', {x1: X(v), x2: X(v), y1: top - 8, y2: plotBottom, class: 'thr'}, svg); });
    el('text', {x: X(DATA.thr.good) - 6, y: top - 14, class: 'zone-label', 'text-anchor': 'end'}, svg, '← 良い（0.909 未満）');
    el('text', {x: (X(DATA.thr.good) + X(DATA.thr.bad)) / 2, y: top - 14, class: 'zone-label', 'text-anchor': 'middle'}, svg, '同等');
    el('text', {x: X(DATA.thr.bad) + 6, y: top - 14, class: 'zone-label'}, svg, '悪い（1.1 超）→');
    el('text', {x: X(1), y: plotBottom + 36, class: 'axis-title', 'text-anchor': 'middle'}, svg, '誤差の比（温度シフトのモデル ÷ C_noTset）。1 ＝ C_noTset と同じ');
    var bars = [], labels = [];
    rows.forEach(function (r) {
      if (r.head) { el('text', {x: 6, y: r.y + 17, class: 'row-head'}, svg, r.m.name + '（' + r.m.shift + '）'); return; }
      var f = r.f, v = f.ratio.shift, st = status(v), cy = r.y + rowH / 2;
      el('text', {x: left - 10, y: cy + 4, class: 'row-label', 'text-anchor': 'end'}, svg, f.T + ' ℃ を除く');
      var xa = X(1), xb = X(v);
      var bar = el('rect', {x: Math.min(xa, xb), y: cy - 8, width: Math.max(1, Math.abs(xb - xa)), height: 16, rx: 3,
                            class: 'bar st-' + st + ' pending ' + (v < 1 ? 'grow-left' : 'grow-right')}, svg);
      var lx = v < 1 ? xa + 8 : xb + 8;
      var lab = el('text', {x: lx, y: cy + 4, class: 'val fade pending'}, svg, ST[st][0] + ' ' + v.toFixed(3) + ' ' + ST[st][1]);
      var hit = el('rect', {x: 0, y: r.y, width: Wd, height: rowH, class: 'hit'}, svg);
      bindTip(hit, function () {
        return '<b>' + r.m.name + '：' + f.T + ' ℃ を除いた回</b>（' + f.pos + '）<br>' + rmseRows(f, r.m) +
               '<br>比（÷ C_noTset）：<b>' + v.toFixed(3) + '</b>（' + ST[st][1] + '）／ C の比：' + f.ratio.C.toFixed(3);
      });
      bars.push(bar); labels.push(lab);
    });
    onVisible(svg, function () {
      bars.forEach(function (b, k) { later(function () { b.classList.remove('pending'); }, 150 + k * 110); });
      labels.forEach(function (l, k) { later(function () { l.classList.remove('pending'); }, 650 + k * 110); });
    });
  })();

  // ---- 3.2 回ごとの係数 E_v（全温度の帯の上へ、点が順に落ちる）
  (function thetaChart() {
    var host = document.getElementById('theta-chart');
    var Wd = 760, H = 340, top = 44, bottom = 62, left = 54, gap = 22;
    var n = DATA.materials.length, pw = (Wd - left - gap * (n - 1) - 8) / n;
    var vals = [];
    DATA.materials.forEach(function (m) { m.folds.concat([m.all]).forEach(function (f) { vals.push(f.theta.v[1], f.theta.v[2]); }); });
    var lo = Math.floor(Math.min.apply(null, vals) / 5) * 5, hi = Math.ceil(Math.max.apply(null, vals) / 5) * 5;
    function Y(v) { return top + (hi - v) / (hi - lo) * (H - top - bottom); }
    var svg = el('svg', {viewBox: '0 0 ' + Wd + ' ' + H, role: 'img', 'aria-label': '回ごとの係数 E_v と 95% 区間。灰色の帯は全温度の 95% 区間。表と同じ値'}, host);
    for (var t = lo; t <= hi; t += 5) {
      el('line', {x1: left, x2: Wd - 8, y1: Y(t), y2: Y(t), class: t === 0 ? 'axis-zero' : 'grid'}, svg);
      el('text', {x: left - 8, y: Y(t) + 4, class: 'tick', 'text-anchor': 'end'}, svg, t < 0 ? '−' + Math.abs(t) : String(t));
    }
    var my = (top + H - bottom) / 2;
    el('text', {x: 14, y: my, class: 'axis-title', 'text-anchor': 'middle', transform: 'rotate(-90 14 ' + my + ')'}, svg, 'E_v（kJ/mol）');
    var bands = [], pts = [];
    DATA.materials.forEach(function (m, i) {
      var px = left + 6 + i * (pw + gap);
      el('text', {x: px + pw / 2, y: top - 18, class: 'row-head', 'text-anchor': 'middle'}, svg, m.name);
      var a = m.all.theta.v;
      var band = el('rect', {x: px, y: Y(a[2]), width: pw, height: Math.max(2, Y(a[1]) - Y(a[2])), class: 'tband fade pending'}, svg);
      bands.push(band);
      var cats = m.folds.concat([m.all]), step = pw / cats.length;
      cats.forEach(function (f, k) {
        var isAll = (k === cats.length - 1), cx = px + step * (k + 0.5), th = f.theta;
        var out = !isAll && !f.inBand;
        el('text', {x: cx, y: H - bottom + 18, class: 'tick', 'text-anchor': 'middle'}, svg, isAll ? '全温度' : f.T + '℃');
        el('text', {x: cx, y: H - bottom + 33, class: 'tick', 'text-anchor': 'middle'}, svg, isAll ? '（参考）' : 'を除く');
        var g0 = el('g', {class: 'tpt pending' + (out ? ' out' : '') + (isAll ? ' isall' : '')}, svg);
        el('line', {x1: cx, x2: cx, y1: Y(th.v[2]), y2: Y(th.v[1]), class: 'whisker'}, g0);
        el('line', {x1: cx - 6, x2: cx + 6, y1: Y(th.v[2]), y2: Y(th.v[2]), class: 'whisker'}, g0);
        el('line', {x1: cx - 6, x2: cx + 6, y1: Y(th.v[1]), y2: Y(th.v[1]), class: 'whisker'}, g0);
        el('circle', {cx: cx, cy: Y(th.v[0]), r: 6.5, class: 'tdot'}, g0);
        if (out) { el('text', {x: cx + 10, y: Y(th.v[0]) + 4, class: 'out-label'}, g0, '帯の外'); }
        pts.push(g0);
        var hit = el('rect', {x: cx - step / 2, y: top - 4, width: step, height: H - top - bottom + 4, class: 'hit'}, svg);
        bindTip(hit, function () {
          var s = '<b>' + m.name + '：' + (isAll ? '全温度（参考）' : f.T + ' ℃ を除いた回') + '</b><br>E_v ＝ <b>' + g(th.v[0]) + '</b>［' + g(th.v[1]) + '〜' + g(th.v[2]) + '］kJ/mol';
          if (th.h) { s += '<br>E_h ＝ ' + g(th.h[0]) + '［' + g(th.h[1]) + '〜' + g(th.h[2]) + '］kJ/mol'; }
          s += '<br>ΔLML ' + th.dLML.toFixed(2) + '（基準 ' + th.lim + '）：' + (th.sup ? '尤度で支持' : '支持されない');
          if (!isAll) { s += '<br>' + (f.inBand ? '全温度の区間に入る' : '<b>全温度の区間の外</b>'); }
          return s;
        });
      });
    });
    el('text', {x: left, y: H - 8, class: 'note-small'}, svg, '点と縦線：その回の最良値と 95% 区間。白抜きの点：全温度で学習したときの値。GPPS の E_h は表に。');
    onVisible(svg, function () {
      bands.forEach(function (b) { b.classList.remove('pending'); });
      pts.forEach(function (p, k) { later(function () { p.classList.remove('pending'); }, 350 + k * 140); });
    });
  })();

  // ---- 原本の図のタブ
  Array.prototype.forEach.call(document.querySelectorAll('[data-figtabs]'), function (box) {
    var btns = box.querySelectorAll('[role="tab"]'), panels = box.querySelectorAll('.tabpanel');
    Array.prototype.forEach.call(btns, function (b, k) {
      b.addEventListener('click', function () {
        Array.prototype.forEach.call(btns, function (x, j) { x.setAttribute('aria-selected', j === k ? 'true' : 'false'); });
        Array.prototype.forEach.call(panels, function (p, j) { p.classList.toggle('active', j === k); });
      });
    });
  });
})();
</script>
</body>
</html>
"""

method_tabs = "".join(f'<button type="button" role="tab" data-mat="{i}" aria-selected="{"true" if i == 0 else "false"}">{esc(m["name"])}</button>'
                      for i, m in enumerate(MATS))
runinfo = (f"{RUN_DATE} に、温度ホールドアウト検証のノート（材料ごとの複製 <code>PP_</code>・<code>MIXPP_</code>・<code>GPPS_温度ホールドアウト検証_…ipynb</code>）を、"
           f"3 材料の最新の学習結果フォルダ（PP 3 回目・粉砕材混合 PP 5 回目・GPPS 3 回目。線形項あり・通常版・全ショット版）で本番の設定で実行した結果（コミット <code>{COMMIT}</code>）から作りました。")
foot = (f"作成：{RUN_DATE}。生成スクリプト <code>04_補助スクリプト/build_report_mat_loto.py</code> が、上の結果の CSV から数値を読んで作りました。"
        "結果を実行し直したら、このファイルと引用した図と表を別の場所へ移してから、スクリプトを実行し直してください（同じ名前のファイルがあると止まります）。")
body = {"RUNINFO": runinfo, "SUMMARY": summary, "CARDS": "".join(card_html(c) for c in CARDS), "METHOD_TABS": method_tabs,
        "TAB5": tab5(), "TABTH": tab_theta(), "TAB3": tab3(), "FIG_RMSE": fig_tabs(FIGS[0], "除いた温度の予測誤差（回ごと）", "figrmse"),
        "FIG_THETA": fig_tabs(FIGS[1], "回ごとのシフトファクターの係数", "figtheta"), "COPYDIR": COPY_DIR, "TABFILES": tab_files(), "FOOT": foot,
        "DATA": json.dumps(DATA, ensure_ascii=False).replace("</", "<\\/")}
page = PAGE
for k, v in body.items():
    page = page.replace(f"@@{k}@@", v)
assert "@@" not in page.split('<script type="application/json"')[0], "置き換えていない項目がある"

# ---------------------------------------------------------------- 書く（コピー → HTML）
for mat, n in copies:
    dst = os.path.join(OUT_DIR, COPY_DIR, MAT_DIR[mat], n)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src(mat, n), dst)
    assert filecmp.cmp(src(mat, n), dst, shallow=False), dst
with open(out_html, "x", encoding="utf-8", newline="\n") as fh:
    fh.write(page)

# HTML の中の、このフォルダの中のファイルへの参照（画像・リンク）が、すべてあるか
import re  # noqa: E402
refs = sorted(set(re.findall(r'(?:src|href)="([^"#:]+)"', page)))
lost = [r for r in refs if not os.path.exists(os.path.join(OUT_DIR, urllib.parse.unquote(r)))]
assert not lost, f"参照先がない: {lost}"
print(f"作成：{rel_repo(out_html)}（{len(page):,} 文字）")
print(f"コピー：{len(copies)} 個（原本と同じ中身を確かめた）→ {rel_repo(os.path.join(OUT_DIR, COPY_DIR))}/")
print(f"HTML の中の参照：{len(refs)} 個、すべてある")
for m in MATS:
    print(m["name"], m["shift"], m["verdict"], "比", [round(f["ratio"]["shift"], 3) for f in m["folds"]],
          "E_v", [f["theta"]["v"][0] for f in m["folds"]], "全温度", m["all"]["theta"]["v"])
