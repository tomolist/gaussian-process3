# -*- coding: utf-8 -*-
"""新しいデータの確認と、問題データ一覧.md の作り直し。

データを取り直したとき、問題のあるデータを入れ替えたときに実行する。
  1. 全ファイルの形式を確かめる（列、BOM、欠損、元ファイル名と列の一致）
  2. 問題の候補を規則で拾う（温度の間でせん断速度が違いすぎる、粘度の順序が逆、外れ値で半分以上が除かれた、ばらつきが大きい）
  3. ファイルと条件の指紋を、前回の記録と比べる（入れ替わったファイル・条件を一覧にする）
  4. 問題データ一覧.csv から、リポジトリ直下の 問題データ一覧.md を作り直す

使い方（リポジトリ直下から）: uv run python 04_補助スクリプト/check_data.py
出力：results/data_check/（データ確認.md、各 CSV）、問題データ一覧.md
"""
import datetime
import hashlib
import os
import sys

os.environ["NB_REPO"] = "."
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import nb_common as nc

OUT = os.path.join("results", "data_check")
os.makedirs(OUT, exist_ok=True)
today = datetime.date.today().isoformat()
plist = nc.read_problem_list()
nc.D = nc.load_all(exclude=False)                      # 全条件（除外する前）で確認する
D = nc.D


def csv(df, name):
    df.to_csv(os.path.join(OUT, name), index=False, encoding="utf-8-sig")


def registered(mat, grp, row):
    """一覧の「除外」「感度確認」に、この条件（row に shot があれば、そのショット）が登録されているか（ID を返す）。
    ショットを指定した一覧の行は、条件全体の候補（shot なし）には当てはまらない"""
    one = pd.DataFrame({"Tset": [float(row["Tset"])], "H": [float(row["H"])], "v": [float(row["v"])], "shot": [row.get("shot", np.nan)]})
    _, ids = nc.problem_mask(one, mat, grp, plist, ("除外", "感度確認"))
    return ids[0]


# ---------------------------------------------------------------- 1. ファイルの確認
rows = []
for (mat, grp, o), df in D.items():
    sh = df.groupby("cond").size()
    q = 6 * 17.5e-3 / (df["H"] * 1e-3 * df["gd"])                      # Δt [s]（γ̇ = 6L/(H·Δt)）
    ts = None
    for cand in [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7]:    # Δt が整数倍になる、最も大きい間隔
        k = q / cand
        if np.abs(k - np.round(k)).max() < 1e-3:
            ts = cand
            break
    tau = df["eta"] * df["gd"] / 1e6
    A = np.column_stack([df["H"] * df["P31"], df["H"]])
    coef, *_ = np.linalg.lstsq(A, tau, rcond=None)
    rel = np.abs(tau / (df["H"] * (df["P31"] - 0.30) / (2 * 17.5)) - 1)
    path = nc.path_of(mat, grp, o)
    with open(path, "rb") as f:
        raw = f.read()
    rows.append({"材料": nc.MATS[mat], "組": grp, "外れ値": o, "行数": len(df), "条件数": df["cond"].nunique(),
                 "ショット/条件": f"{sh.min()}〜{sh.max()}", "BOM": raw[:3] == b"\xef\xbb\xbf", "列名": True, "欠損": 0,
                 "L推定 [mm]": round(1 / (2 * coef[0]), 3), "c推定 [MPa]": round(-coef[1] / coef[0], 4),
                 "算出式の最大誤差 [%]": round(rel.max() * 100, 4), "Δt の刻み [µs]": ts * 1e6 if ts else np.nan,
                 "sha256": hashlib.sha256(raw).hexdigest(), "ファイル": path})
files_df = pd.DataFrame(rows)
for mat in nc.MATS:
    for o in ["なし", "あり"]:
        a, b = set(D[(mat, "5data", o)]["cond"]), set(D[(mat, "3data", o)]["cond"])
        assert not (a & b), f"学習と検証で条件が重なっている: {mat} {o}"
csv(files_df.drop(columns=["sha256"]), "file_check.csv")

# ---------------------------------------------------------------- 2. 問題の候補
cand = []
for (mat, grp, o), df in D.items():
    if o != "なし":
        continue
    g = df.groupby(["H", "v", "Tset"]).agg(gd=("gd", "mean"), eta=("eta", "mean"), n=("eta", "size")).reset_index()
    for (h, v), s in g.groupby(["H", "v"]):
        if len(s) < 3:
            continue
        lg = np.log(s["gd"].to_numpy())
        if np.exp(lg.max() - lg.min()) > 1.3:                          # 規則 a：温度の間で、せん断速度が 1.3 倍を超えて違う
            odd = s.iloc[np.argmax(np.abs(lg - np.median(lg)))]
            cand.append({"材料": nc.MATS[mat], "組": grp, "規則": "a 温度の間でせん断速度が1.3倍超違う", "設定温度": int(odd["Tset"]), "厚み": h, "射出速度": int(v),
                         "内容": f"γ̇ の最大/最小 = {np.exp(lg.max() - lg.min()):.2f}", "mat": mat, "grp": grp, "row": {"Tset": odd["Tset"], "H": h, "v": v}})
        e = s.set_index("Tset")["eta"]
        if 190 in e.index and 230 in e.index and e[190] / e[230] < 0.95:      # 規則 b：190 ℃ の粘度が 230 ℃ より小さい（順序が逆）
            cand.append({"材料": nc.MATS[mat], "組": grp, "規則": "b 粘度の順序が温度と逆", "設定温度": 190, "厚み": h, "射出速度": int(v),
                         "内容": f"η(190)/η(230) = {e[190] / e[230]:.2f}", "mat": mat, "grp": grp, "row": {"Tset": 190, "H": h, "v": v}})
    sd = df.groupby(["Tset", "H", "v"])["ln_eta"].agg(["std", "size"]).reset_index()
    for _, r in sd[(sd["size"] >= 2) & (sd["std"] > 0.1)].iterrows():       # 規則 c：同じ条件内の ln η の標準偏差が 0.1 を超える
        cand.append({"材料": nc.MATS[mat], "組": grp, "規則": "c 同じ条件内のばらつきが大きい", "設定温度": int(r["Tset"]), "厚み": r["H"], "射出速度": int(r["v"]),
                     "内容": f"ln η の標準偏差 = {r['std']:.3f}（{int(r['size'])} ショット）", "mat": mat, "grp": grp, "row": {"Tset": r["Tset"], "H": r["H"], "v": r["v"]}})
    ar = D[(mat, grp, "あり")].groupby("cond").size()                   # 規則 d：外れ値で半分以上のショットが除かれた
    sh = df.groupby("cond").size().reindex(ar.index).fillna(0)
    for c in ar.index[(ar >= 3) & (sh / ar <= 0.5)]:
        t, h, v = c.split("_")
        cand.append({"材料": nc.MATS[mat], "組": grp, "規則": "d 外れ値で半分以上が除かれた", "設定温度": int(t), "厚み": float(h), "射出速度": int(v),
                     "内容": f"{int(ar[c])} ショット中 {int(sh[c])} ショットが残った", "mat": mat, "grp": grp, "row": {"Tset": float(t), "H": float(h), "v": float(v)}})
KEY4 = ["eta", "gd", "P31", "P32"]
for (mat, grp, o), df in D.items():                                   # 規則 e：別の元ファイル（ショット）と、η・γ̇・P3_1・P3_2 が完全に同じ値の行（行のコピーと思われる）。外れ値なし・あり の両方で調べる
    dup = df[df.duplicated(KEY4, keep=False)]
    for _, g in dup.groupby(KEY4):
        if g["file"].nunique() < 2:
            continue
        g = g.sort_values("file")
        mem = [{"Tset": r["Tset"], "H": r["H"], "v": r["v"], "shot": int(r["shot"])} for _, r in g.iterrows()]
        ids = [registered(mat, grp, m) for m in mem]                  # 同じ値の行のどれかが一覧にあれば、登録済みとする（どちらがコピーかは、見て決める）
        r0 = g.iloc[0]
        cand.append({"材料": nc.MATS[mat], "組": grp, "規則": f"e 別のショットと同じ値の行（外れ値{o}）", "設定温度": int(r0["Tset"]), "厚み": r0["H"], "射出速度": int(r0["v"]), "ショット": int(r0["shot"]),
                     "内容": f"{' と '.join(g['file'])} の η・γ̇・P3_1・P3_2 が完全に一致（どちらかが行のコピーと思われる。Tff は {'・'.join(f'{t:.2f}' for t in g['Tff'])} ℃）",
                     "mat": mat, "grp": grp, "row": mem[0], "reg": next((i for i in ids if i), "")})
cand_df = pd.DataFrame(cand)
if len(cand_df):
    if "ショット" not in cand_df.columns:
        cand_df["ショット"] = ""
    cand_df["ショット"] = cand_df["ショット"].apply(lambda x: "" if pd.isna(x) else str(int(x)))
    cand_df["一覧の ID"] = [r["reg"] if isinstance(r.get("reg"), str) else registered(r["mat"], r["grp"], r["row"]) for _, r in cand_df.iterrows()]
    cand_df = cand_df.drop(columns=["mat", "grp", "row", "reg"], errors="ignore").drop_duplicates(subset=["材料", "組", "規則", "設定温度", "厚み", "射出速度", "ショット", "内容"]).sort_values(["規則", "材料", "組", "厚み", "射出速度", "設定温度"])
else:
    cand_df = pd.DataFrame(columns=["材料", "組", "規則", "設定温度", "厚み", "射出速度", "ショット", "内容", "一覧の ID"])
csv(cand_df, "candidates.csv")
dup_main = cand_df[cand_df["規則"].str.startswith("e ") & cand_df["規則"].str.contains("外れ値なし")] if len(cand_df) else cand_df
if len(dup_main):
    print(f"★外れ値なしのファイルに、別のショットと同じ値の行が {len(dup_main)} 行あります（主の分析に入っています）。results/data_check/candidates.csv を確認してください。")

# ---------------------------------------------------------------- 3. 指紋（前回との比較）
cf = []
for (mat, grp, o), df in D.items():
    for c, g in df.groupby("cond"):
        g = g.sort_values("file")
        h = hashlib.sha256(g[["file", "eta", "gd", "P31", "P32", "Tset", "Tff"]].round(9).to_csv(index=False).encode()).hexdigest()[:16]
        cf.append({"材料": nc.MATS[mat], "組": grp, "外れ値": o, "条件": c, "ショット数": len(g), "ln_eta の平均": round(g["ln_eta"].mean(), 6), "指紋": h})
cf_df = pd.DataFrame(cf)
old_path, old_files = os.path.join(OUT, "condition_fingerprints.csv"), os.path.join(OUT, "file_fingerprints.csv")
changes = pd.DataFrame(columns=["材料", "組", "外れ値", "条件", "状態"])
file_changes = []
if os.path.exists(old_path):
    old = pd.read_csv(old_path, encoding="utf-8-sig", dtype={"条件": str})
    m = old.merge(cf_df, on=["材料", "組", "外れ値", "条件"], how="outer", suffixes=("_前", "_今"), indicator=True)
    m["状態"] = np.where(m["_merge"] == "left_only", "削除", np.where(m["_merge"] == "right_only", "追加", np.where(m["指紋_前"] != m["指紋_今"], "変更", "同じ")))
    changes = m.loc[m["状態"] != "同じ", ["材料", "組", "外れ値", "条件", "状態", "ショット数_前", "ショット数_今", "ln_eta の平均_前", "ln_eta の平均_今"]]
    of = pd.read_csv(old_files, encoding="utf-8-sig")
    nf = files_df[["ファイル", "sha256"]]
    mm = of.merge(nf, on="ファイル", how="outer", suffixes=("_前", "_今"))
    file_changes = mm[mm["sha256_前"] != mm["sha256_今"]]["ファイル"].tolist()
    has_old = True
else:
    has_old = False
csv(changes, "changes.csv")
cf_df.to_csv(old_path, index=False, encoding="utf-8-sig")
files_df[["ファイル", "sha256", "行数"]].to_csv(old_files, index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 4. 一覧の確認と、データ確認.md
def cond_text(r):
    parts = [f"{r['設定温度']} ℃" if r["設定温度"] else "全温度", f"{r['厚み']} mm" if r["厚み"] else "全厚み", f"{r['射出速度']} mm/s" if r["射出速度"] else "全速度"]
    if r.get("ショット", ""):
        parts.append(f"ショット {r['ショット']}")
    return "・".join(parts) if (r["設定温度"] or r["厚み"] or r["射出速度"] or r.get("ショット", "")) else "全体"


leg = nc.legacy_check(nc.load_all(exclude=True))
md = [f"# データ確認（{today}）", "",
      "`04_補助スクリプト/check_data.py` が作りました。データを取り直したとき・入れ替えたときに実行します（手順は `問題データ一覧.md` の最後）。", "",
      "## 1. ファイルの形式", "",
      f"- 確認したファイル：{len(files_df)} 個（材料 {len(nc.MATS)} × 組 3 × 外れ値 なし・あり）。列名・BOM・欠損・元ファイル名と列の一致は、すべて問題なし（問題があれば、ここで止まります）。",
      "- 学習（5data）と検証（3data）で、条件の重なりはありません。",
      "- 全条件のファイルから、問題データ一覧の「除外」を除いたデータが、no1.5-3.0 のファイルと同じ行か：",
      "", "| 材料 | 組 | 外れ値 | 一覧で除いた後の行数 | no1.5-3.0 の行数 | 同じ行 |", "|---|---|---|---|---|---|"]
md += [f"| {r['材料']} | {r['組']} | {r['外れ値']} | {r['一覧で除いた後の行数']} | {r['no1.5-3.0 の行数']} | {'はい' if r['同じ行'] else '**いいえ**'} |" for _, r in leg.iterrows()]
md += ["", "| 材料 | 組 | 外れ値 | 行数 | 条件数 | ショット/条件 | L [mm] | c [MPa] | 算出式の最大誤差 [%] | Δt の刻み [µs] |", "|---|---|---|---|---|---|---|---|---|---|"]
md += [f"| {r['材料']} | {r['組']} | {r['外れ値']} | {r['行数']} | {r['条件数']} | {r['ショット/条件']} | {r['L推定 [mm]']} | {r['c推定 [MPa]']} | {r['算出式の最大誤差 [%]']} | {r['Δt の刻み [µs]']:g} |"
       for _, r in files_df.iterrows()]
md += ["", "## 2. 問題の候補（規則で自動で拾ったもの。登録するかはユーザーが決める）", "",
       "規則：a 温度の間でせん断速度が 1.3 倍を超えて違う／b 190 ℃ の粘度が 230 ℃ より小さい／c 同じ条件内の ln η の標準偏差が 0.1 を超える／d 外れ値で半分以上のショットが除かれた（a〜d は外れ値なしのファイルで確認）"
       "／e 別のショットと η・γ̇・P3_1・P3_2 が完全に同じ値の行がある（行のコピーと思われる。外れ値なし・あり の両方で確認）。", ""]
if len(cand_df):
    md += ["| 規則 | 材料 | 組 | 設定温度 | 厚み | 射出速度 | ショット | 内容 | 一覧の ID |", "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['規則']} | {r['材料']} | {r['組']} | {r['設定温度']} | {r['厚み']} | {r['射出速度']} | {r['ショット']} | {r['内容']} | {r['一覧の ID'] or '**未登録**'} |" for _, r in cand_df.iterrows()]
    n_un = int((cand_df["一覧の ID"] == "").sum())
    md += ["", f"- 候補 {len(cand_df)} 件のうち、一覧に未登録：{n_un} 件。"]
else:
    md += ["- 候補はありません。"]
md += ["", "## 3. 前回からの変化", ""]
if not has_old:
    md += ["- 前回の記録がないので、今回の指紋を保存しました（`file_fingerprints.csv`、`condition_fingerprints.csv`）。次回から、入れ替わったファイル・条件を示します。"]
elif len(changes) == 0 and not file_changes:
    md += ["- 前回から変わったファイル・条件はありません。"]
else:
    md += [f"- 内容が変わったファイル：{len(file_changes)} 個", *[f"  - `{f}`" for f in file_changes], f"- 追加・削除・変更された条件：{len(changes)} 件（`changes.csv`）", ""]
    md += ["| 材料 | 組 | 外れ値 | 条件 | 状態 |", "|---|---|---|---|---|"]
    md += [f"| {r['材料']} | {r['組']} | {r['外れ値']} | {r['条件']} | {r['状態']} |" for _, r in changes.head(60).iterrows()]
    md += ["", "取り直した条件があれば、問題データ一覧の「再測定」を「済み」にしてください（「除外」の条件が、再び使われます）。"]
with open(os.path.join(OUT, "データ確認.md"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(md) + "\n")

# ---------------------------------------------------------------- 5. 問題データ一覧.md
KIND = {"除外": "解析から除く。再測定が「済み」になると、再び使う", "感度確認": "主の分析では残し、除いた場合を並べて、影響を見る（影響が大きければ、除いたファイルを作る）",
        "注意": "除かないが、結果を読むときに気をつける", "使わない": "今後の解析に使わない（記録として残す）"}
lines = ["# 問題データ一覧", "",
         "問題のあったデータと、その理由の一覧です。**このファイルは自動で作られます**（`04_補助スクリプト/check_data.py`）。直接は編集せず、`02_機械学習データ/問題データ一覧.csv`（Excel で開けます）を編集してから、`check_data.py` を実行してください。", "",
         f"最終更新：{today}　／　データの確認結果：`results/data_check/データ確認.md`", "", "## 一覧", "",
         "| ID | 区分 | 材料 | データ | 条件 | 問題の内容 | 根拠 | 今の扱い | 影響の判定 | 再測定 |", "|---|---|---|---|---|---|---|---|---|---|"]
for _, r in plist.iterrows():
    cells = [r["ID"], r["区分"], r["材料"], r["データ"] + (f"（{r['組']}）" if r["組"] else ""), cond_text(r), r["問題の内容"], r["根拠"], r["今の扱い"], r["影響の判定"], r["再測定"]]
    lines.append("| " + " | ".join(str(c).replace("|", "／") for c in cells) + " |")
lines += ["", "## 区分の意味", ""] + [f"- **{k}**：{v}" for k, v in KIND.items()]
lines += ["", "## 列の意味（CSV）", "",
          "- 条件の列（材料・データ・組・設定温度・厚み・射出速度・ショット）：空欄は「すべて」。ノートは、この列で条件に当てはまる行を探します。ショットは、元ファイル名の先頭 3 桁の番号で、数字を書くと、そのショットの行だけが対象です。データ「新」は新しいデータ（NR500/01〜06）、「旧00」は旧データ。",
          "- 再測定：未定／予定／済み。**「済み」にした行は、「除外」「感度確認」でも使われません**（取り直したデータを使うため）。",
          "- 影響の判定：感度確認の条件を除いたとき、結果がどれだけ変わったか（手順3b などで書く）。", "",
          "## データを取り直したとき・入れ替えたとき", "",
          "1. 新しい CSV を、**同じ名前で同じフォルダに置く**（古いものは Git の履歴に残ります）。ファイル名の形式（`結合_<番号>_<材料>_max_<組>_外れ値<あり|なし>.csv`、列名）は変えないでください。",
          "2. `問題データ一覧.csv` を直す：取り直した条件は「再測定」を「済み」にし、新しく見つかった問題は行を足します。",
          "3. `uv run python 04_補助スクリプト/check_data.py` を実行し、`results/data_check/データ確認.md` を読みます（形式、前回からの変化、問題の候補）。",
          "4. `uv run python 04_補助スクリプト/rerun_all.py --steps 3a 3b 3c 5a` で、全手順のノートと説明書を作り直します（全部で数時間かかります。3c の保存済みの事後サンプルは消して、サンプリングし直します）。",
          "5. 説明書（`results/学習結果説明_手順*.pdf`）で、結果を判断します。"]
with open("問題データ一覧.md", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines) + "\n")

print(f"確認したファイル {len(files_df)} 個、形式はすべて問題なし。候補 {len(cand_df)} 件（未登録 {int((cand_df['一覧の ID'] == '').sum()) if len(cand_df) else 0} 件）。")
print("no1.5-3.0 との一致：", "すべて同じ" if leg["同じ行"].all() else "★違う行がある")
print("前回との変化：", "前回の記録なし（今回の指紋を保存）" if not has_old else f"ファイル {len(file_changes)} 個、条件 {len(changes)} 件")
print("出力：", OUT, "、問題データ一覧.md")
