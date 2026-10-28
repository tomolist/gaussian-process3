# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR のノート（01_機械学習コードフォルダ/材料別_温度シフトファクター検討/ の下の 5 つ）の説明書を作る。

使い方（リポジトリ直下から）: uv run python 04_補助スクリプト/build_mat_nb_manual.py
report_mat_nb_manual_template.tex の @@項目@@ に、mat_common の設定（学習ファイルの列、元ファイル名の形式、結果のフォルダの名前、材料）と、
05 フォルダの今のデータの概要を入れて、01_機械学習コードフォルダ/材料別_温度シフトファクター検討/説明書_材料別温度シフトGPRのノート.tex を書き、
そのフォルダで LuaLaTeX を 2 回動かす（補助ファイルは _tmp/report_mat_nb_manual/ に出す）。学習の結果は使わないので、ノートを実行していなくても作れる。
2026-10-28（ユーザー依頼）：このフォルダを初めて開いた人向けに、ノートの目的・調べること・使えるデータ・使い方を 1 冊にまとめた。
列・フォルダの名前・材料をコードから取るので、mat_common.py を直したときは、これを実行し直す。
"""
import os
import re
import shutil
import subprocess
import sys

import report_common as rc

os.environ["NB_REPO"] = rc.W                         # mat_common が 05 フォルダを探す場所（リポジトリ直下）
import mat_common as mc

NB_ROOT = os.path.join(rc.W, "01_機械学習コードフォルダ", "材料別_温度シフトファクター検討")
DIR_TRAIN, DIR_STUDY = "材料別_温度シフト機械学習コード", "材料別_温度シフトファクター検討用コード"
NB = {"TRAIN": (DIR_TRAIN, "材料別_温度シフトGPR_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb"),
      "STUDY": (DIR_STUDY, "材料別_温度シフトファクターの検討_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb"),
      "THICK": (DIR_STUDY, "材料別_温度シフトGPR_ARD対応Matern52＋線形＋ホワイトノイズ_厚みごとの比較.ipynb")}
STEM = "説明書_材料別温度シフトGPRのノート"
COL_INFO = {      # 学習ファイルの列（mat_common.COLS の記号）→ (説明, 単位)
    "file": ("1 ショットの計測ファイルの名前（形式は 4.4 節）", "―"),
    "H": ("キャビティ厚み $H$（モデルの入力には使わない。条件の区別・外れ値の検知・厚みごとの比較に使う）", "mm"),
    "v": ("射出速度（条件の区別、検証の低速側・高速側の分け方に使う）", "mm/s"),
    "eta": ("見かけの樹脂粘度 $\\eta$（\\textbf{目的変数}）", "Pa·s"),
    "gd": ("見かけのせん断速度 $\\gd$（説明変数）", "1/s"),
    "P31": ("流動中樹脂圧力 P3\\_1（モデルには使わない。P3\\_3 との相関を見るだけ）", "MPa"),
    "P32": ("流動中樹脂圧力 P3\\_2（同上）", "MPa"),
    "P33": ("流動中樹脂圧力 \\PIII（説明変数）", "MPa"),
    "Tset": ("加熱シリンダ温度 $\\Tset$（説明変数）。2 通りの書き方のどちらか。列がなければ元ファイル名から取る", "℃"),
    "Tff": ("流動中樹脂温度 $\\Tff$（説明変数。型内樹脂温度）", "℃"),
}
V = {}


def esc(s):
    return str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


for key, (folder, name) in NB.items():                 # ノートの場所が変わったら、ここで気づく
    assert os.path.isfile(os.path.join(NB_ROOT, folder, name)), f"ノートが見つからない：{folder}/{name}"
    V[f"NB_{key}"] = name
V["DIR_TRAIN"], V["DIR_STUDY"] = DIR_TRAIN, DIR_STUDY

# ---- 学習ファイルの列（mat_common.COLS・REQUIRED から）
assert set(mc.COLS.values()) == set(COL_INFO), f"mat_common.COLS の列と、この説明の列が違う：{set(mc.COLS.values()) ^ set(COL_INFO)}"
rows = []
for col, sym in mc.COLS.items():
    need = "必須" if sym in mc.REQUIRED else ("どちらか（なくても可）" if sym == "Tset" else "任意")
    rows.append(f"\\file{{{col}}} & {need} & {COL_INFO[sym][1]} & {COL_INFO[sym][0]} \\\\")
V["TABLE_COLS"] = "\n".join(rows)
V["RE_SRC"] = mc.RE_SRC

# ---- 材料（mat_common.MAT_NAMES・MAT_FOLDERS・DEFAULT_SHIFT_CFG から）
rows = []
for mat, name in mc.MAT_NAMES.items():
    cfg = mc.DEFAULT_SHIFT_CFG[mat]
    form = "アレニウス型（係数 $\\Eh$・$\\Ev$、kJ/mol）" if cfg["type"] == "arrhenius" else f"WLF 型（係数 $A_{{1,h}}$・$A_{{1,v}}$。$A_2={cfg['A2']:g}$ K、$D_2=T_g$〔仮置き {cfg['Tg']:g} ℃〕を固定）"
    rows.append(f"\\file{{{mat}}} & {esc(name)} & \\file{{{mc.MAT_FOLDERS[mat]}}} & {form} & {cfg['Tref']:g} ℃ \\\\")
V["TABLE_MATS"] = "\n".join(rows)

# ---- モデルのフォルダの名前（mat_common.MODEL_DIRS・MODEL_JA から）
V["TABLE_MODEL_DIRS"] = "\n".join(f"{esc(k)} & \\file{{{d}}} & {esc(mc.MODEL_JA[k].split('（', 1)[1].rstrip('）'))} \\\\" for k, d in mc.MODEL_DIRS.items())
V["V_SHOT"], V["V_COND"] = mc.VERSION_DIRS["shot"], mc.VERSION_DIRS["cond"]
V["THICK_TOTAL"], V["THICK_EACH"], V["STUDY_DIR"] = mc.THICK_TOTAL, mc.THICK_EACH, mc.STUDY_DIR
for k in ["B_Tset", "B_Tff", "C"]:
    V[f"DIR_{k.upper()}"] = mc.MODEL_DIRS[k]

# ---- 05 フォルダの今のデータ（問題データ一覧の「除外」を除いた後）
mc.load_all(list(mc.MAT_NAMES))
rows = []
for (mat, grp), f in mc.FILE_INFO.items():
    rows.append(f"{esc(f['材料'])} & {'学習（5data）' if grp == '5data' else '検証（3data）'} & {f['使う行数']}（{f['条件数']}） & {f['ショット/条件']} & "
                f"{f['設定温度']} & {f['厚み']} & {f['射出速度']} \\\\")
V["TABLE_DATA"] = "\n".join(rows)

# ---- 原稿を書いて、LuaLaTeX を 2 回
tex = open(os.path.join(rc.HERE, "report_mat_nb_manual_template.tex"), encoding="utf-8").read()
V = {"DATE": rc.today_ja(), **V}
for key, val in V.items():
    tex = tex.replace(f"@@{key}@@", str(val))
left = sorted(set(re.findall(r"@@([A-Za-z0-9_]+)@@", tex)))
assert not left, f"置き換えていない項目: {left}"
with open(os.path.join(NB_ROOT, f"{STEM}.tex"), "w", encoding="utf-8", newline="\n") as f:
    f.write(tex)
out = os.path.join(rc.W, "_tmp", "report_mat_nb_manual")
os.makedirs(out, exist_ok=True)
for ext in ("aux", "toc", "out"):                      # 前回の補助ファイルが残っていると、直した原稿でも前回のエラーを引きずる
    p = os.path.join(out, f"{STEM}.{ext}")
    if os.path.exists(p):
        os.remove(p)
for i in range(2):
    r = subprocess.run([rc.LUALATEX, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={out}", f"{STEM}.tex"], cwd=NB_ROOT, capture_output=True)
    if r.returncode != 0:
        print(r.stdout.decode("utf-8", "replace")[-3000:])
        sys.exit(f"lualatex が失敗しました（{i + 1} 回目）")
shutil.copy2(os.path.join(out, f"{STEM}.pdf"), os.path.join(NB_ROOT, f"{STEM}.pdf"))
print("作成:", os.path.join(NB_ROOT, f"{STEM}.pdf"), "と .tex")
