# -*- coding: utf-8 -*-
"""説明書（results/学習結果説明_手順*.pdf と .tex）を作る生成スクリプト（build_report_step*.py）に共通の部品。

- 数値の整形（負の数は数式モードにして、正しいマイナス記号にする）
- ひな形 .tex の @@項目@@ を置き換え、_tmp/ で LuaLaTeX を2回動かし、できた PDF と tex を results/ に置く
"""
import datetime
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)                              # リポジトリ直下
R = os.path.join(W, "results")
TEX_BIN = r"C:\texlive\2026\bin\windows"
LUALATEX = shutil.which("lualatex") or os.path.join(TEX_BIN, "lualatex.exe")
MAT = {"PP": "PP", "MIX": "粉砕材混合PP"}
MAT_JA = {"PP": "PP", "MIX": "粉砕材混合 PP"}


def rd(step_dir, name):
    return pd.read_csv(os.path.join(R, step_dir, name), encoding="utf-8-sig")


def num(x, nd):
    """数値を文字列に。負の数は数式モードにして正しいマイナス記号にする"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    s = f"{abs(x):,.{nd}f}"
    return f"$-{s}$" if x < 0 and float(s.replace(",", "")) != 0 else s


def rmse(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    return "$<$0.001" if x < 0.0005 else f"{x:.3f}"


def pct(x, nd=0):
    return f"{x * 100:.{nd}f}\\%"


def sci(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def today_ja():
    t = datetime.date.today()
    return f"{t.year}年{t.month}月{t.day}日"


def build_pdf(step, step_dir, template, V, out_stem, extra_files=(), pdf_name=None):
    """ひな形の @@KEY@@ を V の値で置き換え、LuaLaTeX を2回動かして、results/学習結果説明_手順<step>.pdf と .tex を作る。
    step：'3b' など。step_dir：図のフォルダ（results の下。図がなければ None）。template：ひな形のファイル名（04_補助スクリプト）。out_stem：_tmp/ の下の作業名。
    pdf_name：results に置くファイル名（拡張子なし）。None なら 学習結果説明_手順<step>"""
    pdf_name = pdf_name or f"学習結果説明_手順{step}"
    tex = open(os.path.join(HERE, template), encoding="utf-8").read()
    V = {"DATE": today_ja(), **V}
    for key, val in V.items():
        tex = tex.replace(f"@@{key}@@", str(val))
    left = sorted(set(re.findall(r"@@([A-Za-z0-9_]+)@@", tex)))
    assert not left, f"置き換えていない項目: {left}"
    out = os.path.join(W, "_tmp", f"report_{step}")
    os.makedirs(os.path.join(out, step_dir or ""), exist_ok=True)
    with open(os.path.join(out, f"{out_stem}.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write(tex)
    if step_dir:
        src = os.path.join(R, step_dir)
        for fn in os.listdir(src):
            if fn.endswith(".png"):
                shutil.copy2(os.path.join(src, fn), os.path.join(out, step_dir, fn))
    print("原稿を作成:", os.path.join(out, f"{out_stem}.tex"), f"（置き換え {len(V)} 項目）")
    for ext in ("aux", "toc", "out"):                      # 前回の補助ファイルが残っていると、直した原稿でも前回のエラーを引きずる
        p = os.path.join(out, f"{out_stem}.{ext}")
        if os.path.exists(p):
            os.remove(p)
    for i in range(2):
        r = subprocess.run([LUALATEX, "-interaction=nonstopmode", "-halt-on-error", f"{out_stem}.tex"], cwd=out, capture_output=True)
        if r.returncode != 0:
            print(r.stdout.decode("utf-8", "replace")[-3000:])
            sys.exit(f"lualatex が失敗しました（{i + 1} 回目）")
    shutil.copy2(os.path.join(out, f"{out_stem}.pdf"), os.path.join(R, f"{pdf_name}.pdf"))
    shutil.copy2(os.path.join(out, f"{out_stem}.tex"), os.path.join(R, f"{pdf_name}.tex"))
    print("作成:", os.path.join(R, f"{pdf_name}.pdf"), "と .tex")
