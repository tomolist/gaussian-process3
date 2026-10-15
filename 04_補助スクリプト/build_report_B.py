# -*- coding: utf-8 -*-
"""報告書B（05_…/学習結果報告書作成_261015/報告書B_完成版_2枚.tex・報告書B_完成版_3枚.tex）を LuaLaTeX で 2 回ずつ組み、
PDF を同じフォルダに置く。補助ファイル（aux・log）は _tmp/report_B/build に置く。
使い方（リポジトリ直下で）: uv run python 04_補助スクリプト/build_report_B.py [原稿の名前 ...] [--figs]
  原稿の名前を省くと、2 枚版と 3 枚版の両方を組む（例：「報告書B_完成版_3枚」だけを指定することもできる）。
  --figs を付けると、先に検証の図とマスターカーブの図を作り直す（make_report_B_figs.py。図1〜5 の切り出しは、図がないときだけ）。
最後に、原稿ごとにページ数と、ログの注意（はみ出し・字形のない文字・未定義の参照）の件数を表示する。"""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC_DIR = os.path.join(REPO, "05_材料ごとの機械学習データと予測検証データ", "学習結果報告書作成_261015")
STEMS = ["報告書B_完成版_2枚", "報告書B_完成版_3枚"]
BUILD = os.path.join(REPO, "_tmp", "report_B", "build")


def build(lualatex, stem):
    for ext in ("aux", "log", "pdf"):
        p = os.path.join(BUILD, f"{stem}.{ext}")
        if os.path.exists(p):
            os.remove(p)
    for i in range(2):
        r = subprocess.run([lualatex, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={BUILD}", f"{stem}.tex"],
                           cwd=SRC_DIR, capture_output=True)
        if r.returncode != 0:
            print(r.stdout.decode("utf-8", "replace")[-4000:])
            sys.exit(f"lualatex が失敗しました（{stem}、{i + 1} 回目）。ログ：{os.path.join(BUILD, stem + '.log')}")
    shutil.copy2(os.path.join(BUILD, f"{stem}.pdf"), os.path.join(SRC_DIR, f"{stem}.pdf"))
    log = open(os.path.join(BUILD, f"{stem}.log"), encoding="utf-8", errors="replace").read()
    pages = re.search(r"Output written on .*?\((\d+) pages?", log, re.S)
    print("PDF：", os.path.join(SRC_DIR, f"{stem}.pdf"), f"（{pages.group(1) if pages else '?'} ページ）")
    for name, pat in [("Overfull（はみ出し）", r"Overfull \\[hv]box"), ("Underfull", r"Underfull \\[hv]box"),
                      ("字形のない文字", r"Missing character"), ("未定義の参照", r"(?:Reference|Citation) .*? undefined"), ("Float too large", r"Float too large")]:
        hits = re.findall(pat + r".*", log)
        print(f"  {name}: {len(hits)} 件")
        for h in hits[:6]:
            print("    ", h[:160])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--figs" in sys.argv:
        subprocess.run([sys.executable, os.path.join(HERE, "make_report_B_figs.py")], cwd=REPO, check=True)
    lualatex = shutil.which("lualatex")
    assert lualatex, "lualatex が見つからない（TeX Live を入れる）"
    os.makedirs(BUILD, exist_ok=True)
    for stem in (args or STEMS):
        build(lualatex, stem.removesuffix(".tex"))


if __name__ == "__main__":
    main()
