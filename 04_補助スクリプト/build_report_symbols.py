# -*- coding: utf-8 -*-
"""記号表（すべての説明書で使う記号・モデル名・列名の一覧）を作る。

使い方（リポジトリ直下から）: uv run python 04_補助スクリプト/build_report_symbols.py
report_symbols_template.tex から、_tmp/report_symbols で LuaLaTeX を2回動かし、できた PDF と tex を
results/学習結果説明_記号表.pdf・.tex に置く（数値は入らない。新しい記号・モデル名・列名を使ったら、ひな形に足す）。
"""
import report_common as rc

rc.build_pdf("symbols", None, "report_symbols_template.tex", {}, "gakushu_symbols", pdf_name="学習結果説明_記号表")
