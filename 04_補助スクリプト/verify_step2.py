# -*- coding: utf-8 -*-
"""実行済みの手順2ノートを検査し、テキスト出力を表示する (画像は件数だけ)。

使い方: python verify_step2.py <ノートのパス>
"""
import json
import os
import sys

path = sys.argv[1]
nb = json.load(open(path, encoding="utf-8"))
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
errors = [c["id"] for c in code_cells for o in c["outputs"] if o["output_type"] == "error"]
not_run = [c["id"] for c in code_cells if c.get("execution_count") is None]
counts = [c.get("execution_count") for c in code_cells]
print(f"ファイル: {os.path.basename(path)}  {os.path.getsize(path) / 1e6:.2f} MB")
print(f"セル {len(nb['cells'])}（コード {len(code_cells)}）、未実行 {len(not_run)}、エラー {len(errors)}、実行番号 {counts}")
print(f"実行番号が 1 から連番: {counts == list(range(1, len(code_cells) + 1))}")
print(f"kernelspec: {nb['metadata']['kernelspec']}")
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    head = "".join(c["source"]).splitlines()[0]
    n_img = sum(1 for o in c["outputs"] if "data" in o and "image/png" in o["data"])
    print(f"\n===== [{i}] {head}  (画像 {n_img})")
    for o in c["outputs"]:
        if o["output_type"] == "stream":
            txt = "".join(o["text"])
            if o["name"] == "stderr":
                txt = "\n".join("  [stderr] " + l for l in txt.splitlines()[:6])
            print(txt.rstrip())
        elif o["output_type"] == "execute_result":
            print("".join(o["data"].get("text/plain", [])).rstrip())
        elif o["output_type"] == "error":
            print("ERROR:", o["ename"], o["evalue"])
