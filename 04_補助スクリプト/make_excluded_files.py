# -*- coding: utf-8 -*-
"""問題データ一覧の1つの行（ID）の条件だけを除いたデータファイルを、新しく作る（既存のファイルは変更しない）。

「感度確認」の条件の影響が大きいと分かったときに使う（手順3b のノートの判定）。
  例：uv run python 04_補助スクリプト/make_excluded_files.py --id P02 --tag no0.5-20-190
作るもの：各ファイルと同じフォルダの親に、「<番号>_<材料>_max_<組>(<tag>)」のフォルダを作り、
  外れ値あり／なし の結合ファイルを置く（番号は、同じ親フォルダの中の最大の番号 + 1）。
  行の文字列はそのまま残し、条件に当てはまる行だけを除く（BOM と改行は元のまま）。
  --dry-run を付けると、作らずに、作る予定のファイルだけを表示する。
注意：この一覧の他の「除外」（P01 など）は、ここでは除かない（ノートが、読むときに除く）。
"""
import argparse
import os
import re
import sys

os.environ["NB_REPO"] = "."
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nb_common as nc  # noqa: E402

RE_SRC = re.compile(r"^(\d{3})_(\d+)℃_(\d{3})_([\d.]+)mm\.csv$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="問題データ一覧の ID（例：P02）")
    ap.add_argument("--tag", required=True, help="フォルダ名とファイル名に付ける印（例：no0.5-20-190）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pl = nc.read_problem_list()
    row = pl[pl["ID"] == a.id]
    if row.empty:
        sys.exit(f"一覧に {a.id} がありません。")
    r = row.iloc[0]
    if r["データ"] not in ("新", ""):
        sys.exit("新しいデータの行だけ使えます。")
    for (mat, grp), (top, d1, d2, fn) in nc.FILES.items():
        if r["材料"] not in ("すべて", "", nc.MATS[mat]) or r["組"] not in ("", grp):
            continue
        parent = os.path.join(top, d1)
        nums = [int(m.group(1)) for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d)) and (m := re.match(r"^(\d\d)_", d))]
        num = max(max(nums, default=0) + 1, 6)
        stem = d2.split("_", 1)[1]                                # 例：pp_max_5data
        new_dir = f"{num:02d}_{stem}({a.tag})"
        made = 0
        for o in ["なし", "あり"]:
            src = nc.path_of(mat, grp, o)
            raw = open(src, "rb").read()
            lines = raw.split(b"\n")
            keep, removed = [], 0
            for ln in lines:
                m = RE_SRC.match(ln.decode("utf-8-sig", "replace").split(",")[0].strip())
                hit = False
                if m:
                    t, v, h, s = float(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(1))
                    hit = (r["設定温度"] == "" or t == float(r["設定温度"])) and (r["厚み"] == "" or h == float(r["厚み"])) and (r["射出速度"] == "" or v == float(r["射出速度"])) \
                        and (r.get("ショット", "") == "" or s == int(r["ショット"]))
                if hit:
                    removed += 1
                else:
                    keep.append(ln)
            if removed == 0:
                continue
            new_name = f"結合_{num:02d}_{stem}({a.tag})_外れ値{o}.csv"
            dst = os.path.join(parent, new_dir, f"外れ値{o}", new_name)
            print(f"{'（予定）' if a.dry_run else '作成'}：{dst}　元 {len(lines) - 1} 行 → {len(keep) - 1} 行（{removed} 行を除く）")
            if not a.dry_run:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "wb") as f:
                    f.write(b"\n".join(keep))
            made += 1
        if made == 0:
            print(f"（{nc.MATS[mat]} {grp}：該当する条件がないので、作りません）")
    print("\n次に：問題データ一覧.csv の「影響の判定」に結果を書き、check_data.py を実行してください。作ったファイルは、感度確認の記録です（主の分析は、今のファイルのままです）。")


if __name__ == "__main__":
    main()
