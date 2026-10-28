# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR のノート（学習用・シフトファクター検討用）に、Python 初心者向けの解説を 1 行ずつ付けた「解説付き」のノートを生成する（未実行）。

使い方: uv run python 04_補助スクリプト/build_mat_shift_nb_commented.py [学習|検討] [--force]
（学習・検討を書かなければ、両方を作る。既存ファイルがある場合は、--force を付けない限り上書きしない）

- 元のノート（01_機械学習コードフォルダ/材料別_温度シフトファクター検討/ の下の、材料別_温度シフト機械学習コード/ の 材料別_温度シフトGPR_….ipynb〔学習用〕と
  材料別_温度シフトファクター検討用コード/ の 材料別_温度シフトファクターの検討_….ipynb〔シフトファクター検討用。2026-10-27 に学習用から分けた〕）を読み、
  各コードセルの各行の上に、解説の行（「# ▶ 」で始まるコメント）を差し込む。コードそのものは変えない。解説付きの版は、元のノートと同じフォルダに置く。
- 解説は、学習用は 04_補助スクリプト/mat_shift_nb_解説/、シフトファクター検討用は mat_shiftfactor_nb_解説/ の codeNN.txt
  （NN はコードセルの順番。長いセルは codeNN_a.txt・codeNN_b.txt … に分けてある）に、「行番号 ⇒ 解説」の形で 1 行ずつ書いてある。
  行番号は、元のセルの中の行の番号（1 から）。「 ⏎ 」で区切ると、解説を複数行に分けられる。各ファイルの「@sha1」は、元のセルの中身の指紋。
  元のノートを作り直してセルの中身が変わったときは、ここでエラーになる（行番号がずれて、解説が別の行に付くのを防ぐため）。その場合は、解説ファイルを直す。
- シフトファクター検討用のノートで、解説のフォルダに同じ番号のファイルがないセルは、学習用の解説のフォルダから、@sha1 が同じセルの解説を使う
  （ライブラリのセルや共通の部品のセルのように、2 つのノートで中身が同じセル）。
- 解説を付けない行：空行、コメントだけの行、閉じかっこだけの行、複数行の文字列の 2 行目以降と、行末が \\ の次の行（コメントを挟めないため）。
- 差し込んだ後のセルと元のセルの構文木（ast）が同じことを確かめる（＝実行したときの動作は元のノートと同じ）。
"""
import ast
import glob
import hashlib
import io
import json
import os
import sys
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NB_DIR = os.path.join(REPO, "01_機械学習コードフォルダ", "材料別_温度シフトファクター検討")     # ユーザーがノートを移したフォルダ（2026-10-23）
NOTEBOOKS = {     # 名前 → (NB_DIR の下のフォルダ〔2026-10-28 のユーザー依頼で分けた〕, 元のノート, 解説のフォルダ, 解説が見つからないときに @sha1 で探すフォルダ)
    "学習": ("材料別_温度シフト機械学習コード", "材料別_温度シフトGPR_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb", "mat_shift_nb_解説", []),
    "検討": ("材料別_温度シフトファクター検討用コード", "材料別_温度シフトファクターの検討_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb", "mat_shiftfactor_nb_解説", ["mat_shift_nb_解説"]),
}
MARK = "# ▶ "

INTRO = r'''# 解説付きの版について

このノートは、`{name}`（元のノート）の**コードの各行に、Python 初心者向けの解説を付けた版**です。

- **`# ▶` で始まる行が、すぐ下の行の解説**です（灰色のコメントで、実行には影響しません）。もとからあるコメント（`#` の後ろ）は、元のノートの説明です。
- コードそのものは元のノートと同じです（生成するときに、構文が元のノートと同じことを確かめています）。**実行すると、元のノートと同じように、実行の始めにダイアログで選んだ学習結果フォルダに、同じ結果が保存されます（同じフォルダを選ぶと、上書きされます）。**
- 解説を付けていない行：空行、コメントだけの行、閉じかっこ `)`・`]`・`}` だけの行、複数行にまたがる文字列（`"""…"""` など）の 2 行目以降。
- このノートは、`04_補助スクリプト/build_mat_shift_nb_commented.py` で作っています。元のノートを作り直したときは、このスクリプトを実行し直してください（元のセルの中身が変わっていたら、エラーで知らせます。そのときは解説ファイル `04_補助スクリプト/{cdir}/` を直します）。

## Python の読み方の基本（このノートによく出る書き方）

| 書き方 | 意味 |
|---|---|
| `x = 5` | 変数 x に 5 を入れる（`=` は「等しい」ではなく「入れる」） |
| `# …` | コメント（メモ）。実行されない |
| `"文字"`・`'文字'` | 文字列（テキスト） |
| `f"{x} 行"` | f 文字列。`{}` の中の変数や式の値が、文字の中に埋め込まれる |
| `[1, 2, 3]` | リスト（順番のある入れ物）。`a[0]` で最初の要素（番号は 0 から） |
| `(1, 2)` | タプル（中身を変えられないリスト） |
| `{"PP": 1, "MIX": 2}` | 辞書（名前と値の組の入れ物）。`d["PP"]` で 1 が取れる |
| `import numpy as np` | 部品の集まり（ライブラリ）を読み込み、`np` という短い名前で使う |
| `def 名前(引数):` | 関数（処理のまとまり）を作る。字下げした行が中身。`return` で結果を返す |
| `名前(値)` | 関数を呼ぶ（実行する） |
| `for x in リスト:` | リストの要素を 1 つずつ x に入れて、字下げした行をくり返す |
| `if 条件:` ／ `elif` ／ `else:` | 条件が成り立つときだけ、字下げした行を実行する |
| `[式 for x in リスト]` | リスト内包表記。リストの各要素で式を計算して、新しいリストを作る |
| `lambda x: 式` | 名前のない小さな関数 |
| `a.b` | a の中の b（a が持つ値や関数） |
| `np.…` | NumPy（数値計算のライブラリ）の機能。配列（数の並び）をまとめて計算する |
| `pd.…`・`DataFrame` | pandas（表を扱うライブラリ）。`DataFrame` は Excel の表のようなもの。`df["列名"]` で列を取り出す |
| `plt.…`・`fig`・`ax` | matplotlib（グラフを描くライブラリ）。`fig` は図全体、`ax` はその中の 1 つのグラフ |
| `**kw` | 辞書 kw の中身を、名前付きの引数としてまとめて渡す |
| `a if 条件 else b` | 条件が成り立てば a、そうでなければ b |
| `with open(…) as f:` | ファイルを開き、字下げした行が終わったら自動で閉じる |
'''


def blocked_rows(src):
    """解説を前に入れられない行：複数行の文字列の 2 行目以降と、行末が \\ の次の行"""
    rows = set()
    fs = []
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        name = tokenize.tok_name[t.type]
        if name == "FSTRING_START":
            fs.append(t.start[0])
        elif name == "FSTRING_END":
            rows.update(range(fs.pop() + 1, t.end[0] + 1))
        elif t.type == tokenize.STRING and t.end[0] > t.start[0]:
            rows.update(range(t.start[0] + 1, t.end[0] + 1))
    for i, ln in enumerate(src.split("\n"), 1):
        if ln.rstrip().endswith("\\") and i not in rows:
            rows.add(i + 1)
    return rows


def commentable(src):
    """解説を付ける行の番号（1 から）"""
    blk = blocked_rows(src)
    out = []
    for i, ln in enumerate(src.split("\n"), 1):
        s = ln.strip()
        if not s or s.startswith("#") or i in blk or all(c in ")]}, " for c in s):
            continue
        out.append(i)
    return out


def sha1(src):
    return hashlib.sha1(src.encode("utf-8")).hexdigest()[:12]


def read_comments(paths):
    """解説ファイル（1 つのセルの解説を、codeNN_a.txt・codeNN_b.txt のように分けてもよい）→ (sha1, {行番号: [解説の行, …]})"""
    hs, com = set(), {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for k, raw in enumerate(f, 1):
                s = raw.rstrip("\n")
                if not s.strip() or s.startswith("#"):
                    continue
                if s.startswith("@sha1"):
                    hs.add(s.split()[1])
                    continue
                if "⇒" not in s:
                    raise SystemExit(f"{os.path.basename(path)} の {k} 行目：「行番号 ⇒ 解説」の形でない：{s}")
                no, text = s.split("⇒", 1)
                no = int(no.strip())
                if no in com:
                    raise SystemExit(f"{os.path.basename(path)} の {k} 行目：行番号 {no} が重複")
                com[no] = [t.strip() for t in text.split("⏎")]
    return (hs.pop() if len(hs) == 1 else None), com


def for_ast(src):
    """%matplotlib などの行（IPython の命令）は、構文の確認では pass に置き換える"""
    return "\n".join(("pass" if ln.lstrip().startswith("%") else ln) for ln in src.split("\n"))


def annotate(src, com):
    lines = src.split("\n")
    out = []
    for i, ln in enumerate(lines, 1):
        if i in com:
            ind = ln[: len(ln) - len(ln.lstrip())]
            out.extend(ind + MARK + c for c in com[i])
        out.append(ln)
    return "\n".join(out)


def to_lines(s):
    lines = s.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def by_sha1(dirs):
    """解説のフォルダの中の解説ファイルを、セルの指紋（@sha1）ごとにまとめる：{sha1: [ファイル, …]}（codeNN_a・codeNN_b … は 1 つのセルの分）"""
    out = {}
    for d in dirs:
        groups = {}
        for p in sorted(glob.glob(os.path.join(HERE, d, "code*.txt"))):
            groups.setdefault(os.path.basename(p)[:6], []).append(p)
        for paths in groups.values():
            h, _ = read_comments(paths)
            if h:
                out.setdefault(h, paths)
    return out


def build(folder, name, cdir_name, fallback):
    src_nb, out_nb = os.path.join(NB_DIR, folder, name), os.path.join(NB_DIR, folder, "解説付き_" + name)
    cdir = os.path.join(HERE, cdir_name)
    if os.path.exists(out_nb) and "--force" not in sys.argv[1:]:
        sys.exit(f"既にファイルがあるため上書きしません（上書きするときは --force）: {out_nb}")
    nb = json.load(open(src_nb, encoding="utf-8"))
    other = by_sha1(fallback)
    errors, k, n_com = [], 0, 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        k += 1
        src = "".join(c["source"])
        paths = sorted(glob.glob(os.path.join(cdir, f"code{k:02d}*.txt"))) or other.get(sha1(src), [])
        if not paths:
            errors.append(f"code{k:02d}：解説ファイルがない（{cdir}）")
            continue
        h, com = read_comments(paths)
        if h != sha1(src):
            errors.append(f"code{k:02d}：元のセルの中身が変わった（解説ファイルの @sha1 {h}、今のセル {sha1(src)}）。行番号を確かめて、解説ファイルを直す")
            continue
        need = commentable(src)
        miss, extra = sorted(set(need) - set(com)), sorted(set(com) - set(need))
        if miss or extra:
            errors.append(f"code{k:02d}：解説のない行 {miss}、解説を付けられない（または不要な）行 {extra}")
            continue
        new = annotate(src, com)
        if ast.dump(ast.parse(for_ast(new))) != ast.dump(ast.parse(for_ast(src))):
            errors.append(f"code{k:02d}：解説を入れた後の構文が、元のセルと違う")
            continue
        c["source"] = to_lines(new)
        n_com += len(need)
    if errors:
        sys.exit(f"エラー（{name}）：\n" + "\n".join(errors))
    text = INTRO.replace("{name}", name).replace("{cdir}", cdir_name)     # （INTRO には { } を含む Python の例があるので、format は使わない）
    intro = {"cell_type": "markdown", "id": "commented-intro", "metadata": {}, "source": to_lines(text.strip("\n"))}
    nb["cells"].insert(0, intro)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    with open(out_nb, "wb") as f:
        f.write((json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"作成: {os.path.relpath(out_nb, REPO)}（コードセル {k} 個、解説を付けた行 {n_com} 行）")


def main():
    names = [a for a in sys.argv[1:] if a in NOTEBOOKS] or list(NOTEBOOKS)
    for n in names:
        build(*NOTEBOOKS[n])


if __name__ == "__main__":
    main()
