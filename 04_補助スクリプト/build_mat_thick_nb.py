# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR の「厚みごとの比較」のノートブック (未実行) を生成する。

使い方: python build_mat_thick_nb.py <出力する .ipynb のパス> [<もう 1 つの出力パス> ...] [--force]
（成果物は 01_機械学習コードフォルダ/材料別_温度シフトファクター検討/材料別_温度シフトファクター検討用コード/
 材料別_厚みごとの比較_温度シフトファクターの検討_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb に置く〔シフトファクター検討用のノートと同じフォルダ。
 2026-10-28 のユーザー依頼。名前は 2026-11-05 にユーザーが変えたもの〕。ノートは、置いた場所から上へたどってリポジトリ直下を探すので、どこに置いても動く）
既存ファイルがある場合は、--force を付けない限り上書きしない。
シフトファクター検討用のノート（材料別_温度シフトファクターの検討_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb）が保存した学習済みモデルと
予測値を読み、厚みごとに、温度シフトファクターが効いているか、シフトに使う温度は流動中樹脂温度 Tff と加熱シリンダ温度 Tset のどちらが適しているか、
厚みごとに学習したモデルと全厚みの総合データのモデルのどちらがよいかを調べる（ユーザー依頼、2026-10-22。学習 5data・検証 3data のまま）。
2026-10-27：読む場所と保存先を新しい結果の構成（<学習結果フォルダ>/<版>/全厚み総合/シフトファクターの検討/ → <版>/厚みごと/）にし、
学習結果フォルダをダイアログで選ぶ形にした。図は 600 dpi だけを保存する。
2026-11-02（ユーザー決定で、シフトファクターの型を材料ごとに分けなくなった）：設定 SHIFT_FORM の型のモデルを読み、結果は <版>/厚みごと/<型>/ に保存した。
2026-11-05（ユーザー依頼）：シフトファクター検討用のノートと同じ設定（SHIFT_COLS・STRUCTS・SHIFT_FORMS・MAIN_KEY など）で、読むモデルを決める形にした。
解析の中身（厚みごとのデータで、温度シフトファクターを、対象の列と式〔シフトの形・型〕を変えて、それぞれ調べる）は変えず、モデルごとに行う。
「Tff か Tset か」は、シフトの形と型が同じモデルの組（例：B_Tset_Arr と B_Tff_Arr）ごとに比べる。結果は <版>/厚みごと/ に、組の名前をファイルの名前に入れて保存する。
"""
import json
import os
import random
import sys

import mat_nb_cells as nbc

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src})


HERE = os.path.dirname(os.path.abspath(__file__))
_common = open(os.path.join(HERE, "mat_common.py"), encoding="utf-8").read().split('"""', 2)[2].lstrip("\n")
COMMON_CELL = "# セル：共通の部品（読み込み・シフトファクター・モデル・評価・図）。04_補助スクリプト/mat_common.py の中身。直すときは、そのファイルを直して、ノートを作り直す\n" + _common
_thick = open(os.path.join(HERE, "mat_thickness.py"), encoding="utf-8").read().split('"""', 2)[2].lstrip("\n")
_thick = "\n".join(l for l in _thick.split("\n") if not l.startswith("from mat_common import"))
THICK_CELL = "# セル：厚みごとの比較の部品。04_補助スクリプト/mat_thickness.py の中身（import の行を除く）。直すときは、そのファイルを直して、ノートを作り直す\n" + _thick.lstrip("\n")
MAIN_NB = "材料別_温度シフトファクターの検討_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb"     # このノートが読むモデルを学習するノート（シフトファクター検討用。このノートと同じフォルダ）
PDF = "05_材料ごとの機械学習データと予測検証データ/05_学習結果説明書/学習結果説明_厚みごとの温度シフト.pdf"   # 2026-11-12 にユーザーが 05_学習結果説明書/ に移した
MANUAL = "説明書_材料別温度シフトGPRのノート.pdf"     # 1 つ上のフォルダの、3 つのノートの説明書（build_mat_nb_manual.py。2026-10-28）

# ---------------------------------------------------------------- 0. タイトル
md(rf'''# 材料別の温度シフト GPR：厚みごとの比較

シフトファクター検討用のノート（同じフォルダの `{MAIN_NB}`）で学習したモデルを、**キャビティ厚みごと**に見直す。学習は 5data、検証は 3data のまま変えない。調べることは 3 つ（ユーザー依頼、2026-10-22）。

1. **厚みごとに、温度シフトファクターが効いているか**（学習済みのモデル。4 節）
2. **厚みごとに、シフトに使う温度は、流動中樹脂温度 Tff と加熱シリンダ温度 Tset のどちらが適しているか**（5〜8 節）
3. **厚みごとに学習したモデルと、全厚みの総合データで学習したモデルの、どちらがよいか**（9 節）

**比べるモデル**：シフトファクター検討用のノートと**同じ設定**（`SHIFT_COLS`〔シフトに使う列〕・`STRUCTS`〔シフトの形〕・`SHIFT_FORMS`〔シフトファクターの型＝式〕）で選ぶ（2026-11-05 から。それまでは、構造 B の 2 つのモデルと、型 1 つ〔`SHIFT_FORM`〕だけだった）。そのノートが学習した、列・形・型の組み合わせごとのモデルと、基準の C を読み、1〜3 を**モデルごとに**行う（解析の中身は、それまでと同じ）。記号の末尾 `_Arr` はアレニウス型、`_WLF` は WLF 型。

| 列（`SHIFT_COLS` の書き方） | モデル | 式 | シフトに使う温度 | GP の入力に入る温度 |
|---|---|---|---|---|
| `"加熱シリンダ温度[℃]"` | **B_Tset_Arr**・**B_Tset_WLF** | ln η = ln a_v(Tset) ＋ f( ln γ̇ ＋ ln a_h(Tset), ln P3_3, ln Tff ) | 加熱シリンダ温度 Tset | 流動中樹脂温度 Tff |
| `"流動中樹脂温度(℃)"` | **B_Tff_Arr**・**B_Tff_WLF** | ln η = ln a_v(Tff) ＋ f( ln γ̇ ＋ ln a_h(Tff), ln P3_3, Tset ) | 流動中樹脂温度 Tff | 加熱シリンダ温度 Tset |
| （`STRUCTS` に `"A"` があるとき。既定） | **A_Tset_Arr**・**A_Tff_WLF** など | B の横のシフトを 0 にしたもの（縦だけずらす。係数 1 つ） | （列のとおり） | （列のとおり） |
| （基準。いつも読む） | **C** | ln η = f( ln γ̇, ln P3_3, ln Tff, Tset ) | なし | 両方 |

横のシフトファクター a_h、縦のシフトファクター a_v（基準温度 Tref で 1）。アレニウス型の係数は E_h・E_v [kJ/mol]、WLF 型は A1_h・A1_v。基準温度・WLF 型の Tg・係数の格子は、学習済みモデルに記録されたもの（シフトファクター検討用のノートの `SHIFT_CFG`・`FORM_CFG`）を使う。

**組**：シフトの形と型が同じで、**シフトに使う温度だけが違う**モデルの組（例：組 `B_Arr` は B_Tset_Arr と B_Tff_Arr、組 `A_WLF` は A_Tset_WLF と A_Tff_WLF）。2 の「Tff か Tset か」は、組ごとに比べる（`SHIFT_COLS` が 1 つの列だけなら、比べる相手がないので「—」）。

**1 の見方（4 節）**：モデルごと・厚みごとに、次の 3 つを見る。
- 温度の順序：同じせん断速度で、低温ほど粘度が高いか（実測と予測）
- 温度による粘度の比 $R = \exp\left(\frac{{1}}{{G}}\sum_g [\ln\eta_{{低温}}(\dot\gamma_g) - \ln\eta_{{高温}}(\dot\gamma_g)]\right)$：最低と最高の加熱シリンダ温度の、成形条件ごとの平均を、せん断速度 γ̇ の重なる範囲の共通の格子（G = 50 点）で比べる。実測の比 R_実測 と、モデルの予測平均から同じ手順で求めた比 R_予測 を比べる
- 換算前後のずれ：学習済みの係数で換算した後に、温度の違うデータが 1 本の曲線（厚みごとの 3 次式）に重なるか（シフトファクター検討用のノートの「重ね合わせの残差」を厚みごとに分けたもの。粘度の % で表す）

判定：○＝順序が一致し、|ln R_予測 − ln R_実測| が同じ成形条件のショット間の ln η の標準偏差（純粋誤差。約 0.04）以内で、換算後のずれが換算前より小さい。×＝予測の順序が実測と逆。△＝それ以外。（許容差は、10-23 までモデルのノイズの標準偏差だったが、ショットごとの入力ではノイズが約 0.006 と小さく出るので、入力の取り方によらない純粋誤差に変えた。条件平均の入力では、両者はほぼ同じ値）

**2 の基準（5〜8 節）**：組ごと・厚みごとに、次の 3 つで「Tff／Tset／差なし」を決め、多数決でまとめる。
- (a) 重ね合わせ：厚みごとに係数を決め直したときの、最小のずれ（データだけで決まる。構造 A は縦の係数だけ）。差が `A_CLEAR`（既定 0.5 ポイント）以上なら差あり
- (b) 対数周辺尤度：厚みごとの 5data だけで組の 2 つのモデルを学習し、係数をプロファイル尤度で決めたときの、最大の対数周辺尤度の差 ΔL = LML(Tff のモデル) − LML(Tset のモデル)。|ΔL| が `DL_CLEAR`（既定 3.0）以上なら差あり
- (c) 検証データの誤差：保存済みの 3data の予測を、厚みごとに集計した予測誤差（粘度の %）。差が `ERR_CLEAR`（既定 1.0 ポイント）以上なら差あり

**3 の比べ方（9 節）**：厚みごとの 5data だけで、すべてのモデル（C を含む）を学習し（シフトのあるモデルの係数は 6 節で決めたもの）、同じ厚みの 3data を予測する。総合データのモデル（シフトファクター検討用のノートのモデル）の予測と、厚みごと・全厚みで比べる。

**読むときの注意**：流動中樹脂温度 Tff は、せん断発熱で射出速度とともに変わる。そのため、Tff でのシフトは、温度の効果だけでなく、曲線の形の調整も兼ねうる。(a) だけでなく、(b)(c) も見るのはこのため。''')

# ---------------------------------------------------------------- 1. 使い方・設定
md(f'''## 使い方

**初めての人へ**：1 つ上のフォルダ（`材料別_温度シフトファクター検討/`）の説明書 `{MANUAL}` に、3 つのノートの目的・関係・使えるデータ（学習ファイルの条件）・使い方をまとめてある。先に読む。

1. **先にシフトファクター検討用のノート `{MAIN_NB}` を実行しておく**（このノートは、その学習済みモデル `<モデルのフォルダ>/学習済みモデル_<モデル>.joblib` と予測値 `予測検証_3data_予測値.csv`、指標の表を読む）。設定エリアの `SHIFT_COLS`・`STRUCTS`・`SHIFT_FORMS`・`INPUT_MODE`・`MATERIALS` は、**そのノートと同じ書き方・同じ意味**（既定も同じ）。そのノートを既定のまま実行していれば、このノートも既定のままでよい。そのノートで学習したモデルの一部だけを調べたいときは、ここで列・形・型を減らす（そのノートで学習していないモデルを選ぶと、読み込みで止まる）。`INPUT_MODE` は、そのノートと同じにする。
2. メニューの「すべて実行」で、上から順に実行する（目安：既定の設定〔8 つのモデルと C〕・ショットごとの入力で、3 材料 約 45 分〔2026-11-05 に GPPS〔厚み 3 つ〕で計測した約 12 分からの見積もり。厚みが 4 つの PP・粉砕材混合 PP は約 16 分。要確認〕。大部分は 6 節のプロファイル尤度〔特に構造 B の WLF 型〕。型を 1 つにするか、`STRUCTS = ["B"]` にすると短くなる）。カーネルは、このプロジェクトの `.venv` を選ぶ。
3. **実行の始めに（「2. 読み込み」のセルで）、材料ごとにダイアログが 1 回開く**：シフトファクター検討用のノートで選んだ**学習結果フォルダ**を選ぶ。学習データ・検証データの場所は、学習済みモデルに記録されたものを使う（モデルを学習したデータと必ず同じになるように。ファイルが見つからないときだけ、ダイアログで選ぶ）。ダイアログが見当たらないときは、VS Code の後ろに隠れているので、タスクバーで探す。設定 `FILES` に書けば、ダイアログは出ない。
4. 設定を変えたいときは、最初の「設定エリア」のセルだけを直して、最初から実行し直す（時間を短くする：`THICK_PROFILE = False`。6・9 節と、8 節の (b) を飛ばす）。
5. 各節の最後に、**判定**が出る（`>> 判定：OK` か `>> 判定：★要確認`）。OK は「想定どおり」、★ は「想定と違う、または結果を読むときに注意が要る点」。★ が出ても、エラーではない。最後の「判定の一覧」に一覧が出る。
6. 図と表は、学習結果フォルダの **`<版>/厚みごと/`**（版は `INPUT_MODE` で、`全ショット版` か `条件ごとの平均版`）に、次の 4 つのフォルダに分けて保存される。`判定の一覧.csv` は `厚みごと/` の直下。表には、すべてのモデルが入る（列「組」「モデル」）。組ごとの図は、ファイルの名前に組の名前が入る（例：`学習結果_B_Arr_厚み0.5mm.png`）。**図は、すべて 600 dpi の 1 枚だけ**（設定 `FIG_DPI`）。ノートに表示するのは、`MAIN_KEY` の組の図だけ（ほかの図も保存する）。
   - `学習結果グラフ/`（3 節）、`温度シフトの確認/`（4 節）、`シフトに使う温度/`（5〜8 節の (a)(b)(c) とまとめ）、`厚みごとのモデルと総合の比較/`（9 節）
7. 用語や図の見方は、説明書 `{PDF}` にある（10-26 までの結果で書いたもの。そのときのモデルは B_Tset・B_Tff・C だけで、結果のフォルダの名前は、そのときの構成のまま）。''')
md(r'''## 1. 設定''')
code(r'''# セル1：設定エリア  ▼▼▼ 変えてよいのは、ここだけ ▼▼▼
from pathlib import Path      # （下の REPO で、フォルダをたどるのに使う）

SHIFT_COLS = ["加熱シリンダ温度[℃]", "流動中樹脂温度(℃)"]   # 調べるシフトファクターの列（学習ファイルの列名。シフトファクター検討用のノートと同じ書き方。そのノートで学習した列から選ぶ）。基準として、シフトなしの C も読む

STRUCTS = ["B", "A"]          # シフトの形："B"＝横・縦を別々に推定／"A"＝縦だけ（シフトファクター検討用のノートと同じ。そのノートで学習した形から選ぶ）

SHIFT_FORMS = ["arrhenius", "wlf"]   # シフトファクターの型："arrhenius"＝アレニウス型（記号の末尾 _Arr）／"wlf"＝WLF 型（_WLF）。選んだすべての材料に、ここの型をすべて当てはめる（シフトファクター検討用のノートと同じ。そのノートで学習した型から選ぶ）

MAIN_KEY = "B_Tset_Arr"       # 図をノートにも表示するモデル（このモデルの組〔シフトの形と型が同じモデル〕の図を表示する。ほかの図も保存する）

SEED = 42                     # 乱数シード

NOISE_LB = 1e-6               # ノイズ分散の下限（シフトファクター検討用のノートと同じにする）

USE_LINEAR = True             # 線形項（DotProduct）を足すか（シフトファクター検討用のノートと同じにする）

INPUT_MODE = "shot"           # 読む学習済みモデルの入力の取り方（シフトファクター検討用のノートと同じにする。"shot"：ショットごと〔既定。全ショット版/〕、"cond"：条件平均〔条件ごとの平均版/〕）

N_RESTARTS = 20               # 9 節：厚みごとの最終モデルの学習での、最適化のやり直し回数（シフトファクター検討用のノートと同じ）

PROFILE_RESTARTS = 1          # 6 節：プロファイル尤度の格子の各点での、やり直し回数（シフトファクター検討用のノートと同じ）

USE_EXCLUSIONS = True         # 問題データ一覧（05 フォルダ）の「除外」を除く（シフトファクター検討用のノートと同じ）

REPO = str(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "04_補助スクリプト").is_dir()))   # リポジトリ直下（このノートの場所から上へたどって、04_補助スクリプト があるフォルダを探す）

MATERIALS = ["PP", "MIX", "GPPS"]    # 調べる材料（PP：バージン PP、MIX：粉砕材混合 PP、GPPS）。材料ごとに、実行の始めにダイアログで、学習結果フォルダ（シフトファクター検討用のノートで選んだもの）を選ぶ

MAT_FOLDERS = {"PP": "01_バージンPP再々実験(データと結果)", "MIX": "02_粉砕材混合PP実験(データと結果)", "GPPS": "03_GPPS実験(データと結果)"}   # 05 フォルダの中の材料フォルダ名（ダイアログの最初のフォルダ）

FILES = {}                    # ダイアログを使わずに決めるとき：{"PP": {"結果": r"…"}}（相対パスはリポジトリ直下から。"学習"・"検証" も書くと、学習済みモデルの記録より、そちらを使う）
FINE_B = (4.0, 1.0)           # B の細かい格子：最良の点の ±4 を 1 刻み（シフトファクター検討用のノートと同じ。係数の粗い格子と型の設定は、学習済みモデルの記録を使う。その設定に "fine" があれば〔WLF 型〕、そちらを使う）
FINE_A = (4.5, 0.5)           # A の細かい格子：最良の点の ±4.5 を 0.5 刻み（シフトファクター検討用のノートと同じ）
MAX_REFINE = 4                # 最良の点が細かい格子の端に当たったとき、取り直す回数の上限（シフトファクター検討用のノートと同じ）
N_JOBS = 8                    # プロファイル尤度と、9 節の厚みごとのモデルを並列に計算する数（CPU の数に合わせる）
FIG_DPI = 600                 # 保存する図の解像度（すべての図を、この解像度の 1 枚だけで保存する）
SHOW_FIGS = True              # True：MAIN_KEY の組の図をノートにも表示する（False：保存だけ）
# ---- ここから下は、厚みごとの比較だけの設定 ----
THICK_PROFILE = True          # 6 節：厚みごとの 5data で GPR を学習し、係数をプロファイル尤度で決める（False：6・9 節と 8 節の (b) を飛ばす）
THICK_MODELS = True           # 9 節：厚みごとのモデルと、総合データのモデルを比べる（6 節の結果を使う）
A_CLEAR = 0.5                 # (a) 重ね合わせのずれ [%] の差が、これ以上なら「差あり」
DL_CLEAR = 3.0                # (b) 対数周辺尤度の差が、これ以上なら「差あり」
ERR_CLEAR = 1.0               # (c)・9 節：予測誤差 [%] の差が、これ以上なら「差あり」
FLOW_POINTS = "shot"          # 3 節の学習結果グラフの点："shot"＝ショットごとの計測値（既定）／"cond"＝成形条件ごとの平均
# ▲▲▲ 設定エリア終了 ▲▲▲''')
code("# セル2：ライブラリ、フォント\n" + nbc.LIBS + '\nprint("材料:", MATERIALS, "／入力の取り方:", INPUT_MODE)')
md(r'''### 共通の部品

シフトファクター検討用のノートと同じ部品（`04_補助スクリプト/mat_common.py`）と、厚みごとの比較の部品（`04_補助スクリプト/mat_thickness.py`）。''')
code(COMMON_CELL)
code(THICK_CELL)
md(r'''### 読むモデルと、結果の置き場所

設定 `SHIFT_COLS`・`STRUCTS`・`SHIFT_FORMS` から、読むモデルを決める（シフトファクター検討用のノートと同じ決め方：列と形と型の組み合わせごとのシフトのあるモデルと、基準の C）。シフトのあるモデルは、シフトの形と型が同じものを**組**にまとめる（組の中で違うのは、シフトに使う温度だけ）。シフトファクター検討用のノートの結果（学習結果フォルダの `<版>/全厚み総合/シフトファクターの検討/`）を読み、このノートの結果を `<版>/厚みごと/` に保存する（版は `INPUT_MODE` で決まる：`全ショット版` か `条件ごとの平均版`）。''')
code(r'''# セル2b：読むモデルと、結果の置き場所
SHIFT_KEYS = [k for k in model_keys(SHIFT_COLS, STRUCTS, SHIFT_FORMS) if k != "C"]   # 読むシフトのあるモデル（列・形・型の組み合わせ。シフトファクター検討用のノートと同じ記号）
KEYS = SHIFT_KEYS + ["C"]                       # それと、基準の C（シフトなし）
assert SHIFT_KEYS, "SHIFT_COLS・STRUCTS・SHIFT_FORMS から、シフトのあるモデルが 1 つもできない（3 つとも 1 つ以上書く）"
assert MAIN_KEY in KEYS, f"MAIN_KEY（{MAIN_KEY}）は {KEYS} のどれか"
GROUPS = {}                                     # 組の名前（シフトの形と型。例：B_Arr）→ その組のモデル（SHIFT_COLS の順。違うのは、シフトに使う温度だけ）
for k in SHIFT_KEYS:
    GROUPS.setdefault(group_of(k), []).append(k)
MAIN_GROUP = group_of(MAIN_KEY) if MAIN_KEY != "C" else next(iter(GROUPS))


def shown(g):
    """組 g の図を、ノートにも表示するか（SHOW_FIGS のとき、MAIN_KEY の組だけ）"""
    return SHOW_FIGS and g == MAIN_GROUP


def study_dir(mat):
    """シフトファクター検討用のノートの結果のフォルダ：<学習結果フォルダ>/<版>/全厚み総合/シフトファクターの検討/（読むだけなので、フォルダは作らない）"""
    return result_dir(mat, VERSION_DIRS[INPUT_MODE], THICK_TOTAL, STUDY_DIR, make=False)


def model_dir(mat, key):
    """シフトファクター検討用のノートが保存した、モデルのフォルダ（study_dir の下の「記号_説明」のフォルダ）"""
    return os.path.join(study_dir(mat), MODEL_DIRS[key])


def out_dir(mat, sub=None):
    """このノートの結果の置き場所：<学習結果フォルダ>/<版>/厚みごと/（sub があれば、その下のフォルダ）"""
    return result_dir(mat, VERSION_DIRS[INPUT_MODE], THICK_EACH, *([sub] if sub else []))


print("読むモデル：" + "、".join(KEYS))
print("組（シフトに使う温度だけが違うモデル）：" + "／".join(f"{g}〔{group_ja(g)}〕{'・'.join(ks)}" for g, ks in GROUPS.items()))
print(f"読む結果：<学習結果フォルダ>/{VERSION_DIRS[INPUT_MODE]}/{THICK_TOTAL}/{STUDY_DIR}/")
print(f"結果の置き場所：<学習結果フォルダ>/{VERSION_DIRS[INPUT_MODE]}/{THICK_EACH}/（ノートに表示するのは、組 {MAIN_GROUP} の図）")''')

# ---------------------------------------------------------------- 2. 読み込み
md(r'''## 2. 学習済みモデルとデータの読み込み

**このセルで、材料ごとにダイアログが 1 回開く**（シフトファクター検討用のノートで選んだ学習結果フォルダを選ぶ。設定 `FILES` に書いたものは開かない）。学習済みモデル（`KEYS` のすべて）と 3data の予測値を読み、学習済みモデルに記録された学習データ・検証データを、シフトファクター検討用のノートと同じ手順で読む（問題データ一覧の「除外」を除く）。シフトファクターの設定（基準温度・WLF 型の Tg・係数の格子）は、学習済みモデルに記録されたものを使う。学習済みモデルがないとき、材料が違うとき（学習結果フォルダの選び間違い）、モデルによって学習データ・検証データの記録が違うときは、止める。

**判定**：学習済みモデルの入力の取り方が `INPUT_MODE` と同じ／読んだモデルで 3data を予測し直すと、保存済みの予測値と一致する（データとモデルの組み合わせが正しい）／同じ型のモデルは、同じシフトファクターの設定で学習された。''')
code(r'''# セル3：読み込み
import joblib
JUDGES.clear()
display(choose_inputs(MATERIALS, FILES, need_data=False))      # 材料ごとに、学習結果フォルダを決める（設定 FILES に書いていなければ、ダイアログで選ぶ）
FITS, PRED, OBJ, CFG = {}, {}, {}, {}
for mat in MATERIALS:
    for key in KEYS:
        path = os.path.join(model_dir(mat, key), f"学習済みモデル_{key}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} がありません。先にシフトファクター検討用のノートを、INPUT_MODE = '{INPUT_MODE}' で、{key} ができる設定（SHIFT_COLS・STRUCTS・SHIFT_FORMS）で実行し、"
                                    "同じ学習結果フォルダを選んでください（または、このノートの SHIFT_COLS・STRUCTS・SHIFT_FORMS を、そのノートで学習したモデルに合わせてください）")
        OBJ[(mat, key)] = joblib.load(path)
        if OBJ[(mat, key)]["材料"] != MAT_NAMES[mat]:
            raise ValueError(f"{path} は {OBJ[(mat, key)]['材料']} のモデルです（{MAT_NAMES[mat]} の学習結果フォルダを選んでください）")
        FITS[(mat, key)] = OBJ[(mat, key)]["fit"]
        PRED[(mat, key)] = pd.read_csv(os.path.join(model_dir(mat, key), "予測検証_3data_予測値.csv"), encoding="utf-8-sig")
        if key != "C":
            CFG.setdefault((mat, key_form(key)), FITS[(mat, key)]["cfg"])     # シフトファクターの設定は、学習済みモデルの記録（型ごと。最初に読んだモデルのもの）
    recs = {(OBJ[(mat, k)].get("学習データのファイル"), OBJ[(mat, k)].get("検証データのファイル")) for k in KEYS}
    if len(recs) > 1:
        raise ValueError(f"{MAT_NAMES[mat]}：読んだモデルの学習データ・検証データの記録が {len(recs)} 通りあります（{sorted(recs)}）。"
                         "シフトファクター検討用のノートを、同じデータで、すべてのモデルについて実行し直してください")
    data_from_model(mat, OBJ[(mat, SHIFT_KEYS[0])])         # 学習データ・検証データの場所は、学習済みモデルの記録から取る（設定 FILES に書いてあれば、そちら）


def kcfg(mat, key):
    """モデル key のシフトファクターの設定（学習済みモデルに記録されたもの。材料の Tref・Tg と、型の格子）。C と換算前（None）は、最初のモデルの型の設定（シフトには使わない）"""
    return CFG[(mat, key_form(key) or key_form(SHIFT_KEYS[0]))]


D = load_all(MATERIALS)
TRAIN, TEST, POOL = {}, {}, {}
for mat in MATERIALS:
    TRAIN[mat] = D[(mat, "5data")]
    TEST[mat] = add_groups(TRAIN[mat], D[(mat, "3data")])
    POOL[mat] = pd.concat([TRAIN[mat], TEST[mat]], ignore_index=True)
    print(f"【{MAT_NAMES[mat]}】学習データ：{show_path(SOURCES[mat]['5data'])}／検証データ：{show_path(SOURCES[mat]['3data'])}")
    for key in KEYS:
        judge(FITS[(mat, key)].get("mode") == INPUT_MODE, f"{key}：学習済みモデルの入力の取り方が {INPUT_MODE}", mat, "2 読み込み")
        mu, _, _, _ = predict(FITS[(mat, key)], TEST[mat])
        chk = TEST[mat][["file"]].assign(mu=mu).merge(PRED[(mat, key)][["file", "予測_ln_eta"]], on="file")
        judge(len(chk) == len(TEST[mat]) and np.allclose(chk["mu"], chk["予測_ln_eta"], atol=1e-6),
              f"{key}：3data を予測し直した値が、保存済みの予測値と一致（{len(chk)}/{len(TEST[mat])} 行）", mat, "2 読み込み")
    bad = [k for k in SHIFT_KEYS if FITS[(mat, k)]["cfg"] != kcfg(mat, k)]
    judge(not bad, "同じ型のモデルは、同じシフトファクターの設定（基準温度 Tref・WLF 型の Tg・係数の格子）で学習された" + (f"（違うモデル：{'、'.join(bad)}）" if bad else ""),
          mat, "2 読み込み")
    cnt = TRAIN[mat].groupby("H")["cond"].nunique()
    cnt_te = TEST[mat].groupby("H")["cond"].nunique()
    print(f"【{MAT_NAMES[mat]}】厚み {', '.join(f'{h:g}' for h in thicknesses(TRAIN[mat]))} mm。学習の条件数：" +
          "、".join(f"{h:g} mm {cnt[h]}" for h in cnt.index) + "／検証の条件数：" + "、".join(f"{h:g} mm {cnt_te.get(h, 0)}" for h in cnt.index))
    print("  学習済みの係数（横, 縦）：" + "、".join(f"{k} ({FITS[(mat, k)]['th']:g}, {FITS[(mat, k)]['tv']:g} {theta_names(kcfg(mat, k))[2]})" for k in SHIFT_KEYS) +
          f"。Tref = {kcfg(mat, SHIFT_KEYS[0])['Tref']:g} ℃")''')

# ---------------------------------------------------------------- 3. 厚みごとの学習結果グラフ
md(r'''## 3. 厚みごとの学習結果グラフ（組ごとに、シフトに使う温度の違うモデルを並べる）

組ごと・厚みごとに、せん断速度 ― 粘度の学習結果を、シフトに使う温度の違うモデル（`SHIFT_COLS` の順。既定は左が Tset、右が Tff）で並べる。点は、ショットごとの計測値（塗り＝学習の 5data、白抜き＝検証の 3data。設定 `FLOW_POINTS = "cond"` にすると、成形条件ごとの平均の点になる）、線は加熱シリンダ温度ごとの予測平均値、帯は 95% 予測区間。曲線の入力（P3_3・Tff）は、同じ厚み・温度の条件平均を、せん断速度で内挿した値（範囲の外は端の値）。ファイルの名前は `学習結果_<組>_厚み<厚み>mm.png`。

**見方**：3 本の線が、実測の点と同じ順序（低温ほど上）・同じ間隔で並んでいれば、温度の効果を表せている。''')
code(r'''# セル4：厚みごとの学習結果グラフ
for mat in MATERIALS:
    for g, ks in GROUPS.items():
        for h in thicknesses(POOL[mat]):
            thick_flow_pair({k: FITS[(mat, k)] for k in ks}, mat, POOL[mat], set(TRAIN[mat]["cond"]), h, out_dir(mat, "学習結果グラフ"), points=FLOW_POINTS,
                            tag=g, show=shown(g))''')

# ---------------------------------------------------------------- 4. 温度シフトの確認
md(r'''## 4. 温度シフトファクターが効いているか（厚みごと、学習済みの係数）

シフトのあるモデルごと・厚みごとに、温度による粘度の比 R（実測・予測）、温度の順序、学習済みの係数で換算する前後のずれを出し、○・△・× で判定する（基準は冒頭）。全厚みをまとめたずれ（シフトファクター検討用のノートの重ね合わせの残差）も、照合のために表示する。図は組ごと（`温度による粘度の比_<組>.png`、`重ね合わせのずれ_学習済みの係数_<組>.png`、`重ね合わせ_学習済みの係数_<組>_厚み<厚み>mm.png`）。

**判定**：判定が ○ の厚みは OK、△・× は ★（何が満たされなかったかを、文に書く）。''')
code(r'''# セル5：温度シフトの確認
CK = []
for mat in MATERIALS:
    r0_ = superpose_rmse(shifted_cond_means(POOL[mat], None, kcfg(mat, None), 0.0, 0.0))
    for key in SHIFT_KEYS:
        fit = FITS[(mat, key)]
        t = thick_check(fit, POOL[mat]).assign(材料=MAT_NAMES[mat], 組=group_of(key))
        CK.append(t)
        r1_ = superpose_rmse(shifted_cond_means(POOL[mat], key, kcfg(mat, key), fit["th"], fit["tv"]))
        print(f"【{MAT_NAMES[mat]}・{key}】全厚みのずれ（照合用。シフトファクター検討用のノートの重ね合わせの残差を % にしたもの）：換算前 {pct_err(r0_):.1f}% → 換算後 {pct_err(r1_):.1f}%")
        for _, r in t.iterrows():
            why = "、".join(s for s, ok in [("温度の順序が実測と予測で違う", r["順序が一致"]), ("比 R の大きさが合わない", r["比の大きさが一致"]),
                                              ("換算でずれが減らない", r["換算でずれが減る"])] if not ok)
            judge(r["判定"] == "○", f"{key} 厚み {r['厚み']:g} mm：R 実測 {r['R_実測']:.3f}・予測 {r['R_予測']:.3f}、ずれ {pct_err(r['ずれ_換算前']):.1f} → "
                  f"{pct_err(r['ずれ_換算後']):.1f}%（判定 {r['判定']}{'。' + why if why else ''}）", mat, "4 温度シフトの確認")
CK = pd.concat(CK, ignore_index=True)
CK["ずれ_換算前_%"] = pct_err(CK["ずれ_換算前"])
CK["ずれ_換算後_%"] = pct_err(CK["ずれ_換算後"])
show_cols = ["材料", "組", "モデル", "厚み", "条件数", "比べた γ̇ の範囲", "R_実測", "R_予測", "lnR の差", "許容差（ショット間の標準偏差）", "ノイズの標準偏差", "ずれ_換算前_%",
             "ずれ_換算後_%", "判定"]
display(CK[show_cols].round(3))
for mat in MATERIALS:
    c = CK[CK["材料"] == MAT_NAMES[mat]]
    to_csv(c, os.path.join(out_dir(mat, "温度シフトの確認"), "温度シフトの確認_厚みごと.csv"))
    for g, ks in GROUPS.items():
        cg = c[c["組"] == g]
        w = cg.pivot(index="厚み", columns="モデル", values="R_予測").add_prefix("R_予測_").reset_index()
        w["R_実測"] = cg.groupby("厚み")["R_実測"].first().to_numpy()
        thick_bars(w, None, {"R_実測": "実測", **{f"R_予測_{k}": f"予測（{k}）" for k in ks}},
                   f"{MAT_NAMES[mat]}　{group_ja(g)}\n温度による粘度の比 R（最低温度 ÷ 最高温度）", "粘度の比 R", out_dir(mat, "温度シフトの確認"), f"温度による粘度の比_{g}",
                   fmt="{:.2f}", hline=1.0, show=shown(g))
        z = cg.pivot(index="厚み", columns="モデル", values="ずれ_換算後_%").add_prefix("換算後_").reset_index()
        z["換算前"] = cg.groupby("厚み")["ずれ_換算前_%"].first().to_numpy()
        thick_bars(z, None, {"換算前": "換算前", **{f"換算後_{k}": f"{TEMP_JA[temp_of(k)]}で換算（{k}、学習済みの係数）" for k in ks}},
                   f"{MAT_NAMES[mat]}　{group_ja(g)}\n重ね合わせのずれ（学習済みの係数）", "ずれ (%)", out_dir(mat, "温度シフトの確認"), f"重ね合わせのずれ_学習済みの係数_{g}",
                   show=shown(g))
        for h in thicknesses(POOL[mat]):
            thick_master(mat, POOL[mat], kcfg(mat, ks[0]), h, {k: (FITS[(mat, k)]["th"], FITS[(mat, k)]["tv"]) for k in ks}, out_dir(mat, "温度シフトの確認"),
                         f"重ね合わせ_学習済みの係数_{g}", show=shown(g))''')

# ---------------------------------------------------------------- 5. (a) 重ね合わせ
md(r'''## 5. (a) 厚みごとに係数を決め直した重ね合わせ

組ごと・厚みごとのデータ（5data＋3data の条件平均）だけで、ずれが最小になる係数（構造 B は横・縦、構造 A は縦だけ）を、シフトファクター検討用のノートのプロファイル尤度と同じ格子（その型の格子）で探す。加熱シリンダ温度 Tset で換算した場合と、流動中樹脂温度 Tff で換算した場合の、最小のずれを比べる。モデル（GP）を使わず、データだけで決まる。

**見方**：ずれが小さい温度ほど、その厚みで、温度の違う粘度曲線を 1 本に重ねやすい。

**判定**：最良の係数が、粗い格子の端に当たっていない（当たっていれば ★。尾根の上で、さらに外にずれが小さい点がありうる）。''')
code(r'''# セル6：(a) 厚みごとの重ね合わせ
SUP = []
for mat in MATERIALS:
    for h in thicknesses(POOL[mat]):
        ph = sub_h(POOL[mat], h)
        for g, ks in GROUPS.items():
            cfg = kcfg(mat, ks[0])
            th_n, tv_n, unit = theta_names(cfg)
            row = {"材料": MAT_NAMES[mat], "厚み": h, "組": g, "条件数": ph["cond"].nunique()}
            for key in ks:
                s = temp_of(key)
                b = best_superpose(ph, key, cfg)
                row.update({"ずれ_換算前_%": pct_err(b["rmse_0"]), f"ずれ_{s}_%": pct_err(b["rmse"]), f"横の係数_{s}": b["th"], f"縦の係数_{s}": b["tv"],
                            f"格子の端_{s}": b["at_edge"]})
                judge(not b["at_edge"], f"(a) {key} 厚み {h:g} mm：最良の係数（{th_n} {b['th']:g}・{tv_n} {b['tv']:g} {unit}）が、粗い格子の端に当たっていない", mat, "5 (a) 重ね合わせ")
            row["単位"] = unit
            row["適した温度(a)"] = decide_temps({temp_of(k): row[f"ずれ_{temp_of(k)}_%"] for k in ks}, A_CLEAR, False)
            SUP.append(row)
            thick_master(mat, POOL[mat], cfg, h, {k: (row[f"横の係数_{temp_of(k)}"], row[f"縦の係数_{temp_of(k)}"]) for k in ks},
                         out_dir(mat, "シフトに使う温度"), f"重ね合わせ_厚みごとの係数_{g}", show=shown(g))
SUP = pd.DataFrame(SUP)
display(SUP.round(2))
for mat in MATERIALS:
    s = SUP[SUP["材料"] == MAT_NAMES[mat]]
    to_csv(s, os.path.join(out_dir(mat, "シフトに使う温度"), "重ね合わせ_厚みごと.csv"))
    for g, ks in GROUPS.items():
        thick_bars(s[s["組"] == g], None, {"ずれ_換算前_%": "換算前", **{f"ずれ_{temp_of(k)}_%": f"{TEMP_JA[temp_of(k)]}で換算（{k}）" for k in ks}},
                   f"{MAT_NAMES[mat]}　{group_ja(g)}\n(a) 厚みごとに係数を決め直したときの重ね合わせのずれ", "ずれ (%)", out_dir(mat, "シフトに使う温度"),
                   f"重ね合わせのずれ_厚みごとの係数_{g}", show=shown(g))''')

# ---------------------------------------------------------------- 6. (b) プロファイル尤度
md(r'''## 6. (b) 厚みごとのプロファイル尤度（GPR）

厚みごとの 5data だけで、シフトのあるモデルを 1 つずつ学習し、係数をシフトファクター検討用のノートと同じプロファイル尤度で決める（2026-11-03 からの、最良の点から外へたどる細かい格子と、全点の見直しも同じ。このノートは 95% 区間を使わないので、区間の延長はしない）。組ごとに、最大の対数周辺尤度 LML を比べ、ΔL = LML(Tff のモデル) − LML(Tset のモデル) を求める。あわせて、それぞれのシフトなし（係数 0）との差も出す（足した係数の効果。尤度比検定の 5% 基準は、係数が 2 つの B が 3.0 以上、1 つの A が 1.92 以上。表の「支持の基準」）。

**見方**：ΔL が正で大きいほど、その厚みでは流動中樹脂温度でずらしたほうが、学習データをよく説明する（負なら加熱シリンダ温度）。

**判定**：最良の係数が粗い格子の端に当たっていない／細かい格子の取り直しが収束した。''')
code(r'''# セル7：(b) 厚みごとのプロファイル尤度
PROF, PROF_PTS, BESTH, PL = [], [], {}, []
if THICK_PROFILE:
    t0 = time.time()
    for mat in MATERIALS:
        for h in thicknesses(TRAIN[mat]):
            for g, ks in GROUPS.items():
                for key in ks:
                    th_n, tv_n, unit = theta_names(kcfg(mat, key))
                    pts, b = thick_profile(key, TRAIN[mat], kcfg(mat, key), h, n_jobs=N_JOBS, restarts=PROFILE_RESTARTS, mode=INPUT_MODE)
                    BESTH[(mat, h, key)] = b
                    PROF.append({"材料": MAT_NAMES[mat], "厚み": h, "組": g, "モデル": key, "学習の条件数": sub_h(TRAIN[mat], h)["cond"].nunique(), "横の係数": b["th_best"],
                                 "縦の係数": b["tv_best"], "単位": unit, "LML": b["LML_best"], "係数0との差": b["dLML_vs_0"], "支持の基準": lrt_limit(key), "格子の端": b["at_edge"],
                                 "取り直しが収束": b["refine_converged"], "計算した点": b["n_points"], "見直しで上がった点": b["n_review_up"]})
                    PROF_PTS.append(pts.drop(columns="theta").assign(材料=MAT_NAMES[mat], 厚み=h, 組=g, モデル=key))
                    ok = not b["at_edge"] and b["refine_converged"]
                    why = "格子の端でなく、取り直しも収束" if ok else "、".join(
                        s for s in ["粗い格子の端に当たった" if b["at_edge"] else "",
                                    f"細かい格子の取り直しが {MAX_REFINE} 回で収束しなかった（最良の点が動き続けた。尤度の面が平らで、係数が決まりにくい）"
                                    if not b["refine_converged"] else ""] if s)
                    judge(ok, f"(b) {key} 厚み {h:g} mm：最良の係数 {th_n} {b['th_best']:g}・{tv_n} {b['tv_best']:g} {unit}。"
                          f"{why}（LML {b['LML_best']:.1f}、係数 0 との差 {b['dLML_vs_0']:.1f}）", mat, "6 (b) プロファイル尤度")
                lml = {temp_of(k): BESTH[(mat, h, k)]["LML_best"] for k in ks}
                d0 = {temp_of(k): BESTH[(mat, h, k)]["dLML_vs_0"] for k in ks}
                PL.append({"材料": MAT_NAMES[mat], "厚み": h, "組": g, "LML_Tset": lml.get("Tset", np.nan), "LML_Tff": lml.get("Tff", np.nan),
                           "ΔL（Tff − Tset）": lml.get("Tff", np.nan) - lml.get("Tset", np.nan), "係数0との差_Tset": d0.get("Tset", np.nan), "係数0との差_Tff": d0.get("Tff", np.nan),
                           "適した温度(b)": decide_temps(lml, DL_CLEAR, True)})
                print(f"  {MAT_NAMES[mat]} 厚み {h:g} mm・{g}：ΔL = {PL[-1]['ΔL（Tff − Tset）']:.1f}（経過 {time.time() - t0:.0f} 秒）")
    PROF, PL = pd.DataFrame(PROF), pd.DataFrame(PL)
    display(PROF.round(2))
    display(PL.round(2))
    for mat in MATERIALS:
        to_csv(PROF[PROF["材料"] == MAT_NAMES[mat]], os.path.join(out_dir(mat, "シフトに使う温度"), "プロファイル尤度_厚みごと.csv"))
        to_csv(pd.concat(PROF_PTS)[lambda d: d["材料"] == MAT_NAMES[mat]], os.path.join(out_dir(mat, "シフトに使う温度"), "プロファイル尤度_格子_厚みごと.csv"))
        p = PL[PL["材料"] == MAT_NAMES[mat]]
        to_csv(p, os.path.join(out_dir(mat, "シフトに使う温度"), "対数周辺尤度の差_厚みごと.csv"))
        for g, ks in GROUPS.items():
            labels = {f"係数0との差_{temp_of(k)}": f"{TEMP_JA[temp_of(k)]}（{k}）：シフトなしとの差" for k in ks}
            if len(ks) > 1:
                labels["ΔL（Tff − Tset）"] = "ΔL（流動中樹脂温度 − 加熱シリンダ温度）"
            thick_bars(p[p["組"] == g], None, labels, f"{MAT_NAMES[mat]}　{group_ja(g)}\n(b) 厚みごとの対数周辺尤度の差", "対数周辺尤度の差", out_dir(mat, "シフトに使う温度"),
                       f"対数周辺尤度の差_厚みごと_{g}", hline=DL_CLEAR, show=shown(g))
    print(f"合計 {(time.time() - t0) / 60:.1f} 分")
else:
    print("THICK_PROFILE = False のため、飛ばした（8 節の (b) と 9 節も飛ばす）")''')

# ---------------------------------------------------------------- 7. (c) 検証データの誤差
md(r'''## 7. (c) 検証データの誤差（厚みごと）

シフトファクター検討用のノートで保存した 3data の予測値（すべてのモデル。総合データで学習したモデル）を、厚みごとに集計する。予測誤差は、ln η の RMSE を粘度の % に換算した値（e^RMSE − 1）。図は組ごと（その組のモデルと C を並べる。`検証誤差_厚みごと_<組>.png`）。

**見方**：誤差が小さい温度ほど、その厚みで、学習に用いていない射出率条件の予測に向いている。低射出率側（3・5 mm/s）の検証は、厚み 1.5・2.0 mm にしかない。

**判定**：全厚みをまとめた RMSE が、シフトファクター検討用のノートの指標の表（`シフトファクターの検討/予測検証_3data_指標.csv`）と一致する。''')
code(r'''# セル8：(c) 検証データの誤差
VAL = []
for mat in MATERIALS:
    met = pd.read_csv(os.path.join(study_dir(mat), "予測検証_3data_指標.csv"), encoding="utf-8-sig")
    for key in KEYS:
        p = PRED[(mat, key)]
        for h in thicknesses(p):
            VAL += err_rows(sub_h(p, h), {"材料": MAT_NAMES[mat], "モデル": key, "学習": "総合", "厚み": h})
        allr = err_rows(p, {"材料": MAT_NAMES[mat], "モデル": key, "学習": "総合", "厚み": "全厚み"})
        VAL += allr
        ref = float(met[(met["モデル"] == key) & (met["範囲"] == "全体")]["RMSE"].iloc[0])
        judge(np.isclose(allr[0]["RMSE"], ref), f"(c) {key}：全厚みの RMSE（{allr[0]['RMSE']:.4f}）が、シフトファクター検討用のノートの指標の表（{ref:.4f}）と一致", mat, "7 (c) 検証データ")
VAL = pd.DataFrame(VAL)
VC = []
for mat in MATERIALS:
    v = VAL[(VAL["材料"] == MAT_NAMES[mat]) & (VAL["範囲"] == "全体") & (VAL["厚み"] != "全厚み")]
    w = v.pivot(index="厚み", columns="モデル", values="誤差_%").reset_index()
    w["厚み"] = w["厚み"].astype(float)
    w = w.sort_values("厚み").reset_index(drop=True)
    for g, ks in GROUPS.items():
        for _, r in w.iterrows():
            e = {temp_of(k): float(r[k]) for k in ks}
            VC.append({"材料": MAT_NAMES[mat], "厚み": float(r["厚み"]), "組": g, **{f"誤差_{t}_%": x for t, x in e.items()}, "誤差_C_%": float(r["C"]),
                       "適した温度(c)": decide_temps(e, ERR_CLEAR, False)})
        thick_bars(w, None, {"C": "C（シフトなし）", **{k: f"{k}（{TEMP_JA[temp_of(k)]}）" for k in ks}},
                   f"{MAT_NAMES[mat]}　{group_ja(g)}\n(c) 検証データの予測誤差（厚みごと）", "予測誤差 (%)", out_dir(mat, "シフトに使う温度"), f"検証誤差_厚みごと_{g}", show=shown(g))
    to_csv(VAL[VAL["材料"] == MAT_NAMES[mat]], os.path.join(out_dir(mat, "シフトに使う温度"), "検証誤差_厚みごと.csv"))
VC = pd.DataFrame(VC)
display(VC.round(2))
display(VAL[VAL["厚み"] != "全厚み"].pivot_table(index=["材料", "厚み", "範囲"], columns="モデル", values="誤差_%", sort=False).round(1))''')

# ---------------------------------------------------------------- 8. まとめ
md(r'''## 8. まとめ：厚みごとに適した温度

(a)(b)(c) の判定を、材料・厚み・組ごとに並べ、多数決で「Tff（流動中樹脂温度）／Tset（加熱シリンダ温度）／差なし」を決める（同数なら「差なし」。「—」は数えない。どれも「—」なら「—」）。`シフトに使う温度/厚みごとの比較_まとめ.csv` に保存する。''')
code(r'''# セル9：まとめ（Tff か Tset か）
ON = ["材料", "厚み", "組"]
SUMT = SUP.reindex(columns=ON + ["ずれ_Tset_%", "ずれ_Tff_%", "適した温度(a)"]).merge(
    VC.reindex(columns=ON + ["誤差_Tset_%", "誤差_Tff_%", "適した温度(c)"]), on=ON, how="left")
if len(PL):
    SUMT = SUMT.merge(PL[ON + ["ΔL（Tff − Tset）", "適した温度(b)"]], on=ON, how="left")
else:
    SUMT["ΔL（Tff − Tset）"], SUMT["適した温度(b)"] = np.nan, "—"
SUMT["総合の判定"] = [majority([a, b, c]) for a, b, c in zip(SUMT["適した温度(a)"], SUMT["適した温度(b)"], SUMT["適した温度(c)"])]
SUMT = SUMT[ON + ["ずれ_Tset_%", "ずれ_Tff_%", "適した温度(a)", "ΔL（Tff − Tset）", "適した温度(b)", "誤差_Tset_%", "誤差_Tff_%", "適した温度(c)", "総合の判定"]]
display(SUMT.round(2))
for mat in MATERIALS:
    s = SUMT[SUMT["材料"] == MAT_NAMES[mat]]
    to_csv(s, os.path.join(out_dir(mat, "シフトに使う温度"), "厚みごとの比較_まとめ.csv"))
    for g in GROUPS:
        sg = s[s["組"] == g]
        print(f"【{MAT_NAMES[mat]}・{g}〔{group_ja(g)}〕】" + "、".join(f"{h:g} mm：{j}" for h, j in zip(sg["厚み"], sg["総合の判定"])))''')

# ---------------------------------------------------------------- 9. 厚みごとのモデルと総合データのモデル
md(r'''## 9. 厚みごとのモデルと総合データのモデルの比較

学習 5data・検証 3data のまま、厚みごとの 5data だけで、すべてのモデル（`KEYS`。C を含む）を学習し（シフトのあるモデルの係数は 6 節で決めたもの。最終モデルは `N_RESTARTS` 回の最適化。`N_JOBS` で並列に計算する）、同じ厚みの 3data を予測する。総合データのモデル（全厚みの 5data で学習した、シフトファクター検討用のノートのモデル）の予測と、モデルごとに、厚みごと・全厚みで比べる。

**見方**：予測誤差が、どちらかの学習のしかたで、多くの厚みで `ERR_CLEAR` ポイント以上小さければ、その学習のしかたが向いている。差が小さければ、データの量（1 つのモデルに使える条件の数）と使いやすさ（総合モデルは 1 つで全厚みを予測できる）から、総合データで学習するほうがよい。

**判定**：すべての厚み・モデルで学習と予測ができた／総合データのモデルの値が 7 節と一致する。''')
code(r'''# セル10：厚みごとのモデルと総合データのモデル
CMP, TPRED, REC = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
if THICK_MODELS and THICK_PROFILE:
    from joblib import Parallel, delayed
    t0 = time.time()
    jobs = [(mat, h, key) for mat in MATERIALS for h in thicknesses(TRAIN[mat]) for key in KEYS]
    res = Parallel(n_jobs=min(N_JOBS, len(jobs)))(
        delayed(thick_model_job)(key, TRAIN[mat], TEST[mat], kcfg(mat, key), h, BESTH.get((mat, h, key)) if key != "C" else None, N_RESTARTS, INPUT_MODE)
        for mat, h, key in jobs)
    TPRED = pd.concat([pr.assign(材料=MAT_NAMES[mat], モデル=key, 学習="厚みごと") for (mat, h, key), pr in zip(jobs, res)], ignore_index=True)
    print(f"厚みごとのモデルの学習：{len(jobs)} 個、{(time.time() - t0) / 60:.1f} 分")
    rows = []
    for mat in MATERIALS:
        for key in KEYS:
            p = TPRED[(TPRED["材料"] == MAT_NAMES[mat]) & (TPRED["モデル"] == key)]
            judge(len(p) == len(TEST[mat]), f"9 {key}：厚みごとのモデルで、3data の全行（{len(p)}/{len(TEST[mat])}）を予測できた", mat, "9 厚みごと・総合")
            for h in thicknesses(p):
                rows += err_rows(sub_h(p, h), {"材料": MAT_NAMES[mat], "モデル": key, "学習": "厚みごと", "厚み": h})
            rows += err_rows(p, {"材料": MAT_NAMES[mat], "モデル": key, "学習": "厚みごと", "厚み": "全厚み"})
    CMP = pd.concat([pd.DataFrame(rows), VAL], ignore_index=True)
    CMP["厚みの名前"] = [h if isinstance(h, str) else f"{h:g}" for h in CMP["厚み"]]     # 数と「全厚み」が混ざると並べ替えられないので、表にするときは文字で
    rec = []
    for mat in MATERIALS:
        for key in KEYS:
            c = CMP[(CMP["材料"] == MAT_NAMES[mat]) & (CMP["モデル"] == key) & (CMP["範囲"] == "全体")]
            w = c.pivot(index="厚みの名前", columns="学習", values="誤差_%")
            hs = [h for h in w.index if h != "全厚み"]
            d = w.loc[hs, "厚みごと"] - w.loc[hs, "総合"]
            n_thick, n_total = int((d <= -ERR_CLEAR).sum()), int((d >= ERR_CLEAR).sum())
            verdict = "厚みごと" if n_thick > n_total and n_thick >= len(hs) / 2 else ("総合" if n_total > n_thick and n_total >= len(hs) / 2 else "差が小さい（総合を勧める）")
            rec.append({"材料": MAT_NAMES[mat], "モデル": key, "厚みの数": len(hs), "厚みごとがよい厚み": n_thick, "総合がよい厚み": n_total,
                        "全厚みの誤差_厚みごと_%": float(w.loc["全厚み", "厚みごと"]), "全厚みの誤差_総合_%": float(w.loc["全厚み", "総合"]), "勧める学習": verdict})
            fig_df = w.loc[hs].reset_index().rename(columns={"厚みごと": "厚みごとのモデル", "総合": "総合データのモデル"})
            fig_df["厚み"] = fig_df["厚みの名前"].astype(float)
            fig_df = fig_df.sort_values("厚み")
            thick_bars(fig_df, None, {"総合データのモデル": "総合データで学習", "厚みごとのモデル": "厚みごとに学習"},
                       f"{MAT_NAMES[mat]}　{key}：厚みごとのモデルと総合データのモデルの予測誤差", "予測誤差 (%)", out_dir(mat, "厚みごとのモデルと総合の比較"), f"厚みごとと総合_{key}",
                       show=SHOW_FIGS and key in (MAIN_KEY, "C"))
    REC = pd.DataFrame(rec)
    display(REC.round(2))
    display(CMP[CMP["範囲"] == "全体"].pivot_table(index=["材料", "モデル", "厚みの名前"], columns="学習", values="誤差_%", sort=False).round(1))
    for mat in MATERIALS:
        to_csv(CMP[CMP["材料"] == MAT_NAMES[mat]], os.path.join(out_dir(mat, "厚みごとのモデルと総合の比較"), "厚みごとと総合の比較.csv"))
        to_csv(REC[REC["材料"] == MAT_NAMES[mat]], os.path.join(out_dir(mat, "厚みごとのモデルと総合の比較"), "厚みごとと総合の比較_まとめ.csv"))
        to_csv(TPRED[TPRED["材料"] == MAT_NAMES[mat]], os.path.join(out_dir(mat, "厚みごとのモデルと総合の比較"), "厚みごとのモデル_予測値.csv"))
    if len(PROF):
        print("\n厚みごとに決めた係数（6 節。横, 縦）と、総合データのモデルの係数：")
        for mat in MATERIALS:
            for key in SHIFT_KEYS:
                p = PROF[(PROF["材料"] == MAT_NAMES[mat]) & (PROF["モデル"] == key)]
                print(f"  {MAT_NAMES[mat]}・{key}：" + "、".join(f"{h:g} mm ({a:g}, {b:g})" for h, a, b in zip(p["厚み"], p["横の係数"], p["縦の係数"])) +
                      f"／総合 ({FITS[(mat, key)]['th']:g}, {FITS[(mat, key)]['tv']:g})")
else:
    print("THICK_MODELS または THICK_PROFILE が False のため、飛ばした")''')

# ---------------------------------------------------------------- 10. 判定の一覧
md(r'''## 10. 判定の一覧

★ の行を確かめる。材料ごとに `厚みごと/判定の一覧.csv` に保存する。''')
code(r'''# セル11：判定の一覧
JT = pd.DataFrame(JUDGES)
print(f"判定 {len(JT)} 件のうち、★要確認 {int((JT['判定'] != 'OK').sum())} 件")
display(JT[JT["判定"] != "OK"])
for mat in MATERIALS:
    to_csv(JT[JT["材料"].isin([MAT_NAMES[mat], ""])], os.path.join(out_dir(mat), "判定の一覧.csv"))
print("結果の置き場所：" + "、".join(show_path(out_dir(m)) for m in MATERIALS))''')


# ---------------------------------------------------------------- 書き出し
def to_lines(s):
    lines = s.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


rnd = random.Random(20261022)
for c in cells:
    c["id"] = "%08x" % rnd.getrandbits(32)
    c["source"] = to_lines(c["source"])

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "ガウス過程回帰3 (Python 3.12)", "language": "python", "name": "gaussian-process3"},
        "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python",
                          "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.12.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

outs = [a for a in sys.argv[1:] if a != "--force"]
assert outs, "出力する .ipynb のパスを指定する"
assert len({c["id"] for c in cells}) == len(cells)
for c in cells:  # コードセルの構文チェック (%matplotlib の行は除く)
    if c["cell_type"] == "code":
        compile("".join(l for l in c["source"] if not l.startswith("%")), "<cell>", "exec")
for out in outs:
    if os.path.exists(out) and "--force" not in sys.argv[1:]:
        sys.exit(f"既にファイルがあるため上書きしません（上書きするときは --force）: {out}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as f:
        f.write((json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"作成: {out}（{len(cells)} セル）")
