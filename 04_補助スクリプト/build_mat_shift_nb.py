# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR（05_材料ごとの機械学習データと予測検証データ）の「学習用」のノートブック (未実行) を生成する。

使い方: python build_mat_shift_nb.py <出力する .ipynb のパス> [--force]
（成果物は 01_機械学習コードフォルダ/材料別_温度シフトファクター検討/材料別_温度シフト機械学習コード/ に置く〔2026-10-28 のユーザー依頼〕。ノートは、置いた場所から上へたどってリポジトリ直下を探すので、どこに置いても動く）
既存ファイルがある場合は、--force を付けない限り上書きしない。
本体の検討（手順1〜5b）とは別のプロジェクト。説明変数は γ̇・P3_3・Tff・Tset（ユーザー指定、2026-10-14）。
2026-10-27（ユーザー依頼）：シフトファクターに使う列を設定 SHIFT_COL で決めて、その 1 つのモデルを学習する形にした。列ごとに比べる部分は、
シフトファクター検討用のノート（build_mat_shiftfactor_nb.py）に移した。2 つのノートで同じセルの中身は mat_nb_cells.py にある。
読むデータと結果の置き場所はダイアログで選び、図は 600 dpi だけを保存する。
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
COMMON_CELL = "# セル：共通の部品（読み込み・問題データ一覧・シフトファクター・モデル・評価・図）。04_補助スクリプト/mat_common.py の中身。直すときは、そのファイルを直して、ノートを作り直す\n" + _common
PDF = "05_材料ごとの機械学習データと予測検証データ/学習結果説明_材料別温度シフトGPR.pdf"
STUDY_NB = "../材料別_温度シフトファクター検討用コード/材料別_温度シフトファクターの検討_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb"     # このノートの場所からの相対パス
THICK_NB = "../材料別_温度シフトファクター検討用コード/材料別_温度シフトGPR_ARD対応Matern52＋線形＋ホワイトノイズ_厚みごとの比較.ipynb"

# ---------------------------------------------------------------- 0. タイトル
md(rf'''# 材料別の温度シフト GPR（ARD Matern 5/2 ＋ 線形 ＋ ホワイトノイズ）：学習用

`05_材料ごとの機械学習データと予測検証データ` の材料ごとのデータ（**5data で学習し、3data で予測検証**）で、**温度のシフトファクターを表せる**ガウス過程回帰（GPR）の樹脂粘度モデルを作る。6月のノート（`GPR6月_ARD対応Matern52カーネル ＋ ホワイトノイズ.ipynb`）を基準にした。

**シフトファクターに使うデータ列は、実行の前に、設定エリアの `SHIFT_COL` で決める**（このノートは、その 1 つのモデルを学習して、図と表を一式作る）。どの列が適しているかは、別のノート **シフトファクター検討用**（`{STUDY_NB}`）で調べる（2026-10-27 に、このノートから分けた）。

- **目的変数**：樹脂粘度 η34 [Pa·s]（ln をとる）
- **説明変数**：せん断速度 γ34 [1/s]（ln）、流動中樹脂圧力 P3_3 [MPa]（Pa に直して ln）、流動中樹脂温度 Tff [℃]（K に直して ln。6月と同じ）、加熱シリンダ温度 Tset [℃]。**キャビティ厚み H は入れない**（ユーザー指定、2026-10-14）。γ̇・P3_3・Tff は、設定 `INPUT_MODE` で、ショットごとの値（`"shot"`。**既定**。6月のノートと同じ入れ方。1 ショットを 1 つのデータとして学習する。結果は `全ショット版/`）か、条件ごとの平均（`"cond"`。参考。結果は `条件ごとの平均版/`）かを選ぶ。注意：ショットごとの入力では、GP が同じ条件の中のゆれ（ショットごとの γ̇ と η が一緒に動く）に合わせるので、ln γ̇ の長さスケールが短くなり、曲線が波打って、3data の低速側の予測が外れやすい（10-15 の比較）
- **材料**：PP（01_バージンPP再々実験）、粉砕材混合 PP（02_粉砕材混合PP実験）、GPPS（03_GPPS実験）

**モデル**（記号は、ノートと結果のフォルダの名前で使う。本体の検討の B_1 などとは別物）

| 設定 `SHIFT_COL` | モデル | 式 | 結果のフォルダ |
|---|---|---|---|
| `"加熱シリンダ温度[℃]"`（既定） | **B_Tset** | ln η = ln a_v(Tset) ＋ f( ln γ̇ ＋ ln a_h(Tset), ln P3_3, ln Tff ) | `B_Tset_温度シフトに加熱シリンダ温度採用/` |
| `"流動中樹脂温度(℃)"` | **B_Tff** | ln η = ln a_v(Tff) ＋ f( ln γ̇ ＋ ln a_h(Tff), ln P3_3, Tset ) | `B_Tff_温度シフトに型内樹脂温度採用/` |
| `None` | **C** | ln η = f( ln γ̇, ln P3_3, ln Tff, Tset ) | `C_温度シフトなし/` |

列名は、学習ファイルの列名のとおりに書く（GPPS のファイルの `"加熱シリンダ温度(℃)"` と書いてもよい。短い名前 `"Tset"`・`"Tff"` も使える）。温度でない列を書くと、エラーで止まる。`SHIFT_STRUCT = "A"` にすると、縦だけずらす対案（A_Tset・A_Tff）になる。型内樹脂温度は、流動中樹脂温度 Tff のこと（発表資料では「金型内樹脂最高温度」）。

f は GP（ARD Matern 5/2 ＋ 線形項 ＋ ホワイトノイズ）。**横のシフト** a_h は「温度が変わると、同じ粘度になるせん断速度が a_h 倍ずれる」、**縦のシフト** a_v は「同じせん断速度で、粘度が a_v 倍になる」を表す（Tref で 1）。
- PP・粉砕材混合 PP：**アレニウス型** ln a = (E/R)(1/T − 1/Tref)。θ = E_h・E_v [kJ/mol]（「見かけのシフトエネルギー」。文献の活性化エネルギーとは別物）。
- GPPS：**WLF 型** ln a = −A1·[(T − D2)/(A2 + T − D2) − (Tref − D2)/(A2 + Tref − D2)]、A2 = 51.6 K、D2 = Tg（仮置き 100 ℃）を固定し、θ = A1_h・A1_v を推定。

**推定**：θ はプロファイル尤度で決める（θ の格子の各点で GPR を学習し、対数周辺尤度 LML が最大の θ を採る。最大から 1.92 以内をおおよその 95% 区間とする）。粗い格子で全体を見て、最良の点を中心に細かい格子を取り、最良の点が動いたら（斜めの尾根の上で、細かい格子の端に当たったとき）、その点を中心に取り直す（最大 `MAX_REFINE` 回）。横と縦のシフトは、別々には決まりにくく、**差 θ_v − θ_h** が最もよく決まる（尤度の山が斜めの帯になる）。標準化と中心化は、学習データ（5data）だけで、θ によらない定数で行う。θ = 0（シフトなし）との LML の差が 3.0 以上なら、2 つの係数を足した効果が、尤度比検定の 5% 基準（2·ΔLML > 5.99）で認められる。

**読むときの注意**：P3_3 と Tff は、流れの結果として測られる量で、成形の前には分からない。このモデルは「成形条件 → 粘度」ではなく、「**測った流れの状態（γ̇・P3_3・Tff）と設定温度 → 粘度**」のモデル。3data の予測にも、3data で測った P3_3・Tff を使う。''')

# ---------------------------------------------------------------- 1. 使い方・設定
md(f'''## 使い方（このノートを一人で実行して、結果を判断するために）

1. **実行の前に**、最初の「設定エリア」のセルで、シフトファクターに使う列 `SHIFT_COL` を決める（上の表。迷ったら、先にシフトファクター検討用のノートを実行し、最後のまとめに出る「SHIFT_COL の候補」を見る）。入力の取り方 `INPUT_MODE` と、学習する材料 `MATERIALS` も、ここで決める。ほかの設定を変えるときも、このセルだけを直して、最初から実行し直す（時間を短くする：`PROFILE_RESTARTS = 0` など）。
2. メニューの「すべて実行」で、上から順に実行する。カーネルは、このプロジェクトの `.venv` を選ぶ。
3. **実行の始めに（「2. データ」のセルで）、材料ごとにダイアログが 3 回開く**：①学習データ（5data）の CSV、②検証データ（3data）の CSV、③学習結果フォルダ（結果を保存するフォルダ。新しく作るときは、ダイアログの「新しいフォルダー」で作る。**材料ごとに別のフォルダ**を選ぶ）。3 材料なら 9 回。全部選ぶと、あとは最後まで自動で進む（目安：3 材料で、ショットごとの入力は約 25 分、条件平均の入力は約 45 分。大部分は「5. プロファイル尤度」。C は約 10 分）。
   - ダイアログが見当たらないときは、VS Code の後ろに隠れているので、タスクバーで探す。キャンセルすると、そこで止まる（そのセルから実行し直す）。
   - 毎回同じファイルを使うときは、設定 `FILES` に書くと、ダイアログは出ない（プロジェクトの外のファイルは、ここに書かずにダイアログで選ぶ）。
4. 結果は、選んだ学習結果フォルダの中に、次の形で保存される（同じフォルダを選んで実行し直すと、上書きされる）。
   - `データ確認/`：データの概要・外れ値の検知・P3_3 の性質（入力の取り方によらない）
   - `全ショット版/`（`INPUT_MODE = "shot"`）または `条件ごとの平均版/`（`"cond"`）の下の **`全厚み総合/<モデルの記号_説明>/`**（例：`B_Tset_温度シフトに加熱シリンダ温度採用/`）：このノートの結果のすべて（図・表・判定の一覧・まとめ・学習済みモデル）。中に `せん断速度_粘度/`・`せん断速度_粘度_拡張版/`・`依存性グラフ_セル9-2/`・`3次元グラフ/`。
   - 同じ版の `全厚み総合/シフトファクターの検討/` には、シフトファクター検討用のノートの結果、`厚みごと/` には、厚みごとの比較のノート（`{THICK_NB}`）の結果が入る。
   - **図は、すべて 600 dpi の 1 枚だけ**（報告書・発表資料にそのまま使える。設定 `FIG_DPI`）。
5. 各節の最後に、**判定**が出る（`>> 判定：OK` か `>> 判定：★要確認`）。OK は「想定どおり」、★ は「想定と違う、または結果を読むときに注意が要る点」。基準は、判定の文の中に書いてある。★ が出ても、エラーではない。最後の「まとめ」に一覧が出る。
6. **データを取り直したとき・入れ替えたとき**：新しい CSV を、ダイアログで選ぶ（ファイルの名前と場所は自由。列名は今のまま）。問題のあるデータは `05_材料ごとの機械学習データと予測検証データ/問題データ一覧.csv` に 1 行ずつ書く（どの場所のファイルを選んでも、この一覧を使う。区分「除外」の行は、ノートが除く。「ショット」列にショット番号を書くと、そのショットだけ。`2;3` のように複数書ける）。その後、最初から実行し直す。
7. 外れ値：「3. 外れ値の検知」で、規則 f（条件平均が、ほかの条件からの予測と大きく違う）で拾った条件が一覧に未登録なら ★ が出る。PP・粉砕材混合 PP は、一覧に「除外」で登録して実行し直す（4 件以上なら相談）。GPPS は参考表示だけ。
8. **せん断速度 ― 粘度のグラフ**（6月のノートのセル9-2 と同じ見せ方）と **3次元グラフ**のセルは、セルの先頭に設定エリアがあり、**そこだけを直して、そのセルだけを実行し直せる**（全体を一度実行した後で。学習はやり直さない）。
9. 用語や図の見方は、説明書 `{PDF}` にある（10-26 までの結果で書いたもの。結果のフォルダの名前は、そのときの構成のまま）。''')
md(r'''## 1. 設定''')
code(r'''# セル1：設定エリア  ▼▼▼ 変えてよいのは、ここだけ ▼▼▼
from pathlib import Path      # （下の REPO で、フォルダをたどるのに使う）
SHIFT_COL = "加熱シリンダ温度[℃]"   # シフトファクターに使うデータ列（学習ファイルの列名）："加熱シリンダ温度[℃]"（Tset。GPPS のファイルの書き方 "加熱シリンダ温度(℃)" でもよい）／"流動中樹脂温度(℃)"（Tff。型内樹脂温度）／None（シフトなし。4 変数を並べた通常の GPR〔C〕）。どれが適しているかは、シフトファクター検討用のノートで調べる
SHIFT_STRUCT = "B"            # シフトの形："B"＝横・縦を別々に推定／"A"＝縦だけ（対案）
SEED = 42                     # 乱数シード
NOISE_LB = 1e-6               # ノイズ分散の下限（6月のノートは 1e-3）
USE_LINEAR = True             # True：6月のカーネル（ARD Matern 5/2 ＋ ホワイトノイズ）に線形項（DotProduct）を足す（3data は射出速度の両端の外挿のため）／False：6月と同じ
INPUT_MODE = "shot"           # "shot"：γ̇・P3_3・Tff をショットごとの値で入れる（6月のノートと同じ入れ方。既定。ユーザー決定、10-23。結果は 全ショット版/）／"cond"：条件（設定温度・厚み・射出速度）ごとの平均で入れる（参考。結果は 条件ごとの平均版/）
N_RESTARTS = 20               # 最終モデルの学習での、最適化のやり直し回数（6月は 5 回。ハイパーパラメータの尤度の面に局所解が多く、5 回では、尤度の高い解に着かないことがあったので 20 回にした。LML が最大の解を採る）
PROFILE_RESTARTS = 1          # プロファイル尤度の格子の各点での、やり直し回数（0 にすると約半分の時間。尤度の面ががたつきやすい）
USE_EXCLUSIONS = True         # True：問題データ一覧（05 フォルダ）の「除外」を除く／False：全部使う（0 以下の値の行だけは、対数がとれないので除く）
REPO = str(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "04_補助スクリプト").is_dir()))   # リポジトリ直下（このノートの場所から上へたどって、04_補助スクリプト があるフォルダを探す）
MATERIALS = ["PP", "MIX", "GPPS"]    # 学習する材料（PP：バージン PP、MIX：粉砕材混合 PP、GPPS）。材料ごとに、実行の始めにダイアログで、学習データ・検証データの CSV と、学習結果フォルダを選ぶ
MAT_FOLDERS = {"PP": "01_バージンPP再々実験", "MIX": "02_粉砕材混合PP実験", "GPPS": "03_GPPS実験"}   # 05 フォルダの中の材料フォルダ名（ダイアログの最初のフォルダ）
FILES = {}                    # ダイアログを使わずに決めるとき：{"PP": {"学習": r"…\結合_03_pp_max_5data_外れ値なし.csv", "検証": r"…", "結果": r"…"}}（相対パスはリポジトリ直下から）。書いていないものは、ダイアログで選ぶ
SHIFT_CFG = {                 # 材料ごとのシフトファクターの型、基準温度 Tref [℃]（None なら中央の設定温度）、尤度を調べる範囲（粗い格子：(下限, 上限, 刻み)。h が横、v が縦）
    "PP":   {"type": "arrhenius", "Tref": 210.0, "h_grid": (-20.0, 40.0, 5.0), "v_grid": (-20.0, 60.0, 5.0)},          # θ = E [kJ/mol]
    "MIX":  {"type": "arrhenius", "Tref": 210.0, "h_grid": (-20.0, 40.0, 5.0), "v_grid": (-20.0, 60.0, 5.0)},
    "GPPS": {"type": "wlf", "Tref": 220.0, "Tg": 100.0, "A2": 51.6, "h_grid": (-20.0, 40.0, 5.0), "v_grid": (-20.0, 40.0, 5.0)},   # θ = A1 [−]。D2 = Tg [℃]（仮置き。分かれば直す）、A2 [K] は固定
}
FINE_B = (4.0, 1.0)           # B の細かい格子：最良の点の ±4 を 1 刻み
FINE_A = (2.5, 0.5)           # A の細かい格子：最良の点の ±2.5 を 0.5 刻み
MAX_REFINE = 4                # 最良の点が細かい格子の端に当たったとき、その点を中心に細かい格子を取り直す回数の上限（取り直しても動かなければ、そこで終わり）
OUTLIER_REGISTER = ["PP", "MIX"]   # 外れ値（規則 f）を問題データ一覧に登録して除く材料（GPPS は参考表示だけ。ユーザー決定、10-14）
CWLF_T = "Tff"                # 参考の Cross-WLF に使う温度（6月と同じく流動中樹脂温度。"Tset" も可）
CWLF_STARTS = 12              # 参考の Cross-WLF の当てはめの初期値の数（6月に近い値 1 つ ＋ 乱数。残差が最小のものを採る）
CWLF_FIXED = {                # Cross-WLF の係数（D3 = 0）。数＝固定、None＝5data で探索。D2 は K
    "PP":   {"n": None, "tau_star": None, "D1": None, "D2": None, "A1": 30.9, "A2": 43.4486},     # 6月のノートと同じ固定値
    "MIX":  {"n": None, "tau_star": None, "D1": None, "D2": None, "A1": 30.9, "A2": 43.4486},
    "GPPS": {"n": None, "tau_star": None, "D1": None, "D2": 373.15, "A1": None, "A2": 51.6},      # D2 = Tg（仮置き 100 ℃）
}
N_JOBS = 8                    # プロファイル尤度を並列に計算する数（CPU の数に合わせる）
FIG_DPI = 600                 # 保存する図の解像度（すべての図を、この解像度の 1 枚だけで保存する。報告書・発表資料用）
SHOW_FIGS = True              # True：図をノートにも表示する（False：保存だけ）
# ▲▲▲ 設定エリア終了 ▲▲▲''')
code("# セル2：ライブラリ、フォント\n" + nbc.LIBS)
md(r'''### 共通の部品

ファイルを選ぶダイアログ、データの読み込み（加熱シリンダ温度の列がないファイルは、元ファイル名から取る）、問題データ一覧の「除外」、結果の置き場所、シフトファクター（アレニウス型・WLF 型）、モデルの学習と予測、プロファイル尤度、評価、外れ値の検知、Cross-WLF、図の保存。中身は `04_補助スクリプト/mat_common.py` と同じ。''')
code(COMMON_CELL)
md(r'''### 学習するモデルと、結果の置き場所

設定 `SHIFT_COL`・`SHIFT_STRUCT` から、学習するモデル（1 つ）を決める。結果は、学習結果フォルダの `<版>/全厚み総合/<モデルのフォルダ>/` にまとめる（版は `INPUT_MODE` で決まる：`全ショット版` か `条件ごとの平均版`）。''')
code(r'''# セル2b：学習するモデルと、結果の置き場所
KEY = shift_key(SHIFT_COL, SHIFT_STRUCT)       # シフトファクターに使う列 → モデルの記号（加熱シリンダ温度 → B_Tset、流動中樹脂温度 → B_Tff、None → C）
KEYS = [KEY]                                   # このノートで学習するモデル（1 つ）
SHIFT_KEYS = [k for k in KEYS if k != "C"]     # そのうち、シフトのあるモデル（C なら空）
MAIN_KEY = KEY                                 # 図をノートに表示するモデル


def out_dir(mat, key=None, sub=None):
    """このノートの結果の置き場所：<学習結果フォルダ>/<版>/全厚み総合/<モデルのフォルダ>/（sub があれば、その下のフォルダ）。
    モデルが 1 つなので、key によらず同じフォルダ（モデルの表・Cross-WLF との比較・判定の一覧・まとめも、ここに置く）"""
    return result_dir(mat, VERSION_DIRS[INPUT_MODE], THICK_TOTAL, KEY, *([sub] if sub else []))


print(f"学習するモデル：{model_title(KEY)}（シフトファクターに使う列：{SHIFT_COL}）／材料：{MATERIALS}")
print(f"結果の置き場所：<学習結果フォルダ>/{VERSION_DIRS[INPUT_MODE]}/{THICK_TOTAL}/{MODEL_DIRS[KEY]}/")''')

# ---------------------------------------------------------------- 2. データ
md(r'''## 2. ファイルを選んで、データを読み込む

**このセルで、材料ごとにダイアログが 3 回開く**（①学習データ〔5data〕の CSV、②検証データ〔3data〕の CSV、③学習結果フォルダ。設定 `FILES` に書いたものは開かない）。選んだファイルと、読んだデータの概要を表で示す。問題データ一覧の「除外」の行と、η・γ̇・P3_3 が 0 以下の行（対数がとれない）を除く。検証の行は、学習データの射出速度の範囲（10〜240 mm/s）より遅い側を**低速側**（3・5 mm/s）、速い側を**高速側**（280・320 mm/s）に分ける。

**判定**：設定温度が学習・検証とも 3 水準ある／0 以下の値の行がすべて一覧に登録済み（未登録なら自動で除いて ★）／基準温度 Tref が設定温度の範囲の中にある（外なら、材料とファイルの組み合わせを確かめる）／検証データに、学習データの射出速度の範囲の外の条件がある（ないなら、学習と検証のファイルを逆に選んだおそれがある）。''')
code("# セル3：ファイルを選んで、読み込む\n" + nbc.READ)

# ---------------------------------------------------------------- 3. 外れ値
md(r'''## 3. 外れ値の検知（これまでと同じ規則）

**規則 f**（本体の `check_data.py` と同じ）：5data と 3data をまとめた条件平均の点で、GPR（入力 ln γ̇〔条件平均〕・ln H・Tset、線形項あり）を学習し、条件を 1 つずつ抜いた予測と実測の差（残差）を、予測の標準偏差で割った **z** を求める。**|z| > 3.5 かつ |残差| > 0.1**（粘度で約 10%）の条件を拾う。一覧の「除外」を除く前と後の両方で調べる。※ この検知だけは、データを確かめるための道具なので、これまでと同じく H を使う（モデルの説明変数ではない）。

**問題の候補**（見直しの候補であって、外れ値と判定したものではない。一覧に載せるかはユーザーが決める）：a 温度の間で γ̇ が 1.3 倍を超えて違う／b 最も低い設定温度の粘度が、最も高い温度より小さい／c 同じ条件内の ln η の標準偏差が 0.1 を超える／e 別のショットと値が完全に同じ行／g P3_3 が条件の中央値から 20% 以上離れたショット（P3_3 は新しい列なので足した）。

**判定**：PP・粉砕材混合 PP は、規則 f で拾った条件がすべて一覧に登録済みなら OK（未登録なら ★。一覧に「除外」で登録して、最初から実行し直す）。GPPS は参考表示だけ。

**図の見方**（`データ確認/外れ値の検知_規則f.png`）：横軸は条件（厚み → 射出速度 → 設定温度の順）、縦軸は z。破線（±3.5）の外で、赤い丸で囲んだ点が拾った条件。''')
code(r'''# セル4：外れ値の検知（規則 f）と、問題の候補
PLIST = read_problem_list()
D_ALL = load_all(MATERIALS, exclude=False)            # 一覧の「除外」を除く前
for mat in MATERIALS:
    chk_dir = result_dir(mat, "データ確認")
    pool_all = pd.concat([D_ALL[(mat, "5data")], D_ALL[(mat, "3data")]], ignore_index=True)
    pool_ex = pd.concat([TRAIN[mat], TEST[mat]], ignore_index=True)
    f_all, f_ex = outlier_scan(pool_all), outlier_scan(pool_ex)
    ftab = pd.concat([f_all.assign(除外後に再計算=False), f_ex.assign(除外後に再計算=True)], ignore_index=True)
    ftab.insert(0, "材料", MAT_NAMES[mat])
    to_csv(ftab, os.path.join(chk_dir, "外れ値の検知_規則f.csv"))
    flag = ftab[ftab["flag"]].drop_duplicates(subset=["cond"]).copy()
    flag["一覧の ID"] = [registered(mat, r["grp"], {"Tset": r["Tset"], "H": r["H"], "v": r["v"]}, PLIST) for _, r in flag.iterrows()]
    cand = candidates(mat, pool_ex, PLIST)
    to_csv(cand, os.path.join(chk_dir, "問題の候補.csv"))
    # 図：条件ごとの z（除く前の全条件）
    order = f_all.sort_values(["H", "v", "Tset"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    for t, (col, mk) in t_style(order["Tset"].unique()).items():
        m = (order["Tset"] == t).to_numpy()
        ax.scatter(np.flatnonzero(m), order.loc[m, "z"], color=col, marker=mk, s=40, edgecolors="k", linewidths=0.4, label=f"{int(t)} ℃", zorder=3)
    fl = order[order["flag"]]
    if len(fl):
        ax.scatter(fl.index, fl["z"], s=260, facecolors="none", edgecolors="red", linewidths=2, label="規則 f で拾った条件", zorder=4)
        for i, r in fl.iterrows():
            ax.annotate(f"{int(r['Tset'])} ℃・{r['H']:g} mm・{int(r['v'])} mm/s", (i, r["z"]), textcoords="offset points", xytext=(8, 4), fontsize=10, color="red")
    for z0 in (-OUTLIER_Z, OUTLIER_Z):
        ax.axhline(z0, color="gray", ls="--", lw=1)
    ax.axhline(0, color="k", lw=0.6)
    ymax = max(5.0, float(order["z"].abs().max()) * 1.1)
    for h, g in order.groupby("H"):
        ax.axvline(g.index.min() - 0.5, color="gray", lw=0.6)
        ax.text(g.index.to_numpy().mean(), -ymax * 0.93, f"厚み {h:g} mm", ha="center", fontsize=11)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("条件（厚み → 射出速度 → 設定温度の順）")
    ax.set_ylabel("z（実測 − 条件を抜いた予測）／予測の標準偏差")
    ax.set_title(f"{MAT_NAMES[mat]}　外れ値の検知（規則 f。一覧の「除外」を除く前の全 {len(order)} 条件）")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, ls=":", alpha=0.5)
    save_fig(fig, chk_dir, "外れ値の検知_規則f")
    txt = "、".join(f"{int(r['Tset'])} ℃・{r['H']:g} mm・{int(r['v'])} mm/s（{r['grp']}、z = {r['z']:.1f}、残差 {r['resid']:+.2f}、{r['一覧の ID'] or '未登録'}）" for _, r in flag.iterrows()) or "なし"
    unreg = flag[flag["一覧の ID"] == ""]
    if mat in OUTLIER_REGISTER:
        judge(len(unreg) == 0, f"規則 f（|z| > {OUTLIER_Z} かつ |残差| > {OUTLIER_RESID}）で拾った条件は、すべて問題データ一覧に登録済み：{txt}。未登録があれば、一覧に「除外」で登録して、最初から実行し直す（4 件以上なら相談）", mat, "外れ値")
    else:
        judge(len(flag) == 0, f"（参考）規則 f で拾った条件：{txt}。この材料は参考表示だけで、一覧には登録しない", mat, "外れ値")
    n_un = int((cand["一覧の ID"] == "").sum())
    print(f"  問題の候補：{len(cand)} 件（一覧に未登録 {n_un} 件。規則ごと {cand.groupby('規則').size().to_dict()}）→ データ確認/問題の候補.csv")
    if len(cand):
        display(cand[~cand["規則"].str.startswith("g")].head(20))''')

# ---------------------------------------------------------------- 4. P3_3
md(r'''## 4. 参考：P3_3 の性質（リークの目安）

本体の検討では、粘度の算出式に入っている圧力（P3_1）を説明変数に入れると、GPR が「算出式」を覚えてしまう（**リーク**）ことが分かっている。P3_3 がそれに当たるかを確かめる。
- **算出式の圧力か**：見かけのせん断応力 η·γ̇ を、a·H·P3_3 ＋ b·H（スリットの式の形）で表したときの最大の相対誤差。算出式の圧力なら 0.1% 未満になる（P3_1 は 0.03%）。
- **線形回帰の検証 R²**：ln η を ln P3_3 と ln γ̇ で線形回帰（5data で当てはめ）したときの 3data の R²。ln γ̇ だけの場合と並べる。
- P3_1・P3_2 の列があれば、ln P3_3 との相関（P3_2：センサ4が最大値になった時刻のセンサ3の圧力）。

**判定**：算出式の誤差が 5% を超えれば、P3_3 は算出式の圧力ではない（OK）。ただし、P3_3 は流れの結果で、成形の前には分からない値であることに変わりはない。''')
code(r'''# セル5：P3_3 の性質
rows = []
for mat in MATERIALS:
    chk_dir = result_dir(mat, "データ確認")
    pool = pd.concat([TRAIN[mat], TEST[mat]], ignore_index=True)
    tau = (pool["eta"] * pool["gd"]).to_numpy()                       # 見かけのせん断応力 [Pa]
    A = np.column_stack([pool["H"] * pool["P33"] * 1e6, pool["H"]])
    coef, *_ = np.linalg.lstsq(A, tau, rcond=None)
    rel = float(np.max(np.abs(A @ coef / tau - 1)))

    def r2_test(cols):
        Xtr = np.column_stack([np.ones(len(TRAIN[mat]))] + [TRAIN[mat][c] for c in cols])
        b, *_ = np.linalg.lstsq(Xtr, TRAIN[mat]["ln_eta"], rcond=None)
        Xte = np.column_stack([np.ones(len(TEST[mat]))] + [TEST[mat][c] for c in cols])
        return float(r2_score(TEST[mat]["ln_eta"], Xte @ b)), b

    r_pg, b_pg = r2_test(["ln_P33", "ln_gd"])
    r_g, _ = r2_test(["ln_gd"])
    row = {"材料": MAT_NAMES[mat], "算出式の形で表したときの最大誤差 [%]": round(rel * 100, 1),
           "検証 R²（ln P3_3・ln γ̇）": round(r_pg, 4), "係数（ln P3_3, ln γ̇）": f"({b_pg[1]:.2f}, {b_pg[2]:.2f})", "検証 R²（ln γ̇ だけ）": round(r_g, 4)}
    for c in ["P31", "P32"]:
        if c in pool.columns and pool[c].notna().all():
            row[f"相関 ln P3_3 と ln P3_{c[-1]}"] = round(float(np.corrcoef(pool["ln_P33"], np.log(pool[c]))[0, 1]), 3)
    rows.append(row)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for h in sorted(pool["H"].unique()):
        m = (pool["H"] == h).to_numpy()
        kw = dict(s=16, color=H_COLORS.get(h, "gray"), marker=H_MARKS.get(h, "o"), alpha=0.7, edgecolors="none", label=f"{h:g} mm")
        axes[0].scatter(pool.loc[m, "P33"], tau[m] / 1e3, **kw)
        axes[1].scatter(pool.loc[m, "gd"], pool.loc[m, "P33"], **kw)
    axes[0].set_xlabel("流動中樹脂圧力 P3_3 (MPa)")
    axes[0].set_ylabel("見かけのせん断応力 η·γ̇ (kPa)")
    axes[0].set_title("η·γ̇ と P3_3（算出式の圧力なら、厚みごとに直線に乗る）")
    axes[1].set_xlabel(r"せん断速度 $\dot{\gamma}$ (1/s)")
    axes[1].set_ylabel("流動中樹脂圧力 P3_3 (MPa)")
    axes[1].set_title("P3_3 とせん断速度")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        log_ticks(ax)                                 # 目盛りの数字が重ならないようにする
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(title="キャビティ厚み")
    fig.suptitle(f"{MAT_NAMES[mat]}　P3_3 の性質（5data と 3data）")
    save_fig(fig, chk_dir, "P3_3の性質")
    judge(rel > 0.05, f"P3_3 は粘度の算出式の圧力ではない（η·γ̇ = a·H·P3_3 + b·H で表すと、誤差が最大 {rel * 100:.0f}%。算出式の圧力なら 0.1% 未満）", mat, "P3_3")
P33_TAB = pd.DataFrame(rows)
display(P33_TAB)
for mat in MATERIALS:
    to_csv(P33_TAB[P33_TAB["材料"] == MAT_NAMES[mat]], os.path.join(result_dir(mat, "データ確認"), "P3_3の性質.csv"))''')

# ---------------------------------------------------------------- 5. プロファイル尤度・学習
md(r'''## 5. 学習

### 5.1 プロファイル尤度（シフトファクター θ の推定。時間がかかる）

シフトのあるモデル（B_Tset・B_Tff など）では、θ = (θ_h, θ_v)（アレニウス型は E_h・E_v [kJ/mol]、WLF 型は A1_h・A1_v）の格子の各点で GPR を学習し、対数周辺尤度（LML）を求める。粗い格子（設定 `SHIFT_CFG`）で全体を見て、最良の点を中心に細かい格子（`FINE_B`）を取って計算し直す。最良の点が動いたら（斜めの尾根の上では、細かい格子の端に当たることがある）、新しい最良の点を中心に取り直す（最大 `MAX_REFINE` 回）。前の格子点の解を初期値にして、局所解でのがたつきを減らす。C（`SHIFT_COL = None`）では、θ がないので、この節は計算しない。

**θ = 0 の意味**：シフトなし、つまり B_Tset なら「Tset を使わないモデル」、B_Tff なら「Tff を使わないモデル」。最良の LML と θ = 0 の LML の差が、その温度を（シフトの形で）入れる効果を表す。**差が 3.0 以上**なら、2 つの係数を足した効果が、尤度比検定の 5% 基準（2·ΔLML > 5.99）で認められる。

**尾根**：θ_h と θ_v は別々には決まりにくく、尤度の山が斜めの帯（尾根）になる。尾根に沿って尤度がほとんど変わらないので、個々の係数の 95% 区間は、計算した範囲に左右される。**差 θ_v − θ_h** が最もよく決まるので、表では、この差の範囲も示す。

**判定**：最良の θ が、粗い格子の端にも、細かい格子の端にもない（端なら `SHIFT_CFG` の範囲を広げる。取り直しが収束していなければ `MAX_REFINE` を増やす）。

### 5.1b 最良の θ と θ = 0 の、念入りな最適化

ハイパーパラメータ（振幅・長さスケール・線形項・ノイズ）の尤度の面には、**局所解が多い**。格子の各点では、最適化のやり直しが少ない（`PROFILE_RESTARTS`）ので、LML が 1〜3 ほど、局所解でずれることがある。そのまま「最良の点」と「θ = 0」の LML を比べると、**差がずれて見える**。そこで、**最良の θ と θ = 0 の 2 点だけ**を、最適化のやり直し `N_RESTARTS` 回で、念入りに学習し直す（並列。格子の解も初期値の 1 つにする）。最良の θ の学習が、そのまま最終モデルになる。θ = 0 との LML の差（ΔLML）は、この 2 つから求める。

**判定**：ΔLML が 3.0 以上（尤度比検定の 5% 基準。小さければ、温度の効果はほかの入力が担っている。そのときは、シフトファクター検討用のノートで、ほかの列や C と比べる）。''')
code("# セル6：プロファイル尤度（目安：材料 1 つで 5〜10 分）\n" + nbc.PROFILE)
code("# セル6b：最良の θ と θ = 0 の念入りな最適化（並列。目安：数分）\n" + nbc.CAREFUL)
md(r'''**図の見方**（`プロファイル尤度.png`）：色は LML − 最大値（明るいほど尤度が高い）。赤線の内側が、おおよその 95% 区間（最大から 1.92 以内）。★ が最良の点、× が θ = 0（シフトなし）。帯が斜めに伸びるときは、θ_h と θ_v が別々には決まりにくく、差 θ_v − θ_h だけがよく決まる（本体の手順5a と同じ）。''')
code("# セル7：尤度の面\n" + nbc.SURFACE)
md(r'''### 5.2 最終モデルの学習（6月のノートのセル5・6）

シフトのあるモデルは、5.1b の念入りな学習（最良の θ。最適化のやり直し `N_RESTARTS` 回）が、そのまま最終モデル。C（シフトなし）は θ がないので、ここで同じ回数で学習する。表は、学習したハイパーパラメータ（振幅、長さスケール、線形項、ノイズ）と対数周辺尤度（LML）。長さスケールが上限（100）に届いた入力は、粘度にほとんど効いていない（線形項が担う場合もある）。

**判定**：最終モデルの LML が、尤度の格子の最良以上／ノイズ分散が下限（`NOISE_LB`）に張り付いていない。''')
code("# セル8：最終モデルの学習（C。シフトのあるモデルは 5.1b で済み）\n" + nbc.NOSHIFT)
md(r'''### 5.3 損失関数の推移・特徴スケール・カーネル行列（6月のノートのセル7・7.5・8）

- **損失関数の推移**：負の対数周辺尤度（小さいほど良い）の、最適化のステップごとの値。青い太線が、最も良い解に着いた回（灰色は、ほかの初期値からの回）。
- **特徴スケール**：重要度 = 1 / 長さスケール（大きいほど、その入力の変化に粘度が敏感）。シフトのあるモデルでは、シフトに使った温度の効果は θ が担うので、ここには出ない。
- **カーネル行列**：学習データの点どうしの共分散（ノイズを除く）。並べる順を変えた 2 枚（左：設定温度 → 厚み → 射出速度、右：厚み → 設定温度 → 射出速度）。白線は、最初の量が変わる区切り。''')
code(r'''# セル9：損失関数の推移（6月のノートのセル7）
for mat in MATERIALS:
    oks = []
    for key in KEYS:
        runs = [r for r in LOSS[(mat, key)] if len(r)]
        fig, ax = plt.subplots(figsize=(10, 6))
        if runs:
            best = int(np.argmin([r[-1] for r in runs]))
            for i, r in enumerate(runs):
                if i != best:
                    ax.plot(r, marker="o", ms=2, lw=0.8, color="gray", alpha=0.5, label="ほかの初期値" if i == (1 if best == 0 else 0) else None)
            rb = runs[best]
            ax.plot(rb, marker="o", ms=4, lw=1.8, color="blue", label="最良の回")
            ax.text(len(rb) - 1, rb[-1], f"{rb[-1]:.2f}", ha="left", va="bottom", color="red", fontsize=12)
            lo, hi = min(min(r) for r in runs), np.percentile(np.concatenate(runs), 90)
            ax.set_ylim(lo - 0.05 * (hi - lo + 1), hi + 0.05 * (hi - lo + 1))
            oks.append(rb[-1] <= rb[0])
        ax.set_title(f"{MAT_NAMES[mat]}　{model_title(key)}\n学習プロセスにおける損失関数の推移（負の対数周辺尤度）")
        ax.set_xlabel("計算ステップ数 (Iteration)")
        ax.set_ylabel("損失（値が小さいほど良い）")
        ax.grid(True, which="both", ls="--", alpha=0.7)
        ax.legend()
        save_fig(fig, out_dir(mat, key), "損失関数の推移", show=SHOW_FIGS and key == MAIN_KEY)
    judge(all(oks), f"損失が減少して、学習が正常に進んだ（{len(oks)} モデル）", mat, "学習")''')
code(r'''# セル10：説明変数ごとの特徴スケール（6月のノートのセル7.5）
for mat in MATERIALS:
    for key in KEYS:
        prod, lin, _ = kernel_parts(FITS[(mat, key)]["gp"])
        ls = np.atleast_1d(prod.k2.length_scale)
        imp = pd.DataFrame({"説明変数": input_labels(key), "長さスケール": ls, "重要度（1/長さスケール）": 1.0 / ls}).sort_values("重要度（1/長さスケール）", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.barh(imp["説明変数"][::-1], imp["重要度（1/長さスケール）"][::-1], color=plt.cm.viridis(np.linspace(0.25, 0.85, len(imp))))
        ax.set_title(f"{MAT_NAMES[mat]}　{model_title(key)}\nGP の長さスケールによる重要度（ARD）")
        ax.set_xlabel("重要度（感度 = 1 / 長さスケール）")
        ax.grid(axis="x", ls="--", alpha=0.7)
        save_fig(fig, out_dir(mat, key), "特徴スケール", show=SHOW_FIGS and key == MAIN_KEY)
        print(f"{MAT_NAMES[mat]} {key}：" + "、".join(f"{r['説明変数']} {r['長さスケール']:.3g}" for _, r in imp.iterrows())
              + (f"（線形項の振幅 {np.sqrt(lin.k1.constant_value):.3g}）" if lin is not None else ""))''')
code(r'''# セル11：カーネル行列のヒートマップ（6月のノートのセル8）
for mat in MATERIALS:
    tr = TRAIN[mat]
    for key in KEYS:
        f = FITS[(mat, key)]
        _, Xs, _ = design(tr, key, f["cfg"], f["th"], f["tv"], f["mode"])
        Xn = (Xs - f["mu_x"]) / f["sd_x"]
        fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
        for ax, cols, ttl in [(axes[0], ["Tset", "H", "v"], "設定温度 → 厚み → 射出速度の順"), (axes[1], ["H", "Tset", "v"], "厚み → 設定温度 → 射出速度の順")]:
            idx = tr.sort_values(cols + ["shot"]).index.to_numpy()
            K = f["gp"].kernel_.k1(Xn[idx])                          # ノイズを除いた共分散
            im = ax.imshow(K, cmap="viridis", interpolation="nearest")
            first = tr.loc[idx, cols[0]].to_numpy()
            for bb in np.flatnonzero(first[1:] != first[:-1]) + 0.5:
                ax.axhline(bb, color="w", lw=0.8)
                ax.axvline(bb, color="w", lw=0.8)
            ax.set_title(ttl)
            ax.set_xlabel("学習データの行（並べ替え後）")
            ax.set_ylabel("学習データの行（並べ替え後）")
            fig.colorbar(im, ax=ax, shrink=0.8, label="共分散")
        fig.suptitle(f"{MAT_NAMES[mat]}　{model_title(key)}　カーネル行列（共分散行列）")
        save_fig(fig, out_dir(mat, key), "カーネル行列", show=SHOW_FIGS and key == MAIN_KEY)''')

# ---------------------------------------------------------------- 6. シフトファクター
md(r'''## 6. 温度のシフトファクターと、マスターカーブ

- **シフトファクターの図**（`シフトファクター.png`。C にはない）：左が横のシフト ln a_h(T)、右が縦のシフト ln a_v(T)。線が最良の θ、帯が尤度の 95% 区間（赤線の内側の θ の範囲）。点線は設定温度、薄い橙は Tff の範囲。B_Tset は設定温度、B_Tff は流動中樹脂温度で読む。ln a_v(T) > 0 は「その温度では、同じせん断速度で粘度が Tref より高い」。
- **マスターカーブ**（`マスターカーブ.png`）：条件平均の点を、最良の θ でずらしたもの（横：γ̇·a_h、縦：η/a_v）。温度のシフトが効いていれば、厚みごとに、温度の違う点が 1 本の曲線に近づく（シフト前は `流動曲線_シフト前.png`）。GP は、さらに P3_3 と、もう一方の温度でも補正する。
- **重ね合わせの残差**：厚みごとに、全温度の条件平均の点を 3 次式で当てはめた残差の RMSE（ln η）。シフト前より小さければ、シフトで曲線が重なった。

**判定**：縦のシフトの向きが「低温ほど粘い」（最も低い設定温度で ln a_v > 0）／重ね合わせの残差が、シフト前より小さい。''')
code("# セル12：シフトファクターとマスターカーブ\n" + nbc.SHIFT)

# ---------------------------------------------------------------- 7. 依存性
md(r'''## 7. 依存性グラフ（6月のノートのセル9・9-2）

1 つの説明変数だけを動かし、ほかは学習データの中央値（設定温度は Tref）に固定したときの、予測の平均（赤線）と 95% 予測区間（帯）。点は実測（5data と 3data。色は厚み）。ほかの変数を固定しているので、実際の成形では起こらない組み合わせも含む（例：せん断速度だけを大きくして、P3_3 は中央値のまま）。曲線は「その変数の効果の向きと大きさ」の目安として読む。

**温度の効果は 3 通りで見る**：Tset だけを動かす（Tff は固定。Tff を入力やシフトに使うモデルでは、設定温度を上げても、流動中樹脂温度は変わらないことになる）／Tff だけを動かす／**Tset と Tff を同時に動かす**（設定温度を上げると、流動中樹脂温度も同じだけ上がるとした。成形で起こることに近い）。判定には、同時に動かしたときの効果を使う。

**判定**：せん断速度を大きくすると粘度が下がる（ずり流動化）／Tset と Tff を同時に上げると粘度が下がる。Tset だけ・Tff だけを動かしたときの比は、参考として表示する。''')
code(r'''# セル13：依存性グラフ
DEP_VARS = ["gd", "P33", "Tff", "Tset", "Tboth"]
for mat in MATERIALS:
    tr, pool = TRAIN[mat], pd.concat([TRAIN[mat], TEST[mat]], ignore_index=True)
    base = base_point(tr, CFG[mat])
    tl, th_ = pool["Tset"].min(), pool["Tset"].max()
    res_gd, res_both, res_only = [], [], []
    for key in KEYS:
        f = FITS[(mat, key)]
        fig, axes = plt.subplots(2, 3, figsize=(22, 12))
        for ax, var in zip(axes.flat, DEP_VARS):
            name, unit, logx = VAR_JA[var]
            g = var_grid(pool, var, 120)
            if var == "Tboth":      # 設定温度を上げると、流動中樹脂温度も同じだけ上がるとして、2 つを一緒に動かす
                X = model_inputs(**{**base, "Tset": g, "Tff": base["Tff"] + (g - base["Tset"])})
            else:
                X = model_inputs(**{**base, var: g})
            mu, sd, _, _ = predict(f, X)
            ax.fill_between(g, np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd), color="red", alpha=0.2, label="95% 予測区間", zorder=4)
            ax.plot(g, np.exp(mu), color="red", lw=2.2, label="予測平均値", zorder=5)
            xcol = "Tset" if var == "Tboth" else var
            for h in sorted(pool["H"].unique()):
                m = (pool["H"] == h).to_numpy()
                ax.scatter(pool.loc[m, xcol], pool.loc[m, "eta"], s=12, color=H_COLORS.get(h, "gray"), marker=H_MARKS.get(h, "o"), alpha=0.4, edgecolors="none", label=f"実測 {h:g} mm", zorder=2)
            if logx:
                ax.set_xscale("log")
            ax.set_yscale("log")
            log_ticks(ax, "xy" if logx else "y")
            ax.set_title(f"{name} 依存性")
            ax.set_xlabel(f"{name} {unit}")
            ax.set_ylabel(r"樹脂粘度 $\eta$ (Pa·s)")
            ax.grid(True, which="both", ls="--", alpha=0.4)
            ax.legend(fontsize=8, loc="best")
            if var == "gd":
                res_gd.append((key, float(np.exp(mu[0] - mu[-1]))))
            if var in ("Tset", "Tboth"):
                lo_i, hi_i = np.argmin(np.abs(g - tl)), np.argmin(np.abs(g - th_))
                (res_only if var == "Tset" else res_both).append((key, float(np.exp(mu[lo_i] - mu[hi_i]))))
        fixed = f"γ̇ {base['gd']:.0f} 1/s、P3_3 {base['P33']:.2f} MPa、Tff {base['Tff']:.1f} ℃、Tset {base['Tset']:g} ℃"
        axes.flat[5].axis("off")
        axes.flat[5].text(0.0, 0.9, "動かさない変数の値（学習データの中央値）\n\n" + fixed.replace("、", "\n") + "\n\n同時に動かす図では、\nTff = Tff の値 + (Tset − Tset の値)\nとして、2 つを一緒に動かした。",
                          va="top", fontsize=13)
        fig.suptitle(f"{MAT_NAMES[mat]}　{model_title(key)}　依存性グラフ")
        save_fig(fig, out_dir(mat, key), "依存性グラフ", show=SHOW_FIGS and key == MAIN_KEY)
    judge(all(r > 1 for _, r in res_gd), "せん断速度を大きくすると粘度が下がる（グラフの範囲の両端での粘度の比：" + "、".join(f"{k} {r:.1f} 倍" for k, r in res_gd) + "）", mat, "依存性")
    judge(all(r > 1 for _, r in res_both), f"Tset と Tff を同時に上げると粘度が下がる（η({tl:g} ℃)/η({th_:g} ℃)：" + "、".join(f"{k} {r:.3f}" for k, r in res_both) + "。γ̇・P3_3 は固定）", mat, "依存性")
    print(f"  （参考）Tset だけを動かしたときの η({tl:g} ℃)/η({th_:g} ℃)：" + "、".join(f"{k} {r:.3f}" for k, r in res_only) + "。Tff を入力やシフトに使うモデルでは、Tff が固定なので、設定温度の効果が小さく・逆向きに出ることがある")''')

# ---------------------------------------------------------------- 8. 検証
md(r'''## 8. 3data による予測検証（6月のノートのセル10）

5data で学習したモデルで、3data（射出速度の両端。低速側 3・5 mm/s、高速側 280・320 mm/s）を予測する。3data で測った P3_3・Tff を入力に使う。
- **指標**（ln η）：RMSE（小さいほど良い。0.1 は粘度で約 10%）、R²、95% 予測区間に入る割合（被覆率。0.95 に近いほど、予測のばらつきの見積もりが正しい）、NLPD（予測分布の当てはまり。小さいほど良い）。**R²（実数）**は、6月のノートと同じく粘度の実数値で計算した R²。
- **パリティ図**（`予測検証_3data_パリティ.png`）：横軸が実測、縦軸が予測の粘度。縦線が 95% 予測区間。破線（予測 = 実測）に近いほど良い。
- ほかのモデル（ほかの列のシフトや C）との比較は、シフトファクター検討用のノートで行う。

**判定**：被覆率が 0.8 以上。''')
code("# セル14：3data による予測検証\n" + nbc.VALID)

# ---------------------------------------------------------------- 9. Cross-WLF
md(r'''## 9. 参考：Cross-WLF との比較（6月のノートのセル10.5・11）

Cross-WLF 式（D3 = 0）：η0 = D1·exp(−A1(T − D2)/(A2 + T − D2))、η = η0 / (1 + (η0·γ̇/τ*)^(1−n))。設定 `CWLF_FIXED` で固定した係数以外を、5data に（ln η の最小二乗で）当てはめ、3data で評価する。温度 T は `CWLF_T`（既定は 6月と同じ流動中樹脂温度）。予測区間は、学習データの残差の標準偏差で付ける。

**図の見方**（`CrossWLFとの比較_パリティ.png`）：同じ 3data で、Cross-WLF（青）と GPR（このノートのモデル、赤）の予測を重ねる。`予測検証_RMSEの比較.png` は、このノートのモデルと Cross-WLF の RMSE を、全体・低速側・高速側で並べたもの。''')
code(r'''# セル15：Cross-WLF との比較
CW, CROWS = {}, []
for mat in MATERIALS:
    try:
        cw = fit_cross_wlf(TRAIN[mat], CWLF_T, CWLF_FIXED[mat])
    except Exception as e:
        print(f"{MAT_NAMES[mat]}：Cross-WLF の当てはめに失敗しました（{e}）。CWLF_FIXED の固定値を見直す")
        judge(False, f"Cross-WLF の当てはめに失敗（{e}）", mat, "Cross-WLF")
        continue
    CW[mat] = cw
    mu, sd = predict_cross_wlf(cw, TEST[mat])
    rows = metrics_rows(mat, "Cross-WLF", TEST[mat], mu, sd, np.full(len(mu), np.nan))
    CROWS += rows
    prm = pd.DataFrame([{"係数": k, "値": v, "扱い": ("探索（範囲の端）" if k in cw["at_bound"] else "探索") if k in cw["free"] else "固定"} for k, v in cw["params"].items()])
    to_csv(prm, os.path.join(out_dir(mat), "CrossWLF_係数.csv"))
    print(f"\n【{MAT_NAMES[mat]}】Cross-WLF（温度 {CWLF_T}）：" + "、".join(f"{r['係数']} = {r['値']:.4g}（{r['扱い']}）" for _, r in prm.iterrows()) + f"、学習の残差 {cw['resid_sd']:.3f}")
    p = PRED[(mat, MAIN_KEY)]
    r_cw = rows[0]["RMSE"]
    r_gp = float(MT[(MT["材料"] == MAT_NAMES[mat]) & (MT["モデル"] == MAIN_KEY) & (MT["範囲"] == "全体")]["RMSE"].iloc[0])
    fig, ax = plt.subplots(figsize=(9.5, 9))
    ax.scatter(TEST[mat]["eta"], np.exp(mu), color="#1f77b4", s=30, alpha=0.7, label=f"Cross-WLF（RMSE {r_cw:.3f}）", zorder=2)
    ax.scatter(p["eta"], p["予測η"], color="#d62728", s=30, alpha=0.7, label=f"GPR {MAIN_KEY}（RMSE {r_gp:.3f}）", zorder=3)
    lim = [min(TEST[mat]["eta"].min(), np.exp(mu).min(), p["予測η"].min()) / 1.3, max(TEST[mat]["eta"].max(), np.exp(mu).max(), p["予測η"].max()) * 1.3]
    ax.plot(lim, lim, "k--", lw=1, label="予測 = 実測")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    log_ticks(ax)
    ax.set_xlabel(r"実測の粘度 $\eta$ (Pa·s)")
    ax.set_ylabel(r"予測の粘度 $\eta$ (Pa·s)")
    ax.set_title(f"{MAT_NAMES[mat]}　予測モデルの精度比較（同じ 3data による評価。RMSE は ln η）")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(loc="upper left")
    save_fig(fig, out_dir(mat), "CrossWLFとの比較_パリティ")
    judge(r_gp <= r_cw, f"（参考）GPR {MAIN_KEY} の 3data の RMSE（{r_gp:.3f}）が、Cross-WLF（{r_cw:.3f}。温度 {CWLF_T}）以下", mat, "Cross-WLF")
CT = pd.DataFrame(CROWS)
MT_ALL = pd.concat([MT, CT], ignore_index=True)
for mat in MATERIALS:
    t = MT_ALL[MT_ALL["材料"] == MAT_NAMES[mat]]
    to_csv(t, os.path.join(out_dir(mat), "予測検証_3data_指標.csv"))
    pv = t[~t["範囲"].str.contains("感度")].pivot_table(index="モデル", columns="範囲", values="RMSE", sort=False)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    w = 0.8 / len(pv.index)
    for i, (k, row) in enumerate(pv.iterrows()):
        ax.bar(np.arange(len(pv.columns)) + i * w, row.to_numpy(), width=w, label=k)
    ax.set_xticks(np.arange(len(pv.columns)) + w * (len(pv.index) - 1) / 2)
    ax.set_xticklabels(pv.columns)
    ax.set_ylabel("3data の RMSE（ln η）")
    ax.set_title(f"{MAT_NAMES[mat]}　3data の予測誤差の比較（小さいほど良い）")
    ax.grid(axis="y", ls="--", alpha=0.5)
    ax.legend()
    save_fig(fig, out_dir(mat), "予測検証_RMSEの比較")''')

# ---------------------------------------------------------------- 10. せん断速度 ― 粘度
md(r'''## 10. せん断速度 ― 粘度の学習結果グラフ（6月のノートのセル9-2 と同じ見せ方）

線は予測の平均、帯は 95% 区間、塗りつぶしの点は学習（5data）、白抜きの点は検証（3data）、× は問題データ一覧の「感度確認」の条件。点は、ショットごとの計測値（設定 `FLOW_POINTS = "cond"` にすると、条件平均の点になる）。予測線の P3_3・Tff は、同じ設定温度・厚みの条件平均（5data と 3data）を ln γ̇ で内挿した値（データの範囲の外は、端の値のまま）。
- **レイアウト**：`"temperature"`＝設定温度ごとに 1 枚（厚みごとの線。6月のノートと同じ）、`"thickness"`＝厚みごとに 1 枚（設定温度ごとの線）。
- **全厚み共通の図**（`FLOW_COMMON = True`）：①`…_全厚み共通_<温度>C.png`＝加熱シリンダ温度ごとに 1 枚。全キャビティ厚みの**学習データ**の点（色は厚み）を重ね、**厚みによらない共通の平均関数と 95% 区間**を 1 本描く（6月のセル9-2 と同じ見せ方。検証の点は描かない）。②`…_全厚み共通_全温度.png`＝①の曲線と帯を、**全温度を 1 枚に重ねて**、温度ごとに線・帯の色を変えた総合図（温度ごとの平均関数と予測幅を、1 つのグラフで比べられる。線・帯の色は、低い温度から**赤・青・緑**。点の色・形は厚み）。**グラフの画像には、タイトルも凡例も付けず**、凡例は別の画像（`…_全厚み共通_全温度_凡例.png`（日本語）と `…_全温度_凡例_English.png`（英語）。予測平均値_温度、95% 予測区間_温度、キャビティ厚み、固定した値）に保存する。図の大きさ（12 × 10 インチ。6月は 15 × 10）・文字・点の大きさ（300）・両軸の 10 の累乗の目盛りは、6月のセル9-2 と同じ。曲線は、せん断速度だけを動かし、P3_3 は学習データの中央値、Tff はその設定温度の学習データの中央値に固定したもの（温度ごとの図では図の右上の枠に、総合図では凡例の画像に、固定した値を書く）。厚みの違いは、P3_3・Tff の違いを通してしか入らないので、厚みごとの点は、共通の曲線の上下に散らばる。
- 保存先は、モデルのフォルダの中の `せん断速度_粘度/`（図は 600 dpi）。
- **このセルの先頭の設定エリアだけを直して、このセルだけを実行し直せる**（全体を一度実行した後で。学習はやり直さない）。''')
code(r'''# セル16：せん断速度 ― 粘度の学習結果グラフ
# ==========================================
# ▼▼ 設定エリア（グラフの見た目。このセルの設定だけを直して、このセルだけを実行し直せる） ▼▼
# ==========================================
FLOW_MODELS = KEYS                  # 描くモデル（このノートで学習したモデル）
FLOW_MATERIALS = MATERIALS          # 描く材料
FLOW_LAYOUTS = ["temperature", "thickness"]   # "temperature"：設定温度ごとに 1 枚（厚みごとの線）／"thickness"：厚みごとに 1 枚（設定温度ごとの線）
FLOW_COMMON = True                  # True：全厚み共通の図も作る。加熱シリンダ温度ごとに 1 枚（…_全厚み共通_<温度>C.png）と、全温度を 1 枚に重ねた総合図（…_全厚み共通_全温度.png。温度ごとに色分け）。検証（3data）の点は描かない
FLOW_COMMON_XLIM = None                  # 全厚み共通の図の横軸（せん断速度 [1/s]）の範囲。(下限, 上限)＝固定／None＝データを含む 10 の累乗の範囲（目盛りは、6月のセル9-2 と同じ 10 の累乗）
FLOW_COMMON_YLIM = None                    # 同じ、縦軸（粘度 [Pa·s]）の範囲。例：(0.1, 10000)
FLOW_COMMON_NCOL = 1                # 総合図の凡例の列数（凡例が大きくて点に重なるときは 2 にする）
FLOW_BAND = "predict"               # 帯："predict"＝95% 予測区間（ショット間のばらつき込み）／"latent"＝95% 信頼区間（平均の不確かさだけ）／None＝帯なし
FLOW_POINTS = "shot"                # 点："shot"＝ショットごとの計測値（既定。10-26 に変更）／"cond"＝条件平均
FLOW_SHOW_TEST = True               # 検証（3data）の点を、白抜きで重ねるか
FLOW_MARK_SENS = True               # 問題データ一覧の「感度確認」の条件を、× で重ねるか
FLOW_X_RANGE = None                 # 軸の範囲を自動で決めるときの目安 [1/s]（予測線の範囲。None なら、各線のデータの範囲を FLOW_EXTEND 倍）。予測線と帯は、どの場合も、グラフエリアの端（軸の範囲。FLOW_XLIM・FLOW_COMMON_XLIM で変えても追従）まで描く
FLOW_EXTEND = 1.3                  # 自動の軸の範囲を決めるときの、データの範囲の延長の倍率
FLOW_XLIM = {"temperature": None, "thickness": None}   # 横軸（せん断速度 [1/s]）の範囲。(下限, 上限)＝固定（10 の累乗の目盛り）／None＝その図のデータに合わせる（目盛りは、範囲が 2.5 桁以下なら 1・2・5 × 10ⁿ、それより広いか、目盛りの数字が重なるなら 10 の累乗。10-26）
FLOW_YLIM = {"temperature": None, "thickness": None}   # 縦軸（粘度 [Pa·s]）の範囲。同じ書き方
FLOW_FIGSIZE = (12, 10)             # 図の大きさ（幅, 高さ）[インチ]（6月は 15 × 10。10-15 に 12 × 10 へ変更）
FLOW_DPI = None                     # 保存する PNG の解像度（None なら FIG_DPI〔既定 600 dpi〕）
FLOW_FONT = {"axis": 24, "legend": 24, "title": 24, "tick": 28}   # 軸名・凡例・タイトル・目盛り数字の文字サイズ（6月のセル9-2 と同じ）
FLOW_MARKER = 300                   # 点の大きさ（条件平均のとき。面積）
FLOW_MARKER_SHOT = 70               # 点の大きさ（ショットごとのとき）
FLOW_LINE = 2.0                     # 予測線の太さ
FLOW_SHOW_TITLE = True              # タイトルを付けるか
FLOW_SHOW_XLABEL = False            # 軸名を付けるか（6月のノートと同じく、既定は付けない）
FLOW_SHOW_YLABEL = False
FLOW_FORMATS = ["png"]              # 保存形式。"png"・"pdf"・"svg" から選ぶ
FLOW_SHOW_MODELS = [MAIN_KEY]       # 画面にも表示するモデル（全部表示すると重い）
FLOW_EXTENDED = True                # True：軸の範囲を広げた拡張版（せん断速度_粘度_拡張版/ フォルダ。ここまでの設定はそのまま使う）も作る
FLOW_EXT_XLIM = (1.0, 1.0e5)          # 拡張版の横軸（せん断速度 [1/s]）の範囲。予測線と帯は、この範囲いっぱいに描く
FLOW_EXT_YLIM = (0.5, 3000.0)         # 拡張版の縦軸（粘度 [Pa·s]）の範囲
# ==========================================
n_saved = 0
for mat in FLOW_MATERIALS:
    pool = pd.concat([TRAIN[mat], TEST[mat]], ignore_index=True)
    train_c = set(TRAIN[mat]["cond"])
    for key in FLOW_MODELS:
        saved = flow_plot(FITS[(mat, key)], mat, pool, train_c, out_dir(mat, key, "せん断速度_粘度"), "せん断速度_粘度",
                          layouts=FLOW_LAYOUTS, band=FLOW_BAND, points=FLOW_POINTS, show_test=FLOW_SHOW_TEST, mark_sens=FLOW_MARK_SENS,
                          x_range=FLOW_X_RANGE, extend=FLOW_EXTEND, xlim=FLOW_XLIM, ylim=FLOW_YLIM, figsize=FLOW_FIGSIZE, dpi=FLOW_DPI, font=FLOW_FONT,
                          marker=FLOW_MARKER, marker_shot=FLOW_MARKER_SHOT, line=FLOW_LINE, show_title=FLOW_SHOW_TITLE, show_xlabel=FLOW_SHOW_XLABEL,
                          show_ylabel=FLOW_SHOW_YLABEL, formats=FLOW_FORMATS, show=SHOW_FIGS and key in FLOW_SHOW_MODELS)
        n_saved += len(saved)
        if FLOW_COMMON:        # 加熱シリンダ温度ごとに、全厚みの点を重ね、厚みによらない共通の平均関数と区間を 1 本描く（6月のセル9-2 と同じ）
            saved = flow_plot_common(FITS[(mat, key)], mat, pool, train_c, out_dir(mat, key, "せん断速度_粘度"), "せん断速度_粘度", tr=TRAIN[mat],
                                     band=FLOW_BAND, points=FLOW_POINTS, common_xlim=FLOW_COMMON_XLIM, common_ylim=FLOW_COMMON_YLIM, legend_ncol=FLOW_COMMON_NCOL, x_range=FLOW_X_RANGE, extend=FLOW_EXTEND, figsize=FLOW_FIGSIZE, dpi=FLOW_DPI,
                                     font=FLOW_FONT, marker=FLOW_MARKER, marker_shot=FLOW_MARKER_SHOT, line=FLOW_LINE, show_title=FLOW_SHOW_TITLE,
                                     show_xlabel=FLOW_SHOW_XLABEL, show_ylabel=FLOW_SHOW_YLABEL, formats=FLOW_FORMATS, show=SHOW_FIGS and key in FLOW_SHOW_MODELS)
            n_saved += len(saved)
        if FLOW_EXTENDED:      # 拡張版：軸の範囲だけを変えて、同じ図（厚みごと・設定温度ごと・全厚み共通・総合）を別フォルダに保存する
            ext_dir = out_dir(mat, key, "せん断速度_粘度_拡張版")
            ext_kw = dict(band=FLOW_BAND, points=FLOW_POINTS, x_range=FLOW_X_RANGE, extend=FLOW_EXTEND, figsize=FLOW_FIGSIZE, dpi=FLOW_DPI, font=FLOW_FONT, marker=FLOW_MARKER,
                          marker_shot=FLOW_MARKER_SHOT, line=FLOW_LINE, show_title=FLOW_SHOW_TITLE, show_xlabel=FLOW_SHOW_XLABEL, show_ylabel=FLOW_SHOW_YLABEL, formats=FLOW_FORMATS, show=False)
            saved = flow_plot(FITS[(mat, key)], mat, pool, train_c, ext_dir, "せん断速度_粘度_拡張版", layouts=FLOW_LAYOUTS, show_test=FLOW_SHOW_TEST, mark_sens=FLOW_MARK_SENS,
                              xlim={"thickness": FLOW_EXT_XLIM, "temperature": FLOW_EXT_XLIM}, ylim={"thickness": FLOW_EXT_YLIM, "temperature": FLOW_EXT_YLIM}, **ext_kw)
            n_saved += len(saved)
            if FLOW_COMMON:
                saved = flow_plot_common(FITS[(mat, key)], mat, pool, train_c, ext_dir, "せん断速度_粘度_拡張版", tr=TRAIN[mat], common_xlim=FLOW_EXT_XLIM, common_ylim=FLOW_EXT_YLIM,
                                         legend_ncol=FLOW_COMMON_NCOL, **ext_kw)
                n_saved += len(saved)
print(f"保存した図：{n_saved} 枚（モデルのフォルダの せん断速度_粘度/ と せん断速度_粘度_拡張版/）")''')

md(r'''### 10.2 6月のセル9-2 と全く同じ図（目的変数に対する各説明変数の依存性）

6月のノートのセル9-2 と同じ図を、**学習データ（5data）の全ショットの点**と学習済みモデルで作る。説明変数（せん断速度・P3_3・Tff・Tset）ごとに 1 枚。その変数だけを動かし、ほかは**学習データの平均**（対数の変数は、対数の平均）に固定した、予測平均（赤線）と 95% 区間（帯。6月と同じく、ノイズを含む予測区間）。点の色・形は厚み。大きさ（12 × 10 インチ。6月は 15 × 10）、文字、点（300）、10 の累乗の目盛り、凡例、タイトルは、6月と同じ。保存先は、モデルのフォルダの中の `依存性グラフ_セル9-2/`。**このセルの先頭の設定だけを直して、このセルだけを実行し直せる**。固定した値と軸の範囲を変えられる。''')
code(r'''# セル16-2：6月のセル9-2 と同じ依存性グラフ
# ==========================================
# ▼▼ 設定エリア（6月のセル9-2 と同じ） ▼▼
# ==========================================
DEP92_MODELS = KEYS                       # 描くモデル
DEP92_MATERIALS = MATERIALS               # 描く材料
DEP92_FIGSIZE = (12, 10)                  # 図の大きさ [インチ]
DEP92_DPI = None                          # PNG の解像度（None なら FIG_DPI〔既定 600 dpi〕）
DEP92_FONT = dict(axis_fs=24, legend_fs=24, title_fs=24, tick_fs=28)     # 軸名・凡例・タイトル・目盛りの文字の大きさ
DEP92_MARKER = dict(plot_marker=300, legend_marker=300)                  # 点の大きさ（グラフ上・凡例内）
DEP92_YLIM = (0.1, 10000.0)               # 粘度の軸の範囲 [Pa·s]（None なら自動）
DEP92_XLIM = {"gd": (10.0, 100000.0)}     # 横軸の範囲：変数（"gd"・"P33"・"Tff"・"Tset"）→ (下限, 上限)。書かない変数は、データの範囲の両側に 10% 広げる
DEP92_SHOW_XLABEL, DEP92_SHOW_YLABEL = False, False   # 軸名を付けるか（6月と同じ：付けない）
# ==========================================
n92 = 0
for mat in DEP92_MATERIALS:
    for key in DEP92_MODELS:
        n92 += len(dependency_plots_92(FITS[(mat, key)], mat, TRAIN[mat], out_dir(mat, key, "依存性グラフ_セル9-2"), figsize=DEP92_FIGSIZE, dpi=DEP92_DPI,
                                       ylim=DEP92_YLIM, xlim=DEP92_XLIM, show_xlabel=DEP92_SHOW_XLABEL, show_ylabel=DEP92_SHOW_YLABEL, show=False, **DEP92_FONT, **DEP92_MARKER))
print(f"保存した図：{n92} 枚（モデルのフォルダの 依存性グラフ_セル9-2/）")''')

# ---------------------------------------------------------------- 11. 3次元
md(r'''## 11. 3次元グラフ（6月のノートのセル12〜15）

2 つの説明変数を動かし、ほかは学習データの中央値（設定温度は Tref）に固定したときの、予測の粘度（log10）の曲面。保存先は、モデルのフォルダの中の `3次元グラフ/`。**このセルの先頭の設定エリアだけを直して、このセルだけを実行し直せる**。''')
code(r'''# セル17：3次元グラフ
# ==========================================
# ▼▼ 設定エリア（このセルの設定だけを直して、このセルだけを実行し直せる） ▼▼
# ==========================================
SURF_MODELS = KEYS                                       # 描くモデル
SURF_PAIRS = [("gd", "Tset"), ("gd", "P33"), ("gd", "Tff")]   # 動かす 2 つの変数（gd：せん断速度、P33：P3_3、Tff：流動中樹脂温度、Tset：設定温度）
SURF_N = 40                                              # 格子の細かさ
SURF_VIEW = (25, -60)                                    # 視点（仰角, 方位角）[度]
SURF_FIX = {}                                            # 固定する値を変えるとき。例：{"P33": 5.0, "Tff": 220.0}（空なら学習データの中央値、設定温度は Tref）
SURF_SHOW_MODELS = [MAIN_KEY]                            # 画面にも表示するモデル
# ==========================================
for mat in MATERIALS:
    pool = pd.concat([TRAIN[mat], TEST[mat]], ignore_index=True)
    base = {**base_point(TRAIN[mat], CFG[mat]), **SURF_FIX}
    for key in SURF_MODELS:
        for va, vb in SURF_PAIRS:
            ga, gb = var_grid(pool, va, SURF_N), var_grid(pool, vb, SURF_N)
            A, B = np.meshgrid(ga, gb)
            mu, _, _, _ = predict(FITS[(mat, key)], model_inputs(**{**base, va: A, vb: B}))
            Z = (mu / np.log(10)).reshape(A.shape)
            xa = np.log10(A) if VAR_JA[va][2] else A
            yb = np.log10(B) if VAR_JA[vb][2] else B
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(xa, yb, Z, cmap="viridis", edgecolor="none", alpha=0.95)
            lab = lambda v: ("log10 " if VAR_JA[v][2] else "") + VAR_JA[v][0] + " " + VAR_JA[v][1]
            ax.set_xlabel(lab(va), labelpad=12)
            ax.set_ylabel(lab(vb), labelpad=12)
            ax.set_zlabel("log10 粘度 η (Pa·s)", labelpad=10)
            ax.view_init(*SURF_VIEW)
            fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="log10 η")
            others = "、".join(f"{VAR_JA[k][0].split(' ')[-1]} {v:.3g}" for k, v in base.items() if k not in (va, vb))
            ax.set_title(f"{MAT_NAMES[mat]}　{model_title(key)}\n{VAR_JA[va][0]} と {VAR_JA[vb][0]}（固定：{others}）")
            save_fig(fig, out_dir(mat, key, "3次元グラフ"), f"3次元_{va}_{vb}", show=SHOW_FIGS and key in SURF_SHOW_MODELS)
print("3次元グラフを保存しました")''')

# ---------------------------------------------------------------- 12. 保存
md(r'''## 12. 学習済みモデルの保存（6月のノートのセル16）

モデルのフォルダに `学習済みモデル_<モデル>.joblib` として保存する（学習データ・検証データの場所も記録する。Git には入れない。このノートを実行すれば、いつでも作り直せる）。使い方（このノートの「共通の部品」のセルを実行した後で。ノートの外で使うときは、先に `import sys; sys.path.insert(0, "04_補助スクリプト"); from mat_common import *` で、同じ部品を読み込む）：

```python
import joblib
obj = joblib.load(r"…/全ショット版/全厚み総合/B_Tset_温度シフトに加熱シリンダ温度採用/学習済みモデル_B_Tset.joblib")
X = model_inputs(gd=[1000, 5000], P33=[5.0, 3.0], Tff=[210.0, 215.0], Tset=[210.0, 210.0])   # γ̇ [1/s]、P3_3 [MPa]、Tff [℃]、Tset [℃]
mu, sd, sd_latent, noise_sd = predict(obj["fit"], X)       # ln η の予測平均と、予測標準偏差（ノイズ込み）
eta = np.exp(mu)                                         # 粘度 [Pa·s]。95% 予測区間は np.exp(mu ± 1.96 * sd)
```
入力の取り方がショットごと（`INPUT_MODE = "shot"`。既定）のモデルでは、入れる γ̇・P3_3・Tff は、1 ショットで測った値にする。条件平均（`"cond"`）のモデルでは、予測したい条件の平均の値（測った値の平均）にする。''')
code("# セル18：学習済みモデルの保存\n" + nbc.SAVE)

# ---------------------------------------------------------------- 13. まとめ
md(r'''## 13. まとめ

判定の一覧（★ の行を確かめる）と、モデルの主な結果（θ と 95% 区間、3data の RMSE・被覆率。参考の Cross-WLF も）。モデルのフォルダの `判定の一覧.csv`・`まとめ.csv` に保存する。''')
code(r'''# セル19：まとめ
JT = pd.DataFrame(JUDGES)
print(f"判定 {len(JT)} 件のうち、★要確認 {int((JT['判定'] != 'OK').sum())} 件")
display(JT[JT["判定"] != "OK"])
SUM = []
for mat in MATERIALS:
    thn, tvn, un = theta_names(CFG[mat])
    for key in KEYS + (["Cross-WLF"] if mat in CW else []):
        row = {"材料": MAT_NAMES[mat], "モデル": key, "入力": VERSION_DIRS[INPUT_MODE]}
        if key in SHIFT_KEYS:
            b = BEST[(mat, key)]
            row.update({"θ": f"{thn} {b['th_best']:g}［{b['th_lo']:g}〜{b['th_hi']:g}］・{tvn} {b['tv_best']:g}［{b['tv_lo']:g}〜{b['tv_hi']:g}］{un}", "LML": round(b["LML_best"], 1),
                        "θ=0 との LML の差": round(b["dLML_vs_0"], 1)})
        elif key == "C":
            row["LML"] = round(float(KT[(KT["材料"] == MAT_NAMES[mat]) & (KT["model"] == "C")]["LML"].iloc[0]), 1)
        t = MT_ALL[(MT_ALL["材料"] == MAT_NAMES[mat]) & (MT_ALL["モデル"] == key)].set_index("範囲")
        for s in ["全体", "低速側", "高速側"]:
            if s in t.index:
                row[f"RMSE {s}"] = round(float(t.loc[s, "RMSE"]), 4)
        row["被覆率 全体"] = round(float(t.loc["全体", "coverage95"]), 3)
        row["R²(実数) 全体"] = round(float(t.loc["全体", "R2_real"]), 4)
        SUM.append(row)
SUM = pd.DataFrame(SUM)
display(SUM)
for mat in MATERIALS:
    to_csv(JT[JT["材料"].isin([MAT_NAMES[mat], ""])], os.path.join(out_dir(mat), "判定の一覧.csv"))
    to_csv(SUM[SUM["材料"] == MAT_NAMES[mat]], os.path.join(out_dir(mat), "まとめ.csv"))
print("結果の置き場所：" + "、".join(show_path(out_dir(m)) for m in MATERIALS))''')


# ---------------------------------------------------------------- 書き出し
def to_lines(s):
    lines = s.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


rnd = random.Random(20261014)
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

out = sys.argv[1]
if out.startswith("--"):
    sys.exit("1 つ目の引数には、出力する .ipynb のパスを書く（例：python build_mat_shift_nb.py <出力.ipynb> --force）")
if os.path.exists(out) and "--force" not in sys.argv[2:]:
    sys.exit(f"既にファイルがあるため上書きしません（上書きするときは --force）: {out}")
assert len({c["id"] for c in cells}) == len(cells)
for c in cells:  # コードセルの構文チェック (%matplotlib の行は除く)
    if c["cell_type"] == "code":
        compile("".join(l for l in c["source"] if not l.startswith("%")), "<cell>", "exec")
with open(out, "wb") as f:
    f.write((json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
print(f"作成: {out}（{len(cells)} セル）")
