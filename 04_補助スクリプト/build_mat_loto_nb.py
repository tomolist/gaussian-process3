# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR の「温度ホールドアウト検証」のノートブック (未実行) を生成する。

使い方: python build_mat_loto_nb.py <出力する .ipynb のパス> [--material PP|MIX|GPPS] [--force]
（成果物は 01_機械学習コードフォルダ/材料別_温度シフトファクター検討/材料別_温度シフト機械学習コード/
 材料別_温度ホールドアウト検証_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb に置く〔学習用のノートと同じフォルダ〕。
 ノートは、置いた場所から上へたどってリポジトリ直下を探すので、どこに置いても動く）
--material を付けると、その材料だけの版（材料ごとの複製）を作る（2026-11-16 のユーザー依頼。学習用のノートの複製 PP_・MIXPP_・GPPS_ と同じ分け方）。
 設定エリアの MATERIALS・STRUCTS・MAIN_KEY を、その材料と本番の形（PP・MIX は A、GPPS は B。ユーザーの決定）に置き換え、冒頭に 1 段落を足す。
 置き場所の名前は、元の名前の「材料別_」を PP_・MIXPP_・GPPS_ に替えたもの。付けなければ、今までと同じノート。
既存ファイルがある場合は、--force を付けない限り上書きしない。
2026-11-15 のユーザーの問い（温度シフトファクターが本当に必要かを、さらに確実にするため、3 つの加熱シリンダ温度のうち 1 つを
予測精度の検証に使い、残りの温度で学習することを、全パターン〔3 通り〕行うノートを作れるか）に答えるためのノート。
ユーザーの決定（同日）：学習は残り 2 温度の 5data だけ。検証は、除いた温度の 5data（温度だけの外挿）。除いた温度の 3data は、別に集計する。
シフトのあるモデル（A_Tset_Arr・B_Tset_Arr など）、設定温度を GP の普通の入力にした C、設定温度を使わない C_noTset（入力 ln γ̇・ln P3_3・ln Tff。
A_Tset・B_Tset の θ = 0 と同じモデル）を、回ごとに学習して、除いた温度での誤差を比べる。
C_noTset は、mat_common.py を変えず、このノートの中だけで足す（layout を包む。Tff の入力の有無の比較のノートの C_noTff と同じやり方）。
学習用・シフトファクター検討用のノートと同じセルの中身（ライブラリ・読み込み）は mat_nb_cells.py から使う。
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
STUDY_NB = "../材料別_温度シフトファクター検討用コード/材料別_温度シフトファクターの検討_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb"   # このノートの場所からの相対パス
TRAIN_NB = "材料別_温度シフトGPR_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb"     # 学習用のノート（このノートと同じフォルダ）
MANUAL = "説明書_材料別温度シフトGPRのノート.pdf"     # 1 つ上のフォルダの、ノートの説明書（build_mat_nb_manual.py）
OUT_NAME = "温度ホールドアウト検証"                    # 結果のフォルダの名前（<版>/全厚み総合/ の下）

# ---------------------------------------------------------------- 0. タイトル
md(rf'''# 材料別の温度シフト GPR：加熱シリンダ温度を 1 つずつ除いて検証する（温度ホールドアウト検証）

**温度シフトファクターが本当に必要か**を、温度の方向の予測力で確かめる（2026-11-15 のユーザーの問いから）。今のノート（学習用 `{TRAIN_NB}`・シフトファクター検討用）は「5data で学習し、3data で検証」で、3data は**射出速度の両端**の外挿。温度は、学習と検証で同じ 3 水準なので、**学習にない温度をどれだけ当てられるか**（シフトファクターの本来の役目）は、まだ直接には確かめていない。

そこで、材料ごとに、3 つの加熱シリンダ温度のうち **1 つを学習から除き**、残りの 2 温度の 5data で学習して、**除いた温度を予測する**。これを 3 つの温度すべてで行う（3 回）。

| 回 | 学習（5data） | 検証の主（5data） | 別に集計（3data） |
|---|---|---|---|
| 低い温度を除く（PP：190 ℃） | 残りの 2 温度（210・230 ℃） | 190 ℃ の 5data（**温度だけの外挿**。射出速度は学習と同じ） | 190 ℃ の 3data（温度と射出速度の両方の外挿） |
| 中の温度を除く（PP：210 ℃） | 190・230 ℃ | 210 ℃ の 5data（温度の**内挿**） | 210 ℃ の 3data |
| 高い温度を除く（PP：230 ℃） | 190・210 ℃ | 230 ℃ の 5data（温度だけの外挿） | 230 ℃ の 3data |
| （参考）全温度 | 3 温度すべて | ― | 3data すべて（シフトファクター検討用のノートと同じ学習） |

粉砕材混合 PP も 190・210・230 ℃、GPPS は 200・220・240 ℃。

**比べるモデル**（回ごとに、その回の学習データだけで学習する。カーネルは、どれも ARD Matern 5/2 ＋ 線形項 ＋ ホワイトノイズ）：

| モデル | 式 | 設定温度 Tset の使い方 |
|---|---|---|
| **A_Tset_Arr** | ln η = ln a_v(Tset) ＋ f( ln γ̇, ln P3_3, ln Tff ) | 縦のシフト（係数 1 つ。アレニウス型） |
| **B_Tset_Arr** | ln η = ln a_v(Tset) ＋ f( ln γ̇ ＋ ln a_h(Tset), ln P3_3, ln Tff ) | 横・縦のシフト（係数 2 つ） |
| **C** | ln η = f( ln γ̇, ln P3_3, ln Tff, **Tset** ) | GP の普通の入力 |
| **C_noTset**（このノートで足す） | ln η = f( ln γ̇, ln P3_3, ln Tff ) | **使わない**（A_Tset・B_Tset の θ = 0 と同じモデル） |

- **シフトが要るなら**、除いた温度の誤差は、シフトのあるモデルが C_noTset より小さい（温度の効果を、シフトファクターの式で、学習にない温度へ運べる）。
- C と比べると、「温度の効果を、シフトの式で表すか、GP の入力として自由に表すか」の違いが分かる（GP は、学習にない温度では線形項に頼る）。
- シフトファクターの係数 θ が、回によって（どの温度を除いても）大きく変わらないかも見る（全温度の 95% 区間に入るか）。

**読むときの注意**：
- 学習は **2 温度だけ**なので、温度の効果は 2 点の差でしか決まらない。A の係数は 2 温度の粘度の比で決まり、B の横と縦の係数（θ_h・θ_v）は、全温度のときよりさらに別々には決まりにくい。
- **両端の温度を除く回は、温度の外挿**（シフトの式の形〔アレニウス型〕に頼る）、中の温度を除く回は内挿。3data は、温度と射出速度の両方の外挿になる。
- C_noTset は設定温度を使わないが、**流動中樹脂温度 Tff と P3_3 を入力に使う**ので、温度の情報の一部は Tff・P3_3 が運ぶ（「温度をまったく使わない」モデルではない）。C_noTset の誤差が小さければ、「設定温度の効果は、Tff・P3_3 でほぼ説明できる」ことを表す。
- 尤度（LML）は、**同じ回（同じ学習データ）のモデルどうしだけ**で比べる。回が違うと、学習データの行数が違うので比べない。''')

# ---------------------------------------------------------------- 1. 使い方・設定
md(f'''## 使い方

**初めての人へ**：1 つ上のフォルダ（`材料別_温度シフトファクター検討/`）の説明書 `{MANUAL}` に、ノートの目的・関係・使えるデータ・使い方をまとめてある（このノートは、まだ説明書に載っていない）。

1. 最初の「設定エリア」のセルで、除く温度 `HOLDOUT_TEMPS`（既定 None＝3 つとも）、シフトの形 `STRUCTS`（既定 A と B）、型 `SHIFT_FORMS`（既定はアレニウス型）、材料 `MATERIALS`、入力の取り方 `INPUT_MODE` を決める。ほかの設定も、このセルだけを直して、最初から実行し直す。
2. メニューの「すべて実行」で、上から順に実行する。カーネルは、このプロジェクトの `.venv` を選ぶ。重いのは「4. プロファイル尤度」（回 × シフトのモデルの数だけ。B が大部分）。目安は下の「所要時間」。
3. **実行の始めに（「2. データ」のセルで）、材料ごとにダイアログが 3 回開く**：①学習データ（5data）の CSV、②検証データ（3data）の CSV、③学習結果フォルダ（**材料ごとに別のフォルダ**）。学習用のノートと同じ学習結果フォルダを選んでよい（このノートの結果は、別のフォルダ `{OUT_NAME}/` に入るので、ほかのノートの結果は上書きしない）。通常版と温度適切版は、データのファイルが別なので、版ごとに 1 回ずつ実行する。設定 `FILES` に書けば、ダイアログは出ない。ダイアログが見当たらないときは、VS Code の後ろに隠れているので、タスクバーで探す。
4. 各節の最後に、**判定**が出る（`>> 判定：OK` か `>> 判定：★要確認`）。OK は「想定どおり」、★ は「想定と違う、または結果を読むときに注意が要る点」（このノートでは、「シフトが除いた温度の誤差を小さくしない」ことも ★ で知らせる）。★ が出ても、エラーではない。最後の「判定の一覧」に一覧が出る。
5. 結果は、学習結果フォルダの **`<版>/全厚み総合/{OUT_NAME}/`** に保存される（版は `INPUT_MODE` で、`全ショット版` か `条件ごとの平均版`）。
   - 直下：回とモデルを比べる表と図（`分け方.csv`・`シフトファクターの推定_回ごと.csv`・`ハイパーパラメータ.csv`・`予測の指標_回ごと.csv`・`予測のRMSE_回ごと.png`・`シフトファクターの係数_回ごと.png`・`まとめ.csv`・`判定の一覧.csv`）
   - その下の回のフォルダ（`除いた温度190℃/`・…・`全温度/`）：除いた温度のせん断速度 ― 粘度の図。その下のモデルのフォルダ（`A_Tset_Arr_…/`・`C_温度シフトなし/`・`C_noTset_…/` など）に、プロファイル尤度の表、予測値、パリティ図、学習済みモデル。
   - 図は 600 dpi の 1 枚だけ（設定 `FIG_DPI`）。図と学習済みモデルは Git の管理外、CSV は管理する。
6. 最後の「まとめ」に、材料とモデルごとに、**「設定温度を（シフトの形で／GP の入力で）使うのは、要る／どちらでもよい／要らない」**の目安が出る。決めるのはユーザー。

**所要時間**（目安。2026-11-16 の試しの実行〔やり直しの回数を減らした設定〕からの見積もり。要確認）：既定の設定（3 材料・A と B・全温度の回あり）で約 1.5〜2 時間。大部分は「4. プロファイル尤度」の B。`STRUCTS = ["A"]` にすると約 30 分、`REF_ALL = False` にすると約 3/4 になる。

**関係するノート**：シフトファクター検討用のノート（`{STUDY_NB}`）の「全温度」の学習と同じ設定なら、このノートの「全温度」の回の θ と LML は、そのノートの `シフトファクターの推定.csv` に近くなる（乱数と最適化の局所解の分だけ違うことがある）。''')
md(r'''## 1. 設定''')
code(r'''# セル1：設定エリア  ▼▼▼ 変えてよいのは、ここだけ ▼▼▼
from pathlib import Path      # （下の REPO で、フォルダをたどるのに使う）

HOLDOUT_TEMPS = None          # 学習から除いて、検証に回す加熱シリンダ温度 [℃] のリスト。None＝材料の設定温度を 1 つずつ、すべて（3 回）。例：[210]＝210 ℃ を除く回だけ（その温度がない材料〔GPPS は 200・220・240 ℃〕は、除く回がない）

REF_ALL = True                # True：参考に、全温度の 5data でも学習する（「全温度」の回。シフトファクター検討用のノートと同じ学習。係数の安定の判定に使う）／False：除く回だけ（速い）

SHIFT_COLS = ["加熱シリンダ温度[℃]"]   # シフトに使う温度の列（シフトファクター検討用のノートと同じ書き方。GPPS の列名 "加熱シリンダ温度(℃)" と同じ扱い）。このノートの問いは「設定温度のシフトが要るか」なので、既定は加熱シリンダ温度だけ

STRUCTS = ["A", "B"]          # シフトの形："A"＝縦だけ（係数 1 つ）／"B"＝横・縦を別々（係数 2 つ）。["A"] だけにすると、ずっと速い

SHIFT_FORMS = ["arrhenius"]   # シフトファクターの型："arrhenius"＝アレニウス型（記号の末尾 _Arr）／"wlf"＝WLF 型（_WLF）。両方も書ける

MAIN_KEY = "B_Tset_Arr"       # 図をノートにも表示するモデル（ほかのモデルの図も保存する。STRUCTS に B がないときは、最初のシフトのモデル）

SEED = 42                     # 乱数シード

NOISE_LB = 1e-6               # ノイズ分散の下限（ほかのノートと同じ）

USE_LINEAR = True             # True：6月のカーネル（ARD Matern 5/2 ＋ ホワイトノイズ）に線形項（DotProduct）を足す（ほかのノートと同じ）

INPUT_MODE = "shot"           # "shot"：γ̇・P3_3・Tff をショットごとの値で入れる（既定。結果は 全ショット版/）／"cond"：条件ごとの平均（条件ごとの平均版/）

N_RESTARTS = 20               # 最終モデルの学習での、最適化のやり直し回数（ほかのノートと同じ）

PROFILE_RESTARTS = 1          # プロファイル尤度の格子の各点での、やり直し回数（0 にすると約半分の時間）

USE_EXCLUSIONS = True         # True：問題データ一覧（05 フォルダ）の「除外」を除く（ほかのノートと同じ）

REPO = str(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "04_補助スクリプト").is_dir()))   # リポジトリ直下（このノートの場所から上へたどって、04_補助スクリプト があるフォルダを探す）

MATERIALS = ["PP", "MIX", "GPPS"]    # 調べる材料（PP：バージン PP、MIX：粉砕材混合 PP、GPPS）。材料ごとに、実行の始めにダイアログで、学習データ・検証データの CSV と、学習結果フォルダを選ぶ

MAT_FOLDERS = {"PP": "01_バージンPP再々実験(データと結果)", "MIX": "02_粉砕材混合PP実験(データと結果)", "GPPS": "03_GPPS実験(データと結果)"}   # 05 フォルダの中の材料フォルダ名（ダイアログの最初のフォルダ）

FILES = {}                    # ダイアログを使わずに決めるとき：{"PP": {"学習": r"…\結合_03_pp_max_5data_外れ値なし.csv", "検証": r"…", "結果": r"…"}}（相対パスはリポジトリ直下から）。書いていないものは、ダイアログで選ぶ
SHIFT_CFG = {                 # 材料ごとの基準温度 Tref [℃]（どの回でも、学習の温度の範囲の中か端にある）と、WLF 型の D2 = Tg [℃]（仮置き。アレニウス型では使わない）。ほかのノートと同じ
    "PP":   {"Tref": 210.0, "Tg": -10.0},
    "MIX":  {"Tref": 210.0, "Tg": -10.0},
    "GPPS": {"Tref": 220.0, "Tg": 100.0},
}
FORM_CFG = {                  # 型ごとの、尤度を調べる範囲（粗い格子：(下限, 上限, 刻み)。h が横、v が縦）と、細かい格子 "fine"（ほかのノートと同じ）
    "arrhenius": {"h_grid": (-20.0, 40.0, 5.0), "v_grid": (-20.0, 60.0, 5.0)},                            # θ = E [kJ/mol]
    "wlf":       {"A2": 51.6, "h_grid": (-20.0, 40.0, 5.0), "v_grid": (-20.0, 40.0, 5.0), "fine": (2.0, 0.25)},   # θ = A1 [−]
}
FINE_B = (4.0, 1.0)           # B の細かい格子：最良の点の ±4 を 1 刻み
FINE_A = (4.5, 0.5)           # A の細かい格子：最良の点の ±4.5 を 0.5 刻み
MAX_REFINE = 4                # 最良の点が細かい格子の端に当たったとき、細かい格子を取り直す回数の上限
MAX_EXTEND = 20               # 95% 区間の端が細かい格子の中で決まるまで、区間の外側に点を足す回数の上限
N_JOBS = 8                    # 並列に計算する数（CPU の数に合わせる）
FIG_DPI = 600                 # 保存する図の解像度（すべての図を、この解像度の 1 枚だけで保存する）
SHOW_FIGS = True              # True：図をノートにも表示する（False：保存だけ）
# ---- ここから下は、このノートだけの設定 ----
RMSE_RATIO = 1.1              # まとめ：除いた温度の RMSE の比（モデル ÷ C_noTset）が 1/1.1 より小さければ「良い」、1.1 より大きければ「悪い」、その間は「同等」（ほかのノートの「C の 1.1 倍以内」と同じ考え方）
COVER_MIN = 0.8               # 95% 予測区間に入る割合（被覆率）の目安（ほかのノートと同じ）
LML_CHECK = 1.0               # C_noTset の学習の LML が、同じモデル（シフトの θ = 0）の念入りな最適化の LML より、これ以上低ければ ★（局所解の目安。そのときは、LML の高いほうを C_noTset に使う）
SAVE_MODELS = True            # True：回ごと・モデルごとの学習済みモデル（.joblib）を保存する（1 つ数 MB。Git の管理外）
PARITY_FIGS = True            # True：回ごと・モデルごとのパリティ図を保存する（図の数＝回 × モデル × 材料）
# ▲▲▲ 設定エリア終了 ▲▲▲''')
code("# セル2：ライブラリ、フォント\n" + nbc.LIBS)
md(r'''### 共通の部品

ファイルを選ぶダイアログ、データの読み込み、問題データ一覧の「除外」、結果の置き場所、シフトファクター、モデルの学習と予測、プロファイル尤度、評価、図の保存。中身は `04_補助スクリプト/mat_common.py` と同じ（ほかのノートと同じ）。''')
code(COMMON_CELL)
md(r'''### 設定温度を使わないモデル（C_noTset）を足す・比べるモデル・結果の置き場所

共通の部品は変えず、このノートの中だけで、モデルの記号 `C_noTset` を足す（GP の入力の列を決める `layout` を包み、`C_noTset` のときだけ、C から `Tset` を外した列〔ln γ̇, ln P3_3, ln Tff〕を返す）。学習・予測・ハイパーパラメータの表は、共通の部品の関数がそのまま使える。結果は、学習結果フォルダの `<版>/全厚み総合/温度ホールドアウト検証/` にまとめる。''')
code(rf'''# セル3：C_noTset（C から加熱シリンダ温度 Tset を外したモデル）を足す・比べるモデル・結果の置き場所（このノートの中だけ。mat_common.py は変えない）
NO_TSET = "C_noTset"
OUT_NAME = "{OUT_NAME}"
_layout_common = layout


def layout(key):
    """GP の入力の列と、シフトに使う温度の列。C_noTset だけ、C の入力から Tset を外したもの（ln γ̇, ln P3_3, ln Tff。シフトなし。A_Tset・B_Tset の θ = 0 と同じ）。ほかのモデルは共通の部品のまま"""
    if key == NO_TSET:
        return ["ln_gd", "ln_P33", "ln_Tff"], None
    return _layout_common(key)


MODEL_JA[NO_TSET] = "C_noTset（シフト構造なし。C から加熱シリンダ温度 Tset を外し、3 変数を並べた通常の GPR。A_Tset・B_Tset の θ = 0 と同じ）"
MODEL_DIRS[NO_TSET] = "C_noTset_温度シフトなし_加熱シリンダ温度を入力に使わない"
SHIFT_KEYS = [k for k in model_keys(SHIFT_COLS, STRUCTS, SHIFT_FORMS) if k != "C"]   # シフトのあるモデル
KEYS = SHIFT_KEYS + ["C", NO_TSET]               # 比べるモデル（シフトのあるモデル、C、C_noTset）
if MAIN_KEY not in KEYS:
    print(f"MAIN_KEY（{{MAIN_KEY}}）は比べるモデルにないので、{{KEYS[0]}} を表示する")
    MAIN_KEY = KEYS[0]
_pal = ["#1f77b4", "#9467bd", "#2ca02c", "#17becf", "#8c564b", "#e377c2"]
MCOL = {{**{{k: _pal[i % len(_pal)] for i, k in enumerate(SHIFT_KEYS)}}, "C": "#7f7f7f", NO_TSET: "#ff7f0e"}}   # 図のモデルの色（C は灰色、C_noTset は橙）
ALL_T = "全温度"                                 # 参考の回（全温度で学習）の名前


def fold_name(T):
    """回の名前（回のフォルダの名前にも使う）：除いた温度 T [℃]。None は全温度"""
    return ALL_T if T is None else f"除いた温度{{T:g}}℃"


def out_dir(mat, fold=None, key=None, sub=None):
    """このノートの結果の置き場所：<学習結果フォルダ>/<版>/全厚み総合/温度ホールドアウト検証/。fold があれば回のフォルダ、key があればその下のモデルのフォルダ（sub があれば、さらにその下）"""
    return result_dir(mat, VERSION_DIRS[INPUT_MODE], THICK_TOTAL, OUT_NAME, *[p for p in (fold, key, sub) if p])


print("比べるモデル：")
for k in KEYS:
    print(f"  {{k}}：入力 {{'、'.join(input_labels(k))}}（シフトに使う温度：{{layout(k)[1] or 'なし'}}）")
print(f"結果の置き場所：<学習結果フォルダ>/{{VERSION_DIRS[INPUT_MODE]}}/{{THICK_TOTAL}}/{{OUT_NAME}}/")''')

# ---------------------------------------------------------------- 2. データ
md(r'''## 2. ファイルを選んで、データを読み込む

**このセルで、材料ごとにダイアログが 3 回開く**（①学習データ〔5data〕の CSV、②検証データ〔3data〕の CSV、③学習結果フォルダ。設定 `FILES` に書いたものは開かない）。読み方は、学習用・シフトファクター検討用のノートと同じ（問題データ一覧の「除外」の行と、η・γ̇・P3_3 が 0 以下の行を除く。3data の行は、5data の射出速度の範囲より遅い側を**低速側**、速い側を**高速側**に分ける）。

**判定**：設定温度が学習・検証とも 3 水準ある／0 以下の値の行がすべて一覧に登録済み／基準温度 Tref が設定温度の範囲の中にある／検証データに、学習データの射出速度の範囲の外の条件がある。''')
code("# セル4：ファイルを選んで、読み込む\n" + nbc.READ)

# ---------------------------------------------------------------- 3. 分け方
md(r'''## 3. 分け方（回ごとの学習データと検証データ）

材料ごとに、5data の設定温度を 1 つずつ除いて、残りを学習データ、除いた温度の 5data を検証データ（主）、除いた温度の 3data を別の検証データにする。`REF_ALL = True` なら、全温度で学習する回（検証は 3data すべて）も足す。表 `分け方.csv` に、回ごとの温度・行数・条件数を書く。

**判定**：学習の温度が 1 つ減り、除いた温度を含まない／検証は除いた温度だけ／学習と検証で条件が重ならない／除いた温度の 5data の射出速度が、すべて学習の射出速度にある（温度だけの外挿）／基準温度 Tref が、学習の温度の範囲（端を含む）にある／除いた温度の 5data・3data を全部の回で合わせると、元の行数になる（`HOLDOUT_TEMPS = None` のとき）。''')
code(r'''# セル5：分け方
FOLDS, FOLD_NAMES, FROWS = {}, {}, []
for mat in MATERIALS:
    full_tr, full_te = TRAIN[mat], TEST[mat]
    temps = sorted(full_tr["Tset"].unique())
    hold = temps if HOLDOUT_TEMPS is None else [t for t in temps if any(np.isclose(t, float(h)) for h in HOLDOUT_TEMPS)]
    if HOLDOUT_TEMPS is not None and not hold:
        print(f"★ {MAT_NAMES[mat]}：HOLDOUT_TEMPS {HOLDOUT_TEMPS} の温度が、この材料の設定温度（{'・'.join(f'{t:g}' for t in temps)} ℃）にないので、除く回はない")
    FOLD_NAMES[mat] = []
    for T in hold:
        is_t5, is_t3 = np.isclose(full_tr["Tset"], T), np.isclose(full_te["Tset"], T)
        v3 = full_te[is_t3].reset_index(drop=True)
        FOLDS[(mat, fold_name(T))] = {"T": float(T), "位置": "低い温度" if np.isclose(T, temps[0]) else ("高い温度" if np.isclose(T, temps[-1]) else "中の温度"),
                                      "学習": full_tr[~is_t5].reset_index(drop=True),
                                      "検証5": full_tr[is_t5].reset_index(drop=True).assign(検証="5data", group="除いた温度の5data"),
                                      "検証3": v3.assign(検証="3data", group="3data " + v3["group"])}
        FOLD_NAMES[mat].append(fold_name(T))
    if REF_ALL:
        FOLDS[(mat, ALL_T)] = {"T": None, "位置": ALL_T, "学習": full_tr, "検証5": full_tr.iloc[:0].assign(検証="5data", group="除いた温度の5data"),
                               "検証3": full_te.assign(検証="3data", group="3data " + full_te["group"])}
        FOLD_NAMES[mat].append(ALL_T)
    tref = CFG[(mat, FORMS_USED[0])]["Tref"]
    for fd in FOLD_NAMES[mat]:
        F = FOLDS[(mat, fd)]
        tr, v5, v3, T = F["学習"], F["検証5"], F["検証3"], F["T"]
        tt = sorted(tr["Tset"].unique())
        pos = {"低い温度": "低い温度（外挿）", "中の温度": "中の温度（内挿）", "高い温度": "高い温度（外挿）"}.get(F["位置"], "―")
        FROWS.append({"材料": MAT_NAMES[mat], "回": fd, "除いた温度 [℃]": T, "温度の位置": pos, "学習の温度 [℃]": "・".join(f"{t:g}" for t in tt), "Tref [℃]": tref,
                      "学習 行数": len(tr), "学習 条件数": tr["cond"].nunique(), "検証5 行数": len(v5), "検証5 条件数": v5["cond"].nunique(),
                      "検証3 行数": len(v3), "検証3 条件数": v3["cond"].nunique(),
                      "検証3 低速側 行数": int((v3["group"] == "3data 低速側").sum()), "検証3 高速側 行数": int((v3["group"] == "3data 高速側").sum()),
                      "学習の射出速度 [mm/s]": f"{tr['v'].min():g}〜{tr['v'].max():g}"})
        judge(tt[0] <= tref <= tt[-1], f"{fd}：基準温度 Tref = {tref:g} ℃ が、学習の温度（{'・'.join(f'{t:g}' for t in tt)} ℃）の範囲（端を含む）にある", mat, "分け方")
        if T is None:
            continue
        ok_t = len(tt) == len(temps) - 1 and not np.isclose(tr["Tset"], T).any() and np.isclose(v5["Tset"], T).all() and np.isclose(v3["Tset"], T).all()
        both = (set(tr["cond"]) & set(v5["cond"])) | (set(tr["cond"]) & set(v3["cond"]))
        judge(ok_t and not both, f"{fd}：学習は {len(tt)} 温度で {T:g} ℃ を含まず、検証（5data {len(v5)} 行・3data {len(v3)} 行）は {T:g} ℃ だけ。学習と検証で重なる条件 {len(both)} 個", mat, "分け方")
        extra_v = sorted(set(v5["v"]) - set(tr["v"]))
        judge(not extra_v, f"{fd}：除いた温度の 5data の射出速度が、すべて学習の射出速度にある（温度だけの外挿。学習にない射出速度：{extra_v or 'なし'}）", mat, "分け方")
    hold_f = [fd for fd in FOLD_NAMES[mat] if fd != ALL_T]
    if HOLDOUT_TEMPS is None and hold_f:
        n5, n3 = sum(len(FOLDS[(mat, fd)]["検証5"]) for fd in hold_f), sum(len(FOLDS[(mat, fd)]["検証3"]) for fd in hold_f)
        judge(n5 == len(full_tr) and n3 == len(full_te), f"除いた温度の 5data を全部の回で合わせると {n5} 行（元 {len(full_tr)} 行）、3data は {n3} 行（元 {len(full_te)} 行）", mat, "分け方")
FT = pd.DataFrame(FROWS)
display(FT)
for mat in MATERIALS:
    to_csv(FT[FT["材料"] == MAT_NAMES[mat]], os.path.join(out_dir(mat), "分け方.csv"))
print(f"回の数：{len(FOLDS)}（" + "、".join(f"{MAT_NAMES[m]} {len(FOLD_NAMES[m])}" for m in MATERIALS) + "）")''')

# ---------------------------------------------------------------- 4. プロファイル尤度
md(r'''## 4. 回ごとのシフトファクターの推定（プロファイル尤度。時間がかかる）

回ごとに、その回の学習データだけで、シフトのあるモデルの θ（アレニウス型は E_h・E_v [kJ/mol]。A は E_h = 0）を、シフトファクター検討用のノートと同じプロファイル尤度（粗い格子 → 細かい格子 → 見直し → 区間の延長。`mat_common.run_profile`）で推定する。学習が 2 温度だけの回は、温度の効果が 2 点の差でしか決まらないので、尤度の山は全温度より平らになりやすい。

**判定**：最良の θ が、粗い格子の端にも、細かい格子の端にもない（端なら `FORM_CFG` の範囲を広げる）。''')
code(r'''# セル6：プロファイル尤度（目安：回 1 つ・モデル 1 つで、A は 1〜2 分、B は 5〜10 分〔要確認〕）
PROF, BEST = {}, {}
t_all = time.time()
for (mat, fd), F in FOLDS.items():
    for key in SHIFT_KEYS:
        cfg = kcfg(mat, key)
        thn, tvn, un = theta_names(cfg)
        t0 = time.time()
        pts, conv = run_profile(key, F["学習"], cfg)
        PROF[(mat, fd, key)] = pts
        BEST[(mat, fd, key)] = b = profile_best(pts, key, cfg, conv)
        to_csv(pts.drop(columns="theta"), os.path.join(out_dir(mat, fd, key), "プロファイル尤度_格子.csv"))
        print(f"{MAT_NAMES[mat]} {fd} {key}：{len(pts)} 点、{time.time() - t0:.0f} 秒（取り直しが{'収束' if conv else '収束せず'}、見直しで LML が上がった点 {b['n_review_up']}）。"
              f"最良 {thn} = {b['th_best']:g}、{tvn} = {b['tv_best']:g} {un}、格子の LML {b['LML_best']:.1f}")
print(f"合計 {(time.time() - t_all) / 60:.1f} 分")
for (mat, fd, key), b in BEST.items():
    thn, tvn, un = theta_names(kcfg(mat, key))
    judge((not b["at_edge"]) and b["refine_converged"], f"{fd} {key}：最良の θ が、粗い格子の端にも、細かい格子の端にもない（{thn} = {b['th_best']:g}、{tvn} = {b['tv_best']:g} {un}。"
          f"取り直しが{'収束' if b['refine_converged'] else '収束せず'}）。端なら FORM_CFG の範囲を広げる。収束していなければ MAX_REFINE を増やす", mat, "尤度")''')

# ---------------------------------------------------------------- 5. 念入りな最適化と、C・C_noTset
md(r'''## 5. 念入りな最適化と、C・C_noTset の学習

回ごとに、次をまとめて並列に学習する（どれも、最適化のやり直し `N_RESTARTS` 回）。
- シフトのあるモデルの、**最良の θ** と **θ = 0** の 2 点（シフトファクター検討用のノートの 3.1b と同じ）。念入りな最適化の LML を格子の表に入れ、ΔLML（最良 − θ = 0）と 95% 区間を、その表から出し直す（`profile_careful`）。最良の θ の学習が、その回の最終モデル。
- **C**（Tset を GP の入力に使う）と **C_noTset**（Tset を使わない）。

C_noTset は、シフトのモデルの θ = 0 と同じモデルなので、念入りな最適化の θ = 0 の学習（プロファイル尤度の点の解から始めたもの）と、LML がほぼ同じになるはず。**LML の高いほうを C_noTset に使う**（同じモデルの、最適化がよく進んだほうを使うだけなので、比べ方は変わらない。どちらを使ったかは表の「C_noTset の出どころ」）。

**局所解の見直し**（2 温度だけの学習では、LML の低い解に止まりやすい。2026-11-16 の試しの実行で、粉砕材混合 PP の 210 ℃ を除く回の A_Tset_Arr が、C_noTset より約 14 低い解に止まっていた）：
- (a) C_noTset の学習の LML が、シフトのモデルの θ = 0 の念入りな最適化より `LML_CHECK` 以上高いとき：そのシフトのモデルのプロファイル尤度の全点を、C_noTset の解から最適化し直し（LML が上がった点を置き換える。表の `review_gain` に足す）、95% 区間の延長もやり直して（最良の点が動くと、区間の外側の隣の点が足りなくなるため）、最良の θ を決め直して、最良の θ と θ = 0 を念入りに学習し直す（「★ 局所解の見直し」と表示する）。
- (b) C の LML が C_noTset より `LML_CHECK` 以上低いとき：C_noTset の解（Tset の長さスケールを長くしたもの）からも C を学習し、LML の高いほうを C に使う。C は C_noTset に Tset を足したものだが、線形項が Tset も含むので、C の LML が C_noTset より少し低いこともある。

**判定**：
- ΔLML が基準以上（A は 1.92、B は 3.0。その回の学習データで、温度のシフトが尤度で支持される）。
- 95% 区間の端が、細かい格子の中で決まった。
- C_noTset の学習の LML が、θ = 0 の念入りな最適化の LML より `LML_CHECK` 以上低くない（低ければ、C_noTset の学習が局所解に止まった。そのときは θ = 0 の学習を使う）。
- シフトのモデルの最良の LML が、C_noTset 以上（`LML_CHECK` の余裕を見る。θ = 0 で C_noTset と同じになるので、低ければ、見直しの後もシフトのモデルが局所解に止まっている）。''')
code(r'''# セル7：念入りな最適化（最良の θ と θ = 0）と、C・C_noTset の学習（並列。目安：数分〜十数分）
from joblib import Parallel, delayed
t0 = time.time()
FITS, ZERO = {}, {}
jobs = []
for (mat, fd) in FOLDS:
    for key in SHIFT_KEYS:
        b = BEST[(mat, fd, key)]
        jobs += [(mat, fd, key, "best", b["th_best"], b["tv_best"], b["theta_best"]), (mat, fd, key, "zero", 0.0, 0.0, b["theta_zero"])]
    jobs += [(mat, fd, key, "fit", 0.0, 0.0, None) for key in ["C", NO_TSET]]
res = Parallel(n_jobs=min(N_JOBS, len(jobs)))(delayed(fit_job)(key, FOLDS[(mat, fd)]["学習"], kcfg(mat, key), th, tv, N_RESTARTS, th0, INPUT_MODE)
                                              for mat, fd, key, kind, th, tv, th0 in jobs)
for (mat, fd, key, kind, *_), f in zip(jobs, res):
    (ZERO if kind == "zero" else FITS)[(mat, fd, key)] = f
print(f"念入りな最適化と C・C_noTset の学習：{len(jobs)} 回、{(time.time() - t0) / 60:.1f} 分")


def lml_of(f):
    return float(f["gp"].log_marginal_likelihood_value_)


# 局所解の見直し（a）：Tset のシフトのモデルの θ = 0 は、C_noTset と同じモデル。C_noTset の学習の LML が、θ = 0 の念入りな最適化より LML_CHECK 以上高ければ、
# シフトのモデルのプロファイル尤度が LML の低い解に止まっている。そのときは、格子の全点を C_noTset の解から最適化し直し（LML が上がった点を置き換える）、
# 最良の θ を決め直して、最良の θ と θ = 0 を念入りに学習し直す
REDO = [(mat, fd, key) for (mat, fd) in FOLDS for key in SHIFT_KEYS
        if layout(key)[1] == "Tset" and lml_of(FITS[(mat, fd, NO_TSET)]) > lml_of(ZERO[(mat, fd, key)]) + LML_CHECK]
jobs2 = []
for mat, fd, key in REDO:
    cfg, tr = kcfg(mat, key), FOLDS[(mat, fd)]["学習"]
    th_no = np.asarray(FITS[(mat, fd, NO_TSET)]["gp"].kernel_.theta, float)
    pts, old = PROF[(mat, fd, key)], BEST[(mat, fd, key)]
    items = [(float(h), float(v), 0, th_no) for h, v in zip(pts["th"], pts["tv"])]
    n = min(N_JOBS, len(items))
    rows = [r for rr in Parallel(n_jobs=n)(delayed(prof_points)(key, tr, cfg, items[i::n], INPUT_MODE) for i in range(n)) for r in rr]
    pos = {(round(float(h), 6), round(float(v), 6)): i for i, (h, v) in enumerate(zip(pts["th"], pts["tv"]))}
    L, NS, G, TH = pts["LML"].to_numpy(float).copy(), pts["noise_sd"].to_numpy(float).copy(), pts["review_gain"].to_numpy(float).copy(), pts["theta"].tolist()
    for r in rows:
        i = pos[(round(r["th"], 6), round(r["tv"], 6))]
        if r["LML"] > L[i] + 1e-3:
            G[i] += r["LML"] - L[i]
            L[i], NS[i], TH[i] = r["LML"], r["noise_sd"], r["theta"]
    PROF[(mat, fd, key)] = pts.assign(LML=L, noise_sd=NS, review_gain=G, theta=TH)
    for _ in range(MAX_EXTEND):                     # 最良の点が動くと、95% 区間の外側の隣の点が足りなくなるので、区間の延長をやり直す（run_profile の 4 と同じ）
        need = interval_gaps(PROF[(mat, fd, key)], key, cfg)
        if not need:
            break
        items = [(h, v, PROFILE_RESTARTS, np.asarray(t, float)) for (h, v), t in need.items()]
        n = min(N_JOBS, len(items))
        rows = [r for rr in Parallel(n_jobs=n)(delayed(prof_points)(key, tr, cfg, items[i::n], INPUT_MODE) for i in range(n)) for r in rr]
        PROF[(mat, fd, key)] = pd.concat([PROF[(mat, fd, key)], pd.DataFrame([{**r, "stage": "区間の延長", "review_gain": 0.0} for r in rows])],
                                         ignore_index=True).sort_values(["th", "tv"]).reset_index(drop=True)
    BEST[(mat, fd, key)] = b = profile_best(PROF[(mat, fd, key)], key, cfg, old["refine_converged"])
    print(f"★ 局所解の見直し：{MAT_NAMES[mat]} {fd} {key}：C_noTset の学習の LML {lml_of(FITS[(mat, fd, NO_TSET)]):.2f} が、θ = 0 の念入りな最適化 {lml_of(ZERO[(mat, fd, key)]):.2f} より高いので、"
          f"格子の {len(pts)} 点を C_noTset の解から最適化し直した（LML が上がった点 {int((L > pts['LML'].to_numpy(float) + 1e-3).sum())}）。"
          f"最良の θ：{old['th_best']:g}・{old['tv_best']:g} → {b['th_best']:g}・{b['tv_best']:g}、格子の最良の LML {old['LML_best']:.2f} → {b['LML_best']:.2f}")
    jobs2 += [(mat, fd, key, "best", b["th_best"], b["tv_best"], b["theta_best"]), (mat, fd, key, "zero", 0.0, 0.0, th_no)]
if jobs2:
    res2 = Parallel(n_jobs=min(N_JOBS, len(jobs2)))(delayed(fit_job)(key, FOLDS[(mat, fd)]["学習"], kcfg(mat, key), th, tv, N_RESTARTS, th0, INPUT_MODE)
                                                    for mat, fd, key, kind, th, tv, th0 in jobs2)
    for (mat, fd, key, kind, th, tv, _), f in zip(jobs2, res2):
        dct = ZERO if kind == "zero" else FITS
        prev = dct[(mat, fd, key)]
        moved = kind == "best" and not (np.isclose(prev["th"], th) and np.isclose(prev["tv"], tv))   # 最良の θ が動いたら、新しい θ の学習を最終モデルにする
        if moved or lml_of(f) > lml_of(prev):
            dct[(mat, fd, key)] = f
EROWS = []
for (mat, fd) in FOLDS:
    for key in SHIFT_KEYS:
        cfg = kcfg(mat, key)
        PROF[(mat, fd, key)], BEST[(mat, fd, key)] = profile_careful(PROF[(mat, fd, key)], BEST[(mat, fd, key)], key, cfg, FITS[(mat, fd, key)], ZERO[(mat, fd, key)])
        to_csv(PROF[(mat, fd, key)].drop(columns="theta"), os.path.join(out_dir(mat, fd, key), "プロファイル尤度_格子.csv"))
        b = BEST[(mat, fd, key)]
        thn, tvn, un = theta_names(cfg)
        EROWS.append({"材料": MAT_NAMES[mat], "回": fd, "除いた温度 [℃]": FOLDS[(mat, fd)]["T"], "モデル": key, "型": FORM_JA[cfg["type"]], "θ の名前": f"{thn}・{tvn}", "単位": un,
                      "支持の基準": lrt_limit(key), "尤度で支持": bool(b["dLML_vs_0"] >= lrt_limit(key)),
                      **{k: v for k, v in b.items() if k not in ("theta_best", "theta_zero")}})
        span = (f"{thn} {b['th_lo']:g}〜{b['th_hi']:g}・" if key.startswith("B") else "") + f"{tvn} {b['tv_lo']:g}〜{b['tv_hi']:g} {un}"
        judge(b["dLML_vs_0"] >= lrt_limit(key), f"{fd} {key}：学習の温度で、温度のシフトが尤度で支持される（ΔLML {b['dLML_vs_0']:+.2f}、基準 {lrt_limit(key):g}。"
              f"{tvn} = {b['tv_best']:g} {un}、95% 区間 {span}）", mat, "尤度")
        judge(b["interval_closed"] and not b["interval_at_range"], f"{fd} {key}：95% 区間（{span}）の端が、細かい格子の中で決まった"
              f"{'' if b['interval_closed'] else '（★ 計算していない隣の点が残った：MAX_EXTEND を増やす）'}{'（★ 粗い格子の範囲の端に届いた：FORM_CFG の範囲を広げる）' if b['interval_at_range'] else ''}", mat, "尤度")
    # C_noTset：このノートの学習と、シフトのモデルの θ = 0 の念入りな最適化（同じモデル）のうち、LML の高いほうを使う
    own = FITS[(mat, fd, NO_TSET)]
    cands = [("このノートの学習", own)] + [(f"{k} の θ = 0", ZERO[(mat, fd, k)]) for k in SHIFT_KEYS if layout(k)[1] == "Tset"]
    lml = [float(f["gp"].log_marginal_likelihood_value_) for _, f in cands]
    i = int(np.argmax(lml))
    if i > 0:
        FITS[(mat, fd, NO_TSET)] = {**cands[i][1], "key": NO_TSET, "th": 0.0, "tv": 0.0, "cols": layout(NO_TSET)[0]}
    FOLDS[(mat, fd)]["C_noTset の出どころ"] = cands[i][0]
    if len(cands) > 1:
        judge(lml[0] >= max(lml) - LML_CHECK, f"{fd}：C_noTset の学習の LML {lml[0]:.2f} が、同じモデル（θ = 0）の念入りな最適化の LML {max(lml[1:]):.2f} より {LML_CHECK:g} 以上低くない"
              f"（使ったもの：{cands[i][0]}）", mat, "学習")
        for k in [k for k in SHIFT_KEYS if layout(k)[1] == "Tset"]:     # シフトのモデルは C_noTset を含む（θ = 0）ので、最良の LML は C_noTset 以上のはず
            lb = BEST[(mat, fd, k)]["LML_best"]
            judge(lb >= max(lml) - LML_CHECK, f"{fd} {k}：最良の θ の LML {lb:.2f} が、C_noTset（θ = 0 と同じモデル）の LML {max(lml):.2f} 以上（低ければ、シフトのモデルが局所解に止まった。"
                  f"N_RESTARTS・PROFILE_RESTARTS を増やす）" + ("。局所解の見直しをした" if (mat, fd, k) in REDO else ""), mat, "学習")
# 局所解の見直し（b）：C は C_noTset に Tset の入力を足したもの。C の LML が C_noTset より LML_CHECK 以上低ければ、C_noTset の解（Tset の長さスケールを長く〔50〕したもの）からも学習し、
# LML の高いほうを C に使う（線形項は Tset も含むので、C の LML は C_noTset より少し低くなることもある。この見直しは、局所解に止まった C を拾うため）
jobs3 = []
for (mat, fd) in FOLDS:
    if lml_of(FITS[(mat, fd, "C")]) < lml_of(FITS[(mat, fd, NO_TSET)]) - LML_CHECK:
        th_c = np.insert(np.asarray(FITS[(mat, fd, NO_TSET)]["gp"].kernel_.theta, float), 1 + len(layout(NO_TSET)[0]), np.log(50.0))
        jobs3.append((mat, fd, th_c))
if jobs3:
    res3 = Parallel(n_jobs=min(N_JOBS, len(jobs3)))(delayed(fit_job)("C", FOLDS[(mat, fd)]["学習"], kcfg(mat, "C"), 0.0, 0.0, N_RESTARTS, th_c, INPUT_MODE) for mat, fd, th_c in jobs3)
    for (mat, fd, _), f in zip(jobs3, res3):
        prev = lml_of(FITS[(mat, fd, "C")])
        if lml_of(f) > prev:
            FITS[(mat, fd, "C")] = f
        print(f"局所解の見直し：{MAT_NAMES[mat]} {fd} C：LML {prev:.2f}（C_noTset {lml_of(FITS[(mat, fd, NO_TSET)]):.2f}）→ C_noTset の解からの学習 {lml_of(f):.2f}。"
              f"{'こちらを使う' if lml_of(f) > prev else '元のまま'}")
ET = pd.DataFrame(EROWS)
if len(ET):
    display(ET[["材料", "回", "モデル", "th_best", "th_lo", "th_hi", "tv_best", "tv_lo", "tv_hi", "LML_best", "LML_at_0", "dLML_vs_0", "支持の基準", "尤度で支持", "interval_closed", "n_review_up"]].round(2))
for mat in MATERIALS:
    if len(ET):
        to_csv(ET[ET["材料"] == MAT_NAMES[mat]], os.path.join(out_dir(mat), "シフトファクターの推定_回ごと.csv"))''')

# ---------------------------------------------------------------- 6. ハイパーパラメータと保存
md(r'''## 6. ハイパーパラメータと、学習済みモデルの保存

回 × モデルごとの、振幅・長さスケール・線形項・ノイズ・LML（`ハイパーパラメータ.csv`）。列「C_noTset との LML の差」は、同じ回の中だけで意味がある（C は C_noTset に Tset の長さスケールを 1 つ足したもの、A は係数 1 つ、B は係数 2 つを足したもの）。学習済みモデルは、回のフォルダの下のモデルのフォルダに `学習済みモデル_<モデル>.joblib` として保存する（`SAVE_MODELS`。学習した温度・除いた温度・データの場所も記録する）。

**判定**：ノイズ分散が下限（`NOISE_LB`）に張り付いていない。''')
code(r'''# セル8：ハイパーパラメータと、学習済みモデルの保存
import joblib
import sklearn
KROWS = []
for (mat, fd), F in FOLDS.items():
    lml_no = float(FITS[(mat, fd, NO_TSET)]["gp"].log_marginal_likelihood_value_)
    for key in KEYS:
        f = FITS[(mat, fd, key)]
        r = kernel_row(f)
        if key in ("C", NO_TSET):                       # シフトの係数はない
            th_n, tv_n, _ = theta_names(f["cfg"])
            r[th_n], r[tv_n] = np.nan, np.nan
        KROWS.append({"材料": MAT_NAMES[mat], "回": fd, **r, "C_noTset との LML の差": r["LML"] - lml_no})
        if SAVE_MODELS:
            joblib.dump({"fit": f, "材料": MAT_NAMES[mat], "モデル": key, "説明": model_title(key), "入力": input_labels(key), "入力の取り方": INPUT_MODE,
                         "シフトファクター": kcfg(mat, key), "回": fd, "除いた温度": F["T"], "学習した温度": sorted(F["学習"]["Tset"].unique().tolist()),
                         "学習データのファイル": show_path(SOURCES[mat]["5data"]), "検証データのファイル": show_path(SOURCES[mat]["3data"]),
                         "scikit-learn": sklearn.__version__, "作成日": time.strftime("%Y-%m-%d")},
                        os.path.join(out_dir(mat, fd, key), f"学習済みモデル_{key}.joblib"))
        judge(not r["noise_at_lower_bound"], f"{fd} {key}：ノイズ分散（{r['noise_var']:.2e}）が下限 {NOISE_LB:g} に張り付いていない", mat, "学習")
KT = pd.DataFrame(KROWS)
for mat in MATERIALS:
    t = KT[KT["材料"] == MAT_NAMES[mat]]
    to_csv(t.dropna(axis=1, how="all"), os.path.join(out_dir(mat), "ハイパーパラメータ.csv"))
    print(f"\n【{MAT_NAMES[mat]}】LML（同じ回の中だけで比べる）と、C_noTset との差")
    display(t.pivot_table(index="回", columns="model", values="LML", sort=False)[KEYS].round(2))
    display(t.pivot_table(index="回", columns="model", values="C_noTset との LML の差", sort=False)[KEYS].round(2))
print("保存した学習済みモデル：" + (f"{len(KT)} 個（回のフォルダの下のモデルのフォルダ）" if SAVE_MODELS else "なし（SAVE_MODELS = False）"))''')

# ---------------------------------------------------------------- 7. 予測と指標
md(r'''## 7. 除いた温度の予測と指標

回ごとに、その回の学習で作った各モデルで、除いた温度の 5data と 3data を予測する（そこで測った P3_3・Tff を入力に使う）。指標（ln η）：RMSE（小さいほど良い。0.1 は粘度で約 10%）、R²、95% 予測区間に入る割合（被覆率）、NLPD。範囲は「5data」（除いた温度の 5data。**主**）、「3data 全体・低速側・高速側」、「合計」。全温度の回は、3data だけ（シフトファクター検討用のノートの 3data の検証と同じ）。

**判定**（除いた温度の 5data）：シフトのあるモデルの RMSE が、C の `RMSE_RATIO` 倍以内（シフトの式で温度を運んでも、GP の入力で表すより悪くならない）／各モデルの被覆率が `COVER_MIN` 以上。''')
code(r'''# セル9：除いた温度の予測と指標
PRED, MROWS = {}, []
COLS_P = ["file", "cond", "Tset", "H", "v", "検証", "group", "eta", "gd", "P33", "Tff", "ln_eta"]
RANGES = ["5data", "3data 全体", "3data 低速側", "3data 高速側", "3data 範囲内", "合計"]


def loto_rows(mat, fd, key, te, mu, sd):
    """指標を、範囲（除いた温度の 5data・3data 全体・低速側・高速側・合計）ごとに返す"""
    y = te["ln_eta"].to_numpy()
    subs = {"5data": (te["検証"] == "5data").to_numpy(), "3data 全体": (te["検証"] == "3data").to_numpy()}
    for g in ["低速側", "高速側", "範囲内"]:
        subs[f"3data {g}"] = (te["group"] == f"3data {g}").to_numpy()
    subs["合計"] = np.ones(len(te), bool)
    return [{"材料": MAT_NAMES[mat], "回": fd, "除いた温度 [℃]": FOLDS[(mat, fd)]["T"], "モデル": key, "範囲": s, "行数": int(m.sum()), "条件数": int(te.loc[m, "cond"].nunique()),
             **evaluate(y[m], mu[m], sd[m])} for s, m in subs.items() if m.sum() > 0]


for (mat, fd), F in FOLDS.items():
    te = pd.concat([F["検証5"], F["検証3"]], ignore_index=True)
    for key in KEYS:
        mu, sd, sdf, nsd = predict(FITS[(mat, fd, key)], te)
        p = te[COLS_P].assign(予測_ln_eta=mu, 予測標準偏差=sd, 潜在関数の標準偏差=sdf, 予測η=np.exp(mu), 下限95=np.exp(mu - 1.96 * sd), 上限95=np.exp(mu + 1.96 * sd))
        PRED[(mat, fd, key)] = p
        to_csv(p, os.path.join(out_dir(mat, fd, key), "予測値.csv"))
        MROWS += loto_rows(mat, fd, key, te, mu, sd)
MT = pd.DataFrame(MROWS)


def metric_of(mat, fd, key, rng="5data", col="RMSE"):
    """指標の表から 1 つの値（なければ NaN）"""
    t = MT[(MT["材料"] == MAT_NAMES[mat]) & (MT["回"] == fd) & (MT["モデル"] == key) & (MT["範囲"] == rng)]
    return float(t[col].iloc[0]) if len(t) else np.nan


for mat in MATERIALS:
    t = MT[MT["材料"] == MAT_NAMES[mat]]
    to_csv(t, os.path.join(out_dir(mat), "予測の指標_回ごと.csv"))
    for rng in ["5data", "3data 全体", "3data 低速側", "3data 高速側"]:
        s = t[t["範囲"] == rng]
        if len(s):
            print(f"\n【{MAT_NAMES[mat]}】{rng} の ln η の RMSE（行：回、列：モデル）")
            display(s.pivot_table(index="回", columns="モデル", values="RMSE", sort=False)[[k for k in KEYS if k in set(s["モデル"])]].round(4))
    s = t[t["範囲"] == "5data"]
    if len(s):
        print(f"【{MAT_NAMES[mat]}】5data の 95% 予測区間に入る割合（被覆率）")
        display(s.pivot_table(index="回", columns="モデル", values="coverage95", sort=False)[KEYS].round(3))
    for fd in [f for f in FOLD_NAMES[mat] if f != ALL_T]:
        rc = metric_of(mat, fd, "C")
        for key in SHIFT_KEYS:
            rk = metric_of(mat, fd, key)
            judge(rk <= RMSE_RATIO * rc, f"{fd} {key}：除いた温度の 5data の RMSE {rk:.4f} が、C（{rc:.4f}）の {RMSE_RATIO:g} 倍以内", mat, "検証")
        cov = {k: metric_of(mat, fd, k, col="coverage95") for k in KEYS}
        judge(all(v >= COVER_MIN for v in cov.values()), f"{fd}：除いた温度の 5data の被覆率が、どのモデルも {COVER_MIN:g} 以上（" + "、".join(f"{k} {v:.2f}" for k, v in cov.items()) + "）", mat, "検証")''')

# ---------------------------------------------------------------- 8. 図：RMSE と係数
md(r'''## 8. 図：回ごとの誤差と、シフトファクターの係数

- **`予測のRMSE_回ごと.png`**（材料ごと）：左が除いた温度の 5data（主）、右が除いた温度の 3data（全温度の回は 3data すべて）の RMSE。横軸が回、棒がモデル（C は灰色、C_noTset は橙）。**シフトのモデルの棒が C_noTset より低いほど、シフトが学習にない温度の予測に効いている**。
- **`シフトファクターの係数_回ごと.png`**（材料ごと）：シフトのモデルごとに、回ごとの係数（縦 θ_v は丸、B の横 θ_h は四角）の最良値と 95% 区間。灰色の帯は、全温度の回の θ_v の 95% 区間。どの温度を除いても帯の近くにあれば、係数は安定している。

**判定**：除いた回の θ_v（最良値）が、全温度の回の θ_v の 95% 区間に入る（`REF_ALL = True` のとき）。''')
code(r'''# セル10：図（回ごとの RMSE と、シフトファクターの係数）
for mat in MATERIALS:
    hold = [fd for fd in FOLD_NAMES[mat] if fd != ALL_T]
    # 回ごとの RMSE
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.2))
    for ax, rng, ttl in [(axes[0], "5data", "除いた温度の 5data（温度だけの外挿・内挿。主）"), (axes[1], "3data 全体", "除いた温度の 3data（温度と射出速度の外挿）")]:
        fds = hold + ([ALL_T] if rng != "5data" and ALL_T in FOLD_NAMES[mat] else [])
        if not fds:
            ax.axis("off")
            continue
        w = 0.8 / len(KEYS)
        for i, key in enumerate(KEYS):
            ax.bar(np.arange(len(fds)) + i * w, [metric_of(mat, fd, key, rng) for fd in fds], width=w, color=MCOL[key], label=key, edgecolor="k", linewidth=0.4)
        ax.set_xticks(np.arange(len(fds)) + w * (len(KEYS) - 1) / 2)
        ax.set_xticklabels([fd.replace("除いた温度", "除いた ") for fd in fds])
        ax.set_ylabel("RMSE（ln η）")
        ax.set_title(ttl)
        ax.grid(axis="y", ls="--", alpha=0.5)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=MCOL[k], edgecolor="k", linewidth=0.4) for k in KEYS], labels=KEYS,
               loc="lower center", ncol=len(KEYS), fontsize=11, framealpha=1.0)   # 凡例は図の下に 1 つ（棒に重ならないように）
    fig.suptitle(f"{MAT_NAMES[mat]}　加熱シリンダ温度を 1 つ除いて学習したモデルの、除いた温度の予測誤差（小さいほど良い。{VERSION_DIRS[INPUT_MODE]}）")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    save_fig(fig, out_dir(mat), "予測のRMSE_回ごと")
    # 回ごとの係数
    if not SHIFT_KEYS:
        continue
    fds = FOLD_NAMES[mat]
    fig, axes = plt.subplots(1, len(SHIFT_KEYS), figsize=(7.5 * len(SHIFT_KEYS), 5.8), squeeze=False)
    for ax, key in zip(axes[0], SHIFT_KEYS):
        thn, tvn, un = theta_names(kcfg(mat, key))
        x = np.arange(len(fds))
        for w, nm, mk, off, col in [("tv", tvn, "o", -0.08, "#1f77b4")] + ([("th", thn, "s", 0.08, "#d62728")] if key.startswith("B") else []):
            bb = [BEST[(mat, fd, key)] for fd in fds]
            y = np.array([b[f"{w}_best"] for b in bb])
            lo, hi = np.array([b[f"{w}_lo"] for b in bb]), np.array([b[f"{w}_hi"] for b in bb])
            ax.errorbar(x + off, y, yerr=[y - lo, hi - y], fmt=mk, color=col, ms=9, capsize=5, elinewidth=1.8, label=f"{nm}（最良値と 95% 区間）")
        if ALL_T in fds:
            ba = BEST[(mat, ALL_T, key)]
            ax.axhspan(ba["tv_lo"], ba["tv_hi"], color="gray", alpha=0.18, label=f"全温度の {tvn} の 95% 区間（{ba['tv_lo']:g}〜{ba['tv_hi']:g}）")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([fd.replace("除いた温度", "除いた ") for fd in fds])
        ax.set_ylabel(f"シフトファクターの係数 ({un})")
        ax.set_title(f"{key}（{FORM_JA[kcfg(mat, key)['type']]}）")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=9, loc="best", framealpha=0.9)
    fig.suptitle(f"{MAT_NAMES[mat]}　回ごとのシフトファクターの係数（その回の学習の温度だけで推定）")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, out_dir(mat), "シフトファクターの係数_回ごと")
    if ALL_T in fds:
        for key in SHIFT_KEYS:
            ba = BEST[(mat, ALL_T, key)]
            thn, tvn, un = theta_names(kcfg(mat, key))
            for fd in hold:
                v = BEST[(mat, fd, key)]["tv_best"]
                judge(ba["tv_lo"] <= v <= ba["tv_hi"], f"{fd} {key}：{tvn} の最良値 {v:g} {un} が、全温度の 95% 区間（{ba['tv_lo']:g}〜{ba['tv_hi']:g}）に入る（係数が、除いた温度によらず安定）", mat, "係数")''')

# ---------------------------------------------------------------- 9. 図：除いた温度の予測曲線とパリティ図
md(r'''## 9. 図：除いた温度のせん断速度 ― 粘度と、パリティ図

- **`除いた温度<T>℃/せん断速度_粘度_除いた温度の予測.png`**：厚みごとに、除いた温度の実測（塗り：5data、白抜き：3data。ショットごと）と、その回の各モデルの予測線（平均）と 95% 予測区間（薄い帯）。予測線の P3_3・Tff は、除いた温度・その厚みの条件平均を ln γ̇ で内挿した断面（学習用のノートの図と同じ作り方。`flow_inputs`）。
- **`除いた温度<T>℃/<モデル>/予測検証_パリティ.png`**（`PARITY_FIGS`）：除いた温度の実測と予測の粘度（縦の線は 95% 予測区間）。点の形は、5data（丸）、3data の低速側（三角）・高速側（四角）。全温度の回は 3data だけ。''')
code(r'''# セル11：図（除いた温度のせん断速度 ― 粘度、パリティ図）
n_saved = 0
PGROUPS = {"除いた温度の5data": ("#2ca02c", "o"), "3data 低速側": ("#1f77b4", "^"), "3data 高速側": ("#d62728", "s"), "3data 範囲内": ("#9467bd", "D")}
for (mat, fd), F in FOLDS.items():
    T = F["T"]
    if T is not None:                                   # 除いた温度のせん断速度 ― 粘度（厚みごと）
        pool = pd.concat([F["検証5"], F["検証3"]], ignore_index=True)
        Hs = sorted(pool["H"].unique())
        ncol = 2 if len(Hs) == 4 else len(Hs)
        nrow = int(np.ceil(len(Hs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(7.8 * ncol, 6.2 * nrow), squeeze=False)
        for ax, h in zip(axes.flat, Hs):
            sub = pool[np.isclose(pool["H"], h)]
            lg = np.linspace(sub["ln_gd"].min() - 0.3, sub["ln_gd"].max() + 0.3, 120)
            X = flow_inputs(sub, lg, h, T)
            for key in KEYS:
                mu, sd, _, _ = predict(FITS[(mat, fd, key)], X)
                ax.fill_between(X["gd"], np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd), color=MCOL[key], alpha=0.10, lw=0)
                ax.plot(X["gd"], np.exp(mu), color=MCOL[key], lw=2.2, ls="--" if key == NO_TSET else "-", label=key)
            for vk, face, lab in [("5data", "0.25", f"{T:g} ℃ の 5data（実測）"), ("3data", "white", f"{T:g} ℃ の 3data（実測）")]:
                m = (sub["検証"] == vk).to_numpy()
                if m.any():
                    ax.scatter(sub.loc[m, "gd"], sub.loc[m, "eta"], s=38, facecolors=face, edgecolors="k", linewidths=0.9, zorder=5, label=lab)
            ax.set_xscale("log")
            ax.set_yscale("log")
            log_ticks(ax)
            ax.set_title(f"厚み {h:g} mm")
            ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ (1/s)")
            ax.set_ylabel(r"粘度 $\eta$ (Pa·s)")
            ax.grid(True, which="both", ls=":", alpha=0.5)
            ax.legend(fontsize=9)
        for ax in list(axes.flat)[len(Hs):]:
            ax.axis("off")
        fig.suptitle(f"{MAT_NAMES[mat]}　{T:g} ℃ を除いて学習したモデルの、{T:g} ℃ の予測（学習：{'・'.join(f'{t:g}' for t in sorted(F['学習']['Tset'].unique()))} ℃ の 5data。帯：95% 予測区間）")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        n_saved += len(save_fig(fig, out_dir(mat, fd), "せん断速度_粘度_除いた温度の予測"))
    if PARITY_FIGS:                                     # パリティ図（回 × モデル）
        for key in KEYS:
            p = PRED[(mat, fd, key)]
            r5 = metric_of(mat, fd, key, "5data")
            r3 = metric_of(mat, fd, key, "3data 全体")
            ttl = (f"{MAT_NAMES[mat]}　{fd}　{model_title(key)}\n" + (f"5data：RMSE(ln η) {r5:.3f}、被覆率 {metric_of(mat, fd, key, '5data', 'coverage95'):.2f}　" if np.isfinite(r5) else "")
                   + (f"3data：RMSE(ln η) {r3:.3f}" if np.isfinite(r3) else ""))
            n_saved += len(parity_plot(p, ttl, out_dir(mat, fd, key), "予測検証_パリティ", show=SHOW_FIGS and key == MAIN_KEY, groups=PGROUPS))
print(f"保存した図：{n_saved} 枚")''')

# ---------------------------------------------------------------- 10. まとめ
md(r'''## 10. まとめ：設定温度のシフトは要るか

材料とモデルごとに、除いた温度の 5data の RMSE を、**C_noTset（設定温度を使わない）と回ごとに比べる**（比 = モデル ÷ C_noTset）。比が 1/`RMSE_RATIO`（1/1.1）より小さい回を「良い」、`RMSE_RATIO` より大きい回を「悪い」、その間を「同等」として数え、次の目安を出す（n は回の数）。

| 良い回 | 悪い回 | 結論（設定温度を、そのモデルの形で使うのは） |
|---|---|---|
| n | 0 | 要る |
| 1 以上（n 未満） | 0 | どちらかといえば要る |
| 0 | 0 | どちらでもよい（除いた温度の誤差は同等） |
| 0 | 1 以上（n 未満） | どちらかといえば要らない |
| 0 | n | 要らない |
| 1 以上 | 1 以上 | 決まらない（★ 回によって分かれる） |

あわせて、学習の尤度で支持された回の数（シフトのモデルは ΔLML が基準以上、C は C − C_noTset の LML の差が 1.92 以上）、除いた温度の 3data の比（参考）、C との比、係数の安定（全温度の区間に入った回の数）を並べる。C の行は「温度の効果を、シフトの式でなく GP の入力で表す」場合。結果は `まとめ.csv`。

**判定**：シフトのあるモデルの除いた温度の 5data の RMSE が、すべての回で C_noTset より小さい（★ なら、設定温度のシフトが効かない回がある）。''')
code(r'''# セル12：まとめ
SUM = []


def verdict(n, nb, nw):
    """良い回 nb・悪い回 nw・回の数 n → 結論"""
    if nb and nw:
        return "決まらない（★ 回によって分かれる）"
    if nb:
        return "要る" if nb == n else "どちらかといえば要る"
    if nw:
        return "要らない" if nw == n else "どちらかといえば要らない"
    return "どちらでもよい（除いた温度の誤差は同等）"


for mat in MATERIALS:
    hold = [fd for fd in FOLD_NAMES[mat] if fd != ALL_T]
    if not hold:
        print(f"【{MAT_NAMES[mat]}】除く回がないので、まとめはない")
        continue
    for key in SHIFT_KEYS + ["C"]:
        r5 = {fd: metric_of(mat, fd, key) / metric_of(mat, fd, NO_TSET) for fd in hold}
        r3 = {fd: metric_of(mat, fd, key, "3data 全体") / metric_of(mat, fd, NO_TSET, "3data 全体") for fd in hold}
        nb, nw = sum(v < 1 / RMSE_RATIO for v in r5.values()), sum(v > RMSE_RATIO for v in r5.values())
        P = {fd: FOLDS[(mat, fd)]["位置"] for fd in hold}       # 列の名前は、材料によらない「低い温度・中の温度・高い温度」（除いた温度の値は、列「除いた温度」）
        row = {"材料": MAT_NAMES[mat], "版": VERSION_DIRS[INPUT_MODE], "モデル": key, "回の数": len(hold),
               "除いた温度": "・".join(f"{P[fd]} {FOLDS[(mat, fd)]['T']:g} ℃" for fd in hold),
               **{f"5data の RMSE {P[fd]}": metric_of(mat, fd, key) for fd in hold},
               **{f"5data の RMSE の比（÷ C_noTset）{P[fd]}": v for fd, v in r5.items()},
               "5data の比の幾何平均": float(np.exp(np.mean(np.log(list(r5.values()))))),
               "良い回": nb, "同等の回": len(hold) - nb - nw, "悪い回": nw, "結論": verdict(len(hold), nb, nw),
               **{f"3data の比（÷ C_noTset）{P[fd]}": v for fd, v in r3.items()}}
        if key in SHIFT_KEYS:
            row["尤度で支持された回"] = sum(BEST[(mat, fd, key)]["dLML_vs_0"] >= lrt_limit(key) for fd in hold)
            row.update({f"5data の RMSE の比（÷ C）{P[fd]}": metric_of(mat, fd, key) / metric_of(mat, fd, "C") for fd in hold})
            if ALL_T in FOLD_NAMES[mat]:
                ba = BEST[(mat, ALL_T, key)]
                row["係数が全温度の区間に入った回"] = sum(ba["tv_lo"] <= BEST[(mat, fd, key)]["tv_best"] <= ba["tv_hi"] for fd in hold)
        else:
            d = KT[(KT["材料"] == MAT_NAMES[mat]) & (KT["model"] == "C") & KT["回"].isin(hold)]["C_noTset との LML の差"]
            row["尤度で支持された回"] = int((d >= LRT_DLML_1).sum())
        SUM.append(row)
        txt = "、".join(f"{fd.replace('除いた温度', '')} {v:.3f}" for fd, v in r5.items())
        if key in SHIFT_KEYS:
            judge(all(v < 1 for v in r5.values()), f"{key}：除いた温度の 5data の RMSE が、全 {len(hold)} 回で C_noTset より小さい（比 {txt}）→ 結論：{row['結論']}", mat, "まとめ")
        else:
            print(f"[{MAT_NAMES[mat]}] C（Tset を GP の入力に使う）の、C_noTset との比：{txt} → 結論：{row['結論']}")
ST = pd.DataFrame(SUM)
if len(ST):
    show = ["材料", "モデル", "回の数", "良い回", "同等の回", "悪い回", "結論", "5data の比の幾何平均", "尤度で支持された回"] + (["係数が全温度の区間に入った回"] if "係数が全温度の区間に入った回" in ST.columns else [])
    display(ST[show].round(3))
    display(ST[["材料", "モデル"] + [c for c in ST.columns if "の比（÷" in c]].round(3))
    for mat in MATERIALS:
        to_csv(ST[ST["材料"] == MAT_NAMES[mat]].dropna(axis=1, how="all"), os.path.join(out_dir(mat), "まとめ.csv"))''')

# ---------------------------------------------------------------- 11. 判定の一覧
md(r'''## 11. 判定の一覧

★ の行を確かめる。材料ごとに、結果のフォルダの `判定の一覧.csv` に保存する。''')
code(r'''# セル13：判定の一覧
JT = pd.DataFrame(JUDGES)
print(f"判定 {len(JT)} 件のうち、★要確認 {int((JT['判定'] != 'OK').sum())} 件")
display(JT[JT["判定"] != "OK"])
for mat in MATERIALS:
    to_csv(JT[JT["材料"].isin([MAT_NAMES[mat], ""])], os.path.join(out_dir(mat), "判定の一覧.csv"))
print("結果の置き場所：" + "、".join(show_path(out_dir(m)) for m in MATERIALS))''')


# ---------------------------------------------------------------- 材料ごとの複製（--material。2026-11-16 のユーザー依頼）
# 学習用のノートの複製（PP_・MIXPP_・GPPS_。ユーザーが 11-12 に作ったもの）と同じく、材料を 1 つにし、シフトの形を本番の設定（本番学習の設定の決定.pdf の
# 通常版：PP・粉砕材混合 PP は A_Tset_Arr、GPPS は B_Tset_Arr）だけにする（ユーザーの決定、11-16）。C と C_noTset は、どの版でも学習する
MAT_PRESETS = {"PP": {"prefix": "PP_", "name": "PP（バージン PP）", "struct": "A", "time": "約 10〜15 分"},
               "MIX": {"prefix": "MIXPP_", "name": "粉砕材混合 PP", "struct": "A", "time": "約 10〜15 分"},
               "GPPS": {"prefix": "GPPS_", "name": "GPPS", "struct": "B", "time": "約 20〜30 分"}}
args = sys.argv[1:]
MATERIAL = None
if "--material" in args:
    i = args.index("--material")
    assert i + 1 < len(args) and args[i + 1] in MAT_PRESETS, f"--material の後には {list(MAT_PRESETS)} のどれかを書く"
    MATERIAL = args[i + 1]
    del args[i:i + 2]
if MATERIAL:
    ps = MAT_PRESETS[MATERIAL]
    key = f"{ps['struct']}_Tset_Arr"
    sc = next(c for c in cells if c["cell_type"] == "code" and c["source"].startswith("# セル1：設定エリア"))
    for old, new in [('MATERIALS = ["PP", "MIX", "GPPS"]    # ', f'MATERIALS = ["{MATERIAL}"]    # （{ps["prefix"]}の版：この材料だけ。2026-11-16）'),
                     ('STRUCTS = ["A", "B"]          # ', f'STRUCTS = ["{ps["struct"]}"]               # （{ps["prefix"]}の版：本番の形〔{key}〕だけ。2026-11-16 のユーザーの決定）'),
                     ('MAIN_KEY = "B_Tset_Arr"       # ', f'MAIN_KEY = "{key}"       # ')]:
        assert sc["source"].count(old) == 1, f"設定エリアに、置き換える行が見つからない：{old}"
        sc["source"] = sc["source"].replace(old, new)
    head, rest = cells[0]["source"].split("\n\n", 1)
    cells[0]["source"] = (head + "\n\n" + f"> **この版は {ps['name']} だけ**（設定 `MATERIALS = [\"{MATERIAL}\"]`）。シフトのモデルは、本番の設定（`本番学習の設定の決定.pdf`）の **{key}** だけ"
                          f"（`STRUCTS = [\"{ps['struct']}\"]`）で、C と C_noTset も学習する。学習用のノートの `{ps['prefix']}温度シフトGPR_…ipynb` と同じ分け方（2026-11-16 のユーザー依頼）。"
                          f"3 材料をまとめて調べるとき、A と B の両方を比べるときは、同じフォルダの `{os.path.basename(TRAIN_NB).replace('温度シフトGPR', '温度ホールドアウト検証')}` を使う。"
                          f"所要時間の目安：{ps['time']}（要確認）。\n\n" + rest)


# ---------------------------------------------------------------- 書き出し
def to_lines(s):
    lines = s.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


rnd = random.Random(20261115)
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

outs = [a for a in args if a != "--force"]
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
