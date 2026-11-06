# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR の「Tff の入力の有無の比較」のノートブック (未実行) を生成する。

使い方: python build_mat_tff_input_nb.py <出力する .ipynb のパス> [<もう 1 つの出力パス> ...] [--force]
（成果物は 01_機械学習コードフォルダ/材料別_温度シフトファクター検討/材料別_温度シフトファクター検討用コード/
 材料別_Tffの入力の有無の比較_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb に置く〔シフトファクター検討用のノートと同じフォルダ〕。
 ノートは、置いた場所から上へたどってリポジトリ直下を探すので、どこに置いても動く）
既存ファイルがある場合は、--force を付けない限り上書きしない。
2026-11-06 のユーザーの問い（Tff がシフトの役目を果たさなくても、GP の入力に使ったほうがよいか）に答えるためのノート。
シフトファクター検討用のノートの結果（学習済みの C・指標の表）を読み、C から流動中樹脂温度 Tff を外したモデル C_noTff
（入力 ln γ̇・ln P3_3・Tset）を同じ 5data で学習して、対数周辺尤度と 3data の誤差を、検討のすべてのモデルと比べる。
C_noTff は、mat_common.py を変えず、このノートの中だけで足す（layout を包む）。
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
MAIN_NB = "材料別_温度シフトファクターの検討_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb"     # このノートが読む結果を作るノート（シフトファクター検討用。このノートと同じフォルダ）
MANUAL = "説明書_材料別温度シフトGPRのノート.pdf"     # 1 つ上のフォルダの、ノートの説明書（build_mat_nb_manual.py）
OUT_NAME = "Tffの入力の有無の比較"                    # 結果のフォルダの名前（<版>/全厚み総合/ の下）

# ---------------------------------------------------------------- 0. タイトル
md(rf'''# 材料別の温度シフト GPR：流動中樹脂温度 Tff を GP の入力に使うかの比較

シフトファクター検討用のノート（同じフォルダの `{MAIN_NB}`）の結果を読み、**流動中樹脂温度 Tff を GP の入力から外したモデル**を学習して、検討のすべてのモデルと比べる（2026-11-06 のユーザーの問い：「Tff はシフトの役目を果たさなくても、入力に使ったほうがよいか」）。学習は 5data、検証は 3data のまま変えない。

**今のモデルの入力**（材料・版によらず同じ。`mat_common.layout`）：

| モデル | GP の入力 | シフトに使う温度 |
|---|---|---|
| B_Tset_* | ln γ̇ ＋ ln a_h(Tset)、ln P3_3、**ln Tff** | Tset（横・縦） |
| A_Tset_* | ln γ̇、ln P3_3、**ln Tff** | Tset（縦だけ） |
| B_Tff_* | ln γ̇ ＋ ln a_h(Tff)、ln P3_3、Tset | Tff（横・縦） |
| A_Tff_* | ln γ̇、ln P3_3、Tset | Tff（縦だけ） |
| C | ln γ̇、ln P3_3、**ln Tff**、Tset | なし |
| **C_noTff**（このノートで学習） | ln γ̇、ln P3_3、Tset | なし |

どのモデルも、Tff を、GP の入力かシフトのどちらかに使っている。**C_noTff** は、C から ln Tff を外したもので、**Tff をまったく使わない**。カーネルは、ほかのモデルと同じ（ARD Matern 5/2 ＋ 線形項 ＋ ノイズ）。C_noTff は、A_Tff・B_Tff の係数を 0 にした点（θ = 0）と同じモデルなので、その対数周辺尤度は、検討のノートの `シフトファクターの推定.csv` の `LML_at_0` にも出ている（3 節で、一致することを確かめる）。ただし、θ = 0 のモデルは保存されていないので、3data の予測と図は、このノートで学習し直して作る。

**比べること**：
- 4 節：**対数周辺尤度（LML）**。C（Tff あり）と C_noTff（Tff なし）の差 = **Tff を入力に足した効果**。検討のほかのモデル（本番の候補を含む）とも並べる。
- 5 節：**3data の誤差**（ln η の RMSE・95% 予測区間に入る割合）。全体・低速側・高速側。
- 6 節：ハイパーパラメータ（長さスケールなど）。7 節：C_noTff の学習結果グラフ。8 節：まとめ（「Tff を入れたほうがよい／どちらでもよい／外したほうがよい」）。

**LML を比べてよい範囲**：同じ材料・同じ版（通常版／温度適切版）・同じ入力の取り方のモデルどうしだけ（目的変数の ln η の行が同じ。シフトは ln η から ln a_v を引く平行移動なので、尤度の大きさは変わらない。中心化は固定の定数）。**材料や版が違うと、LML の大きさはデータの行数とばらつきで決まるので、比べない**（比べるなら、同じ物差しの 3data の誤差か、それぞれの中での差を並べる）。パラメータの数が多いモデルほど LML が大きく出やすいので、差には基準（パラメータ 1 つで 1.92、2 つで 3.0。尤度比検定の 5% の目安。モデルが入れ子でないときは目安）を当てる。

**読むときの注意**：Tff は流動の結果（せん断発熱で射出速度とともに変わる）で、予測のときには事前に分かっている必要がある。このプロジェクトの説明変数（γ34・P3_3・Tff・Tset）はユーザーの指定なので、本番の入力から Tff を外すかは、このノートの結果を見て、ユーザーが決める（外す場合は、学習用のノートに設定を足す別の作業になる）。''')

# ---------------------------------------------------------------- 1. 使い方・設定
md(f'''## 使い方

**初めての人へ**：1 つ上のフォルダ（`材料別_温度シフトファクター検討/`）の説明書 `{MANUAL}` に、ノートの目的・関係・使えるデータ・使い方をまとめてある。

1. **先にシフトファクター検討用のノート `{MAIN_NB}` を実行しておく**（このノートは、その結果のフォルダ `<学習結果フォルダ>/<版>/全厚み総合/シフトファクターの検討/` の、学習済みの C〔`C_温度シフトなし/学習済みモデル_C.joblib`〕と予測値、表〔`ハイパーパラメータ.csv`・`シフトファクターの推定.csv`・`予測検証_3data_指標.csv`〕を読む）。`INPUT_MODE` は、そのノートと同じにする。
2. メニューの「すべて実行」で、上から順に実行する。重いのは 3 節の学習だけ（材料ごとに GP を 1 回。プロファイル尤度はない）。目安：3 材料で数分〜十数分（要確認）。カーネルは、このプロジェクトの `.venv` を選ぶ。
3. **実行の始めに（「2. 読み込み」のセルで）、材料ごとにダイアログが 1 回開く**：シフトファクター検討用のノートで選んだ**学習結果フォルダ**を選ぶ。学習データ・検証データの場所は、学習済みの C に記録されたものを使う。**通常版と温度適切版は、学習結果フォルダが別なので、版ごとに 1 回ずつ実行する**（2 回目は、温度適切版の学習結果フォルダを選ぶ）。設定 `FILES` に書けば、ダイアログは出ない。ダイアログが見当たらないときは、VS Code の後ろに隠れているので、タスクバーで探す。
4. 各節の最後に、**判定**が出る（`>> 判定：OK` か `>> 判定：★要確認`）。OK は「想定どおり」、★ は「想定と違う、または結果を読むときに注意が要る点」（このノートでは、「Tff を外しても変わらない・良くなる」ことも ★ で知らせる）。★ が出ても、エラーではない。最後の「判定の一覧」に一覧が出る。
5. 図と表は、学習結果フォルダの **`<版>/全厚み総合/{OUT_NAME}/`** に保存される（検討の結果のフォルダ `シフトファクターの検討/` は変えない）。C_noTff の学習済みモデルも、ここに保存する。図は 600 dpi の 1 枚だけ（設定 `FIG_DPI`）。''')
md(r'''## 1. 設定''')
code(r'''# セル1：設定エリア  ▼▼▼ 変えてよいのは、ここだけ ▼▼▼
from pathlib import Path      # （下の REPO で、フォルダをたどるのに使う）

SEED = 42                     # 乱数シード（シフトファクター検討用のノートと同じ）

NOISE_LB = 1e-6               # ノイズ分散の下限（シフトファクター検討用のノートと同じにする）

USE_LINEAR = True             # 線形項（DotProduct）を足すか（シフトファクター検討用のノートと同じにする）

INPUT_MODE = "shot"           # 読む結果の入力の取り方（シフトファクター検討用のノートと同じにする。"shot"：ショットごと〔既定。全ショット版/〕、"cond"：条件平均〔条件ごとの平均版/〕）

N_RESTARTS = 20               # C_noTff の学習での、最適化のやり直し回数（シフトファクター検討用のノートの最終モデルと同じ）

USE_EXCLUSIONS = True         # 問題データ一覧（05 フォルダ）の「除外」を除く（シフトファクター検討用のノートと同じ）

REPO = str(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "04_補助スクリプト").is_dir()))   # リポジトリ直下（このノートの場所から上へたどって、04_補助スクリプト があるフォルダを探す）

MATERIALS = ["PP", "MIX", "GPPS"]    # 調べる材料（PP：バージン PP、MIX：粉砕材混合 PP、GPPS）。材料ごとに、実行の始めにダイアログで、学習結果フォルダ（シフトファクター検討用のノートで選んだもの）を選ぶ

MAT_FOLDERS = {"PP": "01_バージンPP再々実験", "MIX": "02_粉砕材混合PP実験", "GPPS": "03_GPPS実験"}   # 05 フォルダの中の材料フォルダ名（ダイアログの最初のフォルダ）

FILES = {}                    # ダイアログを使わずに決めるとき：{"PP": {"結果": r"…"}}（相対パスはリポジトリ直下から。"学習"・"検証" も書くと、学習済みの C の記録より、そちらを使う）
SAVE_ROOT = None              # 結果の置き場所：None＝学習結果フォルダの <版>/全厚み総合/Tffの入力の有無の比較/（既定）／フォルダを書く＝その下の <材料>/<版>/全厚み総合/Tffの入力の有無の比較/（確かめの実行用）
N_JOBS = 8                    # 材料ごとの学習を並列に計算する数（CPU の数に合わせる）
FIG_DPI = 600                 # 保存する図の解像度（すべての図を、この解像度の 1 枚だけで保存する）
SHOW_FIGS = True              # True：図をノートにも表示する（False：保存だけ）
# ---- ここから下は、このノートだけの設定 ----
LML_CLEAR = 1.92              # 4・8 節：パラメータ 1 つの差の基準（尤度比検定の 5% の目安。C と C_noTff の差は、長さスケール 1 つ分）。2 つ以上の差は、カイ二乗分布の 95% 点 ÷ 2 を使う
RMSE_RATIO = 1.1              # 5・8 節：3data の RMSE の比が、この倍率の内側（1/1.1〜1.1 倍）なら「同等」（シフトファクター検討用のノートの「C の 1.1 倍以内」と同じ考え方）
LML_CHECK = 1.0               # 3 節：C_noTff の LML が、検討のノートの θ = 0 の値（A_Tff・B_Tff の LML_at_0）より、これ以上低ければ ★（最適化の局所解の目安）
# ▲▲▲ 設定エリア終了 ▲▲▲''')
code("# セル2：ライブラリ、フォント\n" + nbc.LIBS + '\nprint("材料:", MATERIALS, "／入力の取り方:", INPUT_MODE)')
md(r'''### 共通の部品

シフトファクター検討用のノートと同じ部品（`04_補助スクリプト/mat_common.py`）。''')
code(COMMON_CELL)
md(r'''### Tff を入力から外した C（C_noTff）を足す

共通の部品は変えず、このノートの中だけで、モデルの記号 `C_noTff` を足す（GP の入力の列を決める `layout` を包み、`C_noTff` のときだけ、C から `ln_Tff` を外した列を返す）。学習・予測・ハイパーパラメータの表・図は、共通の部品の関数がそのまま使える。''')
code(r'''# セル2a：C_noTff（C から流動中樹脂温度 Tff を外したモデル）を足す（このノートの中だけ。mat_common.py は変えない）
NO_TFF = "C_noTff"
_layout_common = layout


def layout(key):
    """GP の入力の列と、シフトに使う温度の列。C_noTff だけ、C の入力から ln Tff を外したもの（ln γ̇, ln P3_3, Tset。シフトなし）。ほかのモデルは共通の部品のまま"""
    if key == NO_TFF:
        return ["ln_gd", "ln_P33", "Tset"], None
    return _layout_common(key)


MODEL_JA[NO_TFF] = "C_noTff（シフト構造なし。C から流動中樹脂温度 Tff を外し、3 変数を並べた通常の GPR）"
MODEL_DIRS[NO_TFF] = "C_noTff_温度シフトなし_型内樹脂温度を入力に使わない"
print("C の入力：" + "、".join(input_labels("C")))
print("C_noTff の入力：" + "、".join(input_labels(NO_TFF)))''')

# ---------------------------------------------------------------- 2. 読み込み
md(rf'''## 2. 検討の結果とデータの読み込み

**このセルで、材料ごとにダイアログが 1 回開く**（シフトファクター検討用のノートで選んだ学習結果フォルダを選ぶ。設定 `FILES` に書いたものは開かない）。検討の結果（学習済みの C、3data の予測値、表）を読み、学習済みの C に記録された学習データ・検証データを、シフトファクター検討用のノートと同じ手順で読む（問題データ一覧の「除外」を除く）。データの版（通常版／温度適切版）は、学習データのフォルダの名前（「温度適切」を含むか）で表示する。

**判定**：学習済みの C の入力の取り方が `INPUT_MODE` と同じ／読んだ C で 3data を予測し直すと、保存済みの予測値と一致する（データとモデルの組み合わせが正しい）／C の LML が `ハイパーパラメータ.csv` の値と一致する。''')
code(rf'''# セル3：読み込み
import joblib
JUDGES.clear()
display(choose_inputs(MATERIALS, FILES, need_data=False))      # 材料ごとに、学習結果フォルダを決める（設定 FILES に書いていなければ、ダイアログで選ぶ）
OUT_NAME = "{OUT_NAME}"
STUDY_CSVS = {{"hyper": "ハイパーパラメータ.csv", "est": "シフトファクターの推定.csv", "metrics": "予測検証_3data_指標.csv"}}   # 検討のノートの表（読むだけ）


def study_dir(mat):
    """シフトファクター検討用のノートの結果のフォルダ：<学習結果フォルダ>/<版>/全厚み総合/シフトファクターの検討/（読むだけなので、フォルダは作らない）"""
    return result_dir(mat, VERSION_DIRS[INPUT_MODE], THICK_TOTAL, STUDY_DIR, make=False)


def out_dir(mat, sub=None):
    """このノートの結果の置き場所：<学習結果フォルダ>/<版>/全厚み総合/Tffの入力の有無の比較/（SAVE_ROOT を書いたときは、その下の <材料>/<版>/全厚み総合/…）"""
    base = os.path.join(full_path(SAVE_ROOT), mat, VERSION_DIRS[INPUT_MODE], THICK_TOTAL) if SAVE_ROOT else result_dir(mat, VERSION_DIRS[INPUT_MODE], THICK_TOTAL, make=False)
    d = os.path.join(base, OUT_NAME, *([sub] if sub else []))
    os.makedirs(d, exist_ok=True)
    return d


OBJ_C, FIT_C, PRED_C, TAB, DATA_VER = {{}}, {{}}, {{}}, {{}}, {{}}
for mat in MATERIALS:
    for name, fn in STUDY_CSVS.items():
        p = os.path.join(study_dir(mat), fn)
        if not os.path.exists(p):
            raise FileNotFoundError(f"{{p}} がありません。先にシフトファクター検討用のノートを、INPUT_MODE = '{{INPUT_MODE}}' で実行し、同じ学習結果フォルダを選んでください")
        TAB[(mat, name)] = pd.read_csv(p, encoding="utf-8-sig")
    path = os.path.join(study_dir(mat), MODEL_DIRS["C"], "学習済みモデル_C.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{{path}} がありません（シフトファクター検討用のノートで、基準の C を学習していない）")
    OBJ_C[mat] = joblib.load(path)
    if OBJ_C[mat]["材料"] != MAT_NAMES[mat]:
        raise ValueError(f"{{path}} は {{OBJ_C[mat]['材料']}} のモデルです（{{MAT_NAMES[mat]}} の学習結果フォルダを選んでください）")
    FIT_C[mat] = OBJ_C[mat]["fit"]
    PRED_C[mat] = pd.read_csv(os.path.join(study_dir(mat), MODEL_DIRS["C"], "予測検証_3data_予測値.csv"), encoding="utf-8-sig")
    data_from_model(mat, OBJ_C[mat])         # 学習データ・検証データの場所は、学習済みの C の記録から取る（設定 FILES に書いてあれば、そちら）

D = load_all(MATERIALS)
TRAIN, TEST, POOL = {{}}, {{}}, {{}}
for mat in MATERIALS:
    TRAIN[mat] = D[(mat, "5data")]
    TEST[mat] = add_groups(TRAIN[mat], D[(mat, "3data")])
    POOL[mat] = pd.concat([TRAIN[mat], TEST[mat]], ignore_index=True)
    DATA_VER[mat] = "温度適切版" if "温度適切" in os.path.basename(os.path.dirname(SOURCES[mat]["5data"])) else "通常版"
    print(f"【{{MAT_NAMES[mat]}}・{{DATA_VER[mat]}}】学習データ：{{show_path(SOURCES[mat]['5data'])}}（{{len(TRAIN[mat])}} 行・{{TRAIN[mat]['cond'].nunique()}} 条件）"
          f"／検証データ：{{show_path(SOURCES[mat]['3data'])}}（{{len(TEST[mat])}} 行）")
    judge(FIT_C[mat].get("mode") == INPUT_MODE, f"学習済みの C の入力の取り方が {{INPUT_MODE}}", mat, "2 読み込み")
    mu, _, _, _ = predict(FIT_C[mat], TEST[mat])
    chk = TEST[mat][["file"]].assign(mu=mu).merge(PRED_C[mat][["file", "予測_ln_eta"]], on="file")
    judge(len(chk) == len(TEST[mat]) and np.allclose(chk["mu"], chk["予測_ln_eta"], atol=1e-6),
          f"C で 3data を予測し直した値が、保存済みの予測値と一致（{{len(chk)}}/{{len(TEST[mat])}} 行）", mat, "2 読み込み")
    h = TAB[(mat, "hyper")].set_index("model")
    lml_c = float(FIT_C[mat]["gp"].log_marginal_likelihood_value_)
    judge("C" in h.index and abs(lml_c - float(h.loc["C", "LML"])) < 1e-6, f"学習済みの C の LML（{{lml_c:.2f}}）が、ハイパーパラメータ.csv の値と一致", mat, "2 読み込み")
    print(f"  検討のモデル：{{'、'.join(h.index)}}")''')

# ---------------------------------------------------------------- 3. 学習
md(r'''## 3. C_noTff の学習

材料ごとに、C と同じ 5data・同じカーネル・同じ設定（`N_RESTARTS` 回のやり直し、乱数シード）で、入力から ln Tff を外したモデル C_noTff を学習する（シフトなし）。学習済みモデルは、結果のフォルダに保存する。

**判定**：C_noTff の LML が、検討のノートの θ = 0 の LML（A_Tff・B_Tff の `LML_at_0`。同じモデルを、プロファイル尤度の点の解から念入りに最適化した値）より `LML_CHECK` 以上低くない（低ければ、最適化が局所解に止まったおそれ。`N_RESTARTS` を増やして実行し直す）。''')
code(r'''# セル4：C_noTff の学習
from joblib import Parallel, delayed
import sklearn
t0 = time.time()
res = Parallel(n_jobs=min(N_JOBS, len(MATERIALS)))(
    delayed(fit_job)(NO_TFF, TRAIN[mat], FIT_C[mat]["cfg"], 0.0, 0.0, N_RESTARTS, None, INPUT_MODE) for mat in MATERIALS)
FIT_N = dict(zip(MATERIALS, res))
print(f"学習：{len(MATERIALS)} 材料、{(time.time() - t0) / 60:.1f} 分")
LML_REF = {}
for mat in MATERIALS:
    f = FIT_N[mat]
    lml = float(f["gp"].log_marginal_likelihood_value_)
    path = os.path.join(out_dir(mat), f"学習済みモデル_{NO_TFF}.joblib")
    joblib.dump({"fit": f, "材料": MAT_NAMES[mat], "モデル": NO_TFF, "説明": model_title(NO_TFF), "入力": input_labels(NO_TFF), "入力の取り方": INPUT_MODE,
                 "学習データのファイル": show_path(SOURCES[mat]["5data"]), "検証データのファイル": show_path(SOURCES[mat]["3data"]),
                 "scikit-learn": sklearn.__version__, "作成日": time.strftime("%Y-%m-%d")}, path)
    est = TAB[(mat, "est")]
    ref = est[est["モデル"].astype(str).str.contains("_Tff")]["LML_at_0"]
    LML_REF[mat] = float(ref.max()) if len(ref) else np.nan
    print(f"【{MAT_NAMES[mat]}・{DATA_VER[mat]}】C_noTff の LML {lml:.2f}（検討の θ = 0 の LML {LML_REF[mat]:.2f}）。保存：{show_path(path)}")
    if np.isfinite(LML_REF[mat]):
        judge(lml >= LML_REF[mat] - LML_CHECK, f"C_noTff の LML {lml:.2f} が、検討の θ = 0 の LML {LML_REF[mat]:.2f}（A_Tff・B_Tff の LML_at_0。同じモデル）より {LML_CHECK:g} 以上低くない", mat, "3 学習")
    else:
        print("  （検討のノートで Tff のシフトのモデルを学習していないので、θ = 0 の LML と照らし合わせられない）")''')

# ---------------------------------------------------------------- 4. 尤度の比較
md(r'''## 4. 対数周辺尤度（LML）の比較（同じ材料・同じ版の中だけ）

検討のすべてのモデル（`ハイパーパラメータ.csv` の LML。シフトのあるモデルは、最良の係数での念入りな最適化の値）と C_noTff を、1 つの表に並べる。

- **パラメータの数**：カーネルのハイパーパラメータ（Matern の振幅 1・長さスケール〔入力の数〕・線形項 2・ノイズ 1）＋ シフトの係数（A は 1、B は 2）。C は 8、C_noTff は 7、A は 8、B は 9。
- **C_noTff との差** = LML(モデル) − LML(C_noTff)。**基準** = パラメータの数の差 k に対する、カイ二乗分布の 95% 点 ÷ 2（k = 1 で 1.92、k = 2 で 3.0。モデルが入れ子でないときは目安）。差が基準以上なら、「そのモデルは、Tff を使わない C_noTff より、尤度で支持される」。
- **本番の候補**：検討のノートと同じ規則（`shift_candidates`。候補がなければ C）で、`シフトファクターの推定.csv` から決め直したもの。

**判定**（★ は「Tff が尤度に効いていない」ことの知らせ）：C（Tff あり）が C_noTff より基準以上大きい／本番の候補が C_noTff より基準以上大きい。''')
code(r'''# セル5：対数周辺尤度の比較
from scipy.stats import chi2


def n_params(key):
    """モデルのパラメータの数：カーネル（振幅 1 ＋ 長さスケール〔入力の数〕＋ 線形項 2〔使うとき〕＋ ノイズ 1）＋ シフトの係数（A 1・B 2・C 0）"""
    d = len(layout(key)[0])
    s = {"A": 1, "B": 2}.get(str(key).split("_")[0], 0)
    return 1 + d + (2 if USE_LINEAR else 0) + 1 + s


def lml_crit(k):
    """パラメータの数の差 k に対する、LML の差の基準（カイ二乗分布の 95% 点 ÷ 2。k = 1 で 1.92）"""
    return float(chi2.ppf(0.95, k) / 2) if k > 0 else 0.0


LML_TAB, CAND = {}, {}
for mat in MATERIALS:
    h = TAB[(mat, "hyper")]
    est = TAB[(mat, "est")]
    cand, _, notes = shift_candidates([{"モデル": r["モデル"], "LML": r["LML_best"], "dLML_vs_0": r["dLML_vs_0"]} for _, r in est.iterrows()])
    CAND[mat] = cand or "C"
    lml_n = float(FIT_N[mat]["gp"].log_marginal_likelihood_value_)
    rows = [{"モデル": r["model"], "LML": float(r["LML"])} for _, r in h.iterrows()] + [{"モデル": NO_TFF, "LML": lml_n}]
    t = pd.DataFrame(rows)
    t["GP の入力"] = ["、".join(input_labels(k)) for k in t["モデル"]]
    t["シフトに使う温度"] = [layout(k)[1] or "なし" for k in t["モデル"]]
    t["Tff を使うか"] = ["使わない" if k == NO_TFF else ("シフトに使う" if layout(k)[1] == "Tff" else "GP の入力に使う") for k in t["モデル"]]
    t["パラメータの数"] = [n_params(k) for k in t["モデル"]]
    t["C_noTff との差"] = t["LML"] - lml_n
    t["基準（目安）"] = [lml_crit(n - n_params(NO_TFF)) for n in t["パラメータの数"]]
    t["C_noTff より支持される"] = [None if k == NO_TFF else bool(d >= c) for k, d, c in zip(t["モデル"], t["C_noTff との差"], t["基準（目安）"])]
    t["本番の候補"] = t["モデル"] == CAND[mat]
    t.insert(0, "版", DATA_VER[mat])
    t.insert(0, "材料", MAT_NAMES[mat])
    LML_TAB[mat] = t
    to_csv(t, os.path.join(out_dir(mat), "対数周辺尤度の比較.csv"))
    print(f"\n【{MAT_NAMES[mat]}・{DATA_VER[mat]}】本番の候補（検討の規則）：{CAND[mat]}" + ("（注意：" + "／".join(notes) + "）" if notes else ""))
    display(t.drop(columns=["材料", "版"]).round(2))
    rc = t.set_index("モデル").loc["C"]
    judge(bool(rc["C_noTff より支持される"]), f"C（Tff を GP の入力に使う）が C_noTff（Tff を使わない）より尤度で支持される（差 {rc['C_noTff との差']:+.2f}、基準 {rc['基準（目安）']:.2f}）。"
          "★ なら、Tff を入力に足しても尤度は上がらない", mat, "4 尤度")
    if CAND[mat] != "C":
        rk = t.set_index("モデル").loc[CAND[mat]]
        judge(bool(rk["C_noTff より支持される"]), f"本番の候補 {CAND[mat]} が C_noTff より尤度で支持される（差 {rk['C_noTff との差']:+.2f}、基準 {rk['基準（目安）']:.2f}）。"
              "★ なら、Tff を使わない C_noTff と尤度で差がない", mat, "4 尤度")''')

# ---------------------------------------------------------------- 5. 3data の誤差
md(r'''## 5. 3data の予測の比較

C_noTff で 3data を予測し、検討のノートと同じ指標（`metrics_rows`：ln η の RMSE・R²・95% 予測区間に入る割合〔被覆率〕など。全体・低速側・高速側）を出して、検討のすべてのモデル（`予測検証_3data_指標.csv`）と並べる。RMSE は、材料や版をまたいでも同じ物差し（ln η。0.1 で粘度の約 10%）。

図：①モデルごとの RMSE の棒グラフ（全体・低速側・高速側。C は灰色、C_noTff は橙、本番の候補は太枠）、②C と C_noTff のパリティ図（実測と予測の粘度。縦の線は 95% 予測区間）。

**判定**：RMSE（全体）の比 C_noTff / C が 1/`RMSE_RATIO`〜`RMSE_RATIO` の中（★ なら、Tff を外すと誤差がはっきり変わる。比が大きければ Tff は誤差を減らしている、小さければ Tff を外したほうが誤差が小さい）／C_noTff の被覆率（全体）が 0.8 以上。''')
code(r'''# セル6：3data の予測の比較
PRED_N, MT_ALL, RATIO = {}, {}, {}
RANGES = ["全体", "低速側", "高速側"]
GCOL = {"低速側": ("#1f77b4", "o"), "高速側": ("#d62728", "s"), "範囲内": ("#2ca02c", "^")}
for mat in MATERIALS:
    te = TEST[mat]
    mu, sd, sdf, nsd = predict(FIT_N[mat], te)
    PRED_N[mat] = te[["file", "cond", "Tset", "H", "v", "group", "eta", "gd", "P33", "Tff", "ln_eta"]].assign(
        予測_ln_eta=mu, 予測標準偏差=sd, 潜在関数の標準偏差=sdf, 予測η=np.exp(mu), 下限95=np.exp(mu - 1.96 * sd), 上限95=np.exp(mu + 1.96 * sd))
    to_csv(PRED_N[mat], os.path.join(out_dir(mat), f"予測検証_3data_予測値_{NO_TFF}.csv"))
    mt = pd.concat([TAB[(mat, "metrics")], pd.DataFrame(metrics_rows(mat, NO_TFF, te, mu, sd, sdf, {"ノイズの標準偏差": nsd}))], ignore_index=True)
    mt.insert(1, "版", DATA_VER[mat])
    MT_ALL[mat] = mt
    to_csv(mt, os.path.join(out_dir(mat), "予測検証_3data_指標_全モデル.csv"))
    rng = [r for r in RANGES if r in set(mt["範囲"])]
    piv = mt.pivot_table(index="モデル", columns="範囲", values="RMSE", sort=False)[rng]
    cov = mt.pivot_table(index="モデル", columns="範囲", values="coverage95", sort=False)[rng]
    RATIO[mat] = (piv.loc[NO_TFF] / piv.loc["C"]).to_dict()
    print(f"\n【{MAT_NAMES[mat]}・{DATA_VER[mat]}】3data の ln η の RMSE（本番の候補：{CAND[mat]}）")
    display(piv.round(4))
    print("95% 予測区間に入る割合（被覆率）")
    display(cov.round(3))
    print("RMSE の比 C_noTff / C：" + "、".join(f"{r} {RATIO[mat][r]:.3f}" for r in rng))
    # ① RMSE の棒グラフ
    fig, axes = plt.subplots(1, len(rng), figsize=(6.5 * len(rng), 6), squeeze=False)
    for ax, r in zip(axes[0], rng):
        vals = piv[r]
        cols = ["#ff7f0e" if k == NO_TFF else ("#7f7f7f" if k == "C" else "#1f77b4") for k in vals.index]
        bars = ax.bar(range(len(vals)), vals.to_numpy(), color=cols)
        for b, k in zip(bars, vals.index):
            if k == CAND[mat]:
                b.set_edgecolor("k")
                b.set_linewidth(2.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=60, ha="right")
        ax.set_ylabel("3data の RMSE（ln η）")
        ax.set_title(f"{r}")
        ax.grid(True, axis="y", ls="--", alpha=0.4)
    fig.suptitle(f"{MAT_NAMES[mat]}・{DATA_VER[mat]}　3data の RMSE（灰：C、橙：C_noTff、太枠：本番の候補 {CAND[mat]}）")
    fig.tight_layout(rect=(0, 0, 1, 0.94))              # 上に、全体のタイトルの場所を空ける
    save_fig(fig, out_dir(mat), "予測検証_RMSEの比較_全モデル")
    # ② パリティ図（C と C_noTff）
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    for ax, (k, p) in zip(axes, [("C", PRED_C[mat]), (NO_TFF, PRED_N[mat])]):
        rr = mt[(mt["モデル"] == k) & (mt["範囲"] == "全体")].iloc[0]
        for g, (col, mk) in GCOL.items():
            m = (p["group"] == g).to_numpy()
            if m.any():
                ax.errorbar(p.loc[m, "eta"], p.loc[m, "予測η"], yerr=[p.loc[m, "予測η"] - p.loc[m, "下限95"], p.loc[m, "上限95"] - p.loc[m, "予測η"]],
                            fmt=mk, color=col, ms=6, elinewidth=1.2, capsize=3, alpha=0.8, label=f"{g}（{int(m.sum())} ショット）")
        lim = [min(p["eta"].min(), p["下限95"].min()) / 1.3, max(p["eta"].max(), p["上限95"].max()) * 1.3]
        ax.plot(lim, lim, "k--", lw=1, label="予測 = 実測")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        log_ticks(ax)
        ax.set_xlabel(r"実測の粘度 $\eta$ (Pa·s)")
        ax.set_ylabel(r"予測の粘度 $\eta$ (Pa·s)")
        ax.set_title(f"{k}（Tff を{'使わない' if k == NO_TFF else '入力に使う'}）\nRMSE(ln η) {rr['RMSE']:.3f}、R²(実数) {rr['R2_real']:.4f}、被覆率 {rr['coverage95']:.2f}")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(loc="upper left")
    fig.suptitle(f"{MAT_NAMES[mat]}・{DATA_VER[mat]}　3data のパリティ図：C と C_noTff", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))              # 上に、全体のタイトルの場所を空ける（各図のタイトルが 2 行なので広めに）
    save_fig(fig, out_dir(mat), "予測検証_3data_パリティ_CとC_noTff")
    q = RATIO[mat]["全体"]
    judge(1 / RMSE_RATIO <= q <= RMSE_RATIO, f"3data の RMSE（全体）の比 C_noTff / C = {q:.3f} が {1 / RMSE_RATIO:.3f}〜{RMSE_RATIO:g} の中（Tff を外しても、誤差は同等）。"
          "★ なら、比が大きいと Tff が誤差を減らしている、小さいと Tff を外したほうが誤差が小さい", mat, "5 検証")
    cv = float(cov.loc[NO_TFF, "全体"])
    judge(cv >= 0.8, f"C_noTff の 95% 予測区間に入る割合（全体）が 0.8 以上（{cv:.3f}。C は {float(cov.loc['C', '全体']):.3f}）", mat, "5 検証")''')

# ---------------------------------------------------------------- 6. ハイパーパラメータ
md(r'''## 6. ハイパーパラメータ（C と C_noTff）

C（検討の `ハイパーパラメータ.csv`）と C_noTff の、振幅・長さスケール・線形項・ノイズ・LML を並べる。入力は学習データで標準化してあるので、長さスケールはデータの幅（約 ±2）と比べて読む（長いほど、その入力の方向に、Matern の部分がゆっくりしか変わらない。線形項は入力ごとの長さスケールを持たない〔全入力に共通〕ので、長さスケールが長くても、その入力が線形に効いている場合がある）。上限 100 に張り付いたものは、表の `ls_at_bound` が True。''')
code(r'''# セル7：ハイパーパラメータ
HP = {}
for mat in MATERIALS:
    rn = kernel_row(FIT_N[mat])
    th_n, tv_n, _ = theta_names(FIT_N[mat]["cfg"])
    rn[th_n], rn[tv_n] = np.nan, np.nan                     # C_noTff にシフトの係数はない
    hc = TAB[(mat, "hyper")]
    rc = hc[hc["model"] == "C"].iloc[0].to_dict()
    t = pd.DataFrame([rc, {"材料": MAT_NAMES[mat], **rn}]).dropna(axis=1, how="all")
    t.insert(1, "版", DATA_VER[mat])
    HP[mat] = t
    to_csv(t, os.path.join(out_dir(mat), "ハイパーパラメータ_CとC_noTff.csv"))
    print(f"\n【{MAT_NAMES[mat]}・{DATA_VER[mat]}】")
    display(t.set_index("model").drop(columns=["材料", "版"]).T)''')

# ---------------------------------------------------------------- 7. 学習結果グラフ
md(r'''## 7. C_noTff の学習結果グラフ（せん断速度 ― 粘度。厚みごと）

学習用・検討用のノートと同じ関数（`flow_plot`）で、厚みごとに 1 枚、設定温度ごとの予測線と 95% 予測区間を描く（塗りつぶしの点は学習〔5data〕、白抜きは検証〔3data〕。点はショットごとの計測値）。曲線は、P3_3 を条件ごとの値で動かした断面（Tff は入力にないので使わない）。検討のノートの C の図は、そのノートの結果のフォルダにある。''')
code(r'''# セル8：C_noTff の学習結果グラフ
n_saved = 0
for mat in MATERIALS:
    saved = flow_plot(FIT_N[mat], mat, POOL[mat], set(TRAIN[mat]["cond"]), out_dir(mat, "せん断速度_粘度"), f"せん断速度_粘度_{NO_TFF}",
                      layouts=["thickness"], points="shot", show=SHOW_FIGS)
    n_saved += len(saved)
print(f"保存した図：{n_saved} 枚（せん断速度_粘度/）")''')

# ---------------------------------------------------------------- 8. まとめ
md(r'''## 8. まとめ：Tff を GP の入力に使ったほうがよいか

材料ごとに、次の 2 つから結論を出す（同じ材料・同じ版の中だけで比べる）。
- **尤度**：LML(C) − LML(C_noTff) が `LML_CLEAR`（1.92）以上なら「Tff ありが支持される」、そうでなければ「支持されない」。
- **誤差**：3data の RMSE（全体）の比 C_noTff / C が `RMSE_RATIO`（1.1）より大きければ「Tff ありが良い」、1/1.1 より小さければ「Tff なしが良い」、その間なら「同等」。

| 尤度 | 誤差 | 結論 |
|---|---|---|
| 支持される | 同等・Tff ありが良い | 入れたほうがよい |
| 支持されない | Tff ありが良い | 入れたほうがよい（誤差で） |
| 支持されない | 同等 | どちらでもよい（外しても精度は同じ） |
| 支持されない | Tff なしが良い | 外したほうがよい |
| 支持される | Tff なしが良い | 決まらない（★） |

参考に、本番の候補（Tset でシフトするモデルは、Tff を GP の入力に使う）と C_noTff の差も並べる。結果は `Tffの入力の有無_まとめ.csv`。''')
code(r'''# セル9：まとめ
SUM = []
for mat in MATERIALS:
    t = LML_TAB[mat].set_index("モデル")
    d_c = float(t.loc["C", "C_noTff との差"])
    q = RATIO[mat]
    lik = "支持される" if d_c >= LML_CLEAR else "支持されない"
    err = "Tff ありが良い" if q["全体"] > RMSE_RATIO else ("Tff なしが良い" if q["全体"] < 1 / RMSE_RATIO else "同等")
    concl = {("支持される", "同等"): "入れたほうがよい", ("支持される", "Tff ありが良い"): "入れたほうがよい",
             ("支持されない", "Tff ありが良い"): "入れたほうがよい（誤差で）", ("支持されない", "同等"): "どちらでもよい（外しても精度は同じ）",
             ("支持されない", "Tff なしが良い"): "外したほうがよい", ("支持される", "Tff なしが良い"): "決まらない（★）"}[(lik, err)]
    piv = MT_ALL[mat].pivot_table(index="モデル", columns="範囲", values="RMSE", sort=False)
    ck = CAND[mat]
    SUM.append({"材料": MAT_NAMES[mat], "版": DATA_VER[mat], "LML C": float(t.loc["C", "LML"]), "LML C_noTff": float(t.loc[NO_TFF, "LML"]),
                "LML の差 C − C_noTff": d_c, "尤度": lik,
                **{f"RMSE C {r}": float(piv.loc["C", r]) for r in RANGES if r in piv.columns},
                **{f"RMSE C_noTff {r}": float(piv.loc[NO_TFF, r]) for r in RANGES if r in piv.columns},
                "RMSE の比 C_noTff / C（全体）": q["全体"], "誤差": err, "結論": concl,
                "本番の候補": ck, "LML の差 本番の候補 − C_noTff": float(t.loc[ck, "C_noTff との差"]), "その基準（目安）": float(t.loc[ck, "基準（目安）"]),
                "RMSE の比 C_noTff / 本番の候補（全体）": float(piv.loc[NO_TFF, "全体"] / piv.loc[ck, "全体"])})
    judge(concl != "決まらない（★）", f"結論：{concl}（尤度：{lik}〔差 {d_c:+.2f}〕、誤差：{err}〔比 {q['全体']:.3f}〕）", mat, "8 まとめ")
ST = pd.DataFrame(SUM)
display(ST.round(4))
for mat in MATERIALS:
    to_csv(ST[ST["材料"] == MAT_NAMES[mat]], os.path.join(out_dir(mat), "Tffの入力の有無_まとめ.csv"))''')

# ---------------------------------------------------------------- 9. 判定の一覧
md(r'''## 9. 判定の一覧

★ の行を確かめる。材料ごとに、結果のフォルダの `判定の一覧.csv` に保存する。''')
code(r'''# セル10：判定の一覧
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


rnd = random.Random(20261106)
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
