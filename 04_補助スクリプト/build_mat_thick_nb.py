# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR の「厚みごとの比較」のノートブック (未実行) を生成する。

使い方: python build_mat_thick_nb.py <出力する .ipynb のパス> [<もう 1 つの出力パス> ...] [--force]
既存ファイルがある場合は、--force を付けない限り上書きしない。
元のノート（材料別_温度シフトGPR_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb）が保存した学習済みモデル（B_Tset・B_Tff・C）と予測値を読み、
厚みごとに、温度シフトファクターが効いているか、シフトに使う温度は流動中樹脂温度 Tff と加熱シリンダ温度 Tset のどちらが適しているか、
厚みごとに学習したモデルと全厚みの総合データのモデルのどちらがよいかを調べる（ユーザー依頼、2026-10-22。学習 5data・検証 3data のまま）。
"""
import json
import os
import random
import sys

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
MAIN_NB = "材料別_温度シフトGPR_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb"
PDF = "05_材料ごとの機械学習データと予測検証データ/学習結果説明_厚みごとの温度シフト.pdf"

# ---------------------------------------------------------------- 0. タイトル
md(rf'''# 材料別の温度シフト GPR：厚みごとの比較

元のノート（`{MAIN_NB}`）で学習したモデルを、**キャビティ厚みごと**に見直す。学習は 5data、検証は 3data のまま変えない。調べることは 3 つ（ユーザー依頼、2026-10-22）。

1. **厚みごとに、温度シフトファクターが効いているか**（学習済みのモデル。4 節）
2. **厚みごとに、シフトに使う温度は、流動中樹脂温度 Tff と加熱シリンダ温度 Tset のどちらが適しているか**（5〜8 節）
3. **厚みごとに学習したモデルと、全厚みの総合データで学習したモデルの、どちらがよいか**（9 節）

**比べるモデル**（元のノートと同じ。どちらも両方の温度を使っていて、違うのは**シフトに使う温度**だけ）

| モデル | 式 | シフトに使う温度 | GP の入力に入る温度 |
|---|---|---|---|
| B_Tset | ln η = ln a_v(Tset) ＋ f( ln γ̇ ＋ ln a_h(Tset), ln P3_3, ln Tff ) | 加熱シリンダ温度 Tset | 流動中樹脂温度 Tff |
| B_Tff | ln η = ln a_v(Tff) ＋ f( ln γ̇ ＋ ln a_h(Tff), ln P3_3, Tset ) | 流動中樹脂温度 Tff | 加熱シリンダ温度 Tset |
| C（基準） | ln η = f( ln γ̇, ln P3_3, ln Tff, Tset ) | なし | 両方 |

横のシフトファクター a_h、縦のシフトファクター a_v（基準温度 Tref で 1）。PP・粉砕材混合 PP はアレニウス型（係数 E_h・E_v [kJ/mol]）、GPPS は WLF 型（係数 A1_h・A1_v）。

**1 の見方（4 節）**：厚みごとに、次の 3 つを見る。
- 温度の順序：同じせん断速度で、低温ほど粘度が高いか（実測と予測）
- 温度による粘度の比 $R = \exp\left(\frac{{1}}{{G}}\sum_g [\ln\eta_{{低温}}(\dot\gamma_g) - \ln\eta_{{高温}}(\dot\gamma_g)]\right)$：最低と最高の加熱シリンダ温度の、成形条件ごとの平均を、せん断速度 γ̇ の重なる範囲の共通の格子（G = 50 点）で比べる。実測の比 R_実測 と、モデルの予測平均から同じ手順で求めた比 R_予測 を比べる
- 換算前後のずれ：学習済みの係数で換算した後に、温度の違うデータが 1 本の曲線（厚みごとの 3 次式）に重なるか（元のノートの「重ね合わせの目安」を厚みごとに分けたもの。粘度の % で表す）

判定：○＝順序が一致し、|ln R_予測 − ln R_実測| がモデルのノイズの標準偏差以内で、換算後のずれが換算前より小さい。×＝予測の順序が実測と逆。△＝それ以外。

**2 の基準（5〜8 節）**：厚みごとに、次の 3 つで「Tff／Tset／差なし」を決め、多数決でまとめる。
- (a) 重ね合わせ：厚みごとに係数（横・縦）を決め直したときの、最小のずれ（データだけで決まる）。差が `A_CLEAR`（既定 0.5 ポイント）以上なら差あり
- (b) 対数周辺尤度：厚みごとの 5data だけで B_Tset・B_Tff を学習し、係数をプロファイル尤度で決めたときの、最大の対数周辺尤度の差 ΔL = LML(B_Tff) − LML(B_Tset)。|ΔL| が `DL_CLEAR`（既定 3.0）以上なら差あり
- (c) 検証データの誤差：保存済みの 3data の予測を、厚みごとに集計した予測誤差（粘度の %）。差が `ERR_CLEAR`（既定 1.0 ポイント）以上なら差あり

**3 の比べ方（9 節）**：厚みごとの 5data だけで B_Tset・B_Tff・C を学習し（B の係数は 6 節で決めたもの）、同じ厚みの 3data を予測する。総合データのモデル（元のノートのモデル）の予測と、厚みごと・全厚みで比べる。

**読むときの注意**：流動中樹脂温度 Tff は、せん断発熱で射出速度とともに変わる。そのため、Tff でのシフトは、温度の効果だけでなく、曲線の形の調整も兼ねうる。(a) だけでなく、(b)(c) も見るのはこのため。''')

# ---------------------------------------------------------------- 1. 使い方・設定
md(f'''## 使い方

1. **先に元のノート `{MAIN_NB}` を実行しておく**（このノートは、その学習済みモデル `<モデル>/学習済みモデル_<モデル>.joblib` と予測値 `予測検証_3data_予測値.csv` を読む。入力の取り方 `INPUT_MODE` は、元のノートと同じにする）。
2. メニューの「すべて実行」で、上から順に実行する（3 材料で約 15〜20 分。大部分は 6 節のプロファイル尤度と、9 節の厚みごとのモデルの学習）。カーネルは、このプロジェクトの `.venv` を選ぶ。
3. 設定を変えたいときは、最初の「設定エリア」のセルだけを直して、最初から実行し直す（時間を短くする：`THICK_PROFILE = False`。6・9 節と、8 節の (b) を飛ばす）。
4. 各節の最後に、**判定**が出る（`>> 判定：OK` か `>> 判定：★要確認`）。OK は「想定どおり」、★ は「想定と違う、または結果を読むときに注意が要る点」。★ が出ても、エラーではない。最後の「判定の一覧」に一覧が出る。
5. 図と表は、材料ごとに **`<材料フォルダ名>_学習結果/<入力の取り方>/厚みごとの比較/`** に保存される（例：`05_材料ごとの機械学習データと予測検証データ/01_バージンPP再々実験/01_バージンPP再々実験_学習結果/条件平均の入力/厚みごとの比較/`）。どの図も、600 dpi の同じ図が `highres/` にある（Git には入れない）。
6. 用語や図の見方は、説明書 `{PDF}` にある。''')
md(r'''## 1. 設定''')
code(r'''# セル1：設定エリア  ▼▼▼ 変えてよいのは、ここだけ ▼▼▼
SEED = 42                     # 乱数シード
NOISE_LB = 1e-6               # ノイズ分散の下限（元のノートと同じ）
USE_LINEAR = True             # 線形項（DotProduct）を足すか（元のノートと同じ）
INPUT_MODE = "cond"           # 読む学習済みモデルの入力の取り方（元のノートと同じにする。"cond"：条件平均、"shot"：ショットごと）
N_RESTARTS = 20               # 9 節：厚みごとの最終モデルの学習での、最適化のやり直し回数（元のノートと同じ）
PROFILE_RESTARTS = 1          # 6 節：プロファイル尤度の格子の各点での、やり直し回数（元のノートと同じ）
USE_EXCLUSIONS = True         # 問題データ一覧（05 フォルダ）の「除外」を除く（元のノートと同じ）
REPO = ".."                   # リポジトリ直下（このノートは 01_機械学習コードフォルダ にある）
MATERIALS = ["PP", "MIX", "GPPS"]    # 調べる材料
MAT_FOLDERS = {"PP": "01_バージンPP再々実験", "MIX": "02_粉砕材混合PP実験", "GPPS": "03_GPPS実験"}
FINE_B = (4.0, 1.0)           # 細かい格子：最良の点の ±4 を 1 刻み（元のノートと同じ。係数の粗い格子と型は、学習済みモデルの設定を使う）
FINE_A = (2.5, 0.5)
MAX_REFINE = 4                # 最良の点が細かい格子の端に当たったとき、取り直す回数の上限（元のノートと同じ）
THICK_PROFILE = True          # 6 節：厚みごとの 5data で GPR を学習し、係数をプロファイル尤度で決める（False：6・9 節と 8 節の (b) を飛ばす）
THICK_MODELS = True           # 9 節：厚みごとのモデルと、総合データのモデルを比べる（6 節の結果を使う）
A_CLEAR = 0.5                 # (a) 重ね合わせのずれ [%] の差が、これ以上なら「差あり」
DL_CLEAR = 3.0                # (b) 対数周辺尤度の差が、これ以上なら「差あり」
ERR_CLEAR = 1.0               # (c)・9 節：予測誤差 [%] の差が、これ以上なら「差あり」
N_JOBS = 8                    # プロファイル尤度を並列に計算する数
FIG_DPI = 200                 # 保存する PNG の解像度
HR_ALL = True                 # すべての図を、600 dpi でも highres/ に保存する（Git の管理外）
HR_ALL_DPI = 600
SHOW_FIGS = True              # 図をノートにも表示する
# ▲▲▲ 設定エリア終了 ▲▲▲''')
code(r'''# セル2：ライブラリ、フォント
%matplotlib inline
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from IPython.display import display

plt.rcParams["figure.dpi"] = 80
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
try:
    fm.findfont("IPAexGothic", fallback_to_default=False)
except ValueError:
    raise RuntimeError("日本語フォント IPAexGothic が見つかりません。インストールしてからカーネルを再起動してください。")
plt.rcParams["font.family"] = ["IPAexGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
KEYS = ["B_Tset", "B_Tff", "C"]
print("Python:", sys.executable, "/", sys.version.split()[0])
print("材料:", MATERIALS, "／入力の取り方:", INPUT_MODE)''')
md(r'''### 共通の部品

元のノートと同じ部品（`04_補助スクリプト/mat_common.py`）と、厚みごとの比較の部品（`04_補助スクリプト/mat_thickness.py`）。''')
code(COMMON_CELL)
code(THICK_CELL)

# ---------------------------------------------------------------- 2. 読み込み
md(r'''## 2. データと学習済みモデルの読み込み

元のノートと同じ手順でデータを読み（問題データ一覧の「除外」を除く）、学習済みモデル（B_Tset・B_Tff・C）と、3data の予測値を読む。

**判定**：学習済みモデルの入力の取り方が `INPUT_MODE` と同じ／読んだモデルで 3data を予測し直すと、保存済みの予測値と一致する（データとモデルの組み合わせが正しい）。''')
code(r'''# セル3：読み込み
import joblib
JUDGES.clear()
D = load_all(MATERIALS)
TRAIN, TEST, POOL, FITS, PRED, CFG, OUT = {}, {}, {}, {}, {}, {}, {}
for mat in MATERIALS:
    TRAIN[mat] = D[(mat, "5data")]
    TEST[mat] = add_groups(TRAIN[mat], D[(mat, "3data")])
    POOL[mat] = pd.concat([TRAIN[mat], TEST[mat]], ignore_index=True)
    OUT[mat] = result_dir(mat, "厚みごとの比較")
    for key in KEYS:
        path = os.path.join(result_dir(mat, key), f"学習済みモデル_{key}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} がありません。先に元のノートを INPUT_MODE = '{INPUT_MODE}' で実行してください")
        FITS[(mat, key)] = joblib.load(path)["fit"]
        PRED[(mat, key)] = pd.read_csv(os.path.join(result_dir(mat, key), "予測検証_3data_予測値.csv"), encoding="utf-8-sig")
        judge(FITS[(mat, key)].get("mode") == INPUT_MODE, f"{key}：学習済みモデルの入力の取り方が {INPUT_MODE}", mat, "2 読み込み")
        mu, _, _, _ = predict(FITS[(mat, key)], TEST[mat])
        chk = TEST[mat][["file"]].assign(mu=mu).merge(PRED[(mat, key)][["file", "予測_ln_eta"]], on="file")
        judge(len(chk) == len(TEST[mat]) and np.allclose(chk["mu"], chk["予測_ln_eta"], atol=1e-6),
              f"{key}：3data を予測し直した値が、保存済みの予測値と一致（{len(chk)}/{len(TEST[mat])} 行）", mat, "2 読み込み")
    CFG[mat] = FITS[(mat, "B_Tff")]["cfg"]
    cnt = TRAIN[mat].groupby("H")["cond"].nunique()
    cnt_te = TEST[mat].groupby("H")["cond"].nunique()
    print(f"【{MAT_NAMES[mat]}】厚み {', '.join(f'{h:g}' for h in thicknesses(TRAIN[mat]))} mm。学習の条件数：" +
          "、".join(f"{h:g} mm {cnt[h]}" for h in cnt.index) + "／検証の条件数：" + "、".join(f"{h:g} mm {cnt_te.get(h, 0)}" for h in cnt.index))
    print("  学習済みの係数：" + "、".join(f"{k} ({FITS[(mat, k)]['th']:g}, {FITS[(mat, k)]['tv']:g})" for k in THICK_KEYS) +
          f"（{'アレニウス型、kJ/mol' if CFG[mat]['type'] == 'arrhenius' else 'WLF 型の A1'}）")''')

# ---------------------------------------------------------------- 3. 厚みごとの学習結果グラフ
md(r'''## 3. 厚みごとの学習結果グラフ（B_Tset と B_Tff）

厚みごとに、せん断速度 ― 粘度の学習結果を、B_Tset（左）と B_Tff（右）で並べる。点は成形条件ごとの平均（塗り＝学習の 5data、白抜き＝検証の 3data）、線は加熱シリンダ温度ごとの予測平均値、帯は 95% 予測区間。曲線の入力（P3_3・Tff）は、同じ厚み・温度の条件平均を、せん断速度で内挿した値（範囲の外は端の値）。

**見方**：3 本の線が、実測の点と同じ順序（低温ほど上）・同じ間隔で並んでいれば、温度の効果を表せている。''')
code(r'''# セル4：厚みごとの学習結果グラフ
for mat in MATERIALS:
    for h in thicknesses(POOL[mat]):
        thick_flow_pair({k: FITS[(mat, k)] for k in THICK_KEYS}, mat, POOL[mat], set(TRAIN[mat]["cond"]), h, OUT[mat])''')

# ---------------------------------------------------------------- 4. 温度シフトの確認
md(r'''## 4. 温度シフトファクターが効いているか（厚みごと、学習済みの係数）

厚みごとに、温度による粘度の比 R（実測・予測）、温度の順序、学習済みの係数で換算する前後のずれを出し、○・△・× で判定する（基準は冒頭）。全厚みをまとめたずれ（元のノート・発表資料の値）も、照合のために表示する。

**判定**：判定が ○ の厚みは OK、△・× は ★（何が満たされなかったかを、文に書く）。''')
code(r'''# セル5：温度シフトの確認
CK = []
for mat in MATERIALS:
    for key in THICK_KEYS:
        fit = FITS[(mat, key)]
        t = thick_check(fit, POOL[mat]).assign(材料=MAT_NAMES[mat])
        CK.append(t)
        r0_ = superpose_rmse(shifted_cond_means(POOL[mat], None, CFG[mat], 0.0, 0.0))
        r1_ = superpose_rmse(shifted_cond_means(POOL[mat], key, CFG[mat], fit["th"], fit["tv"]))
        print(f"【{MAT_NAMES[mat]}・{key}】全厚みのずれ（照合用）：換算前 {pct_err(r0_):.1f}% → 換算後 {pct_err(r1_):.1f}%")
        for _, r in t.iterrows():
            why = "、".join(s for s, ok in [("温度の順序が実測と予測で違う", r["順序が一致"]), ("比 R の大きさが合わない", r["比の大きさが一致"]),
                                              ("換算でずれが減らない", r["換算でずれが減る"])] if not ok)
            judge(r["判定"] == "○", f"{key} 厚み {r['厚み']:g} mm：R 実測 {r['R_実測']:.3f}・予測 {r['R_予測']:.3f}、ずれ {pct_err(r['ずれ_換算前']):.1f} → "
                  f"{pct_err(r['ずれ_換算後']):.1f}%（判定 {r['判定']}{'。' + why if why else ''}）", mat, "4 温度シフトの確認")
CK = pd.concat(CK, ignore_index=True)
CK["ずれ_換算前_%"] = pct_err(CK["ずれ_換算前"])
CK["ずれ_換算後_%"] = pct_err(CK["ずれ_換算後"])
show_cols = ["材料", "モデル", "厚み", "条件数", "比べた γ̇ の範囲", "R_実測", "R_予測", "lnR の差", "ノイズの標準偏差", "ずれ_換算前_%", "ずれ_換算後_%", "判定"]
display(CK[show_cols].round(3))
for mat in MATERIALS:
    c = CK[CK["材料"] == MAT_NAMES[mat]]
    to_csv(c, os.path.join(OUT[mat], "温度シフトの確認_厚みごと.csv"))
    w = c.pivot(index="厚み", columns="モデル", values="R_予測").add_prefix("R_予測_").reset_index()
    w["R_実測"] = c.groupby("厚み")["R_実測"].first().to_numpy()
    thick_bars(w, None, {"R_実測": "実測", "R_予測_B_Tset": "予測（B_Tset）", "R_予測_B_Tff": "予測（B_Tff）"},
               f"{MAT_NAMES[mat]}　温度による粘度の比 R（最低温度 ÷ 最高温度）", "粘度の比 R", OUT[mat], "温度による粘度の比", fmt="{:.2f}", hline=1.0)
    z = c.pivot(index="厚み", columns="モデル", values="ずれ_換算後_%").add_prefix("換算後_").reset_index()
    z["換算前"] = c.groupby("厚み")["ずれ_換算前_%"].first().to_numpy()
    thick_bars(z, None, {"換算前": "換算前", "換算後_B_Tset": "加熱シリンダ温度で換算（学習済みの係数）", "換算後_B_Tff": "流動中樹脂温度で換算（学習済みの係数）"},
               f"{MAT_NAMES[mat]}　重ね合わせのずれ（学習済みの係数）", "ずれ (%)", OUT[mat], "重ね合わせのずれ_学習済みの係数")
    for h in thicknesses(POOL[mat]):
        thick_master(mat, POOL[mat], CFG[mat], h, {k: (FITS[(mat, k)]["th"], FITS[(mat, k)]["tv"]) for k in THICK_KEYS}, OUT[mat], "重ね合わせ_学習済みの係数")''')

# ---------------------------------------------------------------- 5. (a) 重ね合わせ
md(r'''## 5. (a) 厚みごとに係数を決め直した重ね合わせ

厚みごとのデータ（5data＋3data の条件平均）だけで、ずれが最小になる係数（横・縦）を、元のノートのプロファイル尤度と同じ格子で探す。加熱シリンダ温度 Tset で換算した場合と、流動中樹脂温度 Tff で換算した場合の、最小のずれを比べる。モデル（GP）を使わず、データだけで決まる。

**見方**：ずれが小さい温度ほど、その厚みで、温度の違う粘度曲線を 1 本に重ねやすい。

**判定**：最良の係数が、粗い格子の端に当たっていない（当たっていれば ★。尾根の上で、さらに外にずれが小さい点がありうる）。''')
code(r'''# セル6：(a) 厚みごとの重ね合わせ
SUP = []
for mat in MATERIALS:
    for h in thicknesses(POOL[mat]):
        ph = sub_h(POOL[mat], h)
        row = {"材料": MAT_NAMES[mat], "厚み": h, "条件数": ph["cond"].nunique()}
        th_n, tv_n, unit = theta_names(CFG[mat])
        for key in THICK_KEYS:
            s = TEMP_SHORT[key]
            b = best_superpose(ph, key, CFG[mat])
            row.update({"ずれ_換算前_%": pct_err(b["rmse_0"]), f"ずれ_{s}_%": pct_err(b["rmse"]), f"横の係数_{s}": b["th"], f"縦の係数_{s}": b["tv"],
                        f"格子の端_{s}": b["at_edge"]})
            judge(not b["at_edge"], f"(a) {key} 厚み {h:g} mm：最良の係数（{th_n} {b['th']:g}・{tv_n} {b['tv']:g} {unit}）が、粗い格子の端に当たっていない", mat, "5 (a) 重ね合わせ")
        row["単位"] = unit
        row["適した温度(a)"] = decide(row["ずれ_Tff_%"], row["ずれ_Tset_%"], A_CLEAR, False)
        SUP.append(row)
        thick_master(mat, POOL[mat], CFG[mat], h, {k: (row[f"横の係数_{TEMP_SHORT[k]}"], row[f"縦の係数_{TEMP_SHORT[k]}"]) for k in THICK_KEYS},
                     OUT[mat], "重ね合わせ_厚みごとの係数")
SUP = pd.DataFrame(SUP)
display(SUP.round(2))
for mat in MATERIALS:
    s = SUP[SUP["材料"] == MAT_NAMES[mat]]
    to_csv(s, os.path.join(OUT[mat], "重ね合わせ_厚みごと.csv"))
    thick_bars(s, None, {"ずれ_換算前_%": "換算前", "ずれ_Tset_%": "加熱シリンダ温度で換算", "ずれ_Tff_%": "流動中樹脂温度で換算"},
               f"{MAT_NAMES[mat]}　(a) 厚みごとに係数を決め直したときの重ね合わせのずれ", "ずれ (%)", OUT[mat], "重ね合わせのずれ_厚みごとの係数")''')

# ---------------------------------------------------------------- 6. (b) プロファイル尤度
md(r'''## 6. (b) 厚みごとのプロファイル尤度（GPR）

厚みごとの 5data だけで B_Tset・B_Tff を学習し、係数を元のノートと同じプロファイル尤度で決める。最大の対数周辺尤度 LML を比べ、ΔL = LML(B_Tff) − LML(B_Tset) を求める。あわせて、それぞれのシフトなし（係数 0）との差（2 つの係数を足した効果。3.0 以上で、尤度比検定の 5% 基準）も出す。

**見方**：ΔL が正で大きいほど、その厚みでは流動中樹脂温度でずらしたほうが、学習データをよく説明する（負なら加熱シリンダ温度）。

**判定**：最良の係数が粗い格子の端に当たっていない／細かい格子の取り直しが収束した。''')
code(r'''# セル7：(b) 厚みごとのプロファイル尤度
PROF, PROF_PTS, BESTH, PL = [], [], {}, []
if THICK_PROFILE:
    t0 = time.time()
    for mat in MATERIALS:
        th_n, tv_n, unit = theta_names(CFG[mat])
        for h in thicknesses(TRAIN[mat]):
            for key in THICK_KEYS:
                pts, b = thick_profile(key, TRAIN[mat], CFG[mat], h, n_jobs=N_JOBS, restarts=PROFILE_RESTARTS, mode=INPUT_MODE)
                BESTH[(mat, h, key)] = b
                PROF.append({"材料": MAT_NAMES[mat], "厚み": h, "モデル": key, "学習の条件数": sub_h(TRAIN[mat], h)["cond"].nunique(), "横の係数": b["th_best"],
                             "縦の係数": b["tv_best"], "単位": unit, "LML": b["LML_best"], "係数0との差": b["dLML_vs_0"], "格子の端": b["at_edge"],
                             "取り直しが収束": b["refine_converged"], "計算した点": b["n_points"]})
                PROF_PTS.append(pts.drop(columns="theta").assign(材料=MAT_NAMES[mat], 厚み=h, モデル=key))
                ok = not b["at_edge"] and b["refine_converged"]
                why = "格子の端でなく、取り直しも収束" if ok else "、".join(
                    s for s in ["粗い格子の端に当たった" if b["at_edge"] else "",
                                f"細かい格子の取り直しが {MAX_REFINE} 回で収束しなかった（最良の点が動き続けた。尤度の面が平らで、係数が決まりにくい）"
                                if not b["refine_converged"] else ""] if s)
                judge(ok, f"(b) {key} 厚み {h:g} mm：最良の係数 {th_n} {b['th_best']:g}・{tv_n} {b['tv_best']:g} {unit}。"
                      f"{why}（LML {b['LML_best']:.1f}、係数 0 との差 {b['dLML_vs_0']:.1f}）", mat, "6 (b) プロファイル尤度")
            l_ff, l_set = BESTH[(mat, h, "B_Tff")]["LML_best"], BESTH[(mat, h, "B_Tset")]["LML_best"]
            PL.append({"材料": MAT_NAMES[mat], "厚み": h, "LML_Tset": l_set, "LML_Tff": l_ff, "ΔL（Tff − Tset）": l_ff - l_set,
                       "係数0との差_Tset": BESTH[(mat, h, "B_Tset")]["dLML_vs_0"], "係数0との差_Tff": BESTH[(mat, h, "B_Tff")]["dLML_vs_0"],
                       "適した温度(b)": decide(l_ff, l_set, DL_CLEAR, True)})
            print(f"  {MAT_NAMES[mat]} 厚み {h:g} mm：ΔL = {l_ff - l_set:.1f}（経過 {time.time() - t0:.0f} 秒）")
    PROF, PL = pd.DataFrame(PROF), pd.DataFrame(PL)
    display(PROF.round(2))
    display(PL.round(2))
    for mat in MATERIALS:
        to_csv(PROF[PROF["材料"] == MAT_NAMES[mat]], os.path.join(OUT[mat], "プロファイル尤度_厚みごと.csv"))
        to_csv(pd.concat(PROF_PTS)[lambda d: d["材料"] == MAT_NAMES[mat]], os.path.join(OUT[mat], "プロファイル尤度_格子_厚みごと.csv"))
        p = PL[PL["材料"] == MAT_NAMES[mat]]
        to_csv(p, os.path.join(OUT[mat], "対数周辺尤度の差_厚みごと.csv"))
        thick_bars(p, None, {"係数0との差_Tset": "加熱シリンダ温度：シフトなしとの差", "係数0との差_Tff": "流動中樹脂温度：シフトなしとの差",
                             "ΔL（Tff − Tset）": "ΔL（流動中樹脂温度 − 加熱シリンダ温度）"},
                   f"{MAT_NAMES[mat]}　(b) 厚みごとの対数周辺尤度の差", "対数周辺尤度の差", OUT[mat], "対数周辺尤度の差_厚みごと", hline=DL_CLEAR)
else:
    print("THICK_PROFILE = False のため、飛ばした（8 節の (b) と 9 節も飛ばす）")''')

# ---------------------------------------------------------------- 7. (c) 検証データの誤差
md(r'''## 7. (c) 検証データの誤差（厚みごと）

元のノートで保存した 3data の予測値（B_Tset・B_Tff・C。総合データで学習したモデル）を、厚みごとに集計する。予測誤差は、ln η の RMSE を粘度の % に換算した値（e^RMSE − 1）。

**見方**：誤差が小さい温度ほど、その厚みで、学習に用いていない射出率条件の予測に向いている。低射出率側（3・5 mm/s）の検証は、厚み 1.5・2.0 mm にしかない。

**判定**：全厚みをまとめた RMSE が、元のノートの指標の表（`比較/予測検証_3data_指標.csv`）と一致する。''')
code(r'''# セル8：(c) 検証データの誤差
VAL = []
for mat in MATERIALS:
    met = pd.read_csv(os.path.join(result_dir(mat, "比較"), "予測検証_3data_指標.csv"), encoding="utf-8-sig")
    for key in KEYS:
        p = PRED[(mat, key)]
        for h in thicknesses(p):
            VAL += err_rows(sub_h(p, h), {"材料": MAT_NAMES[mat], "モデル": key, "学習": "総合", "厚み": h})
        allr = err_rows(p, {"材料": MAT_NAMES[mat], "モデル": key, "学習": "総合", "厚み": "全厚み"})
        VAL += allr
        ref = float(met[(met["モデル"] == key) & (met["範囲"] == "全体")]["RMSE"].iloc[0])
        judge(np.isclose(allr[0]["RMSE"], ref), f"(c) {key}：全厚みの RMSE（{allr[0]['RMSE']:.4f}）が、元のノートの指標の表（{ref:.4f}）と一致", mat, "7 (c) 検証データ")
VAL = pd.DataFrame(VAL)
VC = []
for mat in MATERIALS:
    v = VAL[(VAL["材料"] == MAT_NAMES[mat]) & (VAL["範囲"] == "全体") & (VAL["厚み"] != "全厚み")]
    w = v.pivot(index="厚み", columns="モデル", values="誤差_%").reset_index()
    w["厚み"] = w["厚み"].astype(float)
    w.insert(0, "材料", MAT_NAMES[mat])
    w["適した温度(c)"] = [decide(a, b, ERR_CLEAR, False) for a, b in zip(w["B_Tff"], w["B_Tset"])]
    VC.append(w)
    to_csv(VAL[VAL["材料"] == MAT_NAMES[mat]], os.path.join(OUT[mat], "検証誤差_厚みごと.csv"))
    thick_bars(w, None, {"C": "C（シフトなし）", "B_Tset": "B_Tset（加熱シリンダ温度）", "B_Tff": "B_Tff（流動中樹脂温度）"},
               f"{MAT_NAMES[mat]}　(c) 検証データの予測誤差（厚みごと）", "予測誤差 (%)", OUT[mat], "検証誤差_厚みごと")
VC = pd.concat(VC, ignore_index=True)
display(VC.round(2))
display(VAL[VAL["厚み"] != "全厚み"].pivot_table(index=["材料", "厚み", "範囲"], columns="モデル", values="誤差_%").round(1))''')

# ---------------------------------------------------------------- 8. まとめ
md(r'''## 8. まとめ：厚みごとに適した温度

(a)(b)(c) の判定を、材料・厚みごとに並べ、多数決で「Tff（流動中樹脂温度）／Tset（加熱シリンダ温度）／差なし」を決める（同数なら「差なし」。「—」は数えない）。`厚みごとの比較_まとめ.csv` に保存する。''')
code(r'''# セル9：まとめ（Tff か Tset か）
SUMT = SUP[["材料", "厚み", "ずれ_Tset_%", "ずれ_Tff_%", "適した温度(a)"]].merge(VC[["材料", "厚み", "B_Tset", "B_Tff", "適した温度(c)"]].rename(
    columns={"B_Tset": "誤差_Tset_%", "B_Tff": "誤差_Tff_%"}), on=["材料", "厚み"], how="left")
if len(PL):
    SUMT = SUMT.merge(PL[["材料", "厚み", "ΔL（Tff − Tset）", "適した温度(b)"]], on=["材料", "厚み"], how="left")
else:
    SUMT["ΔL（Tff − Tset）"], SUMT["適した温度(b)"] = np.nan, "—"
SUMT["総合の判定"] = [majority([a, b, c]) for a, b, c in zip(SUMT["適した温度(a)"], SUMT["適した温度(b)"], SUMT["適した温度(c)"])]
SUMT = SUMT[["材料", "厚み", "ずれ_Tset_%", "ずれ_Tff_%", "適した温度(a)", "ΔL（Tff − Tset）", "適した温度(b)", "誤差_Tset_%", "誤差_Tff_%", "適した温度(c)", "総合の判定"]]
display(SUMT.round(2))
for mat in MATERIALS:
    s = SUMT[SUMT["材料"] == MAT_NAMES[mat]]
    to_csv(s, os.path.join(OUT[mat], "厚みごとの比較_まとめ.csv"))
    print(f"【{MAT_NAMES[mat]}】" + "、".join(f"{h:g} mm：{j}" for h, j in zip(s["厚み"], s["総合の判定"])))''')

# ---------------------------------------------------------------- 9. 厚みごとのモデルと総合データのモデル
md(r'''## 9. 厚みごとのモデルと総合データのモデルの比較

学習 5data・検証 3data のまま、厚みごとの 5data だけで B_Tset・B_Tff・C を学習し（B の係数は 6 節で決めたもの。最終モデルは `N_RESTARTS` 回の最適化）、同じ厚みの 3data を予測する。総合データのモデル（全厚みの 5data で学習した、元のノートのモデル）の予測と、厚みごと・全厚みで比べる。

**見方**：予測誤差が、どちらかの学習のしかたで、多くの厚みで `ERR_CLEAR` ポイント以上小さければ、その学習のしかたが向いている。差が小さければ、データの量（1 つのモデルに使える条件の数）と使いやすさ（総合モデルは 1 つで全厚みを予測できる）から、総合データで学習するほうがよい。

**判定**：すべての厚み・モデルで学習と予測ができた／総合データのモデルの値が 7 節と一致する。''')
code(r'''# セル10：厚みごとのモデルと総合データのモデル
CMP, TPRED, REC = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
if THICK_MODELS and THICK_PROFILE:
    t0 = time.time()
    tp = []
    for mat in MATERIALS:
        for h in thicknesses(TRAIN[mat]):
            for key in KEYS:
                b = BESTH.get((mat, h, key)) if key != "C" else None
                fit_h, pr = thick_model(key, TRAIN[mat], TEST[mat], CFG[mat], h, b, restarts=N_RESTARTS, mode=INPUT_MODE)
                tp.append(pr.assign(材料=MAT_NAMES[mat], モデル=key, 学習="厚みごと"))
            print(f"  {MAT_NAMES[mat]} 厚み {h:g} mm（経過 {time.time() - t0:.0f} 秒）")
    TPRED = pd.concat(tp, ignore_index=True)
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
                       f"{MAT_NAMES[mat]}　{key}：厚みごとのモデルと総合データのモデルの予測誤差", "予測誤差 (%)", OUT[mat], f"厚みごとと総合_{key}")
    REC = pd.DataFrame(rec)
    display(REC.round(2))
    display(CMP[CMP["範囲"] == "全体"].pivot_table(index=["材料", "モデル", "厚みの名前"], columns="学習", values="誤差_%").round(1))
    for mat in MATERIALS:
        to_csv(CMP[CMP["材料"] == MAT_NAMES[mat]], os.path.join(OUT[mat], "厚みごとと総合の比較.csv"))
        to_csv(REC[REC["材料"] == MAT_NAMES[mat]], os.path.join(OUT[mat], "厚みごとと総合の比較_まとめ.csv"))
        to_csv(TPRED[TPRED["材料"] == MAT_NAMES[mat]], os.path.join(OUT[mat], "厚みごとのモデル_予測値.csv"))
    if len(PROF):
        print("\n厚みごとに決めた係数（6 節）と、総合データのモデルの係数：")
        for mat in MATERIALS:
            for key in THICK_KEYS:
                p = PROF[(PROF["材料"] == MAT_NAMES[mat]) & (PROF["モデル"] == key)]
                print(f"  {MAT_NAMES[mat]}・{key}：" + "、".join(f"{h:g} mm ({a:g}, {b:g})" for h, a, b in zip(p["厚み"], p["横の係数"], p["縦の係数"])) +
                      f"／総合 ({FITS[(mat, key)]['th']:g}, {FITS[(mat, key)]['tv']:g})")
else:
    print("THICK_MODELS または THICK_PROFILE が False のため、飛ばした")''')

# ---------------------------------------------------------------- 10. 判定の一覧
md(r'''## 10. 判定の一覧

★ の行を確かめる。材料ごとに `厚みごとの比較/判定の一覧.csv` に保存する。''')
code(r'''# セル11：判定の一覧
JT = pd.DataFrame(JUDGES)
print(f"判定 {len(JT)} 件のうち、★要確認 {int((JT['判定'] != 'OK').sum())} 件")
display(JT[JT["判定"] != "OK"])
for mat in MATERIALS:
    to_csv(JT[JT["材料"].isin([MAT_NAMES[mat], ""])], os.path.join(OUT[mat], "判定の一覧.csv"))
print("結果の置き場所：" + "、".join(os.path.relpath(OUT[m], REPO) for m in MATERIALS))''')


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
