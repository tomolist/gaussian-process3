# -*- coding: utf-8 -*-
"""ノート生成スクリプト（build_step*_nb.py）に共通の、ノートの先頭部分。

- 使い方の説明（ユーザーが一人でノートを実行して、結果を判断するための案内）
- 設定エリアのセル（GPR6月ノートの「設定エリア」にならう）
- 共通の部品のセル（nb_common.py の中身をそのまま入れる。ノートは単独で動く）
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def usage_md(step, out_dir, pdf, minutes, hr=True):
    hr_item = f"""
7. **高解像度のグラフ**（報告書・発表資料用。GPR6月ノートのセル9-2 と同じ見せ方）は、「高解像度のグラフ」の節のセルで作る。**このセルだけは、セルの先頭にも設定エリアがあり、そこだけを直して、このセルだけを実行し直せる**（学習はやり直さない。全体を一度実行した後で）。保存先は `{out_dir}/highres/`（Git には入れない）。""" if hr else ""
    return f"""## 使い方（このノートを一人で実行して、結果を判断するために）

1. メニューの「すべて実行」で、上から順に実行する（約 {minutes} 分）。カーネルは、このプロジェクトの `.venv` を選ぶ。
2. 設定を変えたいときは、最初の「設定エリア」のセルだけを直して、最初から実行し直す（高解像度のグラフの設定は、下の 7）。
3. 各節の最後に、**判定**が出る（`>> 判定：OK` か `>> 判定：★要確認`）。OK は「想定どおり」、★ は「想定と違う、または結果を読むときに注意が要る点」。基準は、判定の文の中に書いてある。★ が出ても、エラーではない。説明書の「判定の読み方」で、★ の意味を確かめる。
4. 図と表は `{out_dir}/` に保存される。図の見方は、説明書 `{pdf}` にある。
5. データを取り直したとき、問題のあるデータを入れ替えたときは、`問題データ一覧.md` の手順に従う。このノートは、全条件のファイルを読み、`02_機械学習データ/問題データ一覧.csv` の「除外」を除く。
6. 用語や図の見方で分からないところは、説明書の用語集で調べる。記号（Tff、Tset、R3、E_h など）の意味は、`results/学習結果説明_記号表.pdf` に一覧がある。{hr_item}"""


def settings_cell(extra=""):
    return f"""# セル1：設定エリア  ▼▼▼ 変えてよいのは、ここだけ ▼▼▼
SEED = 42                     # 乱数シード
NOISE_LB = 1e-6               # ノイズ分散の下限（旧ノートは 1e-3）
N_RESTARTS = 5                # GPR の最適化のやり直し回数（増やすと遅いが、安定する）
USE_EXCLUSIONS = True         # True：問題データ一覧の「除外」を使う／False：全条件を使う（取り直した後の確認用）
REPO = ".."                   # リポジトリ直下（このノートは 01_機械学習コードフォルダ にある）
FIG_DPI = 200                 # 保存する PNG の解像度
{extra}# ▲▲▲ 設定エリア終了 ▲▲▲"""


HR_MD = r"""## 高解像度のグラフ（報告書・発表資料用）

せん断速度 ― 粘度の学習結果グラフを、GPR6月ノートのセル9-2 と同じ見せ方で、**600 dpi（15 × 10 インチ）**で 1 枚ずつ保存する。線は予測の平均、帯は 95% 区間、塗りつぶしの点は学習に使った条件、白抜きの点は学習に使わなかった条件（検証）、× は問題データ一覧の「感度確認」の条件。

- **レイアウト**：`"thickness"`＝厚みごとに 1 枚（3 つの設定温度の曲線）、`"temperature"`＝設定温度ごとに 1 枚（4 つの厚みの曲線。6月のノートと同じ）。
- **変えられるもの**（セルの先頭の設定エリア）：描くモデル・材料・学習の種類、帯（予測区間／信頼区間）、点（条件平均／ショットごと）、軸の範囲、予測線の延長、図の大きさ・解像度・文字・マーカーの大きさ、タイトルと軸名の表示、保存形式（png／pdf／svg）。
- 全体を一度実行して、学習が終わった後なら、このセルだけを何度でも実行し直せる。"""

HR_SETTINGS = r"""# ==========================================
# ▼▼ 設定エリア（グラフの見た目・文字サイズ。このセルの設定だけを直して、このセルだけを実行し直せる） ▼▼
# ==========================================
@@SPECIFIC@@
HR_MATERIALS = ["PP", "MIX"]       # 描く材料（PP：PP、MIX：粉砕材混合 PP）
HR_LAYOUTS = ["thickness", "temperature"]   # "thickness"：厚みごとに 1 枚（3 つの設定温度）／"temperature"：設定温度ごとに 1 枚（4 つの厚み。6月のノートと同じ）
HR_BAND = "predict"                # 帯："predict"＝95% 予測区間（ショット間のばらつき込み）／"latent"＝95% 信頼区間（平均の不確かさだけ）／None＝帯なし
HR_POINTS = "cond"                 # 点："cond"＝条件平均／"shot"＝ショットごと
HR_SHOW_TEST = True                # 学習に使わなかった条件（検証）を、白抜きで重ねるか
HR_MARK_SENS = True                # 問題データ一覧の「感度確認」の条件を、× で重ねるか
HR_X_RANGE = None                  # 予測線の範囲 [1/s]。None なら、各線のデータの範囲を HR_EXTEND 倍に延長。(30, 50000) のように指定すると、全部の線をその範囲に延長（外挿の確認用）
HR_EXTEND = 1.3
HR_XLIM = {"thickness": None, "temperature": (30, 50000)}   # 横軸（せん断速度 [1/s]）の表示範囲。(下限, 上限)＝固定（10 の累乗の目盛り。6月と同じ）／None＝その図のデータに合わせる（1・2・5 の目盛り）。
HR_YLIM = {"thickness": None, "temperature": (0.5, 5000)}   # 縦軸（粘度 [Pa·s]）の表示範囲。同じ書き方。レイアウトごとに変えられる（全部を同じ範囲にするなら、(30, 50000) のように書く）
HR_FIGSIZE = (15, 10)              # 図の大きさ（幅, 高さ）[インチ]
HR_DPI = 600                       # 解像度
HR_FONT = {"axis": 24, "legend": 24, "title": 24, "tick": 28}   # 軸名・凡例・タイトル・目盛り数字の文字サイズ
HR_MARKER = 200                    # 点の大きさ（条件平均のとき。面積）
HR_MARKER_SHOT = 70                # 点の大きさ（ショットごとのとき）
HR_LINE = 3.0                      # 予測線の太さ
HR_SHOW_TITLE = True               # タイトルを付けるか
HR_SHOW_XLABEL = False             # 軸名を付けるか（6月のノートと同じく、既定は付けない。資料に貼るとき、別に付ける場合のため）
HR_SHOW_YLABEL = False
HR_FORMATS = ["png"]               # 保存形式。"png"・"pdf"・"svg" から選ぶ（pdf・svg は、拡大しても粗くならない）
HR_SHOW = False                    # True なら、画面にも表示する（重い）
# ==========================================
HR_KW = dict(layouts=HR_LAYOUTS, band=HR_BAND, points=HR_POINTS, show_test=HR_SHOW_TEST, mark_sens=HR_MARK_SENS, x_range=HR_X_RANGE, extend=HR_EXTEND,
             xlim=HR_XLIM, ylim=HR_YLIM, figsize=HR_FIGSIZE, dpi=HR_DPI, font=HR_FONT, marker=HR_MARKER, marker_shot=HR_MARKER_SHOT, line=HR_LINE,
             show_title=HR_SHOW_TITLE, show_xlabel=HR_SHOW_XLABEL, show_ylabel=HR_SHOW_YLABEL, formats=HR_FORMATS, show=HR_SHOW)
HR_DIR = os.path.join(OUT_DIR, "highres")
"""


def hr_cell(step_comment, specific, driver):
    """高解像度のグラフのセル（設定エリア ＋ 手順ごとの描画部分）。specific：手順ごとの設定の行、driver：描画のコード"""
    return step_comment + "\n" + HR_SETTINGS.replace("@@SPECIFIC@@", specific) + "\n" + driver


def common_cell():
    src = open(os.path.join(HERE, "nb_common.py"), encoding="utf-8").read()
    body = src.split('"""', 2)[2].lstrip("\n")          # 先頭の docstring は除く
    return "# セル：共通の部品（読み込み・モデル・評価）。04_補助スクリプト/nb_common.py の中身。直すときは、そのファイルを直して、ノートを作り直す\n" + body


COMMON_CELL = common_cell()
