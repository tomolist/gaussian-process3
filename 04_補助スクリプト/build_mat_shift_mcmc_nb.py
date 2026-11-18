# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR の「学習用」のノートの MCMC 版（未実行）を生成する（2026-11-17 のユーザー依頼）。

使い方（リポジトリ直下で）: uv run python 04_補助スクリプト/build_mat_shift_mcmc_nb.py <出力する .ipynb> [--material PP|MIX|GPPS]
  （成果物は 01_機械学習コードフォルダ/材料別_温度シフトファクター検討/02_材料別_温度シフト機械学習コード/02_MCMC(適応型RWM)/ に置く。
   既存のファイルは上書きしない。--material を付けると、MATERIALS をその材料だけにし、SHIFT_STRUCT を本番の形〔PP・MIX は "A"、GPPS は "B"〕にする。
   勾配法版の PP_・MIXPP_・GPPS_ のノート〔ユーザーが 11-12 に手で複製したもの〕と同じ分け方）

作り方：勾配法版の生成スクリプト build_mat_shift_nb.py を _tmp/build_mat_shift_mcmc_nb/<日時>/ に実行して、勾配法版のノートを作り、
そのセルを置き換える・足す（置き換える文は、ちょうど 1 回あることを確かめる）。勾配法版の共通のセルを直したら、これを実行し直すだけで MCMC 版にも入る。
ユーザーの決定（11-17）：温度シフトの係数（E_h・E_v など）も、カーネルのハイパーパラメータと一緒に、適応型 RWM の MCMC でサンプリングする
（プロファイル尤度の格子は使わない）。予測は事後分布の混合。結果は、名前に「MCMC」を含む学習結果フォルダ（例：4回目_…_学習結果_MCMC）に置く。
MCMC の部品は 04_補助スクリプト/mat_mcmc.py（ノートの「MCMC の部品」のセルに、そのまま入る）。
"""
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MATS = {"PP": "A", "MIX": "A", "GPPS": "B"}      # --material → 本番の SHIFT_STRUCT（本番学習の設定の決定.pdf。通常版）
GRAD_NB = "../01_勾配法（L-BFGS-B）/材料別_温度シフトGPR_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb"   # このノートの場所からの、勾配法版の共通のノート

_mcmc = open(os.path.join(HERE, "mat_mcmc.py"), encoding="utf-8").read().split('"""', 2)[2].lstrip("\n")
MCMC_CELL = "# セル：MCMC の部品（適応型 RWM・事後分布の要約・混合分布の予測）。04_補助スクリプト/mat_mcmc.py の中身。直すときは、そのファイルを直して、ノートを作り直す\n" + _mcmc


def to_lines(s):
    lines = s.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def src(c):
    return "".join(c["source"])


def set_src(c, s):
    c["source"] = to_lines(s)


def rep(c, old, new, what):
    s = src(c)
    n = s.count(old)
    if n != 1:
        sys.exit(f"置き換える文が {n} 回（1 回のはず）：{what}。build_mat_shift_nb.py か mat_nb_cells.py の文が変わったので、このスクリプトを直す")
    set_src(c, s.replace(old, new))


def rep_between(c, start, end, new, what):
    """start（含む）から end（含まない）までを new に置き換える"""
    s = src(c)
    i, j = s.find(start), s.find(end)
    if s.count(start) != 1 or s.count(end) != 1 or not (0 <= i < j):
        sys.exit(f"置き換える範囲が見つからない：{what}")
    set_src(c, s[:i] + new + s[j:])


def find(cells, head, kind="code"):
    hit = [i for i, c in enumerate(cells) if c["cell_type"] == kind and src(c).startswith(head)]
    if len(hit) != 1:
        sys.exit(f"セル「{head}」が {len(hit)} 個（1 個のはず）")
    return hit[0]


def md_cell(s):
    return {"cell_type": "markdown", "metadata": {}, "source": to_lines(s)}


def code_cell(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": to_lines(s)}


# ---------------------------------------------------------------- 説明の文（マークダウン）
TITLE = "# 材料別の温度シフト GPR（ARD Matern 5/2 ＋ 線形 ＋ ホワイトノイズ）：学習用・MCMC 版（適応型 RWM）"
INTRO_MCMC = rf'''**MCMC 版**（2026-11-17 のユーザー依頼）：勾配法版（`{GRAD_NB}`）と同じデータ・モデル・図で、**対数周辺尤度（LML）の扱いだけを、勾配法（L-BFGS-B）での最大化から、適応型 RWM の MCMC でのサンプリングに変えた**ノート。
- 勾配法版：係数 θ（E_h・E_v など）をプロファイル尤度の格子で決め、各点でカーネルのハイパーパラメータを L-BFGS-B で最大化して、**1 組の値**を使う。
- MCMC 版：**θ とカーネルのハイパーパラメータを一緒に MCMC でサンプリング**し（ユーザーの決定）、**ありうる値の分布（事後分布）**として求める。予測は、事後分布から選んだ多くの組の予測を混ぜた分布（混合分布）で出すので、ハイパーパラメータと θ の決めきれなさが、予測幅に入る。
- 結果は、**名前に「MCMC」を含む学習結果フォルダ**（例：最新の学習結果フォルダの隣の `4回目_01_バージンPP再々実験_学習結果_MCMC`）に置く。中の構成と図・表の名前は勾配法版と同じなので、勾配法版のフォルダを選ぶと上書きになる。名前に MCMC がないフォルダを選ぶと止まる（設定 `REQUIRE_MCMC_FOLDER`）。'''

EST_MCMC = r'''**推定**（MCMC 版）：θ とカーネルのハイパーパラメータを、適応型 RWM の MCMC でまとめてサンプリングする（「5. 学習」）。事後分布は、LML（勾配法版で最大化していたものと同じ式）に、事前分布（カーネルは探索範囲の中で対数一様、θ は `SHIFT_PRIOR` の範囲で一様）を掛けたもの。θ の推定値は事後分布の中央値、95% 区間は 2.5〜97.5%。**支持の判定**は、A（縦だけ）は E_v の 95% 区間が 0 を含まないこと、B（縦と横）は (E_h, E_v) = (0, 0) が事後分布の 95% の楕円の外にあること（勾配法版の「θ = 0 との ΔLML が 3.0・1.92 以上」の代わり）。標準化と中心化は、勾配法版と同じく、学習データ（5data）だけで、θ によらない定数で行う。

'''

SEC5 = r'''## 5. 学習（MCMC：適応型 RWM）

### 5.1 MCMC による事後分布のサンプリング（時間がかかる）

勾配法版では、係数 θ をプロファイル尤度の格子で決め、各点でハイパーパラメータを L-BFGS-B で最大化した。この MCMC 版では、**係数 θ（B は E_h・E_v、A は E_v だけ、C はなし。WLF 型は A1）と、カーネルのハイパーパラメータ（振幅・長さスケール・線形項・ノイズ）を、まとめて MCMC でサンプリングする**（2026-11-17 のユーザーの決定。本体の手順5b と同じ考え方）。
- **事後分布**：対数事後確率 = 対数周辺尤度（LML。勾配法版で最大化していたものと同じ式）＋ 事前分布。事前分布は、カーネルのハイパーパラメータは探索範囲（`kernel.bounds`。勾配法版と同じ）の中で対数一様、係数は `SHIFT_PRIOR` の範囲で一様。範囲の中では、事後分布は LML に比例する。
- **適応型 RWM**（ランダムウォーク・メトロポリス法）：今の点から、正規分布で少しずらした点を提案し、事後確率の比で受け入れるかを決める、を繰り返す。調整期間（`MCMC_WARM` 回）に、提案の分布の形（パラメータどうしの相関）を、それまでの足跡から学び、歩幅を、採択率が `MCMC_TARGET_ACC`（0.234）に近づくように合わせる。本番（`MCMC_SAMPLES` 回）は、提案の分布を固定する。チェーンを `MCMC_CHAINS` 本、別々の初期値から並列に回す（本体の手順3c・5b と同じ方式）。
- **初期値**（`MCMC_INIT`）：既定の `"mle"` は、係数 0 で最尤推定（L-BFGS-B）したカーネルに、チェーンごとのずれを足した点から始める（勾配法は、出発点を決めるだけに使う。結果は MCMC のサンプルから出す）。係数は 0 の近くから始める。`"default"` は、カーネルの既定値から始める。
- **収束**：チェーンどうしの食い違い **R-hat** が `MCMC_RHAT`（1.01）以下、**有効サンプル数（ESS bulk）**が `MCMC_ESS`（400）以上なら、収束とみなす。届かなければ、本番を続きから足す（上限 `MCMC_MAX_SAMPLES`）。
- **出力**（モデルのフォルダ）：`MCMC_事後サンプル.csv`（全チェーンから等間隔に最大 1 万行。列 lp は対数事後確率、chain・draw はチェーンと本番の何回目か）、`MCMC_収束の診断.csv`（パラメータごとの R-hat・ESS、反復数、採択率、時間）。
- 時間の目安：PP の全厚み（460 行）で、1 チェーン 2.5 万回に約 10 分（チェーンは並列。2026-11-17 の測定で、尤度 1 回 約 23 ms）。収束の基準に届かず本番を足すと、最大で約 4 倍。厚みごとは行が少ないので速い。

**判定**：MCMC が収束した／採択率（本番）が 0.15〜0.40 の中。

### 5.1b 事後分布の要約と、係数の支持

係数の事後分布の**中央値**を推定値、**2.5〜97.5% を 95% 区間**とする（勾配法版の「LML が最大から 1.92 以内」の代わり）。**支持の判定**：A（縦だけ）は、E_v の 95% 区間が 0 を含まなければ支持。B（縦と横）は、2 つの係数を合わせて、(E_h, E_v) = (0, 0) が事後分布の 95% の楕円の外（マハラノビス距離² が 5.99〔自由度 2 のカイ二乗分布の 95% 点〕を超える）なら支持。横と縦は別々には決まりにくい（斜めの尾根）ので、差 E_v − E_h の区間も出す。表は `シフトファクターの推定.csv`（中央値・区間・P(E_v > 0)・支持）と、`事後分布の要約.csv`（全パラメータ。カーネルは対数と元の値）。

**判定**：温度のシフトが事後分布で支持される（支持されなければ、温度の効果はほかの入力が担っている）／係数の 95% 区間が、事前分布の範囲の端に届いていない（届いたら `SHIFT_PRIOR` を広げる）。'''

FIG_POST = r'''**図の見方**（`事後分布_係数.png`・`事後分布_ハイパーパラメータ.png`）：係数の事後分布のヒストグラム（赤の実線が中央値、破線が 95% 区間、黒線が 0〔シフトなし〕）。B は E_h–E_v の散布図も描く（× が (0, 0)、★ が中央値。斜めに伸びるときは、差 E_v − E_h だけがよく決まる）。ハイパーパラメータは常用対数で描く（点線は、事前分布の範囲の端が近いとき）。端に張り付くハイパーパラメータ（線形項の σ0 など）は、その方向にはデータで値が決まらないことを表す（本体の手順5b でも同じ。予測への影響は小さいことが多い）。'''

SEC52 = r'''### 5.2 最終モデル（事後分布の混合）とハイパーパラメータの表

最終モデルは、**事後サンプルを混ぜた予測**（混合分布）。全チェーンから等間隔に `N_PRED_DRAWS` 組を選び、それぞれのハイパーパラメータと係数で予測して、平均と分散を混ぜる（平均 = 予測の平均、分散 = 各予測の分散の平均 ＋ 予測どうしのばらつき）。ハイパーパラメータと係数の決めきれなさが、予測幅に入る。このあとのセル（依存性グラフ・3data の予測検証・学習結果のグラフ・3次元グラフ）の予測は、すべて混合分布。
表と、カーネルの図（特徴スケール・カーネル行列）には、**代表点**（事後サンプルの中で、対数事後確率が最大の点）を使う。表の「中央値・2.5%・97.5%」の列は事後分布の要約（`amp2`〔振幅の 2 乗〕・`lin_c`〔線形項の係数〕・`noise`〔ノイズ〕は分散の値）。

**判定**：ノイズ分散の 95% 区間が、下限（`NOISE_LB`）に届いていない。'''

SEC53 = r'''### 5.3 トレースプロット・特徴スケール・カーネル行列（6月のノートのセル7.5・8）

- **トレースプロット**（`MCMC_トレース.png`。勾配法版の「損失関数の推移」の代わり）：チェーンごとの、対数事後確率（いちばん上）と係数の値の推移。破線の左が調整期間（対数事後確率だけ）。本番で、チェーンどうしが同じ範囲を行き来していれば、収束している。
- **特徴スケール**（代表点）：重要度 = 1 / 長さスケール（大きいほど、その入力の変化に粘度が敏感）。シフトのあるモデルでは、シフトに使った温度の効果は θ が担うので、ここには出ない。長さスケールの事後分布の幅は `事後分布_ハイパーパラメータ.png`。
- **カーネル行列**（代表点）：学習データの点どうしの共分散（ノイズを除く）。並べる順を変えた 2 枚（左：設定温度 → 厚み → 射出速度、右：厚み → 設定温度 → 射出速度）。白線は、最初の量が変わる区切り。'''

USAGE_MCMC = r'''10. **MCMC 版の読み方**（勾配法版との違い）：「5. 学習」は、プロファイル尤度の代わりに MCMC。係数は事後分布の中央値と 95% 区間、支持は 95% 区間（B は 95% の楕円）で判定する。収束の判定（R-hat・ESS）が ★ のときは、本番の反復数（`MCMC_SAMPLES`・`MCMC_MAX_SAMPLES`）を増やすか、`MCMC_トレース.png` でチェーンが別々の所に止まっていないかを見る。予測（3data の検証・学習結果のグラフ）は混合分布なので、勾配法版より予測幅が広くなることがある（特にデータのない低速側）。'''

# ---------------------------------------------------------------- 新しいコードセル
CELL6 = r'''# セル6：MCMC（適応型 RWM）で、係数とカーネルのハイパーパラメータの事後分布をサンプリングする（目安：PP の全厚みで約 10 分。収束しなければ本番を足す）
t_all = time.time()
MCMC, SPACE, FITS = {}, {}, {}
for mat in MATERIALS:
    for key in KEYS:
        sp = SPACE[(mat, key)] = mcmc_space(key, TRAIN[mat], kcfg(mat, key))
        print(f"{MAT_NAMES[mat]} {key}：{len(TRAIN[mat])} 行、パラメータ {len(sp['names'])} 個（{'、'.join(sp['names'])}）。"
              f"{MCMC_CHAINS} チェーン × 調整 {MCMC_WARM} 回 ＋ 本番 {MCMC_SAMPLES} 回", flush=True)
        res = MCMC[(mat, key)] = run_mcmc(sp, TRAIN[mat], SEED)
        FITS[(mat, key)] = mixture_fit(sp, res)
        dg = res["diag"]
        print(f"  {res['seconds'] / 60:.1f} 分。本番 {res['n_samples']} 回（足した回数 {res['n_extend']}）。R-hat 最大 {dg['r_hat'].max():.3f}、"
              f"ESS(bulk) 最小 {dg['ess_bulk'].min():.0f}、採択率（本番）{res['acc_samp'].min():.2f}〜{res['acc_samp'].max():.2f}")
        flat = res["draws"].reshape(-1, len(sp["names"]))
        keep = np.linspace(0, len(flat) - 1, min(MCMC_SAVE_ROWS, len(flat))).astype(int)
        n_it = res["draws"].shape[1]
        samp = pd.DataFrame(flat[keep], columns=sp["names"]).assign(chain=keep // n_it + 1, draw=keep % n_it + 1, lp=res["lp"].reshape(-1)[keep])
        to_csv(samp, os.path.join(out_dir(mat, key), "MCMC_事後サンプル.csv"))
        diag = dg.rename_axis("パラメータ").reset_index()
        diag["チェーン数"], diag["調整の反復数"], diag["本番の反復数"] = MCMC_CHAINS, MCMC_WARM, res["n_samples"]
        diag["採択率（本番）の最小"], diag["採択率（本番）の最大"], diag["時間 [分]"] = res["acc_samp"].min(), res["acc_samp"].max(), res["seconds"] / 60
        to_csv(diag, os.path.join(out_dir(mat, key), "MCMC_収束の診断.csv"))
        judge(res["converged"], f"{key}：MCMC が収束した（R-hat 最大 {dg['r_hat'].max():.3f}〔基準 {MCMC_RHAT} 以下〕、ESS(bulk) 最小 {dg['ess_bulk'].min():.0f}"
              f"〔{MCMC_ESS} 以上〕。本番 {res['n_samples']} 回）。収束しなければ、MCMC_MAX_SAMPLES を増やすか、MCMC_トレース.png でチェーンを見る", mat, "MCMC")
        judge(0.15 <= res["acc_samp"].min() and res["acc_samp"].max() <= 0.40,
              f"{key}：採択率（本番）が 0.15〜0.40 の中（{res['acc_samp'].min():.2f}〜{res['acc_samp'].max():.2f}。目標 {MCMC_TARGET_ACC}）", mat, "MCMC")
print(f"合計 {(time.time() - t_all) / 60:.1f} 分")'''

CELL6B = r'''# セル6b：事後分布の要約と、係数の支持
BEST, POST_THETA, PSUM = {}, {}, {}
for mat in MATERIALS:
    rows = []
    for key in KEYS:
        sp, res = SPACE[(mat, key)], MCMC[(mat, key)]
        s = post_summary(sp, res)
        PSUM[(mat, key)] = s["table"]
        to_csv(s["table"], os.path.join(out_dir(mat, key), "事後分布の要約.csv"))
        edge = [n for n, e in zip(s["table"]["パラメータ"], s["table"]["範囲の端に届く"]) if e and n.startswith("ln_")]
        if edge:
            print(f"（参考）{MAT_NAMES[mat]} {key}：95% 区間が事前分布の範囲の端に届いたハイパーパラメータ：{'、'.join(edge)}（その方向にはデータで決まらない）")
        if key not in SHIFT_KEYS:
            continue
        thn, tvn, un = theta_names(kcfg(mat, key))
        flat = res["draws"].reshape(-1, len(sp["names"]))
        sel = np.linspace(0, len(flat) - 1, min(2000, len(flat))).astype(int)
        POST_THETA[(mat, key)] = {"th": flat[sel, sp["nk"]] if key.startswith("B") else np.zeros(len(sel)), "tv": flat[sel, -1]}   # 係数の事後サンプル（シフトファクターの図の帯）
        BEST[(mat, key)] = b = {k: v for k, v in s.items() if k != "table"}
        b["LML_best"] = FITS[(mat, key)]["rep_lp"]               # 代表点（事後サンプルの中で対数事後確率が最大の点）の LML
        rows.append({"材料": MAT_NAMES[mat], "モデル": key, "入力": VERSION_DIRS[INPUT_MODE], "型": FORM_JA[kcfg(mat, key)["type"]],
                     "θ の名前": f"{thn}・{tvn}", "単位": un, **b})
        span = (f"{thn} {b['th_lo']:.3g}〜{b['th_hi']:.3g}・" if key.startswith("B") else "") + f"{tvn} {b['tv_lo']:.3g}〜{b['tv_hi']:.3g} {un}"
        print(f"{MAT_NAMES[mat]} {key}：事後分布の中央値 " + (f"{thn} {b['th_best']:.3g}・" if key.startswith("B") else "") + f"{tvn} {b['tv_best']:.3g} {un}"
              f"、95% 区間 {span}、P({tvn} > 0) = {b['P_tv_pos']:.3f}" + (f"、差 {tvn} − {thn} {b['diff_best']:.3g}〔{b['diff_lo']:.3g}〜{b['diff_hi']:.3g}〕、(0, 0) のマハラノビス距離² {b['d2_zero']:.1f}" if key.startswith("B") else ""))
        judge(b["supported"], f"{key}：温度のシフトが事後分布で支持される（{b['support_rule']}。95% 区間 {span}）。支持されなければ、温度の効果はほかの入力が担っている", mat, "MCMC")
        judge(not b["theta_at_edge"], f"{key}：係数の 95% 区間が、事前分布の範囲（SHIFT_PRIOR）の端に届いていない。届いたら SHIFT_PRIOR を広げる", mat, "MCMC")
    if rows:
        t = pd.DataFrame(rows)
        to_csv(t, os.path.join(out_dir(mat), "シフトファクターの推定.csv"))
        display(t[["材料", "モデル", "型", "θ の名前", "単位", "th_best", "th_lo", "th_hi", "tv_best", "tv_lo", "tv_hi", "diff_best", "diff_lo", "diff_hi",
                   "P_tv_pos", "d2_zero", "supported", "theta_at_edge", "LML_best"]].round(3))'''

CELL7 = r'''# セル7：事後分布の図（係数とハイパーパラメータ）
for mat in MATERIALS:
    for key in KEYS:
        sp, res = SPACE[(mat, key)], MCMC[(mat, key)]
        flat = res["draws"].reshape(-1, len(sp["names"]))
        if key in SHIFT_KEYS:
            thn, tvn, un = theta_names(kcfg(mat, key))
            b = BEST[(mat, key)]
            pairs = [("tv", tvn, flat[:, -1])] if key.startswith("A") else [("th", thn, flat[:, sp["nk"]]), ("tv", tvn, flat[:, -1])]
            fig, axes = plt.subplots(1, len(pairs) + (1 if key.startswith("B") else 0), figsize=(6.5 * (len(pairs) + (1 if key.startswith("B") else 0)), 5.5), squeeze=False)
            axes = list(axes[0])
            if key.startswith("B"):
                ax = axes.pop(0)
                sel = np.linspace(0, len(flat) - 1, min(5000, len(flat))).astype(int)
                ax.scatter(flat[sel, sp["nk"]], flat[sel, -1], s=4, alpha=0.3, color="#1f77b4")
                ax.plot(0, 0, marker="x", color="k", ms=12, mew=2.5, ls="none", label="(0, 0)（シフトなし）")
                ax.plot(b["th_best"], b["tv_best"], marker="*", color="red", ms=16, mec="k", ls="none", label="中央値")
                ax.set_xlabel(f"横のシフト {thn} ({un})")
                ax.set_ylabel(f"縦のシフト {tvn} ({un})")
                ax.set_title(f"(0, 0) のマハラノビス距離² {b['d2_zero']:.1f}（5.99 を超えれば支持）")
                ax.legend(loc="best", fontsize=10)
                ax.grid(True, ls=":", alpha=0.5)
            for ax, (w, nm, v) in zip(axes, pairs):
                ax.hist(v, bins=60, color="#1f77b4", alpha=0.8)
                for q, ls in [(b[f"{w}_lo"], "--"), (b[f"{w}_best"], "-"), (b[f"{w}_hi"], "--")]:
                    ax.axvline(q, color="red", ls=ls, lw=1.8)
                ax.axvline(0, color="k", lw=1)
                ax.set_xlabel(f"{'横' if w == 'th' else '縦'}のシフト {nm} ({un})")
                ax.set_ylabel("サンプルの数")
                ax.set_title(f"中央値 {b[f'{w}_best']:.3g}、95% 区間 {b[f'{w}_lo']:.3g}〜{b[f'{w}_hi']:.3g}")
                ax.grid(True, ls=":", alpha=0.5)
            fig.suptitle(f"{MAT_NAMES[mat]}　{model_title(key)}　係数の事後分布（{FORM_JA[kcfg(mat, key)['type']]}。全チェーン {len(flat)} サンプル。"
                         f"赤の実線：中央値、破線：95% 区間、黒：0）")
            save_fig(fig, out_dir(mat, key), "事後分布_係数", show=SHOW_FIGS and key == MAIN_KEY)
        nk, ncol = sp["nk"], 4
        nrow = int(np.ceil(nk / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.8 * nrow), squeeze=False)
        for i, ax in enumerate(axes.flat):
            if i >= nk:
                ax.axis("off")
                continue
            v = flat[:, i] / np.log(10)                         # 常用対数に直す
            ax.hist(v, bins=60, color="#2ca02c", alpha=0.8)
            w = v.max() - v.min() + 1e-9
            for lim in (sp["lo"][i] / np.log(10), sp["hi"][i] / np.log(10)):
                if v.min() - 0.3 * w <= lim <= v.max() + 0.3 * w:   # 事前分布の範囲の端が近いときだけ描く
                    ax.axvline(lim, color="k", ls=":", lw=1.5)
            ax.set_title(sp["names"][i].replace("ln_", "log10 "))
            ax.grid(True, ls=":", alpha=0.5)
        fig.suptitle(f"{MAT_NAMES[mat]}　{model_title(key)}　カーネルのハイパーパラメータの事後分布（常用対数。amp2・lin_c・noise は分散。点線：事前分布の範囲の端）")
        save_fig(fig, out_dir(mat, key), "事後分布_ハイパーパラメータ", show=SHOW_FIGS and key == MAIN_KEY)'''

CELL8 = r'''# セル8：最終モデル（代表点）のハイパーパラメータと、事後分布の中央値・95% 区間の表
KROWS = []
for mat in MATERIALS:
    for key in KEYS:
        f = FITS[(mat, key)]
        r = {"材料": MAT_NAMES[mat], **kernel_row(f)}
        tab = PSUM[(mat, key)].set_index("パラメータ")
        for nm in [n for n in tab.index if n.startswith("ln_")]:
            r[f"{nm[3:]} 中央値"], r[f"{nm[3:]} 2.5%"], r[f"{nm[3:]} 97.5%"] = tab.loc[nm, "中央値（元の値）"], tab.loc[nm, "2.5%（元の値）"], tab.loc[nm, "97.5%（元の値）"]
        KROWS.append(r)
        print(f"{MAT_NAMES[mat]} {key}（代表点。予測は {len(f['draws'])} 組の混合分布）：{f['gp'].kernel_}")
KT = pd.DataFrame(KROWS)
display(KT.round(4))
for mat in MATERIALS:
    to_csv(KT[KT["材料"] == MAT_NAMES[mat]], os.path.join(out_dir(mat), "ハイパーパラメータ.csv"))
    for key in KEYS:
        nz = PSUM[(mat, key)].set_index("パラメータ").loc["ln_noise"]
        judge(nz["2.5%（元の値）"] > NOISE_LB * 1.05, f"{key}：ノイズ分散の 95% 区間（{nz['2.5%（元の値）']:.2e}〜{nz['97.5%（元の値）']:.2e}）が、下限 {NOISE_LB:g} に届いていない", mat, "学習")'''

CELL9 = r'''# セル9：トレースプロット（チェーンごとの、対数事後確率と係数の推移。勾配法版の「損失関数の推移」の代わり）
for mat in MATERIALS:
    for key in KEYS:
        sp, res = SPACE[(mat, key)], MCMC[(mat, key)]
        show = ["lp"] + sp["names"][sp["nk"]:]
        fig, axes = plt.subplots(len(show), 1, figsize=(12, 3.2 * len(show)), squeeze=False)
        n_w, n_c = res["warm_lp"].shape[1], res["draws"].shape[0]
        for ax, nm in zip(axes[:, 0], show):
            for c in range(n_c):
                if nm == "lp":
                    y = np.r_[res["warm_lp"][c], res["lp"][c]]   # 調整期間も描く
                    ax.plot(np.arange(len(y)), y, lw=0.5, alpha=0.8, label=f"チェーン {c + 1}")
                else:
                    ax.plot(np.arange(n_w, n_w + res["draws"].shape[1]), res["draws"][c, :, sp["names"].index(nm)], lw=0.5, alpha=0.8)
            ax.axvline(n_w, color="k", ls="--", lw=1)
            ax.set_ylabel("対数事後確率" if nm == "lp" else nm)
            ax.grid(True, ls=":", alpha=0.5)
        lp = res["lp"]
        axes[0, 0].set_ylim(np.percentile(lp, 0.5) - 0.5 * (lp.max() - np.percentile(lp, 0.5)) - 1, lp.max() + 2)
        axes[0, 0].legend(loc="lower right", fontsize=9, ncol=n_c)
        axes[-1, 0].set_xlabel("反復の回数（破線の左が調整期間）")
        fig.suptitle(f"{MAT_NAMES[mat]}　{model_title(key)}　トレースプロット（{n_c} チェーン。本番 {res['n_samples']} 回）")
        save_fig(fig, out_dir(mat, key), "MCMC_トレース", show=SHOW_FIGS and key == MAIN_KEY)'''

MCMC_SETTINGS = r'''MCMC_CHAINS = 4               # MCMC のチェーンの数（並列に回す。N_JOBS 以下にする）
MCMC_WARM = 5000              # 調整期間の反復数（提案の分布の大きさと形を合わせる。サンプルには使わない）
MCMC_SAMPLES = 20000          # 本番の反復数（1 チェーンあたり。本体の手順5b と同じ）。時間を短くするときは減らす（収束の判定が ★ になりやすい）
MCMC_MAX_SAMPLES = 100000     # 収束の基準に届かないとき、本番を続きから足す上限（1 チェーンあたり）
MCMC_RHAT = 1.01              # 収束の基準：R-hat（チェーンどうしの食い違い）がこれ以下
MCMC_ESS = 400                # 収束の基準：有効サンプル数（ESS bulk）がこれ以上
MCMC_TARGET_ACC = 0.234       # 調整期間に合わせる、目標の採択率
MCMC_INIT = "mle"             # 初期値："mle"＝係数 0 で最尤推定（L-BFGS-B）したカーネルに、チェーンごとのずれを足した点から（勾配法は出発点を決めるだけに使う）／"default"＝カーネルの既定値から。係数はどちらも 0 の近くから
MCMC_INIT_RESTARTS = 5        # "mle" の最尤推定のやり直し回数
SHIFT_PRIOR = {               # 係数の事前分布（一様分布）の範囲（下限, 上限）。h が横、v が縦。既定は勾配法版の格子の範囲と同じ。事後分布が端に届いたら（★）広げる
    "arrhenius": {"h": (-20.0, 40.0), "v": (-20.0, 60.0)},   # E [kJ/mol]
    "wlf":       {"h": (-20.0, 40.0), "v": (-20.0, 40.0)},   # A1 [−]
}
N_PRED_DRAWS = 200            # 予測に使う事後サンプルの数（全チェーンから等間隔に選ぶ。予測は、これらの予測を混ぜた分布〔混合分布〕）
REQUIRE_MCMC_FOLDER = True    # True：学習結果フォルダの名前に「MCMC」がなければ止める（勾配法版の結果の上書きを防ぐ）'''


def build(out, material=None):
    if os.path.exists(out):
        sys.exit(f"既にファイルがあるため上書きしません: {out}")
    work = os.path.join(REPO, "_tmp", "build_mat_shift_mcmc_nb", time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")   # 同時に動かしても重ならないように
    os.makedirs(work, exist_ok=False)
    base = os.path.join(work, "base.ipynb")
    r = subprocess.run([sys.executable, os.path.join(HERE, "build_mat_shift_nb.py"), base], capture_output=True)
    if r.returncode != 0 or not os.path.exists(base):
        sys.exit(f"勾配法版のノートを作れなかった：{r.stderr.decode('utf-8', 'replace')}")
    nb = json.load(open(base, encoding="utf-8"))
    cells = nb["cells"]

    # ---- 題と使い方
    c = cells[0]
    rep(c, "# 材料別の温度シフト GPR（ARD Matern 5/2 ＋ 線形 ＋ ホワイトノイズ）：学習用\n", TITLE + "\n\n" + INTRO_MCMC + "\n", "題")
    rep_between(c, "**推定**：θ はプロファイル尤度で決める", "**読むときの注意**", EST_MCMC, "題の推定の段落")
    for cc in cells:                                          # ノートの場所が 2 段深くなった（02_材料別_温度シフト機械学習コード/02_MCMC(適応型RWM)/）
        if cc["cell_type"] == "markdown":
            set_src(cc, src(cc).replace("../材料別_温度シフトファクター検討用コード/", "../../01_材料別_温度シフトファクター検討用コード/"))
    c = cells[find(cells, "## 使い方", "markdown")]
    rep(c, "1 つ上のフォルダ（`材料別_温度シフトファクター検討/`）", "2 つ上のフォルダ（`材料別_温度シフトファクター検討/`）", "説明書の場所")
    rep(c, "（時間を短くする：`PROFILE_RESTARTS = 0` など）", "（時間を短くする：`MCMC_SAMPLES` を減らすなど。収束の判定が ★ になりやすい）", "時間を短くする")
    rep(c, "**材料ごとに別のフォルダ**を選ぶ）", "**材料ごとに別のフォルダ**を選ぶ。MCMC 版では、**名前に「MCMC」を含むフォルダ**〔例：最新の学習結果フォルダの隣の "
        "`4回目_01_バージンPP再々実験_学習結果_MCMC`〕を選ぶ。名前に MCMC がないと止まる〔勾配法版の結果の上書きを防ぐ。設定 `REQUIRE_MCMC_FOLDER`〕）", "結果のフォルダ")
    rep_between(c, "（目安：3 材料で、ショットごとの入力は約 35 分", "\n   - ダイアログが見当たらないときは",
                "（目安：MCMC は、PP の全厚み〔460 行〕で 1 チェーン 2.5 万回に約 10 分〔4 チェーンを並列〕。収束の基準に届かないと本番を足すので、最大で約 4 倍。"
                "厚みごとは行が少ないので速い。予測を混合分布で出すので、図のセルも勾配法版より時間がかかる。要確認）。", "時間の目安")
    set_src(c, src(c).rstrip("\n") + "\n" + USAGE_MCMC)

    c = cells[find(cells, "## 4b. 学習する厚み", "markdown")]
    rep(c, "- 厚みごとの比較のノート（6 節・9 節）の「厚みごとのモデル」と同じ考え方で、このノートでは、95% 区間の延長と、最良の θ・θ = 0 の念入りな最適化も行う。",
        "- MCMC 版では、選んだ厚みのデータで、全厚みのときと同じように MCMC でサンプリングする（行が少ないので速い）。", "4b 節")

    # ---- 設定
    c = cells[find(cells, "# セル1：設定エリア")]
    for head in ("N_RESTARTS = ", "PROFILE_RESTARTS = ", "FINE_B = ", "FINE_A = ", "MAX_REFINE = ", "MAX_EXTEND = "):
        lines = src(c).split("\n")
        hit = [l for l in lines if l.startswith(head)]
        if len(hit) != 1:
            sys.exit(f"設定の行「{head}」が {len(hit)} 行")
        set_src(c, "\n".join(l for l in lines if not l.startswith(head)))
    rep(c, "FORM_CFG = {                  # 型ごとの、尤度を調べる範囲（粗い格子：(下限, 上限, 刻み)。h が横、v が縦）と、細かい格子 \"fine\"（±幅, 刻み。書いた型だけ、下の FINE_B・FINE_A の代わりに使う）",
        "FORM_CFG = {                  # 型ごとの設定。MCMC 版では、WLF 型の A2 [K]（固定）だけを使う（h_grid・v_grid・fine は勾配法版の格子の設定で、使わない。係数の事前分布の範囲は SHIFT_PRIOR）", "FORM_CFG の説明")
    rep(c, "\nUSE_EXCLUSIONS = ", "\n" + MCMC_SETTINGS + "\nUSE_EXCLUSIONS = ", "MCMC の設定の場所")
    rep(c, "N_JOBS = 8                    # プロファイル尤度を並列に計算する数（CPU の数に合わせる）", "N_JOBS = 8                    # MCMC のチェーンを並列に回す数の上限（CPU の数に合わせる）", "N_JOBS の説明")
    if material:
        rep(c, 'MATERIALS = ["PP", "MIX", "GPPS"]    #', f'MATERIALS = ["{material}"]    #', "MATERIALS")
        if MATS[material] == "A":
            rep(c, 'SHIFT_STRUCT = "B"            #', 'SHIFT_STRUCT = "A"            #', "SHIFT_STRUCT")

    # ---- 共通の部品の後に、MCMC の部品
    i = find(cells, "# セル：共通の部品")
    cells[i + 1:i + 1] = [md_cell("### MCMC の部品\n\n適応型 RWM・収束の診断・事後分布の要約・混合分布の予測（`04_補助スクリプト/mat_mcmc.py` の中身）。共通の部品の `predict`（予測）と `choose_inputs`（ファイルとフォルダを選ぶ）を、MCMC 版に包み直す（事後サンプルがあれば混合分布で予測する／学習結果フォルダの名前に「MCMC」があるかを確かめる）。"),
                          code_cell(MCMC_CELL)]

    # ---- 5. 学習：プロファイル尤度・念入りな最適化・尤度の面・C の学習 → MCMC・要約・事後分布の図・表
    set_src(cells[find(cells, "## 5. 学習", "markdown")], SEC5)
    set_src(cells[find(cells, "# セル6：プロファイル尤度")], CELL6)
    set_src(cells[find(cells, "# セル6b：最良の θ と θ = 0 の念入りな最適化")], CELL6B)
    set_src(cells[find(cells, "**図の見方**（`プロファイル尤度.png`）", "markdown")], FIG_POST)
    set_src(cells[find(cells, "# セル7：尤度の面")], CELL7)
    set_src(cells[find(cells, "### 5.2 最終モデルの学習", "markdown")], SEC52)
    set_src(cells[find(cells, "# セル8：最終モデルの学習")], CELL8)
    set_src(cells[find(cells, "### 5.3 損失関数の推移", "markdown")], SEC53)
    set_src(cells[find(cells, "# セル9：損失関数の推移")], CELL9)

    # ---- 6. シフトファクター：帯を事後サンプルの 2.5〜97.5% に、マスターカーブは中央値の θ で
    c = cells[find(cells, "# セル12：シフトファクターとマスターカーブ")]
    rep(c, "pts, b, cfg = PROF[(mat, key)], BEST[(mat, key)], kcfg(mat, key)", "b, cfg = BEST[(mat, key)], kcfg(mat, key)", "セル12 の PROF")
    rep(c, 'inside = pts[pts["LML"] >= b["LML_best"] - 1.92]', "post = POST_THETA[(mat, key)]                 # 係数の事後サンプル（全サンプルから等間隔に 2000 組）", "セル12 の区間の点")
    rep(c, "cur = np.array([shift_ln(Tg, v, cfg) for v in inside[w]])", "cur = np.array([shift_ln(Tg, v, cfg) for v in post[w]])", "セル12 の帯の線")
    rep(c, "ax.fill_between(Tg, cur.min(axis=0), cur.max(axis=0), color=col, alpha=0.18)",
        "ax.fill_between(Tg, np.percentile(cur, 2.5, axis=0), np.percentile(cur, 97.5, axis=0), color=col, alpha=0.18)", "セル12 の帯")
    rep(c, "帯：尤度の 95% 区間", "帯：事後分布の 95% 区間", "セル12 のタイトル")
    rep(c, "（最良の θ でずらした条件平均）", "（事後分布の中央値の θ でずらした条件平均）", "マスターカーブのタイトル")
    rep(c, """{nm} = {b[f'{w}_best']:g} {un}、95%：{b[f'{w}_lo']:g}〜{b[f'{w}_hi']:g}）""",
        """{nm} 中央値 {b[f'{w}_best']:.3g} {un}、95%：{b[f'{w}_lo']:.3g}〜{b[f'{w}_hi']:.3g}）""", "セル12 の凡例の桁（事後分布の値は格子の値と違い、桁が長いため）")
    rep(c, """{tvn} = {b['tv_best']:g} {un}）""", """{tvn} = {b['tv_best']:.3g} {un}〔中央値〕）""", "セル12 の判定の桁")
    c = cells[find(cells, "## 6. 温度のシフトファクターと、マスターカーブ", "markdown")]
    rep(c, "線が最良の θ、帯が尤度の 95% 区間（赤線の内側の θ の範囲）", "線が事後分布の中央値の θ、帯が事後分布の 95% 区間（係数の事後サンプルで計算したずれの 2.5〜97.5%）", "6 節の帯")
    rep(c, "条件平均の点を、最良の θ でずらしたもの", "条件平均の点を、事後分布の中央値の θ でずらしたもの", "6 節のマスターカーブ")

    # ---- 8. 予測検証・12. 保存・13. まとめの説明
    c = cells[find(cells, "## 8. 3data による予測検証", "markdown")]
    rep(c, "\n**判定**：被覆率が 0.8 以上。", "\n- **MCMC 版の予測**：事後サンプル `N_PRED_DRAWS` 組それぞれで予測し、平均と分散を混ぜた分布（混合分布）で出す。95% 予測区間は、その平均 ± 1.96 × 標準偏差（正規分布で近似）。"
        "ハイパーパラメータと係数の決めきれなさが予測幅に入るので、勾配法版より区間が広くなることがある。\n\n**判定**：被覆率が 0.8 以上。", "8 節の判定")
    c = cells[find(cells, "## 12. 学習済みモデルの保存", "markdown")]
    rep(c, "（このノートの「共通の部品」のセルを実行した後で。ノートの外で使うときは、先に `import sys; sys.path.insert(0, \"04_補助スクリプト\"); from mat_common import *` で、同じ部品を読み込む）",
        "（このノートの「共通の部品」と「MCMC の部品」のセルを実行した後で。ノートの外で使うときは、先に `import sys; sys.path.insert(0, \"04_補助スクリプト\"); from mat_mcmc import *` で、同じ部品を読み込む。"
        "mat_mcmc は mat_common も読み込む）。MCMC 版の学習済みモデルには、代表点の GPR のほかに、予測に使う事後サンプル（`draws`）と、サンプルごとの GPR を作り直す材料（`mix_base`）が入っていて、"
        "MCMC の部品の `predict` は混合分布で予測する（共通の部品だけの `predict` では、代表点の 1 点の予測になる）", "12 節の使い方")
    c = cells[find(cells, "## 13. まとめ", "markdown")]
    rep(c, "（θ と 95% 区間、3data の RMSE・被覆率。参考の Cross-WLF も）", "（θ の事後分布の中央値と 95% 区間、支持の判定、MCMC の収束、3data の RMSE・被覆率。参考の Cross-WLF も）", "13 節")
    c = cells[find(cells, "# セル19：まとめ")]
    rep(c, '''            row.update({"θ": f"{thn} {b['th_best']:g}［{b['th_lo']:g}〜{b['th_hi']:g}］・{tvn} {b['tv_best']:g}［{b['tv_lo']:g}〜{b['tv_hi']:g}］{un}", "LML": round(b["LML_best"], 1),
                        "θ=0 との LML の差": round(b["dLML_vs_0"], 2), "支持の基準": lrt_limit(key), "尤度で支持": bool(b["dLML_vs_0"] >= lrt_limit(key))})''',
        '''            row.update({"θ": (f"{thn} {b['th_best']:.3g}［{b['th_lo']:.3g}〜{b['th_hi']:.3g}］・" if key.startswith("B") else "") + f"{tvn} {b['tv_best']:.3g}［{b['tv_lo']:.3g}〜{b['tv_hi']:.3g}］{un}（事後分布の中央値［95% 区間］）",
                        "代表点の LML": round(b["LML_best"], 1), "支持の判定の方法": b["support_rule"], "事後分布で支持": bool(b["supported"]), "P(θ_v>0)": round(b["P_tv_pos"], 3)})''', "まとめの θ の列")
    rep(c, '''            row["LML"] = round(float(KT[(KT["材料"] == MAT_NAMES[mat]) & (KT["model"] == "C")]["LML"].iloc[0]), 1)''',
        '''            row["代表点の LML"] = round(float(KT[(KT["材料"] == MAT_NAMES[mat]) & (KT["model"] == "C")]["LML"].iloc[0]), 1)''', "まとめの C の LML")
    rep(c, '''        t = MT_ALL[(MT_ALL["材料"] == MAT_NAMES[mat]) & (MT_ALL["モデル"] == key)].set_index("範囲")''',
        '''        if (mat, key) in MCMC:
            row.update({"MCMC の収束": bool(MCMC[(mat, key)]["converged"]), "本番の反復数": MCMC[(mat, key)]["n_samples"]})
        t = MT_ALL[(MT_ALL["材料"] == MAT_NAMES[mat]) & (MT_ALL["モデル"] == key)].set_index("範囲")''', "まとめの収束の列")

    # ---- 確かめ：勾配法の格子・最適化の名前が、コード（共通の部品と MCMC の部品を除く）に残っていない
    for k, cc in enumerate(cells):
        s = src(cc)
        if cc["cell_type"] != "code" or s.startswith("# セル：共通の部品") or s.startswith("# セル：MCMC の部品"):
            continue
        for w in ("PROF[", "ZERO[", "LOSS[", "lrt_limit(", "N_RESTARTS", "PROFILE_RESTARTS", "FINE_A", "FINE_B", "MAX_REFINE", "MAX_EXTEND", "run_profile", "fit_job"):
            if w in s:
                sys.exit(f"コードセル {k} に、勾配法版の名前「{w}」が残っている")
    if material:
        mat_line = [l for l in src(cells[find(cells, "# セル1：設定エリア")]).split("\n") if l.startswith("MATERIALS")]
        assert mat_line and mat_line[0].startswith(f'MATERIALS = ["{material}"]'), mat_line
    left = [(k, w) for k, cc in enumerate(cells) if cc["cell_type"] == "markdown" for w in ("プロファイル尤度", "L-BFGS-B", "損失関数", "最良の θ") if w in src(cc)]
    print("（確かめ）マークダウンに残る勾配法の言葉（説明として残してよいか見る）：", left)

    # ---- 書き出し
    rnd = random.Random(20261117)
    for cc in cells:
        cc["id"] = "%08x" % rnd.getrandbits(32)
        if cc["cell_type"] == "code":
            cc["outputs"], cc["execution_count"] = [], None
            compile("".join(l for l in cc["source"] if not l.startswith("%")), "<cell>", "exec")
    assert len({cc["id"] for cc in cells}) == len(cells)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as f:
        f.write((json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"作成: {out}（{len(cells)} セル。元の勾配法版のノート：{os.path.relpath(base, REPO)}）")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if not args or args[0].startswith("--"):
        sys.exit("1 つ目の引数には、出力する .ipynb のパスを書く（例：uv run python 04_補助スクリプト/build_mat_shift_mcmc_nb.py <出力.ipynb> --material PP）")
    mat = None
    if "--material" in args:
        mat = args[args.index("--material") + 1]
        if mat not in MATS:
            sys.exit(f"--material は {list(MATS)} のどれか（指定：{mat}）")
    build(args[0], mat)
