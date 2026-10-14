# -*- coding: utf-8 -*-
"""材料別の温度シフト GPR（05_材料ごとの機械学習データと予測検証データ）のノートで共通に使う部品。

ノート（01_機械学習コードフォルダ/材料別_温度シフトGPR_ARD対応Matern52＋線形＋ホワイトノイズ.ipynb）の「共通の部品」のセルに、
この中身がそのまま入る（ノートは単独で動く）。直すときは、このファイルを直して、ノートを作り直す（04_補助スクリプト/build_mat_shift_nb.py）。
本体の検討（nb_common.py・step5_core.py）とは別のプロジェクト。部品の多くは、それらを写して、このデータ
（説明変数 γ̇・P3_3・Tff・Tset。H は入れない）に合わせて直したもの。モジュールとして import しても使える。
設定値（SEED など）は、ノートの「設定エリア」で先に決めた値があれば、それを使う。
"""
import glob
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, DotProduct, Matern, Sum, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- 設定（ノートの設定エリアの値があれば、それを使う）
SEED = globals().get("SEED", 42)
NOISE_LB = globals().get("NOISE_LB", 1e-6)                  # ノイズ分散の下限（6月のノートは 1e-3）
N_RESTARTS = globals().get("N_RESTARTS", 20)                # 最終モデルの学習での、最適化のやり直し回数（ハイパーパラメータの局所解が多いので、6月の 5 回より多くする）
PROFILE_RESTARTS = globals().get("PROFILE_RESTARTS", 1)     # プロファイル尤度の格子の各点での、やり直し回数（前の点の解も初期値に使う）
USE_LINEAR = globals().get("USE_LINEAR", True)              # True：6月のカーネルに線形項（DotProduct）を足す
USE_EXCLUSIONS = globals().get("USE_EXCLUSIONS", True)      # 問題データ一覧の「除外」を使うか
REPO = globals().get("REPO", os.environ.get("NB_REPO", ".."))
N_JOBS = globals().get("N_JOBS", 8)
FIG_DPI = globals().get("FIG_DPI", 200)
HR_ALL = globals().get("HR_ALL", True)                      # すべての図を、600 dpi でも highres/ に保存するか
HR_ALL_DPI = globals().get("HR_ALL_DPI", 600)
SHOW_FIGS = globals().get("SHOW_FIGS", True)                # 図をノートに表示するか（False なら保存だけ）
INPUT_MODE = globals().get("INPUT_MODE", "cond")            # "cond"：γ̇・P3_3・Tff を条件（設定温度・厚み・射出速度）ごとの平均で入れる（本体の手順3b の R3 と同じ考え方）／"shot"：ショットごとの値で入れる（6月のノートと同じ）
MAX_REFINE = globals().get("MAX_REFINE", 4)                 # プロファイル尤度で、最良の点が細かい格子の端に当たったとき、その点を中心に細かい格子を取り直す回数の上限
FINE_B = globals().get("FINE_B", (4.0, 1.0))               # 構造 B の細かい格子：最良の点の ±4 を 1 刻み
FINE_A = globals().get("FINE_A", (2.5, 0.5))                # 構造 A の細かい格子：最良の点の ±2.5 を 0.5 刻み
OUTLIER_Z = globals().get("OUTLIER_Z", 3.5)                 # 規則 f：|z| がこれを超え、かつ |残差|（ln η）が次の値を超える条件
OUTLIER_RESID = globals().get("OUTLIER_RESID", 0.1)
P33_DEV = globals().get("P33_DEV", 0.2)                     # 規則 g：P3_3 が条件の中央値から、この割合以上離れたショットを候補にする
warnings.filterwarnings("ignore", category=ConvergenceWarning)   # 下限・上限への張り付きは、ハイパーパラメータの表に記録する

# ---------------------------------------------------------------- データの場所と列
DATA_ROOT = os.path.join(REPO, "05_材料ごとの機械学習データと予測検証データ")
PROBLEM_CSV = os.path.join(DATA_ROOT, "問題データ一覧.csv")
MAT_FOLDERS = globals().get("MAT_FOLDERS", {"PP": "01_バージンPP再々実験", "MIX": "02_粉砕材混合PP実験", "GPPS": "03_GPPS実験"})
MAT_NAMES = {"PP": "PP", "MIX": "粉砕材混合PP", "GPPS": "GPPS"}     # 問題データ一覧の「材料」列の名前
GROUP_DIRS = {"5data": "学習データ_*", "3data": "検証データ_*"}      # 材料フォルダの中の、学習データと検証データのフォルダ（CSV が 1 つずつ）
COLS = {"元ファイル": "file", "キャビ厚": "H", "射出速度[mm/s]": "v", "樹脂粘度η34(Pa・s)": "eta", "せん断速度γ34(1/s)": "gd",
        "流動中樹脂圧力P3_1(MPa)": "P31", "流動中樹脂圧力P3_2(MPa)": "P32", "流動中樹脂圧力P3_3(MPa)": "P33",
        "加熱シリンダ温度[℃]": "Tset", "加熱シリンダ温度(℃)": "Tset", "流動中樹脂温度(℃)": "Tff"}
REQUIRED = ["file", "H", "v", "eta", "gd", "P33", "Tff"]          # 加熱シリンダ温度の列がなければ、元ファイル名から取る
RE_SRC = r"^(\d{3})_(\d+)℃_(\d{3})_([\d.]+)mm[A-Za-z]*\.csv$"     # ショット_設定温度_射出速度_厚み（GPPS は末尾に NR が付く）
R_GAS = 8.314                                                      # 気体定数 [J/mol/K]
D = {}             # (材料, 組) → DataFrame。組は "5data"（学習）と "3data"（検証）
FILE_INFO = {}     # (材料, 組) → 読んだファイルの情報
REMOVED = {}       # (材料, 組) → 一覧の「除外」で除いた行（元ファイルと ID）
INVALID = {}       # (材料, 組) → 0 以下の値（η・γ̇・P3_3）で対数がとれず、一覧に登録がないまま除いた行
JUDGES = []        # 判定の記録（まとめの表に使う）


MODE_JA = {"shot": "ショットごとの入力", "cond": "条件平均の入力"}         # 入力の取り方（INPUT_MODE）の呼び名。結果のサブフォルダの名前にも使う


def result_dir(mat, sub=""):
    """結果の保存先（学習データ・検証データのフォルダと同じ場所の <材料フォルダ>_学習結果/）。
    データの確認（sub = "データ確認"）は入力の取り方によらないので直下、モデルと比較の結果は、入力の取り方ごとのサブフォルダ（MODE_JA）の下"""
    folder = MAT_FOLDERS[mat]
    base = os.path.join(DATA_ROOT, folder, f"{folder}_学習結果")
    d = os.path.join(base, sub) if sub in ("", "データ確認") else os.path.join(base, MODE_JA[INPUT_MODE], sub)
    os.makedirs(d, exist_ok=True)
    return d


def path_of(mat, grp):
    fs = sorted(glob.glob(os.path.join(DATA_ROOT, MAT_FOLDERS[mat], GROUP_DIRS[grp], "*.csv")))
    assert len(fs) == 1, f"{MAT_FOLDERS[mat]}/{GROUP_DIRS[grp]} に CSV が 1 つだけあるはず（見つかったもの: {fs}）"
    return fs[0]


# ---------------------------------------------------------------- 問題データ一覧
def read_problem_list(path=None):
    """問題データ一覧.csv（05 フォルダ）を読む（空欄は空文字）"""
    return pd.read_csv(path or PROBLEM_CSV, encoding="utf-8-sig", dtype=str).fillna("")


def problem_mask(df, mat, grp, plist, kinds):
    """区分が kinds のどれかで、再測定が「済み」でない一覧の行のうち、この材料・組・条件に当てはまる行の真偽値と、その ID を返す。
    「ショット」列は、空欄なら条件全体、数字ならそのショット（元ファイル名の先頭 3 桁）。"2;3" のように複数書ける"""
    mask = np.zeros(len(df), bool)
    ids = [[] for _ in range(len(df))]
    for _, r in plist.iterrows():
        if r["区分"] not in kinds or r["再測定"] == "済み" or r["データ"] not in ("新", ""):
            continue
        if r["材料"] not in ("すべて", "", MAT_NAMES[mat]) or r["組"] not in ("", grp):
            continue
        m = np.ones(len(df), bool)
        if r["設定温度"] != "":
            m &= df["Tset"].to_numpy() == float(r["設定温度"])
        if r["厚み"] != "":
            m &= np.isclose(df["H"].to_numpy(), float(r["厚み"]))
        if r["射出速度"] != "":
            m &= df["v"].to_numpy() == float(r["射出速度"])
        if r.get("ショット", "") != "":
            shots = [int(s) for s in re.split(r"[;；,、]", r["ショット"]) if s.strip()]
            m &= np.isin(df["shot"].to_numpy(), shots) if "shot" in df.columns else np.zeros(len(df), bool)
        mask |= m
        for i in np.flatnonzero(m):
            ids[i].append(r["ID"])
    return mask, [";".join(x) for x in ids]


# ---------------------------------------------------------------- 読み込み
def load(mat, grp, plist=None, exclude=None):
    """結合ファイルを読み、形式を確かめ、問題データ一覧の「除外」を除く。
    加熱シリンダ温度の列がないファイル（PP の 3data）は、元ファイル名から設定温度を取る。列があれば、ファイル名と一致するかを確かめる。
    η・γ̇・P3_3 が 0 以下の行は対数がとれないので、一覧に登録がなくても除き、INVALID に記録する（判定で ★）。
    列 sens には「感度確認」の ID が入る（なければ空文字）。圧力は Pa に直してから ln をとる。Tff の ln は K で"""
    exclude = USE_EXCLUSIONS if exclude is None else exclude
    path = path_of(mat, grp)
    raw = pd.read_csv(path, encoding="utf-8-sig")
    unknown = [c for c in raw.columns if c not in COLS]
    assert not unknown, f"想定にない列がある: {unknown}（mat_common.py の COLS に足す）"
    df = raw.rename(columns=COLS)
    missing = [c for c in REQUIRED if c not in df.columns]
    assert not missing, f"必要な列がない: {missing}（{path}）"
    assert df[REQUIRED].isna().sum().sum() == 0, f"欠損がある（{path}）"
    p = df["file"].str.extract(RE_SRC)
    assert p.notna().all().all(), f"元ファイル名の形式に合わない行がある（{path}）"
    assert np.allclose(p[2].astype(float), df["v"]) and np.allclose(p[3].astype(float), df["H"]), "元ファイル名と、厚み・射出速度の列が一致しない"
    has_tset = "Tset" in df.columns
    if has_tset:
        assert np.allclose(p[1].astype(float), df["Tset"].astype(float)), "元ファイル名と、加熱シリンダ温度の列が一致しない"
    else:
        df["Tset"] = p[1].astype(float)
    df["Tset"] = df["Tset"].astype(float)
    df["shot"] = p[0].astype(int)
    plist = read_problem_list() if plist is None else plist
    n_raw = len(df)
    if exclude:
        drop, ids = problem_mask(df, mat, grp, plist, ("除外",))
        REMOVED[(mat, grp)] = pd.DataFrame({"元ファイル": df.loc[drop, "file"].to_numpy(), "ID": np.array(ids, dtype=object)[drop]})
        df = df[~drop].reset_index(drop=True)
    else:
        REMOVED[(mat, grp)] = pd.DataFrame(columns=["元ファイル", "ID"])
    bad = (df[["eta", "gd", "P33"]] <= 0).any(axis=1).to_numpy()
    INVALID[(mat, grp)] = df.loc[bad, ["file", "Tset", "H", "v", "shot", "eta", "gd", "P33"]].reset_index(drop=True)
    df = df[~bad].reset_index(drop=True)
    _, df["sens"] = problem_mask(df, mat, grp, plist, ("感度確認",))
    df["is_sens"] = df["sens"] != ""
    df["cond"] = df["Tset"].astype(int).astype(str) + "_" + df["H"].astype(str) + "_" + df["v"].astype(int).astype(str)
    df["ln_eta"], df["ln_gd"], df["ln_H"], df["ln_v"] = np.log(df["eta"]), np.log(df["gd"]), np.log(df["H"]), np.log(df["v"])
    df["ln_P33"] = np.log(df["P33"] * 1e6)                 # CSV の圧力は MPa。Pa に直してから ln をとる
    df["ln_Tff"] = np.log(df["Tff"] + 273.15)               # 流動中樹脂温度 [K] の ln（6月のノートと同じ）
    for c in ["gd", "P33", "Tff"]:                          # 条件ごとの平均（INPUT_MODE = "cond" の入力、図の予測線、外れ値の検知に使う）
        df[f"ln_{c}_cm"] = df.groupby("cond")[f"ln_{c}"].transform("mean")
    df["Tff_cm"] = np.exp(df["ln_Tff_cm"]) - 273.15
    df["grp"] = grp
    FILE_INFO[(mat, grp)] = {"材料": MAT_NAMES[mat], "組": grp, "ファイル": os.path.relpath(path, DATA_ROOT), "読んだ行数": n_raw,
                             "一覧で除外": n_raw - len(df) - int(bad.sum()), "0以下で除外（未登録）": int(bad.sum()), "使う行数": len(df),
                             "条件数": df["cond"].nunique(), "ショット/条件": f"{df.groupby('cond').size().min()}〜{df.groupby('cond').size().max()}",
                             "設定温度": "・".join(str(int(t)) for t in sorted(df["Tset"].unique())), "設定温度の取り方": "列" if has_tset else "元ファイル名",
                             "厚み": "・".join(f"{h:g}" for h in sorted(df["H"].unique())), "射出速度": "・".join(str(int(v)) for v in sorted(df["v"].unique())),
                             "感度確認の行": int(df["is_sens"].sum())}
    return df


def load_all(mats, exclude=None):
    """材料ごとに 5data（学習）と 3data（検証）を読む。学習と検証で条件が重ならないことも確かめる"""
    plist = read_problem_list()
    out = {(mat, grp): load(mat, grp, plist, exclude) for mat in mats for grp in ["5data", "3data"]}
    for mat in mats:
        both = set(out[(mat, "5data")]["cond"]) & set(out[(mat, "3data")]["cond"])
        assert not both, f"{MAT_NAMES[mat]}：学習と検証で条件が重なっている: {sorted(both)}"
    return out


def add_groups(tr, te):
    """検証の行を、学習データの射出速度の範囲より遅い側（低速側）・速い側（高速側）・範囲内に分ける（列 group）"""
    te = te.copy()
    vmin, vmax = tr["v"].min(), tr["v"].max()
    te["group"] = np.where(te["v"] < vmin, "低速側", np.where(te["v"] > vmax, "高速側", "範囲内"))
    return te


# ---------------------------------------------------------------- 温度のシフトファクター
DEFAULT_SHIFT_CFG = {     # h_grid・v_grid：尤度を調べる粗い格子（下限, 上限, 刻み）。h が横、v が縦のシフトの係数
    "PP": {"type": "arrhenius", "Tref": 210.0, "h_grid": (-20.0, 40.0, 5.0), "v_grid": (-20.0, 60.0, 5.0)},
    "MIX": {"type": "arrhenius", "Tref": 210.0, "h_grid": (-20.0, 40.0, 5.0), "v_grid": (-20.0, 60.0, 5.0)},
    "GPPS": {"type": "wlf", "Tref": 220.0, "Tg": 100.0, "A2": 51.6, "h_grid": (-20.0, 40.0, 5.0), "v_grid": (-20.0, 40.0, 5.0)},
}
SHIFT_CFG = globals().get("SHIFT_CFG", DEFAULT_SHIFT_CFG)


def shift_ln(T_c, theta, cfg):
    """ln a(T)（T は ℃。Tref で 0）。
    アレニウス型：ln a = (E/R)(1/T − 1/Tref)、theta = E [kJ/mol]。
    WLF 型：ln a = −A1·[(T − D2)/(A2 + T − D2) − (Tref − D2)/(A2 + Tref − D2)]、theta = A1 [−]、D2 = Tg、A2 は固定"""
    T = np.asarray(T_c, float) + 273.15
    Tr = float(cfg["Tref"]) + 273.15
    if cfg["type"] == "arrhenius":
        return float(theta) * 1000.0 / R_GAS * (1.0 / T - 1.0 / Tr)
    D2, A2 = float(cfg["Tg"]) + 273.15, float(cfg["A2"])
    return -float(theta) * ((T - D2) / (A2 + T - D2) - (Tr - D2) / (A2 + Tr - D2))


def theta_names(cfg):
    """(横の名前, 縦の名前, 単位)"""
    return ("E_h", "E_v", "kJ/mol") if cfg["type"] == "arrhenius" else ("A1_h", "A1_v", "−")


# ---------------------------------------------------------------- モデル
MODEL_JA = {"B_Tset": "B_Tset（設定温度 Tset で横・縦にシフト）", "B_Tff": "B_Tff（流動中樹脂温度 Tff で横・縦にシフト）",
            "A_Tset": "A_Tset（Tset で縦だけシフト）", "A_Tff": "A_Tff（Tff で縦だけシフト）",
            "C": "C（シフト構造なし。4 変数を並べた通常の GPR）"}
COL_JA = {"ln_gd": "ln γ̇", "ln_P33": "ln P3_3", "ln_Tff": "ln Tff", "Tset": "Tset"}


def layout(key):
    """GP の入力の列（シフト前の名前）と、シフトに使う温度の列（C は None）。
    B_Tset・A_Tset：(ln γ̇ + ln a_h(Tset), ln P3_3, ln Tff)、B_Tff・A_Tff：(ln γ̇ + ln a_h(Tff), ln P3_3, Tset)、C：(ln γ̇, ln P3_3, ln Tff, Tset)"""
    if key == "C":
        return ["ln_gd", "ln_P33", "ln_Tff", "Tset"], None
    temp = key.split("_")[1]
    return (["ln_gd", "ln_P33", "ln_Tff"], "Tset") if temp == "Tset" else (["ln_gd", "ln_P33", "Tset"], "Tff")


def model_title(key, mode=None):
    """図のタイトルに使うモデルの名前（入力の取り方つき）"""
    return f"{MODEL_JA[key]}・{MODE_JA[INPUT_MODE if mode is None else mode]}"


def input_labels(key):
    cols, tcol = layout(key)
    lab = [COL_JA[c] for c in cols]
    if tcol is not None and key.startswith("B"):
        lab[0] = f"ln γ̇ + ln a_h({tcol})"
    return lab


def kernel_for(d, linear=None):
    """6月のノートのカーネル（ARD Matern 5/2 ＋ ホワイトノイズ）。linear=True なら線形項（DotProduct）を足す（射出速度の両端の外挿のため）"""
    linear = USE_LINEAR if linear is None else linear
    k = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0] * d, length_scale_bounds=(1e-2, 1e2), nu=2.5)
    if linear:
        k = k + C(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-5, 1e3))
    return k + WhiteKernel(noise_level=0.1, noise_level_bounds=(NOISE_LB, 1e1))


def mode_col(df, c, mode):
    """入力の列名。mode = "cond" で、条件平均の列（<名前>_cm）があればそれを使う（予測用に組み立てた表には _cm がないので、そのまま）"""
    return f"{c}_cm" if mode == "cond" and f"{c}_cm" in df.columns else c


def design(df, key, cfg, th=0.0, tv=0.0, mode=None):
    """データ → (標準化の定数を決めるための入力 X0〔シフト前〕、実際の入力 Xs〔シフト後〕、出力のシフト量 ysh = ln a_v(T))。
    mode（既定 INPUT_MODE）が "cond" なら、γ̇・P3_3・Tff は条件ごとの平均を使う"""
    mode = INPUT_MODE if mode is None else mode
    cols, tcol = layout(key)
    X0 = np.column_stack([df[mode_col(df, c, mode)].to_numpy(float) for c in cols])
    Xs, ysh = X0.copy(), np.zeros(len(df))
    if tcol is not None:
        T = df[mode_col(df, tcol, mode)].to_numpy(float)
        Xs[:, 0] = X0[:, 0] + shift_ln(T, th, cfg)
        ysh = shift_ln(T, tv, cfg)
    return X0, Xs, ysh


class LossRecorder:
    """6月のノートのカスタムオプティマイザと同じ（L-BFGS-B）。各ステップの損失（負の対数周辺尤度）を、最適化 1 回ごとに runs に記録する"""

    def __init__(self):
        self.runs = []

    def __call__(self, obj_func, initial_theta, bounds):
        from scipy.optimize import minimize
        hist = []
        res = minimize(obj_func, initial_theta, bounds=bounds, method="L-BFGS-B", jac=True,
                       callback=lambda xk: hist.append(float(obj_func(xk, eval_gradient=False))))
        self.runs.append(hist)
        return res.x, res.fun


def fit_gp(key, tr, cfg, th=0.0, tv=0.0, restarts=None, theta0=None, record=False, mode=None):
    """学習。標準化・中心化は、シフトの量によらない定数（シフト前の入力の平均・標準偏差、ln η の平均。学習データだけで計算）。
    出力の自動標準化（normalize_y）は使わない（シフトの量ごとに尤度が比べられなくなるため）。theta0：カーネルの初期値（最初の最適化はここから、
    残りの restarts 回は、探索範囲の中の乱数の初期値から。対数周辺尤度が最大の解を採る）。record=True なら、損失の推移を戻り値の loss_runs に入れる。
    mode：入力の取り方（既定 INPUT_MODE。"shot" か "cond"）。予測にも同じ取り方を使う"""
    mode = INPUT_MODE if mode is None else mode
    X0, Xs, ysh = design(tr, key, cfg, th, tv, mode)
    mu_x, sd_x = X0.mean(axis=0), X0.std(axis=0)
    y = tr["ln_eta"].to_numpy()
    y0 = float(y.mean())
    k = kernel_for(X0.shape[1])
    if theta0 is not None:
        k = k.clone_with_theta(theta0)
    rec = LossRecorder() if record else None
    gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=N_RESTARTS if restarts is None else restarts, random_state=SEED,
                                  optimizer=rec if record else "fmin_l_bfgs_b")
    gp.fit((Xs - mu_x) / sd_x, y - ysh - y0)
    gp.optimizer = "fmin_l_bfgs_b"                          # 保存（joblib）できるように戻す
    out = {"gp": gp, "mu_x": mu_x, "sd_x": sd_x, "y0": y0, "key": key, "cfg": dict(cfg), "th": float(th), "tv": float(tv), "cols": layout(key)[0], "mode": mode}
    if record:
        out["loss_runs"] = [list(h) for h in rec.runs]
    return out


def fit_job(key, tr, cfg, th, tv, restarts, theta0, mode, record=False):
    """fit_gp を並列に動かすための仕事（joblib から呼ぶ。設定は引数で渡す）"""
    warnings.simplefilter("ignore")
    return fit_gp(key, tr, cfg, th, tv, restarts, theta0, record, mode)


def predict(fit, df):
    """予測平均（ln η）、予測標準偏差（ノイズ込み）、潜在関数の標準偏差、ノイズの標準偏差"""
    _, Xs, ysh = design(df, fit["key"], fit["cfg"], fit["th"], fit["tv"], fit.get("mode", "shot"))
    mu, sd = fit["gp"].predict((Xs - fit["mu_x"]) / fit["sd_x"], return_std=True)
    noise = fit["gp"].kernel_.k2.noise_level
    return mu + fit["y0"] + ysh, sd, np.sqrt(np.maximum(sd ** 2 - noise, 0.0)), float(np.sqrt(noise))


def kernel_parts(gp):
    """学習したカーネルを、(Matern の部分〔振幅² × Matern〕, 線形の部分〔なければ None〕, ノイズ) に分ける"""
    k = gp.kernel_
    if isinstance(k.k1, Sum):
        return k.k1.k1, k.k1.k2, k.k2
    return k.k1, None, k.k2


def kernel_row(fit):
    """ハイパーパラメータ（振幅、長さスケール、線形項、ノイズ、対数周辺尤度、下限・上限への張り付き）"""
    gp = fit["gp"]
    prod, lin, white = kernel_parts(gp)
    ls = np.atleast_1d(prod.k2.length_scale)
    th_n, tv_n, _ = theta_names(fit["cfg"])
    row = {"model": fit["key"], th_n: fit["th"] if fit["key"].startswith("B") else np.nan, tv_n: fit["tv"] if fit["key"] != "C" else np.nan,
           "LML": float(gp.log_marginal_likelihood_value_), "amplitude": float(np.sqrt(prod.k1.constant_value)),
           "noise_var": float(white.noise_level), "noise_sd": float(np.sqrt(white.noise_level)),
           "noise_at_lower_bound": bool(white.noise_level < NOISE_LB * 1.05), "ls_at_bound": bool(np.any(ls > 99) or np.any(ls < 0.0105))}
    if lin is not None:
        row.update({"lin_amplitude": float(np.sqrt(lin.k1.constant_value)), "lin_sigma0": float(lin.k2.sigma_0)})
    row.update({f"ls_{c}": float(l) for c, l in zip(input_labels(fit["key"]), ls)})
    return row


# ---------------------------------------------------------------- プロファイル尤度（本体の手順5と同じ考え方）
def grid_of(lo, hi, step):
    return np.round(np.arange(lo, hi + step / 2.0, step), 6)


def prof_chain(key, tr, cfg, points, restarts, mode):
    """格子の点 points = [(θ_h, θ_v), …] を順に学習して LML を返す。前の点の解を初期値にする（局所解でがたつくのを減らす）。
    theta はその点で学習したカーネルの値（最終モデルの学習の初期値に使う）"""
    warnings.simplefilter("ignore")
    out, theta = [], None
    for th, tv in points:
        f = fit_gp(key, tr, cfg, th, tv, restarts, theta, mode=mode)
        theta = f["gp"].kernel_.theta
        out.append({"th": float(th), "tv": float(tv), "LML": float(f["gp"].log_marginal_likelihood_value_),
                    "noise_sd": float(np.sqrt(f["gp"].kernel_.k2.noise_level)), "theta": theta.tolist()})
    return out


def run_profile(key, tr, cfg, n_jobs=None, restarts=None, mode=None):
    """粗い格子で全体を見て、最良の点の近くを細かい格子で計算し直す。構造 A は θ_h = 0 の 1 次元、B は (θ_h, θ_v) の 2 次元。
    細かい格子は、最良の点を中心に取る。計算した結果、最良の点が動いたら（細かい格子の端に当たったとき）、新しい最良の点を中心に取り直す
    （最良の点が動かなくなるか、MAX_REFINE 回まで。斜めの尾根の上では、尤度がほとんど変わらないまま最良の点が動くことがある）。
    E_h ごとの列（chain）を並列に計算する。mode（既定 INPUT_MODE）は、並列の計算にも引数で渡す。
    返すもの：(表, 収束したか)。表の列：th, tv, LML, noise_sd, theta（カーネルの値。CSV には入れない）, stage（粗い／細かい）。
    収束したか：最良の点の周り（±細かい格子の幅）の点が、すべて計算済みか"""
    from joblib import Parallel, delayed
    n_jobs = N_JOBS if n_jobs is None else n_jobs
    restarts = PROFILE_RESTARTS if restarts is None else restarts
    mode = INPUT_MODE if mode is None else mode
    struct = key.split("_")[0]
    hg, vg = grid_of(*cfg["h_grid"]), grid_of(*cfg["v_grid"])
    rng, step = cfg.get("fine") or (FINE_A if struct == "A" else FINE_B)

    def run(points, stage):
        by = {}
        for th, tv in points:
            by.setdefault(float(th), []).append(float(tv))
        chs = [[(th, tv) for tv in sorted(tvs)] for th, tvs in sorted(by.items())]
        res = Parallel(n_jobs=min(n_jobs, len(chs)))(delayed(prof_chain)(key, tr, cfg, ch, restarts, mode) for ch in chs)
        t = pd.DataFrame([r for rr in res for r in rr])
        t["stage"] = stage
        return t

    pts = run([(0.0, v) for v in vg] if struct == "A" else [(h, v) for h in hg for v in vg], "粗い")
    converged = False
    for _ in range(MAX_REFINE):
        b = pts.loc[pts["LML"].idxmax()]
        hs = [0.0] if struct == "A" else grid_of(b["th"] - rng, b["th"] + rng, step)
        vs = grid_of(b["tv"] - rng, b["tv"] + rng, step)
        have = set(zip(np.round(pts["th"], 6), np.round(pts["tv"], 6)))
        new = [(h, v) for h in hs for v in vs if (round(float(h), 6), round(float(v), 6)) not in have]
        if new:
            pts = pd.concat([pts, run(new, "細かい")], ignore_index=True)
        b2 = pts.loc[pts["LML"].idxmax()]
        if (b2["th"], b2["tv"]) == (b["th"], b["tv"]):       # 最良の点が動かなかった：周りがすべて計算済みで、最良
            converged = True
            break
    pts = pts.sort_values("LML", ascending=False).drop_duplicates(subset=["th", "tv"]).sort_values(["th", "tv"]).reset_index(drop=True)
    return pts, converged


# 尤度比検定の目安：2 つの係数（横・縦）を足したときの 5% 基準。2·ΔLML > 5.99（自由度 2 のカイ二乗分布の 95% 点）→ ΔLML > 3.0
LRT_DLML = 3.0


def profile_best(pts, key, cfg, converged=True, lml_zero=None):
    """最良の θ と、LML が最大から 1.92 以内の範囲（おおよその 95% 区間）。at_edge：最良の点が粗い格子の端。refine_converged：細かい格子の取り直しが収束したか。
    範囲は、計算した点の中だけで決まる（斜めの尾根の上では、実際の範囲はもっと長い）。尾根に沿った方向は決まりにくく、差 θ_v − θ_h（diff_*）が最もよく決まる"""
    r = pts.loc[pts["LML"].idxmax()]
    inside = pts[pts["LML"] >= r["LML"] - 1.92]
    hg, vg = grid_of(*cfg["h_grid"]), grid_of(*cfg["v_grid"])
    z = pts[(pts["th"] == 0.0) & (pts["tv"] == 0.0)]
    lz = float(z["LML"].iloc[0]) if len(z) else (np.nan if lml_zero is None else lml_zero)
    diff = inside["tv"] - inside["th"]
    edge = bool(r["tv"] in (vg.min(), vg.max()) or (key.startswith("B") and r["th"] in (hg.min(), hg.max())))
    return {"th_best": float(r["th"]), "tv_best": float(r["tv"]), "diff_best": float(r["tv"] - r["th"]), "LML_best": float(r["LML"]),
            "LML_at_0": lz, "dLML_vs_0": float(r["LML"]) - lz, "th_lo": float(inside["th"].min()), "th_hi": float(inside["th"].max()),
            "tv_lo": float(inside["tv"].min()), "tv_hi": float(inside["tv"].max()), "diff_lo": float(diff.min()), "diff_hi": float(diff.max()),
            "n_in_95": int(len(inside)), "n_points": int(len(pts)), "at_edge": edge, "refine_converged": bool(converged),
            "theta_best": np.asarray(r["theta"], float) if "theta" in pts.columns else None,
            "theta_zero": np.asarray(z["theta"].iloc[0], float) if (len(z) and "theta" in pts.columns) else None}


# ---------------------------------------------------------------- 評価
def evaluate(y, mu, sd):
    """ln η の指標：R2、RMSE、95% 予測区間の被覆率、NLPD、予測標準偏差の平均。R2_real は粘度の実数値での R2（6月のノートと同じ）"""
    z = (y - mu) / sd
    return {"R2": r2_score(y, mu) if len(y) > 1 else np.nan, "RMSE": float(np.sqrt(mean_squared_error(y, mu))),
            "coverage95": float(np.mean(np.abs(z) <= 1.96)), "NLPD": float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + 0.5 * z ** 2)),
            "mean_sd": float(np.mean(sd)), "R2_real": r2_score(np.exp(y), np.exp(mu)) if len(y) > 1 else np.nan}


def metrics_rows(mat, key, te, mu, sd, sd_f, extra=None):
    """検証の指標を、全体・低速側・高速側（と、感度確認の条件を除いたもの）ごとに返す"""
    subsets = {"全体": np.ones(len(te), bool)}
    for g in ["低速側", "高速側", "範囲内"]:
        if (te["group"] == g).any():
            subsets[g] = (te["group"] == g).to_numpy()
    if te["is_sens"].any():
        subsets.update({s + "（感度確認を除く）": m & ~te["is_sens"].to_numpy() for s, m in list(subsets.items())})
    rows = []
    y = te["ln_eta"].to_numpy()
    for s, m in subsets.items():
        if m.sum() == 0:
            continue
        rows.append({"材料": MAT_NAMES[mat], "モデル": key, "範囲": s, "行数": int(m.sum()), "条件数": int(te.loc[m, "cond"].nunique()),
                     **evaluate(y[m], mu[m], sd[m]), "mean_sd_latent": float(np.nanmean(sd_f[m])) if np.isfinite(sd_f[m]).any() else np.nan, **(extra or {})})
    return rows


def judge(ok, text, mat=None, section=""):
    """結果の判定を 1 行で表示し、記録する。ok が True なら「OK」、False なら「★要確認」"""
    head = f"[{MAT_NAMES[mat]}] " if mat else ""
    print(f">> 判定：{'OK' if ok else '★要確認'}　{head}{text}")
    JUDGES.append({"節": section, "材料": MAT_NAMES[mat] if mat else "", "判定": "OK" if ok else "★要確認", "内容": text})


def to_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------- 外れ値の検知（本体の check_data.py と同じ規則。05 のデータ用）
def cond_means(df):
    """条件ごとの平均（ln η と ln γ̇ をショットで平均した点）"""
    g = df.groupby("cond", sort=False)
    t = g.agg(Tset=("Tset", "first"), H=("H", "first"), v=("v", "first"), ln_eta=("ln_eta", "mean"), ln_gd=("ln_gd", "mean"),
              n=("ln_eta", "size"), grp=("grp", "first")).reset_index()
    t["ln_H"] = np.log(t["H"])
    return t


def outlier_scan(df):
    """規則 f：条件平均が、ほかの条件から予測した値と大きく違う条件を拾う（本体の nb_common.outlier_scan と同じ）。
    条件平均の GPR（入力 ln γ̇（条件平均）・ln H・Tset、線形項あり）の、条件を 1 つずつ抜いた予測（閉じた式）との残差を、予測標準偏差で割って z とする。
    ※ この検知だけは、データの確認のための道具なので、本体と同じく H を使う（モデルの説明変数ではない）"""
    t = cond_means(df)
    X, y = t[["ln_gd", "ln_H", "Tset"]].to_numpy(), t["ln_eta"].to_numpy()
    Xs, yc = StandardScaler().fit_transform(X), y - y.mean()
    gp = GaussianProcessRegressor(kernel=kernel_for(3, True), n_restarts_optimizer=N_RESTARTS, random_state=SEED).fit(Xs, yc)
    Kinv = np.linalg.inv(gp.kernel_(Xs))
    d = np.diag(Kinv)
    t["resid"], t["sd"] = (Kinv @ yc) / d, np.sqrt(1.0 / d)       # 条件 i を抜いた予測：平均 = y_i − (K⁻¹y)_i / K⁻¹_ii、分散 = 1 / K⁻¹_ii
    t["z"] = t["resid"] / t["sd"]
    t["flag"] = (t["z"].abs() > OUTLIER_Z) & (t["resid"].abs() > OUTLIER_RESID)
    return t[["cond", "grp", "Tset", "H", "v", "n", "resid", "sd", "z", "flag"]]


def registered(mat, grp, row, plist):
    """一覧の「除外」「感度確認」に、この条件（row に shot があれば、そのショット）が登録されているか（ID を返す）"""
    one = pd.DataFrame({"Tset": [float(row["Tset"])], "H": [float(row["H"])], "v": [float(row["v"])], "shot": [row.get("shot", -1)]})
    if pd.isna(one["shot"].iloc[0]):
        one["shot"] = -1
    _, ids = problem_mask(one, mat, grp, plist, ("除外", "感度確認"))
    return ids[0]


def candidates(mat, pool, plist):
    """問題の候補（見直しの候補であって、外れ値と判定したものではない。一覧に載せるかはユーザーが決める）。
    a 温度の間で γ̇ が 1.3 倍を超えて違う／b 最低温度の粘度が最高温度より小さい（順序が逆）／c 同じ条件内の ln η の標準偏差 > 0.1／
    e 別のショットと値が完全に同じ行／g P3_3 が条件の中央値から P33_DEV 以上離れたショット"""
    cand = []
    g = pool.groupby(["H", "v", "Tset"]).agg(gd=("gd", "mean"), eta=("eta", "mean"), grp=("grp", "first")).reset_index()
    t_lo, t_hi = pool["Tset"].min(), pool["Tset"].max()
    for (h, v), s in g.groupby(["H", "v"]):
        if len(s) < 3:
            continue
        lg = np.log(s["gd"].to_numpy())
        if np.exp(lg.max() - lg.min()) > 1.3:
            odd = s.iloc[np.argmax(np.abs(lg - np.median(lg)))]
            cand.append({"規則": "a 温度の間でせん断速度が1.3倍超違う", "組": odd["grp"], "設定温度": int(odd["Tset"]), "厚み": h, "射出速度": int(v), "ショット": "",
                         "内容": f"γ̇ の最大/最小 = {np.exp(lg.max() - lg.min()):.2f}"})
        e = s.set_index("Tset")["eta"]
        if t_lo in e.index and t_hi in e.index and e[t_lo] / e[t_hi] < 0.95:
            cand.append({"規則": "b 粘度の順序が温度と逆", "組": s["grp"].iloc[0], "設定温度": int(t_lo), "厚み": h, "射出速度": int(v), "ショット": "",
                         "内容": f"η({int(t_lo)})/η({int(t_hi)}) = {e[t_lo] / e[t_hi]:.2f}"})
    sd = pool.groupby(["Tset", "H", "v"]).agg(std=("ln_eta", "std"), size=("ln_eta", "size"), grp=("grp", "first")).reset_index()
    for _, r in sd[(sd["size"] >= 2) & (sd["std"] > 0.1)].iterrows():
        cand.append({"規則": "c 同じ条件内のばらつきが大きい", "組": r["grp"], "設定温度": int(r["Tset"]), "厚み": r["H"], "射出速度": int(r["v"]), "ショット": "",
                     "内容": f"ln η の標準偏差 = {r['std']:.3f}（{int(r['size'])} ショット）"})
    key = [c for c in ["eta", "gd", "P31", "P32", "P33"] if c in pool.columns and pool[c].notna().all()]
    for grp, df in pool.groupby("grp"):
        dup = df[df.duplicated(key, keep=False)]
        for _, gg in dup.groupby(key):
            if gg["file"].nunique() < 2:
                continue
            r0 = gg.sort_values("file").iloc[0]
            cand.append({"規則": "e 別のショットと同じ値の行", "組": grp, "設定温度": int(r0["Tset"]), "厚み": r0["H"], "射出速度": int(r0["v"]), "ショット": str(int(r0["shot"])),
                         "内容": f"{' と '.join(gg['file'])} の {'・'.join(key)} が完全に一致"})
    med = pool.groupby("cond")["P33"].transform("median")
    n = pool.groupby("cond")["P33"].transform("size")
    dev = (pool["P33"] / med - 1.0).abs()
    for _, r in pool[(n >= 3) & (dev >= P33_DEV)].iterrows():
        cand.append({"規則": f"g P3_3 が条件の中央値から{int(P33_DEV * 100)}%以上離れている", "組": r["grp"], "設定温度": int(r["Tset"]), "厚み": r["H"], "射出速度": int(r["v"]),
                     "ショット": str(int(r["shot"])), "内容": f"P3_3 = {r['P33']:.2f} MPa（条件の中央値 {med[r.name]:.2f}、{(r['P33'] / med[r.name] - 1) * 100:+.0f}%）"})
    out = pd.DataFrame(cand, columns=["規則", "組", "設定温度", "厚み", "射出速度", "ショット", "内容"])
    out.insert(0, "材料", MAT_NAMES[mat])
    out["一覧の ID"] = [registered(mat, r["組"], {"Tset": r["設定温度"], "H": r["厚み"], "v": r["射出速度"],
                                                   "shot": int(r["ショット"]) if r["ショット"] != "" else np.nan}, plist) for _, r in out.iterrows()]
    return out


# ---------------------------------------------------------------- Cross-WLF（6月のノートのセル10.5・11。D3 = 0）
CWLF_NAMES = ["n", "tau_star", "D1", "D2", "A1", "A2"]


def cross_wlf_ln(T_c, gd, n, ln_tau, ln_D1, D2, A1, A2):
    """ln η（Cross-WLF。D3 = 0 なので T* = D2）。数値の安定のため、τ* と D1 は ln で受け取る。T は ℃、D2 は K"""
    diff = np.maximum(np.asarray(T_c, float) + 273.15 - D2, 1.0)
    ln_eta0 = ln_D1 - A1 * diff / (A2 + diff)
    return ln_eta0 - np.logaddexp(0.0, (1.0 - n) * (ln_eta0 + np.log(gd) - ln_tau))


CWLF_STARTS = globals().get("CWLF_STARTS", 12)            # Cross-WLF の当てはめの初期値の数（局所解を避けるため、乱数の初期値を何度か試して、残差が最小のものを採る）


def fit_cross_wlf(tr, t_col, fixed):
    """Cross-WLF の係数を、学習データ（ln η の最小二乗）で当てはめる。fixed：{名前: 値 または None（探索）}。
    初期値は、6月のノートに近い値と、探索範囲の中の乱数（SEED で固定）の CWLF_STARTS 個で、残差が最小のものを採る。
    返す辞書：係数（実数）、学習データでの残差の標準偏差"""
    from scipy.optimize import curve_fit
    init = {"n": 0.3, "tau_star": np.log(3e4), "D1": np.log(1e12), "D2": 263.15, "A1": 25.0, "A2": 51.6}
    lo = {"n": 0.0, "tau_star": np.log(1e2), "D1": np.log(1e3), "D2": 150.0, "A1": 0.5, "A2": 10.0}
    hi = {"n": 0.95, "tau_star": np.log(1e7), "D1": np.log(1e30), "D2": 450.0, "A1": 80.0, "A2": 200.0}
    conv = {k: (np.log(v) if k in ("tau_star", "D1") and v is not None else v) for k, v in fixed.items()}
    free = [k for k in CWLF_NAMES if conv.get(k) is None]

    def full(args):
        vals = dict(conv)
        vals.update(dict(zip(free, args)))
        return [vals[k] for k in CWLF_NAMES]

    T, gd, y = tr[t_col].to_numpy(float), tr["gd"].to_numpy(float), tr["ln_eta"].to_numpy()
    f = lambda X, *args: cross_wlf_ln(X[0], X[1], *full(args))
    rs = np.random.RandomState(SEED)
    starts = [[init[k] for k in free]] + [[rs.uniform(lo[k], hi[k]) for k in free] for _ in range(max(CWLF_STARTS - 1, 0))]
    best = None
    for p0 in starts:
        try:
            p, _ = curve_fit(f, np.vstack([T, gd]), y, p0=p0, bounds=([lo[k] for k in free], [hi[k] for k in free]), maxfev=20000)
        except Exception:
            continue
        sse = float(np.sum((y - f(np.vstack([T, gd]), *p)) ** 2))
        if best is None or sse < best[0]:
            best = (sse, p)
    if best is None:
        raise RuntimeError("Cross-WLF の当てはめがすべての初期値で失敗しました")
    popt = best[1]
    vals = full(popt)
    resid = y - cross_wlf_ln(T, gd, *vals)
    at_bound = [k for k, v in zip(free, popt) if np.isclose(v, lo[k], rtol=0, atol=1e-6 * max(1.0, abs(lo[k]))) or np.isclose(v, hi[k], rtol=0, atol=1e-6 * max(1.0, abs(hi[k])))]
    out = dict(zip(CWLF_NAMES, vals))
    out["tau_star"], out["D1"] = float(np.exp(out["tau_star"])), float(np.exp(out["D1"]))
    return {"params": out, "free": free, "at_bound": at_bound, "resid_sd": float(np.sqrt(np.mean(resid ** 2))), "t_col": t_col}


def predict_cross_wlf(cw, df):
    p = cw["params"]
    mu = cross_wlf_ln(df[cw["t_col"]].to_numpy(float), df["gd"].to_numpy(float), p["n"], np.log(p["tau_star"]), np.log(p["D1"]), p["D2"], p["A1"], p["A2"])
    return mu, np.full(len(df), cw["resid_sd"])


# ---------------------------------------------------------------- 予測の入力を組み立てる（依存性グラフ・3次元グラフ・保存したモデルの使い方）
VAR_JA = {"gd": ("せん断速度 γ̇", "(1/s)", True), "P33": ("流動中樹脂圧力 P3_3", "(MPa)", True),
          "Tff": ("流動中樹脂温度 Tff", "(℃)", False), "Tset": ("加熱シリンダ温度 Tset", "(℃)", False),
          "Tboth": ("Tset と Tff を同時に", "(℃。横軸は Tset)", False)}     # (名前, 単位, 対数軸か)。Tboth：設定温度を上げると、流動中樹脂温度も同じだけ上がるとして、2 つを一緒に動かす


def model_inputs(gd, P33, Tff, Tset):
    """説明変数の実数値（γ̇ [1/s]、P3_3 [MPa]、Tff [℃]、Tset [℃]。配列でも数でもよい）→ predict に渡せる表"""
    gd, P33, Tff, Tset = (np.asarray(a, float).ravel() for a in np.broadcast_arrays(gd, P33, Tff, Tset))
    return pd.DataFrame({"gd": gd, "ln_gd": np.log(gd), "P33": P33, "ln_P33": np.log(P33 * 1e6), "Tff": Tff, "ln_Tff": np.log(Tff + 273.15), "Tset": Tset})


def base_point(tr, cfg):
    """依存性グラフで固定する値：学習データの中央値（γ̇・P3_3 は対数の中央値）。設定温度は基準温度 Tref"""
    return {"gd": float(np.exp(tr["ln_gd"].median())), "P33": float(np.exp(tr["ln_P33"].median()) / 1e6),
            "Tff": float(np.exp(tr["ln_Tff"].median()) - 273.15), "Tset": float(cfg["Tref"])}


def var_grid(pool, var, n, pad=0.1):
    """動かす説明変数の格子（データの範囲を少し広げる。対数軸の変数は対数で等間隔。設定温度は ±10 ℃）"""
    if var in ("Tset", "Tboth"):
        return np.linspace(pool["Tset"].min() - 10.0, pool["Tset"].max() + 10.0, n)
    x = pool[var].to_numpy(float)
    if VAR_JA[var][2]:
        lo, hi = np.log(x.min()), np.log(x.max())
        return np.exp(np.linspace(lo - pad * (hi - lo), hi + pad * (hi - lo), n))
    lo, hi = x.min(), x.max()
    return np.linspace(lo - pad * (hi - lo), hi + pad * (hi - lo), n)


def superpose_rmse(t):
    """重ね合わせの目安：厚みごとに、条件平均の点（x = ln γ̇〔換算〕、y = ln η〔換算〕）を 3 次式で当てはめた残差の RMSE（全温度をまとめて）。
    温度のシフトで曲線が重なるほど小さくなる"""
    res = []
    for _, g in t.groupby("H"):
        if len(g) < 6:
            continue
        c = np.polyfit(g["x"], g["y"], 3)
        res += list(g["y"] - np.polyval(c, g["x"]))
    return float(np.sqrt(np.mean(np.square(res)))) if res else np.nan


def shifted_cond_means(df, key, cfg, th, tv):
    """条件平均の、換算した ln γ̇（x）と ln η（y）。key が None ならシフト前。Tff 版はショットごとの Tff でずらしてから平均する"""
    x, y = df["ln_gd"].to_numpy(), df["ln_eta"].to_numpy()
    if key is not None:
        T = df[layout(key)[1]].to_numpy(float)
        x = x + shift_ln(T, th, cfg)
        y = y - shift_ln(T, tv, cfg)
    return df.assign(x=x, y=y).groupby("cond").agg(x=("x", "mean"), y=("y", "mean"), Tset=("Tset", "first"), H=("H", "first")).reset_index()


# ---------------------------------------------------------------- 図の保存と、色・記号
H_COLORS = {0.5: "#d62728", 1.0: "#1f77b4", 1.5: "#f0b400", 2.0: "#2ca02c"}       # 厚みごとの色（6月のノートと同じ並び）
H_MARKS = {0.5: "o", 1.0: "s", 1.5: "^", 2.0: "D"}
T_PALETTE = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]               # 設定温度ごとの色（低い順）
T_MARKS = ["o", "s", "^", "D", "v"]


def t_style(levels):
    """設定温度の水準 → (色, 記号) の辞書（低い順に青・緑・赤）"""
    lv = sorted(levels)
    return {t: (T_PALETTE[i % len(T_PALETTE)], T_MARKS[i % len(T_MARKS)]) for i, t in enumerate(lv)}


def log_ticks(ax, axis="xy"):
    """対数軸の目盛りを、数字が重ならない形にする（軸の範囲を決めた後で呼ぶ）。範囲が 2.5 桁以下なら、1・2・5 × 10^n を普通の数で、
    それより広ければ、10 の累乗だけ。小さい目盛りには数字を付けない"""
    import matplotlib.ticker as ticker
    for name in axis:
        a = ax.xaxis if name == "x" else ax.yaxis
        lo, hi = ax.get_xlim() if name == "x" else ax.get_ylim()
        if hi > lo > 0 and np.log10(hi / lo) <= 2.5:
            a.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            a.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            a.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))
            a.set_major_formatter(ticker.LogFormatterSciNotation())
        a.set_minor_formatter(ticker.NullFormatter())


def save_fig(fig, out_dir, stem, show=None):
    """図を out_dir に PNG で保存する（FIG_DPI）。HR_ALL が True なら、同じ図を HR_ALL_DPI（既定 600 dpi）で out_dir/highres/ にも保存する。
    show（既定 SHOW_FIGS）なら表示し、閉じる"""
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, f"{stem}.png")]
    fig.savefig(paths[0], dpi=FIG_DPI, bbox_inches="tight")
    if HR_ALL:
        os.makedirs(os.path.join(out_dir, "highres"), exist_ok=True)
        paths.append(os.path.join(out_dir, "highres", f"{stem}.png"))
        fig.savefig(paths[1], dpi=HR_ALL_DPI, bbox_inches="tight")
    if SHOW_FIGS if show is None else show:
        plt.show()
    plt.close(fig)
    return paths


DEP92_STYLE = {0.5: ("#d62728", "o", 14), 1.0: ("#1f77b4", "s", 13), 1.5: ("#ffd30ed9", "^", 12), 2.0: ("#41ea4f", "D", 11)}      # 6月のノートのセル9-2 の style_dict（色・形・zorder）
DEP92_VARS = [("gd", "せん断速度", r"$\gamma$", "(1/s)", True, "Shear_Rate"), ("P33", "流動中樹脂圧力", r"$P_{3\_3}$", "(MPa)", True, "P33"),
              ("Tff", "流動中樹脂温度", r"$T_{ff}$", "(℃)", False, "Tff"), ("Tset", "加熱シリンダ温度", r"$T_{set}$", "(℃)", False, "Tset")]      # (変数, 名前, 記号, 単位, 対数軸か, ファイル名)


def dependency_plots_92(fit, mat, tr, out_dir, stem="依存性グラフ", **kw):
    """GPR6月ノートのセル9-2（目的変数に対する各説明変数の依存性グラフ）と同じ図を、学習データ（tr）と学習済みモデルで作る。
    説明変数ごとに 1 枚：その変数だけを動かし、ほかは学習データの平均（対数の変数は、対数の平均）に固定した、予測平均（赤線）と 95% 区間（帯）に、
    学習データ全部の点（色・形は厚み）を重ねる。図の大きさ・文字・点・目盛り（対数軸は 10 の累乗）・凡例・タイトルは、6月のセル9-2 と同じ。
    設定（kw）：figsize、dpi、axis_fs・legend_fs・title_fs・tick_fs、plot_marker・legend_marker、show_xlabel・show_ylabel、ylim、xlim（変数 → (下限, 上限)）、n_grid、formats、show。
    保存したファイルの一覧を返す"""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    cfg = dict(figsize=(12, 10), dpi=None, axis_fs=24, legend_fs=24, title_fs=24, tick_fs=28, plot_marker=300, legend_marker=300, show_xlabel=False, show_ylabel=False,
               ylim=(0.1, 10000.0), xlim={"gd": (10.0, 100000.0)}, n_grid=100, formats=["png"], show=None)
    cfg.update(kw)
    dpi = cfg["dpi"] or FIG_DPI
    base = {"gd": float(np.exp(tr["ln_gd"].mean())), "P33": float(np.exp(tr["ln_P33"].mean()) / 1e6), "Tff": float(np.exp(tr["ln_Tff"].mean()) - 273.15), "Tset": float(tr["Tset"].mean())}
    saved = []
    for var, name, sym, unit, logx, fname in DEP92_VARS:
        fig, ax = plt.subplots(figsize=cfg["figsize"], dpi=dpi)
        x = tr[var].to_numpy(float)
        lim = cfg["xlim"].get(var)
        if lim is None:                                           # 6月と同じ：データの範囲を、両側に 10% 広げる（対数軸の変数は、対数で）
            lo, hi = (np.log(x.min()), np.log(x.max())) if logx else (x.min(), x.max())
            lo, hi = lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)
            grid = np.exp(np.linspace(lo, hi, cfg["n_grid"])) if logx else np.linspace(lo, hi, cfg["n_grid"])
        else:
            grid = np.exp(np.linspace(np.log(lim[0]), np.log(lim[1]), cfg["n_grid"])) if logx else np.linspace(lim[0], lim[1], cfg["n_grid"])
        mu, sd, _, _ = predict(fit, model_inputs(**{**base, var: grid}))
        lower, upper = np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd)
        line = ax.plot(grid, np.exp(mu), color="red", linewidth=2, label="予測平均値", zorder=20)
        fill = ax.fill_between(grid, lower, upper, color="red", alpha=0.25, label="95% 信頼区間", zorder=15)
        ax.plot(grid, lower, color="red", linewidth=0.5, alpha=0.6, zorder=16)
        ax.plot(grid, upper, color="red", linewidth=0.5, alpha=0.6, zorder=16)
        hs, labs = [], []
        for h in sorted(tr["H"].unique()):
            m = np.isclose(tr["H"].to_numpy(), h)
            col, mk, z = DEP92_STYLE.get(float(h), ("gray", "x", 10))
            sc = ax.scatter(x[m], tr.loc[m, "eta"], color=col, s=cfg["plot_marker"], marker=mk, alpha=0.8, edgecolors="k", linewidths=0.5, zorder=z)
            hs.append(sc)
            labs.append(f"{h:g} mm")
        scale = np.sqrt(cfg["legend_marker"] / cfg["plot_marker"])
        ax.legend([line[0], fill, Patch(color="none", label="")] + hs, ["予測平均値", "95% 信頼区間", "キャビティ厚み"] + labs, loc="upper right", fontsize=cfg["legend_fs"],
                  framealpha=1.0, edgecolor="black", markerscale=scale)
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=cfg["tick_fs"], pad=8)
        ax.set_title("ガウス過程回帰による予測モデル\n（せん断速度と粘度の関係）" if var == "gd" else f"樹脂粘度の {name} 依存性", fontsize=cfg["title_fs"], fontweight="bold", pad=15)
        if cfg["show_xlabel"]:
            ax.set_xlabel(f"{name} {sym} {unit}", fontsize=cfg["axis_fs"], fontweight="bold", labelpad=15)
        if cfg["show_ylabel"]:
            ax.set_ylabel("樹脂粘度 (Pa·s)", fontsize=cfg["axis_fs"], fontweight="bold", labelpad=15)
        ax.set_yscale("log")
        if cfg["ylim"] is not None:
            ax.set_ylim(*cfg["ylim"])
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        if lim is not None:
            ax.set_xlim(*lim)
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))
            ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax.tick_params(axis="x", which="both", direction="in", top=True, labelsize=cfg["tick_fs"], pad=8)
        ax.grid(True, which="major", linestyle="-", alpha=0.6, color="gray", zorder=1)
        ax.grid(True, which="minor", linestyle=":", alpha=0.4, zorder=1)
        fig.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        for fmt in cfg["formats"]:
            path = os.path.join(out_dir, f"{stem}_{fname}.{fmt}")
            fig.savefig(path, bbox_inches="tight", dpi=dpi)
            saved.append(path)
        if HR_ALL and dpi < HR_ALL_DPI:
            os.makedirs(os.path.join(out_dir, "highres"), exist_ok=True)
            fig.savefig(os.path.join(out_dir, "highres", f"{stem}_{fname}.png"), bbox_inches="tight", dpi=HR_ALL_DPI)
        if SHOW_FIGS if cfg["show"] is None else cfg["show"]:
            plt.show()
        plt.close(fig)
    return saved


def flow_inputs(sub, lg, h, t):
    """予測線の入力（ln γ̇ = lg の点）。P3_3・Tff は、同じ設定温度・厚みの条件平均を ln γ̇ で内挿する（範囲の外は端の値のまま）"""
    cm = sub.groupby("cond").agg(lg=("ln_gd", "mean"), p=("ln_P33", "mean"), tf=("ln_Tff", "mean")).sort_values("lg")
    ip = lambda c: np.interp(lg, cm["lg"], cm[c])
    ln_tff = ip("tf")
    return pd.DataFrame({"ln_gd": lg, "gd": np.exp(lg), "ln_P33": ip("p"), "ln_Tff": ln_tff, "Tff": np.exp(ln_tff) - 273.15, "Tset": float(t), "H": float(h)})


FLOW_DEFAULT = dict(layouts=["temperature", "thickness"], band="predict", points="cond", show_test=True, mark_sens=True, x_range=None, extend=1.3,
                    xlim={"thickness": None, "temperature": None}, ylim={"thickness": None, "temperature": None}, figsize=(12, 10), dpi=None,
                    font={"axis": 24, "legend": 24, "title": 24, "tick": 28}, marker=300, marker_shot=70, line=2.0, show_title=True,
                    show_xlabel=False, show_ylabel=False, formats=["png"], n_grid=120, show=None)


T_CURVE_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]       # 総合図の、加熱シリンダ温度ごとの線・帯の色（低い温度から：赤・青・緑）。点の色は厚みごと（赤・青・黄・緑）なので、色は同じでも、点は形で見分ける


def flow_plot_common(fit, mat, pool, train_conds, out_dir, stem, tr=None, **kw):
    """6月のノートのセル9-2 と同じ仕様（図の大きさ〔既定 12 × 10 インチ。6月は 15 × 10〕、文字の大きさ、点の大きさ〔既定 300〕、軸の目盛り〔両軸とも 10 の累乗〕、凡例の並べ方）で、
    全キャビティ厚みの**学習データ**の点（色・形は厚み）を重ね、**厚みによらない共通の平均関数と 95% 区間**を描く。検証（3data）の点は描かない。
    曲線は、せん断速度だけを動かし、ほかの入力を固定したもの：P3_3 は学習データ（tr）の中央値、Tff は、その設定温度の学習データの中央値、Tset はその温度。
    曲線の範囲は、その設定温度の学習データの範囲を extend 倍に延ばしたもの（x_range があればそれ）。軸の範囲は、common_xlim・common_ylim（None なら、データを含む 10 の累乗の範囲）。
    2 種類の図を保存する：①加熱シリンダ温度ごとに 1 枚（…_全厚み共通_<温度>C）、②全温度を 1 枚に重ね、温度ごとに線・帯の色を変えた総合図（…_全厚み共通_全温度）。
    保存したファイルの一覧を返す"""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    cfg = {**FLOW_DEFAULT, "common_xlim": None, "common_ylim": None, "legend_ncol": 1, **kw}
    dpi = cfg["dpi"] or FIG_DPI
    tr = pool if tr is None else tr
    ft, saved = cfg["font"], []
    band_name = "95% 予測区間" if cfg["band"] == "predict" else "95% 信頼区間"
    p_med = float(np.exp(tr["ln_P33"].median()) / 1e6)
    levels = sorted(tr["Tset"].unique())
    tcol = {float(t): T_CURVE_COLORS[i % len(T_CURVE_COLORS)] for i, t in enumerate(levels)}
    curves = {}
    for t, sub in tr.groupby("Tset"):
        lgc = sub.groupby("cond")["ln_gd"].mean()
        lo, hi = (np.log(cfg["x_range"][0]), np.log(cfg["x_range"][1])) if cfg["x_range"] else (lgc.min() - np.log(cfg["extend"]), lgc.max() + np.log(cfg["extend"]))
        lg = np.linspace(lo, hi, cfg["n_grid"])
        tff_med = float(np.exp(sub["ln_Tff"].median()) - 273.15)
        mu, sd, sdf, _ = predict(fit, model_inputs(np.exp(lg), p_med, tff_med, float(t)))
        curves[float(t)] = (np.exp(lg), mu, sd if cfg["band"] == "predict" else sdf, tff_med)

    def points_of(sub):
        """学習データの点（px, py, 面積）。cfg["points"] が "cond" なら条件平均、"shot" ならショットごと"""
        if cfg["points"] == "cond":
            a = sub.groupby("cond").agg(lg=("ln_gd", "mean"), le=("ln_eta", "mean")).reset_index()
            return np.exp(a["lg"].to_numpy()), np.exp(a["le"].to_numpy()), cfg["marker"]
        return sub["gd"].to_numpy(), sub["eta"].to_numpy(), cfg["marker_shot"]

    def thickness_points(ax, sub, xs, ys, z=14):
        """厚みごとに色・形を変えて、点を描く（6月のセル9-2 と同じ：alpha 0.8、黒い縁）。凡例用の記号の一覧を返す"""
        hs = []
        for h in sorted(sub["H"].unique()):
            px, py, ms = points_of(sub[np.isclose(sub["H"], h)])
            col, mk = H_COLORS.get(float(h), "gray"), H_MARKS.get(float(h), "x")
            xs += list(px)
            ys += list(py)
            ax.scatter(px, py, s=ms, color=col, marker=mk, alpha=0.8, edgecolors="k", linewidths=0.5, zorder=z)
            hs.append(Line2D([], [], color=col, ls="none", marker=mk, markersize=np.sqrt(cfg["marker"]), markeredgecolor="k", markeredgewidth=0.5, label=f"{h:g} mm"))
        return hs

    def decade_lim(v, user):
        if user is not None:
            return user
        return 10.0 ** np.floor(np.log10(min(v))), 10.0 ** np.ceil(np.log10(max(v)))

    def save(fg, base):
        os.makedirs(out_dir, exist_ok=True)
        for fmt in cfg["formats"]:
            path = os.path.join(out_dir, f"{base}.{fmt}")
            fg.savefig(path, bbox_inches="tight", dpi=dpi)
            saved.append(path)
        if HR_ALL and dpi < HR_ALL_DPI:
            os.makedirs(os.path.join(out_dir, "highres"), exist_ok=True)
            fg.savefig(os.path.join(out_dir, "highres", f"{base}.png"), bbox_inches="tight", dpi=HR_ALL_DPI)

    def finish(fig, ax, xs, ys, base, title, handles, separate_legend=False):
        """軸・目盛り・凡例・タイトルを整えて保存する。separate_legend が True なら、グラフには凡例とタイトルを付けず、凡例だけを別の画像（base + "_凡例"）に保存する"""
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=ft["tick"], pad=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*decade_lim(xs, cfg["common_xlim"]))
        ax.set_ylim(*decade_lim(ys, cfg["common_ylim"]))
        for axis in (ax.xaxis, ax.yaxis):                        # 6月のセル9-2 と同じ：両軸とも 10 の累乗の目盛り
            axis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))
            axis.set_major_formatter(ticker.LogFormatterSciNotation())
            axis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(True, which="major", linestyle="-", alpha=0.6, color="gray", zorder=1)
        ax.grid(True, which="minor", linestyle=":", alpha=0.4, zorder=1)
        if not separate_legend:
            ax.legend(handles=handles, loc="lower left", fontsize=ft["legend"], framealpha=1.0, edgecolor="black", ncol=cfg["legend_ncol"])
            if cfg["show_title"] and title:
                ax.set_title(title, fontsize=ft["title"], fontweight="bold", pad=15)
        if cfg["show_xlabel"]:
            ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ (1/s)", fontsize=ft["axis"], fontweight="bold", labelpad=15)
        if cfg["show_ylabel"]:
            ax.set_ylabel(r"樹脂粘度 $\eta$ (Pa·s)", fontsize=ft["axis"], fontweight="bold", labelpad=15)
        fig.tight_layout()
        save(fig, base)
        if SHOW_FIGS if cfg["show"] is None else cfg["show"]:
            plt.show()
        plt.close(fig)
        if separate_legend:                                       # 凡例だけの画像
            lf = plt.figure(figsize=(6, 6))
            lf.legend(handles=handles, loc="center", fontsize=ft["legend"], framealpha=1.0, edgecolor="black", ncol=cfg["legend_ncol"])
            save(lf, base + "_凡例")
            plt.close(lf)

    # ① 加熱シリンダ温度ごと（曲線は赤。6月のセル9-2 と同じ）
    for t, (x, mu, s, tff_med) in curves.items():
        fig, ax = plt.subplots(figsize=cfg["figsize"])
        ax.fill_between(x, np.exp(mu - 1.96 * s), np.exp(mu + 1.96 * s), color="red", alpha=0.25, zorder=15)
        ax.plot(x, np.exp(mu - 1.96 * s), color="red", lw=0.5, alpha=0.6, zorder=16)
        ax.plot(x, np.exp(mu + 1.96 * s), color="red", lw=0.5, alpha=0.6, zorder=16)
        ax.plot(x, np.exp(mu), color="red", lw=cfg["line"], zorder=20)
        xs, ys = list(x), list(np.exp(mu))
        hs = thickness_points(ax, tr[tr["Tset"] == t], xs, ys)
        head = [Line2D([], [], color="red", lw=cfg["line"], label="予測平均値"), Patch(facecolor="red", alpha=0.25, label=band_name), Patch(color="none", label="キャビティ厚み")]
        ax.text(0.98, 0.97, f"固定した値\nP3_3：{p_med:.2f} MPa\nTff：{tff_med:.1f} ℃", transform=ax.transAxes, ha="right", va="top", fontsize=ft["legend"] * 0.75,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"), zorder=30)
        finish(fig, ax, xs, ys, f"{stem}_全厚み共通_{int(t)}C",
               f"{MAT_NAMES[mat]}　せん断速度と粘度の関係（設定温度 {int(t)} ℃）\n{model_title(fit['key'], fit.get('mode'))}", head + hs)

    # ② 全温度を 1 枚に重ねた総合図（線・帯の色は温度、点の色・形は厚み）
    fig, ax = plt.subplots(figsize=cfg["figsize"])
    xs, ys = [], []
    for t, (x, mu, s, tff_med) in curves.items():
        col = tcol[t]
        ax.fill_between(x, np.exp(mu - 1.96 * s), np.exp(mu + 1.96 * s), color=col, alpha=0.25, zorder=15)
        ax.plot(x, np.exp(mu - 1.96 * s), color=col, lw=0.5, alpha=0.6, zorder=16)
        ax.plot(x, np.exp(mu + 1.96 * s), color=col, lw=0.5, alpha=0.6, zorder=16)
        ax.plot(x, np.exp(mu), color=col, lw=cfg["line"], zorder=20)
        xs += list(x)
        ys += list(np.exp(mu))
    hs = thickness_points(ax, tr, xs, ys)
    head = [Line2D([], [], color=tcol[t], lw=cfg["line"], label=f"予測平均値_{int(t)} ℃") for t in curves]
    head += [Patch(facecolor=tcol[t], alpha=0.25, label=f"{band_name}_{int(t)} ℃") for t in curves]
    head += [Patch(color="none", label="キャビティ厚み")]
    note = [Patch(color="none", label="固定した値（学習データの中央値）"), Patch(color="none", label=f"　P3_3：{p_med:.2f} MPa")]
    note += [Patch(color="none", label=f"　Tff（{int(t)} ℃）：{v[3]:.1f} ℃") for t, v in curves.items()]
    finish(fig, ax, xs, ys, f"{stem}_全厚み共通_全温度", None, head + hs + note, separate_legend=True)
    return saved


def flow_plot(fit, mat, pool, train_conds, out_dir, stem, **kw):
    """せん断速度 ― 粘度の学習結果グラフ（6月のノートのセル9-2 と同じ見せ方。本体の nb_common.hr_flow を、このデータ用に直したもの）。
    "temperature"：設定温度ごとに 1 枚（厚みごとの線）／"thickness"：厚みごとに 1 枚（設定温度ごとの線）。
    塗りつぶしの点は学習（5data）、白抜きは検証（3data）、× は問題データ一覧の「感度確認」。保存したファイルの一覧を返す"""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    cfg = {**FLOW_DEFAULT, **kw}
    dpi = cfg["dpi"] or FIG_DPI
    ts = t_style(pool["Tset"].unique())
    curves = {}
    for (h, t), sub in pool.groupby(["H", "Tset"]):
        lgc = sub.groupby("cond")["ln_gd"].mean()
        if len(lgc) < 2:
            continue
        lo, hi = (np.log(cfg["x_range"][0]), np.log(cfg["x_range"][1])) if cfg["x_range"] else (lgc.min() - np.log(cfg["extend"]), lgc.max() + np.log(cfg["extend"]))
        lg = np.linspace(lo, hi, cfg["n_grid"])
        mu, sd, sdf, _ = predict(fit, flow_inputs(sub, lg, h, t))
        curves[(float(h), float(t))] = (np.exp(lg), mu, sd if cfg["band"] == "predict" else sdf)
    saved, ft = [], cfg["font"]
    band_name = "95% 予測区間" if cfg["band"] == "predict" else "95% 信頼区間"
    for layout_ in cfg["layouts"]:
        by_h = layout_ == "thickness"
        for gv in sorted({k[0] if by_h else k[1] for k in curves}):
            fig, ax = plt.subplots(figsize=cfg["figsize"])
            series, any_test, any_sens, xs, ys = [], False, False, [], []
            for sv in sorted({(k[1] if by_h else k[0]) for k in curves if (k[0] if by_h else k[1]) == gv}):
                h, t = (gv, sv) if by_h else (sv, gv)
                col, mk = ts[t] if by_h else (H_COLORS.get(h, "gray"), H_MARKS.get(h, "x"))
                x, mu, s = curves[(float(h), float(t))]
                if cfg["band"] in ("predict", "latent"):
                    ax.fill_between(x, np.exp(mu - 1.96 * s), np.exp(mu + 1.96 * s), color=col, alpha=0.22, zorder=15)
                ax.plot(x, np.exp(mu), color=col, lw=cfg["line"], zorder=20)
                sub = pool[np.isclose(pool["H"], h) & (pool["Tset"] == t)]
                if cfg["points"] == "cond":
                    a = sub.groupby("cond").agg(lg=("ln_gd", "mean"), le=("ln_eta", "mean"), sens=("is_sens", "max")).reset_index()
                    px, py, sens, ms = np.exp(a["lg"].to_numpy()), np.exp(a["le"].to_numpy()), a["sens"].to_numpy().astype(bool), cfg["marker"]
                    trm = a["cond"].isin(train_conds).to_numpy()
                else:
                    px, py, sens, ms = sub["gd"].to_numpy(), sub["eta"].to_numpy(), sub["is_sens"].to_numpy().astype(bool), cfg["marker_shot"]
                    trm = sub["cond"].isin(train_conds).to_numpy()
                xs += list(x) + list(px)
                ys += list(np.exp(mu)) + list(py)
                ax.scatter(px[trm], py[trm], s=ms, color=col, marker=mk, alpha=0.85, edgecolors="k", linewidths=0.6, zorder=14)
                if cfg["show_test"] and (~trm).any():
                    ax.scatter(px[~trm], py[~trm], s=ms, facecolors="white", edgecolors=col, marker=mk, linewidths=2.5, zorder=14)
                    any_test = True
                if cfg["mark_sens"] and sens.any():
                    ax.scatter(px[sens], py[sens], s=ms * 1.3, color="k", marker="x", linewidths=3, zorder=25)
                    any_sens = True
                series.append((sv, col, mk))
            head = "キャビティ厚み" if not by_h else "設定温度"
            handles = [Patch(facecolor="gray", alpha=0.3, label=band_name), Patch(color="none", label=head)]
            handles += [Line2D([], [], color=c, lw=cfg["line"], marker=m, markersize=np.sqrt(cfg["marker"]) * 0.8, markeredgecolor="k", markeredgewidth=0.6,
                               label=(f"{sv:g} mm" if not by_h else f"{int(sv)} ℃")) for sv, c, m in series]
            if any_test:
                handles.append(Line2D([], [], color="gray", ls="none", marker="o", markersize=np.sqrt(cfg["marker"]) * 0.8, markerfacecolor="white", markeredgewidth=2.5, label="検証（3data。白抜き）"))
            if any_sens:
                handles.append(Line2D([], [], color="k", ls="none", marker="x", markersize=np.sqrt(cfg["marker"]) * 0.8, markeredgewidth=3, label="感度確認の条件"))
            if cfg["band"] not in ("predict", "latent"):
                handles = handles[1:]
            ax.legend(handles=handles, loc="upper right", fontsize=ft["legend"], framealpha=1.0, edgecolor="black")
            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=ft["tick"], pad=8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            xl = cfg["xlim"][layout_] if isinstance(cfg["xlim"], dict) else cfg["xlim"]
            yl = cfg["ylim"][layout_] if isinstance(cfg["ylim"], dict) else cfg["ylim"]
            ax.set_xlim(*((min(xs) / 1.4, max(xs) * 1.4) if xl is None else xl))
            ax.set_ylim(*((min(ys) / 1.5, max(ys) * 1.5) if yl is None else yl))
            for axis, auto in ((ax.xaxis, xl is None), (ax.yaxis, yl is None)):
                if auto:
                    axis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
                    axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
                else:
                    axis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))
                    axis.set_major_formatter(ticker.LogFormatterSciNotation())
                axis.set_minor_formatter(ticker.NullFormatter())
            ax.grid(True, which="major", linestyle="-", alpha=0.6, color="gray", zorder=1)
            ax.grid(True, which="minor", linestyle=":", alpha=0.4, zorder=1)
            gtxt = f"厚み {gv:g} mm" if by_h else f"設定温度 {int(gv)} ℃"
            if cfg["show_title"]:
                ax.set_title(f"{MAT_NAMES[mat]}　せん断速度と粘度の関係（{gtxt}）\n{model_title(fit['key'], fit.get('mode'))}", fontsize=ft["title"], pad=15)
            if cfg["show_xlabel"]:
                ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ (1/s)", fontsize=ft["axis"], labelpad=15)
            if cfg["show_ylabel"]:
                ax.set_ylabel(r"樹脂粘度 $\eta$ (Pa·s)", fontsize=ft["axis"], labelpad=15)
            fig.tight_layout()
            base = f"{stem}_{'厚みごと' if by_h else '設定温度ごと'}_{(f'{gv:g}mm') if by_h else (str(int(gv)) + 'C')}"
            os.makedirs(out_dir, exist_ok=True)
            for fmt in cfg["formats"]:
                path = os.path.join(out_dir, f"{base}.{fmt}")
                fig.savefig(path, bbox_inches="tight", dpi=dpi)
                saved.append(path)
            if HR_ALL and dpi < HR_ALL_DPI:
                os.makedirs(os.path.join(out_dir, "highres"), exist_ok=True)
                fig.savefig(os.path.join(out_dir, "highres", f"{base}.png"), bbox_inches="tight", dpi=HR_ALL_DPI)
            if SHOW_FIGS if cfg["show"] is None else cfg["show"]:
                plt.show()
            plt.close(fig)
    return saved
