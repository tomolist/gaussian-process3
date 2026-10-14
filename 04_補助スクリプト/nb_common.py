# -*- coding: utf-8 -*-
"""ノートブックで共通に使う部品（データの読み込み、問題データの除外、モデル、評価）。

各手順のノートには、このファイルの中身がそのまま「共通の部品」のセルとして入る（ノート単独で動かすため）。
直すときは、このファイルを直して、ノートを作り直す（04_補助スクリプト/build_step*_nb.py）。
モジュールとして import しても使える（check_data.py など）。その場合は、nb_common.D = nb_common.load_all() でデータを入れる。

設定値（SEED など）は、ノートの「設定エリア」で先に決めた値があれば、それを使う。
"""
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, DotProduct, Hyperparameter, Kernel, Matern, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- 設定（ノートの設定エリアの値があれば、それを使う）
SEED = globals().get("SEED", 42)
NOISE_LB = globals().get("NOISE_LB", 1e-6)                 # ノイズ分散の下限（CLAUDE.md §5。旧ノートは 1e-3）
N_RESTARTS = globals().get("N_RESTARTS", 5)                # GPR の最適化のやり直し回数
REPO = globals().get("REPO", os.environ.get("NB_REPO", ".."))   # リポジトリ直下（ノートは 01_機械学習コードフォルダ にあるので ".."。スクリプトは環境変数 NB_REPO で指定）
USE_EXCLUSIONS = globals().get("USE_EXCLUSIONS", True)     # 問題データ一覧の「除外」を使うか
warnings.filterwarnings("ignore", category=ConvergenceWarning)   # 下限への張り付きは kernels.csv に記録する

# ---------------------------------------------------------------- データの場所
DATA_DIR = os.path.join(REPO, "02_機械学習データ", "NR500")
TEST_DIR = os.path.join(REPO, "03_予測検証用テストデータ", "NR500")
PROBLEM_CSV = os.path.join(REPO, "02_機械学習データ", "問題データ一覧.csv")
MATS = {"PP": "PP", "MIX": "粉砕材混合PP"}
FILES = {   # (材料, 組): (上位フォルダ, 中間フォルダ, フォルダ, ファイル名)。どれも全条件のファイル（問題のある条件は、一覧で除く）
    ("PP", "5data"): (DATA_DIR, "02_バージンPP再々実験_5data", "03_pp_max_5data", "結合_03_pp_max_5data_外れ値{o}.csv"),
    ("PP", "3data"): (TEST_DIR, "02_バージンPP再々実験_5data用検証データ(3data)", "02_pp_max_3data", "結合_02_pp_max_3data_外れ値{o}.csv"),
    ("PP", "Alldata"): (DATA_DIR, "01_バージンPP再々実験_alldata", "01_pp_max_Alldata", "結合_01_pp_max_Alldata_外れ値{o}.csv"),
    ("MIX", "5data"): (DATA_DIR, "04_粉砕材混合PP実験_5data", "03_mixpp_max_5data", "結合_03_mixpp_max_5data_外れ値{o}.csv"),
    ("MIX", "3data"): (TEST_DIR, "04_粉砕材混合PP実験_5data用検証データ(3data)", "02_mixpp_max_3data", "結合_02_mixpp_max_3data_外れ値{o}.csv"),
    ("MIX", "Alldata"): (DATA_DIR, "03_粉砕材混合PP実験_alldata", "01_mixpp_max_Alldata", "結合_01_mixpp_max_Alldata_外れ値{o}.csv"),
}
LEGACY_FILES = {   # 1.5 mm・3 mm/s を除いてあるファイル（no1.5-3.0）。一覧の「除外」で同じデータになるかの確認用
    ("PP", "3data"): (TEST_DIR, "02_バージンPP再々実験_5data用検証データ(3data)", "05_pp_max_3data(no1.5-3.0)", "結合_05_pp_max_3data(no1.5-3.0)_外れ値{o}.csv"),
    ("PP", "Alldata"): (DATA_DIR, "01_バージンPP再々実験_alldata", "04_pp_max_Alldata(no1.5-3.0)", "結合_04_pp_max_Alldata(no1.5-3.0)_外れ値{o}.csv"),
}
COLS = {"元ファイル": "file", "キャビ厚": "H", "射出速度[mm/s]": "v", "樹脂粘度η34(Pa・s)": "eta", "せん断速度γ34(1/s)": "gd",
        "流動中樹脂圧力P3_1(MPa)": "P31", "流動中樹脂圧力P3_2(MPa)": "P32", "加熱シリンダ温度[℃]": "Tset", "流動中樹脂温度(℃)": "Tff"}
RE_SRC = r"^(\d{3})_(\d+)℃_(\d{3})_([\d.]+)mm\.csv$"
D = {}                  # (材料, 組, 外れ値) → DataFrame。ノートで D = load_all() として入れる
REMOVED = {}            # (材料, 組, 外れ値) → 一覧の「除外」で除いた行数


def path_of(mat, grp, o, files=None):
    top, d1, d2, fn = (files or FILES)[(mat, grp)]
    return os.path.join(top, d1, d2, f"外れ値{o}", fn.format(o=o))


# ---------------------------------------------------------------- 問題データ一覧
def read_problem_list(path=None):
    """問題データ一覧.csv を読む（空欄は空文字）"""
    return pd.read_csv(path or PROBLEM_CSV, encoding="utf-8-sig", dtype=str).fillna("")


def problem_mask(df, mat, grp, plist, kinds):
    """区分が kinds のどれかで、再測定が「済み」でない一覧の行のうち、この材料・組・条件に当てはまる行の真偽値と、その ID を返す"""
    mask = np.zeros(len(df), bool)
    ids = [[] for _ in range(len(df))]
    for _, r in plist.iterrows():
        if r["区分"] not in kinds or r["再測定"] == "済み" or r["データ"] not in ("新", ""):
            continue
        if r["材料"] not in ("すべて", "", MATS[mat]) or r["組"] not in ("", grp):
            continue
        m = np.ones(len(df), bool)
        if r["設定温度"] != "":
            m &= df["Tset"].to_numpy() == float(r["設定温度"])
        if r["厚み"] != "":
            m &= np.isclose(df["H"].to_numpy(), float(r["厚み"]))
        if r["射出速度"] != "":
            m &= df["v"].to_numpy() == float(r["射出速度"])
        if r.get("ショット", "") != "":                      # 空欄は条件全体。数字なら、そのショット（元ファイル名の先頭 3 桁）だけ
            m &= (df["shot"].to_numpy() == int(r["ショット"])) if "shot" in df.columns else np.zeros(len(df), bool)
        mask |= m
        for i in np.flatnonzero(m):
            ids[i].append(r["ID"])
    return mask, [";".join(x) for x in ids]


def first_shots(mat, grp):
    """条件ごとの、最初のショットの番号（外れ値ありのファイル。ショットが除かれる前の番号）。ショットの順番 pos の基準に使う"""
    ref = pd.read_csv(path_of(mat, grp, "あり"), encoding="utf-8-sig").rename(columns=COLS)
    ref["shot0"] = ref["file"].str.extract(RE_SRC)[0].astype(int)
    ref["Tset"] = ref["Tset"].astype(float)
    return ref.groupby(["Tset", "H", "v"], as_index=False)["shot0"].min()


def load(mat, grp, o, plist=None, exclude=None):
    """結合ファイルを読み、形式を確かめ、問題データ一覧の「除外」を除く。列 sens には、「感度確認」の ID が入る（なければ空文字）。
    列 shot は元ファイル名の先頭 3 桁（設定温度・厚みごとの通し番号で、同じ条件のショットは連続した番号）。列 pos は、その条件の中での順番（最初のショット = 1。外れ値で除かれたショットの番号は欠ける）"""
    exclude = USE_EXCLUSIONS if exclude is None else exclude
    df = pd.read_csv(path_of(mat, grp, o), encoding="utf-8-sig")
    assert list(df.columns) == list(COLS), f"列名が想定と違う: {list(df.columns)}"
    df = df.rename(columns=COLS)
    assert df.isna().sum().sum() == 0, "欠損がある"
    p = df["file"].str.extract(RE_SRC)
    assert p.notna().all().all(), "ファイル名の形式に合わない行がある"
    assert np.allclose(p[1].astype(float), df["Tset"]) and np.allclose(p[2].astype(float), df["v"]) \
        and np.allclose(p[3].astype(float), df["H"]), "ファイル名と列の値が一致しない"
    df["shot"] = p[0].astype(int)
    df["Tset"] = df["Tset"].astype(float)
    df = df.merge(first_shots(mat, grp), on=["Tset", "H", "v"], how="left")
    assert df["shot0"].notna().all(), "外れ値ありのファイルにない条件がある"
    df["pos"] = df["shot"] - df["shot0"].astype(int) + 1
    df = df.drop(columns="shot0")
    plist = read_problem_list() if plist is None else plist
    n0 = len(df)
    if exclude:
        drop, _ = problem_mask(df, mat, grp, plist, ("除外",))
        df = df[~drop].reset_index(drop=True)
    REMOVED[(mat, grp, o)] = n0 - len(df)
    _, df["sens"] = problem_mask(df, mat, grp, plist, ("感度確認",))
    df["is_sens"] = df["sens"] != ""
    df["cond"] = df["Tset"].astype(int).astype(str) + "_" + df["H"].astype(str) + "_" + df["v"].astype(int).astype(str)
    df["ln_eta"], df["ln_gd"], df["ln_H"] = np.log(df["eta"]), np.log(df["gd"]), np.log(df["H"])
    df["ln_P31"], df["ln_P32"] = np.log(df["P31"] * 1e6), np.log(df["P32"] * 1e6)   # CSV の圧力は MPa。Pa に直してから ln をとる
    df["ln_v"] = np.log(df["v"])
    df["ln_Tff"] = np.log(df["Tff"] + 273.15)          # 流動先端温度 [K]
    for c in ["gd", "Tff", "P32"]:   # 条件ごとの平均（成形条件から見たばらつきを見るモデル用。cm は condition mean）
        df[f"ln_{c}_cm"] = df.groupby("cond")[f"ln_{c}"].transform("mean")
    return df


def load_all(exclude=None):
    """全材料・全組・外れ値なし/あり を読む。学習と検証で条件が重ならないことも確かめる"""
    plist = read_problem_list()
    out = {(mat, grp, o): load(mat, grp, o, plist, exclude) for mat in MATS for grp in ["5data", "3data", "Alldata"] for o in ["なし", "あり"]}
    for mat in MATS:
        for o in ["なし", "あり"]:
            assert not (set(out[(mat, "5data", o)]["cond"]) & set(out[(mat, "3data", o)]["cond"])), "学習と検証で条件が重なっている"
    return out


def legacy_check(D_):
    """全条件のファイルから一覧で除いたデータが、1.5 mm・3 mm/s を除いてあるファイル（no1.5-3.0）と同じ行か（PP の 3data・Alldata）。
    no1.5-3.0 のファイルにも、1.5 mm・3 mm/s 以外の一覧の「除外」（P03 など）を同じように適用してから比べる"""
    rows = []
    plist = read_problem_list()
    plist = plist[~((plist["厚み"] == "1.5") & (plist["射出速度"] == "3") & (plist["設定温度"] == ""))]      # no1.5-3.0 のファイルで、すでに除かれている分
    for (mat, grp), _ in LEGACY_FILES.items():
        for o in ["なし", "あり"]:
            old = pd.read_csv(path_of(mat, grp, o, LEGACY_FILES), encoding="utf-8-sig")
            o2 = old.rename(columns=COLS)
            o2["shot"] = o2["file"].str.extract(RE_SRC)[0].astype(int)
            o2["Tset"] = o2["Tset"].astype(float)
            drop, _ = problem_mask(o2, mat, grp, plist, ("除外",)) if USE_EXCLUSIONS else (np.zeros(len(o2), bool), None)
            old = old[~drop]
            same = set(old["元ファイル"]) == set(D_[(mat, grp, o)]["file"])
            rows.append({"材料": MATS[mat], "組": grp, "外れ値": o, "一覧で除いた後の行数": len(D_[(mat, grp, o)]), "no1.5-3.0 の行数": len(old), "同じ行": same})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- モデル
def kernel_for(d, linear=False):
    """ARD Matern 5/2 ＋ ノイズ。linear=True なら、線形カーネル（DotProduct）も足す（低速側への外挿のため。CLAUDE.md §5）"""
    k = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0] * d, length_scale_bounds=(1e-2, 1e2), nu=2.5)
    if linear:
        k = k + C(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-5, 1e3))
    return k + WhiteKernel(noise_level=0.1, noise_level_bounds=(NOISE_LB, 1e1))


class OnDims(Kernel):
    """入力の一部の列 dims にだけ当てるカーネル（足し算のカーネル m(…) + c(…) を作るための部品。手順5）。
    ハイパーパラメータは、中のカーネルのものをそのまま使う（scikit-learn の Exponentiation と同じ作り）。tag は、どの部品かの印（"m"・"c" など）"""

    def __init__(self, kernel, dims, tag=""):
        self.kernel = kernel
        self.dims = dims                                   # 列番号のリスト（scikit-learn の clone の決まりで、受け取ったものをそのまま持つ）
        self.tag = tag

    def get_params(self, deep=True):
        params = dict(kernel=self.kernel, dims=self.dims, tag=self.tag)
        if deep:
            params.update(("kernel__" + k, val) for k, val in self.kernel.get_params().items())
        return params

    @property
    def hyperparameters(self):
        return [Hyperparameter("kernel__" + h.name, h.value_type, h.bounds, h.n_elements) for h in self.kernel.hyperparameters]

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        self.kernel.theta = theta

    @property
    def bounds(self):
        return self.kernel.bounds

    def __eq__(self, b):
        return type(self) == type(b) and self.dims == b.dims and self.kernel == b.kernel

    def __hash__(self):
        return hash((tuple(self.dims), self.tag))

    def __call__(self, X, Y=None, eval_gradient=False):
        Xd = np.asarray(X)[:, self.dims]
        Yd = None if Y is None else np.asarray(Y)[:, self.dims]
        return self.kernel(Xd, Yd, eval_gradient=eval_gradient)

    def diag(self, X):
        return self.kernel.diag(np.asarray(X)[:, self.dims])

    def is_stationary(self):
        return self.kernel.is_stationary()

    def __repr__(self):
        return f"OnDims({self.tag}, dims={self.dims}, {self.kernel!r})"


def make_gpr(d, linear=False, restarts=None):
    return GaussianProcessRegressor(kernel=kernel_for(d, linear), n_restarts_optimizer=N_RESTARTS if restarts is None else restarts, random_state=SEED)


def fit_ols(X_tr, y_tr):
    """切片ありの最小二乗回帰。係数、(XᵀX)⁻¹、残差分散を返す"""
    A_tr = np.column_stack([np.ones(len(X_tr)), X_tr])
    beta, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
    resid = y_tr - A_tr @ beta
    return beta, np.linalg.inv(A_tr.T @ A_tr), resid @ resid / (len(y_tr) - A_tr.shape[1])


def predict_ols(model, X):
    """予測平均と、新しい1ショットに対する予測標準偏差"""
    beta, XtX_inv, s2 = model
    A_new = np.column_stack([np.ones(len(X)), X])
    lev = np.einsum("ij,jk,ik->i", A_new, XtX_inv, A_new)
    return A_new @ beta, np.sqrt(s2 * (1.0 + lev))


MODELS = {}             # 記号 → {"kind": "ols" か "gpr", "cols": [入力の列], "linear": 線形カーネルを足すか}。ノートで決める


def fit_model(key, tr, restarts=None):
    spec = MODELS[key]
    X, y = tr[spec["cols"]].to_numpy(), tr["ln_eta"].to_numpy()
    if spec["kind"] == "ols":
        return {"kind": "ols", "model": fit_ols(X, y)}
    sc = StandardScaler().fit(X)              # 標準化と中心化は学習データだけで計算する
    y0 = y.mean()
    gp = make_gpr(X.shape[1], spec.get("linear", False), restarts).fit(sc.transform(X), y - y0)
    return {"kind": "gpr", "gp": gp, "sc": sc, "y0": y0}


def predict_model(key, fit, te):
    """予測平均、予測標準偏差（ノイズ込み）、潜在関数の標準偏差、ノイズの標準偏差"""
    X = te[MODELS[key]["cols"]].to_numpy()
    if fit["kind"] == "ols":
        mu, sd = predict_ols(fit["model"], X)
        return mu, sd, np.full(len(X), np.nan), float(np.sqrt(fit["model"][2]))
    mu, sd = fit["gp"].predict(fit["sc"].transform(X), return_std=True)
    noise = fit["gp"].kernel_.k2.noise_level
    return mu + fit["y0"], sd, np.sqrt(np.maximum(sd ** 2 - noise, 0.0)), float(np.sqrt(noise))


def evaluate(y, mu, sd):
    """ln η の指標：R2、RMSE、95% 予測区間の被覆率、NLPD（各ショットの予測分布の負の対数密度の平均）、予測標準偏差の平均"""
    z = (y - mu) / sd
    return {"R2": r2_score(y, mu) if len(y) > 1 else np.nan, "RMSE": float(np.sqrt(mean_squared_error(y, mu))),
            "coverage95": float(np.mean(np.abs(z) <= 1.96)),
            "NLPD": float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + 0.5 * z ** 2)), "mean_sd": float(np.mean(sd))}


def kernel_row(key, fit):
    """学習した GPR のハイパーパラメータ（振幅、長さスケール、ノイズ、線形項、対数周辺尤度、下限・上限への張り付き）"""
    gp = fit["gp"]
    k = gp.kernel_
    lin = MODELS[key].get("linear", False)
    prod, lin_prod = (k.k1.k1, k.k1.k2) if lin else (k.k1, None)
    ls = np.atleast_1d(prod.k2.length_scale)
    row = {"amplitude": float(np.sqrt(prod.k1.constant_value)), "noise_var": float(k.k2.noise_level),
           "noise_sd": float(np.sqrt(k.k2.noise_level)), "LML": float(gp.log_marginal_likelihood_value_),
           "noise_at_lower_bound": bool(k.k2.noise_level < NOISE_LB * 1.05),
           "ls_at_bound": bool(np.any(ls > 99) or np.any(ls < 0.0105))}
    if lin_prod is not None:
        row.update({"lin_amplitude": float(np.sqrt(lin_prod.k1.constant_value)), "lin_sigma0": float(lin_prod.k2.sigma_0)})
    row.update({f"ls_{c}": float(l) for c, l in zip(MODELS[key]["cols"], ls)})
    return row


def within_stats(df):
    """同じ条件内のばらつき（プール）。dof：自由度、var_after_gd：ln γ̇ で説明した残りの分散"""
    g = df.groupby("cond")
    dof = int((g.size() - 1).sum())
    de = df["ln_eta"] - g["ln_eta"].transform("mean")
    dg = df["ln_gd"] - g["ln_gd"].transform("mean")
    var_e = float((de ** 2).sum() / dof)
    b = float((de * dg).sum() / (dg ** 2).sum())
    return {"dof": dof, "var_ln_eta": var_e, "sd_ln_eta": var_e ** 0.5, "corr_with_ln_gd": float((de * dg).sum() / np.sqrt((de ** 2).sum() * (dg ** 2).sum())),
            "slope_vs_ln_gd": b, "var_after_gd": float(((de - b * dg) ** 2).sum() / (dof - 1))}


# ---------------------------------------------------------------- データの分け方
def split_V(mat, o):
    """5data で学習し、3data で検証する（射出速度の両端の外挿）。te の group 列は、低速側（3・5 mm/s）か高速側（280・320 mm/s）"""
    tr, te = D[(mat, "5data", o)].copy(), D[(mat, "3data", o)].copy()
    te["group"] = np.where(te["v"] <= 5, "低速側", np.where(te["v"] >= 280, "高速側", "その他"))
    assert (te["group"] != "その他").all(), "検証データに、低速側・高速側のどちらでもない条件がある"
    return tr, te


def split_random(df):
    """行単位のランダム 20%（従来の分け方。random_state = SEED）"""
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED)
    te = df.iloc[idx_te].copy()
    te["group"] = "全体"
    return df.iloc[idx_tr].copy(), te


def split_T(mat, o, t_test=210):
    """設定温度 t_test の全条件をテストにし、5data の残りの温度で学習する（温度の内挿の検証）。3data の同じ温度は te3 として別に返す"""
    d5, d3 = D[(mat, "5data", o)], D[(mat, "3data", o)]
    tr, te = d5[d5["Tset"] != t_test].copy(), d5[d5["Tset"] == t_test].copy()
    te["group"] = "全体"
    te3 = d3[d3["Tset"] == t_test].copy()
    te3["group"] = np.where(te3["v"] <= 5, "低速側", "高速側")
    return tr, te, te3


def pool_of(mat, o):
    """5data と 3data をまとめたデータ"""
    return pd.concat([D[(mat, "5data", o)], D[(mat, "3data", o)]], ignore_index=True)


# ---------------------------------------------------------------- 学習と評価の記録
METRICS, KERNELS, OLSROWS, PRED, FITS = [], [], [], {}, {}      # FITS：(材料, split, データ, モデル) → 学習した予測器
SENS_SUFFIX = "（感度確認の条件を除く）"


def _record(mat, name, data, key, fit, tr, te, pure):
    """学習した予測器 fit で te を予測し、指標（subset ごと）と予測を記録する"""
    subsets = {"全体": np.ones(len(te), bool)}
    for g in te["group"].unique():
        if g != "全体":
            subsets[g] = (te["group"] == g).to_numpy()
    if te["is_sens"].any():
        subsets.update({s + SENS_SUFFIX: m & ~te["is_sens"].to_numpy() for s, m in list(subsets.items())})
    y_te = te["ln_eta"].to_numpy()
    mu, sd, sd_f, noise_sd = predict_model(key, fit, te)
    PRED[(mat, name, data, key)] = pd.DataFrame({"y": y_te, "mu": mu, "sd": sd, "sd_f": sd_f, "v": te["v"].to_numpy(), "H": te["H"].to_numpy(),
                                                 "Tset": te["Tset"].to_numpy(), "cond": te["cond"].to_numpy(), "group": te["group"].to_numpy(),
                                                 "is_sens": te["is_sens"].to_numpy()})
    for sub, mask in subsets.items():
        if mask.sum() == 0:
            continue
        ev = evaluate(y_te[mask], mu[mask], sd[mask])
        METRICS.append({"material": MATS[mat], "split": name, "data": data, "model": key, "subset": sub, "n_train": len(tr),
                        "n_test": int(mask.sum()), "n_cond_test": int(te.loc[mask, "cond"].nunique()), **ev,
                        "mean_sd_latent": float(np.nanmean(sd_f[mask])) if fit["kind"] == "gpr" else np.nan, "noise_sd": noise_sd})


def run_split(mat, split, data, tr, te, keys=None, tag="", extra_te=None):
    """全モデルを学習し、指標（subset ごと）・カーネル・予測を記録する。te には group 列がある。
    te に「感度確認」の条件があれば、各 subset から除いた指標（subset 名の後ろに SENS_SUFFIX）も記録する。
    tag は split 名の後ろに付ける印。extra_te（{split 名: テスト}）があれば、同じ学習で、それらも評価する"""
    keys = keys or ALL_KEYS
    name = split + tag
    tests = {name: te, **{s + tag: t for s, t in (extra_te or {}).items()}}
    pure = within_stats(tr)
    for key in keys:
        t0 = time.time()
        fit = fit_model(key, tr)
        FITS[(mat, name, data, key)] = fit
        for nm, te_ in tests.items():
            _record(mat, nm, data, key, fit, tr, te_, pure)
        base = {"material": MATS[mat], "split": name, "data": data, "model": key, "n_train": len(tr)}
        if fit["kind"] == "gpr":
            KERNELS.append({**base, **kernel_row(key, fit), "pure_sd_ln_eta": pure["sd_ln_eta"], "pure_sd_after_gd": pure["var_after_gd"] ** 0.5})
        else:
            names = ["intercept"] + MODELS[key]["cols"]
            OLSROWS.append({**base, "resid_sd": float(np.sqrt(fit["model"][2])), **dict(zip(names, map(float, fit["model"][0])))})
        print(f"{MATS[mat]} {name} {data} {key}: 学習 {len(tr)} 行、{time.time() - t0:.1f} 秒")


# ---------------------------------------------------------------- 条件を1つずつ抜く交差検証（LOCO）
def loco_gpr(key, pool):
    """GPR：ハイパーパラメータを全データで決めて固定し、条件を抜いた予測を閉じた式で計算する（平均 = y_g − [K⁻¹_gg]⁻¹(K⁻¹y)_g、共分散 = [K⁻¹_gg]⁻¹）"""
    spec = MODELS[key]
    X, y = pool[spec["cols"]].to_numpy(), pool["ln_eta"].to_numpy()
    sc = StandardScaler().fit(X)
    Xs, yc = sc.transform(X), y - y.mean()
    gp = make_gpr(X.shape[1], spec.get("linear", False)).fit(Xs, yc)
    Kinv = np.linalg.inv(gp.kernel_(Xs))              # ノイズ込みの共分散行列の逆行列
    alpha = Kinv @ yc
    mu, sd = np.empty(len(y)), np.empty(len(y))
    for idx in pool.groupby("cond").indices.values():
        Minv = np.linalg.inv(Kinv[np.ix_(idx, idx)])
        mu[idx] = y[idx] - Minv @ alpha[idx]
        sd[idx] = np.sqrt(np.diag(Minv))
    return mu, sd, gp


def loco_ols(key, pool):
    """線形回帰：条件ごとに、その条件を除いて当てはめ直す"""
    cols = MODELS[key]["cols"]
    X, y = pool[cols].to_numpy(), pool["ln_eta"].to_numpy()
    mu, sd = np.empty(len(y)), np.empty(len(y))
    for idx in pool.groupby("cond").indices.values():
        keep = np.ones(len(y), bool)
        keep[idx] = False
        mu[idx], sd[idx] = predict_ols(fit_ols(X[keep], y[keep]), X[idx])
    return mu, sd


# ---------------------------------------------------------------- 外れ値の検知（規則 f）と、データの版（5data 版・Alldata 版）
OUTLIER_Z = globals().get("OUTLIER_Z", 3.5)             # 規則 f：|z| がこれを超え、かつ |残差|（ln η）が次の値を超える条件を候補にする
OUTLIER_RESID = globals().get("OUTLIER_RESID", 0.1)     # 0.1 は粘度で約 10%


def cond_means(df):
    """条件ごとの平均（ln η と ln γ̇ をショットで平均した点）"""
    g = df.groupby("cond", sort=False)
    t = g.agg(Tset=("Tset", "first"), H=("H", "first"), v=("v", "first"), ln_eta=("ln_eta", "mean"), ln_gd=("ln_gd", "mean"), n=("ln_eta", "size")).reset_index()
    t["ln_H"] = np.log(t["H"])
    return t


def outlier_scan(df, z_thr=None, resid_thr=None):
    """規則 f：条件平均が、ほかの条件から予測した値と大きく違う条件を拾う。
    条件平均の GPR（入力 ln γ̇（条件平均）・ln H・Tset。R3 と同じ形）の、条件を 1 つずつ抜いた予測（閉じた式）との残差を、予測標準偏差で割って z とする。
    返す表の列：cond, Tset, H, v, n（ショット数）, resid（ln η の実測 − 予測）, sd, z, flag（|z| と |resid| が両方しきい値を超えたか）"""
    z_thr = OUTLIER_Z if z_thr is None else z_thr
    resid_thr = OUTLIER_RESID if resid_thr is None else resid_thr
    t = cond_means(df)
    X, y = t[["ln_gd", "ln_H", "Tset"]].to_numpy(), t["ln_eta"].to_numpy()
    Xs, yc = StandardScaler().fit_transform(X), y - y.mean()
    gp = make_gpr(3, True).fit(Xs, yc)
    Kinv = np.linalg.inv(gp.kernel_(Xs))
    d = np.diag(Kinv)
    t["resid"], t["sd"] = (Kinv @ yc) / d, np.sqrt(1.0 / d)           # 条件 i を抜いた予測：平均 = y_i − (K⁻¹y)_i / K⁻¹_ii、分散 = 1 / K⁻¹_ii
    t["z"] = t["resid"] / t["sd"]
    t["flag"] = (t["z"].abs() > z_thr) & (t["resid"].abs() > resid_thr)
    return t[["cond", "Tset", "H", "v", "n", "resid", "sd", "z", "flag"]]


DATA_VERSIONS = globals().get("DATA_VERSIONS", ["5data", "Alldata"])   # 検討するデータの版。"5data"：5data で学習し、3data で外挿を検証する版／"Alldata"：全条件で学習する版
CV_K = globals().get("CV_K", 10)                                       # Alldata 版の交差検証の分割数（条件単位）


def split_cv(pool, k=None):
    """条件単位の k 分割交差検証。同じ条件のショットは同じ側に入れる（SEED で条件を並べ替える）。(学習, テスト) の組の一覧を返す"""
    k = CV_K if k is None else k
    conds = np.array(sorted(pool["cond"].unique()))
    np.random.RandomState(SEED).shuffle(conds)
    out = []
    for i in range(k):
        te_c = set(conds[i::k])
        te = pool[pool["cond"].isin(te_c)].copy()
        te["group"] = "全体"
        out.append((pool[~pool["cond"].isin(te_c)].copy(), te))
    return out


def version_sets(mat, version, data="なし", t_test=210):
    """データの版ごとの分割を返す。{"pool": 最終モデル（ALL）の学習データ, "splits": {分割の名前: (学習, {集計の名前: テスト})}}
    5data 版：S_V（5data → 3data。低速側・高速側）、S_T（190・230 ℃ → 210 ℃。5data の 210 ℃ と 3data の 210 ℃ を別に集計）、CV（5data の中の条件単位の CV_K 分割）。最終モデルは 5data（3data は検証専用）。
    Alldata 版：S_T（Alldata の 190・230 ℃ で学習。テストは 5data 版と同じ行：5data ファイルの 210 ℃ と 3data ファイルの 210 ℃。そのまま比べられる）、
    S_Vall（Alldata のうち 5data の条件だけで学習 → 3data の条件を検証。PP のみ意味がある〔粉砕材混合 PP は 5data と同じ〕）、CV（Alldata の条件単位の CV_K 分割）。最終モデルは Alldata（全条件）"""
    d5, d3 = D[(mat, "5data", data)], D[(mat, "3data", data)]
    if version == "5data":
        tr_v, te_v = split_V(mat, data)
        tr_t, te_t, te_t3 = split_T(mat, data, t_test)
        sp = {"S_V": (tr_v, {"S_V": te_v}), "S_T": (tr_t, {"S_T": te_t, "S_T3": te_t3})}
        pool = d5
    else:
        assert version == "Alldata", version
        pool = D[(mat, "Alldata", data)]
        _, te_t, te_t3 = split_T(mat, data, t_test)
        sp = {"S_T": (pool[pool["Tset"] != t_test].copy(), {"S_T": te_t, "S_T3": te_t3})}
        te_v = d3.copy()
        te_v["group"] = np.where(te_v["v"] <= 5, "低速側", np.where(te_v["v"] >= 280, "高速側", "その他"))
        sp["S_Vall"] = (pool[pool["cond"].isin(set(d5["cond"]))].copy(), {"S_Vall": te_v})
    for i, (tr_i, te_i) in enumerate(split_cv(pool), 1):
        sp[f"CV{i}"] = (tr_i, {f"CV{i}": te_i})
    return {"pool": pool, "splits": sp}


# ---------------------------------------------------------------- 判定の表示（GPR6月ノートの「判定」にならう）
def judge(ok, text):
    """結果の判定を1行で表示する。ok が True なら「OK」、False なら「要確認」"""
    print(f">> 判定：{'OK' if ok else '★要確認'}　{text}")


# ---------------------------------------------------------------- 高解像度のグラフ（GPR6月ノートのセル9-2 にならう。報告書・発表資料用）
HR_COLORS = {"H": {0.5: "#d62728", 1.0: "#1f77b4", 1.5: "#f0b400", 2.0: "#2ca02c"},       # 厚みごとの色（6月のノートと同じ並び。黄は、線で見やすい濃さにした）
             "Tset": {190: "#1f77b4", 210: "#2ca02c", 230: "#d62728"}}                       # 設定温度ごとの色
HR_MARKS = {"H": {0.5: "o", 1.0: "s", 1.5: "^", 2.0: "D"}, "Tset": {190: "o", 210: "s", 230: "^"}}
HR_DEFAULT = dict(layouts=["thickness", "temperature"], band="predict", points="cond", show_test=True, mark_sens=True, x_range=None, extend=1.3,
                  xlim={"thickness": None, "temperature": (30, 50000)}, ylim={"thickness": None, "temperature": (0.5, 5000)}, figsize=(15, 10), dpi=600, font={"axis": 24, "legend": 24, "title": 24, "tick": 28},
                  marker=200, marker_shot=70, line=3.0, show_title=True, show_xlabel=False, show_ylabel=False, formats=["png"], show=False, n_grid=120)


def hr_split_conds(mat, data, split, t_test=210):
    """学習に使った条件の集合。S_V：5data／S_T：5data の t_test 以外の温度／ALL：5data と 3data"""
    d5, d3 = D[(mat, "5data", data)], D[(mat, "3data", data)]
    if split == "S_V":
        return set(d5["cond"])
    if split == "S_T":
        return set(d5.loc[d5["Tset"] != t_test, "cond"])
    return set(d5["cond"]) | set(d3["cond"])


def hr_inputs(sub, lg, h, t):
    """予測線の入力（ln γ̇ = lg の点）。Tff・P3_2・射出速度は、同じ設定温度・厚みの条件平均を ln γ̇ で内挿する（どの入力のモデルでも描けるように）"""
    cm = sub.groupby("cond").agg(lg=("ln_gd", "mean"), tff=("ln_Tff", "mean"), p32=("ln_P32", "mean"), lv=("ln_v", "mean")).sort_values("lg")
    ip = lambda c: np.interp(lg, cm["lg"], cm[c])
    return pd.DataFrame({"ln_gd": lg, "ln_gd_cm": lg, "ln_H": np.log(h), "Tset": float(t), "ln_Tff": ip("tff"), "ln_Tff_cm": ip("tff"),
                         "ln_P32": ip("p32"), "ln_P32_cm": ip("p32"), "ln_v": ip("lv")})


HR_ALL = globals().get("HR_ALL", True)                 # すべての図を、高解像度でも保存するか（True：out_dir/highres/ に HR_ALL_DPI で保存。Git の管理外）
HR_ALL_DPI = globals().get("HR_ALL_DPI", 600)


def save_fig(fig, out_dir, stem):
    """図を out_dir に PNG で保存する（解像度は FIG_DPI）。HR_ALL が True なら、同じ図を HR_ALL_DPI（既定 600 dpi）で out_dir/highres/ にも保存する。保存したファイルの一覧を返す"""
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, f"{stem}.png")]
    fig.savefig(paths[0], dpi=globals().get("FIG_DPI", 200), bbox_inches="tight")
    if HR_ALL:
        os.makedirs(os.path.join(out_dir, "highres"), exist_ok=True)
        paths.append(os.path.join(out_dir, "highres", f"{stem}.png"))
        fig.savefig(paths[1], dpi=HR_ALL_DPI, bbox_inches="tight")
    return paths


def hr_flow(predict, mat, data, train_conds, title_model, stem, out_dir, pool=None, **kw):
    """せん断速度 ― 粘度の学習結果グラフを、高解像度で 1 枚ずつ保存する（6月のノートのセル9-2 と同じ見せ方）。
    predict(Xg) → (予測平均 ln η, 予測標準偏差（ノイズ込み）, 潜在関数の標準偏差)。Xg は hr_inputs の DataFrame。
    train_conds：学習に使った条件の集合（それ以外は、検証の点として白抜きで重ねる）。pool：点と曲線の範囲に使うデータ（None なら 5data＋3data）。
    設定は HR_DEFAULT を参照（kw で上書き）。保存したファイルの一覧を返す"""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    cfg = {**HR_DEFAULT, **kw}
    os.makedirs(out_dir, exist_ok=True)
    pool = pool_of(mat, data) if pool is None else pool
    curves = {}                                                         # (厚み, 設定温度) → (x, 予測平均, 帯の幅の標準偏差)。レイアウトが違っても同じ線なので、1 回だけ計算する
    for (h, t), sub in pool.groupby(["H", "Tset"]):
        lgc = sub.groupby("cond")["ln_gd"].mean()
        if len(lgc) < 2:
            continue
        lo, hi = (np.log(cfg["x_range"][0]), np.log(cfg["x_range"][1])) if cfg["x_range"] else (lgc.min() - np.log(cfg["extend"]), lgc.max() + np.log(cfg["extend"]))
        lg = np.linspace(lo, hi, cfg["n_grid"])
        mu, sd, sdf = predict(hr_inputs(sub, lg, h, t))
        curves[(float(h), int(t))] = (np.exp(lg), np.asarray(mu), np.asarray(sd if cfg["band"] == "predict" else sdf))
    saved = []
    ft = cfg["font"]
    band_name = "95% 予測区間" if cfg["band"] == "predict" else "95% 信頼区間"
    for layout in cfg["layouts"]:
        by, sk = ("H", "Tset") if layout == "thickness" else ("Tset", "H")        # by：1 枚ごとに固定する量、sk：1 枚の中で線を分ける量
        for gv in sorted({k[0] if by == "H" else k[1] for k in curves}):
            fig, ax = plt.subplots(figsize=cfg["figsize"], dpi=cfg["dpi"])
            series, any_test, any_sens, xs, ys = [], False, False, [], []
            for sv in sorted({(k[1] if by == "H" else k[0]) for k in curves if (k[0] if by == "H" else k[1]) == gv}):
                h, t = (gv, sv) if by == "H" else (sv, gv)
                col, mk = HR_COLORS[sk][sv], HR_MARKS[sk][sv]
                x, mu, s = curves[(float(h), int(t))]
                if cfg["band"] in ("predict", "latent"):
                    ax.fill_between(x, np.exp(mu - 1.96 * s), np.exp(mu + 1.96 * s), color=col, alpha=0.22, zorder=15)
                    ax.plot(x, np.exp(mu - 1.96 * s), color=col, lw=0.6, alpha=0.7, zorder=16)
                    ax.plot(x, np.exp(mu + 1.96 * s), color=col, lw=0.6, alpha=0.7, zorder=16)
                ax.plot(x, np.exp(mu), color=col, lw=cfg["line"], zorder=20)
                sub = pool[(np.isclose(pool["H"], h)) & (pool["Tset"] == t)]
                if cfg["points"] == "cond":
                    a = sub.groupby("cond").agg(lg=("ln_gd", "mean"), le=("ln_eta", "mean"), sens=("is_sens", "max")).reset_index()
                    px, py, tr_m, se_m, msize = np.exp(a["lg"].to_numpy()), np.exp(a["le"].to_numpy()), a["cond"].isin(train_conds).to_numpy(), a["sens"].to_numpy().astype(bool), cfg["marker"]
                else:
                    px, py, tr_m, se_m, msize = sub["gd"].to_numpy(), sub["eta"].to_numpy(), sub["cond"].isin(train_conds).to_numpy(), sub["is_sens"].to_numpy().astype(bool), cfg["marker_shot"]
                xs += list(x) + list(px)
                ys += list(np.exp(mu)) + list(py)
                ax.scatter(px[tr_m], py[tr_m], s=msize, color=col, marker=mk, alpha=0.85, edgecolors="k", linewidths=0.6, zorder=14 - list(HR_COLORS[sk]).index(sv))
                if cfg["show_test"] and (~tr_m).any():
                    ax.scatter(px[~tr_m], py[~tr_m], s=msize, facecolors="white", edgecolors=col, marker=mk, linewidths=2.5, zorder=14 - list(HR_COLORS[sk]).index(sv))
                    any_test = True
                if cfg["mark_sens"] and se_m.any():
                    ax.scatter(px[se_m], py[se_m], s=msize * 1.3, color="k", marker="x", linewidths=3, zorder=25)
                    any_sens = True
                series.append((sv, col, mk))
            head = "キャビティ厚み" if sk == "H" else "設定温度"
            handles = [Patch(facecolor="gray", alpha=0.3, label=band_name), Patch(color="none", label=head)]
            handles += [Line2D([], [], color=col, lw=cfg["line"], marker=mk, markersize=np.sqrt(cfg["marker"]), markeredgecolor="k", markeredgewidth=0.6,
                               label=(f"{sv} mm" if sk == "H" else f"{int(sv)} ℃")) for sv, col, mk in series]
            if any_test:
                handles.append(Line2D([], [], color="gray", ls="none", marker="o", markersize=np.sqrt(cfg["marker"]), markerfacecolor="white", markeredgewidth=2.5, label="検証の条件（白抜き）"))
            if any_sens:
                handles.append(Line2D([], [], color="k", ls="none", marker="x", markersize=np.sqrt(cfg["marker"]), markeredgewidth=3, label="感度確認の条件"))
            if cfg["band"] not in ("predict", "latent"):
                handles = handles[1:]
            ax.legend(handles=handles, loc="upper right", fontsize=ft["legend"], framealpha=1.0, edgecolor="black")
            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=ft["tick"], pad=8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            xl = cfg["xlim"][layout] if isinstance(cfg["xlim"], dict) else cfg["xlim"]       # 範囲：(下限, 上限)＝固定／None＝その図のデータに合わせる
            yl = cfg["ylim"][layout] if isinstance(cfg["ylim"], dict) else cfg["ylim"]
            auto_x, auto_y = xl is None, yl is None
            ax.set_xlim(*((min(xs) / 1.4, max(xs) * 1.4) if auto_x else xl))
            ax.set_ylim(*((min(ys) / 1.5, max(ys) * 1.5) if auto_y else yl))
            for axis, auto in ((ax.xaxis, auto_x), (ax.yaxis, auto_y)):
                if auto:                               # 範囲が狭いので、1・2・5 × 10^n に目盛りを付けて、普通の数で表す
                    axis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
                    axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
                else:                                  # 6月のノートと同じ（10 の累乗の目盛り）
                    axis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))
                    axis.set_major_formatter(ticker.LogFormatterSciNotation())
                axis.set_minor_formatter(ticker.NullFormatter())
            ax.grid(True, which="major", linestyle="-", alpha=0.6, color="gray", zorder=1)
            ax.grid(True, which="minor", linestyle=":", alpha=0.4, zorder=1)
            gtxt = f"厚み {gv} mm" if by == "H" else f"設定温度 {int(gv)} ℃"
            if cfg["show_title"]:
                ax.set_title(f"{MATS[mat]}　せん断速度と粘度の関係（{gtxt}）\n{title_model}", fontsize=ft["title"], pad=15)
            if cfg["show_xlabel"]:
                ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ (1/s)", fontsize=ft["axis"], labelpad=15)
            if cfg["show_ylabel"]:
                ax.set_ylabel("樹脂粘度 (Pa·s)", fontsize=ft["axis"], labelpad=15)
            fig.tight_layout()
            base = f"{stem}_{'by-thickness' if by == 'H' else 'by-temperature'}_{(str(gv) + 'mm') if by == 'H' else (str(int(gv)) + 'C')}"
            for fmt in cfg["formats"]:
                path = os.path.join(out_dir, f"{base}.{fmt}")
                fig.savefig(path, bbox_inches="tight", dpi=cfg["dpi"])
                saved.append(path)
            if cfg["show"]:
                plt.show()
            plt.close(fig)
    return saved
