# -*- coding: utf-8 -*-
"""発表資料（05_…/学習結果報告書作成_261015/パワーポイント/報告書B_5章以降_発表資料.pptx）の図を作る（日本語。学習はやり直さない）。

内容は、ポスター「報告書Bから作成」の 5 章（機械学習）以降。使うモデルは B_Tset（加熱シリンダ温度でシフト）・ショットごとの入力（ユーザー決定、2026-10-23。
報告書B・ポスターは B_Tff・条件平均の入力のまま）。保存済みの 学習済みモデル_B_Tset.joblib と、予測検証の表を読む。build_slides.py が、この図を貼って発表資料を組み立てる。
図は、スライドに貼る大きさ（mm）で作り、文字はすべて 18pt 以上にする（スライドの文字の最小の大きさに合わせる）。
  説明の図（人工のデータ・曲線）：GPR概念図、ベイズ_例、ベイズ_事前、ベイズ_事後、周辺尤度_比較、平均関数_線形項なし、平均関数_線形項あり、
                                 カーネル_距離、カーネル例_<名前>、Matern_サンプル、ARD_サンプル、線形_サンプル、ノイズ_サンプル、合計_サンプル、
                                 シフト概念図
  学習結果の図：シフトファクター_推定値、プロファイル尤度_PP、検証の位置_PP、学習結果_<材料>、凡例_学習結果_<材料>、
               マスターカーブ（S_マスターカーブ_PP・S_凡例_温度）、温度シフトの効果、予測検証_<材料>、凡例_予測検証、予測誤差の比較、
               外挿の範囲（S_苦手な範囲）、表紙の飾り（S_題目の飾り）、比較_従来・比較_GPR（従来の粘度式との違いのイメージ）
  式_*.png：数式（STIX の書体。日本語の部分は IPAexGothic）
  スライドの数値.csv：スライドに書いた数値の元
使い方（リポジトリ直下で）: uv run python 04_補助スクリプト/make_slide_figs.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import make_poster_figs as mp  # noqa: E402  （図の書式〔IPAexGothic など〕と、共通の部品を使う）
import make_poster_design_figs as mpd  # noqa: E402

mp.MODE = "shot"                                   # 発表資料は、ショットごとの入力のモデルで作る（ユーザー決定、2026-10-23。報告書B・ポスターは条件平均のまま）
mp.KEY = "B_Tset"                                  # 主のモデル：加熱シリンダ温度でシフト（ショットごとの入力では、金型内樹脂最高温度のシフトはデータで支持されず、
                                                   # 加熱シリンダ温度のシフトが支持されたため。ユーザー決定、2026-10-23）
SHIFT_T, SHIFT_JA = "Tset", "加熱シリンダ温度"       # シフトに使う温度（列名と呼び名）
LS_COLS = [("換算せん断速度", "ls_ln γ̇ + ln a_h(Tset)"), ("樹脂圧力差", "ls_ln P3_3"), ("金型内樹脂最高温度", "ls_ln Tff")]   # 長さスケール（スライドの呼び名、ハイパーパラメータの表の列）

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, DotProduct, Matern  # noqa: E402

import mat_common as mc  # noqa: E402

OUT = os.path.join(mc.DATA_ROOT, "学習結果報告書作成_261015", "パワーポイント", "図")
FS = 18                                            # 図の文字の大きさ [pt]（スライドの最小の文字と同じ）
PAL = {"main": "#17324D", "accent": "#E07A1F", "sub": "#5B6B7B", "light": "#EEF3F8"}   # build_slides.py と同じ配色
mpd.PAL["S"] = PAL                                 # make_poster_design_figs の図を、この配色・保存先で使う
mpd.OUT = OUT
T_COLS = mpd.T_COLS                                # 加熱シリンダ温度（低・中・高）：青・灰・赤
LOW_COL, HIGH_COL = mpd.LOW_COL, mpd.HIGH_COL
DATA_BG = "#E9EDF1"                                # 「データのない範囲」の背景


def rc(fs=FS):
    return plt.rc_context({"font.size": fs, "xtick.labelsize": fs, "ytick.labelsize": fs, "axes.labelsize": fs + 1, "legend.fontsize": fs,
                           "axes.titlesize": fs + 1})


def save(fig, name, **kw):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=mp.DPI, **kw)
    plt.close(fig)


def bare(ax, keep_frame=True):
    """目盛りを消し、上と右の枠を消す（説明の図）"""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.minorticks_off()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if not keep_frame:
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)


def samples(kernel, x, n, seed):
    """カーネルの事前分布からの標本（列ごとに 1 本）"""
    K = kernel(np.atleast_2d(x).T if np.ndim(x) == 1 else x) + 1e-8 * np.eye(len(x))
    return np.linalg.cholesky(K) @ np.random.default_rng(seed).standard_normal((len(x), n))


def posterior(kernel, noise, xd, yd, xs):
    """ガウス過程の事後分布（潜在関数の平均と共分散。ノイズの分散 noise は、学習データの対角にだけ足す）"""
    K = kernel(xd[:, None]) + noise * np.eye(len(xd))
    Ks = kernel(xs[:, None], xd[:, None])
    a = np.linalg.solve(K, yd)
    mu = Ks @ a
    cov = kernel(xs[:, None]) - Ks @ np.linalg.solve(K, Ks.T)
    return mu, cov


# ---------------------------------------------------------------- 数式
FORMULAS = {
    "式_平均関数": (r"$m(\mathbf{x}) \;=\; \overline{\ln\eta} \;+\; \ln a_v(T_{\mathrm{set}})$", 28),
    "式_事前分布": (r"$f(\mathbf{x}) \;\sim\; \mathcal{GP}\,\left(\,m(\mathbf{x}),\; k(\mathbf{x},\mathbf{x}')\,\right)$", 28),
    "式_観測": (r"$\ln\eta \;=\; f(\mathbf{x}) \;+\; \varepsilon, \qquad \varepsilon \sim \mathcal{N}\,(0,\ \sigma_n^2)$", 28),
    "式_カーネル": (r"$k(\mathbf{x}_i,\mathbf{x}_j) \;=\; \sigma_f^2\,k_{\mathrm{M52}}(\mathbf{x}_i,\mathbf{x}_j) \;+\; "
                r"\sigma_l^2\,(\sigma_0^2 + \mathbf{x}_i\cdot\mathbf{x}_j) \;+\; \sigma_n^2\,\delta_{ij}$", 26),
    "式_Matern": (r"$k_{\mathrm{M52}}(r) \;=\; \left(1 + \sqrt{5}\,r + \frac{5}{3}\,r^2\right)\,\exp\left(-\sqrt{5}\,r\right)$", 26),
    "式_距離": (r"$r \;=\; \sqrt{\sum_{m} \dfrac{(x_{m} - x_{m}')^2}{\ell_{m}^{\,2}}}$", 26),
    "式_線形": (r"$k_{\mathrm{lin}}(\mathbf{x}_i,\mathbf{x}_j) = \sigma_l^2\,(\sigma_0^2 + \mathbf{x}_i\cdot\mathbf{x}_j)$", 24),
    "式_ノイズ": (r"$k_{\mathrm{noise}}(\mathbf{x}_i,\mathbf{x}_j) = \sigma_n^2\,\delta_{ij}$", 24),
    "式_アレニウス": (r"$\ln a(T) = \dfrac{E}{R}\left(\dfrac{1}{T} - \dfrac{1}{T_{\mathrm{ref}}}\right)$", 26),
    "式_WLF": (r"$\ln a(T) = -A_1\left(\dfrac{T - T_g}{A_2 + T - T_g} - \dfrac{T_{\mathrm{ref}} - T_g}{A_2 + T_{\mathrm{ref}} - T_g}\right)$", 26),
    "式_モデル": (r"$\ln\eta \;=\; \ln a_v(T_{\mathrm{set}}) \;+\; f\left(\ln\dot{\gamma} + \ln a_h(T_{\mathrm{set}}),\ \ln\Delta P_2,\ \ln T\right) \;+\; \varepsilon$", 28),
    "式_周辺尤度": (r"$\ln p(\mathbf{y}\mid\theta) = -\frac{1}{2}\,\mathbf{y}^{\top}K_{\theta}^{-1}\mathbf{y} \;-\; \frac{1}{2}\ln\left|K_{\theta}\right| "
               r"\;-\; \frac{n}{2}\ln 2\pi$", 24),
    "式_ベイズ_GP": (r"$p(f\mid D) = \dfrac{p(D\mid f)\;p(f)}{p(D)}$", 28),
    "式_ベイズ_例_平均": (r"$\mu_{\mathrm{post}} = \dfrac{\mu_0/\sigma_0^2 \;+\; n\,\bar{y}/\sigma^2}{1/\sigma_0^2 \;+\; n/\sigma^2}$", 26),
    "式_ベイズ_例_分散": (r"$\dfrac{1}{\sigma_{\mathrm{post}}^2} = \dfrac{1}{\sigma_0^2} + \dfrac{n}{\sigma^2}$", 26),
    "式_予測平均": (r"$\mu(\mathbf{x}_*) = \mathbf{k}_*^{\top}\left(K + \sigma_n^2 I\right)^{-1}\mathbf{y}$", 28),
    "式_予測分散": (r"$\sigma^2(\mathbf{x}_*) = k(\mathbf{x}_*,\mathbf{x}_*) - \mathbf{k}_*^{\top}\left(K + \sigma_n^2 I\right)^{-1}\mathbf{k}_* + \sigma_n^2$", 28),
    "式_予測区間": (r"95% 予測区間 $=\ \mu(\mathbf{x}_*) \;\pm\; 1.96\;\sigma(\mathbf{x}_*)$", 28),
    "式_尤度": (r"$p(\mathbf{y}\mid f) = \prod_{i=1}^{n}\,\mathcal{N}\left(y_i \mid f(\mathbf{x}_i),\ \sigma_n^2\right)$", 26),
    "式_共分散": (r"$\mathrm{cov}\left[\,f(\mathbf{x}),\ f(\mathbf{x}')\,\right] = k(\mathbf{x},\mathbf{x}'), \qquad \rho = k(\mathbf{x},\mathbf{x}')\,/\,\sigma_f^2$", 26),
    "式_標準化": (r"$\tilde{x}_m = \dfrac{x_m - \bar{x}_m}{s_m}$", 26),
    "式_条件平均": (r"$\overline{\ln\dot{\gamma}}_c = \dfrac{1}{n_c}\sum_{k=1}^{n_c} \ln\dot{\gamma}_{c,k}$", 26),
    "式_入力": (r"$\mathbf{x} = \left(\,\ln\dot{\gamma} + \ln a_h(T_{\mathrm{set}}),\ \ \ln\Delta P_2,\ \ \ln T\,\right)$", 26),
    "式_TTS": (r"$\eta(\dot{\gamma},\,T) \;=\; a_v(T)\;\eta_M\!\left(a_h(T)\,\dot{\gamma}\right)$", 30),
    "式_換算": (r"$\dot{\gamma}_r = a_h(T_{\mathrm{set}})\,\dot{\gamma}, \qquad \eta_r = \eta\,/\,a_v(T_{\mathrm{set}})$", 28),
    "式_CrossWLF": (r"Cross-WLF モデル：$\ a_h(T) = a_v(T) = a_T(T)$", 24),
    "式_プロファイル": (r"$(\hat{E}_h,\ \hat{E}_v) = \mathrm{arg\,max}_{E_h,\,E_v}\ \ln p(\mathbf{y}\mid E_h,\,E_v)$", 26),
    "式_尤度差": (r"$\Delta L = \ln p(\mathbf{y}\mid \hat{E}_h,\,\hat{E}_v) - \ln p(\mathbf{y}\mid 0,\,0) \;>\; 3.0$", 26),
    "式_カイ二乗": (r"$2\,\Delta L \sim \chi^2_2, \qquad \chi^2_{2}\,$ の 95% 点 $= 5.99$", 24),
    "式_RMSE": (r"$\mathrm{RMSE} = \sqrt{\dfrac{1}{n}\sum_{i=1}^{n}\left(\ln\eta_i - \ln\hat{\eta}_i\right)^2}$", 26),
    "式_誤差": (r"平均的な予測誤差 $= \left(e^{\mathrm{RMSE}} - 1\right) \times 100$ %", 26),
    "式_被覆率": (r"区間に入った割合 $= \dfrac{1}{n}\sum_{i=1}^{n} \mathbf{1}\left[\,L_i \leq \eta_i \leq U_i\,\right]$", 26),
    "式_短_RBF": (r"$\exp\left(-\,r^2/2\right)$", 20),
    "式_短_Matern": (r"$\left(1+\sqrt{5}\,r+\frac{5}{3}r^2\right)e^{-\sqrt{5}\,r}$", 20),
    "式_短_線形": (r"$\sigma_0^2 + \mathbf{x}_i\cdot\mathbf{x}_j$", 20),
    "式_短_ノイズ": (r"$\delta_{ij}$", 20),
    "式_温度比": (r"温度による粘度の比 $R = \exp\left(\dfrac{1}{G}\sum_{g=1}^{G}\left[\,\ln\eta_{\mathrm{low}}(\dot{\gamma}_g) - \ln\eta_{\mathrm{high}}(\dot{\gamma}_g)\,\right]\right)$", 24),
    "式_ずれ最小": (r"最小のずれ $=\ \min_{E_h,\ E_v}$ ずれ$(E_h,\ E_v)$", 24),
    "式_尤度差_温度": (r"$\Delta L = \ln p(\mathbf{y}_H \mid \mathrm{B\_Tff}) \;-\; \ln p(\mathbf{y}_H \mid \mathrm{B\_Tset})$", 24),
    "式_ずれ": (r"ずれ $= \sqrt{\dfrac{1}{n}\sum_{i=1}^{n}\left(\ln\eta_{r,i} - g_H\!\left(\ln\dot{\gamma}_{r,i}\right)\right)^2}$", 26),
}


def fig_formulas():
    with plt.rc_context({"mathtext.fontset": "stix"}):
        for name, (tex, fs) in FORMULAS.items():
            fig = plt.figure(figsize=(0.1, 0.1))
            fig.text(0, 0, tex, fontsize=fs, color="#1F2933")
            save(fig, f"{name}.png", bbox_inches="tight", pad_inches=0.03, transparent=True)


# ---------------------------------------------------------------- ガウス過程回帰・ベイズの定理の説明
def fig_gpr_concept(w=188.0, h=112.0, margins=(11, 3, 11, 15)):
    """人工のデータでの、予測平均値と 95% 予測区間（データの多い所は幅が狭く、ない所は広い。ポスターの図と同じデータ）"""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import WhiteKernel
    rng = np.random.default_rng(3)
    f = lambda x: 2.2 - 0.32 * x + 0.35 * np.sin(1.1 * x)
    x = np.sort(np.concatenate([rng.uniform(1.6, 6.6, 26), [0.6]]))
    y = f(x) + rng.normal(0, 0.06, len(x))
    gp = GaussianProcessRegressor(C(1.0) * RBF(1.5) + WhiteKernel(0.01), normalize_y=True, random_state=0).fit(x[:, None], y)
    xs = np.linspace(0, 10, 400)
    mu, sd = gp.predict(xs[:, None], return_std=True)
    with rc():
        fig, ax = mpd.new_fig(w, h, *margins)
        ax.fill_between(xs, mu - 1.96 * sd, mu + 1.96 * sd, color=PAL["accent"], alpha=0.25, lw=0, label="95% 予測区間")
        ax.plot(xs, mu, color=PAL["accent"], lw=2.8, label="予測平均値")
        ax.scatter(x, y, s=70, color=PAL["main"], edgecolors="white", linewidths=0.6, zorder=5, label="学習データ")
        i4, i9 = np.argmin(np.abs(xs - 4.0)), np.argmin(np.abs(xs - 9.0))
        ax.annotate("データが多い\n→ 幅が狭い", xy=(4.0, mu[i4] - 1.2 * sd[i4]), xytext=(3.0, -1.0), ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="0.2"))
        ax.annotate("データがない\n→ 幅が広い", xy=(8.9, mu[i9] - 1.6 * sd[i9]), xytext=(6.8, -1.0), ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="0.2"))
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.6, 3.0)
        bare(ax)
        ax.set_xlabel("入力（例：せん断速度）", labelpad=4)
        ax.set_ylabel("出力（例：樹脂粘度）", labelpad=4)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, borderaxespad=0.2, columnspacing=1.2, handletextpad=0.4,
                  handlelength=1.5)
        save(fig, "GPR概念図.png")


def fig_compare_mini(w=120.0, h=56.0):
    """従来の粘度式（1 本の曲線）と、ガウス過程回帰（曲線 ＋ 予測区間）のイメージ（人工の曲線。文字なし。スライドのカードに貼る）"""
    g = np.logspace(0, 4.6, 300)
    lg = np.log10(g)
    eta = 2000 / (1 + (g / 150) ** 0.72)
    sd = 0.05 + 0.32 * ((lg - 2.6) / 2.2) ** 2                       # 中央（データの多い所）は狭く、両端は広い
    for name, band, col in [("比較_従来.png", False, PAL["sub"]), ("比較_GPR.png", True, PAL["main"])]:
        fig = plt.figure(figsize=mp.mm(w, h))
        ax = fig.add_axes([0.03, 0.05, 0.94, 0.9])
        if band:
            ax.fill_between(g, eta * np.exp(-1.96 * sd), eta * np.exp(1.96 * sd), color=PAL["accent"], alpha=0.3, lw=0)
        ax.plot(g, eta, color=col, lw=3.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(g[0], g[-1])
        ax.set_ylim(3, 2e4)
        bare(ax)
        save(fig, name, transparent=True)


def fig_bayes_example(w=176.0, h=112.0, margins=(10, 4, 19, 4)):
    """ベイズの定理の例（架空の数値）：ある成形条件の粘度について、事前分布・尤度（3 ショットの計測値）・事後分布（いずれも ln η の正規分布）"""
    lx = np.linspace(np.log(40.0), np.log(2500.0), 800)
    pm, ps = np.log(180.0), 0.30                                   # 事前分布：計測前の予想（180 Pa·s 前後、幅が広い）
    shots = np.log([290.0, 335.0, 300.0])
    lm, ls = shots.mean(), 0.35 / np.sqrt(len(shots))              # 尤度：計測値（ショット間のばらつき 0.35）の平均
    v = 1.0 / (1.0 / ps ** 2 + 1.0 / ls ** 2)
    qm, qs = v * (pm / ps ** 2 + lm / ls ** 2), np.sqrt(v)         # 事後分布
    dens = lambda m, s: np.exp(-0.5 * ((lx - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    with rc():
        fig, ax = mpd.new_fig(w, h, *margins)
        ax.set_xscale("log")
        ax.plot(np.exp(lx), dens(pm, ps), color="0.45", lw=2.6, ls="--")
        ax.fill_between(np.exp(lx), dens(pm, ps), color="0.6", alpha=0.12, lw=0)
        ax.plot(np.exp(lx), dens(lm, ls), color=LOW_COL, lw=2.6)
        ax.plot(np.exp(lx), dens(qm, qs), color=PAL["accent"], lw=3.2)
        ax.fill_between(np.exp(lx), dens(qm, qs), color=PAL["accent"], alpha=0.25, lw=0)
        ax.scatter(np.exp(shots), [0.06] * 3, marker="^", s=110, color=LOW_COL, edgecolors="k", linewidths=0.6, zorder=6, clip_on=False)
        top = dens(qm, qs).max()
        ax.text(105.0, dens(pm, ps).max() * 1.08, "事前分布\n（計測前の予想）", ha="center", va="bottom", color="0.3")
        ax.text(np.exp(lm) * 1.95, dens(lm, ls).max() * 0.80, "尤度\n（計測データ）", ha="center", va="center", color=LOW_COL)
        ax.text(np.exp(qm), top * 1.04, "事後分布（計測後の予想）", ha="center", va="bottom", color=PAL["accent"], fontweight="bold")
        ax.annotate("計測値\n（3 ショット）", xy=(np.exp(shots.max()) * 1.04, 0.08), xytext=(560.0, 0.42), ha="left", va="center",
                    color=LOW_COL, arrowprops=dict(arrowstyle="-", color=LOW_COL, lw=1.2))
        ax.set_xlim(60, 1500)
        ax.set_ylim(0, top * 1.22)
        ax.set_yticks([])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xticks([100, 300, 1000])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="x", which="both", top=False, direction="out")
        ax.set_xlabel(r"ある成形条件の樹脂粘度 $\eta$ (Pa·s)", labelpad=3)
        save(fig, "ベイズ_例.png")
    return float(np.exp(pm)), float(np.exp(lm)), float(np.exp(qm))


TOY_K = C(0.30) * Matern(1.6, nu=2.5) + C(0.012) * DotProduct(sigma_0=1.0)       # 説明用のカーネル（本学習と同じ形：Matern ＋ 線形）
TOY_NOISE = 0.05 ** 2


def toy_data(n=14, seed=5):
    f = lambda x: 2.4 - 0.22 * x + 0.25 * np.sin(1.3 * x)
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(1.5, 7.5, n))
    return x, f(x) + rng.normal(0, 0.05, n), f


def fig_bayes_samples(w=150.0, h=100.0, margins=(11, 4, 11, 3)):
    """事前分布からの曲線の標本（データを見る前）と、事後分布からの標本（データを見た後）。説明用のカーネル（Matern ＋ 線形）"""
    xs = np.linspace(0, 10, 300)
    x, y, _ = toy_data()
    ybar = y.mean()
    cols = ["#2F6DB5", "#D64545", "#2CA02C", "#9467BD", "#8C564B", "#E07A1F"]
    prior = samples(TOY_K, xs - 5.0, 6, seed=11) + ybar
    mu, cov = posterior(TOY_K, TOY_NOISE, x - 5.0, y - ybar, xs - 5.0)
    post = mu[:, None] + np.linalg.cholesky(cov + 1e-8 * np.eye(len(xs))) @ np.random.default_rng(12).standard_normal((len(xs), 6)) + ybar
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    for name, curves, data in [("ベイズ_事前.png", prior, False), ("ベイズ_事後.png", post, True)]:
        with rc():
            fig, ax = mpd.new_fig(w, h, *margins)
            if data:
                ax.fill_between(xs, mu + ybar - 1.96 * sd, mu + ybar + 1.96 * sd, color="0.6", alpha=0.18, lw=0, zorder=1)
            for i in range(curves.shape[1]):
                ax.plot(xs, curves[:, i], color=cols[i], lw=1.8, alpha=0.9, zorder=2)
            ax.axhline(ybar, color="0.25", lw=1.6, ls="--", zorder=3)
            if data:
                ax.scatter(x, y, s=70, color=PAL["main"], edgecolors="white", linewidths=0.7, zorder=5)
            ax.set_xlim(0, 10)
            ax.set_ylim(ybar - 1.9, ybar + 1.9)
            bare(ax)
            ax.set_xlabel("入力（例：せん断速度）", labelpad=4)
            ax.set_ylabel("出力（例：粘度）", labelpad=4)
            save(fig, name)


def fig_lml_compare(w=310.0, h=100.0, margins=(6, 4, 4, 30), gap=8.0):
    """長さスケール ℓ だけを変えたときの当てはまりと、対数周辺尤度（振幅とノイズは固定。人工のデータ）"""
    rng = np.random.default_rng(7)
    f = lambda x: np.sin(x) + 0.3 * x
    x = np.sort(rng.uniform(0.3, 7.7, 16))
    y = f(x) + rng.normal(0, 0.15, len(x))
    xs = np.linspace(0, 8, 300)
    noise = 0.15 ** 2
    yc = y - y.mean()

    def lml(l):
        K = C(1.0)(x[:, None]) * Matern(l, nu=2.5)(x[:, None]) + noise * np.eye(len(x))
        L = np.linalg.cholesky(K)
        a = np.linalg.solve(L.T, np.linalg.solve(L, yc))
        return float(-0.5 * yc @ a - np.log(np.diag(L)).sum() - 0.5 * len(x) * np.log(2 * np.pi))

    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
    best = float(grid[np.argmax([lml(l) for l in grid])])
    cases = [(0.12, "短すぎる"), (best, "ちょうどよい"), (8.0, "長すぎる")]
    out = []
    with rc():
        fig = plt.figure(figsize=mp.mm(w, h))
        l0, r0_, b0, t0 = margins
        pw = (w - l0 - r0_ - 2 * gap) / 3
        for k, (l, lab) in enumerate(cases):
            ax = fig.add_axes([(l0 + k * (pw + gap)) / w, b0 / h, pw / w, (h - b0 - t0) / h])
            kern = C(1.0) * Matern(l, nu=2.5)
            mu, cov = posterior(kern, noise, x, yc, xs)
            sd = np.sqrt(np.clip(np.diag(cov) + noise, 0, None))
            good = k == 1
            col = PAL["accent"] if good else "0.35"
            ax.fill_between(xs, mu + y.mean() - 1.96 * sd, mu + y.mean() + 1.96 * sd, color=col, alpha=0.2, lw=0)
            ax.plot(xs, mu + y.mean(), color=col, lw=2.6)
            ax.scatter(x, y, s=55, color=PAL["main"], edgecolors="white", linewidths=0.5, zorder=5)
            ax.set_xlim(0, 8)
            ax.set_ylim(-1.6, 4.4)
            bare(ax)
            v = lml(l)
            out.append((lab, l, v))
            ax.set_title(f"長さスケール ℓ = {l:.2g}\n（{lab}）\n対数周辺尤度 {v:.1f}".replace("-", "−"), color=PAL["accent"] if good else "0.2",
                         fontweight="bold" if good else "normal", pad=6, fontsize=FS, linespacing=1.15)
            if good:
                for sp in ("left", "bottom"):
                    ax.spines[sp].set_color(PAL["accent"])
                    ax.spines[sp].set_linewidth(2.0)
        save(fig, "周辺尤度_比較.png")
    return out


def fig_mean_function(w=150.0, h=96.0, margins=(11, 4, 11, 3)):
    """平均関数の役割：データから離れると、予測は平均関数に戻る（線形項なし）／線形項を足すと、傾向に沿って延びる（人工のデータ）"""
    rng = np.random.default_rng(4)
    f = lambda x: 3.9 - 0.42 * x + 0.03 * (x - 5.0) ** 2
    x = np.sort(rng.uniform(3.2, 6.8, 16))
    y = f(x) + rng.normal(0, 0.05, len(x))
    xs = np.linspace(0, 10, 300)
    ybar = y.mean()
    cases = [("平均関数_線形項なし.png", C(0.6) * Matern(1.5, nu=2.5)),
             ("平均関数_線形項あり.png", C(0.05) * Matern(1.5, nu=2.5) + C(0.3) * DotProduct(sigma_0=0.5))]
    for name, kern in cases:
        mu, cov = posterior(kern, 0.003, x - 5.0, y - ybar, xs - 5.0)
        sd = np.sqrt(np.clip(np.diag(cov) + 0.003, 0, None))
        with rc():
            fig, ax = mpd.new_fig(w, h, *margins)
            for a, b in [(0, x.min()), (x.max(), 10)]:
                ax.axvspan(a, b, color=DATA_BG, lw=0, zorder=0)
            ax.plot(xs, f(xs), color="0.55", lw=1.6, ls=":", zorder=1)
            ax.fill_between(xs, mu + ybar - 1.96 * sd, mu + ybar + 1.96 * sd, color=PAL["accent"], alpha=0.22, lw=0, zorder=2)
            ax.plot(xs, mu + ybar, color=PAL["accent"], lw=2.8, zorder=3)
            ax.scatter(x, y, s=60, color=PAL["main"], edgecolors="white", linewidths=0.6, zorder=5)
            if "なし" in name:
                ax.axhline(ybar, color="0.2", lw=1.8, ls="--", zorder=4)
                ax.text(9.85, ybar + 0.18, "平均関数（一定値）", ha="right", va="bottom", color="0.15")
            ax.text(1.6, -0.05 + 0.0, "データなし", ha="center", va="bottom", color="0.35")
            ax.text(8.4, -0.05, "データなし", ha="center", va="bottom", color="0.35")
            ax.set_xlim(0, 10)
            ax.set_ylim(-0.3, 5.3)
            bare(ax)
            ax.set_xlabel("入力（例：せん断速度の対数）", labelpad=4)
            ax.set_ylabel("出力（例：粘度の対数）", labelpad=4)
            save(fig, name)


# ---------------------------------------------------------------- カーネル関数
def fig_kernel_distance(w=160.0, h=108.0, margins=(27, 5, 20, 4)):
    r = np.linspace(0, 4, 300)
    with rc():
        fig, ax = mpd.new_fig(w, h, *margins)
        for l, col, ls in [(0.5, LOW_COL, "-"), (1.0, PAL["accent"], "-"), (2.0, HIGH_COL, "--")]:
            k = Matern(l, nu=2.5)(np.zeros((1, 1)), r[:, None])[0]
            ax.plot(r, k, color=col, lw=2.8, ls=ls, label=f"ℓ = {l:g}")
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_yticks([0, 0.5, 1.0])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.grid(True, ls=":", lw=0.6, color="0.75")
        ax.tick_params(top=False, right=False)
        ax.set_xlabel(r"2 つの入力の距離 $|x - x'|$", labelpad=3)
        ax.set_ylabel("相関（似ている度合い）", labelpad=3)
        ax.legend(loc="upper right", frameon=False, title="長さスケール ℓ", title_fontsize=FS)
        save(fig, "カーネル_距離.png")


def fig_kernel_examples(w=74.0, h=46.0):
    """代表的なカーネルの事前分布からの標本（文字なし。スライドのカードに貼る）"""
    xs = np.linspace(0, 6, 300)
    cols = ["#2F6DB5", "#D64545", "#2CA02C"]
    cases = {"RBF": C(1.0) * RBF(0.9), "Matern": C(1.0) * Matern(0.9, nu=2.5), "線形": C(0.4) * DotProduct(sigma_0=0.4)}
    for name, kern in list(cases.items()) + [("ノイズ", None)]:
        fig = plt.figure(figsize=mp.mm(w, h))
        ax = fig.add_axes([0.03, 0.06, 0.94, 0.88])
        if kern is None:
            rng = np.random.default_rng(3)
            xp = np.linspace(0, 6, 28)
            for i, c in enumerate(cols):
                ax.scatter(xp, rng.standard_normal(len(xp)) * 0.8 + (i - 1) * 0.15, s=14, color=c, alpha=0.85)
        else:
            xx = xs - 3.0 if name == "線形" else xs
            ss = samples(kern, xx, 3, seed=21)
            for i, c in enumerate(cols):
                ax.plot(xs, ss[:, i], color=c, lw=2.0)
        ax.set_xlim(0, 6)
        ax.set_ylim(-2.8, 2.8)
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        bare(ax, keep_frame=False)
        save(fig, f"カーネル例_{name}.png", transparent=True)


def fig_matern_samples(w=310.0, h=80.0, margins=(6, 4, 4, 12), gap=8.0):
    xs = np.linspace(0, 5, 300)
    cols = ["#2F6DB5", "#D64545", "#2CA02C"]
    z = np.random.default_rng(31).standard_normal((len(xs), 3))
    with rc():
        fig = plt.figure(figsize=mp.mm(w, h))
        l0, r0_, b0, t0 = margins
        pw = (w - l0 - r0_ - 2 * gap) / 3
        for k, (l, lab) in enumerate([(0.3, "短い"), (1.0, "中くらい"), (3.0, "長い")]):
            ax = fig.add_axes([(l0 + k * (pw + gap)) / w, b0 / h, pw / w, (h - b0 - t0) / h])
            K = Matern(l, nu=2.5)(xs[:, None]) + 1e-8 * np.eye(len(xs))
            ss = np.linalg.cholesky(K) @ z
            for i, c in enumerate(cols):
                ax.plot(xs, ss[:, i], color=c, lw=2.0)
            ax.axhline(0, color="0.8", lw=0.8, zorder=0)
            ax.set_xlim(0, 5)
            ax.set_ylim(-3, 3)
            bare(ax)
            ax.set_title(f"長さスケール ℓ = {l:g}（{lab}）", pad=5, fontsize=FS)
        save(fig, "Matern_サンプル.png")


def fig_ard(w=310.0, h=100.0, side=66.0, bottom=10.0):
    """2 つの入力の Matern 5/2 の標本（色＝出力）。左：ℓ1 = ℓ2、右：ℓ1 が短く ℓ2 が長い（入力 1 の影響が大きい）。
    図は正方形（一辺 side [mm]）で、左右の半分の中央に置く（題は 2 行）"""
    g = np.linspace(0, 4, 42)
    X = np.array([(a, b) for b in g for a in g])
    z = np.random.default_rng(41).standard_normal(len(X))
    with rc():
        fig = plt.figure(figsize=mp.mm(w, h))
        for k, (ls, title) in enumerate([((1.0, 1.0), "入力 1 の長さスケール ℓ₁ = 1\n入力 2 の長さスケール ℓ₂ = 1"),
                                         ((0.35, 4.0), "入力 1 の長さスケール ℓ₁ = 0.35\n入力 2 の長さスケール ℓ₂ = 4")]):
            x0 = w / 4 + k * w / 2 - side / 2 + 4.0
            ax = fig.add_axes([x0 / w, bottom / h, side / w, side / h])
            K = Matern(list(ls), nu=2.5)(X) + 1e-7 * np.eye(len(X))
            s = (np.linalg.cholesky(K) @ z).reshape(len(g), len(g))
            ax.imshow(s, origin="lower", extent=(0, 4, 0, 4), cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("入力 1", labelpad=3)
            ax.set_ylabel("入力 2", labelpad=3)
            ax.set_title(title, pad=6, fontsize=FS, linespacing=1.2)
        save(fig, "ARD_サンプル.png")


def fig_linear_noise(w=150.0, h=84.0, margins=(6, 4, 6, 3)):
    xs = np.linspace(-3, 3, 200)
    cols = ["#2F6DB5", "#D64545", "#2CA02C", "#9467BD", "#E07A1F"]
    ss = samples(C(0.25) * DotProduct(sigma_0=0.6), xs, 5, seed=51)
    with rc():
        fig, ax = mpd.new_fig(w, h, *margins)
        for i, c in enumerate(cols):
            ax.plot(xs, ss[:, i], color=c, lw=2.2)
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        bare(ax)
        save(fig, "線形_サンプル.png")
    rng = np.random.default_rng(52)
    xp = np.linspace(-3, 3, 40)
    with rc():
        fig, ax = mpd.new_fig(w, h, *margins)
        for c in cols[:2]:                                           # 点ごとに独立な値（線は、点の並びを見やすくするための補助）
            v = rng.standard_normal(len(xp)) * 0.9
            ax.plot(xp, v, color=c, lw=0.8, alpha=0.45, zorder=2)
            ax.scatter(xp, v, s=40, color=c, edgecolors="white", linewidths=0.4, zorder=3)
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        bare(ax)
        save(fig, "ノイズ_サンプル.png")


def fig_sum_samples(w=196.0, h=114.0, margins=(11, 4, 11, 22)):
    """本学習と同じ形のカーネル（Matern 5/2 ＋ 線形 ＋ ノイズ）の事前分布からの標本：線＝Matern ＋ 線形、点＝ノイズを足した値"""
    xs = np.linspace(0, 10, 300)
    xp = np.linspace(0.3, 9.7, 26)
    kern = C(0.25) * Matern(1.3, nu=2.5) + C(0.03) * DotProduct(sigma_0=0.5)
    cols = ["#2F6DB5", "#D64545", "#2CA02C"]
    allx = np.concatenate([xs, xp])
    ss = samples(kern, allx - 5.0, 3, seed=61)
    rng = np.random.default_rng(62)
    with rc():
        fig, ax = mpd.new_fig(w, h, *margins)
        for i, c in enumerate(cols):
            ax.plot(xs, ss[:len(xs), i], color=c, lw=2.4)
            ax.scatter(xp, ss[len(xs):, i] + rng.normal(0, 0.12, len(xp)), s=42, color=c, edgecolors="white", linewidths=0.5, zorder=4)
        ax.set_xlim(0, 10)
        bare(ax)
        ax.set_xlabel("入力", labelpad=4)
        ax.set_ylabel("出力", labelpad=4)
        h1 = Line2D([], [], color="0.3", lw=2.4, label="線：Matern 5/2 ＋ 線形（なめらかな形と傾向）")
        h2 = Line2D([], [], color="0.3", ls="", marker="o", ms=7, label="点：ノイズを足した値（ばらつき）")
        ax.legend(handles=[h1, h2], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, frameon=False, borderaxespad=0.2, handlelength=1.5,
                  handletextpad=0.5, labelspacing=0.2)
        save(fig, "合計_サンプル.png")


# ---------------------------------------------------------------- 温度シフトファクター
def fig_shift_concept(w=180.0, h=114.0, margins=(11, 4, 11, 4)):
    """低温と高温の粘度曲線（Cross 型の人工の曲線）。高温の曲線は、横（高せん断速度側）と縦（低粘度側）にずれる。
    本モデルの推定値（横・縦の係数とも正）と同じ向き：高温ほど a_h < 1・a_v < 1"""
    g = np.logspace(0, 5, 300)
    cross = lambda g: 3000.0 / (1 + (3000.0 * g / 3e4) ** 0.72)
    ah, av = 0.3, 0.15                                           # 高温の a_h・a_v（1 より小さい）
    hi = lambda g: av * cross(g * ah)
    with rc():
        fig, ax = mpd.new_fig(w, h, *margins)
        ax.plot(g, cross(g), color=T_COLS[0], lw=3.2, label="低温")
        ax.plot(g, hi(g), color=T_COLS[2], lw=3.2, ls="--", label="高温")
        gx = 400.0
        p0 = (gx, hi(gx))                                        # 高温の曲線の点
        p1 = (gx * ah, p0[1])                                    # 横に a_h 倍
        p2 = (gx * ah, cross(gx * ah))                           # 縦に 1/a_v 倍すると、低温の曲線に乗る
        for a, b, col in [(p0, p1, PAL["main"]), (p1, p2, PAL["accent"])]:
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=30, lw=3.0, color=col, zorder=6, shrinkA=0, shrinkB=2))
        box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9)
        ax.text(np.sqrt(p0[0] * p1[0]) * 1.6, p0[1] / 2.2, "横のシフトファクター $a_h$", ha="center", va="top", color=PAL["main"], fontweight="bold",
                bbox=box, zorder=7)
        ax.text(p1[0] / 1.4, np.sqrt(p1[1] * p2[1]), "縦のシフト\nファクター $a_v$", ha="right", va="center", color=PAL["accent"],
                fontweight="bold", bbox=box, zorder=7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 1e5)
        ax.set_ylim(2, 8000)
        bare(ax)
        ax.set_xlabel("せん断速度（対数）", labelpad=4)
        ax.set_ylabel("樹脂粘度（対数）", labelpad=4)
        ax.legend(loc="lower left", frameon=False, borderaxespad=0.3)
        save(fig, "シフト概念図.png")


def fig_shift_factors(d, w=310.0, h=100.0, margins=(24, 4, 20, 25), gap=14.0):
    """推定したシフトファクター a_h・a_v と、シフトに使う温度（SHIFT_T）の関係（学習データの温度の範囲）"""
    with rc():
        fig = plt.figure(figsize=mp.mm(w, h))
        l0, r0_, b0, t0 = margins
        pw = (w - l0 - r0_ - 2 * gap) / 3
        for k, mat in enumerate(mp.MATS):
            fit, tr = d[mat]["fit"], d[mat]["tr"]
            cfg = fit["cfg"]
            ax = fig.add_axes([(l0 + k * (pw + gap)) / w, b0 / h, pw / w, (h - b0 - t0) / h])
            T = np.linspace(tr[SHIFT_T].min(), tr[SHIFT_T].max(), 200)
            ax.axhline(1.0, color="0.6", lw=1.0, zorder=0)
            ax.axvline(cfg["Tref"], color="0.6", lw=1.0, ls=":", zorder=0)
            ax.plot(T, np.exp(mc.shift_ln(T, fit["tv"], cfg)), color=PAL["main"], lw=2.8, label="縦のシフトファクター $a_v$（粘度方向）")
            ax.plot(T, np.exp(mc.shift_ln(T, fit["th"], cfg)), color=PAL["accent"], lw=2.8, ls="--", label="横のシフトファクター $a_h$（せん断速度方向）")
            ax.set_ylim(0.6, 1.8)
            ax.set_xlim(174, 250)                          # 左下の角で、横と縦の目盛りの数字が重ならないように、180 より少し左から
            ax.set_xticks([180, 200, 220, 240])
            ax.set_yticks([0.6, 1.0, 1.4, 1.8])
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
            ax.tick_params(top=False, right=False)
            ax.grid(True, ls=":", lw=0.6, color="0.8")
            ax.set_title(f"{mp.MAT_JA[mat]}（基準 {cfg['Tref']:.0f} ℃）", pad=5, fontsize=FS)
            if k == 0:
                ax.set_ylabel("シフトファクター", labelpad=2)
            ax.set_xlabel(f"{SHIFT_JA} (℃)", labelpad=2, fontsize=FS)
        hdl, lab = ax.get_legend_handles_labels()
        fig.legend(hdl, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2, columnspacing=2.0)
        save(fig, "シフトファクター_推定値.png")


def fig_profile(mat="PP", w=168.0, h=124.0, margins=(26, 32, 20, 4)):
    """横・縦の係数の格子での、対数周辺尤度（最大との差）。★＝最大、×＝シフトなし（0, 0）"""
    p = pd.read_csv(os.path.join(mp.model_dir(mat), "プロファイル尤度_格子.csv"), encoding="utf-8-sig")
    p = p.groupby(["th", "tv"], as_index=False)["LML"].max()
    best = p.loc[p["LML"].idxmax()]
    z = np.clip(p["LML"] - best["LML"], -20, 0)
    zero = p[(p["th"] == 0) & (p["tv"] == 0)]["LML"].iloc[0] - best["LML"]
    with rc():
        fig, ax = mpd.new_fig(w, h, *margins)
        cs = ax.tricontourf(p["th"], p["tv"], z, levels=np.linspace(-20, 0, 21), cmap="viridis")
        ax.tricontour(p["th"], p["tv"], p["LML"] - best["LML"], levels=[-3.0], colors="white", linewidths=1.6, linestyles="--")
        ax.plot(best["th"], best["tv"], marker="*", ms=22, color="white", mec="k", mew=0.8, zorder=5)
        ax.plot(0, 0, marker="X", ms=15, color="white", mec="k", mew=0.8, zorder=5)
        ax.annotate(f"最大（{best['th']:g}, {best['tv']:g}）", xy=(best["th"], best["tv"]), xytext=(best["th"] + 6, best["tv"] + 16), color="white",
                    fontweight="bold", ha="left", va="center", arrowprops=dict(arrowstyle="-", color="white", lw=1.2))
        ax.annotate(f"シフトなし\n（差 {zero:.1f}）".replace("-", "−"), xy=(0, 0), xytext=(-17, -12), color="white", ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="white", lw=1.2))
        ax.set_xlim(p["th"].min(), p["th"].max())
        ax.set_ylim(p["tv"].min(), p["tv"].max())
        ax.set_xticks([t for t in range(-10, int(p["th"].max()) + 1, 10)])          # 左下の角で、横と縦の目盛りの数字が重ならないように
        minus = plt.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−"))
        ax.xaxis.set_major_formatter(minus)
        ax.yaxis.set_major_formatter(minus)
        ax.tick_params(top=False, right=False, colors="0.15")
        ax.set_xlabel(r"横の係数 $E_h$ (kJ/mol)", labelpad=2)
        ax.set_ylabel(r"縦の係数 $E_v$ (kJ/mol)", labelpad=2)
        cax = fig.add_axes([(w - margins[1] + 6) / w, margins[2] / h, 5 / w, (h - margins[2] - margins[3]) / h])
        cb = fig.colorbar(cs, cax=cax, ticks=[-20, -10, -3, 0])
        cb.ax.set_yticklabels(["−20", "−10", "−3", "0"])
        cb.set_label("対数周辺尤度の、最大との差", labelpad=4)
        save(fig, f"プロファイル尤度_{mat}.png")
    return float(best["th"]), float(best["tv"]), float(-zero)


# ---------------------------------------------------------------- 学習結果
def fig_flow(d, mat, w=150.0, h=110.0, margins=(27, 5, 19, 3)):
    """全厚み共通の曲線（加熱シリンダ温度ごと。ΔP2 は学習データの中央値、金型内樹脂最高温度は、その温度の学習データの中央値に固定。
    報告書B の図5・ポスターと同じ固定値）と、学習データのショットごとの値（学習に使った値。色＝温度、形＝厚み）"""
    tr, fit = d["tr"], d["fit"]
    xl, yl = (1.0, 1e5), (0.5, 3000.0)
    p_med = float(np.exp(tr["ln_P33"].median()) / 1e6)
    lg = np.linspace(np.log(xl[0]), np.log(xl[1]), 500)
    info = []
    with rc():
        fig, ax = mpd.new_fig(w, h, *margins)
        mp.log_axes(ax, xl, yl)
        for i, (t, sub) in enumerate(tr.groupby("Tset")):
            tff = float(np.exp(sub["ln_Tff"].median()) - 273.15)
            mu, sd, _, _ = mc.predict(fit, mc.model_inputs(np.exp(lg), p_med, tff, float(t)))
            c = T_COLS[i]
            ax.fill_between(np.exp(lg), np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd), color=c, alpha=0.18, lw=0, zorder=2)
            ax.plot(np.exp(lg), np.exp(mu), color=c, lw=2.4, zorder=4)
            for hh in sorted(sub["H"].unique()):
                a = sub[np.isclose(sub["H"], hh)]
                ax.scatter(np.exp(a["ln_gd"]), np.exp(a["ln_eta"]), s=30, color=c, marker=mc.H_MARKS[float(hh)], edgecolors="k", linewidths=0.4,
                           alpha=0.85, zorder=5)
            info.append((int(t), tff))
        ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ (1/s)", labelpad=2)
        ax.set_ylabel(r"樹脂粘度 $\eta$ (Pa·s)", labelpad=2)
        save(fig, f"学習結果_{mat}.png")
    return p_med, info


def legend_flow(d, mat, w=156.0, h=21.0, lab_w=27.0):
    """学習結果の図の凡例（2 行）。行の名前（温度・厚み）は左に文字で書き、その右に凡例を並べる"""
    temps = sorted(d[mat]["tr"]["Tset"].unique())
    hs = sorted(d[mat]["tr"]["H"].unique())
    h_t = [(Line2D([], [], color=T_COLS[i], lw=2.4), Patch(facecolor=T_COLS[i], alpha=0.25)) for i in range(len(temps))]
    h_h = [Line2D([], [], ls="", marker=mc.H_MARKS[float(h)], ms=9, color="0.55", mec="k", mew=0.5) for h in hs]
    with rc():
        fig = plt.figure(figsize=mp.mm(w, h))
        kw = dict(frameon=False, handletextpad=0.35, columnspacing=0.9, borderaxespad=0.0, handlelength=1.6, loc="center left")
        for k, (lab, hdl, labs) in enumerate([("温度", h_t, [f"{int(t)} ℃" for t in temps]), ("厚み (mm)", h_h, [f"{h:.1f}" for h in hs])]):
            yc = 0.75 - 0.5 * k
            fig.text(0.0, yc, lab, ha="left", va="center")
            lg = fig.legend(hdl, labs, ncol=len(hdl), handler_map={tuple: HandlerTuple(ndivide=None)}, bbox_to_anchor=(lab_w / w, yc), **kw)
            fig.add_artist(lg)
        save(fig, f"凡例_学習結果_{mat}.png", transparent=True)


def fig_test_position(d, w=150.0, h=100.0, H=2.0, margins=(27, 6, 19, 3)):
    """学習データ（塗り）と検証データ（白抜き）の、せん断速度 ― 粘度の上での位置（PP、1 つの厚み H の、全温度の成形条件ごとの平均）"""
    pool, train_c = d["pool"], set(d["tr"]["cond"])
    pool = pool[np.isclose(pool["H"], H)]
    a = pool.groupby("cond").agg(lg=("ln_gd", "mean"), le=("ln_eta", "mean")).reset_index()
    trm = a["cond"].isin(train_c).to_numpy()
    with rc():
        fig, ax = mpd.new_fig(w, h, *margins)
        mp.log_axes(ax, (20.0, 1e4), mpd.half_decade(np.exp(a["le"].min()), np.exp(a["le"].max()), pad=1.6))
        ax.scatter(np.exp(a["lg"][trm]), np.exp(a["le"][trm]), s=55, color=PAL["main"], alpha=0.8, edgecolors="none", zorder=3)
        ax.scatter(np.exp(a["lg"][~trm]), np.exp(a["le"][~trm]), s=85, facecolors="white", edgecolors=PAL["accent"], linewidths=2.0, zorder=4)
        lo = a[~trm & (a["le"] > a["le"].median())]
        hi = a[~trm & (a["le"] <= a["le"].median())]
        p_lo = (np.exp(lo["lg"].max()), np.exp(lo["le"][lo["lg"].idxmax()]))
        p_hi = (np.exp(hi["lg"].min()), np.exp(hi["le"][hi["lg"].idxmin()]))
        arrow = dict(arrowstyle="-", lw=1.2, color=PAL["accent"], shrinkA=1, shrinkB=5)
        ax.annotate("検証（低射出率側）", xy=p_lo, xytext=(p_lo[0] * 2.0, p_lo[1] * 1.6), color=PAL["accent"], fontweight="bold", va="center",
                    ha="left", arrowprops=arrow)
        ax.annotate("検証（高射出率側）", xy=p_hi, xytext=(p_hi[0] / 2.0, p_hi[1] / 2.6), color=PAL["accent"], fontweight="bold", va="center",
                    ha="right", arrowprops=arrow)
        ax.text(0.58, 0.80, "学習データ（塗り）", transform=ax.transAxes, color=PAL["main"], fontweight="bold", ha="left", va="center")
        ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ (1/s)", labelpad=1)
        ax.set_ylabel(r"樹脂粘度 $\eta$ (Pa·s)", labelpad=1)
        save(fig, "検証の位置_PP.png")


def fig_shift_effect(rows, w=180.0, h=112.0, margins=(26, 4, 13, 16)):
    with rc():
        fig, ax = mpd.new_fig(w, h, *margins)
        x = np.arange(len(rows))
        b0 = [mp.pct(r["残差（換算前）"]) for r in rows]
        b1 = [mp.pct(r["残差（換算後）"]) for r in rows]
        ax.bar(x - 0.2, b0, 0.38, color="#B8C0C8", edgecolor="k", lw=0.6, label="換算前")
        ax.bar(x + 0.2, b1, 0.38, color=PAL["main"], edgecolor="k", lw=0.6, label="換算後")
        for i in range(len(rows)):
            ax.text(x[i] - 0.2, b0[i] + 0.25, f"{b0[i]:.1f}", ha="center", va="bottom")
            ax.text(x[i] + 0.2, b1[i] + 0.25, f"{b1[i]:.1f}", ha="center", va="bottom", color=PAL["main"], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([mp.MAT_JA[r["mat"]] for r in rows])
        ax.tick_params(axis="x", which="both", top=False, bottom=False)
        ax.tick_params(axis="y", right=False)
        ax.set_ylabel("温度によるずれ (%)")
        ax.set_ylim(0, max(b0) * 1.2)
        ax.legend(loc="lower left", frameon=False, ncol=2, bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2)
        ax.grid(True, axis="y", ls="-", lw=0.5, color="0.85", zorder=0)
        ax.set_axisbelow(True)
        save(fig, "温度シフトの効果.png")


def fig_parity(d, mat, w=128.0, h=118.0, margins=(27, 4, 19, 3)):
    p = d["pred"]
    rmse = float(np.sqrt(np.mean((p["ln_eta"] - p["予測_ln_eta"]) ** 2)))
    cover = float(np.mean((p["eta"] >= p["下限95"]) & (p["eta"] <= p["上限95"])))
    lo = 10.0 ** np.floor(np.log10(min(p["eta"].min(), p["下限95"].min()) / 1.3))
    hi = 10.0 ** np.ceil(np.log10(max(p["eta"].max(), p["上限95"].max()) * 1.3))
    with rc():
        fig, ax = mpd.new_fig(w, h, *margins)
        mp.log_axes(ax, (lo, hi), (lo, hi))
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.4, color="0.35", zorder=1)
        for g, col, mk in [("低速側", LOW_COL, "o"), ("高速側", HIGH_COL, "s")]:
            q = p[p["group"] == g]
            yerr = np.vstack([q["予測η"] - q["下限95"], q["上限95"] - q["予測η"]])
            ax.errorbar(q["eta"], q["予測η"], yerr=yerr, ls="", marker=mk, ms=7, mfc=col, mec="k", mew=0.4, ecolor=col, elinewidth=1.1,
                        capsize=0, alpha=0.85, zorder=3)
        ax.set_aspect("equal")
        ax.set_xlabel(r"実測値 $\eta$ (Pa·s)", labelpad=2)
        ax.set_ylabel(r"予測値 $\eta$ (Pa·s)", labelpad=2)
        save(fig, f"予測検証_{mat}.png")
    return rmse, cover, len(p)


def legend_parity(w=300.0, h=12.0):
    hdl = [Line2D([], [], ls="", marker="o", ms=9, mfc=LOW_COL, mec="k", mew=0.4, label="検証（低射出率側）"),
           Line2D([], [], ls="", marker="s", ms=9, mfc=HIGH_COL, mec="k", mew=0.4, label="検証（高射出率側）"),
           Line2D([], [], color="0.4", lw=1.4, label="縦線：95% 予測区間"),
           Line2D([], [], color="0.35", lw=1.4, ls="--", label="予測値 ＝ 実測値")]
    with rc():
        fig = plt.figure(figsize=mp.mm(w, h))
        fig.legend(handles=hdl, loc="center", ncol=4, frameon=False, handletextpad=0.35, columnspacing=1.4, borderaxespad=0.0, handlelength=1.6)
        save(fig, "凡例_予測検証.png", transparent=True)


def fig_error_compare(rows, w=190.0, h=116.0, margins=(26, 4, 13, 23)):
    with rc():
        fig, ax = mpd.new_fig(w, h, *margins)
        x = np.arange(len(rows))
        g = [mp.pct(r["RMSE 全体"]) for r in rows]
        c = [mp.pct(r["Cross-WLF RMSE 全体"]) for r in rows]
        ax.bar(x - 0.2, g, 0.38, color=PAL["main"], edgecolor="k", lw=0.6, label="本モデル（ガウス過程回帰）")
        ax.bar(x + 0.2, c, 0.38, color="#B8C0C8", edgecolor="k", lw=0.6, label="Cross-WLF モデル（参考）")
        for i in range(len(rows)):
            ax.text(x[i] - 0.2, g[i] + 1.5, f"{mp.r0(g[i])}", ha="center", va="bottom", color=PAL["main"], fontweight="bold")
            ax.text(x[i] + 0.2, c[i] + 1.5, f"{mp.r0(c[i])}", ha="center", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels([mp.MAT_JA[r["mat"]] for r in rows])
        ax.tick_params(axis="x", which="both", top=False, bottom=False)
        ax.tick_params(axis="y", right=False)
        ax.set_ylabel("平均的な予測誤差 (%)")
        ax.set_ylim(0, max(c) * 1.18)
        ax.legend(loc="lower left", frameon=False, ncol=1, bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2, labelspacing=0.2)
        ax.grid(True, axis="y", ls="-", lw=0.5, color="0.85", zorder=0)
        ax.set_axisbelow(True)
        save(fig, "予測誤差の比較.png")


# ---------------------------------------------------------------- 厚みごとの比較（ノート「…_厚みごとの比較.ipynb」の CSV を使う）
def thick_dir(mat):
    """ノート「材料別_温度シフトGPR_…_厚みごとの比較.ipynb」の結果の置き場所"""
    return os.path.join(os.path.dirname(mp.compare_dir(mat)), "厚みごとの比較")


def thick_csv(mat, name):
    return pd.read_csv(os.path.join(thick_dir(mat), name), encoding="utf-8-sig")


def fig_flow_thick(d, mat, H, w=150.0, h=86.0, margins=(27, 5, 19, 3)):
    """1 つの厚みの学習結果（主のモデル mp.KEY）：加熱シリンダ温度ごとの曲線と 95% 予測区間（入力は、同じ厚み・温度の条件平均をせん断速度で内挿。
    ノートの「厚みごと」の図と同じ）。点＝ショットごとの値（学習に使った値。色・形＝温度、塗り＝学習、白抜き＝検証）"""
    fit, pool, train_c = d["fit"], d["pool"], set(d["tr"]["cond"])
    ph = pool[np.isclose(pool["H"], H)]
    xs, ys = [], []
    with rc():
        fig, ax = mpd.new_fig(w, h, *margins)
        for i, t in enumerate(sorted(ph["Tset"].unique())):
            sub = ph[ph["Tset"] == t]
            lg = np.linspace(sub["ln_gd"].min() - np.log(1.3), sub["ln_gd"].max() + np.log(1.3), 300)
            mu, sd, _, _ = mc.predict(fit, mc.flow_inputs(sub, lg, H, t))
            c, mk = T_COLS[i], mpd.T_MARKS[i]
            ax.fill_between(np.exp(lg), np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd), color=c, alpha=0.16, lw=0, zorder=2)
            ax.plot(np.exp(lg), np.exp(mu), color=c, lw=2.4, zorder=4)
            trm = sub["cond"].isin(train_c).to_numpy()
            ax.scatter(np.exp(sub["ln_gd"][trm]), np.exp(sub["ln_eta"][trm]), s=32, color=c, marker=mk, edgecolors="k", linewidths=0.4, zorder=6)
            ax.scatter(np.exp(sub["ln_gd"][~trm]), np.exp(sub["ln_eta"][~trm]), s=32, facecolors="white", edgecolors=c, marker=mk, linewidths=1.6,
                       zorder=6)
            xs += [lg.min(), lg.max()]
            ys += list(sub["ln_eta"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(np.exp(min(xs)), np.exp(max(xs)))
        ax.set_ylim(*mpd.half_decade(np.exp(min(ys)), np.exp(max(ys)), pad=1.4))
        mc.log_ticks(ax)
        # 横軸の目盛り：範囲が 1.6 桁を超えるときは 1・5 × 10^k だけにし、両端に近い目盛りは省く（数字の重なりと、図の外へのはみ出しを防ぐ）
        lo, hi = np.log10(ax.get_xlim())
        mults = [1, 2, 5] if hi - lo <= 1.6 else [1, 5]
        t = [m * 10.0 ** k for k in range(int(np.floor(lo)), int(np.ceil(hi)) + 1) for m in mults]
        t = [v for v in t if lo + 0.03 * (hi - lo) <= np.log10(v) <= hi - 0.03 * (hi - lo)]
        ax.set_xticks(t, [f"{v:g}" for v in t])
        # 縦軸：1.6 桁を超えるときは 10^k の目盛りだけにする（厚みによって書き方がそろわないのを防ぐ）
        lo, hi = np.log10(ax.get_ylim())
        if hi - lo > 1.6:
            ks = range(int(np.ceil(lo)), int(np.floor(hi)) + 1)
            ax.set_yticks([10.0 ** k for k in ks], [f"$10^{{{k}}}$" for k in ks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which="both", top=False, right=False)
        ax.grid(True, which="major", ls="-", lw=0.6, color="0.8", zorder=0)
        ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ (1/s)", labelpad=2)
        ax.set_ylabel(r"樹脂粘度 $\eta$ (Pa·s)", labelpad=2)
        save(fig, f"厚み_学習結果_{mat}_{H:g}mm.png")


def legend_thick(d, mat, w=310.0, h=11.0):
    temps = sorted(d[mat]["pool"]["Tset"].unique())
    hdl = [Line2D([], [], color=T_COLS[i], lw=2.4, marker=mpd.T_MARKS[i], ms=9, mec="k", mew=0.5, label=f"{int(t)} ℃") for i, t in enumerate(temps)]
    hdl += [Line2D([], [], ls="", marker="o", ms=9, color="0.5", mec="k", mew=0.5, label="学習データ（塗り）"),
            Line2D([], [], ls="", marker="o", ms=9, mfc="white", mec="0.4", mew=1.8, label="検証データ（白抜き）")]
    with rc():
        fig = plt.figure(figsize=mp.mm(w, h))
        fig.legend(handles=hdl, loc="center", ncol=len(hdl), frameon=False, handletextpad=0.35, columnspacing=1.3, borderaxespad=0.0, handlelength=1.8)
        save(fig, f"凡例_厚み_{mat}.png", transparent=True)


def bars_by_h(ax, hs, series, labels=False):
    """厚みごとの棒（series：[(値の配列, 色, 凡例), ...]）"""
    x = np.arange(len(hs))
    wd = 0.8 / len(series)
    for i, (v, c, lab) in enumerate(series):
        xx = x + (i - (len(series) - 1) / 2) * wd
        ax.bar(xx, v, wd * 0.95, color=c, edgecolor="k", lw=0.5, label=lab, zorder=3)
        if labels:
            for xi, vi in zip(xx, v):
                ax.text(xi, vi, f"{r0_(vi)}", ha="center", va="bottom", fontsize=FS - 2 if len(series) > 2 else FS)
    ax.set_xticks(x)
    ax.set_xticklabels([h if isinstance(h, str) else f"{h:.1f}" for h in hs])
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.tick_params(axis="y", right=False)
    ax.grid(True, axis="y", ls="-", lw=0.5, color="0.85", zorder=0)
    ax.set_axisbelow(True)


def r0_(v):
    return mp.r0(v) if np.isfinite(v) else ""


def fig_thick_criteria(mat, w=150.0, h=108.0, margins=(24, 4, 19, 26)):
    """(a) 厚みごとに係数を決め直したときの重ね合わせのずれ、(c) 検証データの予測誤差（厚みごと）の 2 枚"""
    s = thick_csv(mat, "重ね合わせ_厚みごと.csv").sort_values("厚み")
    v = thick_csv(mat, "検証誤差_厚みごと.csv")
    v = v[(v["範囲"] == "全体") & (v["厚み"] != "全厚み")].assign(厚み=lambda t: t["厚み"].astype(float))
    pv = v.pivot(index="厚み", columns="モデル", values="誤差_%").sort_index()
    for name, hs, series, ylab in [
            ("厚み_重ね合わせ", s["厚み"].tolist(), [(s["ずれ_換算前_%"], "#B8C0C8", "換算前"), (s["ずれ_Tset_%"], "#7FA7D0", "加熱シリンダ温度で換算"),
                                                 (s["ずれ_Tff_%"], PAL["main"], "金型内樹脂最高温度で換算")], "重ね合わせのずれ (%)"),
            ("厚み_検証誤差", pv.index.tolist(), [(pv["C"], "#B8C0C8", "シフトなし（C）"), (pv["B_Tset"], "#7FA7D0", "加熱シリンダ温度でシフト（B_Tset）"),
                                               (pv["B_Tff"], PAL["main"], "金型内樹脂最高温度でシフト（B_Tff）")], "予測誤差 (%)")]:
        with rc():
            fig, ax = mpd.new_fig(w, h, *margins)
            bars_by_h(ax, hs, series)
            ax.set_xlabel("キャビティ厚み (mm)", labelpad=2)
            ax.set_ylabel(ylab, labelpad=2)
            ax.legend(loc="lower left", bbox_to_anchor=(-0.12, 1.0), frameon=False, ncol=1, borderaxespad=0.2, labelspacing=0.15, handlelength=1.2)
            save(fig, f"{name}_{mat}.png")


THICK_MODELS = ["C", "B_Tset", "B_Tff"]


def fig_thick_total(w=310.0, h=80.0, margins=(24, 4, 11, 21), gap=12.0):
    """厚みごとに学習したモデルと、総合データ（全厚み）で学習したモデルの、検証データ全体の予測誤差。材料ごとに 1 枠、モデルごとに 2 本"""
    p = {}
    for mat in mp.MATS:
        c = thick_csv(mat, "厚みごとと総合の比較.csv")
        c = c[(c["範囲"] == "全体") & (c["厚みの名前"] == "全厚み")]
        p[mat] = c.pivot(index="モデル", columns="学習", values="誤差_%").loc[THICK_MODELS]
    top = max(float(v.max().max()) for v in p.values()) * 1.22
    with rc():
        fig = plt.figure(figsize=mp.mm(w, h))
        l0, r_, b0, t0 = margins
        pw = (w - l0 - r_ - 2 * gap) / 3
        for k, mat in enumerate(mp.MATS):
            ax = fig.add_axes([(l0 + k * (pw + gap)) / w, b0 / h, pw / w, (h - b0 - t0) / h])
            bars_by_h(ax, THICK_MODELS, [(p[mat]["総合"], PAL["main"], "総合データ（全厚み）で学習"), (p[mat]["厚みごと"], PAL["accent"], "厚みごとに学習")],
                      labels=True)
            ax.set_ylim(0, top)
            ax.set_title(mp.MAT_JA[mat], fontsize=FS, pad=4)
            if k == 0:
                ax.set_ylabel("予測誤差 (%)", labelpad=2)
        hd, lb = ax.get_legend_handles_labels()
        fig.legend(hd, lb, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2, columnspacing=2.0)
        save(fig, "厚みと総合.png")


def thick_figs(d):
    """厚みごとの比較の図（ノートの CSV がなければ、作らずに知らせる）"""
    if not all(os.path.exists(os.path.join(thick_dir(m), "厚みごとの比較_まとめ.csv")) for m in mp.MATS):
        print("★ 厚みごとの比較の CSV がない。先にノート「材料別_温度シフトGPR_ARD対応Matern52＋線形＋ホワイトノイズ_厚みごとの比較.ipynb」を実行する")
        return
    for mat in mp.MATS:
        for H in sorted(d[mat]["pool"]["H"].unique()):
            fig_flow_thick(d[mat], mat, float(H))
        legend_thick(d, mat)
        fig_thick_criteria(mat)
    if all(os.path.exists(os.path.join(thick_dir(m), "厚みごとと総合の比較.csv")) for m in mp.MATS):
        fig_thick_total()


# ---------------------------------------------------------------- まとめ
def main():
    os.makedirs(OUT, exist_ok=True)
    fig_formulas()
    fig_gpr_concept()
    bayes = fig_bayes_example()
    fig_bayes_samples()
    lmls = fig_lml_compare()
    fig_mean_function()
    fig_kernel_distance()
    fig_kernel_examples()
    fig_matern_samples()
    fig_ard()
    fig_linear_noise()
    fig_sum_samples()
    fig_shift_concept()
    d = mpd.load()
    fig_shift_factors(d)
    prof = fig_profile("PP")
    mpd.fig_header_art(d["PP"], "PP", "S", 138.0, 112.0, line="#FFFFFF", band=PAL["accent"])
    fig_compare_mini()
    fig_test_position(d["PP"])
    mpd.fig_master(d["PP"], "PP", "S", 310.0, 92.0, FS, H=2.0, gap=34.0, margins=(27, 4, 20, 9))
    mpd.legend_temps(d, "S", 310.0, 12.0, FS)
    mpd.fig_limit(d["MIX"], "MIX", "S", 172.0, 122.0, 18.5, margins=(28, 5, 20, 4))
    legend_parity()
    rows = []
    for mat in mp.MATS:
        dm = d[mat]
        fit = dm["fit"]
        p_med, info = fig_flow(dm, mat)
        legend_flow(d, mat)
        rmse, cover, n = fig_parity(dm, mat)
        met = pd.read_csv(os.path.join(mp.compare_dir(mat), "予測検証_3data_指標.csv"), encoding="utf-8-sig")
        get = lambda m, rng: float(met[(met["モデル"] == m) & (met["範囲"] == rng)]["RMSE"].iloc[0])
        assert np.isclose(rmse, get(mp.KEY, "全体")), f"{mat}：RMSE が指標の表と違う"
        summ = pd.read_csv(os.path.join(mp.compare_dir(mat), "まとめ.csv"), encoding="utf-8-sig")
        dl = lambda m: float(summ[summ["モデル"] == m]["θ=0 との LML の差"].iloc[0])
        r_before, _ = mpd.shift_resid(dm, False)
        r_after, _ = mpd.shift_resid(dm, True)
        noise_sd = float(np.sqrt(fit["gp"].kernel_.k2.noise_level))
        g = dm["pool"].groupby("cond")["ln_eta"]
        shot_sd = float(np.sqrt((g.var(ddof=1) * (g.count() - 1)).sum() / (g.count() - 1).sum()))     # 同じ成形条件のショット間の標準偏差（純粋誤差）
        hp = pd.read_csv(os.path.join(mp.compare_dir(mat), "ハイパーパラメータ.csv"), encoding="utf-8-sig").set_index("model").loc[mp.KEY]
        ls = {k: float(hp[c]) for k, c in LS_COLS}
        cm = dm["tr"].groupby("cond")["ln_gd"].mean()
        T = dm["tr"][SHIFT_T]
        rows.append({"mat": mat, "材料": mp.MAT_JA[mat], "シフトの型": fit["cfg"]["type"], "基準温度": fit["cfg"]["Tref"], "横の係数": fit["th"],
                     "縦の係数": fit["tv"], "LML の差（主）": dl(mp.KEY), "LML の差（B_Tset）": dl("B_Tset"), "LML の差（B_Tff）": dl("B_Tff"),
                     "残差（換算前）": float(np.sqrt(np.mean(r_before["r"] ** 2))), "残差（換算後）": float(np.sqrt(np.mean(r_after["r"] ** 2))),
                     "RMSE 全体": rmse, "RMSE 低速側": get(mp.KEY, "低速側"), "RMSE 高速側": get(mp.KEY, "高速側"), "被覆率": cover, "検証の行数": n,
                     "Cross-WLF RMSE 全体": get("Cross-WLF", "全体"), "ノイズの標準偏差": noise_sd, "ショット間の標準偏差": shot_sd,
                     **{f"長さスケール（{k}）": v for k, v in ls.items()}, "振幅の 2 乗": float(hp["amplitude"]) ** 2,
                     "線形項の大きさ": float(hp["lin_amplitude"]) ** 2, "線形項の切片": float(hp["lin_sigma0"]), "ノイズの分散": float(hp["noise_var"]),
                     "ln η の平均": fit["y0"],
                     "学習の行数": len(dm["tr"]), "学習の条件数": dm["tr"]["cond"].nunique(), "γ̇ 最小": float(np.exp(cm.min())), "γ̇ 最大": float(np.exp(cm.max())),
                     "シフト温度 最小": float(T.min()), "シフト温度 最大": float(T.max()),
                     "a_v（最小の温度）": float(np.exp(mc.shift_ln(T.min(), fit["tv"], fit["cfg"]))),
                     "a_v（最大の温度）": float(np.exp(mc.shift_ln(T.max(), fit["tv"], fit["cfg"]))),
                     "a_h（最小の温度）": float(np.exp(mc.shift_ln(T.min(), fit["th"], fit["cfg"]))),
                     "a_h（最大の温度）": float(np.exp(mc.shift_ln(T.max(), fit["th"], fit["cfg"]))),
                     "ΔP2 の中央値 [MPa]": p_med, "Tff の中央値 [℃]": "・".join(f"{t}:{v:.1f}" for t, v in info)})
    fig_shift_effect(rows)
    fig_error_compare(rows)
    thick_figs(d)
    tab = pd.DataFrame(rows)
    for c in ["RMSE 全体", "RMSE 低速側", "RMSE 高速側", "Cross-WLF RMSE 全体", "残差（換算前）", "残差（換算後）", "ノイズの標準偏差", "ショット間の標準偏差"]:
        tab[f"{c} [%]"] = [mp.pct(v) for v in tab[c]]
    tab.drop(columns="mat").to_csv(os.path.join(OUT, "スライドの数値.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(tab.drop(columns=["mat", "Tff の中央値 [℃]"]).round(3).T)
    print("ベイズの例（事前・尤度・事後の中心 Pa·s）:", np.round(bayes, 1))
    print("周辺尤度の比較:", [(lab, round(l, 3), round(v, 2)) for lab, l, v in lmls])
    print("プロファイル尤度（PP）: 最大", prof[:2], "θ=0 との差", round(prof[2], 2))
    print("保存先:", OUT)


if __name__ == "__main__":
    main()
