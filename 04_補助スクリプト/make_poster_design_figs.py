# -*- coding: utf-8 -*-
"""オリジナルデザインのポスター 3 案（build_poster_designs.py が作る）の図を作る（日本語。学習はやり直さない）。

使うモデルは、報告書B・前のポスターと同じ B_Tff・条件平均の入力（保存済みの 学習済みモデル_B_Tff.joblib と、予測検証の表）。
図は、案ごとに、ポスターに貼る大きさ（mm）で作る（文字の大きさが、そのままポスター上の pt になる）。色は案ごとの配色に合わせる。
  <案>_予測曲線_<材料>.png    ：厚みごとの予測曲線（中央の加熱シリンダ温度）。点＝成形条件ごとの平均（塗り＝学習、白抜き＝検証）、線＝予測平均値、帯＝95% 予測区間
  <案>_予測検証_<材料>.png    ：学習に用いていない射出率条件の、実測と予測の粘度（縦線＝95% 予測区間）
  <案>_温度のずれ_<材料>.png  ：温度シフトファクターで換算する前と後の、温度によるずれ（厚みごとに、全温度のデータへ 1 本の曲線〔3 次式〕を当てはめたときの、
                               曲線からのずれ。前のポスター・報告書B の「換算前の約 4〜5 割に減少」と同じ計算）
  <案>_苦手な範囲.png         ：学習データのない低・高せん断速度側（外挿）の予測（全厚み共通の曲線。報告書B の図5 と同じ固定値）
  <案>_検証の位置_<材料>.png   ：学習データと検証データの、せん断速度 ― 粘度の上での位置（1 つの厚み）
  <案>_題目の飾り.png         ：題目の帯の飾り（実際の予測曲線。背景は透明）
  <案>_GPR概念図.png、<案>_温度シフト概念図.png、<案>_従来と本研究.png（人工のデータ・曲線による説明の図）、<案>_誤差の比較.png、<案>_凡例_*.png
  ポスターの数値_オリジナル.csv：ポスターに書いた数値の元
（fig_master〔換算前と換算後の粘度曲線〕・legend_temps は、今の 3 案では使っていない）
使い方（リポジトリ直下で）: uv run python 04_補助スクリプト/make_poster_design_figs.py
出力: 05_材料ごとの機械学習データと予測検証データ/学習結果報告書作成_261015/ポスター/図_オリジナル/
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import make_poster_figs as mp  # noqa: E402  （図の書式〔IPAexGothic など〕と、共通の部品を使う）

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch  # noqa: E402

import mat_common as mc  # noqa: E402

MATS = mp.MATS
MAT_JA = mp.MAT_JA
OUT = os.path.join(mp.POSTER, "図_オリジナル")
PAL = {   # 案ごとの配色（build_poster_designs.py と同じ）
    "A": {"main": "#14213D", "accent": "#F28F1C", "sub": "#5B6577", "light": "#E8EEF7"},
    "B": {"main": "#00695C", "accent": "#F4511E", "sub": "#546E7A", "light": "#E0F2F1"},
    "C": {"main": "#22303C", "accent": "#D1495B", "sub": "#4A6274", "light": "#EEF1F4"},
}
T_COLS = ["#2F6DB5", "#7B8794", "#D64545"]       # 加熱シリンダ温度（低・中・高）の色：青・灰・赤
T_MARKS = ["o", "^", "s"]                         # 加熱シリンダ温度の形（報告書B の図4 と同じ：低い順に ●・▲・■）
LOW_COL, HIGH_COL = "#2F6DB5", "#D64545"          # 検証の低射出率側・高射出率側
GRAY = "#9AA3AE"


# ---------------------------------------------------------------- 共通の部品
def new_fig(w, h, left, right, bottom, top):
    """大きさ w × h [mm] の図。余白（left・right・bottom・top）も mm で指定する（図をそろえて並べるため）"""
    fig, ax = plt.subplots(figsize=mp.mm(w, h))
    fig.subplots_adjust(left=left / w, right=1 - right / w, bottom=bottom / h, top=1 - top / h)
    return fig, ax


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=mp.DPI)
    plt.close(fig)


def fonts(fs):
    """目盛り fs [pt] を基準にした、軸名・凡例の文字の大きさ"""
    return {"xtick.labelsize": fs, "ytick.labelsize": fs, "axes.labelsize": fs + 1.5, "legend.fontsize": fs - 1, "font.size": fs}


def half_decade(lo, hi, pad=1.15):
    """データの範囲を少し広げて、10^(0.5 刻み) で切った範囲"""
    return 10.0 ** (np.floor(2 * np.log10(lo / pad)) / 2), 10.0 ** (np.ceil(2 * np.log10(hi * pad)) / 2)


def log_axes(ax, xlim, ylim):
    mp.log_axes(ax, xlim, ylim)


def load():
    mc.INPUT_MODE = mp.MODE
    mc.SHOW_FIGS = False
    mc.D = mc.load_all(MATS)
    out = {}
    for mat in MATS:
        tr = mc.D[(mat, "5data")]
        te = mc.add_groups(tr, mc.D[(mat, "3data")])
        fit = joblib.load(os.path.join(mp.model_dir(mat), f"学習済みモデル_{mp.KEY}.joblib"))["fit"]
        assert fit.get("mode") == mp.MODE, f"{mc.MODE_JA[mp.MODE]}のモデルのはず"
        pred = pd.read_csv(os.path.join(mp.model_dir(mat), "予測検証_3data_予測値.csv"), encoding="utf-8-sig")
        out[mat] = {"tr": tr, "te": te, "pool": pd.concat([tr, te], ignore_index=True), "fit": fit, "pred": pred}
    return out


# ---------------------------------------------------------------- 学習結果：厚みごとの予測曲線
def fig_flow(d, mat, des, w, h, fs, margins=(24, 4, 17, 4), xlim=(20.0, 5e4), labels=True, name=None):
    """中央の加熱シリンダ温度での、厚みごとの予測曲線。ΔP2・金型内樹脂最高温度は、同じ厚み・温度の条件ごとの平均を、せん断速度で内挿した値
    （ノートの「せん断速度_粘度_設定温度ごと」と同じ。平均関数と帯は、グラフエリアの端まで描く）"""
    tr, pool, fit = d["tr"], d["pool"], d["fit"]
    t = sorted(tr["Tset"].unique())[1]
    train_c = set(tr["cond"])
    with plt.rc_context(fonts(fs)):
        fig, ax = new_fig(w, h, *margins)
        sub_t = pool[pool["Tset"] == t]
        a_all = sub_t.groupby("cond")["ln_eta"].mean()
        ylim = half_decade(np.exp(a_all.min()), np.exp(a_all.max()))
        log_axes(ax, xlim, ylim)
        lg = np.linspace(np.log(xlim[0]), np.log(xlim[1]), 400)
        for hh in sorted(sub_t["H"].unique()):
            sub = sub_t[np.isclose(sub_t["H"], hh)]
            col, mk = mc.H_COLORS[float(hh)], mc.H_MARKS[float(hh)]
            mu, sd, _, _ = mc.predict(fit, mc.flow_inputs(sub, lg, hh, t))
            ax.fill_between(np.exp(lg), np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd), color=col, alpha=0.18, lw=0, zorder=2)
            ax.plot(np.exp(lg), np.exp(mu), color=col, lw=2.0, zorder=4)
            a = sub.groupby("cond").agg(lg=("ln_gd", "mean"), le=("ln_eta", "mean")).reset_index()
            trm = a["cond"].isin(train_c).to_numpy()
            ms = (fs * 0.62) ** 2
            ax.scatter(np.exp(a["lg"][trm]), np.exp(a["le"][trm]), s=ms, color=col, marker=mk, edgecolors="k", linewidths=0.5, zorder=6)
            ax.scatter(np.exp(a["lg"][~trm]), np.exp(a["le"][~trm]), s=ms, facecolors="white", edgecolors=col, marker=mk, linewidths=1.8, zorder=6)
        if labels:
            ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ (1/s)", labelpad=2)
            ax.set_ylabel(r"樹脂粘度 $\eta$ (Pa·s)", labelpad=2)
        save(fig, name or f"{des}_予測曲線_{mat}.png")
    return int(t)


def legend_flow(des, w, h, fs, hs=(0.5, 1.0, 1.5, 2.0), ncol=None):
    """予測曲線の図の凡例（横 1 列）：厚み（色・形）、学習・検証（塗り・白抜き）、予測平均値・95% 予測区間"""
    hdl = [Line2D([], [], ls="", marker=mc.H_MARKS[h], ms=fs * 0.75, color=mc.H_COLORS[h], mec="k", mew=0.5, label=f"{h:g} mm") for h in hs]
    hdl += [Line2D([], [], ls="", marker="o", ms=fs * 0.75, color="0.45", mec="k", mew=0.5, label="学習データ"),
            Line2D([], [], ls="", marker="o", ms=fs * 0.75, mfc="white", mec="0.35", mew=1.6, label="検証データ（学習に不使用）"),
            Line2D([], [], color="0.3", lw=2.0, label="予測平均値"), Patch(facecolor="0.3", alpha=0.22, label="95% 予測区間")]
    with plt.rc_context(fonts(fs)):
        fig = plt.figure(figsize=mp.mm(w, h))
        fig.legend(handles=hdl, loc="center", ncol=ncol or len(hdl), frameon=False, fontsize=fs, handletextpad=0.3, columnspacing=1.1,
                   borderaxespad=0.0, handlelength=1.6)
        save(fig, f"{des}_凡例_予測曲線.png")


# ---------------------------------------------------------------- 予測検証
def fig_parity(d, mat, des, w, h, fs, margins=(22, 4, 16, 4), legend=True, note=True):
    p = d["pred"]
    rmse = float(np.sqrt(np.mean((p["ln_eta"] - p["予測_ln_eta"]) ** 2)))
    cover = float(np.mean((p["eta"] >= p["下限95"]) & (p["eta"] <= p["上限95"])))
    lo = 10.0 ** np.floor(np.log10(min(p["eta"].min(), p["下限95"].min()) / 1.3))
    hi = 10.0 ** np.ceil(np.log10(max(p["eta"].max(), p["上限95"].max()) * 1.3))
    with plt.rc_context(fonts(fs)):
        fig, ax = new_fig(w, h, *margins)
        log_axes(ax, (lo, hi), (lo, hi))
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color="0.35", zorder=1, label="予測値 ＝ 実測値")
        for g, col, mk, lab in [("低速側", LOW_COL, "o", "低射出率側"), ("高速側", HIGH_COL, "s", "高射出率側")]:
            q = p[p["group"] == g]
            yerr = np.vstack([q["予測η"] - q["下限95"], q["上限95"] - q["予測η"]])
            ax.errorbar(q["eta"], q["予測η"], yerr=yerr, ls="", marker=mk, ms=fs * 0.42, mfc=col, mec="k", mew=0.4, ecolor=col,
                        elinewidth=1.0, capsize=0, alpha=0.85, label=lab, zorder=3)
        ax.set_aspect("equal")
        if legend:
            ax.legend(loc="upper left", frameon=True, framealpha=0.9, borderaxespad=0.3, handletextpad=0.2, fontsize=fs - 1.5, borderpad=0.3,
                      labelspacing=0.25)
        if note:
            ax.text(0.97, 0.04, f"平均的な誤差：約 {mp.r0(mp.pct(rmse))}%\n予測区間に入った割合：{mp.r0(cover * 100)}%", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=fs - 1.5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6"))
        ax.set_xlabel(r"実測値 $\eta$ (Pa·s)", labelpad=2)
        ax.set_ylabel(r"予測値 $\eta$ (Pa·s)", labelpad=2)
        save(fig, f"{des}_予測検証_{mat}.png")
    return rmse, cover


# ---------------------------------------------------------------- 温度シフトの効果
def shift_resid(d, shifted):
    """成形条件ごとの平均（換算前、または B_Tff で換算後）について、厚みごとに全温度のデータへ 3 次式を当てはめた残差（ln η）"""
    fit = d["fit"]
    t = mc.shifted_cond_means(d["pool"], mp.KEY if shifted else None, fit["cfg"], fit["th"] if shifted else 0.0, fit["tv"] if shifted else 0.0)
    rows = []
    for _, g in t.groupby("H"):
        c = np.polyfit(g["x"], g["y"], 3)
        rows.append(g.assign(r=g["y"] - np.polyval(c, g["x"]), fx=np.polyval(c, g["x"])))
    return pd.concat(rows, ignore_index=True), t


def fig_resid(d, mat, des, w, h, fs, gap=8.0, margins=(23, 3, 13, 10)):
    """換算前・換算後の 2 枚：横＝加熱シリンダ温度、縦＝1 本の曲線からのずれ [%]。点＝成形条件（全厚み）、横線＝温度ごとの平均"""
    r0, _ = shift_resid(d, False)
    r1, _ = shift_resid(d, True)
    temps = sorted(d["pool"]["Tset"].unique())
    lim = max(np.abs((np.exp(r0["r"]) - 1) * 100).max(), np.abs((np.exp(r1["r"]) - 1) * 100).max()) * 1.12
    rng = np.random.default_rng(1)
    with plt.rc_context(fonts(fs)):
        fig = plt.figure(figsize=mp.mm(w, h))
        l, r, b, t = margins
        pw = (w - l - r - gap) / 2
        for k, (res, title) in enumerate([(r0, "換算前"), (r1, "換算後")]):
            ax = fig.add_axes([(l + k * (pw + gap)) / w, b / h, pw / w, (h - b - t) / h])
            pct = (np.exp(res["r"]) - 1) * 100
            ax.axhline(0, color="0.3", lw=1.0, zorder=1)
            for i, T in enumerate(temps):
                m = (res["Tset"] == T).to_numpy()
                xj = i + rng.uniform(-0.22, 0.22, m.sum())
                ax.scatter(xj, pct[m], s=(fs * 0.5) ** 2, color=T_COLS[i], alpha=0.75, edgecolors="white", linewidths=0.4, zorder=3)
                ax.plot([i - 0.32, i + 0.32], [pct[m].mean()] * 2, color="k", lw=2.6, zorder=4, solid_capstyle="butt")
            rmse = float(np.sqrt(np.mean(res["r"] ** 2)))
            ax.set_title(f"{title}（ずれの大きさ {mp.pct(rmse):.1f}%）", fontsize=fs + 0.5, pad=4, color="k")
            ax.set_xlim(-0.6, len(temps) - 0.4)
            ax.set_ylim(-lim, lim)
            ax.set_xticks(range(len(temps)))
            ax.set_xticklabels([f"{int(T)} ℃" for T in temps])
            ax.tick_params(axis="x", which="both", top=False, bottom=False)
            ax.tick_params(axis="y", right=False)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5, symmetric=True))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: (f"+{v:.0f}" if v > 0 else f"−{-v:.0f}") if round(v) else "0"))
            ax.grid(True, axis="y", ls=":", lw=0.6, color="0.75", zorder=0)
            if k == 0:
                ax.set_ylabel("1 本の曲線からのずれ (%)", labelpad=2)
            else:
                ax.set_yticklabels([])
        save(fig, f"{des}_温度のずれ_{mat}.png")
    return {T: float(((np.exp(r0["r"]) - 1) * 100)[r0["Tset"] == T].mean()) for T in temps}, \
           {T: float(((np.exp(r1["r"]) - 1) * 100)[r1["Tset"] == T].mean()) for T in temps}


def fig_master(d, mat, des, w, h, fs, H, gap=30.0, margins=(23, 3, 16, 9)):
    """1 つの厚みの、換算前（η と γ̇）と換算後（η/a_v と γ̇·a_h）の粘度曲線。破線＝全温度のデータに当てはめた 1 本の曲線（3 次式）"""
    temps = sorted(d["pool"]["Tset"].unique())
    train_c = set(d["tr"]["cond"])
    with plt.rc_context(fonts(fs)):
        fig = plt.figure(figsize=mp.mm(w, h))
        l, r, b, t = margins
        pw = (w - l - r - gap) / 2
        for k, (shifted, title, xl, yl) in enumerate([(False, "換算前", r"せん断速度 $\dot{\gamma}$ (1/s)", r"樹脂粘度 $\eta$ (Pa·s)"),
                                                      (True, "換算後", r"換算せん断速度 $\dot{\gamma}\,a_h$ (1/s)", r"換算粘度 $\eta\,/\,a_v$ (Pa·s)")]):
            ax = fig.add_axes([(l + k * (pw + gap)) / w, b / h, pw / w, (h - b - t) / h])
            res, _ = shift_resid(d, shifted)
            g = res[np.isclose(res["H"], H)].sort_values("x")
            xs = np.linspace(g["x"].min() - 0.15, g["x"].max() + 0.15, 200)
            c = np.polyfit(g["x"], g["y"], 3)
            log_axes(ax, (np.exp(xs[0]), np.exp(xs[-1])), half_decade(np.exp(g["y"].min()), np.exp(g["y"].max()), pad=1.25))
            ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.plot(np.exp(xs), np.exp(np.polyval(c, xs)), ls="--", lw=1.4, color="0.35", zorder=2)
            for i, T in enumerate(temps):
                q = g[g["Tset"] == T]
                trm = q["cond"].isin(train_c).to_numpy()
                ms = (fs * 0.62) ** 2
                ax.scatter(np.exp(q["x"][trm]), np.exp(q["y"][trm]), s=ms, color=T_COLS[i], marker=T_MARKS[i], edgecolors="k", linewidths=0.5, zorder=4)
                ax.scatter(np.exp(q["x"][~trm]), np.exp(q["y"][~trm]), s=ms, facecolors="white", edgecolors=T_COLS[i], marker=T_MARKS[i],
                           linewidths=1.6, zorder=4)
            ax.set_title(title, fontsize=fs + 1, pad=4)
            ax.set_xlabel(xl, labelpad=2)
            ax.set_ylabel(yl, labelpad=2)
        save(fig, f"{des}_マスターカーブ_{mat}.png")


def legend_temps(d, des, w, h, fs, mat="PP", with_line=True):
    temps = sorted(d[mat]["pool"]["Tset"].unique())
    hdl = [Line2D([], [], ls="", marker=T_MARKS[i], ms=fs * 0.75, color=T_COLS[i], mec="k", mew=0.5, label=f"{int(T)} ℃") for i, T in enumerate(temps)]
    if with_line:
        hdl.append(Line2D([], [], ls="--", lw=1.4, color="0.35", label="全温度のデータに当てはめた 1 本の曲線"))
    with plt.rc_context(fonts(fs)):
        fig = plt.figure(figsize=mp.mm(w, h))
        fig.legend(handles=hdl, loc="center", ncol=len(hdl), frameon=False, fontsize=fs, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.0)
        save(fig, f"{des}_凡例_温度.png")


# ---------------------------------------------------------------- 苦手な範囲（外挿）
def fig_limit(d, mat, des, w, h, fs, margins=(24, 4, 17, 4)):
    """全厚み共通の曲線（中央の加熱シリンダ温度。ΔP2 は学習データの中央値、金型内樹脂最高温度は、その温度の学習データの中央値に固定。
    報告書B の図5 と同じ）。学習データのない範囲（学習データの条件平均のせん断速度の外）を灰色にする"""
    tr, fit = d["tr"], d["fit"]
    t = sorted(tr["Tset"].unique())[1]
    sub = tr[tr["Tset"] == t]
    p_med = float(np.exp(tr["ln_P33"].median()) / 1e6)
    tff = float(np.exp(sub["ln_Tff"].median()) - 273.15)
    gmin = float(np.exp(tr.groupby("cond")["ln_gd"].mean().min()))
    gmax = float(np.exp(tr.groupby("cond")["ln_gd"].mean().max()))
    xl, yl = (1.0, 1e5), (0.5, 3000.0)
    pal = PAL[des]
    with plt.rc_context(fonts(fs)):
        fig, ax = new_fig(w, h, *margins)
        log_axes(ax, xl, yl)
        for x0, x1 in [(xl[0], gmin), (gmax, xl[1])]:
            ax.axvspan(x0, x1, color="0.88", zorder=0, lw=0)
        lg = np.linspace(np.log(xl[0]), np.log(xl[1]), 500)
        mu, sd, _, _ = mc.predict(fit, mc.model_inputs(np.exp(lg), p_med, tff, float(t)))
        ax.fill_between(np.exp(lg), np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd), color=pal["accent"], alpha=0.22, lw=0, zorder=2)
        ax.plot(np.exp(lg), np.exp(mu), color=pal["accent"], lw=2.2, zorder=4)
        for hh in sorted(sub["H"].unique()):
            a = sub[np.isclose(sub["H"], hh)].groupby("cond").agg(lg=("ln_gd", "mean"), le=("ln_eta", "mean"))
            ax.scatter(np.exp(a["lg"]), np.exp(a["le"]), s=(fs * 0.55) ** 2, color=mc.H_COLORS[float(hh)], marker=mc.H_MARKS[float(hh)],
                       edgecolors="k", linewidths=0.4, alpha=0.85, zorder=5)
        ax.text(np.sqrt(xl[0] * gmin), yl[0] * 1.35, "学習データ\nのない範囲", ha="center", va="bottom", fontsize=fs - 0.5, color="0.25")
        ax.text(np.sqrt(gmin * gmax), yl[0] * 1.35, "学習データのある範囲", ha="center", va="bottom", fontsize=fs - 0.5, color="0.25")
        ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ (1/s)", labelpad=2)
        ax.set_ylabel(r"樹脂粘度 $\eta$ (Pa·s)", labelpad=2)
        save(fig, f"{des}_苦手な範囲.png")
    return gmin, gmax, int(t)


# ---------------------------------------------------------------- 説明の図
def fig_test_position(d, mat, des, w, h, fs, H=2.0, margins=(20, 7, 13, 3)):
    """学習データと検証データの、せん断速度 ― 粘度の上での位置（1 つの厚み H の、全温度の成形条件ごとの平均）。
    検証は、同じ厚みの中では両端（学習範囲の外）にある（厚みを混ぜると、薄肉の学習データと重なって見えるので、1 つの厚みで描く）"""
    pal = PAL[des]
    pool, train_c = d["pool"], set(d["tr"]["cond"])
    pool = pool[np.isclose(pool["H"], H)]
    a = pool.groupby("cond").agg(lg=("ln_gd", "mean"), le=("ln_eta", "mean")).reset_index()
    trm = a["cond"].isin(train_c).to_numpy()
    with plt.rc_context(fonts(fs)):
        fig, ax = new_fig(w, h, *margins)
        log_axes(ax, (20.0, 1e4), half_decade(np.exp(a["le"].min()), np.exp(a["le"].max()), pad=1.6))
        ax.scatter(np.exp(a["lg"][trm]), np.exp(a["le"][trm]), s=(fs * 0.45) ** 2, color=pal["main"], alpha=0.75, edgecolors="none", zorder=3,
                   label="学習データ")
        ax.scatter(np.exp(a["lg"][~trm]), np.exp(a["le"][~trm]), s=(fs * 0.62) ** 2, facecolors="white", edgecolors=pal["accent"], linewidths=1.8,
                   zorder=4, label="検証データ（学習に不使用）")
        lo = a[~trm & (a["le"] > a["le"].median())]
        hi = a[~trm & (a["le"] <= a["le"].median())]
        p_lo = (np.exp(lo["lg"].max()), np.exp(lo["le"][lo["lg"].idxmax()]))           # 低射出率側の検証のうち、いちばん右の点
        p_hi = (np.exp(hi["lg"].min()), np.exp(hi["le"][hi["lg"].idxmin()]))           # 高射出率側の検証のうち、いちばん左の点
        arrow = dict(arrowstyle="-", lw=1.2, color=pal["accent"], shrinkA=1, shrinkB=4)
        ax.annotate("検証（低射出率側）", xy=p_lo, xytext=(p_lo[0] * 2.2, p_lo[1] * 2.6), fontsize=fs, color=pal["accent"], fontweight="bold",
                    va="center", ha="left", arrowprops=arrow)
        ax.annotate("検証（高射出率側）", xy=p_hi, xytext=(p_hi[0] / 2.2, p_hi[1] / 3.0), fontsize=fs, color=pal["accent"], fontweight="bold",
                    va="center", ha="right", arrowprops=arrow)
        ax.set_xlabel(r"せん断速度 $\dot{\gamma}$ (1/s)", labelpad=1)
        ax.set_ylabel(r"粘度 $\eta$ (Pa·s)", labelpad=1)
        ax.legend(loc="upper right", frameon=False, fontsize=fs - 0.5, borderaxespad=0.3, handletextpad=0.3, markerscale=1.0)
        save(fig, f"{des}_検証の位置_{mat}.png")


def fig_header_art(d, mat, des, w, h, line="#FFFFFF", band=None):
    """題目の帯の飾り（背景は透明）：実際のモデルの予測曲線（中央の温度、厚み 1.0 mm）と、帯・点"""
    pal = PAL[des]
    band = band or pal["accent"]
    pool, fit = d["pool"], d["fit"]
    t = sorted(d["tr"]["Tset"].unique())[1]
    fig = plt.figure(figsize=mp.mm(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    xl = (25.0, 5e4)
    lg = np.linspace(np.log(xl[0]), np.log(xl[1]), 400)
    for k, hh in enumerate([2.0, 1.0, 0.5]):
        sub = pool[np.isclose(pool["H"], hh) & (pool["Tset"] == t)]
        mu, sd, _, _ = mc.predict(fit, mc.flow_inputs(sub, lg, hh, t))
        ax.fill_between(np.exp(lg), np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd), color=band, alpha=0.5 - 0.1 * k, lw=0)
        ax.plot(np.exp(lg), np.exp(mu), color=line, lw=2.4, alpha=0.95 - 0.2 * k)
        a = sub.groupby("cond").agg(lg=("ln_gd", "mean"), le=("ln_eta", "mean"))
        ax.scatter(np.exp(a["lg"]), np.exp(a["le"]), s=34, color=line, alpha=0.95 - 0.2 * k, edgecolors="none", zorder=5)
    ax.set_xlim(*xl)
    ax.set_ylim(0.4, 2.5e4)
    ax.axis("off")
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{des}_題目の飾り.png"), dpi=mp.DPI, transparent=True)
    plt.close(fig)


def fig_compare_concept(des, w, h, fs, gap=24.0, margins=(8, 3, 7, 11)):
    """従来の粘度式（1 本の曲線）と本研究（曲線 ＋ ばらつきの幅）の、イメージの図（人工の曲線）"""
    pal = PAL[des]
    g = np.logspace(0, 4.6, 300)
    lg = np.log10(g)
    eta = 2000 / (1 + (g / 150) ** 0.72)
    sd = 0.05 + 0.32 * ((lg - 2.6) / 2.2) ** 2                       # 中央（データの多い所）は狭く、両端は広い
    with plt.rc_context(fonts(fs)):
        fig = plt.figure(figsize=mp.mm(w, h))
        l, r, b, t = margins
        pw = (w - l - r - gap) / 2
        for k, (title, band, col) in enumerate([("従来の粘度式：1 本の曲線", False, "0.35"), ("本研究：曲線 ＋ ばらつきの幅", True, pal["main"])]):
            ax = fig.add_axes([(l + k * (pw + gap)) / w, b / h, pw / w, (h - b - t) / h])
            if band:
                ax.fill_between(g, eta * np.exp(-1.96 * sd), eta * np.exp(1.96 * sd), color=pal["accent"], alpha=0.25, lw=0)
            ax.plot(g, eta, color=col, lw=2.6)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(g[0], g[-1])
            ax.set_ylim(3, 2e4)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.minorticks_off()
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_title(title, fontsize=fs, color=col if band else "0.25", fontweight="bold", loc="left", pad=3)
            ax.set_xlabel("せん断速度", fontsize=fs - 2, labelpad=1)
            ax.set_ylabel("粘度", fontsize=fs - 2, labelpad=1)
        fig.text((l + pw + gap / 2) / w, (b + (h - b - t) / 2) / h, "▶", fontsize=fs * 1.6, ha="center", va="center", color=pal["accent"])
        save(fig, f"{des}_従来と本研究.png")


def fig_gpr_concept(des, w, h, fs, margins=(9, 3, 8, 13)):
    """人工のデータでの、ガウス過程回帰の予測平均値と 95% 予測区間（データの多い所は幅が狭く、ない所は広い）"""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    pal = PAL[des]
    rng = np.random.default_rng(3)
    f = lambda x: 2.2 - 0.32 * x + 0.35 * np.sin(1.1 * x)
    x = np.sort(np.concatenate([rng.uniform(1.6, 6.6, 26), [0.6]]))
    y = f(x) + rng.normal(0, 0.06, len(x))
    gp = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(1.5) + WhiteKernel(0.01), normalize_y=True, random_state=0).fit(x[:, None], y)
    xs = np.linspace(0, 10, 400)
    mu, sd = gp.predict(xs[:, None], return_std=True)
    with plt.rc_context(fonts(fs)):
        fig, ax = new_fig(w, h, *margins)
        ax.fill_between(xs, mu - 1.96 * sd, mu + 1.96 * sd, color=pal["accent"], alpha=0.25, lw=0, label="95% 予測区間（ばらつきの幅）")
        ax.plot(xs, mu, color=pal["accent"], lw=2.6, label="予測平均値")
        ax.scatter(x, y, s=(fs * 0.6) ** 2, color=pal["main"], edgecolors="white", linewidths=0.6, zorder=5, label="学習データ")
        i4, i9 = np.argmin(np.abs(xs - 4.0)), np.argmin(np.abs(xs - 9.0))
        ax.annotate("データが多い\n→ 幅が狭い", xy=(4.0, mu[i4] - 1.2 * sd[i4]), xytext=(3.2, -1.0), fontsize=fs, ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="0.2"))
        ax.annotate("データがない\n→ 幅が広い", xy=(8.9, mu[i9] - 1.6 * sd[i9]), xytext=(6.9, -1.0), fontsize=fs, ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="0.2"))
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.6, 3.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("入力（例：せん断速度）", labelpad=3)
        ax.set_ylabel("出力（例：粘度）", labelpad=3)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=fs - 0.5, borderaxespad=0.2, columnspacing=1.0,
                  handletextpad=0.4, handlelength=1.5)
        save(fig, f"{des}_GPR概念図.png")


def fig_shift_concept(des, w, h, fs, margins=(9, 3, 8, 3)):
    """低温と高温の 2 本の粘度曲線と、横（せん断速度）・縦（粘度）方向のずれ"""
    pal = PAL[des]
    g = np.logspace(0, 5, 300)
    kh, kv = 5.0, 3.5
    cross = lambda g: 3000.0 / (1 + (3000.0 * g / 3e4) ** 0.72)
    with plt.rc_context(fonts(fs)):
        fig, ax = new_fig(w, h, *margins)
        ax.plot(g, cross(g), color=T_COLS[0], lw=3, label="低温")
        ax.plot(g, cross(g * kh) / kv, color=T_COLS[2], lw=3, ls="--", label="高温")
        gx = 12.0
        p0 = (gx, cross(gx * kh) / kv)
        p1 = (gx * kh, p0[1])
        p2 = (gx * kh, cross(gx * kh))
        for a, b, col in [(p0, p1, pal["main"]), (p1, p2, pal["accent"])]:
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=fs * 1.7, lw=2.8, color=col, zorder=6, shrinkA=0, shrinkB=2))
        ax.text(0.97, 0.95, "→ 横のずれ（せん断速度の方向）", transform=ax.transAxes, ha="right", va="top", fontsize=fs, color=pal["main"],
                fontweight="bold")
        step = fs * 25.4 / 72 * 1.5 / (h - margins[2] - margins[3])            # 1 行の高さ（軸の高さに対する割合）
        ax.text(0.97, 0.95 - step, "↑ 縦のずれ（粘度の方向）", transform=ax.transAxes, ha="right", va="top", fontsize=fs,
                color=pal["accent"], fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 1e5)
        ax.set_ylim(2, 8000)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.minorticks_off()
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel(r"せん断速度（対数）", labelpad=3)
        ax.set_ylabel(r"粘度（対数）", labelpad=3)
        ax.legend(loc="lower left", fontsize=fs, frameon=False, borderaxespad=0.3)
        save(fig, f"{des}_温度シフト概念図.png")


def fig_error_bars(rows, des, w, h, fs, margins=(20, 3, 12, 14), horizontal=False):
    """本モデルと Cross-WLF モデル（参考）の、平均的な予測誤差（粘度の %）"""
    pal = PAL[des]
    g = [mp.pct(r["RMSE 全体"]) for r in rows]
    c = [mp.pct(r["Cross-WLF RMSE 全体"]) for r in rows]
    names = [MAT_JA[r["mat"]] for r in rows]
    with plt.rc_context(fonts(fs)):
        fig, ax = new_fig(w, h, *margins)
        x = np.arange(len(rows))
        ax.bar(x - 0.2, g, 0.38, color=pal["main"], label="本モデル", zorder=3)
        ax.bar(x + 0.2, c, 0.38, color=GRAY, label="Cross-WLF モデル（参考）", zorder=3)
        for i in range(len(rows)):
            ax.text(x[i] - 0.2, g[i] + 2, f"{mp.r0(g[i])}", ha="center", va="bottom", fontsize=fs, color=pal["main"], fontweight="bold")
            ax.text(x[i] + 0.2, c[i] + 2, f"{mp.r0(c[i])}", ha="center", va="bottom", fontsize=fs, color="0.3")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.tick_params(axis="x", which="both", top=False, bottom=False)
        ax.tick_params(axis="y", right=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_ylabel("平均的な予測誤差 (%)", labelpad=2)
        ax.set_ylim(0, max(c) * 1.2)
        ax.grid(True, axis="y", ls="-", lw=0.5, color="0.88", zorder=0)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=fs - 0.5, borderaxespad=0.2, handlelength=1.2,
                  columnspacing=1.0)
        save(fig, f"{des}_誤差の比較.png")


# ---------------------------------------------------------------- 案ごとの図（大きさは、build_poster_designs.py の配置に合わせる）
def numbers(d):
    rows = []
    for mat in MATS:
        met = pd.read_csv(os.path.join(mp.compare_dir(mat), "予測検証_3data_指標.csv"), encoding="utf-8-sig")
        get = lambda m, rng: float(met[(met["モデル"] == m) & (met["範囲"] == rng)]["RMSE"].iloc[0])
        summ = pd.read_csv(os.path.join(mp.compare_dir(mat), "まとめ.csv"), encoding="utf-8-sig")
        p = d[mat]["pred"]
        r0_, _ = shift_resid(d[mat], False)
        r1_, _ = shift_resid(d[mat], True)
        rows.append({"mat": mat, "材料": MAT_JA[mat], "RMSE 全体": get(mp.KEY, "全体"), "RMSE 低速側": get(mp.KEY, "低速側"), "RMSE 高速側": get(mp.KEY, "高速側"),
                     "Cross-WLF RMSE 全体": get("Cross-WLF", "全体"),
                     "被覆率": float(np.mean((p["eta"] >= p["下限95"]) & (p["eta"] <= p["上限95"]))), "検証の行数": len(p),
                     "LML の差（Tff）": float(summ[summ["モデル"] == mp.KEY]["θ=0 との LML の差"].iloc[0]),
                     "LML の差（Tset）": float(summ[summ["モデル"] == "B_Tset"]["θ=0 との LML の差"].iloc[0]),
                     "ずれ 換算前 [%]": mp.pct(float(np.sqrt(np.mean(r0_["r"] ** 2)))), "ずれ 換算後 [%]": mp.pct(float(np.sqrt(np.mean(r1_["r"] ** 2)))),
                     "学習の条件数": d[mat]["tr"]["cond"].nunique(), "学習の行数": len(d[mat]["tr"]), "検証の条件数": d[mat]["te"]["cond"].nunique()})
    return rows


def main():
    d = load()
    rows = numbers(d)
    # 案A：インフォグラフィック
    for mat in MATS:
        fig_flow(d[mat], mat, "A", 252, 236, 15)
        fig_parity(d[mat], mat, "A", 168, 166, 13.5)
    legend_flow("A", 600, 13, 16)
    fig_gpr_concept("A", 228, 128, 14)
    fig_error_bars(rows, "A", 267, 166, 14)
    fig_test_position(d["PP"], "PP", "A", 228, 112, 13.5)
    fig_header_art(d["PP"], "PP", "A", 270, 118)
    # 案B：Q&A
    for mat in MATS:
        fig_flow(d[mat], mat, "B", 253, 172, 14)
        fig_parity(d[mat], mat, "B", 130, 127, 12, margins=(19, 3, 14, 3), legend=(mat == "PP"), note=False)
    legend_flow("B", 600, 12, 15)
    fig_gpr_concept("B", 222, 120, 13.5)
    fig_shift_concept("B", 122, 84, 12)
    fig_resid(d["PP"], "PP", "B", 375, 86, 12.5)
    fig_limit(d["MIX"], "MIX", "B", 165, 104, 12)
    fig_compare_concept("B", 375, 58, 15)
    # 案C：材料別マトリクス
    for mat in MATS:
        fig_flow(d[mat], mat, "C", 250, 214, 14)
        fig_resid(d[mat], mat, "C", 250, 104, 12.5)
        fig_parity(d[mat], mat, "C", 176, 168, 13, legend=(mat == "PP"), note=False)
    legend_flow("C", 600, 12, 15)
    fig_gpr_concept("C", 244, 118, 13.5)
    fig_error_bars(rows, "C", 244, 148, 13.5)
    tab = pd.DataFrame(rows)
    for c in ["RMSE 全体", "RMSE 低速側", "RMSE 高速側", "Cross-WLF RMSE 全体"]:
        tab[c.replace("RMSE", "誤差 [%]")] = [mp.pct(v) for v in tab[c]]
    tab["被覆率 [%]"] = tab["被覆率"] * 100
    tab.drop(columns="mat").to_csv(os.path.join(OUT, "ポスターの数値_オリジナル.csv"), index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(tab.drop(columns=["mat"]).round(2).T)
    print("保存先:", OUT)


if __name__ == "__main__":
    main()
