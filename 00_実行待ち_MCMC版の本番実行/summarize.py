# -*- coding: utf-8 -*-
"""MCMC 版の本番の実行の結果を集め、勾配法版（同じデータ・同じモデル・同じ厚み）と並べる台本（2026-11-18 作成。計画は同じフォルダの 実行計画.md）。

実行（リポジトリ直下から）:
  uv run python 00_実行待ち_MCMC版の本番実行/summarize.py
  uv run python 00_実行待ち_MCMC版の本番実行/summarize.py --root GPPS=<MCMC の学習結果フォルダ>   # 結果の場所を変えて試すとき
読むもの（モデルのフォルダごと）：まとめ.csv・判定の一覧.csv、MCMC 版は MCMC_収束の診断.csv も。
出力：_tmp/mcmc_prod_run/比較_<日時>.csv（1 行 = 材料 × 厚み × 版）、…_要確認の判定.csv（MCMC 版の ★）。新しい名前で書く（上書きしない）。画面に比べの表。
"""
import argparse
import glob
import os
import sys
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
from run_all import MATS, sub_of  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 70)


def rd(p):
    return pd.read_csv(p, encoding="utf-8-sig") if p and os.path.isfile(p) else None


def model_dir(root, thick, model):
    """モデルのフォルダ（名前が「モデルの記号_」で始まり、まとめ.csv があるもの。全厚み総合/ には 温度ホールドアウト検証/ などもあるため、記号で選ぶ）"""
    fs = glob.glob(os.path.join(root, "全ショット版", *sub_of(thick).split("/"), f"{model}_*", "まとめ.csv"))
    return os.path.dirname(fs[0]) if len(fs) == 1 else (None if not fs else "複数")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="*", default=[], help="MCMC の学習結果フォルダを差し替える（例：GPPS=_tmp/…_MCMC）")
    ap.add_argument("--out", default="_tmp/mcmc_prod_run", help="出力のフォルダ（既定 _tmp/mcmc_prod_run）")
    a = ap.parse_args()
    roots = {m: v["res"] for m, v in MATS.items()}
    for s in a.root:
        k, v = s.split("=", 1)
        roots[k] = v

    rows, flags = [], []
    for mat, m in MATS.items():
        for h in [None] + m["thick"]:
            for ver, root in [("勾配法", m["grad"]), ("MCMC", roots[mat])]:
                d = model_dir(root, h, m["model"])
                if d is None:
                    rows.append({"材料": mat, "厚み": "全厚み" if h is None else f"{h:g}", "版": ver, "メモ": "結果なし"})
                    continue
                if d == "複数":
                    rows.append({"材料": mat, "厚み": "全厚み" if h is None else f"{h:g}", "版": ver, "メモ": "モデルのフォルダが複数ある（見て確かめる）"})
                    continue
                s = rd(os.path.join(d, "まとめ.csv"))
                g = s[s["モデル"] != "Cross-WLF"].iloc[0]
                jd = rd(os.path.join(d, "判定の一覧.csv"))
                star = jd[jd["判定"] != "OK"] if jd is not None else None
                row = {"材料": mat, "厚み": "全厚み" if h is None else f"{h:g}", "版": ver, "モデル": g["モデル"], "θ": g.get("θ"),
                       "LML": g.get("LML", g.get("代表点の LML")),
                       "支持": g.get("尤度で支持", g.get("事後分布で支持")),
                       "支持の根拠": (f"ΔLML {g['θ=0 との LML の差']}（基準 {g['支持の基準']}）" if "θ=0 との LML の差" in g.index
                                  else (f"P(θ_v>0) {g['P(θ_v>0)']}" if "P(θ_v>0)" in g.index else "")),
                       "RMSE 全体": g.get("RMSE 全体"), "RMSE 低速側": g.get("RMSE 低速側"), "RMSE 高速側": g.get("RMSE 高速側"),
                       "被覆率 全体": g.get("被覆率 全体"), "R²(実数) 全体": g.get("R²(実数) 全体"),
                       "★の数": len(star) if star is not None else None, "フォルダ": d}
                if ver == "MCMC":
                    dg = rd(os.path.join(d, "MCMC_収束の診断.csv"))
                    if dg is not None:
                        row.update({"MCMC の収束": g.get("MCMC の収束"), "R-hat 最大": round(float(dg["r_hat"].max()), 4), "ESS 最小": round(float(dg["ess_bulk"].min())),
                                    "本番の反復数": int(dg["本番の反復数"].iloc[0]), "採択率": f"{dg['採択率（本番）の最小'].iloc[0]:.2f}〜{dg['採択率（本番）の最大'].iloc[0]:.2f}",
                                    "MCMC の時間 [分]": round(float(dg["時間 [分]"].iloc[0]), 1)})
                    if star is not None:
                        for _, r in star.iterrows():
                            flags.append({"材料": mat, "厚み": row["厚み"], "節": r.get("節"), "内容": str(r.get("内容"))})
                rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out, out2 = os.path.join(a.out, f"比較_{stamp}.csv"), os.path.join(a.out, f"比較_{stamp}_要確認の判定.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    pd.DataFrame(flags).to_csv(out2, index=False, encoding="utf-8-sig")
    print(f"保存：{out}（{len(df)} 行）、{out2}（{len(flags)} 行）\n")

    cols = [c for c in ["材料", "厚み", "版", "モデル", "θ", "支持", "支持の根拠", "RMSE 全体", "RMSE 低速側", "RMSE 高速側", "被覆率 全体", "★の数", "メモ"] if c in df.columns]
    print("■ 勾配法版と MCMC 版（同じデータ・同じモデル）")
    print(df[cols].round(4).to_string(index=False))
    mc = df[(df["版"] == "MCMC") & df.get("R-hat 最大", pd.Series(dtype=float)).notna()] if "R-hat 最大" in df.columns else df.iloc[0:0]
    print("\n■ MCMC の収束（基準：R-hat 1.01 以下、ESS 400 以上、採択率 0.15〜0.40）")
    print(mc[["材料", "厚み", "MCMC の収束", "R-hat 最大", "ESS 最小", "本番の反復数", "採択率", "MCMC の時間 [分]"]].to_string(index=False) if len(mc) else "（MCMC 版の結果なし）")
    num = [c for c in ["RMSE 全体", "RMSE 低速側", "RMSE 高速側", "被覆率 全体"] if c in df.columns]
    if num and len(mc):
        pv = df[df["メモ"].isna()].pivot_table(index=["材料", "厚み"], columns="版", values=num, aggfunc="first") if "メモ" in df.columns else df.pivot_table(index=["材料", "厚み"], columns="版", values=num, aggfunc="first")
        print("\n■ 指標を版ごとに並べた表")
        print(pv.round(4).to_string())
    print("\n■ MCMC 版の ★要確認の判定")
    print(pd.DataFrame(flags).to_string(index=False) if flags else "なし")


if __name__ == "__main__":
    main()
