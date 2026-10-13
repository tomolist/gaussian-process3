# -*- coding: utf-8 -*-
"""データを取り直したとき・入れ替えたときに、全手順のノートと説明書を作り直す。

使い方（リポジトリ直下から）:
  uv run python 04_補助スクリプト/rerun_all.py --steps 3a 3b 3c 5a
  オプション：--skip-check（データ確認を省く）、--no-report（説明書を作らない）、--only-build（ノートを作るだけで、実行しない）

流れ：データ確認（check_data.py）→ 各手順で、
  ① ノートを2つ作る（01_機械学習コードフォルダ/ に未実行のもの、_tmp/ に実行用のコピー）
  ② _tmp/ のコピーを run_nb.py で実行する（図・表は results/ に出る）
  ③ 実行済みのノートにエラーがないか確かめる
  ④ 説明書（results/学習結果説明_手順*.pdf と .tex）を作り直す
エラーが出たら、そこで止まる。結果の判断は、説明書と、ノートの「判定」（>> 判定：OK / ★要確認）で行う。
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = "01_機械学習コードフォルダ"
STEPS = {   # 手順 → (ノート生成スクリプト, ノート名, 説明書生成スクリプト)
    "3a": ("build_step3a_nb.py", "手順3a_新データ_圧力列の検討.ipynb", "build_report_step3a.py"),
    "3b": ("build_step3b_nb.py", "手順3b_新データ_設定温度の比較.ipynb", "build_report_step3b.py"),
    "3c": ("build_step3c_mcmc_nb.py", "手順3c_最終候補のMCMC.ipynb", "build_report_step3c.py"),
    "5a": ("build_step5a_nb.py", "手順5a_シフト構造の見直し.ipynb", "build_report_step5a.py"),
}
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "RUN_NB_TIMEOUT": os.environ.get("RUN_NB_TIMEOUT", "14400")}   # 1セルの上限 4 時間（3c の MCMC 用）


def run(args, what):
    t0 = time.time()
    r = subprocess.run([sys.executable, *args], env=ENV)
    print(f"  → {what}：{'成功' if r.returncode == 0 else '失敗'}（{(time.time() - t0) / 60:.1f} 分）", flush=True)
    if r.returncode != 0:
        sys.exit(f"{what} が失敗したので止めます。")


def count_errors(nb_path):
    nb = json.load(open(nb_path, encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    errs = [o for c in cells for o in c.get("outputs", []) if o["output_type"] == "error"]
    not_run = sum(1 for c in cells if c.get("execution_count") is None)
    return len(errs), not_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", required=True, choices=list(STEPS))
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--only-build", action="store_true")
    ap.add_argument("--reuse-mcmc", action="store_true", help="3c で、保存済みの事後サンプルを再利用する（既定では消して、サンプリングし直す）")
    a = ap.parse_args()
    os.makedirs("_tmp", exist_ok=True)
    if not a.skip_check:
        print("データ確認", flush=True)
        run([os.path.join(HERE, "check_data.py")], "データ確認")
    for step in a.steps:
        build, nb_name, report = STEPS[step]
        print(f"\n=== 手順{step} ===", flush=True)
        for out in [os.path.join(NB_DIR, nb_name), os.path.join("_tmp", nb_name)]:
            run([os.path.join(HERE, build), out, "--force"], f"ノートの作成（{out}）")
        if a.only_build:
            continue
        if step == "3c" and not a.reuse_mcmc:       # データが変わっていると、古い事後サンプルは使えない
            d = os.path.join("results", "step3c_mcmc")
            old = [f for f in os.listdir(d) if f.startswith("posterior_") and f.endswith(".csv")] if os.path.isdir(d) else []
            for f in old:
                os.remove(os.path.join(d, f))
            print(f"  保存済みの事後サンプル {len(old)} 個を消した（再利用するなら --reuse-mcmc）", flush=True)
        run([os.path.join(HERE, "run_nb.py"), os.path.join("_tmp", nb_name)], "ノートの実行")
        n_err, n_not = count_errors(os.path.join("_tmp", nb_name))
        if n_err or n_not:
            sys.exit(f"実行済みのノートに、エラー {n_err} 件、未実行のセル {n_not} 個があります。")
        print("  実行済みのノート：エラーなし、全セル実行済み", flush=True)
        if not a.no_report and os.path.exists(os.path.join(HERE, report)):
            run([os.path.join(HERE, report)], "説明書の作成")
    print("\n完了。結果は results/、説明書は results/学習結果説明_手順*.pdf。ノートの「判定」と説明書で、結果を判断してください。")


if __name__ == "__main__":
    main()
