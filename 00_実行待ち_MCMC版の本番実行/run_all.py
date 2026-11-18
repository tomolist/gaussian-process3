# -*- coding: utf-8 -*-
"""学習用のノートの MCMC 版を、3 材料で本番の実行をする台本（2026-11-18 作成。計画は同じフォルダの 実行計画.md）。

実行（リポジトリ直下から。1 コマンドずつ）:
  uv run python 00_実行待ち_MCMC版の本番実行/run_all.py --dry                         # 一覧と事前の確認だけ（実行しない）
  uv run python 00_実行待ち_MCMC版の本番実行/run_all.py --only PP_all MIX_all GPPS_all  # 全厚みの 3 回だけ
  uv run python 00_実行待ち_MCMC版の本番実行/run_all.py                               # 残り全部（結果がある実行は飛ばす）
  uv run python 00_実行待ち_MCMC版の本番実行/run_all.py --only PP_t0.5 --set "MCMC_MAX_SAMPLES = 200000"   # 設定を足してやり直す例

やること（1 回の実行ごと）:
  1. 「上書きの行」（SHOW_FIGS・THICKNESS・FILES と、--set の行）を _tmp/mcmc_prod_run/over/ に書く
  2. _tmp/make_run_copy.py で、ノートのコピー（設定エリアの最後に上書きの行を足したもの）を _tmp/mcmc_prod_run/nb/ に作る（元のノートは変えない）
  3. 04_補助スクリプト/run_nb.py でコピーを最初から最後まで実行する（ログは _tmp/mcmc_prod_run/logs/）
  4. 結果（モデルのフォルダの まとめ.csv）が新しくできたかを確かめ、状況を _tmp/mcmc_prod_run/進行状況.csv に追記する
既存のファイルは上書きしない：コピー・ログ・上書きの行は、やり直すたびに _try<N> の新しい名前にする。
結果のモデルのフォルダに まとめ.csv がすでにある実行は飛ばす（やり直すときは、そのモデルのフォルダを _to_delete/ へ移してから。実行計画.md の 6 節）。
同じ学習結果フォルダの データ確認/ の表は、実行のたびに同じ中身で書き直される（ノートの仕様。勾配法版でも同じ）。
既定は 1 つずつ実行する（1 回の実行が MCMC の 4 チェーンを並列に回すため。--parallel で増やせる）。
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

REPO = Path.cwd()
assert (REPO / "04_補助スクリプト").is_dir(), "リポジトリ直下から実行してください"
WORK = Path("_tmp/mcmc_prod_run")
NB_DIR = "01_機械学習コードフォルダ/材料別_温度シフトファクター検討/02_材料別_温度シフト機械学習コード/02_MCMC(適応型RWM)"
D05 = "05_材料ごとの機械学習データと予測検証データ"
STATE = WORK / "進行状況.csv"
STATE_COLS = ["実行", "試行", "開始", "終了", "状態", "秒", "ログ", "メモ"]

# データは、比べる勾配法版の最新の学習結果（PP 4回目・粉砕材混合 PP 5回目・GPPS 3回目）の データ確認/データの概要.csv に記録されたファイルと同じ。
# 結果のフォルダは、2026-11-18 にユーザーの依頼で作った（空）。勾配法版の結果フォルダ（grad）は、比べるときだけ読む。
MATS = {
    "PP": dict(
        nb_glob="PP_*.ipynb", model="A_Tset_Arr",
        train=f"{D05}/01_バージンPP再々実験(データと結果)/01_学習・検証データ/学習データ_pp_max_5data/1117修正_結合_03_pp_max_5data_外れ値なし.csv",
        test=f"{D05}/01_バージンPP再々実験(データと結果)/01_学習・検証データ/検証データ_pp_max_5data用_3data/1117修正_結合_05_pp_max_3data(no1.5-3.0)_外れ値なし.csv",
        res=f"{D05}/01_バージンPP再々実験(データと結果)/01_学習結果/4回目_01_バージンPP再々実験_学習結果_MCMC",
        grad=f"{D05}/01_バージンPP再々実験(データと結果)/01_学習結果/4回目_01_バージンPP再々実験_学習結果",
        thick=[0.5, 1.0, 1.5, 2.0],
    ),
    "MIX": dict(
        nb_glob="MIXPP_*.ipynb", model="A_Tset_Arr",
        train=f"{D05}/02_粉砕材混合PP実験(データと結果)/02_学習・検証データ/学習データ_mixpp_max_5data/結合_03_mixpp_max_5data_外れ値なし.csv",
        test=f"{D05}/02_粉砕材混合PP実験(データと結果)/02_学習・検証データ/検証データ_mixpp_max_5data用_3data/結合_02_mixpp_max_3data_外れ値なし.csv",
        res=f"{D05}/02_粉砕材混合PP実験(データと結果)/02_学習結果/5回目_02_粉砕材混合PP実験_学習結果_MCMC",
        grad=f"{D05}/02_粉砕材混合PP実験(データと結果)/02_学習結果/5回目_02_粉砕材混合PP実験_学習結果",
        thick=[0.5, 1.0, 1.5, 2.0],
    ),
    "GPPS": dict(
        nb_glob="GPPS_*.ipynb", model="B_Tset_Arr",
        train=f"{D05}/03_GPPS実験(データと結果)/03_学習・検証データ/学習データ_gpps_max_5data/結合_gpps_5data_20261014_060353.csv",
        test=f"{D05}/03_GPPS実験(データと結果)/03_学習・検証データ/検証データ_gpps_max_5data用_3data/結合_gpps_3data_20261014_052315.csv",
        res=f"{D05}/03_GPPS実験(データと結果)/03_学習結果/3回目_03_GPPS実験_学習結果_MCMC",
        grad=f"{D05}/03_GPPS実験(データと結果)/03_学習結果/3回目_03_GPPS実験_学習結果",
        thick=[1.0, 1.5, 2.0],          # GPPS は 0.5 mm のデータがない
    ),
}

_lock = threading.Lock()


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def build_runs():
    """材料ごとに、全厚み → 厚みごと（薄い順）。ID は <材料>_all・<材料>_t<厚み>"""
    runs = []
    for mat, m in MATS.items():
        runs.append(dict(id=f"{mat}_all", mat=mat, thick=None))
        for h in m["thick"]:
            runs.append(dict(id=f"{mat}_t{h:g}", mat=mat, thick=h))
    return runs


def sub_of(thick):
    return "全厚み総合" if thick is None else f"厚みごと/厚み{thick:g}mm"


def notebook_of(mat):
    fs = sorted(glob.glob(f"{NB_DIR}/{MATS[mat]['nb_glob']}"))
    assert len(fs) == 1, f"{mat} のノートが 1 つだけあるはず：{fs}"
    return fs[0]


def over_text(run, extra):
    m = MATS[run["mat"]]
    files = {run["mat"]: {"学習": m["train"], "検証": m["test"], "結果": m["res"]}}
    lines = ["SHOW_FIGS = False", f"THICKNESS = {run['thick']!r}", f"FILES = {files!r}"] + list(extra)
    return "\n".join(lines) + "\n"


def result_files(run):
    m = MATS[run["mat"]]
    return glob.glob(f"{m['res']}/全ショット版/{sub_of(run['thick'])}/*/まとめ.csv")


def append_state(row):
    with _lock:
        new = not STATE.exists()
        with open(STATE, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=STATE_COLS)
            if new:
                w.writeheader()
            w.writerow(row)


def run_one(run, sem, extra):
    with sem:
        rid, mat = run["id"], run["mat"]
        n = 1
        while (WORK / "nb" / f"{rid}_try{n}.ipynb").exists() or (WORK / "logs" / f"{rid}_try{n}.txt").exists() or (WORK / "over" / f"{rid}_try{n}.txt").exists():
            n += 1
        for d in ["over", "nb", "logs"]:
            (WORK / d).mkdir(parents=True, exist_ok=True)
        over, nbc, logp = WORK / "over" / f"{rid}_try{n}.txt", WORK / "nb" / f"{rid}_try{n}.ipynb", WORK / "logs" / f"{rid}_try{n}.txt"
        over.write_text(over_text(run, extra), encoding="utf-8")
        t0 = time.time()
        start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log(f"開始 {rid}（試行 {n}）")
        env = dict(os.environ, RUN_NB_TIMEOUT="14400", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        status, note = "異常", ""
        with open(logp, "wb") as lf:
            lf.write(f"# {rid} 試行 {n} 開始 {start}\n# 上書きの行:\n{over.read_text(encoding='utf-8')}\n".encode("utf-8"))
            lf.flush()
            r1 = subprocess.run([sys.executable, "_tmp/make_run_copy.py", notebook_of(mat), str(nbc), str(over)], cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
            if r1.returncode != 0:
                note = "ノートのコピーの作成に失敗"
            else:
                r2 = subprocess.run([sys.executable, "04_補助スクリプト/run_nb.py", str(nbc), f"gp_mcmc_{mat.lower()}"], cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
                made = [p for p in result_files(run) if os.path.getmtime(p) >= t0 - 1]
                if r2.returncode == 0 and made:
                    status = "正常"
                elif r2.returncode == 0:
                    note = "正常終了だが まとめ.csv が新しくできていない"
                else:
                    note = f"run_nb.py が異常終了（コード {r2.returncode}）。ログと、実行したコピーのノートのエラーのセルを見る"
        sec = int(time.time() - t0)
        append_state({"実行": rid, "試行": n, "開始": start, "終了": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "状態": status, "秒": sec, "ログ": str(logp), "メモ": note})
        log(f"終了 {rid}（試行 {n}）：{status}、{sec // 60} 分 {sec % 60} 秒 {note}")
        return status == "正常"


def run_queue(runs, sem, extra):
    fails = 0
    for run in runs:
        ok = run_one(run, sem, extra)
        fails = 0 if ok else fails + 1
        if fails >= 2:
            log(f"★ {run['mat']} の列：2 回続けて失敗したので、この材料の残りは実行しない")
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="実行する ID（例：PP_all、GPPS_t1.5）。省略すると全部")
    ap.add_argument("--parallel", type=int, default=1, help="同時に動かす実行の数の上限（既定 1。材料ごとに 1 本の列）")
    ap.add_argument("--set", nargs="*", default=[], help='設定エリアの最後に足す行（例："MCMC_MAX_SAMPLES = 200000"）')
    ap.add_argument("--dry", action="store_true", help="一覧と事前の確認だけ（実行しない）")
    a = ap.parse_args()

    runs = build_runs()
    if a.only:
        unknown = [x for x in a.only if x not in {r["id"] for r in runs}]
        assert not unknown, f"知らない ID：{unknown}（ID の一覧は --dry で見る）"
        runs = [r for r in runs if r["id"] in a.only]
    # 事前の確認：ノート・データ・結果のフォルダ
    for mat, m in MATS.items():
        nb = notebook_of(mat)
        text = Path(nb).read_text(encoding="utf-8")
        assert "# ▲▲▲ 設定エリア終了 ▲▲▲" in text, f"{nb} に設定エリアの目印がない"
        assert f'MATERIALS = [\\"{mat}\\"]' in text, f"{nb} の MATERIALS が {mat} だけになっていない"
        for k in ["train", "test"]:
            assert Path(m[k]).is_file(), f"{mat} のデータがない：{m[k]}"
        assert Path(m["res"]).is_dir(), f"{mat} の結果のフォルダがない：{m['res']}（実行計画.md の 3 節）"
        assert "MCMC" in Path(m["res"]).name, f"{mat} の結果のフォルダの名前に MCMC がない：{m['res']}"
        assert Path(m["grad"]).is_dir(), f"{mat} の比べる勾配法版の結果フォルダがない：{m['grad']}"
    has = {r["id"] for r in runs if result_files(r)}
    log(f"実行の一覧（{len(runs)} 回。同時 {a.parallel} 本まで。足す行：{a.set or 'なし'}）")
    for r in runs:
        mark = "（結果あり：飛ばす）" if r["id"] in has else ""
        print(f"  {r['id']:<9} {MATS[r['mat']]['model']}  THICKNESS={r['thick']!s:<5} → {MATS[r['mat']]['res']}/全ショット版/{sub_of(r['thick'])}/{mark}")
    if a.dry:
        return
    todo = [r for r in runs if r["id"] not in has]
    sem = threading.Semaphore(a.parallel)
    threads = []
    for mat in MATS:
        q = [r for r in todo if r["mat"] == mat]
        if q:
            t = threading.Thread(target=run_queue, args=(q, sem, a.set), name=mat)
            t.start()
            threads.append(t)
            time.sleep(5)           # 起動を少しずらす
    for t in threads:
        t.join()
    ok = {r["id"] for r in runs if result_files(r)}
    log(f"全部終わり：結果のある実行 {len(ok)} / {len(runs)} 回。次は summarize.py で確かめる")


if __name__ == "__main__":
    main()
