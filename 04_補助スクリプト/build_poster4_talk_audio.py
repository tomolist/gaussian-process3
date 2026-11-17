# -*- coding: utf-8 -*-
"""ポスター4 の台本（build_poster4_talk.py の TALK。省略可の文も含む全部の版）を、男性の声で読み上げた WAV を作る
（2026-11-16 のユーザーの依頼：男性の声で、ふつうの速さと少しゆっくりの 2 つ。2026-11-17 のユーザーの決定で、ふつうの速さだけにした）。

- 音声は Windows 標準の音声合成（System.Speech。声 Microsoft Ichiro）で、PC の中だけで作る（文章を外部に送らない）。
  合成は 04_補助スクリプト/poster4_tts.ps1（pwsh で動かす）。
- 読み上げ文は、台本の文から作る（台本の PDF は変えない）：
  - 【 】の合図は読まず、図を指す間として短い無音を入れる。節の間・段落の間にも無音を入れる（読み始めの無音は入れない）。
  - 読み間違いを防ぐため、読み上げ文だけ言い換える（金型→かながた、①②→1番・2番、%→パーセント、〜→から）。
    2026-11-17 の 3 回目（台本を 4 分 30 秒にし、数値と単位を言うようにした）から、単位も言い換える
    （cm³/s→立方センチメートル毎秒、Pa·s→パスカル秒、MPa→メガパスカル、1/s→毎秒、mm→ミリ、℃→度）。単位は、空白を消す前に言い換える。
- ファイルの名前は、台本と同じ（build_poster4_talk.STEM_TALK。11-17 から ポスター4_台本_4分30秒）。
- 速さ：ふつう＝台本の時間の目安（1 分 300 字）。声の標準の速さで 1 回作って話す速さを測り、
  それに合う合成の速さを選ぶ（SPEEDS。速さを足せば、ほかの速さの版も作れる）。実際の音声の長さと話す速さを WAV から測って表示する。
- 同じ名前のファイルがあれば止める（上書きしない）。合成は _tmp/poster4_audio/<日時>/ で行い、できたファイルをコピーする。

使い方（リポジトリ直下で）: uv run python 04_補助スクリプト/build_poster4_talk_audio.py [--out <フォルダ>]
  （既定：05_…/07_学習結果報告書作成_261015/ポスター/完成版の添削と複数デザイン/ポスター4_発表報告/〔build_poster4_talk.OUT_DEFAULT〕。
   --out はリポジトリ直下からの相対パス）
"""
import argparse
import datetime
import math
import os
import re
import shutil
import subprocess
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_poster4_talk as bt  # noqa: E402
import report_common as rc  # noqa: E402

VOICE = "Microsoft Ichiro"                 # 男性の声（この PC の ja-JP の声）
SPEEDS = [("ふつう", 300, "normal")]   # (名前, 目標の話す速さ〔1 分あたりの字数〕, 作業用の英字の名前)
# ふつう＝台本の時間の目安（1 分 300 字）。2026-11-17 のユーザーの決定で、少しゆっくり（1 分 270 字）の版はやめた。
# 声の標準の速さ（Rate 0）で 1 回作って実際の速さを測り、合成の速さ Rate（-10〜10。Rate を r にすると、話す速さはおよそ 3^(r/10) 倍）を、
# 目標に近い整数に選ぶ
PAUSE_SECTION, PAUSE_PARA, PAUSE_CUE = bt.PAUSE_SECTION, bt.PAUSE_PARA, bt.PAUSE_CUE   # 無音の秒数：節の間・段落の間・図を指す合図（台本の時間の目安と同じ値。11-17 から build_poster4_talk の値を使う）
READ_FIX = [("計測金型", "けいそくかながた"), ("金型", "かながた"), ("①", "1番"), ("②", "2番"), ("\\%", "パーセント"), ("〜", "から")]
UNIT_FIX = [(" cm³/s", "立方センチメートル毎秒"), (" Pa$\\cdot$s", "パスカル秒"), (" MPa", "メガパスカル"), (" 1/s", "毎秒"),
            (" mm", "ミリ"), (" ℃", "度"), ("℃", "度")]   # 単位の読み方（空白を消す前に言い換える。2026-11-17 の 3 回目）
STEM = bt.STEM_TALK
PS1 = "04_補助スクリプト/poster4_tts.ps1"


def reading_lines(N):
    """台本 → 読み上げ文の行（「[間 秒数]」の行は無音）"""
    out = []
    for i, (title, pars) in enumerate(bt.TALK):
        if i:
            out.append(f"[間 {PAUSE_SECTION}]")
        for j, p in enumerate(pars):
            if j:
                out.append(f"[間 {PAUSE_PARA}]")
            s = bt.sub(p, N).replace("[[", "").replace("]]", "")          # 省略可の文も読む（全部の版）
            s = re.sub(r"\\memo\{[^}]*\}", "", s)                           # 言わない（ ）の数値は読まない（2026-11-17 の 4 回目）
            for a, b in UNIT_FIX:                                          # 単位（空白を消す前に。数字の末尾と「1/s」が続いて読めなくなるのを防ぐ）
                s = s.replace(a, b)
            if re.search(r"cm³|Pa|1/s|mm|℃", s):
                sys.exit(f"読み上げ文に、言い換えていない単位が残っている：{s}")
            s = s.replace(" ", "")                                         # 数字と文字の間の空白を消す（無音の印を入れる前に）
            s = re.sub(r"\\cue\{[^}]*\}", f"\n[間 {PAUSE_CUE}]\n", s)
            for a, b in READ_FIX:
                s = s.replace(a, b)
            if re.search(r"[\\${}]", s) or "所属" in s:
                sys.exit(f"読み上げ文に LaTeX の命令か、名乗りの置き場所が残っている：{s}")
            out.extend(x.strip() for x in s.split("\n") if x.strip())
    while out and out[0].startswith("[間"):                                # 読み始めの無音は入れない
        out.pop(0)
    return out


def seconds(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def mmss(sec):
    sec = int(sec + 0.5)
    return f"{sec // 60}:{sec % 60:02d}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=bt.OUT_DEFAULT, help="WAV と読み上げ文を置くフォルダ（リポジトリ直下から）")
    a = ap.parse_args()
    out = os.path.join(rc.W, a.out)
    targets = {name: os.path.join(out, f"{STEM}_男性_{name}.wav") for name, _, _ in SPEEDS}
    txt_dst = os.path.join(out, f"{STEM}_読み上げ文.txt")
    for p in list(targets.values()) + [txt_dst]:
        if os.path.exists(p):
            sys.exit(f"すでにある：{p}（上書きしない。前のファイルを _to_delete/ へ移すか、--out で別のフォルダを指定）")
    N, _ = bt.load()
    lines = reading_lines(N)
    work = os.path.join("_tmp", "poster4_audio", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(rc.W, work), exist_ok=False)
    txt = os.path.join(work, "read.txt").replace("\\", "/")
    with open(os.path.join(rc.W, txt), "x", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    pauses = [float(m) for m in re.findall(r"^\[間 ([0-9.]+)\]$", "\n".join(lines), flags=re.M)]
    n_marks = sum(1 for x in lines if x.startswith("[間"))
    if len(pauses) != n_marks:
        sys.exit("無音の印の書き方がそろっていない")
    pause_sec = sum(pauses)
    n_chars = sum(bt.spoken_len(bt.sub(p, N).replace("[[", "").replace("]]", "")) for _, pars in bt.TALK for p in pars)   # 台本の字数（PDF の表と同じ数え方）

    def synth(tag, rate):
        wav = os.path.join(work, f"{tag}.wav").replace("\\", "/")
        r = subprocess.run(["pwsh", "-NoProfile", "-File", PS1, "-TextFile", txt, "-OutFile", wav, "-Voice", VOICE, "-Rate", str(rate)],
                           cwd=rc.W, capture_output=True)
        if r.returncode != 0 or not os.path.exists(os.path.join(rc.W, wav)):
            print(r.stdout.decode("utf-8", "replace")[-2000:], r.stderr.decode("utf-8", "replace")[-2000:])
            sys.exit(f"音声の合成に失敗しました（{tag}）")
        return os.path.join(rc.W, wav)

    cpm = lambda path: n_chars / (seconds(path) - pause_sec) * 60          # 話す速さ（無音を除く。1 分あたりの字数）
    cpm0 = cpm(synth("calib_rate0", 0))
    print(f"声の標準の速さ（Rate 0）：1 分 {cpm0:.0f} 字（台本 {n_chars} 字、無音 {pause_sec:.1f} 秒）")
    made, used = [], set()
    for name, target, tag in SPEEDS:
        rate = max(-10, min(10, round(10 * math.log(target / cpm0, 3))))
        while rate in used:                    # 2 つが同じ速さにならないように、ゆっくりのほうを 1 つ下げる
            rate -= 1
        used.add(rate)
        made.append((name, target, rate, synth(tag, rate)))
    os.makedirs(out, exist_ok=True)
    shutil.copy2(os.path.join(rc.W, txt), txt_dst)
    for name, target, rate, src in made:
        shutil.copy2(src, targets[name])
        print(f"{os.path.basename(targets[name])}：{mmss(seconds(src))}（{seconds(src):.1f} 秒。話す速さ 1 分 {cpm(src):.0f} 字〔目標 {target}〕、"
              f"声 {VOICE}、Rate {rate}、{os.path.getsize(src) / 1e6:.1f} MB）")
    print(f"読み上げ文：{len(lines)} 行（うち無音 {len(pauses)} か所、合計 {pause_sec:.1f} 秒）→ {os.path.basename(txt_dst)}")
    print("作業フォルダ:", work)
    print("保存:", a.out)


if __name__ == "__main__":
    main()
