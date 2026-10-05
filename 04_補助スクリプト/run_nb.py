# -*- coding: utf-8 -*-
"""jupyter_client でノートブックを上から順に実行し、出力をノートに保存する (nbclient の代わり)。

使い方: python run_nb.py <ノートのパス> [カーネル名 (既定: gaussian-process3)]
カーネルの作業ディレクトリはノートのあるフォルダ。エラーが出たセルで止め、そこまでの出力を保存する。
"""
import json
import os
import sys
import time

from jupyter_client.manager import KernelManager

TIMEOUT = float(os.environ.get("RUN_NB_TIMEOUT", 3600))  # 1セルあたりの上限 [s]（MCMC のセル用に環境変数で延ばせる）


def to_lines(s):
    return s.splitlines(keepends=True)


def conv_data(data):
    """表示データを .ipynb の形式にする (テキストは行のリスト、画像は base64 文字列のまま)"""
    out = {}
    for k, v in data.items():
        if isinstance(v, str) and (k.startswith("text/") or k == "image/svg+xml"):
            out[k] = to_lines(v)
        else:
            out[k] = v
    return out


def main():
    nb_path = os.path.abspath(sys.argv[1])
    kernel_name = sys.argv[2] if len(sys.argv) > 2 else "gaussian-process3"
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    km = KernelManager(kernel_name=kernel_name)
    km.start_kernel(cwd=os.path.dirname(nb_path))
    kc = km.client()
    kc.start_channels()
    ok = True
    try:
        kc.wait_for_ready(timeout=180)
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            t0 = time.time()
            msg_id = kc.execute("".join(cell["source"]), store_history=True, allow_stdin=False, stop_on_error=True)
            outputs = []
            while True:
                msg = kc.get_iopub_msg(timeout=TIMEOUT)
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                mt, c = msg["msg_type"], msg["content"]
                if mt == "status":
                    if c["execution_state"] == "idle":
                        break
                elif mt == "execute_input":
                    cell["execution_count"] = c["execution_count"]
                elif mt == "stream":
                    if outputs and outputs[-1]["output_type"] == "stream" and outputs[-1]["name"] == c["name"]:
                        outputs[-1]["text"] += c["text"]
                    else:
                        outputs.append({"output_type": "stream", "name": c["name"], "text": c["text"]})
                elif mt == "display_data":
                    outputs.append({"output_type": "display_data", "data": conv_data(c["data"]),
                                    "metadata": c.get("metadata", {})})
                elif mt == "execute_result":
                    outputs.append({"output_type": "execute_result", "data": conv_data(c["data"]),
                                    "metadata": c.get("metadata", {}), "execution_count": c["execution_count"]})
                elif mt == "error":
                    outputs.append({"output_type": "error", "ename": c["ename"], "evalue": c["evalue"],
                                    "traceback": c["traceback"]})
                elif mt == "clear_output":
                    outputs = []
            while True:
                reply = kc.get_shell_msg(timeout=TIMEOUT)
                if reply["parent_header"].get("msg_id") == msg_id:
                    break
            status = reply["content"]["status"]
            for o in outputs:
                if o["output_type"] == "stream":
                    o["text"] = to_lines(o["text"])
            cell["outputs"] = outputs
            print(f"[セル {idx:2d}] 実行番号 {cell.get('execution_count')}  {time.time() - t0:6.1f} 秒  {status}", flush=True)
            if status != "ok":
                ok = False
                for o in outputs:
                    if o["output_type"] == "error":
                        print(f"  エラー: {o['ename']}: {o['evalue']}", flush=True)
                break
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)

    with open(nb_path, "wb") as f:
        f.write((json.dumps(nb, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    print("保存:", nb_path, "（正常終了）" if ok else "（エラーで停止）", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
