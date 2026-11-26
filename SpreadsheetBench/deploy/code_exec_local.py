#!/usr/bin/env python3
"""本地代码执行服务（无 Docker 环境下的 code_exec_docker 替代品）。

接口与 code_exec_docker/api.py 完全一致：
    POST /execute  {"convid": "...", "code": "..."}
    -> {"result": "<stdout 或 traceback>", "new_kernel_created": bool}

每个 convid 维护一个持久的命名空间，等价于常驻 Jupyter kernel；被执行代码中
的 /mnt/data 需指向数据集目录（可用软链接实现）。

用法:
    /remote-home/yichengfu/anaconda3/envs/spread/bin/python code_exec_local.py --port 8081
"""

import argparse
import io
import json
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_namespaces = {}
_lock = threading.Lock()


def execute(conv_id, code):
    with _lock:
        ns = _namespaces.get(conv_id)
        created = ns is None
        if created:
            ns = {"__name__": "__main__"}
            _namespaces[conv_id] = ns

        buf = io.StringIO()
        error = None
        with redirect_stdout(buf), redirect_stderr(buf):
            try:
                exec(compile(code, f"<cell {conv_id}>", "exec"), ns)
            except Exception:
                error = traceback.format_exc()

    out = buf.getvalue().rstrip()
    if error is not None:
        result = (out + "\n" if out else "") + error
    else:
        result = out if out else "[Code executed successfully with no output]"
    return {"result": result, "new_kernel_created": created}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        if self.path != "/execute":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length))
        payload = execute(data["convid"], data["code"])
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--host", type=str, default="0.0.0.0")
    args = p.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"code exec server listening on {args.host}:{args.port}")
    server.serve_forever()
