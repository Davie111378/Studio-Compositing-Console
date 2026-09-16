# -*- coding: utf-8 -*-
"""ImageCompose 开发服务器：静态站点 + 同源反向代理到 Agent 服务。

    python tools/dev_server.py
    环境变量：
        IMC_AGENT_URL   Agent 上游地址，仅允许 http/https（默认 http://127.0.0.1:8000）
        IMC_SITE_PORT   站点端口（默认 8899）

代理路径前缀（固定，不接受用户提供的任意 URL，无 SSRF 面）：
    /api/...  /healthz  /artifacts/...   -> Agent（IMC_AGENT_URL）
    /speech/...                          -> speech 语音服务（IMC_SPEECH_URL）
前端因此全部使用同源相对路径请求后端，无跨域、无硬编码主机。
"""
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = os.environ.get("IMC_AGENT_URL", "http://127.0.0.1:8000").rstrip("/")
SPEECH_UPSTREAM = os.environ.get("IMC_SPEECH_URL", "http://127.0.0.1:8200").rstrip("/")
PORT = int(os.environ.get("IMC_SITE_PORT", "8899"))
PROXY_PREFIXES = ("/api/", "/healthz", "/artifacts/", "/speech/")

_scheme = urlsplit(UPSTREAM).scheme
if _scheme not in ("http", "https"):
    sys.exit(f"IMC_AGENT_URL 仅允许 http/https，当前非法：{UPSTREAM!r}")
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 本机直连，绕过系统代理


class Handler(SimpleHTTPRequestHandler):
    # 以 frontend 目录为静态根
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    # ---- 代理判定 ----
    def _proxied(self) -> bool:
        return self.path.split("?")[0].startswith(PROXY_PREFIXES) \
            or self.path.split("?")[0] == "/healthz"

    def _forward(self, method: str) -> None:
        is_speech = self.path.split("?")[0].startswith("/speech/")
        upstream = SPEECH_UPSTREAM if is_speech else UPSTREAM
        # /speech/* 剥掉前缀转发（speech 服务路由为 /healthz、/api/v1/speech/*）
        path = self.path[len("/speech"):] if is_speech else self.path
        if not path.startswith("/"):
            path = "/" + path
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        req = urllib.request.Request(upstream + path, data=body, method=method)
        for hop in ("Host", "Connection", "Content-Length", "Accept-Encoding"):
            del self.headers[hop]
        for key, val in self.headers.items():
            req.add_header(key, val)
        try:
            with _opener.open(req, timeout=300) as resp:
                payload = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Length", str(len(payload)))
                ctype = resp.headers.get("Content-Type")
                if ctype:
                    self.send_header("Content-Type", ctype)
                self.end_headers()
                if method != "HEAD":
                    self.wfile.write(payload)
        except urllib.error.HTTPError as e:
            payload = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:  # 上游不可达等
            payload = json.dumps({"error": "agent upstream unreachable", "detail": str(e)}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def do_GET(self):
        if self._proxied():
            self._forward("GET")
        else:
            super().do_GET()

    def do_HEAD(self):
        if self._proxied():
            self._forward("HEAD")
        else:
            super().do_HEAD()

    def do_POST(self):
        if self._proxied():
            self._forward("POST")
        else:
            self.send_error(405, "POST 仅允许代理到 Agent 服务")

    def end_headers(self):  # 开发期禁缓存
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):  # 精简日志
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


class Server(ThreadingHTTPServer):
    daemon_threads = True
    # Windows 上 SO_REUSEADDR 允许第二个实例静默绑定同一端口，导致部分请求
    # 落到旧实例（/api 返回 404 HTML）。显式关闭，端口被占用时直接启动失败。
    allow_reuse_address = False


if __name__ == "__main__":
    srv = Server(("127.0.0.1", PORT), Handler)
    print(f"ImageCompose dev server  http://127.0.0.1:{PORT}  (静态根: {ROOT})")
    print(f"代理 /api, /healthz, /artifacts  ->  {UPSTREAM}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
