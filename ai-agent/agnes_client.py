# -*- coding: utf-8 -*-
"""
agnes_client.py — OpenAI 兼容 LLM 客户端封装 (多 provider 可切换)。

支持的后端 (provider):
  - agnes : Agnes AI          (https://apihub.agnes-ai.com/v1)
  - qwen  : 通义千问 DashScope (https://dashscope.aliyuncs.com/compatible-mode/v1)

统一能力:
  - 文本对话 + function calling
  - VLM 看图 (content 内含 image_url)
  - 文生图 (/images/generations)

切换方式 (三种, 优先级从高到低):
  1. 代码: get_client("qwen") / AgnesClient(provider="qwen")
  2. 环境变量: LLM_PROVIDER=qwen          (默认 agnes)
  3. 前端: studio_web /agent 页面的后端下拉

依赖仅 urllib.request / json / base64 (零第三方)。
"""
from __future__ import annotations
import base64, io, json, os, time, urllib.request, urllib.error
from pathlib import Path

# .env 加载 (优先真实环境变量；dotenv 缺失时用内置零依赖解析器)
_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv_fallback(path: Path) -> int:
    """零依赖 .env 解析: KEY=VALUE / # 注释 / 引号包裹 / 行内注释。"""
    if not path.exists():
        return 0
    n = 0
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if not k:
            continue
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        elif " #" in v:
            v = v.split(" #", 1)[0].rstrip()
        # 真实环境变量优先, 不覆盖
        if k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env", override=False)
except ImportError:
    _load_dotenv_fallback(_ROOT / ".env")


def _env(k: str, default: str = "") -> str:
    v = os.environ.get(k, "").strip()
    return v or default


def imread_any(path: str):
    """中文路径安全读图 (np.fromfile + cv2.imdecode)；失败回退 PIL 字节直读。"""
    return Path(path).read_bytes()


def _img_data_url(path: str) -> str:
    ext = Path(path).suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp", "bmp": "image/bmp"}.get(ext.lstrip("."), "image/png")
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


# ============================================================ Provider 注册表
# 每个 provider: key / base / chat_model / vl_model / image_model / 能力开关
PROVIDERS: dict[str, dict] = {
    "agnes": {
        "label": "Agnes AI",
        "key": _env("AGNES_API_KEY", "sk-7lP9m75PadBRnATt6zn5I08IQK0s17354aHHionhutT22iTQ"),
        "base": _env("AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1"),
        "chat_model": _env("AGNES_CHAT_MODEL", "agnes-3.0-flash"),
        "vl_model": _env("AGNES_VL_MODEL", _env("AGNES_CHAT_MODEL", "agnes-3.0-flash")),
        "image_model": _env("AGNES_IMAGE_MODEL", "agnes-image-2.5-flash"),
        "supports_vision": True,
        "supports_image_gen": True,
        "note": "多模态旗舰, 文本/看图/文生图全支持",
    },
    "qwen": {
        "label": "通义千问 (DashScope)",
        "key": _env("QWEN_API_KEY", ""),
        "base": _env("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "chat_model": _env("QWEN_CHAT_MODEL", "qwen-flash"),
        "vl_model": _env("QWEN_VL_MODEL", "qwen-vl-max"),
        "image_model": _env("QWEN_IMAGE_MODEL", "wanx2.1-t2i-turbo"),
        "supports_vision": True,
        "supports_image_gen": True,
        "note": "免费额度模型 (qwen-flash 文本 / qwen-vl 看图)",
    },
}

DEFAULT_PROVIDER = _env("LLM_PROVIDER", "agnes")

# 兼容旧变量名 (外部可能直接 import)
DEFAULT_KEY = PROVIDERS[DEFAULT_PROVIDER]["key"]
DEFAULT_BASE = PROVIDERS[DEFAULT_PROVIDER]["base"]
CHAT_MODEL = PROVIDERS[DEFAULT_PROVIDER]["chat_model"]
IMAGE_MODEL = PROVIDERS[DEFAULT_PROVIDER]["image_model"]


def list_providers() -> list[dict]:
    """返回可选后端清单 (供前端下拉/健康检查)。key 只回传是否已配置, 不回传明文。"""
    out = []
    for name, cfg in PROVIDERS.items():
        out.append({
            "name": name,
            "label": cfg["label"],
            "base": cfg["base"],
            "chat_model": cfg["chat_model"],
            "vl_model": cfg["vl_model"],
            "image_model": cfg["image_model"],
            "supports_vision": cfg["supports_vision"],
            "supports_image_gen": cfg["supports_image_gen"],
            "key_configured": bool(cfg["key"]),
            "note": cfg["note"],
            "is_default": name == DEFAULT_PROVIDER,
        })
    return out


def resolve_provider(provider: str | None = None) -> str:
    """规范化 provider 名 (大小写/别名容忍), 未知则回退默认。"""
    if not provider:
        return DEFAULT_PROVIDER if DEFAULT_PROVIDER in PROVIDERS else "agnes"
    p = str(provider).strip().lower()
    alias = {"qwen": "qwen", "tongyi": "qwen", "dashscope": "qwen", "通义": "qwen",
             "千问": "qwen", "qwen-flash": "qwen",
             "agnes": "agnes", "agnes-ai": "agnes", "agnesai": "agnes"}
    p = alias.get(p, p)
    return p if p in PROVIDERS else (DEFAULT_PROVIDER if DEFAULT_PROVIDER in PROVIDERS else "agnes")


# ============================================================ 客户端
class AgnesClient:
    """OpenAI 兼容客户端 (多 provider)。调用失败自动重试。

    构造方式:
      AgnesClient()                      # 用 LLM_PROVIDER (默认 agnes)
      AgnesClient(provider="qwen")       # 显式指定千问
      AgnesClient(key=..., base=...)     # 完全自定义 (测试桩)
    """

    def __init__(self, key: str | None = None, base: str | None = None,
                 chat_model: str | None = None, image_model: str | None = None,
                 provider: str | None = None, vl_model: str | None = None):
        self.provider = resolve_provider(provider)
        cfg = PROVIDERS[self.provider]
        # 显式参数 > provider 配置
        self.key = key or cfg["key"]
        self.base = (base or cfg["base"]).rstrip("/")
        self.chat_model = chat_model or cfg["chat_model"]
        self.vl_model = vl_model or cfg.get("vl_model") or self.chat_model
        self.image_model = image_model or cfg["image_model"]

    # ---------------- 底层 HTTP ----------------
    def _post(self, path: str, body: dict, timeout: int = 120, retries: int = 3) -> dict:
        url = f"{self.base}/{path}"
        last = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    url, data=json.dumps(body).encode(),
                    headers={"Authorization": f"Bearer {self.key}",
                             "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                err = e.read().decode()[:400]
                # 429/5xx 可重试
                if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    last = f"HTTP {e.code}: {err}"
                    continue
                raise RuntimeError(f"[{self.provider}] HTTP {e.code}: {err}")
            except Exception as e:
                last = str(e)
                if attempt < retries - 1:
                    time.sleep(1.0)
                else:
                    raise RuntimeError(f"[{self.provider}] 请求失败: {last}")
        raise RuntimeError(f"[{self.provider}] 重试耗尽: {last}")

    # ---------------- 文本对话 + function calling ----------------
    def chat(self, messages: list, tools: list | None = None,
             max_tokens: int = 1024, temperature: float = 0.2,
             tool_choice: str | None = "auto", extra: dict | None = None,
             model: str | None = None) -> dict:
        """返回 OpenAI 风格的 completion dict。"""
        body = {"model": model or self.chat_model, "messages": messages,
                "max_tokens": max_tokens, "temperature": temperature}
        if tools:
            body["tools"] = tools
            if tool_choice:
                body["tool_choice"] = tool_choice
        if extra:
            body.update(extra)
        return self._post("chat/completions", body)

    # ---------------- 带图 VLM ----------------
    def chat_with_image(self, text: str, image_paths: list, tools: list | None = None,
                        max_tokens: int = 1024) -> dict:
        content: list = [{"type": "text", "text": text}]
        for p in image_paths:
            content.append({"type": "image_url", "image_url": {"url": _img_data_url(p)}})
        # 视觉请求走 vl_model (分体式 provider 必需, agnes 两者相同)
        return self.chat([{"role": "user", "content": content}], tools=tools,
                         max_tokens=max_tokens, model=self.vl_model)

    # ---------------- 文生图 ----------------
    def generate_image(self, prompt: str, out_path: str,
                       size: str = "1024x1024", model: str | None = None) -> dict:
        body = {"model": model or self.image_model, "prompt": prompt,
                "n": 1, "size": size}
        d = self._post("images/generations", body, timeout=180)
        item = (d.get("data") or [{}])[0]
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if item.get("b64_json"):
            out.write_bytes(base64.b64decode(item["b64_json"]))
        elif item.get("url"):
            req = urllib.request.Request(item["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                out.write_bytes(r.read())
        else:
            raise RuntimeError(f"[{self.provider}] 图像生成无返回 (model={self.image_model}): {str(d)[:200]}")
        return {"path": str(out), "prompt": prompt, "provider": self.provider}

    # ---------------- 健康检查 ----------------
    def ping(self, timeout: int = 20) -> dict:
        """轻量连通性测试, 供前端"测试连接"。"""
        try:
            t0 = time.time()
            d = self.chat([{"role": "user", "content": "回复: ok"}],
                          max_tokens=8, temperature=0.0)
            dt = int((time.time() - t0) * 1000)
            txt = (d["choices"][0]["message"].get("content") or "").strip()
            return {"ok": True, "provider": self.provider, "model": self.chat_model,
                    "latency_ms": dt, "reply": txt[:40]}
        except Exception as e:
            return {"ok": False, "provider": self.provider, "error": str(e)[:200]}

    def __repr__(self):
        return f"AgnesClient(provider={self.provider}, model={self.chat_model}, base={self.base})"


# 供旧代码 import 的名字 (语义上就是"LLM 客户端")
LLMClient = AgnesClient


def get_client(provider: str | None = None, **kw) -> AgnesClient:
    """统一工厂: 按 provider 取客户端。provider=None 时走 LLM_PROVIDER。"""
    return AgnesClient(provider=provider, **kw)


if __name__ == "__main__":
    print("默认后端:", DEFAULT_PROVIDER)
    for info in list_providers():
        print(f"  - {info['name']:6s} {info['label']:22s} key={'✓' if info['key_configured'] else '✗'} "
              f"chat={info['chat_model']}")
    for name in PROVIDERS:
        c = AgnesClient(provider=name)
        print(f"\n[{name}] {c}")
        print("  ping:", c.ping())
