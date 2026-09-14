"""T01–T08 Schema 加载与轻量 JSON-Schema 校验。

不引第三方 jsonschema 依赖（避免环境漂移）；只实现本项目 Schema 用到的子集：
type / required / properties / items / enum / minItems / maxItems。

统一信封见 ENVELOPE.md。Agent 对工具的一切调用先过本模块校验（规范 N1 接口先行）。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.errors import AgentError, E_INVALID_INPUT, E_SCHEMA_VALIDATION, E_TOOL_UNKNOWN

SCHEMA_DIR = Path(__file__).resolve().parent

_TYPE_MAP = {
    "object": dict,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
}


@lru_cache(maxsize=1)
def load_registry() -> dict[str, dict[str, Any]]:
    """加载 T01–T08，返回 {tool_name: schema_dict}，并做结构自检。"""
    registry: dict[str, dict[str, Any]] = {}
    for tid in range(1, 9):
        path = SCHEMA_DIR / f"T{tid:02d}.json"
        if not path.exists():
            raise AgentError(E_SCHEMA_VALIDATION, f"缺少工具 Schema 文件: {path.name}")
        data = json.loads(path.read_text(encoding="utf-8"))
        _check_schema_structure(data, path.name)
        if data["tool"] in registry:
            raise AgentError(E_SCHEMA_VALIDATION, f"工具名重复: {data['tool']}")
        registry[data["tool"]] = data
    return registry


def _check_schema_structure(data: dict, filename: str) -> None:
    for key in ("id", "tool", "version", "input", "output", "errors", "latency"):
        if key not in data:
            raise AgentError(E_SCHEMA_VALIDATION, f"{filename} 缺少四件套字段: {key}")
    for section in ("input", "output"):
        if data[section].get("type") != "object":
            raise AgentError(E_SCHEMA_VALIDATION, f"{filename} {section} 必须是 object schema")
    for err in data["errors"]:
        if not {"code", "message", "retryable"} <= set(err):
            raise AgentError(E_SCHEMA_VALIDATION, f"{filename} errors 项缺少 code/message/retryable")
    for key in ("p50_ms", "p95_ms", "timeout_ms"):
        if key not in data["latency"]:
            raise AgentError(E_SCHEMA_VALIDATION, f"{filename} latency 缺少 {key}")


def get_tool_schema(tool: str) -> dict[str, Any]:
    schema = load_registry().get(tool)
    if schema is None:
        raise AgentError(E_TOOL_UNKNOWN, f"未知工具: {tool}（固定 8 个，不得扩张）")
    return schema


def tool_ids() -> list[str]:
    return [s["tool"] for s in load_registry().values()]


def validate_payload(schema: dict[str, Any], payload: Any, path: str = "$") -> list[str]:
    """返回错误列表；空列表 = 校验通过。"""
    errors: list[str] = []
    expected = schema.get("type")
    if expected and not isinstance(payload, _TYPE_MAP[expected]):
        # bool 是 int 子类，排除误判
        if not (expected == "number" and isinstance(payload, bool) is False):
            if not (expected == "integer" and isinstance(payload, bool)):
                errors.append(f"{path}: 期望 {expected}，实际 {type(payload).__name__}")
            return errors
    if expected == "string" and isinstance(payload, str):
        pass
    if isinstance(payload, dict) and expected == "object":
        for req in schema.get("required", []):
            if req not in payload:
                errors.append(f"{path}: 缺少必填字段 '{req}'")
        for name, sub in schema.get("properties", {}).items():
            if name in payload:
                errors.extend(validate_payload(sub, payload[name], f"{path}.{name}"))
    if isinstance(payload, list) and expected == "array":
        if "minItems" in schema and len(payload) < schema["minItems"]:
            errors.append(f"{path}: 元素数 {len(payload)} < minItems {schema['minItems']}")
        if "maxItems" in schema and len(payload) > schema["maxItems"]:
            errors.append(f"{path}: 元素数 {len(payload)} > maxItems {schema['maxItems']}")
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(payload):
                errors.extend(validate_payload(item_schema, item, f"{path}[{i}]"))
    if "enum" in schema and isinstance(payload, str) and payload not in schema["enum"]:
        errors.append(f"{path}: '{payload}' 不在枚举 {schema['enum']} 内")
    return errors


def validate_tool_inputs(tool: str, inputs: dict, options: dict) -> None:
    """校验工具输入与 options，不通过抛 E_INVALID_INPUT。"""
    schema = get_tool_schema(tool)
    errs = validate_payload(schema["input"], inputs)
    opt_schema = schema.get("options", {"type": "object", "properties": {}})
    errs += validate_payload(opt_schema, options)
    if errs:
        raise AgentError(E_INVALID_INPUT, f"工具 {tool} 输入校验失败", detail=errs)


def resolve_uri(uri: str, artifacts_root: Path) -> Path:
    """asset:// / artifact:// -> 本地路径。"""
    for prefix in ("asset://", "artifact://"):
        if uri.startswith(prefix):
            p = artifacts_root / uri[len(prefix):]
            if not p.exists():
                raise AgentError(E_INVALID_INPUT, f"URI 指向的文件不存在: {uri}")
            return p
    # 兼容直接给绝对/相对路径
    p = Path(uri)
    if p.exists():
        return p
    raise AgentError(E_INVALID_INPUT, f"无法解析的 URI: {uri}")


def to_artifact_uri(path: Path, artifacts_root: Path) -> str:
    return "artifact://" + path.resolve().relative_to(artifacts_root.resolve()).as_posix()


def public_artifact_url(uri: str) -> str:
    """artifact:// / asset:// -> 同源 HTTP 路径（Agent 服务 /artifacts 静态挂载）。"""
    for prefix in ("asset://", "artifact://"):
        if uri.startswith(prefix):
            return "/artifacts/" + uri[len(prefix):]
    return uri
