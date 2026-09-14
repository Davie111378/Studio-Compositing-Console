"""A1 验收：8 个 Schema 结构完整 + 校验器行为正确。"""

from agent.schema import get_tool_schema, load_registry, tool_ids, validate_payload, validate_tool_inputs
from agent.errors import AgentError

EXPECTED_TOOLS = ["matting", "background_generate", "lighting_estimate", "relight",
                  "shadow_generate", "harmonize", "enhance", "export"]


def test_registry_loads_all_8_tools():
    registry = load_registry()
    assert set(registry.keys()) == set(EXPECTED_TOOLS)
    assert tool_ids() == EXPECTED_TOOLS  # T01..T08 顺序


def test_each_schema_has_four_pieces():
    for tool, s in load_registry().items():
        assert s["input"].get("type") == "object", tool
        assert s["output"].get("type") == "object", tool
        assert s["errors"] and all({"code", "message", "retryable"} <= set(e) for e in s["errors"]), tool
        assert {"p50_ms", "p95_ms", "timeout_ms"} <= set(s["latency"]), tool


def test_t01_input_validation():
    ok = {"image": "asset://uploads/s1/a1.png", "mode": "auto"}
    validate_tool_inputs("matting", ok, {"quality": "draft"})

    bad = {"mode": "wrong"}  # 缺 image + 非法 mode
    try:
        validate_tool_inputs("matting", bad, {})
        raise AssertionError("应当抛 E_INVALID_INPUT")
    except AgentError as e:
        assert e.code == "E_INVALID_INPUT"
        assert any("image" in err for err in e.detail)


def test_t03_output_contract():
    s = get_tool_schema("lighting_estimate")
    assert s["output"]["required"] == ["sh_coeff", "light_dir", "color_temp", "intensity"]
    errs = validate_payload(s["output"], {
        "sh_coeff": [0.1] * 8,             # 少于 9 个 -> 报错
        "light_dir": {"azimuth": 1.0},      # 缺 polar -> 报错
        "color_temp": 5600, "intensity": 1.0,
    })
    assert len(errs) == 2


def test_unknown_tool_rejected():
    try:
        get_tool_schema("super_res")
        raise AssertionError("8 个工具之外应当拒绝")
    except AgentError as e:
        assert e.code == "E_TOOL_UNKNOWN"
