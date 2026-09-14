# 工具统一信封（Envelope）

所有 T01–T08 工具共用同一请求/响应信封（总规范 §4.3），Agent 只依赖信封，不依赖任何模型实现。

## 请求

```json
{
  "tool": "matting",
  "version": "1.0",
  "request_id": "uuid",
  "inputs": { "image": "asset://uploads/s1/a1.png" },
  "options": { "quality": "draft|normal|fine" },
  "out_dir": "runs/<run_id>/<node_id>/v1"
}
```

`out_dir` 为相对 `data/artifacts/` 的目录，工具产出的文件必须写入该目录，并返回 `artifact://` 相对 URI。

## 响应

```json
{
  "request_id": "uuid",
  "status": "success | failed | timeout",
  "outputs": { "rgba_png": "artifact://runs/.../t_r1_n1_v1_rgba.png", "alpha_png": "...", "mask_png": "..." },
  "error": { "code": "E_MATTING_OOM", "message": "...", "retryable": true },
  "latency_ms": 1830,
  "artifacts": ["artifact://runs/.../t_r1_n1_v1_rgba.png"]
}
```

## 约定

- `status=failed/timeout` 时 `outputs` 为空对象、`error` 必填；`success` 时 `error` 为 null。
- `retryable=true` 的错误由 Executor 自动重试（L1 降级：重试 -> 换低精度档位）。
- 工具内部不得访问 Agent 状态；只读输入 URI、写输出目录。
