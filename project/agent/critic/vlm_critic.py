"""VLM Critic：调用 OpenAI 兼容多模态接口打分；任何失败回落规则 Critic。

扩展点（A4 真正实现时）：
- 配置 VLM_API_BASE / VLM_API_KEY / VLM_MODEL（未配置则直接用规则 Critic）。
- 将 rubric（五维评分标准 + 标定样例）固化到 agent/critic/threshold.md 后更新提示词。
- 与人工分 Spearman >=0.6 的验收在 agent/eval 里做（A8）。
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import httpx

from agent.config import Settings
from agent.critic.base import DIMS, DIM_TO_TOOL, WEIGHTS, BaseCritic
from agent.critic.rule_critic import RuleCritic
from agent.dag.models import CriticResult, NodeStatus, RunState

logger = logging.getLogger("agent.critic.vlm")

RUBRIC = """你是图像合成质量评审。对合成图按四个维度打分（0-100 整数）：
- lighting 光照一致性：前景受光方向/色温是否与背景环境光吻合
- shadow 阴影一致性：接触阴影方向/软硬/深浅是否合理
- color 色彩一致性：前后景色调、噪点、饱和度是否统一
- edge 边缘质量：前景边缘是否有白边/锯齿/残留背景
只输出 JSON：{"lighting": n, "shadow": n, "color": n, "edge": n, "comment": "一句话"}"""


class VLMCritic(BaseCritic):
    def __init__(self, settings: Settings, api_base: str = "", api_key: str = "", model: str = ""):
        self.settings = settings
        self.fallback = RuleCritic(settings)
        self.api_base = (api_base or settings.llm_api_base).rstrip("/")
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        self.timeout_s = settings.llm_timeout_s

    def available(self) -> bool:
        return bool(self.api_base and self.model)

    def score(self, run: RunState) -> CriticResult:
        if not self.available():
            return self._tag(self.fallback.score(run), "VLM 未配置，规则 Critic 兜底")
        try:
            scores = self._vlm_score(run)
        except Exception as e:
            logger.warning("VLM 打分失败，回落规则 Critic: %s", e)
            return self._tag(self.fallback.score(run), f"VLM 失败回落：{e}")
        overall = round(sum(WEIGHTS[d] * scores[d] for d in DIMS), 1)
        lowest = min(DIMS, key=lambda d: scores[d])
        threshold = self.settings.critic_threshold
        if overall >= threshold:
            action, rerun_role = "pass", None
        elif run.replan_count >= self.settings.max_replans:
            action, rerun_role = "force_pass", None
        else:
            rerun_role = DIM_TO_TOOL[lowest]
            action = "rerun" if run.dag.node_by_role(rerun_role) else "force_pass"
        return CriticResult(scores=scores, overall=overall, threshold=threshold,
                            passed=action == "pass", lowest_dim=lowest, action=action,
                            rerun_role=rerun_role, comment=f"VLM：{scores.get('comment', '')}")

    @staticmethod
    def _tag(r: CriticResult, note: str) -> CriticResult:
        r.comment = f"{r.comment} [{note}]" if r.comment else note
        return r

    def _vlm_score(self, run: RunState) -> dict:
        image_uri = self._final_image(run)
        if not image_uri:
            raise ValueError("无可用最终图")
        path: Path = self._resolve(image_uri)
        b64 = base64.b64encode(path.read_bytes()).decode()
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(
                f"{self.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                json={
                    "model": self.model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": RUBRIC},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        out = {d: float(data[d]) for d in DIMS}
        out["comment"] = str(data.get("comment", ""))
        return out

    def _final_image(self, run: RunState) -> str | None:
        for role in ("export", "harmonize", "shadow_generate"):
            node = run.dag.node_by_role(role)
            if node and node.status == NodeStatus.done and node.primary_artifact:
                return node.primary_artifact
        return None

    def _resolve(self, uri: str) -> Path:
        from agent.schema import resolve_uri
        return resolve_uri(uri, self.settings.artifacts_dir)
