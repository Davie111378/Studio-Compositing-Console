# -*- coding: utf-8 -*-
"""
local_planner.py — Stage 4: 本地 SFT 模型 Planner (离线, 不依赖 agnes API)
- 加载 Qwen2.5-0.5B-Instruct + sft_v1 LoRA adapter
- 与 agent/planner.py 的 AgnesLLM 同接口: complete(messages) -> str
用法: planner_mod.plan(..., llm=LocalPlannerLLM())
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent_training"))
MODEL_DIR = ROOT / "agent_training" / "models" / "qwen2.5-0.5b-instruct"
ADAPTER = ROOT / "agent_training" / "checkpoints" / "sft_v1"

# 训练/推理必须使用**完全相同**的 system prompt (否则模型见到的是没见过的工具表 -> 幻觉工具名)。
# 单一事实来源: build_sft_data.SFT_SYSTEM
from build_sft_data import SFT_SYSTEM  # noqa: E402


class LocalPlannerLLM:
    def __init__(self, device: str = "cuda", max_new_tokens: int = 512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR), torch_dtype=torch.bfloat16)
        if ADAPTER.exists():
            model = PeftModel.from_pretrained(model, str(ADAPTER))
        self.model = model.to(device).eval()
        self.device = device
        self.max_new_tokens = max_new_tokens

    def complete(self, messages: list, max_tokens: int | None = None) -> str:
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt").to(self.device)
        n_in = ids["input_ids"].shape[1]
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_tokens or self.max_new_tokens,
                                      do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0][n_in:], skip_special_tokens=True)
