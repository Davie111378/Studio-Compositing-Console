# -*- coding: utf-8 -*-
"""
train_sft.py — Stage 3: LoRA 微调 Qwen2.5-0.5B-Instruct → 演播室合成 Planner
- 数据: agent_training/data/sft_train.jsonl (messages 格式)
- 损失: 仅 assistant 段 (prompt 段 labels=-100)
- 4GB 显存: bf16 + LoRA(r16) + gradient_checkpointing, batch 2 × accum 8
- 产出: agent_training/checkpoints/sft_v1 (adapter)
运行: python agent_training/train_sft.py [--epochs 3]
"""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "agent_training" / "models" / "qwen2.5-0.5b-instruct"
DATA = ROOT / "agent_training" / "data"
CKPT = ROOT / "agent_training" / "checkpoints" / "sft_v1"
MAX_LEN = 1024


def load_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


class SFTDataset(Dataset):
    """prompt(user+system) 与 assistant 分别编码, labels 仅 assistant 段。"""

    def __init__(self, rows, tok, max_len=MAX_LEN):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]["messages"]
        sys_u = r[0]["content"]
        user = r[1]["content"]
        asst = r[2]["content"]
        prompt_ids = self.tok.apply_chat_template(
            [{"role": "system", "content": sys_u}, {"role": "user", "content": user}],
            tokenize=True, add_generation_prompt=True, return_dict=False)
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids.input_ids
        asst_ids = self.tok.encode(asst, add_special_tokens=False) + [self.tok.eos_token_id]
        ids = (prompt_ids + asst_ids)[: self.max_len]
        labels = ([-100] * len(prompt_ids) + asst_ids)[: self.max_len]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    mx = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attn = [], [], []
    for x in batch:
        n = mx - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_id] * n)
        labels.append(x["labels"] + [-100] * n)
        attn.append([1] * len(x["input_ids"]) + [0] * n)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model = model.cuda()          # 模型搬上 GPU (此前漏掉: batch 在 cuda、模型在 cpu)
    model.print_trainable_parameters()

    train_rows = load_jsonl(DATA / "sft_train.jsonl")
    val_rows = load_jsonl(DATA / "sft_val.jsonl")
    train_ds = SFTDataset(train_rows, tok)
    val_ds = SFTDataset(val_rows, tok)
    pad_id = tok.pad_token_id or tok.eos_token_id
    g = torch.Generator().manual_seed(42)

    def dl(ds, shuffle=False):
        return DataLoader(ds, batch_size=a.batch, shuffle=shuffle, generator=g if shuffle else None,
                          collate_fn=lambda b: collate(b, pad_id))

    train_loader = dl(train_ds, shuffle=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    total_steps = math.ceil(len(train_loader) * a.epochs / a.accum) + 2   # +2 余量: 末批 flush 会多一步
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total_steps,
                                                pct_start=0.1)
    CKPT.mkdir(parents=True, exist_ok=True)
    log_path = CKPT / "train_log.txt"
    log_f = log_path.open("w", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        log_f.write(s + "\n"); log_f.flush()

    model.train()
    step, accum_loss, t0 = 0, 0.0, time.time()
    done = False
    for epoch in range(a.epochs):
        for i, batch in enumerate(train_loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / a.accum
            loss.backward()
            accum_loss += loss.item()
            if (i + 1) % a.accum == 0 or i == len(train_loader) - 1:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                log(f"epoch{epoch+1} step{step}/{total_steps} loss={accum_loss:.4f} "
                    f"lr={sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s")
                accum_loss = 0.0
    done = True

    # --- 快速验证损失 ---
    model.eval()
    vl = 0.0
    with torch.no_grad():
        for batch in dl(val_ds):
            batch = {k: v.cuda() for k, v in batch.items()}
            vl += model(**batch).loss.item()
    vl /= max(len(val_ds) // a.batch, 1)
    log(f"val_loss={vl:.4f}")

    model.save_pretrained(str(CKPT))
    tok.save_pretrained(str(CKPT))
    (CKPT / "done.flag").write_text(f"done {time.strftime('%F %T')} val_loss={vl:.4f}", encoding="utf-8")
    log(f"saved adapter -> {CKPT}")
    log_f.close()


if __name__ == "__main__":
    main()
