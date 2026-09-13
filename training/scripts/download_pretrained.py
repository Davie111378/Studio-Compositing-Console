"""
下载公开 matting 数据集和 BiRefNet 预训练权重
使用 hf-mirror.com (HuggingFace 国内镜像)
"""

from __future__ import annotations
import os
import sys
import time
import json
from pathlib import Path

# 设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def download_with_hf_mirror(repo_id: str, filename: str, save_dir: Path, use_hf_hub: bool = True):
    """从 hf-mirror 下载单文件。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / filename
    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return out_path
    if use_hf_hub:
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(save_dir))
            print(f"[hf-mirror] downloaded {p}")
            return Path(p)
        except Exception as e:
            print(f"[hf-mirror] failed: {e}, falling back to curl")
    # 兜底 curl
    import urllib.request
    url = f"https://hf-mirror.com/{repo_id}/resolve/main/{filename}"
    print(f"[curl] {url}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(out_path))
    return out_path


def main():
    out_root = Path("D:/AIcode/生产实习/training/pretrained")
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. 下载 BiRefNet 公开权重（来自 ZhengPeng7/BiRefNet）
    repos = [
        ("ZhengPeng7/BiRefNet", "BiRefNet-general-epoch_120.pth", "general"),
        ("ZhengPeng7/BiRefNet", "BiRefNet-portrait-epoch_120.pth", "portrait"),
        ("ZhengPeng7/BiRefNet", "BiRefNet-dis-epoch_120.pth", "dis"),
    ]
    for repo_id, fname, tag in repos:
        try:
            download_with_hf_mirror(repo_id, fname, out_root / tag)
        except Exception as e:
            print(f"[err] {tag}: {e}")

    # 2. 下载 SAM 2 权重（轻量版）
    sam_repos = [
        ("facebook/sam2.1-hiera-small", "sam2.1_hiera_small.pt", "sam2.1"),
    ]
    for repo_id, fname, tag in sam_repos:
        try:
            download_with_hf_mirror(repo_id, fname, out_root / tag)
        except Exception as e:
            print(f"[err] {tag}: {e}")


if __name__ == "__main__":
    main()
