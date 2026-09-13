"""快速验证 carvekit 真实 BiRefNet 是否可用（含 HF 镜像权重下载）。"""
import os, sys, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from pathlib import Path

img = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/AIcode/生产实习/data/matting/train/hair_0000.png")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("D:/AIcode/生产实习/experiments/baseline/carvekit_test_alpha.png")
out.parent.mkdir(parents=True, exist_ok=True)

from PIL import Image
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
from carvekit.api.high import HiInterface
print(f"[test] building HiInterface(object_type=object, device={device}) ...")
interface = HiInterface(object_type="object", device=device, seg_mask_size=1024, fp16=(device == "cuda"))
print(f"[test] interface built in {time.time()-t0:.1f}s")
t1 = time.time()
res = interface([str(img)])
print(f"[test] inference in {time.time()-t1:.1f}s")
rgba = res[0].convert("RGBA")
alpha = rgba.split()[-1]
alpha.save(str(out))
print(f"[test] alpha saved -> {out}  size={alpha.size}")
print("[test] OK")
