# -*- coding: utf-8 -*-
"""capability_test.py — Agent 抠图/绿幕背景/滤镜特效 能力测试
1. 抠图: P3M-10k + HM-1k 样本 → T01_matting → 与 GT alpha 算 SAD/MSE/Grad
2. 绿幕背景: 绿幕图片(演播室场景)做背景, AI生成前景人物合成 → T02 green_key / T06 greenscreen
3. 滤镜特效: T07_enhance 各 mode
"""
from __future__ import annotations
import sys, json, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
AGENT = ROOT / "agent"
sys.path.insert(0, str(AGENT))
sys.path.insert(0, str(ROOT / "ai-service"))
sys.path.insert(0, str(ROOT / "ai-service" / "src"))
sys.path.insert(0, str(ROOT / "ai-agent"))
sys.path.insert(0, str(ROOT / "training" / "scripts"))

import numpy as np
from PIL import Image
import adapter

TEST_DATA = ROOT / "data" / "dataset_test"
OUT_DIR = ROOT / "outputs" / "capability_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PY = "C:/Users/zhaod/venvs/prod-gpu/Scripts/python.exe"


# ============ 1. 抠图精度测试 ============
def matting_metrics(pred_alpha: np.ndarray, gt_alpha: np.ndarray) -> dict:
    """SAD(×1000) / MSE / Grad(×1000) — 与 evaluate_matting.py 对齐"""
    p = pred_alpha.astype(np.float64) / 255.0
    g = gt_alpha.astype(np.float64) / 255.0
    h, w = p.shape
    # 尺寸对齐
    if p.shape != g.shape:
        g = np.array(Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)).astype(np.float64) / 255.0
    sad = float(np.abs(p - g).sum()) * 1000.0 / (h * w)
    mse = float(((p - g) ** 2).sum()) / (h * w)
    # Grad (Sobel)
    try:
        import cv2
        px = cv2.Sobel(p, cv2.CV_64F, 1, 0, ksize=3)
        py_ = cv2.Sobel(p, cv2.CV_64F, 0, 1, ksize=3)
        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        gy_ = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        grad = float(np.abs(px - gx).sum() + np.abs(py_ - gy_).sum()) * 1000.0 / (h * w)
    except Exception:
        grad = -1.0
    return {"SAD": round(sad, 2), "MSE": round(mse, 4), "Grad": round(grad, 2)}


def test_matting():
    """P3M + HM 抠图精度"""
    results = []
    # P3M
    p3m_img = TEST_DATA / "p3m" / "img"
    p3m_mask = TEST_DATA / "p3m" / "mask"
    for img_f in sorted(p3m_img.glob("*.jpg")):
        mask_f = p3m_mask / (img_f.stem + ".png")
        if not mask_f.exists():
            print(f"  [skip] {img_f.name}: GT mask missing")
            continue
        t0 = time.time()
        ctx = {"run_id": f"matting_p3m_{img_f.stem}"}
        r = adapter.call_tool("T01_matting", "n1",
                              {"image": str(img_f)}, quality="draft", ctx=ctx)
        sec = round(time.time() - t0, 2)
        if not r["ok"]:
            results.append({"dataset": "P3M", "image": img_f.name, "error": r["error"]["message"][:200], "sec": sec})
            print(f"  [FAIL] P3M/{img_f.name}: {r['error']['message'][:100]}")
            continue
        alpha_path = r["data"]["alpha_path"]
        pred = np.array(Image.open(alpha_path).convert("L"))
        gt = np.array(Image.open(mask_f).convert("L"))
        m = matting_metrics(pred, gt)
        m.update({"dataset": "P3M", "image": img_f.name, "sec": sec,
                  "engine": r["data"].get("engine", ""), "fg_ratio": r["data"].get("fg_ratio")})
        results.append(m)
        print(f"  [P3M] {img_f.name}: SAD={m['SAD']} MSE={m['MSE']} Grad={m['Grad']} ({sec}s)")

    # HM
    hm_img = TEST_DATA / "hm" / "img"
    hm_mask = TEST_DATA / "hm" / "mask"
    for img_f in sorted(hm_img.glob("*.jpg")):
        mask_f = hm_mask / (img_f.name.rsplit(".", 1)[0] + ".png")
        if not mask_f.exists():
            continue
        t0 = time.time()
        ctx = {"run_id": f"matting_hm_{img_f.stem}"}
        r = adapter.call_tool("T01_matting", "n1",
                              {"image": str(img_f)}, quality="draft", ctx=ctx)
        sec = round(time.time() - t0, 2)
        if not r["ok"]:
            results.append({"dataset": "HM", "image": img_f.name, "error": r["error"]["message"][:200], "sec": sec})
            print(f"  [FAIL] HM/{img_f.name}: {r['error']['message'][:100]}")
            continue
        alpha_path = r["data"]["alpha_path"]
        pred = np.array(Image.open(alpha_path).convert("L"))
        gt = np.array(Image.open(mask_f).convert("L"))
        m = matting_metrics(pred, gt)
        m.update({"dataset": "HM", "image": img_f.name, "sec": sec,
                  "engine": r["data"].get("engine", ""), "fg_ratio": r["data"].get("fg_ratio")})
        results.append(m)
        print(f"  [HM] {img_f.name}: SAD={m['SAD']} MSE={m['MSE']} Grad={m['Grad']} ({sec}s)")

    # 汇总
    ok = [r for r in results if "SAD" in r]
    if ok:
        p3m = [r for r in ok if r["dataset"] == "P3M"]
        hm = [r for r in ok if r["dataset"] == "HM"]
        for label, subset in [("P3M", p3m), ("HM", hm), ("ALL", ok)]:
            if subset:
                avg_sad = round(np.mean([r["SAD"] for r in subset]), 2)
                avg_mse = round(np.mean([r["MSE"] for r in subset]), 4)
                avg_grad = round(np.mean([r["Grad"] for r in subset if r["Grad"] >= 0]), 2)
                print(f"  [{label}] avg SAD={avg_sad} MSE={avg_mse} Grad={avg_grad} n={len(subset)}")
    return results


# ============ 2. 绿幕背景合成测试 ============
def test_greenscreen_bg():
    """绿幕图片(演播室绿幕场景)做背景, 前景人物合成上去
    方式1: T06 mode=greenscreen (只换绿幕区域, 保留桌台话筒)
    方式2: T02 mode=green_key (HSV色度键直合成)
    """
    results = []
    green_screens = sorted((ROOT / "绿幕图片").glob("*.png"))
    # 前景: AI生成的绿幕人物素材
    fgs = sorted((ROOT / "data" / "ai_generated" / "green_fg").glob("*.png"))
    if not fgs:
        print("  [skip] 无前景素材")
        return results
    fg = str(fgs[0])  # fg_01_anchor_male.png

    for i, bg_path in enumerate(green_screens[:3]):
        bg = str(bg_path)
        t0 = time.time()
        ctx = {"run_id": f"greenscreen_{bg_path.stem}"}
        # 方式1: T06 greenscreen — 把 fg 的绿幕区域换成 bg
        r = adapter.call_tool("T06_harmonize", "n1", {
            "fg_path": fg, "alpha_path": fg, "bg_path": bg, "mode": "greenscreen"
        }, quality="draft", ctx=ctx)
        sec = round(time.time() - t0, 2)
        if r["ok"]:
            comp = r["data"].get("composite_path", "")
            results.append({"mode": "greenscreen", "bg": bg_path.name, "fg": Path(fg).name,
                            "output": comp, "sec": sec, "ok": True})
            print(f"  [greenscreen] bg={bg_path.name} fg={Path(fg).name} → {Path(comp).name} ({sec}s)")
        else:
            results.append({"mode": "greenscreen", "bg": bg_path.name, "fg": Path(fg).name,
                            "error": r["error"]["message"][:200], "sec": sec, "ok": False})
            print(f"  [FAIL greenscreen] bg={bg_path.name}: {r['error']['message'][:100]}")

        # 方式2: T02 green_key — 用前景做色度键合成到背景
        t0 = time.time()
        ctx2 = {"run_id": f"greenkey_{bg_path.stem}"}
        r2 = adapter.call_tool("T02_background_generate", "n1", {
            "mode": "green_key", "app_fg": fg, "bg": bg
        }, quality="draft", ctx=ctx2)
        sec2 = round(time.time() - t0, 2)
        if r2["ok"]:
            comp2 = r2["data"].get("composite_path", "")
            results.append({"mode": "green_key", "bg": bg_path.name, "fg": Path(fg).name,
                            "output": comp2, "sec": sec2, "ok": True})
            print(f"  [green_key] bg={bg_path.name} fg={Path(fg).name} → {Path(comp2).name if comp2 else '?'} ({sec2}s)")
        else:
            results.append({"mode": "green_key", "bg": bg_path.name, "fg": Path(fg).name,
                            "error": r2["error"]["message"][:200], "sec": sec2, "ok": False})
            print(f"  [FAIL green_key] bg={bg_path.name}: {r2['error']['message'][:100]}")
    return results


# ============ 3. 滤镜/特效测试 ============
def test_filters_fx():
    """T07_enhance 各 mode"""
    results = []
    test_img = str(ROOT / "data" / "ai_generated" / "green_fg" / "fg_02_anchor_female.png")
    modes = [
        ("depth_blur", "景深虚化"),
        ("sharpen", "锐化"),
        ("grain", "胶片颗粒"),
        ("vignette", "暗角"),
        ("warm", "暖色调"),
        ("cool", "冷色调"),
    ]
    for mode, label in modes:
        t0 = time.time()
        ctx = {"run_id": f"fx_{mode}"}
        r = adapter.call_tool("T07_enhance", "n1",
                              {"image_path": test_img, "mode": mode},
                              quality="draft", ctx=ctx)
        sec = round(time.time() - t0, 2)
        if r["ok"]:
            out = r["data"].get("enhanced_path", "")
            results.append({"mode": mode, "label": label, "output": out, "sec": sec, "ok": True})
            print(f"  [{mode}] {label} → {Path(out).name if out else '?'} ({sec}s)")
        else:
            results.append({"mode": mode, "label": label,
                            "error": r["error"]["message"][:200], "sec": sec, "ok": False})
            print(f"  [FAIL {mode}] {label}: {r['error']['message'][:100]}")

    # 贴纸 + 水印
    for mode, extra, label in [
        ("sticker", {"sticker": "heart"}, "爱心贴纸"),
        ("watermark_text", {"text": "样片"}, "文字水印"),
    ]:
        t0 = time.time()
        ctx = {"run_id": f"fx_{mode}"}
        params = {"image_path": test_img, "mode": mode, **extra}
        r = adapter.call_tool("T07_enhance", "n1", params, quality="draft", ctx=ctx)
        sec = round(time.time() - t0, 2)
        if r["ok"]:
            out = r["data"].get("enhanced_path", "")
            results.append({"mode": mode, "label": label, "output": out, "sec": sec, "ok": True})
            print(f"  [{mode}] {label} → {Path(out).name if out else '?'} ({sec}s)")
        else:
            results.append({"mode": mode, "label": label,
                            "error": r["error"]["message"][:200], "sec": sec, "ok": False})
            print(f"  [FAIL {mode}] {label}: {r['error']['message'][:100]}")
    return results


# ============ 主入口 ============
def main():
    report = {"matting": [], "greenscreen": [], "filters_fx": []}
    print("=" * 60)
    print("1. 抠图精度测试 (P3M-10k + HM-1k)")
    print("=" * 60)
    report["matting"] = test_matting()

    print("\n" + "=" * 60)
    print("2. 绿幕背景合成测试")
    print("=" * 60)
    report["greenscreen"] = test_greenscreen_bg()

    print("\n" + "=" * 60)
    print("3. 滤镜/特效测试")
    print("=" * 60)
    report["filters_fx"] = test_filters_fx()

    out = OUT_DIR / "capability_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {out}")

    # 汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    m_ok = [r for r in report["matting"] if "SAD" in r]
    g_ok = [r for r in report["greenscreen"] if r.get("ok")]
    f_ok = [r for r in report["filters_fx"] if r.get("ok")]
    print(f"  抠图: {len(m_ok)}/{len(report['matting'])} 成功")
    print(f"  绿幕: {len(g_ok)}/{len(report['greenscreen'])} 成功")
    print(f"  滤镜: {len(f_ok)}/{len(report['filters_fx'])} 成功")


if __name__ == "__main__":
    main()
