# -*- coding: utf-8 -*-
"""
演播室图像合成 · 交互式 GUI (tkinter, 零新增依赖)
功能: 选择绿幕图片 -> (圈选) -> 抠图 / 背景叠加 / 添加特效

运行: python app/studio_gui.py
复用后端: app/studio_cli.task_matting / task_composite / task_fx (与 CLI 同一引擎)
"""
from __future__ import annotations
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent.parent  # project root (app/ -> root)
sys.path.insert(0, str(ROOT / "app"))
import studio_cli  # noqa: E402  (复用任务函数)

OUT_ROOT = ROOT / "outputs" / "gui"
EFFECTS = ["spotlight", "bokeh", "fog", "vignette", "color_temp", "depth_blur"]
FX_CN = {"spotlight": "聚光灯", "bokeh": "光斑散景", "fog": "舞台雾效",
         "vignette": "暗角", "color_temp": "色温滤镜", "depth_blur": "景深虚化"}


def checkerboard(w: int, h: int, cell: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    c = (((yy // cell) + (xx // cell)) % 2).astype(np.uint8)
    g = (c * 40 + 160).astype(np.uint8)
    return np.stack([g] * 3, -1)


class StudioGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("演播室图像合成工作台 · B 组")
        root.geometry("1180x760")
        self.img: np.ndarray | None = None        # 当前工作图 (uint8 RGB)
        self.alpha: np.ndarray | None = None
        self.bg_path: str | None = None
        self.scale = 1.0
        self.disp_ids = []
        self.rect_id = None
        self.rect_start = None
        self.sel_rect = None                       # 原图坐标 (x1,y1,x2,y2)

        # ---- 左侧控制面板 ----
        panel = ttk.Frame(root, padding=8)
        panel.pack(side="left", fill="y")
        ttk.Label(panel, text="① 选图", font=("微软雅黑", 11, "bold")).pack(anchor="w")
        ttk.Button(panel, text="打开绿幕图片…", command=self.open_green).pack(fill="x", pady=2)
        self.engine = tk.StringVar(value="hybrid")
        ttk.Combobox(panel, textvariable=self.engine, state="readonly",
                     values=["hybrid", "chroma", "birefnet", "refiner"], width=18).pack(fill="x")

        ttk.Separator(panel).pack(fill="x", pady=6)
        ttk.Label(panel, text="② 抠图", font=("微软雅黑", 11, "bold")).pack(anchor="w")
        self.sel_mode = tk.StringVar(value="off")
        ttk.Checkbutton(panel, text="圈选模式 (画框后可 GrabCut 圈选抠图)",
                        variable=self.sel_mode, onvalue="matting", offvalue="off").pack(anchor="w")
        ttk.Button(panel, text="自动抠图 (整图)", command=lambda: self.run(self.do_matting)).pack(fill="x", pady=2)
        ttk.Button(panel, text="按圈选 GrabCut 抠图", command=lambda: self.run(self.do_grabcut)).pack(fill="x", pady=2)

        ttk.Separator(panel).pack(fill="x", pady=6)
        ttk.Label(panel, text="③ 背景叠加", font=("微软雅黑", 11, "bold")).pack(anchor="w")
        ttk.Button(panel, text="选择背景…", command=self.pick_bg).pack(fill="x", pady=2)
        self.bg_label = ttk.Label(panel, text="(未选择)", foreground="#888")
        self.bg_label.pack(anchor="w")
        self.opt_harm = tk.BooleanVar(value=True)
        self.opt_light = tk.BooleanVar(value=True)
        self.opt_shadow = tk.BooleanVar(value=True)
        for txt, var in (("和谐化 (防压黑)", self.opt_harm), ("重打光 (方向光)", self.opt_light),
                         ("接触阴影", self.opt_shadow)):
            ttk.Checkbutton(panel, text=txt, variable=var).pack(anchor="w")
        ttk.Button(panel, text="背景叠加", command=lambda: self.run(self.do_composite)).pack(fill="x", pady=2)

        ttk.Separator(panel).pack(fill="x", pady=6)
        ttk.Label(panel, text="④ 添加特效", font=("微软雅黑", 11, "bold")).pack(anchor="w")
        ttk.Label(panel, text="圈选模式 (画框=特效只作用该区域):").pack(anchor="w")
        self.fx_region_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(panel, text="启用特效圈选", variable=self.fx_region_mode).pack(anchor="w")
        self.fx_name = tk.StringVar(value="spotlight")
        ttk.Combobox(panel, textvariable=self.fx_name, state="readonly",
                     values=[f"{FX_CN[k]} ({k})" for k in EFFECTS], width=20).pack(fill="x", pady=2)
        ttk.Label(panel, text="强度:").pack(anchor="w")
        self.fx_intensity = tk.DoubleVar(value=0.6)
        ttk.Scale(panel, from_=0.1, to=1.0, variable=self.fx_intensity,
                  orient="horizontal").pack(fill="x")
        ttk.Button(panel, text="添加特效", command=lambda: self.run(self.do_fx)).pack(fill="x", pady=2)
        ttk.Button(panel, text="↩ 撤销 (回到上一步)", command=self.undo).pack(fill="x", pady=2)

        ttk.Separator(panel).pack(fill="x", pady=6)
        ttk.Button(panel, text="保存结果…", command=self.save).pack(fill="x", pady=2)

        # ---- 右侧画布 ----
        self.canvas = tk.Canvas(root, bg="#3a3a3a", width=820, height=720,
                                highlightthickness=0)
        self.canvas.pack(side="right", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # ---- 状态栏 ----
        self.status = tk.StringVar(value="就绪。第一步: 打开一张绿幕图片。")
        ttk.Label(root, textvariable=self.status, relief="sunken",
                  anchor="w", padding=(6, 3)).pack(side="bottom", fill="x")
        self.history: list[np.ndarray] = []

    # ---------- 工具 ----------
    def log(self, msg: str):
        self.status.set(msg)
        self.root.update_idletasks()

    def show(self, arr: np.ndarray | None):
        if arr is None:
            return
        self.img = arr
        cw = max(self.canvas.winfo_width(), 820)
        ch = max(self.canvas.winfo_height(), 720)
        h, w = arr.shape[:2]
        self.scale = min((cw - 24) / w, (ch - 24) / h, 1.0)
        dw, dh = int(w * self.scale), int(h * self.scale)
        im = Image.fromarray(arr).resize((dw, dh), Image.BILINEAR)
        self.canvas.delete("all")
        self.disp_ids = [self.canvas.create_image(12, 12, anchor="nw", image=None)]
        self._tkimg = ImageTk.PhotoImage(im)
        self.disp_ids = [self.canvas.create_image(12, 12, anchor="nw", image=self._tkimg)]
        self.sel_rect = None

    def run(self, fn):
        """后台线程执行, 防 UI 卡死。"""
        self.log("处理中, 请稍候…")
        self.root.config(cursor="watch")
        t = threading.Thread(target=self._worker, args=(fn,), daemon=True)
        t.start()

    def _worker(self, fn):
        try:
            fn()
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", str(e)))
        finally:
            self.root.after(0, lambda: (self.root.config(cursor=""),
                                        self.log("完成。")))

    # ---------- ①② 打开 / 抠图 ----------
    def open_green(self):
        p = filedialog.askopenfilename(title="选择绿幕图片",
                                       filetypes=[("图片", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not p:
            return
        from PIL import Image as _I
        arr = np.array(_I.open(p).convert("RGB"))
        if arr.shape[0] > 1600:      # 大图先降档, 保交互流畅
            s = 1600 / arr.shape[0]
            arr = np.array(_I.fromarray(arr).resize((int(arr.shape[1] * s), 1600)))
        self.history = []
        self.show(arr)
        self.log(f"已打开 {Path(p).name} ({arr.shape[1]}x{arr.shape[0]})。可自动抠图, 或开圈选模式画框。")

    def do_matting(self):
        if self.img is None:
            return
        tmp = OUT_ROOT / "session"
        tmp.mkdir(parents=True, exist_ok=True)
        src = tmp / "input.png"
        Image.fromarray(self.img).save(src)
        r = studio_cli.task_matting(str(src), str(tmp), engine=self.engine.get())
        self.alpha = np.array(Image.open(r["alpha"]).convert("L")).astype(np.float32) / 255.0
        fg = np.array(Image.open(r["fg"]).convert("RGB"))
        board = checkerboard(*fg.shape[:2][::-1])[::-1]
        a = self.alpha[..., None]
        vis = (fg * a + board * (1 - a)).astype(np.uint8)
        self.history.append(self.img)
        self.show(vis)
        self.log(f"抠图完成 (engine={r['engine']}, 前景占比 {r['fg_ratio']*100:.0f}%)。下一步: 选背景叠加, 或直接加特效。")

    def do_grabcut(self):
        if self.img is None or not self.sel_rect:
            messagebox.showinfo("提示", "请先勾选『圈选模式』并在图上拖拽画框。")
            return
        from matting.interactive_matting import grabcut_matte
        x1, y1, x2, y2 = [int(v) for v in self.sel_rect]
        prompt = {"type": "box", "xyxy": [x1, y1, x2, y2]}
        alpha = grabcut_matte(self.img, prompt)
        self.alpha = alpha
        board = checkerboard(*self.img.shape[:2][::-1])[::-1]
        vis = (self.img * alpha[..., None] + board * (1 - alpha[..., None])).astype(np.uint8)
        self.history.append(self.img)
        self.show(vis)
        self.log("圈选 GrabCut 抠图完成。")

    # ---------- ③ 背景 ----------
    def pick_bg(self):
        p = filedialog.askopenfilename(title="选择背景图片",
                                       filetypes=[("图片", "*.png;*.jpg;*.jpeg;*.bmp")])
        if p:
            self.bg_path = p
            self.bg_label.config(text=Path(p).name)

    def do_composite(self):
        if self.img is None:
            return
        if not self.bg_path:
            messagebox.showinfo("提示", "请先选择背景图片。")
            return
        tmp = OUT_ROOT / "session"
        tmp.mkdir(parents=True, exist_ok=True)
        src = tmp / "input.png"
        Image.fromarray(self.img).save(src)
        alpha_p = None
        if self.alpha is not None:
            alpha_p = str(tmp / "input_alpha.png")
            Image.fromarray((self.alpha * 255).astype(np.uint8)).save(alpha_p)
        r = studio_cli.task_composite(src, self.bg_path, str(tmp / "composite"),
                                      alpha=alpha_p, harmonize=self.opt_harm.get(),
                                      relight=self.opt_light.get(), shadow=self.opt_shadow.get())
        self.history.append(self.img)
        self.show(np.array(Image.open(r["final"]).convert("RGB")))
        self.log(f"背景叠加完成 ({Path(self.bg_path).name}, {r['total_ms']}ms)。可继续加特效。")

    # ---------- ④ 特效 ----------
    def do_fx(self):
        if self.img is None:
            return
        raw = self.fx_name.get()
        effect = raw.split("(")[-1].rstrip(")").strip() if "(" in raw else raw
        region = None
        if self.fx_region_mode.get():
            if not self.sel_rect:
                messagebox.showinfo("提示", "已勾选特效圈选, 请先在图上拖拽画框 (或取消勾选=全图)。")
                return
            x1, y1, x2, y2 = self.sel_rect
            region = {"type": "box", "xyxy": [float(x1), float(y1), float(x2), float(y2)]}
        out = OUT_ROOT / "fx"
        r = studio_cli.task_fx_pil(self.img, str(out), effect,
                                   float(self.fx_intensity.get()), region)
        self.history.append(self.img)
        self.show(r["image"])
        self.log(f"特效 {effect} 已添加 ({'区域' if region else '全图'})。可叠加下一个特效或保存。")

    def undo(self):
        if self.history:
            self.show(self.history.pop())
            self.log("已撤销。")
        else:
            self.log("没有可撤销的步骤。")

    def save(self):
        if self.img is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         initialfile="result.png",
                                         filetypes=[("PNG", "*.png")])
        if p:
            Image.fromarray(self.img).save(p)
            self.log(f"已保存: {p}")

    # ---------- 鼠标圈选 ----------
    def _to_img(self, cx, cy):
        return (int((cx - 12) / self.scale), int((cy - 12) / self.scale))

    def on_press(self, e):
        self.rect_start = (e.x, e.y)

    def on_drag(self, e):
        if not self.rect_start:
            return
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.rect_start[0], self.rect_start[1], e.x, e.y,
            outline="#ff4444", width=2, dash=(6, 4))

    def on_release(self, e):
        if not self.rect_start:
            return
        x1, y1 = self._to_img(*self.rect_start)
        x2, y2 = self._to_img(e.x, e.y)
        if abs(x2 - x1) > 8 and abs(y2 - y1) > 8:
            self.sel_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            mode = "特效区域" if self.fx_region_mode.get() else (
                "GrabCut 圈选" if self.sel_mode.get() == "matting" else "圈选(未启用模式)")
            self.log(f"圈选完成: {self.sel_rect}  (当前模式: {mode})")
        else:
            self.sel_rect = None
        self.rect_start = None


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except Exception:
        pass
    StudioGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
