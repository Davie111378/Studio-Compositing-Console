# -*- coding: utf-8 -*-
import pathlib, subprocess, tempfile, os

html = r"d:\AIcode\生产实习\docs\专用Agent系统架构与工程实现指南_论文化版.html"
pdf = r"d:\AIcode\生产实习\docs\专用Agent系统架构与工程实现指南_论文化版.pdf"
url = pathlib.Path(html).as_uri()
chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
profile = os.path.join(tempfile.gettempdir(), "chrome_agent_print3")
cmd = [
    chrome, "--headless=new", "--disable-gpu",
    f"--user-data-dir={profile}",
    f"--print-to-pdf={pdf}",
    "--no-pdf-header-footer", "--print-to-pdf-no-header",
    url,
]
print("URL:", url)
r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
print("exit:", r.returncode)
print((r.stdout or "")[-1500:])
print((r.stderr or "")[-800:])
print("pdf exists:", os.path.exists(pdf), "| size:", os.path.getsize(pdf) if os.path.exists(pdf) else 0)