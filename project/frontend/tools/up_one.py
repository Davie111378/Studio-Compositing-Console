# -*- coding: utf-8 -*-
"""单文件上传：python up_one.py <name>"""
import sys
from pathlib import Path
import requests

M = Path(r"G:\myself\作业\生产实习\imagecompose-site\media").resolve()


def _safe_in(base: Path, name: str) -> Path:
    """路径校验：解析后必须仍位于基准目录内（防路径穿越，禁止 ../ 逃逸）。"""
    p = (base / name).resolve()
    if p != base and base not in p.parents:
        raise SystemExit(f"路径越界: {p}")
    return p


BASE = "https://design-tmp-1258344699.cos.ap-guangzhou.myqcloud.com/tmp/presigned/718711599973019/"
Q = ("?sign=q-sign-algorithm%3Dsha1%26q-ak%3DAKIDVsHXijW5PUKygAPQC9RZT9ZkhPsDKCoA"
     "%26q-sign-time%3D{t}%26q-key-time%3D{t}"
     "%26q-header-list%3Dhost%26q-url-param-list%3D%26q-signature%3D{s}")

# name: (objectKey, uploadSig, signTime, contentType, localFile)
J = {
    "source_original": ("01M28DGR9X8EXB4J1T0GVA0K3N", "ab8a2bd890cc6eb40c5d39cca5d75f3978cda5ed", "1789136434;1789137394", "image/jpeg", "source_original.jpg"),
    "stage_compose":   ("01M28DH1T8KR0X96G2JVESE978", "0d531159d37341600e97e287bc7226c7a7ed36c7", "1789136444;1789137404", "image/jpeg", "stage_compose.jpg"),
    "detail_feet":     ("01M28DHBWRPG2XHDJ80FNE94JP", "b39016e33b8cbaebb1e8018ba611fe9a1220036a", "1789136454;1789137414", "image/jpeg", "detail_feet.jpg"),
    "stage_final":     ("01M28DHP2898ZP2AC7STXAJD0X", "d2c86f9ad37a896129e7c2c7d646ff6441a949f9", "1789136465;1789137425", "image/jpeg", "stage_final.jpg"),
    "macro_cloud":     ("01M28DHZZ4GGCFJN1QA83S92YA", "aab0b7aded003e427b1d979fe7d3a5c64c996dbd", "1789136475;1789137435", "image/jpeg", "macro_cloud.jpg"),
    "macro_water":     ("01M28DJAK7CMGB2449GQSCDV7G", "c295954e0ad144ba45fe97ddd5974f0cf34b0996", "1789136486;1789137446", "image/jpeg", "macro_water.jpg"),
    "macro_hair":      ("01M28DJPC52KJSCY6EQ62GB1V6", "198e351fa62ea10d7dce99f1e668afbf4ca29b75", "1789136498;1789137458", "image/jpeg", "macro_hair.jpg"),
    "subject_alpha":   ("01M28DK1T9SYV98TCKTNW08MG7", "588cbe1184a231421a9fc9f70d77b7bec0b0094f", "1789136510;1789137470", "image/png",  "subject_transparent.png"),
    "stage_relight":   ("01M28DFY88ZVW2KENKX68NYZNP", "b5151110ed7fc74196de8282d654f666a5cb50ed", "1789136408;1789137368", "image/jpeg", "stage_relight.jpg"),
}

name = sys.argv[1]
key, sig, t, ct, fn = J[name]
url = BASE + key + Q.format(t=t, s=sig)
data = _safe_in(M, fn).read_bytes()
s = requests.Session()
s.trust_env = False
try:
    r = s.put(url, data=data, headers={"Content-Type": ct}, timeout=120)
    print(name, r.status_code)
except Exception as e:
    print(name, "ERR", type(e).__name__, str(e)[:100])
finally:
    s.close()
