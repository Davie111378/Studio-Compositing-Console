# -*- coding: utf-8 -*-
"""上传剩余 ImageStage 资源（绕过系统代理）。"""
import os
import time
import requests

M = r"G:\myself\作业\生产实习\imagecompose-site\media"

def U(key, sig):
    return ("https://design-tmp-1258344699.cos.ap-guangzhou.myqcloud.com/tmp/presigned/"
            "718711599973019/" + key +
            "?sign=q-sign-algorithm%3Dsha1%26q-ak%3DAKIDVsHXijW5PUKygAPQC9RZT9ZkhPsDKCoA"
            "%26q-sign-time%3D1789134272%3B1789135232"
            "%26q-key-time%3D1789134272%3B1789135232"
            "%26q-header-list%3Dhost%26q-url-param-list%3D%26q-signature%3D" + sig)

jobs = [
    ("subject_transparent.png", U("01M28BERGV4DF5CNK4F9GX1R7J", "eb0def03a447aa921c4e01e7b135fcb216766945"), "image/png"),
    ("scene_lakeside.jpg",      U("01M28BERH78S2X7G3MSZZ4FPEM", "e7d7a1fe7878bea2970c00a392dd2a10d8c76600"), "image/jpeg"),
    ("stage_compose.jpg",       U("01M28BERHC49K9GDRR97Y93V16", "62743c35d6f1b0b9d43c2af1a8337cd38b0d0981"), "image/jpeg"),
]

s = requests.Session()
s.trust_env = False

for name, url, ct in jobs:
    data = open(os.path.join(M, name), "rb").read()
    for attempt in range(3):
        try:
            r = s.put(url, data=data, headers={"Content-Type": ct}, timeout=120)
            print(name, attempt, "->", r.status_code, r.text[:100].replace("\n", " "))
            if r.status_code in (200, 201):
                break
        except Exception as e:
            print(name, attempt, "ERR", type(e).__name__, str(e)[:100])
        time.sleep(2)
