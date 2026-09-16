# -*- coding: utf-8 -*-
"""上传 v2 素材，并把 downloadUrl 清单写盘备用。"""
import json
import time
from pathlib import Path
import requests

M = Path(r"G:\myself\作业\生产实习\imagecompose-site\media").resolve()
TOOLS = Path(r"G:\myself\作业\生产实习\imagecompose-site\tools").resolve()


def _safe_out(base: Path, name: str) -> Path:
    """路径校验：解析后必须仍位于基准目录内（防路径穿越，禁止 ../ 逃逸）。"""
    p = (base / name).resolve()
    if p != base and base not in p.parents:
        raise SystemExit(f"路径越界: {p}")
    return p


BASE = "https://design-tmp-1258344699.cos.ap-guangzhou.myqcloud.com/tmp/presigned/718711599973019/"
COMMON = ("?sign=q-sign-algorithm%3Dsha1%26q-ak%3DAKIDVsHXijW5PUKygAPQC9RZT9ZkhPsDKCoA"
          "%26q-sign-time%3D{st}%26q-key-time%3D{kt}"
          "%26q-header-list%3Dhost%26q-url-param-list%3D%26q-signature%3D{sig}")

# key, 上传签名, 下载签名, 生效时间戳(起;止), content-type, 本地文件
JOBS = [
    ("source_original",  "01M28CNWP6ZW00N2Z9MSZCSC9G", "229ec99c2d427e4bf94ac367d498a14b5f2d44e9", "cca28ec311875be7bc63a8ac5d5395f5b488f08f", "1789135554;1789136514", "image/jpeg", "source_original.jpg"),
    ("stage_compose",    "01M28CP67Z0EB5BCGH9F7CER91", "b1ff060237418aeb6997700a1d751a0b46b22ff0", "af8977af066cd8345e18f63986ff5edd26e7c5e8", "1789135564;1789136524", "image/jpeg", "stage_compose.jpg"),
    ("stage_relight",    "01M28CPG6QH5DPZFTZ960SC5QP", "16e45bd21b636de304a0b9799a06a65e25a7ac7f", "23a059d70657046a5d6108b3e3d18ba5883181ea", "1789135574;1789136534", "image/jpeg", "stage_relight.jpg"),
    ("detail_feet",      "01M28CPYAW9DPSDSSSTB1RM14S", "06d9fac256d6c185fafd555213d6eadbea1ab384", "8c34fa779eed15116dde990ddb070a25d47e70ef", "1789135589;1789136549", "image/jpeg", "detail_feet.jpg"),
    ("stage_final",      "01M28CQ7175D7JFQ08GGFF8R4Z", "869217f3bca4d9f07c40ec63cadb17540da2ff6c", "c5cd9f56a5a8e2312fdff0e4eb5d3a688b0b2918", "1789135598;1789136558", "image/jpeg", "stage_final.jpg"),
    ("macro_cloud",      "01M28CQGG8QPP99RTW65YKKET8", "baa437ee80c81c597bcd3fef9cf6ae434f4058bd", "9342d930e79bfb6afafa70615cd1715d53c273f5", "1789135607;1789136567", "image/jpeg", "macro_cloud.jpg"),
    ("macro_water",      "01M28CQSQYDXRKAC6SGCP9Z1GP", "06220ad1c5235766fd7f916aad0f92201076d0af", "6df44213c72b999169efa4bca695d2129b6cb008", "1789135617;1789136577", "image/jpeg", "macro_water.jpg"),
    ("macro_hair",       "01M28CR5NFR7Z4RNJ9WG1X641W", "a1d5d0803b08281716a001ff40ca6324a7913c60", "e65b6f262a9ef64d50af9255fdeb8211535328ff", "1789135629;1789136589", "image/jpeg", "macro_hair.jpg"),
    ("subject_alpha",    "01M28CRFPSDYRG3J9CHE4V9XC1", "424b797b6d114933fbe4d7806f4b44bdd069a80a", "59aab89858be0049c4d95d66bb59370ac6ca1277", "1789135639;1789136599", "image/png",  "subject_transparent.png"),
]

s = requests.Session()
s.trust_env = False
down = {}

for name, key, usig, dsig, ts, ct, fn in JOBS:
    st, kt = ts.split(";")
    put_url = BASE + key + COMMON.format(st=st, kt=kt, sig=usig)
    get_url = BASE + key + COMMON.format(st=st, kt=kt, sig=dsig)
    data = _safe_out(M, fn).read_bytes()
    ok = False
    for attempt in range(3):
        try:
            s = requests.Session()          # 每个文件一个全新连接，复用连接会被拒
            s.trust_env = False
            r = s.put(put_url, data=data, headers={"Content-Type": ct}, timeout=120)
            s.close()
            if r.status_code in (200, 201):
                ok = True
                break
            print(name, attempt, r.status_code, r.text[:200].replace("\n", " "))
        except Exception as e:
            print(name, attempt, "ERR", type(e).__name__, str(e)[:90])
        time.sleep(3)
    print(("OK  " if ok else "FAIL"), name, fn)
    down[name] = get_url
    time.sleep(4)

out_json = _safe_out(TOOLS, "asset_urls_v2.json")
out_json.write_text(json.dumps(down, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote asset_urls_v2.json")
