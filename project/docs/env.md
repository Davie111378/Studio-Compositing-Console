# 环境与依赖（P2 冻结底稿）

> 冻结时间：2026-09-10（A 组环境）。数据冻结由 B 组在 `data/README.md` 另行记录。

## 已验证环境

| 项 | 版本 | 备注 |
|---|---|---|
| OS（开发机） | Windows 10 x64 | 仓库路径避免中文/空格跑 Docker（规范 6.3） |
| Python | 3.12.10 | 3.11+ 均可 |
| fastapi | 0.141.1 | 含 starlette 1.6.0 |
| pydantic | 2.13.4 | |
| uvicorn | 0.52.1 | |
| httpx | 0.28.1 | 含 ASGITransport（内嵌模式依赖） |
| pillow | 12.2.0 | mock 工具与上传校验 |
| python-multipart | 已装 | 上传接口 |
| pytest | 9.1.1 | 37 条用例 |
| Docker | 29.7.2 | compose 部署用 |
| Java | Temurin 17.0.18 | Spring Boot（A6）已装 JRE/JDK，**缺 Maven** |

## 锁版本

首次 Docker 构建后执行 `pip freeze > requirements-lock.txt` 并提交（规范 6.3：租卡镜像锁死版本）。

## 已知环境坑（对应规范 6.3）

- Windows 控制台 GBK：服务日志统一 ASCII/UTF-8，勿在日志里打 emoji；
- 中文路径：Python 直跑兼容，Docker 挂载与 Linux 训练机会乱码，仓库迁移到英文路径后再跑容器；
- 本项目内嵌模式依赖 httpx ASGITransport，升级 httpx 时回归 `pytest tests/`。
