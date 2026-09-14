# backend/springboot（A6，规划 09-22）

业务 API 层：用户 / 项目 / 任务 / 产物四类接口 + MySQL(Flyway)，向下编排 Agent 服务。

## 现状

- Agent 服务（Python）已提供全部 Agent 能力 REST+WS（见 `docs/api-spec.md`），Spring Boot 是它外面的**业务壳**：登录态、项目归档、任务计费/统计、产物入库索引。
- 本机已有 JDK 17（Temurin 17.0.18），**尚无 Maven**（需要安装 Maven 3.9+ 或用 IDE 内置）后再落代码，避免提交无法构建的工程。

## 规划的模块边界

```text
springboot
├── controller   user / project / task / artifact 四类
├── client       AgentClient：对 agent:8000 的 HTTP 封装（会话/指令/运行/回滚/回放）
├── ws           把 agent 的 /ws/{sid} 事件转发给前端（或前端直连 agent WS，二选一）
├── entity/repository  project / artifact 元数据（产物二进制仍走 artifacts 目录/对象存储）
└── migration    Flyway V1__init.sql
```

## 与 Agent 的职责切分（避免重复造轮子）

| 能力 | 归属 |
|---|---|
| 计划/执行/打分/回滚/回放 | Agent 服务（已有） |
| 上传暂存与产物静态服务 | Agent 服务 `/artifacts`（Spring 只记索引） |
| 用户/项目/权限/归档/统计 | Spring Boot |
