"""多模态图像合成 Agent —— A 组核心包。

分层（规范 N2，Agent 与模型解耦）：
    Agent(本包) -> Tool Interface(HTTP 契约, agent/schema) -> ai-service(模型实现)

各子模块：
    schema    工具接口四件套（T01–T08 JSON Schema + 校验器）
    dag       执行计划 DAG 数据模型与校验
    planner   指令 -> DAG（LLM 可插拔 + 规则模板兜底）
    executor  DAG 执行引擎（Run/Pause/Resume/Retry/Skip/Re-run）
    critic    五维评分 + 自动重规划
    rollback  节点级版本树 + 条件回滚
    eval      运行记录可回放（A8）
    server    FastAPI 服务（REST + WebSocket）
    tools     工具调用 Provider（HTTP / 内嵌）
"""

__version__ = "0.1.0"
