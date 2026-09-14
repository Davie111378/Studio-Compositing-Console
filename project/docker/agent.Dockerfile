# Agent 服务镜像：包含 agent 包 + ai-service（内嵌兜底用）
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY agent /app/agent
COPY ai-service /app/ai-service
COPY scripts /app/scripts

ENV PYTHONPATH=/app:/app/ai-service \
    ARTIFACTS_DIR=/workspace/artifacts \
    RUNS_DIR=/workspace/runs \
    PYTHONUTF8=1

EXPOSE 8000
CMD ["python", "scripts/run_agent.py"]
