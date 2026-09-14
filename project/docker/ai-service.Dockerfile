# ai-service 镜像：B 组在此镜像基础上追加 torch/diffusers 等重依赖
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ai-service /app/ai-service

ENV PYTHONPATH=/app \
    ARTIFACTS_DIR=/workspace/artifacts \
    PYTHONUTF8=1

# 真模型接入时在这里追加：RUN pip install torch --index-url https://download.pytorch.org/whl/cu121 等
# 并把权重通过卷挂载到 /models，从 model_registry.yaml 读取路径

EXPOSE 8100
CMD ["python", "ai-service/run.py"]
