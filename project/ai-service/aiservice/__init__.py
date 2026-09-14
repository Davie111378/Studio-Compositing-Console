"""aiservice —— 图像工具服务包（ai-service 的可导入实现层）。

Agent 通过 HTTP 契约调用本包（embedded 模式下走 ASGI 内嵌，同样只过 HTTP 信封）。
B 组替换真实模型时：只改各 impl 的内部实现，函数签名与信封不变（规范 N2）。
"""

__version__ = "0.1.0"
