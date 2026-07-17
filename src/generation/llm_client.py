# -*- coding: utf-8 -*-
"""
llm_client.py
生成器的薄封装，backend 可切换。目前实现:
  - GeminiClient : 调用 Google Gemini（google-genai SDK）
  - DryRunClient : 不调 API，返回占位答案，用于离线打通完整 trace 管道

统一接口: client.generate(prompt) -> dict，包含
  final_answer, model_name, temperature, prompt_version(留空由上层填),
  latency_ms, input_tokens, output_tokens, status, error

依赖(仅 Gemini 需要): pip install google-genai
API key: 环境变量 GEMINI_API_KEY 或 GOOGLE_API_KEY，或构造时传入。
"""
import os
import time
from typing import Dict, Optional


class DryRunClient:
    """离线占位客户端：不调用任何 API，用于验证管道与 trace 结构。"""
    method = "dry_run"

    def __init__(self, model_name: str = "dry-run", **kwargs):
        self.model_name = model_name
        self.temperature = kwargs.get("temperature", 0)

    def generate(self, prompt: str) -> Dict:
        return {
            "final_answer": "[DRY_RUN] 占位答案，未调用真实模型。",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "latency_ms": 0,
            "input_tokens": None,
            "output_tokens": None,
            "status": "success",
            "error": None,
        }


class GeminiClient:
    method = "gemini"

    def __init__(self,
                 model_name: str = "gemini-3.5-flash",
                 api_key: Optional[str] = None,
                 temperature: float = 0.0,
                 max_output_tokens: int = 256,
                 max_retries: int = 3,
                 timeout_s: int = 60):
        from google import genai  # 延迟导入，dry-run 时无需安装
        self.genai = genai
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("未找到 API key：请设置 GEMINI_API_KEY 环境变量或传入 api_key")
        self.client = genai.Client(api_key=key)

    def generate(self, prompt: str) -> Dict:
        from google.genai import types
        cfg = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        last_err = None
        t0 = time.time()
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name, contents=prompt, config=cfg)
                latency_ms = int((time.time() - t0) * 1000)
                usage = getattr(resp, "usage_metadata", None)
                return {
                    "final_answer": (resp.text or "").strip(),
                    "model_name": self.model_name,
                    "temperature": self.temperature,
                    "latency_ms": latency_ms,
                    "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
                    "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
                    "status": "success",
                    "error": None,
                }
            except Exception as e:  # 网络/限流/超时等，指数退避重试
                last_err = str(e)
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        # 重试仍失败：返回错误占位，不让整批中断
        return {
            "final_answer": "",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "latency_ms": int((time.time() - t0) * 1000),
            "input_tokens": None,
            "output_tokens": None,
            "status": "error",
            "error": last_err,
        }


def build_client(backend: str, **kwargs):
    if backend == "gemini":
        return GeminiClient(**kwargs)
    if backend == "dry-run":
        return DryRunClient(**kwargs)
    raise ValueError(f"unknown backend: {backend}")
