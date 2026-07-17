# -*- coding: utf-8 -*-
"""
prompt_builder.py
构造喂给生成器的最终 prompt。提示词做版本管理，写进 trace 便于复现。

grounded_v1: 要求只依据给定 context 作答，不引入无依据事实，
             context 不足时明确说明"无法从上下文确定"。
"""

GROUNDED_V1 = """You are a question-answering assistant.

Answer the question using only the supplied context.
Do not introduce unsupported facts.
If the context is insufficient, explicitly state that the answer
cannot be determined from the provided context.

Question:
{query}

Context:
{final_context}

Answer:"""

PROMPTS = {
    "grounded_v1": GROUNDED_V1,
}


def build_prompt(query: str, final_context: str, prompt_version: str = "grounded_v1") -> str:
    if prompt_version not in PROMPTS:
        raise ValueError(f"未知 prompt 版本: {prompt_version}")
    return PROMPTS[prompt_version].format(query=query or "", final_context=final_context or "")
