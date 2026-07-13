import time
from typing import List
from openai import OpenAI

from app.core.config import LLM_API_KEY, LLM_API_BASE, LLM_MODEL
from app.core.prompts import SYSTEM_PROMPT
from app.core.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class LLMHandler:
    """AI 大脑 - DeepSeek API 调用，支持自动重试"""

    @staticmethod
    def generate(question: str, sources: List[dict]) -> dict:
        """让 AI 根据资料回答问题，失败时自动重试"""
        if not LLM_API_KEY:
            return LLMHandler._local_mode(question, sources)

        context_parts = [f"【资料{i+1}】\n{src['content']}" for i, src in enumerate(sources)]
        context = "\n\n---\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"参考资料：\n\n{context}\n\n问题：{question}"},
        ]

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
                start = time.time()

                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=1000,
                    timeout=30,
                )

                latency = round((time.time() - start) * 1000, 0)
                usage = response.usage

                return {
                    "answer": response.choices[0].message.content,
                    "model_used": LLM_MODEL,
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "latency_ms": latency,
                    "error": None,
                }

            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM 调用失败 (第{attempt}/{MAX_RETRIES}次): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)

        logger.error(f"LLM 调用全部失败: {last_error}")
        return {
            "answer": f"AI服务暂时不可用，请稍后再试。({last_error[:80]})",
            "model_used": "error",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0,
            "error": last_error,
        }

    @staticmethod
    def _local_mode(question: str, sources: List[dict]) -> dict:
        """离线模式 - 整理搜索到的资料"""
        if not sources:
            return {
                "answer": "知识库中没有找到相关信息。请先上传一些文档。",
                "model_used": "本地搜索",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": 0,
                "error": None,
            }

        answer = f"找到 {len(sources)} 条相关资料：\n\n"
        answer += "─" * 40 + "\n\n"
        for i, src in enumerate(sources, 1):
            match_pct = int(src.get("score", 0) * 100)
            stars = "=" * min(5, max(1, match_pct // 20))
            answer += f"第{i}条 (相关度: {stars} {match_pct}%)\n{src['content'][:400]}\n\n"
            answer += "─" * 40 + "\n\n"

        answer += "提示: 设置 LLM_API_KEY 可使用 DeepSeek AI 智能回答。"
        return {
            "answer": answer,
            "model_used": "本地搜索",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0,
            "error": None,
        }


llm_handler = LLMHandler()
