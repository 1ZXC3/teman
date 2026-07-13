import time
from typing import List
from openai import OpenAI
from app.core.config import LLM_API_KEY, LLM_API_BASE, LLM_MODEL


# 回答问题的提示词模板
SYSTEM_PROMPT = """你是一个超棒的知识助手！像个耐心的老师一样回答用户的问题。

📚 规则：
1. 优先使用【参考资料】里的内容来回答
2. 回答要简单易懂，就像给小学生解释一样
3. 如果你用了某条资料，在回答后注明"📖 参考来源"
4. 如果资料里找不到答案，诚实地说"这个问题我暂时还不会，可以上传包含这些知识的文档来教我哦！"
5. 用可爱的语气回答，适当使用 emoji ✨"""


class LLMHandler:
    """AI 大脑 - 根据找到的资料生成回答"""

    @staticmethod
    def generate(question: str, sources: List[dict]) -> dict:
        """
        让 AI 根据资料回答问题
        返回: {answer, model_used, tokens, latency_ms, error}
        """
        # 如果没有 API Key，用本地模式
        if not LLM_API_KEY or LLM_API_KEY == "your-api-key-here":
            return LLMHandler._local_mode(question, sources)

        # 构建上下文
        context_parts = []
        for i, src in enumerate(sources, 1):
            context_parts.append(f"【资料{i}】\n{src['content']}")
        context = "\n\n---\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"参考资料：\n\n{context}\n\n❓ 用户的问题：{question}"},
        ]

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
            return {
                "answer": f"😢 抱歉，AI 服务出了点问题：{e}",
                "model_used": "error",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": 0,
                "error": str(e),
            }

    @staticmethod
    def _local_mode(question: str, sources: List[dict]) -> dict:
        """
        离线模式 - 不需要 API Key
        直接用搜索到的资料拼接回答
        """
        if not sources:
            return {
                "answer": "❓ 我在知识库里没有找到相关信息。\n\n💡 建议：先上传一些文档，我就可以帮你找答案啦！",
                "model_used": "本地搜索模式",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": 0,
                "error": None,
            }

        # 把搜到的资料整理成回答
        answer = f"🔍 我找到了 {len(sources)} 条相关资料，整理如下：\n\n"
        answer += "─" * 40 + "\n\n"

        for i, src in enumerate(sources, 1):
            match_percent = int(src.get("score", 0) * 100)
            stars = "⭐" * min(5, max(1, match_percent // 20))
            answer += f"📄 **第{i}条** (相关度: {stars} {match_percent}%)\n\n"
            answer += f"{src['content'][:400]}\n\n"
            answer += "─" * 40 + "\n\n"

        answer += (
            "\n💡 **提示**: 这是本地搜索模式的结果。\n"
            "如果要获得 AI 总结后的答案，请在 `config.py` 中填入你的 API Key~"
        )
        return {
            "answer": answer,
            "model_used": "本地搜索模式",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0,
            "error": None,
        }


llm_handler = LLMHandler()
