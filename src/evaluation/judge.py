"""LLM-as-a-Judge evaluation for RAG quality assessment.

Scores a RAG response across three dimensions (0–5 each):
- Context Relevance: how relevant the retrieved chunks are to the query.
- Faithfulness: how grounded the answer is in the retrieved context.
- Answer Relevance: how directly and helpfully the answer addresses the query.
"""
import json
import logging
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_llm

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """### ROLE
You are a strict RAG evaluation judge. Score a RAG response across three dimensions.

### SCORING RUBRIC

**Context Relevance (0–5)** — How relevant is the retrieved context to the user's query?
  0 – Completely irrelevant.  2 – Tangentially related.  3 – Partially relevant.
  4 – Mostly relevant.  5 – Highly relevant; directly addresses what the query requires.

**Faithfulness (0–5)** — How well is the answer grounded in the retrieved context?
  0 – Contradicts context or invents facts.  2 – Significant fabrication.  3 – Mostly grounded.
  4 – Well grounded; only trivial details venture beyond context.  5 – Fully grounded.

**Answer Relevance (0–5)** — How well does the answer address the original query?
  0 – Off-topic.  2 – Touches topic but misses core question.  3 – Partially answers.
  4 – Mostly answers; minor omissions.  5 – Comprehensively and accurately answers.

### OUTPUT INSTRUCTIONS
- Return ONLY valid JSON — no markdown fences, no extra text.
- Use this exact structure:
{{
  "context_relevance": <integer 0-5>,
  "faithfulness": <integer 0-5>,
  "answer_relevance": <integer 0-5>,
  "reasoning": "<one concise sentence justifying each of the three scores>"
}}"""
    ),
    (
        "human",
        """**User Query:**
{query}

**Retrieved Context:**
{context}

**Generated Answer:**
{answer}"""
    ),
])


@dataclass
class EvaluationResult:
    """Holds the three RAG quality scores (0-5 each) and optional reasoning."""

    context_relevance: int
    faithfulness: int
    answer_relevance: int
    reasoning: str = ""

    def as_markdown(self) -> str:
        """Render scores as a compact Markdown table with star ratings."""

        def _stars(score: int) -> str:
            return "⭐" * score + "☆" * (5 - score)

        rows = [
            ("Context Relevance", self.context_relevance),
            ("Faithfulness", self.faithfulness),
            ("Answer Relevance", self.answer_relevance),
        ]
        lines = [
            "",
            "---",
            "### 📊 RAG Quality Assessment",
            "",
            "| Metric | Score | Rating |",
            "|--------|:-----:|--------|",
        ]
        for name, score in rows:
            lines.append(f"| {name} | **{score}/5** | {_stars(score)} |")

        if self.reasoning:
            lines += ["", f"> *{self.reasoning}*"]

        return "\n".join(lines)


class RAGJudge:
    """Calls the local LLM to produce structured RAG quality scores."""

    def evaluate(
        self,
        query: str,
        context: str,
        answer: str,
        model: str | None = None,
    ) -> EvaluationResult | None:
        """Score a single RAG interaction.

        Returns ``None`` on any error so callers can degrade gracefully.

        Args:
            query: The original user question.
            context: The retrieved context as returned by the RAG tool.
            answer: The final generated answer.
            model: Ollama model override. Defaults to the configured model.
        """
        try:
            llm = get_llm(model=model, temperature=0.0, require_json=True)
            chain = _JUDGE_PROMPT | llm
            response = chain.invoke({"query": query, "context": context, "answer": answer})
            raw = response.content.strip()
            data = json.loads(raw)
            return EvaluationResult(
                context_relevance=int(data.get("context_relevance", 0)),
                faithfulness=int(data.get("faithfulness", 0)),
                answer_relevance=int(data.get("answer_relevance", 0)),
                reasoning=str(data.get("reasoning", "")),
            )
        except Exception as exc:
            logger.warning("RAG evaluation failed: %s", exc)
            return None

    async def aevaluate(
        self,
        query: str,
        context: str,
        answer: str,
        model: str | None = None,
    ) -> EvaluationResult | None:
        """Async version of evaluate. Uses ainvoke for non-blocking LLM call."""
        try:
            llm = get_llm(model=model, temperature=0.0, require_json=True)
            chain = _JUDGE_PROMPT | llm
            response = await chain.ainvoke({"query": query, "context": context, "answer": answer})
            raw = response.content.strip()
            data = json.loads(raw)
            return EvaluationResult(
                context_relevance=int(data.get("context_relevance", 0)),
                faithfulness=int(data.get("faithfulness", 0)),
                answer_relevance=int(data.get("answer_relevance", 0)),
                reasoning=str(data.get("reasoning", "")),
            )
        except Exception as exc:
            logger.warning("Async RAG evaluation failed: %s", exc)
            return None
