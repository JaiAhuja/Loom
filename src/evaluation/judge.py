"""LLM-as-a-Judge evaluation for RAG quality assessment.

Scores a RAG response across three dimensions (0–5 each):
- Context Relevance: how relevant the retrieved chunks are to the query.
- Faithfulness: how grounded the answer is in the retrieved context.
- Answer Relevance: how directly and helpfully the answer addresses the query.
"""
import json
import logging
from dataclasses import dataclass

from src.llm import get_llm

logger = logging.getLogger(__name__)

_JUDGE_PROMPT_TEMPLATE = """\
You are a strict RAG evaluation judge. Your task is to evaluate the quality of a \
Retrieval-Augmented Generation (RAG) response.

## Input

**User Query:**
{query}

**Retrieved Context:**
{context}

**Generated Answer:**
{answer}

## Scoring Dimensions

Score each dimension as an integer from 0 to 5 using the rubrics below.

**Context Relevance (0–5)**
How relevant is the retrieved context to the user's query?
  0 – Completely irrelevant; context has nothing to do with the query.
  2 – Tangentially related; mentions the topic but misses the point.
  3 – Partially relevant; some useful content but significant gaps.
  4 – Mostly relevant; covers the key aspects with minor gaps.
  5 – Highly relevant; directly and fully addresses what the query requires.

**Faithfulness (0–5)**
How well is the generated answer grounded in the retrieved context (no hallucinations)?
  0 – Answer contradicts the context or invents unsupported facts.
  2 – Significant fabrication; only a small fraction is supported by context.
  3 – Mostly grounded but includes some unsupported claims.
  4 – Well grounded; only trivial details venture beyond the context.
  5 – Fully grounded; every factual claim can be traced to the retrieved context.

**Answer Relevance (0–5)**
How well does the generated answer address the user's original query?
  0 – Answer is off-topic or does not address the query at all.
  2 – Touches the topic but misses the core question.
  3 – Partially addresses the query; some important aspects are missing.
  4 – Mostly answers the query; minor omissions only.
  5 – Comprehensively and accurately answers the query.

## Output Format

Return ONLY a valid JSON object — no markdown fences, no extra text — with this \
exact structure:

{{
  "context_relevance": <integer 0-5>,
  "faithfulness": <integer 0-5>,
  "answer_relevance": <integer 0-5>,
  "reasoning": "<one concise sentence justifying each of the three scores>"
}}
"""


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
        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            query=query,
            context=context,
            answer=answer,
        )
        try:
            llm = get_llm(model=model, temperature=0.0, require_json=True)
            response = llm.invoke(prompt)
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
