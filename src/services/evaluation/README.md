# Evaluation service

The evaluation service provides an LLM-as-judge for retrieval-augmented
answers. It scores context relevance, answer faithfulness, and answer
relevance, then returns a structured result suitable for display in the chat
page.

## Public API

- `RAGJudge.evaluate(...)` — evaluates a question, answer, and retrieved
  context using the configured LLM.
- `EvaluationResult` — stores the scores, reasoning, and Markdown rendering
  used by the UI.

The service depends on `llm` for the judge model and `common` for robust JSON
parsing. It is optional to the core chat flow; a failed evaluation should not
prevent a user from receiving an answer. See
`tests/unit/test_llm_and_evaluation.py`.
