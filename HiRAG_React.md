# ReAct Integration Guidance for HiRAG

This document explains how to add a lightweight ReAct-style retrieval loop into the current HiRAG project:
retrieve once, rewrite the query based on retrieved evidence, retrieve again, then generate the final answer.

---

## 1. High-Level ReAct Idea in HiRAG

This design does not aim for a full agent/tool framework. It focuses on your core requirement:

1. **First Retrieval (Search-1)**
   Run HiRAG retrieval using the original user question `query`, and get initial context (entities/communities/path/text units).
2. **Interaction + Query Rewrite**
   Feed key evidence from Search-1 into an LLM and produce a more retrieval-friendly, more specific `rewritten_query`.
   This is exactly the "rewrite query based on retrieved content" step.
3. **Second Retrieval (Search-2)**
   Run retrieval again (same mode) with `rewritten_query` to get more focused context.
4. **Answer Generation**
   Generate final response using Search-2 context as primary input (optionally merge Search-1 context).

A minimal MVP flow:

- Round 1: set `only_need_context=True` to get `context1`
- Rewrite: `rewritten_query = llm(query + context1)`
- Round 2: set `only_need_context=True` to get `context2`
- Final: use `context2 + rewritten_query` with existing `local_rag_response`

---

## 2. Where to Modify in This Project

The current codebase is already modular enough for clean integration. Recommended changes:

## 2.1 `hirag/base.py`: Extend `QueryParam`

Add ReAct flags and parameters to `QueryParam`:

- `enable_react: bool = False`
- `react_max_iter: int = 1` (start with one rewrite round)
- `react_context_mode: str = "second_only"` (optional: `second_only` / `concat`)

Purpose: preserve backward compatibility and enable ReAct only when explicitly turned on.

---

## 2.2 `hirag/prompt.py`: Add Rewrite Prompt

Add a new prompt key, for example:

- `PROMPTS["react_query_rewrite"]`

Usage: input original query + first-round retrieval context, output a rewritten query.
Recommended constraint: force one-line output to avoid verbose explanations.

---

## 2.3 `hirag/_op.py`: Add ReAct Core Functions (Main Work)

This file already contains retrieval context builders and query pipelines. Add three function groups:

1. **Context Builder by Mode (without answer generation)**
   - `async def build_context_by_mode(...)`
   - Dispatch by `query_param.mode` and reuse existing builders:
     - `_build_hierarchical_query_context`
     - `_build_hibridge_query_context`
     - `_build_hilocal_query_context`
     - `_build_higlobal_query_context`
     - `_build_local_query_context`
     - plus naive mode chunk context logic

2. **Query Rewrite Function**
   - `async def rewrite_query_with_context(query, context, global_config)`
   - Call `global_config["cheap_model_func"]` (or `best_model_func`) with `PROMPTS["react_query_rewrite"]`
   - Return `rewritten_query`

3. **ReAct Query Entry**
   - `async def hierarchical_react_query(...)` (support `hi` first, expand later)
   - Main flow:
     - `context1 = build_context_by_mode(query, ...)`
     - `rewritten_query = rewrite_query_with_context(query, context1, ...)`
     - `context2 = build_context_by_mode(rewritten_query, ...)`
     - Build final context (`context2` only, or `context1 + context2`)
     - Generate final answer with existing `PROMPTS["local_rag_response"]`

Why `_op.py`: it is already the central layer for retrieval context construction and response generation, so this gives minimal code intrusion and maximum reuse.

---

## 2.4 `hirag/hirag.py`: Hook ReAct in `aquery()`

In `HiRAG.aquery()`, add a small gate around the current mode dispatch:

- If `param.enable_react` is `True` and mode is supported (start with `"hi"`):
  - route to `hierarchical_react_query(...)`
- Otherwise keep current flow unchanged:
  - `hierarchical_query / ... / naive_query`

Benefits:

- Existing scripts remain unchanged by default
- New behavior is opt-in only

---

## 2.5 Top-Level Runners: `hi_Search_*.py`

Update calls in `hi_Search_deepseek.py`, `hi_Search_openai.py`, and `hi_Search_glm.py`, for example:

- `QueryParam(mode="hi", enable_react=True, react_max_iter=1)`

Optional: add `hirag.react.*` settings in `config.yaml` and map them to `QueryParam` in each runner.

---

## 3. Recommended Implementation Order

1. Start with `mode="hi"` and single-round ReAct (`react_max_iter=1`)
2. After stable validation, expand to:
   `hi_bridge / hi_local / hi_global / hi_nobridge / naive`
3. Then consider multi-round ReAct (`react_max_iter > 1`) and early-stop policy

---

## 4. Minimal Pseudocode (Aligned with Current Code)

```python
async def hierarchical_react_query(query, knowledge_graph_inst, entities_vdb,
                                   community_reports, text_chunks_db,
                                   query_param, global_config):
    use_model_func = global_config["best_model_func"]

    # 1) first retrieval
    context1 = await _build_hierarchical_query_context(
        query, knowledge_graph_inst, entities_vdb, community_reports, text_chunks_db, query_param
    )
    if context1 is None:
        return PROMPTS["fail_response"]

    # 2) rewrite query
    rewritten_query = await rewrite_query_with_context(query, context1, global_config)
    if not rewritten_query:
        rewritten_query = query

    # 3) second retrieval
    context2 = await _build_hierarchical_query_context(
        rewritten_query, knowledge_graph_inst, entities_vdb, community_reports, text_chunks_db, query_param
    )
    if context2 is None:
        context2 = context1

    final_context = context2
    sys_prompt = PROMPTS["local_rag_response"].format(
        context_data=final_context,
        response_type=query_param.response_type
    )
    return await use_model_func(rewritten_query, system_prompt=sys_prompt)
```

---

## 5. Engineering Notes

- **Compatibility**: when `enable_react=False`, behavior should be identical to current implementation.
- **Cost and latency**: ReAct adds at least one extra retrieval and one LLM rewrite call.
- **Robustness**: if rewrite fails, fallback to original query; if second retrieval fails, fallback to `context1`.
- **Observability**: log `original_query`, `rewritten_query`, and retrieval timings to measure gains.

---

## 6. Validation Suggestion

You can reuse `eval/batch_eval.py` for A/B tests:

- Group A: native `mode="hi"`
- Group B: `mode="hi" + enable_react=True`

Compare:

- answer quality
- retrieval coverage (entity/path hits)
- latency and token cost

If Group B consistently improves key entity/relation hits on complex questions, ReAct integration is effective.
