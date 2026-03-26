import json
import os
import re
import time
from collections import Counter

import networkx as nx
import yaml
from openai import AsyncOpenAI


with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


def _from_env_or_config(env_name: str, config_value: str) -> str:
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        return env_value
    return str(config_value or "").strip()


def _optional(v: str):
    return v if v else None


def tokenize(text: str):
    text = (text or "").lower()
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)


def overlap_score(query_tokens, text):
    tks = tokenize(text)
    if not tks:
        return 0
    c = Counter(tks)
    return sum(c[t] for t in query_tokens)


def top_items(scored_items, top_k):
    scored_items.sort(key=lambda x: x[0], reverse=True)
    return [x for x in scored_items if x[0] > 0][:top_k]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_keyword_context(work_dir: str, query: str, top_k: int = 6) -> str:
    graph_path = os.path.join(work_dir, "graph_chunk_entity_relation.graphml")
    chunks_path = os.path.join(work_dir, "kv_store_text_chunks.json")
    communities_path = os.path.join(work_dir, "kv_store_community_reports.json")

    graph = nx.read_graphml(graph_path)
    chunks = load_json(chunks_path)
    communities = load_json(communities_path)

    q_tokens = tokenize(query)

    entity_hits = []
    for node, data in graph.nodes(data=True):
        txt = f"{node} {data.get('entity_type','')} {data.get('description','')}"
        s = overlap_score(q_tokens, txt)
        if s > 0:
            entity_hits.append((s, node, data))

    community_hits = []
    for cid, c in communities.items():
        txt = c.get("report_string", "")
        s = overlap_score(q_tokens, txt)
        if s > 0:
            community_hits.append((s, cid, c))

    chunk_hits = []
    for chunk_id, ch in chunks.items():
        txt = ch.get("content", "")
        s = overlap_score(q_tokens, txt)
        if s > 0:
            chunk_hits.append((s, chunk_id, ch))

    top_entities = top_items(entity_hits, top_k)
    top_communities = top_items(community_hits, top_k)
    top_chunks = top_items(chunk_hits, top_k)

    entities_section = [["id", "entity", "type", "description", "score"]]
    for i, (s, node, data) in enumerate(top_entities):
        entities_section.append([i, node, data.get("entity_type", "UNKNOWN"), data.get("description", ""), s])

    communities_section = [["id", "cluster", "content", "score"]]
    for i, (s, cid, c) in enumerate(top_communities):
        communities_section.append([i, cid, (c.get("report_string", "") or "").replace("\n", " "), s])

    chunks_section = [["id", "chunk_id", "content", "score"]]
    for i, (s, chunk_id, ch) in enumerate(top_chunks):
        chunks_section.append([i, chunk_id, (ch.get("content", "") or "").replace("\n", " "), s])

    path_text = ""
    if len(top_entities) >= 2:
        src = top_entities[0][1]
        dst = top_entities[1][1]
        try:
            path = nx.shortest_path(graph, source=src, target=dst)
            path_text = " -> ".join(path)
        except Exception:
            path_text = ""

    def to_csv(rows):
        out = []
        for r in rows:
            out.append(",".join([str(x).replace("\n", " ") for x in r]))
        return "\n".join(out)

    return f"""
-----Backgrounds-----
```csv
{to_csv(communities_section)}
```
-----Reasoning Path-----
```text
{path_text}
```
-----Detail Entity Information-----
```csv
{to_csv(entities_section)}
```
-----Source Documents-----
```csv
{to_csv(chunks_section)}
```
"""


async def call_deepseek(prompt: str, system_prompt: str = None) -> str:
    key = _from_env_or_config("DEEPSEEK_API_KEY", config["deepseek"]["api_key"])
    if not key:
        raise ValueError("Missing DeepSeek key. Set DEEPSEEK_API_KEY or config.deepseek.api_key")

    model = config["deepseek"]["model"]
    base_url = _from_env_or_config("DEEPSEEK_BASE_URL", config["deepseek"]["base_url"])

    client = AsyncOpenAI(api_key=key, base_url=_optional(base_url))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    resp = await client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content


async def rewrite_query(query: str, context: str) -> str:
    rewrite_prompt = f"""You rewrite search queries.
Given user query and retrieved evidence, output exactly one rewritten query line.
Do not answer.

Query: {query}
Evidence:\n{context}
"""
    text = await call_deepseek(rewrite_prompt)
    text = (text or "").strip().strip('"').strip("'")
    return text.splitlines()[0].strip() if text else ""


async def answer_with_context(query: str, context: str) -> str:
    sys_prompt = f"""You are a helpful assistant. Use only supported evidence from context.
If uncertain, say you don't know.

Context:\n{context}
"""
    return await call_deepseek(query, system_prompt=sys_prompt)


async def run_compare_once():
    work_dir = config["hirag"]["working_dir"]
    query = str(config["hirag"].get("compare_query", config["hirag"].get("query", ""))).strip()
    if not query:
        raise ValueError("hirag.compare_query and hirag.query are empty")

    # Baseline: retrieve once -> answer
    t0 = time.perf_counter()
    ctx_base = build_keyword_context(work_dir, query, top_k=6)
    ans_base = await answer_with_context(query, ctx_base)
    t1 = time.perf_counter()

    # ReAct: retrieve once -> rewrite -> retrieve again -> answer
    tr0 = time.perf_counter()
    ctx1 = build_keyword_context(work_dir, query, top_k=6)
    rewritten = await rewrite_query(query, ctx1)
    if not rewritten:
        rewritten = query
    ctx2 = build_keyword_context(work_dir, rewritten, top_k=6)
    ans_react = await answer_with_context(rewritten, ctx2)
    tr1 = time.perf_counter()

    print("=" * 80)
    print("Offline Keyword ReAct A/B (DeepSeek generation)")
    print("=" * 80)
    print(f"query: {query}")
    print(f"rewritten_query: {rewritten}")
    print(f"baseline_time_sec: {(t1 - t0):.3f}")
    print(f"react_time_sec:    {(tr1 - tr0):.3f}")
    print(f"react_overhead_sec:{((tr1 - tr0) - (t1 - t0)):.3f}")
    print()
    print("[Baseline Answer]")
    print(ans_base)
    print()
    print("[ReAct Answer]")
    print(ans_react)


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_compare_once())
