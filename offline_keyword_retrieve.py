import argparse
import json
import os
import re
from collections import Counter

import networkx as nx


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str):
    # Simple bilingual-friendly tokenizer: latin words + CJK chars
    text = (text or "").lower()
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)
    return words


def overlap_score(query_tokens, text):
    text_tokens = tokenize(text)
    if not text_tokens:
        return 0
    text_counter = Counter(text_tokens)
    return sum(text_counter[t] for t in query_tokens)


def top_items(scored_items, top_k):
    scored_items.sort(key=lambda x: x[0], reverse=True)
    return [x for x in scored_items if x[0] > 0][:top_k]


def main():
    parser = argparse.ArgumentParser(description="Offline keyword retrieval from an existing HiRAG working_dir")
    parser.add_argument("--work-dir", required=True, help="Path to existing HiRAG working_dir")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--top-k", type=int, default=5, help="Top K for each section")
    args = parser.parse_args()

    work_dir = args.work_dir
    query = args.query
    top_k = args.top_k

    files = {
        "graph": os.path.join(work_dir, "graph_chunk_entity_relation.graphml"),
        "chunks": os.path.join(work_dir, "kv_store_text_chunks.json"),
        "communities": os.path.join(work_dir, "kv_store_community_reports.json"),
    }

    for name, path in files.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {name} file: {path}")

    query_tokens = tokenize(query)
    if not query_tokens:
        raise ValueError("Query is empty after tokenization")

    graph = nx.read_graphml(files["graph"])
    chunks = load_json(files["chunks"])
    communities = load_json(files["communities"])

    entity_hits = []
    for node, data in graph.nodes(data=True):
        text = f"{node} {data.get('entity_type', '')} {data.get('description', '')}"
        score = overlap_score(query_tokens, text)
        if score > 0:
            entity_hits.append((score, node, data))

    chunk_hits = []
    for chunk_id, chunk in chunks.items():
        text = chunk.get("content", "")
        score = overlap_score(query_tokens, text)
        if score > 0:
            chunk_hits.append((score, chunk_id, chunk))

    community_hits = []
    for cid, c in communities.items():
        text = c.get("report_string", "")
        score = overlap_score(query_tokens, text)
        if score > 0:
            community_hits.append((score, cid, c))

    top_entities = top_items(entity_hits, top_k)
    top_chunks = top_items(chunk_hits, top_k)
    top_communities = top_items(community_hits, top_k)

    print("=" * 90)
    print("Offline Retrieval Result (No API)")
    print("=" * 90)
    print(f"query: {query}")
    print(f"work_dir: {work_dir}")
    print()

    print("[Top Entities]")
    if not top_entities:
        print("  (no matches)")
    for score, node, data in top_entities:
        desc = (data.get("description", "") or "").replace("\n", " ")
        print(f"- score={score} entity={node} type={data.get('entity_type', 'UNKNOWN')}")
        print(f"  desc: {desc[:220]}")

    print()
    print("[Top Community Reports]")
    if not top_communities:
        print("  (no matches)")
    for score, cid, c in top_communities:
        title = c.get("title", f"Cluster {cid}")
        report = (c.get("report_string", "") or "").replace("\n", " ")
        print(f"- score={score} community={cid} title={title}")
        print(f"  report: {report[:260]}")

    print()
    print("[Top Source Chunks]")
    if not top_chunks:
        print("  (no matches)")
    for score, chunk_id, chunk in top_chunks:
        content = (chunk.get("content", "") or "").replace("\n", " ")
        print(f"- score={score} chunk_id={chunk_id}")
        print(f"  content: {content[:260]}")

    # Optional quick path between top-2 entities
    if len(top_entities) >= 2:
        src = top_entities[0][1]
        dst = top_entities[1][1]
        try:
            path = nx.shortest_path(graph, source=src, target=dst)
            print()
            print("[Reasoning Path Between Top-2 Entities]")
            print(" -> ".join(path))
        except Exception:
            pass


if __name__ == "__main__":
    main()
