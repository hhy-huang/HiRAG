import os
import sys
import json
import time
import argparse
import numpy as np
import networkx as nx
sys.path.append("../")
from hirag import HiRAG, QueryParam
from openai import AsyncOpenAI, OpenAI
from dataclasses import dataclass
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash
from tqdm import tqdm
from networkx.linalg.algebraicconnectivity import algebraic_connectivity

os.environ["OPENAI_API_KEY"] = "***"
GLM_API_KEY = "***"
MODEL = "deepseek-chat"
DEEPSEEK_API_KEY = "***"


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)

def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro

@wrap_embedding_func_with_attrs(embedding_dim=2048, max_token_size=8192)
async def GLM_embedding(texts: list[str]) -> np.ndarray:
    model_name = "embedding-3"
    client = OpenAI(
        api_key=GLM_API_KEY,
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    ) 
    embedding = client.embeddings.create(
        input=texts,
        model=model_name,
    )
    final_embedding = [d.embedding for d in embedding.data]
    return np.array(final_embedding)


async def deepseepk_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content

DATASET = "legal"

graph_func = HiRAG(
    working_dir=f"../eval/datasets/{DATASET}/work_dir_glm_hi_clustercase",
    enable_llm_cache=False,
    embedding_func=GLM_embedding,
    best_model_func=deepseepk_model_if_cache,
    cheap_model_func=deepseepk_model_if_cache,
    enable_hierachical_mode=True, 
    embedding_func_max_async=16,
    enable_naive_rag=True)

nx_graph = graph_func.chunk_entity_relation_graph._graph
num_nodes = nx_graph.number_of_nodes()
num_edges = nx_graph.number_of_edges()
max_edges_directed = num_nodes * (num_nodes - 1) / 2
sparsity_directed = 1 - (num_edges / max_edges_directed)
alg_connectivity = nx.transitivity(nx_graph)

print("Dataset:", DATASET)
print("Sparsity:", sparsity_directed)
print("Global Clustering Coefficient:", alg_connectivity)
