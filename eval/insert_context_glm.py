import os
import sys
import json
import time
sys.path.append("../")
from hirag import HiRAG, QueryParam
import os
import logging
import numpy as np
import tiktoken
import yaml
from openai import AsyncOpenAI, OpenAI
from dataclasses import dataclass
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash

logging.basicConfig(level=logging.WARNING)
logging.getLogger("HiRAG").setLevel(logging.INFO)

DATASET = "agriculture"
file_path = f"./datasets/{DATASET}/{DATASET}_unique_contexts.json"
WORKING_DIR = f"./datasets/{DATASET}/work_dir_deepseek_hi_clustercase"

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

MODEL = config['glm']['model']
GLM_API_KEY = config['glm']['api_key']
TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0
tokenizer = tiktoken.get_encoding("cl100k_base")


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


async def glm_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    global TOTAL_TOKEN_COST
    global TOTAL_API_CALL_COST

    openai_async_client = AsyncOpenAI(
        api_key=GLM_API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4"
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # logging token cost
    cur_token_cost = len(tokenizer.encode(messages[0]['content']))
    TOTAL_TOKEN_COST += cur_token_cost
    # logging api call cost
    TOTAL_API_CALL_COST += 1
    
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------
    try:
        # request
        response = await openai_async_client.chat.completions.create(
            model=MODEL, messages=messages, **kwargs
        )
    except Exception as e:
        logging.info(e)
        return "<|COMPLETE|>"

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content


if __name__ == "__main__":
    graph_func = HiRAG(
        working_dir=WORKING_DIR, 
        enable_llm_cache=True,
        embedding_func=GLM_embedding,
        best_model_func=glm_model_if_cache,
        cheap_model_func=glm_model_if_cache,
        enable_hierachical_mode=True, 
        embedding_func_max_async=8,
        enable_naive_rag=True)

    with open(file_path, mode="r") as f:
        unique_contexts = json.load(f)
        graph_func.insert(unique_contexts)
        logging.info(f"[Total token cost: {TOTAL_TOKEN_COST}]")
        logging.info(f"[Total api call cost: {TOTAL_API_CALL_COST}]")