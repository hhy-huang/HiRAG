import os
import sys
import json
import time
import argparse
sys.path.append("../")
import os
import logging
import numpy as np
import tiktoken
import yaml
from hirag import HiRAG, QueryParam
from openai import AsyncOpenAI, OpenAI
from dataclasses import dataclass
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash
from tqdm import tqdm

WORKING_DIR = f"./datasets/cs/work_dir_deepseek_hi"
MAX_QUERIES = 100
TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract configurations
MODEL = config['deepseek']['model']
DEEPSEEK_API_KEY = config['deepseek']['api_key']
DEEPSEEK_URL = config['deepseek']['base_url']

GLM_MODEL = config['glm']['model']
GLM_API_KEY = config['glm']['api_key']
GLM_URL = config['glm']['base_url']

OPENAI_MODEL = config['openai']['model']
OPENAI_API_KEY = config['openai']['api_key']
OPENAI_URL = config['openai']['base_url']
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


async def deepseepk_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    global TOTAL_TOKEN_COST
    global TOTAL_API_CALL_COST

    openai_async_client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_URL
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
    # logging token cost
    cur_token_cost = len(tokenizer.encode(messages[0]['content']))
    TOTAL_TOKEN_COST += cur_token_cost
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="legal")
    parser.add_argument("-m", "--mode", type=str, default="hi", help="hi / naive / hi_global / hi_local / hi_bridge / hi_nobridge")
    args = parser.parse_args()

    DATASET = args.dataset
    tok_k = 10
    if DATASET == "mix":
        MAX_QUERIES = 130
    elif DATASET == "cs" or DATASET == "agriculture" or DATASET == "legal":
        MAX_QUERIES = 100
    input_path = f"./datasets/{DATASET}/{DATASET}.jsonl"
    output_path = f"./datasets/{DATASET}/{DATASET}_{args.mode}_result_deepseek_pro.jsonl"
    graph_func = HiRAG(
        working_dir=WORKING_DIR, 
        enable_llm_cache=False,
        embedding_func=GLM_embedding,
        best_model_func=deepseepk_model_if_cache,
        cheap_model_func=deepseepk_model_if_cache,
        enable_hierachical_mode=True, 
        embedding_func_max_async=8,
        enable_naive_rag=True)

    query_list = []
    with open(input_path, encoding="utf-8", mode="r") as f:      # get context
        lines = f.readlines()
        for item in lines:
            item_dict = json.loads(item)
            query_list.append(item_dict["input"])
    query_list = query_list[:MAX_QUERIES]
    answer_list = []

    print(f"Perform {args.mode} search:")
    for query in tqdm(query_list):
        logging.info(f"Q: {query}")
        retry = 3
        while retry:
            try:
                answer = graph_func.query(query=query, param=QueryParam(mode=args.mode, top_k=tok_k))
                retry = 0
            except Exception as e:
                print(e)
                answer = "Error"
                retry -= 1
        logging.info(f"A: {answer} \n ################################################################################################")
        answer_list.append(answer)
    logging.info(f"[Token Cost: {TOTAL_TOKEN_COST}]")

    result_to_write = []
    for query, answer in zip(query_list, answer_list):
        result_to_write.append({"query": query, "answer": answer})
    with open(output_path, "w") as f:
        for item in result_to_write:
            f.write(json.dumps(item) + "\n")
        
