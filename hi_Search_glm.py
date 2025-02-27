import os
import logging
import numpy as np
import yaml
from hirag import HiRAG, QueryParam
from openai import AsyncOpenAI, OpenAI
from dataclasses import dataclass
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash
from tqdm import tqdm

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract configurations
GLM_API_KEY = config['glm']['api_key']
MODEL = config['glm']['model']
GLM_URL = config['glm']['base_url']

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

@wrap_embedding_func_with_attrs(embedding_dim=config['model_params']['glm_embedding_dim'], max_token_size=config['model_params']['max_token_size'])
async def GLM_embedding(texts: list[str]) -> np.ndarray:
    model_name = config['glm']['embedding_model']
    client = OpenAI(
        api_key=GLM_API_KEY,
        base_url=GLM_URL
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
    openai_async_client = AsyncOpenAI(
        api_key=GLM_API_KEY, base_url=GLM_URL
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
    try:
        response = await openai_async_client.chat.completions.create(
            model=MODEL, messages=messages, **kwargs
        )
    except Exception as e:
        logging.info(e)
        return "ERROR"

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content


graph_func = HiRAG(
    working_dir=config['hirag']['working_dir'],
    enable_llm_cache=config['hirag']['enable_llm_cache'],
    embedding_func=GLM_embedding,
    best_model_func=glm_model_if_cache,
    cheap_model_func=glm_model_if_cache,
    enable_hierachical_mode=config['hirag']['enable_hierachical_mode'], 
    embedding_func_max_async=config['hirag']['embedding_func_max_async'],
    enable_naive_rag=config['hirag']['enable_naive_rag'])

with open("./web3_test/txtWhitePapers/aave-v2-whitepaper.pdf.txt") as f:
    graph_func.insert(f.read())

print("Perform hi search:")
print(graph_func.query("What are the top themes in this story?", param=QueryParam(mode="hi")))
