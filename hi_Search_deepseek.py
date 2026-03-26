import os
import logging
import numpy as np
import yaml
from hirag import HiRAG, QueryParam
from openai import AsyncOpenAI
from dataclasses import dataclass
from hirag.base import BaseKVStorage
from hirag._storage import NetworkXStorage
from hirag._utils import compute_args_hash

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def _from_env_or_config(env_name: str, config_value: str) -> str:
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        return env_value
    return str(config_value or "").strip()


def _optional(value: str):
    return value if value else None

# Extract configurations
GLM_API_KEY = _from_env_or_config('GLM_API_KEY', config['glm']['api_key'])
MODEL = config['deepseek']['model']
DEEPSEEK_EMBEDDING_MODEL = str(config['deepseek'].get('embedding_model', '')).strip()
DEEPSEEK_API_KEY = _from_env_or_config('DEEPSEEK_API_KEY', config['deepseek']['api_key'])
DEEPSEEK_URL = _from_env_or_config('DEEPSEEK_BASE_URL', config['deepseek']['base_url'])
GLM_URL = _from_env_or_config('GLM_BASE_URL', config['glm']['base_url'])


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
    model_name = str(config['glm'].get('embedding_model', 'embedding-3')).strip()
    client = AsyncOpenAI(
        api_key=GLM_API_KEY,
        base_url=_optional(GLM_URL)
    ) 
    embedding = await client.embeddings.create(
        input=texts,
        model=model_name,
    )
    final_embedding = [d.embedding for d in embedding.data]
    return np.array(final_embedding)


@wrap_embedding_func_with_attrs(embedding_dim=config['model_params']['glm_embedding_dim'], max_token_size=config['model_params']['max_token_size'])
async def DEEPSEEK_embedding(texts: list[str]) -> np.ndarray:
    client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=_optional(DEEPSEEK_URL)
    )
    embedding = await client.embeddings.create(
        input=texts,
        model=DEEPSEEK_EMBEDDING_MODEL,
    )
    final_embedding = [d.embedding for d in embedding.data]
    return np.array(final_embedding)


async def deepseepk_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY, base_url=_optional(DEEPSEEK_URL)
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


if not DEEPSEEK_API_KEY:
    raise ValueError(
        "DeepSeek API key is missing. Set DEEPSEEK_API_KEY env var or deepseek.api_key in config.yaml"
    )

if DEEPSEEK_EMBEDDING_MODEL:
    embedding_func = DEEPSEEK_embedding
    logging.info(f"Using DeepSeek embeddings with model: {DEEPSEEK_EMBEDDING_MODEL}")
elif GLM_API_KEY:
    embedding_func = GLM_embedding
    logging.info("Using GLM embeddings")
else:
    raise ValueError(
        "No embedding provider available. Set deepseek.embedding_model for DeepSeek embeddings, or set GLM_API_KEY."
    )

working_dir = config['hirag']['working_dir']
input_txt_path = str(config['hirag'].get('input_txt_path', '')).strip()
query = str(config['hirag'].get('query', 'What are the top themes in this story?')).strip()
mode = str(config['hirag'].get('mode', 'hi')).strip()
insert_on_startup = bool(config['hirag'].get('insert_on_startup', False))

react_cfg = config['hirag'].get('react', {})
enable_react = bool(react_cfg.get('enable', False))
react_max_iter = int(react_cfg.get('max_iter', 1))
react_context_mode = str(react_cfg.get('context_mode', 'second_only')).strip()

graph_func = HiRAG(
    working_dir=working_dir,
    enable_llm_cache=config['hirag']['enable_llm_cache'],
    embedding_func=embedding_func,
    best_model_func=deepseepk_model_if_cache,
    cheap_model_func=deepseepk_model_if_cache,
    enable_hierachical_mode=config['hirag']['enable_hierachical_mode'],
    embedding_batch_num=config['hirag']['embedding_batch_num'],
    embedding_func_max_async=config['hirag']['embedding_func_max_async'],
    enable_naive_rag=config['hirag']['enable_naive_rag'],
    graph_storage_cls=NetworkXStorage,
)

if insert_on_startup:
    if not input_txt_path:
        raise ValueError(
            "hirag.input_txt_path is empty in config.yaml. Set it to a local .txt file before indexing."
        )
    if not os.path.isfile(input_txt_path):
        raise FileNotFoundError(f"Input file not found: {input_txt_path}")
    with open(input_txt_path, encoding="utf-8") as f:
        graph_func.insert(f.read())

print(f"Perform {mode} search:")
print(
    graph_func.query(
        query,
        param=QueryParam(
            mode=mode,
            enable_react=enable_react,
            react_max_iter=react_max_iter,
            react_context_mode=react_context_mode,
        ),
    )
)
