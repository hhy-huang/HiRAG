import os
import time
import logging
import numpy as np
import yaml
from dataclasses import dataclass
from openai import AsyncOpenAI

from hirag import HiRAG, QueryParam
from hirag.base import BaseKVStorage
from hirag._storage import NetworkXStorage
from hirag._utils import compute_args_hash


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def _from_env_or_config(env_name: str, config_value: str) -> str:
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        return env_value
    return str(config_value or "").strip()


def _optional(value: str):
    return value if value else None


MODEL = config["deepseek"]["model"]
DEEPSEEK_EMBEDDING_MODEL = str(config["deepseek"].get("embedding_model", "")).strip()
DEEPSEEK_API_KEY = _from_env_or_config("DEEPSEEK_API_KEY", config["deepseek"]["api_key"])
DEEPSEEK_URL = _from_env_or_config("DEEPSEEK_BASE_URL", config["deepseek"]["base_url"])
GLM_API_KEY = _from_env_or_config("GLM_API_KEY", config["glm"]["api_key"])
GLM_URL = _from_env_or_config("GLM_BASE_URL", config["glm"]["base_url"])
GLM_EMBEDDING_MODEL = str(config["glm"].get("embedding_model", "embedding-3")).strip()

METRICS = {
    "embedding_provider": "",
    "glm_embedding_api_calls": 0,
    "deepseek_embedding_api_calls": 0,
    "deepseek_chat_api_calls": 0,
    "deepseek_chat_cache_hits": 0,
}


def reset_metrics():
    METRICS["glm_embedding_api_calls"] = 0
    METRICS["deepseek_embedding_api_calls"] = 0
    METRICS["deepseek_chat_api_calls"] = 0
    METRICS["deepseek_chat_cache_hits"] = 0


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


def wrap_embedding_func_with_attrs(**kwargs):
    def final_decro(func) -> EmbeddingFunc:
        return EmbeddingFunc(**kwargs, func=func)

    return final_decro


@wrap_embedding_func_with_attrs(
    embedding_dim=config["model_params"]["glm_embedding_dim"],
    max_token_size=config["model_params"]["max_token_size"],
)
async def glm_embedding(texts: list[str]) -> np.ndarray:
    METRICS["glm_embedding_api_calls"] += 1
    client = AsyncOpenAI(api_key=GLM_API_KEY, base_url=_optional(GLM_URL))
    embedding = await client.embeddings.create(input=texts, model=GLM_EMBEDDING_MODEL)
    return np.array([d.embedding for d in embedding.data])


@wrap_embedding_func_with_attrs(
    embedding_dim=config["model_params"]["glm_embedding_dim"],
    max_token_size=config["model_params"]["max_token_size"],
)
async def deepseek_embedding(texts: list[str]) -> np.ndarray:
    METRICS["deepseek_embedding_api_calls"] += 1
    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=_optional(DEEPSEEK_URL))
    embedding = await client.embeddings.create(input=texts, model=DEEPSEEK_EMBEDDING_MODEL)
    return np.array([d.embedding for d in embedding.data])


async def deepseek_model_if_cache(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=_optional(DEEPSEEK_URL))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        cache_val = await hashing_kv.get_by_id(args_hash)
        if cache_val is not None:
            METRICS["deepseek_chat_cache_hits"] += 1
            return cache_val["return"]

    METRICS["deepseek_chat_api_calls"] += 1
    response = await client.chat.completions.create(model=MODEL, messages=messages, **kwargs)

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": response.choices[0].message.content, "model": MODEL}})

    return response.choices[0].message.content


def build_graph_func() -> HiRAG:
    if not DEEPSEEK_API_KEY:
        raise ValueError("Missing DeepSeek key. Set DEEPSEEK_API_KEY or config.deepseek.api_key")

    if DEEPSEEK_EMBEDDING_MODEL:
        use_embedding = deepseek_embedding
        METRICS["embedding_provider"] = f"deepseek:{DEEPSEEK_EMBEDDING_MODEL}"
        logging.info(f"Embedding provider: DeepSeek ({DEEPSEEK_EMBEDDING_MODEL})")
    elif GLM_API_KEY:
        use_embedding = glm_embedding
        METRICS["embedding_provider"] = f"glm:{GLM_EMBEDDING_MODEL}"
        logging.info("Embedding provider: GLM")
    else:
        raise ValueError("No embedding provider available. Set deepseek.embedding_model or GLM_API_KEY")

    return HiRAG(
        working_dir=config["hirag"]["working_dir"],
        enable_llm_cache=config["hirag"]["enable_llm_cache"],
        embedding_func=use_embedding,
        best_model_func=deepseek_model_if_cache,
        cheap_model_func=deepseek_model_if_cache,
        enable_hierachical_mode=config["hirag"]["enable_hierachical_mode"],
        embedding_batch_num=config["hirag"]["embedding_batch_num"],
        embedding_func_max_async=config["hirag"]["embedding_func_max_async"],
        enable_naive_rag=config["hirag"]["enable_naive_rag"],
        graph_storage_cls=NetworkXStorage,
    )


def run_once(enable_react: bool):
    query = str(config["hirag"].get("compare_query", config["hirag"].get("query", ""))).strip()
    if not query:
        raise ValueError("hirag.compare_query and hirag.query are both empty")

    reset_metrics()
    graph_func = build_graph_func()
    param = QueryParam(
        mode="hi",
        enable_react=enable_react,
        react_max_iter=int(config["hirag"].get("react", {}).get("max_iter", 1)),
        react_context_mode=str(config["hirag"].get("react", {}).get("context_mode", "second_only")),
    )

    start = time.perf_counter()
    answer = graph_func.query(query, param=param)
    elapsed = time.perf_counter() - start
    return query, answer, elapsed, dict(METRICS)


if __name__ == "__main__":
    q, ans_base, t_base, m_base = run_once(enable_react=False)
    _, ans_react, t_react, m_react = run_once(enable_react=True)

    print("=" * 80)
    print("HiRAG ReAct A/B (single query)")
    print("=" * 80)
    print(f"query: {q}")
    print(f"baseline_time_sec: {t_base:.3f}")
    print(f"react_time_sec:    {t_react:.3f}")
    print(f"react_overhead_sec:{(t_react - t_base):.3f}")
    print()
    print("[Baseline Usage]")
    print(m_base)
    print("[ReAct Usage]")
    print(m_react)
    print()
    print("[Baseline Answer]")
    print(ans_base)
    print()
    print("[ReAct Answer]")
    print(ans_react)
