"""Benchmarking script for GraphRAG."""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import argparse
from collections import Counter
sys.path.append("../")
import numpy as np
import xxhash
import tiktoken
from dotenv import load_dotenv
from hirag import HiRAG, QueryParam
from hirag.hirag import always_get_an_event_loop
from hirag._llm import gpt_4o_mini_complete
from hirag._utils import logging, compute_args_hash
from hirag.base import BaseKVStorage
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from openai import AsyncOpenAI, OpenAI


logging.basicConfig(level=logging.WARNING)
logging.getLogger("HiRAG").setLevel(logging.INFO)

GLM_API_KEY = "dd28703438eb35f06df75ba427be00e8.ENP06LydxFmc2iGq"
DEEPSEEK_SOURCE = "aimlapi" #  siliconflow or deepseek or aimlapi
TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0
if DEEPSEEK_SOURCE == "siliconflow":
    MODEL = "deepseek-ai/DeepSeek-V3"
    DEEPSEEK_API_KEY = "sk-hwjhlnkpfuscwkfmzrnljyxnhuwnxpydbxihtisndhrebbks"
    DEEPSEEK_URL = "https://api.siliconflow.cn/v1"
elif DEEPSEEK_SOURCE == "deepseek":
    MODEL = "deepseek-chat"
    DEEPSEEK_API_KEY = "sk-d064b13decda4bf39af9791405cc2f08"
    DEEPSEEK_URL = "https://api.deepseek.com"
elif DEEPSEEK_SOURCE == "aimlapi":
    MODEL = "deepseek/deepseek-chat"
    DEEPSEEK_API_KEY = "c9018e434dfb429a957d0f931bd9e6c7"
    DEEPSEEK_URL = "https://api.aimlapi.com/v1"
GLM_MODEL = "glm-4-plus"
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

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    try:
        cur_token_cost = len(tokenizer.encode(messages[0]['content']))
        TOTAL_TOKEN_COST += cur_token_cost
        response = await openai_async_client.chat.completions.create(
            model=GLM_MODEL, messages=messages, **kwargs
        )
    except Exception as e:
        logging.info(e)
        return "ERROR"

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    return response.choices[0].message.content

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

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    cur_token_cost = len(tokenizer.encode(messages[0]['content']))
    TOTAL_TOKEN_COST += cur_token_cost
    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    return response.choices[0].message.content

@dataclass
class Query:
    """Dataclass for a query."""

    question: str = field()
    answer: str = field()
    evidence: List[Tuple[str, int]] = field()

def load_dataset(dataset_name: str, subset: int = 0) -> Any:
    """Load a dataset from the datasets folder."""
    with open(f"./datasets/{dataset_name}/{dataset_name}.json", "r") as f:
        dataset = json.load(f)

    if subset:
        return dataset[:subset]
    else:
        return dataset

def get_corpus(dataset: Any, dataset_name: str) -> Dict[int, Tuple[int | str, str]]:
    """Get the corpus from the dataset."""
    if dataset_name == "2wikimultihopqa" or dataset_name == "hotpotqa":
        passages: Dict[int, Tuple[int | str, str]] = {}

        for datapoint in dataset:
            context = datapoint["context"]

            for passage in context:
                title, text = passage
                title = title.encode("utf-8").decode()
                text = "\n".join(text).encode("utf-8").decode()
                hash_t = xxhash.xxh3_64_intdigest(text)
                if hash_t not in passages:
                    passages[hash_t] = (title, text)

        return passages
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")

def get_queries(dataset: Any):
    """Get the queries from the dataset."""
    queries: List[Query] = []

    for datapoint in dataset:
        queries.append(
            Query(
                question=datapoint["question"].encode("utf-8").decode(),
                answer=datapoint["answer"],
                evidence=list(datapoint["supporting_facts"]),
            )
        )

    return queries

def compute_em(pred: str, true: str) -> float:
    """Compute Exact Match score with normalization."""
    pred = re.sub(r'[^\w\s]', '', pred.strip().lower())
    true = re.sub(r'[^\w\s]', '', true.strip().lower())
    pred = ' '.join(pred.split())
    true = ' '.join(true.split())
    return 1.0 if pred == true else 0.0

def compute_f1(pred: str, true: str) -> float:
    """Compute F1 score with token overlap."""
    pred_tokens = tokenizer.encode(pred)
    true_tokens = tokenizer.encode(true)
    
    if not pred_tokens or not true_tokens:
        return 0.0
    
    pred_counter = Counter(pred_tokens)
    true_counter = Counter(true_tokens)
    
    common = pred_counter & true_counter
    overlap = sum(common.values())
    
    precision = overlap / len(pred_tokens)
    recall = overlap / len(true_tokens)
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="HiRAG CLI")
    parser.add_argument("-d", "--dataset", default="hotpotqa", help="Dataset to use.")
    parser.add_argument("-n", type=int, default=0, help="Subset of corpus to use.")
    parser.add_argument("-c", "--create", action="store_true", default="False", help="Create the graph for the given dataset.")
    parser.add_argument("-b", "--benchmark", action="store_true", default="True", help="Benchmark the graph for the given dataset.")
    parser.add_argument("-s", "--score", action="store_true", default="True", help="Report scores after benchmarking.")
    parser.add_argument("-k", "--topk", type=int, default=20, help="top-k entities retrieval.")
    parser.add_argument("--mode", default="hi", help="HiRAG query mode.")
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = load_dataset(args.dataset, subset=args.n)
    working_dir = f"./datasets/{args.dataset}/{args.dataset}_{args.n}_hi"
    corpus = get_corpus(dataset, args.dataset)

    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    if args.create:
        print("Dataset loaded. Corpus:", len(corpus))
        grag = HiRAG(
            working_dir=working_dir, 
            enable_llm_cache=True,
            embedding_func=GLM_embedding,
            best_model_func=glm_model_if_cache,
            cheap_model_func=glm_model_if_cache,
            enable_hierachical_mode=True, 
            embedding_func_max_async=8,
            enable_naive_rag=True
            )
        grag.insert([f"{title}: {corpus}" for _, (title, corpus) in tuple(corpus.items())])
    if args.benchmark:
        queries = get_queries(dataset)
        print("Dataset loaded. Queries:", len(queries))
        grag = HiRAG(
            working_dir=working_dir, 
            enable_llm_cache=False,
            embedding_func=GLM_embedding,
            best_model_func=deepseepk_model_if_cache,
            cheap_model_func=deepseepk_model_if_cache,
            enable_hierachical_mode=True, 
            embedding_func_max_async=1,
            enable_naive_rag=True
            )

        async def _query_task_hi(query: Query, mode: str) -> Dict[str, Any]:
            answer = await grag.aquery(
                query.question, QueryParam(
                    mode=mode,
                    top_k=args.topk,
                    top_m=args.topk,
                    only_need_context=False,
                    max_token_for_text_unit=9000
                )
            )
            return {
                "question": query.question,
                "answer": answer,
                "evidence": answer.lower(),
                "ground_truth": [e[0] for e in query.evidence],
                "true_answer": query.answer
            }

        async def _run_hi(mode: str):
            answers = [
                await a
                for a in tqdm(
                    asyncio.as_completed([_query_task_hi(query, mode=mode) for query in queries]), total=len(queries)
                )
            ]
            return answers

        answers = always_get_an_event_loop().run_until_complete(_run_hi(mode=args.mode))

        with open(f"./datasets/{args.dataset}/{args.dataset}_{args.n}_{args.mode}_hi.json", "w") as f:
            json.dump(answers, f, indent=4)

    if args.benchmark or args.score:
        with open(f"./datasets/{args.dataset}/{args.dataset}_{args.n}_{args.mode}_hi.json", "r") as f:
            answers = json.load(f)

        try:
            with open(f"./questions/{args.dataset}_{args.n}.json", "r") as f:
                questions_multihop = json.load(f)
        except FileNotFoundError:
            questions_multihop = []

        # Compute retrieval metrics
        retrieval_scores: List[float] = []
        retrieval_scores_multihop: List[float] = []
        em_scores: List[float] = []
        f1_scores: List[float] = []

        for answer in answers:
            # Retrieval metrics
            ground_truth = answer["ground_truth"]
            predicted_evidence = answer["evidence"]
            hit_num = 0
            ground_truth = set(ground_truth)
            for item in ground_truth:
                item = item.lower()
                if item in predicted_evidence:
                    hit_num += 1
            p_retrieved: float = hit_num / len(ground_truth)
            retrieval_scores.append(p_retrieved)

            if answer["question"] in questions_multihop:
                retrieval_scores_multihop.append(p_retrieved)

            # Answer quality metrics
            pred = answer.get("answer", "")
            true = answer.get("true_answer", "")
            em_scores.append(compute_em(pred, true))
            f1_scores.append(compute_f1(pred, true))

        print(f"\nRetrieval Metrics:")
        print(f"Recall@{args.topk}: {np.mean([1 if s == 1.0 else 0 for s in retrieval_scores])}")
        if len(retrieval_scores_multihop):
            print(f"[multihop] Percentage of queries with perfect retrieval: {np.mean([1 if s == 1.0 else 0 for s in retrieval_scores_multihop])}")

        print(f"\nAnswer Quality Metrics:")
        print(f"Exact Match (EM): {np.mean(em_scores):.4f}")
        print(f"F1 Score: {np.mean(f1_scores):.4f}")