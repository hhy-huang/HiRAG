import re
import json
import jsonlines
import argparse
import os
import time
import copy
import yaml
import csv
import asyncio
from io import StringIO
from statistics import mean
from dataclasses import dataclass
from typing import Callable
import numpy as np
from openai import OpenAI
from openai import AsyncOpenAI
from hirag import HiRAG, QueryParam
from hirag.base import BaseKVStorage
from hirag._storage import NetworkXStorage
from hirag._utils import compute_args_hash
# os.environ["OPENAI_API_KEY"] = ""

# max test queries
DATASET = "mix"
if DATASET == "mix":
    MAX_QUERIES = 130
elif DATASET == "cs" or DATASET == "agriculture" or DATASET == "legal":
    MAX_QUERIES = 100

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract configurations
DEEPSEEK_MODEL = config['deepseek']['model']
DEEPSEEK_API_KEY = config['deepseek']['api_key']
DEEPSEEK_URL = config['deepseek']['base_url']

GLM_MODEL = config['glm']['model']
GLM_API_KEY = config['glm']['api_key']
GLM_URL = config['glm']['base_url']

OPENAI_MODEL = config['openai']['model']
OPENAI_API_KEY = config['openai']['api_key']
OPENAI_URL = config['openai']['base_url']

GLM_EMBEDDING_MODEL = str(config['glm'].get('embedding_model', 'embedding-3')).strip()
DEEPSEEK_EMBEDDING_MODEL = str(config['deepseek'].get('embedding_model', '')).strip()


def _from_env_or_config(env_name: str, config_value: str) -> str:
    env_value = os.getenv(env_name, '').strip()
    if env_value:
        return env_value
    return str(config_value or '').strip()


def _optional(value: str):
    return value if value else None


METRICS = {
    'embedding_provider': '',
    'glm_embedding_api_calls': 0,
    'deepseek_embedding_api_calls': 0,
    'deepseek_chat_api_calls': 0,
    'deepseek_chat_cache_hits': 0,
    'embedding_prompt_tokens': 0,
    'embedding_total_tokens': 0,
    'chat_prompt_tokens': 0,
    'chat_completion_tokens': 0,
    'chat_total_tokens': 0,
}


def reset_metrics():
    for k in list(METRICS.keys()):
        if k == 'embedding_provider':
            continue
        METRICS[k] = 0


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: Callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


def wrap_embedding_func_with_attrs(**kwargs):
    def final_decorator(func) -> EmbeddingFunc:
        return EmbeddingFunc(**kwargs, func=func)

    return final_decorator


@wrap_embedding_func_with_attrs(
    embedding_dim=config['model_params']['glm_embedding_dim'],
    max_token_size=config['model_params']['max_token_size'],
)
async def glm_embedding(texts: list[str]) -> np.ndarray:
    api_key = _from_env_or_config('GLM_API_KEY', GLM_API_KEY)
    base_url = _from_env_or_config('GLM_BASE_URL', GLM_URL)
    client = AsyncOpenAI(api_key=api_key, base_url=_optional(base_url))
    METRICS['glm_embedding_api_calls'] += 1
    response = await client.embeddings.create(input=texts, model=GLM_EMBEDDING_MODEL)
    usage = getattr(response, 'usage', None)
    if usage is not None:
        METRICS['embedding_prompt_tokens'] += int(getattr(usage, 'prompt_tokens', 0) or 0)
        METRICS['embedding_total_tokens'] += int(getattr(usage, 'total_tokens', 0) or 0)
    return np.array([d.embedding for d in response.data])


@wrap_embedding_func_with_attrs(
    embedding_dim=config['model_params']['glm_embedding_dim'],
    max_token_size=config['model_params']['max_token_size'],
)
async def deepseek_embedding(texts: list[str]) -> np.ndarray:
    api_key = _from_env_or_config('DEEPSEEK_API_KEY', DEEPSEEK_API_KEY)
    base_url = _from_env_or_config('DEEPSEEK_BASE_URL', DEEPSEEK_URL)
    client = AsyncOpenAI(api_key=api_key, base_url=_optional(base_url))
    METRICS['deepseek_embedding_api_calls'] += 1
    response = await client.embeddings.create(input=texts, model=DEEPSEEK_EMBEDDING_MODEL)
    usage = getattr(response, 'usage', None)
    if usage is not None:
        METRICS['embedding_prompt_tokens'] += int(getattr(usage, 'prompt_tokens', 0) or 0)
        METRICS['embedding_total_tokens'] += int(getattr(usage, 'total_tokens', 0) or 0)
    return np.array([d.embedding for d in response.data])


async def deepseek_model_if_cache(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    api_key = _from_env_or_config('DEEPSEEK_API_KEY', DEEPSEEK_API_KEY)
    base_url = _from_env_or_config('DEEPSEEK_BASE_URL', DEEPSEEK_URL)
    client = AsyncOpenAI(api_key=api_key, base_url=_optional(base_url))

    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})

    hashing_kv: BaseKVStorage = kwargs.pop('hashing_kv', None)
    messages.extend(history_messages)
    messages.append({'role': 'user', 'content': prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(DEEPSEEK_MODEL, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            METRICS['deepseek_chat_cache_hits'] += 1
            return cached['return']

    METRICS['deepseek_chat_api_calls'] += 1
    response = await client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        **kwargs,
    )
    usage = getattr(response, 'usage', None)
    if usage is not None:
        METRICS['chat_prompt_tokens'] += int(getattr(usage, 'prompt_tokens', 0) or 0)
        METRICS['chat_completion_tokens'] += int(getattr(usage, 'completion_tokens', 0) or 0)
        METRICS['chat_total_tokens'] += int(getattr(usage, 'total_tokens', 0) or 0)

    text = response.choices[0].message.content
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {'return': text, 'model': DEEPSEEK_MODEL}})
    return text


async def chat_model_if_cache(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    deepseek_key = _from_env_or_config('DEEPSEEK_API_KEY', DEEPSEEK_API_KEY)
    glm_key = _from_env_or_config('GLM_API_KEY', GLM_API_KEY)
    openai_key = _from_env_or_config('OPENAI_API_KEY', OPENAI_API_KEY)

    if deepseek_key:
        model = DEEPSEEK_MODEL
        api_key = deepseek_key
        base_url = _from_env_or_config('DEEPSEEK_BASE_URL', DEEPSEEK_URL)
    elif glm_key:
        model = GLM_MODEL
        api_key = glm_key
        base_url = _from_env_or_config('GLM_BASE_URL', GLM_URL)
    elif openai_key:
        model = OPENAI_MODEL
        api_key = openai_key
        base_url = _from_env_or_config('OPENAI_BASE_URL', OPENAI_URL)
    else:
        raise ValueError('Missing chat model key. Set one of DEEPSEEK_API_KEY/GLM_API_KEY/OPENAI_API_KEY')

    client = AsyncOpenAI(api_key=api_key, base_url=_optional(base_url))

    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})

    hashing_kv: BaseKVStorage = kwargs.pop('hashing_kv', None)
    messages.extend(history_messages)
    messages.append({'role': 'user', 'content': prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            METRICS['deepseek_chat_cache_hits'] += 1
            return cached['return']

    METRICS['deepseek_chat_api_calls'] += 1
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
    usage = getattr(response, 'usage', None)
    if usage is not None:
        METRICS['chat_prompt_tokens'] += int(getattr(usage, 'prompt_tokens', 0) or 0)
        METRICS['chat_completion_tokens'] += int(getattr(usage, 'completion_tokens', 0) or 0)
        METRICS['chat_total_tokens'] += int(getattr(usage, 'total_tokens', 0) or 0)

    text = response.choices[0].message.content
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {'return': text, 'model': model}})
    return text


def build_graph_func_for_ab() -> HiRAG:
    deepseek_key = _from_env_or_config('DEEPSEEK_API_KEY', DEEPSEEK_API_KEY)
    glm_key = _from_env_or_config('GLM_API_KEY', GLM_API_KEY)
    openai_key = _from_env_or_config('OPENAI_API_KEY', OPENAI_API_KEY)

    if not (deepseek_key or glm_key or openai_key):
        raise ValueError('Missing chat key. Set one of DEEPSEEK_API_KEY/GLM_API_KEY/OPENAI_API_KEY')

    if DEEPSEEK_EMBEDDING_MODEL:
        embedding_func = deepseek_embedding
        METRICS['embedding_provider'] = f'deepseek:{DEEPSEEK_EMBEDDING_MODEL}'
    elif glm_key:
        embedding_func = glm_embedding
        METRICS['embedding_provider'] = f'glm:{GLM_EMBEDDING_MODEL}'
    else:
        raise ValueError('No embedding provider available. Set deepseek.embedding_model or GLM_API_KEY')

    return HiRAG(
        working_dir=config['hirag']['working_dir'],
        enable_llm_cache=config['hirag']['enable_llm_cache'],
        embedding_func=embedding_func,
        best_model_func=chat_model_if_cache,
        cheap_model_func=chat_model_if_cache,
        enable_hierachical_mode=config['hirag']['enable_hierachical_mode'],
        embedding_batch_num=config['hirag']['embedding_batch_num'],
        embedding_func_max_async=config['hirag']['embedding_func_max_async'],
        enable_naive_rag=config['hirag']['enable_naive_rag'],
        graph_storage_cls=NetworkXStorage,
    )


def _extract_csv_section(context: str, section_name: str):
    if not context:
        return []
    marker = f'-----{section_name}-----'
    pos = context.find(marker)
    if pos < 0:
        return []
    tail = context[pos + len(marker):]
    m = re.search(r"```csv\s*(.*?)\s*```", tail, flags=re.DOTALL)
    if not m:
        return []
    csv_text = m.group(1).strip()
    if not csv_text:
        return []
    reader = csv.DictReader(StringIO(csv_text))
    return list(reader)


def parse_coverage_from_context(context: str) -> dict:
    entities_rows = _extract_csv_section(context, 'Entities')
    path_rows = _extract_csv_section(context, 'Reasoning Path')
    rel_rows = _extract_csv_section(context, 'Relationships')

    entity_names = [r.get('entity', '').strip() for r in entities_rows if r.get('entity')]
    path_texts = [
        (r.get('path', '') or r.get('description', '') or r.get('content', '')).strip()
        for r in path_rows
    ]
    path_texts = [x for x in path_texts if x]

    return {
        'entity_hits': len(entities_rows),
        'entity_unique_hits': len(set(entity_names)),
        'path_hits': len(path_rows),
        'path_nonempty_hits': len(path_texts),
        'relationship_hits': len(rel_rows),
    }


def is_complex_query(query: str) -> bool:
    q = query.lower()
    markers = [
        'why', 'how', 'relation', 'relationship', 'compare', 'difference', 'impact',
        '原因', '关系', '路径', '比较', '影响', '机制', '如何',
    ]
    if any(m in q for m in markers):
        return True
    return len(query) >= 40


def run_group_once(graph_func: HiRAG, query: str, enable_react: bool) -> dict:
    react_cfg = config['hirag'].get('react', {})
    answer_param = QueryParam(
        mode='hi',
        only_need_context=False,
        enable_react=enable_react,
        react_max_iter=int(react_cfg.get('max_iter', 1)),
        react_context_mode=str(react_cfg.get('context_mode', 'second_only')).strip(),
    )
    context_param = QueryParam(
        mode='hi',
        only_need_context=True,
        enable_react=enable_react,
        react_max_iter=int(react_cfg.get('max_iter', 1)),
        react_context_mode=str(react_cfg.get('context_mode', 'second_only')).strip(),
    )

    reset_metrics()
    t0 = time.perf_counter()
    answer = graph_func.query(query, param=answer_param)
    latency_sec = time.perf_counter() - t0
    answer_metrics = dict(METRICS)

    # Coverage is measured from retrieval context and is tracked separately from answer latency/token usage.
    reset_metrics()
    context = graph_func.query(query, param=context_param)
    coverage = parse_coverage_from_context(context or '')

    return {
        'answer': answer,
        'latency_sec': latency_sec,
        'usage': answer_metrics,
        'coverage': coverage,
    }


def _safe_mean(values):
    return float(mean(values)) if values else 0.0


def run_ab_eval(query_file: str, output_file: str, max_queries: int, quality_api: str, enable_quality_judge: bool):
    queries = []
    with open(query_file, 'r', encoding='utf-8') as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                q = (obj.get('input') or '').strip()
                if q:
                    queries.append(q)
            except json.JSONDecodeError as e:
                print(f'JSON decoding error in {query_file} line {line_number}: {e}')
    queries = queries[:max_queries]
    if not queries:
        raise ValueError(f'No valid queries found in {query_file}')

    graph_func = build_graph_func_for_ab()

    ab_rows = []
    answers_a = []
    answers_b = []
    for idx, query in enumerate(queries, start=1):
        print(f'[A/B] Running query {idx}/{len(queries)}')
        group_a = run_group_once(graph_func, query, enable_react=False)
        group_b = run_group_once(graph_func, query, enable_react=True)

        entity_delta = group_b['coverage']['entity_hits'] - group_a['coverage']['entity_hits']
        path_delta = group_b['coverage']['path_hits'] - group_a['coverage']['path_hits']

        row = {
            'query_id': idx,
            'query': query,
            'complex': is_complex_query(query),
            'group_a': group_a,
            'group_b': group_b,
            'delta': {
                'entity_hits': entity_delta,
                'path_hits': path_delta,
                'latency_sec': group_b['latency_sec'] - group_a['latency_sec'],
                'chat_total_tokens': int(group_b['usage'].get('chat_total_tokens', 0)) - int(group_a['usage'].get('chat_total_tokens', 0)),
                'embedding_total_tokens': int(group_b['usage'].get('embedding_total_tokens', 0)) - int(group_a['usage'].get('embedding_total_tokens', 0)),
            },
        }
        ab_rows.append(row)
        answers_a.append({'answer': group_a['answer']})
        answers_b.append({'answer': group_b['answer']})

    complex_rows = [r for r in ab_rows if r['complex']]
    improved_complex_rows = [
        r for r in complex_rows
        if (r['delta']['entity_hits'] > 0 or r['delta']['path_hits'] > 0)
    ]

    summary = {
        'num_queries': len(ab_rows),
        'num_complex_queries': len(complex_rows),
        'complex_improved_count': len(improved_complex_rows),
        'complex_improved_rate': (
            len(improved_complex_rows) / len(complex_rows) if complex_rows else 0.0
        ),
        'avg_entity_hits_a': _safe_mean([r['group_a']['coverage']['entity_hits'] for r in ab_rows]),
        'avg_entity_hits_b': _safe_mean([r['group_b']['coverage']['entity_hits'] for r in ab_rows]),
        'avg_path_hits_a': _safe_mean([r['group_a']['coverage']['path_hits'] for r in ab_rows]),
        'avg_path_hits_b': _safe_mean([r['group_b']['coverage']['path_hits'] for r in ab_rows]),
        'avg_latency_sec_a': _safe_mean([r['group_a']['latency_sec'] for r in ab_rows]),
        'avg_latency_sec_b': _safe_mean([r['group_b']['latency_sec'] for r in ab_rows]),
        'avg_chat_total_tokens_a': _safe_mean([int(r['group_a']['usage'].get('chat_total_tokens', 0)) for r in ab_rows]),
        'avg_chat_total_tokens_b': _safe_mean([int(r['group_b']['usage'].get('chat_total_tokens', 0)) for r in ab_rows]),
        'avg_embedding_total_tokens_a': _safe_mean([int(r['group_a']['usage'].get('embedding_total_tokens', 0)) for r in ab_rows]),
        'avg_embedding_total_tokens_b': _safe_mean([int(r['group_b']['usage'].get('embedding_total_tokens', 0)) for r in ab_rows]),
    }
    summary['react_effective_on_complex'] = bool(
        summary['complex_improved_rate'] >= 0.6 and summary['num_complex_queries'] > 0
    )

    result_payload = {
        'summary': summary,
        'details': ab_rows,
    }

    with open(output_file.replace('.jsonl', '_ab_result.json'), 'w', encoding='utf-8') as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    a_file = output_file.replace('.jsonl', '_groupA_hi_answer.jsonl')
    b_file = output_file.replace('.jsonl', '_groupB_hi_react_answer.jsonl')
    with jsonlines.open(a_file, mode='w') as w:
        for x in answers_a:
            w.write(x)
    with jsonlines.open(b_file, mode='w') as w:
        for x in answers_b:
            w.write(x)

    print(f'A/B results written to: {output_file.replace(".jsonl", "_ab_result.json")}')
    print(f'Group A answers written to: {a_file}')
    print(f'Group B answers written to: {b_file}')

    if enable_quality_judge:
        print(f'Running quality judge with api={quality_api}...')
        if quality_api == 'openai':
            eval_oq_openai(query_file=query_file, result1_file=a_file, result2_file=b_file, output_file_path=output_file)
            fetch_eval_result_openai(output_file=output_file)
        elif quality_api == 'deepseek':
            eval_oq_deepseek(query_file=query_file, result1_file=a_file, result2_file=b_file, output_file_path=output_file)
            fetch_eval_result_deepseek(output_file=output_file)
        elif quality_api == 'glm':
            eval_oq_glm(query_file=query_file, result1_file=a_file, result2_file=b_file, output_file_path=output_file)
            fetch_eval_result_glm(output_file=output_file)
        else:
            raise ValueError('ab_eval_api must be one of: openai, deepseek, glm')


def eval_oq_openai_batch(query_file, result1_file, result2_file, output_file_path):  # with original query
    client = OpenAI(base_url=OPENAI_URL, api_key=OPENAI_API_KEY)

    queries = []
    with open(query_file, "r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json_obj = json.loads(line)
                query = json_obj.get("input")
                queries.append(query)
            except json.JSONDecodeError as e:
                print(
                f"JSON decoding error in file {query_file} at line {line_number}: {e}"
                )
    queries = queries[:MAX_QUERIES]

    with open(result1_file, "r") as f:
        answers1 = f.readlines()
    answers1 = [json.loads(i)["answer"] for i in answers1][:MAX_QUERIES]

    with open(result2_file, "r") as f:
        answers2 = f.readlines()
    answers2 = [json.loads(i)["answer"] for i in answers2][:MAX_QUERIES]

    # placement of answer 1 and 2 is swapped
    queries += queries
    temp = copy.deepcopy(answers1)
    answers1 += answers2
    answers2 += temp

    requests = []
    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2)):
        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
        """

        prompt = f"""
        You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

        Here is the question:
        {query}

        Here are the two answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}

        Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

        Output your evaluation in the following JSON format:

        {{
            "Comprehensiveness": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Winner": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
            }}
        }}
        """

        request_data = {
            "custom_id": f"request-{i+1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": f"{OPENAI_MODEL}",
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
            },
        }

        requests.append(request_data)

    with jsonlines.open(output_file_path, mode="w") as writer:
        for request in requests:
            writer.write(request)

    print(f"Batch API requests written to {output_file_path}")

    batch_input_file = client.files.create(
        file=open(output_file_path, "rb"), purpose="batch"
    )
    batch_input_file_id = batch_input_file.id

    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "nightly eval job"},
    )

    print(f"Batch {batch.id} has been created.")
    return batch.id


def batch_eval_gq_openai(query_file, result1_file, result2_file, output_file_path):  # with generated query
    client = OpenAI(base_url=OPENAI_URL, api_key=OPENAI_API_KEY)

    with open(query_file, "r") as f:
        data = f.read()

    queries = re.findall(r"- Question \d+: (.+)", data)
    queries = queries[:MAX_QUERIES]

    with open(result1_file, "r") as f:
        answers1 = json.load(f)
    answers1 = [i["result"] for i in answers1]

    with open(result2_file, "r") as f:
        answers2 = json.load(f)
    answers2 = [i["result"] for i in answers2]

    # placement of answer 1 and 2 is swapped
    queries += queries
    temp = copy.deepcopy(answers1)
    answers1 += answers2
    answers2 += temp

    requests = []
    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2)):
        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
        """

        prompt = f"""
        You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

        Here is the question:
        {query}

        Here are the two answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}

        Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

        Output your evaluation in the following JSON format:

        {{
            "Comprehensiveness": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Winner": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
            }}
        }}
        """

        request_data = {
            "custom_id": f"request-{i+1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": f"{OPENAI_MODEL}",
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
            },
        }

        requests.append(request_data)

    with jsonlines.open(output_file_path, mode="w") as writer:
        for request in requests:
            writer.write(request)

    print(f"Batch API requests written to {output_file_path}")

    batch_input_file = client.files.create(
        file=open(output_file_path, "rb"), purpose="batch"
    )
    batch_input_file_id = batch_input_file.id

    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "nightly eval job"},
    )
    print(f"Batch {batch.id} has been created.")
    return batch.id


def eval_oq_glm(query_file, result1_file, result2_file, output_file_path):
    # Openai configuration
    client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_URL)
    
    queries = []
    with open(query_file, "r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json_obj = json.loads(line)
                query = json_obj.get("input")
                queries.append(query)
            except json.JSONDecodeError as e:
                print(
                f"JSON decoding error in file {query_file} at line {line_number}: {e}"
                )
    queries = queries[:MAX_QUERIES]

    with open(result1_file, "r") as f:
        answers1 = f.readlines()
    answers1 = [json.loads(i)["answer"] for i in answers1][:MAX_QUERIES]

    with open(result2_file, "r") as f:
        answers2 = f.readlines()
    answers2 = [json.loads(i)["answer"] for i in answers2][:MAX_QUERIES]

    # placement of answer 1 and 2 is swapped
    queries += queries
    temp = copy.deepcopy(answers1)
    answers1 += answers2
    answers2 += temp

    if not (len(queries) == len(answers1) == len(answers2)):
        print("Warning: the number of query and answer does not match, please check!")
        return

    evaluations = []

    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2), start=1):
        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
        """

        prompt = f"""
        You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

        Here is the question:
        {query}

        Here are the two answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}

        Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

        Output your evaluation in the following JSON format:

        {{
            "Comprehensiveness": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Winner": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
            }}
        }}
        """

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        # try:
        response = client.chat.completions.create(
            model=GLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=4095,
        )

        max_retries = 3  # max retry
        retry_delay = 1

        response = response.choices[0].message.content
        for attempt in range(max_retries):
            try:
                evaluation = json.loads('\n'.join(response.strip().split('\n')[1:-1]))
                evaluations.append(evaluation)
                print(f"Successfully evaluate {i}/{len(queries)}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(e)
                    print("Failed after maximum retries")

    with jsonlines.open(output_file_path.replace(".jsonl", "_result_glm.jsonl"), mode="w") as writer:
        for eval_item in evaluations:
            writer.write(eval_item)

    print(f"All evaluation completed, results are written to {output_file_path}")


def eval_oq_deepseek(query_file, result1_file, result2_file, output_file_path):
    # Openai configuration
    client = OpenAI(api_key=DEEPSEEK_API_KEY,base_url=DEEPSEEK_URL)
    
    queries = []
    with open(query_file, "r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json_obj = json.loads(line)
                query = json_obj.get("input")
                queries.append(query)
            except json.JSONDecodeError as e:
                print(
                f"JSON decoding error in file {query_file} at line {line_number}: {e}"
                )
    queries = queries[:MAX_QUERIES]

    with open(result1_file, "r") as f:
        answers1 = f.readlines()
    answers1 = [json.loads(i)["answer"] for i in answers1][:MAX_QUERIES]

    with open(result2_file, "r") as f:
        answers2 = f.readlines()
    answers2 = [json.loads(i)["answer"] for i in answers2][:MAX_QUERIES]

    # placement of answer 1 and 2 is swapped
    queries += queries
    temp = copy.deepcopy(answers1)
    answers1 += answers2
    answers2 += temp

    if not (len(queries) == len(answers1) == len(answers2)):
        print("Warning: the number of query and answer does not match, please check!")
        return

    evaluations = []

    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2), start=1):
        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
        """

        prompt = f"""
        You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

        Here is the question:
        {query}

        Here are the two answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}

        Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion. And you need to be very fair and have no bias towards the order.

        Output your evaluation in the following JSON format:

        {{
            "Comprehensiveness": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Winner": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
            }}
        }}
        """

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        # try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=6400,
        )

        max_retries = 3  # max retry
        retry_delay = 1

        response = response.choices[0].message.content
        for attempt in range(max_retries):
            try:
                evaluation = json.loads('\n'.join(response.strip().split('\n')[1:-1]))
                evaluations.append(evaluation)
                print(f"Successfully evaluate {i}/{len(queries)}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(e)
                    print("Failed after maximum retries")

    with jsonlines.open(output_file_path.replace(".jsonl", "_result_deepseek.jsonl"), mode="w") as writer:
        for eval_item in evaluations:
            writer.write(eval_item)

    print(f"All evaluation completed, results are written to {output_file_path}")


def eval_oq_openai(query_file, result1_file, result2_file, output_file_path):
    # Openai configuration
    client = OpenAI(base_url=OPENAI_URL, api_key=OPENAI_API_KEY)
    
    queries = []
    with open(query_file, "r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json_obj = json.loads(line)
                query = json_obj.get("input")
                queries.append(query)
            except json.JSONDecodeError as e:
                print(
                f"JSON decoding error in file {query_file} at line {line_number}: {e}"
                )
    queries = queries[:MAX_QUERIES]

    with open(result1_file, "r") as f:
        answers1 = f.readlines()
    answers1 = [json.loads(i)["answer"] for i in answers1][:MAX_QUERIES]

    with open(result2_file, "r") as f:
        answers2 = f.readlines()
    answers2 = [json.loads(i)["answer"] for i in answers2][:MAX_QUERIES]

    # placement of answer 1 and 2 is swapped
    queries += queries
    temp = copy.deepcopy(answers1)
    answers1 += answers2
    answers2 += temp

    if not (len(queries) == len(answers1) == len(answers2)):
        print("Warning: the number of query and answer does not match, please check!")
        return

    evaluations = []

    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2), start=1):
        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
        """

        prompt = f"""
        You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

        Here is the question:
        {query}

        Here are the two answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}

        Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion. And you need to be very fair and have no bias towards the order.

        Output your evaluation in the following JSON format:

        {{
            "Comprehensiveness": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Winner": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
            }}
        }}
        """

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        # try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=6400,
        )

        max_retries = 3  # max retry
        retry_delay = 1

        response = response.choices[0].message.content
        for attempt in range(max_retries):
            try:
                evaluation = json.loads('\n'.join(response.strip().split('\n')[1:-1]))
                evaluations.append(evaluation)
                print(f"Successfully evaluate {i}/{len(queries)}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(e)
                    print("Failed after maximum retries")

    with jsonlines.open(output_file_path.replace(".jsonl", "_result_openai.jsonl"), mode="w") as writer:
        for eval_item in evaluations:
            writer.write(eval_item)

    print(f"All evaluation completed, results are written to {output_file_path}")


def fetch_eval_result_glm(output_file):
    result = []
    with open(output_file.replace(".jsonl", "_result_glm.jsonl"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            result.append(item)
    
    result_0 = result[0]
    comprehensiveness = result_0['Comprehensiveness']['Winner']
    comprehensiveness_explanation = result_0['Comprehensiveness']['Explanation']
    empowerment = result_0['Empowerment']['Winner']
    empowerment_explanation = result_0['Empowerment']['Explanation']
    diversity = result_0['Diversity']['Winner']
    diversity_explanation = result_0['Diversity']['Explanation']
    overall_winner = result_0['Overall Winner']['Winner']
    overall_explanation = result_0['Overall Winner']['Explanation']

    print("===================================Comprehensiveness===================================")
    print(f"Winner:\n{comprehensiveness}")
    print(f"Explanation:\n{comprehensiveness_explanation}")
    print("======================================Empowerment======================================")
    print(f"Winner:\n{empowerment}")
    print(f"Explanation:\n{empowerment_explanation}")
    print("=======================================Diversity=======================================")
    print(f"Winner:\n{diversity}")
    print(f"Explanation:\n{diversity_explanation}")
    print("=========================================Winner=========================================")
    print(f"Winner:\n{overall_winner}")
    print(f"Explanation:\n{overall_explanation}")


    comprehensiveness_winner_ans1 = 0
    comprehensiveness_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Comprehensiveness']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            comprehensiveness_winner_ans1 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            comprehensiveness_winner_ans2 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            comprehensiveness_winner_ans2 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            comprehensiveness_winner_ans1 += 1
    empowerment_winner_ans1 = 0
    empowerment_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Empowerment']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            empowerment_winner_ans1 += 1
        elif item['Empowerment']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            empowerment_winner_ans2 += 1
        elif item['Empowerment']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            empowerment_winner_ans2 += 1
        elif item['Empowerment']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            empowerment_winner_ans1 += 1
    diversity_winner_ans1 = 0
    diversity_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Diversity']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            diversity_winner_ans1 += 1
        elif item['Diversity']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            diversity_winner_ans2 += 1
        elif item['Diversity']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            diversity_winner_ans2 += 1
        elif item['Diversity']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            diversity_winner_ans1 += 1
    overall_winner_ans1 = 0
    overall_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Overall Winner']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            overall_winner_ans1 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            overall_winner_ans2 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            overall_winner_ans2 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            overall_winner_ans1 += 1
    print("======================================Winner Accuracy=========================================")
    print("Comprehensiveness:")
    print(f"Answer 1: {comprehensiveness_winner_ans1 / len(result)}")
    print(f"Answer 2: {comprehensiveness_winner_ans2 / len(result)}")
    print("Empowerment:")
    print(f"Answer 1: {empowerment_winner_ans1 / len(result)}")
    print(f"Answer 2: {empowerment_winner_ans2 / len(result)}")
    print("Diversity:")
    print(f"Answer 1: {diversity_winner_ans1 / len(result)}")
    print(f"Answer 2: {diversity_winner_ans2 / len(result)}")
    print("Overall:")
    print(f"Answer 1: {overall_winner_ans1 / len(result)}")
    print(f"Answer 2: {overall_winner_ans2 / len(result)}")


def fetch_eval_result_deepseek(output_file):
    result = []
    with open(output_file.replace(".jsonl", "_result_deepseek.jsonl"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            result.append(item)
    
    result_0 = result[0]
    comprehensiveness = result_0['Comprehensiveness']['Winner']
    comprehensiveness_explanation = result_0['Comprehensiveness']['Explanation']
    empowerment = result_0['Empowerment']['Winner']
    empowerment_explanation = result_0['Empowerment']['Explanation']
    diversity = result_0['Diversity']['Winner']
    diversity_explanation = result_0['Diversity']['Explanation']
    overall_winner = result_0['Overall Winner']['Winner']
    overall_explanation = result_0['Overall Winner']['Explanation']

    print("===================================Comprehensiveness===================================")
    print(f"Winner:\n{comprehensiveness}")
    print(f"Explanation:\n{comprehensiveness_explanation}")
    print("======================================Empowerment======================================")
    print(f"Winner:\n{empowerment}")
    print(f"Explanation:\n{empowerment_explanation}")
    print("=======================================Diversity=======================================")
    print(f"Winner:\n{diversity}")
    print(f"Explanation:\n{diversity_explanation}")
    print("=========================================Winner=========================================")
    print(f"Winner:\n{overall_winner}")
    print(f"Explanation:\n{overall_explanation}")


    comprehensiveness_winner_ans1 = 0
    comprehensiveness_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Comprehensiveness']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            comprehensiveness_winner_ans1 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            comprehensiveness_winner_ans2 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            comprehensiveness_winner_ans2 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            comprehensiveness_winner_ans1 += 1
    empowerment_winner_ans1 = 0
    empowerment_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Empowerment']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            empowerment_winner_ans1 += 1
        elif item['Empowerment']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            empowerment_winner_ans2 += 1
        elif item['Empowerment']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            empowerment_winner_ans2 += 1
        elif item['Empowerment']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            empowerment_winner_ans1 += 1
    diversity_winner_ans1 = 0
    diversity_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Diversity']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            diversity_winner_ans1 += 1
        elif item['Diversity']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            diversity_winner_ans2 += 1
        elif item['Diversity']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            diversity_winner_ans2 += 1
        elif item['Diversity']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            diversity_winner_ans1 += 1
    overall_winner_ans1 = 0
    overall_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Overall Winner']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            overall_winner_ans1 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            overall_winner_ans2 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            overall_winner_ans2 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            overall_winner_ans1 += 1
    print("======================================Winner Accuracy=========================================")
    print("Comprehensiveness:")
    print(f"Answer 1: {comprehensiveness_winner_ans1 / len(result)}")
    print(f"Answer 2: {comprehensiveness_winner_ans2 / len(result)}")
    print("Empowerment:")
    print(f"Answer 1: {empowerment_winner_ans1 / len(result)}")
    print(f"Answer 2: {empowerment_winner_ans2 / len(result)}")
    print("Diversity:")
    print(f"Answer 1: {diversity_winner_ans1 / len(result)}")
    print(f"Answer 2: {diversity_winner_ans2 / len(result)}")
    print("Overall:")
    print(f"Answer 1: {overall_winner_ans1 / len(result)}")
    print(f"Answer 2: {overall_winner_ans2 / len(result)}")


def fetch_eval_result_openai(output_file):
    result = []
    with open(output_file.replace(".jsonl", "_result_openai.jsonl"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            result.append(item)
    
    result_0 = result[0]
    comprehensiveness = result_0['Comprehensiveness']['Winner']
    comprehensiveness_explanation = result_0['Comprehensiveness']['Explanation']
    empowerment = result_0['Empowerment']['Winner']
    empowerment_explanation = result_0['Empowerment']['Explanation']
    diversity = result_0['Diversity']['Winner']
    diversity_explanation = result_0['Diversity']['Explanation']
    overall_winner = result_0['Overall Winner']['Winner']
    overall_explanation = result_0['Overall Winner']['Explanation']

    print("===================================Comprehensiveness===================================")
    print(f"Winner:\n{comprehensiveness}")
    print(f"Explanation:\n{comprehensiveness_explanation}")
    print("======================================Empowerment======================================")
    print(f"Winner:\n{empowerment}")
    print(f"Explanation:\n{empowerment_explanation}")
    print("=======================================Diversity=======================================")
    print(f"Winner:\n{diversity}")
    print(f"Explanation:\n{diversity_explanation}")
    print("=========================================Winner=========================================")
    print(f"Winner:\n{overall_winner}")
    print(f"Explanation:\n{overall_explanation}")


    comprehensiveness_winner_ans1 = 0
    comprehensiveness_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Comprehensiveness']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            comprehensiveness_winner_ans1 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            comprehensiveness_winner_ans2 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            comprehensiveness_winner_ans2 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            comprehensiveness_winner_ans1 += 1
    empowerment_winner_ans1 = 0
    empowerment_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Empowerment']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            empowerment_winner_ans1 += 1
        elif item['Empowerment']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            empowerment_winner_ans2 += 1
        elif item['Empowerment']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            empowerment_winner_ans2 += 1
        elif item['Empowerment']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            empowerment_winner_ans1 += 1
    diversity_winner_ans1 = 0
    diversity_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Diversity']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            diversity_winner_ans1 += 1
        elif item['Diversity']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            diversity_winner_ans2 += 1
        elif item['Diversity']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            diversity_winner_ans2 += 1
        elif item['Diversity']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            diversity_winner_ans1 += 1
    overall_winner_ans1 = 0
    overall_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Overall Winner']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            overall_winner_ans1 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            overall_winner_ans2 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            overall_winner_ans2 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            overall_winner_ans1 += 1
    print("======================================Winner Accuracy=========================================")
    print("Comprehensiveness:")
    print(f"Answer 1: {comprehensiveness_winner_ans1 / len(result)}")
    print(f"Answer 2: {comprehensiveness_winner_ans2 / len(result)}")
    print("Empowerment:")
    print(f"Answer 1: {empowerment_winner_ans1 / len(result)}")
    print(f"Answer 2: {empowerment_winner_ans2 / len(result)}")
    print("Diversity:")
    print(f"Answer 1: {diversity_winner_ans1 / len(result)}")
    print(f"Answer 2: {diversity_winner_ans2 / len(result)}")
    print("Overall:")
    print(f"Answer 1: {overall_winner_ans1 / len(result)}")
    print(f"Answer 2: {overall_winner_ans2 / len(result)}")


def fetch_eval_result_openai_batch(batch_id, output_file):
    """
    Fetch evaluation result from OpenAI API.
    """ 
    client = OpenAI()
    batch_content = client.batches.retrieve(batch_id)
    print(batch_content.status)
    output_file_id = batch_content.output_file_id
    file_content = client.files.content(output_file_id)
    with open(output_file.replace(".jsonl", "_result_openai.jsonl"), 'wb') as file:
        file.write(file_content.content)

    result = []
    with open(output_file.replace(".jsonl", "_result_openai.jsonl"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            result.append(json.loads(line))
    
    result_0 = json.loads('\n'.join(result[0]['response']['body']['choices'][0]['message']['content'].strip().split('\n')[1:-1]))

    comprehensiveness = result_0['Comprehensiveness']['Winner']
    comprehensiveness_explanation = result_0['Comprehensiveness']['Explanation']
    empowerment = result_0['Empowerment']['Winner']
    empowerment_explanation = result_0['Empowerment']['Explanation']
    diversity = result_0['Diversity']['Winner']
    diversity_explanation = result_0['Diversity']['Explanation']
    overall_winner = result_0['Overall Winner']['Winner']
    overall_explanation = result_0['Overall Winner']['Explanation']

    print("===================================Comprehensiveness===================================")
    print(f"Winner:\n{comprehensiveness}")
    print(f"Explanation:\n{comprehensiveness_explanation}")
    print("======================================Empowerment======================================")
    print(f"Winner:\n{empowerment}")
    print(f"Explanation:\n{empowerment_explanation}")
    print("=======================================Diversity=======================================")
    print(f"Winner:\n{diversity}")
    print(f"Explanation:\n{diversity_explanation}")
    print("=========================================Winner=========================================")
    print(f"Winner:\n{overall_winner}")
    print(f"Explanation:\n{overall_explanation}")

    result_list = []
    for item in result:
        result_list.append(json.loads('\n'.join(item['response']['body']['choices'][0]['message']['content'].strip().split('\n')[1:-1])))
    
    comprehensiveness_winner_ans1 = 0
    comprehensiveness_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Comprehensiveness']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            comprehensiveness_winner_ans1 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            comprehensiveness_winner_ans2 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            comprehensiveness_winner_ans2 += 1
        elif item['Comprehensiveness']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            comprehensiveness_winner_ans1 += 1
    empowerment_winner_ans1 = 0
    empowerment_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Empowerment']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            empowerment_winner_ans1 += 1
        elif item['Empowerment']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            empowerment_winner_ans2 += 1
        elif item['Empowerment']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            empowerment_winner_ans2 += 1
        elif item['Empowerment']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            empowerment_winner_ans1 += 1
    diversity_winner_ans1 = 0
    diversity_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Diversity']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            diversity_winner_ans1 += 1
        elif item['Diversity']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            diversity_winner_ans2 += 1
        elif item['Diversity']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            diversity_winner_ans2 += 1
        elif item['Diversity']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            diversity_winner_ans1 += 1
    overall_winner_ans1 = 0
    overall_winner_ans2 = 0
    for i, item in enumerate(result):
        if item['Overall Winner']['Winner'] == 'Answer 1' and i <= MAX_QUERIES - 1:
            overall_winner_ans1 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 1' and i > MAX_QUERIES - 1:
            overall_winner_ans2 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 2' and i <= MAX_QUERIES - 1:
            overall_winner_ans2 += 1
        elif item['Overall Winner']['Winner'] == 'Answer 2' and i > MAX_QUERIES - 1:
            overall_winner_ans1 += 1
    print("======================================Winner Accuracy=========================================")
    print("Comprehensiveness:")
    print(f"Answer 1: {float(comprehensiveness_winner_ans1 / len(result_list))}")
    print(f"Answer 2: {float(comprehensiveness_winner_ans2 / len(result_list))}")
    print("Empowerment:")
    print(f"Answer 1: {float(empowerment_winner_ans1 / len(result_list))}")
    print(f"Answer 2: {float(empowerment_winner_ans2 / len(result_list))}")
    print("Diversity:")
    print(f"Answer 1: {float(diversity_winner_ans1 / len(result_list))}")
    print(f"Answer 2: {float(diversity_winner_ans2 / len(result_list))}")
    print("Overall:")
    print(f"Answer 1: {float(overall_winner_ans1 / len(result_list))}")
    print(f"Answer 2: {float(overall_winner_ans2 / len(result_list))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query_file", type=str, default=f"./datasets/{DATASET}/{DATASET}.jsonl")
    parser.add_argument("-r1", "--result1_file", type=str, default=f"./datasets/{DATASET}/{DATASET}_kag_result_deepseek.jsonl")
    parser.add_argument("-r2", "--result2_file", type=str, default=f"./datasets/{DATASET}/{DATASET}_hi_bridge_result_deepseek_pro.jsonl")
    parser.add_argument("-o", "--output_file", type=str, default=f"./datasets/{DATASET}/{DATASET}_eval_hi_graphtrag.jsonl")
    parser.add_argument("-m", "--mode", type=str, default="result", help="request or result or ab_run")
    parser.add_argument("-api", "--api", type=str, default="openai", help="openai or deepseek or glm")
    parser.add_argument("-b", "--batch_id", type=str, default="")
    parser.add_argument("--ab_max_queries", type=int, default=MAX_QUERIES)
    parser.add_argument("--ab_eval_api", type=str, default="deepseek", help="openai or deepseek or glm")
    parser.add_argument("--ab_enable_quality_judge", action="store_true")
    args = parser.parse_args()

    if args.ab_max_queries > 0:
        MAX_QUERIES = args.ab_max_queries

    if args.mode == "request":
        if args.api == "openai":
            batch_id = eval_oq_openai(query_file=args.query_file, 
                            result1_file=args.result1_file, 
                            result2_file=args.result2_file, 
                            output_file_path=args.output_file)
        elif args.api == "openai_batch":
            batch_id = eval_oq_openai_batch(query_file=args.query_file, 
                            result1_file=args.result1_file, 
                            result2_file=args.result2_file, 
                            output_file_path=args.output_file)
        elif args.api == "deepseek":
            batch_id = eval_oq_deepseek(query_file=args.query_file, 
                            result1_file=args.result1_file, 
                            result2_file=args.result2_file, 
                            output_file_path=args.output_file)
        elif args.api == "glm":
            batch_id = eval_oq_glm(query_file=args.query_file, 
                            result1_file=args.result1_file, 
                            result2_file=args.result2_file, 
                            output_file_path=args.output_file)
    elif args.mode == "result":
        if args.api == "openai_batch":
            fetch_eval_result_openai_batch(batch_id=args.batch_id, output_file=args.output_file)
        elif args.api == "openai":
            fetch_eval_result_openai(output_file=args.output_file)
        elif args.api == "deepseek":
            fetch_eval_result_deepseek(output_file=args.output_file)
        elif args.api == "glm":
            fetch_eval_result_glm(output_file=args.output_file)
    elif args.mode == "ab_run":
        run_ab_eval(
            query_file=args.query_file,
            output_file=args.output_file,
            max_queries=args.ab_max_queries,
            quality_api=args.ab_eval_api,
            enable_quality_judge=args.ab_enable_quality_judge,
        )
