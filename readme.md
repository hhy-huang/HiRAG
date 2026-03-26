<div align="center">

# 🤗 HiRAG: Retrieval-Augmented Generation with Hierarchical Knowledge

<p align="center">
    <img src="./imgs/icon.jpg" width="20%" style="max-width: 80;">
</p>

[<img align="center" src="https://img.shields.io/badge/arXiv-2503.10150 HiRAG -b31b1b" />](https://arxiv.org/abs/2503.10150)
[<img align="center" src="https://img.shields.io/badge/Dataset-TommyChien/UltraDomain -yellow" />](https://huggingface.co/datasets/TommyChien/UltraDomain)
[<img align="center" src="https://deepwiki.com/badge.svg" />](https://deepwiki.com/hhy-huang/HiRAG)

</div>



- This is the repo for the paper [HiRAG: Retrieval-Augmented Generation with Hierarchical Knowledge](https://arxiv.org/abs/2503.10150).
- **Accepted to [EMNLP 2025](https://2025.emnlp.org/) Findings!🎉**

## Model Pipeline

![image-20240129111934589](./imgs/hirag_ds_trans.drawio.png)

## Install

```bash
# remember clone this repo first
cd HiRAG
pip install -e .
```

## Quick Start

You can just utilize the following code to perform a query with HiRAG.

```python
graph_func = HiRAG(
    working_dir="./your_work_dir",
    enable_llm_cache=True,
    enable_hierachical_mode=True, 
    embedding_batch_num=6,
    embedding_func_max_async=8, # according to your machine
    enable_naive_rag=True
    )
# indexing
with open("path_to_your_context.txt", "r") as f:
    graph_func.insert(f.read())
# retrieval & generation
print("Perform hi search:")
print(graph_func.query("The question you want to ask?", param=QueryParam(mode="hi")))
```

Or if you want to employ HiRAG with DeepSeek, ChatGLM, or other third-party retrieval api, here are the examples in `./hi_Search_deepseek.py`, `./hi_Search_glm.py`, and `./hi_Search_openai.py`. The API keys and the LLM configurations can be set at `./config.yaml`.


## Evaluation

We take the procedure in Mix dataset as an example.

```shell
cd ./HiRAG/eval
```

1. Extract context from original QA datasets.
```shell
python extract_context.py -i ./datasets/mix -o ./datasets/mix
```

2. Insert context to Graph Database.
```shell
python insert_context_deepseek.py
```

Note that the script `insert_context_deepseek.py` is for the setting of generation with DeepSeek-v3 api, you can replace that with `insert_context_openai.py` or `insert_context_glm.py`.

3. Test with different versions of HiRAG.
```shell
# there are different retrieval options
# If you want to employ HiRAG approach, just run:
python test_deepseek.py -d mix -m hi
# If you want to employ naive RAG approach, just run:
python test_deepseek.py -d mix -m naive
# If you want to employ HiRAG approach w/o bridge, just run:
python test_deepseek.py -d mix -m hi_nobridge
# If you want to employ HiRAG approach with retrieving only local knowledge, just run:
python test_deepseek.py -d mix -m hi_local
# If you want to employ HiRAG approach with retrieving only global knowledge, just run:
python test_deepseek.py -d mix -m hi_global
# If you want to employ HiRAG approach with retrieving only bridge knowledge, just run:
python test_deepseek.py -d mix -m hi_bridge
```

Note that the dataset `mix` can be replaced to any other datasets in [Hugging Face link](https://huggingface.co/datasets/TommyChien/UltraDomain/tree/main). And the script `test_deepseek.py` is for the setting of generation with DeepSeek-v3 api, you can replace that with `test_openai.py` or `test_glm.py`.

4. Evaluate the generated answers.

First step, request for evaluations.
```shell
python batch_eval.py -m request -api openai
python batch_eval.py -m request -api deepseek
```

Second step, get the results.
```shell
python batch_eval.py -m result -api openai
python batch_eval.py -m result -api deepseek
```

## ReAct A/B Evaluation

This section is for comparing your ReAct integration with minimal setup and controllable API cost.

### Goal

- Group A: native `mode="hi"`
- Group B: `mode="hi" + enable_react=True`
- Compare:
    - answer quality
    - retrieval coverage (entity/path hits)
    - latency and token cost

### 1. Prerequisites

Run from repository root (`HiRAG/`), because `eval/batch_eval.py` reads `config.yaml` from the current working directory.

Set at least one chat API key and one embedding provider:

- Chat key (any one is enough):
    - `DEEPSEEK_API_KEY` or `GLM_API_KEY` or `OPENAI_API_KEY`
- Embedding provider:
    - `deepseek.embedding_model` in `config.yaml`, or
    - `GLM_API_KEY` for GLM embedding fallback

### 2. Lowest-cost smoke test (recommended first)

Run only 1 query and skip external quality judge to save tokens.

```shell
cd HiRAG
python eval/batch_eval.py \
    -m ab_run \
    -q ./eval/datasets/mix/mix.jsonl \
    -o ./eval/datasets/mix/mix_eval_ab_quick.jsonl \
    --ab_max_queries 1
```

### 3. Small batch A/B

Run 3-10 queries for a more stable trend while still controlling cost.

```shell
python eval/batch_eval.py \
    -m ab_run \
    -q ./eval/datasets/mix/mix.jsonl \
    -o ./eval/datasets/mix/mix_eval_ab.jsonl \
    --ab_max_queries 5
```

### 4. Optional answer-quality judge (extra token cost)

This step uses an external LLM judge on Group A/B answers.

```shell
python eval/batch_eval.py \
    -m ab_run \
    -q ./eval/datasets/mix/mix.jsonl \
    -o ./eval/datasets/mix/mix_eval_ab.jsonl \
    --ab_max_queries 5 \
    --ab_enable_quality_judge \
    --ab_eval_api deepseek
```

`--ab_eval_api` can be `openai`, `deepseek`, or `glm`.

### 5. Output files

Given `-o ./eval/datasets/mix/mix_eval_ab.jsonl`, outputs include:

- `./eval/datasets/mix/mix_eval_ab_ab_result.json`
    - Summary + per-query A/B details
    - Coverage metrics (`entity_hits`, `path_hits`)
    - Latency and token deltas
    - Complex-query improvement rate
- `./eval/datasets/mix/mix_eval_ab_groupA_hi_answer.jsonl`
- `./eval/datasets/mix/mix_eval_ab_groupB_hi_react_answer.jsonl`

If quality judge is enabled, judge result files are also generated with suffixes like `_result_deepseek.jsonl`.

### 6. How to interpret effectiveness

ReAct is effective if Group B shows consistent gains on complex questions in key retrieval coverage, especially:

- higher `path_hits` and/or `entity_hits`
- acceptable latency increase
- acceptable token increase

In `*_ab_result.json`, check:

- `summary.complex_improved_rate`
- `summary.avg_path_hits_a` vs `summary.avg_path_hits_b`
- `summary.avg_entity_hits_a` vs `summary.avg_entity_hits_b`
- `summary.avg_latency_sec_a` vs `summary.avg_latency_sec_b`
- `summary.avg_chat_total_tokens_a` vs `summary.avg_chat_total_tokens_b`

## Results

### Compare with Naive RAG:

With the config `output_file` set as `f"./datasets/{DATASET}/{DATASET}_eval_hi_naive.jsonl"`, just run the command:
```
python batch_eval.py -m result -api openai
```

| Dataset |  Dimension  | NaiveRAG % | HiRAG % |
|----------:|:--------:|--------------:|----------------:|
|         Mix||||
|           |Comprehensiveness|           16.6|             **83.4**|
|           |Empowerment|           11.6|             **88.4**|
|           |Diversity|           12.7|             **87.3**|
|           |Overall|           12.4|             **87.6**|
|        CS||||
|           |Comprehensiveness|           30.0|             **70.0**|
|           |Empowerment|           29.0|             **71.0**|
|           |Diversity|           14.5|             **85.5**|
|           |Overall|           26.5|             **73.5**|
|        Legal||||
|           |Comprehensiveness|           32.5|             **67.5**|
|           |Empowerment|           25.0|             **75.0**|
|           |Diversity|           22.0|             **78.0**|
|           |Overall|           22.5|             **74.5**|
|        Agriculture||||
|           |Comprehensiveness|           34.0|             **66.0**|
|           |Empowerment|           31.0|             **69.0**|
|           |Diversity|           21.0|             **79.0**|
|           |Overall|           28.5|             **71.5**|


### Compare with GraphRAG:

With the config `output_file` set as `f"./datasets/{DATASET}/{DATASET}_eval_hi_graphrag.jsonl"`, just run the command:
```
python batch_eval.py -m result -api openai
```

| Dataset |  Dimension  | GraphRAG % | HiRAG % |
|----------:|:--------:|--------------:|----------------:|
|         Mix||||
|           |Comprehensiveness|           42.1|             **57.9**|
|           |Empowerment|           35.1|             **64.9**|
|           |Diversity|           40.5|             **59.5**|
|           |Overall|           35.9|             **64.1**|
|        CS||||
|           |Comprehensiveness|           40.5|             **59.5**|
|           |Empowerment|           38.5|             **61.5**|
|           |Diversity|           30.5|             **69.5**|
|           |Overall|           36.0|             **64.0**|
|        Legal||||
|           |Comprehensiveness|           48.5|             **51.5**|
|           |Empowerment|           43.5|             **56.5**|
|           |Diversity|           47.0|             **53.0**|
|           |Overall|           45.5|             **54.5**|
|        Agriculture||||
|           |Comprehensiveness|           49.0|             **51.0**|
|           |Empowerment|           48.5|             **51.5**|
|           |Diversity|           45.5|             **54.5**|
|           |Overall|           46.0|             **54.0**|

### Compare with LightRAG:

With the config `output_file` set as `f"./datasets/{DATASET}/{DATASET}_eval_hi_lightrag.jsonl"`, just run the command:
```
python batch_eval.py -m result -api openai
```

| Dataset |  Dimension  | LightRAG % | HiRAG % |
|----------:|:--------:|--------------:|----------------:|
|         Mix||||
|           |Comprehensiveness|           36.8|             **63.2**|
|           |Empowerment|           34.9|             **65.1**|
|           |Diversity|           34.1|             **65.9**|
|           |Overall|           34.1|             **65.9**|
|        CS||||
|           |Comprehensiveness|           44.5|             **55.5**|
|           |Empowerment|           41.5|             **58.5**|
|           |Diversity|           33.0|             **67.0**|
|           |Overall|           41.0|             **59.0**|
|        Legal||||
|           |Comprehensiveness|           49.0|             **51.0**|
|           |Empowerment|           43.5|             **56.5**|
|           |Diversity|           **63.0**|             37.0|
|           |Overall|           48.0|             **52.0**|
|        Agriculture||||
|           |Comprehensiveness|           38.5|             **61.5**|
|           |Empowerment|           36.5|             **63.5**|
|           |Diversity|           37.5|             **62.5**|
|           |Overall|           38.5|             **61.5**|

### Compare with FastGraphRAG:

With the config `output_file` set as `f"./datasets/{DATASET}/{DATASET}_eval_hi_fastgraphrag.jsonl"`, just run the command:
```
python batch_eval.py -m result -api openai
```

| Dataset |  Dimension  | FastGraphRAG % | HiRAG % |
|----------:|:--------:|--------------:|----------------:|
|         Mix||||
|           |Comprehensiveness|           0.8|             **99.2**|
|           |Empowerment|           0.8|             **99.2**|
|           |Diversity|           0.8|             **99.2**|
|           |Overall|           0.8|             **99.2**|
|        CS||||
|           |Comprehensiveness|           0.0|             **100.0**|
|           |Empowerment|           0.0|             **100.0**|
|           |Diversity|           0.5|             **99.5**|
|           |Overall|           0.0|             **100.0**|
|        Legal||||
|           |Comprehensiveness|           1.0|             **99.0**|
|           |Empowerment|           0.0|             **100.0**|
|           |Diversity|              1.5|             **98.5**|
|           |Overall|           0.0|             **100.0**|
|        Agriculture||||
|           |Comprehensiveness|           0.0|             **100.0**|
|           |Empowerment|           0.0|             **100.0**|
|           |Diversity|           0.0|             **100.0**|
|           |Overall|           0.0|             **100.0**|

### Compare with KAG:

With the config `output_file` set as `f"./datasets/{DATASET}/{DATASET}_eval_hi_kag.jsonl"`, just run the command:
```
python batch_eval.py -m result -api openai
```

| Dataset |  Dimension  | KAG % | HiRAG % |
|----------:|:--------:|--------------:|----------------:|
|         Mix||||
|           |Comprehensiveness|           2.3|             **97.7**|
|           |Empowerment|           3.5|             **96.5**|
|           |Diversity|           3.8|             **96.2**|
|           |Overall|           2.3|             **97.7**|
|        CS||||
|           |Comprehensiveness|           1.0|             **99.0**|
|           |Empowerment|           4.5|             **95.5**|
|           |Diversity|           5.0|             **95.0**|
|           |Overall|           1.5|             **98.5**|
|        Legal||||
|           |Comprehensiveness|           16.5|             **83.5**|
|           |Empowerment|           9.0|             **91.0**|
|           |Diversity|              11.0|             **89.0**|
|           |Overall|           8.5|             **91.5**|
|        Agriculture||||
|           |Comprehensiveness|           5.0|             **95.0**|
|           |Empowerment|           5.0|             **95.0**|
|           |Diversity|           3.5|             **96.5**|
|           |Overall|           0.0|             **100.0**|

## Acknowledgement
We gratefully acknowledge the use of the following open-source projects in our work:
- [nano-graphrag](https://github.com/gusye1234/nano-graphrag): a simple, easy-to-hack GraphRAG implementation

- [RAPTOR](https://github.com/parthsarthi03/raptor): a novel approach to retrieval-augmented language models by constructing a recursive tree structure from documents.

## 💡 Cite Us
```python
@article{huang2025retrieval,
  title={Retrieval-Augmented Generation with Hierarchical Knowledge},
  author={Huang, Haoyu and Huang, Yongfeng and Yang, Junjie and Pan, Zhenyu and Chen, Yongqiang and Ma, Kaili and Chen, Hongzhi and Cheng, James},
  journal={arXiv preprint arXiv:2503.10150},
  year={2025}
}
```
## 🌟 Star History

![Star History Chart](https://api.star-history.com/svg?repos=hhy-huang/HiRAG&type=Date)