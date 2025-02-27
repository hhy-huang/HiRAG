# Retrieval-Augmented Generation with Hierarchical Knowledge (HiRAG)

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

Or if you want to employ HiRAG with DeepSeek, ChatGLM, or other third-party retrieval api, here are the examples in `./hi_Search_deepseek.py`, `./hi_search_glm.py`, and `./hi_search_openai.py`. The API keys and the LLM configurations can be set at `config.yaml`.


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