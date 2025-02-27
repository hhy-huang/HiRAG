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


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0

tokenizer = tiktoken.get_encoding("cl100k_base")

if __name__ == "__main__":
    # file_path = f"./datasets/{DATASET}/{DATASET}_unique_contexts.json"
    # with open(file_path, mode="r") as f:
    #     unique_contexts = json.load(f)
    #     TOTAL_TOKEN_COST += len(tokenizer.encode(str(unique_contexts[:100])))
    with open("./datasets/mix/mix_kag_result_deepseek.jsonl", "r") as f:
        doc = f.readlines()
        full_doc = ""
        for item in doc:
            full_doc += json.loads(item)['answer']
    
    TOTAL_TOKEN_COST += len(tokenizer.encode(str(full_doc)))
    logging.info(f"[Total token cost: {TOTAL_TOKEN_COST}]")
    # logging.info(f"[Total document num: {len(unique_contexts)}]")