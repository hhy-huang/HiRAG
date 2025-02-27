import os
import json
import argparse
from tqdm import tqdm
from hirag import HiRAG, QueryParam
os.environ["OPENAI_API_KEY"] = "***"
MAX_QUERIES = 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="mix")
    parser.add_argument("-m", "--mode", type=str, default="hi", help="hi / naive / hi_global / hi_local / hi_bridge / hi_nobridge")
    args = parser.parse_args()
    
    if args.mode == "naive":
        mode = True
    elif args.mode == "global" or "local":
        mode = False

    DATASET = args.dataset
    input_path = f"./datasets/{DATASET}/{DATASET}.jsonl"
    output_path = f"./datasets/{DATASET}/{DATASET}_{args.mode}_result.jsonl"
    graph_func = HiRAG(
        working_dir=f"./datasets/{DATASET}/work_dir", 
        enable_hierachical_mode=False, 
        embedding_func_max_async=4,
        enable_naive_rag=mode)

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
        tqdm.write(f"Q: {query}")
        answer = graph_func.query(query=query, param=QueryParam(mode=args.mode))
        tqdm.write(f"A: {answer} \n ################################################################################################")
        answer_list.append(answer)
    
    result_to_write = []
    for query, answer in zip(query_list, answer_list):
        result_to_write.append({"query": query, "answer": answer})
    with open(output_path, "w") as f:
        for item in result_to_write:
            f.write(json.dumps(item) + "\n")
        
