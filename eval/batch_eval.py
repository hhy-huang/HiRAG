import re
import json
import jsonlines
import argparse
import os
import time
import copy
import yaml
from openai import OpenAI
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
    parser.add_argument("-m", "--mode", type=str, default="result", help="request or result")
    parser.add_argument("-api", "--api", type=str, default="openai", help="openai or deepseek or glm")
    parser.add_argument("-b", "--batch_id", type=str, default="")
    args = parser.parse_args()

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
