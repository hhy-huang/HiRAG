"""
This script extract queries of a dataset from UltraDomain

Example Usage:
    python extract_query.py -i xxx/mix.jsonl -o xxx/mix_query.jsonl

Example Output File (xxx/mix_query.jsonl):
    {"query": "This is a query"}
    {"query": "Another query"}
"""

import json
import argparse

def extract_query(input_file, output_file):
    print(f"Processing file: {input_file}")

    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            with open(output_file, "w", encoding="utf-8") as outfile:
                for line_number, line in enumerate(infile, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json_obj = json.loads(line)
                        query = json_obj.get("input")
                        outfile.write(json.dumps({"query": query}, ensure_ascii=False) + "\n")
                    except json.JSONDecodeError as e:
                        print(f"JSON decoding error in file {input_file} at line {line_number}: {e}")
    except FileNotFoundError:
        print(f"File not found: {input_file}")
    except Exception as e:
        print(f"An error occurred while processing file {input_file}: {e}")

    print(f"Finished processing file: {input_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the input file")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="Path to the output file")

    args = parser.parse_args()

    extract_query(args.input_file, args.output_file)
