import re

content = ""
with open("../log/graphrag_mix_retrieval_time.txt", 'r') as f:
    lines = f.readlines()
    for line in lines:
        content += str(line)
# 使用正则表达式提取时间
pattern = r"\[Retrieval Time: ([0-9.]+) seconds\]"
matches = re.findall(pattern, content)

# 将提取的时间转换为浮点数
times = [float(match) for match in matches]

# 计算平均值
average_time = sum(times) / len(times) if times else 0

print(len(times))
print(f"Average Retrieval Time: {average_time:.6f} seconds")