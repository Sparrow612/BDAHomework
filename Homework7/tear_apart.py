import re
import random

triples = list()
results = list()

with open('FB15k/test.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        triples.append(re.split('\\s+', line))  # 匹配任意空白字符
# 随机替换一项为'_blk_'
for triple in triples:
    triple[random.randint(0, 2)] = '_blk_'  # 随机替换一项为_blk_
    results.append('\t'.join(triple))

with open('priest.txt', 'w') as f:
    f.write('\n'.join(results))
