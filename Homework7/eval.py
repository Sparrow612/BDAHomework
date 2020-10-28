import re

expected = list()
with open('FB15k/test.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        expected.append(re.split('\\s+', line))

actual = list()

with open('completed.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        actual.append(re.split('\\s+', line))

success_num = 0
for i in range(len(expected)):
    exp = expected[i]
    act = actual[i]
    if exp[0] == act[0] and exp[1] == act[1] and exp[2] == act[2]:
        success_num += 1
print(success_num / len(expected))
