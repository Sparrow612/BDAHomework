import codecs
import json
import numpy
import re

# 读取待补完的三元组，切分

blk = '_blk_'

triples = list()
with open('priest.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        triples.append(re.split('\\s+', line))

# 读取实体、关系和其对应的id
entity2id = {}
relation2id = {}
with open('FB15k/entity2id.txt', 'r') as f1, open(
        'FB15k/relation2id.txt', 'r') as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    for line in lines1:
        line = line.strip().split('\t')
        entity2id[line[0]] = line[1]

    for line in lines2:
        line = line.strip().split('\t')
        relation2id[line[0]] = line[1]

# 读取实体id、关系id与其对应的向量
entity_dict = {}
relation_dict = {}
with codecs.open('entity_50dim_batch400') as e_f:
    lines = e_f.readlines()
    for line in lines:
        entity_name, embedding = line.strip().split('\t')
        embedding = json.loads(embedding)
        entity_dict[entity_name] = embedding

with codecs.open('entity_50dim_batch400') as r_f:
    lines = r_f.readlines()
    for line in lines:
        relation, embedding = line.strip().split('\t')
        embedding = json.loads(embedding)
        relation_dict[relation] = embedding

# 根据h+r=t，寻找使||h+r-t||最小的进行补全
# 三元组是(h,t,r)
for triple in triples:
    # 如果h缺失
    if triple[0] == blk:
        h = ''
        min_distance = float('inf')
        for entity_name in entity2id:
            if entity_name == triple[1]:
                continue
            entity_vec = entity_dict[entity2id[entity_name]]
            distance = numpy.linalg.norm(
                numpy.array(entity_vec) + numpy.array(relation_dict[relation2id[triple[2]]]) - numpy.array(
                    entity_dict[entity2id[triple[1]]]))
            if distance < min_distance:
                min_distance = distance
                h = entity_name
        triple[0] = h
        print(h)
    # 如果t缺失
    elif triple[1] == blk:
        t = ''
        min_distance = float('inf')
        for entity_name in entity2id:
            if entity_name == triple[0]:
                continue
            entity_vec = entity_dict[entity2id[entity_name]]
            distance = numpy.linalg.norm(numpy.array(entity_dict[entity2id[triple[0]]]) + numpy.array(
                relation_dict[relation2id[triple[2]]]) - numpy.array(entity_vec))
            if distance < min_distance:
                min_distance = distance
                t = entity_name
        triple[1] = t
        print(t)
    # 如果r缺失
    elif triple[2] == blk:
        r = ''
        min_distance = float('inf')
        for relation_name in relation2id:
            relation_vec = relation_dict[relation2id[relation_name]]
            distance = numpy.linalg.norm(
                numpy.array(entity_dict[entity2id[triple[0]]]) + numpy.array(relation_vec) - numpy.array(
                    entity_dict[entity2id[triple[1]]]))
            if distance < min_distance:
                min_distance = distance
                r = relation_name
        triple[2] = r
        print(r)


results = list()
for triple in triples:
    results.append(' '.join(triple))
with open('completed.txt', 'w') as f:
    f.write('\n'.join(results))
