
import csv
from tqdm import tqdm

dir_name = "data/WN18RR/"


# # TSVファイルの読み込みと特定のパラメータの抽出
# tsv_file_path = dir_name + 'train.tsv'  # TSVファイルのパスを指定
# # tsv_file_path = dir_name + 'dev.tsv'
# # tsv_file_path = dir_name + 'test.tsv'

# triples = []

# # TSVファイルの読み取り
# with open(tsv_file_path, 'r', newline='', encoding='utf-8') as tsv_file:
#     tsv_reader = csv.reader(tsv_file, delimiter='\t')  # タブ文字で区切る
#     for row in tsv_reader:
#         triples.append(row)

# # print("triples = ", triples[0:5])

# file_path1 = dir_name + 'entity2text.txt'

# entity = []

# with open(file_path1, "r", encoding="utf-8") as f:
#     list = f.readlines()
#     for l in list:
#         entity.append(l.split("\t"))

# # print("entity = ", entity[0:5])

# file_path2 = dir_name + 'relation2text.txt'

# relation = []

# with open(file_path2, "r", encoding="utf-8") as f:
#     list = f.readlines()
#     for l in list:
#         relation.append(l.split("\t"))

# # print("relation = ", relation[0:5])

# n = [0, 0, 0]
# str_triples = []

# # for triple in tqdm(triples):
# #     for e in entity:
# #         if str(triple[0]) == str(e[0]):
# #             n[0] += 1
# #             str_head = "[CLS]" + str(e[1]) + "[SEP]"
# #         if str(triple[2]) == str(e[0]):
# #             n[1] += 1
# #             str_tail = str(e[1]) + "[SEP]"
# #     for r in relation:
# #         if str(triple[1]) == str(r[0]):
# #             n[2] += 1
# #             str_relation = str(r[1]) + "[SEP]"
# #     str_triple = str_head + str_relation + str_tail
# #     str_triple = str_triple.replace('\n', '')
# #     str_triples.append(str_triple)

# # print("n = ", n)

# for triple in tqdm(triples):
#     for e in entity:
#         if str(triple[0]) == str(e[0]):
#             n[0] += 1
#             str_head = str(e[1]) + "[SEP]"
#         if str(triple[2]) == str(e[0]):
#             n[1] += 1
#             str_tail = str(e[1])
#     for r in relation:
#         if str(triple[1]) == str(r[0]):
#             n[2] += 1
#             str_relation = str(r[1]) + "[SEP]"
#     str_triple = str_head + str_relation + str_tail
#     str_triple = str_triple.replace('\n', '')
#     str_triples.append(str_triple)

# train_str_triples = str_triples

# print("train_str_triples = ", train_str_triples[0:5])



# # TSVファイルの読み込みと特定のパラメータの抽出
# # tsv_file_path = dir_name + 'train.tsv'  # TSVファイルのパスを指定
# tsv_file_path = dir_name + 'dev.tsv'
# # tsv_file_path = dir_name + 'test.tsv'

# triples = []

# # TSVファイルの読み取り
# with open(tsv_file_path, 'r', newline='', encoding='utf-8') as tsv_file:
#     tsv_reader = csv.reader(tsv_file, delimiter='\t')  # タブ文字で区切る
#     for row in tsv_reader:
#         triples.append(row)

# # print("triples = ", triples[0:5])

# file_path1 = dir_name + 'entity2text.txt'

# entity = []

# with open(file_path1, "r", encoding="utf-8") as f:
#     list = f.readlines()
#     for l in list:
#         entity.append(l.split("\t"))

# # print("entity = ", entity[0:5])

# file_path2 = dir_name + 'relation2text.txt'

# relation = []

# with open(file_path2, "r", encoding="utf-8") as f:
#     list = f.readlines()
#     for l in list:
#         relation.append(l.split("\t"))

# # print("relation = ", relation[0:5])

# str_triples = []

# for triple in tqdm(triples):
#     for e in entity:
#         if str(triple[0]) == str(e[0]):
#             n[0] += 1
#             str_head = str(e[1]) + "[SEP]"
#         if str(triple[2]) == str(e[0]):
#             n[1] += 1
#             str_tail = str(e[1])
#     for r in relation:
#         if str(triple[1]) == str(r[0]):
#             n[2] += 1
#             str_relation = str(r[1]) + "[SEP]"
#     str_triple = str_head + str_relation + str_tail
#     str_triple = str_triple.replace('\n', '')
#     str_triples.append(str_triple)

# dev_str_triples = str_triples

# print("dev_str_triples = ", dev_str_triples[0:5])



# # TSVファイルの読み込みと特定のパラメータの抽出
# # tsv_file_path = dir_name + 'train.tsv'  # TSVファイルのパスを指定
# # tsv_file_path = dir_name + 'dev.tsv'
# tsv_file_path = dir_name + 'test.tsv'

# triples = []

# # TSVファイルの読み取り
# with open(tsv_file_path, 'r', newline='', encoding='utf-8') as tsv_file:
#     tsv_reader = csv.reader(tsv_file, delimiter='\t')  # タブ文字で区切る
#     for row in tsv_reader:
#         triples.append(row)

# # print("triples = ", triples[0:5])

# file_path1 = dir_name + 'entity2text.txt'

# entity = []

# with open(file_path1, "r", encoding="utf-8") as f:
#     list = f.readlines()
#     for l in list:
#         entity.append(l.split("\t"))

# # print("entity = ", entity[0:5])

# file_path2 = dir_name + 'relation2text.txt'

# relation = []

# with open(file_path2, "r", encoding="utf-8") as f:
#     list = f.readlines()
#     for l in list:
#         relation.append(l.split("\t"))

# # print("relation = ", relation[0:5])

# str_triples = []

# for triple in tqdm(triples):
#     for e in entity:
#         if str(triple[0]) == str(e[0]):
#             n[0] += 1
#             str_head = str(e[1]) + "[SEP]"
#         if str(triple[2]) == str(e[0]):
#             n[1] += 1
#             str_tail = str(e[1])
#     for r in relation:
#         if str(triple[1]) == str(r[0]):
#             n[2] += 1
#             str_relation = str(r[1]) + "[SEP]"
#     str_triple = str_head + str_relation + str_tail
#     str_triple = str_triple.replace('\n', '')
#     str_triples.append(str_triple)

# test_str_triples = str_triples
# print("test_str_triples = ", test_str_triples[0:5])
# print("n = ", n)



# # data_neme = "all"

# # file_path3 = dir_name + data_neme + '_triples.txt'

# # with open(file_path3, "w", encoding="utf-8") as f:
# #     f.write('\n'.join(train_str_triples+dev_str_triples+test_str_triples))


# data_neme = "train"

# file_path3 = dir_name + data_neme + '_triples.txt'

# with open(file_path3, "w", encoding="utf-8") as f:
#     f.write('\n'.join(train_str_triples))


# data_neme = "dev"

# file_path3 = dir_name + data_neme + '_triples.txt'

# with open(file_path3, "w", encoding="utf-8") as f:
#     f.write('\n'.join(dev_str_triples))


# data_neme = "test"

# file_path3 = dir_name + data_neme + '_triples.txt'

# with open(file_path3, "w", encoding="utf-8") as f:
#     f.write('\n'.join(test_str_triples))





file_path = dir_name + '/train_triples.txt'
with open(file_path) as f:
    train_triples = [s.rstrip() for s in f.readlines()]

file_path = dir_name + '/dev_triples.txt'
with open(file_path) as f:
    dev_triples = [s.rstrip() for s in f.readlines()]

train_dev_triples = train_triples + dev_triples

data_neme = "train_dev"

file_path3 = dir_name + data_neme + '_triples.txt'

with open(file_path3, "w", encoding="utf-8") as f:
    f.write('\n'.join(train_dev_triples))
