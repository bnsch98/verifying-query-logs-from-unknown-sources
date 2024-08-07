# import ir_datasets
# dataset = ir_datasets.load('msmarco-passage/train')

# # for query in dataset.queries_iter(): # Will download and extract MS-MARCO's queries.tar.gz the first time
# #     ...

# cnt = 0
# for query in dataset.queries_iter(): # Will download and extract MS-MARCO's queries.tar.gz the first time
#     cnt += 1
# print(f"size msmarco data set: {cnt}")

# dataset2 = ir_datasets.load("aol-ia")
# cnt = 0
# for query in dataset2.queries_iter():
#     cnt+= 1
# print(f"size aolia data set: {cnt}")#

import numpy as np

with open('/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/avg_query_chars.npy', 'rb') as f:
    arr = np.load(f)

print(f"avg query length: {arr}")


with open('/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/avg_query_words.npy', 'rb') as f:
    arr2 = np.load(f)

print(f"avg words per query: {arr2}")
