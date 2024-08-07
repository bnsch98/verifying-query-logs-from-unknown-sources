import numpy as np
import matplotlib.pyplot as plt


with open('/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/avg_query_chars_full.npy', 'rb') as f:
    avg_query_chars_full = np.load(f)

with open('/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/hist_chars.npy', 'rb') as f:
    hist_chars = np.load(f)

with open('/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/hist_words.npy', 'rb') as f:
    hist_words = np.load(f)

with open('/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/avg_query_words_full.npy', 'rb') as f:
    avg_query_words_full = np.load(f)

# print(len(hist_chars))
# for i in hist_chars[0:200]:
#     print(i)
# plt.hist(hist_chars[0:100])  # , bins=np.arange(start=1, stop=101, step=1)
# plt.show()
