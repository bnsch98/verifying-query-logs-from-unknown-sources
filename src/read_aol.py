import gzip
import pandas as pd

# with gzip.open('/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/corpus-aol-query-log/corpus-aol-query-log/user-ct-test-collection-01.txt.gz', 'rb') as f:
#     file_content = f.read()
#     print(file_content)

df = pd.read_csv("/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/corpus-aol-query-log/corpus-aol-query-log/user-ct-test-collection-01.txt.gz", compression='gzip', delimiter='\t', on_bad_lines='skip', header=None, low_memory=False)
# Inspect data types
print(df.dtypes)
print(df.head)
# print(df.describe)