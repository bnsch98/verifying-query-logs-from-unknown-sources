import time
import sys
import gzip
import io
import pandas as pd
import json


input_path = '/mnt/ceph/storage/data-in-progress/data-research/web-search/archive-query-log/focused/corpus/full/2023-05-22/serps/part-00000.gz'

with open(input_path, 'rb') as input, open('output.jsonl', 'a') as output:
  for line in input:
    line = json.loads(line)
    line = json.dumps(line)
    output.write(line + '\n')

# d = sys.argv[1]
# e = sys.argv[2]
# start = time.time()
# result = 0

# for i in range(1000000):
#   result += i*i

# end = time.time()


# print(f"result: {result}")
# print(f"elapsed time: {(end-start)} s")
# print(f"Eingage: {d} \t {e}")


# input_path = "/mnt/ceph/storage/data-in-progress/data-research/web-search/archive-query-log/focused/corpus/full/2023-05-22/serps/part-00000.gz"

# result = []
# with open(input_path, 'rb') as f:
#   data = f.read()

# unzipped_data = gzip.decompress(data)

# decoded_data = io.BytesIO(unzipped_data)
# reader = jsonlines.Reader(decoded_data)

# for line in reader:
    
#   result.append(line)


# df = pd.DataFrame(result)
# print(df.head)
# print(df.columns)
# print(df['serp_query_text_url'])
