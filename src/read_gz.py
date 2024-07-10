import pandas as pd
import jsonlines
import gzip
import os
import io


result = []
with open('part-00000.gz', 'rb') as f:
  data = f.read()

unzipped_data = gzip.decompress(data)

decoded_data = io.BytesIO(unzipped_data)
reader = jsonlines.Reader(decoded_data)

for line in reader:
    
  result.append(line)


df = pd.DataFrame(result)
print(df.head)
print(df.columns)
print(df['serp_query_text_url'])

# print(df['serp_query_text_url_language'][0:10])