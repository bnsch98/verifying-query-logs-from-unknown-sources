import gzip

# with gzip.open('part-00000.gz', 'r') as f:
#     print(type(f))

with open('part-00000.gz', 'rb') as f:
  data = f.read()

unzipped_data = gzip.decompress(data)

with open('unzipped', 'wb') as d:
  d.write(unzipped_data)