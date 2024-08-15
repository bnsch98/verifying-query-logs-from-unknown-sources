from ray import init
from ray.data import read_parquet

# Initialize Ray (and connect to cluster).
init()

input_path = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/results_query_length'

ds = read_parquet(
    input_path,
    concurrency=5
)


ds = ds.to_pandas()

print(ds.to_string())
