from ray import init
from ray.data import range
import ray

# Initialize Ray (and connect to cluster).
init()

input_path = "/mnt/ceph/storage/data-in-progress/data-research/web-search/archive-query-log/focused/corpus/full/2023-05-22/serps/part.jsonl.gz"

ds = ray.data.read_json(input_path, arrow_open_stream_args={"compression": "gzip"}) #, file_extensions=['gz','json','jsonl']

print(ds.schema())


