from ray import init
import ray
from ray.data import read_parquet
# Initialize Ray (and connect to cluster).
init()
# init(address = None)


input_path = "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/orcas_output/"


ds = read_parquet(
    input_path,
    # arrow_open_stream_args={"compression": "gzip"},
    file_extensions=['parquet'],
    # partitioning=partitioning,
    # parse_options=ParseOptions(explicit_schema=schema)
    concurrency=5
)
cnt = 0
for i in ds.iter_rows():
    cnt += 1

# # ds.write_parquet(
# #     '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/output_remote_parquetCheck',
# #     num_rows_per_file=5000000)
# # ds.write_json(path='/home/benjamin/studium/masterarbeit/thesis-schneg/data/output_remote_all',
# #               num_rows_per_file=1000000, **json_args)
print(f"\n\n\nNUMBER OF ROWS:\t{cnt}\n\n\n")
# ds.write_parquet(path='/home/benjamin/studium/masterarbeit/thesis-schneg/data/output_remote_all_parquet',
#                  num_rows_per_file=1000000)
