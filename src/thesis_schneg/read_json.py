from ray import init
# from ray.data import range
# import ray
import pyarrow as pa
# from pyarrow.lib import timestamp
from pyarrow.json import ParseOptions
from ray.data import read_json
# from ray.data.datasource.partitioning import Partitioning
# from ray.data.aggregate import Count, AggregateFn
import os

# Initialize Ray (and connect to cluster).
init()
# init(address = None)

schema = pa.schema(
    [
        pa.field("serp_id", pa.string(), nullable=True),
        pa.field("serp_url", pa.string(), nullable=True),
        pa.field("serp_domain", pa.string(), nullable=True),
        pa.field("serp_domain_public_suffix", pa.string(), nullable=True),
        pa.field("serp_timestamp", pa.int64(), nullable=True),
        # pa.field('serp_timestamp', timestamp('s', tz='UTC')),
        pa.field("serp_wayback_url", pa.string(), nullable=True),
        pa.field("serp_wayback_raw_url", pa.string(), nullable=True),
        pa.field("serp_page", pa.int64(), nullable=True),
        pa.field("serp_offset", pa.int64(), nullable=True),
        pa.field("serp_query_text_url", pa.string(), nullable=True),
        pa.field("serp_query_text_url_language", pa.string(), nullable=True),
        pa.field("serp_query_text_html", pa.string(), nullable=True),
        pa.field("serp_warc_relative_path", pa.string(), nullable=True),
        pa.field("serp_warc_byte_offset", pa.int64(), nullable=True),
        pa.field(
            "serp_results",
            pa.list_(
                pa.struct(
                    [
                        pa.field("result_id", pa.string(), nullable=True),
                        pa.field("result_url", pa.string(), nullable=True),
                        pa.field("result_domain", pa.string(), nullable=True),
                        pa.field(
                            "result_domain_public_suffix", pa.string(), nullable=True
                        ),
                        pa.field("result_wayback_url", pa.string(), nullable=True),
                        pa.field("result_wayback_raw_url", pa.string(), nullable=True),
                        pa.field("result_snippet_rank", pa.int64(), nullable=True),
                        pa.field("result_snippet_title", pa.string(), nullable=True),
                        pa.field("result_snippet_text", pa.string(), nullable=True),
                        pa.field(
                            "result_warc_relative_path", pa.string(), nullable=True
                        ),
                        pa.field("result_warc_byte_offset", pa.int64(), nullable=True),
                    ]
                )
            ),
            nullable=True,
        ),
        pa.field("search_provider_name", pa.string(), nullable=True),
        pa.field("search_provider_alexa_domain", pa.string(), nullable=True),
        pa.field(
            "search_provider_alexa_domain_public_suffix", pa.string(), nullable=True
        ),
        pa.field("search_provider_alexa_rank", pa.int64(), nullable=True),
        pa.field("search_provider_category", pa.string(), nullable=True),
    ]
)
# pa.field('result_id', pa.string()),
#                     pa.field('result_url', pa.string()),
# Erhalte eine Liste aller CSV-Dateien im Verzeichnis


input_paths = "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/few_serps/"
# input_path = "/mnt/ceph/storage/data-in-progress/data-research/web-search/archive-query-log/focused/corpus/full/2023-05-22/serps/part-00001.gz"
# input_path = "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/file20.gz"


input_paths = [
    os.path.join(input_paths, f) for f in os.listdir(input_paths) if f.endswith(".gz")
]
print(f"\n\n{input_paths}\n\n")

ds = read_json(
    input_paths,
    arrow_open_stream_args={"compression": "gzip"},
    file_extensions=["gz", "json", "jsonl"],
    # partitioning=partitioning,
    parse_options=ParseOptions(explicit_schema=schema),
)

# cnt = 0
# for i in ds.iter_rows():
#     cnt += 1
# print(f"Dataset has {cnt} rows.")
# drop_cols = ['serp_wayback_url',
# #              'serp_wayback_raw_url']  # , 'result_wayback_raw_url'
# print("\n\n\n\n\n\n\n")
# struc = ds.select_columns(['serp_results'])
# print(f"############## STRUC: {struc.schema()} ##############")
# print("\n\n\n\n\n\n\n")
drop_cols = ['serp_results']
ds = ds.drop_columns(drop_cols)
# print(ds.schema())
# col = ds.select_columns(['serp_id'])

print("\n\n\n\n\n\n\n")
drop_cols = ["serp_offset"]
ds = ds.drop_columns(drop_cols)
print(ds.schema())
print("\n\n\n\n\n\n\n")

# print(ds)
# print("\n\n\n\n\n\n\n")
# ds.drop_columns(drop_cols).write_parquet(
#     '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/output_remote_parquet_loop',
#     num_rows_per_file=5000000)

# ds.write_json(path='/home/benjamin/studium/masterarbeit/thesis-schneg/data/output_remote_all',
#               num_rows_per_file=1000000, **json_args)

# ds.write_parquet(path='/home/benjamin/studium/masterarbeit/thesis-schneg/data/output_remote_all_parquet',
#                  num_rows_per_file=1000000)
