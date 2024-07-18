from ray import init
from ray.data import range
import ray
import pyarrow as pa
from pyarrow.lib import timestamp
from pyarrow.json import ParseOptions
from ray.data import read_json, read_parquet
from ray.data.datasource.partitioning import Partitioning
from ray.data.aggregate import Count, AggregateFn
import os
# Initialize Ray (and connect to cluster).
init()
# init(address = None)

schema = pa.schema([
    pa.field('serp_id', pa.string(), nullable=True),
    pa.field('serp_url', pa.string(), nullable=True),
    pa.field('serp_domain', pa.string(), nullable=True),
    pa.field('serp_domain_public_suffix',
             pa.string(), nullable=True),
    pa.field('serp_timestamp', pa.int64(), nullable=True),
    # pa.field('serp_timestamp', timestamp('s', tz='UTC')),
    pa.field('serp_wayback_url', pa.string(), nullable=True),
    pa.field('serp_wayback_raw_url',
             pa.string(), nullable=True),
    pa.field('serp_page', pa.int64(), nullable=True),
    pa.field('serp_offset', pa.int64(), nullable=True),
    pa.field('serp_query_text_url',
             pa.string(), nullable=True),
    pa.field('serp_query_text_url_language',
             pa.string(), nullable=True),
    pa.field('serp_query_text_html',
             pa.string(), nullable=True),
    pa.field('serp_warc_relative_path',
             pa.string(), nullable=True),
    pa.field('serp_warc_byte_offset',
             pa.int64(), nullable=True),
    pa.field('serp_results', pa.list_(pa.struct([
        pa.field(
            'result_id', pa.string(), nullable=True),
        pa.field(
            'result_url', pa.string(), nullable=True),
        pa.field(
            'result_domain', pa.string(), nullable=True),
        pa.field(
            'result_domain_public_suffix', pa.string(), nullable=True),
        pa.field(
            'result_wayback_url', pa.string(), nullable=True),
        pa.field(
            'result_wayback_raw_url', pa.string(), nullable=True),
        pa.field(
            'result_snippet_rank', pa.int64(), nullable=True),
        pa.field(
            'result_snippet_title', pa.string(), nullable=True),
        pa.field(
            'result_snippet_text', pa.string(), nullable=True),
        pa.field(
            'result_warc_relative_path', pa.string(), nullable=True),
        pa.field(
            'result_warc_byte_offset', pa.int64(), nullable=True),
    ])), nullable=True),
    pa.field('search_provider_name',
             pa.string(), nullable=True),
    pa.field('search_provider_alexa_domain',
             pa.string(), nullable=True),
    pa.field('search_provider_alexa_domain_public_suffix',
             pa.string(), nullable=True),
    pa.field('search_provider_alexa_rank',
             pa.int64(), nullable=True),
    pa.field('search_provider_category',
             pa.string(), nullable=True),
])
# pa.field('result_id', pa.string()),
#                     pa.field('result_url', pa.string()),
# Erhalte eine Liste aller CSV-Dateien im Verzeichnis


input_path = "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/output_remote_parquetCheck"
# input_path = "/mnt/ceph/storage/data-in-progress/data-research/web-search/archive-query-log/focused/corpus/full/2023-05-22/serps/part-00001.gz"
# input_path = "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/file20.gz"

# partitioning = Partitioning("dir", field_names=None, base_dir=input_path)

input_path = [os.path.join(input_path, f)
              for f in os.listdir(input_path) if f.endswith('.parquet')]

cnt = 0
for path in input_path:
    print(path)
    ds = read_parquet(
        path,
        # arrow_open_stream_args={"compression": "gzip"},
        # file_extensions=['gz', 'json', 'jsonl'],
        # partitioning=partitioning,
        # parse_options=ParseOptions(explicit_schema=schema)
    )
    for i in ds.iter_rows():
        cnt += 1

    # ds.write_parquet(
    #     '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/output_remote_parquetCheck',
    #     num_rows_per_file=5000000)
# ds.write_json(path='/home/benjamin/studium/masterarbeit/thesis-schneg/data/output_remote_all',
#               num_rows_per_file=1000000, **json_args)
print(f"\n\n\nNUMBER OF ROWS:\t{cnt}\n\n\n")
# ds.write_parquet(path='/home/benjamin/studium/masterarbeit/thesis-schneg/data/output_remote_all_parquet',
#                  num_rows_per_file=1000000)
