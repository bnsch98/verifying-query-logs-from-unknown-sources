from json import dumps
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
import numpy as np
# Initialize Ray (and connect to cluster).
init()

# init(address = None)
# asasa
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
                        pa.field("result_wayback_url",
                                 pa.string(), nullable=True),
                        pa.field("result_wayback_raw_url",
                                 pa.string(), nullable=True),
                        pa.field("result_snippet_rank",
                                 pa.int64(), nullable=True),
                        pa.field("result_snippet_title",
                                 pa.string(), nullable=True),
                        pa.field("result_snippet_text",
                                 pa.string(), nullable=True),
                        pa.field(
                            "result_warc_relative_path", pa.string(), nullable=True
                        ),
                        pa.field("result_warc_byte_offset",
                                 pa.int64(), nullable=True),
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


def serp_results_to_json_string(row):
    row["serp_results"] = dumps(row["serp_results"])
    return row


def get_word_count(row):
    row['query'] = len(str(row['query']).split())
    return row


def get_query_length(row):
    row['query'] = len(str(row['query']))
    return row


# input_paths = "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/few_serps/"
input_paths = "/mnt/ceph/storage/data-in-progress/data-research/web-search/archive-query-log/focused/corpus/full/2023-05-22/serps/"
# input_paths = "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/file20.gz"


input_paths = [
    os.path.join(input_paths, f) for f in os.listdir(input_paths) if f.endswith(".gz")
]

input_paths = input_paths[0]

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

ds = ds.map(serp_results_to_json_string,)

ds = ds.add_column('query', lambda df:
                   df["serp_query_text_url"])

query_lengths_words = ds.select_columns(
    ['query']).map(get_word_count).to_pandas()

query_lengths = ds.select_columns(
    ['query']).map(get_query_length).to_pandas()

query_lengths = query_lengths["query"].to_numpy()
query_lengths = query_lengths.flatten()
query_lengths_words = query_lengths_words["query"].to_numpy()
query_lengths_words = query_lengths_words.flatten()

avg_query_chars = np.mean(query_lengths)
avg_query_words = np.mean(query_lengths_words)

hist_array_chars = np.histogram(
    query_lengths, bins=np.arange(1, np.max(query_lengths)+2))[0]
hist_array_words = np.histogram(
    query_lengths_words, bins=np.arange(1, np.max(query_lengths_words)+2))[0]

print(
    f"\n\n\nAverage number of characters per query: {avg_query_chars}")
print(
    f"\n\n\nAverage number of words per query: {avg_query_words}")


with open('/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/avg_query_chars_full.npy', 'wb') as f:
    np.save(f, avg_query_chars)

with open('/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/hist_chars.npy', 'wb') as f:
    np.save(f, hist_array_chars)

with open('/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/hist_words.npy', 'wb') as f:
    np.save(f, hist_array_words)

with open('/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/avg_query_words_full.npy', 'wb') as f:
    np.save(f, avg_query_words)


# print(ds)
# print("\n\n\n\n\n\n\n")
# ds.drop_columns(drop_cols).write_parquet(
#     '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/output_remote_parquet_loop',
#     num_rows_per_file=5000000)

# ds.write_json(path='/home/benjamin/studium/masterarbeit/thesis-schneg/data/output_remote_all',
#               num_rows_per_file=1000000, **json_args)

# ds.write_parquet(path='/home/benjamin/studium/masterarbeit/thesis-schneg/data/output_remote_all_parquet',
#                  num_rows_per_file=1000000)
