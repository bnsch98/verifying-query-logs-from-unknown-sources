from ray import init
from typing import Any, Dict
# from ray.data import range
# import ray
import pyarrow as pa
# from pyarrow.lib import timestamp
from pyarrow import json, csv

from ray.data import read_json, read_parquet, read_csv
from ray.data.aggregate import AggregateFn

# from ray.data.datasource.partitioning import Partitioning
# from ray.data.aggregate import Count, AggregateFn
import os
# import matplotlib.pyplot as plt
import numpy as np
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
        pa.field("serp_query_text_html", pa.string(),
                 nullable=True),  # , nullable=True
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
        pa.field("search_provider_alexa_rank",
                 pa.int64(), nullable=True),  # int64
        pa.field("search_provider_category", pa.string(), nullable=True),
    ]
)

schema = pa.schema(
    [
        pa.field("serp_id", pa.string()),
        pa.field("serp_url", pa.string()),
        pa.field("serp_domain", pa.string()),
        pa.field("serp_domain_public_suffix", pa.string()),
        pa.field("serp_timestamp", pa.int64()),
        # pa.field('serp_timestamp', timestamp('s', tz='UTC')),
        pa.field("serp_wayback_url", pa.string()),
        pa.field("serp_wayback_raw_url", pa.string()),
        pa.field("serp_page", pa.int64()),
        pa.field("serp_offset", pa.int64()),
        pa.field("serp_query_text_url", pa.string()),
        pa.field("serp_query_text_url_language", pa.string()),
        pa.field("serp_query_text_html", pa.string()),  # ,
        pa.field("serp_warc_relative_path", pa.string()),
        pa.field("serp_warc_byte_offset", pa.int64()),
        pa.field(
            "serp_results",
            pa.list_(
                pa.struct(
                    [
                        pa.field("result_id", pa.string()),
                        pa.field("result_url", pa.string()),
                        pa.field("result_domain", pa.string()),
                        pa.field(
                            "result_domain_public_suffix", pa.string()
                        ),
                        pa.field("result_wayback_url",
                                 pa.string()),
                        pa.field("result_wayback_raw_url",
                                 pa.string()),
                        pa.field("result_snippet_rank",
                                 pa.int64()),
                        pa.field("result_snippet_title",
                                 pa.string()),
                        pa.field("result_snippet_text",
                                 pa.string()),
                        pa.field(
                            "result_warc_relative_path", pa.string()
                        ),
                        pa.field("result_warc_byte_offset",
                                 pa.int64()),
                    ]
                )
            ),

        ),
        pa.field("search_provider_name", pa.string()),
        pa.field("search_provider_alexa_domain", pa.string()),
        pa.field(
            "search_provider_alexa_domain_public_suffix", pa.string()
        ),
        pa.field("search_provider_alexa_rank",
                 pa.int64()),  # int64
        pa.field("search_provider_category", pa.string()),
    ]
)
# pa.field('result_id', pa.string()),
#                     pa.field('result_url', pa.string()),
# Erhalte eine Liste aller CSV-Dateien im Verzeichnis


class Ray_Dataloader:
    def __init__(self, file_type: str, path_dataset: str, compression: str = None, num_files: int = None, concurrency: int = 5, parse_options=None, multi: bool = True):
        """A uniform dataloader, that manages reading different query log datasets in Ray.

        Args: 
            file_type (str, compulsory): specifies the file extension, eg. json, csv, txt
            path_dataset (str, compulsory): specifies the path to the source data. Should be a folder 
            compression (str, compulsory): specifies whether the source data is compressed or not by passing the file extension. E.g. compressed = 'gz' 
            num_files (int, optional): specifies the number of input files to the Data Loader
            concurrency (int, optional) specifies the max number of processes used to read the source data
             """
        self.file_type = file_type
        self.path_dataset = path_dataset
        self.compression = compression
        self.num_files = num_files
        self.concurrency = concurrency
        self.parse_options = parse_options
        self.multi = multi

        assert self.file_type in ['txt', 'csv', 'tsv', 'json',
                                  'jsonl', 'parquet'], "Specified file type is not supported!"

        assert self.compression in [
            'gz'] or self.compression is None, 'Specified compression is not supported!'

    def read_file(self):
        if self.file_type == 'txt' or self.file_type == 'csv' or self.file_type == 'tsv':
            reader = read_csv
        elif self.file_type == 'json' or self.file_type == 'jsonl':
            reader = read_json
        else:
            reader = read_parquet

        arrow_open_stream_args = None
        file_extensions = []
        if self.compression is not None:
            file_ending = self.compression
            arrow_open_stream_args = {"compression": "gzip"}
            file_extensions.append("gz")
        else:
            file_ending = self.file_type

        file_extensions.append(self.file_type)

        if self.multi:
            input_paths = [os.path.join(self.path_dataset, f) for f in os.listdir(
                self.path_dataset) if f.endswith("."+file_ending)]
        else:
            input_paths = self.path_dataset

        if self.num_files is not None:
            input_paths = input_paths[0:self.num_files]

        if self.parse_options is not None:
            parse_options = self.parse_options
            ds = reader(paths=input_paths, arrow_open_stream_args=arrow_open_stream_args, file_extensions=file_extensions,
                        parse_options=parse_options, concurrency=self.concurrency)
        else:
            ds = reader(paths=input_paths, arrow_open_stream_args=arrow_open_stream_args, file_extensions=file_extensions,
                        concurrency=self.concurrency)

        return ds


def avg_query(row: Dict[str, Any]) -> Dict[str, Any]:
    row['query_length'] = len(str(row['serp_query_text_url']).split())
    return row


input_paths_aql = "/mnt/ceph/storage/data-in-progress/data-research/web-search/archive-query-log/focused/corpus/full/2023-05-22/serps/"
# input_paths_aql = "/mnt/ceph/storage/data-in-progress/data-research/web-search/archive-query-log/focused/corpus/full/2023-05-22/serps/part-00004.gz"


aql_parse_options = json.ParseOptions(
    explicit_schema=schema, unexpected_field_behavior="ignore")

# aql_dataloader = Ray_Dataloader(
#     file_type="jsonl", path_dataset=input_paths_aql, compression="gz", parse_options=aql_parse_options, multi=False)  # num_files=2,

aql_dataloader = Ray_Dataloader(
    file_type="jsonl", path_dataset=input_paths_aql, compression="gz", parse_options=aql_parse_options, num_files=20)  # num_files=2,

ds_aql = aql_dataloader.read_file()

ds_aql = ds_aql.drop_columns(cols=["serp_wayback_url", "serp_wayback_raw_url",
                                   "serp_results", "serp_warc_relative_path", "serp_warc_byte_offset", "search_provider_alexa_rank", "serp_query_text_html", "serp_page"],  concurrency=5)

# ds_aql = ds_aql.drop_columns(cols=["serp_wayback_url", "serp_wayback_raw_url",
#                                    "serp_results", "serp_warc_relative_path", "serp_warc_byte_offset"],  concurrency=5)

ds_aql.add_column('query_length', lambda df:
                  df["serp_query_text_url"])
ds_aql = ds_aql.map(avg_query)
ds_group = ds_aql.groupby('query_length')

# print(ds_group)
# print(ds_aql.schema())
# print(ds_aql.take(5))

output_path_aql = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/aql_output'


# ds_orcas.write_parquet(output_path_orcas, concurrency=5,
#                        num_rows_per_file=500000)

# ds_ms.write_parquet(output_path_ms, concurrency=5,
#                     num_rows_per_file=500000)

# ds_aol.write_parquet(output_path_aol, concurrency=5,
#                      num_rows_per_file=500000)
# def transform_null(row: Dict[str, Any]) -> Dict[str, Any]:
#     for k, v in row.items():
#         if row[k] is None:
#             row[k] = "None"
#     return row


# ds_aql = ds_aql.map(transform_null, concurrency=5)

aggregation_row_count = AggregateFn(
    init=lambda column: 0,
    # Apply this to each row to produce a partial aggregate result
    accumulate_row=lambda a, row: a + 1,
    # Apply this to merge partial aggregate results into a final result
    merge=lambda a1, a2: a1 + a2,
    name="sum_rows"
)

# print(
#     f"SIZE grouped Dataset: {ds_group.aggregate(aggregation_row_count)['sum_rows']}")
# print(f"SIZE Dataset: {ds_aql.aggregate(aggregation_row_count)['sum_rows']}")
# print(ds_group.count().take(5))
query_lengths_dataset = ds_group.count()
query_lengths_dataset.write_parquet(
    '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/results_query_length')
# ds_groupedrows = ds_group.map_groups(
#     lambda g: g.aggregate(aggregation_row_count))

# print(ds_groupedrows.take(5))

# print(f"SIZE Dataset: {ds_aql.aggregate(aggregation_row_count)['sum_rows']}")

# ds_aql.write_parquet(output_path_aql, concurrency=5,
#                      num_rows_per_file=500000)  # , arrow_parquet_args={'': '', }

# ql_dataloader = Ray_Dataloader(
#     file_type="parquet", path_dataset=output_path_aql, parse_options=aql_parse_options)  # num_files=2,

# ds_aql = aql_dataloader.read_file()

# ds_aql = ds_aql.drop_columns(cols=["serp_wayback_url", "serp_wayback_raw_url",
#                                    "serp_results", "serp_warc_relative_path", "serp_warc_byte_offset", "search_provider_alexa_rank", "serp_query_text_html"],  concurrency=5)

# print(f"SIZE Dataset: {ds_aql.aggregate(aggregation_row_count)['sum_rows']}")
