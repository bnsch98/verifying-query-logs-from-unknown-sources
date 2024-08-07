from json import dumps
from ray import init
# from ray.data import range
# import ray
import pyarrow as pa
# from pyarrow.lib import timestamp
from pyarrow import json, csv

from ray.data import read_json, read_parquet, read_csv, read_text
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


class Ray_Dataloader:
    def __init__(self, file_type: str, path_dataset: str, compression: str = None, num_files: int = None, concurrency: int = 5, parse_options=None):
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

        input_paths = [os.path.join(self.path_dataset, f) for f in os.listdir(
            self.path_dataset) if f.endswith("."+file_ending)]

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

        # input_paths = "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/few_serps/"
input_paths_aql = "/mnt/ceph/storage/data-in-progress/data-research/web-search/archive-query-log/focused/corpus/full/2023-05-22/serps/"
# input_path = "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/file20.gz"
# input_paths = "/mnt/ceph/storage/data-in-progress/data-research/web-search/archive-query-log/focused/corpus/full/2023-05-22/serps/part-00004.gz"  # kleine Datei
input_paths_aol = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/corpus-aol/'
input_paths_ms = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/corpus-msmarco/'
# input_paths_ms = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/msmarco_micro/'
# input_paths_orcas = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/orcas2/'
input_paths_orcas = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/orcas/'


schema_orcas = pa.schema([
    pa.field('query_id', pa.string(), nullable=True),
    pa.field('query', pa.string(), nullable=True)

])
aql_parse_options = json.ParseOptions(explicit_schema=schema)
aql_dataloader = Ray_Dataloader(
    file_type="jsonl", path_dataset=input_paths_aql, compression="gz", parse_options=aql_parse_options, num_files=10)
ds_aql = aql_dataloader.read_file()

ds_aql = ds_aql.drop_columns(cols=["serp_wayback_url", "serp_wayback_raw_url",
                                   "serp_results", "serp_warc_relative_path", "serp_warc_byte_offset"],  concurrency=5)

print(ds_aql.schema())
print(ds_aql.take(5))

# aol_parse_options = csv.ParseOptions(delimiter="\t")

# aol_dataloader = Ray_Dataloader(
#     file_type="txt", path_dataset=input_paths_aol, compression='gz', num_files=1, parse_options=aol_parse_options)

# ds_aol = aol_dataloader.read_file()

# print(ds_aol.schema())
# print(ds_aol.take(5))

# ms_parse_options = csv.ParseOptions(delimiter="\t")

# ms_dataloader = Ray_Dataloader(
#     file_type="tsv", path_dataset=input_paths_ms, num_files=1, parse_options=ms_parse_options)

# ds_ms = ms_dataloader.read_file()

# print(ds_ms.schema())
# print(ds_ms.take(5))


# orcas_parse_options = csv.ParseOptions(delimiter="\t")

# orcas_dataloader = Ray_Dataloader(
#     file_type="tsv", path_dataset=input_paths_orcas, num_files=1, parse_options=orcas_parse_options)

# ds_orcas = orcas_dataloader.read_file()

# print(ds_orcas.schema())
# print(ds_orcas.take(20))


# output_path_orcas = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/orcas_output'
# output_path_ms = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/msmarco_output'
# output_path_aol = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/aol_output'
output_path_aql = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/aql_output'


# ds_orcas.write_parquet(output_path_orcas, concurrency=5,
#                        num_rows_per_file=500000)

# ds_ms.write_parquet(output_path_ms, concurrency=5,
#                     num_rows_per_file=500000)

# ds_aol.write_parquet(output_path_aol, concurrency=5,
#                      num_rows_per_file=500000)

ds_aql.write_parquet(output_path_aql, concurrency=5,
                     num_rows_per_file=500000)
