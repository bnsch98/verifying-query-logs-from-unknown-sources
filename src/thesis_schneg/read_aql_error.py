from ray import init
# from ray.data import range
# import ray
import pyarrow as pa
# from pyarrow.lib import timestamp
from pyarrow.json import ParseOptions
from ray.data import read_json, read_parquet, read_csv
# from ray.data.datasource.partitioning import Partitioning
# from ray.data.aggregate import Count, AggregateFn
import json
import sys
import pandas as pd


# # Continue with other operations on ds_aql if needed
# from pandas import DataFrame
# from pyarrow import Table, schema, field, string, struct, list_, timestamp
# from ray.data.block import DataBatch


# Initialize Ray (and connect to cluster).
init()
# init(address = None)

my_schema = pa.schema(
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
        # pa.field("search_provider_alexa_rank", pa.int64(), nullable=True),
        pa.field("search_provider_alexa_rank", pa.float64(), nullable=True),
        pa.field("search_provider_category", pa.string(), nullable=True),
    ]
)


class Ray_Dataloader:
    def __init__(self, file_type: str, path_dataset: str, compression: str = None, num_files: int = None, concurrency: int = 5, parse_options=None, multi: bool = True, includePath: bool = False):
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
        self.includePath = includePath

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

        file_extensions.append(file_ending)

        input_paths = self.path_dataset

        if self.num_files is not None:
            input_paths = input_paths[0:self.num_files]

        if self.parse_options is not None:
            parse_options = self.parse_options
            ds = reader(paths=input_paths, arrow_open_stream_args=arrow_open_stream_args, file_extensions=file_extensions,
                        parse_options=parse_options, concurrency=self.concurrency, include_paths=self.includePath)
        else:
            ds = reader(paths=input_paths, arrow_open_stream_args=arrow_open_stream_args, file_extensions=file_extensions,
                        concurrency=self.concurrency, include_paths=self.includePath)

        return ds

        # input_paths = "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/data/few_serps/"
with open("/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/aql_paths.json", 'r') as f:
    input_paths = json.load(f)

# error_path = input_paths[794]
# input_paths = input_paths[794]
input_paths.pop(794)  # remove the error path -> empty file
input_paths.pop(961)  # remove the error path -> empty file
input_paths = input_paths[int(sys.argv[1]):int(sys.argv[2])]

print(f"argv1: {sys.argv[1]}\targv2: {sys.argv[2]}")


aql_parse_options = ParseOptions(
    explicit_schema=my_schema, unexpected_field_behavior="infer")

# aql_dataloader = Ray_Dataloader(
#     file_type="jsonl", path_dataset=input_paths_aql, compression="gz", parse_options=aql_parse_options, multi=False)  # num_files=2,

aql_dataloader = Ray_Dataloader(
    file_type="jsonl", path_dataset=input_paths, compression="gz", parse_options=aql_parse_options)  # , num_files=2,

ds_aql = aql_dataloader.read_file()

# ds_aql = ds_aql.drop_columns(cols=["serp_wayback_url", "serp_wayback_raw_url",
#                                    "serp_results", "serp_warc_relative_path", "serp_warc_byte_offset"],  concurrency=5)


# Define the function to fill columns with only null values with empty strings


# Define the function to fill columns with only null values based on their type
def fill_null_columns(batch: pd.DataFrame) -> pd.DataFrame:
    for column in batch.columns:
        if batch[column].isnull().all():
            if pd.api.types.is_string_dtype(batch[column]):
                batch[column] = batch[column].fillna("")
            elif pd.api.types.is_integer_dtype(batch[column]):
                batch[column] = batch[column].fillna(0)
            elif pd.api.types.is_float_dtype(batch[column]):
                batch[column] = batch[column].fillna(0.0)
    return batch


def count_rows(batch: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"row_count": [len(batch)]})


# Apply the function to each batch
# ds_aql = ds_aql.map_batches(fill_null_columns, batch_format="pandas")
row_counts = ds_aql.map_batches(count_rows, batch_format="pandas")

total_row_count = row_counts.sum(on="row_count")
# 961 962 fehler
# print(ds_aql.schema())
# print(ds_aql.take(5))

output_path_aql = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/aql_output2'

# ds_aql.write_parquet(output_path_aql, concurrency=5,
#                      num_rows_per_file=500000)


# output_dataloader = Ray_Dataloader(
#     file_type="parquet", path_dataset=output_path_aql)

ds_output = read_parquet(paths=output_path_aql, concurrency=5)

# # ds_output = ds_output.map_batches(fill_null_columns, batch_format="pandas")
row_counts = ds_output.map_batches(count_rows, batch_format="pandas")

total_row_count_out = row_counts.sum(on="row_count")


print(f"\n\n\n\n\n\nTotal number of input_rows: {total_row_count}\n\n\n\n\n\n")
print(
    f"\n\n\n\n\n\nTotal number of output_rows: {total_row_count_out}\n\n\n\n\n\n")
