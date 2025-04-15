from ray import init
import json
import pyarrow as pa
from pyarrow import csv
import pandas as pd
from ray.data import read_json, read_parquet, read_csv
import os
import sys
# this script is used to standardize the datasets for the thesis, namely to align column names and types
init()

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

final_schema = pa.schema(
    [
        pa.field("serp_id", pa.string(), nullable=True),
        pa.field("serp_url", pa.string(), nullable=True),
        pa.field("serp_domain", pa.string(), nullable=True),
        pa.field("serp_domain_public_suffix", pa.string(), nullable=True),
        pa.field("serp_timestamp", pa.int64(), nullable=True),
        pa.field("serp_page", pa.int64(), nullable=True),
        pa.field("serp_offset", pa.int64(), nullable=True),
        pa.field("serp_query_text_url", pa.string(), nullable=True),
        pa.field("serp_query_text_url_language", pa.string(), nullable=True),
        pa.field("serp_query_text_html", pa.string(), nullable=True),
        pa.field("serp_warc_relative_path", pa.string(), nullable=True),
        pa.field("serp_warc_byte_offset", pa.int64(), nullable=True),
        pa.field("search_provider_name", pa.string(), nullable=True),
        pa.field("search_provider_alexa_domain", pa.string(), nullable=True),
        pa.field(
            "search_provider_alexa_domain_public_suffix", pa.string(), nullable=True
        ),
        pa.field("search_provider_alexa_rank", pa.int64(), nullable=True),
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

        file_extensions.append(self.file_type)

        if self.multi:
            input_paths = [os.path.join(self.path_dataset, f) for f in os.listdir(
                self.path_dataset) if f.endswith("."+file_ending)]
        else:
            input_paths = self.path_dataset

        if self.num_files is not None:
            input_paths = input_paths[0:self.num_files]

        if self.file_type != 'parquet':
            if self.parse_options is not None:
                parse_options = self.parse_options
                ds = reader(paths=input_paths, arrow_open_stream_args=arrow_open_stream_args, file_extensions=file_extensions,
                            parse_options=parse_options, concurrency=self.concurrency, include_paths=self.includePath)
            else:
                ds = reader(paths=input_paths, arrow_open_stream_args=arrow_open_stream_args, file_extensions=file_extensions,
                            concurrency=self.concurrency, include_paths=self.includePath)
        else:
            ds = reader(paths=input_paths, file_extensions=file_extensions,
                        concurrency=self.concurrency, include_paths=self.includePath)

        return ds


# aggregation_row_count = AggregateFn(
#     init=lambda column: 0,
#     # Apply this to each row to produce a partial aggregate result
#     accumulate_row=lambda a, row: a + 1,
#     # Apply this to merge partial aggregate results into a final result
#     merge=lambda a1, a2: a1 + a2,
#     name="sum_rows"
# )
# Define a function to fill missing columns with None


def fill_missing_columns(batch: pd.DataFrame) -> pd.DataFrame:
    for field in final_schema:
        if field.name not in batch.columns:
            batch[field.name] = None
    return batch


def int_to_str(batch: pd.DataFrame) -> pd.DataFrame:
    batch["serp_id"] = batch['serp_id'].astype(str)
    return batch


def count_rows(batch: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"row_count": [len(batch)]})
# print(f"SIZE Dataset: {ds_aql.aggregate(aggregation_row_count)['sum_rows']}")


dataset_name = sys.argv[1]
assert len(sys.argv) >= 2, "Please provide the dataset name as an argument.  Please use \"aql\", \"aol\", \"ms-marco\" or \"orcas\" as dataset name."
assert dataset_name in ['aql', 'aol', 'ms-marco',
                        'orcas'], "Dataset name not supported! Please use \"aql\", \"aol\", \"ms-marco\" or \"orcas\" as dataset name."

if dataset_name == 'aql':
    with open("/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/aql_paths.json", 'r') as f:
        input_paths_aql = json.load(f)
        input_paths_aql.pop(794)  # remove the error path -> empty file
        input_paths_aql.pop(961)  # remove the error path -> empty file
elif dataset_name == 'aol':
    input_paths_aol = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/corpus-aol-cleaned/'
elif dataset_name == 'ms-marco':
    input_paths_ms = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/corpus-msmarco/'
elif dataset_name == 'orcas':
    input_paths_orcas = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/orcas_cleaned/'


#   AQL   #####

aql_parse_options = pa.json.ParseOptions(
    explicit_schema=schema, unexpected_field_behavior="infer")


aql_dataloader = Ray_Dataloader(
    file_type="jsonl", path_dataset=input_paths_aql, compression="gz", parse_options=aql_parse_options)  # , num_files=2,

ds_aql = aql_dataloader.read_file()

ds_aql = ds_aql.drop_columns(cols=["serp_wayback_url", "serp_wayback_raw_url",
                                   "serp_results", "serp_warc_relative_path", "serp_warc_byte_offset"],  concurrency=5)

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


ds_aql = ds_aql.map_batches(fill_null_columns, batch_format="pandas")

output_path_aql = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/aql_output'

ds_aql.write_parquet(output_path_aql, concurrency=5,
                     num_rows_per_file=500000)


#     AOL     ###
aol_parse_options = csv.ParseOptions(delimiter="\t")

aol_dataloader = Ray_Dataloader(
    file_type="txt", path_dataset=input_paths_aol, compression='gz', parse_options=aol_parse_options, includePath=True)

ds_aol = aol_dataloader.read_file()

# Define the new column names
new_column_names_aol = {
    'AnonID': 'serp_id',
    'Query': 'serp_query_text_url',
    'QueryTime': 'serp_timestamp',
    'ItemRank': 'serp_offset',
}


def rename_columns_aol(batch: pd.DataFrame) -> pd.DataFrame:
    return batch.rename(columns=new_column_names_aol)


ds_aol = ds_aol.map_batches(rename_columns_aol, batch_format="pandas")

ds_aol = ds_aol.drop_columns(cols=["ClickURL"])

ds_aol = ds_aol.map_batches(fill_missing_columns, batch_format="pandas")

output_path_aol = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/aol_output'

ds_aol.write_parquet(output_path_aol, concurrency=5,
                     num_rows_per_file=500000)

# row_counts = ds_aol.map_batches(count_rows, batch_format="pandas")

# total_row_count = row_counts.sum(on="row_count")

# print(ds_aol.schema())
# print(ds_aol.take(5))

###     MS_MARCO        ###
ms_parse_options = csv.ParseOptions(delimiter="\t")

ms_dataloader = Ray_Dataloader(
    file_type="tsv", path_dataset=input_paths_ms, parse_options=ms_parse_options)

ds_ms = ms_dataloader.read_file()

# Define the new column names
new_column_names_ms = {
    'query_id': 'serp_id',
    'query': 'serp_query_text_url',
    'language': 'serp_query_text_url_language',
}


def rename_columns_ms(batch: pd.DataFrame) -> pd.DataFrame:
    return batch.rename(columns=new_column_names_ms)


def fill_empty_lang(batch: pd.DataFrame) -> pd.DataFrame:
    if batch['serp_query_text_url_language'].isnull().all():
        batch['serp_query_text_url_language'] = batch['serp_query_text_url_language'].fillna(
            "")
    return batch


ds_ms = ds_ms.map_batches(rename_columns_ms, batch_format="pandas")
ds_ms = ds_ms.map_batches(fill_missing_columns, batch_format="pandas")
print(ds_ms.take(5))

ds_ms = ds_ms.map_batches(fill_empty_lang, batch_format="pandas")

output_path_ms = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/msmarco_output'

ds_ms.write_parquet(output_path_ms, concurrency=5,
                    num_rows_per_file=500000)

# row_counts = ds_ms.map_batches(count_rows, batch_format="pandas")

# total_row_count = row_counts.sum(on="row_count")
# print(ds_ms.schema())
# print(ds_ms.take(5))


## ORCAS ####
orcas_dataloader = Ray_Dataloader(
    file_type="csv", path_dataset=input_paths_orcas)

ds_orcas = orcas_dataloader.read_file()

ds_orcas = ds_orcas.map_batches(fill_missing_columns, batch_format="pandas")

# row_counts = ds_orcas.map_batches(count_rows, batch_format="pandas")

# total_row_count = row_counts.sum(on="row_count")

# print(f"\n\n\n\n\n\nTotal number of rows: {total_row_count}\n\n\n\n\n\n")

# print(ds_orcas.schema())
# print(ds_orcas.take(20))
output_path_orcas = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/orcas_output'

ds_orcas.write_parquet(output_path_orcas, concurrency=5,
                       num_rows_per_file=500000)


## CHECK ORCAS ###

# ds_orcas.write_parquet(output_path_orcas, concurrency=5,
#                        num_rows_per_file=500000)


# output_dataloader = Ray_Dataloader(
#     file_type="parquet", path_dataset=output_path_orcas)

# ds_orcas = output_dataloader.read_file()
# # # ,parse_options=orcas_parse_options
# ds_orcas = ds_orcas.map_batches(fill_missing_columns, batch_format="pandas")

# row_counts = ds_orcas.map_batches(count_rows, batch_format="pandas")

# total_row_count_out = row_counts.sum(on="row_count")

# print(f"\n\n\n\n\n\nTotal number of input_rows: {total_row_count}\n\n\n\n\n\n")
# print(
#     f"\n\n\n\n\n\nTotal number of output_rows: {total_row_count_out}\n\n\n\n\n\n")


## CHECK MS_MARCO ###

# ds_ms.write_parquet(output_path_ms, concurrency=5,
# #                     num_rows_per_file=500000)
# output_dataloader = Ray_Dataloader(
#     file_type="parquet", path_dataset=output_path_ms)


# ds_ms = output_dataloader.read_file()
# # # # ,parse_options=orcas_parse_options
# ds_ms = ds_ms.map_batches(fill_missing_columns, batch_format="pandas")

# row_counts = ds_ms.map_batches(count_rows, batch_format="pandas")

# total_row_count_out = row_counts.sum(on="row_count")

# print(f"\n\n\n\n\n\nTotal number of input_rows: {total_row_count}\n\n\n\n\n\n")
# print(
#     f"\n\n\n\n\n\nTotal number of output_rows: {total_row_count_out}\n\n\n\n\n\n")


## CHECK AOL ###

# ds_aol.write_parquet(output_path_aol, concurrency=5,
#                      num_rows_per_file=500000)


# output_dataloader = Ray_Dataloader(
#     file_type="parquet", path_dataset=output_path_aol)


# ds_aol = output_dataloader.read_file()
# # # # ,parse_options=orcas_parse_options
# ds_aol = ds_aol.map_batches(fill_missing_columns, batch_format="pandas")

# row_counts = ds_aol.map_batches(count_rows, batch_format="pandas")

# total_row_count_out = row_counts.sum(on="row_count")

# print(f"\n\n\n\n\n\nTotal number of input_rows: {total_row_count}\n\n\n\n\n\n")
# print(
#     f"\n\n\n\n\n\nTotal number of output_rows: {total_row_count_out}\n\n\n\n\n\n")

# print(ds_aol.schema())
# def transform_null(row: Dict[str, Any]) -> Dict[str, Any]:
#     for k, v in row.items():
#         if row[k] is None:
#             row[k] = "None"
#     return row


# ds_aql = ds_aql.map(transform_null, concurrency=5)

# ds_aql.write_parquet(output_path_aql, concurrency=5,
#                      num_rows_per_file=500000)

# 1835038
