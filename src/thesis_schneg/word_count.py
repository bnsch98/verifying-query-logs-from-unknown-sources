import matplotlib.pyplot as plt
from ray import init
from typing import Any, Dict
from ray.data import read_parquet
import os
import pandas as pd
init()


class read_parquet_data:
    def __init__(self, dataset_name: str, num_files: int = None, concurrency: int = 5, multi: bool = True):
        """A uniform dataloader, that manages reading different query log datasets in Ray.

        Args:
            dataset_name (str, compulsory): specifies which dataset should be read: aol, ms, orcas, aql
            num_files (int, optional): specifies the number of input files to be read
            concurrency (int, optional) specifies the max number of processes used to read the source data
             """
        self.dataset_name = dataset_name
        self.num_files = num_files
        self.concurrency = concurrency
        self.multi = multi

        assert self.dataset_name in [
            'aol', 'ms', 'orcas', 'aql'], "Specified dataset_name is not supported!"
        assert not (self.multi is False and self.num_files is not None), "Can't request single file and simoultenously specify multiple files! For single file, set multi to False and num_files to None!"

        if self.dataset_name == 'aol':
            self.paths = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/aol_output/'
        elif self.dataset_name == 'ms':
            self.paths = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/msmarco_output/'
        elif self.dataset_name == 'orcas':
            self.paths = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/orcas_output/'
        else:
            self.paths = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/aql_output/'

    def read_file(self):

        input_paths = [os.path.join(self.paths, f) for f in os.listdir(
            self.paths) if f.endswith(".parquet")]
        if self.multi:
            if self.num_files is not None:
                input_paths = input_paths[0:self.num_files]
        else:
            input_paths = input_paths[0]

        ds = read_parquet(paths=input_paths, concurrency=self.concurrency)

        return ds


# def query_word_count(batch: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
#     for row in batch:
#         if 'serp_query_text_url' in row:
#             row['word_count'] = len(str(row['serp_query_text_url']).split())
#             row['string_length'] = len(str(row['serp_query_text_url']))
#     return batch

def query_word_count(df: pd.DataFrame) -> pd.DataFrame:
    # df['word_count'] = df['serp_query_text_url'].apply(
    #     lambda x: len(str(x).split()))
    df['string_length'] = df['serp_query_text_url'].apply(
        lambda x: len(str(x)))
    return df
# def query_word_count(row: Dict[str, Any]) -> Dict[str, Any]:
#     if 'serp_query_text_url' in row:
#         row['word_count'] = len(str(row['serp_query_text_url']).split())
#         row['string_length'] = len(str(row['serp_query_text_url']))
#     return row


datasets = ['aol', 'ms', 'orcas', 'aql']
dataframes = []
max_word_count = 0
max_string_count = 0
# analysis = 'word_count'
analysis = 'string_length'

for dataset_name in datasets:
    reader = read_parquet_data(
        dataset_name=dataset_name, concurrency=5)
    ds = reader.read_file()
    # ds = ds.add_column(
    #     'word_count', lambda df: df['serp_query_text_url'], concurrency=5)
    ds = ds.add_column(
        'string_length', lambda df: df['serp_query_text_url'], concurrency=5)

    # ,batch_format="pandas"
    ds = ds.map_batches(query_word_count, batch_format="pandas")
    # print(f"\n\n\n\n\nrows of {dataset_name}: {ds.count()}\n\n\n\n\n")
    ds_group = ds.groupby('string_length').count()
    df_group = ds_group.to_pandas()
    dataframes.append((dataset_name, df_group))
    # print(ds.select_columns(
    #     cols=['word_count', 'serp_query_text_url']).take_batch(1))
    # print(ds.select_columns(
    #     cols=['string_length', 'serp_query_text_url']).take_batch(1))
    # max_word_count = max(max_word_count, df_group['word_count'].max())
    # max_string_count = max(max_string_count, df_group['word_count'].max())


# print(f"\n\nMAX WORD COUNT{max_word_count}\n\n")
# Plot the histograms for each dataset in a single plot
# plt.figure(figsize=(10, 6))

for dataset_name, df_group in dataframes:
    total_rows = df_group['count()'].sum()
    # plt.bar(df_group['word_count'], df_group['count()']/total_rows,
    #         alpha=0.5, label=dataset_name)
    df_group.to_csv(
        '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/results_query_length/' + dataset_name + '_' + analysis + '.csv')

# plt.xlabel(analysis)
# plt.ylabel('Frequency')
# plt.title(f'Histogram of scaled {analysis} for Multiple Datasets')
# plt.legend()

# # Set the x-axis limit to the maximum word count
# plt.xlim(0, 50)

# # Add and customize grid
# plt.minorticks_on()
# plt.grid(True, which='major', linestyle='-', linewidth='0.5', color='black')
# plt.grid(True, which='minor', linestyle=':', linewidth='0.5', color='gray')


# plt.show()

# plt.savefig(
#     f'/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/plots/histogram_scaled_{analysis}.png')
# aql_reader = read_parquet_data(
#     dataset_name='orcas', num_files=1, concurrency=5)

# ds_aql = aql_reader.read_file()


# # ds_aql = ds_aql.map_batches(query_word_count)
# ds_aql = ds_aql.add_column('word_count', lambda row: len(
#     str(row['serp_query_text_url']).split()))


# print(ds_aql.schema())
# print(ds_aql.select_columns(
#     cols=['word_count']).take_batch(1))

# # Gruppieren Sie die Daten nach 'word_count'
# ds_group = ds_aql.groupby('word_count').count()

# # Konvertieren Sie die aggregierten Daten in ein Pandas DataFrame
# df_group = ds_group.to_pandas()

# # Überprüfen Sie die aggregierten Daten
# print("Aggregierte Daten:")
# print(df_group)


# # Erstellen Sie das Histogramm
# plt.figure(figsize=(10, 6))
# plt.bar(df_group['word_count'], df_group['count()'])
# plt.xlabel('Word Count')
# plt.ylabel('Frequency')
# plt.title('Histogram of Word Count')
# plt.show()
# output_path_aql = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/aql_output'


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


# # ds_aql = ds_aql.map(transform_null, concurrency=5)

# aggregation_row_count = AggregateFn(
#     init=lambda column: 0,
#     # Apply this to each row to produce a partial aggregate result
#     accumulate_row=lambda a, row: a + 1,
#     # Apply this to merge partial aggregate results into a final result
#     merge=lambda a1, a2: a1 + a2,
#     name="sum_rows"
# )

# # print(
# #     f"SIZE grouped Dataset: {ds_group.aggregate(aggregation_row_count)['sum_rows']}")
# # print(f"SIZE Dataset: {ds_aql.aggregate(aggregation_row_count)['sum_rows']}")
# # print(ds_group.count().take(5))
# query_lengths_dataset = ds_group.count()
# query_lengths_dataset.write_parquet(
#     '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/results_query_length')
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
