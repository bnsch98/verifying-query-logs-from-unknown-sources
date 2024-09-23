import matplotlib.pyplot as plt
from ray import init
from typing import Any, Dict
from ray.data import read_parquet
import os
import pandas as pd
from ray.data.aggregate import AggregateFn
import sys
import pickle
init()


class read_parquet_data:
    def __init__(self, dataset_name: str, num_files: int = None, concurrency: int = 5, multi: bool = True, debug: bool = False):
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
        self.debug = debug

        assert self.dataset_name in [
            'aol', 'ms-marco', 'orcas', 'aql'], "Specified dataset_name is not supported!"
        assert not (self.multi is False and self.num_files is not None), "Can't request single file and simoultenously specify multiple files! For single file, set multi to False and num_files to None!"

        if self.dataset_name == 'aol':
            self.paths = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/aol_output/'
        elif self.dataset_name == 'ms-marco':
            self.paths = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/msmarco_output/'
        elif self.dataset_name == 'orcas':
            self.paths = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/orcas_output/'
        else:
            self.paths = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/aql_output/'

    def read_file(self):

        input_paths = [os.path.join(self.paths, f) for f in os.listdir(
            self.paths) if f.endswith(".parquet")]

        if self.debug:
            input_paths = input_paths[int(sys.argv[1]):int(sys.argv[2])]

        if self.multi:
            if self.num_files is not None:
                input_paths = input_paths[0:self.num_files]
        else:
            input_paths = input_paths[0]

        ds = read_parquet(paths=input_paths, concurrency=self.concurrency)

        return ds


aggregation_row_count = AggregateFn(
    init=lambda column: 0,
    # Apply this to each row to produce a partial aggregate result
    accumulate_row=lambda a, row: a + 1,
    # Apply this to merge partial aggregate results into a final result
    merge=lambda a1, a2: a1 + a2,
    name="sum_rows"
)


def filter_rows(batch: pd.DataFrame) -> pd.DataFrame:
    return batch.dropna(subset=['serp_query_text_url_language'])


dataframes = {'aql': [], 'ms-marco': [], 'orcas': [], 'aol': []}

for key in dataframes:
    reader = read_parquet_data(
        dataset_name=key, concurrency=5)  # , debug=True , num_files=1 files: 259
    ds = reader.read_file()
    dataframes[key].append(ds.count())
    # print(f"\n\n\n\n\nrows of {dataset_name}: {ds.count()}\n\n\n\n\n")
    # ds = ds.filter(lambda row: row['serp_query_text_url_language'] is not None)
    ds = ds.map_batches(filter_rows, batch_format="pandas", concurrency=5)

    ds_group = ds.groupby('serp_query_text_url').count()

    # ds_group = ds_group.to_pandas()
    # sizes.append(ds_group.shape[0])

    dataframes[key].append(ds_group.aggregate(
        aggregation_row_count)['sum_rows'])
    print(f"\n\n{key}: {dataframes[key]}\n\n")

print(f"\n\n\n\nSIZES: {dataframes}\n\n\n\n")
# Save the dataframes dictionary to a file
with open('/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/unique_queries_full.pkl', 'wb') as f:
    pickle.dump(dataframes, f)
