import matplotlib.pyplot as plt
from ray import init
from typing import Any, Dict
from ray.data.aggregate import AggregateFn
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


aggregation_row_count = AggregateFn(
    init=lambda column: 0,
    # Apply this to each row to produce a partial aggregate result
    accumulate_row=lambda a, row: a + 1,
    # Apply this to merge partial aggregate results into a final result
    merge=lambda a1, a2: a1 + a2,
    name="sum_rows"
)

# datasets = ['aol', 'ms', 'orcas', 'aql']
datasets = ['ms', 'aql']

sizes = []
for dataset_name in datasets:
    reader = read_parquet_data(
        dataset_name=dataset_name, concurrency=5)
    ds = reader.read_file()
    sizes.append(ds.aggregate(aggregation_row_count)['sum_rows'])

    ds = ds.filter(lambda row: row['serp_query_text_url'] is not None)
    sizes.append(ds.aggregate(aggregation_row_count)['sum_rows'])

print(f"\n\n\n\nSIZES: {sizes}\n\n\n\n")
# for dataset_name in datasets:
#     reader = read_parquet_data(
#         dataset_name=dataset_name, concurrency=5, num_files=2)
#     ds = reader.read_file()
#     print(ds.take(1))
