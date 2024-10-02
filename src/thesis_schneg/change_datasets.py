from ray import init
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
        if self.multi:
            if self.num_files is not None:
                input_paths = input_paths[0:self.num_files]
        else:
            input_paths = input_paths[0]

        ds = read_parquet(paths=input_paths, concurrency=self.concurrency)

        return ds


def filter_language(df: pd.DataFrame, language: str) -> pd.DataFrame:
    # Filtern der Zeilen, die den String 'en' in 'serp_query_text_url' enthalten
    return df[df['serp_query_text_url'].str.contains(language, na=False)]


# abbkürzungen sprache:
# aql: en   aol: keine sprache   orcas: keine sprache   ms-marco: en-xxx

datasets = ['ms-marco', 'aql']

for dataset_name in datasets:
    reader = read_parquet_data(
        dataset_name=dataset_name, concurrency=5)  # , num_files=1
    ds = reader.read_file()
    lng = 'en'
    ds = ds.map_batches(lambda df: filter_language(
        df, lng), batch_format="pandas")  # , concurrency=5       no limitations due to previous memory issues
    ds.write_parquet('/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/lng_filtered_'+dataset_name, concurrency=5,
                     num_rows_per_file=500000)
