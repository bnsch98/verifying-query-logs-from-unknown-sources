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
from transformers import pipeline


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
            if arrow_open_stream_args is not None:
                ds = reader(paths=input_paths, arrow_open_stream_args=arrow_open_stream_args, file_extensions=file_extensions,
                            parse_options=parse_options, concurrency=self.concurrency)
            else:
                ds = reader(paths=input_paths, file_extensions=file_extensions,
                            parse_options=parse_options, concurrency=self.concurrency)
        else:
            if arrow_open_stream_args is not None:
                ds = reader(paths=input_paths, arrow_open_stream_args=arrow_open_stream_args, file_extensions=file_extensions,
                            concurrency=self.concurrency)
            else:
                ds = reader(paths=input_paths, file_extensions=file_extensions,
                            concurrency=self.concurrency)
        return ds


class LanguagePredictor:
    def __init__(self):
        model_ckpt = "papluca/xlm-roberta-base-language-detection"
        self.model = pipeline("text-classification", model=model_ckpt)

    def __call__(self, batch: Dict[str, str]) -> Dict[str, list]:
        predictions = self.model(
            list(batch["query"]), top_k=1, truncation=True)

        batch["output"] = [sequences[0]['label']
                           for sequences in predictions]

        return batch


def predict_language(model, row):
    row['language'] = model(row['query'], max_length=20,
                            num_return_sequences=1)[0][0]['label']
    return row


# Initialize Ray (and connect to cluster).
init()

# input_path = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/orcas_output'
input_path = "/mnt/ceph/storage/data-in-progress/data-research/web-search/archive-query-log/focused/corpus/full/2023-05-22/serps/part-00004.gz"


aql_dataloader = Ray_Dataloader(
    file_type="parquet", path_dataset=input_path,  multi=False)  # num_files=2,

ds_aql = aql_dataloader.read_file()
print(ds_aql.schema())

# ds_query = ds_aql.select_columns('serp_query_text_url')
ds_query = ds_aql.select_columns(['query'])

# print(type(ds_query.take_batch(1)))
# ds_query = ds_query.add_column('language', lambda df:
#                                df["query"])
# model_ckpt = "papluca/xlm-roberta-base-language-detection"
# model = pipeline("text-classification", model=model_ckpt)

# ds_query = ds_query.map(predict_language, fn_args=model)
predictions = ds_query.map_batches(
    LanguagePredictor,
    concurrency=2,
)


# Step 2: Map the Predictor over the Dataset to get predictions.
# Use 2 parallel actors for inference. Each actor predicts on a
# different partition of data.
# predictions = ds_query.map_batches(LanguagePredictor, concurrency=2)

predictions.take_batch(5)

# print(ds_query.take(10))
# text = [
#     "Brevity is the soul of wit.",
#     "Amor, ch'a nullo amato amar perdona."
# ]

# model_ckpt = "papluca/xlm-roberta-base-language-detection"
# pipe = pipeline("text-classification", model=model_ckpt)
# print(pipe(text, top_k=1, truncation=True)[0][0]['label'])
# print(pipe(text, top_k=1, truncation=True))
