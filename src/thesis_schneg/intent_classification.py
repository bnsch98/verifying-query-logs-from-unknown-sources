from nltk.tokenize import word_tokenize
from collections import defaultdict
from ray import init
from ray.data import read_parquet
import os
import pandas as pd
import nltk


import argparse
from argparse import ArgumentParser
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import safetensors
import json
# from ray.data import read_json, read_parquet, read_csv

# from ray.data.datasource.partitioning import Partitioning
# from ray.data.aggregate import Count, AggregateFn
import os
# import matplotlib.pyplot as plt
from transformers import pipeline

init()


class read_parquet_data:
    def __init__(self, dataset_name: str, num_files: int = None, concurrency: int = 5, multi: bool = True, only_english: bool = False):
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
        self.only_english = only_english

        assert self.dataset_name in [
            'aol', 'ms-marco', 'orcas', 'aql'], "Specified dataset_name is not supported!"
        assert not (self.multi is False and self.num_files is not None), "Can't request single file and simoultenously specify multiple files! For single file, set multi to False and num_files to None!"

        if self.only_english:
            if self.dataset_name == 'aol':
                self.paths = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/aol_output/'
            elif self.dataset_name == 'ms-marco':
                self.paths = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/lng_filtered_ms-marco/'
            elif self.dataset_name == 'orcas':
                self.paths = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/orcas_output/'
            else:
                self.paths = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/lng_filtered_aql/'
        else:
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


def classify_intent(batch: pd.DataFrame, label_dict, model, tokenizer) -> pd.DataFrame:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inverse_label_dict = {v: k for k, v in label_dict.items()}
    predicted_intent = []
    try:
        for query in batch["serp_query_text_url"]:
            inputs = tokenizer(query, return_tensors="pt").to(device)
            logits = model(**inputs).logits.cpu().detach().numpy()
            predicted_intent.append(inverse_label_dict[np.argmax(logits)])
    except Exception:
        # Load the tokenizer
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True)
        # Load the BERT model
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=len(label_dict),
            output_attentions=False,
            output_hidden_states=False,)
        tensors = {}
        with safetensors.safe_open("/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/finetuned_BERT_first_level.safetensors", framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
            # del tensors["bert.embeddings.position_ids"]
        model.load_state_dict(
            tensors,
            strict=False
        )
        for query in batch["serp_query_text_url"]:
            inputs = tokenizer(query, return_tensors="pt").to(device)
            logits = model(**inputs).logits.cpu().detach().numpy()
            predicted_intent.append(inverse_label_dict[np.argmax(logits)])

    batch["predicted_intent"] = predicted_intent
    return batch


def read_labels(infile):
    with open(infile, "r") as fp:
        return json.load(fp)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_dict = read_labels(
    infile='/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/labels.json')
inverse_label_dict = {v: k for k, v in label_dict.items()}

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", do_lower_case=True)

# load the model and update the weights
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_dict),
    output_attentions=False,
    output_hidden_states=False,
)
# print("base model loaded")
tensors = {}
with safetensors.safe_open("/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/finetuned_BERT_first_level.safetensors", framework="pt", device="cpu") as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
    # del tensors["bert.embeddings.position_ids"]
model.load_state_dict(
    tensors,
    strict=False
)


datasets = ['aol', 'ms-marco', 'orcas', 'aql']
# datasets = ['aol', 'ms-marco', 'orcas']
datasets = ['aol', 'orcas']  # only english
datasets = ['aol']  # only english


for dataset_name in datasets:
    reader = read_parquet_data(
        dataset_name=dataset_name, concurrency=5, only_english=True, num_files=1)  # , num_files=1
    ds = reader.read_file()

    ds = ds.select_columns(['serp_query_text_url'])  # , concurrency=4

    classified_ds = ds.map_batches(classify_intent, fn_kwargs={
                                   "label_dict": label_dict, "model": model, "tokenizer": tokenizer}, batch_format="pandas")  # , concurrency=4

    final_word_counts = classified_ds.to_pandas()
    print(final_word_counts.columns)
    print(final_word_counts.take([0, 1, 2]))
