from nltk.tokenize import word_tokenize
from collections import defaultdict
from ray import init
from ray.data import read_parquet
import os
import pandas as pd
import ray
from typing import Dict

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


class HuggingFacePredictor:
    def __init__(self):
        from transformers import BertTokenizer, BertForSequenceClassification

        label_dict = read_labels(
            infile='/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/labels.json')
        inverse_label_dict = {v: k for k, v in label_dict.items()}

        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True)

        # Load the BERT model
        classifier = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=len(label_dict),
            output_attentions=False,
            output_hidden_states=False,)
        tensors = {}
        with safetensors.safe_open("/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/finetuned_BERT_first_level.safetensors", framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
            # del tensors["bert.embeddings.position_ids"]
        classifier.load_state_dict(
            tensors,
            strict=False
        )
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = classifier.to(self.device)
        self.tokenizer = tokenizer
        self.inverse_label_dict = inverse_label_dict

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        output = []
        for query in batch["serp_query_text_url"]:
            inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
            logits = self.model(**inputs).logits.cpu().detach().numpy()
            output.append(self.inverse_label_dict[np.argmax(logits)])
        batch["predicted_intent"] = output

        return batch


class debugPredictor:
    def __init__(self):
        from transformers import pipeline
        # Initialize a pre-trained GPT2 Huggingface pipeline.
        self.model = pipeline("text-generation", model="gpt2")

    # Logic for inference on 1 batch of data.
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Get the predictions from the input batch.
        predictions = self.model(
            list(batch["serp_query_text_url"]), max_length=20, num_return_sequences=1)
        # `predictions` is a list of length-one lists. For example:
        # [[{'generated_text': 'output_1'}], ..., [{'generated_text': 'output_2'}]]
        # Modify the output to get it into the following format instead:
        # ['output_1', 'output_2']
        batch["output"] = [sequences[0]["generated_text"]
                           for sequences in predictions]
        return batch


class LanguagePredictor:
    def __init__(self):
        model_ckpt = "papluca/xlm-roberta-base-language-detection"
        self.model = pipeline("text-classification",
                              model=model_ckpt, device='cuda:0')

    def __call__(self, batch: Dict[str, str]) -> Dict[str, list]:
        predictions = self.model(
            list(batch["query"]), top_k=1, truncation=True)

        batch["output"] = [sequences[0]['label']
                           for sequences in predictions]

        return batch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# label_dict = read_labels(
#     infile='/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/labels.json')
# inverse_label_dict = {v: k for k, v in label_dict.items()}

# # Load the tokenizer
# tokenizer = BertTokenizer.from_pretrained(
#     "bert-base-uncased", do_lower_case=True)


# # load the model and update the weights
# model = BertForSequenceClassification.from_pretrained(
#     "bert-base-uncased",
#     num_labels=len(label_dict),
#     output_attentions=False,
#     output_hidden_states=False,
# )
# # print("base model loaded")
# tensors = {}
# with safetensors.safe_open("/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/finetuned_BERT_first_level.safetensors", framework="pt", device="cpu") as f:
#     for k in f.keys():
#         tensors[k] = f.get_tensor(k)
#     # del tensors["bert.embeddings.position_ids"]
# model.load_state_dict(
#     tensors,
#     strict=False
# )
datasets = ['aol', 'ms-marco', 'orcas', 'aql']
# datasets = ['aol', 'ms-marco', 'orcas']
datasets = ['aol', 'orcas']  # only english
datasets = ['orcas']  # only english

# ds = ray.data.from_items([
#     {"serp_query_text_url": "how much dollar", "price": 9.34},
#     {"serp_query_text_url": "wikipedia", "price": 5.37},
#     {"serp_query_text_url": "youtube", "price": 0.94}
# ])

for dataset_name in datasets:
    reader = read_parquet_data(
        dataset_name=dataset_name, concurrency=1, only_english=True, num_files=1)  # , num_files=1
    ds = reader.read_file()
    print("data read")
    # ds = ray.data.from_items([
    #     {"serp_query_text_url": "how much dollar", "price": 9.34},
    #     {"serp_query_text_url": "wikipedia", "price": 5.37},
    #     {"serp_query_text_url": "youtube", "price": 0.94}
    # ])
    ds = ds.select_columns(['serp_query_text_url'])  # , concurrency=4
    print("columns selected")
    # classified_ds = ds.map_batches(
    #     HuggingFacePredictor, batch_format="pandas", concurrency=1, num_gpus=1, batch_size=1096)  # , concurrency=4
    classified_ds = ds.map_batches(
        debugPredictor, concurrency=1, num_gpus=1, batch_size=1096)  # , concurrency=4

    print("classifier applied")
    final_word_counts = classified_ds.to_pandas()
    print(final_word_counts.columns)
    print(final_word_counts.take([0, 1, 2]))
