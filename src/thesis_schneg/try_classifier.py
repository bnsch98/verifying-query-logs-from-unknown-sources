# from ray import init
from typing import Dict
# from ray.data import range
# import ray
import argparse
from argparse import ArgumentParser
import numpy as np
import pandas as pd
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
import time


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


def predict_language(model, row):
    row['language'] = model(row['query'], max_length=20,
                            num_return_sequences=1)[0][0]['label']
    return row


# Initialize Ray (and connect to cluster).
# init()

# input_path = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/orcas_output'
# input_path = "/mnt/ceph/storage/data-in-progress/data-research/web-search/archive-query-log/focused/corpus/full/2023-05-22/serps/part-00004.gz"


# aql_dataloader = Ray_Dataloader(
#     file_type="parquet", path_dataset=input_path,  multi=False)  # num_files=2,

# ds_aql = aql_dataloader.read_file()
# print(ds_aql.schema())

# ds_query = ds_aql.select_columns(['query'])

# predictions = ds_query.map_batches(
#     LanguagePredictor,
#     concurrency=2,
# )

# predictions.take_batch(5)
def read_labels(infile):
    with open(infile, "r") as fp:
        return json.load(fp)


# Pfad zum Verzeichnis, das das Modell und die Labels enthält
model_dir = '/home/benjamin/studium/masterarbeit/finetuned_BERT_first_level'

"""Short script describing how to load pretrained bert model for intent prediction on new data."""


# device = torch.device("cpu")
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"Script started at {start_timestamp}")
    parser = ArgumentParser()
    parser.add_argument(
        "--infile", default="/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/orcas_cleaned/orcas_small_5k.csv", type=str)
    parser.add_argument("--model_name", default="bert-base-uncased", type=str)
    parser.add_argument(
        "--model_path",
        default="/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/finetuned_BERT_first_level.model",
        type=str,
    )
    parser.add_argument(
        "--labels_path",
        default="/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/labels.json",
        type=str,
    )
    args = parser.parse_args()

    label_dict = read_labels(infile=args.labels_path)
    inverse_label_dict = {v: k for k, v in label_dict.items()}

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        args.model_name, do_lower_case=True)
    print(f"type of tokenizer: {type(tokenizer)}")
    # load the model and update the weights
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_dict),
        output_attentions=False,
        output_hidden_states=False,
    )
    print(f"type of model: {type(model)}")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"base model loaded at {timestamp}")
    tensors = {}
    with safetensors.safe_open("/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/finetuned_BERT_first_level.safetensors", framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
        # del tensors["bert.embeddings.position_ids"]
    model.load_state_dict(
        tensors,
        strict=False
    )
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"weights loaded at {timestamp}")
    # load your data
    df = pd.read_csv(args.infile, sep=",")
    INPUT_TEXT_COLUMN = "serp_query_text_url"
    queries = df[INPUT_TEXT_COLUMN].tolist()
    model = model.to(device)
    # iterate over the data to get the predictions
    for query in queries:
        if not pd.isna(query):
            inputs = tokenizer(query, return_tensors="pt").to(device)
            logits = model(**inputs).logits.cpu().detach().numpy()
            predicted_intent = inverse_label_dict[np.argmax(logits)]
        print(f"{query=}, {predicted_intent=}")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"model inference completed at {timestamp}")
    print(f"model inference started at {start_timestamp}")
    if torch.cuda.is_available():
        print("cuda used")
