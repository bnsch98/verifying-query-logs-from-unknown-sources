# Use a pipeline as a high-level helper
from transformers import pipeline
from ray import init
from typing import Dict
from ray.data import read_parquet
from ray.data import from_pandas
import os
import pandas as pd

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


class SpamPredictor:
    def __init__(self):
        model_ckpt = "mrm8488/bert-tiny-finetuned-enron-spam-detection"
        self.model = pipeline("text-classification", model=model_ckpt)

    def __call__(self, batch: Dict[str, str]) -> Dict[str, list]:
        predictions = self.model(
            list(batch["serp_query_text_url"]), top_k=1, truncation=True)

        batch["output"] = [sequences[0]['label']
                           for sequences in predictions]

        return batch


def predict_spam(model, row):
    row['output'] = model(row['serp_query_text_url'], max_length=20,
                          num_return_sequences=1)[0][0]['label']
    return row


datasets = ['aol', 'ms-marco', 'orcas', 'aql']
datasets = ['aol', 'ms-marco', 'orcas']
datasets = ['aol', 'orcas']  # only english


datasets = ['aol']

##      TEST    ##
# # Create a test dataset
# test_data = pd.DataFrame({
#     'serp_query_text_url': [
#         "This is a test sentence.",
#         "Another example of a query.",
#         "Spam detection is important.",
#         "How to write unit tests in Python?",
#         "Machine learning models can be complex."
#         "Your account has been compromised. Please enter your personal information to secure it."
#         "Hi Peter, I hope you are doing well. I wanted to ask you about the meeting tomorrow.",
#     ]
# })

# # Convert the test dataset to a Ray Dataset
# test_ds = from_pandas(test_data)

# # Use the LanguagePredictor to classify the test dataset
# predictions = test_ds.map_batches(SpamPredictor, batch_size=2, concurrency=5)
# print(predictions.take_all())
# TEST END        ## (erfolgreich)

for dataset_name in datasets:
    reader = read_parquet_data(
        dataset_name=dataset_name, concurrency=5, num_files=1)  # , num_files=1
    ds = reader.read_file()
    ds_query = ds.select_columns(['serp_query_text_url'])
    predictions = ds_query.map_batches(
        SpamPredictor,
        concurrency=5,)
    predictions.take_batch(5)
