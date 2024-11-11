# Use a pipeline as a high-level helper
# from nltk.tokenize import word_tokenize
from collections import defaultdict
from ray import init
from ray.data import read_parquet
import os
import pandas as pd
# import nltk
import spacy
from string import punctuation

init()

nlp = spacy.load("en_core_web_sm")


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


def count_words(df: pd.DataFrame, column: str) -> dict:
    # try:
    #     nltk.data.find('tokenizers/punkt_tab')
    # except LookupError:
    #     nltk.download('punkt_tab')
    nlp = spacy.load("en_core_web_sm")
    word_freq = defaultdict(int)

    for text in df[column]:
        if pd.notna(text):  # Überprüfen, ob der Text nicht NaN ist
            words = [tok.text for tok in nlp(
                text) if tok.text not in punctuation]
            for word in words:
                # Wörter in Kleinbuchstaben umwandeln und zählen
                word_freq[word.lower()] += 1

    return dict(word_freq)


def count_words_in_batch(batch: pd.DataFrame) -> pd.DataFrame:
    word_freq = count_words(batch, 'serp_query_text_url')
    return pd.DataFrame(list(word_freq.items()), columns=['word', 'count'])


class WordCounter:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.word_freq = defaultdict(int)

    def count_words(self, df: pd.DataFrame, column: str) -> dict:
        for text in df[column]:
            if pd.notna(text):  # Überprüfen, ob der Text nicht NaN ist
                words = [tok.text for tok in self.nlp(
                    text) if tok.text not in punctuation]
                for word in words:
                    # Wörter in Kleinbuchstaben umwandeln und zählen
                    self.word_freq[word.lower()] += 1

        return dict(self.word_freq)

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        word_freq = self.count_words(batch, 'serp_query_text_url')
        return pd.DataFrame(list(word_freq.items()), columns=['word', 'count'])


datasets = ['aol', 'ms-marco', 'orcas', 'aql']
# datasets = ['aol', 'ms-marco', 'orcas']
datasets = ['aol', 'orcas']  # only english
datasets = ['aol']  # only english


for dataset_name in datasets:
    reader = read_parquet_data(
        dataset_name=dataset_name, concurrency=5, only_english=True, num_files=1)  # , num_files=1
    ds = reader.read_file()
    word_counts = ds.map_batches(
        WordCounter, batch_format="pandas", concurrency=1)  # , concurrency=4

    # Gruppieren und Summieren der 'count'-Werte mit Ray
    grouped_word_counts = word_counts.groupby("word").sum("count")

    # # Zusammenführen der Ergebnisse
    # word_counts_df = word_counts.to_pandas()
    # final_word_counts = word_counts_df.groupby('word').sum().reset_index()

    final_word_counts = grouped_word_counts.to_pandas()

    # print(final_word_counts.columns)
    # print(final_word_counts.head())
    final_word_counts.rename(columns={'sum(count)': 'count'}, inplace=True)
    # Sortieren nach der Spalte 'count'
    final_word_counts = final_word_counts.sort_values(
        by='count', ascending=False)
    print(final_word_counts.head())

    # # print(final_word_counts)
    # # print(type(final_word_counts))

    final_word_counts.to_csv(
        '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/results_zipfs_law/' + dataset_name + '.csv')
