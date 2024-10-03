# Use a pipeline as a high-level helper
from nltk.tokenize import word_tokenize
from collections import defaultdict
from ray import init
from ray.data import read_parquet
import os
import pandas as pd
import nltk
import matplotlib.pyplot as plt

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


def count_words(df: pd.DataFrame, column: str) -> dict:
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    word_freq = defaultdict(int)

    for text in df[column]:
        if pd.notna(text):  # Überprüfen, ob der Text nicht NaN ist
            words = word_tokenize(text)
            for word in words:
                # Wörter in Kleinbuchstaben umwandeln und zählen
                word_freq[word.lower()] += 1

    return dict(word_freq)


def count_words_in_batch(batch: pd.DataFrame) -> pd.DataFrame:
    word_freq = count_words(batch, 'serp_query_text_url')
    return pd.DataFrame(list(word_freq.items()), columns=['word', 'count'])


nltk.download('punkt_tab')

datasets = ['aol', 'ms-marco', 'orcas', 'aql']
# datasets = ['aol', 'ms-marco', 'orcas']
# datasets = ['aol', 'orcas']  # only english


# datasets = ['aol']
plt.figure(figsize=(10, 6))

for dataset_name in datasets:
    reader = read_parquet_data(
        dataset_name=dataset_name, concurrency=5, num_files=3, only_english=True)  # , num_files=1
    ds = reader.read_file()
    word_counts = ds.map_batches(count_words_in_batch, batch_format="pandas")

    # Zusammenführen der Ergebnisse
    word_counts_df = word_counts.to_pandas()
    final_word_counts = word_counts_df.groupby('word').sum().reset_index()
    # Sortieren nach der Spalte 'count'
    final_word_counts = final_word_counts.sort_values(
        by='count', ascending=False)

    # print(final_word_counts)
    # print(type(final_word_counts))

    # Plotten der sortierten Spalte 'count'
    plt.plot(range(1, len(final_word_counts) + 1),
             final_word_counts['count'], label=dataset_name)

    final_word_counts.to_csv(
        '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/results_zipfs_law/' + dataset_name + '.csv')


plt.xscale('log')
plt.yscale('log')
plt.xlabel('Rank of the word')
plt.ylabel('Frequency of the word')
plt.title('Zipfs Law: Frequency vs. Rank')
plt.legend(title='Dataset')
plt.grid(True)
plt.show()
