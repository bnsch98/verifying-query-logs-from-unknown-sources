# All analysis conducting aggregation and/or group by operations are done in this script.
from dataclasses import dataclass
from functools import cached_property
from json import load
from pathlib import Path
from random import choices
from typing import Literal, Protocol, Iterable, Mapping, Optional
from spacy import load as load_spacy
from spacy import Language
from collections import defaultdict
from string import punctuation

from pandas import DataFrame
from ray import init
from ray.data import read_parquet
from torch import device, argmax
from torch.cuda import is_available as cuda_is_available

from thesis_schneg.model import DatasetName, AggregatorName

import pandas as pd


class _Aggregator(Protocol):
    def aggregate_batch(self, batch: DataFrame) -> DataFrame:
        raise NotImplementedError()

    def __call__(self, batch: DataFrame) -> DataFrame:
        return self.aggregate_batch(batch)


@dataclass(frozen=True)
class ZipfsLawAggregator(_Aggregator):

    word_freq = defaultdict(int)
    tokenizer_name: str = "en_core_web_sm"

    @cached_property
    def _tokenizer(self) -> Language:
        return load_spacy(self.tokenizer_name)

    def count_words(self, batch: DataFrame) -> dict:
        for text in batch['serp_query_text_url']:
            if pd.notna(text):  # Check if query is not NaN.
                words = [tok.text for tok in self._tokenizer(
                    text) if tok.text not in punctuation]
                for word in words:
                    # Convert words to lowercase and count.
                    self.word_freq[word.lower()] += 1

        return dict(self.word_freq)

    def aggregate_batch(self, batch: DataFrame) -> DataFrame:
        res_dict = self.count_words(batch)
        return pd.DataFrame(list(res_dict.items()), columns=['word', 'count'])


def _get_parquet_paths(
    dataset_name: DatasetName,
    sample_files: Optional[int] = None,
    only_english: bool = False,
) -> Iterable[Path]:
    base_path: Path
    if dataset_name == "aol":
        base_path = Path(
            "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/aol_output/"
        )
    elif dataset_name == "ms-marco":
        if only_english:
            base_path = Path(
                "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/lng_filtered_ms-marco/"
            )
        else:
            base_path = Path(
                "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/msmarco_output/"
            )
    elif dataset_name == "orcas":
        base_path = Path(
            "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/orcas_output/"
        )
    elif dataset_name == "aql":
        if only_english:
            base_path = Path(
                "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/lng_filtered_aql/"
            )
        else:
            base_path = Path(
                "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/aql_output/"
            )

    input_paths = [path for path in base_path.iterdir()
                   if path.suffix == ".parquet"]
    if sample_files is not None:
        input_paths = choices(
            population=input_paths,
            k=min(sample_files, len(input_paths)),
        )
    return input_paths


def _get_aggregator(
    aggregator_name: AggregatorName,
) -> _Aggregator:
    if aggregator_name == "zipfs-law":
        return ZipfsLawAggregator()


def aggregate(
    aggregator_name: AggregatorName,
    dataset_name: DatasetName,
    sample_files: Optional[int] = None,
    only_english: bool = False,
    read_concurrency: Optional[int] = None,
    aggregate_concurrency: Optional[int] = None,
    write_results: bool = False,
    # write_concurrency: Optional[int] = None,

) -> None:
    init()

    # Load dataset.
    dataset = read_parquet(
        paths=[
            str(path)
            for path in _get_parquet_paths(
                dataset_name=dataset_name,
                sample_files=sample_files,
                only_english=only_english,
            )
        ],
        concurrency=read_concurrency,
    )

    # Select just the columns we need.
    # TODO: This might need to be adjusted when using other aggregators.
    dataset = dataset.select_columns(["serp_query_text_url"])

    # Load the aggregator.
    aggregateor = _get_aggregator(aggregator_name=aggregator_name)

    # Aggregate in batches.
    dataset = dataset.map_batches(
        aggregateor,
        concurrency=aggregate_concurrency,
        num_gpus=1,
        batch_size=16,
        batch_format="pandas",
    )
    grouped_dataset = dataset.groupby("word").sum("count")
    results = grouped_dataset.to_pandas()
    results.rename(columns={'sum(count)': 'count'}, inplace=True)
    results = results.sort_values(
        by='count', ascending=False)
    print(results.head())
    if write_results:
        results.to_csv(
            f'/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/results_zipfs_law/{dataset_name}_{aggregator_name}.csv')
