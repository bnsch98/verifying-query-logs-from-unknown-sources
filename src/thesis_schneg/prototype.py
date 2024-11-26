from pathlib import Path
from random import choices
from typing import Iterable, Optional, Callable, Protocol, Union, Any, Dict
from pandas import DataFrame
from ray import init
from ray.data import read_parquet, Dataset
from ray.data.aggregate import AggregateFn
from ray.data.grouped_data import GroupedData
from thesis_schneg.model import DatasetName, AnalysisName
from json import dumps
from functools import cached_property
from dataclasses import dataclass
from spacy import load as spacy_load, Language


############################################    Requirements for basic modules    ############################################
class _spacy_framework(Protocol):
    def get_tokens(self, row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError()

    def __call__(self, row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        return self.get_tokens(row)


@dataclass(frozen=True)
class SpacyModel(_spacy_framework):

    @cached_property
    def spacy_model(self) -> Language:
        return spacy_load("en_core_web_sm")

    def get_tokens(self, row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        # Get tokens from text
        doc = self.spacy_model(row["serp_query_text_url"])
        tokens = [{"word": token.text} for token in doc]
        return tokens


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


############################################    Basic Modules    ############################################
def load_dataset(dataset_name: DatasetName,
                 sample_files: Optional[int] = None,
                 only_english: bool = False,
                 read_concurrency: Optional[int] = None,
                 ) -> Dataset:

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
    return dataset


def filter_columns(dataset: Dataset,
                   columns: Optional[Iterable[str]
                                     ] = ['serp_query_text_url'],
                   # filter out NaN values for fault tolerance
                   filter_NaN: Optional[Iterable[str]] = [
                       'serp_query_text_url'],
                   ) -> Dataset:

    def col_filter(df: DataFrame, col=columns, filter=filter_NaN) -> DataFrame:
        if filter is not None:
            return df.dropna(subset=filter)[col]
        else:
            return df[col]

    return dataset.map_batches(col_filter, batch_format="pandas")


def map_dataset(dataset: Dataset,
                mapping_func: Callable[[DataFrame], DataFrame],
                map_concurrency: Optional[int] = None,
                mapping_batch_size: int = 16,
                num_gpus: int = None) -> Dataset:
    return dataset.map_batches(
        mapping_func,
        concurrency=map_concurrency,
        num_gpus=num_gpus,
        batch_size=mapping_batch_size,
        batch_format="pandas",
    )


def flat_map_dataset(dataset: Dataset,
                     flat_mapping_func: Callable[[Dict[str, Any]], Dict[str, Any]],
                     flatmap_concurrency: Optional[int] = None,
                     num_cpus: Optional[int] = None,
                     num_gpus: Optional[int] = None
                     ) -> Dataset:
    return dataset.flat_map(fn=flat_mapping_func, concurrency=flatmap_concurrency, num_cpus=num_cpus, num_gpus=num_gpus)


def aggregate_dataset(dataset: Dataset, aggregation_func: AggregateFn) -> Optional[Dict[str, Any]]:
    return dataset.aggregate(aggregation_func)


def write_dataset(dataset: Union[Dict, Dataset, DataFrame], write_dir: Path, analysis: str, write_concurrency: Optional[int] = 2) -> None:
    if type(dataset) is dict:
        # Make directory to work around FileNotFoundError
        write_dir.mkdir(parents=True, exist_ok=True)
        # Distinguish between nested dict and flat dict. We rule out deeper nesting.
        if type(dataset[analysis]) is dict:
            # Write json file
            with write_dir.joinpath("result.json").open("w+", encoding="utf-8") as f:
                f.write(dumps(dataset[analysis]))
        else:
            # Write json file
            with write_dir.joinpath("result.json").open("w+", encoding="utf-8") as f:
                f.write(dumps(dataset))
    elif type(dataset) is Dataset:
        dataset.write_parquet(path=str(write_dir),
                              concurrency=write_concurrency)
    elif type(dataset) is DataFrame:
        dataset.to_csv(path_or_buf=write_dir.joinpath(
            "result.csv"), index=False)
    else:
        print("Unknown type of output")


############################################    Task Specific Functions    ############################################

# Mapping functions
def get_length_char(batch: DataFrame) -> DataFrame:
    batch['query_length_chars'] = batch['serp_query_text_url'].apply(len)
    return batch


def get_length_word(batch: DataFrame) -> DataFrame:
    batch['query_length_words'] = batch['serp_query_text_url'].apply(
        lambda x: len(x.split()))
    return batch


# Flat mapping functions
def _extract_chars(row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    return [{"char": char} for char in row['serp_query_text_url'].replace(" ", "")]


# Aggregation functions
sum_rows = AggregateFn(
    init=lambda column: 0,
    # Apply this to each row to produce a partial aggregate result
    accumulate_row=lambda a, row: a + 1,
    # Apply this to merge partial aggregate results into a final result
    merge=lambda a1, a2: a1 + a2,
    name="sum-rows"
)

unique_queries_agg = AggregateFn(
    init=lambda column: {"sum": 0, "unique": 0},
    # Apply this to each row to produce a partial aggregate result

    accumulate_row=lambda a, row: acc_row(a, row),
    # Apply this to merge partial aggregate results into a final result
    merge=lambda a1, a2: sum_dict(a1, a2),

    name="unique-queries"
)


# Helper functions for aggregation
def acc_row(aggr_dict: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    aggr_dict["sum"] += row["count()"]
    aggr_dict["unique"] += 1
    return aggr_dict


def sum_dict(a1: Dict[str, Any], a2: Dict[str, Any]) -> Dict[str, Any]:
    return {k: int(a1.get(k, 0) + a2.get(k, 0)) for k in set(a1) | set(a2)}


# Group-by function
def groupby_queries(dataset: Dataset) -> Dataset:
    return dataset.groupby('serp_query_text_url').count()


def groupby_words(dataset: Dataset) -> Dataset:
    return dataset.groupby('word').count()


def groupby_chars(dataset: Dataset) -> Dataset:
    return dataset.groupby('char').count()


def query_length_chars_groupby(dataset: Dataset) -> Dataset:
    return dataset.groupby('query_length_chars').count()


def query_length_words_groupby(dataset: Dataset) -> Dataset:
    return dataset.groupby('query_length_words').count()


def unique_queries_groupby(dataset: Dataset) -> GroupedData:
    return dataset.groupby('serp_query_text_url').count()


############################################    Get task-specific modules     ############################################
def _get_module_specifics(analysis_name: AnalysisName) -> Dict[str, Any]:
    if analysis_name == "sum-rows":
        return {'groupby_func': None, 'aggregator': sum_rows, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': {'cols': ['serp_query_text_url'], 'nan_filter': ['serp_query_text_url']}}
    elif analysis_name == "zipfs-law-queries":
        return {'groupby_func': groupby_queries, 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': {'cols': ['serp_query_text_url'], 'nan_filter': ['serp_query_text_url']}}
    elif analysis_name == "zipfs-law-words":
        return {'groupby_func': groupby_words, 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': SpacyModel(), 'col_filter': {'cols': ['serp_query_text_url'], 'nan_filter': ['serp_query_text_url']}}
    elif analysis_name == "zipfs-law-chars":
        return {'groupby_func': groupby_chars, 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': _extract_chars, 'col_filter': {'cols': ['serp_query_text_url'], 'nan_filter': ['serp_query_text_url']}}
    elif analysis_name == "query-length-chars":
        return {'groupby_func': query_length_chars_groupby, 'aggregator': None, 'mapping_func': get_length_char, 'flat_mapping_func': None, 'col_filter': {'cols': ['serp_query_text_url'], 'nan_filter': ['serp_query_text_url']}}
    elif analysis_name == "query-length-words":
        return {'groupby_func': query_length_words_groupby, 'aggregator': None, 'mapping_func': get_length_word, 'flat_mapping_func': None, 'col_filter': {'cols': ['serp_query_text_url'], 'nan_filter': ['serp_query_text_url']}}
    elif analysis_name == "unique-queries":
        return {'groupby_func': unique_queries_groupby, 'aggregator': unique_queries_agg, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': {'cols': ['serp_query_text_url'], 'nan_filter': ['serp_query_text_url']}}


############################################    Pipeline    ############################################
def analysis_pipeline(dataset_name: DatasetName,
                      analysis_name: AnalysisName,
                      sample_files: Optional[int] = None,
                      only_english: bool = False,
                      read_concurrency: Optional[int] = None,
                      map_concurrency: Optional[int] = None,
                      mapping_batch_size: int = 16,
                      flatmap_concurrency: Optional[int] = None,
                      num_cpus: Optional[int] = None,
                      num_gpus: Optional[int] = None,
                      write_dir: Path = Path(
        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis"),
    write_concurrency: Optional[int] = 2
) -> None:
    if sample_files is not None:
        write_dir = write_dir.joinpath(
            f"{dataset_name}-{analysis_name}-{sample_files}")
    else:
        write_dir = write_dir.joinpath(f"{dataset_name}-{analysis_name}-all")

    init()

    # Load module specifics
    module_specifics = _get_module_specifics(analysis_name=analysis_name)

    # Load dataset.
    ds = load_dataset(dataset_name=dataset_name, sample_files=sample_files,
                      only_english=only_english, read_concurrency=read_concurrency)

    # Select required columns.
    if module_specifics['col_filter'] is not None:
        ds = filter_columns(
            dataset=ds, columns=module_specifics['col_filter']['cols'], filter_NaN=module_specifics['col_filter']['nan_filter'])

    # # Apply mapping function.
    if module_specifics['mapping_func'] is not None:
        ds = map_dataset(dataset=ds, mapping_func=module_specifics['mapping_func'],
                         map_concurrency=map_concurrency, mapping_batch_size=mapping_batch_size, num_gpus=num_gpus)

    # Apply flat mapping function.
    if module_specifics['flat_mapping_func'] is not None:
        ds = flat_map_dataset(dataset=ds, flat_mapping_func=module_specifics['flat_mapping_func'],
                              flatmap_concurrency=flatmap_concurrency, num_cpus=num_cpus, num_gpus=num_gpus)

    # Group by a column.
    if module_specifics['groupby_func'] is not None:
        ds = module_specifics['groupby_func'](dataset=ds)

    # Apply aggregation function.
    if module_specifics['aggregator'] is not None:
        ds = aggregate_dataset(
            dataset=ds, aggregation_func=module_specifics['aggregator'])

    print(ds.take(10))

    # Write results.
    write_dataset(dataset=ds, write_dir=write_dir,
                  analysis=analysis_name, write_concurrency=write_concurrency)
