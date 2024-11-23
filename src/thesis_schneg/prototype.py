from pathlib import Path
from random import choices
from typing import Iterable, Optional, Callable, Any, Dict
from pandas import DataFrame
from ray import init
from ray.data import read_parquet, Dataset
from ray.data.aggregate import AggregateFn
from ray.data.grouped_data import GroupedData
from thesis_schneg.model import DatasetName, AnalysisName
from json import dumps


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


def write_dataset(dataset: Dataset | GroupedData | DataFrame | dict, write_dir: Path, analysis: str, write_concurrency: Optional[int] = 2) -> None:
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
def _duplicate_row(row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    return [row] * 2


def _extract_words(row: Dict[str, Any]) -> Iterable[str]:
    return row['serp_query_text_url'].split()


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
def query_length_chars_groupby(dataset: Dataset) -> Dataset:
    return dataset.groupby('query_length_chars').count()


def query_length_words_groupby(dataset: Dataset) -> Dataset:
    return dataset.groupby('query_length_words').count()


def unique_queries_groupby(dataset: Dataset) -> GroupedData:
    return dataset.groupby('serp_query_text_url').count()


############################################    Get task-specific modules     ############################################
def _get_col_filter(analysis_name: AnalysisName) -> Optional[Dict[str, Iterable[str]]]:
    if analysis_name == "sum-rows":
        return {'cols': ['serp_query_text_url'], 'nan_filter': ['serp_query_text_url']}
    elif analysis_name == "zipfs-law":
        return {'cols': ['serp_query_text_url'], 'nan_filter': ['serp_query_text_url']}
    elif analysis_name == "query-length-chars":
        return {'cols': ['serp_query_text_url'], 'nan_filter': ['serp_query_text_url']}
    elif analysis_name == "query-length-words":
        return {'cols': ['serp_query_text_url'], 'nan_filter': ['serp_query_text_url']}
    elif analysis_name == "unique-queries":
        return {'cols': ['serp_query_text_url'], 'nan_filter': ['serp_query_text_url']}


def _get_flat_mapping_func(analysis_name: AnalysisName) -> Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]:
    if analysis_name == "sum-rows":
        return None
    elif analysis_name == "zipfs-law":
        return _duplicate_row
    elif analysis_name == "query-length-chars":
        return None
    elif analysis_name == "query-length-words":
        return None
    elif analysis_name == "unique-queries":
        return None


def _get_mapping_func(analysis_name: AnalysisName) -> Optional[Callable[[DataFrame], DataFrame]]:
    if analysis_name == "sum-rows":
        return None
    elif analysis_name == "zipfs-law":
        return None
    elif analysis_name == "query-length-chars":
        return get_length_char
    elif analysis_name == "query-length-words":
        return get_length_word
    elif analysis_name == "unique-queries":
        return None


def _get_aggregator(analysis_name: AnalysisName) -> Optional[AggregateFn]:
    if analysis_name == "sum-rows":
        return sum_rows
    elif analysis_name == "zipfs-law":
        return None
    elif analysis_name == "query-length-chars":
        return None
    elif analysis_name == "query-length-words":
        return None
    elif analysis_name == "unique-queries":
        return unique_queries_agg


def _get_groupby_func(analysis_name: AnalysisName) -> Optional[Callable[[Dataset], Any]]:
    if analysis_name == "sum-rows":
        return None
    elif analysis_name == "zipfs-law":
        return None
    elif analysis_name == "query-length-chars":
        return query_length_chars_groupby
    elif analysis_name == "query-length-words":
        return query_length_words_groupby
    elif analysis_name == "unique-queries":
        return unique_queries_groupby


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

    # Load column filter.
    col_filter = _get_col_filter(analysis_name=analysis_name)

    # Load mapping function.
    mapping_func = _get_mapping_func(analysis_name=analysis_name)

    # Load function for flat mapping.
    flat_mapping_func = _get_flat_mapping_func(analysis_name=analysis_name)

    # Load function for aggregation.
    aggregation_func = _get_aggregator(analysis_name=analysis_name)

    # Load groupby column.
    groupby_func = _get_groupby_func(analysis_name=analysis_name)

    # Load dataset.
    ds = load_dataset(dataset_name=dataset_name, sample_files=sample_files,
                      only_english=only_english, read_concurrency=read_concurrency)

    # Select required columns.
    if col_filter is not None:
        ds = filter_columns(
            dataset=ds, columns=col_filter['cols'], filter_NaN=col_filter['nan_filter'])

    # # Apply mapping function.
    if mapping_func is not None:
        ds = map_dataset(dataset=ds, mapping_func=mapping_func,
                         map_concurrency=map_concurrency, mapping_batch_size=mapping_batch_size, num_gpus=num_gpus)

    # Apply flat mapping function.
    if flat_mapping_func is not None:
        ds = flat_map_dataset(dataset=ds, flat_mapping_func=flat_mapping_func,
                              flatmap_concurrency=flatmap_concurrency, num_cpus=num_cpus, num_gpus=num_gpus)

    # Group by a column.
    if groupby_func is not None:
        ds = groupby_func(dataset=ds)

    # Apply aggregation function.
    if aggregation_func is not None:
        ds = aggregate_dataset(
            dataset=ds, aggregation_func=aggregation_func)

    # Write results.
    write_dataset(dataset=ds, write_dir=write_dir,
                  analysis=analysis_name, write_concurrency=write_concurrency)
