from json import dumps
from thesis_schneg.model import DatasetName, AnalysisName
from ray.data.grouped_data import GroupedData
from ray.data.aggregate import AggregateFn
from ray.data import read_parquet, Dataset
from ray import init
from pandas import DataFrame
from typing import Iterable, Optional, Callable, Union, Any, Dict
from random import choices
from pathlib import Path
from datasketch import HyperLogLog
from datetime import datetime
from functools import partial

############################################    Requirements for basic modules    #####################################


def _get_parquet_paths(
    dataset_name: DatasetName,
    analysis_name: AnalysisName,
    read_dir: Optional[Iterable[str]] = None,
    struc_level: Optional[str] = None,
    sample_files: Optional[int] = None,
    only_english: bool = False,
    which_half: Optional[str] = None
) -> Iterable[Path]:
    base_path: Path
    if read_dir is None:
        if analysis_name in ["character-count-frequencies", "word-count-frequencies", "entity-count-frequencies", "query-count-frequencies", "filter-urls", "aql-anomaly"]:
            assert struc_level is not None, "Structural level must be specified by \"--struc-level\" [queries, named-entities, words]"
            base_path = Path(
                f"/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis/{dataset_name}-get-lengths-{struc_level}-all/"
            )
            assert base_path.is_dir(
            ), f"No directory found for dataset = {dataset_name} and struc_level = {struc_level}"
        else:
            if struc_level in [None, "queries"]:
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
            else:
                base_path = Path(
                    f"/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis/{dataset_name}-extract-{struc_level}-all/"
                )
                assert base_path.is_dir(
                ), f"No directory found for dataset = {dataset_name} and struc_level = {struc_level}"

        input_paths = [path for path in base_path.iterdir()
                       if path.suffix == ".parquet"]
        assert len(input_paths) > 0, f"No parquet files found in {base_path}"

        assert which_half in [None, "first",
                              "second"], "Invalid half specified"
        assert not (
            sample_files is not None and which_half is not None), "Cannot specify both \"sample_files\" and \"which_half\""

        if sample_files is not None and which_half is None:
            input_paths = choices(
                population=input_paths,
                k=min(sample_files, len(input_paths)),
            )
        elif sample_files is None and which_half is not None:
            if which_half == "first":
                input_paths = input_paths[:len(input_paths)//2]
            elif which_half == "second":
                input_paths = input_paths[len(input_paths)//2:]
        assert input_paths, f"No files found in {base_path.name}"
    else:
        input_paths = []
        for path in read_dir:
            print(f"Reading from {path}")
            if isinstance(path, str):
                path = Path(path)
            else:
                print(f"Path is {type(path)}")
            assert path.is_dir(), f"Invalid directory {path}"
            input_paths = input_paths + [dir for dir in path.iterdir()
                                         if dir.suffix == ".parquet"]
        assert len(input_paths) > 0, f"No parquet files found in {read_dir}"

        if sample_files is not None:
            input_paths = choices(
                population=input_paths,
                k=min(sample_files, len(input_paths)),
            )
    return input_paths


############################################    Basic Modules    #######################################
def load_dataset(dataset_name: DatasetName,
                 analysis_name: AnalysisName,
                 struc_level: Optional[str] = None,
                 sample_files: Optional[int] = None,
                 only_english: bool = False,
                 read_concurrency: Optional[int] = None,
                 columns: Optional[Iterable[str]] = None,
                 memory_scaler: float = 1.0,
                 which_half: Optional[str] = None,
                 read_dir: Optional[Iterable[str]] = None
                 ) -> Dataset:

    # Load dataset.
    dataset = read_parquet(
        paths=[
            str(path)
            for path in _get_parquet_paths(
                dataset_name=dataset_name,
                analysis_name=analysis_name,
                read_dir=read_dir,
                struc_level=struc_level,
                sample_files=sample_files,
                only_english=only_english,
                which_half=which_half
            )
        ],
        concurrency=read_concurrency,
        columns=columns,
        ray_remote_args={"memory": memory_scaler*1000*1000*1000}
    )
    return dataset


def map_dataset(dataset: Dataset,
                mapping_func: Callable[[DataFrame], DataFrame],
                concurrency: Optional[int] = None,
                batch_size: int = 16,
                num_gpus: float = None,
                num_cpus: float = None,
                memory_scaler: float = 1.0) -> Dataset:
    return dataset.map_batches(
        mapping_func,
        concurrency=concurrency,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        batch_size=batch_size,
        batch_format="pandas",
        memory=memory_scaler*1000*1000*1000,
    )


def flat_map_dataset(dataset: Dataset,
                     flat_mapping_func: Callable[[Dict[str, Any]], Dict[str, Any]],
                     concurrency: Optional[int] = None,
                     num_cpus: Optional[float] = None,
                     num_gpus: Optional[float] = None,
                     memory_scaler: float = 1.0
                     ) -> Dataset:
    return dataset.flat_map(fn=flat_mapping_func, concurrency=concurrency, num_cpus=num_cpus, num_gpus=num_gpus, memory=memory_scaler*1000*1000*1000)


def aggregate_dataset(dataset: Dataset, aggregation_func: AggregateFn) -> Optional[Dict[str, Any]]:
    return dataset.aggregate(aggregation_func)


def map_groups(dataset: GroupedData, map_group_func: Callable[[Any], Any], memory_scaler: float = 1.0, concurrency: Optional[int] = None) -> Dataset:
    return dataset.map_groups(map_group_func, concurrency=concurrency, memory=memory_scaler*1000*1000*1000)


def write_dataset(dataset: Union[Dict, Dataset, DataFrame], write_dir: Path, analysis_name: str, struc_level: str, dataset_name: str, sample_files: int, which_half: Optional[str], read_dir: Optional[Iterable[str]], _year: Optional[int], write_concurrency: Optional[int] = 2, only_english: bool = False) -> None:
    # check if wirte_dir is Path
    if type(write_dir) is not Path:
        write_dir = Path(write_dir)

    # Specifiy output directory if standard path was parsed. If not, we assume that a specific path was parsed via the CLI and no further specification is necessary.
    if str(write_dir) == "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis":
        output_folder = f"{dataset_name}-{analysis_name}"

        if struc_level is not None:
            output_folder += f"-{struc_level}"
        if which_half is not None:
            output_folder += f"-{which_half}"
        if sample_files is not None:
            output_folder += f"-{sample_files}"
            if read_dir is not None:
                if "google" in str(read_dir):
                    output_folder += "-google"
                elif "english" in str(read_dir):
                    output_folder += "-english"
                else:
                    output_folder += "-special"
        else:
            if read_dir is not None:
                if "google" in str(read_dir):
                    output_folder += "-google"
                elif "english" in str(read_dir):
                    output_folder += "-english"
                else:
                    output_folder += "-special"
            else:
                output_folder += "-all"
        if only_english:
            output_folder += "-english"

        write_dir = write_dir.joinpath(output_folder)

    # Write output
    if isinstance(dataset, dict):
        if not write_dir.exists():
            # Make directory to work around FileNotFoundError
            write_dir.mkdir(parents=True, exist_ok=True)
        # Distinguish between nested dict and flat dict. We rule out deeper nesting.
        if analysis_name in dataset.keys() and isinstance(dataset[analysis_name], dict):
            # Write json file
            with write_dir.joinpath(f"{analysis_name}-{dataset_name}-{_year}.json").open("w+", encoding="utf-8") as f:
                f.write(dumps(dataset[analysis_name]))
        else:
            # Write json file
            with write_dir.joinpath(f"{analysis_name}-{dataset_name}-{_year}.json").open("w+", encoding="utf-8") as f:
                f.write(dumps(dataset))
    elif isinstance(dataset, Dataset):
        # Delete old files
        if write_dir.exists():
            [f.unlink() for f in write_dir.glob("*") if f.is_file()]
        # Write parquet file
        dataset.write_parquet(path=str(write_dir),
                              concurrency=write_concurrency)
    elif isinstance(dataset, DataFrame):
        # Write csv file
        dataset.to_csv(path_or_buf=write_dir.joinpath(
            "result.csv"), index=False)
    elif isinstance(dataset, list):
        # We assume a list of dicts -> write json lines file
        if not write_dir.exists():
            # Make directory to work around FileNotFoundError
            write_dir.mkdir(parents=True, exist_ok=True)
        # Delete old files
        [f.unlink() for f in write_dir.glob("*") if f.is_file()]
        # Write json lines file
        with write_dir.joinpath("result.jsonl").open("w+", encoding="utf-8") as f:
            for item in dataset:
                f.write(dumps(item) + "\n")
    else:
        print("Unknown type of output")


###########################################    Task Specific Functions    ###########################################
# Mapping functions
def filter_by_year(batch: DataFrame, year: Iterable[int]) -> DataFrame:
    # filter out empty timestamps
    batch = batch[~batch['serp_timestamp'].isnull()]
    batch['year'] = batch['serp_timestamp'].apply(
        lambda x: datetime.fromtimestamp(x).year)
    return batch[batch['year'] == year]


def set_lowercase(batch: DataFrame) -> DataFrame:
    batch['serp_query_text_url'] = batch['serp_query_text_url'].str.lower()
    return batch


# Aggregation functions
hyperloglog_agg_row = AggregateFn(
    init=lambda _: HyperLogLog(p=16),
    accumulate_row=lambda hll, row: acc_hyperloglog_row(hll, row),
    merge=lambda hll1, hll2: merge_hyperloglog(hll1, hll2),
    finalize=lambda hll: hll.count(),
    name="hyperloglog-agg"
)


# Helper functions for aggregation
def acc_hyperloglog_row(hll: HyperLogLog, row: Dict[str, Any]) -> HyperLogLog:
    hll.update(row["serp_query_text_url"].encode('utf8'))
    return hll


def merge_hyperloglog(hll1: HyperLogLog, hll2: HyperLogLog) -> HyperLogLog:
    hll1.merge(hll2)
    return hll1


###########################################    Get task-specific modules     #########################################
def _get_module_specifics(analysis_name: AnalysisName) -> Dict[str, Any]:

    if analysis_name == "count-regular":
        return {'groupby_func': None, 'aggregator': hyperloglog_agg_row, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url', 'serp_timestamp']}
    elif analysis_name == "count-lowercase":
        return {'groupby_func': None, 'aggregator': hyperloglog_agg_row, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url', 'serp_timestamp']}


############################################    Pipeline    ################################################
def analysis_pipeline(dataset: Iterable[DatasetName],
                      analysis_name: AnalysisName,
                      struc_level: Optional[str] = None,
                      sample_files: Optional[int] = None,
                      only_english: bool = False,
                      which_half: Optional[str] = None,
                      read_concurrency: Optional[int] = None,
                      concurrency: Optional[int] = None,
                      batch_size: int = 16,
                      memory_scaler: float = 1.0,
                      num_cpus: Optional[float] = None,
                      num_gpus: Optional[float] = None,
                      write_dir: Path = Path(
        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis"),
    write_concurrency: Optional[int] = 2,
    read_dir: Optional[Iterable[str]] = None
) -> None:
    assert struc_level in [None, "words",
                           "named-entities", "queries"], "Invalid structural level"
    assert dataset[0] == 'aql', "'aql' has to be the first dataset in the list"
    init()

    # Load module specifics
    module_specifics = _get_module_specifics(
        analysis_name=analysis_name)

    # load dataset
    ds = load_dataset(dataset_name=dataset[0], struc_level=struc_level, sample_files=sample_files,
                      only_english=only_english, read_concurrency=read_concurrency, columns=module_specifics['col_filter'], memory_scaler=memory_scaler, which_half=which_half, analysis_name=analysis_name, read_dir=read_dir)
    ds_comp = load_dataset(dataset_name=dataset[1], struc_level=struc_level, sample_files=sample_files,
                           only_english=only_english, read_concurrency=read_concurrency, columns=module_specifics['col_filter'], memory_scaler=memory_scaler, which_half=which_half, analysis_name=analysis_name, read_dir=read_dir)
    # preprocess dataset
    if analysis_name == "count-lowercase":
        ds = map_dataset(dataset=ds, mapping_func=set_lowercase,
                         concurrency=concurrency, batch_size=batch_size, num_gpus=num_gpus, num_cpus=num_cpus, memory_scaler=memory_scaler)
        ds_comp = map_dataset(dataset=ds_comp, mapping_func=set_lowercase,
                              concurrency=concurrency, batch_size=batch_size, num_gpus=num_gpus, num_cpus=num_cpus, memory_scaler=memory_scaler)

    ds_comp_agg = aggregate_dataset(
        dataset=ds_comp, aggregation_func=module_specifics['aggregator'])

    years = [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
             2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

    # years = [2021, 2022]

    results = []

    for _year in years:
        try:
            # Load dataset.
            ds2 = map_dataset(dataset=ds, mapping_func=partial(filter_by_year, year=_year),
                              concurrency=concurrency, batch_size=batch_size, num_gpus=num_gpus, num_cpus=num_cpus, memory_scaler=memory_scaler)
            # Aggregate AQL dataset.
            aql_agg = aggregate_dataset(
                dataset=ds2, aggregation_func=module_specifics['aggregator'])
            # Union datasets.
            ds2 = ds2.union(ds_comp)
            # Apply aggregation function to combination of AQL and comparison dataset.
            if module_specifics['aggregator'] is not None:
                ds2 = aggregate_dataset(
                    dataset=ds2, aggregation_func=module_specifics['aggregator'])

            # Add AQL aggregation result to dataset.
            if aql_agg is not None:
                ds2['aql-agg'] = aql_agg['hyperloglog-agg']
                ds2['combined-agg'] = ds2.pop('hyperloglog-agg')
                ds2['compare-agg'] = ds_comp_agg['hyperloglog-agg']
                ds2['aql_agg+comp_agg-combined_agg'] = ds2['aql-agg'] + \
                    ds2['compare-agg'] - ds2['combined-agg']
                if aql_agg['hyperloglog-agg'] < 1:
                    ds2['duplicate-ratio-aql'] = "ratio not defined (AQL agg < 1)"
                else:
                    ds2['duplicate-ratio-aql'] = (ds2['aql_agg+comp_agg-combined_agg'] / aql_agg['hyperloglog-agg'],
                                                  f"{ds2['aql_agg+comp_agg-combined_agg'] / aql_agg['hyperloglog-agg']:.2%}")
                if ds_comp_agg['hyperloglog-agg'] < 1:
                    ds2['duplicate-ratio-comp'] = "ratio not defined (comp agg < 1)"
                else:
                    ds2['duplicate-ratio-comp'] = (ds2['aql_agg+comp_agg-combined_agg'] / ds_comp_agg['hyperloglog-agg'],
                                                   f"{ds2['aql_agg+comp_agg-combined_agg'] / ds_comp_agg['hyperloglog-agg']:.2%}")
                ds2['year'] = _year
                ds2['dataset'] = "-".join(dataset)
                ds2['analysis_name'] = analysis_name
            else:
                raise ValueError(
                    f"No aggregation result of AQL for year {_year} found. Check if AQL contains data for this year.")
            results.append(ds2)
        except Exception as e:
            print(f"Error processing year {_year}: {e}")
            # Write results.
            # Determine dataset name for writing.
            dataset_name = "-".join(dataset)
            write_dataset(dataset=results, write_dir=write_dir,
                          analysis_name=analysis_name, write_concurrency=write_concurrency, struc_level=struc_level, dataset_name=dataset_name, sample_files=sample_files, which_half=which_half, read_dir=read_dir, only_english=only_english, _year=_year)

    # Write results.
    # Determine dataset name for writing.
    dataset_name = "-".join(dataset)
    write_dataset(dataset=results, write_dir=write_dir,
                  analysis_name=analysis_name, write_concurrency=write_concurrency, struc_level=struc_level, dataset_name=dataset_name, sample_files=sample_files, which_half=which_half, read_dir=read_dir, only_english=only_english, _year=_year)
