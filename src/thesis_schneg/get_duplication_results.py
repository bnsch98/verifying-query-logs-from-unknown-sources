from pathlib import Path
from typing import Dict, Iterable, Union, Any
from pandas import DataFrame, read_csv as pd_read_csv, read_parquet as pd_read_parquet, concat as pd_concat
from json import load as json_load, dumps
from thesis_schneg.model import DatasetName, AnalysisName


############################################    Basic Modules    #######################################
def read_data(data_dir: Path) -> Union[Dict[str, Any], Iterable[Any], DataFrame]:
    # check if directory exists
    if not data_dir.exists():
        raise FileNotFoundError(f"The directory {data_dir} does not exist.")
    # check if directory contains a file
    if not any(data_dir.iterdir()):
        raise FileNotFoundError(f"The directory {data_dir} is empty.")
    # get result files
    files = [file for file in data_dir.iterdir() if file.is_file()]
    # check if file extensions are uniform
    extensions = set(file.suffix for file in files)
    if len(extensions) > 1:
        raise ValueError(
            f"Files in {data_dir} have different extensions: {extensions}.")
    # read files based on extension and return list of data
    if files[0].suffix == '.json':
        return [json_load(file.open()) for file in files]
    elif files[0].suffix == '.csv':
        return pd_concat((pd_read_csv(file) for file in files), ignore_index=True)
    elif files[0].suffix == '.txt':
        return [file.read_text() for file in files]
    elif files[0].suffix == '.parquet':
        return pd_concat((pd_read_parquet(file) for file in files), ignore_index=True)
    else:
        raise ValueError(
            f"Unsupported file extension: {files[0].suffix}. Supported extensions are: .json, .csv, .txt, .parquet.")


def write_data(data: Union[Dict[str, Any], Iterable[Any], DataFrame], write_path: Path) -> None:
    # check if write path is a directory
    if not write_path.is_dir():
        raise ValueError(f"The path {write_path} is a file, not a directory.")
    # if path exists and is not empty, delete files
    if write_path.exists() and any(write_path.iterdir()):
        for file in write_path.iterdir():
            file.unlink()
    # write data based on type
    if isinstance(data, dict):
        with write_path.joinpath("result.json").open("w+", encoding="utf-8") as f:
            f.write(dumps(data))
    elif isinstance(data, DataFrame):
        data.to_csv(write_path.joinpath("result.csv"), index=False)
    elif isinstance(data, str):
        with write_path.joinpath("result.txt").open('w') as f:
            f.write(data)
    elif isinstance(data, Iterable):
        with write_path.joinpath("result.txt").open('w') as f:
            for item in data:
                f.write(str(item) + '\n')
    else:
        raise TypeError(f"Unsupported data type: {type(data)}.")


def get_data_path(analysis_name: AnalysisName, dataset_name: Iterable[DatasetName], data_dir: Path = Path("/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis")) -> Path:
    """
    Get the path to the result file for a given analysis and dataset.
    """
    dataset_name = "-".join(dataset_name)
    return data_dir.joinpath(f"{dataset_name}-{analysis_name}-all")


def get_result_path(analysis_name: AnalysisName, dataset_name: Iterable[DatasetName], result_dir: Path = Path("/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/results")) -> Path:
    """
    Get the path to the result file for a given analysis and dataset.
    """
    dataset_name = "-".join(dataset_name)
    return result_dir.joinpath(f"{dataset_name}-{analysis_name}")


############################################    Analysis Specific Processing    ################################################
def get_query_log_intersec_ratio(
        dataset_name: Iterable[DatasetName], analysis_name: AnalysisName, write_results: bool = False) -> None:
    """
    Get the intersection ratio of two sets of queries.
    Supports only two datasets, e.g. aql and another query log,
    or the intersection of the aql and all combined comparison query logs.
    The result is printed to the console.\n
    If write_results is True, the result is written to a file.
    """
    if len(dataset_name) < 2:
        raise ValueError(
            "At least two datasets are required to calculate intersection ratio.")
    # get the lengths of individual datasets
    data = [read_data(get_data_path(analysis_name, [dataset]))[0]
            for dataset in dataset_name]
    # get length of combined dataset
    data.append(read_data(get_data_path(analysis_name, dataset_name))[0])
    # special case for aql and combined comparison query logs
    if dataset_name == ['aql', 'aol', 'ms-marco', 'orcas']:
        data.append(read_data(get_data_path(
            analysis_name, ['aol', 'ms-marco', 'orcas']))[0])
    # Check if data is a list of dicts
    if not all(isinstance(d, dict) for d in data):
        raise TypeError("Expected a list of dicts. Received: " +
                        str([type(d) for d in data if not isinstance(d, dict)]))
    # Calculate intersection ratio from deduplicated query logs
    result_dict = {}
    for result in data:
        result_dict.update({
            result['dataset']: result['hyperloglog-agg']
        })
    # get size of deduplicated aql
    aql_size = result_dict['aql']
    del result_dict['aql']
    if len(dataset_name) == 2:
        # size of comparison query log is the first value in the dict
        other_size = next(iter(result_dict.values()))
        # size of combined query log is the last value in the dict
        combined_size = next(reversed(result_dict.values()))
        intersec_ratio = (other_size+aql_size-combined_size)/other_size
        result = {
            'analysis_name': analysis_name,
            'dataset_name': dataset_name,
            'intersec_ratio': intersec_ratio
        }
        print(
            f"Intersection ratio of {dataset_name} for {analysis_name}: {intersec_ratio:.2%}\n-> {intersec_ratio:.2%} of {dataset_name[1]} queries are also in {dataset_name[0]}.")
    else:
        # we calculate the intersection ratio of the aql and all combined comparison query logs
        all_size = result_dict['aql-aol-ms-marco-orcas']
        comb_size = result_dict['aol-ms-marco-orcas']
        # we estimate conservatively
        intersec_ratio_aql = (aql_size + comb_size - all_size) / (aql_size)
        intersec_ratio_comb = (aql_size + comb_size - all_size) / (comb_size)
        result = {
            'analysis_name': analysis_name,
            'dataset_name': dataset_name,
            'intersec_ratio_aql': intersec_ratio_aql,
            'intersec_ratio_comb': intersec_ratio_comb
        }
        print(f"Intersection ratio of {dataset_name} for {analysis_name}:")
        print(
            f"{intersec_ratio_aql:.2%} of aql queries are also in the combined comparison query logs.")
        print(
            f"{intersec_ratio_comb:.2%} of the combined comparison query logs are also in aql.")

    if write_results:
        result_path = get_result_path(
            analysis_name, dataset_name)
        if not result_path.exists():
            print("create new result directory...")
            result_path.mkdir(parents=True, exist_ok=True)
            if result_path.exists():
                print("result path successfully created.")
        write_data(result, result_path)
        print("result successfully written.")

############################################    Process Results    ################################################


def process_results(analysis_name: AnalysisName, dataset_name: Iterable[DatasetName], write_results: bool = False) -> None:
    """
    Process results for a given analysis and dataset.
    """
    if analysis_name == "count-deduplicated-queries":
        get_query_log_intersec_ratio(
            dataset_name, analysis_name, write_results)
    elif analysis_name == "count-deduplicated-lowercase-queries":
        get_query_log_intersec_ratio(
            dataset_name, analysis_name, write_results)
    else:
        raise ValueError(f"Unknown analysis name: {analysis_name}")
