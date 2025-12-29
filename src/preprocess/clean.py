from pathlib import Path
from typing import Dict, Iterable, Union, Any
from pandas import DataFrame, read_csv as pd_read_csv, read_parquet as pd_read_parquet, concat as pd_concat
from json import load as json_load, dumps
from thesis_schneg.model import DatasetName, AnalysisName


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


if __name__ == "__main__":
    datasets = ["aql-aol", "aql-ms-marco", "aql-orcas",
                "aql-aol-ms-marco-orcas", "aol-ms-marco-orcas"]
    analysis_names = ["count-deduplicated-queries",
                      "count-deduplicated-lowercase-queries"]
    # datasets = ["aql-aol"]
    # analysis_names = ["count-deduplicated-queriess"]
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        for analysis in analysis_names:
            print(f"Running analysis: {analysis}")
            # Example of reading data
            try:
                data_path = get_data_path(analysis, [dataset])
                print(f"Data path: {data_path}")
                data = read_data(data_path)[0]
                print(f"Data read successfully for {dataset}.")
                data["dataset"] = dataset
                # write data to the same path and delete the old file if it exists
                write_data(data, data_path)
                print(f"Data written successfully to {data_path}.")
            except Exception as e:
                print(f"Error reading data for {dataset}: {e}")
