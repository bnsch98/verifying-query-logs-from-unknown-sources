from pathlib import Path
from typing import Iterable
from thesis_schneg.model import DatasetName, AnalysisName
from pandas import DataFrame
import pandas as pd
############################################    Requirements for basic modules    ########################################


def _get_results_paths(
    dataset_name: DatasetName,
    analysis_name: AnalysisName,
) -> Iterable[Path]:

    base_path = Path(
        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis"
    )
    # filter paths by dataset_name and analysis_name
    result_path = [path for path in base_path.glob(
        f'{dataset_name}-{analysis_name}-*')]
    assert result_path, f"No directories found for dataset = {dataset_name} and analysis = {analysis_name}"

    # check if there are multiple directories
    if len(result_path) > 1:
        # get directory of the most significant analysis, e.g. the one with the most samples
        # 1. if directory with suffix "all" exists, take this one
        all_result_paths = [
            path for path in result_path if path.name.endswith("all")]
        # 2. if not, take the one with the highest number of samples
        if not all_result_paths:
            result_path.sort(key=lambda path: int(
                path.name.split("-")[-1]), reverse=True)
            all_result_paths = [result_path[0]]

        result_path = all_result_paths[0]
    else:
        result_path = result_path[0]

    # get all files in the directory
    result_files = [path for path in result_path.iterdir()
                    if path.is_file()]
    assert result_files, f"Selected directory \"{result_path.name}\" is empty"

    return result_files


def load_results(
    result_files: Iterable[Path]
) -> DataFrame:
    # check if there are multiple files
    if len(result_files) > 1:
        # check if result_files contains onyl parquet files
        assert all(
            file.suffix == ".parquet" for file in result_files), "Non-parquet files found"
        result = pd.concat([pd.read_parquet(file) for file in result_files])
    else:
        result = pd.read_json(result_files[0])

    return result


############################################    Pipeline    ##############################################
def visualize(dataset_name: DatasetName,
              analysis_name: AnalysisName,

              ) -> None:
    result_files = _get_results_paths(dataset_name, analysis_name)
    result = load_results(result_files)
    print(result.head())
