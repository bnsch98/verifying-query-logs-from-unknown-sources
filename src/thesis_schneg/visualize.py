from pathlib import Path
from typing import Iterable, Optional, Union
from thesis_schneg.model import DatasetName, AnalysisName
from pandas import DataFrame

############################################    Requirements for basic modules    ########################################


def _get_results_paths(
    dataset_name: DatasetName,
    analysis_name: AnalysisName,
    sample_files: Optional[int] = None,
) -> Iterable[Path]:
    base_path: Path = Path(
        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis"
    )
    if sample_files is not None:
        base_path = base_path.joinpath(
            f"{dataset_name}-{analysis_name}-{sample_files}")
    else:
        base_path = base_path.joinpath(f"{dataset_name}-{analysis_name}-all")

    result_paths = [path for path in base_path.iterdir()
                    if path.is_file()]
    return result_paths


def load_results(
    result_paths: Iterable[Path]
) -> DataFrame:

    # check what kind of result file, e.g. parquet, json, etc.

    # load file accordingly

    # return file
    pass
