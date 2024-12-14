from pathlib import Path
from typing import Tuple, Callable, Optional, Dict, Any
from typing import Iterable
from thesis_schneg.model import DatasetName, AnalysisName
from pandas import DataFrame, concat, read_json, read_parquet
from matplotlib import pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy import sort as np_sort
import scienceplots
import matplotlib


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
        # by now only parquet files are expected as multiple files
        assert all(
            file.suffix == ".parquet" for file in result_files), "Non-parquet files found"
        result = concat([read_parquet(file) for file in result_files])
    else:
        # by now only json files are expected as a single file
        result = read_json(result_files[0])

    return result


def bar_plot(data: DataFrame, subplots: Tuple[Figure, Axes], vis_params: Dict[str, Any], label: str = None, color: str = None) -> Tuple[Figure, Axes]:
    fig, ax = subplots
    height = data[vis_params["dataset-col-y"]].to_numpy()
    x = data[vis_params["dataset-col-x"]].to_numpy()
    if label is not None:
        ax.bar(x=x, height=height, alpha=0.5, label=label, color=color)
        ax.legend(fancybox=False,
                  edgecolor="black").get_frame().set_linewidth(0.5)
    else:
        ax.bar(x=x, height=height,
               alpha=0.5)
    if vis_params["x-label"] is not None:
        ax.set_xlabel(vis_params["x-label"])
    if vis_params["y-label"] is not None:
        ax.set_ylabel(vis_params["y-label"])
    if vis_params["x-lim"] is not None:
        ax.set_xlim(left=vis_params["x-lim"][0], right=vis_params["x-lim"][1])
    if vis_params["y-lim"] is not None:
        ax.set_ylim(left=vis_params["y-lim"][0], right=vis_params["y-lim"][1])
    if vis_params["title"] is not None:
        ax.set_title(vis_params["title"])

    return fig, ax


def log_plot(data: DataFrame, subplots: Tuple[Figure, Axes], vis_params: Dict[str, Any], label: str = None, color: str = None) -> Tuple[Figure, Axes]:
    fig, ax = subplots
    height = data[vis_params["dataset-col-y"]].to_numpy()

    if type(data[vis_params["dataset-col-x"]].iloc[0]) is str:
        x = list(range(1, len(data[vis_params["dataset-col-x"]])+1))
        height = np_sort(a=height, kind='mergesort')[::-1]
    else:
        x = data[vis_params["dataset-col-x"]].to_numpy()
    # total_rows = data[vis_params["dataset-col-y"]].sum()
    # height = height/total_rows
    if label is not None:
        ax.plot(x, height, label=label, color=color)
        ax.legend(fancybox=False,
                  edgecolor="black").get_frame().set_linewidth(0.5)
    else:
        ax.plot(x, height)
    if vis_params["x-label"] is not None:
        ax.set_xlabel(vis_params["x-label"])
    if vis_params["y-label"] is not None:
        ax.set_ylabel(vis_params["y-label"])
    if vis_params["x-lim"] is not None:
        ax.set_xlim(left=vis_params["x-lim"][0], right=vis_params["x-lim"][1])
    if vis_params["y-lim"] is not None:
        ax.set_ylim(left=vis_params["y-lim"][0], right=vis_params["y-lim"][1])
    if vis_params["title"] is not None:
        ax.set_title(vis_params["title"])
    ax.set_yscale('log')
    return fig, ax
