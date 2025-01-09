from pathlib import Path
from typing import Tuple, Callable, Optional, Dict, Any, List
from typing import Iterable, Sequence
from numpy.typing import ArrayLike
from thesis_schneg.model import DatasetName, AnalysisName
from pandas import DataFrame, concat, read_json, read_parquet as pd_read_parquet
from pyarrow.parquet import read_table as pa_read_table
from ray.data import read_parquet as ray_read_parquet
from ray import init
from matplotlib import pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy import sort as np_sort, linspace
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
    result_files: Iterable[Path],
    cols: Optional[List[str]] = None,
    test_data: bool = False,
) -> DataFrame:
    # check if there are multiple files
    if len(result_files) > 1:
        # by now only parquet files are expected as multiple files
        assert all(
            file.suffix == ".parquet" for file in result_files), "Non-parquet files found"
        if test_data:
            # read only the first file
            result = pd_read_parquet(result_files[0])
        else:
            result = concat(objs=[pa_read_table(file, columns=cols).to_pandas()
                                  for file in result_files], axis=0)
    else:
        # by now only json files are expected as a single file
        result = read_json(result_files[0])

    return result


def set_plot_properties(subplots: Tuple[Figure, Axes], vis_params: Dict[str, Any]) -> Tuple[Figure, Axes]:
    fig, ax = subplots
    if vis_params["x-label"] is not None:
        ax.set_xlabel(vis_params["x-label"])
    if vis_params["y-label"] is not None:
        ax.set_ylabel(vis_params["y-label"])
    if vis_params["x-lim"] is not None:
        ax.set_xlim(left=vis_params["x-lim"][0], right=vis_params["x-lim"][1])
    if vis_params["y-lim"] is not None:
        ax.set_ylim(bottom=vis_params["y-lim"][0], top=vis_params["y-lim"][1])
    if vis_params["title"] is not None:
        ax.set_title(vis_params["title"])
    return fig, ax


def get_frequency_dict(counts: List[Any], lengths: List[Any]) -> Dict[str, Any]:
    freq_dict = {str(key): value for key, value in zip(lengths, counts)}
    return freq_dict


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


def hist_plot(x: ArrayLike,  bins: Optional[Sequence[int]], subplots: Tuple[Figure, Axes], vis_params: Dict[str, Any], label: str = None, color: str = None, **kwargs) -> Tuple[Figure, Axes]:
    fig, ax = subplots

    ax.hist(x=x, bins=bins, alpha=0.5, label=label, color=color, **kwargs)
    ax.legend(fancybox=False,
              edgecolor="black").get_frame().set_linewidth(0.5)

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


def log_plot(data: DataFrame, subplots: Tuple[Figure, Axes], vis_params: Dict[str, Any], label: str = None, color: str = None, multi: bool = False) -> Tuple[Figure, Axes]:
    fig, ax = subplots
    height = data[vis_params["dataset-col-y"]].to_numpy()

    if type(data[vis_params["dataset-col-x"]].iloc[0]) is str:
        x = linspace(start=1, stop=len(data[vis_params["dataset-col-x"]])+1, num=len(
            data[vis_params["dataset-col-x"]]), dtype=int)
        height = np_sort(a=height, kind='mergesort')[::-1]
    else:
        x = data[vis_params["dataset-col-x"]].to_numpy()
    assert len(x) == len(height), "Length of x and height must be equal"
    # total_rows = data[vis_params["dataset-col-y"]].sum()
    # height = height/total_rows
    if label is not None:
        ax.plot(x, height, label=label, color=color)
        ax.legend(fancybox=False,
                  edgecolor="black").get_frame().set_linewidth(0.5)
    else:
        ax.plot(x, height)
    if not multi:
        if vis_params["x-label"] is not None:
            ax.set_xlabel(vis_params["x-label"])
        if vis_params["y-label"] is not None:
            ax.set_ylabel(vis_params["y-label"])
    if vis_params["x-lim"] is not None:
        ax.set_xlim(left=vis_params["x-lim"][0], right=vis_params["x-lim"][1])
    if vis_params["y-lim"] is not None:
        ax.set_ylim(left=vis_params["y-lim"][0], right=vis_params["y-lim"][1])
    if vis_params["title"] is not None and not multi:
        ax.set_title(vis_params["title"])
    ax.set_yscale('log')
    ax.set_xscale('log')
    return fig, ax
