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

# latex rendering for matplotlib
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)


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


def _get_vis_func(analysis_name: AnalysisName) -> Optional[Callable[[Any], Tuple[Figure, Axes]]]:
    if analysis_name == "sum-rows":
        return None
    elif analysis_name == "zipfs-law-queries":
        return None
    elif analysis_name == "zipfs-law-words":
        return log_plot
    elif analysis_name == "zipfs-law-chars":
        return log_plot
    elif analysis_name == "unique-queries":
        return None
    elif analysis_name == "heaps-law-words":
        return None
    elif analysis_name == "query-length-chars":
        return bar_plot
    elif analysis_name == "query-length-words":
        return bar_plot
    elif analysis_name == "named-entities":
        return None
    elif analysis_name == "search-operators":
        return None


def _get_vis_parameters(analysis_name: AnalysisName) -> Dict[str, Any]:
    if analysis_name == "sum-rows":
        return {"dataset-col-x": None, "dataset-col-y": "count()", "x-label": None, "y-label": None, "x-lim": None, "y-lim": None, "title": None}
    elif analysis_name == "zipfs-law-queries":
        return {"dataset-col-x": None, "dataset-col-y": "count()", "x-label": None, "y-label": None, "x-lim": None, "y-lim": None, "title": None}
    elif analysis_name == "zipfs-law-words":
        return {"dataset-col-x": "word", "dataset-col-y": "count()", "x-label": "Rank", "y-label": "Frequency", "x-lim": None, "y-lim": None, "title": "Zipf's law for words"}
    elif analysis_name == "zipfs-law-chars":
        return {"dataset-col-x": "char", "dataset-col-y": "count()", "x-label": "Rank", "y-label": "Frequency", "x-lim": None, "y-lim": None, "title": "Zipf's Law for characters"}
    elif analysis_name == "unique-queries":
        return {"dataset-col-x": None, "dataset-col-y": "count()", "x-label": None, "y-label": None, "x-lim": None, "y-lim": None, "title": None}
    elif analysis_name == "heaps-law-words":
        return {"dataset-col-x": None, "dataset-col-y": "count()", "x-label": None, "y-label": None, "x-lim": None, "y-lim": None, "title": None}
    elif analysis_name == "query-length-chars":
        return {"dataset-col-x": "query-length-chars", "dataset-col-y": "count()", "x-label": "Number of characters", "y-label": "Relative frequency", "x-lim": (0, 50), "y-lim": None, "title": "Query length in characters"}
    elif analysis_name == "query-length-words":
        return {"dataset-col-x": "query-length-words", "dataset-col-y": "count()", "x-label": "Number of words", "y-label": "Relative frequency", "x-lim": (0, 20), "y-lim": None, "title": "Query length in words"}
    elif analysis_name == "named-entities":
        return {"dataset-col-x": None, "dataset-col-y": "count()", "x-label": None, "y-label": None, "x-lim": None, "y-lim": None, "title": None}
    elif analysis_name == "search-operators":
        return {"dataset-col-x": None, "dataset-col-y": "count()", "x-label": None, "y-label": None, "x-lim": None, "y-lim": None, "title": None}


def bar_plot(data: DataFrame, subplots: Tuple[Figure, Axes], vis_params: Dict[str, Any], label: str = None, color: str = None) -> Tuple[Figure, Axes]:
    fig, ax = subplots
    x = data[vis_params["dataset-col-x"]].to_numpy()
    height = data[vis_params["dataset-col-y"]].to_numpy()
    total_rows = data[vis_params["dataset-col-y"]].sum()
    height = height/total_rows
    if label is not None:
        ax.bar(x=x, height=height, alpha=0.5, label=label, color=color)
        ax.legend()
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
    # ax.grid(True, which='major', linestyle='-',
    #         linewidth='0.5', color='black')
    # ax.grid(True, which='minor', linestyle=':',
    #         linewidth='0.5', color='gray')

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
        ax.legend()
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


def visualize(analysis_name: AnalysisName,
              dataset_name: DatasetName = None,
              save_vis: bool = False,
              ) -> None:
    # enable pgf format for matplotlib
    if save_vis:
        matplotlib.use("pgf")
    # use science style for plots from scienceplots library
    plt.style.use(["science", "grid"])
    vis_dir = Path(
        f"/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/plots/{analysis_name}")
    # make directory if it does not exist
    if not vis_dir.exists() and save_vis:
        vis_dir.mkdir(parents=True)

    # create visualization for all data sets if dataset_name is not specified
    if dataset_name is None:
        # crawl files from all datasets and load into dictionary
        files = {f"{dataset}": _get_results_paths(dataset, analysis_name) for dataset in [
            "aol", "ms-marco", "orcas", "aql"]}
        # create subplots for each dataset
        fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(16, 4))
        cnt_datasets = 0
        # color palette for datasets
        color = ['yellow', 'orange', 'red', 'purple']

        # iterate over datasets and create visualization
        for dataset, result_files in files.items():
            vis_data = load_results(result_files)
            vis_params = _get_vis_parameters(analysis_name)
            vis_func = _get_vis_func(analysis_name)
            # apply specific visualization function
            fig, ax[cnt_datasets] = vis_func(data=vis_data, subplots=(fig, ax[cnt_datasets]),
                                             vis_params=vis_params, label=dataset, color=color[cnt_datasets])
            cnt_datasets += 1
        fig.suptitle(
            f'{vis_params["title"]} across multiple Datasets', fontsize=16)

        plt.tight_layout()

        if save_vis:
            fig.savefig(vis_dir.joinpath("all.pgf"))
        else:
            plt.show()

    else:  # create visualization for a specific data set

        # enable pgf format for matplotlib
        if save_vis:
            matplotlib.use("pgf")
        # use science style for plots from scienceplots library
        plt.style.use(["science", "grid"])
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        result_files = _get_results_paths(dataset_name, analysis_name)
        vis_data = load_results(result_files)
        vis_params = _get_vis_parameters(analysis_name)
        vis_func = _get_vis_func(analysis_name)
        fig, ax = vis_func(data=vis_data, subplots=(fig, ax),
                           vis_params=vis_params)
        plt.tight_layout()
        if save_vis:
            fig.savefig(vis_dir.joinpath(f"{dataset_name}.pgf"))
        else:
            plt.show()
        # print(vis_data)
        # print(vis_data.head())
        # print(vis_data.info())
        # print(vis_data.columns)
