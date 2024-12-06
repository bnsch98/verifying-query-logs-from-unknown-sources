from pathlib import Path
from typing import Optional, Tuple, Callable, Dict, Any
from typing import Iterable
from thesis_schneg.model import DatasetName, AnalysisName
from pandas import DataFrame, concat, read_json, read_parquet
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.axes import Axes

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
        # by now only parquet files are expected as multiple files
        assert all(
            file.suffix == ".parquet" for file in result_files), "Non-parquet files found"
        result = concat([read_parquet(file) for file in result_files])
    else:
        # by now only json files are expected as a single file
        result = read_json(result_files[0])

    return result


def run_visualization(vis_func: Callable, vis_data: DataFrame, vis_params: Dict[str, Any], vis_dir: Path, save_vis: bool = False, show: bool = True) -> Tuple[Figure, Axes]:
    return vis_func(data=vis_data, vis_dir=vis_dir, save_vis=save_vis, vis_params=vis_params, show=show)


def _get_vis_func(analysis_name: AnalysisName) -> Callable:
    if analysis_name == "sum-rows":
        return None
    elif analysis_name == "zipfs-law-queries":
        return None
    elif analysis_name == "zipfs-law-words":
        return None
    elif analysis_name == "zipfs-law-chars":
        return None
    elif analysis_name == "unique-queries":
        return None
    elif analysis_name == "heaps-law-words":
        return None
    elif analysis_name == "query-length-chars":
        return query_length_vis
    elif analysis_name == "query-length-words":
        return None
    elif analysis_name == "named-entities":
        return None
    elif analysis_name == "search-operators":
        return None


def _get_vis_parameters(analysis_name: AnalysisName) -> Dict[str, Any]:
    if analysis_name == "sum-rows":
        return {"dataset-col": None, "x-label": None, "y-label": None, "x-lim": None, "y-lim": None, "title": None}
    elif analysis_name == "zipfs-law-queries":
        return {"dataset-col": None, "x-label": None, "y-label": None, "x-lim": None, "y-lim": None, "title": None}
    elif analysis_name == "zipfs-law-words":
        return {"dataset-col": None, "x-label": None, "y-label": None, "x-lim": None, "y-lim": None, "title": None}
    elif analysis_name == "zipfs-law-chars":
        return {"dataset-col": None, "x-label": None, "y-label": None, "x-lim": None, "y-lim": None, "title": None}
    elif analysis_name == "unique-queries":
        return {"dataset-col": None, "x-label": None, "y-label": None, "x-lim": None, "y-lim": None, "title": None}
    elif analysis_name == "heaps-law-words":
        return {"dataset-col": None, "x-label": None, "y-label": None, "x-lim": None, "y-lim": None, "title": None}
    elif analysis_name == "query-length-chars":
        return {"dataset-col": "query-length-chars", "x-label": "Number of characters", "y-label": "Relative frequency", "x-lim": (0, 50), "y-lim": None, "title": "Query length in characters"}
    elif analysis_name == "query-length-words":
        return {"dataset-col": None, "x-label": None, "y-label": None, "x-lim": None, "y-lim": None, "title": None}
    elif analysis_name == "named-entities":
        return {"dataset-col": None, "x-label": None, "y-label": None, "x-lim": None, "y-lim": None, "title": None}
    elif analysis_name == "search-operators":
        return {"dataset-col": None, "x-label": None, "y-label": None, "x-lim": None, "y-lim": None, "title": None}


def query_length_vis(data: DataFrame, vis_params: Dict[str, Any], vis_dir: Path, save_vis: bool = False, show: bool = True) -> Tuple[Figure, Axes]:

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    total_rows = data['count()'].sum()
    ax.bar(data[vis_params["dataset-col"]], data['count()']/total_rows,
           alpha=0.5)
    if vis_params["x-label"] is not None:
        ax.set_xlabel(vis_params["x-label"])
    if vis_params["y-label"] is not None:
        ax.set_ylabel(vis_params["y-label"])
    if vis_params["x-lim"] is not None:
        ax.set_xlim(left=vis_params["x-lim"][0], right=vis_params["x-lim"][1])
    if vis_params["y-lim"] is not None:
        ax.set_ylim(left=vis_params["y-lim"][0], right=vis_params["y-lim"][1])
    if vis_params["y-label"] is not None:
        ax.set_title(vis_params["title"])
    ax.grid(True, which='major', linestyle='-',
            linewidth='0.5', color='black')
    ax.grid(True, which='minor', linestyle=':',
            linewidth='0.5', color='gray')
    plt.tight_layout()
    if show:
        plt.show()
    if save_vis:
        with PdfPages(vis_dir) as pdf:
            pdf.savefig(fig)

    return fig, ax


def visualize(analysis_name: AnalysisName,
              dataset_name: DatasetName = None,
              save_vis: bool = False,
              ) -> None:
    # create visualization for all data sets
    if dataset_name is None:
        files = {f"{dataset}": _get_results_paths(dataset, analysis_name) for dataset in [
            "aol", "ms-marco", "orcas", "aql"]}
        glob_fig, glob_axes = plt.subplots(ncols=4, nrows=1, figsize=(16, 4))
        cnt_datasets = 0
        for dataset, result_files in files.items():
            vis_data = load_results(result_files)
            vis_params = _get_vis_parameters(analysis_name)
            vis_func = _get_vis_func(analysis_name)
            vis_dir = Path(
                f"/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/plots/{dataset}-{analysis_name}.pdf")
            fig, ax = run_visualization(vis_func=vis_func, vis_data=vis_data,
                                        vis_params=vis_params, vis_dir=vis_dir, save_vis=save_vis, show=False)
            glob_axes[cnt_datasets] = ax
            cnt_datasets += 1
        plt.show()

    else:  # create visualization for a specific data set
        vis_dir = Path(
            f"/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/plots/{dataset_name}-{analysis_name}.pdf")
        result_files = _get_results_paths(dataset_name, analysis_name)
        vis_data = load_results(result_files)
        vis_params = _get_vis_parameters(analysis_name)
        vis_func = _get_vis_func(analysis_name)
        fig, ax = run_visualization(vis_func=vis_func, vis_data=vis_data,
                                    vis_params=vis_params, vis_dir=vis_dir, save_vis=save_vis)

        print(vis_data)
        print(vis_data.head())
        print(vis_data.info())
        print(vis_data.columns)
