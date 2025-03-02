from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from typing import Iterable, Sequence
from numpy.typing import ArrayLike
from thesis_schneg.model import DatasetName, AnalysisName
from pandas import DataFrame, concat, read_json, read_parquet as pd_read_parquet
from pyarrow.parquet import read_table as pa_read_table
# from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy import sort as np_sort, linspace, log, divide, multiply, sum as np_sum, max as np_max
from numpy.typing import ArrayLike
from scipy.stats import chi2


def _get_results_paths(
    dataset_name: DatasetName,
    analysis_name: AnalysisName,
    cleaned_aql: bool = False,
) -> Iterable[Path]:

    base_path = Path(
        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis"
    )
    if cleaned_aql and dataset_name == 'aql':
        # filter paths by dataset_name and analysis_name
        result_path = [path for path in base_path.glob(
            f'{dataset_name}-{analysis_name}-special')]
        print(result_path[0])
    else:
        # filter paths by dataset_name and analysis_name
        result_path = [path for path in base_path.glob(
            f'{dataset_name}-{analysis_name}-all')]

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
    filter_rows: Optional[List[Tuple]] = None,
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
            result = concat(objs=[pa_read_table(file, columns=cols, filters=filter_rows).to_pandas()
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


def bar_plot(data: DataFrame, subplots: Tuple[Figure, Axes], vis_params: Dict[str, Any], label: str = None, color: str = None, multi: bool = False, bar_width: float = 0.5) -> Tuple[Figure, Axes]:
    fig, ax = subplots
    height = data[vis_params["dataset-col-y"]].to_numpy()
    if type(data[vis_params["dataset-col-x"]].iloc[0]) is str:
        try:
            start = int(data[vis_params["dataset-col-x"]].iloc[0])
            x = linspace(start=start, stop=start+len(height), num=len(
                height), dtype=int)
        except ValueError:
            x = data[vis_params["dataset-col-x"]].to_numpy()
    else:
        x = data[vis_params["dataset-col-x"]].to_numpy()
    if label is not None:
        ax.bar(x=x, height=height, alpha=0.5,
               label=label, color=color, width=bar_width)
        ax.legend(fancybox=False,
                  edgecolor="black").get_frame().set_linewidth(0.5)
    else:
        ax.bar(x=x, height=height,
               alpha=0.5, width=bar_width)
    if not multi:
        if vis_params["x-label"] is not None:
            ax.set_xlabel(vis_params["x-label"])
        if vis_params["y-label"] is not None:
            ax.set_ylabel(vis_params["y-label"])
        if vis_params["title"] is not None:
            ax.set_title(vis_params["title"])

    if vis_params["x-lim"] is not None:
        ax.set_xlim(left=vis_params["x-lim"][0], right=vis_params["x-lim"][1])
    if vis_params["y-lim"] is not None:
        ax.set_ylim(left=vis_params["y-lim"][0], right=vis_params["y-lim"][1])

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


def log_plot(data: DataFrame, subplots: Tuple[Figure, Axes], vis_params: Dict[str, Any], label: str = None, color: str = None, linestyle: str = None, multi: bool = False) -> Tuple[Figure, Axes]:
    fig, ax = subplots
    height = data[vis_params["dataset-col-y"]].to_numpy()
    height = np_sort(a=height, kind='mergesort')[::-1]

    x = linspace(start=1, stop=len(height)+1, num=len(
        height), dtype=int)

    assert len(x) == len(height), "Length of x and height must be equal"
    # total_rows = data[vis_params["dataset-col-y"]].sum()
    # height = height/total_rows
    if label is not None:
        ax.plot(x, height, label=label, color=color,
                linestyle=linestyle, alpha=0.6)
        ax.legend(fancybox=False,
                  edgecolor="black", labelspacing=0.1, handletextpad=0.1, framealpha=0.5).get_frame().set_linewidth(0.5)
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
    if label == "AOL":
        ax.set_title(vis_params["title"])
    ax.set_yscale('log')
    ax.set_xscale('log')
    return fig, ax


def get_xlim(data: DataFrame, vis_params: Dict[str, Any], threshold: float, bar_width: float) -> Tuple[float, float]:
    x_max = data[vis_params["dataset-col-x"]][data[vis_params["dataset-col-y"]]
                                              > threshold*data[vis_params["dataset-col-y"]].max()].max() + bar_width/2
    x_min = 0 - bar_width/2
    return x_min, x_max


def normalize(data: ArrayLike) -> ArrayLike:
    return data / data.sum()


def g_test(test_distr: ArrayLike, exp_distr: ArrayLike, normal: bool = False) -> float:
    """Calculate the G-test for goodness of fit.
    """
    assert len(test_distr) == len(
        exp_distr), f"Length of test and expected distribution must be equal. Length of test distribution = {len(test_distr)}, length of expected distribution = {len(exp_distr)}\nTest Distribution: {test_distr}\nExpected Distribution: {exp_distr}"
    if normal:
        test_distr = normalize(test_distr)
        exp_distr = normalize(exp_distr)
    return 2*np_sum(multiply(test_distr, log(divide(test_distr, exp_distr))))


def ks_test(test_distr1: ArrayLike, test_distr2: ArrayLike, significance_lvl: float) -> Tuple[float, float, bool]:
    """Calculate the Kolmogorov-Smirnov test for goodness of fit.
        Distributions must be numpy-array.
    """
    assert len(test_distr1) == len(
        test_distr2), f"Length of test and expected distribution must be equal. Length of test distribution = {len(test_distr1)}, length of expected distribution = {len(test_distr2)}\nTest Distribution: {test_distr1}\nExpected Distribution: {test_distr2}"
    # compute the constant for the significance level
    assert significance_lvl in [0.2, 0.15, 0.1, 0.05, 0.01,
                                0.001], f"Significance level is {significance_lvl} but must be in [0.2,0.15,0.1,0.05,0.01,0.001]"
    ks_constant = {0.2: 1.07, 0.15: 1.14, 0.1: 1.22,
                   0.05: 1.36, 0.01: 1.63, 0.001: 1.95}
    # get total number of counts of each distribution
    len1 = np_sum(test_distr1)
    len2 = np_sum(test_distr2)

    # normalize the distributions
    test_distr1 = normalize(test_distr1)
    test_distr2 = normalize(test_distr2)

    # get cumulative distribution
    test_distr1 = test_distr1.cumsum()
    test_distr2 = test_distr2.cumsum()

    # compute ks statistic
    test_statistic = max(abs(test_distr1 - test_distr2))

    # compute hypothesis threshold
    threshold = ks_constant[significance_lvl] * (
        (len1 + len2) / (len1 * len2))**0.5
    return test_statistic, threshold, test_statistic <= threshold


def chi2_fit(test_distr: ArrayLike, exp_distr: ArrayLike, significance_lvl: float) -> Tuple[float, float, bool]:
    """Calculate the chi2 test for goodness of fit.
        Distributions must be numpy-array.
    """
    assert len(test_distr) == len(
        exp_distr), f"Length of test and expected distribution must be equal. Length of test distribution = {len(test_distr)}, length of expected distribution = {len(exp_distr)}\nTest Distribution: {test_distr}\nExpected Distribution: {exp_distr}"

    # compute sample size of test distribution
    N = np_sum(test_distr)

    # compute the cumulative distribution of the expected distribution
    rel_exp_distr = exp_distr/exp_distr.sum()

    test_statistic = np_sum(
        (test_distr - N * rel_exp_distr)**2 / (N * rel_exp_distr))
    treshold = chi2.isf(significance_lvl, len(test_distr)-1)
    return test_statistic, treshold, test_statistic <= treshold


def get_max_x(data_dict: Dict[str, DataFrame], x_name: str, threshold: float = 0.999) -> float | int:
    max_x = []
    print('get x_max...')
    for dataset_name, data in data_dict.items():
        print(dataset_name)
        data = data.sort_values(x_name, ascending=True)
        data['cum_dist'] = (data['count()'] / data['count()'].sum()).cumsum()
        data = data.query(f'`cum_dist` < {threshold}')
        max_x.append(data[x_name].max())
    return max(max_x)
