from thesis_schneg.model import DatasetName, AnalysisName
from thesis_schneg.vis_modules import _get_results_paths, load_results
import time
from thesis_schneg.vis_modules import log_plot
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib as mpl
import scienceplots

##### set visualization configuration #####

single_plot = True
# single_plot = False

# save visualization or not
# save_vis: bool = False
save_vis: bool = True

# select dataset (if none is selected all datasets are visualized in a joint plot)
dataset: DatasetName = None
# dataset: DatasetName = 'aol'
# dataset: DatasetName = 'aql'
# dataset: DatasetName = 'ms-marco'
# dataset: DatasetName = 'orcas'

analyses = []
# set analysis that should be visualized
# analyses.append('query-frequencies')
analyses.append('extract-named-entities')
analyses.append('extract-words')
# analyses.append('extract-chars')

# test_data = True
test_data = False

normalize_data = True
# normalize_data = False

color = None
label = None
# load data
analysis_data = []
for analysis_name in analyses:
    if dataset is None:
        result_data = {}
        # crawl files from all datasets and load into dictionary
        if analysis_name == 'query-frequencies':
            paths = {f"{name}": _get_results_paths(name, analysis_name) for name in [
                "aol", "aql"]}
        else:
            paths = {f"{name}": _get_results_paths(name, analysis_name) for name in [
                "aol", "aql", "ms-marco", "orcas"]}
        # iterate over datasets and create visualization
        for name, result_paths in paths.items():
            start_time = time.time()
            print(f"Loading data for {analysis_name} from {name} dataset")
            vis_data = load_results(
                result_paths, test_data=test_data, cols=["count()"])
            result_data.update({name: vis_data})
            end_time = time.time()
            print(f"{name} loaded in {(end_time - start_time)/60} min")
        label = ["AOL", "AQL", "MS-MARCO", "ORCAS"]
        analysis_data.append(result_data)
    else:
        # load data from single dataset
        result_paths = _get_results_paths(dataset, analysis_name)
        start_time = time.time()
        print(f"Loading data from {dataset} dataset")
        result_data = {dataset: load_results(result_paths, cols=["count()"])}
        end_time = time.time()
        print(f"{dataset} loaded in {(end_time - start_time)/60} min")
        analysis_data.append(result_data)


analyses_params = []
for analysis_name in analyses:
    # load visualization parametes into dictionary
    if analysis_name == 'query-frequencies':
        vis_params = {"dataset-col-x": "serp_query_text_url", "dataset-col-y": "count()", "x-label": "Rank",
                      "y-label": "Frequency", "x-lim": None, "y-lim": None, "title": "Zipf's Law Queries"}
    elif analysis_name == 'extract-named-entities':
        vis_params = {"dataset-col-x": "entity", "dataset-col-y": "count()", "x-label": "Rank",
                      "y-label": "Frequency", "x-lim": None, "y-lim": None, "title": "Zipf's Law Named Entities"}
    elif analysis_name == 'extract-words':
        vis_params = {"dataset-col-x": "word", "dataset-col-y": "count()", "x-label": "Rank",
                      "y-label": "Frequency", "x-lim": None, "y-lim": None, "title": "Zipf's Law Words"}
    elif analysis_name == 'extract-chars':
        vis_params = {"dataset-col-x": "char", "dataset-col-y": "count()", "x-label": "Rank",
                      "y-label": "Frequency", "x-lim": None, "y-lim": None, "title": "Zipf's Law Characters"}
    analyses_params.append(vis_params)


# latex rendering for matplotlib
mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)

# set path to save visualization
if len(analyses) > 1:
    if single_plot:
        vis_dir = Path(
            f"/home/benjamin/studium/masterarbeit/thesis-schneg/plots/{analyses[0]}-and-{analyses[1]}-single")
    else:
        vis_dir = Path(
            f"/home/benjamin/studium/masterarbeit/thesis-schneg/plots/{analyses[0]}-and-{analyses[1]}")
else:
    if single_plot:
        vis_dir = Path(
            f"/home/benjamin/studium/masterarbeit/thesis-schneg/plots/{analyses[0]}-single")
    else:
        vis_dir = Path(
            f"/home/benjamin/studium/masterarbeit/thesis-schneg/plots/{analyses[0]}")

# vis_dir = Path(
#         f"/home/benjamin/studium/masterarbeit/thesis-schneg/plots/extract-named-entities")

# make directory if it does not exist
if not vis_dir.exists() and save_vis:
    vis_dir.mkdir(parents=True)

# enable pgf format for matplotlib
if save_vis:
    mpl.use("pgf")
# use science style for plots from scienceplots library
plt.style.use(["science", "grid"])

# create subplots for each dataset
# set size of plot
textwidth = 5.5129
aspect_ratio = 6/8
scale = 1.0
width = textwidth * scale
height = width * aspect_ratio
if single_plot:
    fig, ax = plt.subplots(ncols=len(analyses), nrows=1,
                           figsize=(width*1.03, 4*height/5))
else:
    fig, ax = plt.subplots(ncols=len(analyses), nrows=4,
                           figsize=(width, 2*height))

for i in range(len(analyses)):
    result_data = analysis_data[i]
    vis_params = analyses_params[i]
    print(f"Visualizing {analyses[i]}")
    print(vis_params)
    # create visualization for all data sets if dataset is not specified
    if dataset is None:

        multi = True
        # color palette for datasets
        color = ['tab:blue', 'tab:orange', 'tab:gray', 'tab:red']
        linestyles = ["solid", "dashdot", "dashed", "dotted"]
        # counter for datasets
        cnt_datasets = 0
        # iterate over datasets and create visualization
        for name, vis_data in result_data.items():
            if normalize_data:
                vis_data[vis_params["dataset-col-y"]] = vis_data[vis_params["dataset-col-y"]] / \
                    vis_data[vis_params["dataset-col-y"]].sum()
            # apply specific visualization function
            if not single_plot:
                if len(analyses) > 1:
                    fig, ax[cnt_datasets, i] = log_plot(data=vis_data, subplots=(fig, ax[cnt_datasets, i]),
                                                        vis_params=vis_params, label=label[cnt_datasets], linestyle=linestyles[cnt_datasets], multi=multi, color=color[cnt_datasets])
                else:
                    fig, ax[cnt_datasets] = log_plot(data=vis_data, subplots=(fig, ax[cnt_datasets]),
                                                     vis_params=vis_params, label=name.upper(), multi=multi, linestyle=linestyles[cnt_datasets], color=color[cnt_datasets])  # , color=color[cnt_datasets]
            else:
                if len(analyses) > 1:
                    fig, ax[i] = log_plot(data=vis_data, subplots=(fig, ax[i]),
                                          vis_params=vis_params, label=label[cnt_datasets], multi=multi, linestyle=linestyles[cnt_datasets], color=color[cnt_datasets])
                else:
                    fig, ax = log_plot(data=vis_data, subplots=(fig, ax),
                                       vis_params=vis_params, label=name.upper(), multi=multi, linestyle=linestyles[cnt_datasets], color=color[cnt_datasets])  # , color=color[cnt_datasets]
            cnt_datasets += 1

    # create visualization for a specific data set:
    else:
        multi = False

        # modify title
        vis_params["title"] = f'{vis_params["title"]} ({dataset.upper()})'

        # set size of plot
        textwidth = 5.5129
        aspect_ratio = 6/8
        scale = 1.0
        width = textwidth * scale
        height = width * aspect_ratio

        # create subplot for dataset
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(width, height))

        # apply specific visualization function
        fig, ax = log_plot(data=result_data[dataset], subplots=(fig, ax),
                           vis_params=vis_params, multi=multi)

        # make layout tight
        plt.tight_layout()

        # either save visualization or show it
        if save_vis:
            fig.savefig(vis_dir.joinpath(f"{dataset}.pgf"))
        else:
            plt.show()

if dataset is None:
    fig.supxlabel(analyses_params[0]["x-label"], y=0.05)
    fig.supylabel(analyses_params[0]["y-label"])
# make layout tight
plt.tight_layout()

# either save visualization or show it
if save_vis:
    fig.savefig(vis_dir.joinpath("all.pgf"))
else:
    plt.show()

print("Visualization finished")
