from pandas import DataFrame
from thesis_schneg.vis_modules import get_max_x
from thesis_schneg.vis_modules import _get_results_paths, load_results
from scipy.stats import wasserstein_distance
import numpy as np
from time import time


analyses = []
# set analysis that should be analyzed
# analyses.append('query-frequencies')
# analyses.append('extract-named-entities')
# analyses.append('extract-words')
analyses.append('extract-chars')

# test_data = True
test_data = False

cleaned_aql = True

col = ['count()']

# load data
analysis_data = {}
for analysis in analyses:
    print(f"Start loading \"{analysis}\"")
    datasets = {}
    for dataset in ["aol", "aql", "ms-marco", "orcas"]:
        start_time = time()
        paths = _get_results_paths(dataset, analysis, cleaned_aql)
        result_data = load_results(paths, test_data=test_data, cols=col)
        datasets.update({dataset: result_data})
        end_time = time()
        print(f"{dataset.upper()} loaded in {(end_time - start_time)/60} min")
    analysis_data.update({analysis: datasets})

# calculate wasserstein distance
cnt = 0
distances_data = {}
for analysis, datasets in analysis_data.items():
    print(analysis)
    x_max = get_max_x(datasets, "count()")
    print(x_max)
    # distances = ndarray((len(datasets), len(datasets)))
    distances = DataFrame(np.zeros((len(datasets), len(datasets))),
                          index=datasets.keys(), columns=datasets.keys())
    names = []
    j = 0
    print("start calculating distances")
    for dataset_name, data in datasets.items():
        data1 = {dataset_name: data.sort_values("count()", ascending=True)}
        print(f"{dataset_name.title()} sorted")
        names.append(dataset_name)
        x_vals = np.arange(1, x_max+1)
        y_vals = data1[dataset_name]['count()'][0:x_max].to_numpy()
        if len(y_vals) < x_max:
            y_vals = np.append(y_vals, np.zeros(x_max-len(y_vals)))
        i = 0
        for dataset_name, data in datasets.items():
            if dataset_name in names:
                dist = 0
            else:
                print(f"names: {names}")
                data2 = data.sort_values("count()", ascending=True)
                print(f"{dataset_name.title()} sorted")
                x_vals2 = np.arange(1, x_max+1)
                y_vals2 = data2['count()'][0:x_max].to_numpy()
                if len(y_vals2) < x_max:
                    y_vals2 = np.append(y_vals2, np.zeros(x_max-len(y_vals2)))
                print(
                    f"Calculating Wasserstein distance between {dataset_name} and {names}")
                dist = wasserstein_distance(
                    x_vals, x_vals2, u_weights=y_vals, v_weights=y_vals2)
                print(
                    "Wasserstein distance calculated")
            # distances[i][j] = dist
            distances.iloc[i, j] = dist
            i += 1
        j += 1
    distances = distances + distances.T
    distances_data.update({analysis: distances})
    cnt += 1
for key, value in distances_data.items():
    print(value)

# get avarage wasserstein distances per query log
avg_distances = {}
for analysis, distances in distances_data.items():
    avg_distances.update({analysis: distances.mean().mean()})

for key, value in avg_distances.items():
    print(key, value)
