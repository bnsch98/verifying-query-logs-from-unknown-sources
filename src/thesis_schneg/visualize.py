import matplotlib.pyplot as plt
import pandas as pd

input_path = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/results_query_length/'


variable = ['string_length', 'word_count']
datasets = ['aol', 'ms', 'aql', 'orcas']
colors = ['blue', 'green', 'red', 'purple']  # Define a list of colors
ylims = [0.16, 0.6]
xlims = [100, 20]
cnt = 0
for var in variable:
    fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(16, 4))
    cnt_datasets = 0
    for dataset in datasets:
        file = input_path + dataset + '_' + var + '.csv'
        data = pd.read_csv(file)
        total_rows = data['count()'].sum()
        # print(data.head())
        axes[cnt_datasets].bar(data[var], data['count()']/total_rows,
                               alpha=0.5, label=dataset, color=colors[cnt_datasets])

        axes[cnt_datasets].set_xlabel(var)
        axes[cnt_datasets].set_ylabel('Frequency')
        # axes[cnt_datasets].title(
        #     f'Histogram of scaled {var} for Multiple Datasets')
        axes[cnt_datasets].legend()

        # Set the x-axis limit to the maximum word count
        axes[cnt_datasets].set_xlim(0, xlims[cnt])
        axes[cnt_datasets].set_ylim(0, ylims[cnt])
        # Add and customize grid
        axes[cnt_datasets].minorticks_on()
        axes[cnt_datasets].grid(True, which='major', linestyle='-',
                                linewidth='0.5', color='black')
        axes[cnt_datasets].grid(True, which='minor', linestyle=':',
                                linewidth='0.5', color='gray')
        cnt_datasets += 1
    fig.suptitle(f'Comparison of {var} across Multiple Datasets', fontsize=16)
    plt.tight_layout()
    plt.savefig(
        f'/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/plots/histogram_scaled_{var}.png')
    plt.show()
    cnt += 1
