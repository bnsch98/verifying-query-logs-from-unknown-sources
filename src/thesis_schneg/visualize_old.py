import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


class visualize_results:
    def __init__(self, analysis: str):
        self.analysis = analysis
        assert self.analysis in ['word_count', 'string_length',
                                 'zipfs_law'], "Specified analysis is not supported!"

    def visualize(self, save_plots: bool = False, show_plots: bool = True):
        if self.analysis == 'word_count':

            input_path = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/results_query_length/'
            datasets = ['aol', 'ms', 'aql', 'orcas']
            # Define a list of colors
            colors = ['blue', 'green', 'red', 'purple']
            var = 'word_count'
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
                axes[cnt_datasets].set_xlim(0, 15)
                axes[cnt_datasets].set_ylim(0, 0.6)
                # Add and customize grid
                axes[cnt_datasets].minorticks_on()
                axes[cnt_datasets].grid(True, which='major', linestyle='-',
                                        linewidth='0.5', color='black')
                axes[cnt_datasets].grid(True, which='minor', linestyle=':',
                                        linewidth='0.5', color='gray')
                cnt_datasets += 1
            fig.suptitle(
                f'Comparison of {var} across Multiple Datasets', fontsize=16)
            plt.tight_layout()
            if show_plots:
                plt.show()
            if save_plots:
                with PdfPages(f'/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/plots/histogram_scaled_{var}.pdf') as pdf:
                    pdf.savefig(fig)

        elif self.analysis == 'string_length':

            input_path = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/results_query_length/'
            datasets = ['aol', 'ms', 'aql', 'orcas']
            # Define a list of colors
            colors = ['blue', 'green', 'red', 'purple']
            var = 'string_length'
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
                axes[cnt_datasets].set_xlim(0, 60)
                axes[cnt_datasets].set_ylim(0, 0.16)
                # Add and customize grid
                axes[cnt_datasets].minorticks_on()
                axes[cnt_datasets].grid(True, which='major', linestyle='-',
                                        linewidth='0.5', color='black')
                axes[cnt_datasets].grid(True, which='minor', linestyle=':',
                                        linewidth='0.5', color='gray')
                cnt_datasets += 1
            fig.suptitle(
                f'Comparison of {var} across Multiple Datasets', fontsize=16)
            plt.tight_layout()
            if show_plots:
                plt.show()
            if save_plots:
                with PdfPages(f'/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/plots/histogram_scaled_{var}.pdf') as pdf:
                    pdf.savefig(fig)

        elif self.analysis == 'zipfs_law':
            input_path = '/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/results_zipfs_law/'
            datasets = ['aol', 'ms-marco', 'aql', 'orcas']
            # Define a list of colors
            # colors = ['blue', 'green', 'red', 'purple']

            plt.figure(figsize=(10, 6))
            for dataset in datasets:
                file = input_path + dataset + '.csv'
                data = pd.read_csv(file)
                plt.plot(range(1, len(data) + 1),
                         data['count'], label=dataset)  # , color=colors[datasets.index(dataset)]
            zipf_values = [1/(i*np.log(1.78*len(data)))
                           for i in range(1, len(data) + 1)]
            zipf_values = data['count'][0]/zipf_values[0]*np.array(zipf_values)
            plt.plot(range(1, len(data) + 1),
                     zipf_values, label='zipf distribution')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Rank of the word')
            plt.ylabel('Frequency of the word')
            plt.title('Zipfs Law: Frequency vs. Rank')
            plt.legend(title='Dataset')
            plt.grid(True)
            if show_plots:
                plt.show()
            if save_plots:
                with PdfPages('/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/plots/zipfs_law.pdf') as pdf:
                    pdf.savefig(plt.gcf())


if __name__ == '__main__':

    analysis = 'zipfs_law'
    vis = visualize_results(analysis)
    vis.visualize(save_plots=False, show_plots=True)
