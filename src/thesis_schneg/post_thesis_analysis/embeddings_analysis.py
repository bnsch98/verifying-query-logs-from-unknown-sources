from typing import Any, Optional, Iterable, Tuple
from thesis_schneg.model import DatasetName, EmbeddingsAnalysisName, OTSolverVariant
from pathlib import Path
from dotenv import load_dotenv
from random import sample
from pandas import DataFrame, concat, read_parquet
from jax.random import PRNGKey
from numpy import stack
from time import time
import jax.numpy as jnp
import os
import inspect

################### CONFIGURATION #####################
# use .env to load data path for embeddings
load_dotenv()

# Mapping of dataset names to their corresponding directory names
DIRECTORY_MAP = {
    "aol": "aol-get-embeddings-special",
    "aql": "aql-get-embeddings-special",
    "ms-marco": "ms-marco-get-embeddings-all",
    "orcas": "orcas-get-embeddings-all",
}

OT_PARAMS = {
    "sliced-wasserstein": {
        "n_proj": 1000,  # number of projections for sliced wasserstein computation
        "rng": PRNGKey(42),  # random key for reproducibility
        # "batch_size": 1024,
    },

    # "sinkhorn": {
    #     "epsilon": 0.1,
    #     "num_iters": 100,
    # },
    # "linearized-wasserstein": {
    #     "epsilon": 0.1,
    # },
    # "neural-ot": {
    #     "hidden_layers": [128, 128],
    #     "learning_rate": 0.001,
    #     "num_epochs": 1000,
    # },
}

################### FUNCTIONS #####################


def load_embeddings(dataset: DatasetName, num_input_files: Optional[int], randomized: bool = True, print_memory_usg: bool = True) -> DataFrame:
    # Get file paths
    data_dir = Path(os.getenv("EMBEDDINGS_PATH")) / DIRECTORY_MAP[dataset]
    print(f"Loading embeddings from {data_dir}")
    # yield files
    files = [file for file in data_dir.iterdir()]
    # Adjust num_input_files if None or larger than available files
    if num_input_files is None or num_input_files > len(files):
        num_input_files = len(files)
    # Get random or fixed sample of data files of size <num_input_files>
    if randomized:
        files = sample(files, k=num_input_files)
    else:
        files = files[:num_input_files]
    # Load data into pandas DataFrame
    df = concat([read_parquet(file, engine='pyarrow')
                for file in files], ignore_index=True)
    # Print memory usage
    if print_memory_usg:
        print(
            f"Memory usage of loaded DataFrame: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")
    return df


class OTSolver:
    @staticmethod
    def get_ot_variant(variant: OTSolverVariant):
        if variant == "sliced-wasserstein":
            from ott.tools.sliced import sliced_wasserstein
            return sliced_wasserstein

        # elif variant == "sinkhorn":
        #     return SlicedWassersteinOT(**kwargs)
        # elif variant == "linearized-wasserstein":
        #     return LinearizedWassersteinOT(**kwargs)
        # elif variant == "neural-ot":
        #     return NeuralOT(**kwargs)
        else:
            raise ValueError(f"Unknown OT variant: {variant}")

    def __init__(self, variant: OTSolverVariant, X: DataFrame, Y: DataFrame):
        self.variant = variant
        self.X = jnp.array(stack(X['embeddings'].values))
        self.Y = jnp.array(stack(Y['embeddings'].values))

    def center_pointclouds(self):
        self.X = self.X - jnp.mean(self.X, axis=0)
        self.Y = self.Y - jnp.mean(self.Y, axis=0)

    def compute_distance(self, **kwargs) -> Any:
        ot_function = self.get_ot_variant(self.variant)
        print(f"Computing {self.variant} distance with parameters:")
        for key, value in kwargs.items():
            print(f"    - {key}: {value}")
        distance = ot_function(self.X, self.Y, **kwargs)
        return distance


def get_ot_distance(solver: OTSolver) -> Tuple[Any, float]:
    """
    Compute the OT distance using the provided solver and measure the time taken.

    :param solver: The OT solver instance to use for computing the distance.
    :type solver: OTSolver
    :return: Return the computed distance and the time taken in seconds.
    :rtype: Tuple[Any, float]
    """
    # Center the pointclouds
    solver.center_pointclouds()

    # Compute distance
    start = time()
    distance = solver.compute_distance(**OT_PARAMS[solver.variant])
    end = time()

    return distance, end - start


def embeddings_analysis_pipeline(
    datasets: Iterable[DatasetName],
    analysis: EmbeddingsAnalysisName,
    ot_variant: OTSolverVariant,
    num_input_files: Optional[int],
    shuffle_files: bool = False,
) -> None:
    # Load embeddings
    df_X = load_embeddings(
        dataset=datasets[0], num_input_files=num_input_files, randomized=shuffle_files, print_memory_usg=True)
    df_Y = load_embeddings(
        dataset=datasets[1], num_input_files=num_input_files, randomized=shuffle_files, print_memory_usg=True)

    size_X = len(df_X)
    size_Y = len(df_Y)

    print(
        f"Loaded embeddings: {datasets[0]} with {size_X} samples, {datasets[1]} with {size_Y} samples.")

    if analysis == "embeddings-distance":
        # Initialize OT solver
        solver = OTSolver(variant=ot_variant, X=df_X, Y=df_Y)

        # Compute OT distance
        distance, duration = get_ot_distance(solver)

        print(
            f"Computed {ot_variant} distance for {datasets[0]} ({size_X} samples) vs {datasets[1]} ({size_Y} samples)\nDistance: {distance[0]}\nDuration: {duration:.2f} seconds.")
    elif analysis == "umap-visualization":
        from umap import UMAP
        import plotly.express as px

        reducer = UMAP(n_neighbors=5, min_dist=0.3,
                       n_components=2, random_state=42)

        # combine embeddings and then fit into UMPAP embedding space
        combined_embeddings = df_X['embeddings'].tolist(
        ) + df_Y['embeddings'].tolist()
        combined_embeddings = reducer.fit_transform(combined_embeddings)
        print(f"Size of combined embeddings: {combined_embeddings.shape}")

        # create labels for the two datasets
        labels = [datasets[0]] * size_X + [datasets[1]] * size_Y

        # plot with plotly and add the queries as metadata to each point
        metadata = df_X['serp_query_text_url'].tolist(
        ) + df_Y['serp_query_text_url'].tolist()

        # create scatter plot
        fig = px.scatter(x=combined_embeddings[:, 0], y=combined_embeddings[:, 1],
                         color=labels,
                         hover_data={'Query': metadata},
                         title=f"UMAP Visualization of {datasets[0]} and {datasets[1]} Embeddings",
                         opacity=0.5)  # Wert zwischen 0 (transparent) und 1 (opak)
        fig.show()

    else:
        raise ValueError(f"Unknown analysis type: {analysis}")


class ResultWriter:
    pass

    # plot the results
    # plt.figure(figsize=(10, 6))
    # plt.plot(projs, distances, marker='o')
    # plt.title('Sliced Wasserstein Distance vs Number of Projections')
    # plt.xlabel('Number of Projections (n_proj)')
    # plt.ylabel('Sliced Wasserstein Distance')
    # plt.grid(True)
    # plt.show()

# TODO
# 1. DataLoader
# - größe des datensatzes bestimmen
# - Quelldaten festlegen (AOL, AQl, etc)

# 2. OT Solver
# - sinkhorn
# - sliced wasserstein distance
# - linearized wasserstein distance
# - neural OT?

# 3. Data Writer
# - zielverzeichnis festlegen
# - dateinamen festlegen
# - format festlegen (csv, json, parquet, etc)
