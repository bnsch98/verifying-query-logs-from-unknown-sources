from time import time
from typing import Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from random import sample
from pandas import DataFrame, concat, read_parquet
import jax.numpy as jnp
from jax.random import PRNGKey
from numpy import stack
import matplotlib.pyplot as plt
import os


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


################### FUNCTIONS #####################
def load_embeddings(dataset: str, num_input_files: Optional[int], randomized: bool = True, print_memory_usg: bool = True) -> DataFrame:
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
    df = concat([read_parquet(file, engine='pyarrow', columns=["embeddings"])
                for file in files], ignore_index=True)
    # Print memory usage
    if print_memory_usg:
        print(
            f"Memory usage of loaded DataFrame: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")
    return df


class OTSolver:
    @staticmethod
    def get_ot_variant(variant: str):
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

    def __init__(self, variant: str, X: DataFrame, Y: DataFrame):
        self.variant = variant
        self.X = jnp.array(stack(X['embeddings'].values))
        self.Y = jnp.array(stack(Y['embeddings'].values))

    def center_pointclouds(self):
        self.X = self.X - jnp.mean(self.X, axis=0)
        self.Y = self.Y - jnp.mean(self.Y, axis=0)

    def compute_distance(self, **kwargs) -> Any:
        ot_function = self.get_ot_variant(self.variant)
        print(f"Computing {self.variant} distance with parameters: {kwargs}")
        distance = ot_function(self.X, self.Y, **kwargs)
        return distance


class ResultWriter:
    pass


################### MAIN ANALYSIS #####################
if __name__ == "__main__":
    start = time()
    n = 20
    df_X = load_embeddings(dataset="aol", num_input_files=n,
                           randomized=False, print_memory_usg=True)
    end = time()
    print(
        f"Time taken to load {len(df_X)} embeddings for dataset AOL: {end - start:.2f} seconds")

    start = time()
    df_Y = load_embeddings(
        dataset="ms-marco", num_input_files=n, randomized=False, print_memory_usg=True)
    end = time()
    print(
        f"Time taken to load {len(df_Y)} embeddings for dataset MS-MARCO: {end - start:.2f} seconds")

    start = time()
    # initialize OT solver
    Solver = OTSolver(variant="sliced-wasserstein",
                      X=df_X, Y=df_Y)

    # center the pointclouds
    Solver.center_pointclouds()

    # tracking the influence of n_proj on convergence
    distances = []
    # projs = list(range(10, 100, 10)) + \
    #     list(range(100, 500, 50)) + list(range(500, 3000, 250))
    proj = 1000
    distance = Solver.compute_distance(n_proj=proj, rng=PRNGKey(42))
    end = time()
    print(
        f"Sliced Wasserstein Distance with n_input_files={n} and n_proj={proj}: {distance[0]}\nTime taken: {end - start:.2f} seconds")
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
