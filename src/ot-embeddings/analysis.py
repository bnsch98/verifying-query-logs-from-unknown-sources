from typing import Any
from pathlib import Path
from dotenv import load_dotenv
from random import sample
from pandas import DataFrame, concat, read_parquet
import jax.numpy as jnp
from jax.random import PRNGKey
from numpy import stack
import os


################### LOAD CONFIGURATION #####################
# use .env to load data path for embeddings
load_dotenv()

DIRECTORY_MAP = {
    "aol": "aol-get-embeddings-special",
    "aql": "aql-get-embeddings-special",
    "ms-marco": "ms-marco-get-embeddings-all",
    "orcas": "orcas-get-embeddings-all",
}


################### FUNCTIONS #####################
def load_embeddings(dataset: str, num_input_files: int, randomized: bool = True) -> DataFrame:
    # Get file paths
    data_dir = Path(os.getenv("EMBEDDINGS_PATH")) / DIRECTORY_MAP[dataset]
    print(f"Loading embeddings from {data_dir}")

    # Get random or fixed sample of data files of size <size num_input_files>
    if randomized:
        files = sample([file for file in data_dir.iterdir()],
                       k=num_input_files)
    else:
        files = [file for file in data_dir.iterdir()][:num_input_files]

    # Load data into pandas DataFrame
    df = concat([read_parquet(file) for file in files], ignore_index=True)

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

    def compute_distance(self, center_pointcloud: bool = True, **kwargs) -> Any:
        ot_function = self.get_ot_variant(self.variant)
        if center_pointcloud:
            print("Centering point clouds before computing OT distance.")
            self.X = self.X - jnp.mean(self.X, axis=0)
            self.Y = self.Y - jnp.mean(self.Y, axis=0)
        distance = ot_function(self.X, self.Y, **kwargs)
        return distance


class ResultWriter:
    pass


df_X = load_embeddings(dataset="aol", num_input_files=10, randomized=False)
print(f"Loaded {len(df_X)} embeddings for dataset AOL")

df_Y = load_embeddings(
    dataset="ms-marco", num_input_files=10, randomized=False)
print(f"Loaded {len(df_Y)} embeddings for dataset MS-MARCO")

Solver = OTSolver(variant="sliced-wasserstein",
                  X=df_X, Y=df_Y)

distance = Solver.compute_distance(
    center_pointcloud=True, n_proj=50, rng=PRNGKey(42))

print(f"Sliced Wasserstein Distance: {distance[0]}")
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
