from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from random import sample
from pandas import DataFrame, concat, read_parquet
import os

################### LOAD CONFIGURATION #####################
# use .env to load data path for embeddings
load_dotenv()

DIRECTORY_MAP = {
    "aol": "aol-get-embeddings-special",
    "aql": "aql-get-embeddings-special",
    "ms-marco": "msmarco-get-embeddings-all",
    "orcas": "orcas-get-embeddings-all",
}

################### FUNCTIONS #####################


def load_embeddings(dataset: str, num_input_files: int) -> DataFrame:
    # Get file paths
    data_dir = Path(os.getenv("EMBEDDINGS_PATH")) / DIRECTORY_MAP[dataset]
    print(f"Loading embeddings from {data_dir}")
    files = list(data_dir.iterdir())

    # Get random sample of data files of size <size num_input_files>
    files = sample(files, k=num_input_files)

    # Load data into pandas DataFrame
    df = concat([read_parquet(file) for file in files], ignore_index=True)

    return df


df = load_embeddings("aol", 10)
print(df.head())


class OTSolver:
    def __init__(self, variant: str):
        self.variant = variant
        self.variant_map = {}


class ResultWriter:
    pass


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
