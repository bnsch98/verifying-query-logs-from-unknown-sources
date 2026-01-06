from typing import Any, Optional, Iterable, Tuple, Literal
from thesis_schneg.model import DatasetName, EmbeddingsAnalysisName, OTSolverVariant
from pathlib import Path
from dotenv import load_dotenv
from random import sample
from pandas import DataFrame, concat, read_parquet
from jax.random import PRNGKey
from numpy import stack
import time
import jax.numpy as jnp
import jax
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
        "center_pointclouds": False,
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

# Limit GPU memory usage to 50%
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
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

    def __init__(self, variant: OTSolverVariant, X: DataFrame, Y: DataFrame, device_type: Literal["cpu", "gpu"] = "gpu"):
        self.variant = variant
        self.device_type = device_type

        # 1. Set device
        self.device = jax.devices(self.device_type)[0]
        print(f"Using device: {self.device}")

        # 2. Load data
        print(f"Loading data to {self.device_type.upper()}...")
        self.X = stack(X['embeddings'].values)
        self.Y = stack(Y['embeddings'].values)

        # 3. Get OT function
        if variant == "sliced-wasserstein":
            from ott.tools.sliced import sliced_wasserstein
            self._swd_func = sliced_wasserstein

    # 2. Push data to the selected device
    def push_to_device(self):
        self.X = jax.device_put(
            jnp.array(self.X), self.device)
        self.Y = jax.device_put(
            jnp.array(self.Y), self.device)

    def center_pointclouds(self):
        self.X = self.X - jnp.mean(self.X, axis=0)
        self.Y = self.Y - jnp.mean(self.Y, axis=0)

    # max_samples = 150k
    def compute_distance(self, batch_size: int = 100, **kwargs) -> tuple[float, float]:
        rng = kwargs.get('rng', PRNGKey(42))
        n_proj = kwargs.get('n_proj', 1000)
        start_time = time.perf_counter()

        if self.device_type == "gpu":
            # --- BATCHED GPU VERSION ---
            # We divide projections into smaller chunks to stay below the 4GB VRAM limit.
            # Memory usage is determined by batch_size instead of total n_proj.
            n_batches = n_proj // batch_size

            @jax.jit
            def batched_swd(key):
                def scan_fn(carry, batch_key):
                    # In ott-jax 0.4.x, sliced_wasserstein returns (distance, projections)
                    dist, _ = self._swd_func(
                        self.X, self.Y, n_proj=batch_size, rng=batch_key)
                    return carry + dist, None

                    # Generate n_batches unique keys for the projections
                keys = jax.random.split(key, n_batches)
                # jax.lax.scan is a compiled loop that keeps intermediate results small
                total_dist, _ = jax.lax.scan(scan_fn, 0.0, keys)
                return total_dist / n_batches

            result_array = batched_swd(rng)

        else:
            # --- STANDARD CPU VERSION (Non-batched) ---
            # CPU usually has enough system RAM to handle n_proj=1000 directly.
            @jax.jit
            def simple_swd(key):
                dist, _ = self._swd_func(
                    self.X, self.Y, n_proj=n_proj, rng=key)
                return dist

            result_array = simple_swd(rng)

        # block_until_ready() is mandatory for accurate timing due to JAX's async dispatch
        result_array.block_until_ready()
        duration = time.perf_counter() - start_time

        return float(result_array), duration


def get_ot_distance(solver: OTSolver, batch_size: int = 100, **kwargs) -> Tuple[Any, float]:
    """
    Compute the OT distance using the provided solver and measure the time taken.

    :param solver: The OT solver instance to use for computing the distance.
    :type solver: OTSolver
    :return: Return the computed distance and the time taken in seconds.
    :rtype: Tuple[Any, float]
    """
    # Center the pointclouds if requested
    center_pc = kwargs.get('center_pointclouds', True)
    if center_pc:
        solver.center_pointclouds()

    # Push data to device
    solver.push_to_device()

    # Compute distance
    distance, duration = solver.compute_distance(
        batch_size=batch_size, **kwargs)
    return distance, duration


def embeddings_analysis_pipeline(
    datasets: Iterable[DatasetName],
    analysis: EmbeddingsAnalysisName,
    ot_variant: OTSolverVariant,
    num_input_files: Optional[int],
    shuffle_files: bool = False,
    device_type: Literal["cpu", "gpu"] = "gpu",
    batch_size: int = 100,
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
        solver = OTSolver(variant=ot_variant, X=df_X,
                          Y=df_Y, device_type=device_type)

        # Compute OT distance
        distance, duration = get_ot_distance(
            solver, batch_size=batch_size, **OT_PARAMS[solver.variant])

        print(
            f"Computed {ot_variant} distance for {datasets[0]} ({size_X} samples) vs {datasets[1]} ({size_Y} samples)\nDistance: {distance}\nDuration: {duration:.2f} seconds.")
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
# - neural OT? 0.000562994449865073

# 3. Data Writer
# - zielverzeichnis festlegen
# - dateinamen festlegen
# - format festlegen (csv, json, parquet, etc)
