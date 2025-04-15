import sys
from pathlib import Path
from sentence_transformers import util
import pandas as pd
import time
import json
from ray.data import read_parquet
from ray import init
import numpy as np
import logging


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


logger = logging.getLogger(__name__)
init()

# dataset name
if sys.argv[1] is not None:
    dataset_name = sys.argv[1]
else:
    raise ValueError("Please provide a dataset name as the first argument.\n")

args = len(sys.argv)

parameters = [0.75, 25, 100000]

if args == 3:
    parameters[0] = float(sys.argv[2])
elif args == 4:
    parameters[0] = float(sys.argv[2])
    parameters[1] = int(sys.argv[3])
elif args == 5:
    parameters[0] = float(sys.argv[2])
    parameters[1] = int(sys.argv[3])
    parameters[2] = int(sys.argv[4])

cosine_similarity = parameters[0]/100
min_cluster_size = parameters[1]
target_size = parameters[2]


logging.basicConfig(
    filename=f'logs/{dataset_name}-{cosine_similarity}-{min_cluster_size}-{target_size}.log', encoding='utf-8', level=logging.INFO)


# load data
data_path = f"/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis/{dataset_name}-get-embeddings-"
if dataset_name == "aol" or dataset_name == 'aql':
    data_path += "special/"
else:
    data_path += "all/"

logger.info(f"Loading data from {data_path}\n")
data_path = Path(data_path)

files = [str(path) for path in data_path.iterdir()
         if path.suffix == ".parquet"]

logger.info("Start reading data\n")
start_time = time.time()

# take subset of data
# input_data = pd.read_parquet(data_path)
input_data = read_parquet(files[0:50])
size = input_data.count()
if size > target_size:
    frac = target_size/size
    input_data = input_data.random_sample(fraction=frac, seed=42)

input_data = input_data.to_pandas()
size = len(input_data)
logger.info(size)
queries = input_data["serp_query_text_url"].tolist()
embeddings = input_data["embeddings"].to_list()
embeddings = [np.array(embedding) for embedding in embeddings]
# convert to numpy array
embeddings = np.array(embeddings)

logger.info(f"Data loaded after {time.time() - start_time:.2f} sec\n")
logger.info("Start clustering\n")
cluster_start_time = time.time()


# Two parameters to tune:
# min_cluster_size: Only consider cluster that have at least 25 elements
# threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
clusters = util.community_detection(
    embeddings, min_community_size=min_cluster_size, threshold=cosine_similarity)

logger.info(
    f"Clustering done after {time.time() - cluster_start_time:.2f} sec\n")

# Print for all clusters the top 3 and bottom 3 elements
for i, cluster in enumerate(clusters):
    logger.info(f"\nCluster {i + 1}, #{len(cluster)} Elements \n")
    for sentence_id in cluster[0:3]:
        logger.info(f"     {queries[sentence_id]}")
    logger.info("     ...\n")
    for sentence_id in cluster[-3:]:
        logger.info(f"     {queries[sentence_id]}")


result_dict = {"queries": queries,
               "embeddings": embeddings, "clusters": clusters}

# round size into thousands
if size > 1000:
    size = round(size / 1000) * 1000

# store the result
output_path = f"/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis/{dataset_name}-size-{size}-clusters{min_cluster_size}-sim{cosine_similarity*100}-sentence-transformers/result.json"

if not Path(output_path).exists():
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
else:
    logger.info(f"Output path {output_path} already exists. Overwriting.\n")

result_dict = json.dumps(result_dict, cls=NumpyEncoder)

# save the result
with open(output_path, "w") as f:
    json.dump(result_dict, f)
logger.info(f"Clustering results saved to {output_path}\n")

end_time = time.time()
logger.info(
    f"{dataset_name.upper()} Total time taken: {end_time - start_time:.2f} sec\n")
