import ray
import pandas as pd
import numpy as np

# Get first value per group.
ds = ray.data.from_items([
    {"group": 1, "value": 1},
    {"group": 1, "value": 2},
    {"group": 2, "value": 3},
    {"group": 2, "value": 4}])

ds = ds.groupby("group").map_groups(
    lambda g: {"group": [str(g["group"][0])], "result": np.array([np.sum(g["value"])])})

print(ds.take(5))
