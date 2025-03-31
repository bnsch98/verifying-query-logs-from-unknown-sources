import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

# This script reads all the csv files in the annual folder and concatenates them into one csv file
# base_path = Path(
#     "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/google_trends/")
# result = pd.DataFrame()
# csvpath = base_path / "annual"
# for file in csvpath.iterdir():
#     year = file.stem.split('-')[1]
#     if year < '2023':
#         table = pd.read_csv(file)
#         table['year'] = int(year)
#         result = pd.concat([result, table])

# result = result.sort_values(by=['year', 'score'], ascending=False)
# print(result)

# result.to_csv(base_path / "google_trends_total.csv", index=False)


base_path = Path("/home/benjamin/studium/masterarbeit/thesis-schneg/trends")

data = pd.DataFrame()
for path in base_path.iterdir():
    if path.is_file():
        df = pd.read_csv(path, skiprows=2, nrows=25, header=None)
        df.columns = ['query', 'score']
        df = df.sort_values(by='score', ascending=False)
        df['rank'] = np.arange(1, len(df)+1)
        data = pd.concat([data, df])

write_path = Path(
    "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/google_trends/monthly")

if write_path.exists():
    for file in write_path.iterdir():
        file.unlink()
else:
    write_path.mkdir()

data.to_csv(write_path / "google_trends_total.csv", index=False)
