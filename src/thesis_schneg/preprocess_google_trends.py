import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# This script reads all the csv files in the annual folder and concatenates them into one csv file
base_path = Path(
    "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/google_trends/")
result = pd.DataFrame()
csvpath = base_path / "annual"
for file in csvpath.iterdir():
    year = file.stem.split('-')[1]
    if year < '2023':
        table = pd.read_csv(file)
        table['year'] = int(year)
        result = pd.concat([result, table])

result = result.sort_values(by=['year', 'score'], ascending=False)
print(result)

result.to_csv(base_path / "google_trends_total.csv", index=False)
