from ray import init
from ray.data import range

# Initialize Ray (and connect to cluster).
init()

# Create dataset rows with the "id" field containing ascending numbers.
numbers = range(1_000_000)

# Keep only odd numbers.
odds = numbers.filter(lambda row: row["id"] % 2 == 0)

# Square numbers.


def square(row: dict) -> dict:
    row["id"] = row["id"] * row["id"]
    return row


squares = odds.map(square)

# Calculate the sum.
print(squares.sum("id"))
# squares.repartition(1).write_json('local:///mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/file')