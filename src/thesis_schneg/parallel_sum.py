from ray import init, remote, get

# Initialize Ray (and connect to cluster).
init()

# Define the square task.


@remote
def square(x: int) -> int:
    return x * x


# Launch four parallel square tasks.
futures = [square.remote(i) for i in range(1_000)]

# Retrieve results.
print(sum(get(futures)))
