from typing import Literal, TYPE_CHECKING, Any


# Workaround as TypeAlias is not yet implemented in older Python versions.
if TYPE_CHECKING:
    from typing import TypeAlias
else:
    TypeAlias = Any


DatasetName: TypeAlias = Literal[
    "aol",
    "ms-marco",
    "orcas",
    "aql",
]


PredictorName: TypeAlias = Literal[
    "query-intent",
    "language",
    "hate-speech",
    "spam",
    "query-rating",
]

AggregatorName: TypeAlias = Literal[
    "zipfs-law",
]

AnalysisName: TypeAlias = Literal[
    'sum-rows',
    'zipfs-law',
    "query-length-chars",
]
