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
    "extract-chars",
    "extract-words",
    "extract-named-entities",
    "get-lengths",
    "characters-count-frequencies",
    "words-count-frequencies",
    "entity-count-frequencies",
    "query-count-frequencies",
    "query-intent",
    "zipfs-law-queries",
    "zipfs-law-words",
    "zipfs-law-chars",
    "query-length-chars",
    "query-length-words",
    "unique-queries",
    "named-entities-count",
    "extract-search-operators",
    "search-operators-count",
    "heaps-law-words",
]
