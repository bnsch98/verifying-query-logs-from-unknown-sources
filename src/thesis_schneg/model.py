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

AnalysisName: TypeAlias = Literal[
    "extract-chars",
    "extract-words",
    "extract-named-entities",
    "extract-pii",
    "get-lengths",
    "aql-anomaly",
    "character-count-frequencies",
    "word-count-frequencies",
    "entity-count-frequencies",
    "query-frequencies",
    "query-intent",
    "query-domain",
    "query-quality",
    "query-nsfw",
    "filter-urls",
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
