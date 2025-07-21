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
    "clean-query-log",
    "get-too-short-queries",
    "extract-chars",
    "extract-words",
    "extract-words-merge",
    "extract-named-entities",
    "extract-gliner-pii",
    "extract-presidio-pii",
    "group-presidio-pii",
    "get-lengths",
    "sum-rows",
    "aql-anomaly",
    "filter-aql-outlier",
    "get-temporal-url-proportion",
    "get-email-proportion",
    "aql-get-words-2006",
    "get-repl-char-proportion",
    "filter-by-year-clean-repl-char",
    "character-count-frequencies",
    "word-count-frequencies",
    "entity-count-frequencies",
    "query-frequencies",
    "query-intent",
    "filter-urls",
    "named-entities-count",
    "extract-search-operators",
    "search-operators-count",
    "debug",
    "sort-grouped-data",
    "query-chart-by-year",
    "get-annual-top-queries",
    "filter-google-queries",
    "get-temporal-query-frequency",
    "get-monthly-google-queries",
    "transform-timestamps",
    "get-query-frequency",
    "get-embeddings",
    "questions",
    "deduplicate-queries",
    "deduplicate-lowercase-queries",
    "get-query-overlap",
    "deduplicate-queries-per-year",
    "deduplicate-lowercase-queries-per-year",
]
