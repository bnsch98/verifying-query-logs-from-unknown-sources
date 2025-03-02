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
    "extract-words-group",
    "extract-named-entities",
    "extract-gliner-pii",
    "extract-presidio-pii",
    "get-lengths",
    "aql-anomaly",
    "filter-aql-outlier",
    "get-temporal-url-proportion",
    "get-email-proportion",
    "aql-get-words-2006",
    "get-char-count",
    "get-repl-char-proportion",
    "filter-by-year",
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
    "debug",
    "sort-data",

]
