from pathlib import Path
from random import choices
from typing import Iterable, Optional, Callable, Protocol, Union, Any, Dict
from pandas import DataFrame
from ray import init
from ray.data import read_parquet, Dataset
from ray.data.aggregate import AggregateFn
from thesis_schneg.model import DatasetName, AnalysisName
from json import dumps
from functools import cached_property
from dataclasses import dataclass
from spacy import load as spacy_load, Language, explain
from gliner import GLiNER
from functools import partial
from thesis_schneg.classification_module import QueryIntentPredictor, nvidiaDomainClassifier, nvidiaQualityClassifier, NSFWPredictor

############################################    Requirements for basic modules    #####################################


class _spacy_framework(Protocol):
    def get_spacy_vals(self, row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError()

    def __call__(self, row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        return self.get_spacy_vals(row)


class _gliner_framework(Protocol):
    def get_gliner_vals(self, row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError()

    def __call__(self, row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        return self.get_gliner_vals(row)


@dataclass(frozen=True)
class GlinerGetEntities(_gliner_framework):

    @cached_property
    def gliner_model(self) -> GLiNER:
        return GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")

    @cached_property
    def gliner_labels(self) -> Iterable[str]:
        return ["person", "organization", "phone number", "address", "passport number", "email", "credit card number", "social security number", "health insurance id number", "date of birth", "mobile phone number", "bank account number", "medication", "cpf", "driver's license number", "tax identification number", "medical condition", "identity card number", "national id number", "ip address", "email address", "iban", "credit card expiration date", "username", "health insurance number", "registration number", "student id number",
                "insurance number", "flight number", "landline phone number", "blood type", "cvv", "reservation number", "digital signature", "social media handle", "license plate number", "cnpj", "postal code", "passport_number", "serial number", "vehicle registration number", "credit card brand", "fax number", "visa number", "insurance company", "identity document number", "transaction number", "national health insurance number", "cvc", "birth certificate number", "train ticket number", "passport expiration date", "social_security_number"]

    def get_gliner_vals(self, row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        # Get tokens from text
        doc = self.gliner_model.predict_entities(
            text=row["serp_query_text_url"], labels=self.gliner_labels)

        entities = [{"entity": ent["text"], "entity-label": ent["label"]}
                    for ent in doc if doc]
        return entities


@dataclass(frozen=True)
class SpacyWords(_spacy_framework):

    @cached_property
    def spacy_model(self) -> Language:
        return spacy_load("en_core_web_sm")

    def get_spacy_vals(self, row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        # Get tokens from text
        doc = self.spacy_model(row["serp_query_text_url"])
        tokens = [{"word": token.text} for token in doc]
        return tokens


@dataclass(frozen=True)
class SpacyGetEntities(_spacy_framework):

    @cached_property
    def spacy_model(self) -> Language:
        return spacy_load("en_core_web_sm")

    def get_spacy_vals(self, row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        # Get tokens from text
        doc = self.spacy_model(row["serp_query_text_url"])
        entities = [{"entity": ent.text, "entity-label": ent.label_+'-'+str(explain(ent.label_))}
                    for ent in doc.ents if doc.ents != ()]
        return entities


@dataclass(frozen=True)
class SpacyQueryLevelStructures(_spacy_framework):

    @cached_property
    def spacy_model(self) -> Language:
        return spacy_load("en_core_web_sm")

    def get_spacy_vals(self, batch: DataFrame) -> DataFrame:
        batch['character-count'] = batch['serp_query_text_url'].apply(len)
        batch['word-count'] = batch['serp_query_text_url'].apply(
            lambda x: len(x.split()))
        # Get entity count
        predictions = [len([ent for ent in self.spacy_model(query).ents])
                       for query in list(batch["serp_query_text_url"])]
        batch['entity-count'] = predictions
        return batch


def _get_parquet_paths(
    dataset_name: DatasetName,
    analysis_name: AnalysisName,  # word-count-frequencies
    struc_level: Optional[str] = None,
    sample_files: Optional[int] = None,
    only_english: bool = False,
    which_half: Optional[str] = None
) -> Iterable[Path]:
    base_path: Path

    if analysis_name in ["character-count-frequencies", "word-count-frequencies", "entity-count-frequencies", "query-count-frequencies", "filter-urls", "aql-anomaly"]:
        assert struc_level is not None, "Structural level must be specified by \"--struc-level\" [queries, named-entities, words]"
        base_path = Path(
            f"/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis/{dataset_name}-get-lengths-{struc_level}-all/"
        )
        assert base_path.is_dir(
        ), f"No directory found for dataset = {dataset_name} and struc_level = {struc_level}"
    else:
        if struc_level in [None, "queries"]:
            if dataset_name == "aol":
                base_path = Path(
                    "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/aol_output/"
                )
            elif dataset_name == "ms-marco":
                if only_english:
                    base_path = Path(
                        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/lng_filtered_ms-marco/"
                    )
                else:
                    base_path = Path(
                        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/msmarco_output/"
                    )
            elif dataset_name == "orcas":
                base_path = Path(
                    "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/orcas_output/"
                )
            elif dataset_name == "aql":
                if only_english:
                    base_path = Path(
                        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/lng_filtered_aql/"
                    )
                else:
                    base_path = Path(
                        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/aql_output/"
                    )
        else:
            base_path = Path(
                f"/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis/{dataset_name}-extract-{struc_level}-all/"
            )
            assert base_path.is_dir(
            ), f"No directory found for dataset = {dataset_name} and struc_level = {struc_level}"

    input_paths = [path for path in base_path.iterdir()
                   if path.suffix == ".parquet"]
    assert len(input_paths) > 0, f"No parquet files found in {base_path}"

    assert which_half in [None, "first", "second"], "Invalid half specified"
    assert not (
        sample_files is not None and which_half is not None), "Cannot specify both \"sample_files\" and \"which_half\""

    if sample_files is not None and which_half is None:
        input_paths = choices(
            population=input_paths,
            k=min(sample_files, len(input_paths)),
        )
    elif sample_files is None and which_half is not None:
        if which_half == "first":
            input_paths = input_paths[:len(input_paths)//2]
        elif which_half == "second":
            input_paths = input_paths[len(input_paths)//2:]
    assert input_paths, f"No files found in {base_path.name}"
    return input_paths


############################################    Basic Modules    #######################################
def load_dataset(dataset_name: DatasetName,
                 analysis_name: AnalysisName,
                 struc_level: Optional[str] = None,
                 sample_files: Optional[int] = None,
                 only_english: bool = False,
                 read_concurrency: Optional[int] = None,
                 columns: Optional[Iterable[str]] = None,
                 memory_scaler: float = 1.0,
                 which_half: Optional[str] = None,
                 read_dir: Optional[Path] = None
                 ) -> Dataset:

    # Load dataset.
    if read_dir is not None:
        dataset = read_parquet(
            paths=read_dir,
            concurrency=read_concurrency,
            columns=columns,
            ray_remote_args={"memory": memory_scaler*1000*1000*1000}
        )
    else:
        dataset = read_parquet(
            paths=[
                str(path)
                for path in _get_parquet_paths(
                    dataset_name=dataset_name,
                    analysis_name=analysis_name,
                    struc_level=struc_level,
                    sample_files=sample_files,
                    only_english=only_english,
                    which_half=which_half
                )
            ],
            concurrency=read_concurrency,
            columns=columns,
            ray_remote_args={"memory": memory_scaler*1000*1000*1000}
        )
    return dataset


def map_dataset(dataset: Dataset,
                mapping_func: Callable[[DataFrame], DataFrame],
                concurrency: Optional[int] = None,
                batch_size: int = 16,
                num_gpus: float = None,
                num_cpus: float = None,
                memory_scaler: float = 1.0) -> Dataset:
    return dataset.map_batches(
        mapping_func,
        concurrency=concurrency,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        batch_size=batch_size,
        batch_format="pandas",
        memory=memory_scaler*1000*1000*1000,
    )


def flat_map_dataset(dataset: Dataset,
                     flat_mapping_func: Callable[[Dict[str, Any]], Dict[str, Any]],
                     concurrency: Optional[int] = None,
                     num_cpus: Optional[float] = None,
                     num_gpus: Optional[float] = None,
                     memory_scaler: float = 1.0
                     ) -> Dataset:
    return dataset.flat_map(fn=flat_mapping_func, concurrency=concurrency, num_cpus=num_cpus, num_gpus=num_gpus, memory=memory_scaler*1000*1000*1000)


def aggregate_dataset(dataset: Dataset, aggregation_func: AggregateFn) -> Optional[Dict[str, Any]]:
    return dataset.aggregate(aggregation_func)


def write_dataset(dataset: Union[Dict, Dataset, DataFrame], write_dir: Path, analysis_name: str, struc_level: str, dataset_name: str, sample_files: int, which_half: Optional[str], read_dir: Optional[Path], write_concurrency: Optional[int] = 2) -> None:
    # Specifiy output directory
    if struc_level is not None:
        if sample_files is not None:
            write_dir = write_dir.joinpath(
                f"{dataset_name}-{analysis_name}-{struc_level}-{sample_files}")
        else:
            if which_half is not None:
                write_dir = write_dir.joinpath(
                    f"{dataset_name}-{analysis_name}-{struc_level}-{which_half}")
            else:
                if read_dir is not None:
                    write_dir = write_dir.joinpath(
                        f"{dataset_name}-{struc_level}-{analysis_name}-special")
                else:
                    write_dir = write_dir.joinpath(
                        f"{dataset_name}-{analysis_name}-{struc_level}-all")
    else:
        if sample_files is not None:
            write_dir = write_dir.joinpath(
                f"{dataset_name}-{analysis_name}-{sample_files}")
        else:
            if which_half is not None:
                write_dir = write_dir.joinpath(
                    f"{dataset_name}-{analysis_name}-{which_half}")
            else:
                write_dir = write_dir.joinpath(
                    f"{dataset_name}-{analysis_name}-all")

    # Delete old files
    if write_dir.exists():
        [f.unlink() for f in write_dir.glob("*") if f.is_file()]

    # Write output
    if type(dataset) is dict:
        # Make directory to work around FileNotFoundError
        write_dir.mkdir(parents=True, exist_ok=True)
        # Distinguish between nested dict and flat dict. We rule out deeper nesting.
        if type(dataset[analysis_name]) is dict:
            # Write json file
            with write_dir.joinpath("result.json").open("w+", encoding="utf-8") as f:
                f.write(dumps(dataset[analysis_name]))
        else:
            # Write json file
            with write_dir.joinpath("result.json").open("w+", encoding="utf-8") as f:
                f.write(dumps(dataset))
    elif type(dataset) is Dataset:
        # Write parquet file
        dataset.write_parquet(path=str(write_dir),
                              concurrency=write_concurrency)
    elif type(dataset) is DataFrame:
        # Write csv file
        dataset.to_csv(path_or_buf=write_dir.joinpath(
            "result.csv"), index=False)
    else:
        print("Unknown type of output")


############################################    Task Specific Functions    ###########################################

# Mapping functions
def identity(batch: DataFrame) -> DataFrame:
    return batch


def get_length_char(batch: DataFrame) -> DataFrame:
    batch['query-length-chars'] = batch['serp_query_text_url'].apply(len)
    return batch


def get_length_word(batch: DataFrame) -> DataFrame:
    batch['query-length-words'] = batch['serp_query_text_url'].apply(
        lambda x: len(x.split()))
    return batch


def get_operator_count(batch: DataFrame) -> DataFrame:
    batch['operator-count'] = batch['serp_query_text_url'].apply(
        lambda x: sum([x.count(operator) for operator in ["site:", "filetype:", "intitle:", "allinurl:", "allintitle:",
                                                          "intext:", "allintext:", "related:", "define:", "cache:", "around(", " OR ", " AND "]]))
    return batch


def get_lengths(batch: DataFrame, structural_level: str) -> DataFrame:
    if structural_level == "words":
        batch['character-count'] = batch['word'].apply(len)
    elif structural_level == "named-entities":
        batch['character-count'] = batch['entity'].apply(len)
        batch['word-count'] = batch['entity'].apply(lambda x: len(x.split()))

    return batch


def filter_urls(batch: DataFrame) -> DataFrame:
    return batch[~batch['word'].str.contains(pat="http|www.|.com|.net|.org|.int|.edu", na=False)]


def filter_lengths(batch: DataFrame) -> DataFrame:
    return batch[(batch['character-count'] == 14) | (batch['character-count'] == 16) | (batch['character-count'] == 24)]

# Flat mapping functions


def _extract_chars(row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    return [{"char": char} for char in row['serp_query_text_url'].replace(" ", "")]


def _extract_operators(row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    res = [{"operator": operator} for operator in ["site:", "filetype:", "intitle:", "allinurl:", "allintitle:",
                                                   "intext:", "allintext:", "related:", "define:", "cache:", "around(", " OR ", " AND "] if operator in row['serp_query_text_url']]
    return res


# Aggregation functions
sum_rows = AggregateFn(
    init=lambda: 0,
    # Apply this to each row to produce a partial aggregate result
    accumulate_row=lambda a, row: a + 1,
    # Apply this to merge partial aggregate results into a final result
    merge=lambda a1, a2: a1 + a2,
    name="sum-rows"
)

unique_queries_agg = AggregateFn(
    init=lambda: {"sum": 0, "unique": 0},
    # Apply this to each row to produce a partial aggregate result

    accumulate_row=lambda a, row: acc_row(a, row),
    # Apply this to merge partial aggregate results into a final result
    merge=lambda a1, a2: sum_dict(a1, a2),

    name="unique-queries"
)

unique_words_agg = AggregateFn(
    init=lambda: {"sum": 0, "unique": 0},
    # Apply this to each row to produce a partial aggregate result

    accumulate_row=lambda a, row: acc_row(a, row),
    # Apply this to merge partial aggregate results into a final result
    merge=lambda a1, a2: sum_dict(a1, a2),

    name="unique-words"
)

aggregation_query_length = AggregateFn(
    init=lambda _: {},
    accumulate_row=lambda counts, row: {
        length:
        count + 1 if length == row["length"] else count
        for length, count in counts.items()
    },
    merge=lambda counts1, counts2: {
        length: counts1.get(length, 0) + counts1.get(length, 0)
        for length in {*counts1.keys(), *counts2.keys()}
    },
    name="count"
)


# Helper functions for aggregation
def acc_row(aggr_dict: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    aggr_dict["sum"] += row["count()"]
    aggr_dict["unique"] += 1
    return aggr_dict


def sum_dict(a1: Dict[str, Any], a2: Dict[str, Any]) -> Dict[str, Any]:
    return {k: int(a1.get(k, 0) + a2.get(k, 0)) for k in set(a1) | set(a2)}


# Group-by function
def groupby_count(dataset: Dataset, col: str) -> Dataset:
    return dataset.groupby(key=col).count()

############################################    Get task-specific modules     #########################################


def _get_module_specifics(analysis_name: AnalysisName, struc_level: Optional[int]) -> Dict[str, Any]:
    if analysis_name == "extract-chars":
        return {'groupby_func': partial(groupby_count, col='char'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': _extract_chars, 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "extract-words":
        return {'groupby_func': partial(groupby_count, col='word'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': SpacyWords(), 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "extract-named-entities":
        return {'groupby_func': partial(groupby_count, col=['entity', 'entity-label']), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': SpacyGetEntities(), 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "extract-pii":
        return {'groupby_func': partial(groupby_count, col=['entity', 'entity-label']), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': GlinerGetEntities(), 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "get-lengths":
        assert struc_level is not None, "Structural level must be specified"
        return {'groupby_func': None, 'aggregator': None, 'mapping_func': [partial(get_lengths, structural_level=struc_level)] if struc_level in ["words", "named-entities"] else [SpacyQueryLevelStructures()], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url'] if struc_level == "query" else None}
    elif analysis_name == "aql-anomaly":
        return {'groupby_func': partial(groupby_count, col=['serp_query_text_url', 'character-count']), 'aggregator': None, 'mapping_func': [filter_lengths], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url', 'character-count']}
    elif analysis_name == "character-count-frequencies":
        return {'groupby_func': partial(groupby_count, col='character-count'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': None}
    elif analysis_name == "word-count-frequencies":
        return {'groupby_func': partial(groupby_count, col='word-count'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': None}
    elif analysis_name == "entity-count-frequencies":
        return {'groupby_func': partial(groupby_count, col='entity-count'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': None}
    elif analysis_name == "query-frequencies":
        return {'groupby_func': partial(groupby_count, col='serp_query_text_url'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url']}

    elif analysis_name == "query-intent":
        return {'groupby_func': partial(groupby_count, col='query-intent'), 'aggregator': None, 'mapping_func': [QueryIntentPredictor()], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "query-domain":
        return {'groupby_func': partial(groupby_count, col='query-domain'), 'aggregator': None, 'mapping_func': [nvidiaDomainClassifier()], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "query-quality":
        return {'groupby_func': partial(groupby_count, col='query-quality'), 'aggregator': None, 'mapping_func': [nvidiaQualityClassifier()], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "query-nsfw":
        return {'groupby_func': partial(groupby_count, col='query-nsfw'), 'aggregator': None, 'mapping_func': [NSFWPredictor()], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url']}

    elif analysis_name == "filter-urls":
        return {'groupby_func': None, 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': filter_urls, 'col_filter': None}

    elif analysis_name == "zipfs-law-queries":
        return {'groupby_func': partial(groupby_count, col='serp_query_text_url'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "zipfs-law-words":
        return {'groupby_func': partial(groupby_count, col='word'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': SpacyWords(), 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "zipfs-law-chars":
        return {'groupby_func': partial(groupby_count, col='char'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': _extract_chars, 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "unique-queries":
        return {'groupby_func': partial(groupby_count, col='serp_query_text_url'), 'aggregator': unique_queries_agg, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "heaps-law-words":
        return {'groupby_func': partial(groupby_count, col='word'), 'aggregator': unique_words_agg, 'mapping_func': None, 'flat_mapping_func': SpacyWords(), 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "query-length-chars":
        return {'groupby_func': partial(groupby_count, col='query-length-chars'), 'aggregator': None, 'mapping_func': [get_length_char], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "query-length-words":
        return {'groupby_func': partial(groupby_count, col='query-length-words'), 'aggregator': None, 'mapping_func': [get_length_word], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url']}

    elif analysis_name == "named-entities-count":
        return {'groupby_func': None, 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "extract-search-operators":
        return {'groupby_func': partial(groupby_count, col='operator'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': _extract_operators, 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "search-operators-count":
        return {'groupby_func': partial(groupby_count, col='operator-count'), 'aggregator': None, 'mapping_func': [get_operator_count], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url']}


############################################    Pipeline    #############################################
def analysis_pipeline(dataset_name: DatasetName,
                      analysis_name: AnalysisName,
                      struc_level: Optional[str] = None,
                      sample_files: Optional[int] = None,
                      only_english: bool = False,
                      which_half: Optional[str] = None,
                      read_concurrency: Optional[int] = None,
                      concurrency: Optional[int] = None,
                      batch_size: int = 16,
                      memory_scaler: float = 1.0,
                      num_cpus: Optional[float] = None,
                      num_gpus: Optional[float] = None,
                      write_dir: Path = Path(
        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis"),
    write_concurrency: Optional[int] = 2,
    read_dir: Optional[Path] = None
) -> None:
    assert struc_level in [None, "words",
                           "named-entities", "queries"], "Invalid structural level"

    init()

    # Load module specifics
    module_specifics = _get_module_specifics(
        analysis_name=analysis_name, struc_level=struc_level)

    # Load dataset.
    ds = load_dataset(dataset_name=dataset_name, struc_level=struc_level, sample_files=sample_files,
                      only_english=only_english, read_concurrency=read_concurrency, columns=module_specifics['col_filter'], memory_scaler=memory_scaler, which_half=which_half, analysis_name=analysis_name, read_dir=read_dir)

    # Apply mapping function.
    if module_specifics['mapping_func'] is not None:
        # iterate through list of mapping functions
        for func in module_specifics['mapping_func']:
            ds = map_dataset(dataset=ds, mapping_func=func,
                             concurrency=concurrency, batch_size=batch_size, num_gpus=num_gpus, num_cpus=num_cpus, memory_scaler=memory_scaler)

    # Apply flat mapping function.
    if module_specifics['flat_mapping_func'] is not None:
        ds = flat_map_dataset(dataset=ds, flat_mapping_func=module_specifics['flat_mapping_func'],
                              concurrency=concurrency, num_cpus=num_cpus, num_gpus=num_gpus, memory_scaler=memory_scaler)

    # Group by a column.
    if module_specifics['groupby_func'] is not None:
        ds = module_specifics['groupby_func'](
            dataset=ds)

    # Apply aggregation function.
    if module_specifics['aggregator'] is not None:
        ds = aggregate_dataset(
            dataset=ds, aggregation_func=module_specifics['aggregator'])

    # Print results for debugging.
    if type(ds) is Dataset:
        print(ds.take(10))
    elif type(ds) is dict:
        print(ds)

    # Write results.
    write_dataset(dataset=ds, write_dir=write_dir,
                  analysis_name=analysis_name, write_concurrency=write_concurrency, struc_level=struc_level, dataset_name=dataset_name, sample_files=sample_files, which_half=which_half, read_dir=read_dir)
