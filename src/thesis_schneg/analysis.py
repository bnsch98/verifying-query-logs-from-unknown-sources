# from tirex_tracker import fetch_info
# print(fetch_info())
from functools import partial
from sentence_transformers import SentenceTransformer
from presidio_analyzer import AnalyzerEngine
from gliner import GLiNER
from spacy import load as spacy_load, Language, explain
from dataclasses import dataclass
from functools import cached_property
from json import dumps, load as json_load
from thesis_schneg.model import DatasetName, AnalysisName
from ray.data.grouped_data import GroupedData
from ray.data.aggregate import AggregateFn
from ray.data import read_parquet, Dataset
from ray import init
from pandas import DataFrame
from numpy import array as np_array, sum as np_sum
from typing import Iterable, Optional, Callable, Protocol, Union, Any, Dict, Mapping
from random import choices
from datetime import datetime
from re import compile, findall
from pathlib import Path
from safetensors.torch import load_model
from torch import device
from torch.cuda import is_available as cuda_is_available
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    pipeline,
    Pipeline,
)
# from tirex_tracker import fetch_info
# print(fetch_info())


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


class _presidio_framework(Protocol):
    def get_presidio_vals(self, row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError()

    def __call__(self, row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        return self.get_presidio_vals(row)


class Sentence_Predictor(Protocol):
    def get_embeddings(self, batch: DataFrame) -> DataFrame:
        raise NotImplementedError()

    def __call__(self, batch: DataFrame) -> DataFrame:
        return self.get_embeddings(batch)


class _Predictor(Protocol):
    def predict_batch(self, batch: DataFrame) -> DataFrame:
        raise NotImplementedError()

    def __call__(self, batch: DataFrame) -> DataFrame:
        return self.predict_batch(batch)

    @cached_property
    def _device(self) -> device:
        return device("cuda" if cuda_is_available() else "cpu")


@dataclass(frozen=True)
class QueryIntentPredictor(_Predictor):
    labels_path: Path = Path(
        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/model/labels.json"
    )
    model_path: Path = Path(
        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/BERT_intent_classifier/model/bert-orcas-i-level1-query.safetensors"
    )

    @cached_property
    def _id2label(self) -> Mapping[int, str]:
        # Load labels.
        with self.labels_path.open("rb") as file:
            label_dict = json_load(fp=file)

        # Swap label dict keys/values.
        return {v: k for k, v in label_dict.items()}

    @cached_property
    def _tokenizer(self) -> BertTokenizer:
        return BertTokenizer.from_pretrained(
            pretrained_model_name_or_path="bert-base-uncased",
            do_lower_case=True,
        )

    @cached_property
    def _model(self) -> BertForSequenceClassification:
        # Load the BERT model.
        model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path="bert-base-uncased",
            id2label=self._id2label,
            output_attentions=False,
            output_hidden_states=False,
        )

        model.to(self._device)

        # Load state of trained model.
        load_model(
            model=model,
            filename=str(self.model_path),
            strict=False,
            device=self._device.type,
        )

        return model

    @cached_property
    def _pipeline(self) -> Pipeline:
        return pipeline(
            task="text-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            device=self._device,
            padding=True,
            truncation=True,
            top_k=1,
        )

    def predict_batch(self, batch: DataFrame) -> DataFrame:
        # Reset call count of classifier pipeline.
        self._pipeline.call_count = 0
        predictions = self._pipeline(list(batch["serp_query_text_url"]))
        batch["query-intent"] = [prediction[0]["label"]
                                 for prediction in predictions]
        return batch


@dataclass(frozen=True)
class SentenceEmbedder(Sentence_Predictor):

    @cached_property
    def model(self) -> SentenceTransformer:
        return SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    def get_embeddings(self, batch: DataFrame) -> DataFrame:
        embeddings = self.model.encode(
            list(batch['serp_query_text_url']), convert_to_tensor=True)
        batch['embeddings'] = [embeddings[i].tolist()
                               for i in range(len(embeddings))]
        return batch


@dataclass(frozen=True)
class PresidioGetEntities(_presidio_framework):

    @cached_property
    def presidio_model(self) -> AnalyzerEngine:
        return AnalyzerEngine()

    def get_presidio_vals(self, row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        # Get tokens from text
        doc = self.presidio_model.analyze(
            text=row["serp_query_text_url"], language='en')
        # "entity": row['serp_query_text_url'][ent.to_dict()['start']:ent.to_dict()['end']],
        entities = [{"entity-label": ent.to_dict()["entity_type"]}
                    for ent in doc if doc and ent.to_dict()["score"] >= 0.7]
        return entities


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
        tokens = [{"word": token.text}
                  for token in doc if not token.is_punct]
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
        spacy_doc = [self.spacy_model(row) for row in list(
            batch['serp_query_text_url'])]

        # Get word count
        batch['word-count'] = [len(
            [token.text for token in doc if not token.is_punct]) for doc in spacy_doc]

        # Get entity count
        batch['entity-count'] = [len([ent for ent in doc.ents])
                                 for doc in spacy_doc]

        return batch


@dataclass(frozen=True)
class SpacyEntityLevelStructures(_spacy_framework):

    @cached_property
    def spacy_model(self) -> Language:
        return spacy_load("en_core_web_sm")

    def get_spacy_vals(self, batch: DataFrame) -> DataFrame:
        batch['character-count'] = batch['entity'].apply(len)
        spacy_doc = [self.spacy_model(row) for row in list(
            batch['entity'])]

        # Get word count
        batch['word-count'] = [len(
            [token.text for token in doc if not token.is_punct]) for doc in spacy_doc]

        return batch


def _get_parquet_paths(
    dataset_name: DatasetName,
    analysis_name: AnalysisName,
    read_dir: Optional[Path] = None,
    struc_level: Optional[str] = None,
    sample_files: Optional[int] = None,
    only_english: bool = False,
    which_half: Optional[str] = None
) -> Iterable[Path]:
    base_path: Path
    if read_dir is None:
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

        assert which_half in [None, "first",
                              "second"], "Invalid half specified"
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
    else:
        if read_dir is str:
            read_dir = Path(read_dir)
        assert read_dir.is_dir(), f"Invalid directory {read_dir}"
        input_paths = [path for path in read_dir.iterdir()
                       if path.suffix == ".parquet"]
        assert len(input_paths) > 0, f"No parquet files found in {read_dir}"

        if sample_files is not None:
            input_paths = choices(
                population=input_paths,
                k=min(sample_files, len(input_paths)),
            )
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
    dataset = read_parquet(
        paths=[
            str(path)
            for path in _get_parquet_paths(
                dataset_name=dataset_name,
                analysis_name=analysis_name,
                read_dir=read_dir,
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


def map_groups(dataset: GroupedData, map_group_func: Callable[[Any], Any], memory_scaler: float = 1.0, concurrency: Optional[int] = None) -> Dataset:
    return dataset.map_groups(map_group_func, concurrency=concurrency, memory=memory_scaler*1000*1000*1000)


def write_dataset(dataset: Union[Dict, Dataset, DataFrame], write_dir: Path, analysis_name: str, struc_level: str, dataset_name: str, sample_files: int, which_half: Optional[str], read_dir: Optional[Path], write_concurrency: Optional[int] = 2, only_english: bool = False) -> None:
    # check if wirte_dir is Path
    if type(write_dir) is not Path:
        write_dir = Path(write_dir)

    # Specifiy output directory if standard path was parsed. If not, we assume that a specific path was parsed via the CLI and no further specification is necessary.
    if str(write_dir) == "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis":
        output_folder = f"{dataset_name}-{analysis_name}"

        if struc_level is not None:
            output_folder += f"-{struc_level}"
        if which_half is not None:
            output_folder += f"-{which_half}"
        if sample_files is not None:
            output_folder += f"-{sample_files}"
            if read_dir is not None:
                if "google" in str(read_dir):
                    output_folder += "-google"
                elif "english" in str(read_dir):
                    output_folder += "-english"
                else:
                    output_folder += "-special"
        else:
            if read_dir is not None:
                if "google" in str(read_dir):
                    output_folder += "-google"
                elif "english" in str(read_dir):
                    output_folder += "-english"
                else:
                    output_folder += "-special"
            else:
                output_folder += "-all"
        if only_english:
            output_folder += "-english"

        write_dir = write_dir.joinpath(output_folder)

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


###########################################    Task Specific Functions    ###########################################

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
    return batch


def filter_lengths(batch: DataFrame) -> DataFrame:
    return batch[(batch['character-count'] == 14) | (batch['character-count'] == 16) | (batch['character-count'] == 24)]


def filter_aql(batch: DataFrame) -> DataFrame:
    # filter out some absurtly frequent queries
    return batch[~batch['serp_query_text_url'].isin(["茅聵驴茅聡聦猫聹聵猫聸聸忙卤", "#FreeMariaButina", "พระชัยหลวงพ่อโสธรปี 2505", "é\x98¿é\x87\x8cè\x9c\x98è\x9b\x9bæ±"])]


def filter_empty_timestamps(batch: DataFrame) -> DataFrame:
    return batch[~batch['serp_timestamp'].isnull()]


def filter_empty_queries(batch: DataFrame) -> DataFrame:
    return batch[batch['serp_query_text_url'].apply(len) != 0]


def get_year(batch: DataFrame) -> DataFrame:
    batch['serp_timestamp'] = batch['serp_timestamp'].apply(
        lambda x: datetime.fromtimestamp(x).year)
    batch.rename(columns={'serp_timestamp': 'year'}, inplace=True)
    return batch


def filter_by_year(batch: DataFrame, year: Iterable[int]) -> DataFrame:
    # filter out empty timestamps
    batch = batch[~batch['serp_timestamp'].isnull()]
    batch['year'] = batch['serp_timestamp'].apply(
        lambda x: datetime.fromtimestamp(x).year)
    return batch[batch['year'].isin(year)]


def filter_by_char(batch: DataFrame, char: str) -> DataFrame:
    return batch[~batch['serp_query_text_url'].str.contains(char)]


def filter_by_col_value(batch: DataFrame, col: str, value: str) -> DataFrame:
    return batch[batch[col] == value]


def get_short_queries(batch: DataFrame) -> DataFrame:
    batch['character-count'] = batch['serp_query_text_url'].apply(len)
    return batch[batch['character-count'] < 2]


def get_repl_char(batch: DataFrame) -> DataFrame:
    batch['is-repl-char'] = batch['serp_query_text_url'].apply(
        lambda x: '�' in x)
    return batch


def is_email(batch: DataFrame) -> DataFrame:
    pattern = compile(r"[a-z0-9\w-]+@[a-z0-9\w-]+\.[a-z\.]+")
    batch['is-email'] = batch['serp_query_text_url'].apply(
        lambda x: bool(findall(pattern, x)))
    return batch


def extract_queries(batch: DataFrame, queries: Iterable[str]) -> DataFrame:
    return batch[batch['serp_query_text_url'].isin(queries)]


def transform_timestamp(batch: DataFrame, time_mode: str) -> DataFrame:
    assert time_mode in [
        "yearly", "monthly", "weekly"], f"Invalid time mode \"{time_mode}\". Choose from [yearly, monthly, weekly]"
    # transform timestamp to year, month or week if serp_timestamp is not null
    # batch = batch[~batch['serp_timestamp'].isnull()]
    if time_mode == "yearly":  # get year from timestamp
        batch['time'] = batch['serp_timestamp'].apply(
            lambda x: datetime.fromtimestamp(x).year)
    elif time_mode == "monthly":  # get year and month from timestamp
        batch['time'] = batch['serp_timestamp'].apply(
            lambda x: datetime.fromtimestamp(x).strftime('%Y-%m'))
    elif time_mode == "weekly":  # get year and week from timestamp
        batch['time'] = batch['serp_timestamp'].apply(
            lambda x: datetime.fromtimestamp(x).strftime('%Y-%W'))
    return batch


# Flat mapping functions
def _extract_chars(row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    return [{"char": char} for char in row['serp_query_text_url'].replace(" ", "")]


def _extract_operators(row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    res = [{"operator": operator} for operator in ["site:", "filetype:", "intitle:", "allinurl:", "allintitle:",
                                                   "intext:", "allintext:", "related:", "define:", "cache:", "around(", " OR ", " AND "] if operator in row['serp_query_text_url']]
    return res


def is_url(row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    pattern = compile(
        r"(https?://)?[a-z0-9\w-]+\.[a-z\.]+")
    return [{"year": datetime.fromtimestamp(row['serp_timestamp']).year, "is-url": bool(findall(pattern, row['serp_query_text_url']))}]


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


aggregate_word_counts = AggregateFn(
    init=lambda word_counts: {"word": [], "count()": []},
    accumulate_row=lambda counts, row: acc_word_counts(counts, row),
    merge=lambda counts1, counts2: merge_word_counts(counts1, counts2),
    # finalize=from_pandas,
    name="word-counts"
)


# Helper functions for aggregation
def acc_row(aggr_dict: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    aggr_dict["sum"] += row["count()"]
    aggr_dict["unique"] += 1
    return aggr_dict


def sum_dict(a1: Dict[str, Any], a2: Dict[str, Any]) -> Dict[str, Any]:
    return {k: int(a1.get(k, 0) + a2.get(k, 0)) for k in set(a1) | set(a2)}


def acc_word_counts(aggr_frame: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    if row['word'] in aggr_frame['word']:
        aggr_frame['count()'][aggr_frame['word'] ==
                              row['word']] += row['count()']
    else:
        aggr_frame['word'].append(row['word'])
        aggr_frame['count()'].append(row['count()'])
    return aggr_frame


def merge_word_counts(df1: Dict[str, Any], df2: Dict[str, Any]) -> Dict[str, Any]:

    intersec = list(set(df1['word']).intersection(set(df2['word'])))

    if intersec:
        for word in intersec:
            df1['count()'][df1['word'] == word] += df2['count()'][df2['word'] == word]
            del df2['word'][df2['word'] == word]
            del df2['count()'][df2['word'] == word]
        df1['word'].extend(df2['word'])
        df1['count()'].extend(df2['count()'])
    else:
        df1['word'].extend(df2['word'])
        df1['count()'].extend(df2['count()'])

    return df1


# Group-by function
def groupby(dataset: Dataset, col: str) -> GroupedData:
    return dataset.groupby(key=col)


def groupby_count(dataset: Dataset, col: str) -> Dataset:
    return dataset.groupby(key=col).count()


def groupby_count_sort(dataset: Dataset, col_group: str, col_sort: str) -> Dataset:
    return dataset.groupby(key=col_group).count().sort(key=col_sort, descending=True)


###########################################    Get task-specific modules     #########################################
def _get_module_specifics(analysis_name: AnalysisName, struc_level: Optional[int]) -> Dict[str, Any]:

    # Basic modules: clean data, simple transformations, filters, debug etc.
    if analysis_name == "clean-query-log":
        return {'groupby_func': None, 'aggregator': None, 'mapping_func': [filter_aql, partial(filter_by_char, char='�'), filter_empty_queries], 'flat_mapping_func': None, 'col_filter': None}
    elif analysis_name == "debug":
        return {'groupby_func': None, 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': None}
    elif analysis_name == "filter-aql-outlier":
        return {'groupby_func': None, 'aggregator': None, 'mapping_func': [filter_aql], 'flat_mapping_func': None, 'col_filter': None}
    elif analysis_name == "get-too-short-queries":
        return {'groupby_func': partial(groupby_count_sort, col_group=['serp_query_text_url', 'character-count'], col_sort='count()'), 'aggregator': None, 'mapping_func': [get_short_queries], 'flat_mapping_func': None, 'col_filter': None}
    elif analysis_name == "sort-grouped-data":
        return {'groupby_func': partial(groupby_count_sort, col_group='serp_query_text_url', col_sort='count()'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': None}
    elif analysis_name == "filter-google-queries":
        return {'groupby_func': None, 'aggregator': None, 'mapping_func': [partial(filter_by_col_value, col='search_provider_name', value='google')], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url', 'serp_timestamp', 'search_provider_name']}
    elif analysis_name == "transform-timestamps":
        return {'groupby_func': None, 'aggregator': None, 'mapping_func': [partial(transform_timestamp, time_mode='weekly')], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url', 'serp_timestamp']}
    elif analysis_name == "get-query-frequency":
        return {'groupby_func': partial(groupby_count_sort, col_group='serp_query_text_url', col_sort='count()'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "sum-rows":
        return {'groupby_func': None, 'aggregator': sum_rows, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': None}
    elif analysis_name == "deduplicate-queries":
        return {'groupby_func': partial(groupby, col='serp_query_text_url'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': None, 'map_groups_func': lambda g: {"serp_query_text_url": [g['serp_query_text_url'][0]]}, 'col_filter': ['serp_query_text_url']}

    # Syntactical analysis: Get frequencies of linguistic elements and frequencies of their lengths
    elif analysis_name == "extract-chars":
        return {'groupby_func': partial(groupby_count_sort, col_group='char', col_sort='count()'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': _extract_chars, 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "extract-words":
        return {'groupby_func': partial(groupby_count_sort, col_group='word', col_sort='count()'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': SpacyWords(), 'col_filter': ['serp_query_text_url']}
    # words of aql were too large to be extracted in one go, hence a merge of the two halfes was necessary
    elif analysis_name == "extract-words-merge":
        return {'groupby_func': partial(groupby, col="word"), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': None, 'map_groups_func': lambda g: {"word": [str(g["word"][0])], "count()": np_array([np_sum(g["count()"])])}}
    elif analysis_name == "extract-named-entities":
        return {'groupby_func': partial(groupby_count_sort, col_group=['entity', 'entity-label'], col_sort='count()'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': SpacyGetEntities(), 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "get-lengths":
        assert struc_level is not None, "Structural level must be specified by \"--struc-level\" [queries, named-entities, words]"
        if struc_level == "queries":
            map_func = SpacyQueryLevelStructures()
        elif struc_level == "named-entities":
            map_func = SpacyEntityLevelStructures()
        elif struc_level == "words":
            map_func = partial(get_lengths, structural_level=struc_level)
        return {'groupby_func': None, 'aggregator': None, 'mapping_func': [map_func], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url'] if struc_level == "queries" else None}
    elif analysis_name == "character-count-frequencies":
        return {'groupby_func': partial(groupby_count, col='character-count'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': None}
    elif analysis_name == "word-count-frequencies":
        return {'groupby_func': partial(groupby_count, col='word-count'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': None}
    elif analysis_name == "entity-count-frequencies":
        return {'groupby_func': partial(groupby_count, col='entity-count'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': None}
    elif analysis_name == "query-frequencies":
        return {'groupby_func': partial(groupby_count_sort, col_group=['serp_query_text_url'], col_sort='count()'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "extract-search-operators":
        return {'groupby_func': partial(groupby_count, col='operator'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': _extract_operators, 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "search-operators-count":
        return {'groupby_func': partial(groupby_count, col='operator-count'), 'aggregator': None, 'mapping_func': [get_operator_count], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url']}

    # Inference and Embeddings: Predictions on queries, extraction of inference-based features, embeddings
    elif analysis_name == "query-intent":
        return {'groupby_func': partial(groupby_count, col='query-intent'), 'aggregator': None, 'mapping_func': [QueryIntentPredictor()], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "extract-gliner-pii":
        return {'groupby_func': partial(groupby_count, col=['entity', 'entity-label']), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': GlinerGetEntities(), 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "extract-presidio-pii":
        return {'groupby_func': None, 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': PresidioGetEntities(), 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "group-presidio-pii":
        return {'groupby_func': partial(groupby_count, col='entity-label'), 'aggregator': None, 'mapping_func': None, 'flat_mapping_func': None, 'col_filter': None}
    elif analysis_name == "get-embeddings":
        return {'groupby_func': None, 'aggregator': None, 'mapping_func': [SentenceEmbedder()], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url']}

    # Temporal-based analyses
    elif analysis_name == "query-chart-by-year":
        return {'groupby_func': partial(groupby_count_sort, col_group=['year', 'serp_query_text_url'], col_sort=['year', 'count()']), 'aggregator': None, 'mapping_func': [get_year], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url', 'serp_timestamp']}
    elif analysis_name == "get-annual-top-queries":
        return {'groupby_func': partial(groupby, col='year'),  'aggregator': None, 'mapping_func': None, 'flat_mapping_func': None, 'map_groups_func': lambda g: g.sort_values(by='count()', ascending=False).head(25), 'col_filter': None}
    elif analysis_name == "get-temporal-query-frequency":
        return {'groupby_func': partial(groupby_count, col=['time', 'serp_query_text_url']), 'aggregator': None, 'mapping_func': [partial(extract_queries, queries=['google', 'yahoo', 'weather', 'youtube', 'hotmail', 'facebook', 'gmail', 'news', 'you', 'ebay', 'amazon', 'games', 'free', 'twitter', 'translate', 'mp3', 'maps', 'msn', 'fb', 'mail', 'instagram', 'map', 'face', 'video', 'juegos', 'craigslist', 'facebook login', 'lyrics', 'game', 'traductor', 'you tube', 'as', 'whatsapp', 'videos', 'wikipedia', 'yahoo mail', 'myspace', 'tiempo', 'samsung', 'bbc', 'meteo', 'web', 'music', 'nokia', 'погода', 'clima', 'chat', 'sony', 'download', 'go', 'google translate', 'wiki', 'netflix', 'hot', 'dictionary', 'messenger', 'test', 'whatsapp web', 'satta', 'torrent', 'microsoft', 'radio', 'movies', 'вк', 'hi5', 'java', 'tv', 'jobs', 'mobile', 'halloween', 'iphone', 'linux', 'coronavirus', 'jogos', 'flash', 'world cup', 'apple', 'earth', 'ipl', 'dvd', 'corona', 'cricbuzz', 'web whatsapp', 'hp', 'my', 'wetter', 'nba', 'cheats', 'hotmail.com', 'friv', 'dr', 'fifa', 'trump', 'love', 'christmas', 'black friday', 'olympics', 'covid', 'walmart', 'www']), partial(transform_timestamp, time_mode='monthly')], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url', 'serp_timestamp']}
    elif analysis_name == "get-monthly-google-queries":
        return {'groupby_func': partial(groupby_count, col=['time']), 'aggregator': None, 'mapping_func': [partial(transform_timestamp, time_mode='monthly')], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url', 'serp_timestamp']}

    # Analyses motivated after inspecting result data
    elif analysis_name == "get-temporal-url-proportion":
        return {'groupby_func': partial(groupby_count, col=['year', 'is-url']), 'aggregator': None, 'mapping_func': [filter_empty_timestamps], 'flat_mapping_func': is_url, 'col_filter': ['serp_query_text_url', 'serp_timestamp']}
    elif analysis_name == "get-email-proportion":
        return {'groupby_func': partial(groupby_count, col=['is-email']), 'aggregator': None, 'mapping_func': [is_email], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url']}
    elif analysis_name == "get-repl-char-proportion":
        return {'groupby_func': partial(groupby_count, col=['is-repl-char', 'year']), 'aggregator': None, 'mapping_func': [partial(filter_by_year, year=[1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]), get_repl_char], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url', 'serp_timestamp']}
    elif analysis_name == "aql-get-words-2006":
        return {'groupby_func': None, 'aggregator': None, 'mapping_func': [partial(filter_by_year, year=[2006, 2005, 2004, 2003, 2002, 2001, 2000]), partial(filter_by_char, char='�')], 'flat_mapping_func': SpacyWords(), 'col_filter': ['serp_query_text_url', 'serp_timestamp']}
    elif analysis_name == "aql-anomaly":
        return {'groupby_func': partial(groupby_count_sort, col_group=['serp_query_text_url', 'character-count'], col_sort='count()'), 'aggregator': None, 'mapping_func': [filter_lengths], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url', 'character-count']}
    elif analysis_name == "filter-by-year-clean-repl-char":
        return {'groupby_func': partial(groupby_count_sort, col_group=['serp_query_text_url', 'serp_url'], col_sort='count()'), 'aggregator': None, 'mapping_func': [partial(filter_by_year, year=[2006]), partial(filter_by_char, char='�')], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url', 'serp_timestamp', 'serp_url']}


############################################    Pipeline    ################################################
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

    # take random sample of dataset
    if analysis_name == "get-embeddings":
        if dataset_name == "aol":  # Size: 10 M
            ds = ds.random_sample(fraction=0.1, seed=42)
        elif dataset_name == "aql":  # Size: 62,4 M
            ds = ds.random_sample(fraction=1/62.4, seed=42)
        elif dataset_name == "ms-marco":  # Size: 10 M
            ds = ds.random_sample(fraction=0.1, seed=42)
        elif dataset_name == "orcas":  # Size: 10 M
            ds = ds.random_sample(fraction=0.1, seed=42)

    # Apply mapping function.
    if module_specifics['mapping_func'] is not None:
        # iterate through list of mapping functions
        if type(module_specifics['mapping_func']) is list:
            for func in module_specifics['mapping_func']:
                ds = map_dataset(dataset=ds, mapping_func=func,
                                 concurrency=concurrency, batch_size=batch_size, num_gpus=num_gpus, num_cpus=num_cpus, memory_scaler=memory_scaler)
        else:
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

    # Map groups.
    if 'map_groups_func' in module_specifics.keys() and module_specifics['map_groups_func'] is not None:
        ds = map_groups(dataset=ds, map_group_func=module_specifics['map_groups_func'],
                        concurrency=concurrency, memory_scaler=memory_scaler)

    # Apply aggregation function.
    if module_specifics['aggregator'] is not None:
        ds = aggregate_dataset(
            dataset=ds, aggregation_func=module_specifics['aggregator'])

    # Print results for debugging.
    # if type(ds) is Dataset:
    #     print(ds.take(5))
    #     print(ds.columns())
    #     print(f"\n\n\n\n\n{ds.count()}\n\n\n\n\n")
    # elif type(ds) is dict:
    #     print(ds)

    # Get sample of results.
    # ds = ds.take(50)

    # print number of rows
    # print(ds.count())
    # print(ds.columns())
    # print(ds.take(1))

    # Write results.
    write_dataset(dataset=ds, write_dir=write_dir,
                  analysis_name=analysis_name, write_concurrency=write_concurrency, struc_level=struc_level, dataset_name=dataset_name, sample_files=sample_files, which_half=which_half, read_dir=read_dir, only_english=only_english)
