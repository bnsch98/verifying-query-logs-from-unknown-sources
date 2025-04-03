from typing import final
from nltk import sent_tokenize, word_tokenize, pos_tag
from typing import Optional
from nltk.downloader import Downloader
from urllib.error import URLError
from urllib.request import urlopen
from abc import abstractmethod, ABC

from functools import partial
from dataclasses import dataclass
from functools import cached_property
from json import dumps
from thesis_schneg.model import DatasetName, AnalysisName
from ray.data.grouped_data import GroupedData
from ray.data.aggregate import AggregateFn
from ray.data import read_parquet, Dataset
from ray import init
from pandas import DataFrame
from typing import Iterable, Optional, Callable, Protocol, Union, Any, Dict
from random import choices
from pathlib import Path


def has_internet_connection(host: str = "https://webis.de") -> bool:
    try:
        urlopen(host)
        return True
    except URLError:
        return False


def _load_downloader() -> Optional[Downloader]:
    downloader = Downloader()
    if has_internet_connection(downloader.url):
        return downloader
    else:
        return None


_DOWNLOADER: Optional[Downloader] = _load_downloader()


def download_nltk_dependencies(*dependencies: str):
    if _DOWNLOADER is None:
        print("NLTK Downloader is not available. Cannot download dependencies.")
        return
    for dependency in dependencies:
        if not _DOWNLOADER.is_installed(dependency):
            _DOWNLOADER.download(dependency)
            print(f"Downloaded NLTK dependency: {dependency}")


class QuestionClassifier(ABC):
    @abstractmethod
    def is_question(self, question: str) -> bool:
        pass


class ClosedQuestionClassifier(ABC):
    @abstractmethod
    def is_closed_question(self, question: str) -> bool:
        pass


INTERROGATIVE_DETERMINERS = ["which", "what", "whose"]
INTERROGATIVE_PRONOUNS = ["who", "whom", "whose", "what", "which"]
INTERROGATIVE_ADVERBS = ["where", "when", "how", "why", "whether"]
EMPATHIC_SUFFIXES = ["ever", "so", "soever"]
AUXILIARY_VERBS = [
    # be
    "am", "are", "aren't", "is", "isn't", "was", "wasn't", "were", "weren't",
    # do
    "do", "don't", "does", "doesn't", "did", "didn't",
    # can
    "can", "can't",
    # could
    "could", "couldn't",
    # have
    "have", "haven't", "has", "hasn't", "had", "hadn't",
    # may
    "may",
    # might
    "might",
    # must
    "must", "mustn't",
    # need
    "need", "needn't",
    # shall
    "shall", "shan't",
    # should
    "should", "shouldn't",
    # will
    "will", "won't",
    # would
    "would", "wouldn't",
]
HELPER_WORDS = ["in", "for", "at", "on", "after", "as", "with", "if"]


def _is_interrogative_word(word: str) -> bool:
    # Remove empathic suffixes.
    for suffix in EMPATHIC_SUFFIXES:
        word = word.removesuffix(suffix)

    if word in INTERROGATIVE_DETERMINERS:
        return True
    elif word in INTERROGATIVE_PRONOUNS:
        return True
    elif word in INTERROGATIVE_ADVERBS:
        return True
    else:
        return False


def _starts_with_interrogative_word(sentence: str) -> bool:
    sentence = sentence.lower()
    words = word_tokenize(sentence, preserve_line=True)
    if len(words) == 0:
        return False

    # Skip helping phrase.
    if words[0] in HELPER_WORDS and "," in sentence:
        return _starts_with_interrogative_word(sentence.split(",", 1)[1])

    # Remove helper words that might come before the interrogative word.
    if words[0] in HELPER_WORDS:
        words = words[1:]

    if len(words) == 0:
        return False
    return _is_interrogative_word(words[0])


def _ends_with_question_mark(sentence: str) -> bool:
    return sentence.endswith("?")


def _contains_verb(sentence: str) -> bool:
    sentence = sentence.lower()
    words = word_tokenize(sentence, preserve_line=True)
    parts_of_speech = pos_tag(words, tagset="universal")
    for word, pos in parts_of_speech:
        if word in AUXILIARY_VERBS:
            return True
        elif pos == "VERB":
            return True
    return False


def _starts_with_auxiliary_verb(sentence: str) -> bool:
    sentence = sentence.lower()
    words = word_tokenize(sentence, preserve_line=True)
    if len(words) == 0:
        return False

    # Skip helping phrase.
    if words[0] in HELPER_WORDS and "," in sentence:
        return _starts_with_auxiliary_verb(sentence.split(",", 1)[1])

    return words[0] in AUXILIARY_VERBS


def _is_question(sentence: str) -> bool:
    if not _contains_verb(sentence):
        return False
    elif _starts_with_interrogative_word(sentence):
        return True
    elif not _ends_with_question_mark(sentence):
        return False
    elif _starts_with_auxiliary_verb(sentence):
        return True
    else:
        return False


def _is_closed_question(sentence: str) -> bool:
    if not _ends_with_question_mark(sentence):
        return False
    elif not _contains_verb(sentence):
        return False
    elif _starts_with_interrogative_word(sentence):
        return False
    elif _starts_with_auxiliary_verb(sentence):
        return True
    else:
        return False


@final
class RuleBasedQuestionClassifier(
    QuestionClassifier,
    ClosedQuestionClassifier,
):
    def __init__(self):
        download_nltk_dependencies(
            "punkt",
            "averaged_perceptron_tagger",
            "universal_tagset",
        )

    def is_question(self, question: str) -> bool:
        sentences = [question, *sent_tokenize(question)]
        return any(
            _is_question(sentence)
            for sentence in sentences
        )

    def is_closed_question(self, question: str) -> bool:
        sentences = [question, *sent_tokenize(question)]
        return any(
            _is_closed_question(sentence)
            for sentence in sentences
        )

    def __repr__(self) -> str:
        return RuleBasedQuestionClassifier.__name__


class _Question_Predictor(Protocol):
    def is_question(self, batch: DataFrame) -> DataFrame:
        raise NotImplementedError()

    def __call__(self, batch: DataFrame) -> DataFrame:
        return self.is_question(batch)


@dataclass(frozen=True)
class Question_Predictor(_Question_Predictor):

    @cached_property
    def model(self) -> RuleBasedQuestionClassifier:
        return RuleBasedQuestionClassifier()

    def is_question(self, batch: DataFrame) -> DataFrame:
        batch['is-queston'] = [self.model.is_question(query)
                               for query in list(batch['serp_query_text_url'])]
        return batch


def _get_parquet_paths(
    dataset_name: DatasetName,
    analysis_name: AnalysisName,
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


def aggregate_dataset(dataset: Dataset, aggregation_func: AggregateFn, concurrency: Optional[int] = None) -> Optional[Dict[str, Any]]:
    return dataset.aggregate(aggregation_func, concurrency=concurrency)


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


def groupby_count(dataset: Dataset, col: str) -> Dataset:
    return dataset.groupby(key=col).count()

###########################################    Get task-specific modules     #########################################


def _get_module_specifics(analysis_name: AnalysisName, struc_level: Optional[int]) -> Dict[str, Any]:

    if analysis_name == "questions":
        return {'groupby_func': partial(groupby_count, col='is-question'), 'aggregator': None, 'mapping_func': [Question_Predictor()], 'flat_mapping_func': None, 'col_filter': ['serp_query_text_url']}


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
            dataset=ds, aggregation_func=module_specifics['aggregator'], concurrency=concurrency)

    # Print results for debugging.
    # if type(ds) is Dataset:
    #     print(ds.take(50))
    #     print(ds.columns())
    # elif type(ds) is dict:
    #     print(ds)

    # # Write results.
    write_dataset(dataset=ds, write_dir=write_dir,
                  analysis_name=analysis_name, write_concurrency=write_concurrency, struc_level=struc_level, dataset_name=dataset_name, sample_files=sample_files, which_half=which_half, read_dir=read_dir, only_english=only_english)
