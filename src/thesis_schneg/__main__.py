from typing import Optional, Iterable, Literal
from pathlib import Path
from cyclopts import App
from cyclopts.types import Directory
from thesis_schneg.model import (
    DatasetName,
    ThesisAnalysisName,
    PostThesisAnalysisName,
    EmbeddingsAnalysisName,
    OTSolverVariant,
)

app = App()


@app.command
def analyser(
    dataset: Iterable[DatasetName],
    analysis: ThesisAnalysisName,
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
    write_concurrency: Optional[int] = 2,
    write_dir: Directory = Path(
        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis"),
    read_dir: Optional[Iterable[str]] = None
) -> None:
    from thesis_schneg.analysis import analysis_pipeline as _analysis_pipeline

    _analysis_pipeline(
        dataset=dataset,
        analysis_name=analysis,
        struc_level=struc_level,
        sample_files=sample_files,
        only_english=only_english,
        which_half=which_half,
        read_concurrency=read_concurrency,
        concurrency=concurrency,
        batch_size=batch_size,
        memory_scaler=memory_scaler,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        write_dir=write_dir,
        write_concurrency=write_concurrency,
        read_dir=read_dir,
    )


@app.command
def presidio_analysis(
    dataset: Iterable[DatasetName],
    analysis: ThesisAnalysisName,
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
    write_concurrency: Optional[int] = 2,
    write_dir: Directory = Path(
        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis"),
    read_dir: Optional[Iterable[str]] = None
) -> None:
    from thesis_schneg.presidio_analyse import analysis_pipeline as _presidio_analysis_pipeline

    _presidio_analysis_pipeline(
        dataset=dataset,
        analysis_name=analysis,
        struc_level=struc_level,
        sample_files=sample_files,
        only_english=only_english,
        which_half=which_half,
        read_concurrency=read_concurrency,
        concurrency=concurrency,
        batch_size=batch_size,
        memory_scaler=memory_scaler,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        write_dir=write_dir,
        write_concurrency=write_concurrency,
        read_dir=read_dir,
    )


@app.command
def questions(
    dataset: Iterable[DatasetName],
    analysis: ThesisAnalysisName,
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
    write_concurrency: Optional[int] = 2,
    write_dir: Directory = Path(
        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis"),
    read_dir: Optional[Iterable[str]] = None
) -> None:
    from thesis_schneg.question_classifier import analysis_pipeline as question_pipeline

    question_pipeline(
        dataset=dataset,
        analysis_name=analysis,
        struc_level=struc_level,
        sample_files=sample_files,
        only_english=only_english,
        which_half=which_half,
        read_concurrency=read_concurrency,
        concurrency=concurrency,
        batch_size=batch_size,
        memory_scaler=memory_scaler,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        write_dir=write_dir,
        write_concurrency=write_concurrency,
        read_dir=read_dir,
    )


@app.command
def annual_deduplication(
    dataset: Iterable[DatasetName],
    analysis: PostThesisAnalysisName,
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
    write_concurrency: Optional[int] = 2,
    write_dir: Directory = Path(
        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis"),
    read_dir: Optional[Iterable[str]] = None
) -> None:
    from thesis_schneg.post_thesis_analysis.annual_deduplicate import analysis_pipeline as annual_deduplication_pipeline

    annual_deduplication_pipeline(
        dataset=dataset,
        analysis_name=analysis,
        struc_level=struc_level,
        sample_files=sample_files,
        only_english=only_english,
        which_half=which_half,
        read_concurrency=read_concurrency,
        concurrency=concurrency,
        batch_size=batch_size,
        memory_scaler=memory_scaler,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        write_dir=write_dir,
        write_concurrency=write_concurrency,
        read_dir=read_dir,
    )


@app.command
def get_deduplication_results(analysis: ThesisAnalysisName, dataset: Iterable[DatasetName], write_results: bool = False) -> None:
    """
    Process results for a given analysis and dataset.
    If write_results is True, write the processed results to a file.
    """
    from thesis_schneg.post_thesis_analysis.get_duplication_results import process_results as _process_results

    _process_results(
        dataset_name=dataset,
        analysis_name=analysis,
        write_results=write_results,
    )


@app.command
def embeddings_analysis(
    datasets: Iterable[DatasetName],
    analysis: EmbeddingsAnalysisName,
    ot_variant: OTSolverVariant = "sliced-wasserstein",
    num_input_files: Optional[int] = 12,
    shuffle_files: bool = False,
    device_type: Literal["cpu", "gpu"] = "gpu",
    batch_size: int = 100,
) -> None:

    from thesis_schneg.post_thesis_analysis.embeddings_analysis import embeddings_analysis_pipeline as _embeddings_analysis_pipeline
    _embeddings_analysis_pipeline(
        datasets=datasets,
        analysis=analysis,
        ot_variant=ot_variant,
        num_input_files=num_input_files,
        shuffle_files=shuffle_files,
        device_type=device_type,
        batch_size=batch_size,
    )


# solely for testing purposes, not part of the main functionality
@app.command
def test_cli(
    arg1: Optional[Iterable[str]] = None,
    arg2: Optional[str] = None,
    arg3: Iterable[DatasetName] = None,
) -> None:

    print(f"arg1: {arg1}")
    print(f"type arg1: {type(arg1)}")
    print(f"arg2: {arg2}")
    print(f"type arg2: {type(arg2)}")
    print(f"arg3: {arg3}")
    print(f"type arg3: {type(arg3)}")

    if arg1:
        for item in arg1:
            print(f"Item from arg1: {item}")
    else:
        print("No items in arg1")
    if arg2:
        print(f"Value of arg2: {arg2}")
    else:
        print("No value for arg2")
    if arg3:
        print(f"Value of arg3: {arg3}")
        if isinstance(arg3, Iterable):
            print(f"arg3 is an iterable with {len(arg3)} items")
            for item in arg3:
                print(f"Item from arg3: {item}")
    else:
        print("No items in arg3")


if __name__ == "__main__":
    app()
