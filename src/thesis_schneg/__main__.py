from typing import Optional
from pathlib import Path
from cyclopts import App
from cyclopts.types import Directory
from thesis_schneg.model import (
    DatasetName,
    AnalysisName,
)

app = App()


@app.command
def analyser(
    dataset: DatasetName,
    analysis: AnalysisName,
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
    read_dir: Optional[Directory] = None
) -> None:
    from thesis_schneg.analysis import analysis_pipeline as _analysis_pipeline

    _analysis_pipeline(
        dataset_name=dataset,
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
    dataset: DatasetName,
    analysis: AnalysisName,
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
    read_dir: Optional[Directory] = None
) -> None:
    from thesis_schneg.presidio_analyse import analysis_pipeline as _presidio_analysis_pipeline

    _presidio_analysis_pipeline(
        dataset_name=dataset,
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
    dataset: DatasetName,
    analysis: AnalysisName,
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
    read_dir: Optional[Directory] = None
) -> None:
    from thesis_schneg.question_classifier import analysis_pipeline as question_pipeline

    question_pipeline(
        dataset_name=dataset,
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


if __name__ == "__main__":
    app()
