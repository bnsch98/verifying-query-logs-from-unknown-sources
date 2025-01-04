from typing import Optional
from pathlib import Path
from cyclopts import App
from cyclopts.types import Directory
from thesis_schneg.model import (
    DatasetName,
    AggregatorName,
    AnalysisName,
)

app = App()


@app.command
def aggregate(
    aggregator: AggregatorName,
    dataset: DatasetName,
    sample_files: Optional[int] = None,
    only_english: bool = False,
    read_concurrency: Optional[int] = None,
    aggregate_concurrency: Optional[int] = None,
    write_results: bool = False,
    write_concurrency: Optional[int] = 2,
    write_dir: Directory = Path(
        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/aggregation"),
) -> None:
    from thesis_schneg.aggregate import aggregate as _aggregate

    _aggregate(
        aggregator_name=aggregator,
        dataset_name=dataset,
        sample_files=sample_files,
        only_english=only_english,
        read_concurrency=read_concurrency,
        aggregate_concurrency=aggregate_concurrency,
        write_results=write_results,
        write_dir=write_dir,
        write_concurrency=write_concurrency,
    )


@app.command
def analyser(
    dataset: DatasetName,
    analysis: AnalysisName,
    struc_level: Optional[str] = None,
    sample_files: Optional[int] = None,
    only_english: bool = False,
    read_concurrency: Optional[int] = None,
    map_concurrency: Optional[int] = None,
    batch_size: int = 16,
    memory_scaler: float = 1.0,
    num_cpus: Optional[float] = None,
    num_gpus: Optional[float] = None,
    write_concurrency: Optional[int] = 2,
    write_dir: Directory = Path(
        "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis"),
) -> None:
    from thesis_schneg.analysis import analysis_pipeline as _analysis_pipeline

    _analysis_pipeline(
        dataset_name=dataset,
        analysis_name=analysis,
        struc_level=struc_level,
        sample_files=sample_files,
        only_english=only_english,
        read_concurrency=read_concurrency,
        map_concurrency=map_concurrency,
        batch_size=batch_size,
        memory_scaler=memory_scaler,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        write_dir=write_dir,
        write_concurrency=write_concurrency,
    )


@app.command
def visualize(
    analysis: AnalysisName,
    dataset: DatasetName = None,
    save_vis: bool = False,

) -> None:
    from thesis_schneg.visualize import visualize as _visualize

    _visualize(
        dataset_name=dataset,
        analysis_name=analysis,
        save_vis=save_vis,
    )


if __name__ == "__main__":
    app()
