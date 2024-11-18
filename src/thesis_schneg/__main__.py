from typing import Optional, Dict, Any, Iterable, Callable
from ray.data.aggregate import AggregateFn
from pathlib import Path
from cyclopts import App
from cyclopts.types import ResolvedExistingDirectory
from pandas import DataFrame
from thesis_schneg.classification import (
    DatasetName,
    PredictorName,
    classify as _classify,
)

from thesis_schneg.aggregate import (
    DatasetName,
    AggregatorName,
    aggregate as _aggregate,
)

from thesis_schneg.prototype import (
    DatasetName,
    AnalysisName,
    analysis_pipeline as _analysis_pipeline,
)

app = App()


@app.command
def classify(
    predictor: PredictorName,
    dataset: DatasetName,
    sample_files: Optional[int] = None,
    only_english: bool = False,
    read_concurrency: Optional[int] = None,
    predict_concurrency: Optional[int] = None,
    write_concurrency: Optional[int] = None,
    write_dir: ResolvedExistingDirectory = Path(
        f"/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/classification/{DatasetName}_{PredictorName}"),
) -> None:
    _classify(
        predictor_name=predictor,
        dataset_name=dataset,
        sample_files=sample_files,
        only_english=only_english,
        read_concurrency=read_concurrency,
        predict_concurrency=predict_concurrency,
        write_concurrency=write_concurrency,
        write_dir=write_dir,
    )


@app.command
def aggregate(
    aggregator: AggregatorName,
    dataset: DatasetName,
    sample_files: Optional[int] = None,
    only_english: bool = False,
    read_concurrency: Optional[int] = None,
    aggregate_concurrency: Optional[int] = None,
    write_results: bool = False,
    # write_concurrency: Optional[int] = None,
) -> None:
    _aggregate(
        aggregator_name=aggregator,
        dataset_name=dataset,
        sample_files=sample_files,
        only_english=only_english,
        read_concurrency=read_concurrency,
        aggregate_concurrency=aggregate_concurrency,
        write_results=write_results,
        # write_concurrency=write_concurrency,
    )


@app.command
def analyser(
    dataset: DatasetName,
    analysis: AnalysisName,
    sample_files: Optional[int] = None,
    only_english: bool = False,
    read_concurrency: Optional[int] = None,
    map_concurrency: Optional[int] = None,
    mapping_batch_size: int = 16,
    flatmap_concurrency: Optional[int] = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
    write_dir: ResolvedExistingDirectory = Path(
        f"/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/classification/{DatasetName}-{AnalysisName}"),
) -> None:
    _analysis_pipeline(
        dataset_name=dataset,
        analysis_name=analysis,
        sample_files=sample_files,
        only_english=only_english,
        read_concurrency=read_concurrency,
        map_concurrency=map_concurrency,
        mapping_batch_size=mapping_batch_size,
        flatmap_concurrency=flatmap_concurrency,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        write_dir=write_dir,
    )


if __name__ == "__main__":
    app()
