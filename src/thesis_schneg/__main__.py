from typing import Optional

from cyclopts import App

from thesis_schneg.model import (
    DatasetName,
    PredictorName,
    AggregatorName,
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
    write_results: bool = False,
    write_concurrency: Optional[int] = None,
) -> None:
    from thesis_schneg.classification import classify as _classify

    _classify(
        predictor_name=predictor,
        dataset_name=dataset,
        sample_files=sample_files,
        only_english=only_english,
        read_concurrency=read_concurrency,
        predict_concurrency=predict_concurrency,
        write_results=write_results,
        write_concurrency=write_concurrency,
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
    from thesis_schneg.aggregate import aggregate as _aggregate

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


if __name__ == "__main__":
    app()
