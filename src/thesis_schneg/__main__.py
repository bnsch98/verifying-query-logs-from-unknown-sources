from typing import Optional

from cyclopts import App

from thesis_schneg.classification import (
    DatasetName,
    PredictorName,
    classify as _classify,
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


if __name__ == "__main__":
    app()
