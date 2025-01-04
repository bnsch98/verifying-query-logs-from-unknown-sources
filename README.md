# thesis-schneg

This thesis aims at verifying web search query logs from unknown sources by comparing them to trustworthy query logs.

This repository contains the python code that is required to replicate the experiments, an instruction on how to set up a working environment to run the code as well as the thesis itself.

## Installation

### Python Installation

Follow these steps to set up the Python environment that is required for the provided code:

1. Install [Python 3.11](https://www.python.org/downloads/release/python-3110/)
2. Create and activate a virtual environment:

    ```shell
    python3.11 -m venv venv/
    source venv/bin/activate
    ```

3. Install project dependencies:

    ```shell
    pip install -e .
    ```

## Usage

### CLI

Run the CLI with:

```shell
ray job submit --runtime-env ray-runtime-env.yml --no-wait -- python -m thesis_schneg --<Input-Parameter> --<Parameter-Value>
```

Available Input Parameters and their corresponding parameter values

| Identifier | Input-Parameter | Input-Values | Description |
| :--- | :---: | :--- | ---: | :---: | ---: | :--- |
|Analysis mode|`--classify`, `--aggregate`|❌|Determines which analysis mode should be performed on the data|
|Dataset|`--dataset`|`aql`,`aol`,`ms-marco`,`orcas`|Determines the data set on which the analysis is performed|
|Analysis mode|`classify`, `aggregate`|Determines which analysis mode should be performed on the data|
|Analysis mode|`classify`, `aggregate`|Determines which analysis mode should be performed on the data|


### Classify queries

Classify the queries with multiple pre-trained text classifiers, and store the output label as a Parquet file.

Query intent:

```shell
ray job submit --no-wait --runtime-env ray-runtime-env.yml -- python -m thesis_schneg classify --dataset aql --predictor query-intent --predict-batch-size 32 --sample-files 1
```
