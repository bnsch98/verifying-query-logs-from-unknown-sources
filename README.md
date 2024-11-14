# thesis-schneg
This thesis aims at verfiying web search query logs from unknown sources by comparing them to trustworthy query logs.

This repository contains the python code that is required to replicate the experiments, an instruction on how to set up a working environment to run the code as well as the thesis itself.

Overleaf-Project: https://de.overleaf.com/read/sqcbzpjgyjpb#eb0738

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

<detail>
<summary>

Available Input Parameters and their corresponding parameter values

</summary>

| Identifier | Input-Parameter | Input-Values | Description |
| :--- | :---: | :--- | ---: | :---: | ---: | :--- |
|Analysis mode|`--classify`, `--aggregate`|❌|Determines which analysis mode should be performed on the data|
|Data Set|`--dataset`|`aql`,`aol`,`ms-marco`,`orcas`|Determines the data set on which the anaylsis is performed|
|Analysis mode|`classify`, `aggregate`|Determines which analysis mode should be performed on the data|
|Analysis mode|`classify`, `aggregate`|Determines which analysis mode should be performed on the data|


