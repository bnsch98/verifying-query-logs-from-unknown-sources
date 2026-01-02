# Verifying Query Logs from Unknown Sources

Master's thesis by Benjamin Schneg

<!-- TOC -->
  * [Installation](#installation)
  * [Structure](#structure)
  * [Usage](#usage)
<!-- TOC -->

## Installation

### Python Installation

Follow these steps to set up the project's working environment:

1. Connect to the Webis [VPN](https://webis.de/facilities.html#?q=vpn) and mount the [Ceph FS](https://faq.webis.de#how-to-use-ceph-cephfs-s3) at `/mnt/ceph/storage`. [Contact staff](https://webis.de/people.html) if you need assistance.
2. Install [LaTeX](https://www.tug.org/texlive/)
3. Install [Python 3.10](https://www.python.org/downloads/release/python-31012/)
4. Navigate to the `src` directory: 
    ```shell script
    cd src/
    ```
5. Create and activate a virtual environment:
    ```shell script
    python3 -m venv venv/
    source venv/bin/activate
    ```
6. Install project dependencies:
    ```shell
    pip install -e .
    ```

## Structure
This repository is structured as follows:
| Folder                     | Purpose                                     |
|----------------------------|---------------------------------------------|
| [`literature`](literature) | Literature collection/review.               |
| [`plots`](plots)           | Plot collection.                            |
| [`src`](src)               | Experimentation and evaluation source code. |
| [`thesis`](thesis)         | The Master's thesis.                        |
| [`utility`](utility)       | Configuration utilities.                    |
| others                     | SDK & build configuration.                  |

## Usage
The source code is executable via a CLI.
### CLI

Usage of the CLI:

```shell script
python -m thesis_schneg --<command> <command-value>
```
Run experiments on a [Ray Cluster](https://docs.ray.io/en/latest/): 
```shell script
ray job submit --runtime-env ray-runtime-env.yml --no-wait -- python -m thesis_schneg --<command> <command-value>
```

### CLI Commands

| Identifier     | `<command>`                            | `<command-value>`                   | Description                                                                             |
|----------------|---------------------------------------------|------------------------------------|-----------------------------------------------------------------------------------------| 
|Analysis mode. Most Analyses are invoked by this command.  |`--analysis`                                 |[`<analysis-type>`](#analysis-modes)|Specify a desired analysis. Get an overview from the provided table. Find the script containing the analysis [here](src/thesis_schneg/analysis.py).|
|Analysis mode: Questions   |`--questions`                                |`questions`                         |Carry out question classification. |
|Analysis mode: Presidio PII Extraction.   |`--presidio_analysis`                        |`extract-presidio-pii`              |Carry out PII Entity Extraction. |
|Dataset         |`--dataset`                                  |`aql`,`aol`,`ms-marco`,`orcas`      |Specify the data set to be analyzed |
|Concurrency     |`--concurrency`                              |`<int>`                             |Set the concurrency to transform data.|
|Read concurrency|`--read-concurrency`                         |`<int>`                             |Set the concurrency to read data. |
|Write concurrency|`--write-concurrency`                       |`<int>`                             |Set the concurrency to write data.  |
|Batch size      |`--batch-size`                               |`<int>`                             |Set the batch size to transform data.|
|Memory scaler   |`--memory-scaler`                            |`<float>`                           |Minimum number of available memory in GB for involved nodes.   |
|Sample files    |`--sample-files`                             |`<int>`                             |Number of sample files. |
|Num CPUs        |`--num-cpus`                                 |`<float>`                           |Number of CPUs per node. |
|Num GPUs        |`--num-gpus`                                 |`<float>`                           |Number of GPUs per node. |
|Struc Level     |`--struc-level`                              |`queries`, `named-entities`, `words`|Only relevant for the anaysis "`get-lengths`". Provides the structural level on which the lengths should be determined. E.g. for named entities we can measure the length in words or in characters. |
|Write Directory |`--write-dir`                                |`<path>`                            |Specify where to write the results.   |
|Read Directory |`--read-dir`                                 |`<path>`                            |Specify path to read data.   |
| Consider english subsets |`--only-english`                   | `None`                            | Option to consider only english subsets for an analysis.   |






### Analysis Modes

#### Structure-related Analysis
| `<analysis-type>`                                         | Description                                                                                                                    |
|-----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `extract-named-entities`                                  | Extract and group named entities.                                                                                              |
| `extract-words`                                           | Extract and group words.                                                                                                       | 
| `extract-chars`                                           | Extract and group characters.                                                                                                  |
| `extract-search-operators`                                | Extract and group search operators like `site:`, `filetype:` etc.                                                              | 
| `get-lengths`                                             | Measure the length of queries, named entities or words.                                                                        |
| `search-operators-count`                                  | Count the number of search operators per query and group by their count.                                                       | 
| `character-count-frequencies`                             | Group data by the character count and count their frequency.                                                                   | 
| `word-count-frequencies`                                  | Group data by the word count and count their frequency.                                                                        | 
| `entity-count-frequency`                                  | Group data by the entity count and count their frequency.                                                                      | 


#### Inference-based Analysis
| `<analysis-type>`                                         | Description                                                                                                                    |
|-----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `query-intent`                                            | Classify queries with regard to their intent acoording to [Alexander, Kusa, de Vries](https://dl.acm.org/doi/10.1145/3477495.3531737).| 
| `get-embeddings`                                  | Get sentence embeddings from [Multilingual Sentence Transformer for Symmetric Semantic Search](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2).                                                                                              |
| `group-presidio-pii`                                           | Group Data according to PII Entity Labels.

#### Temporal-based Analysis
| `<analysis-type>`                                         | Description                                                                                                                    |
|-----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `query-chart-by-year`                                  | Group and count queries so obtain the frequency per year.                                                                                              |
| `get-annual-top-queries`                                           | Extract top 25 queries per year.                                                                                                       | 
| `get-temporal-query-frequency`                                           | Retrieve frequency of top monthly google queries in the AQL. Google queries from the AQL are stored in `/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis/aql-filter-google-queries-all`.                                                                                                 |
| `get-monthly-google-queries`                                | Get monthly frequencies of google queries in the AQL. Google queries from the AQL are stored in `/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/analysis_data/analysis/aql-filter-google-queries-all`. | 
