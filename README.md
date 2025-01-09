# Thesis Title

Master's thesis by Benjamin Schneg

<!-- TOC -->
  * [Installation](#installation)
  * [Structure](#structure)
  * [Usage](#usage)
<!-- TOC -->

## Installation

### Python Installation

Follow these steps to set up the environment to reproduce our experiments:

1. Connect to the Webis [VPN](https://webis.de/facilities.html#?q=vpn) and mount the [Ceph FS](https://faq.webis.de#how-to-use-ceph-cephfs-s3) at `/mnt/ceph/storage`. [Contact staff](https://webis.de/people.html) if you need assistance.
2. Install [LaTeX](https://www.tug.org/texlive/)
3. Install [Python 3.11](https://www.python.org/downloads/release/python-3110/)
4. Navigate to the `src` directory: 
    ```shell script
    cd src/
    ```
5. Create and activate a virtual environment:
    ```shell script
    python3.11 -m venv venv/
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

Run the CLI with:

```shell script
python -m thesis_schneg --<Input-Parameter> <Parameter-Value>
```
Run the CLI on a [Ray Cluster](https://docs.ray.io/en/latest/) with: 
```shell script
ray job submit --runtime-env ray-runtime-env.yml --no-wait -- python -m thesis_schneg --<Input-Parameter> <Parameter-Value>
```

Available Input Parameters and their corresponding parameter values

| Identifier     | Input-Parameter                             | Parameter-Values                   | Description                                                                             |
|----------------|---------------------------------------------|------------------------------------|-----------------------------------------------------------------------------------------| 
|Analysis mode   |`--analysis`                                 |[`<analysis-type>`](#analysis-table)|Specify the desired analysis. Get an overview from the provided table. Find the script containing the analysis [here](src/thesis_schneg/analysis.py).                  |
|Dataset         |`--dataset`                                  |`aql`,`aol`,`ms-marco`,`orcas`      |Specify the data set on which the analysis is performed.                                 |
|Concurrency     |`--concurrency`                              |`<int>`                             |Set the concurrency to transform data.                           |
|Read concurrency|`--read-concurrency`                         |`<int>`                             |Set the concurrency to read data.                                               |
|Write concurrency|`--write-concurrency`                       |`<int>`                             |Set the concurrency to write data.                                              |
|Batch size      |`--batch-size`                               |`<int>`                             |Set the batch size to transform data.                                               |
|Memory scaler   |`--memory-scaler`                            |`<float>`                           |Minimum number of memory in GB for involved nodes.                                               |
|Sample files    |`--sample-files`                             |`<int>`                             |Number of sample files.                                                |
|Num CPUs        |`--num-cpus`                                 |`<float>`                           |Number of CPUs per node.                                                |
|Num GPUs        |`--num-gpus`                                 |`<float>`                           |Number of GPUs per node.                                                |
|Struc Level     |`--struc-level`                              |`queries`, `named-entities`, `words`|Only relevant for the anaysis "`get-lengths`". Provides the structural level on which the lengths should be determinde. E.g. for named entities we can measure the length in words or in characters.                                                |
|Write Directory |`--write-dir`                                |`<path>`                            |Specify where to write the results.                                                |




## Analysis-Table

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
| `query-intent`                                            | Classify queries with regard to their intent acoording to [Alexander, Kusa, de Vries](https://dl.acm.org/doi/10.1145/3477495.3531737)| 
| `query-quality`                                           | Classify queries in terms of their grammatical quality                                                                         | 
| `query-domain`                                            | Classify queries into a domain taxonomy                                                                                        | 
| `query-nsfw`                                              | Determine if the query is *safe for work* or *nots safe for work*, e.g. if it uses inappropriate language                      | 
