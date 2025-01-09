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

```shell
ray job submit --runtime-env ray-runtime-env.yml --no-wait -- python -m thesis_schneg --<Input-Parameter> <Parameter-Value>
```

Available Input Parameters and their corresponding parameter values

| Identifier     | Input-Parameter             | Input-Values                     | Description                                                                             |
|----------------|-----------------------------|----------------------------------|-----------------------------------------------------------------------------------------| 
|Analysis mode   |`--classify`, `--aggregate`  |                                  |Determines which analysis mode should be performed on the data                           |
|Dataset         |`--dataset`                  |`aql`,`aol`,`ms-marco`,`orcas`    |Determines the data set on which the analysis is performed                               |
|Analysis mode   |`classify`, `aggregate`      |                                  |Determines which analysis mode should be performed on the data                           |
|Analysis mode   |`classify`, `aggregate`      |                                   |Determines which analysis mode should be performed on the data                          |


