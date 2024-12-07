**Experiments**




| Experiment Type       | Data Set  | CLI Command | Start Time | End Time | Duration | Success |
|-----------------------|---------- |-------------|------------|----------|----------|---------|
| query length chars    | aql       | `python -m thesis_schneg analyser --dataset aql --analysis query-length-chars --read-concurrency 100 --map-concurrency 32 --write-concurrency 100 --memory-scaler 5 --batch-size 1024` | `2024/12/07 17:54:15`   | `2024/12/07 18:50:55` | 56m 42s  | ✔️       |
| query length chars    |  aol       | -    | -| - | - | ✔️       |
| query length chars    | ms-marco  | -    | -| - | - | ✔️       |
| query length chars    | orcas     | -    | -| - | - | ✔️       |
| query length words    | aql       | `python -m thesis_schneg analyser --dataset aql --analysis query-length-words --read-concurrency 100 --map-concurrency 32 --write-concurrency 100 --memory-scaler 5 --batch-size 256`    | `2024/12/07 18:50:16` | `	2024/12/07 19:12:52` | 22m 40s | ✔️       |
| query length words    | aol       | `python -m thesis_schneg analyser --dataset aol --analysis query-length-words --read-concurrency 100 --map-concurrency 32 --write-concurrency 100 --memory-scaler 5 --batch-size 256`    | `2024/12/07 18:35:02`  | `2024/12/07 18:43:07` | 8m 7s  | ✔️       |
| query length words    | ms-marco  | `python -m thesis_schneg analyser --dataset ms-marco --analysis query-length-words --read-concurrency 100 --map-concurrency 32 --write-concurrency 100 --memory-scaler 5 --batch-size 256`    |`2024/12/07 18:35:12`   | `2024/12/07 18:41:29` | 6m 18s  | ✔️       |
| query length words    | orcas     | `python -m thesis_schneg analyser --dataset orcas --analysis query-length-words --read-concurrency 100 --map-concurrency 32 --write-concurrency 100 --memory-scaler 5 --batch-size 256`    | `2024/12/07 18:35:22`   | `2024/12/07 18:39:40` | 4m 20s | ✔️       |
| zipfs law chars       | aql       | `python -m thesis_schneg analyser --dataset aql --analysis zipfs-law-chars --read-concurrency 100 --map-concurrency 32 --write-concurrency 100 --memory-scaler 5 --batch-size 256`    | `2024/12/07 21:15:30`   | `2024/12/07 22:28:00` | 1h 12m  | ✔️       |
| zipfs law chars       | aol       | `python -m thesis_schneg analyser --dataset aol --analysis zipfs-law-chars --read-concurrency 100 --map-concurrency 32 --write-concurrency 100 --memory-scaler 5 --batch-size 256`    | `2024/12/07 20:49:44`   | `2024/12/07 21:07:24` | 17m 44s  | ✔️       |
| zipfs law chars       | ms-marco  | `python -m thesis_schneg analyser --dataset ms-marco --analysis zipfs-law-chars --read-concurrency 100 --map-concurrency 32 --write-concurrency 100 --memory-scaler 5 --batch-size 256`    | `2024/12/07 20:55:06`   | `2024/12/07 21:06:05` | 11m 10s  | ✔️       |
| zipfs law chars       | orcas     | `python -m thesis_schneg analyser --dataset orcas --analysis zipfs-law-chars --read-concurrency 100 --map-concurrency 32 --write-concurrency 100 --memory-scaler 5 --batch-size 256`    | `2024/12/07 20:55:17`   | `2024/12/07 21:18:30` | 8m 45s  | ✔️       |
| zipfs law words       | aql       | `python -m thesis_schneg analyser --dataset aql --analysis zipfs-law-words --read-concurrency 100 --map-concurrency 64 --write-concurrency 100 --memory-scaler 7 --batch-size 256`    | `2024/12/08 00:11:39`   | 00:00 AM | 00 mins  | ❌      |
| zipfs law words       | aol       | `python -m thesis_schneg analyser --dataset aol --analysis zipfs-law-words --read-concurrency 100 --map-concurrency 64 --write-concurrency 100 --memory-scaler 5 --batch-size 256`    | `2024/12/07 21:38:07`   | `2024/12/07 23:13:24` | 1h 35m  | ✔️       |
| zipfs law words       | ms-marco  | -    | 00:00 PM   | 00:00 PM | 00 mins  | ❌       |
| zipfs law words       | orcas     | -    | 00:00 PM   | 00:00 PM | 00 mins  | ❌       |
| zipfs law queries     | aql       | -    | 00:00 AM   | 00:00 AM | 00 mins  | ❌       |
| zipfs law queries     | aol       | -    | 00:00 AM   | 00:00 AM | 00 mins  | ❌       |
| zipfs law queries     | ms-marco  | -    | 00:00 PM   | 00:00 PM | 00 mins  | ❌       |
| zipfs law queries     | orcas     | -    | 00:00 PM   | 00:00 PM | 00 mins  | ❌       |

