from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import json
import gzip


# SparkSession erstellen
spark = SparkSession.builder \
    .appName("ProcessGzippedJsonl") \
    .config("spark.master", "local[*]") \
    .config("spark.driver.host", "127.0.1.1") \
    .getOrCreate()


# spark = SparkSession.builder.config(conf=conf).getOrCreate()

# .gz Datei einlesen
input_path = "/mnt/ceph/storage/data-in-progress/data-research/web-search/archive-query-log/focused/corpus/full/2023-05-22/serps/part-00000.gz"
# /mnt/ceph/storage/data-in-progress/data-research/web-search/archive-query-log/focused/corpus/full/2023-05-22/serps
df = spark.read.json(input_path)


df.show(2)
print(df.columns)

df.drop('serp_wayback_raw_url')
df.drop('serp_wayback_url')
df.write.format('json').mode("overwrite").save('/home/benjamin/studium/masterarbeit/thesis-schneg/src/output_spark_new')

# /mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/unzipped_aql