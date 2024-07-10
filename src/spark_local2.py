from pyspark import SparkConf
from pyspark.sql import SparkSession


# # Konfiguration für lokale Ausführung
# conf = SparkConf() \
#     .setAppName("MyApp") \
#     .setMaster("local[*]") \
#     .set("spark.driver.bindAddress", "127.0.0.1") \
#     .set("spark.driver.host", "127.0.0.1")

conf = (SparkConf().setMaster('k8s://https://k8s.srv.webis.de')
        .set('spark.kubernetes.container.image',
             'registry.webis.de/code-lib/public-images/webis/spark')
        .set('spark.kubernetes.namespace', 'spark-jobs')
        .set('spark.kubernetes.authenticate.driver.serviceAccountName', 'spark')
        .set('spark.kubernetes.driver.annotation.yunikorn.apache.org/allow-preemption', 'false')
        .set('spark.kubernetes.file.upload.path', 'file:///mnt/ceph/storage/data-tmp/current/za53top/spark-upload')
        .set('spark.kubernetes.driver.volumes.hostPath.cephfs.options.path', '/mnt/ceph/storage/data-tmp/current/za53top/spark-upload')
        .set('spark.kubernetes.driver.volumes.hostPath.cephfs.mount.path', '/mnt/ceph/storage/data-tmp/current/za53top/spark-upload')
        .set('spark.executor.instances', 3)
        .set('spark.app.name', 'sum-numbers')
        .set('log4j.rootCategory', 'INFO, console')
        .set('log4j.appender.console', 'org.apache.log4j.ConsoleAppender')
        .set('log4j.appender.console.target', 'System.err')
        .set('log4j.appender.console.layout', 'org.apache.log4j.PatternLayout')
        .set('log4j.appender.console.ConversionPattern', '%d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n')
        .set('log4j.logger.org.spark_project.jetty', 'WARN')
        .set('log4j.logger.org.spark_project.jetty.util.component.AbstractLifeCycle', 'ERROR')
        .set('log4j.logger.org.apache.spark.repl.Main', 'WARN')
        .set('log4j.logger.org.apache.spark.repl.SparkIMain$exprTyper', 'INFO')
        .set('log4j.logger.org.apache.spark.repl.SparkILoop$SparkILoopInterpreter', 'INFO')
        .set('log4j.logger.org.apache.parquet', 'ERROR')
        .set('log4j.logger.parquet', 'ERROR')
        .set('log4j.logger.org.apache.hadoop.hive.metastore.RetryingHMSHandler', 'FATAL')
        .set('log4j.logger.org.apache.hadoop.hive.ql.exec.FunctionRegistry', 'ERROR')
        )

spark = SparkSession.builder.config(conf=conf).getOrCreate()

# .gz Datei einlesen
input_path = "/mnt/ceph/storage/data-in-progress/data-research/web-search/archive-query-log/focused/corpus/full/2023-05-22/serps/part-00000.gz"
# /mnt/ceph/storage/data-in-progress/data-research/web-search/archive-query-log/focused/corpus/full/2023-05-22/serps
df = spark.read.json(input_path)


df.show(2)
print(df.columns)

df.drop('serp_wayback_raw_url')
df.drop('serp_wayback_url')

df.write.format('json').mode("overwrite").save('/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/unzipped_aql/output_spark')

# /mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/unzipped_aql