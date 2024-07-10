from pyspark import SparkConf, SparkContext
import logging

# Einrichten des Loggings
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
logger.info('Starting Spark Application')

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
        )

sc = SparkContext(conf = conf)
rdd = sc.textFile("part-00000.gz")

print(rdd.take(2))
s = sc.parallelize(range(10000), numSlices=3).sumApprox(0)
logger.info(f'Approximate sum: {s}\n\n\n\n######################################################')
print(f'Approximate sum: {s}\n\n\n\n######################################################')


