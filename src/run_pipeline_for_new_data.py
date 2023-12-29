import os
import warnings

warnings.filterwarnings("ignore")


import socket
from pyspark.sql import SparkSession
from us_used_cars_ml_pipeline.pipelines.new_data_pipeline import run


LOCAL_IP = socket.gethostbyname(socket.gethostname())

name_space = "eabraham-373705"

# Master node
kubernetes_master_url = "k8s://https://10.32.7.103:6443"

# Resource settings
driver_cores = "6"
executor_cores = "6"
driver_memory = "22g"
executor_memory = "22g"
executor_memory_overhead = "2g"

# These are the limits
cpu_limit = "3"  # 12 cores
memory_limit = "32g"  # Upto 32 GB
executor_limit = "8"

APP_NAME = 'scalables_executor'


spark = SparkSession\
    .builder\
    .appName(APP_NAME)\
    .master(kubernetes_master_url)\
    .config("spark.driver.host", LOCAL_IP)\
    .config("spark.driver.bindAddress", "0.0.0.0")\
    .config("spark.executor.instances", "2")\
    .config("spark.executor.cores", executor_cores)\
    .config("spark.executor.memory", executor_memory)\
    .config("spark.memory.fraction", "0.8")\
    .config("spark.memory.storageFraction", "0.2")\
    .config("spark.kubernetes.executor.limit.cores", executor_limit)\
    .config("spark.kubernetes.namespace", name_space)\
    .config("spark.kubernetes.authenticate.driver.serviceAccountName", "spark")\
    .config("spark.kubernetes.driver.label.appname", APP_NAME)\
    .config("spark.kubernetes.executor.label.appname", APP_NAME)\
    .config("spark.kubernetes.executor.deleteOnTermination", "false") \
    .config("spark.kubernetes.container.image.pullPolicy", "Always") \
    .config("spark.kubernetes.container.image", "node03.st:5000/pyspark-hdfs-jupyter:eabraham-373705-v4-executor")\
    .config("spark.local.dir", "/tmp/spark")\
    .config("spark.kubernetes.driver.volumes.emptyDir.spark-local-dir-tmp-spark.mount.path", "/tmp/spark")\
    .config("spark.kubernetes.driver.volumes.emptyDir.spark-local-dir-tmp-spark.mount.readOnly", "false")\
    .config("spark.kubernetes.executor.volumes.emptyDir.spark-local-dir-tmp-spark.mount.path", "/tmp/spark")\
    .config("spark.kubernetes.executor.volumes.emptyDir.spark-local-dir-tmp-spark.mount.readOnly", "false")\
    .getOrCreate()

run(spark, do_conversion=False, calculate_metrics=True)

spark.stop()