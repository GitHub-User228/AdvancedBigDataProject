from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.column import Column
from pyspark.sql.types import StringType, FloatType, StructType, StructField
from pyspark.sql import functions as F
from us_used_cars_ml_pipeline import logger

from pyspark.ml.functions import array_to_vector
from pyspark.ml.evaluation import RegressionEvaluator

from us_used_cars_ml_pipeline.entity.config_entity import MetricsCalculationConfig

import os
import json



class MetricsCalculation:

    
    def __init__(self, config: MetricsCalculationConfig):
        """
        Initializes the MetricsCalculationConfig component.
        
        Args:
        - config (StackingRegressorModelingConfig): Configuration settings.
        - model (StackingRegressorCV): Stacking Regression model based on cross-validation approach
        """
        self.config = config


    def read_raw_data_from_hdfs(self, 
                                spark: SparkSession,
                                is_new_data: bool = False) -> DataFrame:
        """
        Reads raw parquet data from HDFS using the provided SparkSession.

        Parameters:
        - spark (SparkSession): SparkSession object.

        Returns:
        - DataFrame: Spark DataFrame containing the read data.
        """
     
        prefix = ''
        if is_new_data:
            prefix = 'NEW_'
        
        try:
            df = spark.read.parquet(os.path.join(self.config.path_to_raw_data, prefix+self.config.raw_data_filename), 
                                    header=True, inferSchema=True)
            return df
        except Exception as e:
            logger.error(f"Failed to read parquet data from HDFS. Error: {e}")
            raise e


    def read_predictions_from_hdfs(self, 
                                   spark: SparkSession,
                                   is_new_data: bool = False) -> DataFrame:
        """
        Reads parquet data with predictions from HDFS using the provided SparkSession.

        Parameters:
        - spark (SparkSession): SparkSession object.
        - filename(str): Name of parquet file with predictions

        Returns:
        - DataFrame: Spark DataFrame containing the read data.
        """
        
        prefix = ''
        if is_new_data:
            prefix = 'NEW_'
        
        try:
            df = spark.read.parquet(os.path.join(self.config.path_to_predictions, prefix+self.config.predictions_filename), 
                                    header=True, inferSchema=True)
            return df
        except Exception as e:
            logger.error(f"Failed to read parquet data from HDFS. Error: {e}")
            raise e
            
            
    def str_to_float(self, col: Column) -> Column:
        """
        Utility function to extract float from strings with units.

        Parameters:
        - col (Column): Spark DataFrame column.

        Returns:
        - Column: Transformed column with float values.
        """
        return F.regexp_extract(col, r"(\d+\.?\d*)", 1).cast(FloatType())
            
            
    def save_metrics(self,
                     spark: SparkSession,
                     metrics: dict,
                     is_new_data: bool = False):
        
        prefix = ''
        if is_new_data:
            prefix = 'NEW_'
        
        schema = StructType([StructField("metric", StringType()), StructField("value", FloatType(), True)])
        metrics = spark.createDataFrame(metrics.items(), schema)
        metrics.repartition(1) \
               .write.mode('overwrite').format('json') \
               .save(os.path.join(self.config.path_to_metrics, prefix+self.config.metrics_filename))
        

    def calculate(self, 
                  spark: SparkSession,
                  df_raw: DataFrame, 
                  df_pred: DataFrame,
                  is_new_data: bool = False):
        
        # Selecting only necessary columns from df_raw
        df_raw = df_raw.select(['vin', 'price'])
        df_raw = df_raw.withColumn('price', self.str_to_float(F.col('price')))
        
        # Merging
        df = df_pred.join(df_raw, on='vin', how='left')
        
        # Initializationg of the evaluator
        evaluator = RegressionEvaluator(labelCol='price', predictionCol='prediction')
        
        # Calculating metrics
        metrics = {}
        for metric_name in self.config.metrics:    
            
            # Calculating metric
            metrics[metric_name] = evaluator.evaluate(df, {evaluator.metricName: metric_name})
        
        # Saving values for metrics as json
        self.save_metrics(spark, metrics, is_new_data)


    def run_stage(self, 
                  spark: SparkSession, 
                  is_new_data=False):

        # Lodaing raw data
        df_raw = self.read_raw_data_from_hdfs(spark, is_new_data)
        logger.info("Raw data has been loaded")
        
        # Loading predictions
        df_pred = self.read_predictions_from_hdfs(spark, is_new_data)
        logger.info("Predictions have been loaded")
        
        # Calculating metrics and saving them
        self.calculate(spark, df_raw, df_pred, is_new_data)
        logger.info("Metrics have been calculated and saved") 