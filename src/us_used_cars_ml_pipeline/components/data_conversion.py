import os
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from us_used_cars_ml_pipeline.entity.config_entity import DataConversionConfig



class DataConversion:
    
    
    def __init__(self, config: DataConversionConfig):
        """
        Initializes the DataConversion component with the given configuration.

        Args:
            config (DataConversionConfig): Configuration parameters for data conversion.
        """
        self.config = config

        
        
    def read_data_from_hdfs(self, 
                            spark: SparkSession, 
                            is_new_data: bool = False,
                            filename: str = None) -> DataFrame:
        """
        Reads data from HDFS and returns it as a DataFrame.

        Args:
            spark (SparkSession): Active SparkSession for data processing.

        Returns:
            DataFrame: Spark DataFrame containing the read data.

        Raises:
            Exception: If there's an error during the data reading process.
        """
        
        prefix = ''
        if is_new_data:
            prefix = 'NEW_'
            
        if filename == None:
            filename = 'data'
            
        try:
            df = spark.read.csv(os.path.join(self.config.path_to_csv_data, prefix+f'{filename}.csv'), 
                                             header=True, escape='"', multiLine=True)
            return df
        except Exception as e:
            logger.error(f"Failed to read data from HDFS. Error: {e}")
            raise e
            
            
    def run_stage(self, 
                  spark: SparkSession, 
                  is_new_data: bool = False,
                  filename: str = None):
        
        prefix = ''
        if is_new_data:
            prefix = 'NEW_'
        
        df = self.read_data_from_hdfs(spark, is_new_data, filename)

        df = df.replace('NaN', None).replace('None', None).replace(float('nan'), None)
        df = df.fillna('None', 'interior_color')
        df = df.fillna('None', 'exterior_color')

        df.write.mode('overwrite').format('parquet') \
                                  .save(os.path.join(self.config.path_to_parquet_data, prefix+'raw_data.parquet'))