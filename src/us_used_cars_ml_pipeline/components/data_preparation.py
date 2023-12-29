from pathlib import Path

from pyspark.sql.types import FloatType, ArrayType
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from us_used_cars_ml_pipeline.utils.common import read_yaml, save_yaml
from us_used_cars_ml_pipeline import logger

from pyspark.ml.functions import vector_to_array, array_to_vector
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, MinMaxScalerModel
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator

from us_used_cars_ml_pipeline.entity.config_entity import DataPreparationConfig


import os
import sys
from tqdm.notebook import tqdm

import time


def str_to_class(classname: str):
    return getattr(sys.modules[__name__], classname)



class DataPreparation:
    """
    Class for data preparation before it can be used for later stages.
    Parts:
    
    1. Scaling features_for_imputers group
    
    2. Training imputer models and saving them (for features with nans)

    3.1. Filling in missing data with imputer models
    3.2. Scaling features with nans

    4.1 Scaling other_features group
    4.2. Grouping all features together
    4.3. Saving prepared data

    In case when input data is new, fitted scalers and imputer models are loaded (no training is peformed)
    """
    
    def __init__(self, config: DataPreparationConfig):
        """
        Initializes the DataPreparation component.
        
        Args:
        - config (DataPreparationConfig): Configuration settings.
        """
        self.config = config


    def read_data_from_hdfs(self, 
                            spark: SparkSession, 
                            filename: str,
                            is_new_data: bool = False) -> DataFrame:
        """
        Reads the data from HDFS using the provided SparkSession.

        Parameters:
        - spark (SparkSession): SparkSession object.
        - filename (str): Name of the parquet file to read

        Returns:
        - DataFrame: Spark DataFrame containing the read data.
        """
        
        prefix = ''
        if is_new_data:
            prefix = 'NEW_'
        
        try:
            df = spark.read.parquet(os.path.join(self.config.path_to_cleaned_data, prefix+filename), 
                                    header=True, inferSchema=True)
            return df
        except Exception as e:
            logger.error(f"Failed to read data from HDFS. Error: {e}")
            raise e


    def read_prepared_parquet_data_from_hdfs(self, 
                                             spark: SparkSession, 
                                             filename: str,
                                             is_new_data: bool = False) -> DataFrame:
        """
        Reads prepared parquet data from HDFS using the provided SparkSession.

        Parameters:
        - spark (SparkSession): SparkSession object.
        - filename (str): Parquet filename.

        Returns:
        - DataFrame: Spark DataFrame containing the read data.
        """
  
        prefix = ''
        if is_new_data:
            prefix = 'NEW_'
        
        try:
            df = spark.read.parquet(os.path.join(self.config.path_to_prepared_data, prefix+filename), 
                                    header=True, inferSchema=True)
            return df
        except Exception as e:
            logger.error(f"Failed to read parquet data from HDFS. Error: {e}")
            raise e


    def fit_scaler(self, 
                   df: DataFrame, 
                   group: list):
        """
        Function to fit a scaler on group of features of DataFrame.

        Parameters:
        - df (DataFrame): Spark DataFrame
        - group (str): Name of the column (with features in a vector form) to scale.
        """

        # Initialiazing scaler
        self.scaler = str_to_class(self.config.scaler_name)(inputCol=group, outputCol=f'scaled_{group}')

        # Fitting data
        self.scaler = self.scaler.fit(df)

        # Saving scaler
        self.scaler.write().overwrite().save(os.path.join(self.config.path_to_scalers, f'{group}.parquet'))


    def load_scaler(self, 
                    group: list):
        """
        Function to load a scaler, which was fitted a group of features.

        Parameters:
        - group (str): Name of the column (with features in a vector form) to scale.
        """
        
        # Loading scaler
        self.scaler = str_to_class(self.config.scaler_name+'Model').load(os.path.join(self.config.path_to_scalers, f'{group}.parquet'))


    def train_model(self, 
                    df: DataFrame, 
                    target_col_name: str,
                    model_name: str):
        """
        Function, that is used to train and save an imputer model for specified target_col_name feature.

        Parameters:
        - df (DataFrame): Spark DataFrame
        - target_col_name (str): Name of the target column
        - model_name (str): Name of ML model to be trained
        """

        # Initializing a model
        self.model = str_to_class(model_name)(**self.config.models[model_name], 
                                                featuresCol='features_for_imputers',
                                                labelCol=target_col_name)

        # Fitting the model
        self.model = self.model.fit(df)

        # Saving the model
        self.model.write().overwrite().save(os.path.join(self.config.path_to_imputers, f'{target_col_name}.parquet'))


    def eval_model(self,
                   df: DataFrame, 
                   target_col_name: str,
                   model_name: str,
                   is_numerical_target_type: bool):
        """
        Function to evaluate recently trained imputer model.

        Parameters:
        - df (DataFrame): Spark DataFrame
        - target_col_name (str): Name of the target column
        - model_name (str): Name of fitted ML model
        - is_numerical_target_type (bool): Whether the target column is numerical.
        
        Returns:
        - scores (dict): Dictionary with scores for different metrics
        """

        if is_numerical_target_type:

            if model_name == 'LinearRegression':
                scores = {}
                scores['RMSE'] = self.model.summary.rootMeanSquaredError
                scores['MAE'] = self.model.summary.meanAbsoluteError
                scores['R2'] = self.model.summary.r2

            else:
                df = self.model.transform(df)
                evaluator = RegressionEvaluator(labelCol=target_col_name, predictionCol='prediction')
                scores = {}
                scores['RMSE'] = evaluator.evaluate(df, {evaluator.metricName: 'rmse'})
                scores['MAE'] = evaluator.evaluate(df, {evaluator.metricName: 'mae'})
                scores['R2'] = evaluator.evaluate(df, {evaluator.metricName: 'r2'})

            print(f"{target_col_name} / {scores['RMSE']} / {scores['MAE']} / {scores['R2']}")

        else:

            if model_name == 'LogisticRegression':
                scores = {}
                scores['accuracy'] = self.model.summary.accuracy
                scores['precision'] = self.model.summary.weightedPrecision
                scores['recall'] = self.model.summary.weightedRecall

            else:
                df = self.model.transform(df)
                evaluator = MulticlassClassificationEvaluator(labelCol=target_col_name, predictionCol='prediction')
                scores = {}
                scores['accuracy'] = evaluator.evaluate(df, {evaluator.metricName: 'accuracy'})
                scores['precision'] = evaluator.evaluate(df, {evaluator.metricName: 'precision'})
                scores['recall'] = evaluator.evaluate(df, {evaluator.metricName: 'recall'})       

            print(f"{target_col_name} / {scores['accuracy']} / {scores['precision']} / {scores['recall']}")

        return scores         


    def load_model(self, 
                   target_col_name: str,
                   model_name: str):
        """
        Function, that is used to load fitted imputer model for specified target_col_name feature.

        Parameters:
        - target_col_name (str): Name of the target column
        - model_name (str): Name of ML model to be loaded
        """

        if model_name == 'RandomForestRegressor':
            self.model = RandomForestRegressionModel.load(os.path.join(self.config.path_to_imputers, f'{target_col_name}.parquet'))
        else:
            self.model = str_to_class(model_name+'Model').load(os.path.join(self.config.path_to_imputers, f'{target_col_name}.parquet'))
    

    def predict(self, 
                df: DataFrame, 
                target_col_name: str, 
                is_numerical_target_type: bool) -> DataFrame:
        """
        Function, that is used to make predictions using fitted imputer model for specified target_col_name feature 
        for rows with missing data in that column.

        Parameters:
        - df (DataFrame): Spark DataFrame
        - target_col_name (str): Name of the target column
        - is_numerical_target_type (bool): Whether the target column is categorical.

        Returns:
        - df (DataFrame): Spark DataFrame with no missing data in the target_col_name column.
        """
        
        cols_to_drop = ['prediction']
        if not is_numerical_target_type:
            cols_to_drop += ['rawPrediction', 'probability']

        df = self.model.transform(df) \
                       .withColumn(target_col_name, F.when(F.isnull(F.col(target_col_name)), F.col('prediction')) \
                                                     .otherwise(F.col(target_col_name))) \
                       .drop(*cols_to_drop)
        
        return df


    def save_data(self, 
                  df: DataFrame, 
                  filename: str, 
                  is_new_data: bool):
        """
        Function to save a prepared part of DataFrame as parquet.

        Parameters:
        - df (DataFrame): Spark DataFrame
        - filename (str): Name of parquet file.
        """

        if is_new_data: filename = 'NEW_' + filename
        df.write.mode('overwrite').format('parquet').save(os.path.join(self.config.path_to_prepared_data, filename))


    def prepare_part1(self,                       
                      df1: DataFrame, 
                      df2: DataFrame,
                      is_new_data: bool,
                      as_separate: bool = True) -> DataFrame:
        """
        Function to train imputer models and some scalers

        Parameters:
        - df1 (DataFrame): Spark DataFrame with features_for_imputers group of features
        - df2 (DataFrame): Spark DataFrame with features_with_nans group of features
        - is_new_data (bool): Whether input DataFrame is new.
        """

        if not is_new_data:

            # Reading yaml file with the list of features
            features = read_yaml(Path(self.config.path_to_features_list))
    
            # Grouping
            assembler = VectorAssembler(inputCols=features['features_for_imputers'].to_list(), outputCol='features_for_imputers')
            df1 = assembler.transform(df1).drop(*features['features_for_imputers'])
            
            # Scaling
            self.fit_scaler(df1, 'features_for_imputers')
            df1 = self.scaler.transform(df1) \
                             .drop('features_for_imputers') \
                             .withColumnRenamed('scaled_features_for_imputers', 'features_for_imputers')
            logger.info(f"1. features_for_imputers group has been scaled")
    
            # Merging
            df1 = df1.join(df2, on='ID', how='left')
            logger.info(f"2. Data has been merged")

            # Caching
            df1.cache().count()
            logger.info(f"3. Data has been cached")

            # Training
            scores = {'numerical_feats': {}, 'categorical_feats': {}}
            print("-"*100)
            print(f"column / RMSE / MAE / R2")
            for (col, model_name) in tqdm(features['numerical_features_with_nans'].to_dict().items()): 
                df1_subset = df1.select('features_for_imputers', col).where(~F.isnull(F.col(col)))
                self.train_model(df1_subset, col, model_name) 
                scores['numerical_feats'][col] = self.eval_model(df1_subset, col, model_name, True)
    
            print("-"*100)
            print(f"column / accuracy / precision / recall")
            for (col, model_name) in tqdm(features['categorical_features_with_nans'].to_dict().items()):
                df1_subset = df1.select('features_for_imputers', col).where(~F.isnull(F.col(col)))
                self.train_model(df1_subset, col, model_name) 
                scores['categorical_feats'][col] = self.eval_model(df1_subset, col, model_name, False)

            save_yaml(Path(self.config.path_to_scores_file), scores)
            logger.info("4. Imputer models have been trained")

            if as_separate: 
                # Uncaching
                df1.unpersist()
                logger.info(f"5. Data has been uncached")

            else:
                logger.info(f"5. Data won't be uncached")
                return df1
            
        else:
            logger.info("1-5. Scalers and imputer models have already been fitted. Skipping this part")
            if not as_separate: 
                return None
        

    def prepare_part2(self, 
                      df1: DataFrame, 
                      df2: DataFrame,
                      df3: DataFrame,
                      is_new_data: bool,
                      as_separate: bool = True):
        """
        Function to scale cleaned data, fill in missing data using fitted imputer models,
        merge and save resulting data.

        Parameters:
        - df1 (DataFrame): Spark DataFrame with features_for_imputers group of features
        - df2 (DataFrame): Spark DataFrame with features_with_nans group of features
        - df3 (DataFrame): Spark DataFrame with other_features group of features
        - is_new_data (bool): Whether input DataFrame is new.
        """

        # Reading yaml file with the list of features
        features = read_yaml(Path(self.config.path_to_features_list))

        if as_separate or is_new_data:

            # Grouping features_for_imputers    
            assembler = VectorAssembler(inputCols=features['features_for_imputers'].to_list(), outputCol='features_for_imputers')
            df1 = assembler.transform(df1).drop(*features['features_for_imputers'])
            
            # Scaling features_for_imputers
            self.load_scaler('features_for_imputers')
            df1 = self.scaler.transform(df1) \
                             .drop('features_for_imputers') \
                             .withColumnRenamed('scaled_features_for_imputers', 'features_for_imputers')
            logger.info(f"6. features_for_imputers group has been scaled")
    
            # Merging features_for_imputers and features_with_nans
            df1 = df1.join(df2, on='ID', how='left')   

        # Filling in missing data
        for it, (group, is_num) in enumerate(zip(['categorical_features_with_nans', 'numerical_features_with_nans'], [False, True])):
            for (col, model_name) in tqdm(features[group].to_dict().items()):   
                self.load_model(col, model_name)           
                df1 = self.predict(df1, col, is_num)      
            logger.info(f"7.{it+1} Missing data in {group.split('_')[0]} features has been filled in")

        # Scaling features_with_nans   
        for it, group in enumerate(['numerical_features_with_nans', 'categorical_features_with_nans']):
            
            feats = list(features[group].to_dict().keys())
            assembler = VectorAssembler(inputCols=feats, outputCol=group)
            df1 = assembler.transform(df1).drop(*feats)
            logger.info(f"8.{it+1}.1. {group} features has been grouped")
    
            if not is_new_data:
                self.fit_scaler(df1, group)
            else:
                self.load_scaler(group)
            df1 = self.scaler.transform(df1) \
                             .drop(group) \
                             .withColumnRenamed(f'scaled_{group}', group)
            logger.info(f"8.{it+1}.2. {group} group has been scaled")

        # Grouping other_features
        other_features = [k for k in features['other_features'].to_list() if k != 'price']
        if not is_new_data:
            df3 = df3.select('ID', 'price', F.array(*other_features).alias('other_features'))
        else:
            df3 = df3.select('ID', F.array(*other_features).alias('other_features'))
        logger.info(f"9. other_features features has been grouped")

        # Merging
        df1 = df1.join(df3, on='ID', how='left')
        logger.info(f"10. Data has been merged")
        
        # Regrouping
        if not is_new_data: 
            df1 = df1.select('ID', 
                             'price',
                              F.concat(vector_to_array(F.col('features_for_imputers')).cast(ArrayType(FloatType())),
                                       vector_to_array(F.col('numerical_features_with_nans')).cast(ArrayType(FloatType())), 
                                       vector_to_array(F.col('categorical_features_with_nans')).cast(ArrayType(FloatType())),
                                       F.col('other_features')).alias('features'))
        else:
            df1 = df1.select('ID',
                              F.concat(vector_to_array(F.col('features_for_imputers')).cast(ArrayType(FloatType())),
                                       vector_to_array(F.col('numerical_features_with_nans')).cast(ArrayType(FloatType())), 
                                       vector_to_array(F.col('categorical_features_with_nans')).cast(ArrayType(FloatType())),
                                       F.col('other_features')).alias('features'))
        logger.info(f"11. Data has been regrouped")

        # Saving
        self.save_data(df1, 'prepared_data.parquet', is_new_data=is_new_data)   
        logger.info(f"12. Prepared data has been saved")


    def run_stage(self, 
                  spark: SparkSession, 
                  is_new_data: bool = False):

        # Loading data
        df1 = self.read_data_from_hdfs(spark, 'features_for_imputers.parquet', is_new_data)
        df2 = self.read_data_from_hdfs(spark, 'features_with_nans.parquet', is_new_data)
        df3 = self.read_data_from_hdfs(spark, 'other_features.parquet', is_new_data)
        logger.info("Cleaned data has been read")

        # Performing part 1
        logger.info("PART I. STARTING")
        df12 = self.prepare_part1(df1, df2, is_new_data=is_new_data, as_separate=False)
        logger.info("PART I. COMPLETED")

        # Performing part 2
        logger.info("PART II. STARTING")
        if not is_new_data:
            self.prepare_part2(df12, None, df3, is_new_data=is_new_data, as_separate=False)
        else:
            self.prepare_part2(df1, df2, df3, is_new_data=is_new_data, as_separate=False)
        logger.info("PART II. COMPLETED")