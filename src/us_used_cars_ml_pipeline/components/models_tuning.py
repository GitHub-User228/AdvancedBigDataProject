from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from us_used_cars_ml_pipeline import logger

from pyspark.ml.functions import array_to_vector

from us_used_cars_ml_pipeline.utils.common import read_yaml, save_yaml
from us_used_cars_ml_pipeline.utils.tuners import CVTuner, TVSTuner
from us_used_cars_ml_pipeline.entity.config_entity import ModelsTuningConfig

import os
import sys
from tqdm.notebook import tqdm



def sort_dict_by_values(dictionary, descending=False):
    return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=descending)}

def str_to_class(classname: str):
    return getattr(sys.modules[__name__], classname)



class ModelsTuning:
    """
    Class to perform ML model tuning process via TVStuner or CVTuner.
    It consists of the following parts:
    - selecting best subset of features for specified ML model based on results of feature selection stage
    - performing grid search on specified grid of hyperparameters
    - saving the results of the previous part as yaml files
    
    Attributes:
        config (ModelsTuningConfig): Configuration parameters.
    """

    
    def __init__(self, config: ModelsTuningConfig):
        """
        Initializes the ModelsTuning component.
        
        Args:
        - config (ModelsTuningConfig): Configuration settings.
        """
        self.config = config


    def read_data_from_hdfs(self, 
                            spark: SparkSession) -> DataFrame:
        """
        Reads prepared parquet data from HDFS using the provided SparkSession.

        Parameters:
        - spark (SparkSession): SparkSession object.

        Returns:
        - DataFrame: Spark DataFrame containing the read data.
        """
        try:
            df = spark.read.parquet(self.config.path_to_prepared_data, header=True, inferSchema=True)
            return df
        except Exception as e:
            logger.error(f"Failed to read parquet data from HDFS. Error: {e}")
            raise e


    def select_best_features(self, 
                             df: DataFrame, 
                             model_name: str,
                             n_feats: int):
        """
        Function to select best set of features for specified ML model based on results of the feature selection stage

        Parameters:
        - df (DataFrame): Spark DataFrame.
        - model_name (str): Name of ML model.
        - n_feats (int): Number of features to be selected

        Returns:
        - df_tmp (DataFrame): Spark DataFrame with only best features.
        """

        # Selecting the best features set based on importances and best number of top features
        importances = read_yaml(Path(os.path.join(self.config.path_to_importances, f'{model_name}.yaml')))
        topFeaturesIds = list(sort_dict_by_values(importances, descending=True).keys())[:n_feats]
        df_tmp = df.select('price', F.array(*[F.col('features')[featureId] for featureId in range(n_feats)]).alias('features'))

        # array -> vector
        df_tmp = df_tmp.select('price', array_to_vector(F.col('features')).alias('features'))

        return df_tmp


    
    def tune(self, 
             df: DataFrame, 
             skip_features_selection: bool = False,
             models_to_tune=None,
             tuner_name=None):
        """
        Function to tune specified ML model on input DataFrame

        Parameters:
        - df (DataFrame): Spark DataFrame.
        - skip_features_selection (bool): Whether not to select best set of features.
        - models_to_tune (list): List of models to be tuned (in case there are some models that have already been tuned)
        - tuner_name (str): Name of the tuner to be applied (CVTuner or TVSTuner). If None, then default tuner from configs is used
        """

        # Selecting only necessary columns
        df = df.select('price', 'features')

        if models_to_tune is None:
            models_to_tune = self.config.models

        # Tuning each model
        for model_id, model_name in enumerate(tqdm(self.config.models, desc='Tuning models', total=len(self.config.models))):

            if model_name in models_to_tune:

                # Selecting best features for the current model
                if not skip_features_selection:
                    df_tmp = self.select_best_features(df, model_name, self.config.n_feats[model_name])
                    logger.info(f'1.{model_id+1}.1. Best set of features for {model_name} model has been selected')
                else:
                    df_tmp = df.select('price', array_to_vector(F.col('features')).alias('features'))
                    logger.info(f'1.{model_id+1}.1. Best set of features selection part has been skipped')
    
                # Initialization of tuner
                if tuner_name is not None:
                    tuner = str_to_class(tuner_name)(model_name, model_id, self.config)
                else:
                    tuner = str_to_class(self.config.tuner)(model_name, model_id, self.config)
                logger.info(f'2.{model_id+1}.2. {self.config.tuner} has been initialized.')
    
                # Tuning a model
                tuner.tune(df_tmp)

            else:
                if tuner_name is not None:
                    logger.info(f'1-2.{model_id+1} {model_name} has already been tuned using {tuner_name}. Skipping that model.')
                else:
                    logger.info(f'1-2.{model_id+1} {model_name} has already been tuned using {self.config.tuner}. Skipping that model.')

    
    def run_stage(self, spark: SparkSession):

        # Reading prepared data
        df = self.read_data_from_hdfs(spark)
        logger.info("Prepared data has been read")

        # Tuning all base models
        logger.info("STARTING")
        self.tune(df, skip_features_selection=False, models_to_tune=None, tuner_name=None)
        logger.info("COMPLETED")