from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from us_used_cars_ml_pipeline import logger

from pyspark.ml.functions import array_to_vector

from us_used_cars_ml_pipeline.utils.tuners import TVSTuner, CVTuner, MetaModelTuner
from us_used_cars_ml_pipeline.utils.common import read_yaml, save_yaml
from us_used_cars_ml_pipeline.entity.config_entity import StackingRegressorModelingConfig
from us_used_cars_ml_pipeline.models.stacking_regressor import StackingRegressorCV

import os
import sys
from tqdm.notebook import tqdm



def str_to_class(classname: str):
    return getattr(sys.modules[__name__], classname)



class StackingRegressorModeling:

    
    def __init__(self, config: StackingRegressorModelingConfig):
        """
        Initializes the StackingRegressorModeling component.
        
        Args:
        - config (StackingRegressorModelingConfig): Configuration settings.
        - model (StackingRegressorCV): Stacking Regression model based on cross-validation approach
        """
        self.config = config
        self.model = StackingRegressorCV(config)


    def read_data_from_hdfs(self, 
                            spark: SparkSession,
                            is_new_data: bool = False) -> DataFrame:
        """
        Reads prepared parquet data from HDFS using the provided SparkSession.

        Parameters:
        - spark (SparkSession): SparkSession object.

        Returns:
        - DataFrame: Spark DataFrame containing the read data.
        """
     
        prefix = ''
        if is_new_data:
            prefix = 'NEW_'
        
        try:
            df = spark.read.parquet(os.path.join(self.config.path_to_prepared_data, prefix+'prepared_data.parquet'), 
                                    header=True, inferSchema=True)
            return df
        except Exception as e:
            logger.error(f"Failed to read parquet data from HDFS. Error: {e}")
            raise e


    def read_predictions_from_hdfs(self, 
                                   spark: SparkSession,
                                   filename: str,
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
            df = spark.read.parquet(os.path.join(self.config.path_to_predictions, prefix+filename), 
                                    header=True, inferSchema=True)
            return df
        except Exception as e:
            logger.error(f"Failed to read parquet data from HDFS. Error: {e}")
            raise e


    def get_base_models_params(self):
        """
        Returns best parameters for base models by reading corresponding yaml files

        Returns:
        - params (dict): Dictionary with parameters for base models
        """
        params = {}

        for model_name in self.config.base_models_names:

            default_params = read_yaml(Path(os.path.join(self.config.path_to_parameters_grid, f'{model_name}.yaml')))['CVTuner']['default']
            best_params = read_yaml(Path(os.path.join(self.config.path_to_best_params, f'{model_name}.yaml')))['params']
            params[model_name] = dict([(k, v) for d in [default_params, best_params] for (k, v) in d.items()])

        return params


    def tune_meta_model(self, 
                        df: DataFrame, 
                        df2: DataFrame=None):
        """
        Tunes meta model on input data with tuner specified in configs.

        Parameters:
        - df (DataFrame): Spark DataFrame.
        - df2 (DataFrame): Spark DataFrame with test (eval) data.
        """
        
        # array -> vector
        df = df.select('ID', 'price', array_to_vector(F.col('first_level_predictions')).alias('features'))
        if df2 is not None:
            df2 = df2.select('ID', 'price', array_to_vector(F.col('first_level_predictions')).alias('features'))

        # Initialization of a model
        if self.config.use_best_params == 1:
            params = self.config.meta_model_params
        if self.config.use_best_params == 0:
            params = read_yaml(Path(os.path.join(self.config.tuner_config.path_to_best_params, 
                                                      f'{self.config.meta_model_name}.yaml')))['params']

        # Initialization of a tuner
        tuner = str_to_class(self.config.tuner)(self.config.meta_model_name, 0, self.config.tuner_config)

        # Tuning
        if df2 is None:
            tuner.tune(df)
        else:
            tuner.tune(df, df2)


    def run_stage(self, 
                  spark: SparkSession, 
                  is_new_data=False):

        # Lodaing prepared data
        df = self.read_data_from_hdfs(spark, is_new_data)
        logger.info("Prepared data has been loaded")
        
        if not is_new_data:
            
            # Loading best params for base models
            base_models_params = self.get_base_models_params()
            logger.info("Best sets of hyperparameters have been loaded")

            # Creating fold column
            df = df.withColumn('fold', (F.rand(self.config.seed) * self.config.n_folds).cast('int'))
            logger.info("Fold column has been created")

            # 1. Training base models (for meta model training)
            logger.info("PART I. STARTING")
            self.model.fit_base_models_for_meta_model_training(df, base_models_params, as_separate=False)
            logger.info("PART I. COMPLETED")

            # 2. Making first level predictions using base models (for meta model training)
            logger.info("PART II. STARTING")
            self.model.predict_with_base_models_for_meta_model_training(df, as_separate=False)
            logger.info("PART II. COMPLETED")

            # 3. Merging first level predictions (for meta model training)
            logger.info("PART III. STARTING")
            self.model.merge_predictions(spark, is_new_data=is_new_data, is_training=True)
            logger.info("PART III. COMPLETED")

            # 4. Training base models
            logger.info("PART IV. STARTING")
            self.model.fit_base_models(df, base_models_params)
            logger.info("PART IV. COMPLETED")

            # 5. Making first level predictions using base models
            logger.info("PART V. STARTING")
            self.model.predict_with_base_models(df, is_new_data=is_new_data)
            logger.info("PART V. COMPLETED")

            # 6. Merging first level predictions
            logger.info("PART VI. STARTING")
            self.model.merge_predictions(spark, is_new_data=is_new_data, is_training=False)
            logger.info("PART VI. COMPLETED")

            # Reading data with predictions
            df_train = self.read_predictions_from_hdfs(spark, 'first_level_predictions_for_meta_model_training.parquet')
            df_test = self.read_predictions_from_hdfs(spark, 'first_level_predictions.parquet')
            logger.info("Data with predictions has been loaded")

            # 7. Tuning meta model
            logger.info("PART VII. STARTING")
            self.tune_meta_model(df_train, df_test)
            logger.info("PART VII. COMPLETED")

            # 8. Training meta model
            logger.info("PART VIII. STARTING")
            self.model.fit_predict_meta_model(df_train, df_test)
            logger.info("PART VIII. COMPLETED")
          
        else:
            
            # 1. Making first level predictions using base models
            self.model.predict_with_base_models(df, is_new_data=is_new_data)
            logger.info("First level predictions have been calculated")

            # 2. Merging first level predictions
            self.model.merge_predictions(spark, is_new_data=is_new_data, is_training=False)
            logger.info("First level predictions have been merged") 
            
            # 3. Reading data with merged first level predictions
            df = self.read_predictions_from_hdfs(spark, 'first_level_predictions.parquet', is_new_data=is_new_data)
            
            # 4. Making second level prediction
            self.model.predict_with_meta_model(df, is_new_data=is_new_data)
            logger.info("Second level prediction has been calculated")