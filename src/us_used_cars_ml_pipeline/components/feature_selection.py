from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from us_used_cars_ml_pipeline import logger

from pyspark.ml.functions import vector_to_array, array_to_vector
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from us_used_cars_ml_pipeline.entity.config_entity import FeatureSelectionConfig

from us_used_cars_ml_pipeline.utils.selectors import PermutationImportance
from us_used_cars_ml_pipeline.utils.common import save_yaml, read_yaml

import os
import sys
import math
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd



def str_to_class(classname: str):
    return getattr(sys.modules[__name__], classname)



class FeatureSelection:
    """
    Class to perform feature selection process via Permutation Importance algorithm.
    It consists of the following parts:
    - calculating importance values for all features in the prepared data for chosen ML model
    - evaluating chosen ML model on different sets of the most important features (number of them is a variable)
    - saving the results of the previous parts as yaml files
    
    Attributes:
        config (FeatureSelectionConfig): Configuration parameters.
    """

    
    def __init__(self, config: FeatureSelectionConfig):
        """
        Initializes the FeatureSelection component with the given configuration.

        Args:
            config (FeatureSelectionConfig): Configuration parameters for features selection.
        """        
        self.config = config


    def read_data_from_hdfs(self, spark: SparkSession) -> DataFrame:
        """
        Reads prepared data from HDFS and returns it as a DataFrame.

        Parameters:
            spark (SparkSession): Active SparkSession.

        Returns:
            DataFrame: Spark DataFrame containing the read data.

        Raises:
            Exception: If there's an error during the data reading process.
        """
        try:
            df = spark.read.parquet(self.config.path_to_prepared_data, header=True, inferSchema=True)
            return df
        except Exception as e:
            logger.error(f"Failed to read data from HDFS. Error: {e}")
            raise e


    def train_eval_split(self, df: DataFrame):
        """
        Splits input dataframe into training and evaluating sets 
        according to specified test_ratio and seed in configs

        Parameters:
            df (DataFrame): Spark DataFrame.

        Returns:
            df_train (DataFrame): Training set.
            df_eval (DataFrame): Evaluating set.
        """
        
        df = df.withColumn('fold', (F.rand(self.config.seed) * int(1/self.config.test_ratio)).cast('int'))
        df_train = df.filter(F.col('fold') != 0).drop('fold')
        df_eval = df.filter(F.col('fold') == 0).drop('fold')

        return df_train, df_eval


    def train_model(self, 
                    df_train: DataFrame, 
                    model_name: str, 
                    model_parameters: dict):
        """
        Trains ML model on input training data

        Parameters:
            df (DataFrame): Spark DataFrame.
            model_name (str): Name of the ML model to be trained.
            model_parameters (dict): Parameters of the ML model.

        Returns:
            model: Fitted ML model.
        """
        
        # Initializing a model
        model = str_to_class(model_name)(**model_parameters, featuresCol='features', labelCol='price')

        # Fitting the model
        model = model.fit(df_train)

        return model


    def eval_model(self, model, df_eval, evaluator):
        """
        Evaluates ML model on input evaluating data using metrics specified in configs

        Parameters:
            model (object): Fitted ML model to be evaluated.
            df_eval (DataFrame): Evaluating data as Spark DataFrame.
            evaluator (oblect): Initialized evalutor.

        Returns:
            scores (dict): Dictionary with scores for different metrics.
        """
        scores = {}  
        df_eval = model.transform(df_eval)
        for metric in self.config.metrics:
            scores[metric] = evaluator.evaluate(df_eval, {evaluator.metricName: metric}) 
        return scores                
        

#     def compute(self, 
#                 df: DataFrame, 
#                 spark: SparkSession,
#                 model_name: str,
#                 model_parameters: dict,
#                 include_importance_calculation: bool = True,
#                 include_search: bool = True):
#         """
#         Function to perform feature selection.

#         Parameters:
#             df (DataFrame): Data for which features selection should be done.
#             spark (SparkSession): Active SparkSession.
#             model_name (str): Name of the ML model to be used.
#             model_parameters (dict): Parameters of the ML model.
#             include_importance_calculation (bool): Whether to include part, where importance values are calculated
#             include_search (bool): Whether to include part, where a search for the best features set is performed
#         """
        
#         # Train Test Split
#         df_train, df_eval = self.train_eval_split(df)
#         logger.info("1. Data has been splitted into training and testing sets")

#         if include_importance_calculation:

#             # Converting values features column from array type to vector type
#             df_train = df_train.select('ID', 'price', array_to_vector('features').alias('features'))
#             df_eval = df_eval.select('ID', 'price', array_to_vector('features').alias('features'))
#             logger.info("2. Features has been converted to vectors")
            
#             # Training ML model
#             model = self.train_model(df_train, model_name, model_parameters)
#             logger.info(f"3. Model {model_name} has been trained on the training set")
    
#             # Retrieving r2 score
#             if model_name != 'LinearRegression':
#                 evaluator = RegressionEvaluator(labelCol='price', predictionCol='prediction')
#                 df_eval = model.transform(df_eval)
#                 basis_score = evaluator.evaluate(df_eval, {evaluator.metricName: 'r2'})
#                 df_eval = df_eval.drop('prediction')
#                 df_eval = df_eval.select('ID', 'price', vector_to_array('features').alias('features'))
#             else:
#                 basis_score = model.summary.r2
#             logger.info(f"4. Basis r2 score has been calculated -> {basis_score}")

#             # Caching evaluation set
#             df_eval = df_eval.select('ID', 'price', vector_to_array('features').alias('features'))
#             df_eval.cache().count()
#             logger.info("5. Evaluation set has been cached")
    
#             # Initialization of selector model
#             selector = PermutationImportance(targetCol='price', featuresCol='features', indexCol='ID', 
#                                              nfeats=self.config.n_feats,
#                                              n_permutations=self.config.n_permutations)
#             logger.info("6. Selector has been initialized")
    
#             # Computing importances
#             importances = selector.compute(model=model, model_name=model_name, df=df_eval, spark=spark, 
#                                            verbose=self.config.verbose, basis_score=basis_score,
#                                            path_to_save_importances=self.config.path_to_save_importances)
#             logger.info("7. Importances have been calculated")

#         else:
            
#             importances = read_yaml(Path(os.path.join(self.config.path_to_save_importances, f'{model_name}.yaml')))
#             logger.info("2-7. Importance calculation part has been skipped. Using stored values")

#         if include_search:
        
#             # Creating a dictionary to store score of the later evalutaions
#             scores = dict([(metric, {}) for metric in self.config.metrics])

#             # Extracting searching configs for the current ML model
#             curr_configs = self.config.searching_params[model_name]
    
#             # Calculating a set of number of top features that will be considered
#             n_features_sets = [curr_configs.max_feats - k*curr_configs.step for k \
#                                in range(math.ceil((curr_configs.max_feats - curr_configs.min_feats + 1)/curr_configs.step))]
    
#             # Retrieving features ids in the descending order of their importances
#             featuresIds = list(importances.keys())
    
#             # Initialization of evaluator
#             evaluator = RegressionEvaluator(labelCol='price', predictionCol='prediction')
#             logger.info("8. Preparations have been done before the search for the best top features set")

#             # Dropping ID column
#             df_train = df_train.drop('ID')
#             df_eval = df_eval.drop('ID')

#             # Converting values features column from vector type to array type for the training set
#             if include_importance_calculation:
#                 df_train = df_train.select('price', vector_to_array('features').alias('features'))
#                 logger.info("9. Features in the training set has been converted to arrays")
#             else:
#                 logger.info("9. No need for vector_to_array conversion in the training data")

#             # Recaching evalutation set
#             df_eval.cache().count()
#             logger.info("10. Evaluation set has been cached")
    
#             # Caching training set
#             df_train.cache().count()
#             logger.info("11. Training set has been cached")
    
#             # Search for the best top features set
#             for it, n_feats in enumerate(tqdm(n_features_sets, desc='Evaluating best features set')):
    
#                 # Selecting only defined subset of features
#                 df_train_curr = df_train.select('price', F.array(*[F.col('features')[featureId] for featureId in featuresIds[:n_feats]]).alias('features'))
#                 df_eval_curr = df_eval.select('price', F.array(*[F.col('features')[featureId] for featureId in featuresIds[:n_feats]]).alias('features'))

#                 # Array -> vector
#                 df_train_curr = df_train_curr.select('price', array_to_vector(F.col('features')).alias('features'))
#                 df_eval_curr = df_eval_curr.select('price', array_to_vector(F.col('features')).alias('features'))
                
#                 # Training model
#                 model = self.train_model(df_train_curr, model_name, model_parameters)
    
#                 # Evaluating model
#                 current_scores = self.eval_model(model, df_eval_curr, evaluator)
    
#                 # Appending scores
#                 for metric in self.config.metrics:
#                     scores[metric][n_feats] = current_scores[metric]
    
#                 # Saving current state of scores
#                 save_yaml(Path(os.path.join(self.config.path_to_feature_selection_scores, f'{model_name}.yaml')), scores)
                
#             logger.info("12. All scores have been calculated and saved")
            
#             # Uncaching train set
#             df_train.unpersist()
#             logger.info("13. Training set has been uncached")

#         else:

#             logger.info("8-13. Searching part has been skipped")
            
#         # Uncaching eval set
#         df_eval.unpersist()
#         logger.info("14. Evaluation set has been uncached")
        
        
    def run_stage(self, 
                  spark: SparkSession, 
                  include_importance_calculation: bool = True,
                  include_search: bool = True):
        
        # Reading prepared data
        df = self.read_data_from_hdfs(spark)
        logger.info("Prepared data has been read")
        
        # Train Test Split
        df_train, df_eval = self.train_eval_split(df)
        logger.info("Data has been splitted into training and testing sets")
        
        if include_importance_calculation:
        
            # Converting values features column from array type to vector type
            df_train = df_train.select('ID', 'price', array_to_vector('features').alias('features'))
            df_eval = df_eval.select('ID', 'price', array_to_vector('features').alias('features'))
            logger.info("Features has been converted to vectors")

            # Training ML models and getting scores
            basis_scores = {}
            for model_name in tqdm(self.config.models.keys()):

                model = self.train_model(df_train, model_name, self.config.models[model_name])
                logger.info(f"Model {model_name} has been trained on the training set")

                if model_name != 'LinearRegression':
                    evaluator = RegressionEvaluator(labelCol='price', predictionCol='prediction')
                    df_eval = model.transform(df_eval)
                    basis_score = evaluator.evaluate(df_eval, {evaluator.metricName: 'r2'})
                    df_eval = df_eval.select('ID', 'price', 'features')
                else:
                    basis_score = model.summary.r2

                basis_scores[model_name] = basis_score
                logger.info(f"r2_score = {basis_score}")

            # Caching evaluation set
            df_eval = df_eval.select('ID', 'price', vector_to_array('features').alias('features'))
            df_eval.cache().count()
            logger.info("Evaluation set has been cached")

            # Calculating importance values for each model
            for model_name in tqdm(self.config.models.keys()):

                # Initialization of selector model
                selector = PermutationImportance(targetCol='price', featuresCol='features', indexCol='ID', 
                                                 nfeats=self.config.n_feats,
                                                 n_permutations=self.config.n_permutations)
                logger.info(f"Selector has been initialized for {model_name}")

                # Computing importances
                importances = selector.compute(model=model, model_name=model_name, df=df_eval, spark=spark, 
                                               verbose=self.config.verbose, basis_score=basis_scores[model_name],
                                               path_to_save_importances=self.config.path_to_save_importances)
                logger.info(f"Importances have been calculated for {model_name}")
                
        else:
            
            # Loading precomputed importance values
            importances = {}
            for model_name in tqdm(self.config.models.keys()):
                importances[model_name] = read_yaml(Path(os.path.join(self.config.path_to_save_importances, f'{model_name}.yaml')))
            logger.info("Importance calculation part has been skipped. Using stored values")
        
        if include_search:
        
            # Dropping ID column
            df_train = df_train.drop('ID')
            df_eval = df_eval.drop('ID')

            # Caching training set
            df_train.cache().count()
            logger.info("Training set has been cached")        

            # Caching evaluation set
            df_eval.cache().count()
            logger.info("Evaluation set has been cached")

            # 
            for model_name in tqdm(self.config.models.keys()):

                # Creating a dictionary to store score of the later evalutaions
                scores = dict([(metric, {}) for metric in self.config.metrics])

                # Extracting searching configs for the current ML model
                curr_configs = self.config.searching_params[model_name]

                # Calculating a set of number of top features that will be considered
                n_features_sets = [curr_configs.max_feats - k*curr_configs.step for k \
                                   in range(math.ceil((curr_configs.max_feats - curr_configs.min_feats + 1)/curr_configs.step))]

                # Retrieving features ids in the descending order of their importances
                featuresIds = list(importances[model_name].keys())

                # Initialization of evaluator
                evaluator = RegressionEvaluator(labelCol='price', predictionCol='prediction')
                logger.info(f"Preparations have been done before the search for the best top features set for {model_name}")

                # Search for the best top features set
                for it, n_feats in enumerate(tqdm(n_features_sets, desc='Evaluating best features set')):

                    # Selecting only defined subset of features
                    df_train_curr = df_train.select('price', F.array(*[F.col('features')[featureId] for featureId in featuresIds[:n_feats]]).alias('features'))
                    df_eval_curr = df_eval.select('price', F.array(*[F.col('features')[featureId] for featureId in featuresIds[:n_feats]]).alias('features'))

                    # Array -> vector
                    df_train_curr = df_train_curr.select('price', array_to_vector(F.col('features')).alias('features'))
                    df_eval_curr = df_eval_curr.select('price', array_to_vector(F.col('features')).alias('features'))

                    # Training model
                    model = self.train_model(df_train_curr, model_name, self.config.models[model_name])

                    # Evaluating model
                    current_scores = self.eval_model(model, df_eval_curr, evaluator)

                    # Appending scores
                    for metric in self.config.metrics:
                        scores[metric][n_feats] = current_scores[metric]

                    # Saving current state of scores
                    save_yaml(Path(os.path.join(self.config.path_to_feature_selection_scores, f'{model_name}.yaml')), scores)

            logger.info("All scores have been calculated and saved")               
        


#     def run_stage2(self, spark: SparkSession):

#         # Reading prepared data
#         df = self.read_data_from_hdfs(spark)
#         logger.info("Prepared data has been read")

#         # Calculating importance values
#         logger.info("STARTING")
#         for model_name in tqdm(self.config.models.keys()):
#             self.compute(df, spark, model_name, self.config.models[model_name], include_importance_calculation=True, include_search=False)
#         logger.info("COMPLETED")