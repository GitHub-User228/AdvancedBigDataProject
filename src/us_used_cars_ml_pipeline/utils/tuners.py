from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from us_used_cars_ml_pipeline.utils.common import read_yaml, save_yaml
from us_used_cars_ml_pipeline import logger

from pyspark.ml.functions import array_to_vector
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.regression import GBTRegressor, GBTRegressionModel
from xgboost.spark import SparkXGBRegressor, SparkXGBRegressorModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit


import os
import sys
import itertools
import numpy as np
from functools import reduce
from tqdm.notebook import tqdm



def str_to_class(classname: str):
    return getattr(sys.modules[__name__], classname)

def rgetattr(obj, attr):
    return reduce(getattr, attr.split('.'), obj)



class CVTuner:
    """
    Tuning algorithm based on cross validation.
    
    Attributes:
        model_name (str): Name of ML model to be tuned.
        model_id (id): ML model ID.
        config (dict): Dictionary with configs.
    """

    
    def __init__(self, 
                 model_name: str,
                 model_id: int,
                 config: dict):
        """
        Initializes CVTuner
        
        Attributes:
            model_name (str): Name of ML model to be tuned.
            model_id (id): ML model ID.
            config (dict): Dictionary with configs.
        """

        self.model_name = model_name
        self.model_id = model_id
        self.config = config

    
    def init_tuner(self):
        """
        Initializes cross-validation based tuner (CrossValidator)
        
        Returns:
            tuner (CrossValidator): Initialized tuner.
        """
        
        # Reading file with grid
        params = read_yaml(Path(os.path.join(self.config.path_to_parameters_grid, f'{self.model_name}.yaml')))['CVTuner']
        default_params = params['default']
        params_grid = params['grid']

        # Initializing model with default parameters
        model = str_to_class(self.model_name)(**default_params)

        # Creating a grid
        paramGrid = ParamGridBuilder()
        for param in params_grid.keys():
            paramGrid = paramGrid.addGrid(rgetattr(model, param), params_grid[param])
        paramGrid = paramGrid.build()

        # Creating cross validation based tuner
        tuner = CrossValidator(estimator=model,
                               estimatorParamMaps=paramGrid,
                               evaluator=RegressionEvaluator(labelCol='price', 
                                                             predictionCol='prediction',
                                                             metricName=self.config.metric),
                               numFolds=self.config.n_folds,
                               seed=self.config.seed)  

        return tuner


    def get_scores_and_best_set(self, tuner):
        """
        Function to extract best set of hyperparameters and scores for different sets of hyperparameters
        
        Parameters:
            tuner (CrossValidator): Fitted tuner.

        Returns:
            scores (dict): Dictionary with scores for different sets of hyperparameters
            best_params (dict): Dictionary with the best set of hyperparameters and corresponding score
        """
        
        metric = tuner.getEvaluator().getMetricName()
        scores = {'params': [], metric: []}
        for (param_set, score) in zip(tuner.getEstimatorParamMaps(), tuner.avgMetrics):
            scores[metric].append(float(score))
            scores['params'].append(dict([(k.name, v) for (k, v) in param_set.items()]))
        scores = {'params': [scores['params'][k] for k in np.argsort(scores[metric])], metric: sorted(scores[metric], reverse=True)}
        best_params = {'params': scores['params'][0], metric: scores[metric][0]}
        
        return scores, best_params
        

    def tune(self, df: DataFrame):
        """
        Function to tune ML model on input dataframe using specified configs.
        
        Parameters:
            df (DataFrame): Spark DataFrame.
        """

        # Initialization of cross validator
        tuner = self.init_tuner()
        logger.info(f'2.{self.model_id+1}.3. Cross Validation based tuner has been initialized')

        # Running cross-validation
        tuner = tuner.fit(df)
        logger.info(f'2.{self.model_id+1}.4. Tuner has been fitted')

        # Getting scores and best params set
        scores, best_params = self.get_scores_and_best_set(tuner)
        logger.info(f'2.{self.model_id+1}.5. Scores and best params set have been extracted from tuner')

        # Saving scores
        save_yaml(Path(os.path.join(self.config.path_to_scores, f'CVTuner/{self.model_name}.yaml')), scores)
        logger.info(f'2.{self.model_id+1}.6. Scores for {self.model_name} model have been saved')

        # Saving the best model's parameters
        save_yaml(Path(os.path.join(self.config.path_to_best_params, f'CVTuner/{self.model_name}.yaml')), best_params)
        logger.info(f'2.{self.model_id+1}.7. Best set of parameters for {self.model_name} model has been saved')

        # # Saving the best model
        # tuner.bestModel.write().overwrite().save(os.path.join(self.config.path_to_best_models, f'CVTuner/{self.model_name}.parquet'))
        # logger.info(f'2.{self.model_id+1}.8. Best {self.model_name} model has been saved')
        


class TVSTuner:
    """
    Tuning algorithm based on train-validation split.
    
    Attributes:
        model_name (str): Name of ML model to be tuned.
        model_id (id): ML model ID.
        config (dict): Dictionary with configs.
    """

    def __init__(self, 
                 model_name: str,
                 model_id: int,
                 config: dict):
        """
        Initializes TVSTuner
        
        Attributes:
            model_name (str): Name of ML model to be tuned.
            model_id (id): ML model ID.
            config (dict): Dictionary with configs.
        """

        self.model_name = model_name
        self.model_id = model_id
        self.config = config

    
    def init_tuner(self):
        """
        Initializes train-validation split based tuner (TrainValidationSplit)
        
        Returns:
            tuner (TrainValidationSplit): Initialized tuner.
        """   
        
        # Reading file with grid
        params = read_yaml(Path(os.path.join(self.config.path_to_parameters_grid, f'{self.model_name}.yaml')))['TVSTuner']
        default_params = params['default']
        params_grid = params['grid']

        # Initializing model with default parameters
        model = str_to_class(self.model_name)(**default_params)

        # Creating a grid
        paramGrid = ParamGridBuilder()
        for param in params_grid.keys():
            paramGrid = paramGrid.addGrid(rgetattr(model, param), params_grid[param])
        paramGrid = paramGrid.build()

        # Creating Train Validation Split based tuner
        tuner = TrainValidationSplit(estimator=model,
                                   estimatorParamMaps=paramGrid,
                                   evaluator=RegressionEvaluator(labelCol='price', 
                                                                 predictionCol='prediction',
                                                                 metricName=self.config.metric),
                                   trainRatio=1-self.config.test_ratio,
                                   seed=self.config.seed)  

        return tuner


    def get_scores_and_best_set(self, tuner):
        """
        Function to extract best set of hyperparameters and scores for different sets of hyperparameters
        
        Parameters:
            tuner (TrainValidationSplit): Fitted tuner.

        Returns:
            scores (dict): Dictionary with scores for different sets of hyperparameters
            best_params (dict): Dictionary with the best set of hyperparameters and corresponding score
        """
        
        metric = tuner.getEvaluator().getMetricName()
        scores = {'params': [], metric: []}
        for (param_set, score) in zip(tuner.getEstimatorParamMaps(), tuner.validationMetrics):
            scores[metric].append(float(score))
            scores['params'].append(dict([(k.name, v) for (k, v) in param_set.items()]))
        scores = {'params': [scores['params'][k] for k in np.argsort(scores[metric])], metric: sorted(scores[metric], reverse=True)}
        best_params = {'params': scores['params'][0], metric: scores[metric][0]}
        
        return scores, best_params

        
    def tune(self, df: DataFrame):
        """
        Function to tune ML model on input dataframe using specified configs.
        
        Parameters:
            df (DataFrame): Spark DataFrame.
        """
        
        # Initialization of cross validator
        tuner = self.init_tuner()
        logger.info(f'2.{self.model_id+1}.3. Train Validation Split based tuner has been initialized')

        # Running cross-validation
        tuner = tuner.fit(df)
        logger.info(f'2.{self.model_id+1}.4. Tuner has been fitted')

        # Getting scores and best params set
        scores, best_params = self.get_scores_and_best_set(tuner)
        logger.info(f'2.{self.model_id+1}.5. Scores and best params set have been extracted from tuner')

        # Saving the best model's parameters
        save_yaml(Path(os.path.join(self.config.path_to_scores, f'TVSTuner/{self.model_name}.yaml')), scores)
        logger.info(f'2.{self.model_id+1}.6. Scores for {self.model_name} model have been saved')

        # Saving the best model's parameters
        save_yaml(Path(os.path.join(self.config.path_to_best_params, f'TVSTuner/{self.model_name}.yaml')), best_params)
        logger.info(f'2.{self.model_id+1}.7. Best set of parameters for {self.model_name} model has been saved')

        # # Saving the best model
        # tuner.bestModel.write().overwrite().save(os.path.join(self.config.path_to_best_models, f'TVSTuner/{self.model_name}.parquet'))
        # logger.info(f'2.{self.model_id+1}.8. Best {self.model_name} model has been saved')



class MetaModelTuner:
    """
    Tuning algorithm for meta model.
    
    Attributes:
        model_name (str): Name of ML model to be tuned.
        model_id (id): ML model ID.
        config (dict): Dictionary with configs.
    """

    
    def __init__(self, 
                 model_name: str,
                 model_id: int,
                 config: dict):
        """
        Initializes 
        
        Attributes:
            model_name (str): Name of ML model to be tuned.
            model_id (id): ML model ID.
            config (dict): Dictionary with configs.
        """

        self.model_name = model_name
        self.model_id = model_id
        self.config = config


    def get_combinations(self, params_grid: dict) -> list:
        keys, values = zip(*params_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return combinations


    def tune(self, df1: DataFrame, df2: DataFrame):

        # Reading file with grid
        params = read_yaml(Path(os.path.join(self.config.path_to_parameters_grid, f'{self.model_name}.yaml')))['MetaModelTuner']
        default_params = params['default']
        params_grid = params['grid']

        # Initialization of evaluator
        evaluator = RegressionEvaluator(labelCol='price', predictionCol='prediction', metricName=self.config.metric)

        # Getting all combinations from grid
        combinations = self.get_combinations(params_grid)

        # Main loop
        scores = []
        for params in tqdm(combinations):

            # Initializing model
            model = str_to_class(self.model_name)(**params, **default_params)
    
            # Fitting model
            model = model.fit(df1)

            # Making predictions
            df2_pred = model.transform(df2)
    
            # Evaluating
            score = evaluator.evaluate(df2_pred)
            scores.append(score)
            # logger.info(f'{params} -> {self.config.metric}: {score}')

        scores = {'params': combinations, self.config.metric: scores}
        scores = {'params': [scores['params'][k] for k in np.argsort(scores[self.config.metric])], 
                  self.config.metric: sorted(scores[self.config.metric], reverse=True)}
        best_params = {'params': scores['params'][0], 
                       self.config.metric: scores[self.config.metric][0]}

        # Saving scores
        save_yaml(Path(os.path.join(self.config.path_to_scores, f'{self.model_name}.yaml')), scores)
        logger.info(f'Scores for {self.model_name} model have been saved')

        # Saving the best model's parameters
        save_yaml(Path(os.path.join(self.config.path_to_best_params, f'{self.model_name}.yaml')), best_params)
        logger.info(f'Best set of parameters for {self.model_name} model has been saved')