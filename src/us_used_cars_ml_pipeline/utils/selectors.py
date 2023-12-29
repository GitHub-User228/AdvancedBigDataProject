from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.functions import vector_to_array, array_to_vector

from us_used_cars_ml_pipeline.utils.common import save_yaml

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import os
from pathlib import Path



class PermutationImportance:
    """
    Implementation of Permutation Importance algorithm.
    
    Attributes:
        n_permutations (int): Number of permutations.
        targetCol (str): Name of the target column.
        featuresCol (str): Name of the column with features.
        indexCol (str): Name of the index column.
        nfeats (int): Number of features
    """

    def __init__(self, 
                 targetCol: str, 
                 featuresCol: str, 
                 indexCol: str,
                 nfeats: int,
                 n_permutations: int = 5):
        """
        Initialises Permutation Importance algorithm.

        Parameters:
            n_permutations (int): Number of permutations.
            targetCol (str): Name of the target column.
            featuresCol (str): Name of the column with features.
            indexCol (str): Name of the index column.
            nfeats (int): Number of features
        """   

        self.n_permutations = n_permutations
        self.targetCol = targetCol
        self.featuresCol = featuresCol
        self.indexCol = indexCol
        self.nfeats = nfeats


    def sort_dict_by_values(self, dictionary, descending=False):
        """
        Utility function to sort dictionary by values

        Parameters:
            dictionary (dict): Dictionary to be sorted.
            descending (bool): Whether to do sorting in descending order

        Returns:
            dict: Sorted dictionary.
        """   
        return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=descending)}

    
    def create_shuffles(self, 
                        df: DataFrame, 
                        featureId: int, 
                        spark: SparkSession):
        """
        Utility function to create n_permutations shuffles for specified column.

        Parameters:
            df (DataFrame): Spark DataFrame.
            featureId (int): ID of the column to be shuffled.
            spark (SparkSession): Active SparkSession.

        Returns:
            df (DataFrame): Spark DataFrame with shuffles stored as new columns.
        """  
        
        df_tmp = df.select(self.indexCol, F.col(self.featuresCol)[featureId].alias(f'{self.featuresCol}_{featureId}')).toPandas()
        
        for it in range(self.n_permutations):
            df_tmp[f'shuffle_{it}'] = np.random.permutation(df_tmp[f'{self.featuresCol}_{featureId}'].values)
            
        df_tmp = spark.createDataFrame(df_tmp)
        df = df.join(df_tmp, on=self.indexCol, how='left')
        df = df.drop(self.indexCol)
        
        return df

    
    def replace_column_with_another(self, 
                                    df: DataFrame, 
                                    featureId: int, 
                                    anotherCol: str):
        """
        Utility function to replace a column (with specified ID) in features column with another column.

        Parameters:
            df (DataFrame): Spark DataFrame.
            featureId (int): ID of the column to be replaced.
            anotherCol (str): Name of the column which will replace specified column.

        Returns:
            df (DataFrame): Spark DataFrame with replaced featureId column from features column and dropped anotherCol column.
        """   
        expr = F.array(*[F.col(anotherCol) if it==featureId else F.col(self.featuresCol)[it] for it in range(self.nfeats)])
        df = df.withColumn(self.featuresCol, expr).drop(anotherCol) 
        return df   


    def eval_column(self, 
                    model,
                    df: DataFrame, 
                    spark: SparkSession, 
                    featureId: int, 
                    basis_score: float,
                    verbose: int):
        """
        Function to evaluate the importance of specified feature using fitted ML model.

        Parameters:
            model (object): Fitted ML model.
            df (DataFrame): Spark DataFrame.
            spark (SparkSession): Active SparkSession.
            featureId (int): ID of the column from features column which importance will be calculated.
            basis_score (float): R2 score computed when using all features.
            verbose (int): Verbosity paremeter.

        Returns:
            importance (float): Importance value.
        """   
    
        evaluator = RegressionEvaluator(labelCol=self.targetCol, predictionCol=model.getPredictionCol())
        df = self.create_shuffles(df, featureId, spark)
        scores = []
        
        # for it in tqdm(range(self.n_permutations), desc=f'permutations_{featureId}'):
        for it in range(self.n_permutations):
            
            # print(1)
            df = self.replace_column_with_another(df, featureId, f'shuffle_{it}')
            # print(2)
            df = df.select(self.targetCol, 
                           f'{self.featuresCol}_{featureId}',
                           *[f'shuffle_{k}' for k in range(it+1, self.n_permutations)],
                           array_to_vector(F.col(self.featuresCol)).alias(self.featuresCol))
            # print(3)
            df = model.transform(df)
            # print(4)
            scores.append(evaluator.evaluate(df, {evaluator.metricName: 'r2'}))
            # print(5)
    
            if verbose == 2: 
                print(f'r2: {scores[-1]}')   

            df = df.select(self.targetCol,
                           f'{self.featuresCol}_{featureId}',
                           *[f'shuffle_{k}' for k in range(it+1, self.n_permutations)],
                           vector_to_array(F.col(self.featuresCol)).alias(self.featuresCol))
            # print(6)
    
        df = self.replace_column_with_another(df, featureId, f'{self.featuresCol}_{featureId}')
        importance = basis_score - sum(scores)/self.n_permutations
    
        return importance


    def compute(self,
                model,
                model_name: str,
                basis_score: float,
                df: DataFrame, 
                spark: SparkSession, 
                path_to_save_importances: str,
                verbose: int = 1):
        """
        Function to evaluate the importance of all feature using fitted ML model.

        Parameters:
            model (object): Fitted ML model.
            model_name (str): Name of fitted ML model
            basis_score (float): R2 score computed when using all features.
            df (DataFrame): Spark DataFrame.
            spark (SparkSession): Active SparkSession.
            path_to_save_importances (str): Path where to save importance values.
            verbose (int): Verbosity paremeter.

        Returns:
            importance (dict): Importance values.
        """   

        importances = {}

        iterator = tqdm(range(self.nfeats), desc='features')
        # iterator = tqdm(range(10), desc='features')
        
        for featureId in iterator:

            df_tmp = df.select(self.indexCol, self.targetCol, self.featuresCol)

            importance = self.eval_column(model, df_tmp, spark, featureId, basis_score, verbose)

            if verbose == 1: 
                print(f'{featureId} -> ' + f'r2: {importance}')

            importances[featureId] = importance

            importances = self.sort_dict_by_values(importances, descending=True)
            
            save_yaml(Path(os.path.join(path_to_save_importances, f'{model_name}.yaml')), importances)
    
        return importances 