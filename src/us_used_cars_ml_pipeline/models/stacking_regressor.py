from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from us_used_cars_ml_pipeline import logger
from us_used_cars_ml_pipeline.utils.common import read_yaml, save_yaml

from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.regression import GBTRegressor, GBTRegressionModel
from xgboost.spark import SparkXGBRegressor, SparkXGBRegressorModel

import os
import sys
from tqdm.notebook import tqdm



def str_to_class(classname: str):
    return getattr(sys.modules[__name__], classname)



def sort_dict_by_values(dictionary, descending=False):
    return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=descending)}
    
    

class StackingRegressorCV:
    """
    Implementaion of Stacking Regressor model based on cross-validation approach

    Attributes:
    - config: Configuration settings.
    """

    
    def __init__(self, config: dict):
        """
        Initializes Stacking Regressor model.
        
        Args:
        - config: Configuration settings.
        """
        self.config = config

    
    def save_data(self, 
                  df: DataFrame, 
                  path: str,
                  filename: str, 
                  is_new_data: bool):
        """
        Saves input data as parquet file using speified path and filename. 
        In case of new data, adds prefix NEW_ to the filename.
        
        Parameters:
        - df (DataFrame): Spark DataFrame to be saved.
        - path (str): Path where to save input data.
        - filename (str): Filename.
        - is_new_data (bool): Whether input data is new.
        """
        
        if is_new_data: filename = 'NEW_' + filename
        df.write.mode('overwrite').format('parquet').save(os.path.join(path, filename))
            
    
    def load_base_model(self, model_name, fold=None):
        """
        Loads fitted base model stored in HDFS
        
        Parameters:
        - model_name (str): Name of the base model
        - fold (int): Fold number for which a base model should be loaded
                      If None, the final base model trained on the whole data is loaded.

        Returns:
        - model (object): Base model.
        """
        
        if fold is not None:
            filename = f'{model_name}_{fold}.parquet'
        else:
            filename = f'{model_name}.parquet'
        if 'Regressor' in model_name:
            model_name_2 = model_name.replace('Regressor', 'Regression')
            model = str_to_class(model_name_2+'Model').load(os.path.join(self.config.path_to_stacking_models, filename))
        else:
            model = str_to_class(model_name+'Model').load(os.path.join(self.config.path_to_stacking_models, filename))
        return model

    
    def init_base_model(self, model_name, model_params, fold=None):
        """
        Initializes base model
        
        Parameters:
        - model_name (str): Name of the base model.
        - model_params (dict): Dictionary with parameters and hyperparameters of base model.
        - fold (int): Fold number for which a base model should be initialized
                      If None, the final base model is initialized.

        Returns:
        - model (object): Base model.
        """
        
        if fold is not None:
            predictionCol=f'{model_name}_prediction_{fold}'
        else:
            predictionCol=f'{model_name}_prediction'
        model = str_to_class(model_name)(**model_params, predictionCol=predictionCol)
        return model        

    
    def load_meta_model(self):
        """
        Loads meta model stored in HDFS.

        Returns:
        - model (object): Meta model.
        """
        
        if 'Regressor' in self.config.meta_model_name:
            model_name = self.config.meta_model_name.replace('Regressor', 'Regression')
            model = str_to_class(model_name+'Model').load(os.path.join(self.config.path_to_stacking_models, 
                                                                       f'{self.config.meta_model_name}_meta.parquet'))
        else:
            model = str_to_class(self.config.meta_model_name+'Model').load(os.path.join(self.config.path_to_stacking_models, 
                                                                                        f'{self.config.meta_model_name}_meta.parquet'))
        return model

    
    def init_meta_model(self, model_params):
        """
        Initializes meta model.
        
        Parameters:
        - model_params (dict): Dictionary with parameters and hyperparameters of meta model.
        
        Returns:
        - model (object): Meta model.
        """
        
        model = str_to_class(self.config.meta_model_name)(**model_params)
        return model        


    def select_top_features(self, 
                             df: DataFrame, 
                             model_name: str,
                             n_feats: int):
        """
        Selects best set of features for specified ML model based on results of the feature selection stage

        Parameters:
        - df (DataFrame): Spark DataFrame.
        - model_name (str): Name of ML model.
        - n_feats (int): Number of features to be selected

        Returns:
        - df_tmp (DataFrame): Spark DataFrame with only best features.
        """

        cols_to_keep = ['ID']
        if 'price' in df.columns:
            cols_to_keep.append('price')
        if 'fold' in df.columns:
            cols_to_keep.append('fold')
            
        # Selecting the best features set based on importances and best number of top features
        importances = read_yaml(Path(os.path.join(self.config.path_to_importances, f'{model_name}.yaml')))
        topFeaturesIds = list(sort_dict_by_values(importances, descending=True).keys())[:n_feats]
        df_tmp = df.select(*cols_to_keep, F.array(*[F.col('features')[featureId] for featureId in range(n_feats)]).alias('features'))
    
        # array -> vector
        df_tmp = df_tmp.select(*cols_to_keep, array_to_vector(F.col('features')).alias('features'))

        return df_tmp
        
    
    def predict_with_base_model(self, model, model_name, df, fold=None):
        """
        Makes predicitions using base model for input data
        
        Parameters:
        - model (object): Fitted base model
        - model_name (str): Name of the base model.
        - model_params (dict): Dictionary with parameters and hyperparameters of meta model.
        - df (Dataframe): Spark DataFrame
        - fold (int): Fold number for which a base model should be used
        
        Returns:
        - df (Dataframe): DataFrame with predictions in a new column.
        """
        df = model.transform(df)
        if fold != None:
            df = df.withColumn(f'{model_name}_prediction', F.when(F.col('fold')==fold, F.col(f'{model_name}_prediction_{fold}')) \
                                                            .otherwise(F.col(f'{model_name}_prediction')))      
        return df


    def fit_base_models_for_meta_model_training(self, 
                                                df: DataFrame, 
                                                base_models_params: dict,
                                                as_separate: bool = True):
        """
        Fits and saves base models, which are used in making a training set for a meta model.
        
        Parameters:
        - df (Dataframe): Spark DataFrame
        - base_models_params (dict): Dictionary with parameters and hyperparameters of base models
        """

        if as_separate:
            
            # Creating fold column
            df = df.withColumn('fold', (F.rand(self.config.seed) * self.config.n_folds).cast('int'))

        # Training each base model
        for model_name in tqdm(self.config.base_models_names, desc='Training base models'):

            # Selecting only a subset of data for the current base model
            df_tmp = self.select_top_features(df, model_name, self.config.n_feats[model_name])

            # Caching
            df_tmp.cache().count()

            # Training base model for each fold
            for fold in tqdm(range(self.config.n_folds), desc=f'Training {model_name} models'):

                # Getting only training data
                df_tmp_train = df_tmp.filter(F.col('fold') != fold).drop('fold')

                # Initialization of a model
                model = self.init_base_model(model_name, base_models_params[model_name], fold)

                # Training model
                model = model.fit(df_tmp_train)

                # Saving model
                model.write().overwrite().save(os.path.join(self.config.path_to_stacking_models, f'{model_name}_{fold}.parquet'))

            # Uncaching
            df_tmp.unpersist()


    def predict_with_base_models_for_meta_model_training(self, 
                                                         df: DataFrame,
                                                         as_separate: bool = True):
        """
        Makes and saves predictions using base models, which are used in making a training set for a meta model.
        
        Parameters:
        - df (Dataframe): Spark DataFrame
        """

        if as_separate:
        
            # Creating fold column
            df = df.withColumn('fold', (F.rand(self.config.seed) * self.config.n_folds).cast('int'))       
    
        # Making predictions using each base model
        for model_name in tqdm(self.config.base_models_names, desc='Predicting using base models'):
    
                # Selecting only a subset of data for the current base model
                df_tmp = self.select_top_features(df, model_name, self.config.n_feats[model_name])

                # Creating empty col to store predictions
                df_tmp = df_tmp.withColumn(f'{model_name}_prediction', F.lit(None))
    
                # Caching
                df_tmp.cache().count()
    
                # Training base model for each fold
                for fold in tqdm(range(self.config.n_folds), desc=f'{model_name} models'):
    
                    # Loading fitted model
                    model = self.load_base_model(model_name, fold)
    
                    # Making predictions
                    df_tmp = self.predict_with_base_model(model, model_name, df_tmp, fold)

                # Omitting unnecessary columns
                df_tmp = df_tmp.select('ID', 'price', f'{model_name}_prediction')
    
                # Saving predictions
                self.save_data(df_tmp, self.config.path_to_predictions, f'{model_name}_predictions_for_meta_model_training.parquet', is_new_data=False)

                # Uncaching
                df_tmp.unpersist()


    def merge_predictions(self, spark, is_new_data=False, is_training=False):    
        """
        Merges predictions made by base models.
        
        Parameters:
        - is_training (bool): Whether predictions data is a training data for a meta model
        - is_new_data (bool): Whether predictions data is new
        """
        
        if is_training: postfix = '_predictions_for_meta_model_training'
        else: postfix = '_predictions'
        if is_new_data: prefix = 'NEW_'
        else: prefix = ''
            
        df = None
        for model_name in tqdm(self.config.base_models_names):
            df_tmp = spark.read.parquet(os.path.join(self.config.path_to_predictions, f'{prefix}{model_name}{postfix}.parquet'), header=True, inferSchema=True)
            df = df_tmp if df is None else df.join(df_tmp.select('ID', f'{model_name}_prediction'), on='ID')  
            
        if not is_new_data:
            df = df.select('ID', 'price', F.array(*[f'{model_name}_prediction' for model_name in self.config.base_models_names]).alias('first_level_predictions'))
        else:
            df = df.select('ID', F.array(*[f'{model_name}_prediction' for model_name in self.config.base_models_names]).alias('first_level_predictions'))
       
        self.save_data(df, self.config.path_to_predictions, f'first_level{postfix}.parquet', is_new_data)         


    def fit_base_models(self, df, base_models_params):
        """
        Fits and saves final base models.
        
        Parameters:
        - df (Dataframe): Spark DataFrame
        - base_models_params (dict): Dictionary with parameters and hyperparameters of base models
        """
        
        # Training each base model
        for model_name in tqdm(self.config.base_models_names, desc='Training base models'):

            # Selecting only a subset of data for the current base model
            df_tmp = self.select_top_features(df, model_name, self.config.n_feats[model_name])

            # Caching
            df_tmp.cache().count()

            # Initialization of a model
            model = self.init_base_model(model_name, base_models_params[model_name])

            # Training model
            model = model.fit(df_tmp)

            # Saving model
            model.write().overwrite().save(os.path.join(self.config.path_to_stacking_models, f'{model_name}.parquet'))


    def predict_with_base_models(self, df, is_new_data=False):
        """
        Makes and saves predictions using final base models.
        
        Parameters:
        - df (Dataframe): Spark DataFrame
        - is_new_data (bool): Whether input data is new
        """
        
        # Making predictions using each base model
        for model_name in tqdm(self.config.base_models_names, desc='Predicting using base models'):

            # Selecting only a subset of data for the current base model
            df_tmp = self.select_top_features(df, model_name, self.config.n_feats[model_name])

            # Caching
            df_tmp.cache().count()

            # Loading model
            model = self.load_base_model(model_name)

            # Making predictions
            df_tmp = model.transform(df_tmp)     

            # Saving predictions
            self.save_data(df_tmp, self.config.path_to_predictions, f'{model_name}_predictions.parquet', is_new_data)      


    def fit_meta_model(self, df, params):
        """
        Fits and saves meta model.
        
        Parameters:
        - df (Dataframe): Spark DataFrame
        - params (dict): Dictionary with parameters and hyperparameters of meta model
        """
        
        # array -> vector
        df = df.select('ID', 'price', array_to_vector(F.col('first_level_predictions')).alias('first_level_predictions'))

        # Initialization of a model
        if self.config.use_best_params == 1:
            params = self.config.meta_model_params
        if self.config.use_best_params == 0:
            params = read_yaml(Path(os.path.join(self.config.tuner_config.path_to_best_params, 
                                                      f'{model_name}.yaml')))['params']
        
        # Initialization of a model
        model = self.init_meta_model(**params)

        # Training model
        model = model.fit(df)

        # Saving model
        model.write().overwrite().save(os.path.join(self.config.path_to_stacking_models, f'{self.config.meta_model_name}_meta.parquet'))    

    
    def predict_with_meta_model(self, df, is_new_data=False):
        """
        Makes and saves predictions using meta model
        
        Parameters:
        - df (Dataframe): Spark DataFrame
        - is_new_data (bool): Whether input data is new
        """
        
        # array -> vector
        if not is_new_data:
            df = df.select('ID', 'price', array_to_vector(F.col('first_level_predictions')).alias('first_level_predictions'))
        else:
            df = df.select('ID', array_to_vector(F.col('first_level_predictions')).alias('first_level_predictions'))

        # Loading model
        model = self.load_meta_model()

        # Making predictions
        df = model.transform(df)

        # vector -> array -> separate columns
        if not is_new_data:
            df = df.select('ID', 
                           'price', 
                           vector_to_array(F.col('first_level_predictions')).alias('first_level_predictions'), 'prediction')
            df = df.select('ID', 
                           'price', 
                           *[F.col('first_level_predictions')[k].alias(f'{model_name}_1Lpred') for k, model_name in enumerate(self.config.base_models_names)], F.col('prediction').alias(f'{self.config.meta_model_name}_2Lpred'))
        else:
            df = df.select('ID', 'prediction')
            df = df.withColumnRenamed('ID', 'vin')

        # Saving predictions
        self.save_data(df, self.config.path_to_predictions, 'second_level_predictions.parquet', is_new_data)   


    def fit_predict_meta_model(self, df_train, df_test):
        """
        Fits meta model on training data, saves meta model.
        Also makes and saves predictions using that model on testing data.
        
        Parameters:
        - df_train (Dataframe): Spark DataFrame with training data
        - df_test (Dataframe): Spark DataFrame with testing data
        """
        
        # array -> vector
        df_train = df_train.select('ID', 'price', array_to_vector(F.col('first_level_predictions')).alias('first_level_predictions'))
        df_test = df_test.select('ID', 'price', array_to_vector(F.col('first_level_predictions')).alias('first_level_predictions'))

        # Initialization of a model
        model = self.init_meta_model(self.config.meta_model_params)

        # Training model
        model = model.fit(df_train)

        # Saving model
        model.write().overwrite().save(os.path.join(self.config.path_to_stacking_models, f'{self.config.meta_model_name}_meta.parquet'))    

        # Making predictions
        df_test = model.transform(df_test)

        # vector -> array -> separate columns
        df_test = df_test.select('ID', 'price', vector_to_array(F.col('first_level_predictions')).alias('first_level_predictions'), 'prediction')
        df_test = df_test.select('ID', 'price', *[F.col('first_level_predictions')[k].alias(f'{model_name}_1Lpred') for k, model_name in enumerate(self.config.base_models_names)], F.col('prediction').alias(f'{self.config.meta_model_name}_2Lpred'))

        # Saving predictions
        self.save_data(df_test, self.config.path_to_predictions, 'second_level_predictions.parquet', is_new_data=False) 