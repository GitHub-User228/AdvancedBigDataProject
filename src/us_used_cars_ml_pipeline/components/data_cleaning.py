from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.column import Column
from pyspark.sql.functions import broadcast
from pyspark.sql.types import IntegerType, FloatType, BooleanType, ArrayType, DateType, StringType

from pyspark.ml.feature import StringIndexer

from us_used_cars_ml_pipeline import logger
from us_used_cars_ml_pipeline.entity.config_entity import CleanDataConfig
from us_used_cars_ml_pipeline.utils.common import save_yaml, read_yaml, load_json, save_json
from us_used_cars_ml_pipeline.utils.encoders import kfold_mean_target_encoder

import os
import pandas as pd
from pathlib import Path
from itertools import chain
from tqdm.notebook import tqdm
from datetime import datetime, date
from workalendar.usa import UnitedStates
from gensim.models.keyedvectors import KeyedVectors




class CleanData:
    """
    Class for cleaning the used cars dataset.
    What is done:
    1) Dropping rows with nans in specified columns
    2) Creating index column
    3) Converting data to desired data types
    4) Splitting power and torque columns into two columns (4 numeric columns in total)
    5) Creating time-related features from listed_date column
    6) Creating columns, which indicate the presence of popular options (retrieved from major_options column)
    7) Creating column to indicate the presence of nans in the columns
    8) Replacing nans with median values for numerical features with low number of nans
    9) Replacing nans with top class for categorical features with low number of nans
    10) Encoding specified categorical features using glove-twitter-25 model
    11) Encoding specified categorical features using KFold Mean Target Encoder
    12) Encoding specified categorical features using LabelEncoder
    13) Droppping specified columns which are not necessary
    14) Splitting data into 3 separate groups and saving them

    Attributes:
    - config (CleanDataConfig): Configuration for the data cleaning process.
    """

    
    def __init__(self, config: CleanDataConfig):
        """
        Initializes the CleanData object with the given configuration.

        Parameters:
        - config (CleanDataConfig): Configuration for the data cleaning process.
        """
        self.config = config

    
    def read_data_from_hdfs(self, 
                            spark: SparkSession, 
                            is_new_data: bool = False) -> DataFrame:
        """
        Reads the data from HDFS using the provided SparkSession.

        Parameters:
        - spark (SparkSession): SparkSession object.

        Returns:
        - DataFrame: Spark DataFrame containing the read data.
        """
        
        prefix = ''
        if is_new_data:
            prefix = 'NEW_'
        
        try:
            df = spark.read.parquet(os.path.join(self.config.path_to_raw_data, prefix+'raw_data.parquet'))
            return df
        except Exception as e:
            logger.error(f"Failed to read data from HDFS. Error: {e}")
            raise e
            

    def read_parquet_data_from_hdfs(self, 
                                    spark: SparkSession,
                                    filename: str,
                                    is_new_data: bool = False) -> DataFrame:
        """
        Reads cleaned parquet data from HDFS using the provided SparkSession.

        Parameters:
        - spark (SparkSession): SparkSession object.

        Returns:
        - DataFrame: Spark DataFrame containing the read data.
        """
        
        prefix = ''
        if is_new_data:
            prefix = 'NEW_'
        
        try:
            df = spark.read.parquet(os.path.join(self.config.path_to_cleaned_data, prefix+filename))
            return df
        except Exception as e:
            logger.error(f"Failed to read data from HDFS. Error: {e}")
            raise e
            

    def save_data(self, 
                  df: DataFrame, 
                  path: str,
                  filename: str, 
                  is_new_data: bool):
        """
        Saves unput data to HDFS as parquet.

        Parameters:
        - df (DataFrame): Spark DataFrame.
        - path (str): Path where to save data
        - filename (str): Name of the parquet file
        - is_new_data (bool): Whether input data is new.
        """
        if is_new_data: filename = 'NEW_' + filename
        df.write.mode('overwrite').format('parquet').save(os.path.join(path, filename))


    def to_int(self, col: Column) -> Column:
        """
        Utility function to convert values to int.

        Parameters:
        - col (Column): Spark DataFrame column.

        Returns:
        - Column: Transformed column with int values.
        """
        return col.cast(IntegerType())


    def to_float(self, col: Column) -> Column:
        """
        Utility function to convert value to flaot.

        Parameters:
        - col (Column): Spark DataFrame column.

        Returns:
        - Column: Transformed column with float values.
        """
        return col.cast(FloatType())


    def str_to_int(self, col: Column) -> Column:
        """
        Utility function to extract int from strings with units.

        Parameters:
        - col (Column): Spark DataFrame column.

        Returns:
        - Column: Transformed column with int values.
        """
        return F.regexp_extract(col, r"(\d+)", 1).cast(IntegerType())
        
    
    def str_to_float(self, col: Column) -> Column:
        """
        Utility function to extract float from strings with units.

        Parameters:
        - col (Column): Spark DataFrame column.

        Returns:
        - Column: Transformed column with float values.
        """
        return F.regexp_extract(col, r"(\d+\.?\d*)", 1).cast(FloatType())


    def to_date(self, col: Column) -> Column:
        """
        Utility function to convert to date. Date must be in yyyy-MM-dd format

        Parameters:
        - col (Column): Spark DataFrame column.

        Returns:
        - Column: Transformed column with boolean values.
        """
        return F.to_date(col, 'yyyy-MM-dd').cast(DateType())

    
    def split_power_torque(self, 
                           df: DataFrame, 
                           col_name: str) -> DataFrame:
        """
        Utility function to split power and torque into value and rpm.

        Parameters:
        - df (DataFrame): Spark DataFrame.
        - col_name (str): Name of the column to split.

        Returns:
        - DataFrame: DataFrame with new columns for value and rpm.
        """
        value = F.regexp_extract(df[col_name], r"(\d+)", 1).cast(IntegerType())
        rpm = F.regexp_replace(F.regexp_extract(df[col_name], r"@ ([\d,]+)", 1), ",", "").cast(IntegerType())
        return df.withColumn(f"{col_name}_value", value).withColumn(f"{col_name}_rpm", rpm).drop(col_name)
        
    
    def get_new_features_from_date(self, 
                                   df: DataFrame, 
                                   col_name: str,
                                   holidays: list,
                                   country: UnitedStates) -> DataFrame:
        """
        Utility function to extract new time-related features from listed_date feature

        Parameters:
        - df (DataFrame): Spark DataFrame.
        - col_name (str): Name of the column of DateType, which denotes date in yyyy-MM-dd format
        - holidays (list): List of dates with holidays
        - country (UnitedStates): Object to be used for checking if some day is a is_working_day

        Returns:
        - DataFrame: DataFrame with new time-related features
        """

        @F.udf(returnType=IntegerType())
        def is_working_day(input_date: date) -> int:
            """
            Utility UDF to check if a date is a working day in United States
    
            Parameters:
            - input_date (date): date.
    
            Returns:
            - int: Whether a date is a working day in United States (1: True, 0: False)
            """
            try:
                return 1 if country.is_working_day(input_date) else 0
            except:
                return None

        @F.udf(returnType=IntegerType())
        def is_holiday(input_date: date) -> int:
            """
            Utility UDF to check if a date is a holiday in United States
        
            Parameters:
            - input_date (date): date.
        
            Returns:
            - int: Whether a date is a holiday in United States (1: True, 0: False)
            """
            try:
                return 1 if input_date in holidays else 0
            except:
                return None

        # Functions to extract new time-related features
        year = F.year(df[col_name]).cast(IntegerType())
        month = F.month(df[col_name]).cast(IntegerType())
        day = F.dayofmonth(df[col_name]).cast(IntegerType())
        weekday = F.dayofweek(df[col_name]).cast(IntegerType())
        is_weekend = F.when((weekday == 5) | (weekday == 6), 1).otherwise(0).cast(IntegerType())

        # Applying all functions
        df = df.withColumn('year', year) \
               .withColumn('month', month) \
               .withColumn('day', day) \
               .withColumn('weekday', weekday) \
               .withColumn('is_weekend', is_weekend) \
               .withColumn('is_holiday', is_holiday(col_name)) \
               .withColumn('is_working_day', is_working_day(col_name))

        df = df.drop(col_name)

        return df

    
    def get_popular_options(self, 
                            df: DataFrame, 
                            col_name: str,
                            popular_options: list) -> DataFrame:
        """
        Utility function to extract an info whether popular options from major_options feature exist
        for each observation from major_options feature. 
        The result is splitted into several BooleanType columns with names denoted as popular options names
        In a case of is_new_data = False:
            - No calculations were performed to retrieve popular_options (upcoming data is training data).
            - They will be found and stored in popular_options.yaml 

        Parameters:
        - df (DataFrame): Spark DataFrame.
        - col_name (str): Name of the column with major_options data
        - popular_options (list): List of popular options.
        - is_new_data (bool): Whether input data is new.
        
        Returns:
        - DataFrame: DataFrame with new features
        """

        @F.udf(returnType=ArrayType(IntegerType()))
        def is_popular_options_in_options_list(options_list: list) -> list:
            """
            Utility function to check if popular options are in options_list

            Parameters:
            - options_list (list): Array of options

            Returns:
            - list: list of 1 and 0 values (1: True, 0: False), which define whether each popular option is in options_list
            """
            # Make sure popular_options is accessible within the UDF
            nonlocal popular_options
            if options_list is None:
                return [None] * len(popular_options)
            return [1 if option in options_list else 0 for option in popular_options]
    
        # Make sure popular_options is not None
        if popular_options is None:
            raise ValueError("Popular options have not been defined.")
    
        # Apply the UDF to get a list of popular options presence
        df = df.withColumn('popular_options', is_popular_options_in_options_list(F.col(col_name)))
        
        # Create individual columns for each popular option
        for it, option in enumerate(popular_options):
            df = df.withColumn(option, F.col('popular_options')[it])
        
        # Drop the intermediate 'popular_options' column and col_name column
        df = df.drop('popular_options', col_name)

        # Renaming
        df = df.withColumnRenamed('Sunroof/Moonroof', 'Sunroof_Moonroof')
        for feat in popular_options:
            if ' ' in feat:
                df = df.withColumnRenamed(feat, feat.replace(' ', '_'))
        
        return df
    

    def encode_via_glove(self, 
                         df: DataFrame, 
                         col_name: str,
                         glove_model: KeyedVectors,
                         is_new_data: bool) -> DataFrame:
        """
        Utility function to encode corresponding columns using glove-twitter-25 model.
        Each value in encoding (which is in a vector format) will be splitted in a separate column

        Parameters:
        - df (DataFrame): Spark DataFrame.
        - col_name (str): Name of the column to be encoded.
        - glove_model (KeyedVectors): Glove model to be used
        - is_new_data (bool): Whether input data is new

        Returns:
        - DataFrame: DataFrame with encodings placed in new columns
        """ 

        @F.udf(returnType=ArrayType(FloatType()))
        def glove_encoding(string: str) -> list:
            """
            Utility function to encode string using glove-twitter-25 model.
    
            Parameters:
            - string (str): String to encode.
    
            Returns:
            - list: Encoded string as a list of floats
            """      
            try:
                return glove_model.get_mean_vector(string).tolist()
            except:
                return [None for _ in range(glove_model.vector_size)]

        # Getting a column with only unique values
        encodings = df.select(col_name).distinct()

        # Calculating encodings
        encodings = encodings.withColumn(f'{col_name}_encoded', glove_encoding(col_name))

        # Splitting encodings stored in a single column into multiple columns
        # cols = dict([(f'glove_{col_name}_{it}', encodings[f'{col_name}_encoded'][it]) for it in range(glove_model.vector_size)])
        for it in range(glove_model.vector_size):
            encodings = encodings.withColumn(f'glove_{col_name}_{it}', encodings[f'{col_name}_encoded'][it])
        encodings = encodings.drop(F.col(f'{col_name}_encoded'))

        # Joining dataframe with encodings with main dataframe in order to get columns with encodings
        if is_new_data:
            df = df.join(broadcast(encodings), on=col_name, how='left')
        else:
            if col_name in ['city', 'model_name']:
                df = df.join(broadcast(encodings), on=col_name, how='left')
            else:
                df = df.join(encodings, on=col_name, how='left')

        df = df.drop(col_name)
        
        return df

    
    def extract_rare_classes(self, 
                             df: DataFrame, 
                             cols: list, 
                             rare_classes_count: list):
        """
        Utility function to extract classes with specified number of occurences for desired categorical columns
        
        Parameters:
        - df (DataFrame): Spark DataFrame.
        - cols (list): Names of the categorical columns.
        - rare_classes_count (list): List of integers, where each denotes the number of occurences.

        Returns:
        - output (dict): Dictionary with extracted classes
        """
        output = {}
        for col in tqdm(cols):
            tmp = {}
            df_tmp = df.select(col).groupBy(col).count()
            for v in rare_classes_count:
                classes = [row[col] for row in df_tmp.where(F.col('count') == v).select(col).collect()]
                if len(classes) > 0:
                    tmp[v] = classes
            output[col] = tmp
        return output
        

    def encode_via_kfold_mean_target_encoder(self, 
                                             df: DataFrame, 
                                             spark: SparkSession,
                                             cols_to_encode: str,
                                             target_col_name: str,
                                             is_new_data: bool) -> DataFrame:
        """
        Utility function to encode corresponding categorical columns using KFold Mean Target Encoder.
        Also replaces rare classes with generalized classes (e.g. classes with only 1 occurence -> rare_1)
        Cases:
        1) is_new_data = False
            No calculations were performed to calculate encodings (upcoming data is training data).
            They will be computed and averaged results for each class from column col_name will be saved as parquet.
        2) is_new_data = True
            Averaged encodings have been computed and results have been stored.
            Classes of column col_name will be encoded using stored encdoings.

        Parameters:
        - df (DataFrame): Spark DataFrame.
        - col_name (str): Name of the column to be encoded.
        - target_col_name (str): Name of the target column.
        - spark (SparkSession): SparkSession object.
        - is_new_data (bool): Whether input data is new. 

        Returns:
        - DataFrame: DataFrame with new column with encodings
        """      

        if not is_new_data:
    
            # Extracting rare classes for columns
            rare_classes = self.extract_rare_classes(df, cols_to_encode, self.config.rare_classes_count)
            logger.info(f"1. Rare classes have been extracted")
    
            # Saving
            save_yaml(Path(self.config.rare_classes), rare_classes)
            logger.info(f"2. Rare classes names have been saved")
            
            # Adding a 'fold' column to the DataFrame
            df = df.withColumn('fold', (F.rand(self.config.seed) * self.config.n_folds).cast('int'))
            logger.info(f"3. Fold column has been created")
            
            # Selecting only necessary columns
            df_subset = df.select('ID', 'fold', target_col_name, *cols_to_encode)
    
            # Replacing rare classes
            for col in tqdm(cols_to_encode):
    
                rare_classes_ = pd.DataFrame([(v1, f'rare_{k}') for (k, v) in rare_classes[col].items() for v1 in rare_classes[col][k]], 
                                             columns=[col, 'new_class'])
    
                df_subset = df_subset.join(broadcast(spark.createDataFrame(rare_classes_)), on=col, how='left') \
                                         .withColumn(col, F.when(F.isnull('new_class'), F.col(col)).otherwise(F.col('new_class'))) \
                                         .drop('new_class')
                
            logger.info(f"4. Rare classes have been replaced")
    
            df_subset.cache().count()
            logger.info(f"5. Temporary data has been cached")
        
            for col in tqdm(cols_to_encode):  
                encodings = kfold_mean_target_encoder(df_subset, col, target_col_name, self.config.n_folds)  
                encodings.write.mode('overwrite').format('parquet').save(os.path.join(self.config.kfold_encodings, f'{col}.parquet'))
            logger.info(f"6. Encodings have been calculated and saved")
            
            df_subset.unpersist()
            logger.info(f"7. Temporary data has been uncached")
    
            for col in tqdm(cols_to_encode):
                
                rare_classes_ = pd.DataFrame([(v1, f'rare_{k}') for (k, v) in rare_classes[col].items() for v1 in rare_classes[col][k]], 
                                             columns=[col, f'new_{col}'])
    
                df = df.join(broadcast(spark.createDataFrame(rare_classes_)), on=col, how='left') \
                       .withColumn(col, F.when(F.isnull(f'new_{col}'), F.col(col)).otherwise(F.col(f'new_{col}'))) \
                       .drop(f'new_{col}')
                logger.info(f"8.1. Rare classes have been replaced")
    
                encodings = spark.read.parquet(os.path.join(self.config.kfold_encodings, f'{col}.parquet'))
                logger.info(f"8.2. Encodings have been read")
    
                df = df.join(broadcast(encodings), on=[col, 'fold'], how='left')
                logger.info(f"8.3. {col} column has been encoded with encodings")
    
                encodings = df.select(col, f'targetencoding_{col}') \
                              .groupBy(col) \
                              .agg(F.mean(f'targetencoding_{col}').alias(f'avg_targetencoding_{col}'))
                df = df.drop(col)
                logger.info(f"8.4. Average encodings have been calculated")
    
                encodings.write.mode('overwrite').format('parquet').save(os.path.join(self.config.kfold_encodings, f'avg_{col}.parquet'))
                logger.info(f"8.5. Average encodings have been saved")    
    
            df = df.drop('fold', 'price')
    
        else:
    
            # Loading data about rare classes
            rare_classes = read_yaml(Path(self.config.rare_classes))
            logger.info(f"1. Data about rare classes has been read")
    
            for col in tqdm(cols_to_encode):
    
                rare_classes_ = pd.DataFrame([(v1, f'rare_{k}') for (k, v) in rare_classes[col].items() for v1 in rare_classes[col][k]], 
                                             columns=[col, f'new_{col}'])
    
                # Replacing rare classes
                df = df.join(broadcast(spark.createDataFrame(rare_classes_)), on=col, how='left') \
                       .withColumn(col, F.when(F.isnull(f'new_{col}'), F.col(col)).otherwise(F.col(f'new_{col}'))) \
                       .drop(f'new_{col}')
                logger.info(f"2.1. Rare classes have been replaced")
    
                # Loading stored encodings
                encodings = spark.read.parquet(os.path.join(self.config.kfold_encodings, f'avg_{col}.parquet')) \
                                      .withColumnRenamed(f'avg_targetencoding_{col}', f'targetencoding_{col}')
        
                # Joining dataframe with encodings with main dataframe in order to get a column with encodings
                df = df.join(broadcast(encodings), on=col, how='left').drop(col)
                logger.info(f"2.2. {col} column has been encoded with average encodings")
            
            # double check if price does not exist
            if 'price' in df.columns:
                df = df.drop('price')
    
        return df


    def encode_via_label_encoder(self, 
                                 df: DataFrame, 
                                 col_name: str, 
                                 spark: SparkSession,
                                 is_new_data: bool) -> DataFrame:
        """
        Utility function to encode corresponding categorical columns using Label Encoder
        In a case of is_new_data = False:
            - No calculations were performed to encode classes of columns
            - Encodings will be calculated and saved

        Parameters:
        - df (DataFrame): Spark DataFrame.
        - col_name (str): Name of the column with major_options data
        - spark (SparkSession): SparkSession object.
        - is_new_data (bool): Whether upcoming data is new.
        
        Returns:
        - DataFrame: DataFrame with new features
        """

        # Case: upcoming data is training data
        if not is_new_data:

            # Initialaizing and fitting encoder
            encodings = StringIndexer(inputCol=col_name, outputCol=f'labelencoding_{col_name}')
            encodings = encodings.fit(df)

            # Converting encoder to dict and saving it as json
            encodings = dict(zip(encodings.labels, range(len(encodings.labels))))
            save_json(Path(f'{self.config.label_encodings}/{col_name}.json'), encodings)

        else:
            
            encodings = load_json(Path(f'{self.config.label_encodings}/{col_name}.json'))

        # Converting encodings from dict to DataFrame
        encodings = spark.createDataFrame(data=encodings.items(), schema = [col_name, f'labelencoding_{col_name}'])
        encodings = encodings.withColumn(f'labelencoding_{col_name}', encodings[f'labelencoding_{col_name}'].cast(IntegerType()))
        
        # Joining dataframe with encodings with main dataframe in order to get a column with encodings
        df = df.join(broadcast(encodings), on=col_name, how='left')
        df = df.drop(col_name)
    
        return df


    def cleaning_part1(self, 
                       df: DataFrame,
                       spark: SparkSession,
                       is_new_data: bool = False,
                       as_separate: bool = True) -> DataFrame:
        """
        Function to perform the first part of the cleaning stage:
        1) Dropping rows with nans in specified columns
        2) Creating index column
        3) Converting data to desired data types
        4) Splitting power and torque columns into two columns (4 numeric columns in total)
        5) Creating columns, which indicate the presence of popular options
        6) Encoding specified categorical features using LabelEncoder
        7) Saving features_with_nans group
        8) Saving the rest columns

        Parameters:
        - df (DataFrame): Spark DataFrame.
        - spark (SparkSession): SparkSession object.
        - is_new_data (bool): Whether upcoming data is new.
        """

        # Loading necessary files
        data_types_dictionary = read_yaml(Path(self.config.data_types))
        features_with_nans = read_yaml(Path(self.config.features_with_nans))
        popular_options = read_yaml(Path(self.config.popular_options))['options']
        features_to_encode = read_yaml(Path(self.config.features_to_encode))

        # Selecting only necessary columns  
        if not is_new_data:
            df = df.select('vin', *list(data_types_dictionary.keys()))
        else:
            df = df.select('vin', *[k for k in list(data_types_dictionary.keys()) if k!='price'])

        # Dropping rows with nans in specified columns
        df = df.dropna(how = 'any', subset = features_with_nans['subset_for_dropping'])
        logger.info("1. Rows with nans in specified columns have been dropped")

        # Creating index column
        df = df.withColumnRenamed('vin', 'ID')
        # df = df.withColumn('ID', F.monotonically_increasing_id())
        logger.info("2. Index column has been created")
        
        # Apply conversion functions to corresponding columns
        function_mapping = {
            'to_int': self.to_int,
            'to_float': self.to_float,
            'str_to_int': self.str_to_int,
            'str_to_float': self.str_to_float,
            'to_date': self.to_date
        }
        for col, func_name in tqdm(data_types_dictionary.items()):
            if is_new_data and (col == 'price'):
                pass
            else:
                if func_name != 'None':
                    if func_name in function_mapping:
                        conversion_func = function_mapping[func_name]
                        df = df.withColumn(col, conversion_func(F.col(col)))
                    else:
                        raise ValueError(f"No conversion function defined for {func_name}")
        logger.info("3. Values in specified columns have been converted")

        # Split power and torque into value and rpm, and add new columns
        df = self.split_power_torque(df, 'power')
        df = self.split_power_torque(df, 'torque')
        logger.info("4. Features power and torque have been splitted")

        # Extracting new features from major_options column
        df = self.get_popular_options(df, 'major_options', popular_options)
        logger.info("5. Popular options features have been extracted")

        # Encoding some categorical columns using LabelEncoder
        for col in tqdm(features_to_encode['label_encoding']):
            df = self.encode_via_label_encoder(df, col, spark, is_new_data=is_new_data)
        logger.info("6. Specified features have been encoded using Label Encoder")   

        # Saving features with nans, where modeling is required
        self.save_data(df.select('ID', *features_with_nans['model'].to_list()), self.config.path_to_cleaned_data, 
                                   'features_with_nans.parquet', is_new_data)
        logger.info("7. Features with nans, for which modeling is required, have been saved") 
        
        if as_separate:
            # Saving other columns
            df = df.drop(*features_with_nans['model'].to_list())
            self.save_data(df, self.config.path_to_cleaned_data, 'tmp_data.parquet', is_new_data)
            logger.info("8. Other columns have been saved")   

        else:
            return df    


    def cleaning_part2(self, 
                       df1: DataFrame,
                       df2: DataFrame = None,
                       is_new_data: bool = False,
                       as_separate: bool = True):
        """
        Function to perform the second part of the cleaning stage:
        1) Creating column to indicate the presence of nans in the columns
        2) Encoding specified categorical features using glove-twitter-25 model
        3) Saving data as other_features group

        Parameters:
        - df1 (DataFrame): Spark DataFrame with features_with_nans group (and the rest data if as_separate = False.
        - df2 (DataFrame): Spark DataFrame with the rest data (if as_separate = True)
        - spark (SparkSession): SparkSession object.
        - is_new_data (bool): Whether upcoming data is new.
        
        5) Creating time-related features from listed_date column        
        7) Replacing nans with median values for numerical features with low number of nans
        8) Replacing nans with top class for categorical features with low number of nans
        10) Encoding specified categorical features using KFold Mean Target Encoder
        13) Splitting data into 3 separate groups and saving them

        """
    
        # Loading necessary files and models
        features_with_nans = read_yaml(Path(self.config.features_with_nans))
        features_to_encode = read_yaml(Path(self.config.features_to_encode))
        features_to_drop = read_yaml(Path(self.config.features_to_drop))['features']
        feats_still_with_nans = [v if type(v)==str else list(v.keys())[0] for k in ['median', 'top_class', 'model'] for v in features_with_nans[k]]
        features = list(set(feats_still_with_nans + features_to_encode['glove']))
        if not is_new_data:
            features = features + ['price']
        glove_model = KeyedVectors.load(self.config.glove_model)

        
        if as_separate:
            
            # Selecting necessary columns
            df1 = df1.select('ID', *[k for k in features if k in df1.columns])
            df2 = df2.select('ID', *[k for k in features if k in df2.columns])
    
            # Creating columns to indicate the presence of missing values
            for col in tqdm(feats_still_with_nans):
                if col in df1.columns:  # Check if the column exists in the DataFrame
                    df1 = df1.withColumn(col, F.when(F.isnull(F.col(col)), 0).otherwise(1).cast(IntegerType())) \
                             .withColumnRenamed(col, f'isnotnan_{col}')
            for col in tqdm(feats_still_with_nans):
                if col in df2.columns:  # Check if the column exists in the DataFrame
                    df2 = df2.withColumn(col, F.when(F.isnull(F.col(col)), 0).otherwise(1).cast(IntegerType())) \
                             .withColumnRenamed(col, f'isnotnan_{col}')
            logger.info("1. Columns to indicate nans presence have been created")
            
            # Merging
            df1 = df1.join(df2, on='ID', how='left')
            logger.info("2. DataFrames have been merged")

        else:

            # Selecting necessary columns
            df1 = df1.select('ID', *[k for k in features if k in df1.columns])
    
            # Creating columns to indicate the presence of missing values
            for col in tqdm(feats_still_with_nans):
                if col in df1.columns:  # Check if the column exists in the DataFrame
                    df1 = df1.withColumn(col, F.when(F.isnull(F.col(col)), 0).otherwise(1).cast(IntegerType())) \
                             .withColumnRenamed(col, f'isnotnan_{col}')
            logger.info("1. Columns to indicate nans presence have been created")
            
            # Merging
            logger.info("2. No need to merge data")            

        # Caching
        df1_cached = df1.cache()
        df1_cached.count()
        logger.info("3. Data has been cached")

        # Encoding specified features using glove-twitter-25 model
        for col in tqdm(features_to_encode['glove']):
            df1_cached = self.encode_via_glove(df1_cached, col, glove_model, is_new_data)
        logger.info("4. Specified features have been encoded using glove-twitter-25 model")

        # Saving data as other_features group
        self.save_data(df1_cached, self.config.path_to_cleaned_data, 'other_features.parquet', is_new_data)
        logger.info("5. Data has been saved")   

        # Uncaching
        df1_cached.unpersist()
        logger.info("6. Data has been uncached")


    def cleaning_part3(self, 
                       df: DataFrame,
                       spark: SparkSession,
                       is_new_data: bool = False,
                       as_separate: bool = True) -> DataFrame:
        """
        Function to perform the third part of the cleaning stage:
        1) Creating time-related features from listed_date column
        2) Replacing nans with median value or top class
        3) Encoding specified categorical features using KFold Mean Target Encoder
        4) Saving data as features_for_imputers group

        Parameters:
        - df (DataFrame): Spark DataFrame.
        - spark (SparkSession): SparkSession object.
        - is_new_data (bool): Whether upcoming data is new.
        """
        
        # Loading necessary files and models
        features_with_nans = read_yaml(Path(self.config.features_with_nans))
        features_to_encode = read_yaml(Path(self.config.features_to_encode))
        country = UnitedStates()
        holidays = [item[0] for k in [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021] for item in country.holidays(k)]
        
        # Dropping columns in case when stage is running whole at once
        if not as_separate:
            df = df.drop(*features_with_nans['model'].to_list())

        # Getting new time-related features from listed_date column
        df = self.get_new_features_from_date(df, 'listed_date', holidays, country)
        logger.info("1. Time-related features have been computed")

        # Replacing nans with median value or top class in corresponding columns
        mapper = dict([list(kv.items())[0] for kv in features_with_nans['median'].to_list()] + \
                      [list(kv.items())[0] for kv in features_with_nans['top_class'].to_list()])
        for (col, val) in tqdm(mapper.items()):
            if 'labelencoding' in col:
                encodings = load_json(Path(f"{self.config.label_encodings}/{col.replace('labelencoding_', '')}.json"))
                mapper[col] = encodings[val]
        df = df.fillna(mapper)
        logger.info("2. Nans have been replaced with median value or top class")

        # Encoding some features using k-fold mean target encoder
        df = self.encode_via_kfold_mean_target_encoder(df, spark, features_to_encode['kfold_mean_target_encoding'], 
                                                       'price', is_new_data)
        logger.info("3. Specified features have been encoded using k-fold mean target encoder")
        
        # Saving data as features_for_imputers group
        self.save_data(df, self.config.path_to_cleaned_data, 'features_for_imputers.parquet', is_new_data)


    def run_stage(self, 
                  spark: SparkSession, 
                  is_new_data: bool = False):

        # Reading raw data
        df = self.read_data_from_hdfs(spark, is_new_data)
        logger.info("Raw data has been read")
        
        # Dealing with nans
        df = df.replace('NaN', None).replace('None', None).replace(float('nan'), None)
        df = df.fillna('None', 'interior_color')
        df = df.fillna('None', 'exterior_color')

        # Performing part 1
        logger.info("PART I. STARTING")
        df = self.cleaning_part1(df, spark, is_new_data, as_separate=False)
        logger.info("PART I. COMPLETED")
        
        # Caching data
        df.cache().count()
        logger.info("Data has been cached")

        # Performing part 2
        logger.info("PART II. STARTING")
        self.cleaning_part2(df, None, is_new_data, as_separate=False)
        logger.info("PART II. COMPLETED")

        # Performing part 3
        logger.info("PART III. STARTING")
        self.cleaning_part3(df, spark, is_new_data, as_separate=False)
        logger.info("PART III. COMPLETED")
        
        # Uncaching data
        df.unpersist()