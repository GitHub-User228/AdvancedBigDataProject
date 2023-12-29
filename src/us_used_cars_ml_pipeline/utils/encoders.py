from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, ArrayType

from us_used_cars_ml_pipeline import logger



def kfold_mean_target_encoder(df: DataFrame,
                              col_to_encode: str,
                              target_col_name: str,
                              cv: int = 5):
    
    # Validation check on number of folds
    if cv <= 1:
        raise ValueError("cv must be greater than 1")
    
    logger.info(f"Starting K-Fold Mean Target Encoding for column '{col_to_encode}'...")

    # Initialize an empty DataFrame to store encodings
    encodings = None

    # Process each fold
    for fold in range(cv):
        # logger.info(f"Processing fold {fold+1}/{cv} for column '{col_to_encode}'...")

        df_train = df.filter(F.col('fold') != fold).select(col_to_encode, target_col_name)
        
        df_train = df_train.groupBy(col_to_encode) \
                             .agg(F.mean(target_col_name).cast(FloatType()).alias(f'targetencoding_{col_to_encode}')) \
                             .withColumn('fold', F.lit(fold))

        encodings = df_train if encodings is None else encodings.union(df_train)
        # logger.info(f"Applied encodings to validation part of fold {fold+1} for column '{col_to_encode}'")

    return encodings