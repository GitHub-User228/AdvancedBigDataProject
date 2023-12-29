from pyspark.sql import SparkSession

from us_used_cars_ml_pipeline import logger
from us_used_cars_ml_pipeline.config.configuration import ConfigurationManager
from us_used_cars_ml_pipeline.components.data_conversion import DataConversion
from us_used_cars_ml_pipeline.components.data_cleaning import CleanData
from us_used_cars_ml_pipeline.components.data_preparation import DataPreparation
from us_used_cars_ml_pipeline.components.stacking_regressor_modeling import StackingRegressorModeling
from us_used_cars_ml_pipeline.components.metrics_calculation import MetricsCalculation



def run(spark: SparkSession, 
        new_csv_data_filename: str = None,
        do_conversion: bool = True,
        calculate_metrics: bool = False):
    
    logger.info('=== STARTING PROCESSING NEW DATA ===')
    
    # Initialization of necessary components
    config_manager = ConfigurationManager()
    if do_conversion:
        data_conversion = DataConversion(config_manager.get_data_conversion_config())
    data_cleaning = CleanData(config_manager.get_clean_data_config())
    data_preparation = DataPreparation(config_manager.get_data_preparation_config())
    stacking_regressor_modeling = StackingRegressorModeling(config_manager.get_stacking_regressor_modeling_config())
    if calculate_metrics:
        metrics_calculation = MetricsCalculation(config_manager.get_metrics_calculation_config())
    logger.info('=== I. COMPONENTS HAVE BEEN INITIALIZED ===')
    
    # STAGE 1
    if do_conversion:
        data_conversion.run_stage(spark, is_new_data=True, filename=new_csv_data_filename)
        logger.info('=== II. DATA CONVERSION STAGE HAS BEEN COMPLETED ===')
    else:
        logger.info('=== II. DATA CONVERSION STAGE HAS BEEN SKIPPED ===')
        
    
    # STAGE 2
    data_cleaning.run_stage(spark, is_new_data=True)
    logger.info('=== III. DATA CLEANING STAGE HAS BEEN COMPLETED ===')
    
    # STAGE 3
    data_preparation.run_stage(spark, is_new_data=True)
    logger.info('=== IV. DATA PREPARATION STAGE HAS BEEN COMPLETED ===')    
    
    # STAGE 6
    stacking_regressor_modeling.run_stage(spark, is_new_data=True)
    logger.info('=== V. PREDICTIONS HAVE BEEN CALCULATED AND SAVED===')   
    
    # STAGE 7
    if calculate_metrics:
        metrics_calculation.run_stage(spark, is_new_data=True)
        logger.info('=== VI. METRICS CALCULATION STAGE HAS BEEN COMPLETED ===')
    else:
        logger.info('=== VI. METRICS CALCULATION STAGE HAS BEEN SKIPPED ===')        
    
    logger.info('=== COMPLETED PROCESSING NEW DATA ===')